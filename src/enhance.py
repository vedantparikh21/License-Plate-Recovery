"""
Enhancement stage: takes a degraded plate crop and applies a pipeline of
corrections. Each function is independent so the pipeline can be
ablated (report shows per-stage contribution).
"""
import cv2
import numpy as np
import os

_SR_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "data", "models", "FSRCNN_x3.pb")
_sr_instance = None


def _get_sr():
    global _sr_instance
    if _sr_instance is None:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_SR_MODEL_PATH)
        sr.setModel("fsrcnn", 3)
        _sr_instance = sr
    return _sr_instance


def deconvolve_wiener(img, kernel_size=15, angle=0, K=0.01):
    """Simple Wiener-style deconvolution to reverse a known-direction motion
    blur kernel. Works per-channel in the frequency domain."""
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
    k /= k.sum()

    out = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        h, w = channel.shape
        kernel_padded = np.zeros((h, w), dtype=np.float32)
        kh, kw = k.shape
        kernel_padded[:kh, :kw] = k
        kernel_padded = np.roll(kernel_padded, -kh // 2, axis=0)
        kernel_padded = np.roll(kernel_padded, -kw // 2, axis=1)

        H = np.fft.fft2(kernel_padded)
        G = np.fft.fft2(channel)
        H_conj = np.conj(H)
        F_hat = (H_conj / (np.abs(H) ** 2 + K)) * G
        restored = np.real(np.fft.ifft2(F_hat))
        out[:, :, c] = restored
    return np.clip(out, 0, 255).astype(np.uint8)


def deblur_unsharp(img, sigma=2.0, amount=1.5):
    """Fallback / complementary sharpening for defocus-type blur where the
    exact kernel isn't known: unsharp masking."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def correct_perspective_auto(img):
    """Detect the dominant quadrilateral in the crop and warp it to a
    front-facing rectangle. Falls back to the original image if no
    reliable quadrilateral is found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4 or cv2.contourArea(approx) < 0.3 * img.shape[0] * img.shape[1]:
        return img

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left

    w = int(max(np.linalg.norm(ordered[0] - ordered[1]), np.linalg.norm(ordered[3] - ordered[2])))
    h = int(max(np.linalg.norm(ordered[0] - ordered[3]), np.linalg.norm(ordered[1] - ordered[2])))
    if w < 10 or h < 10:
        return img
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(img, M, (w, h))


def super_resolve(img, max_input_area=25000):
    """Learned lightweight super-resolution (FSRCNN, OpenCV dnn_superres).
    Guards against huge inputs (FSRCNN is meant for small crops)."""
    h, w = img.shape[:2]
    if h * w > max_input_area:
        scale = (max_input_area / (h * w)) ** 0.5
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    sr = _get_sr()
    return sr.upsample(img)


def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                            templateWindowSize=7, searchWindowSize=21)


def enhance_contrast_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def multiframe_average(frames):
    """Average a short burst of jittery frames after aligning them to the
    first frame via ECC (a simple video-stabilization proxy)."""
    ref = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    aligned = [frames[0].astype(np.float32)]
    warp_mode = cv2.MOTION_TRANSLATION
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    for f in frames[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp_matrix = cv2.findTransformECC(ref, gray, warp_matrix, warp_mode, criteria)
            h, w = f.shape[:2]
            f_aligned = cv2.warpAffine(f, warp_matrix, (w, h),
                                        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                        borderMode=cv2.BORDER_REPLICATE)
        except cv2.error:
            f_aligned = f  # alignment failed on this frame, use as-is
        aligned.append(f_aligned.astype(np.float32))
    stacked = np.mean(aligned, axis=0)
    return np.clip(stacked, 0, 255).astype(np.uint8)


def enhance_scene_level(img, degradation_type="generic"):
    """Corrections applied at full-frame resolution, BEFORE re-running the
    plate detector. Only stages that make sense (and are affordable) at
    full-image scale live here -- deconvolution, denoising, contrast.
    Perspective correction and super-resolution are crop-level (see
    enhance_crop_level) since they need the plate region isolated first."""
    out = img.copy()

    if degradation_type == "motion_blur":
        out = deconvolve_wiener(out, kernel_size=17, angle=15, K=0.02)
    elif degradation_type == "defocus_blur":
        out = deblur_unsharp(out, sigma=2.5, amount=1.6)
    elif degradation_type == "combined_hard":
        out = deconvolve_wiener(out, kernel_size=13, angle=10, K=0.03)

    if degradation_type in ("under_exposure", "over_exposure"):
        out = enhance_contrast_clahe(out)
    if degradation_type in ("noise", "combined_hard", "jpeg_compression"):
        out = denoise(out)

    return out


def enhance_crop_level(crop, degradation_type="generic", do_perspective=False):
    """Corrections applied to an already-isolated plate crop: optional
    perspective un-warp, a final contrast pass, then learned
    super-resolution."""
    out = crop.copy()
    if do_perspective:
        out = correct_perspective_auto(out)
    out = enhance_contrast_clahe(out)
    out = super_resolve(out)
    return out


def estimate_blur_score(img):
    """Higher = sharper. Variance of Laplacian, a standard blur proxy."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_noise_score(img):
    """Rough noise estimate: std of the high-frequency residual
    (image minus a median-blurred version of itself)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
    residual = gray - smooth
    return float(np.std(residual))


def estimate_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def enhance_adaptive(img, blur_thresh=500.0, noise_thresh=22.5,
                      dark_thresh=80.0, bright_thresh=180.0):
    """
    For an image of UNKNOWN degradation type (e.g. a live demo input where
    we don't get to cheat and know what was applied): measure blur, noise,
    and exposure directly and only apply the corrections that measurement
    actually calls for. This directly addresses a finding from the offline
    evaluation (report section 4/5): unconditionally applying every
    enhancement stage can slightly hurt an already-good image. Returns
    (enhanced_image, dict of what was applied and the measured scores).
    """
    out = img.copy()
    applied = []

    blur_score = estimate_blur_score(out)
    if blur_score < blur_thresh:
        out = deblur_unsharp(out, sigma=2.0, amount=1.4)
        applied.append("deblur (unsharp)")

    noise_score = estimate_noise_score(out)
    if noise_score > noise_thresh:
        out = denoise(out)
        applied.append("denoise")

    brightness = estimate_brightness(out)
    if brightness < dark_thresh or brightness > bright_thresh:
        out = enhance_contrast_clahe(out)
        applied.append("CLAHE contrast")

    scores = {"blur_score": round(blur_score, 1), "noise_score": round(noise_score, 2),
              "brightness": round(brightness, 1), "applied": applied}
    return out, scores
    """Legacy single-shot pipeline (scene-level + crop-level combined),
    kept for the perspective / video code paths that enhance an
    already-cropped region directly rather than re-detecting on a full
    scene."""
    out = img.copy()

    if degradation_type == "motion_blur":
        out = deconvolve_wiener(out, kernel_size=17, angle=15, K=0.02)
    elif degradation_type == "defocus_blur":
        out = deblur_unsharp(out, sigma=2.5, amount=1.6)
    elif degradation_type == "perspective":
        out = correct_perspective_auto(out)
    elif degradation_type == "combined_hard":
        out = deconvolve_wiener(out, kernel_size=13, angle=10, K=0.03)

    if degradation_type in ("under_exposure", "over_exposure"):
        out = enhance_contrast_clahe(out)
    if degradation_type in ("noise", "combined_hard", "jpeg_compression"):
        out = denoise(out)

    out = enhance_contrast_clahe(out)
    out = super_resolve(out)
    return out