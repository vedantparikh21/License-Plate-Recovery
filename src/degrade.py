"""
Degradation functions simulating real-world capture problems.
Each function takes a clean BGR scene (np.ndarray) and returns a degraded
version, applied to the whole scene (as a real camera issue would affect
the full frame, not just the plate crop).
"""
import numpy as np
import cv2


def motion_blur(img, kernel_size=15, angle=0):
    k = np.zeros((kernel_size, kernel_size))
    k[kernel_size // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    k = cv2.warpAffine(k, M, (kernel_size, kernel_size))
    k = k / k.sum()
    return cv2.filter2D(img, -1, k)


def defocus_blur(img, radius=6):
    return cv2.GaussianBlur(img, (0, 0), sigmaX=radius)


def low_resolution(img, factor=4):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def gaussian_noise(img, sigma=18):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_compression(img, quality=15):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def under_exposure(img, factor=0.4):
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def over_exposure(img, add=90):
    return np.clip(img.astype(np.float32) + add, 0, 255).astype(np.uint8)


def perspective_skew(img, bbox_xywh, strength=0.35):
    """
    Warp the whole scene with a projective transform that skews the plate
    region (simulates the plate being photographed off-angle rather than
    front-on). bbox_xywh is used only to keep the skew roughly centered on
    the plate so the rest of the scene doesn't get destroyed.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_xywh
    cx, cy = x + bw / 2, y + bh / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = strength * bw
    dst = np.float32([
        [0, 0],
        [w, 0],
        [w, h - dx * 0.6],
        [0, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def camera_shake_sequence(img, n_frames=6, max_shift=4, noise_sigma=6):
    """
    Simulate a short handheld/jittery video: n_frames slightly shifted +
    independently noised versions of the same scene. Used to test
    multi-frame averaging / stabilization.
    """
    h, w = img.shape[:2]
    frames = []
    for _ in range(n_frames):
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        frames.append(gaussian_noise(shifted, sigma=noise_sigma))
    return frames


DEGRADATIONS = {
    "motion_blur": lambda img, bbox: motion_blur(img, kernel_size=17, angle=15),
    "defocus_blur": lambda img, bbox: defocus_blur(img, radius=5),
    "low_res": lambda img, bbox: low_resolution(img, factor=5),
    "noise": lambda img, bbox: gaussian_noise(img, sigma=22),
    "jpeg_compression": lambda img, bbox: jpeg_compression(img, quality=12),
    "under_exposure": lambda img, bbox: under_exposure(img, factor=0.35),
    "over_exposure": lambda img, bbox: over_exposure(img, add=100),
    "perspective": lambda img, bbox: perspective_skew(img, bbox, strength=0.4),
    "combined_hard": lambda img, bbox: gaussian_noise(
        low_resolution(motion_blur(img, kernel_size=13, angle=10), factor=3), sigma=12
    ),
}
