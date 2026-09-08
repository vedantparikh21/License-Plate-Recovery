"""
Live demo: feed a real image (file or webcam snapshot) and see the full
pipeline run in real time -- detected plate region, plain vs enhanced
OCR, side by side. Meant to be run on your own machine during a live
demo/interview, not in the offline evaluation harness.

Usage:
    python src/live_demo.py                  # interactive menu (file or webcam)
    python src/live_demo.py path/to/image.jpg # process one image directly, then exit

Controls in webcam mode:
    SPACE = capture current frame and run the pipeline on it
    ESC   = quit webcam mode

The pipeline applied here is the ADAPTIVE one (enhance.enhance_adaptive):
it measures blur/noise/exposure on the actual input and only applies the
corrections that measurement calls for, then always finishes with
super-resolution on the detected crop -- this is deliberately different
from the fixed per-degradation-type recipe used in the offline evaluation
(pipeline_real.py), because a live input's degradation type is unknown,
unlike the controlled offline experiment.
"""
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect
import enhance
import pretrained_models as pm

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "live_demo_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def label(img, text, color=(30, 30, 30)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), color, -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_labeled(items, target_h=260):
    import numpy as np
    resized = []
    for img, txt in items:
        h, w = img.shape[:2]
        scale = target_h / h
        img_r = cv2.resize(img, (max(1, int(w * scale)), target_h))
        resized.append(label(img_r, txt))
    return np.hstack(resized)


def best_box(dets):
    if not dets:
        return None
    return sorted(dets, key=lambda d: -d[4])[0][:4]


def try_multiple_enhancements(crop):
    """
    For a live, unknown-degradation crop: try several candidate corrections
    and keep whichever the pretrained OCR model itself is most confident
    about. This is a pragmatic stand-in for true blind deconvolution --
    we don't know the real blur angle/kernel for a live input the way we
    do in the controlled offline evaluation, so we search a small set of
    plausible candidates instead of guessing one.
    Returns (best_crop, best_text, best_conf, description).
    """
    candidates = [("original (no correction)", crop)]
    for angle in range(0, 180, 15):
        try:
            candidates.append((f"deconvolution @ {angle} deg",
                                enhance.deconvolve_wiener(crop, kernel_size=15, angle=angle, K=0.02)))
        except Exception:
            pass
    candidates.append(("unsharp mask", enhance.deblur_unsharp(crop, sigma=2.0, amount=1.4)))
    candidates.append(("denoise + CLAHE", enhance.enhance_contrast_clahe(enhance.denoise(crop))))

    best = None
    for desc, cand in candidates:
        sr = enhance.super_resolve(enhance.enhance_contrast_clahe(cand))
        text, conf = pm.read_plate_text(sr)
        score = conf * (1.0 if text else 0.0)  # empty text can't win regardless of confidence
        if best is None or score > best[0]:
            best = (score, sr, text, conf, desc)

    _, best_crop, best_text, best_conf, best_desc = best
    return best_crop, best_text, best_conf, best_desc


def process_image(image):
    """Runs: plain detect+OCR  vs  adaptive-enhance -> re-detect -> best-of-N -> OCR.
    Returns (comparison_panel_bgr, info_dict)."""
    t0 = time.time()

    # --- Plain (no enhancement) ---
    plain_dets = pm.detect_plates(image)
    plain_box = best_box(plain_dets)
    if plain_box is not None:
        plain_crop = detect.crop_with_margin(image, plain_box, margin_frac=0.1)
        plain_text, plain_conf = pm.read_plate_text(plain_crop)
    else:
        plain_crop = image
        plain_text, plain_conf = pm.read_plate_text(image)  # give it a fair shot even without a detection box

    # --- Adaptive scene-level correction, re-detect, then best-of-N crop correction ---
    enhanced_scene, scores = enhance.enhance_adaptive(image)
    enh_dets = pm.detect_plates(enhanced_scene)
    enh_box = best_box(enh_dets)
    if enh_box is not None:
        raw_crop = detect.crop_with_margin(enhanced_scene, enh_box, margin_frac=0.1)
    else:
        raw_crop = detect.crop_with_margin(enhanced_scene, plain_box, margin_frac=0.1) if plain_box else enhanced_scene
    enh_crop, enh_text, enh_conf, best_desc = try_multiple_enhancements(raw_crop)
    scores["applied"] = scores["applied"] + [f"crop-level: {best_desc}"]

    elapsed = time.time() - t0

    # --- Build visual panel ---
    full_annotated = image.copy()
    box_to_draw = enh_box or plain_box
    if box_to_draw:
        x, y, w, h = box_to_draw
        cv2.rectangle(full_annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(full_annotated, "detected", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(full_annotated, "NO PLATE DETECTED", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    top = hstack_labeled([(full_annotated, "INPUT + detected region")])
    bottom = hstack_labeled([
        (plain_crop, f"PLAIN (no enhancement) | OCR: '{plain_text}' (conf {plain_conf:.2f})"),
        (enh_crop, f"ENHANCED ({'+'.join(scores['applied']) or 'none needed'}) | OCR: '{enh_text}' (conf {enh_conf:.2f})"),
    ], target_h=220)

    import numpy as np
    if bottom.shape[1] != top.shape[1]:
        scale = top.shape[1] / bottom.shape[1]
        bottom = cv2.resize(bottom, (top.shape[1], int(bottom.shape[0] * scale)))
    panel = np.vstack([top, bottom])

    info = {
        "plain_text": plain_text, "plain_conf": plain_conf,
        "enhanced_text": enh_text, "enhanced_conf": enh_conf,
        "measured": scores, "elapsed_sec": round(elapsed, 2),
        "detected": box_to_draw is not None,
    }
    return panel, info


def run_on_file(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Could not read image: {path}")
        return
    panel, info = process_image(image)
    print(f"\n--- Result ({info['elapsed_sec']}s) ---")
    print(f"Plain OCR:    '{info['plain_text']}'  (confidence {info['plain_conf']:.2f})")
    print(f"Enhanced OCR: '{info['enhanced_text']}'  (confidence {info['enhanced_conf']:.2f})")
    print(f"Corrections applied: {info['measured']['applied'] or 'none (image already looked fine)'}")
    print(f"Measured -- blur: {info['measured']['blur_score']}, "
          f"noise: {info['measured']['noise_score']}, brightness: {info['measured']['brightness']}")

    out_path = os.path.join(OUTPUT_DIR, f"live_{int(time.time())}.png")
    cv2.imwrite(out_path, panel)
    print(f"Saved: {out_path}")

    cv2.imshow("License Plate Recovery - Live Demo (press any key to close)", panel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_on_directory(dir_path):
    """Process every image in a directory, save each result panel, print a summary table."""
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = sorted(f for f in os.listdir(dir_path) if f.lower().endswith(exts))
    if not files:
        print(f"No images found in {dir_path}")
        return

    print(f"Found {len(files)} images. Processing...\n")
    results = []
    for fname in files:
        path = os.path.join(dir_path, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"  SKIP (couldn't read): {fname}")
            continue
        panel, info = process_image(image)
        out_name = f"batch_{os.path.splitext(fname)[0]}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), panel)
        results.append((fname, info))
        print(f"  {fname:30s} plain: '{info['plain_text']}'  ->  enhanced: '{info['enhanced_text']}'  "
              f"({info['elapsed_sec']}s)")

    print(f"\nDone. {len(results)} panels saved to {OUTPUT_DIR}")


def run_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return
    print("Webcam live. SPACE = capture + run pipeline, ESC = quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preview = frame.copy()
        cv2.putText(preview, "SPACE = capture & run   ESC = quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Webcam - live", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            panel, info = process_image(frame)
            out_path = os.path.join(OUTPUT_DIR, f"live_{int(time.time())}.png")
            cv2.imwrite(out_path, panel)
            print(f"\nPlain OCR: '{info['plain_text']}' (conf {info['plain_conf']:.2f})  "
                  f"| Enhanced OCR: '{info['enhanced_text']}' (conf {info['enhanced_conf']:.2f})  "
                  f"| saved: {out_path}")
            cv2.imshow("Result (press any key to return to webcam)", panel)
            cv2.waitKey(0)
            cv2.destroyWindow("Result (press any key to return to webcam)")
    cap.release()
    cv2.destroyAllWindows()


def main():
    if len(sys.argv) > 1:
        run_on_file(sys.argv[1])
        return

    print("License Plate Recovery -- Live Demo")
    print("1) Process an image file")
    print("2) Use webcam")
    choice = input("Choose (1/2): ").strip()
    if choice == "2":
        run_webcam()
    else:
        path = input("Path to image file or folder: ").strip().strip('"')
        if os.path.isdir(path):
            run_on_directory(path)
        else:
            run_on_file(path)


if __name__ == "__main__":
    main()
