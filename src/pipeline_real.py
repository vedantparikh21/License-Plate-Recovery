"""
End-to-end evaluation pipeline (v2): real annotated photographs +
pretrained detection/OCR models, our own enhancement pipeline in between.

For every case x every degradation:
  1. Degrade the real photo (whole frame).
  2. BASELINE: run the pretrained YOLOv9 detector + CCT OCR directly on
     the degraded frame -- this is what you'd get with no enhancement
     at all.
  3. ENHANCED: run our scene-level correction (deconvolution / denoise /
     CLAHE, chosen per degradation type), re-run the pretrained detector
     on the corrected frame, crop, run learned super-resolution on the
     crop, then run the pretrained OCR again.
  4. Score both against the real ground-truth box + text.

No model is trained or fine-tuned anywhere in this file -- detection and
OCR are frozen, pretrained, off-the-shelf inference (see
pretrained_models.py for exactly which ones and where they come from).
"""
import os
import sys
import json
import csv
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
import degrade
import detect  # only used for iou() and crop_with_margin() helpers
import enhance
import metrics
import pretrained_models as pm

np.random.seed(7)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_dataset")
RES_DIR = os.path.join(PROJECT_ROOT, "data", "results_real")
os.makedirs(RES_DIR, exist_ok=True)


def load_manifest():
    with open(f"{DATA_DIR}/manifest.json") as f:
        return json.load(f)


def best_box_by_gt(dets_xywhc, gt_bbox):
    """dets_xywhc: list of (x,y,w,h,conf). Returns (best_box_xywh_or_None, best_iou)."""
    if not dets_xywhc:
        return None, 0.0
    scored = [(d[:4], detect.iou(d[:4], gt_bbox)) for d in dets_xywhc]
    scored.sort(key=lambda t: -t[1])
    return scored[0]


def normalize_text(t):
    return "".join(ch for ch in t.upper() if ch.isalnum())


def run_case(entry, degradation_name):
    case_id = entry["case_id"]
    bbox_gt = tuple(entry["bbox_xywh"])
    text_gt = normalize_text(entry["text_gt"])

    image = cv2.imread(f"{DATA_DIR}/{entry['image_file']}")
    clean_crop = detect.crop_with_margin(image, bbox_gt, margin_frac=0.15)

    degrade_fn = degrade.DEGRADATIONS[degradation_name]
    degraded_scene = degrade_fn(image, bbox_gt)

    # ---------- BASELINE: pretrained model directly on degraded frame ----------
    baseline_dets = pm.detect_plates(degraded_scene)
    baseline_box, baseline_iou = best_box_by_gt(baseline_dets, bbox_gt)
    if baseline_box is not None:
        baseline_crop = detect.crop_with_margin(degraded_scene, baseline_box, margin_frac=0.1)
        baseline_text, baseline_conf = pm.read_plate_text(baseline_crop)
    else:
        baseline_crop = detect.crop_with_margin(degraded_scene, bbox_gt, margin_frac=0.1)
        baseline_text, baseline_conf = "", 0.0
    baseline_text = normalize_text(baseline_text)

    # ---------- ENHANCED ----------
    if degradation_name == "perspective":
        # Perspective correction is inherently crop-level: isolate the
        # region first (using the baseline detection if it's plausible,
        # else fall back to ground truth so OCR-level effect can still be
        # measured even when detection itself fails), un-warp, then OCR.
        # The plate detector is not re-run for this path -- see report.
        crop_src_box = baseline_box if (baseline_box is not None and baseline_iou >= 0.15) else bbox_gt
        raw_crop = detect.crop_with_margin(degraded_scene, crop_src_box, margin_frac=0.15)
        enhanced_crop = enhance.enhance_crop_level(raw_crop, degradation_name, do_perspective=True)
        enhanced_iou = baseline_iou
        detector_rerun = False
    else:
        enhanced_scene = enhance.enhance_scene_level(degraded_scene, degradation_name)
        enhanced_dets = pm.detect_plates(enhanced_scene)
        enhanced_box, enhanced_iou = best_box_by_gt(enhanced_dets, bbox_gt)
        crop_box = enhanced_box if enhanced_box is not None else bbox_gt
        raw_crop = detect.crop_with_margin(enhanced_scene, crop_box, margin_frac=0.1)
        enhanced_crop = enhance.enhance_crop_level(raw_crop, degradation_name, do_perspective=False)
        detector_rerun = True

    enhanced_text, enhanced_conf = pm.read_plate_text(enhanced_crop)
    enhanced_text = normalize_text(enhanced_text)

    # ---------- Metrics ----------
    degraded_crop_gt_geom = detect.crop_with_margin(degraded_scene, bbox_gt, margin_frac=0.15)
    p_before, s_before = metrics.compute_psnr_ssim(degraded_crop_gt_geom, clean_crop)
    p_after, s_after = metrics.compute_psnr_ssim(enhanced_crop, clean_crop)
    acc_before = metrics.char_accuracy(baseline_text, text_gt)
    acc_after = metrics.char_accuracy(enhanced_text, text_gt)

    prefix = f"{RES_DIR}/{case_id}_{degradation_name}"
    cv2.imwrite(f"{prefix}_1_degraded_crop.png", baseline_crop)
    cv2.imwrite(f"{prefix}_2_enhanced_crop.png", enhanced_crop)
    cv2.imwrite(f"{prefix}_3_clean_reference_crop.png", clean_crop)

    return {
        "case_id": case_id, "region": entry["region"], "degradation": degradation_name,
        "detector_rerun": detector_rerun,
        "det_iou_before": round(baseline_iou, 3), "det_iou_after": round(enhanced_iou, 3),
        "gt_text": text_gt,
        "ocr_before": baseline_text, "ocr_after": enhanced_text,
        "ocr_conf_before": round(baseline_conf, 3), "ocr_conf_after": round(enhanced_conf, 3),
        "char_acc_before": round(acc_before, 3), "char_acc_after": round(acc_after, 3),
        "exact_before": baseline_text == text_gt, "exact_after": enhanced_text == text_gt,
        "psnr_before": round(p_before, 2), "psnr_after": round(p_after, 2),
        "ssim_before": round(s_before, 3), "ssim_after": round(s_after, 3),
    }


def run_video_case(entry, n_frames=6, max_shift=4, noise_sigma=8):
    case_id = entry["case_id"]
    bbox_gt = tuple(entry["bbox_xywh"])
    text_gt = normalize_text(entry["text_gt"])
    image = cv2.imread(f"{DATA_DIR}/{entry['image_file']}")
    clean_crop = detect.crop_with_margin(image, bbox_gt, margin_frac=0.15)

    frames = degrade.camera_shake_sequence(image, n_frames=n_frames, max_shift=max_shift, noise_sigma=noise_sigma)

    # Baseline: pretrained model on the single first shaky frame
    base_dets = pm.detect_plates(frames[0])
    base_box, base_iou = best_box_by_gt(base_dets, bbox_gt)
    base_crop = detect.crop_with_margin(frames[0], base_box if base_box else bbox_gt, margin_frac=0.1)
    base_text, base_conf = pm.read_plate_text(base_crop)
    base_text = normalize_text(base_text)

    # Enhanced: ECC-align + average the full frames, then re-detect + SR + OCR
    averaged_scene = enhance.multiframe_average(frames)
    enh_dets = pm.detect_plates(averaged_scene)
    enh_box, enh_iou = best_box_by_gt(enh_dets, bbox_gt)
    enh_raw_crop = detect.crop_with_margin(averaged_scene, enh_box if enh_box else bbox_gt, margin_frac=0.1)
    enh_crop = enhance.enhance_crop_level(enh_raw_crop, "generic", do_perspective=False)
    enh_text, enh_conf = pm.read_plate_text(enh_crop)
    enh_text = normalize_text(enh_text)

    degraded_crop_gt_geom = detect.crop_with_margin(frames[0], bbox_gt, margin_frac=0.15)
    p_before, s_before = metrics.compute_psnr_ssim(degraded_crop_gt_geom, clean_crop)
    p_after, s_after = metrics.compute_psnr_ssim(enh_crop, clean_crop)

    prefix = f"{RES_DIR}/{case_id}_video_multiframe"
    cv2.imwrite(f"{prefix}_1_degraded_crop.png", base_crop)
    cv2.imwrite(f"{prefix}_2_enhanced_crop.png", enh_crop)
    cv2.imwrite(f"{prefix}_3_clean_reference_crop.png", clean_crop)

    return {
        "case_id": case_id, "region": entry["region"], "degradation": "video_multiframe",
        "detector_rerun": True,
        "det_iou_before": round(base_iou, 3), "det_iou_after": round(enh_iou, 3),
        "gt_text": text_gt,
        "ocr_before": base_text, "ocr_after": enh_text,
        "ocr_conf_before": round(base_conf, 3), "ocr_conf_after": round(enh_conf, 3),
        "char_acc_before": round(metrics.char_accuracy(base_text, text_gt), 3),
        "char_acc_after": round(metrics.char_accuracy(enh_text, text_gt), 3),
        "exact_before": base_text == text_gt, "exact_after": enh_text == text_gt,
        "psnr_before": round(p_before, 2), "psnr_after": round(p_after, 2),
        "ssim_before": round(s_before, 3), "ssim_after": round(s_after, 3),
    }


def main():
    manifest = load_manifest()
    degradation_names = list(degrade.DEGRADATIONS.keys())
    all_results = []
    t0 = time.time()

    for i, entry in enumerate(manifest):
        for deg_name in degradation_names:
            try:
                res = run_case(entry, deg_name)
                all_results.append(res)
                print(f"[{i+1}/{len(manifest)}] {entry['case_id']:32s} {deg_name:16s} "
                      f"IoU {res['det_iou_before']:.2f}->{res['det_iou_after']:.2f}  "
                      f"acc {res['char_acc_before']:.2f}->{res['char_acc_after']:.2f}  "
                      f"'{res['ocr_before']}'->'{res['ocr_after']}' (gt='{res['gt_text']}')")
            except Exception as e:
                print(f"FAILED {entry['case_id']} {deg_name}: {e}")

    for entry in manifest[:8]:
        try:
            res = run_video_case(entry)
            all_results.append(res)
            print(f"{entry['case_id']:32s} {'video_multiframe':16s} "
                  f"IoU {res['det_iou_before']:.2f}->{res['det_iou_after']:.2f}  "
                  f"acc {res['char_acc_before']:.2f}->{res['char_acc_after']:.2f}")
        except Exception as e:
            print(f"FAILED {entry['case_id']} video: {e}")

    print(f"\nTotal time: {time.time()-t0:.1f}s for {len(all_results)} instances")

    keys = list(all_results[0].keys())
    with open(f"{RES_DIR}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_results)

    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_results:
        agg[r["degradation"]].append(r)

    with open(f"{RES_DIR}/summary.md", "w") as f:
        f.write("| Degradation | N | Det.IoU before->after | Exact-match before->after | "
                "Char-acc before->after | SSIM before->after | PSNR before->after |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for deg_name, rows in agg.items():
            n = len(rows)
            iou_b = np.mean([r["det_iou_before"] for r in rows])
            iou_a = np.mean([r["det_iou_after"] for r in rows])
            exact_b = sum(r["exact_before"] for r in rows) / n
            exact_a = sum(r["exact_after"] for r in rows) / n
            acc_b = np.mean([r["char_acc_before"] for r in rows])
            acc_a = np.mean([r["char_acc_after"] for r in rows])
            ssim_b = np.mean([r["ssim_before"] for r in rows])
            ssim_a = np.mean([r["ssim_after"] for r in rows])
            psnr_b = np.mean([r["psnr_before"] for r in rows])
            psnr_a = np.mean([r["psnr_after"] for r in rows])
            f.write(f"| {deg_name} | {n} | {iou_b:.2f}->{iou_a:.2f} | {exact_b:.0%}->{exact_a:.0%} | "
                    f"{acc_b:.2f}->{acc_a:.2f} | {ssim_b:.2f}->{ssim_a:.2f} | "
                    f"{psnr_b:.1f}->{psnr_a:.1f} |\n")

    print("Wrote results.csv and summary.md")


if __name__ == "__main__":
    main()
