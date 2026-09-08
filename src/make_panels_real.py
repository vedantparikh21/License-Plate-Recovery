"""Before/after comparison panels for the real-data pipeline (v2)."""
import os
import sys
import csv
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(PROJECT_ROOT, "data", "results_real")
PANEL_DIR = f"{RES_DIR}/panels"
os.makedirs(PANEL_DIR, exist_ok=True)


def label(img, text, color=(30, 30, 30)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), color, -1)
    cv2.putText(out, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_labeled(panels_with_labels, target_h=140):
    resized = []
    for img, txt in panels_with_labels:
        h, w = img.shape[:2]
        scale = target_h / h
        img_r = cv2.resize(img, (max(1, int(w * scale)), target_h))
        resized.append(label(img_r, txt))
    return np.hstack(resized)


def build_panel(case_id, degradation, row):
    prefix = f"{RES_DIR}/{case_id}_{degradation}"
    degraded = cv2.imread(f"{prefix}_1_degraded_crop.png")
    enhanced = cv2.imread(f"{prefix}_2_enhanced_crop.png")
    clean = cv2.imread(f"{prefix}_3_clean_reference_crop.png")
    if degraded is None or enhanced is None or clean is None:
        return None
    panel = hstack_labeled([
        (clean, "clean reference (real photo)"),
        (degraded, f"degraded: {degradation} | OCR: '{row['ocr_before']}'"),
        (enhanced, f"enhanced | OCR: '{row['ocr_after']}' (gt: {row['gt_text']})"),
    ])
    out_path = f"{PANEL_DIR}/{case_id}_{degradation}_panel.png"
    cv2.imwrite(out_path, panel)
    return out_path


def main():
    rows = list(csv.DictReader(open(f"{RES_DIR}/results.csv")))
    by_key = {(r["case_id"], r["degradation"]): r for r in rows}

    selection = [
        ("us_wts-lg-000016", "motion_blur"),       # 0% -> 100%, headline win
        ("us_wts-lg-000010", "noise"),              # strong baseline degraded by our enhancement
        ("us_wts-lg-000148", "defocus_blur"),       # genuine failure case
        ("br_OYJ9557", "combined_hard"),            # compounded-degradation recovery
        ("us_wts-lg-000016", "perspective"),        # pretrained OCR already robust; correction doesn't hurt
        ("us_wts-lg-000016", "video_multiframe"),   # multi-frame stabilization
    ]
    for case_id, deg in selection:
        row = by_key.get((case_id, deg))
        if row is None:
            print("missing:", case_id, deg)
            continue
        path = build_panel(case_id, deg, row)
        print(case_id, deg, "->", path)


if __name__ == "__main__":
    main()
