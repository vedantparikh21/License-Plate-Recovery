"""
Curate a diverse, real subset from the OpenALPR benchmark dataset
(https://github.com/openalpr/benchmarks) for our evaluation.

For each candidate image we also sanity-check that the pretrained
detector can actually find the ground-truth plate on the CLEAN
(undegraded) image with reasonable IoU -- this keeps our primary
quantitative table focused on genuine, working real-world cases, while
we deliberately keep a couple of known-hard ones for the failure-case
discussion instead of silently dropping every hard case.
"""
import os
import glob
import json
import shutil
import random

import cv2
import sys
sys.path.insert(0, os.path.dirname(__file__))
from open_image_models import create_detector

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "benchmarks-master", "endtoend")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "real_dataset")

random.seed(11)


def load_region_annotations(region):
    entries = []
    for txt_path in sorted(glob.glob(f"{SRC_ROOT}/{region}/*.txt")):
        parts = open(txt_path).read().strip().split("\t")
        if len(parts) < 6:
            continue
        fn, x, y, w, h, text = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), parts[5]
        img_path = f"{SRC_ROOT}/{region}/{fn}"
        if os.path.exists(img_path):
            entries.append({"region": region, "filename": fn, "bbox_xywh": [x, y, w, h], "text_gt": text})
    return entries


def iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main(n_us=12, n_eu=6, n_br=6):
    os.makedirs(OUT_DIR, exist_ok=True)
    detector = create_detector("yolo-v9-t-384-license-plate-end2end")

    all_candidates = []
    for region, n in [("us", n_us), ("eu", n_eu), ("br", n_br)]:
        entries = load_region_annotations(region)
        random.shuffle(entries)
        # widen the search a bit beyond n so we can filter by detectability
        picked, hard_examples = [], []
        for e in entries:
            img = cv2.imread(f"{SRC_ROOT}/{region}/{e['filename']}")
            if img is None:
                continue
            dets = detector.predict(img)
            best_iou = max((iou(tuple(e["bbox_xywh"]),
                                 (d.bounding_box.x1, d.bounding_box.y1,
                                  d.bounding_box.x2 - d.bounding_box.x1,
                                  d.bounding_box.y2 - d.bounding_box.y1))
                             for d in dets), default=0.0)
            e["clean_detection_iou"] = round(best_iou, 3)
            if best_iou >= 0.5 and len(picked) < n:
                picked.append(e)
            elif best_iou < 0.3 and len(hard_examples) < 2:
                hard_examples.append(e)  # keep a couple of genuine hard cases too
            if len(picked) >= n and len(hard_examples) >= 2:
                break
        all_candidates.extend(picked)
        all_candidates.extend(hard_examples)
        print(f"{region}: picked {len(picked)} clean-working cases + {len(hard_examples)} hard cases "
              f"(scanned {entries.index(e) + 1} annotations)")

    manifest = []
    for e in all_candidates:
        src = f"{SRC_ROOT}/{e['region']}/{e['filename']}"
        case_id = f"{e['region']}_{os.path.splitext(e['filename'])[0]}"[:40]
        dst = f"{OUT_DIR}/{case_id}.jpg"
        shutil.copy(src, dst)
        manifest.append({
            "case_id": case_id,
            "region": e["region"],
            "image_file": f"{case_id}.jpg",
            "bbox_xywh": e["bbox_xywh"],
            "text_gt": e["text_gt"],
            "clean_detection_iou": e["clean_detection_iou"],
        })

    with open(f"{OUT_DIR}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nTotal curated cases: {len(manifest)} -> {OUT_DIR}/manifest.json")


if __name__ == "__main__":
    main()
