"""
Compares the model combination actually used throughout this project
(YOLOv9-t-384 detector + CCT-XS-v2 OCR) against the strongest currently
available combination in the same two libraries (YOLOv9-s-608 -- the
highest published mAP50 in open-image-models' own plate-detection model
table -- + CCT-S-v2 -- the fast-plate-ocr maintainer's now-recommended
default for new integrations), on the SAME 25 curated cases x 9
degradations (+ a clean/no-degradation pass) used everywhere else in
this project.

No enhancement is applied here on purpose: this isolates the effect of
the model choice itself from the enhancement pipeline, which is a
separate, orthogonal variable already measured in full in
pipeline_real.py / summary.md. Mixing the two would make it impossible
to say whether a change in accuracy came from the bigger model or from
enhancement.

Both accuracy (exact-match, char-accuracy) AND wall-clock CPU latency
(detector forward pass and OCR forward pass, timed separately, after a
warm-up call so model-load time doesn't pollute the measurement) are
recorded per instance, so the comparison can show not just "is the
bigger model more accurate" but "how much more accurate, for how much
extra time per plate, on the same CPU this whole project runs on."
"""
import os
import sys
import json
import csv
import time

import numpy as np
import cv2
from open_image_models import create_detector
from fast_plate_ocr import LicensePlateRecognizer

sys.path.insert(0, os.path.dirname(__file__))
import degrade
import detect
import metrics

np.random.seed(7)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "real_dataset")
RES_DIR = os.path.join(PROJECT_ROOT, "data", "results_real")
os.makedirs(RES_DIR, exist_ok=True)

# The two combinations being compared. "current" is what pretrained_models.py
# and every other result in this project uses. "best_available" swaps in the
# top of each library's own model table as of when this was written --
# double check ankandrew/open-image-models and ankandrew/fast-plate-ocr's
# model zoos if you re-run this later, since "best available" can change.
COMBOS = {
    "current (yolo-v9-t-384 + cct-xs-v2)": {
        "detector": "yolo-v9-t-384-license-plate-end2end",
        "ocr": "cct-xs-v2-global-model",
    },
    "best-available (yolo-v9-s-608 + cct-s-v2)": {
        "detector": "yolo-v9-s-608-license-plate-end2end",
        "ocr": "cct-s-v2-global-model",
    },
}


def normalize_text(t):
    return "".join(ch for ch in t.upper() if ch.isalnum())


def timed_detect(detector, img):
    t0 = time.perf_counter()
    dets = detector.predict(img)
    dt = time.perf_counter() - t0
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2
        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(d.confidence)))
    return out, dt


def timed_ocr(ocr, crop_bgr):
    t0 = time.perf_counter()
    results = ocr.run(crop_bgr, return_confidence=True)
    dt = time.perf_counter() - t0
    if not results:
        return "", 0.0, dt
    pred = results[0]
    text = pred.plate or ""
    conf = float(sum(pred.char_probs) / len(pred.char_probs)) if len(pred.char_probs) else 0.0
    return text, conf, dt


def best_box_by_gt(dets_xywhc, gt_bbox):
    if not dets_xywhc:
        return None, 0.0
    scored = [(d[:4], detect.iou(d[:4], gt_bbox)) for d in dets_xywhc]
    scored.sort(key=lambda t: -t[1])
    return scored[0]


def run_instance(detector, ocr, entry, degradation_name):
    bbox_gt = tuple(entry["bbox_xywh"])
    text_gt = normalize_text(entry["text_gt"])
    image = cv2.imread(f"{DATA_DIR}/{entry['image_file']}")

    if degradation_name == "clean":
        scene = image
    else:
        scene = degrade.DEGRADATIONS[degradation_name](image, bbox_gt)

    dets, det_dt = timed_detect(detector, scene)
    box, iou_val = best_box_by_gt(dets, bbox_gt)
    crop_box = box if box is not None else bbox_gt
    crop = detect.crop_with_margin(scene, crop_box, margin_frac=0.1)
    text, conf, ocr_dt = timed_ocr(ocr, crop)
    text = normalize_text(text)

    return {
        "case_id": entry["case_id"], "region": entry["region"], "degradation": degradation_name,
        "det_iou": round(iou_val, 3),
        "ocr_text": text, "gt_text": text_gt,
        "exact_match": text == text_gt,
        "char_acc": round(metrics.char_accuracy(text, text_gt), 3),
        "det_latency_ms": round(det_dt * 1000, 2),
        "ocr_latency_ms": round(ocr_dt * 1000, 2),
        "total_latency_ms": round((det_dt + ocr_dt) * 1000, 2),
    }


def main():
    with open(f"{DATA_DIR}/manifest.json") as f:
        manifest = json.load(f)
    degradation_names = ["clean"] + list(degrade.DEGRADATIONS.keys())

    all_rows = []
    for combo_name, spec in COMBOS.items():
        print(f"\n=== {combo_name} ===")
        detector = create_detector(spec["detector"])
        ocr = LicensePlateRecognizer(spec["ocr"])

        # Warm-up call so the first-call overhead (ONNX session lazy init)
        # doesn't get baked into the very first timed measurement.
        warm_img = cv2.imread(f"{DATA_DIR}/{manifest[0]['image_file']}")
        detector.predict(warm_img)
        ocr.run(warm_img, return_confidence=True)

        t0 = time.time()
        for entry in manifest:
            for deg_name in degradation_names:
                try:
                    row = run_instance(detector, ocr, entry, deg_name)
                    row["combo"] = combo_name
                    all_rows.append(row)
                except Exception as e:
                    print(f"FAILED {combo_name} {entry['case_id']} {deg_name}: {e}")
        print(f"{combo_name}: {time.time() - t0:.1f}s for "
              f"{len(manifest) * len(degradation_names)} instances")

    keys = ["combo"] + [k for k in all_rows[0].keys() if k != "combo"]
    with open(f"{RES_DIR}/model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    # ---------- Aggregate summary ----------
    from collections import defaultdict
    by_combo = defaultdict(list)
    for r in all_rows:
        by_combo[r["combo"]].append(r)

    summary_rows = []
    for combo_name, rows in by_combo.items():
        n = len(rows)
        exact = sum(r["exact_match"] for r in rows) / n
        char_acc = np.mean([r["char_acc"] for r in rows])
        iou = np.mean([r["det_iou"] for r in rows])
        det_ms = np.mean([r["det_latency_ms"] for r in rows])
        ocr_ms = np.mean([r["ocr_latency_ms"] for r in rows])
        total_ms = np.mean([r["total_latency_ms"] for r in rows])
        summary_rows.append({
            "combo": combo_name, "n": n,
            "exact_match": round(exact, 3), "char_acc": round(float(char_acc), 3),
            "det_iou": round(float(iou), 3),
            "det_latency_ms": round(float(det_ms), 2), "ocr_latency_ms": round(float(ocr_ms), 2),
            "total_latency_ms": round(float(total_ms), 2),
            "throughput_ips": round(1000.0 / total_ms, 2),
        })

    with open(f"{RES_DIR}/model_comparison_summary.md", "w") as f:
        f.write("# Model combination comparison\n\n")
        f.write("Same 25 curated cases x 9 degradations + 1 clean pass, no enhancement "
                "applied (isolates the model-choice effect). CPU latency, single image "
                "at a time, after a warm-up call. See src/compare_models.py.\n\n")
        f.write("| Combination | N | Exact-match | Char-acc | Det. IoU | "
                "Detector (ms) | OCR (ms) | Total (ms/plate) | Throughput (plates/s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| {r['combo']} | {r['n']} | {r['exact_match']:.0%} | {r['char_acc']:.2f} | "
                    f"{r['det_iou']:.2f} | {r['det_latency_ms']:.1f} | {r['ocr_latency_ms']:.1f} | "
                    f"{r['total_latency_ms']:.1f} | {r['throughput_ips']:.1f} |\n")

    print("\nWrote model_comparison.csv and model_comparison_summary.md")
    for r in summary_rows:
        print(r)

    # ---------- Chart: accuracy vs latency trade-off ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        names = [r["combo"] for r in summary_rows]
        short_names = ["Current\n(t-384 + xs-v2)", "Best-available\n(s-608 + s-v2)"]
        exact = [r["exact_match"] * 100 for r in summary_rows]
        characc = [r["char_acc"] * 100 for r in summary_rows]
        latency = [r["total_latency_ms"] for r in summary_rows]
        colors = ["#4C72B0", "#DD8452"]

        ax = axes[0]
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width / 2, exact, width, label="Exact-match %", color=colors[0])
        ax.bar(x + width / 2, characc, width, label="Char-accuracy %", color=colors[1])
        ax.set_xticks(x)
        ax.set_xticklabels(short_names)
        ax.set_ylabel("%")
        ax.set_title("Accuracy: current vs. best-available models")
        ax.legend()
        ax.set_ylim(0, 100)
        for i, v in enumerate(exact):
            ax.text(i - width / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
        for i, v in enumerate(characc):
            ax.text(i + width / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=9)

        ax2 = axes[1]
        for i, (name, sn) in enumerate(zip(names, short_names)):
            ax2.scatter(latency[i], exact[i], s=160, color=colors[i], label=sn.replace("\n", " "))
            ax2.annotate(f"{latency[i]:.0f} ms/plate", (latency[i], exact[i]),
                         textcoords="offset points", xytext=(8, -4), fontsize=9)
        ax2.set_xlabel("Total latency (ms/plate, detector + OCR, CPU)")
        ax2.set_ylabel("Exact-match %")
        ax2.set_title("Accuracy vs. latency trade-off")
        ax2.legend(loc="lower right", fontsize=8)
        ax2.set_ylim(0, 100)

        fig.suptitle("Model choice: what swapping detector + OCR actually buys you", y=1.02)
        fig.tight_layout()
        chart_path = f"{RES_DIR}/model_comparison_chart.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {chart_path}")
    except ImportError:
        print("matplotlib not installed -- skipped chart generation "
              "(CSV and summary.md were still written).")


if __name__ == "__main__":
    main()
