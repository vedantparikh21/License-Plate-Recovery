"""
Shared geometry helpers used by the rest of the pipeline.

Plate *detection* itself is not done here -- it's the pretrained YOLOv9-t
detector wrapped in pretrained_models.py (see report/REPORT.md and
README.md). This module only holds the geometry utilities every other
stage needs on top of that detector's output: IoU scoring against
ground truth, and margin-padded cropping around a box.

(An earlier iteration of this project used classical contour-based
rectangle detection and a Haar cascade as the plate localizer, before
the pretrained detector replaced them in v2. That code has been removed
from this file since it's no longer part of the pipeline -- see
archive_v1_synthetic/ if you want to see what it looked like.)
"""


def iou(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union = aw * ah + bw * bh - inter_area
    return inter_area / union if union > 0 else 0.0


def crop_with_margin(img, bbox, margin_frac=0.12):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    mx, my = int(bw * margin_frac), int(bh * margin_frac)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    return img[y0:y1, x0:x1]
