"""
License-plate region detection.

Primary method: classical contour-based rectangle detection (edge map ->
contours -> filter by aspect ratio / area / rectangularity). This is used
as the primary detector because it is explainable, needs no training data,
and — as shown in the report — generalizes reasonably to plate-like
rectangles without being tied to one country's plate design.

Secondary method: OpenCV's bundled Haar cascade, included for comparison.
Both are evaluated in the report; the Haar cascade is shown to perform
worse on our (partly non-Russian-style) test scenes, which is discussed
as a failure case.
"""
import cv2
import numpy as np


def detect_candidates_contour(img, min_area_frac=0.001, max_area_frac=0.08):
    h, w = img.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * img_area or area > max_area_frac * img_area:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / float(ch)
        if 2.0 <= aspect <= 6.0:  # plates are wide rectangles
            rect_fill = area / float(cw * ch)
            if rect_fill > 0.5:
                candidates.append((x, y, cw, ch, rect_fill))

    # Highest rectangularity first
    candidates.sort(key=lambda t: -t[4])
    return [(x, y, cw, ch) for (x, y, cw, ch, _) in candidates]


def detect_candidates_haar(img, cascade_path=None):
    if cascade_path is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 12))
    return [tuple(b) for b in boxes]


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


def best_candidate_by_gt(candidates, gt_bbox):
    """Pick the candidate with highest IoU vs ground truth (evaluation only —
    a real deployment wouldn't have access to ground truth, see notes)."""
    if not candidates:
        return None, 0.0
    scored = [(c, iou(c, gt_bbox)) for c in candidates]
    scored.sort(key=lambda t: -t[1])
    return scored[0]


def crop_with_margin(img, bbox, margin_frac=0.12):
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    mx, my = int(bw * margin_frac), int(bh * margin_frac)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + bw + mx), min(h, y + bh + my)
    return img[y0:y1, x0:x1]
