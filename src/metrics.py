"""Quantitative evaluation metrics.

- PSNR / SSIM: image-quality similarity of enhanced output vs the known
  clean ground-truth plate crop. Chosen because they are the standard,
  widely-reported metrics for image restoration tasks (PSNR captures pixel-
  level fidelity / noise suppression, SSIM captures perceptual structural
  similarity, which correlates better with human-judged readability).
- OCR character accuracy: 1 - (normalized Levenshtein edit distance)
  between recovered text and ground-truth text. Chosen because the end
  goal is a readable plate string, not just a visually clean image, so a
  text-level metric is required alongside the pixel-level ones.
"""
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def _match_size(a, b):
    h, w = b.shape[:2]
    return cv2.resize(a, (w, h))


def compute_psnr_ssim(enhanced, ground_truth):
    enhanced_r = _match_size(enhanced, ground_truth)
    gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    en_gray = cv2.cvtColor(enhanced_r, cv2.COLOR_BGR2GRAY)
    p = psnr(gt_gray, en_gray, data_range=255)
    s = ssim(gt_gray, en_gray, data_range=255)
    return float(p), float(s)


def levenshtein(a, b):
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def char_accuracy(recovered_text, gt_text):
    if len(gt_text) == 0:
        return 1.0 if len(recovered_text) == 0 else 0.0
    dist = levenshtein(recovered_text, gt_text)
    return max(0.0, 1.0 - dist / len(gt_text))
