# Model combination comparison

Same 25 curated cases x 9 degradations + 1 clean pass, no enhancement applied (isolates the model-choice effect). CPU latency, single image at a time, after a warm-up call. See src/compare_models.py.

| Combination | N | Exact-match | Char-acc | Det. IoU | Detector (ms) | OCR (ms) | Total (ms/plate) | Throughput (plates/s) |
|---|---|---|---|---|---|---|---|---|
| current (yolo-v9-t-384 + cct-xs-v2) | 250 | 60% | 0.71 | 0.77 | 30.9 | 4.1 | 35.0 | 28.6 |
| best-available (yolo-v9-s-608 + cct-s-v2) | 250 | 64% | 0.72 | 0.79 | 233.4 | 37.2 | 270.6 | 3.7 |
