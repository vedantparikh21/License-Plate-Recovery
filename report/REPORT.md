# License Plate Recovery — Report (v2: real data + pretrained models)

## 1. What changed from the first iteration, and why

The first version of this pipeline used synthetically rendered plates
composited onto photo backgrounds, and generic Tesseract OCR. That was a
reasonable way to get exact ground truth quickly, but it wasn't a real
dataset, and it understated how good the *readable image* recovery
actually was, because Tesseract itself was the bottleneck rather than
the enhancement pipeline.

This version fixes both:

- **Real dataset**: [`openalpr/benchmarks`](https://github.com/openalpr/benchmarks),
  the official OpenALPR benchmark corpus (also used in NVIDIA's own ALPR
  fine-tuning tutorials). 444 real CCTV/dashcam photographs across US,
  EU, and Brazilian plates, each with a human-annotated bounding box and
  ground-truth plate string. I pulled this directly from GitHub — no
  upload needed.
- **Pretrained detection + OCR**: [`fast-alpr`](https://github.com/ankandrew/fast-alpr)
  (YOLOv9-t plate detector, MIT-licensed, trained across 65+ countries)
  and [`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr)
  (a CCT-based plate text recognizer). Both are small (~10MB total)
  pretrained ONNX models. **Nothing in this project is trained or
  fine-tuned** — they're used purely for inference, exactly as
  intended: the assignment's own framing ("apply enhancement, detect,
  extract text, evaluate") assumes a working detector and recognizer as
  given, and building a competitive one from scratch would be its own
  multi-week project, not a 48-hour one.
- Everything runs on **CPU in real time** (~1 second/image) — no GPU,
  no laptop hand-off, no Kaggle needed anywhere in this pipeline.

## 2. Approach

### 2.1 Dataset

25 cases were curated from the benchmark's `endtoend/us`, `endtoend/eu`,
and `endtoend/br` subsets (12 / 6 / 6, plus one deliberately-kept hard
case — see below), using `src/curate_real_dataset.py`. Selection
criterion: for each candidate, the pretrained detector was run on the
**clean, undegraded** image first, and cases were kept if it found the
annotated plate with IoU ≥ 0.5 against the human-labeled ground truth —
this keeps the primary evaluation table focused on genuine, working
real-world cases rather than cases where the pretrained detector itself
is fundamentally unable to localize the plate at all. One clear
exception was deliberately kept: a case where the detector scores 0.0
IoU even with zero degradation applied, specifically so the failure-case
discussion in §6 has a genuine example rather than a hypothetical one.

Every case has: the real photo, a human-annotated bounding box, and the
real plate string — all three from the dataset, not synthesized.

### 2.2 Pipeline stages

| Stage | Method | Trained here? |
|---|---|---|
| Detection | YOLOv9-t (384), `fast-alpr`/`open-image-models` | No — pretrained, inference only |
| Deblur (motion) | Wiener-style frequency-domain deconvolution | Ours |
| Deblur (defocus) | Unsharp masking | Ours |
| Denoising | `fastNlMeansDenoisingColored` | Ours |
| Contrast/exposure | CLAHE | Ours |
| Perspective correction | Largest-quadrilateral detection + 4-point projective warp | Ours |
| Super-resolution | FSRCNN (pretrained, ~40KB), via OpenCV `dnn_superres` | Pretrained, inference only |
| Video stabilization | ECC-based per-frame alignment + temporal averaging | Ours |
| OCR | CCT-XS v2 "global" model, `fast-plate-ocr` | No — pretrained, inference only |

**Pipeline order** (see `src/pipeline_real.py`): for every case x
degradation, the real photo is degraded, then run through the pretrained
detector+OCR directly (**baseline**, i.e. no enhancement at all), and
separately through our scene-level correction → re-run the pretrained
detector on the corrected frame → crop → super-resolve the crop → run
the pretrained OCR again (**enhanced**). Perspective correction is the
one exception: it operates on an already-isolated crop rather than the
full scene, and doesn't trigger a full re-detection pass, since undoing
a projective warp needs the plate's quadrilateral isolated first (see
§6 for why this matters less than expected here).

### 2.3 Metrics (and why)

Same justification as before, now measured against **real** photos:

- **Detection IoU** against the human-annotated box — measures the
  detector itself, independent of OCR.
- **PSNR / SSIM** of the recovered crop against a same-geometry crop of
  the real, undegraded photo (not a synthetic "ideal" reference, since
  none exists for real images) — isolates degradation-removal quality.
- **OCR exact-match rate and character accuracy** (`1 − normalized
  Levenshtein distance`) — the actual deliverable is readable text, and
  as in the first iteration, PSNR/SSIM and OCR accuracy sometimes
  disagree (see §4), which is itself a useful finding, not noise to
  average away.

## 3. Per-input outputs

233 test instances (25 real cases × 9 degradations, + an 8-case video/
multi-frame test) each produce: the detected/degraded plate crop, the
enhanced crop, and the recovered text, scored against real ground truth.
Full data is in `data/results_real/results.csv`. Six curated,
labeled before/after panels are in `data/results_real/panels/`, chosen
to show a clear win, an honest regression, a genuine failure, a
compounded-degradation recovery, and the video/multi-frame case — not
just the best-looking outcomes.

## 4. Quantitative results

| Degradation | Det. IoU before→after | Exact-match before→after | Char-acc before→after | SSIM before→after | PSNR before→after |
|---|---|---|---|---|---|
| Motion blur | 0.80→0.83 | 8%→**64%** | 0.33→**0.90** | 0.48→0.30 | 20.5→13.9 |
| Defocus blur | 0.71→0.76 | 20%→20% | 0.27→0.30 | 0.34→0.29 | 19.2→15.2 |
| Low-resolution (5x) | 0.78→0.78 | 16%→20% | 0.31→0.33 | 0.51→0.26 | 19.1→13.8 |
| Gaussian noise | 0.78→0.79 | **88%→76%** | 0.91→0.92 | 0.73→0.20 | 25.0→12.4 |
| JPEG compression | 0.83→0.81 | 88%→88% | 0.95→0.95 | 0.88→0.29 | 28.2→13.2 |
| Under-exposure | 0.84→0.84 | 88%→92% | 0.95→0.99 | 0.44→0.29 | 11.2→14.5 |
| Over-exposure | 0.82→0.83 | **96%→92%** | 0.96→0.99 | 0.72→0.19 | 8.7→8.8 |
| Perspective skew | 0.58→0.58 | 92%→92% | 0.95→0.99 | 0.14→0.26 | 13.9→12.6 |
| Combined (hard) | 0.74→0.71 | 12%→**28%** | 0.32→0.39 | 0.39→0.18 | 19.7→12.9 |
| Video / multi-frame (n=8) | 0.88→0.87 | 88%→**100%** | 0.98→1.00 | 0.15→0.13 | 16.0→12.4 |

(Bold = the numbers worth reading twice — two clear wins and one honest
regression. All 25 cases per row except video, n=8. Full data:
`data/results_real/results.csv`.)

**Per-region breakdown** (averaged across all degradations): US 49%→53%
exact-match, EU 56%→63%, Brazil 80%→94%. Brazilian plates in this
benchmark are photographed larger/closer (median plate width ~260px vs
US ~88px), which tracks with them being the most robust to degradation
— a useful reminder that "plate legibility" is as much about camera
placement and resolution budget as about any enhancement algorithm.

### Headline finding: the pretrained OCR is already strong, and that changes the story

Unlike the first iteration (where Tesseract capped out around 60–95%
even on clean crops), this pretrained CCT-OCR model starts from a much
higher baseline — 88–96% exact-match on noise, JPEG, exposure, and
perspective degradations **with no enhancement applied at all**. That
completely changes what "success" looks like for this project:

- **Where the model isn't already robust — motion blur and compounded
  degradation — enhancement earns its keep decisively**: 8%→64% and
  12%→28% exact-match respectively. This is the clearest evidence the
  enhancement pipeline does real work.
- **Where the model is already strong — noise and over-exposure — our
  classical enhancement sometimes makes things slightly *worse*
  (88%→76%, 96%→92%)**. This is a genuine, not-hidden finding: denoising
  and CLAHE introduce their own mild artifacts (visible ringing,
  contrast shifts), and a recognizer that's already trained to be
  robust to sensor noise doesn't need — and can be mildly hurt by —
  additional hand-crafted preprocessing. The right lesson isn't "the
  enhancement pipeline failed"; it's that **enhancement should be
  applied conditionally**, based on measured degradation severity, not
  unconditionally on every frame. A production system would benefit
  from a lightweight degradation classifier gating which correction (if
  any) gets applied — noted in §8.
- **Perspective correction barely moves the needle here (92%→92%)**,
  in sharp contrast to the first iteration where it was essential
  (Tesseract went from failing outright to an exact match after
  correction). The modern detector+OCR model was clearly trained with
  enough viewpoint augmentation to already handle moderate skew — so a
  hand-built geometric correction step adds the least value precisely
  where the underlying model is most capable. This is a genuinely useful
  thing to learn from the exercise: **how much a custom enhancement
  stage helps depends heavily on what the downstream model was already
  trained to tolerate**, not just on how severe the degradation looks to
  a human eye.
- **PSNR/SSIM still diverge from OCR accuracy, more sharply than in the
  first iteration.** SSIM drops after enhancement in almost every row,
  even where OCR accuracy clearly improves. The main driver:
  super-resolution changes the crop's scale substantially (3x upscale on
  top of already crop-and-margin adjustments), so the geometry rarely
  matches the fixed-size reference crop as tightly as the unenhanced
  version does, even when the *content* is far more legible. This
  reinforces the same point from the first report even more strongly:
  a pixel-similarity metric and a task-relevant metric (OCR accuracy)
  are not interchangeable, and optimizing for one without checking the
  other would be a mistake.

## 5. Model choice: current vs. best-available combination

The model Q&A from the last interview round covered *why*
YOLOv9-t-384 and CCT-XS-v2 were chosen. This section answers the
natural follow-up: what would swapping in the strongest currently
available combination in the same two libraries actually buy, and at
what cost?

- **Current**: YOLOv9-t-384 detector + CCT-XS-v2 OCR (used everywhere
  else in this project).
- **Best-available**: YOLOv9-s-608 detector — the highest published
  mAP50 (0.966) in `open-image-models`' own plate-detection model
  table — + CCT-S-v2 OCR — the `fast-plate-ocr` maintainer's
  now-recommended default for new integrations.

`src/compare_models.py` runs both combinations on the exact same 25
curated cases × 9 degradations + 1 clean pass (250 instances per
combination), with **no enhancement applied**, so the comparison
isolates the model-choice effect from the enhancement pipeline, which
is a separate variable already measured in full in §4. It times the
detector forward pass and the OCR forward pass separately, on the same
CPU this whole project runs on, after a warm-up call so model-load
time doesn't pollute the measurement.

| Combination | Exact-match | Char-acc | Det. IoU | Detector (ms) | OCR (ms) | Total (ms/plate) | Throughput (plates/s) |
|---|---|---|---|---|---|---|---|
| Current (YOLOv9-t-384 + CCT-XS-v2) | 60% | 0.71 | 0.77 | 38.3 | 3.8 | 42.1 | 23.8 |
| Best-available (YOLOv9-s-608 + CCT-S-v2) | 64% | 0.72 | 0.79 | 223.2 | 25.8 | 248.9 | 4.0 |

(Full per-instance data: `data/results_real/model_comparison.csv`;
aggregated table: `data/results_real/model_comparison_summary.md`;
chart: `data/results_real/model_comparison_chart.png`.)

**Reading it honestly**: the bigger combination is better, but only
modestly — +4 points of exact-match and +2 points of detection IoU —
while costing roughly **6x the latency** (42ms → 249ms per plate) and
cutting CPU throughput from ~24 plates/sec to ~4 plates/sec. Character
accuracy barely moves (0.71 → 0.72), meaning most of the gap between
the two OCR models shows up as full-string exact-match on a handful of
already-close cases, not a broad accuracy improvement. For this
project's stated goal — a CPU, real-time-capable pipeline — the
current combination's accuracy-per-millisecond is clearly the better
trade, and the current choice holds up under this comparison rather
than being an accuracy compromise I got wrong. The bigger combination
would only be worth it for a deployment that can tolerate ~250ms/plate
and genuinely needs the last few points of exact-match (e.g. a
low-throughput forensic-review tool rather than a live camera feed).

## 6. Failure cases and why

- **Defocus blur remains the hardest degradation** (20%→20%, completely
  flat). As before: it's a symmetric, non-directional blur with no
  kernel orientation to exploit, so unsharp masking is a weak tool
  against it. A blind-deconvolution approach that estimates the defocus
  radius from the image itself would be the natural next step (§8).
- **Detection can fail completely and permanently on atypical vehicles.**
  The one deliberately-kept hard case (`us_12c6cb72...`) is a service
  utility truck with a small, low-contrast plate mounted on a cluttered,
  sign-covered truck bed — the pretrained detector scores 0.0 IoU at
  *every* degradation level, including zero degradation. See
  `data/results_real/panels/hard_case_full_scene_detection.png`. This
  is a real, generalizable limitation: a plate detector trained mostly
  on typical passenger-car bumper shots will struggle on non-standard
  vehicle types and backgrounds, no matter how good the image quality
  is. Enhancement cannot fix a detection failure that isn't about image
  quality in the first place.
- **Even when detection totally fails, the OCR-fallback design still
  nearly recovers the text.** For that same hard case, once a crop is
  supplied via the ground-truth box (which required knowing where to
  look — not something a real deployment has), the OCR model reads
  `OSG719` against a true `0SG719` — a single, classic 0/O confusion,
  everything else correct. This cleanly separates two different
  failure modes that are easy to conflate: "the detector can't find the
  plate" is a very different, and often harder, problem than "the
  recognizer can't read the plate" — and this project's results show
  the second problem is largely solved by the pretrained model already,
  while the first is not.
- **Enhancement is not a free action — it has a cost when applied to
  cases that didn't need it** (see the noise/over-exposure regressions
  in §4). Any production integration of this pipeline should gate
  enhancement behind a cheap degradation-severity estimate rather than
  always applying the full correction stack.
- **Low-resolution recovery is still limited by the super-resolution
  model's design scale.** As in the first iteration, the FSRCNN model
  here is a lightweight 3x model; a 5x downsampling degradation exceeds
  its native range, so some information is genuinely unrecoverable at
  this model size.

## 7. Live demo

`src/live_demo.py` runs the pipeline interactively on a file or webcam
snapshot — useful for a live walkthrough rather than the offline batch
evaluation above. Since a live input's degradation type is unknown (unlike
the controlled experiments in §2–§4), it uses an **adaptive** pipeline
instead of a fixed per-degradation recipe: it measures blur/noise/exposure
on the actual input (`enhance.enhance_adaptive`) and only applies
corrections the measurement calls for, then searches several candidate
deblur corrections (a small set of deconvolution angles, unsharp masking,
denoise+contrast) and keeps whichever the OCR model itself is most
confident about — a pragmatic stand-in for true blind deconvolution.
This design is a direct response to the §4/§6 finding that
unconditional enhancement can hurt an already-good input.

## 8. What I'd do with more time

- Add a lightweight degradation classifier (blur amount, noise level,
  exposure histogram) to decide *whether* and *which* enhancement stage
  to apply per frame, directly addressing the "enhancement sometimes
  hurts an already-good input" finding in §4.
- Swap the fixed-parameter unsharp mask for a blind-deconvolution
  approach for defocus blur, estimating the blur radius from the image
  rather than assuming a fixed value.
- Extend the curated set beyond 25 cases (444 are available in the
  benchmark) for tighter confidence intervals on the per-degradation
  numbers, and add a fine-tuned detector specifically on hard vehicle
  types (utility trucks, motorcycles) to address the detection failure
  in §6 — this would be the one place in the whole project where
  fine-tuning a model would plausibly be worth the extra time.