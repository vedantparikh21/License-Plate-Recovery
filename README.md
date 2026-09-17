# License Plate Recovery — Forensic Enhancement Pipeline (real data + pretrained models)

A pipeline that detects a license plate in an image, corrects for blur,
perspective distortion, low resolution, noise, exposure issues, and
camera shake, then recovers the plate text — using a **real, publicly
available benchmark dataset** and **off-the-shelf pretrained models**
for detection and OCR (no training or fine-tuning anywhere in this
project).

See `report/REPORT.md` for the full write-up.
<!-- This is the second iteration of this project — an earlier version used a synthetic,
composited dataset and classical Tesseract OCR; that's preserved in
`archive_v1_synthetic/` for reference, but superseded by everything
below. -->

## Getting started:

If you're setting this up on a new machine for the first time (you must have python installed on your computer to run this project):

```powershell
# 1. Clone this repo
git clone https://github.com/vedantparikh21/License-Plate-Recovery.git
cd License-Plate-Recovery

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

That's the entire dependency list — no GPU, no other setup. From here,
jump to **Quick start** below to try the live demo immediately, or
**How to run** further down to reproduce the full offline evaluation.

## Quick start: 

`src/live_demo.py` is the main interactive entry point — point it at
any image (or your webcam) and it runs the full pipeline live: detects
the plate, shows plain OCR vs. enhanced OCR side by side, and saves the
result. This is the fastest way to see the whole thing work end to end
without running the full offline evaluation below.

**Note**: you would need to have image(s) of License Plate for testing out this file!.

```powershell
python src/live_demo.py
```

A handful of pre-generated example outputs from this are in
`data/live_demo_outputs/` if you want to see results without running it
yourself. Full usage details (folder mode, webcam mode) are further
down in this README under **Live demo**.

The rest of this document covers the full, reproducible offline
evaluation (25 real photos × 9 degradations, quantitative metrics,
before/after galleries) that this live demo is built on top of.

## Why this version is different

- **Dataset**: [`openalpr/benchmarks`](https://github.com/openalpr/benchmarks)
  — the official OpenALPR benchmark (also used in NVIDIA's own ALPR
  fine-tuning tutorials). 444 real CCTV/dashcam photos across US, EU,
  and Brazilian plates, each with a human-annotated bounding box and
  ground-truth plate string. 25 of these were curated for this project
  (see `src/curate_real_dataset.py`), spanning all three regions.
- **Detection + OCR**: [`fast-alpr`](https://github.com/ankandrew/fast-alpr)
  — a pretrained YOLOv9-t (384) license-plate detector (MIT license,
  trained across 65+ countries) plus a CCT-based OCR model
  ([`fast-plate-ocr`](https://github.com/ankandrew/fast-plate-ocr)).
  Both are ~10MB ONNX models, run on CPU in ~1 second/image, and are
  used purely for **inference** — nothing is trained or fine-tuned.
- **Our own contribution**: the image-enhancement layer in between —
  Wiener deconvolution, denoising, CLAHE, auto perspective correction,
  and lightweight learned super-resolution (FSRCNN) — applied to
  degraded frames before re-running the pretrained models, to measure
  whether enhancement actually helps a strong, already-trained
  recognizer, and where it doesn't.

**No GPU needed anywhere in this pipeline** — everything above runs on
CPU. There's nothing here that needs to be handed off to a laptop or to
Kaggle.

## Project layout

```
src/
  curate_real_dataset.py   Pulls + filters real cases from the OpenALPR benchmark
  pretrained_models.py     Thin wrapper around the pretrained detector + OCR
  degrade.py               Degradation simulators (blur, noise, perspective, ...)
  enhance.py                Enhancement pipeline (deblur, deskew, SR, denoise, CLAHE)
  metrics.py                PSNR / SSIM / character-accuracy scoring
  detect.py                 Shared geometry helpers (IoU, margin-cropping)
  pipeline_real.py           End-to-end experiment runner (25 cases x 9 degradations + video)
  compare_models.py         Current vs. best-available detector+OCR combo: accuracy AND latency
  make_panels_real.py       Before/after comparison panels
data/
  real_dataset/              25 curated real photos + manifest.json (bbox + ground-truth text)
  models/                    Pretrained FSRCNN super-resolution model (our enhancement stage)
  results_real/              All pipeline outputs (see below)
report/
  REPORT.md                  Full written report
archive_v1_synthetic/        Earlier synthetic-dataset iteration (superseded, kept for reference)
```

## How to run

Clone the benchmark dataset repo into the project root first (needs
internet access once):

```powershell
cd "path\to\this\project"
git clone https://github.com/openalpr/benchmarks benchmarks-master
```

All paths in the code are computed automatically relative to the
project folder — nothing needs manual editing, on Windows or Linux.

Then, from your venv, in order:

```powershell
# 1. Curate the real dataset from the OpenALPR benchmark repo
#    (picks 25 diverse real cases, sanity-checks that the pretrained
#    detector can find each plate on the CLEAN image first)
python src/curate_real_dataset.py

# 2. Run the full experiment: 25 real cases x 9 degradations
#    (motion blur, defocus blur, low-res, noise, JPEG compression,
#    under/over-exposure, perspective skew, a combined "hard" case)
#    plus an 8-case video/camera-shake + multi-frame-averaging test
python src/pipeline_real.py

# 3. Build the before/after comparison panels (6 curated highlight cases)
python src/make_panels_real.py

# 4. Build the complete "input -> detected region -> enhanced -> text"
#    gallery, one case per degradation type (10 total) -- this is the
#    single document to open to see the full qualitative pipeline output
python src/make_full_case_gallery.py

# 5. Optional: compare the current detector+OCR combo against the
#    strongest currently-available one in the same libraries, on
#    accuracy AND CPU latency (see report §5 for the results and how
#    to read them)
python src/compare_models.py
```

Step 1 needs internet access once (to clone the benchmark repo). Step 2
also needs internet access the *first* time you run it, because
`fast-alpr` auto-downloads its two pretrained ONNX models on first use.

### Where the pretrained detection/OCR models actually live

They are **not** stored in this repo's `data/models/` folder (that
folder only holds the small FSRCNN super-resolution model, which is
*our* enhancement stage). The detector and OCR model are downloaded
automatically by the `fast-alpr`/`open-image-models`/`fast-plate-ocr`
packages the first time `pretrained_models.py` is used, and cached at:

- Windows: `C:\Users\<you>\.cache\open-image-models\` and `C:\Users\<you>\.cache\fast-plate-ocr\`
- Linux/Mac: `~/.cache/open-image-models/` and `~/.cache/fast-plate-ocr/`

You'll see a one-time download progress bar (~11MB total) the first
time you run `pipeline_real.py` or anything else that calls
`pretrained_models.get_detector()` / `get_ocr()`. After that it's
cached and loads instantly.

## Where to look for each deliverable

- **Detected region, enhanced image, recovered text — per input**:
  `data/results_real/BEFORE_AFTER_GALLERY.md` (open this first) plus
  the images it embeds in `data/results_real/full_case_panels/` — one
  complete input→detection→enhancement→text panel per degradation type.
  The raw per-instance files for all 233 test cases are also in
  `data/results_real/` as `<case>_<degradation>_{1_degraded_crop,
  2_enhanced_crop, 3_clean_reference_crop}.png`.
- **Quantitative evaluation (PSNR/SSIM/OCR accuracy)**:
  `data/results_real/results.csv` (every one of the 233 instances,
  every metric) and `data/results_real/summary.md` (the aggregated
  table, same one quoted in the report).
- **Qualitative before/after comparisons**: both
  `data/results_real/panels/` (6 curated highlight cases — a clear win,
  an honest regression, a genuine failure, etc.) and the
  `full_case_panels/` gallery above.
- **Live, real-time examples** (not part of the offline evaluation, but
  useful to look at): `data/live_demo_outputs/` — handpicked screenshots
  from running `src/live_demo.py` directly, showing the same
  detect→enhance→OCR flow on arbitrary images rather than the curated
  dataset.
- **Written analysis, metric justification, failure-case discussion**:
  `report/REPORT.md`.
- **Model choice comparison (current vs. best-available, accuracy vs.
  latency)**: `report/REPORT.md` §5, backed by
  `data/results_real/model_comparison_summary.md`,
  `model_comparison.csv`, and `model_comparison_chart.png`.

## Live demo

`src/live_demo.py` runs the pipeline interactively — good for showing
someone the pipeline working on a real image on the spot, rather than
the offline batch evaluation above.

```powershell
python src/live_demo.py                    # interactive menu: file, folder, or webcam
python src/live_demo.py path\to\image.jpg  # process one image directly
```

- **File mode**: pick any image; it shows detected region, plain OCR,
  and enhanced OCR side by side in a popup window (share that window on
  a call), and also saves the panel to `data/live_demo_outputs/`.
- **Folder mode**: point it at a directory and it processes every image
  in it, saving one result panel per image and printing a summary table.
- **Webcam mode**: SPACE captures the current frame and runs the
  pipeline on it, ESC quits.

Unlike the offline evaluation, this uses an **adaptive** pipeline
(`enhance.enhance_adaptive` + a best-of-N deblur search in
`live_demo.try_multiple_enhancements`) since a live input's degradation
type isn't known in advance — see report §6.

## Design notes / where I made judgment calls

- **Why controlled degradation on real photos, rather than a dataset of
  already-degraded plates.** No public dataset pairs a real clean plate
  photo with a matched, deliberately-degraded version *and* ground-truth
  text for scoring — that pairing is exactly what's needed to compute
  PSNR/SSIM and OCR accuracy quantitatively. So the base photos and their
  ground truth are 100% real (from the OpenALPR benchmark); only the
  degradation is synthesized on top of them, with full control over type
  and severity for the evaluation. This is different from the first
  iteration, where the *underlying plate images themselves* were
  synthetic composites — here they're genuine photographs.
- **Detection is evaluated two ways.** Every case's ground-truth box lets
  us measure real, standalone detector IoU before and after enhancement
  (not just "did OCR happen to work"). One curated case is a genuine
  detector failure (IoU 0.0 at every degradation level, including zero
  degradation) — kept deliberately rather than filtered out, because a
  dataset that only contains cases the model already handles well isn't
  a real evaluation.
- **Perspective correction is applied at the crop level, not the full
  scene**, and does not trigger a full re-detection pass (see
  `pipeline_real.py`) — explained in more detail in the report, along
  with why this matters less than expected once you're evaluating a
  strong pretrained OCR model rather than generic Tesseract.
- **PSNR/SSIM are computed against a same-geometry crop of the real,
  undegraded photo** at the ground-truth box (not a synthetic "ideal"
  reference, since none exists for real photos) — this isolates
  degradation-removal quality from detector localization noise.
- **Detector/OCR model choice was re-checked, not just asserted.**
  `src/compare_models.py` runs the current combination (YOLOv9-t-384 +
  CCT-XS-v2) against the strongest currently-available one in the same
  libraries (YOLOv9-s-608 + CCT-S-v2) on the same 25 cases, with
  latency measured alongside accuracy. The bigger combination is only
  modestly more accurate (+4 points exact-match) for ~6x the per-plate
  latency — see report §5 for the full numbers and reasoning.
