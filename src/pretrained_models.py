"""
Thin, cached wrapper around the pretrained fast-alpr components so the
rest of the code can call a plain function instead of juggling model
objects. Both models are used purely for inference -- no training or
fine-tuning happens anywhere in this project.

  - Detector: YOLOv9-t (384) license-plate detector, MIT-licensed,
    trained across 65+ countries' plate formats.
    https://github.com/ankandrew/open-image-models
  - OCR: CCT-XS v2 "global" plate text recognizer.
    https://github.com/ankandrew/fast-plate-ocr
Both ship as small ONNX files (~7MB / ~3MB) and run on CPU in
real time -- no GPU is required for this project.
"""
from open_image_models import create_detector
from fast_plate_ocr import LicensePlateRecognizer

_detector = None
_ocr = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = create_detector("yolo-v9-t-384-license-plate-end2end")
    return _detector


def get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = LicensePlateRecognizer("cct-xs-v2-global-model")
    return _ocr


def detect_plates(img):
    """Returns a list of (x, y, w, h, confidence) in xywh format."""
    dets = get_detector().predict(img)
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.bounding_box.x1, d.bounding_box.y1, d.bounding_box.x2, d.bounding_box.y2
        out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(d.confidence)))
    return out


def read_plate_text(crop_bgr):
    """Returns (text, mean_char_confidence 0-1) for a single plate crop."""
    results = get_ocr().run(crop_bgr, return_confidence=True)
    if not results:
        return "", 0.0
    pred = results[0]
    text = pred.plate or ""
    conf = float(sum(pred.char_probs) / len(pred.char_probs)) if len(pred.char_probs) else 0.0
    return text, conf
