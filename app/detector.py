import os
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from ultralytics import YOLO

register_heif_opener()

_model: YOLO | None = None

TARGET_BRIGHTNESS = 150
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def load_model(model_path: str) -> None:
    global _model
    _model = YOLO(model_path)


def get_model() -> YOLO:
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


def _normalize_brightness(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    if mean_brightness == 0:
        return img
    factor = TARGET_BRIGHTNESS / mean_brightness
    return np.clip(img * factor, 0, 255).astype(np.uint8)


def analyze_image(image_bytes: bytes, suffix: str, confidence_threshold: float) -> dict:
    suffix = suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        pil_img = Image.open(tmp_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img = _normalize_brightness(img)

        start = time.perf_counter()
        results = get_model().predict(img, conf=confidence_threshold, verbose=False)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        boxes = results[0].boxes
        detections = []
        confidences = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "confidence": conf,
            })
            confidences.append(conf)

        return {
            "varroa_count": len(detections),
            "confidence_mean": round(float(np.mean(confidences)), 4) if confidences else 0.0,
            "detections": detections,
            "processing_time_ms": elapsed_ms,
        }
    finally:
        os.unlink(tmp_path)
