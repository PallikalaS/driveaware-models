"""Crops used for Model A input.

* center crop          – the naive baseline.
* dynamic upper-body   – box derived from the face, extended down to the lap so
                         a phone held low (texting) and both hands stay in view.

The Android app MUST use the same geometry (see android/.../vision/Crop.kt).
If you change a factor here, change it there too and note it in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Geometry, in units of face-box width (w) / height (h). Tuned on Wed by
# Mahitha with crop_compare.py; keep in sync with Crop.kt.
WIDTH_FACTOR = 4.0      # crop width  = 4.0 * face width
TOP_FACTOR = 0.6        # starts 0.6 * face height above the face top
BOTTOM_FACTOR = 4.5     # ends 4.5 * face height below the face top (lap)
OUT_SIZE = 224


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


class FaceFinder:
    """MediaPipe face detector if a model file is given, else OpenCV Haar cascade."""

    def __init__(self, mediapipe_model: str | None = None):
        self._mp = None
        if mediapipe_model and Path(mediapipe_model).exists():
            import mediapipe as mp
            from mediapipe.tasks.python import vision, BaseOptions
            opts = vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=mediapipe_model),
                min_detection_confidence=0.5)
            self._mp = vision.FaceDetector.create_from_options(opts)
            self._mp_image = mp.Image
            self._mp_format = mp.ImageFormat.SRGB
        else:
            path = Path(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "") \
                / "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(str(path)) if path.exists() else None
            if self._haar is None or self._haar.empty():
                self._haar = None
                print("! No face detector available (pass --face-model blaze_face_short_range.tflite). "
                      "Dynamic crop will fall back to center crop.")

    def find(self, bgr: np.ndarray) -> Box | None:
        if self._mp is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = self._mp.detect(self._mp_image(image_format=self._mp_format, data=rgb))
            if not res.detections:
                return None
            b = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
            return Box(b.origin_x, b.origin_y, b.width, b.height)
        if self._haar is None:
            return None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return Box(int(x), int(y), int(w), int(h))


def center_crop(img: np.ndarray, size: int = OUT_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y0:y0 + s, x0:x0 + s], (size, size), interpolation=cv2.INTER_AREA)


def upper_body_box(face: Box, img_w: int, img_h: int) -> Box:
    cx = face.x + face.w / 2
    bw = WIDTH_FACTOR * face.w
    x0 = int(max(0, cx - bw / 2))
    x1 = int(min(img_w, cx + bw / 2))
    y0 = int(max(0, face.y - TOP_FACTOR * face.h))
    y1 = int(min(img_h, face.y + BOTTOM_FACTOR * face.h))
    return Box(x0, y0, x1 - x0, y1 - y0)


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def dynamic_crop(img: np.ndarray, face: Box | None, size: int = OUT_SIZE) -> tuple[np.ndarray, bool]:
    """Returns (crop, used_face). Falls back to center crop when no face is found."""
    if face is None:
        return center_crop(img, size), False
    b = upper_body_box(face, img.shape[1], img.shape[0])
    roi = img[b.y:b.y + b.h, b.x:b.x + b.w]
    return cv2.resize(pad_to_square(roi), (size, size), interpolation=cv2.INTER_AREA), True


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian; low = blurry."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def dhash(img: np.ndarray, size: int = 8) -> int:
    """Difference hash for near-duplicate removal."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size))
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
