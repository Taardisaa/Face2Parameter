"""Shared image loading for inference (predict.py / infer.py).

Centralizes the preprocessing so both entry points behave identically:
  - tolerate non-ASCII paths (imdecode), grayscale, and RGBA (drop alpha)
  - optionally mtcnn-align the face (matches the training crop); if that's
    unavailable or finds no face, fall back to an aspect-preserving center crop
    (NEVER a plain stretch-to-square, which distorts the face)
  - resize to img_size
"""

from __future__ import annotations

import cv2
import numpy as np


def center_square(img: np.ndarray) -> np.ndarray:
    """Crop the central square so a later resize doesn't distort aspect ratio."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


def load_face_rgb(path: str, img_size: int, use_detector: bool = True) -> np.ndarray:
    """Return an (img_size, img_size, 3) RGB uint8 face crop ready for the backbone."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img is None:
        raise RuntimeError(f"could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)  # drop alpha if present

    crop = None
    if use_detector:
        try:
            from src.face_data_utils.FaceCrop import FaceCrop  # needs mtcnn-onnxruntime
            faces = FaceCrop().crop(img)
            if faces and np.ndim(faces[0]) == 3:
                crop = faces[0]
            else:
                print("[img] no face detected; falling back to center-crop")
        except Exception as exc:  # noqa: BLE001
            print(f"[img] detector unavailable ({type(exc).__name__}); center-crop fallback")
    img = crop if crop is not None else center_square(img)

    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)
