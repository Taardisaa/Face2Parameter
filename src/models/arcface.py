"""ArcFace face-recognition embeddings (expression-invariant identity features).

Used as an expression-invariant backbone (see docs/expression-invariance.md). We run the
InsightFace `w600k_r50` ArcFace model (512-d) on onnxruntime — no heavy `insightface`
build — and align faces with mtcnn_ort 5 landmarks to the canonical ArcFace 112x112
template (same SimilarityTransform machinery as FaceCrop).

The weights live in InsightFace's `buffalo_l` pack; `arcface_onnx_path()` downloads and
extracts it once to a user cache dir.
"""

from __future__ import annotations

import os
import zipfile

import cv2
import numpy as np
import onnxruntime
from skimage import transform as trans

# Canonical ArcFace 5-point destination template for a 112x112 aligned crop
# (left eye, right eye, nose, left mouth, right mouth) — InsightFace standard.
ARCFACE_DST = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

_BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
_MODEL_IN_ZIP = "w600k_r50.onnx"


def _cache_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), ".cache", "face2param", "arcface")
    os.makedirs(d, exist_ok=True)
    return d


def arcface_onnx_path() -> str:
    """Return the local path to w600k_r50.onnx, downloading buffalo_l once if needed."""
    dst = os.path.join(_cache_dir(), _MODEL_IN_ZIP)
    if os.path.exists(dst):
        return dst

    import requests  # local import: only needed on first run

    zip_path = os.path.join(_cache_dir(), "buffalo_l.zip")
    if not os.path.exists(zip_path):
        print(f"[arcface] downloading buffalo_l (~288MB) -> {zip_path}")
        with requests.get(_BUFFALO_L_URL, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        member = next(n for n in z.namelist() if n.endswith(_MODEL_IN_ZIP))
        with z.open(member) as src, open(dst, "wb") as out:
            out.write(src.read())
    print(f"[arcface] extracted -> {dst}")
    return dst


def align_112(img_rgb: np.ndarray, landmark5: np.ndarray) -> np.ndarray:
    """Warp an RGB image to the 112x112 ArcFace template using 5 landmarks.

    landmark5: the raw mtcnn output array (reshaped to (5,2) the same way FaceCrop does).
    """
    lmk = landmark5.reshape(2, 5).T.astype(np.float32)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, ARCFACE_DST)
    M = tform.params[0:2, :]
    return cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0)


class ArcFaceONNX:
    """w600k_r50 ArcFace recognizer: aligned 112x112 RGB -> L2-normalized 512-d embedding."""

    def __init__(self, onnx_path: str | None = None, providers=None):
        onnx_path = onnx_path or arcface_onnx_path()
        if providers is None:
            # Prefer GPU (CUDA, else DirectML on Windows), fall back to CPU.
            avail = onnxruntime.get_available_providers()
            pref = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
            providers = [p for p in pref if p in avail] or avail
        self.sess = onnxruntime.InferenceSession(onnx_path, providers=providers)
        self.providers = self.sess.get_providers()
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        # The model is exported with a fixed batch=1; DirectML errors on batched input
        # (CPU/CUDA tolerate it). On DML we loop per image — still ~3.4ms/img on GPU.
        self._per_image = bool(self.providers) and self.providers[0] == "DmlExecutionProvider"

    def embed(self, aligned_rgb: np.ndarray) -> np.ndarray:
        """aligned_rgb: (N,112,112,3) or (112,112,3) uint8 RGB -> (N,512) float32, L2-normed."""
        if aligned_rgb.ndim == 3:
            aligned_rgb = aligned_rgb[None]
        # InsightFace recognizer: (x-127.5)/127.5 on RGB, NCHW.
        blob = (aligned_rgb.astype(np.float32) - 127.5) / 127.5
        blob = np.transpose(blob, (0, 3, 1, 2))
        if self._per_image:
            out = np.concatenate(
                [self.sess.run([self.output_name], {self.input_name: blob[i:i + 1]})[0]
                 for i in range(blob.shape[0])], axis=0)
        else:
            out = self.sess.run([self.output_name], {self.input_name: blob})[0]
        out = out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)
        return out.astype(np.float32)
