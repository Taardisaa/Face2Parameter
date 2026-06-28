"""Optional expression neutralizer: relax a smiling face to neutral before Stage 1.

When every available photo of a subject is smiling, multi-image aggregation can't cancel the
(systematic) smile and it leaks into the predicted geometry. This neutralizer edits the input
*image* toward a neutral expression first; the result then feeds the existing backbone+head
unchanged (see docs/expression-invariance.md).

Method = **delta-zeroing** (driver-free), implemented in scripts/lp_neutralize.py: take the
subject's OWN LivePortrait keypoints and zero the expression deviation `exp` while keeping their
pose/scale/translation, so *their* face relaxes to neutral. This avoids the cross-identity warping
that a driver-image transfer produces. LivePortrait runs in THIS venv (py3.13/torch2.11/numpy2
verified); we invoke the script as a subprocess (simple + isolated) and exchange images on disk.

An ArcFace identity gate rejects edits that drift identity (falling back to the original crop).

Configuration (env vars, overridable per call):
    LIVEPORTRAIT_DIR     path to the cloned LivePortrait repo (with pretrained_weights/)
    LIVEPORTRAIT_PYTHON  python to run it with (defaults to the current interpreter)
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

import numpy as np

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
# scripts/lp_neutralize.py, resolved relative to the repo root (this file is src/neutralize.py)
_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "scripts", "lp_neutralize.py")


class NeutralizerError(RuntimeError):
    """Raised when the LivePortrait install is missing or the edit subprocess fails."""


class Neutralizer:
    """Relax expressions to neutral via LivePortrait delta-zeroing, with an ArcFace identity gate.

    ``mode="off"`` is a no-op passthrough (the default everywhere), so importing this module and
    constructing a Neutralizer never touches torch/onnx or the external tool unless used.
    """

    def __init__(self, mode: str = "off", lp_dir: str | None = None, lp_python: str | None = None,
                 alpha: float = 0.0, gate_threshold: float = 0.6,
                 cache_dir: str = "outputs/_neutralized", img_size: int = 112,
                 neutral_template=None):  # neutral_template accepted but unused (driver-free now)
        if mode not in ("off", "liveportrait"):
            raise ValueError(f"unknown neutralize mode '{mode}' (off|liveportrait)")
        self.mode = mode
        self.lp_dir = lp_dir or os.environ.get("LIVEPORTRAIT_DIR")
        self.lp_python = lp_python or os.environ.get("LIVEPORTRAIT_PYTHON") or sys.executable
        self.alpha = alpha               # 0 = full neutral, up to 1 = unchanged
        self.gate_threshold = gate_threshold
        self.cache_dir = os.path.abspath(cache_dir)
        self.img_size = img_size
        self._arc = None                 # lazy ArcFaceONNX for the gate

    # -- public ----------------------------------------------------------------
    def __call__(self, image_paths: list) -> tuple:
        """Return (kept_paths, report). Each kept path is the neutralized image if it passed the
        identity gate, else the original (so the pipeline always gets a usable image)."""
        if self.mode == "off":
            return image_paths, {"mode": "off", "items": []}

        self._require_install()
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_of = {src: self._cache_path(src) for src in image_paths}
        uncached = [s for s in image_paths if self._stale(s, cache_of[s])]
        if uncached:
            print(f"[neutralize] delta-zeroing {len(uncached)} image(s) (alpha={self.alpha}) ...")
            self._neutralize_batch(uncached, cache_of)

        kept, items = [], []
        for src in image_paths:
            neut = cache_of[src]
            if not os.path.exists(neut):
                kept.append(src)
                items.append({"src": src, "neutralized": None, "sim": None,
                              "kept": False, "reason": "no neutralized output (no face?)"})
                print(f"[neutralize]   {os.path.basename(src)}: no output; using original")
                continue
            try:
                sim = self._identity_sim(src, neut)
            except Exception as exc:  # noqa: BLE001
                sim = None
                print(f"[neutralize]   {os.path.basename(src)}: gate error ({exc})")
            if sim is not None and sim >= self.gate_threshold:
                kept.append(neut)
                items.append({"src": src, "neutralized": neut, "sim": sim, "kept": True})
                print(f"[neutralize]   {os.path.basename(src)}: ok (id sim {sim:.3f}) -> neutralized")
            else:
                kept.append(src)
                reason = "no face in neutralized" if sim is None else f"id sim {sim:.3f} < {self.gate_threshold}"
                items.append({"src": src, "neutralized": neut, "sim": sim, "kept": False, "reason": reason})
                print(f"[neutralize]   {os.path.basename(src)}: REJECT ({reason}); using original")
        n_ok = sum(1 for it in items if it["kept"])
        print(f"[neutralize] {n_ok}/{len(image_paths)} images neutralized (rest fell back to original)")
        return kept, {"mode": self.mode, "alpha": self.alpha,
                      "threshold": self.gate_threshold, "n_neutralized": n_ok, "items": items}

    # -- internals -------------------------------------------------------------
    def _require_install(self) -> None:
        if not self.lp_dir or not os.path.isdir(self.lp_dir):
            raise NeutralizerError(
                "LivePortrait not found. Set LIVEPORTRAIT_DIR to the cloned repo. "
                "See docs/expression-invariance.md for one-time setup.")
        if not os.path.exists(os.path.join(self.lp_dir, "src", "live_portrait_wrapper.py")):
            raise NeutralizerError(f"{self.lp_dir} doesn't look like a LivePortrait checkout")
        if not os.path.exists(_SCRIPT):
            raise NeutralizerError(f"missing neutralization script: {_SCRIPT}")

    def _cache_path(self, src_path: str) -> str:
        """Readable cache name: <input-stem>_neutralized[.aNN].jpg (alpha tag only when nonzero)."""
        stem = os.path.splitext(os.path.basename(src_path))[0]
        tag = "" if self.alpha == 0 else f".a{int(round(self.alpha * 100))}"
        return os.path.join(self.cache_dir, f"{stem}_neutralized{tag}.jpg")

    @staticmethod
    def _stale(src_path: str, cache_path: str) -> bool:
        """Regenerate if the cache is missing or older than the source (named cache can't self-invalidate)."""
        return (not os.path.exists(cache_path)) or os.path.getmtime(src_path) > os.path.getmtime(cache_path)

    def _neutralize_batch(self, srcs: list, cache_of: dict) -> None:
        """Run lp_neutralize.py once over all uncached sources (models load a single time)."""
        work_in = os.path.join(self.cache_dir, "_batch_in")
        work_out = os.path.join(self.cache_dir, "_batch_out")
        for d in (work_in, work_out):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        # copy each source in under its cache stem so the script's <stem>.jpg output maps back cleanly
        namemap = {}
        for src in srcs:
            cpath = cache_of[src]
            stem = os.path.splitext(os.path.basename(cpath))[0]   # e.g. yua_ariga_01_neutralized
            ext = os.path.splitext(src)[1].lower()
            ext = ext if ext in _IMAGE_EXTS else ".jpg"
            shutil.copyfile(src, os.path.join(work_in, stem + ext))
            namemap[stem + ".jpg"] = cpath                        # script writes <stem>.jpg

        cmd = [self.lp_python, _SCRIPT, "--lp-dir", self.lp_dir,
               "--in", work_in, "--out", work_out, "--alpha", str(self.alpha)]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise NeutralizerError(
                f"neutralization subprocess failed (exit {proc.returncode}).\n"
                f"stderr tail:\n{proc.stderr[-1000:]}")
        for outname, cpath in namemap.items():
            produced = os.path.join(work_out, outname)
            if os.path.exists(produced):
                shutil.move(produced, cpath)
        shutil.rmtree(work_in, ignore_errors=True)
        shutil.rmtree(work_out, ignore_errors=True)

    def _embed(self, path: str):
        """Aligned 112 crop -> L2-normalized ArcFace embedding, or None if no face."""
        from src.img_utils import load_face_rgb
        if self._arc is None:
            from src.models.arcface import ArcFaceONNX
            self._arc = ArcFaceONNX()
        try:
            crop, detected = load_face_rgb(path, self.img_size, use_detector=True, return_detected=True)
        except Exception:  # noqa: BLE001 - unreadable
            return None
        if not detected:
            return None
        return self._arc.embed(np.ascontiguousarray(crop))[0]  # (512,), already L2-normalized

    def _identity_sim(self, orig_path: str, neut_path: str):
        a, b = self._embed(orig_path), self._embed(neut_path)
        if a is None or b is None:
            return None
        return float(np.dot(a, b))  # both unit-norm -> cosine similarity
