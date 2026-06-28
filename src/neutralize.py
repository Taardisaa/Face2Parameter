"""Optional expression neutralizer: relax a smiling face to neutral before Stage 1.

When every available photo of a subject is smiling, multi-image aggregation can't cancel the
(systematic) smile and it leaks into the predicted geometry. This neutralizer edits the input
*image* toward a neutral expression first; the result then feeds the existing backbone+head
unchanged (see docs/expression-invariance.md).

Pluggable backends (all shell out to an external env, exchange images on disk, then pass an ArcFace
identity gate that falls back to the original on drift):
  - "liveportrait" : LivePortrait delta-zeroing (keypoint warp; identity ~0.98, residual cheek volume).
  - "kontext"      : FLUX.1 Kontext [dev] instruction edit (GGUF, low-VRAM); strongest identity-preserving
                     generative editor. Gated base components need an HF token (hf_token.txt / HF_TOKEN).
  - "off"          : no-op passthrough (default everywhere).

Config via env vars (overridable per call):
  LIVEPORTRAIT_DIR / LIVEPORTRAIT_PYTHON     (liveportrait)
  FLUX_PYTHON  -> the FLUX env's python ; FLUX_MODEL -> path to the Kontext GGUF   (kontext)
"""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import subprocess
import sys

import numpy as np

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LP_SCRIPT = os.path.join(_ROOT, "scripts", "lp_neutralize.py")
_FLUX_SCRIPT = os.path.join(_ROOT, "scripts", "flux_neutralize.py")

DEFAULT_KONTEXT_PROMPT = (
    "Change only the facial expression to a calm, neutral expression with the mouth gently "
    "closed and relaxed cheeks (no smile). If the cheeks are raised or puffed up from smiling "
    "(apple cheeks / bulging cheekbone area), flatten them back to a relaxed, non-smiling state. "
    "Keep the same person, identity, and other facial features, "
    "hairstyle, makeup, head pose, camera angle, lighting, and background unchanged."
)


def _read_hf_token() -> str | None:
    """HF token for gated FLUX base components: hf_token.txt at repo root, else HF_TOKEN env."""
    f = os.path.join(_ROOT, "hf_token.txt")
    if os.path.exists(f):
        tok = open(f, encoding="utf-8").read().strip()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


class NeutralizerError(RuntimeError):
    """Raised when a backend's external env is missing or the edit subprocess fails."""


class Neutralizer:
    """Relax expressions to neutral via a pluggable external backend, with an ArcFace identity gate.

    ``mode="off"`` is a no-op passthrough (the default everywhere), so importing this module and
    constructing a Neutralizer never touches torch/onnx or any external tool unless used.
    """

    def __init__(self, mode: str = "off", *, alpha: float = 0.0, gate_threshold: float = 0.6,
                 cache_dir: str = "outputs/_neutralized", img_size: int = 112,
                 lp_dir=None, lp_python=None,
                 flux_python=None, flux_model=None, prompt=None, steps: int = 28, guidance: float = 2.5,
                 true_cfg: float = 1.0, negative=None,
                 neutral_template=None):  # neutral_template accepted but unused (driver-free)
        if mode not in ("off", "liveportrait", "kontext"):
            raise ValueError(f"unknown neutralize mode '{mode}' (off|liveportrait|kontext)")
        self.mode = mode
        self.alpha = alpha               # liveportrait: 0=full neutral .. 1=unchanged
        self.gate_threshold = gate_threshold
        self.cache_dir = os.path.abspath(cache_dir)
        self.img_size = img_size
        # liveportrait
        self.lp_dir = lp_dir or os.environ.get("LIVEPORTRAIT_DIR")
        self.lp_python = lp_python or os.environ.get("LIVEPORTRAIT_PYTHON") or sys.executable
        # kontext
        self.flux_python = flux_python or os.environ.get("FLUX_PYTHON")
        self.flux_model = flux_model or os.environ.get("FLUX_MODEL")
        self.prompt = prompt or DEFAULT_KONTEXT_PROMPT
        self.steps = steps
        self.guidance = guidance
        self.true_cfg = true_cfg
        self.negative = negative         # None -> script default negative prompt
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
            print(f"[neutralize] {self.mode}: editing {len(uncached)} image(s) ...")
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
        return kept, {"mode": self.mode, "threshold": self.gate_threshold,
                      "n_neutralized": n_ok, "items": items}

    # -- internals -------------------------------------------------------------
    def _require_install(self) -> None:
        if self.mode == "liveportrait":
            if not self.lp_dir or not os.path.isdir(self.lp_dir):
                raise NeutralizerError("LivePortrait not found. Set LIVEPORTRAIT_DIR to the cloned "
                                       "repo. See docs/expression-invariance.md.")
            if not os.path.exists(os.path.join(self.lp_dir, "src", "live_portrait_wrapper.py")):
                raise NeutralizerError(f"{self.lp_dir} doesn't look like a LivePortrait checkout")
            if not os.path.exists(_LP_SCRIPT):
                raise NeutralizerError(f"missing script: {_LP_SCRIPT}")
        elif self.mode == "kontext":
            if not self.flux_python or not os.path.exists(self.flux_python):
                raise NeutralizerError("FLUX env not found. Set FLUX_PYTHON to the FLUX venv's python. "
                                       "See docs/expression-invariance.md.")
            if not self.flux_model or not os.path.exists(self.flux_model):
                raise NeutralizerError("FLUX GGUF not found. Set FLUX_MODEL to the Kontext .gguf path.")
            if not os.path.exists(_FLUX_SCRIPT):
                raise NeutralizerError(f"missing script: {_FLUX_SCRIPT}")

    def _backend_tag(self) -> str:
        if self.mode == "liveportrait":
            return "lp" if self.alpha == 0 else f"lp.a{int(round(self.alpha * 100))}"
        # kontext: fold the edit params into the tag so changing prompt/guidance/cfg/steps regenerates
        # (a fixed "kontext" tag silently reused stale images when only the prompt changed).
        key = repr((self.prompt, self.steps, self.guidance, self.true_cfg, self.negative))
        return "kontext." + hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]

    def _cache_path(self, src_path: str) -> str:
        """Readable, backend-tagged cache name: <input-stem>_neutralized.<backend>.jpg."""
        stem = os.path.splitext(os.path.basename(src_path))[0]
        return os.path.join(self.cache_dir, f"{stem}_neutralized.{self._backend_tag()}.jpg")

    @staticmethod
    def _stale(src_path: str, cache_path: str) -> bool:
        """Regenerate if cache is missing or older than the source (named cache can't self-invalidate)."""
        return (not os.path.exists(cache_path)) or os.path.getmtime(src_path) > os.path.getmtime(cache_path)

    def _build_cmd(self, work_in: str, work_out: str):
        """Return (cmd, env) for the active backend; both scripts take --in/--out dirs, write <stem>.jpg."""
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        if self.mode == "liveportrait":
            cmd = [self.lp_python, _LP_SCRIPT, "--lp-dir", self.lp_dir,
                   "--in", work_in, "--out", work_out, "--alpha", str(self.alpha)]
        else:  # kontext
            cmd = [self.flux_python, _FLUX_SCRIPT, "--model", self.flux_model,
                   "--in", work_in, "--out", work_out, "--prompt", self.prompt,
                   "--steps", str(self.steps), "--guidance", str(self.guidance),
                   "--true-cfg", str(self.true_cfg)]
            if self.negative:
                cmd += ["--negative-prompt", self.negative]
            tok = _read_hf_token()
            if tok:
                env["HF_TOKEN"] = env["HUGGING_FACE_HUB_TOKEN"] = tok
        return cmd, env

    def _neutralize_batch(self, srcs: list, cache_of: dict) -> None:
        """Run the backend once over all uncached sources (model loads a single time)."""
        work_in = os.path.join(self.cache_dir, "_batch_in")
        work_out = os.path.join(self.cache_dir, "_batch_out")
        for d in (work_in, work_out):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d, exist_ok=True)
        # copy each source in under its cache stem so the script's <stem>.jpg output maps back cleanly
        namemap = {}
        for src in srcs:
            cpath = cache_of[src]
            stem = os.path.splitext(os.path.basename(cpath))[0]
            ext = os.path.splitext(src)[1].lower()
            ext = ext if ext in _IMAGE_EXTS else ".jpg"
            shutil.copyfile(src, os.path.join(work_in, stem + ext))
            namemap[stem + ".jpg"] = cpath

        cmd, env = self._build_cmd(work_in, work_out)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise NeutralizerError(
                f"{self.mode} subprocess failed (exit {proc.returncode}).\n"
                f"stderr tail:\n{proc.stderr[-1200:]}")
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
