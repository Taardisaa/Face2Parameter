"""Why is the beauty scorer unstable on game renders — alignment jitter, or the model itself?

The control run in hs2_baldness_ablation.py found that a 3-degree yaw — a nuisance
perturbation, visually almost nothing — drops the scorer's self-consistency to rho ~= 0.48 on
game renders. That is the real blocker for Phase 5: it puts a ceiling on any reward signal
derived from renders, and it is upstream of every remaining renderer fidelity gap.

Two candidate causes, and they call for completely different fixes:

  (a) mtcnn 5-point alignment jitters between the two renders, so DINOv2 sees differently
      cropped faces. Fix: stabilise preprocessing (fixed crop from the known camera, since we
      CONTROL the camera for renders — no detection needed).
  (b) The DINOv2->MLP head genuinely does not transfer to rendered faces. Fix: fine-tune or
      re-train the reward on renders; no amount of preprocessing saves it.

Discriminating test: re-score the SAME cached image pairs with the detector switched off, so
both images get an identical deterministic centre crop. If rho jumps, it is (a). If it stays
low, it is (b).

    .venv/Scripts/python.exe scripts/hs2_scorer_stability.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beauty_score import BeautyScorer  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "hs2_gt", "_baldness")


def pairs(suffix: str):
    out = []
    for h in sorted(glob.glob(os.path.join(OUT_DIR, f"*__hair{suffix}.png"))):
        b = h.replace(f"__hair{suffix}.png", f"__bald{suffix}.png")
        if os.path.exists(b):
            out.append((h, b))
    return out


def run(scorer, ps, use_detector: bool):
    a, b = [], []
    for ph, pb in ps:
        a.append(scorer.score(ph, use_detector=use_detector))
        b.append(scorer.score(pb, use_detector=use_detector))
    return np.array(a), np.array(b)


def main():
    scorer = BeautyScorer()

    # Determinism check first: if the scorer is not even repeatable on one image, every other
    # number here is meaningless.
    probe = pairs("__ctrl")[0][0] if pairs("__ctrl") else pairs("")[0][0]
    s1, s2 = scorer.score(probe, use_detector=False), scorer.score(probe, use_detector=False)
    print(f"[determinism] same image twice: {s1:.6f} vs {s2:.6f}  "
          f"({'deterministic' if abs(s1 - s2) < 1e-6 else 'NON-DETERMINISTIC — stop and fix'})")

    for label, suffix in (("CONTROL  (hair kept, 3-deg yaw)", "__ctrl"),
                          ("BALDNESS (hair removed)", "")):
        ps = pairs(suffix)
        if not ps:
            continue
        print(f"\n{label}  n={len(ps)}")
        for name, det in (("mtcnn align (as trained)", True), ("fixed centre crop", False)):
            a, b = run(scorer, ps, det)
            rho, p = spearmanr(a, b)
            print(f"    {name:26s} rho = {rho:+.4f}  (p={p:.2g})   "
                  f"sd(diff)={np.abs(a - b).std():.3f}")

    print("\nReading it: if 'fixed centre crop' >> 'mtcnn align', the instability is "
          "preprocessing\nand is fixable. If both are low, the reward model does not transfer "
          "to renders.")


if __name__ == "__main__":
    main()
