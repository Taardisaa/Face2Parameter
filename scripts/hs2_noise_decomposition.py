"""Where does the evaluation noise come from? Four layers, isolated one at a time.

The protocol was documented as delivering sd ~0.010 on a 5-view mean. The 8-repeat self-check in
`hs2_slider_sensitivity.py` measured **0.0425 bald / 0.0379 haired**, with a range of 0.11 -- and
against that floor the MEDIAN slider moves the score by only 0.039, so 54 of 57 sliders fall below
2 sigma. An optimiser run on that floor would be fitting noise while the score climbed.

Three repeats had earlier suggested 0.0125, so the excursions are intermittent and a short check
misses them. Guessing which layer is responsible is how this project has previously lost days, so
each layer is measured separately here:

  A  SCORER      score the identical PNG N times          -> expect exactly 0
  B  RENDER      same framing, same yaw, N captures       -> pixel diff + score sd
  C  FRAMING     call aligned_framing() N times           -> sd of ortho_size and target
  D  FULL        the real protocol, framing recomputed    -> the 0.0425 being explained

If D >> B, framing jitter is the cause and the fix is to hold the framing fixed across an
optimisation run. If B is already large, the render itself is not reproducible and the pose/blink
freeze is not holding. If A > 0 the scorer is not deterministic, which would invalidate far more
than this.

Needs the game running with the Character Maker open.

    .venv/Scripts/python.exe scripts/hs2_noise_decomposition.py --card <card.png> -n 10
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call, aligned_framing  # noqa: E402
from src.hs2_ingame_eval import InGameEvaluator, YAWS_A   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--hair", choices=("off", "on"), default="off")
    args = ap.parse_args()

    work = os.path.join(os.environ.get("TEMP", "."), "hs2_noise_decomp")
    os.makedirs(work, exist_ok=True)
    hide = 1 if args.hair == "off" else 0

    from beauty_score import BeautyScorer
    scorer = BeautyScorer()
    ev = InGameEvaluator(res=args.res, hide_hair=(args.hair == "off"), scorer=scorer,
                         work_dir=work)
    ev.load_card(args.card)
    N = args.n

    print(f"\n{'=' * 70}\nNOISE DECOMPOSITION -- {N} repeats, {args.hair} hair\n{'=' * 70}")

    # ---- A. is the scorer deterministic?
    probe = os.path.join(work, "_A.png")
    call(f"/maker/render?w={args.res}&h={args.res}&yaw=0&hide_hair={hide}&out={probe}")
    a = np.array([scorer.score(probe, use_detector=True) for _ in range(N)])
    print(f"\nA  SCORER   same PNG x{N}: mean {a.mean():.4f}  sd {a.std(ddof=1):.6f}"
          f"   {'deterministic' if a.std() < 1e-6 else '** NOT DETERMINISTIC **'}")

    # ---- C. how much does the framing itself move? (measured before B so B can use one of them)
    frames, orthos, targets = [], [], []
    for _ in range(N):
        f = aligned_framing(res=args.res)
        frames.append(f)
        orthos.append(float(f.split("ortho_size=")[1].split("&")[0]))
        targets.append([float(v) for v in f.split("target=")[1].split(",")])
    orthos = np.array(orthos)
    targets = np.array(targets)
    print(f"\nC  FRAMING  ortho_size: mean {orthos.mean():.6f}  sd {orthos.std(ddof=1):.6f}  "
          f"rel sd {orthos.std(ddof=1) / orthos.mean() * 100:.3f}%")
    print(f"            target sd  {np.round(targets.std(0, ddof=1), 6)}   "
          f"(world units; head is ~{orthos.mean() * 2:.3f} tall in frame)")

    # ---- B. render reproducibility at ONE fixed framing
    fixed = frames[0]
    imgs, paths = [], []
    for i in range(N):
        p = os.path.join(work, f"_B{i}.png")
        call(f"/maker/render?w={args.res}&h={args.res}&yaw=0&hide_hair={hide}&{fixed}&out={p}")
        paths.append(p)
        imgs.append(np.asarray(Image.open(p).convert("RGB"), np.float64))
    stack = np.stack(imgs)
    pix = np.abs(stack - stack.mean(0)).mean()
    b1 = np.array([scorer.score(p, use_detector=True) for p in paths])
    print(f"\nB  RENDER   fixed framing, yaw=0 x{N}: mean pixel deviation {pix:.4f} levels")
    print(f"            single-view score: mean {b1.mean():.4f}  sd {b1.std(ddof=1):.4f}")

    # ---- B5. the same, but the full 5-view mean at fixed framing
    b5 = []
    for i in range(N):
        s = []
        for j, y in enumerate(YAWS_A):
            p = os.path.join(work, f"_B5_{i}_{j}.png")
            call(f"/maker/render?w={args.res}&h={args.res}&yaw={y}&hide_hair={hide}"
                 f"&{fixed}&out={p}")
            s.append(scorer.score(p, use_detector=True))
        b5.append(float(np.mean(s)))
    b5 = np.array(b5)
    print(f"            5-view mean, FIXED framing: mean {b5.mean():.4f}  sd {b5.std(ddof=1):.4f}")

    # ---- D. the protocol as the experiments actually run it
    d = np.array([ev.evaluate(sets=("a",), tag=f"D{i}")["S_a"] for i in range(N)])
    print(f"\nD  FULL     5-view mean, framing RECOMPUTED each time: "
          f"mean {d.mean():.4f}  sd {d.std(ddof=1):.4f}")

    print(f"\n{'-' * 70}")
    sd_fixed, sd_full = b5.std(ddof=1), d.std(ddof=1)
    print(f"  fixed framing  sd {sd_fixed:.4f}")
    print(f"  recomputed     sd {sd_full:.4f}")
    if sd_fixed > 1e-9:
        print(f"  ratio          {sd_full / sd_fixed:.2f}x")
    extra = sd_full ** 2 - sd_fixed ** 2
    if extra > 0:
        print(f"  variance attributable to framing: {np.sqrt(extra):.4f} sd "
              f"({extra / max(sd_full ** 2, 1e-12) * 100:.0f}% of the total)")

    if sd_full > 2 * max(sd_fixed, 1e-6):
        print("\n  VERDICT: framing jitter dominates. Hold the framing FIXED for the whole run --\n"
              "  it is computed from the head bake, which moves slightly every capture, and the\n"
              "  scorer is very sensitive to alignment (fresh vs cached alignment correlate at\n"
              "  only rho 0.15). A per-step reframe injects that sensitivity into every step.")
    elif sd_fixed > 0.02:
        print("\n  VERDICT: the render itself is not reproducible even at fixed framing. The pose\n"
              "  freeze / blink pin is not holding; fix that before anything else.")
    else:
        print("\n  VERDICT: neither layer explains the floor -- look at the scorer input pipeline.")

    ev.restore()


if __name__ == "__main__":
    main()
