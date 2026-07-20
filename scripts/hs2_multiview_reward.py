"""Does averaging the beauty score over several viewpoints make the reward usable?

The reward's failure mode is specific (see docs/beauty-guided-generation.md): a 3-degree yaw --
pure nuisance -- already scrambles half the ranking. Nuisance that is view-dependent averages
down; the character's actual beauty does not. So scoring K viewpoints and averaging should cut
the nuisance standard deviation by ~sqrt(K) while leaving the signal alone. That is the one
intervention aimed directly at the measured defect, rather than at a guess about it.

Honest test design. Averaging over K views and then correlating those same K views with
themselves would be circular. Instead each character is shot at 2K yaws, split into two DISJOINT
view sets, and we ask how well the K-view mean over set A ranks characters the same way the
K-view mean over set B does. K=1 reproduces the original single-view measurement, so the whole
curve is comparable, and the number to watch is where (if anywhere) it crosses a usable rho.

Requires the game running with the Character Maker open.

    .venv/Scripts/python.exe scripts/hs2_multiview_reward.py --n 48
    .venv/Scripts/python.exe scripts/hs2_multiview_reward.py --score-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "hs2_gt", "_multiview")
CARD_DIR = os.environ.get("HS2_CARD_DIR",
                          r"E:\HoneySelect2_ArcticFox\UserData\chara\female")

# Two disjoint interleaved sets of 5 yaws each. Interleaving (rather than left-half/right-half)
# keeps the two sets from differing systematically in pose, which would confound the comparison.
YAWS_A = [-20.0, -10.0, 0.0, 10.0, 20.0]
YAWS_B = [-16.0, -6.0, 3.0, 13.0, 24.0]


def capture(cards, res):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for i, card in enumerate(cards):
        stem = os.path.splitext(os.path.basename(card))[0]
        want = {f"{stem}__{tag}{j}.png": y
                for tag, ys in (("a", YAWS_A), ("b", YAWS_B))
                for j, y in enumerate(ys)}
        paths = {k: os.path.join(OUT_DIR, k) for k in want}
        if all(os.path.exists(p) for p in paths.values()):
            rows.append({"stem": stem, "paths": paths})
            continue
        try:
            call("/maker/card/load", "POST", {"path": card}, timeout=180)
            for k, p in paths.items():
                if not os.path.exists(p):
                    call(f"/maker/render?w={res}&h={res}&yaw={want[k]}&hide_hair=0&out={p}",
                         timeout=120)
        except SystemExit as e:
            print(f"  [{i + 1}/{len(cards)}] {stem}: FAILED — {e}")
            continue
        rows.append({"stem": stem, "paths": paths})
        print(f"  [{i + 1}/{len(cards)}] {stem}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        stems = sorted({os.path.basename(p).rsplit("__", 1)[0]
                        for p in glob.glob(os.path.join(OUT_DIR, "*__a0.png"))})
        rows = [{"stem": s,
                 "paths": {f"{s}__{t}{j}.png": os.path.join(OUT_DIR, f"{s}__{t}{j}.png")
                           for t in "ab" for j in range(5)}} for s in stems]
        rows = [r for r in rows if all(os.path.exists(p) for p in r["paths"].values())]
    else:
        st = call("/status")
        if not st.get("inside_maker"):
            raise SystemExit("Character Maker is not open — this needs the game running.")
        cards = sorted(glob.glob(os.path.join(CARD_DIR, "*.png")))[: args.n]
        print(f"[capture] {len(cards)} cards x {len(YAWS_A) + len(YAWS_B)} yaws, {args.res}^2")
        rows = capture(cards, args.res)
    if not rows:
        raise SystemExit("nothing to score")

    from scipy.stats import spearmanr
    from beauty_score import BeautyScorer
    scorer = BeautyScorer()

    print(f"\n[score] {len(rows)} characters x 10 views")
    SA, SB = [], []
    for r in rows:
        s = r["stem"]
        SA.append([scorer.score(r["paths"][f"{s}__a{j}.png"], use_detector=True) for j in range(5)])
        SB.append([scorer.score(r["paths"][f"{s}__b{j}.png"], use_detector=True) for j in range(5)])
    SA, SB = np.array(SA), np.array(SB)

    print(f"\n{'=' * 62}\nMULTI-VIEW AVERAGING — disjoint view sets, {len(rows)} characters\n{'=' * 62}")
    print("  K   rho(mean of K views in A, mean of K views in B)")
    base = None
    for K in (1, 2, 3, 4, 5):
        rho, _ = spearmanr(SA[:, :K].mean(1), SB[:, :K].mean(1))
        base = rho if K == 1 else base
        print(f"  {K}   rho = {rho:+.4f}" + ("      <- single view, the original measurement"
                                             if K == 1 else ""))
    rho5, _ = spearmanr(SA.mean(1), SB.mean(1))
    print(f"\n  single view -> 5 views: rho {base:+.4f} -> {rho5:+.4f}")
    print(f"  per-character sd across the 10 views: {np.concatenate([SA, SB], 1).std(1).mean():.4f}")
    print(f"  sd of the 10-view mean across characters: {np.concatenate([SA, SB], 1).mean(1).std():.4f}")
    print("\n  If rho climbs toward ~0.85 the reward is salvageable by averaging views — expensive\n"
          "  per optimisation step, but sound. If it plateaus well short, the nuisance is not\n"
          "  view-noise and averaging cannot fix it.")

    with open(os.path.join(OUT_DIR, "scores.json"), "w", encoding="utf-8") as f:
        json.dump({"stems": [r["stem"] for r in rows],
                   "A": SA.tolist(), "B": SB.tolist(), "rho_1": float(base),
                   "rho_5": float(rho5)}, f, indent=2)


if __name__ == "__main__":
    main()
