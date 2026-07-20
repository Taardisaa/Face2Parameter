"""Re-baseline: does beauty(our render) rank characters like beauty(game render)?

This decides whether more renderer-fidelity work pays at all. Phase 5 optimises
`param -> OUR render -> beauty`; that is only the same problem the reward was validated on if the
reward ranks characters the same way on our renders as on the game's.

The first attempt at this (hs2_renderer_as_proxy.py) returned rho = -0.24, but every input to it
was contaminated and the number is void:

  * the game side blinked — repeat captures of an UNCHANGED character differed by a mean of 13.8
    pixel levels, concentrated entirely at the eyes;
  * the game's camera was 27% zoomed in relative to ours, with the eye line 0.055 of frame height
    off, so the two sides were not even looking at the same thing;
  * our side predates 2x supersampling, the BRDF1_Unity_PBS port, and the specular fix.

All four are addressed, so both sides are captured fresh here. The game side uses the clean
protocol throughout: hide_hair + freeze_pose + settle, a discarded warm-up pass after each card
load, and per-character `aligned_framing()` (the bridge's own auto-fit under-measures the head —
it applies the transform scale twice).

    .venv/Scripts/python.exe scripts/hs2_proxy_v2.py --n 40
    .venv/Scripts/python.exe scripts/hs2_proxy_v2.py --score-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call, aligned_framing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAME_DIR = os.path.join(ROOT, "data", "hs2_gt", "_proxy_game_v2")
OUR_DIR = os.path.join(ROOT, "data", "hs2_gt", "_proxy_ours_v2")
CARD_DIR = os.environ.get("HS2_CARD_DIR", r"E:\HoneySelect2_ArcticFox\UserData\chara\female")

# Two disjoint interleaved view sets: set A scores the character, set B gives the reliability
# ceiling for free (how well the game agrees with ITSELF across viewpoints).
YAWS_A = [-20.0, -10.0, 0.0, 10.0, 20.0]
YAWS_B = [-16.0, -6.0, 3.0, 13.0, 24.0]
BG = np.array([134, 135, 140], np.float64)


def capture_game(cards, res):
    os.makedirs(GAME_DIR, exist_ok=True)
    done = []
    for i, card in enumerate(cards):
        stem = os.path.splitext(os.path.basename(card))[0]
        want = {f"{stem}__{t}{j}.png": y
                for t, ys in (("a", YAWS_A), ("b", YAWS_B)) for j, y in enumerate(ys)}
        if all(os.path.exists(os.path.join(GAME_DIR, k)) for k in want):
            done.append(stem)
            continue
        try:
            call("/maker/card/load", "POST", {"path": card}, timeout=180)
            # The character keeps settling well past what one render can wait for.
            call(f"/maker/render?w={res}&h={res}&yaw=0&hide_hair=1&out={GAME_DIR}/_warm.png",
                 timeout=90)
            frame = aligned_framing(res=res)
            for k, y in want.items():
                call(f"/maker/render?w={res}&h={res}&yaw={y}&hide_hair=1&{frame}"
                     f"&out={os.path.join(GAME_DIR, k)}", timeout=90)
        except SystemExit as e:
            print(f"  [{i + 1}/{len(cards)}] {stem}: FAILED — {str(e)[:70]}")
            continue
        done.append(stem)
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(cards)}] game")
    return done


def render_ours(stems, res):
    from src.render.scene import HeadScene, render
    os.makedirs(OUR_DIR, exist_ok=True)
    done, skipped = [], []
    for i, stem in enumerate(stems):
        card = os.path.join(CARD_DIR, f"{stem}.png")
        want = {f"{stem}__{t}{j}.png": y
                for t, ys in (("a", YAWS_A), ("b", YAWS_B)) for j, y in enumerate(ys)}
        if all(os.path.exists(os.path.join(OUR_DIR, k)) for k in want):
            done.append(stem)
            continue
        try:
            scene = HeadScene(card)
            meshes = scene.deform()
            for k, y in want.items():
                img = np.clip(render(scene, meshes, yaw=y, res=res).detach().cpu().numpy(), 0, 1)
                if img.shape[2] == 4:
                    a = img[..., 3:4]
                    img = img[..., :3] * a + (BG / 255.0) * (1 - a)
                Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(OUR_DIR, k))
        except (Exception, SystemExit) as e:
            skipped.append((stem, str(e)[:60]))
            continue
        done.append(stem)
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(stems)}] ours")
    if skipped:
        print(f"  skipped {len(skipped)} (head not extracted / card unreadable) — reported, not hidden")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        stems = sorted({os.path.basename(p).rsplit("__", 1)[0]
                        for p in glob.glob(os.path.join(GAME_DIR, "*__a0.png"))})
    else:
        prev = json.load(open(os.path.join(ROOT, "data", "hs2_gt", "_multiview", "scores.json"),
                              encoding="utf-8"))
        cards = [os.path.join(CARD_DIR, f"{s}.png") for s in prev["stems"]][: args.n]
        cards = [c for c in cards if os.path.exists(c)]
        print(f"[capture] {len(cards)} cards x 10 views, both sides, clean protocol")
        stems = capture_game(cards, args.res)
        stems = render_ours(stems, args.res)

    stems = [s for s in stems
             if all(os.path.exists(os.path.join(d, f"{s}__{t}{j}.png"))
                    for d in (GAME_DIR, OUR_DIR) for t in "ab" for j in range(5))]
    print(f"\n[score] {len(stems)} characters renderable and captured on both sides")

    from scipy.stats import spearmanr, pearsonr
    from beauty_score import BeautyScorer
    scorer = BeautyScorer()

    def mean_score(d, stem, sets="ab"):
        return float(np.mean([scorer.score(os.path.join(d, f"{stem}__{t}{j}.png"),
                                           use_detector=True)
                              for t in sets for j in range(5)]))

    g = np.array([mean_score(GAME_DIR, s) for s in stems])
    o = np.array([mean_score(OUR_DIR, s) for s in stems])
    ga = np.array([mean_score(GAME_DIR, s, "a") for s in stems])
    gb = np.array([mean_score(GAME_DIR, s, "b") for s in stems])

    print(f"\n{'=' * 68}\nRENDERER AS A PROXY — re-baselined on clean captures, "
          f"{len(stems)} characters\n{'=' * 68}")
    print(f"  game beauty: mean {g.mean():.4f}  sd {g.std():.4f}")
    print(f"  ours beauty: mean {o.mean():.4f}  sd {o.std():.4f}   "
          f"(offset {o.mean() - g.mean():+.4f}, harmless — only order matters)")

    rho, p = spearmanr(g, o)
    r, _ = pearsonr(g, o)
    ceil, _ = spearmanr(ga, gb)
    print(f"\n  Spearman rho = {rho:+.4f}  (p={p:.2g})    Pearson r = {r:+.4f}")
    print(f"  previous attempt, contaminated: rho = -0.2386")
    print(f"\n  CEILING: the game against ITSELF across the two disjoint view sets = {ceil:+.4f}.")
    print(f"  No proxy can beat that, so read rho against it, not against 1.0.")
    if ceil > 0.05:
        print(f"  ceiling-normalised: {rho / ceil:+.3f}")

    verdict = ("VALID PROXY — optimise our renders; residual shading gaps are cosmetic."
               if rho >= 0.75 else
               "MARGINAL — usable with care; verify winners in-game."
               if rho >= 0.5 else
               "NOT A PROXY — optimising our renders does not optimise game beauty.")
    print(f"\n  VERDICT: {verdict}")

    json.dump({"spearman": float(rho), "pearson": float(r), "ceiling": float(ceil),
               "stems": stems, "game": g.tolist(), "ours": o.tolist()},
              open(os.path.join(OUR_DIR, "proxy_v2.json"), "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main()
