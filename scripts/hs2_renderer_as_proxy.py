"""Is our offline renderer a valid stand-in for the game, as far as the reward is concerned?

This is the measurement that decides whether shading fidelity is worth more work.

Phase 5 optimises `param -> OUR render -> beauty`. The reward's reliability (rho 0.88 with 4-5
views) was established on GAME renders. Those are only the same problem if the reward ranks
characters the same way on our renders as on the game's. Wherever the two renderers differ
systematically -- our heavier eyelashes, our pinker skin -- the optimiser is free to chase
whatever our renderer does that the game does not, and the score will rise while the actual
character does not improve.

    rho high -> our renderer is a valid proxy; residual shading gaps are cosmetic.
    rho low  -> optimising our renders does not optimise game beauty; fix fidelity first.

Both sides use the SAME K viewpoints and the same background, so the comparison isolates
rendering differences rather than presentation. Ceiling note: neither side is perfectly reliable,
so the observed correlation is attenuated -- it is bounded by ~sqrt(r_ours * r_game), and the
game's own K=5 reliability is 0.88. A result near that bound is as good as this test can show.

Needs the game captures from scripts/hs2_multiview_reward.py; does NOT need the game running.

    .venv/Scripts/python.exe scripts/hs2_renderer_as_proxy.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MV_DIR = os.path.join(ROOT, "data", "hs2_gt", "_multiview")
OUR_DIR = os.path.join(ROOT, "data", "hs2_gt", "_ours")
CARD_DIR = os.environ.get("HS2_CARD_DIR", r"E:\HoneySelect2_ArcticFox\UserData\chara\female")

YAWS_A = [-20.0, -10.0, 0.0, 10.0, 20.0]
YAWS_B = [-16.0, -6.0, 3.0, 13.0, 24.0]


def game_background(sample_png: str) -> np.ndarray:
    """Sample the game capture's backdrop so our renders can sit on the identical colour."""
    a = np.asarray(Image.open(sample_png).convert("RGB"))
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], axis=0)
    return np.median(edge, axis=0).astype(np.uint8)


def render_ours(cards, res, bg):
    from src.render.scene import HeadScene, render
    from src.hs2_mesh import card_head_id

    os.makedirs(OUR_DIR, exist_ok=True)
    done, skipped = [], []
    for i, card in enumerate(cards):
        stem = os.path.splitext(os.path.basename(card))[0]
        want = {f"{stem}__{t}{j}.png": y
                for t, ys in (("a", YAWS_A), ("b", YAWS_B)) for j, y in enumerate(ys)}
        paths = {k: os.path.join(OUR_DIR, k) for k in want}
        if all(os.path.exists(p) for p in paths.values()):
            done.append(stem)
            continue
        try:
            scene = HeadScene(card)
            meshes = scene.deform()
            for k, p in paths.items():
                img = render(scene, meshes, yaw=want[k], res=res).detach().cpu().numpy()
                img = np.clip(img, 0, 1)
                if img.shape[2] == 4:
                    a = img[..., 3:4]
                    img = img[..., :3] * a + (bg / 255.0) * (1 - a)
                Image.fromarray((img * 255).astype(np.uint8)).save(p)
        except (Exception, SystemExit) as e:
            # Every card needs its own extraction pass (the head mesh is shared, but the manifest
            # and the card-selected textures are not). SystemExit is in the tuple because
            # HeadScene raises it for a missing manifest — `except Exception` silently let it
            # kill the run. A card we cannot render is a coverage gap; report it, don't hide it.
            try:                      # a card that fails to load also fails to report its headId
                hid = card_head_id(card)
            except Exception:
                hid = "unreadable"
            skipped.append((stem, hid, str(e)[:60]))
            continue
        done.append(stem)
        print(f"  [{i + 1}/{len(cards)}] {stem}")
    if skipped:
        print(f"\n  skipped {len(skipped)} cards whose head is not extracted "
              f"(headIds {sorted({str(h) for _, h, _ in skipped})}) — coverage gap, not hidden")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()

    gt = json.load(open(os.path.join(MV_DIR, "scores.json"), encoding="utf-8"))
    game = {s: (np.array(a) + np.array(b)).tolist()
            for s, a, b in zip(gt["stems"], gt["A"], gt["B"])}      # only the stems matter here
    game_mean = {s: float(np.mean(a + b)) for s, a, b in zip(gt["stems"], gt["A"], gt["B"])}

    bg = game_background(os.path.join(MV_DIR, f"{gt['stems'][0]}__a0.png"))
    print(f"[bg] matching the game backdrop {tuple(int(v) for v in bg)}")

    if not args.score_only:
        cards = [os.path.join(CARD_DIR, f"{s}.png") for s in gt["stems"]]
        cards = [c for c in cards if os.path.exists(c)]
        print(f"[render] {len(cards)} cards x 10 yaws with our renderer")
        render_ours(cards, args.res, bg)

    from scipy.stats import spearmanr, pearsonr
    from beauty_score import BeautyScorer
    scorer = BeautyScorer()

    stems = [s for s in gt["stems"]
             if all(os.path.exists(os.path.join(OUR_DIR, f"{s}__{t}{j}.png"))
                    for t in "ab" for j in range(5))]
    print(f"\n[score] {len(stems)} of {len(gt['stems'])} characters renderable by us")

    ours_all = {}
    for s in stems:
        ours_all[s] = [scorer.score(os.path.join(OUR_DIR, f"{s}__{t}{j}.png"), use_detector=True)
                       for t in "ab" for j in range(5)]

    g = np.array([game_mean[s] for s in stems])
    o = np.array([float(np.mean(ours_all[s])) for s in stems])

    print(f"\n{'=' * 64}\nOUR RENDERER AS A PROXY FOR THE GAME — {len(stems)} characters, "
          f"10 views each\n{'=' * 64}")
    print(f"  game  beauty: mean {g.mean():.4f}  sd {g.std():.4f}")
    print(f"  ours  beauty: mean {o.mean():.4f}  sd {o.std():.4f}   "
          f"(offset {o.mean() - g.mean():+.4f} — harmless, only order matters)")
    rho, p = spearmanr(g, o)
    r, _ = pearsonr(g, o)
    print(f"\n  Spearman rho = {rho:+.4f}  (p={p:.2g})    Pearson r = {r:+.4f}")
    print(f"  attenuation ceiling: the game's own K=5 reliability is {gt['rho_5']:.3f}, so a\n"
          f"  perfect proxy would still land near sqrt(0.88*r_ours) < 0.94, not at 1.0.")

    verdict = ("VALID PROXY — residual shading gaps are cosmetic; Phase 5 can optimise our renders."
               if rho >= 0.75 else
               "MARGINAL — usable, but expect the optimiser to exploit some renderer-only artifacts."
               if rho >= 0.5 else
               "NOT A PROXY — optimising our renders does not optimise game beauty. Fix fidelity.")
    print(f"\n  VERDICT: {verdict}")

    worst = sorted(stems, key=lambda s: abs(float(np.mean(ours_all[s])) - game_mean[s]),
                   reverse=True)[:5]
    print("\n  Largest disagreements (inspect these renders for renderer-only artifacts):")
    for s in worst:
        print(f"    {s[:44]:44s} game {game_mean[s]:.3f}  ours {np.mean(ours_all[s]):.3f}")

    with open(os.path.join(OUR_DIR, "proxy_scores.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman": float(rho), "pearson": float(r),
                   "stems": stems, "game": g.tolist(), "ours": o.tolist()}, f, indent=2)


if __name__ == "__main__":
    main()
