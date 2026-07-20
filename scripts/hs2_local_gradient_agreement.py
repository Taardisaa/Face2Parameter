"""Do our renderer and the game AGREE about which way is uphill?

The cross-character proxy test (hs2_renderer_as_proxy.py) found beauty(our render) uncorrelated
with beauty(game render) over 40 characters. But that answers a question optimisation does not
ask. Gradient ascent never compares character A to character B; it nudges ONE character and asks
"did that help?". Two renderers can disagree about global ranking — systematic per-character
offsets will do that — while still agreeing about the effect of a nudge.

So this measures the property optimisation actually depends on:

    perturb one card's sliders, and correlate  Δ_ours  against  Δ_game.

Δ is a PAIRED difference against the same baseline through the same K viewpoints, so the view
nuisance that dominates absolute scores is common-mode and largely cancels. That makes this test
far more sensitive than the absolute-score one, which is the point.

    rho high -> local directions agree; optimising our renders moves the game character the same
                way, and the cross-character disagreement is a harmless per-character offset.
    rho low  -> the disagreement reaches into the gradient itself. Fix the renderer, and the
                per-slider breakdown below says where to look first.

Requires the game running with the Character Maker open. Leaves the maker's sliders where it
found them (it reloads the card at the end).

    .venv/Scripts/python.exe scripts/hs2_local_gradient_agreement.py --n 60 --sigma 0.10
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call, aligned_framing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "hs2_gt", "_localgrad")
YAWS = [-16.0, -8.0, 0.0, 8.0, 16.0]


def set_shapes(vals, only=None):
    """Push slider values to the maker. `only` limits the POSTs to indices that changed."""
    for i, v in enumerate(vals):
        if only is not None and i not in only:
            continue
        try:
            call("/maker/face/shapes", "POST", {"index": int(i), "value": float(v)}, timeout=20)
        except SystemExit as e:
            # The endpoint's 422 does not say WHICH slider it rejected; say it here rather than
            # leave the next person guessing at the value.
            raise SystemExit(f"slider index={i} value={float(v)!r}\n{e}")


def game_score(scorer, tag, res):
    """Score the character currently in the maker, under the protocol the drift study forced.

    A game render of an UNCHANGED character was not reproducible. The dominant cause turned out to
    be BLINKING -- consecutive captures differed almost entirely at the eyes, open in one and shut
    in the next -- not the idle sway I first blamed. Bridge v0.13.0 pins the blink, freezes the
    animation and, crucially, lets the character settle for a few frames before capturing, because
    those settings only reach the mesh through its own LateUpdate.

        mean pixel difference between repeat captures, blinking     13.8
        the same, with bridge v0.13.0's freeze + settle              0.014

    `freeze_pose` and `settle` default to on/3 server-side, so a plain render request already gets
    them. What the server cannot do is know that the CALLER just reloaded a card: that needs longer
    to settle than a render can wait, so the caller discards its first scoring pass (see main).
    Residual score noise is sd ~0.024 on a 5-view mean (NOT the 0.010 this line used to claim --
    see scripts/hs2_noise_decomposition.py; three repeats had under-sampled it), against a
    perturbation effect of sd ~0.11. So the signal-to-noise for a joint perturbation is ~4.5x.
    """
    # Re-fit the framing for THIS slider set, because our renderer does: it recomputes the head
    # bbox from the deformed vertices on every call. Holding the game's camera fixed while ours
    # refits would put a camera difference back into every Delta, which is the confound this whole
    # alignment exercise removed. The bridge's own auto-fit is wrong (see aligned_framing).
    frame = aligned_framing(res=res)
    paths = []
    for j, y in enumerate(YAWS):
        p = os.path.join(OUT_DIR, f"game_{tag}_{j}.png")
        call(f"/maker/render?w={res}&h={res}&yaw={y}&hide_hair=1&{frame}&out={p}", timeout=90)
        paths.append(p)
    return float(np.mean([scorer.score(p, use_detector=True) for p in paths]))


def ours_score(scorer, scene, sf, ab, tag, res, bg):
    from src.render.scene import render
    from PIL import Image
    meshes = scene.deform(sf, ab)
    out = []
    for j, y in enumerate(YAWS):
        img = np.clip(render(scene, meshes, yaw=y, res=res).detach().cpu().numpy(), 0, 1)
        if img.shape[2] == 4:
            a = img[..., 3:4]
            img = img[..., :3] * a + (bg / 255.0) * (1 - a)
        p = os.path.join(OUT_DIR, f"ours_{tag}_{j}.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(p)
        out.append(scorer.score(p, use_detector=True))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--sigma", type=float, default=0.10, help="slider-space perturbation sd")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    card = os.path.abspath(args.card)
    st = call("/status")
    if not st.get("inside_maker"):
        raise SystemExit("Character Maker is not open")
    call("/maker/card/load", "POST", {"path": card}, timeout=180)

    got = call("/maker/face/shapes")
    base_game = np.array([s["value"] for s in got["shapes"]], np.float64)
    print(f"[shapes] game reports {len(base_game)} sliders")

    from src.render.scene import HeadScene
    from beauty_score import BeautyScorer
    scene = HeadScene(card)
    sf0, ab = scene.trig.from_card(card)
    base_ours = sf0[0].detach().cpu().numpy().astype(np.float64)

    # The two sides must index the same sliders, or Δ_ours and Δ_game describe different edits and
    # a low correlation would mean nothing. Check it rather than assume it.
    d = np.abs(base_game - base_ours[: len(base_game)]).max()
    print(f"[check] max |game slider - our slider| = {d:.6f} "
          f"{'OK — same indexing' if d < 1e-3 else 'MISMATCH — indices do not line up, stop'}")
    if d >= 1e-3:
        worst = int(np.argmax(np.abs(base_game - base_ours[: len(base_game)])))
        raise SystemExit(f"slider {worst}: game {base_game[worst]:.4f} vs ours "
                         f"{base_ours[worst]:.4f}")

    # SliderUnlocker cards store values outside [0,1]; the game loads them happily but the bridge
    # endpoint validates [0,1] and refuses to write them. Perturbing such a slider would leave the
    # game clamped at 1.0 while our renderer used the card's 1.09 — the two sides would no longer
    # be evaluating the same parameters, which is exactly what this test must not do. Freeze them.
    # Nothing is lost: the keyframe table is constant past its ends, so their gradient is 0 anyway.
    railed = np.nonzero((base_game < 0.0) | (base_game > 1.0))[0]
    movable = np.setdiff1d(np.arange(len(base_game)), railed)
    if len(railed):
        print(f"[rails] {len(railed)} slider(s) sit outside [0,1] "
              f"({railed.tolist()}) — frozen, the bridge cannot write them back")

    scorer = BeautyScorer()
    bg = np.array([134, 135, 140], np.float64)

    # Discard one full pass: the character keeps settling for a while after a card load, and the
    # first scoring lands measurably low (3.5985 against 3.69-3.71 for the five that followed).
    game_score(scorer, "warmup", args.res)
    g0 = game_score(scorer, "base", args.res)
    o0 = ours_score(scorer, scene, sf0, ab, "base", args.res, bg)
    print(f"[baseline] game {g0:.4f}   ours {o0:.4f}\n")

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        delta = np.zeros(len(base_game))
        delta[movable] = rng.normal(0.0, args.sigma, size=len(movable))
        vals = base_game.copy()
        vals[movable] = np.clip(base_game[movable] + delta[movable], 0.0, 1.0)
        changed = set(np.nonzero(np.abs(vals - base_game) > 1e-9)[0].tolist())

        set_shapes(vals, only=changed)
        g = game_score(scorer, f"p{i}", args.res)

        sf = sf0.clone()
        sf[0, : len(vals)] = torch.tensor(vals, dtype=sf.dtype, device=sf.device)
        o = ours_score(scorer, scene, sf, ab, f"p{i}", args.res, bg)

        rows.append({"i": i, "dg": g - g0, "do": o - o0,
                     "delta": (vals - base_game).tolist()})
        if (i + 1) % 10 == 0:
            dg = np.array([r["dg"] for r in rows]); do = np.array([r["do"] for r in rows])
            from scipy.stats import spearmanr
            print(f"  [{i + 1}/{args.n}] running rho = {spearmanr(dg, do)[0]:+.3f}")

        # put the sliders back before the next draw, so perturbations do not compound
        set_shapes(base_game, only=changed)

    from scipy.stats import spearmanr, pearsonr
    dg = np.array([r["dg"] for r in rows])
    do = np.array([r["do"] for r in rows])
    rho, p = spearmanr(dg, do)
    r, _ = pearsonr(dg, do)

    print(f"\n{'=' * 64}\nLOCAL GRADIENT AGREEMENT — 1 card, {args.n} perturbations, "
          f"sigma={args.sigma}\n{'=' * 64}")
    print(f"  game  score change: mean {dg.mean():+.4f}  sd {dg.std():.4f}  "
          f"range [{dg.min():+.3f}, {dg.max():+.3f}]")
    print(f"  ours  score change: mean {do.mean():+.4f}  sd {do.std():.4f}  "
          f"range [{do.min():+.3f}, {do.max():+.3f}]")
    print(f"\n  Spearman rho = {rho:+.4f}  (p={p:.2g})    Pearson r = {r:+.4f}")
    print(f"  sign agreement: {(np.sign(dg) == np.sign(do)).mean() * 100:.1f}% "
          f"(50% = coin flip)")
    if dg.std() < 0.05:
        print("  !! the game's own score barely moved — sigma is too small to measure anything; "
              "re-run with a larger --sigma before reading the correlation")

    print(f"\n  VERDICT: " + (
        "DIRECTIONS AGREE — optimise our renders; the cross-character gap is a harmless offset."
        if rho >= 0.7 else
        "PARTIAL — usable with care; expect drift, verify winners in-game."
        if rho >= 0.4 else
        "DIRECTIONS DISAGREE — our renderer cannot guide optimisation as it stands."))

    with open(os.path.join(OUT_DIR, "local_grad.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman": float(rho), "pearson": float(r), "sigma": args.sigma,
                   "baseline": {"game": g0, "ours": o0}, "rows": rows}, f, indent=2)
    print(f"  -> {os.path.join(OUT_DIR, 'local_grad.json')}")

    call("/maker/card/load", "POST", {"path": card}, timeout=180)   # leave the maker as found
    print("  maker restored to the card's own sliders")


if __name__ == "__main__":
    main()
