"""Does hair rewrite the beauty ranking? Hair is the ONLY difference here.

Why this has to run before any optimisation. The reward's reliability (rho 0.88 over 5 views) was
measured on hair-ON captures (`hs2_multiview_reward.py` passes `hide_hair=0`). The proxy verdict
(rho 0.006) was measured on hair-OFF captures. Our offline renderer is bald by design. Those three
have never been the same domain, and comparing the two existing capture sets suggests the gap is
first-order:

    bald  mean 3.810 sd 0.255      hair  mean 3.680 sd 0.211
    rho(bald, hair) = +0.186       mean |rank shift| = 9.7 places out of 34
    #1 -> #30,  #3 -> #24,  #32 -> #2

But those two sets ALSO differ in framing (the multiview run predates the aligned_framing fix, so
its camera was ~27% zoomed in), so that 0.186 is hair AND framing and cannot be charged to hair.
This script removes the confound: one card load, ONE `aligned_framing()`, then 10 views with hair
and 10 without. Framing is computed from the o_head bake, which no hair renderer contributes to,
so it is identical for both conditions by construction — hair is the only thing that varies.

What decides the evaluation domain:

    rho(on, off) near the ceilings  -> hair is an additive offset; optimise bald (more face visible)
    rho(on, off) below the ceilings -> hair rewrites the ranking, so a face optimised bald need not
                                       be better in the state the user actually looks at

The ceilings are not optional. Each condition's own A-vs-B view-set agreement bounds what
rho(on, off) could possibly reach; without them a low number cannot be told apart from "this
condition is just unreliable". Same reason the proxy verdict needed its 0.935.

    .venv/Scripts/python.exe scripts/hs2_hair_domain.py --n 34
    .venv/Scripts/python.exe scripts/hs2_hair_domain.py --score-only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call, aligned_framing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "hs2_gt", "_hair_domain")
CARD_DIR = os.environ.get("HS2_CARD_DIR", r"E:\HoneySelect2_ArcticFox\UserData\chara\female")

# Same disjoint interleaved view sets as every other experiment here, so the ceilings are
# comparable with the 0.935 from hs2_proxy_v2.py.
YAWS_A = [-20.0, -10.0, 0.0, 10.0, 20.0]
YAWS_B = [-16.0, -6.0, 3.0, 13.0, 24.0]

# hide_hair does not reach accessory-slot hair (bridge <= v0.15.0), so for this card the "off"
# condition is not actually bald and the comparison would be meaningless. Excluded until B3.
KNOWN_HIDE_HAIR_LEAK = {"HS2ChaF_20240525091904656"}


def view_names(stem):
    """(filename, yaw, hide_hair) for all 20 captures of one character."""
    for cond, hide in (("off", 1), ("on", 0)):
        for tag, yaws in (("a", YAWS_A), ("b", YAWS_B)):
            for j, y in enumerate(yaws):
                yield f"{stem}__{cond}__{tag}{j}.png", y, hide


def capture(cards, res):
    os.makedirs(OUT_DIR, exist_ok=True)
    done = []
    for i, card in enumerate(cards):
        stem = os.path.splitext(os.path.basename(card))[0]
        want = list(view_names(stem))
        if all(os.path.exists(os.path.join(OUT_DIR, k)) for k, _, _ in want):
            done.append(stem)
            continue
        try:
            call("/maker/card/load", "POST", {"path": card}, timeout=180)
            # The character keeps settling past what a single render waits for; the first pass
            # after a load is discarded everywhere in this project for that reason.
            call(f"/maker/render?w={res}&h={res}&yaw=0&hide_hair=1&out={OUT_DIR}/_warm.png",
                 timeout=90)
            # ONE framing for both conditions. It is derived from the o_head bake, which no hair
            # renderer contributes to, so hair cannot move it -- that is what makes hair the only
            # variable below.
            frame = aligned_framing(res=res)
            for k, y, hide in want:
                call(f"/maker/render?w={res}&h={res}&yaw={y}&hide_hair={hide}&{frame}"
                     f"&out={os.path.join(OUT_DIR, k)}", timeout=90)
        except SystemExit as e:
            print(f"  [{i + 1}/{len(cards)}] {stem}: FAILED -- {str(e)[:70]}")
            continue
        done.append(stem)
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(cards)}] captured")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=34)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--include-leaky", action="store_true",
                    help="do NOT exclude cards whose accessory-slot hair survives hide_hair")
    args = ap.parse_args()

    if args.score_only:
        stems = sorted({os.path.basename(p).rsplit("__", 2)[0]
                        for p in glob.glob(os.path.join(OUT_DIR, "*__off__a0.png"))})
    else:
        # Reuse the proxy run's character list so the numbers sit next to its 0.935 ceiling.
        prev = json.load(open(os.path.join(ROOT, "data", "hs2_gt", "_proxy_ours_v2",
                                           "proxy_v2.json"), encoding="utf-8"))
        stems = prev["stems"]
        if not args.include_leaky:
            dropped = [s for s in stems if s in KNOWN_HIDE_HAIR_LEAK]
            stems = [s for s in stems if s not in KNOWN_HIDE_HAIR_LEAK]
            if dropped:
                print(f"[skip] {len(dropped)} card(s) whose accessory hair survives hide_hair: "
                      f"{', '.join(dropped)}")
        cards = [os.path.join(CARD_DIR, f"{s}.png") for s in stems][: args.n]
        cards = [c for c in cards if os.path.exists(c)]
        st = call("/status")
        if not st.get("inside_maker"):
            raise SystemExit("Character Maker is not open -- this needs the game running.")
        print(f"[capture] {len(cards)} cards x 20 views (10 hair-off + 10 hair-on), "
              f"identical framing")
        stems = capture(cards, args.res)

    stems = [s for s in stems
             if all(os.path.exists(os.path.join(OUT_DIR, k)) for k, _, _ in view_names(s))]
    if not args.include_leaky:
        stems = [s for s in stems if s not in KNOWN_HIDE_HAIR_LEAK]
    if len(stems) < 8:
        raise SystemExit(f"only {len(stems)} characters captured -- too few to correlate")

    from scipy.stats import spearmanr
    from beauty_score import BeautyScorer
    scorer = BeautyScorer()

    print(f"\n[score] {len(stems)} characters x 20 views")

    # Score each view exactly once, then aggregate -- the A/B means and the AB mean are all
    # views of the same 20 numbers per character.
    raw = {}
    for c in ("off", "on"):
        for t in "ab":
            raw[c, t] = np.array([[scorer.score(
                os.path.join(OUT_DIR, f"{s}__{c}__{t}{j}.png"), use_detector=True)
                for j in range(5)] for s in stems])
    S = {c: {"a": raw[c, "a"].mean(1), "b": raw[c, "b"].mean(1),
             "ab": np.concatenate([raw[c, "a"], raw[c, "b"]], 1).mean(1)}
         for c in ("off", "on")}

    off, on = S["off"]["ab"], S["on"]["ab"]
    ceil_off = spearmanr(S["off"]["a"], S["off"]["b"])[0]
    ceil_on = spearmanr(S["on"]["a"], S["on"]["b"])[0]
    rho, p = spearmanr(off, on)

    print(f"\n{'=' * 70}\nDOES HAIR REWRITE THE RANKING? -- {len(stems)} characters, "
          f"identical framing\n{'=' * 70}")
    print(f"  hair OFF: mean {off.mean():.4f}  sd {off.std():.4f}   "
          f"own view-set ceiling {ceil_off:+.4f}")
    print(f"  hair ON : mean {on.mean():.4f}  sd {on.std():.4f}   "
          f"own view-set ceiling {ceil_on:+.4f}")
    print(f"\n  rho(hair-off, hair-on) = {rho:+.4f}  (p={p:.2g})")

    # Both sides are noisy, so the observed correlation is attenuated by their reliabilities.
    # Dividing it out estimates what the agreement would be with perfectly reliable scores --
    # the honest way to ask whether the residual disagreement is real or just noise.
    bound = float(np.sqrt(max(ceil_off, 0) * max(ceil_on, 0)))
    print(f"  attenuation bound sqrt(ceil_off * ceil_on) = {bound:+.4f}"
          f"  -- read rho against THIS, not against 1.0")
    if bound > 0.05:
        print(f"  disattenuated: {rho / bound:+.4f}")

    idx = {s: i for i, s in enumerate(stems)}
    rank_off = {s: r for r, s in enumerate(sorted(stems, key=lambda s: -off[idx[s]]))}
    rank_on = {s: r for r, s in enumerate(sorted(stems, key=lambda s: -on[idx[s]]))}
    shift = np.array([abs(rank_off[s] - rank_on[s]) for s in stems])
    print(f"  mean |rank shift| when hair goes on: {shift.mean():.1f} places out of {len(stems)}")

    if bound <= 0.05:
        verdict = ("INCONCLUSIVE -- at least one condition has no usable view-set reliability, so "
                   "rho cannot be read. Fix the capture protocol before drawing any conclusion.")
    elif rho / bound >= 0.75:
        verdict = ("HAIR IS AN OFFSET -- the ranking survives it. Optimise on BALD renders: more "
                   "face is visible, so the per-slider signal should be larger.")
    elif rho / bound >= 0.4:
        verdict = ("PARTIAL -- hair reorders characters substantially. Prefer the hair-ON domain "
                   "(it is what the user judges and what rho=0.88 was measured in), but confirm "
                   "with the per-slider signal-to-noise from hs2_slider_sensitivity.py.")
    else:
        verdict = ("HAIR REWRITES THE RANKING -- a face optimised bald need not be better in the "
                   "state the user looks at. Evaluate WITH hair, and treat every bald-render "
                   "result (including our offline renderer's) as a different quantity.")
    print(f"\n  VERDICT: {verdict}")

    biggest = sorted(stems, key=lambda s: -abs(rank_off[s] - rank_on[s]))[:5]
    print("\n  Largest reorderings (bald rank -> haired rank):")
    for s in biggest:
        i = idx[s]
        print(f"    #{rank_off[s] + 1:2d} -> #{rank_on[s] + 1:2d}   "
              f"off {off[i]:.3f}  on {on[i]:.3f}   {s[:44]}")

    out = os.path.join(OUT_DIR, "hair_domain.json")
    json.dump({"stems": stems, "off": off.tolist(), "on": on.tolist(),
               "ceiling_off": float(ceil_off), "ceiling_on": float(ceil_on),
               "rho": float(rho), "p": float(p), "attenuation_bound": bound,
               "mean_rank_shift": float(shift.mean())},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
