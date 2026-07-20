"""Does removing hair destroy the beauty scorer's *ranking* of characters?

This settles the one risk that can invalidate Phase 5 of the offline-renderer plan.

The offline renderer is bald by design (the user chose "face parity, bald"). The beauty
scorer, though, was trained on real portrait photographs — every one of them with hair. So
`beauty(our_render)` is being asked to judge an image from a domain it has never seen.

Gradient ascent on a reward does not need the reward to be *accurate*; it needs it to be
*monotone in the right thing*. So the question is NOT "how much does the score drop when the
hair goes away" — a constant drop is harmless. The question is whether the score still ORDERS
characters the same way. Formally, over a set of cards:

    rho = spearman( beauty(render(card, hair)), beauty(render(card, bald)) )

  * rho high  -> baldness is an offset. Optimising against bald renders is sound, and the
                 renderer needs no hair.
  * rho low   -> the bald reward is noise with respect to the haired one. Optimising it
                 chases artifacts of missing hair, and Phase 5 needs hair (or a scorer
                 fine-tuned on bald renders) before it means anything.

Both images come from the GAME, not from our renderer — that isolates the hair variable from
every fidelity gap our renderer still has. Requires the game running with the Character Maker
open (see scripts/hs2_capture_gt.py).

    .venv/Scripts/python.exe scripts/hs2_baldness_ablation.py --n 50
    .venv/Scripts/python.exe scripts/hs2_baldness_ablation.py --score-only   # re-score cached
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call  # noqa: E402  (shares the bridge client)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "hs2_gt", "_baldness")
CARD_DIR = os.environ.get(
    "HS2_CARD_DIR", r"E:\HoneySelect2_ArcticFox\UserData\chara\female")


def capture(cards, res: int, yaw: float, control=False):
    """For each card: load it once, then shoot it twice.

    Normally the two shots are haired vs bald. With `control=True` BOTH shots keep their hair
    and differ only by a 3-degree yaw — a nuisance perturbation far smaller than removing a
    hairstyle. That measures the scorer's own stability on this domain, which is the ceiling
    any hair result has to be read against: if the scorer cannot even rank the same faces
    consistently across 3 degrees, a low haired-vs-bald rho says nothing about hair.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for i, card in enumerate(cards):
        stem = os.path.splitext(os.path.basename(card))[0]
        suffix = "__ctrl" if control else ""
        haired = os.path.join(OUT_DIR, f"{stem}__hair{suffix}.png")
        bald = os.path.join(OUT_DIR, f"{stem}__bald{suffix}.png")
        if os.path.exists(haired) and os.path.exists(bald):
            rows.append({"stem": stem, "hair": haired, "bald": bald})
            continue
        try:
            call("/maker/card/load", "POST", {"path": card}, timeout=180)
            # Same camera for both shots; hide_hair is the ONLY thing that differs.
            # (HeadBounds() frames on the `o_head` SMR alone, so hiding hair renderers cannot
            #  move the camera — the two shots are pixel-aligned outside the hair.)
            shots = (((haired, 0, yaw), (bald, 0, yaw + 3.0)) if control
                     else ((haired, 0, yaw), (bald, 1, yaw)))
            for path, hide, y in shots:
                call(f"/maker/render?w={res}&h={res}&yaw={y}&hide_hair={hide}&out={path}",
                     timeout=120)
        except SystemExit as e:
            print(f"  [{i + 1}/{len(cards)}] {stem}: FAILED — {e}")
            continue
        rows.append({"stem": stem, "hair": haired, "bald": bald})
        print(f"  [{i + 1}/{len(cards)}] {stem}")
    return rows


def score(rows, config: str, head: str | None):
    from beauty_score import BeautyScorer
    scorer = BeautyScorer(config=config, head=head)

    out = []
    for r in rows:
        rec = dict(r)
        for key in ("hair", "bald"):
            try:
                # use_detector=True is the scorer's trained-on preprocessing (mtcnn 5-point
                # align). If it FAILS on bald renders, that is itself a finding, not a bug to
                # paper over — record it instead of silently falling back to a centre crop.
                rec[f"s_{key}"] = scorer.score(r[key], use_detector=True)
                rec[f"det_{key}"] = True
            except Exception as e:
                rec[f"s_{key}"] = scorer.score(r[key], use_detector=False)
                rec[f"det_{key}"] = False
                rec[f"err_{key}"] = str(e)[:120]
        out.append(rec)
    return out


def report(scored):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    ok = [r for r in scored if r.get("s_hair") is not None and r.get("s_bald") is not None]
    h = np.array([r["s_hair"] for r in ok])
    b = np.array([r["s_bald"] for r in ok])
    n_fail = sum(1 for r in ok if not (r["det_hair"] and r["det_bald"]))

    print(f"\n{'=' * 66}\nBALDNESS ABLATION — {len(ok)} cards, both shots from the GAME\n{'=' * 66}")
    if n_fail:
        print(f"  !! mtcnn failed to detect a face on {n_fail} render(s); those fell back to a "
              f"centre crop, which is NOT the scorer's trained preprocessing.")
    print(f"  beauty with hair : mean {h.mean():.4f}  sd {h.std():.4f}  range [{h.min():.3f}, {h.max():.3f}]")
    print(f"  beauty bald      : mean {b.mean():.4f}  sd {b.std():.4f}  range [{b.min():.3f}, {b.max():.3f}]")
    print(f"  mean shift (bald - hair) = {(b - h).mean():+.4f}   sd of shift = {(b - h).std():.4f}")

    rho, p_s = spearmanr(h, b)
    pear, p_p = pearsonr(h, b)
    print(f"\n  Spearman rho = {rho:.4f}  (p={p_s:.2g})   <-- the number that decides Phase 5")
    print(f"  Pearson  r   = {pear:.4f}  (p={p_p:.2g})")

    # A constant offset is harmless; scatter around it is what destroys the ranking.
    print(f"\n  If baldness were a pure offset, sd-of-shift would be ~0 relative to sd-of-score:")
    print(f"      sd(shift)/sd(hair) = {(b - h).std() / max(h.std(), 1e-9):.3f}   "
          f"(<<1 = offset-like, >~1 = the ranking is being scrambled)")

    verdict = ("RANKING SURVIVES — bald renders are a sound optimisation target."
               if rho >= 0.85 else
               "PARTIAL — usable but lossy; expect the optimiser to chase some hair artifacts."
               if rho >= 0.6 else
               "RANKING DOES NOT SURVIVE — optimising bald renders does not optimise beauty.")
    print(f"\n  VERDICT: rho={rho:.3f} -> {verdict}")

    worst = sorted(ok, key=lambda r: abs(r["s_bald"] - r["s_hair"]), reverse=True)[:5]
    print("\n  Largest disagreements (these are what the optimiser would be misled by):")
    for r in worst:
        print(f"    {r['stem'][:44]:44s} hair {r['s_hair']:.3f} -> bald {r['s_bald']:.3f} "
              f"({r['s_bald'] - r['s_hair']:+.3f})")

    with open(os.path.join(OUT_DIR, "scores.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman": float(rho), "pearson": float(pear),
                   "mean_shift": float((b - h).mean()), "rows": ok}, f, indent=2)
    print(f"\n  -> {os.path.join(OUT_DIR, 'scores.json')}")
    return rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--config", default="beauty_dinov2_vits14")
    ap.add_argument("--head", default=None, help="beauty head weights (default: latest in exp/)")
    ap.add_argument("--score-only", action="store_true", help="skip capture, use cached PNGs")
    ap.add_argument("--control", action="store_true",
                    help="control run: both shots haired, differing only by a 3-degree yaw. "
                         "Measures the scorer's own stability on game renders.")
    args = ap.parse_args()

    if args.score_only:
        stems = sorted({os.path.basename(p).rsplit("__", 1)[0]
                        for p in glob.glob(os.path.join(OUT_DIR, "*__hair.png"))})
        rows = [{"stem": s,
                 "hair": os.path.join(OUT_DIR, f"{s}__hair.png"),
                 "bald": os.path.join(OUT_DIR, f"{s}__bald.png")} for s in stems]
        rows = [r for r in rows if os.path.exists(r["bald"])]
    else:
        st = call("/status")
        if not st.get("inside_maker"):
            raise SystemExit("Character Maker is not open — this needs the game running.")
        cards = sorted(glob.glob(os.path.join(CARD_DIR, "*.png")))[: args.n]
        if not cards:
            raise SystemExit(f"no cards in {CARD_DIR}")
        print(f"[capture] {len(cards)} cards, {args.res}^2, yaw={args.yaw:+.0f}"
              f"{'  [CONTROL: hair kept on both, 3-deg yaw only]' if args.control else ''}")
        rows = capture(cards, args.res, args.yaw, control=args.control)

    if not rows:
        raise SystemExit("nothing to score")
    print(f"\n[score] {len(rows)} pairs")
    report(score(rows, args.config, args.head))


if __name__ == "__main__":
    main()
