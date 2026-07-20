"""Which of the 59 face sliders actually move the beauty score, and in which hair domain?

Three questions, one experiment, because they share every render:

  1. THE NOISE FLOOR. Re-evaluate an unchanged character N times. The protocol is supposed to
     deliver sd ~0.010 on a 5-view mean. This doubles as the pre-flight check for every
     optimisation run: if the floor is not where it should be, the capture broke and the run
     would be fitting noise.

  2. PER-SLIDER SENSITIVITY. Move each slider +/-delta from p0 and measure |dS|. Sliders whose
     whole range stays under the floor are being read as signal when they are noise; freezing
     them shrinks the search space honestly. Single-slider screening cannot see interaction-only
     effects, so only sliders that are flat in BOTH directions get frozen, and the count and
     names are always printed -- a silent cap reads as "we covered everything" when it did not.

  3. WHICH DOMAIN CARRIES MORE SIGNAL. Run the whole thing bald and haired and compare
     median|dS| / noise floor. Bald shows more face, haired is what the user judges and what the
     reward's rho=0.88 was measured in; that trade is settled by measurement, not by argument.
     Pair this with scripts/hs2_hair_domain.py, which asks whether hair reorders characters
     at all.

Needs the game running with the Character Maker open. Works on bridge v0.15.0 (each probe moves
ONE slider, so the batch endpoint is not required).

    .venv/Scripts/python.exe scripts/hs2_slider_sensitivity.py \
        --card "E:/HoneySelect2_ArcticFox/UserData/chara/female/5850ddb4146caa467bd74800445b4561da118ab2.png" \
        --hair both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hs2_ingame_eval import InGameEvaluator, N_SLIDERS, slider_names  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def screen(card, hide_hair, delta, noise_n, res, scorer, sets=("a",)):
    """Noise floor + per-slider |dS| for one hair condition."""
    label = "bald" if hide_hair else "haired"
    print(f"\n{'=' * 70}\n{label.upper()}  (hide_hair={int(hide_hair)})\n{'=' * 70}")

    ev = InGameEvaluator(res=res, hide_hair=hide_hair, scorer=scorer,
                         work_dir=os.path.join(os.environ.get("TEMP", "."),
                                               f"hs2_sens_{label}"))
    try:
        ev.load_card(card)
        p0 = ev.p0.copy()
        print(f"  p0: {N_SLIDERS} sliders, "
              f"{int((p0 <= 1e-6).sum())} at 0.0, {int((p0 >= 1 - 1e-6).sum())} at 1.0")

        print(f"\n  [1/2] noise floor, {noise_n} repeats of an unchanged character")
        floor = ev.noise_floor(n=noise_n, sets=sets)
        sd = floor[sets[0]]["sd"]
        base = floor[sets[0]]["mean"]

        names = slider_names()
        print(f"\n  [2/2] per-slider sensitivity, +/-{delta} on each of {N_SLIDERS}")
        t0 = time.time()
        rows = []
        for i in range(N_SLIDERS):
            if i in ev.frozen_oob:
                # SliderUnlocker value; the bridge cannot write it, so it cannot be probed and
                # cannot be optimised either. Recorded so the count still adds up to 59.
                rows.append({"index": i, "name": names[i] if i < len(names) else f"#{i}",
                             "p0": float(p0[i]), "max_abs_dS": None, "n_probes": 0,
                             "probes": [], "unwritable": True})
                continue
            probes = []
            for sign in (+1.0, -1.0):
                v = float(np.clip(p0[i] + sign * delta, 0.0, 1.0))
                if abs(v - p0[i]) < 1e-6:
                    continue                      # already hard against that bound
                ev.set_shapes([v], indices=[i])
                r = ev.evaluate(sets=sets, tag=f"s{i:02d}{'p' if sign > 0 else 'm'}")
                probes.append({"value": v, "S": r[f"S_{sets[0]}"],
                               "dS": r[f"S_{sets[0]}"] - base})
            ev.set_shapes([float(p0[i])], indices=[i])   # restore just this slider
            best = max((abs(p["dS"]) for p in probes), default=0.0)
            rows.append({"index": i, "name": names[i] if i < len(names) else f"#{i}",
                         "p0": float(p0[i]), "max_abs_dS": float(best),
                         "n_probes": len(probes), "probes": probes, "unwritable": False})
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                print(f"      {i + 1}/{N_SLIDERS}   {el:.0f}s elapsed, "
                      f"~{el / (i + 1) * (N_SLIDERS - i - 1):.0f}s left")

        probed = [r for r in rows if not r["unwritable"]]
        unwritable = [r for r in rows if r["unwritable"]]
        d = np.array([r["max_abs_dS"] for r in probed])
        flat = [r for r in probed if r["max_abs_dS"] < 2 * sd]
        one_sided = [r for r in probed if r["n_probes"] == 1]

        print(f"\n  noise floor sd            {sd:.4f}   (2sd = {2 * sd:.4f})")
        print(f"  |dS| median / max         {np.median(d):.4f} / {d.max():.4f}")
        print(f"  SIGNAL-TO-NOISE (med/sd)  {np.median(d) / max(sd, 1e-9):.2f}")
        print(f"  below 2sd (freeze these)  {len(flat)} of {len(probed)} probed")
        if one_sided:
            print(f"  probed one-sided only     {len(one_sided)} (slider sits on a bound at p0)")
        if unwritable:
            print(f"  unwritable (out of [0,1]) {len(unwritable)}: "
                  f"{', '.join(str(r['index']) for r in unwritable)}")

        top = sorted(probed, key=lambda r: -r["max_abs_dS"])[:10]
        print("\n  most sensitive:")
        for r in top:
            print(f"    {r['max_abs_dS']:.4f}  p0={r['p0']:.3f}  [{r['index']:2d}] {r['name']}")
        if flat:
            print(f"\n  frozen ({len(flat)}): "
                  f"{', '.join(r['name'] for r in flat[:12])}"
                  f"{' ...' if len(flat) > 12 else ''}")

        print(f"\n  {ev.timing()}")
        return {"hide_hair": int(hide_hair), "noise_sd": sd, "base": base,
                "delta": delta, "sliders": rows,
                "frozen": [r["index"] for r in flat],
                "signal_to_noise": float(np.median(d) / max(sd, 1e-9))}
    finally:
        ev.restore()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--hair", choices=("off", "on", "both"), default="both",
                    help="off = bald only, on = haired only, both = compare the two domains")
    ap.add_argument("--delta", type=float, default=0.15)
    ap.add_argument("--noise-n", type=int, default=8)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "slider_sensitivity.json"))
    args = ap.parse_args()

    if not os.path.exists(args.card):
        raise SystemExit(f"no such card: {args.card}")

    from beauty_score import BeautyScorer
    scorer = BeautyScorer()          # load once, share across both conditions

    modes = {"off": [True], "on": [False], "both": [True, False]}[args.hair]
    results = {}
    for hide in modes:
        key = "bald" if hide else "haired"
        results[key] = screen(args.card, hide, args.delta, args.noise_n, args.res, scorer)

    if len(results) == 2:
        b, h = results["bald"], results["haired"]
        print(f"\n{'=' * 70}\nWHICH DOMAIN CARRIES MORE SIGNAL\n{'=' * 70}")
        print(f"  {'':10s} {'noise sd':>9s} {'median |dS|':>12s} {'S/N':>7s} {'frozen':>7s}")
        for k, r in (("bald", b), ("haired", h)):
            # unwritable sliders carry max_abs_dS = None (never probed) -- excluded, not counted
            med = float(np.median([s["max_abs_dS"] for s in r["sliders"]
                                   if s["max_abs_dS"] is not None]))
            print(f"  {k:10s} {r['noise_sd']:9.4f} {med:12.4f} "
                  f"{r['signal_to_noise']:7.2f} {len(r['frozen']):7d}")
        better = "bald" if b["signal_to_noise"] >= h["signal_to_noise"] else "haired"
        ratio = max(b["signal_to_noise"], h["signal_to_noise"]) / \
            max(min(b["signal_to_noise"], h["signal_to_noise"]), 1e-9)
        print(f"\n  {better} carries {ratio:.2f}x the signal-to-noise.")
        print("  Decide the domain from THIS together with hs2_hair_domain.py's rho: a domain with")
        print("  more signal is still the wrong one if hair reorders characters, because then a")
        print("  face optimised bald need not be better in the state the user looks at.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"card": os.path.abspath(args.card), **results},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
