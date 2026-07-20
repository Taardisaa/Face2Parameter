"""CMA-ES over the 59 face sliders, evaluated by rendering IN THE GAME.

Why in the game and not on our differentiable renderer: `scripts/hs2_proxy_v2.py` measured
`beauty(our render)` against `beauty(game render)` at Spearman rho = +0.006, against a 0.935
ceiling from the game agreeing with itself across disjoint view sets. A whole session of fidelity
work moved that from -0.24 to +0.01, so "keep improving the renderer until it proxies" has strong
evidence against it. The game renders a view in ~100 ms and is valid by construction.

    maximize   S_A(p)  -  lambda * mean((p - p0)^2)

THE HOLDOUT IS THE POINT. Running a few thousand gradient-free steps against a DINOv2+MLP trained
on real photographs is a weak adversarial attack on that model. Three defences, and only the third
can detect failure:

  1. trust region (lambda) keeps the solution near p0;
  2. K>=4 views averaged -- a single view ranks characters at rho 0.479, so a single-view
     optimiser fits render noise while the score climbs convincingly;
  3. view set B is scored but NEVER optimised. S_A rising with S_B flat is the signature of the
     optimiser finding a hole in the reward rather than a better face. B is evaluated for each
     generation's best candidate, so the trajectory costs ~1/popsize of the run.

And none of that establishes VALIDITY. rho=0.88 says the reward ranks consistently, not correctly;
it is still a photo-trained model looking at game renders. The output goes to human review.

    .venv/Scripts/python.exe scripts/hs2_optimize_ingame.py \
        --card "E:/HoneySelect2_ArcticFox/UserData/chara/female/5850ddb4....png" \
        --budget 1200 --lam 0.5
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

# Measured floor for a 5-view mean is sd ~0.024 (scripts/hs2_noise_decomposition.py, 10 repeats),
# NOT the 0.010 this project's docs long claimed. The residual is sub-pixel geometry jitter: with
# the camera held byte-identical the captures still differ, all of it on edges (silhouette, lids,
# lip line), and it does NOT average down over views -- a 5-view mean has the same spread as one
# view, so the drift is common to the views within a round. More settle frames do not help
# (swept 3/10/30/60, no trend). Well above this limit means something else broke on top of it.
NOISE_SD_LIMIT = 0.060


def free_indices(ev, sensitivity_json, noise_sd):
    """Which sliders the optimiser may move, and why the others are excluded.

    Freezing by the single-slider screen is OFF by default, and that is a measured decision, not
    caution. `hs2_slider_sensitivity.py` moved one slider at a time by +/-0.15 and found 54 of 57
    below 2 sigma -- which would freeze almost the entire search space. But moving the WHOLE
    vector by sigma=0.15 spreads the score with sd 0.107 against a 0.024 floor (S/N 4.5), and at
    sigma=0.30, 0.133 (S/N 5.5). One slider at a time is simply the weakest available probe; its
    verdict does not transfer to the joint search CMA-ES actually performs. Pass an explicit
    --sensitivity path to override, knowing that.
    """
    free = [i for i in range(N_SLIDERS) if i not in ev.frozen_oob]
    excluded = {i: "outside [0,1] (SliderUnlocker); the bridge cannot write it"
                for i in ev.frozen_oob}

    if sensitivity_json and os.path.exists(sensitivity_json):
        s = json.load(open(sensitivity_json, encoding="utf-8"))
        cond = "bald" if ev.hide_hair else "haired"
        if cond in s:
            for row in s[cond]["sliders"]:
                if row.get("unwritable") or row["max_abs_dS"] is None:
                    continue
                if row["max_abs_dS"] < 2 * noise_sd and row["index"] in free:
                    free.remove(row["index"])
                    excluded[row["index"]] = (
                        f"|dS|={row['max_abs_dS']:.4f} < 2*noise ({2 * noise_sd:.4f})")
    return free, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--budget", type=int, default=1200, help="evaluation budget (S_A calls)")
    ap.add_argument("--lam", type=float, nargs="+", default=[0.5],
                    help="trust-region weight(s); several values sweep a Pareto front")
    ap.add_argument("--hair", choices=("off", "on"), default="off")
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--popsize", type=int, default=None, help="default 4+3*ln(n)")
    ap.add_argument("--noise-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--sensitivity", default="",
                    help="OFF by default -- the single-slider screen is underpowered and would "
                         "freeze 54 of 57. Pass a slider_sensitivity.json to freeze anyway.")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "optimize_ingame"))
    args = ap.parse_args()

    import cma
    from beauty_score import BeautyScorer

    if not os.path.exists(args.card):
        raise SystemExit(f"no such card: {args.card}")
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.card))[0]
    scorer = BeautyScorer()

    for lam in args.lam:
        run(args, lam, stem, scorer, cma)


def run(args, lam, stem, scorer, cma):
    tag = f"{stem[:24]}_lam{lam:g}_{args.hair}"
    log_path = os.path.join(args.out_dir, f"{tag}.jsonl")
    print(f"\n{'=' * 72}\nCMA-ES  card={stem[:40]}  lambda={lam:g}  hair={args.hair}"
          f"\n{'=' * 72}")

    ev = InGameEvaluator(res=args.res, hide_hair=(args.hair == "off"), scorer=scorer,
                         work_dir=os.path.join(os.environ.get("TEMP", "."), f"hs2_opt_{tag}"))
    log = open(log_path, "w", encoding="utf-8")
    t_start = time.time()
    try:
        ev.load_card(args.card)
        p0 = ev.p0.copy()

        # ---- self-check. A broken protocol produces a beautiful-looking climb over noise.
        print(f"\n[self-check] noise floor, {args.noise_n} repeats of the unchanged character")
        floor = ev.noise_floor(n=args.noise_n, sets=("a",))
        noise_sd = floor["a"]["sd"]
        base_S = floor["a"]["mean"]
        if noise_sd > NOISE_SD_LIMIT:
            raise SystemExit(
                f"noise sd {noise_sd:.4f} exceeds {NOISE_SD_LIMIT} -- the capture protocol is not "
                f"delivering its usual ~0.010. Fix that before optimising; every result from here "
                f"would be fitting the breakage.")

        free, excluded = free_indices(ev, args.sensitivity, noise_sd)
        names = slider_names()
        n = len(free)
        print(f"\n[dims] optimising {n} of {N_SLIDERS} sliders; {len(excluded)} excluded:")
        for i in sorted(excluded):
            print(f"    [{i:2d}] {names[i] if i < len(names) else '':22s} {excluded[i]}")

        p0_free = p0[free]
        popsize = args.popsize or int(4 + 3 * np.log(n))
        es = cma.CMAEvolutionStrategy(
            list(p0_free), args.sigma0,
            {"bounds": [0, 1], "popsize": popsize, "seed": args.seed, "verbose": -9})
        print(f"[cma] n={n}  popsize={popsize}  sigma0={args.sigma0}  "
              f"budget={args.budget} evals (~{args.budget // popsize} generations)")

        def full(x_free):
            p = p0.copy()
            p[free] = x_free
            return p

        def objective(x_free, sets=("a",), tag_="c"):
            r = ev.evaluate(full(x_free), sets=sets, tag=tag_)
            msd = float(np.mean((np.asarray(x_free) - p0_free) ** 2))
            r["msd"] = msd
            r["rms"] = float(np.sqrt(msd))
            r["fitness"] = -(r["S_a"] - lam * msd)
            return r

        best = {"S_a": base_S, "S_b": None, "x": p0_free.copy(), "rms": 0.0,
                "fitness": -base_S}
        n_eval, gen, history = 0, 0, []
        # +1 per generation for the holdout evaluation. Stop on whole generations: cma's tell()
        # expects the population it handed out, so a half-finished generation cannot be reported.
        while n_eval + popsize + 1 <= args.budget and not es.stop():
            xs = es.ask()
            recs = []
            for k, x in enumerate(xs):
                r = objective(x, tag_=f"g{gen}_{k}")
                r["x"] = list(map(float, x))
                r["gen"], r["eval"] = gen, n_eval
                recs.append(r)
                n_eval += 1
                log.write(json.dumps({k2: v for k2, v in r.items() if k2 != "views"}) + "\n")
            if not recs:
                break
            es.tell([r["x"] for r in recs], [r["fitness"] for r in recs])

            # Holdout on this generation's best only: a trajectory of "does the un-optimised view
            # set agree" for ~1/popsize of the cost.
            top = min(recs, key=lambda r: r["fitness"])
            hb = ev.evaluate(full(top["x"]), sets=("a", "b"), tag=f"g{gen}_hold")
            top["S_b"] = hb["S_b"]
            n_eval += 1
            history.append({"gen": gen, "eval": n_eval, "S_a": top["S_a"], "S_b": hb["S_b"],
                            "rms": top["rms"], "sigma": float(es.sigma)})
            log.write(json.dumps({"holdout": True, **history[-1]}) + "\n")
            log.flush()
            if top["S_a"] - lam * top["msd"] > -best["fitness"]:
                best = {"S_a": top["S_a"], "S_b": hb["S_b"], "x": np.array(top["x"]),
                        "rms": top["rms"], "fitness": top["fitness"]}
            print(f"  gen {gen:3d}  eval {n_eval:5d}  S_A {top['S_a']:.4f}  "
                  f"S_B(holdout) {hb['S_b']:.4f}  rms {top['rms']:.4f}  "
                  f"sigma {es.sigma:.4f}")
            gen += 1

        # ---- report
        p_best = full(best["x"])
        dt = time.time() - t_start
        print(f"\n{'-' * 72}\n  p0        S_A {base_S:.4f}")
        print(f"  best      S_A {best['S_a']:.4f}  ({best['S_a'] - base_S:+.4f})   "
              f"S_B(holdout) {best['S_b']:.4f}")
        print(f"  edit size RMS {best['rms']:.4f} per slider, max "
              f"{np.abs(p_best - p0).max():.4f}")
        print(f"  {n_eval} evals in {dt / 60:.1f} min ({dt / max(n_eval, 1):.2f} s/eval)")
        print(f"  {ev.timing()}")

        if len(history) >= 4:
            from scipy.stats import spearmanr
            a = np.array([h["S_a"] for h in history])
            b = np.array([h["S_b"] for h in history])
            rho = spearmanr(a, b)[0]
            print(f"\n  HOLDOUT AGREEMENT over {len(history)} generations: "
                  f"rho(S_A, S_B) = {rho:+.4f}")
            print(f"    S_A {a[0]:+.4f} -> {a[-1]:.4f} ({a[-1] - a[0]:+.4f})")
            print(f"    S_B {b[0]:+.4f} -> {b[-1]:.4f} ({b[-1] - b[0]:+.4f})")
            if a[-1] - a[0] > 3 * noise_sd and b[-1] - b[0] < noise_sd:
                print("\n  ** S_A climbed while the HOLDOUT did not. That is the signature of the\n"
                      "     optimiser exploiting the reward, not of a better face. Do not ship\n"
                      "     this; tighten lambda / add views / shorten the run.")

        out = os.path.join(args.out_dir, f"{tag}.json")
        json.dump({"card": os.path.abspath(args.card), "lambda": lam, "hair": args.hair,
                   "noise_sd": noise_sd, "base_S_a": base_S,
                   "free": free, "excluded": {str(k): v for k, v in excluded.items()},
                   "p0": p0.tolist(), "p_best": p_best.tolist(),
                   "best": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in best.items()},
                   "history": history, "n_eval": n_eval, "seconds": dt},
                  open(out, "w", encoding="utf-8"), indent=2)
        print(f"\n  -> {out}\n  -> {log_path}")
    finally:
        log.close()
        ev.restore()      # never leave the user's maker character in a mutated state


if __name__ == "__main__":
    main()
