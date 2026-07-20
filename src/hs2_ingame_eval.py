"""One place where `59 sliders -> beauty score` is defined, for every in-game experiment.

The measurement protocol here is not a style choice; each clause was bought with a wrong result:

  * `hide_hair`/`freeze_pose`/`settle` -- repeat captures of an UNCHANGED character once differed
    by 13.8 mean pixel levels, all of it at the eyes. Blinking. Settings reach the mesh through
    LateUpdate, so a same-frame capture records the state they replaced.
  * discard the first pass after a card load -- the character keeps settling past one render.
  * per-evaluation `aligned_framing()` -- the bridge's own auto-fit applies the transform scale
    twice, so its camera sat 27% too close. Geometry moves when sliders move, so the framing has
    to be recomputed every evaluation, not once per run.
  * K >= 4 views averaged -- a single view ranks characters at rho 0.479 (a 3-degree yaw scrambles
    half the order). Optimising one view fits render noise while the score climbs convincingly.
  * disjoint view sets A and B -- B is scored but never optimised, so "the score went up" can be
    told apart from "the optimiser found a hole in the reward".

`set_shapes` prefers `POST /maker/face/shapes/batch` and falls back to the per-slider endpoint, so
this works against bridge v0.15.0 today and picks up the batch speedup with no code change once
v0.16.0 is deployed. `--force-single` exists so the two paths can be diffed against each other.

Needs the game running with the Character Maker open.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_capture_gt import call, aligned_framing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The interleaved disjoint view sets used by every experiment in this project, so ceilings and
# reliabilities stay comparable across them.
YAWS_A = [-20.0, -10.0, 0.0, 10.0, 20.0]
YAWS_B = [-16.0, -6.0, 3.0, 13.0, 24.0]

N_SLIDERS = 59      # cf_headshapename; the rig needs all 59, NOT the 54-dim ML label vector


class InGameEvaluator:
    """Set the 59 face sliders in the running game, render K views, score them."""

    def __init__(self, res=512, hide_hair=True, scorer=None, work_dir=None,
                 yaws_a=None, yaws_b=None, force_single=False):
        self.res = res
        self.hide_hair = 1 if hide_hair else 0
        self.yaws = {"a": list(yaws_a or YAWS_A), "b": list(yaws_b or YAWS_B)}
        self.work_dir = work_dir or os.path.join(
            os.environ.get("TEMP", "."), "hs2_ingame_eval")
        os.makedirs(self.work_dir, exist_ok=True)
        self._scorer = scorer
        self._batch = None if not force_single else False   # None = not probed yet
        self.p0 = None
        # Cards saved with SliderUnlocker can hold values outside [0,1], and the bridge endpoint
        # refuses to write those. Such sliders are FROZEN: never written, so they keep the card's
        # value. Writing them is not merely rejected -- even restoring p0 would fail.
        self.frozen_oob = []
        self.n_render = 0
        self.n_eval = 0
        self.t_render = 0.0
        self.t_score = 0.0

        st = call("/status")
        if not st.get("inside_maker"):
            raise SystemExit("Character Maker is not open -- this needs the game running.")

    # ---------------------------------------------------------------- model (loaded lazily)
    @property
    def scorer(self):
        if self._scorer is None:
            from beauty_score import BeautyScorer
            self._scorer = BeautyScorer()
        return self._scorer

    # ---------------------------------------------------------------- character / sliders
    def load_card(self, path):
        """Load a card and take p0 from it, discarding the settling first render."""
        r = call("/maker/card/load", "POST", {"path": os.path.abspath(path)}, timeout=180)
        call(f"/maker/render?w={self.res}&h={self.res}&yaw=0&hide_hair=1"
             f"&out={os.path.join(self.work_dir, '_warm.png')}", timeout=120)
        self.p0 = self.get_shapes()
        self.frozen_oob = [i for i in range(N_SLIDERS)
                           if self.p0[i] < 0.0 or self.p0[i] > 1.0]
        if self.frozen_oob:
            vals = ", ".join(f"[{i}]={self.p0[i]:+.3f}" for i in self.frozen_oob)
            print(f"  [oob] {len(self.frozen_oob)} slider(s) outside [0,1] (SliderUnlocker): "
                  f"{vals}\n        frozen -- the bridge cannot write them, so they keep the "
                  f"card's value.")
        return r

    def get_shapes(self) -> np.ndarray:
        r = call("/maker/face/shapes")
        vals = np.zeros(N_SLIDERS, np.float64)
        for s in r["shapes"]:
            if s["index"] < N_SLIDERS:
                vals[s["index"]] = float(s["value"])
        if r["count"] != N_SLIDERS:
            # Loud rather than silent: a different count means the rig assumption is wrong, and
            # that exact assumption ("54 is enough") already produced deformed ears once.
            raise SystemExit(f"game reports {r['count']} face shapes, expected {N_SLIDERS}")
        return vals

    def set_shapes(self, values, indices=None):
        """Write sliders, skipping the out-of-range ones. Uses the batch endpoint if available."""
        values = np.asarray(values, np.float64)
        idx = list(range(N_SLIDERS)) if indices is None else [int(i) for i in indices]
        keep = [k for k, i in enumerate(idx) if i not in self.frozen_oob]
        if len(keep) != len(idx):
            idx = [idx[k] for k in keep]
            values = values[keep]
        if values.size == 0:
            return
        if np.any(values < -1e-9) or np.any(values > 1 + 1e-9):
            raise ValueError(
                f"slider values must lie in [0,1]; got "
                f"[{values.min():.4f}, {values.max():.4f}]")
        values = np.clip(values, 0.0, 1.0)

        if self._batch is not False:
            # Always send indices: frozen out-of-range sliders may have been dropped above, so
            # `values` is not guaranteed to be the full 0..58 run even when the caller passed all.
            body = {"values": [float(v) for v in values], "indices": [int(i) for i in idx]}
            try:
                call("/maker/face/shapes/batch", "POST", body, timeout=60)
                self._batch = True
                return
            except SystemExit as e:
                if "404" not in str(e)[:12]:
                    raise
                self._batch = False      # bridge <= v0.15.0; fall through, once

        for i, v in zip(idx, values):
            call("/maker/face/shapes", "POST", {"index": int(i), "value": float(v)}, timeout=30)

    # ---------------------------------------------------------------- render + score
    def _render(self, sets, tag):
        """Capture the requested view sets at one framing; returns {set: [paths]}."""
        t = time.time()
        # Framing must follow the geometry: sliders move the head, and the bridge's own auto-fit
        # is the one with the double-scale bug.
        frame = aligned_framing(res=self.res)
        out = {}
        for s in sets:
            paths = []
            for j, y in enumerate(self.yaws[s]):
                p = os.path.join(self.work_dir, f"{tag}__{s}{j}.png")
                call(f"/maker/render?w={self.res}&h={self.res}&yaw={y}"
                     f"&hide_hair={self.hide_hair}&{frame}&out={p}", timeout=90)
                paths.append(p)
                self.n_render += 1
            out[s] = paths
        self.t_render += time.time() - t
        return out

    def evaluate(self, values=None, sets=("a",), tag="eval") -> dict:
        """Set sliders (if given), render, score. Returns per-view scores and per-set means."""
        if values is not None:
            self.set_shapes(values)
        paths = self._render(sets, tag)
        t = time.time()
        views = {s: [self.scorer.score(p, use_detector=True) for p in ps]
                 for s, ps in paths.items()}
        self.t_score += time.time() - t
        self.n_eval += 1
        return {"views": views, **{f"S_{s}": float(np.mean(v)) for s, v in views.items()}}

    # ---------------------------------------------------------------- protocol self-check
    def noise_floor(self, n=8, sets=("a",), verbose=True) -> dict:
        """Re-evaluate the CURRENT character n times without changing anything.

        Run this before every optimisation. If the spread is not what the protocol is supposed to
        deliver (~0.010 for a 5-view mean), something in the capture broke and the run would be
        fitting noise -- stop instead of collecting bad data.
        """
        S = {s: [] for s in sets}
        for i in range(n):
            r = self.evaluate(sets=sets, tag=f"noise{i}")
            for s in sets:
                S[s].append(r[f"S_{s}"])
        out = {s: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                   "min": float(np.min(v)), "max": float(np.max(v))}
               for s, v in S.items()}
        if verbose:
            for s, d in out.items():
                print(f"  [noise] set {s}: mean {d['mean']:.4f}  sd {d['sd']:.4f}  "
                      f"range {d['max'] - d['min']:.4f}  (n={n})")
        return out

    def restore(self):
        """Put the character back the way it was found. Call this from a `finally`."""
        if self.p0 is not None:
            try:
                self.set_shapes(self.p0)
            except (SystemExit, Exception) as e:      # never mask the original failure
                print(f"  [warn] could not restore p0: {str(e)[:80]}")

    def timing(self) -> str:
        n = max(self.n_eval, 1)
        return (f"{self.n_eval} evals, {self.n_render} renders | "
                f"render {self.t_render / n * 1000:.0f} ms/eval, "
                f"score {self.t_score / n * 1000:.0f} ms/eval | "
                f"slider path: {'batch' if self._batch else 'per-slider'}")


def slider_names():
    """Names as the game reports them, for readable output."""
    return [s["name"] for s in sorted(call("/maker/face/shapes")["shapes"],
                                      key=lambda s: s["index"])][:N_SLIDERS]
