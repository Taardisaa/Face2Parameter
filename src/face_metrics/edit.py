"""Ratio-knob face editor: nudge a card's params to hit named facial-ratio targets.

Turns the facial-ratio metric (face_metrics) into an editor. The forward model params->ratios is exe-free
(set_from_vector -> hs2_mesh.fd_to_inputs -> landmarks.mesh_landmarks -> ratios.ratio_vector), so we
finite-difference a Jacobian and take damped, regularized Gauss-Newton steps over the normalized 205-dim
vector, holding non-targeted ratios and clamping params to [0,1] (min-max normalization => in-range == [0,1]).
without_right=True auto-mirrors left->right, so edits stay symmetric. See the plan + docs/facial-harmony-metric.md.
"""
from __future__ import annotations

import re

import numpy as np

from .landmarks import mesh_landmarks
from .ratios import ratio_vector

# ---------- masked-index <-> human name ----------
_PARAM_KIND = ["scale.x", "scale.y", "scale.z", "length",
               "pos.x", "pos.y", "pos.z", "rot.x", "rot.y", "rot.z"]
_KEPT = None
_BONES = None


def _kept():
    """Full-vector indices kept by is_simplify=True, without_right=True (the 205 masked positions)."""
    global _KEPT, _BONES
    if _KEPT is None:
        from ..face_data_utils.utils import BONE_NAME_LIST, ONLY_RIGHT_INDEX, PARAMETER_FLAGS
        nfull = len(BONE_NAME_LIST) * 10 + 54
        right = set(ONLY_RIGHT_INDEX)
        _KEPT = [i for i in range(nfull) if i not in right and PARAMETER_FLAGS[i] != 0]
        _BONES = BONE_NAME_LIST
    return _KEPT


def masked_param_name(m):
    full = _kept()[m]
    if full < 54:
        return f"shapeValueFace[{full}]"
    b, k = divmod(full - 54, 10)
    return f"{_BONES[b]}.{_PARAM_KIND[k]}"


# ---------- forward model ----------
def current_p(fd):
    """The card's current normalized 205-vec (same space as labels.json / infer)."""
    return np.asarray(fd.to_vector(is_simplify=True, without_right=True,
                                   normalize=True, use_gaussian=False), dtype=float)


def forward_ratios(rig, fd, p, subset, names_order):
    """Set p into fd (in place) and return the ratio vector in names_order."""
    from ..hs2_mesh import fd_to_inputs
    fd.set_from_vector(np.asarray(p, np.float32), is_simplify=True, without_right=True,
                       denormalize=True, use_gaussian=False)
    shape_face, ab_data = fd_to_inputs(fd)
    vals, names = ratio_vector(mesh_landmarks(rig, shape_face, ab_data), subset)
    by = dict(zip(names, vals))
    return np.array([by[n] for n in names_order], dtype=float)


def jacobian(rig, fd, p, subset, names_order, eps=1e-3, free_idx=None):
    """Forward-difference J (K x M) of ratios wrt p over free_idx; leaves fd set back to p."""
    r0 = forward_ratios(rig, fd, p, subset, names_order)
    M = len(p)
    free_idx = range(M) if free_idx is None else free_idx
    J = np.zeros((len(r0), M))
    for j in free_idx:
        pj = p.copy()
        pj[j] = pj[j] + eps if p[j] + eps <= 1.0 else p[j] - eps   # probe inward at the upper bound
        step = pj[j] - p[j]
        if step == 0:
            continue
        J[:, j] = (forward_ratios(rig, fd, pj, subset, names_order) - r0) / step
    forward_ratios(rig, fd, p, subset, names_order)                # restore fd state to p
    return r0, J


# ---------- targets ----------
_TARGET_RE = re.compile(r"^\s*([\w:/]+)\s*=\s*([+-]?)\s*([0-9.]+)\s*(%|sigma|σ|s)?\s*$")

PRESETS = {
    "thinner":      [("face_height/width", +1.0), ("morpho_height/width", +0.7)],
    "wider":        [("face_height/width", -1.0), ("morpho_height/width", -0.7)],
    "eyes-closer":  [("icd/facewidth", -1.0)],
    "eyes-wider":   [("icd/facewidth", +1.0)],
    "bigger-eyes":  [("eyewidth/facewidth", +1.0), ("eye_aspect", +0.6)],
    "smaller-eyes": [("eyewidth/facewidth", -1.0), ("eye_aspect", -0.6)],
    "longer-face":  [("face_height/width", +1.0), ("lowerface_fraction", +0.4)],
    "rounder-face": [("face_height/width", -1.0)],
    "slimmer-jaw":  [("jaw/facewidth", -1.0), ("jaw/cheekbone_taper", -0.7)],
    "narrower-nose":[("nosewidth/facewidth", -1.0)],
}


def parse_target(spec, names, r0_by, mad_by):
    """'<ratio>=<expr>' -> (name, goal_value). expr: 0.85 | +10% | -1.5sigma | +0.03(additive)."""
    m = _TARGET_RE.match(spec)
    if not m:
        raise ValueError(f"bad --set '{spec}' (expected name=VALUE[%|sigma])")
    name, sign, mag, unit = m.group(1), m.group(2), float(m.group(3)), m.group(4)
    if name not in names:
        raise ValueError(f"unknown ratio '{name}'. valid: {', '.join(names)}")
    signed = -mag if sign == "-" else mag
    r0 = r0_by[name]
    if unit in ("sigma", "σ", "s"):
        goal = r0 + signed * mad_by[name]
    elif unit == "%":
        goal = r0 * (1 + signed / 100.0)
    elif sign:                       # bare additive delta, e.g. +0.03
        goal = r0 + signed
    else:                            # absolute target, e.g. 0.85
        goal = mag
    return name, goal


def expand_presets(active, strength, r0_by, mad_by):
    """active preset names -> {ratio: goal}. Each preset step is in sigma * strength."""
    goals = {}
    for name in active:
        for ratio, sig in PRESETS[name]:
            goals[ratio] = r0_by[ratio] + sig * strength * mad_by[ratio]
    return goals


# ---------- harmony plausibility ----------
def harmony_pctile(r_named, names, ref):
    by = dict(zip(names, r_named))
    x = np.array([by[n] for n in ref["ratio_names"]], dtype=float)
    d = x - ref["mean"]
    D = float(np.sqrt(d @ ref["cov_inv"] @ d))
    return 100.0 * float(np.mean(ref["maha_sorted"] <= D))


# ---------- solver ----------
def solve(rig, fd, p0, targets, ref, lam=1e-3, hold_weight=0.05, eps=1e-3,
          max_iters=10, tol_sigma=0.05, max_step=0.5, free_idx=None):
    """targets: {ratio_name: goal}. Returns (p1, info). Uses the reference's ratio order/subset/MAD."""
    names = list(ref["ratio_names"])
    subset = ref["subset"]
    mad = np.asarray(ref["mad"], float)
    idx_of = {n: i for i, n in enumerate(names)}
    K, M = len(names), len(p0)
    free_idx = list(range(M)) if free_idx is None else list(free_idx)

    valid = mad > 1e-6                                   # near-constant ratios (e.g. symmetry) -> no weight
    tgt_mask = np.zeros(K, bool)
    for n in targets:
        tgt_mask[idx_of[n]] = True
    w = np.where(tgt_mask, 1.0, hold_weight) / np.where(valid, mad, np.inf)
    w[~valid] = 0.0
    W2 = w * w

    p = p0.copy()
    r_before = forward_ratios(rig, fd, p, subset, names)
    goalv = r_before.copy()
    for n, g in targets.items():
        goalv[idx_of[n]] = g

    J0, history = None, []
    for it in range(max_iters):
        r0, J = jacobian(rig, fd, p, subset, names, eps=eps, free_idx=free_idx)
        if J0 is None:
            J0 = J
        resid = goalv - r0
        A = (J.T * W2) @ J + lam * np.eye(M)
        dp = np.linalg.solve(A, (J.T * W2) @ resid)
        obj0 = float(np.sum(W2 * resid ** 2))
        alpha, accepted, pn = 1.0, False, p
        for _ in range(4):
            cand = np.clip(p + alpha * dp, 0.0, 1.0)
            if np.sum(W2 * (goalv - forward_ratios(rig, fd, cand, subset, names)) ** 2) < obj0:
                pn, accepted = cand, True
                break
            alpha *= 0.5
        if not accepted:
            break
        p = pn
        rcur = forward_ratios(rig, fd, p, subset, names)
        tsig = np.abs((goalv - rcur) / np.where(valid, mad, np.inf))[tgt_mask]
        history.append({"iter": it, "obj": float(np.sum(W2 * (goalv - rcur) ** 2)),
                        "max_tgt_sigma": float(tsig.max()) if tsig.size else 0.0})
        if (tsig.size == 0 or tsig.max() < tol_sigma) or np.max(np.abs(alpha * dp)) < 1e-4:
            break

    capped = False
    disp = p - p0
    nrm = float(np.linalg.norm(disp))
    if nrm > max_step:
        p = np.clip(p0 + (max_step / nrm) * disp, 0.0, 1.0)
        disp = p - p0
        capped = True
    r_after = forward_ratios(rig, fd, p, subset, names)

    leverage = {n: float(np.linalg.norm(J0[idx_of[n], free_idx])) for n in targets}
    info = {
        "names": names, "mad": mad, "tgt_mask": tgt_mask, "targets": dict(targets),
        "r_before": r_before, "r_after": r_after, "goalv": goalv, "valid": valid,
        "p0": p0, "p1": p, "disp": disp, "free_idx": free_idx, "leverage": leverage,
        "dp_norm": float(np.linalg.norm(disp)), "capped": capped,
        "n_clamped": int(np.sum((p <= 1e-9) | (p >= 1 - 1e-9))), "history": history,
        "harmony_before": harmony_pctile(r_before, names, ref),
        "harmony_after": harmony_pctile(r_after, names, ref),
    }
    return p, info


def format_edit_report(info, card, out, mode):
    names, mad = info["names"], info["mad"]
    idx = {n: i for i, n in enumerate(names)}
    L = [f"EDIT  {card}  ->  {out}",
         f"  mode={mode}  iters={len(info['history'])}  ||dp||={info['dp_norm']:.3f}"
         f"  clamped={info['n_clamped']}" + ("  [max_step capped]" if info["capped"] else ""),
         f"  harmony percentile: {info['harmony_before']:.0f} -> {info['harmony_after']:.0f}"
         "  (low=typical; big jump = less plausible)",
         "  TARGETS  (before -> after / goal, residual in sigma):"]
    for n, g in info["targets"].items():
        i = idx[n]
        b, a = info["r_before"][i], info["r_after"][i]
        res_sig = (g - a) / mad[i] if mad[i] > 1e-6 else float("nan")
        lev = info["leverage"][n]
        status = ("infeasible(no leverage)" if lev < 1e-4
                  else "OK" if abs(res_sig) < 0.1 else "partial")
        L.append(f"    {n:<22} {b:7.3f} -> {a:7.3f} / {g:7.3f}   resid={res_sig:+5.2f}sig  {status}")

    # held-ratio collateral drift
    leaks = []
    for n in names:
        if info["tgt_mask"][idx[n]] or not info["valid"][idx[n]]:
            continue
        d = (info["r_after"][idx[n]] - info["r_before"][idx[n]]) / mad[idx[n]]
        if abs(d) > 0.5:
            leaks.append((n, d))
    if leaks:
        L.append("  COLLATERAL DRIFT (held ratios moved >0.5 sigma):")
        for n, d in sorted(leaks, key=lambda x: -abs(x[1])):
            L.append(f"    {n:<22} {d:+5.2f}sigma")

    # top params moved
    order = np.argsort(-np.abs(info["disp"]))
    L.append("  PARAMS MOVED (top, normalized before -> after):")
    for m in order[:10]:
        if abs(info["disp"][m]) < 1e-3:
            break
        L.append(f"    {masked_param_name(m):<28} {info['p0'][m]:6.3f} -> {info['p1'][m]:6.3f}"
                 f"  ({info['disp'][m]:+.3f})")
    return "\n".join(L)
