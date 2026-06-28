"""Scoring + human-readable reports. Interpretable diagnostics only — never a fake "8.2/10".

  harmony(card)        -> typicality of one card's ratios vs the reference population
  compare(cardA,cardB) -> per-ratio signed delta (in reference-sigma units) between two cards
  fidelity(card,photo) -> per-ratio delta between a card and an input photo (Phase 2)
"""
from __future__ import annotations

import os

import numpy as np

from .landmarks import card_landmarks
from .ratios import ratio_vector
from .reference import load_reference


def _aligned(lm, ref):
    """Ratio vector for `lm` reordered to the reference's stored ratio order."""
    vals, names = ratio_vector(lm, ref["subset"])
    by = dict(zip(names, vals))
    return np.array([by[n] for n in ref["ratio_names"]], dtype=float)


def harmony(card, ref=None):
    ref = ref or load_reference()
    x = _aligned(card_landmarks(card), ref)
    z = (x - ref["median"]) / ref["mad"]
    d = x - ref["mean"]
    D = float(np.sqrt(d @ ref["cov_inv"] @ d))
    pct = 100.0 * float(np.mean(ref["maha_sorted"] <= D))
    return {"names": ref["ratio_names"], "values": x, "z": z, "maha": D, "pctile": pct}


def compare(card_a, card_b, ref=None):
    ref = ref or load_reference()
    xa = _aligned(card_landmarks(card_a), ref)
    xb = _aligned(card_landmarks(card_b), ref)
    return {"names": ref["ratio_names"], "a": xa, "b": xb,
            "delta": xa - xb, "dsigma": (xa - xb) / ref["mad"]}


# ---------- text formatting ----------
def format_harmony(res, card):
    out = [f"HARMONY  {os.path.basename(card)}",
           f"  typicality: Mahalanobis at the {res['pctile']:.0f}th percentile "
           f"(low = more typical / coordinated; high = unusual combination)",
           "  per-ratio robust z  (|z|>2 = unusual vs population), sorted by |z|:"]
    for i in np.argsort(-np.abs(res["z"])):
        flag = "  <-- unusual" if abs(res["z"][i]) > 2 else ""
        out.append(f"    {res['names'][i]:<22} {res['values'][i]:8.3f}   z={res['z'][i]:+5.2f}{flag}")
    return "\n".join(out)


def fidelity(card, photo, subset="STRUCTURAL", render_backend=False, ref=None):
    """Per-ratio difference between a card and an input photo, in reference-sigma + percent units."""
    from .landmarks import card_landmarks, photo_landmarks, render_landmarks
    ref = ref or load_reference()
    mad_by = dict(zip(ref["ratio_names"], ref["mad"]))
    clm = render_landmarks(card) if render_backend else card_landmarks(card)
    cv, names = ratio_vector(clm, subset)
    pv, _ = ratio_vector(photo_landmarks(photo), subset)
    delta = cv - pv
    dsigma = np.array([delta[i] / mad_by.get(names[i], np.nan) for i in range(len(names))])
    return {"names": names, "char": cv, "photo": pv, "delta": delta, "dsigma": dsigma,
            "dpct": 100.0 * delta / pv, "subset": subset, "render_backend": render_backend}


def format_fidelity(res, card, photo):
    backend = "render-68" if res["render_backend"] else "mesh-bones"
    out = [f"FIDELITY  card={os.path.basename(card)}  vs  photo={os.path.basename(photo)}",
           f"  subset={res['subset']}  char-backend={backend}",
           "  caveats: photo has pose/expression/perspective; mesh is neutral & frontal. De-rolled by",
           "           inner-canthi (roll only) — use a frontal photo. STRUCTURAL drops expression ratios.",
           "  per-ratio  char - photo  (+ = char larger than the person), sorted by |sigma|:"]
    for i in np.argsort(-np.abs(np.nan_to_num(res["dsigma"]))):
        out.append(f"    {res['names'][i]:<22} char={res['char'][i]:7.3f} photo={res['photo'][i]:7.3f}"
                   f"   d={res['delta'][i]:+6.3f} ({res['dsigma'][i]:+5.2f}sigma, {res['dpct'][i]:+5.1f}%)")
    return "\n".join(out)


def format_compare(res, card_a, card_b):
    out = [f"COMPARE  A={os.path.basename(card_a)}   B={os.path.basename(card_b)}",
           "  per-ratio  A - B  in reference-sigma units, sorted by |delta|:"]
    for i in np.argsort(-np.abs(res["dsigma"])):
        out.append(f"    {res['names'][i]:<22} A={res['a'][i]:7.3f} B={res['b'][i]:7.3f}"
                   f"   d={res['delta'][i]:+7.3f}  ({res['dsigma'][i]:+5.2f}sigma)")
    return "\n".join(out)
