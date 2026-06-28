"""Scale-invariant anthropometric ratios over the canonical landmark dict (landmarks.py).

Each ratio is a pure function of {canonical: (x,y[,z])}, so the SAME definition serves the mesh and photo
backends. Distances are 2D (x,y); xspan uses |Δx|, ydiff uses |Δy|. Normalizers: ICD = inner-canthal
distance, FW = face width (ear-base span).

Tags:
  computable: "mesh" (needs z or the rig, photo can't) or "both"
  expr:       True if expression-sensitive (mouth/lip/eye-openness) — excluded by STRUCTURAL
  weak:       True if the photo (dlib-68) landmark is noisy (cheek/jaw contour) — excluded by STRUCTURAL

Subsets: HARMONY = everything mesh can compute (mesh is neutral, so expr ratios are valid resting-shape
features); FULL = everything the photo can compute; STRUCTURAL = FULL minus expr & weak (robust fidelity
against a smiling/imperfect photo).
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

Ratio = namedtuple("Ratio", "name fn computable expr weak")


def _d(lm, a, b):       # 2D distance
    return float(np.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1]))


def _xspan(lm, a, b):
    return float(abs(lm[a][0] - lm[b][0]))


def _ydiff(lm, a, b):
    return float(abs(lm[a][1] - lm[b][1]))


def _meany(lm, a, b):
    return 0.5 * (lm[a][1] + lm[b][1])


def _icd(lm):
    return _d(lm, "EYE_INNER_L", "EYE_INNER_R")


def _fw(lm):
    return _xspan(lm, "FACE_W_L", "FACE_W_R")


def _eyew(lm):          # mean left/right palpebral-fissure width
    return 0.5 * (_xspan(lm, "EYE_OUTER_L", "EYE_INNER_L") + _xspan(lm, "EYE_OUTER_R", "EYE_INNER_R"))


_SYM_PAIRS = [("EYE_INNER_L", "EYE_INNER_R"), ("EYE_OUTER_L", "EYE_OUTER_R"),
              ("NOSE_WING_L", "NOSE_WING_R"), ("MOUTH_L", "MOUTH_R"),
              ("CHEEK_L", "CHEEK_R"), ("JAW_L", "JAW_R"), ("BROW_L", "BROW_R")]


def _symmetry(lm):      # mesh: _L has -x, _R has +x, so |x_L + x_R| ~ 0 when symmetric
    return float(np.mean([abs(lm[a][0] + lm[b][0]) for a, b in _SYM_PAIRS])) / _fw(lm)


RATIOS = [
    Ratio("face_height/width",   lambda lm: _ydiff(lm, "GLABELLA", "GNATHION") / _fw(lm), "both", False, False),
    Ratio("morpho_height/width", lambda lm: _ydiff(lm, "NASION", "GNATHION") / _fw(lm),   "both", False, False),
    Ratio("icd/facewidth",       lambda lm: _icd(lm) / _fw(lm),                            "both", False, False),
    Ratio("biocular/facewidth",  lambda lm: _d(lm, "EYE_OUTER_L", "EYE_OUTER_R") / _fw(lm),"both", False, False),
    Ratio("eyewidth/icd",        lambda lm: _eyew(lm) / _icd(lm),                          "both", False, False),
    Ratio("eyewidth/facewidth",  lambda lm: _eyew(lm) / _fw(lm),                           "both", False, False),
    Ratio("eye_aspect",          lambda lm: 0.5 * (_ydiff(lm, "EYE_TOP_L", "EYE_BOT_L")
                                                    + _ydiff(lm, "EYE_TOP_R", "EYE_BOT_R")) / _eyew(lm),
                                 "both", True, False),
    Ratio("eye_vert_level",      lambda lm: abs(lm["GLABELLA"][1] - _meany(lm, "EYE_INNER_L", "EYE_INNER_R"))
                                            / _ydiff(lm, "GLABELLA", "GNATHION"),
                                 "both", False, False),
    Ratio("nosewidth/facewidth", lambda lm: _xspan(lm, "NOSE_WING_L", "NOSE_WING_R") / _fw(lm),  "both", False, False),
    Ratio("nosewidth/icd",       lambda lm: _xspan(lm, "NOSE_WING_L", "NOSE_WING_R") / _icd(lm), "both", False, False),
    Ratio("noselength/facewidth",lambda lm: _ydiff(lm, "NASION", "SUBNASALE") / _fw(lm),          "both", False, False),
    Ratio("noselength/midface",  lambda lm: _ydiff(lm, "NASION", "SUBNASALE") / _ydiff(lm, "NASION", "GNATHION"),
                                 "both", False, False),
    Ratio("nose_projection",     lambda lm: (lm["NOSE_TIP"][2] - lm["SUBNASALE"][2]) / _fw(lm), "mesh", False, False),
    Ratio("mouthwidth/facewidth",lambda lm: _xspan(lm, "MOUTH_L", "MOUTH_R") / _fw(lm),  "both", True, False),
    Ratio("mouthwidth/icd",      lambda lm: _xspan(lm, "MOUTH_L", "MOUTH_R") / _icd(lm), "both", True, False),
    Ratio("mouth/nose_width",    lambda lm: _xspan(lm, "MOUTH_L", "MOUTH_R") / _xspan(lm, "NOSE_WING_L", "NOSE_WING_R"),
                                 "both", True, False),
    Ratio("lipheight/mouthwidth",lambda lm: _ydiff(lm, "LIP_UP", "LIP_LOW") / _xspan(lm, "MOUTH_L", "MOUTH_R"),
                                 "both", True, False),
    Ratio("mid:lower_thirds",    lambda lm: _ydiff(lm, "GLABELLA", "SUBNASALE") / _ydiff(lm, "SUBNASALE", "GNATHION"),
                                 "both", False, False),
    Ratio("lowerface_fraction",  lambda lm: _ydiff(lm, "SUBNASALE", "GNATHION") / _ydiff(lm, "GLABELLA", "GNATHION"),
                                 "both", False, False),
    Ratio("chinheight/lowerface",lambda lm: _ydiff(lm, "LIP_LOW", "GNATHION") / _ydiff(lm, "SUBNASALE", "GNATHION"),
                                 "both", False, False),
    Ratio("cheekbone/facewidth", lambda lm: _xspan(lm, "CHEEK_L", "CHEEK_R") / _fw(lm), "both", False, True),
    Ratio("jaw/facewidth",       lambda lm: _xspan(lm, "JAW_L", "JAW_R") / _fw(lm),     "both", False, True),
    Ratio("jaw/cheekbone_taper", lambda lm: _xspan(lm, "JAW_L", "JAW_R") / _xspan(lm, "CHEEK_L", "CHEEK_R"),
                                 "both", False, True),
    Ratio("browheight/icd",      lambda lm: abs(_meany(lm, "BROW_L", "BROW_R") - _meany(lm, "EYE_TOP_L", "EYE_TOP_R"))
                                            / _icd(lm),
                                 "both", False, False),
    Ratio("symmetry_index",      _symmetry, "mesh", False, False),
]


def select(subset):
    if subset == "HARMONY":
        return [r for r in RATIOS if r.computable in ("mesh", "both")]
    if subset == "FULL":
        return [r for r in RATIOS if r.computable == "both"]
    if subset == "STRUCTURAL":
        return [r for r in RATIOS if r.computable == "both" and not r.expr and not r.weak]
    raise ValueError(f"unknown subset: {subset!r} (use HARMONY / FULL / STRUCTURAL)")


def ratio_vector(lm, subset):
    """Return (values (K,), names [K]) for the ratios in `subset`."""
    rs = select(subset)
    return np.array([r.fn(lm) for r in rs], dtype=float), [r.name for r in rs]
