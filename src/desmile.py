"""Parameter-space smile neutralization (no models, no GPU, deterministic).

Every photo of the subject is smiling, so the regression bakes the smile into the *static* face
shape — raised mouth corners and puffed apple cheeks. Instead of editing the input image (the FLUX /
LivePortrait detour), we relax it directly in parameter space: after prediction, pull the smile-related
bones toward their population mean (the neutral resting face), by a factor ``alpha``.

This is allowed because we no longer require exact identity preservation — a slightly adjusted face is
fine. ``alpha=0`` is a no-op; ``alpha=1`` fully replaces the smile bones with the dataset-average
(neutral) values; values in between blend.

Only mouth + cheek bones are touched (the smile carriers). Eyes are left alone — eye geometry is strong
identity and not where a closed-mouth smile mainly lives.
"""

from __future__ import annotations

from .face_data_utils.utils import STATISTICAL_BONE_DATA

# Smile lives here: mouth corners/shape + the apple-cheek / lower-cheek volume.
SMILE_BONES = (
    "cf_J_Mouth_L", "cf_J_Mouth_R", "cf_J_Mouthup", "cf_J_MouthLow",
    "cf_J_MouthBase_tr", "cf_J_MouthMove",
    "cf_J_CheekUp_L", "cf_J_CheekUp_R", "cf_J_CheekLow_L", "cf_J_CheekLow_R",
)

_VEC3 = ("scale", "position", "rotation")  # blended component-wise (x/y/z)

# A component's stored mean is only trustworthy if enough cards actually customized it. A low ``count``
# means "almost nobody sets this, so the neutral value is the rig DEFAULT, not the mean of the few
# outliers who did" — and some are count=0 with mean=0.0 (e.g. MouthMove length), which would otherwise
# zero out a length. So we only pull components with a decent sample size; the rest are left untouched.
# At the default threshold this keeps all `scale` (counts in the thousands, where the smile lives) plus
# the well-sampled mouth-corner rotations, and drops the noisy position/length means.
MIN_COUNT = 100


def _blend(cur, target, alpha):
    return (1.0 - alpha) * float(cur) + alpha * float(target)


def _reliable(stat_comp) -> bool:
    return (isinstance(stat_comp, dict) and "mean" in stat_comp
            and stat_comp.get("count", 0) >= MIN_COUNT)


def desmile_face(face_data, alpha: float = 1.0, bones=SMILE_BONES) -> list:
    """Blend each smile bone's deformation toward its population mean, in place.

    Operates on ``face_data.ab_data`` (a {bone_name: ABData} dict) AFTER ``set_from_vector`` has
    written the predicted, denormalized values. Only well-sampled components (count >= MIN_COUNT) are
    moved; sparsely-customized ones keep the predicted value. Returns the list of bones adjusted.
    """
    if alpha <= 0.0:
        return []
    touched = []
    for name in bones:
        ab = face_data.ab_data.get(name)
        stat = STATISTICAL_BONE_DATA.get(name)
        if ab is None or stat is None:
            continue
        changed = False
        for kind in _VEC3:
            comp = stat.get(kind)
            if not comp:
                continue
            cur = list(getattr(ab, kind))
            new = list(cur)
            for i, axis in enumerate(("x", "y", "z")):
                if _reliable(comp.get(axis)):
                    new[i] = _blend(cur[i], comp[axis]["mean"], alpha)
                    changed = True
            setattr(ab, kind, new)
        if _reliable(stat.get("length")):
            ab.length = _blend(ab.length, stat["length"]["mean"], alpha)
            changed = True
        if changed:
            touched.append(name)
    return touched
