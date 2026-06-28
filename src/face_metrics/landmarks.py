"""Canonical facial landmarks + backends.

A *canonical landmark* is an anatomical point (eye inner corner, nose tip, …) named once and resolved by
either backend:
  - MESH (character side): the world position of a named rig bone after deform stages 1-4 — see
    src/hs2_mesh_deform.bone_world. The rig's bones ARE the anthropometric landmarks, so no vertex
    hand-labeling is needed, and the mesh is always expression-neutral.
  - DLIB68 (photo side, Phase 2): the matching iBUG/dlib 68-point index (or mean of indices).

Both backends emit the SAME canonical dict {name: (x,y[,z])}, so ratios (ratios.py) are defined once.
Coordinates: x = subject left(-)/right(+), y = up(+), z = forward(+). Mesh is frontal & axis-aligned, so
ratios use the (x,y) projection; z is used only by the mesh-only nose-projection ratio.
"""
from __future__ import annotations

import numpy as np

# canonical point -> mesh rig bone (its world translation = the landmark)
MESH_BONE = {
    "EYE_INNER_L": "cf_J_Eye01_s_L", "EYE_INNER_R": "cf_J_Eye01_s_R",
    "EYE_OUTER_L": "cf_J_Eye03_s_L", "EYE_OUTER_R": "cf_J_Eye03_s_R",
    "EYE_TOP_L":   "cf_J_Eye02_s_L", "EYE_TOP_R":   "cf_J_Eye02_s_R",
    "EYE_BOT_L":   "cf_J_Eye04_s_L", "EYE_BOT_R":   "cf_J_Eye04_s_R",
    "NASION":      "cf_J_NoseBridge_s",
    "NOSE_TIP":    "cf_J_Nose_tip",
    "SUBNASALE":   "cf_J_NoseBase_s",
    "NOSE_WING_L": "cf_J_NoseWing_tx_L", "NOSE_WING_R": "cf_J_NoseWing_tx_R",
    "MOUTH_L":     "cf_J_Mouth_L", "MOUTH_R": "cf_J_Mouth_R",
    "LIP_UP":      "cf_J_Mouthup",
    "LIP_LOW":     "cf_J_MouthLow",
    "CHEEK_L":     "cf_J_CheekUp_L", "CHEEK_R": "cf_J_CheekUp_R",
    "JAW_L":       "cf_J_CheekLow_L", "JAW_R": "cf_J_CheekLow_R",
    "BROW_L":      "cf_J_Mayu_L", "BROW_R": "cf_J_Mayu_R",
    "FACE_W_L":    "cf_J_EarBase_s_L", "FACE_W_R": "cf_J_EarBase_s_R",
    "GNATHION":    "cf_J_ChinTip_s",
}

# dlib / iBUG-68 indices (0-based), subject-anatomical. int -> that point; tuple -> mean of those points.
# GLABELLA / GNATHION exist here; on the mesh GLABELLA is synthesized (no inner-brow bone).
DLIB68 = {
    "EYE_INNER_L": 42, "EYE_INNER_R": 39,
    "EYE_OUTER_L": 45, "EYE_OUTER_R": 36,
    "EYE_TOP_L": (43, 44), "EYE_TOP_R": (37, 38),
    "EYE_BOT_L": (46, 47), "EYE_BOT_R": (40, 41),
    "NASION": 27, "NOSE_TIP": 30, "SUBNASALE": 33,
    "NOSE_WING_L": 35, "NOSE_WING_R": 31,
    "MOUTH_L": 54, "MOUTH_R": 48, "LIP_UP": 51, "LIP_LOW": 57,
    "CHEEK_L": 14, "CHEEK_R": 2, "JAW_L": 12, "JAW_R": 4,
    "BROW_L": (24, 25), "BROW_R": (18, 19),
    "FACE_W_L": 16, "FACE_W_R": 0,
    "GNATHION": 8, "GLABELLA": (21, 22),
}


def mesh_landmarks(rig, shape_face=None, ab_data=None):
    """{canonical: (x,y,z)} from named rig bones (deform stages 1-4, no skinning)."""
    from ..hs2_mesh_deform import bone_world
    pos = bone_world(rig, shape_face, ab_data, list(set(MESH_BONE.values())))
    lm = {name: pos[bone].astype(float) for name, bone in MESH_BONE.items()}
    # GLABELLA: no inner-brow bone -> synthesize a midline (x=0) point at brow height
    brow = 0.5 * (lm["BROW_L"] + lm["BROW_R"])
    lm["GLABELLA"] = np.array([0.0, brow[1], brow[2]])
    return lm


def card_landmarks(card_path):
    """{canonical: (x,y,z)} for the head described by a HS2 card."""
    from ..hs2_mesh import _rig, fd_to_inputs
    from ..face_data_utils.utils import FaceData
    fd = FaceData(card_path)
    shape_face, ab_data = fd_to_inputs(fd)
    return mesh_landmarks(_rig(), shape_face, ab_data)


def _pt68(pts, spec):
    return pts[spec].astype(float) if isinstance(spec, int) else pts[list(spec)].mean(0)


def _canonical_from_68(pts):
    """68x2 image-coord landmarks -> {canonical: (x,y)}, y-flipped to y-up and de-rolled by inner-canthi."""
    pts = np.asarray(pts, dtype=float)[:, :2].copy()
    pts[:, 1] *= -1.0  # image y is down; flip to match the mesh's y-up convention
    inl, inr = pts[42], pts[39]  # dlib inner canthi (subject left, right)
    ang = np.arctan2(inl[1] - inr[1], inl[0] - inr[0])
    c, s = np.cos(-ang), np.sin(-ang)
    pts = pts @ np.array([[c, -s], [s, c]]).T  # rotate so the inter-canthal line is horizontal
    return {name: _pt68(pts, spec) for name, spec in DLIB68.items()}


def photo_landmarks(img_path):
    """{canonical: (x,y)} from a real photo via face_alignment-68.

    Lazy-imports face_alignment (heavy; downloads weights once). Scale is irrelevant (ratios are
    dimensionless); de-roll removes in-plane roll only — large yaw/pitch is not corrected, so use frontal
    photos (see report header caveats).
    """
    import cv2
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # CJK-safe
    if img is None:
        raise FileNotFoundError(f"could not read image: {img_path}")
    preds = _get_fa().get_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not preds:
        raise RuntimeError(f"no face detected in {img_path}")
    return _canonical_from_68(preds[0])


def render_landmarks(card_path, yaw=0.0):
    """Char-side landmarks by running the SAME 68-detector on a render of the card.

    Most comparable to photo_landmarks (one detector both sides -> definitional bias cancels), unlike the
    bone backend. Reuses the numpy rasterizer in scripts/hs2_render_mesh.
    """
    import cv2
    from ..hs2_mesh import card_to_mesh
    from scripts.hs2_render_mesh import render
    verts, faces = card_to_mesh(card_path)
    gray = render(verts, faces, yaw=yaw)  # (H,W) uint8
    preds = _get_fa().get_landmarks(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    if not preds:
        raise RuntimeError("face_alignment found no face in the render; use the default bone backend")
    return _canonical_from_68(preds[0])


_FA = None


def _get_fa():
    global _FA
    if _FA is None:
        import face_alignment
        if not hasattr(face_alignment, "FaceAlignment"):
            raise RuntimeError(
                "Phase-2 fidelity needs a 68-point face-landmark detector, but `import face_alignment` "
                f"resolves to the local repo file ({getattr(face_alignment, '__file__', '?')}, an mtcnn "
                "5-point shim) — not the bulat `face-alignment` library. Install a real 68-pt detector "
                "(e.g. mediapipe) to enable --photo. Harmony/compare work without it.")
        try:                       # no MSVC `cl` on this box -> fall back to eager, don't crash on compile
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
        try:                       # API differs across versions
            lt = face_alignment.LandmarksType.TWO_D
        except AttributeError:
            lt = face_alignment.LandmarksType._2D
        _FA = face_alignment.FaceAlignment(lt, flip_input=False, device="cpu")
    return _FA
