"""card -> offline 3D face mesh. Thin wrapper over hs2_mesh_deform using the existing FaceData reader.

    from src.hs2_mesh import card_to_mesh
    verts, faces = card_to_mesh("outputs/xxx_out.png")

Pure offline (no game): see docs/hs2-renderer-and-mesh.md. The rig is per-head — a card's `headId`
picks both the mesh and its shapeValueFace keyframe table — so each head must be extracted once
(scripts/hs2_extract_head.py --card <card>) before it can be built.
"""
from __future__ import annotations

import numpy as np

from .hs2_mesh_deform import HeadRig, available_heads, build_mesh

_RIGS: dict[int, HeadRig] = {}


def _rig(head_id: int | None = None) -> HeadRig:
    """Cached rig for `head_id`. With no id, uses the only extracted head (else raises)."""
    if head_id is None:
        heads = available_heads()
        if len(heads) != 1:
            raise ValueError(f"head_id required — extracted heads: {heads}")
        head_id = heads[0]
    if head_id not in _RIGS:
        _RIGS[head_id] = HeadRig(head_id)
    return _RIGS[head_id]


def card_head_id(card_path: str) -> int:
    """The card's headId — which head mesh/rig the game would load for it."""
    from .face_data_utils.utils import AiSyoujyoCharaData
    return int(AiSyoujyoCharaData.load(card_path, True).Custom["face"]["headId"])


def fd_to_inputs(fd):
    """(shape_face(59), ab_data) from a loaded FaceData — the inputs build_mesh / landmarks consume.

    Reads fd.base_data (not card_data.Custom): at load it equals the stored shapeValueFace, and unlike
    card_data.Custom it also reflects an in-memory set_from_vector (which only writes base_data/ab_data).

    ...for the first 54. `FaceData.base_data` deliberately stops there ("without ear data") because
    the ML LABEL vector excludes the five ear knobs — but the RIG is not the label. Categories 54-58
    drive `cf_s_Ear*`, and `_fk_world` OVERWRITES a driven bone's rest transform with its source
    bone's value, so an absent slider does not leave the ear at rest: it writes the default
    (pos 0 / rot 0 / scl 1) over the rest pose, which is not the same thing as sampling the
    keyframes at the card's value — even when that value is the neutral 0.5. That silently
    collapsed and crumpled the ears. Read the ear block straight off the card; the model never
    writes it, so base_data has nothing newer to offer.
    """
    from .face_data_utils.utils import BONE_NAME_LIST
    shape_face = list(fd.base_data[:54])
    shape_face += [float(v) for v in fd.card_data.Custom["face"]["shapeValueFace"][54:59]]
    ab_data = {}
    for name in BONE_NAME_LIST:
        ab = fd.ab_data.get(name)
        if ab is None:
            continue
        ab_data[name] = {"scale": list(ab.scale), "length": float(ab.length),
                         "position": list(ab.position), "rotation": list(ab.rotation)}
    return shape_face, ab_data


def card_to_mesh(card_path: str):
    """Return (verts (V,3), faces (F,3)) for the head described by a HS2 card."""
    from .face_data_utils.utils import FaceData
    fd = FaceData(card_path)
    shape_face, ab_data = fd_to_inputs(fd)
    verts, faces = build_mesh(_rig(card_head_id(card_path)), shape_face=shape_face, ab_data=ab_data)
    return verts, faces


def save_obj(verts, faces, path: str):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/yua_desmile08_out.png")
    ap.add_argument("--out", default="outputs/yua_desmile08_head.obj")
    args = ap.parse_args()

    rig = _rig()
    neutral, _ = build_mesh(rig, shape_face=None, ab_data=None)
    verts, faces = card_to_mesh(args.card)

    assert np.isfinite(verts).all(), "non-finite vertices!"
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    save_obj(verts, faces, args.out)

    d = np.linalg.norm(verts - neutral, axis=1)
    bb = verts.max(0) - verts.min(0)
    print(f"card: {args.card}")
    print(f"verts={len(verts)} faces={len(faces)}  finite={np.isfinite(verts).all()}")
    print(f"bbox(WxHxD)={bb.round(4).tolist()}")
    print(f"deform vs neutral: max={d.max():.4f} mean={d.mean():.4f} moved>{0.001}: {(d>1e-3).sum()} verts")
    print(f"saved -> {args.out}")
