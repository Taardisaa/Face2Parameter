"""card -> offline 3D face mesh. Thin wrapper over hs2_mesh_deform using the existing FaceData reader.

    from src.hs2_mesh import card_to_mesh
    verts, faces = card_to_mesh("outputs/xxx_out.png")

Pure offline (no game): see docs/hs2-renderer-and-mesh.md. v1 supports the stock female head (headId via
fo_head_00 / cf_anmShapeHead_00); other heads need their own extracted rig.
"""
from __future__ import annotations

import numpy as np

from .hs2_mesh_deform import HeadRig, build_mesh

_RIG = None


def _rig() -> HeadRig:
    global _RIG
    if _RIG is None:
        _RIG = HeadRig()
    return _RIG


def fd_to_inputs(fd):
    """(shape_face(54), ab_data) from a loaded FaceData — the inputs build_mesh / landmarks consume.

    Reads fd.base_data (not card_data.Custom): at load it equals the stored shapeValueFace, and unlike
    card_data.Custom it also reflects an in-memory set_from_vector (which only writes base_data/ab_data).
    """
    from .face_data_utils.utils import BONE_NAME_LIST
    shape_face = list(fd.base_data[:54])
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
    verts, faces = build_mesh(_rig(), shape_face=shape_face, ab_data=ab_data)
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
