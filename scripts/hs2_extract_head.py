"""Extract the HS2 female head rig needed for the offline face-mesh constructor.

Pulls everything the deform needs (pure offline, via UnityPy) from the game's asset bundles and caches
it under data/hs2_head/:
  - o_head mesh: vertices, triangles, per-vertex bone indices+weights, bindposes, skin-bone order
  - the head bone hierarchy (name, parent, REST local pos/rot/scale) for FK
  - cf_anmShapeHead_00: per-cf_s_-bone keyframe table (binary) -> JSON
  - cf_customhead: slider(category) -> cf_s_ bone + DOF-use flags (CSV) -> JSON

See docs/hs2-renderer-and-mesh.md for the reverse-engineered pipeline. Run with the project venv:
    .venv/Scripts/python.exe scripts/hs2_extract_head.py
"""
from __future__ import annotations

import io
import json
import os
import struct

import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

HS2 = os.environ.get("HS2_DIR", r"E:\HoneySelect2_ArcticFox")
AB = os.path.join(HS2, "abdata")
FO_HEAD = os.path.join(AB, "chara", "00", "fo_head_00.unity3d")
CUSTOMSHAPE = os.path.join(AB, "list", "customshape.unity3d")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hs2_head")


def _name_of(transform):
    """GameObject name for a Transform object reader-result."""
    try:
        return transform.m_GameObject.read().m_Name
    except Exception:
        return None


def extract_mesh_and_skeleton():
    env = UnityPy.load(FO_HEAD)
    objs = list(env.objects)

    # index Transforms by path_id for hierarchy walking
    trans = {}   # path_id -> dict(name, parent_pid, pos, rot(xyzw), scale)
    raw_t = {}   # path_id -> typed transform (for father/children)
    for o in objs:
        if o.type.name != "Transform":
            continue
        t = o.read()
        pid = o.path_id
        father_pid = None
        try:
            father_pid = t.m_Father.path_id if t.m_Father and t.m_Father.path_id else None
        except Exception:
            father_pid = None
        lp, lr, ls = t.m_LocalPosition, t.m_LocalRotation, t.m_LocalScale
        trans[pid] = {
            "name": _name_of(t),
            "parent": father_pid,
            "pos": [lp.x, lp.y, lp.z],
            "rot": [lr.x, lr.y, lr.z, lr.w],
            "scale": [ls.x, ls.y, ls.z],
        }
        raw_t[pid] = t

    # find the MAIN o_head SMR (mesh name o_head, max vertex count)
    best = None  # (vcount, smr, mesh)
    for o in objs:
        if o.type.name != "SkinnedMeshRenderer":
            continue
        smr = o.read()
        try:
            mesh = smr.m_Mesh.read()
        except Exception:
            continue
        if getattr(mesh, "m_Name", "") != "o_head":
            continue
        h = MeshHandler(mesh)
        h.process()
        vc = len(h.m_Vertices)
        if best is None or vc > best[0]:
            best = (vc, smr, mesh, h)
    if best is None:
        raise SystemExit("o_head SMR not found in fo_head_00")
    vc, smr, mesh, h = best
    print(f"[mesh] o_head: {vc} verts")

    verts = np.asarray(h.m_Vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(h.m_IndexBuffer, dtype=np.int64).reshape(-1, 3)
    bidx = np.asarray(h.m_BoneIndices, dtype=np.int64).reshape(-1, 4)
    bw = np.asarray(h.m_BoneWeights, dtype=np.float32).reshape(-1, 4)
    # bindpose (Matrix4x4) list -> (N,4,4)
    bind = []
    for m4 in mesh.m_BindPose:
        # UnityPy Matrix4x4: e00.. row-major accessors
        bind.append([[m4.e00, m4.e01, m4.e02, m4.e03],
                     [m4.e10, m4.e11, m4.e12, m4.e13],
                     [m4.e20, m4.e21, m4.e22, m4.e23],
                     [m4.e30, m4.e31, m4.e32, m4.e33]])
    bind = np.asarray(bind, dtype=np.float32)

    # skin-bone order from the SMR's m_Bones (matches bindpose order)
    skin_bone_pids = []
    skin_bone_names = []
    for b in smr.m_Bones:
        pid = b.path_id
        skin_bone_pids.append(pid)
        skin_bone_names.append(trans.get(pid, {}).get("name"))
    print(f"[mesh] skin bones: {len(skin_bone_names)}; bindposes: {len(bind)}")

    # collect the skeleton subtree we need: skin bones + all ancestors to root
    need = set()
    for pid in skin_bone_pids:
        cur = pid
        while cur is not None and cur in trans and cur not in need:
            need.add(cur)
            cur = trans[cur]["parent"]
    skel = {str(pid): trans[pid] for pid in need}
    # also stringify parent ids
    for v in skel.values():
        v["parent"] = str(v["parent"]) if v["parent"] is not None else None
    print(f"[skel] {len(skel)} bones (skin + ancestors)")

    return {
        "verts": verts, "faces": faces, "bone_idx": bidx, "bone_w": bw,
        "bindpose": bind,
        "skin_bone_pids": [str(p) for p in skin_bone_pids],
        "skin_bone_names": skin_bone_names,
        "skel": skel,
    }


def _read_dotnet_string(br):
    # .NET BinaryReader.ReadString: 7-bit-encoded length prefix, then UTF-8 bytes
    length = 0; shift = 0
    while True:
        b = br.read(1)[0]
        length |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return br.read(length).decode("utf-8")


def parse_anm_shape(env_bundle, asset_name):
    env = UnityPy.load(env_bundle)
    data = None
    for o in env.objects:
        if o.type.name == "TextAsset":
            tt = o.read_typetree()
            if tt.get("m_Name") == asset_name:
                s = tt.get("m_Script")
                data = s.encode("utf-8", "surrogateescape") if isinstance(s, str) else bytes(s)
                break
    if data is None:
        raise SystemExit(f"{asset_name} not found in {env_bundle}")
    br = io.BytesIO(data)
    n = struct.unpack("<i", br.read(4))[0]
    out = {}
    for _ in range(n):
        name = _read_dotnet_string(br)
        k = struct.unpack("<i", br.read(4))[0]
        frames = []
        for _ in range(k):
            no = struct.unpack("<i", br.read(4))[0]
            vals = struct.unpack("<9f", br.read(36))
            frames.append({"no": no, "pos": vals[0:3], "rot": vals[3:6], "scl": vals[6:9]})
        out[name] = frames
    print(f"[anm] {asset_name}: {len(out)} bones, keyframes/bone e.g. {len(next(iter(out.values())))}")
    return out


def parse_customhead():
    env = UnityPy.load(CUSTOMSHAPE)
    txt = None
    for o in env.objects:
        if o.type.name == "TextAsset":
            tt = o.read_typetree()
            if tt.get("m_Name") == "cf_customhead":
                s = tt.get("m_Script")
                txt = s if isinstance(s, str) else bytes(s).decode("utf-8", "replace")
                break
    if txt is None:
        raise SystemExit("cf_customhead not found")
    rows = []
    for line in txt.splitlines():
        line = line.rstrip("\r")
        if not line.strip():
            continue
        c = line.split("\t")
        rows.append({"category": int(c[0]), "bone": c[1],
                     "use": [int(x != "0") for x in c[2:11]]})  # pos.xyz, rot.xyz, scl.xyz
    print(f"[customhead] {len(rows)} rows")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    m = extract_mesh_and_skeleton()
    np.savez(os.path.join(OUT, "o_head_mesh.npz"),
             verts=m["verts"], faces=m["faces"], bone_idx=m["bone_idx"],
             bone_w=m["bone_w"], bindpose=m["bindpose"])
    with open(os.path.join(OUT, "skeleton.json"), "w", encoding="utf-8") as f:
        json.dump({"skin_bone_pids": m["skin_bone_pids"],
                   "skin_bone_names": m["skin_bone_names"],
                   "bones": m["skel"]}, f)
    anm = parse_anm_shape(FO_HEAD, "cf_anmShapeHead_00")
    with open(os.path.join(OUT, "anmShapeHead_00.json"), "w", encoding="utf-8") as f:
        json.dump(anm, f)
    ch = parse_customhead()
    with open(os.path.join(OUT, "customhead.json"), "w", encoding="utf-8") as f:
        json.dump(ch, f)
    print(f"\n[done] cached -> {OUT}")


if __name__ == "__main__":
    main()
