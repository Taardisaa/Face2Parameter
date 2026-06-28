"""Offline HS2 face-mesh deform (pure numpy) — the reverse-engineered pipeline.

Given the cached head rig (data/hs2_head/, from scripts/hs2_extract_head.py + hs2_parse_update.py) plus a
card's shapeValueFace(54) + ABMX bones, reproduce the deformed o_head mesh exactly as the game does:

  shapeValueFace[i] --(cf_customhead: bone+DOF)--> sample cf_anmShapeHead keyframes (rate=value)
     --> cf_s_ source-bone values --(ShapeHeadInfoFemale.Update equations)--> cf_J_ bone LOCAL transforms
     --> ABMX multiply (BoneModifier.cs) --> FK world matrices --> linear-blend-skin o_head.

See docs/hs2-renderer-and-mesh.md. Expression blendshapes are left at 0 (neutral).
"""
from __future__ import annotations

import json
import os

import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hs2_head")


# ---------- math (Unity conventions) ----------
def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _axis_quat(axis, deg):
    r = np.radians(deg) * 0.5
    s = np.sin(r)
    q = [0.0, 0.0, 0.0, np.cos(r)]
    q[axis] = s
    return np.array(q)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def euler_zxy_quat(x, y, z):
    """Unity Quaternion.Euler(x,y,z): apply Z, then X, then Y -> q = qy * qx * qz."""
    return _qmul(_axis_quat(1, y), _qmul(_axis_quat(0, x), _axis_quat(2, z)))


def _trs(pos, quat, scl):
    m = np.eye(4)
    m[:3, :3] = _quat_to_mat(quat) * np.asarray(scl)  # column-scale (M = R * S)
    m[:3, 3] = pos
    return m


# ---------- data ----------
class HeadRig:
    def __init__(self, data_dir=_DATA):
        d = np.load(os.path.join(data_dir, "o_head_mesh.npz"))
        self.verts = d["verts"].astype(np.float64)
        self.faces = d["faces"]
        self.bone_idx = d["bone_idx"]
        self.bone_w = d["bone_w"].astype(np.float64)
        self.bindpose = d["bindpose"].astype(np.float64)
        sk = json.load(open(os.path.join(data_dir, "skeleton.json")))
        self.skin_bone_names = sk["skin_bone_names"]
        self.bones = sk["bones"]  # pid(str) -> {name,parent,pos,rot,scale}
        self.name2pid = {}
        for pid, b in self.bones.items():
            self.name2pid.setdefault(b["name"], pid)
        self.enums = json.load(open(os.path.join(data_dir, "enums.json")))
        self.customhead = json.load(open(os.path.join(data_dir, "customhead.json")))
        self.anm = json.load(open(os.path.join(data_dir, "anmShapeHead_00.json")))
        self.eqns = json.load(open(os.path.join(data_dir, "update_eqns.json")))
        # topo order of bones (parents first)
        self._topo = self._toposort()

    def _toposort(self):
        order, seen = [], set()
        def visit(pid):
            if pid in seen or pid not in self.bones:
                return
            p = self.bones[pid]["parent"]
            if p in self.bones:
                visit(p)
            seen.add(pid); order.append(pid)
        for pid in self.bones:
            visit(pid)
        return order


def _sample(frames, rate):
    """AnimationKeyInfo.GetInfo: evenly-spaced keyframes over [0,1] -> (pos,rot,scl)."""
    n = len(frames)
    if rate <= 0:
        f = frames[0]; return f["pos"], f["rot"], f["scl"]
    if rate >= 1:
        f = frames[-1]; return f["pos"], f["rot"], f["scl"]
    x = (n - 1) * rate
    i = int(np.floor(x)); t = x - i
    a, b = frames[i], frames[i + 1]
    pos = [a["pos"][k] * (1 - t) + b["pos"][k] * t for k in range(3)]
    scl = [a["scl"][k] * (1 - t) + b["scl"][k] * t for k in range(3)]
    # rot: LerpAngle per component
    rot = []
    for k in range(3):
        da = a["rot"][k] % 360; db = b["rot"][k] % 360
        diff = (db - da + 540) % 360 - 180
        rot.append(da + diff * t)
    return pos, rot, scl


def _fk_world(rig: HeadRig, shape_face=None, ab_data=None):
    """Deform stages 1-4 -> {pid: 4x4 world matrix} for every bone (no skinning).

    shape_face: (54,) floats; ab_data: {cf_J_name: dict(scale[3],length,position[3],rotation[3])} or None.
    """
    src_names = rig.enums["src"]; dst_names = rig.enums["dst"]
    src_idx = {n: i for i, n in enumerate(src_names)}

    # 1. shapeValueFace -> cf_s_ source-bone values (defaults: pos 0, rot 0, scl 1)
    src_pos = np.zeros((len(src_names), 3)); src_rot = np.zeros((len(src_names), 3))
    src_scl = np.ones((len(src_names), 3))
    if shape_face is not None:
        for row in rig.customhead:
            cat = row["category"]
            if cat >= len(shape_face):
                continue
            bone = row["bone"]; use = row["use"]
            if bone not in rig.anm or bone not in src_idx:
                continue
            p, r, s = _sample(rig.anm[bone], float(shape_face[cat]))
            sid = src_idx[bone]
            for k in range(3):
                if use[k]:     src_pos[sid][k] = p[k]
                if use[3 + k]: src_rot[sid][k] = r[k]
                if use[6 + k]: src_scl[sid][k] = s[k]

    # 2. local transforms for every bone = REST, then Update() overrides for dst cf_J_ bones
    loc_pos, loc_quat, loc_scl = {}, {}, {}
    for pid, b in rig.bones.items():
        loc_pos[pid] = np.array(b["pos"], float)
        loc_quat[pid] = np.array(b["rot"], float)
        loc_scl[pid] = np.array(b["scale"], float)

    def term_val(term):
        if term[0] == "const":
            return term[1]
        _, si, fld, ax = term
        return {"pos": src_pos, "rot": src_rot, "scl": src_scl}[fld][si][ax]

    # group eqns per dst bone and apply
    if shape_face is not None:
        for e in rig.eqns:
            dname = dst_names[e["dst"]]
            pid = rig.name2pid.get(dname)
            if pid is None:
                continue
            if e["target"] == "pos":
                loc_pos[pid][e["axis"]] = sum(term_val(t) for t in e["args"][0])
            elif e["target"] == "rot":
                ex, ey, ez = (sum(term_val(t) for t in arg) for arg in e["args"])
                loc_quat[pid] = euler_zxy_quat(ex, ey, ez)
            else:  # scl
                loc_scl[pid] = np.array([sum(term_val(t) for t in arg) for arg in e["args"]])

    # 3. ABMX multiply onto the post-shape local transform (BoneModifier.cs)
    if ab_data:
        for name, ab in ab_data.items():
            pid = rig.name2pid.get(name)
            if pid is None:
                continue
            loc_scl[pid] = loc_scl[pid] * np.asarray(ab["scale"], float)
            loc_pos[pid] = loc_pos[pid] * float(ab["length"]) + np.asarray(ab["position"], float)
            rx, ry, rz = ab["rotation"]
            loc_quat[pid] = _qmul(loc_quat[pid], euler_zxy_quat(rx, ry, rz))

    # 4. FK world matrices
    world = {}
    for pid in rig._topo:
        local = _trs(loc_pos[pid], loc_quat[pid], loc_scl[pid])
        par = rig.bones[pid]["parent"]
        world[pid] = (world[par] @ local) if par in world else local
    return world


def build_mesh(rig: HeadRig, shape_face=None, ab_data=None):
    """shape_face: (54,) floats; ab_data: {cf_J_name: dict(...)} or None. Returns (verts(V,3), faces)."""
    world = _fk_world(rig, shape_face, ab_data)

    # 5. LBS
    skin_world = np.zeros((len(rig.skin_bone_names), 4, 4))
    for i, nm in enumerate(rig.skin_bone_names):
        pid = rig.name2pid.get(nm)
        skin_world[i] = (world[pid] @ rig.bindpose[i]) if pid in world else np.eye(4)

    vh = np.concatenate([rig.verts, np.ones((len(rig.verts), 1))], axis=1)  # (V,4)
    out = np.zeros((len(rig.verts), 3))
    for k in range(4):
        bi = rig.bone_idx[:, k]; w = rig.bone_w[:, k]
        M = skin_world[bi]                      # (V,4,4)
        tv = np.einsum("vij,vj->vi", M, vh)[:, :3]
        out += w[:, None] * tv
    return out, rig.faces


def bone_world(rig: HeadRig, shape_face=None, ab_data=None, names=None):
    """World-space (x,y,z) of each named bone after deform stages 1-4 (no skinning) — landmark source."""
    world = _fk_world(rig, shape_face, ab_data)
    names = list(rig.name2pid) if names is None else names
    out = {}
    for nm in names:
        pid = rig.name2pid.get(nm)
        if pid is not None and pid in world:
            out[nm] = world[pid][:3, 3].copy()
    return out


if __name__ == "__main__":
    rig = HeadRig()
    # NEUTRAL SANITY: rest locals (no shape, no abmx) must reproduce the extracted bind-pose mesh.
    out, faces = build_mesh(rig, shape_face=None, ab_data=None)
    err = np.linalg.norm(out - rig.verts, axis=1)
    print(f"[neutral-sanity] verts={len(out)}  max_err={err.max():.6e}  mean_err={err.mean():.6e}")
    print("  -> FK+bindpose+LBS consistent" if err.max() < 1e-3 else "  -> MISMATCH (plumbing bug)")
