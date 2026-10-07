"""Differentiable (torch) HS2 face deform — a line-for-line port of src/hs2_mesh_deform.py.

Same reverse-engineered pipeline, same data tables (data/hs2_head/{head_N/anmShapeHead,customhead,
enums,update_eqns}.json); only the arithmetic moves to torch so gradients reach the *parameters*:

    shapeValueFace(59) + ABMX(30x10)  ->  keyframe lerp  ->  Update() equations  ->  ABMX
                                      ->  FK world matrices  ->  LBS  ->  verts

NB **59, not 54**. The ML label vector carries 54 sliders; the rig is driven by 59 — categories
54-58 are the ear knobs (`FaceData.base_data` drops them, commented "without ear data"). Feeding
the rig 54 does not leave the ears at rest: the Update equations OVERWRITE each driven bone's rest
transform from its source bone, so absent sliders write neutral defaults over the rest pose and
the ears come out collapsed and crumpled. Both entry points now refuse a short vector.

Every stage is smooth in the parameters (piecewise-linear interpolation, quaternion composition,
matrix products), so `verts` is differentiable w.r.t. both inputs — that is what makes
`param -> render -> beauty` a gradient problem instead of a black-box search.

The whole thing is precomputed into flat index/matrix buffers at construction, so a forward pass is
a handful of batched ops (no Python loop over sliders/equations; only the 68-bone FK chain, which is
inherently sequential).

    from src.hs2_deform_torch import TorchHeadRig
    trig = TorchHeadRig(HeadRig(2), device="cuda")
    verts = trig(shape_face, ab)          # (B,59), (B,30,10) -> (B,V,3), autograd-ready

Self-check (torch vs numpy, and autograd vs finite differences):
    .venv/Scripts/python.exe -m src.hs2_deform_torch --card tests/HS2ChaF_20240901192905747.png

Sliders parked outside [0,1] clamp under vanilla and may have zero gradient. This is NOT
the behavior of an installed SliderUnlocker: select sampling_profile="slider_unlocker_18_2"
explicitly for that runtime. A zero gradient at a particular card or rail is not evidence
that a control never affects o_head. The neutral local atlas measures all 59 directions,
including 27/29/36/38; descendant transforms matter even if a direct target is unskinned.
See docs/hs2_parameter_atlas.md for the head/input-specific evidence and limits.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .hs2_mesh_deform import HeadRig, available_heads
from .hs2_sampling import (
    rotation_is_exempt, unlocker_rotation_delta, validate_keyframes, validate_sampling_profile,
)

# ABData.to_vector layout, per bone: scale(3) | length(1) | position(3) | rotation(3)
AB_SCALE, AB_LENGTH, AB_POS, AB_ROT = slice(0, 3), 3, slice(4, 7), slice(7, 10)
_FIELDS = ("pos", "rot", "scl")   # order of the concatenated source-value vector


def _axis_quat(axis: int, deg: torch.Tensor) -> torch.Tensor:
    """Quaternion (x,y,z,w) for a rotation of `deg` about a principal axis. deg: (...,)."""
    r = torch.deg2rad(deg) * 0.5
    q = [torch.zeros_like(deg)] * 3 + [torch.cos(r)]
    q[axis] = torch.sin(r)
    return torch.stack(q, dim=-1)


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of (x,y,z,w) quaternions, broadcasting over leading dims."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def euler_zxy_quat(x, y, z) -> torch.Tensor:
    """Unity Quaternion.Euler(x,y,z): apply Z, then X, then Y -> q = qy * qx * qz."""
    return qmul(_axis_quat(1, y), qmul(_axis_quat(0, x), _axis_quat(2, z)))


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(...,4) xyzw -> (...,3,3)."""
    x, y, z, w = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], dim=-2)


def trs(pos: torch.Tensor, quat: torch.Tensor, scl: torch.Tensor) -> torch.Tensor:
    """(...,3),(...,4),(...,3) -> (...,4,4) with M = T * R * diag(S) (Unity column scale)."""
    m = torch.zeros(*pos.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
    m[..., :3, :3] = quat_to_mat(quat) * scl.unsqueeze(-2)
    m[..., :3, 3] = pos
    m[..., 3, 3] = 1.0
    return m


class TorchHeadRig:
    """Precompiled, batched, differentiable version of `HeadRig` + `build_mesh`."""

    def __init__(self, rig: HeadRig, device="cuda", dtype=torch.float32, *, sampling_profile=None):
        self.rig, self.device, self.dtype = rig, torch.device(device), dtype
        self.sampling_profile = validate_sampling_profile(
            rig.sampling_profile if sampling_profile is None else sampling_profile)
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), dtype=dt or dtype, device=self.device)
        ti = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.long, device=self.device)

        # ---- bone table, in FK (topological, parents-first) order
        self.pids = list(rig._topo)
        pid2i = {p: i for i, p in enumerate(self.pids)}
        self.n_bones = len(self.pids)
        self.parent = ti([pid2i.get(rig.bones[p]["parent"], -1) for p in self.pids])
        self.rest_pos = t([rig.bones[p]["pos"] for p in self.pids])          # (n,3)
        self.rest_quat = t([rig.bones[p]["rot"] for p in self.pids])         # (n,4) xyzw
        self.rest_scl = t([rig.bones[p]["scale"] for p in self.pids])        # (n,3)
        self.name2bone = {rig.bones[p]["name"]: i for i, p in enumerate(self.pids)}

        # ---- stage 1: shapeValueFace -> cf_s_ source values, via evenly-spaced keyframes
        src_names = rig.enums["src"]
        self.n_src = len(src_names)
        src_idx = {n: i for i, n in enumerate(src_names)}
        # Every category the table drives, NOT just the 54 the ML label vector carries: dropping
        # categories 54-58 does not leave the ear bones at rest, it lets the Update equations write
        # neutral defaults over their rest pose. See src/hs2_mesh.fd_to_inputs.
        rows = [r for r in rig.customhead
                if r["bone"] in rig.anm and r["bone"] in src_idx]
        for row in rows:
            validate_keyframes(rig.anm[row["bone"]])
        self.n_slider = max(r["category"] for r in rig.customhead) + 1
        # kept so the self-check can guarantee it probes the ear knobs (categories 54-58)
        self.ear_cats = sorted({r["category"] for r in rows if "Ear" in r["bone"]})
        ks = {len(rig.anm[r["bone"]]) for r in rows}
        if len(ks) != 1:
            raise ValueError(f"expected a uniform keyframe count, got {sorted(ks)}")
        self.n_key = ks.pop()
        if self.n_key < 1:
            raise ValueError("Animation needs at least one keyframe")
        self.row_cat = ti([r["category"] for r in rows])                      # (R,) slider index
        self.rotation_exempt = torch.as_tensor(
            [rotation_is_exempt(r["bone"]) or not any(r["use"][3:6]) for r in rows],
            dtype=torch.bool, device=self.device)
        self.extrap_first, self.extrap_last, self.extrap_delta = {}, {}, {}

        # Per row: base value at each keyframe + the delta to the next one, so sampling is
        # `a[i] + d[i]*t` (rotation deltas use Mathf.LerpAngle's shortest-arc rule, folded in here).
        a_all, d_all = {}, {}
        for f, key in (("pos", "pos"), ("rot", "rot"), ("scl", "scl")):
            a = np.array([[fr[key] for fr in rig.anm[r["bone"]]] for r in rows], np.float64)
            self.extrap_first[f], self.extrap_last[f] = t(a[:, 0]), t(a[:, -1])
            delta = (np.array([unlocker_rotation_delta(rig.anm[r["bone"]]) for r in rows])
                     if f == "rot" and self.n_key > 1 else a[:, -1] - a[:, 0])
            self.extrap_delta[f] = t(delta)
            if f == "rot":
                a = a % 360.0
                d = (np.diff(a, axis=1) + 540.0) % 360.0 - 180.0
            else:
                d = np.diff(a, axis=1)
            a_all[f], d_all[f] = t(a), t(d)                                   # (R,K,3), (R,K-1,3)
        self.key_a, self.key_d = a_all, d_all

        # scatter: sampled row values -> the (n_src,3) source arrays, honouring the DOF-use flags.
        # Later rows win on collision (matches the numpy loop's assignment order).
        self.scatter = {}
        for fi, field in enumerate(_FIELDS):
            pairs = {}
            for ri, r in enumerate(rows):
                sid = src_idx[r["bone"]]
                for ax in range(3):
                    if r["use"][fi * 3 + ax]:
                        pairs[sid * 3 + ax] = ri * 3 + ax                     # dst <- src (last wins)
            self.scatter[field] = (ti(sorted(pairs)), ti([pairs[k] for k in sorted(pairs)]))
        self.src_default = {"pos": 0.0, "rot": 0.0, "scl": 1.0}

        # ---- stage 2: the hardcoded Update() equations, as one affine map
        # every equation argument is `const + sum(selected source entries)`, so all 209 of them are
        # a single (n_args, 3*3*n_src) matrix product over the concatenated source vector.
        S = 3 * self.n_src
        args, self.arg_of = [], []      # arg rows; per-equation slice into them
        A = np.zeros((sum(len(e["args"]) for e in rig.eqns), 3 * S), np.float64)
        c = np.zeros(A.shape[0], np.float64)
        k = 0
        for e in rig.eqns:
            first = k
            for arg in e["args"]:
                for term in arg:
                    if term[0] == "const":
                        c[k] += term[1]
                    else:
                        _, si, fld, ax = term
                        A[k, _FIELDS.index(fld) * S + si * 3 + ax] += 1.0
                k += 1
            self.arg_of.append((first, k))
            args.append(e)
        self.eqn_A, self.eqn_c = t(A), t(c)

        dst_names = rig.enums["dst"]
        pos_dst, pos_arg, scl_dst, scl_arg, rot_bone, rot_arg = [], [], [], [], [], []
        for e, (lo, hi) in zip(args, self.arg_of):
            b = self.name2bone.get(dst_names[e["dst"]])
            if b is None:
                continue                                     # dst bone absent from this head's rig
            if e["target"] == "pos":
                pos_dst.append(b * 3 + e["axis"])
                pos_arg.append(lo)
            elif e["target"] == "scl":
                scl_dst.extend([b * 3 + a for a in range(3)])
                scl_arg.extend(range(lo, hi))
            else:                                            # rot: (ex, ey, ez) -> quaternion
                rot_bone.append(b)
                rot_arg.append(list(range(lo, hi)))
        self.pos_dst, self.pos_arg = ti(pos_dst), ti(pos_arg)
        self.scl_dst, self.scl_arg = ti(scl_dst), ti(scl_arg)
        self.rot_bone, self.rot_arg = ti(rot_bone), ti(rot_arg)

        # ---- stage 3: ABMX bones (in the 205-vector's BONE_NAME_LIST order)
        from .face_data_utils.utils import BONE_NAME_LIST
        self.ab_names = list(BONE_NAME_LIST)
        keep = [(i, self.name2bone[n]) for i, n in enumerate(self.ab_names) if n in self.name2bone]
        self.ab_slot = ti([i for i, _ in keep])
        self.ab_bone = ti([b for _, b in keep])

        # ---- stage 5: skinning
        self.skin_bone = ti([self.name2bone[n] for n in rig.skin_bone_names])
        self.bindpose = t(rig.bindpose)                                        # (43,4,4)
        self.verts_h = torch.cat([t(rig.verts), torch.ones(len(rig.verts), 1,
                                                           dtype=dtype, device=self.device)], 1)
        self.bone_idx = ti(rig.bone_idx)                                       # (V,4)
        self.bone_w = t(rig.bone_w)                                            # (V,4)
        self.faces = rig.faces

    # ------------------------------------------------------------------ forward
    def source_values(self, shape_face: torch.Tensor) -> torch.Tensor:
        """(B,59) sliders -> (B, 3*3*n_src) concatenated cf_s_ [pos|rot|scl] source values.

        `AnimationKeyInfo.GetInfo`: keyframes are evenly spaced over [0,1]; the segment index is
        piecewise-constant (no gradient) and the fraction `t` carries it — exactly the derivative
        of the piecewise-linear interpolant the game evaluates.
        """
        B = shape_face.shape[0]
        if not torch.isfinite(shape_face).all():
            raise ValueError("shape_face must contain only finite values")
        if shape_face.shape[1] < self.n_slider:
            raise ValueError(
                f"shape_face has {shape_face.shape[1]} sliders, the rig drives {self.n_slider}. "
                f"Categories 54..58 are the ear knobs and are NOT part of the 54-dim label vector; "
                f"a short vector silently deforms the ears rather than leaving them at rest.")
        raw_rate = shape_face.index_select(1, self.row_cat)
        outside = (raw_rate < 0) | (raw_rate > 1)
        unlocked = self.sampling_profile == "slider_unlocker_18_2"
        if unlocked and self.n_key == 1 and (outside & ~self.rotation_exempt).any():
            raise ValueError("SliderUnlocker rotation extrapolation requires at least two keyframes")
        rate = raw_rate.clamp(0, 1)                                         # (B,R)
        x = rate * (self.n_key - 1)
        i = x.floor().clamp(max=max(self.n_key - 2, 0)).detach().long()        # (B,R) segment
        frac = (x - i).unsqueeze(-1)                                           # (B,R,1)

        out = []
        for field in _FIELDS:
            a, d = self.key_a[field], self.key_d[field]                        # (R,K,3),(R,K-1,3)
            gi = i.unsqueeze(-1).expand(-1, -1, 3)                             # (B,R,3)
            if self.n_key == 1:
                val = a[:, 0].unsqueeze(0).expand(B, -1, -1) + raw_rate.unsqueeze(-1) * 0
            else:
                av = a.unsqueeze(0).expand(B, -1, -1, -1).gather(2, gi.unsqueeze(2)).squeeze(2)
                dv = d.unsqueeze(0).expand(B, -1, -1, -1).gather(2, gi.unsqueeze(2)).squeeze(2)
                val = av + dv * frac
            if unlocked:
                first = self.extrap_first[field].unsqueeze(0)
                last = self.extrap_last[field].unsqueeze(0)
                delta = self.extrap_delta[field].unsqueeze(0)
                raw = raw_rate.unsqueeze(-1)
                extrap = first + delta * raw
                if field == "rot":
                    endpoint = torch.where(raw < 0, first, last)
                    extrap = torch.where(raw < 0, extrap, last + delta * (raw - 1))
                    extrap = torch.where(self.rotation_exempt[None, :, None], endpoint, extrap)
                val = torch.where(outside.unsqueeze(-1), extrap, val)
            val = val.reshape(B, -1)                                         # (B,R*3)

            dst, src = self.scatter[field]
            flat = torch.full((B, 3 * self.n_src), self.src_default[field],
                              dtype=self.dtype, device=self.device)
            out.append(flat.index_copy(1, dst, val.index_select(1, src)))
        return torch.cat(out, dim=1)                                           # (B, 3*3*n_src)

    def local_transforms(self, shape_face=None, ab=None):
        """Stages 1-3 -> per-bone local (pos, quat, scale), each (B, n_bones, ...)."""
        B = 1 if shape_face is None else shape_face.shape[0]
        if ab is not None:
            B = max(B, ab.shape[0])
        pos = self.rest_pos.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        scl = self.rest_scl.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
        quat = self.rest_quat.unsqueeze(0).expand(B, -1, -1)

        if shape_face is not None:
            src = self.source_values(shape_face)                               # (B,S)
            argv = src @ self.eqn_A.T + self.eqn_c                             # (B,n_args)
            pos = pos.index_copy(1, self.pos_dst, argv.index_select(1, self.pos_arg))
            scl = scl.index_copy(1, self.scl_dst, argv.index_select(1, self.scl_arg))
            if len(self.rot_bone):
                e = argv[:, self.rot_arg]                                      # (B,n_rot,3)
                q = euler_zxy_quat(e[..., 0], e[..., 1], e[..., 2])
                quat = quat.index_copy(1, self.rot_bone, q)
        pos, scl = pos.view(B, -1, 3), scl.view(B, -1, 3)

        if ab is not None:                                                     # BoneModifier.cs
            sel = ab.index_select(1, self.ab_slot)                             # (B,n_ab,10)
            scl = scl.index_copy(1, self.ab_bone,
                                 scl.index_select(1, self.ab_bone) * sel[..., AB_SCALE])
            pos = pos.index_copy(1, self.ab_bone,
                                 pos.index_select(1, self.ab_bone) * sel[..., AB_LENGTH:AB_LENGTH + 1]
                                 + sel[..., AB_POS])
            r = sel[..., AB_ROT]
            quat = quat.index_copy(1, self.ab_bone,
                                   qmul(quat.index_select(1, self.ab_bone),
                                        euler_zxy_quat(r[..., 0], r[..., 1], r[..., 2])))
        return pos, quat, scl

    def bone_world(self, shape_face=None, ab=None) -> torch.Tensor:
        """Stages 1-4 -> (B, n_bones, 4, 4) world matrices (FK, parents already resolved)."""
        pos, quat, scl = self.local_transforms(shape_face, ab)
        local = trs(pos, quat, scl)                                            # (B,n,4,4)
        parent = self.parent.tolist()
        world = [None] * self.n_bones
        for i, p in enumerate(parent):                                         # topo order
            world[i] = local[:, i] if p < 0 else world[p] @ local[:, i]
        return torch.stack(world, dim=1)

    def forward(self, shape_face=None, ab=None) -> torch.Tensor:
        """(B,59) sliders + (B,30,10) ABMX -> (B,V,3) skinned vertices."""
        world = self.bone_world(shape_face, ab)
        skin = world.index_select(1, self.skin_bone) @ self.bindpose           # (B,43,4,4)
        M = skin[:, self.bone_idx.reshape(-1)].view(
            world.shape[0], *self.bone_idx.shape, 4, 4)                        # (B,V,4,4,4)
        tv = torch.einsum("bvkij,vj->bvki", M, self.verts_h)[..., :3]          # (B,V,4,3)
        return (self.bone_w.unsqueeze(0).unsqueeze(-1) * tv).sum(dim=2)

    __call__ = forward

    # ------------------------------------------------------------------ submeshes
    def load_submesh(self, name: str) -> dict:
        """One submesh (eyes / lashes / tear / eyeshadow / teeth / tongue) as GPU tensors.

        They ride the SAME skeleton as o_head, so they reuse `bone_world()` — only their own skin
        weights and bindposes differ. Requires the extractor's skeleton to cover their bones (it
        walks every submesh's `m_Bones`, not just the head's).
        """
        if not hasattr(self, "_subs"):
            self._subs = {}
        if name in self._subs:
            return self._subs[name]
        path = os.path.join(self.rig.data_dir, "submeshes", f"{name}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"no submesh {name} at {path}")
        d = np.load(path, allow_pickle=True)
        pid2i = {p: i for i, p in enumerate(self.pids)}
        pids = [str(p) for p in d["skin_bone_pids"]]
        missing = [p for p in pids if p not in pid2i]
        if missing:
            raise KeyError(f"submesh {name}: {len(missing)} skin bones absent from the skeleton "
                           f"— re-run scripts/hs2_extract_head.py")
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), dtype=dt or self.dtype, device=self.device)
        sub = {
            "name": name,
            "verts_h": torch.cat([t(d["verts"]), torch.ones(len(d["verts"]), 1,
                                                            dtype=self.dtype, device=self.device)], 1),
            "faces": d["faces"],
            "uv": d["uv"], "normals": d["normals"], "tangents": d["tangents"],
            "bone_idx": torch.as_tensor(d["bone_idx"], dtype=torch.long, device=self.device),
            "bone_w": t(d["bone_w"]),
            "bindpose": t(d["bindpose"]),
            "skin_bone": torch.as_tensor([pid2i[p] for p in pids], dtype=torch.long,
                                         device=self.device),
        }
        self._subs[name] = sub
        return sub

    def skin(self, sub: dict, world: torch.Tensor) -> torch.Tensor:
        """LBS one submesh with already-computed bone world matrices. -> (B,V,3)"""
        skin = world.index_select(1, sub["skin_bone"]) @ sub["bindpose"]
        M = skin[:, sub["bone_idx"].reshape(-1)].view(
            world.shape[0], *sub["bone_idx"].shape, 4, 4)
        tv = torch.einsum("bvkij,vj->bvki", M, sub["verts_h"])[..., :3]
        return (sub["bone_w"].unsqueeze(0).unsqueeze(-1) * tv).sum(dim=2)

    # ------------------------------------------------------------------ input helpers
    def ab_tensor(self, ab_data: dict, batch=1) -> torch.Tensor:
        """{bone: {scale,length,position,rotation}} -> (batch, len(ab_names), 10)."""
        v = np.zeros((len(self.ab_names), 10), np.float64)
        v[:, AB_SCALE], v[:, AB_LENGTH] = 1.0, 1.0
        for i, name in enumerate(self.ab_names):
            ab = ab_data.get(name)
            if ab is None:
                continue
            v[i, AB_SCALE] = ab["scale"]
            v[i, AB_LENGTH] = ab["length"]
            v[i, AB_POS] = ab["position"]
            v[i, AB_ROT] = ab["rotation"]
        return torch.as_tensor(v, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(batch, 1, 1)

    def from_card(self, card_path: str):
        """(shape_face (1,59), ab (1,n_ab,10)) for a card — mirrors src/hs2_mesh.fd_to_inputs."""
        from .face_data_utils.utils import FaceData
        from .hs2_mesh import fd_to_inputs
        shape_face, ab_data = fd_to_inputs(FaceData(card_path))
        sf = torch.as_tensor(np.asarray(shape_face, np.float64),
                             dtype=self.dtype, device=self.device).unsqueeze(0)
        return sf, self.ab_tensor(ab_data)


# ---------------------------------------------------------------------- self-check
def _check(card, head_id=None, seed=0):
    from .hs2_mesh_deform import build_mesh
    from .hs2_mesh import card_head_id, fd_to_inputs
    from .face_data_utils.utils import FaceData

    head_id = card_head_id(card) if head_id is None else head_id
    rig = HeadRig(head_id)
    trig = TorchHeadRig(rig, device="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=torch.float64)
    print(f"[rig] headId={head_id}  bones={trig.n_bones}  verts={len(rig.verts)}  "
          f"device={trig.device}")

    fd = FaceData(card)
    shape_face, ab_data = fd_to_inputs(fd)
    sf, ab = trig.from_card(card)

    # 1) torch must reproduce numpy exactly (same tables, same math)
    ref_neutral, _ = build_mesh(rig, None, None)
    ref_card, _ = build_mesh(rig, shape_face, ab_data)
    for name, (a, b) in {
        "neutral": (trig().detach().cpu().numpy()[0], ref_neutral),
        "card": (trig(sf, ab).detach().cpu().numpy()[0], ref_card),
    }.items():
        err = np.abs(a - b).max()
        print(f"[torch-vs-numpy] {name:8s} max|dv| = {err:.3e}  "
              f"{'OK' if err < 1e-9 else 'MISMATCH'}")

    # 2) autograd vs finite differences, on BOTH halves of the parameter vector
    #    (59 rig sliders and the 30x10 ABMX block — the latter is 150 of the 205 label dims)
    rng = np.random.default_rng(seed)
    sfv, abv = sf.clone().requires_grad_(True), ab.clone().requires_grad_(True)
    trig(sfv, abv).square().sum().backward()
    grads = {"slider": (sfv.grad[0].detach().cpu().numpy(), sf, "sf"),
             "abmx": (abv.grad[0].detach().cpu().numpy(), ab, "ab")}

    # Central differences on a float64 loss of magnitude L resolve gradients no finer than
    # ~L*2.2e-16/eps; comparing purely relatively below that floor just measures FD noise, so the
    # tolerance is the usual mixed form  |dg| <= atol + rtol*|g|  with atol = that floor.
    eps, rtol = 1e-6, 1e-3
    loss0 = trig(sf, ab).square().sum().item()
    atol = abs(loss0) * 2.22e-16 / eps
    print(f"[jacobian] loss={loss0:.4e}  fd noise floor (atol) = {atol:.2e}, rtol = {rtol}")

    # Probe the FULL slider range, not the label vector's 54: the ear knobs live at 54..58 and a
    # probe set that stopped at 54 is part of why they stayed broken while this check kept passing.
    # The two ear categories are always included so the coverage cannot regress by luck of the draw.
    ear_cats = [c for c in trig.ear_cats if c >= 54][:2]
    others = [int(p) for p in rng.choice(trig.n_slider, 5, replace=False) if p not in ear_cats]
    probes = [("slider", (int(p),)) for p in ear_cats + others[:5 - len(ear_cats)]]
    probes += [("abmx", (int(b), int(c)))
               for b, c in zip(rng.choice(len(trig.ab_names), 5), rng.choice(10, 5))]
    bad = 0
    for kind, idx in probes:
        g_auto_all, base, which = grads[kind]
        g_auto = float(g_auto_all[idx])
        gs = []
        for s in (+1, -1):
            x = base.clone()
            x[(0,) + idx] += s * eps
            gs.append(trig(x if which == "sf" else sf,
                           ab if which == "sf" else x).square().sum().item())
        g_fd = (gs[0] - gs[1]) / (2 * eps)
        ok = abs(g_fd - g_auto) <= atol + rtol * abs(g_fd)
        bad += not ok
        note = "" if abs(g_auto) > atol else "  (below fd resolution)"
        print(f"[jacobian] {kind:6s} {str(idx):9s} autograd={g_auto:+.6e}  fd={g_fd:+.6e}  "
              f"|d|={abs(g_fd - g_auto):.2e}  {'OK' if ok else 'FAIL'}{note}")
    print(f"[jacobian] {len(probes) - bad}/{len(probes)} probes within tolerance"
          f"  {'OK' if bad == 0 else 'FAIL'}")
    for kind, (g, _, _) in grads.items():
        nz = int((np.abs(g) > 0).sum())
        print(f"[jacobian] {kind}: {nz}/{g.size} entries have non-zero gradient")

    # 3) batched timing (float32, the shape training/optimization actually uses)
    if torch.cuda.is_available():
        trig32 = TorchHeadRig(rig, device="cuda", dtype=torch.float32)
        sf32 = sf.float().repeat(32, 1).requires_grad_(True)
        ab32 = ab.float().repeat(32, 1, 1)
        import time
        for _ in range(3):
            trig32(sf32, ab32).square().sum().backward()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            trig32(sf32, ab32).square().sum().backward()
        torch.cuda.synchronize()
        print(f"[timing] batch=32 fwd+bwd: {(time.time() - t0) / 10 * 1e3:.1f} ms")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--head-id", type=int, default=None)
    args = ap.parse_args()
    print(f"[heads] extracted: {available_heads()}")
    _check(args.card, args.head_id)
