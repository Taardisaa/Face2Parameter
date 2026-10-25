"""Bridge-render (pose-frozen) renderer matrices and bounds generated from local TRS.

During ``MakerRenderStyleService.Capture`` the pose is frozen
(``MakerRenderService.PoseFreeze.Apply``: every Animator ``Play(hash, 0, 0)`` +
``Update(0)``), then the camera renders 16 times.  Probe v6.1 exports the character
local transforms at every bridge render's ``pre_cull`` callback.

For a SkinnedMeshRenderer whose bounds come from its local AABB
(``updateWhenOffscreen`` false; prepared flag +4 clear), the per-render values are
(UnityPlayer.dll 2018.4.11f1, sha 5034cdc2...):

* ``unity_ObjectToWorld`` = TRS(rootBone world position, world rotation, 1)
  (CalculateGlobalPositionAndRotation 0xdda60, the same as
  ``unity_skin_emu.skin_from_locals``);
* world AABB: ``SkinnedMeshRendererManager::CalculateTransformInfo_Prepared``
  0xbdaca0 -> CalculateGlobalMatrix 0xdd620 of the root bone (affineX columns
  C0..C3, scale included), then the inline non-bone path 0xbdbe7f..0xbdc057 on the
  prepared local AABB (+0x58 centre, +0x64 extent; copied by InitializePreparedInfo
  0xbdce90 from SetLocalAABB 0x788010):

      centre = ((C1*cy + C2*cz) + C0*cx) + C3
      extent = (|C0*ex| + |C1*ey|) + |C2*ez|        (float32 lanes, abs after multiply)

``global_matrix`` runs CalculateGlobalMatrix itself in the emulator on a fake
TransformHierarchy (the layout ``unity_skin_emu.global_position_rotation`` uses).
"""
from __future__ import annotations

import struct

import numpy as np

F = np.float32
GLOBAL_MATRIX = 0xdd620   # affineX CalculateGlobalMatrix(TransformAccessReadOnly) (hidden return in rcx)


def global_matrix(emu, chain):
    """Columns C0..C3 (float32 x4 each) of the root's affineX world matrix.

    chain: local (t, q, s) of the transform, then its parent, ... up to the top."""
    emu.reset_heap()
    trs = b''.join(np.asarray(list(t) + [0.0], F).tobytes() + np.asarray(q, F).tobytes()
                   + np.asarray(list(s) + [0.0], F).tobytes() for t, q, s in chain)
    parents = list(range(1, len(chain))) + [-1]
    p_trs = emu.alloc(trs)
    p_par = emu.alloc(np.asarray(parents, np.int32).tobytes())
    hier = emu.alloc(struct.pack('<5Q', 0, 0, 0, p_trs, p_par))
    access = emu.alloc(struct.pack('<QI4x', hier, 0))
    out = emu.alloc(bytes(64))
    emu.call(GLOBAL_MATRIX, (out, access))
    return np.frombuffer(emu.read(out, 64), F).reshape(4, 4).copy()


def world_aabb(cols, local_center, local_extent):
    """Inline non-bone path of CalculateTransformInfo_Prepared (0xbdbe7f..0xbdbf5c)."""
    c0, c1, c2, c3 = (np.asarray(c, F) for c in cols)
    cx, cy, cz = (F(v) for v in local_center)
    ex, ey, ez = (F(v) for v in local_extent)
    centre = ((c1 * cy + c2 * cz) + c0 * cx) + c3
    extent = (np.abs(c0 * ex) + np.abs(c1 * ey)) + np.abs(c2 * ez)
    return centre[:3].astype(F), extent[:3].astype(F)


def chain_of(locals_by_id, parent_of, tid):
    """[(t, q, s) of tid, of its parent, ...] from {id: (t, q, s)} and {id: parent id | None}."""
    out = []
    while tid is not None:
        out.append(locals_by_id[tid])
        tid = parent_of[tid]
    return out


# ---------------------------------------------------------------- bone-driven bounds
def prepared_local_matrix(t, q, s):
    """Columns of one transform's local matrix as the bone-bounds loop builds it
    (CalculateTransformInfo_Prepared 0xbdb5d0..0xbdb6fd): rotation columns from the
    quaternion with the lane constants (2,2,-2,0) / (-2,2,2,0) / (-2,2,-2,0) / (2,-2,2,0) /
    (-2,-2,2,0) / (2,-2,-2,0), plus the identity column, times the scale lane, w cleared;
    translation (t, 1)."""
    x, y, z, w = (F(v) for v in q)
    col0 = np.array([F(F(F(-2 * y) * y) + F(F(-2 * z) * z)), F(F(F(2 * y) * x) + F(F(2 * z) * w)),
                     F(F(F(-2 * y) * w) + F(F(2 * z) * x)), F(0)], F)
    col0 = col0 + np.array([1, 0, 0, 0], F)
    col1 = np.array([F(F(F(-2 * z) * w) + F(F(2 * x) * y)), F(F(F(-2 * z) * z) + F(F(-2 * x) * x)),
                     F(F(F(2 * z) * y) + F(F(2 * x) * w)), F(0)], F)
    col1 = col1 + np.array([0, 1, 0, 0], F)
    col2 = np.array([F(F(F(2 * x) * z) + F(F(2 * y) * w)), F(F(F(-2 * x) * w) + F(F(2 * y) * z)),
                     F(F(F(-2 * x) * x) + F(F(-2 * y) * y)), F(0)], F)
    col2 = col2 + np.array([0, 0, 1, 0], F)
    sx, sy, sz = (F(v) for v in s)
    c0 = (col0 * sx).astype(F); c1 = (col1 * sy).astype(F); c2 = (col2 * sz).astype(F)
    c0[3] = c1[3] = c2[3] = F(0)
    c3 = np.array([F(t[0]), F(t[1]), F(t[2]), F(1)], F)
    return [c0, c1, c2, c3]


def prepared_compose(parent_cols, local_cols):
    """r15[k] column j = (L.x*P0 + L.y*P1) + (L.z*P2 + L.w*P3) (0xbdb751..0xbdb858)."""
    P0, P1, P2, P3 = parent_cols
    out = []
    for L in local_cols:
        out.append(((P0 * L[0] + P1 * L[1]) + (P2 * L[2] + P3 * L[3])).astype(F))
    return out


def bone_aabbs(vertices, weights, indices, bindposes_rowmajor):
    """Mesh::GetCachedBonesBounds 0x7a4770 -> 0x7a2890 (no blend shapes): per vertex and per
    influence with weight > 0, p = bindpose[bone] * v ((y*m01 + x*m00) + z*m02 + m03, ...),
    per-bone min/max (minss/maxss against the running value); TryPrepareStandardRenderer
    0xbde941 converts to centre = (min + max) * 0.5, extent = (max - min) * 0.5 and skips
    bones whose box stayed empty (0xbde8dc)."""
    nb = len(bindposes_rowmajor)
    mn = np.full((nb, 3), np.inf, F)
    mx = np.full((nb, 3), -np.inf, F)
    for i in range(len(vertices)):
        x, y, z = (F(c) for c in vertices[i])
        for k in range(4):
            if not (weights[i, k] > 0):
                continue
            b = int(indices[i, k])
            m = bindposes_rowmajor[b]
            p = np.array([F(F(F(y * m[0, 1]) + F(x * m[0, 0])) + F(z * m[0, 2])) + m[0, 3],
                          F(F(F(x * m[1, 0]) + F(y * m[1, 1])) + F(z * m[1, 2])) + m[1, 3],
                          F(F(F(x * m[2, 0]) + F(y * m[2, 1])) + F(z * m[2, 2])) + m[2, 3]], F)
            mn[b] = np.where(p < mn[b], p, mn[b])
            mx[b] = np.where(p > mx[b], p, mx[b])
    out = []
    for b in range(nb):
        if not np.isfinite(mn[b]).all():
            out.append(None)
            continue
        out.append((((mn[b] + mx[b]) * F(0.5)).astype(F), ((mx[b] - mn[b]) * F(0.5)).astype(F)))
    return out


def bone_world_aabb(bone_cols, aabbs):
    """0xbdb8d0..0xbdb9c3 + 0xbdb9dc..0xbdba06: centre' = ((cy*M1 + cz*M2) + cx*M0) + M3,
    extent' = (|ex*M0| + |ey*M1|) + |ez*M2|; running min/max of centre' -/+ extent';
    result centre = (max + min) * 0.5, extent = (max - min) * 0.5."""
    lo = np.full(4, np.inf, F)
    hi = np.full(4, -np.inf, F)
    for M, ab in zip(bone_cols, aabbs):
        if ab is None:
            continue
        c, e = ab
        M0, M1, M2, M3 = M
        cc = (((M1 * c[1]) + (M2 * c[2])) + (M0 * c[0])) + M3
        ee = (np.abs(M0 * e[0]) + np.abs(M1 * e[1])) + np.abs(M2 * e[2])
        lo = np.minimum(lo, (cc - ee).astype(F))
        hi = np.maximum(hi, (ee + cc).astype(F))
    return ((hi + lo) * F(0.5)).astype(F)[:3], ((hi - lo) * F(0.5)).astype(F)[:3]
