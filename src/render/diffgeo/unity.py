"""Unity 2018.4.11f1 transform / skinning arithmetic over torch (both arithmetic modes).

Native routines (UnityPlayer.dll) use the lifted lane formulas of ``native_generated``
(``tools/diffgeo/derive_native.py``); managed routines follow the ports already accepted in the
exact chain:

* ``normalize_quaternion``  -- ``src.render.unity_transform_math.normalize_quaternion``
  (SetLocalR 0x1b12f0: per-lane horizontal sums, sqrtps, divps, 1e-30 fallback);
* ``managed_quaternion_mul`` -- ``Quaternion.operator*`` under Mono (double, one rounding per field);
* ``quaternion_euler_degrees`` -- managed ``euler * Deg2Rad`` (binary32 products), then native
  Internal_FromEulerRad 0x9541d0 -> EulerToQuaternion 0x7cc120 (order 4): half angles
  (mulss by 0.5), CRT sinf/cosf, ``euler_case4``;
* ``global_position_rotation``  -- CalculateGlobalPositionAndRotation 0xdda60 (``gpr_step`` per parent);
* ``world_to_local_noscale``    -- ``inv_noscale``;
* ``prepared_palette``          -- CalculateSkinningMatrices prepared path (``local_compose`` per
  entry, parents first; ``bind_mul`` per bone), evaluated level by level of the hierarchy;
* ``skin_vertices4``            -- SkinVerticesGenericCPU<4,1,1> (``skin4``), vectorised over vertices;
* ``object_to_world``           -- unity_ObjectToWorld = TRS(root position, root rotation, 1) with
  ``Matrix4x4.Rotate``'s binary32 expression order (``gpu_skin_inputs.unity_rotation_matrix``).
"""
from __future__ import annotations

import numpy as np
import torch

from . import native_generated as ng
from .fp import DT

NORM_EPS = float(np.float32(1e-30))
MEMO_LIMIT = 200000
DEG2RAD = float(np.float32(0.0174532924))
ZERO_SQR = float(np.float32(9.9999994e-11))


def stack(xs):
    return torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=DT) for x in xs])


def normalize_quaternion(A, q):
    """q: tensor (..., 4) -> SetLocalR-normalised tensor (..., 4)."""
    R = A.R
    v = [R(q[..., i] * q[..., i]) for i in range(4)]
    t = [R(v[i] + v[(i + 1) % 4]) for i in range(4)]
    s = [R(t[i] + t[(i + 2) % 4]) for i in range(4)]
    out = []
    for i in range(4):
        safe = torch.where(s[i] > NORM_EPS, s[i], torch.ones_like(s[i]))
        n = R(q[..., i] / R(torch.sqrt(safe)))
        out.append(torch.where(s[i] > NORM_EPS, n, torch.full_like(n, 1.0 if i == 3 else 0.0)))
    return torch.stack(out, -1)


def managed_quaternion_mul(A, a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return torch.stack([A.R(aw * bx + ax * bw + ay * bz - az * by),
                        A.R(aw * by + ay * bw + az * bx - ax * bz),
                        A.R(aw * bz + az * bw + ax * by - ay * bx),
                        A.R(aw * bw - ax * bx - ay * by - az * bz)], -1)


_EULER_MEMO = {}


def quaternion_euler_degrees(A, deg):
    """deg: tensor (..., 3) of binary32 degrees -> quaternion (..., 4).  A constant argument (no
    grad) is memoized by its bits (the routine is a pure function)."""
    if not deg.requires_grad:
        key = (A.mode, deg.detach().numpy().tobytes(), tuple(deg.shape))
        hit = _EULER_MEMO.get(key)
        if hit is None:
            hit = _EULER_MEMO[key] = _euler(A, deg)
        return hit
    return _euler(A, deg)


def _euler(A, deg):
    R = A.R
    half = [R(R(deg[..., i] * DEG2RAD) * 0.5) for i in range(3)]
    sc = []
    for h in half:
        sc += [A.sinf(h), A.cosf(h)]
    return torch.stack(ng.euler_case4(R, A.S, A.sqrt, *sc), -1)


def sqrmag(A, v):
    """Vector3.SqrMagnitude under Mono: x*x + y*y + z*z in double, rounded once."""
    return A.R(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])


def unity_zero(A, v):
    """Vector3 == Vector3.zero (sqrMagnitude < 9.99999944e-11); a branch condition (bool).

    Decided on the binary32 value in both arithmetic modes, so the float64 mode follows the same
    branches as the exact mode."""
    x, y, z = (float(np.float32(c)) for c in v.detach().cpu().numpy())
    return bool(np.float32(x * x + y * y + z * z) < np.float32(ZERO_SQR))


def global_position_rotation(A, chain, memo=None):
    """chain: [(t(3), q(4), s(3)) of the transform, its parent, ... the top] -> (pos(3), rot(4)).
    ``memo``: a chain without gradient is looked up by the bits of all its TRS."""
    key = None
    if memo is not None and not any(x.requires_grad for c in chain for x in c):
        key = (A.mode,) + tuple(_bits(x) for c in chain for x in c)
        hit = memo.get(key)
        if hit is not None:
            return hit
    out = _global_position_rotation(A, chain)
    if key is not None:
        if len(memo) > MEMO_LIMIT:
            memo.clear()
        memo[key] = out
    return out


def _global_position_rotation(A, chain):
    t, q, _ = chain[0]
    pos = [t[..., i] for i in range(3)]
    rot = [q[..., i] for i in range(4)]
    for pt, pq, ps in chain[1:]:
        out = ng.gpr_step(A.R, A.S, A.sqrt, pos, rot, [pt[..., i] for i in range(3)], [pq[..., i] for i in range(4)],
                          [ps[..., i] for i in range(3)])
        pos, rot = list(out[:3]), list(out[3:])
    return torch.stack(pos, -1), torch.stack(rot, -1)


def _full(x, like):
    return x if isinstance(x, torch.Tensor) else torch.full_like(like, x)


def world_to_local_noscale(A, pos, rot):
    """-> (16,) column-major."""
    out = ng.inv_noscale(A.R, A.S, A.sqrt, [pos[..., i] for i in range(3)], [rot[..., i] for i in range(4)])
    return torch.stack([_full(x, pos[..., 0]) for x in out], -1)


def _bits(x):
    return x.detach().numpy().tobytes()


def prepared_poses(A, base, entries, memo=None):
    """Pose rows of the prepared hierarchy: rows[0] = base (16,) column-major; entries:
    [(parent entry index (0 = base), t(3), q(4), s(3))] for entry 1..N, parents first.
    rows[i] = rows[parent] * TRS(entry i) (``local_compose``), evaluated level by level.

    ``memo`` (optional dict): rows whose inputs carry no gradient are pure functions of the bits
    of the base and of the TRS on their path; they are looked up by exactly those bits (a
    nested key of the path) and evaluated only once."""
    n = len(entries) + 1
    depth = [0] * n
    for i, (p, *_ ) in enumerate(entries, start=1):
        if not 0 <= p < i:
            raise ValueError('parent must precede child')
        depth[i] = depth[p] + 1
    rows = [None] * n
    rows[0] = base
    keys = [None] * n
    if memo is not None and len(memo) > MEMO_LIMIT:
        memo.clear()
    if memo is not None and not base.requires_grad:
        keys[0] = (A.mode, _bits(base))
    for i in range(1, n):
        p, t, q, s_ = entries[i - 1]
        if keys[p] is not None and not (t.requires_grad or q.requires_grad or s_.requires_grad):
            keys[i] = (keys[p], _bits(t), _bits(q), _bits(s_))
            hit = memo.get(keys[i])
            if hit is not None:
                rows[i] = hit
    levels = {}
    for i in range(1, n):
        if rows[i] is None:
            levels.setdefault(depth[i], []).append(i)
    for d in sorted(levels):
        idx = levels[d]
        P = torch.stack([rows[entries[i - 1][0]] for i in idx])
        T = torch.stack([entries[i - 1][1] for i in idx])
        Q = torch.stack([entries[i - 1][2] for i in idx])
        Sc = torch.stack([entries[i - 1][3] for i in idx])
        out = ng.local_compose(A.R, A.S, A.sqrt, [T[:, j] for j in range(3)], [Q[:, j] for j in range(4)],
                               [Sc[:, j] for j in range(3)], [P[:, j] for j in range(16)])
        M = torch.stack([_full(x, P[:, 0]) for x in out], -1)
        for k, i in enumerate(idx):
            rows[i] = M[k]
            if keys[i] is not None:
                memo[keys[i]] = M[k].detach()
    return rows


def palette_from_poses(A, pose_rows, bindposes):
    """Bone loop: palette[k] = pose_rows[k] * bindposes[k] (``bind_mul``); (K, 16) column-major."""
    Pm = torch.stack(list(pose_rows))
    B = bindposes
    out = ng.bind_mul(A.R, A.S, A.sqrt, [Pm[:, j] for j in range(16)], [B[:, j] for j in range(16)])
    return torch.stack(list(out), -1)


def prepared_palette(A, base, entries, bone_map, bindposes):
    """CalculateSkinningMatrices prepared path for one renderer: (K, 16) column-major palette."""
    rows = prepared_poses(A, base, entries)
    return palette_from_poses(A, [rows[b] for b in bone_map], bindposes)


def skin_vertices4(A, src, palette, weights, indices):
    """src (n, 10) pos/normal/tangent xyzw; palette (K, 16); weights (n, 4); indices (n, 4) int.
    Returns (n, 10) -- the 40-byte skinned vertices."""
    idx = torch.as_tensor(np.asarray(indices, np.int64))
    if int(idx.min()) < 0 or int(idx.max()) >= palette.shape[0]:
        raise ValueError('bone index outside the palette')
    M = [palette[idx[:, k]] for k in range(4)]
    w = [weights[:, k] for k in range(4)]
    out = ng.skin4(A.R, A.S, A.sqrt, w, *[[m[:, j] for j in range(16)] for m in M], [src[:, j] for j in range(10)])
    return torch.stack(list(out), -1)


def object_to_world(A, pos, rot):
    """unity_ObjectToWorld (4, 4) row-major = TRS(pos, rot, 1), Matrix4x4.Rotate binary32 order."""
    R = A.R
    x, y, z, w = (rot[..., i] for i in range(4))
    X, Y, Z = R(x * 2.0), R(y * 2.0), R(z * 2.0)
    xx, yy, zz = R(x * X), R(y * Y), R(z * Z)
    xy, xz, yz = R(x * Y), R(x * Z), R(y * Z)
    wx, wy, wz = R(w * X), R(w * Y), R(w * Z)
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    rows = [[R(1.0 - R(yy + zz)), R(xy - wz), R(xz + wy), pos[..., 0]],
            [R(xy + wz), R(1.0 - R(xx + zz)), R(yz - wx), pos[..., 1]],
            [R(xz - wy), R(yz + wx), R(1.0 - R(xx + yy)), pos[..., 2]],
            [zero, zero, zero, one]]
    return torch.stack([torch.stack(r, -1) for r in rows], -2)
