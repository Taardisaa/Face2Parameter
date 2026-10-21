"""Float32-exact transcription of Unity 2018.4.11f1 ``CalculateFocusedLightHull``.

Source: ``E:/HoneySelect2_ArcticFox/UnityPlayer.dll`` (SHA-256 ``5034cdc2…``,
CodeView GUID ``BFA91A8C44734BB4AABC5A975B304CFE`` age 1, public PDB
``UnityPlayer_Win64_mono_x64.pdb``).  Disassembly dumps reproduced with
``tools/offline_renderer/unityplayer_disasm.py`` live in
``outputs/offline_renderer_unity_forward_20261020/disasm/hull/``.

Transcribed functions (all callees of the root, recursively):

=========  =====================================================================
RVA        symbol
=========  =====================================================================
0x6deb40   ``CalculateFocusedLightHull(const Vector3f*, const Vector3f&,
           const MinMaxAABB&, dynamic_array<Vector3f>&)``
0x6df9d0   ``ClipPolysByPlane(UInt8, const Vector3f*, const Plane&, UInt8*,
           Vector3f*, UInt8*, Vector3f*)``
0x6eb280   ``IntersectSegmentPlane(const Vector3f&, const Vector3f&,
           const Plane&, Vector3f*)``
0x6e0830   unnamed (no public symbol): ray / MinMaxAABB slab clip returning the
           far (exit) point, here :func:`ray_minmax_aabb_far_point`
0x0e4580   ``MinMaxAABB::IsValid`` (checked by the caller, provided for the
           layout proof and for callers)
=========  =====================================================================

Non-arithmetic callees (``__chkstk``, ``memcpy``,
``dynamic_array_data::reserve``, ``dynamic_array<float3_storage>::grow``) only
manage storage; they never change a value and are represented by Python lists.

Recovered semantics
-------------------
* ``MinMaxAABB`` layout: ``m_Min`` (x, y, z) at +0x0..+0x8 then ``m_Max``
  (x, y, z) at +0xc..+0x14 (``IsValid`` compares +0x0.. with +inf and
  +0xc.. with -inf; the hull reads [+0]/[+4]/[+8] as min and
  [+0xc]/[+0x10]/[+0x14] as max).
* ``frustum`` = 8 world-space corners in ``GetFrustumPoints`` order (NDC
  (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), then the same with z = +1),
  here the output of ``GetFrustumPortion`` (point i = lerp(near i, far i+4))
  computed in fn 0x6a4400 from the inverse of the camera matrix at
  shadow-job +0x108.
* ``light_dir`` = the vector at light data +0x160..+0x168.  fn 0x6a4400 uses
  the same vector as the Z axis in ``SetPositionAndOrthoNormalBasis`` with the
  light-space origin placed at ``center - k*z`` (k > 0), so it is the light's
  forward vector, i.e. the direction light travels.  The hull extrudes along
  ``-light_dir`` (towards the light).
* ``aabb`` = the MinMaxAABB at shadow-job header +0x18 (``r13``).
* Algorithm: the six frustum faces (quads, see ``_FRUSTUM_QUADS``) are clipped
  successively by the six half-spaces of the AABB, in the order
  y >= min.y, y <= max.y, x >= min.x, x <= max.x, z >= min.z, z <= max.z.
  Each plane stage appends a "cap" polygon formed by the intersection points
  in encounter order (planes 0..4 only; no cap after the last plane).  For
  every vertex of every surviving polygon the hull receives the vertex and,
  when the ray ``vertex + t*(-light_dir)`` (t >= 0) overlaps the AABB, its far
  exit point.  The result is therefore a point cloud whose convex hull is the
  (frustum ∩ AABB) volume swept towards the light up to the AABB boundary; it
  is not deduplicated and is not itself a hull in face form.
* Output is appended to the given dynamic_array (the caller in 0x6a4400 passes
  a freshly constructed empty one at [rbp-0x38]).

Exactness: every SSE scalar op is IEEE binary32 round-to-nearest, reproduced
with numpy ``float32`` scalars in the machine's operand order; no double or
FMA arithmetic occurs in these routines.  Branches mirror ``comiss`` flag
semantics: ``ja``/``seta`` are false for unordered operands, ``jb``/``jbe`` are
taken for unordered operands, ``jae`` is not taken for unordered operands.
``maxss``/``minss`` return the second (source) operand unless the first is
strictly greater/less (so NaN in either operand yields the source).
"""
from __future__ import annotations

import struct

import numpy as np

F = np.float32
ZERO = F(0.0)
ONE = F(1.0)


def _bits(h):
    return F(struct.unpack('<f', struct.pack('<I', h))[0])


EPS_SEGMENT_PLANE = _bits(0x358637BD)   # __real@358637bd  (1e-6)
RAY_TMAX_INIT = _bits(0x7149F2CA)       # __real@7149f2ca  (1e30)
RAY_EPS = _bits(0x2EDBE6FF)             # __real@2edbe6ff  (1e-10)
RAY_NEG_EPS = _bits(0xAEDBE6FF)         # __real@aedbe6ff  (-1e-10)
POS_INF = F(np.inf)

# Frustum faces built on the stack at [rbp+0x190] (indices into the 8 points),
# each with vertex count 4 (bytes at [rbp+0x10..0x15]).
_FRUSTUM_QUADS = (
    (0, 1, 2, 3),   # near   (NDC z = -1)
    (7, 6, 5, 4),   # far    (NDC z = +1)
    (0, 3, 7, 4),   # x = -1
    (1, 5, 6, 2),   # x = +1
    (4, 5, 1, 0),   # y = -1
    (6, 7, 3, 2),   # y = +1
)

# Stack buffer capacities in the root function (bytes / 12): both ping-pong
# point buffers are 0xc00 bytes, the cap buffer at [rbp+0x1990] is 0xc60 bytes.
_STAGE_POINT_CAPACITY = 0xC00 // 12
_CAP_POINT_CAPACITY = 0xC60 // 12


class LightHullCapacityError(RuntimeError):
    """A stage exceeded a stack buffer or wrapped a UInt8 counter.

    The game does no bounds checks there (it would corrupt its own stack); that
    undefined behaviour is an explicit, unreproduced gap.  It cannot occur for
    clipped 6-quad frusta of practical size.
    """


def _v3(p):
    return (F(p[0]), F(p[1]), F(p[2]))


def _max_ss(a, b):
    """``maxss a, b`` -> a if a > b (ordered) else b."""
    return a if a > b else b


def _min_ss(a, b):
    """``minss a, b`` -> a if a < b (ordered) else b."""
    return a if a < b else b


def minmax_aabb_planes(aabb_min, aabb_max):
    """Planes (nx, ny, nz, d) built at [rbp-0x80..-0x24] of 0x6deb40.

    Inside means ``dot(n, p) + d >= 0`` (see :func:`clip_polys_by_plane`).
    The distances are ``-min`` (xorps sign flip) and ``max`` unchanged.
    """
    mn = _v3(aabb_min)
    mx = _v3(aabb_max)
    m1 = F(-1.0)
    return (
        (ZERO, ONE, ZERO, -mn[1]),
        (ZERO, m1, ZERO, mx[1]),
        (ONE, ZERO, ZERO, -mn[0]),
        (m1, ZERO, ZERO, mx[0]),
        (ZERO, ZERO, ONE, -mn[2]),
        (ZERO, ZERO, m1, mx[2]),
    )


def minmax_aabb_is_valid(aabb_min, aabb_max):
    """``MinMaxAABB::IsValid`` (RVA 0xe4580).

    False iff m_Min == (+inf, +inf, +inf) or m_Max == (-inf, -inf, -inf)
    (``ucomiss``/``jp``/``jne``: any NaN component makes that vector unequal).
    """
    mn = _v3(aabb_min)
    mx = _v3(aabb_max)
    if mn[0] == POS_INF and mn[1] == POS_INF and mn[2] == POS_INF:
        return False
    if mx[0] == -POS_INF and mx[1] == -POS_INF and mx[2] == -POS_INF:
        return False
    return True


def intersect_segment_plane(p0, p1, plane):
    """``IntersectSegmentPlane`` (RVA 0x6eb280).  Returns (ok, point or None).

    d = p1 - p0; denom = (nx*dx + ny*dy) + nz*dz; reject when
    1e-6 >= |denom - 0| (jae; NaN continues).  t = (-((ny*p0.y + nx*p0.x) +
    nz*p0.z) - plane.d) / denom; reject when 0 > t or t > 1 (ja; NaN
    continues).  point = (dx*t + p0.x, dy*t + p0.y, dz*t + p0.z).
    """
    nx, ny, nz, pd = plane
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    dz = p1[2] - p0[2]
    denom = (nx * dx + ny * dy) + nz * dz
    if EPS_SEGMENT_PLANE >= abs(denom - ZERO):
        return False, None
    t = (-((ny * p0[1] + nx * p0[0]) + nz * p0[2]) - pd) / denom
    if ZERO > t:
        return False, None
    if t > ONE:
        return False, None
    return True, (dx * t + p0[0], dy * t + p0[1], dz * t + p0[2])


def clip_polys_by_plane(points, plane, cap_points):
    """``ClipPolysByPlane`` (RVA 0x6df9d0) for one polygon.

    ``points``: the polygon's vertices; ``cap_points``: the shared cap list of
    the current plane stage (appended in place, UInt8 count in the game).
    Returns (kept, out_points).  ``kept`` is ``len(out_points) != 0``; a
    polygon with fewer than 3 vertices (unsigned ``cmp cl, 3; jae``) yields
    (False, []).

    Vertex i is outside when ``0 > ((p.y*ny + p.x*nx) + p.z*nz) + d``
    (``comiss``/``seta``: NaN counts as inside).  Edge i -> j = (i+1) mod n:
    inside->inside emits p[j]; inside->outside emits the intersection (also
    appended to the cap); outside->inside emits the intersection (also to the
    cap) then p[j]; outside->outside emits nothing.  IntersectSegmentPlane is
    always called as (p[i], p[j]); a rejected intersection emits nothing.
    """
    n = len(points)
    if n > 255:
        raise LightHullCapacityError('polygon vertex count exceeds UInt8')
    if n < 3:
        return False, []
    nx, ny, nz, pd = plane
    outside = []
    for p in points:
        dist = ((p[1] * ny + p[0] * nx) + p[2] * nz) + pd
        outside.append(bool(ZERO > dist))
    out = []
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        if not outside[i]:
            if not outside[j]:
                out.append(points[j])
            else:
                ok, q = intersect_segment_plane(points[i], points[j], plane)
                if ok:
                    out.append(q)
                    cap_points.append(q)
        else:
            if outside[j]:
                continue
            ok, q = intersect_segment_plane(points[i], points[j], plane)
            if ok:
                out.append(q)
                cap_points.append(q)
            out.append(points[j])
    if len(out) > 255 or len(cap_points) > 255:
        raise LightHullCapacityError('UInt8 vertex counter would wrap')
    return len(out) != 0, out


def _slab(num, denom, tmin, tmax):
    """One half-space step of fn 0x6e0830; returns (ok, tmin, tmax).

    ``-1e-10 > denom``: t = num/denom, fail if t > tmax, tmin = maxss(t, tmin).
    ``denom > 1e-10``:  t = num/denom, fail if tmin > t, tmax = minss(t, tmax).
    otherwise:          fail if not (num >= 0)  (jb: NaN fails).
    """
    if RAY_NEG_EPS > denom:
        t = num / denom
        if t > tmax:
            return False, tmin, tmax
        return True, _max_ss(t, tmin), tmax
    if denom > RAY_EPS:
        t = num / denom
        if tmin > t:
            return False, tmin, tmax
        return True, tmin, _min_ss(t, tmax)
    if not (num >= ZERO):
        return False, tmin, tmax
    return True, tmin, tmax


def ray_minmax_aabb_far_point(origin, direction, aabb_min, aabb_max, out=None):
    """Unnamed fn at RVA 0x6e0830: ``bool f(Vector3f* out, const Vector3f&
    origin, const Vector3f& dir, const MinMaxAABB&)``.

    Slab clip of t in [0, 1e30] against the half-spaces, in machine order
    (z>=min.z, z<=max.z, y>=min.y, y<=max.y, x>=min.x, x<=max.x), each with
    num = origin - min (denominator -dir) or max - origin (denominator dir).
    Then: if tmin >= 0, out = (dir*tmin + origin), result true; and if
    tmax >= 0, out = (tmax*dir + origin), result true (overwriting).  On
    failure ``out`` is left untouched.  Returns (ok, out).
    """
    o = origin
    d = direction
    mn = aabb_min
    mx = aabb_max
    tmin = ZERO
    tmax = RAY_TMAX_INIT
    steps = (
        (lambda: o[2] - mn[2], lambda: -d[2]),
        (lambda: mx[2] - o[2], lambda: d[2]),
        (lambda: o[1] - mn[1], lambda: -d[1]),
        (lambda: mx[1] - o[1], lambda: d[1]),
        (lambda: o[0] - mn[0], lambda: -d[0]),
        (lambda: mx[0] - o[0], lambda: d[0]),
    )
    for num_f, den_f in steps:
        ok, tmin, tmax = _slab(num_f(), den_f(), tmin, tmax)
        if not ok:
            return False, out
    result = False
    if tmin >= ZERO:
        out = (d[0] * tmin + o[0], d[1] * tmin + o[1], d[2] * tmin + o[2])
        result = True
    if not (tmax >= ZERO):
        return result, out
    out = (tmax * d[0] + o[0], tmax * d[1] + o[1], tmax * d[2] + o[2])
    return True, out


def frustum_polygons(frustum_points8):
    """The six quads written at [rbp+0x190] of 0x6deb40."""
    pts = [_v3(p) for p in frustum_points8]
    if len(pts) != 8:
        raise ValueError('CalculateFocusedLightHull needs 8 frustum points')
    return [[pts[i] for i in quad] for quad in _FRUSTUM_QUADS]


def clip_frustum_by_aabb(frustum_points8, aabb_min, aabb_max):
    """The six-stage ping-pong clip loop of 0x6deb40 (0x6dee90..0x6defba).

    Returns the list of surviving polygons of the last stage.
    """
    polys = frustum_polygons(frustum_points8)
    planes = minmax_aabb_planes(aabb_min, aabb_max)
    for plane_index, plane in enumerate(planes):
        cap = []
        dst = []
        remaining = len(polys)
        if remaining:
            for poly in polys:
                kept, out = clip_polys_by_plane(poly, plane, cap)
                if kept:
                    dst.append(out)
                else:
                    remaining -= 1
                    if remaining == 0:
                        break
        # cap appended only when non-empty and plane index < 5 (cmp r9d, 5; jge)
        if len(cap) != 0 and plane_index < 5:
            dst.append(list(cap))
        if sum(len(p) for p in dst) > _STAGE_POINT_CAPACITY or len(cap) > _CAP_POINT_CAPACITY:
            raise LightHullCapacityError('stage exceeds the stack point buffers')
        polys = dst
    return polys


def calculate_focused_light_hull(frustum_points8, light_dir, aabb, out=None):
    """``CalculateFocusedLightHull`` (RVA 0x6deb40).

    ``frustum_points8``: 8 world-space corners (GetFrustumPoints order).
    ``light_dir``: the light forward vector (light data +0x160).
    ``aabb``: MinMaxAABB as ``(min_xyz, max_xyz)`` or 6 floats
    ``[min.x, min.y, min.z, max.x, max.y, max.z]`` (memory order).
    ``out``: optional list to append to (dynamic_array append semantics).

    Returns the list of float32 3-tuples: for every vertex of every polygon of
    the clipped frustum (polygon order, vertex order), the vertex followed by
    its far exit point along ``-light_dir`` through the AABB when that ray
    overlaps the AABB.
    """
    if len(aabb) == 6:
        aabb_min, aabb_max = _v3(aabb[0:3]), _v3(aabb[3:6])
    else:
        aabb_min, aabb_max = _v3(aabb[0]), _v3(aabb[1])
    ld = _v3(light_dir)
    if out is None:
        out = []
    with np.errstate(all='ignore'):
        polys = clip_frustum_by_aabb(frustum_points8, aabb_min, aabb_max)
        if not polys:
            return out
        # [rsp+0x58] initialised once from Vector3f::zero; it is only read
        # after a call that returned true, which always writes it first.
        tmp = (ZERO, ZERO, ZERO)
        neg_dir = (-ld[0], -ld[1], -ld[2])
        for poly in polys:
            for v in poly:
                out.append(v)
                ok, tmp = ray_minmax_aabb_far_point(v, neg_dir, aabb_min, aabb_max, tmp)
                if ok:
                    out.append(tmp)
    return out
