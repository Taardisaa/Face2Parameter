"""Directional shadow cascade fitting of UnityPlayer 2018.4.11f1 (float32-exact).

Transcribed from the installed UnityPlayer.dll (see ``unity_math`` for the
provenance and arithmetic conventions):

* per-cascade setup: unnamed function at RVA 0x6a4400, called from
  ``PrepareShadowMaps`` (0x69f4f0) at 0x6a0309 once per cascade;
* ``GetFrustumPoints`` 0x6e03d0, ``Matrix4x4f::PerspectiveMultiplyPoint3``
  0x2a17c0, ``GetFrustumPortion`` 0x6e0520,
  ``CalculateBoundingSphereFromFrustumPoints`` 0x6de970,
  ``CalculateSphereFrom4Points`` 0x6df190, ``Matrix4x4f::GetDeterminant``
  0x7cca00, ``CalculateShadowSphereOffset`` 0x679580,
  ``Matrix4x4f::SetPositionAndOrthoNormalBasis`` 0x7d0ce0.

The ShadowMapJobHeader fields read by 0x6a4400 are passed explicitly in
``CascadeInputs``; their mapping to Unity state (from ``SetupShadowCullData``
0x6815f0, ``Camera::CustomCull`` 0x660f70 and helper 0x69f180) is documented
on each field.  The close-fit hull (``CalculateFocusedLightHull`` 0x6deb40) is
provided by ``unity_light_hull``; the two MinMaxAABB inputs by
``unity_shadow_bounds`` (both separate transcriptions).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import unity_math as um

F = np.float32
EPS_VECTOR3 = F(1e-5)        # ?epsilon@Vector3f@@2MB, read from the DLL .rdata at RVA 0x1257060 (0x3727c5ac)
INF = F(np.inf)


def _sqrtf(x):
    return F(math.sqrt(float(x))) if x >= F(0.0) else F(np.nan)


def perspective_multiply_point3(m, v):
    """RVA 0x2a17c0: returns (ok, point)."""
    vx, vy, vz = v
    x = ((vy * m[4] + vx * m[0]) + vz * m[8]) + m[12]
    y = ((vx * m[1] + vy * m[5]) + vz * m[9]) + m[13]
    w = ((vx * m[3] + vy * m[7]) + vz * m[11]) + m[15]
    z = ((vx * m[2] + vy * m[6]) + vz * m[10]) + m[14]
    if abs(w) > F(1e-7):
        r = F(1.0) / w
        return True, (x * r, r * y, r * z)
    return False, (F(0.0), F(0.0), F(0.0))


FRUSTUM_CORNERS = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                   (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))


def get_frustum_points(clip_to_world):
    """RVA 0x6e03d0: the eight clip-cube corners through PerspectiveMultiplyPoint3."""
    return [perspective_multiply_point3(clip_to_world, tuple(F(c) for c in corner))[1] for corner in FRUSTUM_CORNERS]


def get_frustum_portion(pts, a, b):
    """RVA 0x6e0520: near quad lerped at a, far quad at b: t*far + (1-t)*near."""
    a, b = F(a), F(b)
    one = F(1.0)
    out = []
    for t in (a, b):
        u = one - t
        for i in range(4):
            n, f = pts[i], pts[i + 4]
            out.append(tuple(t * f[k] + u * n[k] for k in range(3)))
    return out


def get_determinant(m):
    """``Matrix4x4f::GetDeterminant`` (RVA 0x7cca00): 24-term expansion in double,
    left-to-right, products ((a*b)*c)*d, rounded to float."""
    g = lambda r, c: float(m[c * 4 + r])  # noqa: E731
    m00, m01, m02, m03 = g(0, 0), g(0, 1), g(0, 2), g(0, 3)
    m10, m11, m12, m13 = g(1, 0), g(1, 1), g(1, 2), g(1, 3)
    m20, m21, m22, m23 = g(2, 0), g(2, 1), g(2, 2), g(2, 3)
    m30, m31, m32, m33 = g(3, 0), g(3, 1), g(3, 2), g(3, 3)
    r = (m03 * m12 * m21 * m30 - m02 * m13 * m21 * m30 - m03 * m11 * m22 * m30 + m01 * m13 * m22 * m30 +
         m02 * m11 * m23 * m30 - m01 * m12 * m23 * m30 - m03 * m12 * m20 * m31 + m02 * m13 * m20 * m31 +
         m03 * m10 * m22 * m31 - m00 * m13 * m22 * m31 - m02 * m10 * m23 * m31 + m00 * m12 * m23 * m31 +
         m03 * m11 * m20 * m32 - m01 * m13 * m20 * m32 - m03 * m10 * m21 * m32 + m00 * m13 * m21 * m32 +
         m01 * m10 * m23 * m32 - m00 * m11 * m23 * m32 - m02 * m11 * m20 * m33 + m01 * m12 * m20 * m33 +
         m02 * m10 * m21 * m33 - m00 * m12 * m21 * m33 - m01 * m10 * m22 * m33 + m00 * m11 * m22 * m33)
    return F(r)


def _rows_matrix(rows):
    """Matrix4x4f whose row i is rows[i] (m[c*4 + i] = rows[i][c])."""
    m = [F(0.0)] * 16
    for i, row in enumerate(rows):
        for c in range(4):
            m[c * 4 + i] = F(row[c])
    return m


def calculate_sphere_from_4_points(p):
    """RVA 0x6df190: circumsphere through determinants (rows as in the binary)."""
    one = F(1.0)
    s = [(q[1] * q[1] + q[0] * q[0]) + q[2] * q[2] for q in p]
    a = get_determinant(_rows_matrix([(q[0], q[1], q[2], one) for q in p]))
    dx = get_determinant(_rows_matrix([(s[i], p[i][1], p[i][2], one) for i in range(4)]))
    dy = get_determinant(_rows_matrix([(p[i][0], s[i], p[i][2], one) for i in range(4)]))
    dz = get_determinant(_rows_matrix([(p[i][0], p[i][1], s[i], one) for i in range(4)]))
    c = get_determinant(_rows_matrix([(s[i], p[i][0], p[i][1], p[i][2]) for i in range(4)]))
    da = float(a)
    cx = F(float(dx) * 0.5 / da)
    cz = F(float(dz) * 0.5 / da)
    cy = F(float(dy) * 0.5 / da)
    r2 = ((cy * cy + cx * cx) + cz * cz) - c / a
    return (cx, cy, cz), _sqrtf(r2)


def calculate_bounding_sphere_from_frustum_points(pts):
    """RVA 0x6de970: try midpoints of (p5,p7) then (p0,p3); else 4-point sphere of p0,p3,p5,p7."""
    four = [pts[0], pts[3], pts[5], pts[7]]
    half = F(0.5)
    for a_i, b_i, o1, o2 in ((2, 3, 0, 1), (0, 1, 2, 3)):
        pa, pb = four[a_i], four[b_i]
        c = ((pa[0] + pb[0]) * half, (pa[1] + pb[1]) * half, (pa[2] + pb[2]) * half)

        def d2(q):
            dx, dy, dz = q[0] - c[0], q[1] - c[1], q[2] - c[2]
            return (dy * dy + dx * dx) + dz * dz
        r2 = d2(pa)
        if r2 >= d2(four[o1]) and r2 >= d2(four[o2]):
            return c, _sqrtf(r2)
    return calculate_sphere_from_4_points(four)


def calculate_shadow_sphere_offset(orthographic, fov):
    """RVA 0x679580."""
    if orthographic:
        return F(0.4)
    x = F(1.0) - F(fov) / F(180.0)
    if F(0.0) > x:
        x = F(0.0)
    else:
        x = F(1.0) if F(1.0) < x else x
    return x * F(0.4)


@dataclass
class CascadeInputs:
    """ShadowMapJobHeader fields read by 0x6a4400 (offsets relative to the header)."""
    projection_type: str                    # +0x20 QualitySettings.shadowProjection ('CloseFit'|'StableFit')
    clip_to_world: list                     # +0x108 camera clip->world (cull+0x40 <- camera cull params +0x25c)
    camera_near: object                     # +0x430 (cull+0x368)
    camera_far: object                      # +0x434 (cull+0x36c)
    shadow_distance: object                 # +0x42c min(camera far, QualitySettings.shadowDistance)
    split_percentages: list                 # +0x10d4c (helper 0x69f180 / split function)
    cascade_count: int                      # +0x10d74
    near_plane_offset: object               # +0x38 QualitySettings.shadowNearPlaneOffset
    reversed_z: bool                        # +0x11143
    light_axes: tuple                       # SharedLightData +0x140/+0x150/+0x160 (light localToWorld columns 0..2)
    caster_bounds: tuple                    # 5th argument MinMaxAABB (min3, max3) or None
    receiver_bounds: tuple = None           # header +0x18 MinMaxAABB (close fit only)
    camera_projection: list = None          # +0x44 (StableFit only: camera projection)
    camera_to_world: list = None            # +0x84 (StableFit only)
    camera_orthographic: bool = False       # +0x345
    camera_fov: object = None               # +0x440
    hull_function: object = None            # unity_light_hull.calculate_focused_light_hull


@dataclass
class CascadeResult:
    ok: bool
    light_matrix: list = None               # out+4: SetPositionAndOrthoNormalBasis
    projection: list = None                 # out+0x84: SetOrtho
    view: list = None                       # out+0x44: inverse(light matrix with Z negated)
    shadow_vp_tex: list = None              # out+0xc4: proj' * view (reversed-Z adjusted, no texture scale)
    world_to_shadow: list = None            # out+0x104: T * proj' * view
    sphere: tuple = None                    # out+0x144..0x150 (centre, radius)  (StableFit)
    texel_size: tuple = None                # out+0x154/+0x158
    split_near: object = None               # out+0x15c
    split_far: object = None                # out+0x160
    light_near: object = None               # out+0x164
    light_far: object = None                # out+0x168
    notes: list = field(default_factory=list)


def _identity_result():
    return CascadeResult(False, light_matrix=um.identity(), projection=um.set_ortho(-1, 1, -1, 1, 0.1, 10.0))


def directional_cascade(inp, cascade_index, width, height):
    """Transcription of RVA 0x6a4400 for one cascade."""
    if inp.caster_bounds is None:
        return _identity_result()
    bmin, bmax = [tuple(F(v) for v in t) for t in inp.caster_bounds]
    stable = inp.projection_type == 'StableFit'
    far = F(inp.camera_far)
    near = F(inp.camera_near)
    shadow_distance = F(inp.shadow_distance)
    factor = F(1.0)
    frustum_matrix = inp.clip_to_world
    res = CascadeResult(True)
    if stable:
        ok, inv = um.invert_full(inp.camera_projection)
        w = ((inv[3] + inv[7]) + inv[11]) + inv[15]
        x = ((inv[0] + inv[4]) + inv[8]) + inv[12]
        y = ((inv[1] + inv[5]) + inv[9]) + inv[13]
        z = ((inv[2] + inv[6]) + inv[10]) + inv[14]
        if abs(w) > F(1e-7):
            frustum_matrix = inv
            r = F(1.0) / w
            yy, xx, zz = y * r, x * r, z * r
            k = _sqrtf((yy * yy + xx * xx) + zz * zz) / far
            o = calculate_shadow_sphere_offset(inp.camera_orthographic, inp.camera_fov)
            k2 = k * k
            t = ((k2 * (F(1.0) - o)) * (F(1.0) - o) - (k2 * o) * o) + o * o
            factor = (_sqrtf(t) + o) / k2
    depth = far - near
    frac = (shadow_distance * factor - near) / depth
    if not (EPS_VECTOR3 < frac or np.isnan(frac)):   # comiss eps, frac; jb continue (unordered continues)
        res.ok = False
        res.notes.append('shadow range fraction <= Vector3f::epsilon: cascade skipped')
        return res
    pts = get_frustum_points(frustum_matrix)
    s0 = frac * F(inp.split_percentages[cascade_index])
    s1 = frac * F(inp.split_percentages[cascade_index + 1])
    res.split_near = depth * s0 + near
    res.split_far = depth * s1 + near
    portion = get_frustum_portion(pts, s0, s1)
    X, Y, Z = [tuple(F(v) for v in a) for a in inp.light_axes]
    hull = None
    if not stable:
        if inp.hull_function is None:
            raise NotImplementedError('CloseFit requires unity_light_hull.calculate_focused_light_hull')
        hull = inp.hull_function(portion, Z, inp.receiver_bounds)
        if not hull:
            r = _identity_result()
            r.notes.append('empty focused light hull')
            return r
    # initial light matrix from the caster bounds
    dx, dy, dz = bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2]
    diag = _sqrtf((dy * dy + dx * dx) + dz * dz)
    half = diag * F(0.5)
    pos = tuple((bmin[k] + bmax[k]) * F(0.5) - (half * Z[k]) * F(1.2) for k in range(3))
    L = [X[0], X[1], X[2], F(0.0), Y[0], Y[1], Y[2], F(0.0), Z[0], Z[1], Z[2], F(0.0), pos[0], pos[1], pos[2], F(1.0)]

    def to_light(p):
        ex, ey, ez = p[0] - L[12], p[1] - L[13], p[2] - L[14]
        lx = (L[0] * ex + L[1] * ey) + L[2] * ez
        ly = (L[4] * ex + L[5] * ey) + L[6] * ez
        lz = (L[8] * ex + L[9] * ey) + L[10] * ez
        return lx, ly, lz
    mn = [INF, INF, INF]
    mx = [-INF, -INF, -INF]
    if not stable:
        for p in hull:
            lp = to_light(tuple(F(v) for v in p))
            for k in range(3):
                mn[k] = lp[k] if lp[k] < mn[k] else mn[k]
                mx[k] = lp[k] if lp[k] > mx[k] else mx[k]
    else:
        center, radius = calculate_bounding_sphere_from_frustum_points(portion)
        cz_abs = abs(center[2]) + radius
        res.split_far = shadow_distance if shadow_distance < cz_abs else cz_abs
        c2w = inp.camera_to_world
        cx, cy, cz = center
        wc = (((cy * c2w[4] + cx * c2w[0]) + cz * c2w[8]) + c2w[12],
              ((cy * c2w[5] + cx * c2w[1]) + cz * c2w[9]) + c2w[13],
              ((cy * c2w[6] + cx * c2w[2]) + cz * c2w[10]) + c2w[14])
        res.sphere = (wc, radius)
        lp = to_light(wc)
        for k in range(3):
            mn[k] = lp[k] - radius
            mx[k] = lp[k] + radius
    ex = mx[0] - mn[0]
    ey = mx[1] - mn[1]
    ez = mx[2] - mn[2]
    Wf, Hf = F(width), F(height)
    if inp.cascade_count > 1:
        ez = ez + F(0.0)
        ex = (ex / Wf) * F(10.0) + ex
        ey = (ey / Hf) * F(10.0) + ey
    cy = (mx[1] + mn[1]) * F(0.5)
    cz = (mx[2] + mn[2]) * F(0.5)
    cx = (mx[0] + mn[0]) * F(0.5)
    tex_x = ex / Wf
    tex_y = ey / Hf
    res.texel_size = (tex_x, tex_y)
    wx = ((cy * L[4] + cx * L[0]) + cz * L[8]) + L[12]
    wy = ((cy * L[5] + cx * L[1]) + cz * L[9]) + L[13]
    wz = ((cy * L[6] + cx * L[2]) + cz * L[10]) + L[14]
    d_x = (float(X[1]) * float(wy) + float(X[0]) * float(wx)) + float(X[2]) * float(wz)
    from .unity_crt import fmod as crt_fmod
    snap_x = F(crt_fmod(d_x, float(tex_x)))
    d_y = (float(Y[0]) * float(wx) + float(Y[1]) * float(wy)) + float(Y[2]) * float(wz)
    snap_y = F(crt_fmod(d_y, float(tex_y)))
    half_x = ex * F(0.5)
    half_z = ez * F(0.5)
    half_y = ey * F(0.5)
    k12 = F(1.2)
    px = ((wx - snap_x * X[0]) - snap_y * Y[0]) - (Z[0] * half_z) * k12
    py = ((wy - snap_x * X[1]) - snap_y * Y[1]) - (Z[1] * half_z) * k12
    pz = ((wz - snap_x * X[2]) - snap_y * Y[2]) - (Z[2] * half_z) * k12
    light_near = half_z * F(0.1)
    light_far = half_z * F(2.3)
    off = F(inp.near_plane_offset)
    off = off if off > F(0.0) else F(0.0)
    light_near = light_near - off * F(10.0)
    res.light_near, res.light_far = light_near, light_far
    L[12], L[13], L[14] = px, py, pz
    res.light_matrix = list(L)
    proj = um.set_ortho(-half_x, half_x, -half_y, half_y, light_near, light_far)
    res.projection = proj
    vsrc = list(L)
    vsrc[8], vsrc[9], vsrc[10] = -vsrc[8], -vsrc[9], -vsrc[10]
    ok, view = um.invert_full(vsrc)
    res.view = view
    texm = um.identity()
    for i in (0, 5, 10):
        texm[i] = F(0.5)
    texm[12] = texm[13] = texm[14] = F(0.5)
    pr = list(proj)
    if inp.reversed_z:
        for c in range(4):
            pr[2 + 4 * c] = -pr[2 + 4 * c]
    res.shadow_vp_tex = um.multiply(pr, view)
    res.world_to_shadow = um.multiply(texm, res.shadow_vp_tex)
    return res


def shadow_caster_view_projection(result, reversed_z=True):
    """VP used while rendering the cascade into the shadow map: the GfxDevice
    projection path (CalculateDeviceProjectionMatrix with invertY because the
    shadow map is a render texture) times the cascade view."""
    dev = um.calculate_device_projection(result.projection, False, True, reversed_z)
    return um.multiply(dev, result.view), dev


def unused_cascade_world_to_shadow():
    """unity_WorldToShadow entries beyond the cascade count (capture_observed:
    zero 3x3 with column 3 = (0,0,1,1); source writer not yet located: gap)."""
    m = [F(0.0)] * 16
    m[14] = F(1.0)
    m[15] = F(1.0)
    return m
