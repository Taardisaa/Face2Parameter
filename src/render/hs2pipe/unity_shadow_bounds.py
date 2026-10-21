"""Float32-exact reimplementation of the bounds that drive Unity 2018.4.11f1
built-in forward directional shadows (installed ``E:/HoneySelect2_ArcticFox/
UnityPlayer.dll``, SHA-256 ``5034cdc2...``, CodeView GUID
``BFA91A8C44734BB4AABC5A975B304CFE`` age 1).

Two MinMaxAABBs reach the per-cascade function ``0x6a4400``:

* **caster bounds** -- the job function ``0x69fab0`` scheduled by
  ``PrepareShadowMaps`` (``0x69f4f0``) builds it on its stack at ``rsp+0x50``
  (``0x69fbf4..0x69fc76`` init, ``fn 0x69cfa0`` node casters, loop
  ``0x69fcf0`` terrain custom-cull casters) and passes it as the 5th argument
  of ``0x6a4400`` (``0x6a02dc``).  See :func:`caster_bounds`.
* **receiver bounds** -- ``DoForwardShaderRenderLoop`` (``0x6c8d60``)
  encapsulates the world AABB of every shadow-receiving opaque object whose
  forward main light is the light (``0x6c9507..0x6c9585``), ``0x6cff90``
  expands it by 0.01 and passes it to ``PrepareShadowMaps`` (stored as
  ``ShadowMapJobHeader+0x18``, read by ``CalculateFocusedLightHull``).
  See :func:`receiver_bounds` / :func:`forward_receiver_bounds`.

Every rule below is transcribed from the disassembly; RVAs are given in the
docstrings and the evidence dumps are under
``outputs/offline_renderer_unity_forward_20261020/disasm/bounds/``.  The full
specification, including what is *not* recovered, is
``outputs/offline_renderer_unity_forward_20261020/shadow_bounds_spec.md``.

Conventions
-----------
* An AABB is ``[cx, cy, cz, ex, ey, ez]`` (Unity ``AABB``: centre, extent).
* A MinMaxAABB is ``(min3, max3)`` with numpy ``float32`` components.
* A plane is ``(nx, ny, nz, d)``; a point ``p`` is inside when
  ``n.p + d >= 0``.
* Matrices are 16 float32 in Unity memory layout ``m[col*4 + row]``
  (as in :mod:`unity_math`).
* SSE ``minss a, b`` returns ``a`` iff ``a < b`` else ``b``; ``maxss a, b``
  returns ``a`` iff ``a > b`` else ``b`` (second operand on equality / NaN).
  The operand order of every union is transcribed because it decides the
  sign of zero and NaN propagation.

Explicit gaps (raise :class:`ShadowBoundsGap`, never approximated):

* ``caster_cull_set`` -- membership of a renderer in the light's shadow
  caster ``CullingOutput`` (kernel ``ProcessShadowCasterNodeVisibilityAnd
  CullWithoutUmbra`` 0x6ac320 with planes from ``PrepareShadowedLight``
  0x67eb30 / ``CalculateShadowCasterCull`` 0x679140, Umbra path, LOD) is not
  transcribed; callers must supply ``shadow_cull_visible``.  The directional
  post-cull predicate (``0x6a9eb0``) *is* transcribed in
  :func:`directional_caster_post_cull`.
* ``local_light_receiver_entry`` -- creation of a receiver-bounds entry for a
  non-directional light (``0x6cda58..0x6cdb76``) needs ``RenderLoop+0x748``,
  whose producer is not recovered.
"""
from __future__ import annotations

import numpy as np

from . import unity_math as um

F = np.float32
ZERO = F(0.0)
ONE = F(1.0)
INF = F(np.inf)                      # ?infinityVec@Vector3f@@2V1@B (0x14b5208)
NEG_INF = F(-np.inf)                 # infinityVec ^ __xmm@8000...
RECEIVER_EXPAND = F(0.00999999977648258)   # __real@3c23d70a, 0x6cffc5
VEC3_EPSILON = F(9.999999747378752e-06)    # ?epsilon@Vector3f@@2MB (0x1257060)
DEG_180 = F(180.0)                   # __real@43340000, 0x681ba2
STABLE_FIT_K = F(0.4000000059604645)  # __real@3ecccccd, 0x681b8e / 0x681bc3
CLOSE_FIT_RADIUS = F(999999986991104.0)    # __real@58635fa9, 0x681a48

LIGHT_DIRECTIONAL = 'directional'
SHADOW_CASTING_MODES = ('off', 'on', 'two_sided', 'shadows_only')   # Renderer flags bits 6..7 (0x73fd73)
RENDERING_PATH_VERTEX_LIT = 0          # CullResults+0x5b8 == Camera::CalculateRenderingPath (0x661130 -> 0x661216)
SHADOW_PROJECTION_CLOSE_FIT = 0        # QualitySettings+0x34 passed to SetupShadowCullData (0x661360)
SHADOW_PROJECTION_STABLE_FIT = 1


class ShadowBoundsGap(RuntimeError):
    """A required engine branch that has not been recovered from the binary."""


# ----------------------------------------------------------------- scalars
def _f(x):
    return F(x)


def _minss(a, b):
    return a if a < b else b


def _maxss(a, b):
    return a if a > b else b


def _sqrtf(x):
    # CRT sqrtf (0x116c720): non-negative finite input -> sqrtss (0x116c7b4),
    # IEEE correctly rounded; negative -> NaN via _handle_errorf.
    return F(np.sqrt(F(x)))


def _aabb(values):
    vals = [F(v) for v in values]
    if len(vals) != 6:
        raise ValueError('AABB requires [cx, cy, cz, ex, ey, ez]')
    return vals


def _plane(values):
    vals = [F(v) for v in values]
    if len(vals) != 4:
        raise ValueError('Plane requires (nx, ny, nz, d)')
    return vals


# ------------------------------------------------------------- MinMaxAABB
def empty_minmax():
    """MinMaxAABB default state: min = infinityVec, max = -infinityVec
    (0x69fbf4..0x69fc76, 0x69cfb9..0x69d02b, 0x6c8efd..0x6c8f58)."""
    return [INF, INF, INF], [NEG_INF, NEG_INF, NEG_INF]


def is_valid(mm):
    """``MinMaxAABB::IsValid`` (0xe4580): false iff min == (inf, inf, inf)
    exactly or max == (-inf, -inf, -inf) exactly (ucomiss; NaN compares
    unequal, so a NaN component makes the box 'valid')."""
    mn, mx = mm
    if mn[0] == INF and mn[1] == INF and mn[2] == INF:
        return False
    if mx[0] == NEG_INF and mx[1] == NEG_INF and mx[2] == NEG_INF:
        return False
    return True


def encapsulate(mm, aabb):
    """``MinMaxAABB::Encapsulate(const AABB&)`` (0x292d60), in place.

    min_i = minss(c_i - e_i, min_i); max_i = maxss(e_i + c_i, max_i).
    The same expressions and operand order are inlined in fn 0x69cfa0
    (0x69d10d..0x69d18c visible casters, 0x69d194..0x69d213 fallback), in
    the terrain loop of 0x69fab0 (comiss/cmovbe form, 0x69fd38..0x69feb6:
    'new if acc > new else acc' == minss(new, acc)) and in 0x6cdab2."""
    c0, c1, c2, e0, e1, e2 = _aabb(aabb)
    mn, mx = mm
    mn[0] = _minss(c0 - e0, mn[0])
    mn[2] = _minss(c2 - e2, mn[2])
    mn[1] = _minss(c1 - e1, mn[1])
    mx[2] = _maxss(e2 + c2, mx[2])
    mx[0] = _maxss(e0 + c0, mx[0])
    mx[1] = _maxss(e1 + c1, mx[1])
    return mm


def minmax_from_aabb(aabb):
    """Direct conversion used when 0x6cda00 creates a new per-light entry
    (0x6cdbc0..0x6cdc47): min = c - e, max = c + e (no union with infinity)."""
    c0, c1, c2, e0, e1, e2 = _aabb(aabb)
    return [c0 - e0, c1 - e1, c2 - e2], [c0 + e0, c1 + e1, c2 + e2]


def expand_receiver_bounds(mm):
    """0x6cffc5..0x6d0032 (inside fn 0x6cff90, before PrepareShadowMaps):
    min -= 0.01f, max += 0.01f, in place (also applied to an invalid box)."""
    mn, mx = mm
    for i in range(3):
        mn[i] = mn[i] - RECEIVER_EXPAND
        mx[i] = mx[i] + RECEIVER_EXPAND
    return mm


def _result(mm):
    if not is_valid(mm):
        return None
    return tuple(mm[0]), tuple(mm[1])


# ------------------------------------------------------- intersection tests
def _plane_keeps(aabb, plane):
    c0, c1, c2, e0, e1, e2 = aabb
    n0, n1, n2, d = plane
    dist = ((n0 * c0 + n1 * c1) + n2 * c2) + d
    rad = (abs(n1) * e1 + abs(n0) * e0) + abs(n2) * e2
    s = rad + dist
    return not (ZERO > s)


def intersect_aabb_plane_bounds(aabb, planes, count):
    """``IntersectAABBPlaneBounds(aabb, planes, n)`` (0x6eaa50): false as soon
    as 0 > (|n.y|e.y + |n.x|e.x) + |n.z|e.z + (((n.x c.x + n.y c.y) + n.z c.z) + d)."""
    box = _aabb(aabb)
    for i in range(int(count)):
        if not _plane_keeps(box, _plane(planes[i])):
            return False
    return True


def intersect_aabb_frustum_full(aabb, planes):
    """``IntersectAABBFrustumFull(aabb, planes[6])`` (0x6ea930): identical
    per-plane expression to :func:`intersect_aabb_plane_bounds` over 6 planes."""
    if len(planes) != 6:
        raise ValueError('IntersectAABBFrustumFull takes 6 planes')
    return intersect_aabb_plane_bounds(aabb, planes, 6)


def intersect_aabb_sphere(aabb, center, radius):
    """``IntersectAABBSphere`` (0x6eaca0): v_i = max(|s_i - c_i|, e_i) - e_i;
    lane 3 is masked by the xyz mask initialised at 0x80510 from .rdata
    0x1224ca0 (ffffffff x3, 0); horizontal sum (v2 + v3) + (v0 + v1) with
    v3 = 0; true iff sum < r*r (cmpltps)."""
    c0, c1, c2, e0, e1, e2 = _aabb(aabb)
    s = [F(v) for v in center]
    r = F(radius)
    v = []
    for si, ci, ei in ((s[0], c0, e0), (s[1], c1, e1), (s[2], c2, e2)):
        t = _maxss(abs(si - ci), ei) - ei      # maxps(|d|, e) - e
        v.append(t * t)
    total = (v[2] + ZERO) + (v[0] + v[1])
    return bool(total < r * r)


# ---------------------------------------------------------- camera planes
def extract_projection_planes(m):
    """``ExtractProjectionPlanes`` -> ``ExtractProjectionPlanesHelper<0>::Apply``
    (0x68dc40 -> 0x68b480).  row_r = (m[r], m[4+r], m[8+r], m[12+r]);
    planes: 0 = row0+row3, 1 = row3-row0, 2 = row1+row3, 3 = row3-row1,
    4 = row2+row3, 5 = row3-row2; each scaled by inv = 1/sqrtf((x*x + y*y) + z*z)."""
    m = um.as_f32_list(m)
    r0 = (m[0], m[4], m[8], m[12])
    r1 = (m[1], m[5], m[9], m[13])
    r2 = (m[2], m[6], m[10], m[14])
    r3 = (m[3], m[7], m[11], m[15])
    raw = [
        [r0[k] + r3[k] for k in range(4)],
        [r3[k] - r0[k] for k in range(4)],
        [r1[k] + r3[k] for k in range(4)],
        [r3[k] - r1[k] for k in range(4)],
        [r2[k] + r3[k] for k in range(4)],
        [r3[k] - r2[k] for k in range(4)],
    ]
    out = []
    for p in raw:
        length = _sqrtf((p[0] * p[0] + p[1] * p[1]) + p[2] * p[2])
        inv = ONE / length
        out.append([p[0] * inv, p[1] * inv, p[2] * inv, p[3] * inv])
    return out


def calculate_frustum_planes(world_to_clip, world_to_camera, max_distance, camera_near):
    """``Camera::CalculateFrustumPlanes(planes, worldToClip, maxDist, &baseFar,
    implicitNearFar=true)`` (0x65e950) as called from
    ``GetScriptableCullingParameters`` (0x7b9535).

    Planes 0..3 from :func:`extract_projection_planes`; with the flag set
    (0x7b951b) planes 4/5 are rebuilt from cameraToWorld =
    InvertMatrix4x4_Full(worldToCamera): a = column 2 normalised by division
    (only when its length > Vector3f::epsilon, else zero), p = column 3;
    near = (-a, -(((-a.y) p.y + (-a.x) p.x) + (-a.z) p.z) - near);
    baseFar = -((a.y p.y + a.x p.x) + a.z p.z); far = (a, maxDist + baseFar).
    Returns (planes, baseFar)."""
    planes = extract_projection_planes(world_to_clip)
    ok, c2w = um.invert_full(um.as_f32_list(world_to_camera))
    if not ok:
        raise ValueError('worldToCamera is singular (InvertMatrix4x4_Full failed)')
    ax, ay, az = c2w[8], c2w[9], c2w[10]
    px, py, pz = c2w[12], c2w[13], c2w[14]
    length = _sqrtf((ay * ay + ax * ax) + az * az)
    if length > VEC3_EPSILON:
        ax, ay, az = ax / length, ay / length, az / length
    else:
        ax = ay = az = ZERO
    nx, ny, nz = -ax, -ay, -az
    near_d = -((ny * py + nx * px) + nz * pz)
    near_d = near_d - F(camera_near)
    planes[4] = [nx, ny, nz, near_d]
    base_far = -((ay * py + ax * px) + az * pz)
    planes[5] = [ax, ay, az, F(max_distance) + base_far]
    return planes, base_far


def setup_shadow_cull_data(*, cull_world_to_clip, world_to_camera, camera_near, camera_far,
                           shadow_distance_for_rendering, shadow_projection,
                           cull_position, is_orthographic, field_of_view):
    """The ShadowCullData fields used by the bounds (built-in RP, no SRP asset).

    * planes_b (ShadowCullData+0x2ec, from ScriptableCullingParameters+0x420):
      :func:`calculate_frustum_planes` with maxDist from ``comiss sd, far;
      ja keep far`` (0x7b94e1..0x7b94ee): maxDist = far if sd > far else sd,
      where far = Camera::GetFar() (+0x440, stored at +0x1a0 by 0x7b8cae).
    * planes_a (ShadowCullData+0x28c, copy +0x480 made at 0x7b9555) with plane
      5 d replaced by baseFar + far (0x681aff..0x681b0f).
    * base_far (ShadowCullData+0x288 = ScriptableCullingParameters+0x4e0).
    * shadow_distance = minss(camera far [cam+0x440], sd) (0x661390..0x66139f).
    * Stable fit (projection != CloseFit, 0x681b3c..0x681cbc): k = 0.4 for an
      orthographic camera, else clamp01(1 - fov/180) * 0.4; centre =
      position + col0*0 + col1*0 + col2*(-(k*dist)) of cameraToWorld
      (ScriptableCullingParameters+0x1dc, 0x7b8f4f); radius = (1-k)*dist.
      Close fit: use_sphere false, centre = zero, radius = 1e15f (0x681a48).

    ``cull_world_to_clip`` is ScriptableCullingParameters+0x29c (selected at
    0x7b9320 between the camera worldToClip and implicit projection x
    worldToCamera; the selection test itself is not transcribed, so the
    caller passes the matrix that the engine selected).  ``cull_position`` is
    ScriptableCullingParameters+0x194 (copied to ShadowCullData+0x244/+0x34c).
    """
    far = F(camera_far)
    sd = F(shadow_distance_for_rendering)
    max_dist = far if sd > far else sd
    planes_b, base_far = calculate_frustum_planes(cull_world_to_clip, world_to_camera, max_dist, camera_near)
    planes_a = [list(p) for p in planes_b]
    planes_a[5][3] = base_far + far
    dist = _minss(far, sd)
    pos = [F(v) for v in cull_position]
    if int(shadow_projection) != SHADOW_PROJECTION_CLOSE_FIT:
        if is_orthographic:
            k = STABLE_FIT_K
        else:
            t = ONE - F(field_of_view) / DEG_180
            k = ZERO if ZERO > t else _minss(ONE, t)
            k = k * STABLE_FIT_K
        ok, c2w = um.invert_full(um.as_f32_list(world_to_camera))
        if not ok:
            raise ValueError('worldToCamera is singular (InvertMatrix4x4_Full failed)')
        s = -(k * dist)
        x = ((c2w[4] * ZERO + c2w[0] * ZERO) + s * c2w[8]) + pos[0]
        y = ((c2w[1] * ZERO + c2w[5] * ZERO) + c2w[9] * s) + pos[1]
        z = ((c2w[2] * ZERO + c2w[6] * ZERO) + c2w[10] * s) + pos[2]
        radius = (ONE - k) * dist
        use_sphere, center = True, [x, y, z]
    else:
        radius = CLOSE_FIT_RADIUS
        use_sphere, center = False, [ZERO, ZERO, ZERO]
    return {
        'planes_a': planes_a,
        'planes_b': planes_b,
        'base_far': base_far,
        'shadow_distance': dist,
        'use_sphere': use_sphere,
        'sphere_center': center,
        'sphere_radius': radius,
        'sphere_radius_sq': radius * radius,
        'eye_position': pos,
    }


def is_object_within_shadow_range(shadow_cull_data, aabb):
    """``IsObjectWithinShadowRange(ShadowJobData, AABB)`` (0x6abb60).

    Stable fit: d = c - centre; true if r^2 > ((d.y^2 + d.x^2) + d.z^2), else
    IntersectAABBSphere.  Close fit: IntersectAABBPlaneBounds(aabb,
    ShadowCullData+0x33c (= planes_b[5]), 1)."""
    box = _aabb(aabb)
    if shadow_cull_data['use_sphere']:
        s = [F(v) for v in shadow_cull_data['sphere_center']]
        dz = box[2] - s[2]
        dx = box[0] - s[0]
        dy = box[1] - s[1]
        dist_sq = (dy * dy + dx * dx) + dz * dz
        if F(shadow_cull_data['sphere_radius_sq']) > dist_sq:
            return True
        return intersect_aabb_sphere(box, s, shadow_cull_data['sphere_radius'])
    return intersect_aabb_plane_bounds(box, [shadow_cull_data['planes_b'][5]], 1)


# ----------------------------------------------------------- caster bounds
def directional_caster_post_cull(caster, ctx):
    """``CullDirectionalShadows`` per-node predicate (0x6a9eb0, applied by
    0x6a9fc0 from 0x67bcc2 after the culling kernel).

    caster: ``shadow_casting_mode``, ``layer`` (0..31), ``world_bounds``,
    ``lightmap_index_is_none`` (renderer+0xec u16 == 0xffff).
    ctx: ``layer_mask`` (DirectionalCullContext+0x10 = light.cullingMask &
    caster CullingParameters+0xc4, 0x67b25f..0x67b276), ``skip_baked``
    (ctx+0x14 = fn 0x682580(light, SceneCullingParameters+0x238); fn not
    transcribed, so the caller supplies it), ``layer_cull_distances`` (32
    floats, ShadowCullData+0x388), ``layer_cull_spherical``
    (ShadowCullData+0x408), and the :func:`setup_shadow_cull_data` dict as
    ``shadow_cull_data``."""
    mode = caster['shadow_casting_mode']
    if mode not in SHADOW_CASTING_MODES:
        raise ValueError('unknown shadow_casting_mode %r' % (mode,))
    if mode == 'off':                       # SceneNode+0x1c < 0x40000000
        return False
    if ctx['skip_baked'] and not caster['lightmap_index_is_none']:
        return False
    layer = int(caster['layer'])
    if not (int(ctx['layer_mask']) >> layer) & 1:
        return False
    dist = F(ctx['layer_cull_distances'][layer])
    if dist == ZERO:
        return True
    box = _aabb(caster['world_bounds'])
    scd = ctx['shadow_cull_data']
    if ctx['layer_cull_spherical']:
        eye = [F(v) for v in scd['eye_position']]
        dy = box[1] - eye[1]
        dx = box[0] - eye[0]
        dz = box[2] - eye[2]
        d2 = (dy * dy + dx * dx) + dz * dz
        return not (d2 > dist * dist)
    far = scd['planes_b'][5]
    plane = [F(far[0]), F(far[1]), F(far[2]), dist + F(scd['base_far'])]
    return intersect_aabb_plane_bounds(box, [plane], 1)


def caster_bounds_minmax(casters, *, light_type, camera_frustum_planes=None,
                         terrain_custom_cull_bounds=()):
    """Caster MinMaxAABB exactly as passed to 0x6a4400 (may be invalid).

    ``casters``: the light's extracted shadow RenderNodeQueue (header+0x5d8)
    in queue order; each dict needs ``world_bounds`` (RenderNode+0x80),
    ``shadow_cull_visible`` (membership in the light's shadow CullingOutput;
    gap ``caster_cull_set``), ``has_shadow_caster_pass`` (fn 0x69d310: some
    material's active subshader returns a pass from
    Shader::GetShadowCasterPassToUse) and ``shadow_casting_mode``.

    fn 0x69cfa0: for a caster with a ShadowCaster pass, a directional light
    encapsulates it into the result when IntersectAABBFrustumFull(aabb,
    ShadowCullData+0x28c = planes_a) holds, otherwise into a local fallback;
    every other light type uses the fallback for all casters.  If the result
    is not valid afterwards: result = (minss(fallback.min, result.min),
    maxss(fallback.max, result.max)) (0x69d294..0x69d2f7).  Then loop
    0x69fcf0 encapsulates every terrain custom-cull caster AABB
    (CullingOutput list 5 = TerrainManager::CollectCustomCullRenderers,
    bounds from CullResults+0x590) unconditionally."""
    directional = light_type == LIGHT_DIRECTIONAL
    if directional and camera_frustum_planes is None:
        raise ValueError('directional caster bounds need ShadowCullData+0x28c planes')
    result = empty_minmax()
    fallback = empty_minmax()
    for caster in casters:
        if 'shadow_cull_visible' not in caster:
            raise ShadowBoundsGap('caster_cull_set: shadow caster culling (0x6ac320/0x67eb30) is not '
                                  'transcribed; supply shadow_cull_visible per caster')
        if caster['shadow_casting_mode'] not in SHADOW_CASTING_MODES:
            raise ValueError('unknown shadow_casting_mode %r' % (caster['shadow_casting_mode'],))
        if not caster['shadow_cull_visible']:
            continue
        if directional and caster['shadow_casting_mode'] == 'off':
            raise ValueError('a directional caster with castShadows Off cannot pass 0x6a9eb4')
        if not caster['has_shadow_caster_pass']:
            continue
        box = _aabb(caster['world_bounds'])
        if directional and intersect_aabb_frustum_full(box, camera_frustum_planes):
            encapsulate(result, box)
        else:
            encapsulate(fallback, box)
    if not is_valid(result):
        rmn, rmx = result
        fmn, fmx = fallback
        result = ([_minss(fmn[i], rmn[i]) for i in range(3)],
                  [_maxss(fmx[i], rmx[i]) for i in range(3)])
    for box in terrain_custom_cull_bounds:
        encapsulate(result, _aabb(box))
    return result


def caster_bounds(casters, *, light_type, camera_frustum_planes=None, terrain_custom_cull_bounds=()):
    """(min3, max3) of :func:`caster_bounds_minmax`, or None when
    ``MinMaxAABB::IsValid`` fails (0x6a4441: 0x6a4400 then writes an
    identity view and SetOrtho(-1, 1, -1, 1, 0.1, 10) and returns false)."""
    return _result(caster_bounds_minmax(casters, light_type=light_type,
                                        camera_frustum_planes=camera_frustum_planes,
                                        terrain_custom_cull_bounds=terrain_custom_cull_bounds))


# --------------------------------------------------------- receiver bounds
def should_use_shadows(*, build_has_shadows, caps_0x39b, caps_0x399, caps_0x390,
                       quality_shadows, shadow_distance_for_rendering, check_distance=True):
    """``ShouldUseShadows(bool)`` (0x6a57f0); Camera::CalculateCanDoShadows
    (0x65e1d0) calls it with true.  BuildSettings+0x101, GraphicsCaps
    +0x39b/+0x399 bytes and +0x390 int, QualitySettings current +0x2c
    (shadows) != 0, and shadowDistance > 0 when checking distance."""
    if not (build_has_shadows and caps_0x39b and caps_0x399 and int(caps_0x390) != 0):
        return False
    if int(quality_shadows) == 0:
        return False
    if check_distance and not (F(shadow_distance_for_rendering) > ZERO):
        return False
    return True


def forward_receiver_gate(*, shadow_pass_argument, camera_can_do_shadows, rendering_path):
    """[rbp-0x60] of DoForwardShaderRenderLoop: 4th argument (1 for the opaque
    call at 0x6b9c88, 0 for the transparent call at 0x6b9dc7) && camera
    CalculateCanDoShadows (0x6c8dc5) && CullResults+0x5b8 (rendering path)
    != 0 (0x6c915e..0x6c918d)."""
    return bool(shadow_pass_argument and camera_can_do_shadows
                and int(rendering_path) != RENDERING_PATH_VERTEX_LIT)


def forward_receiver_bounds(objects, lights, shadow_cull_data, *, receiver_gate,
                            additional_light_gate=False):
    """Replays the receiver part of DoForwardShaderRenderLoop (0x6c9200 loop).

    ``objects``: opaque RenderObjectData entries in list order (ExtractObjects
    0x6b65c0: one entry per material with render queue <= 2500, nodes with
    castShadows ShadowsOnly skipped).  Each dict: ``world_bounds``
    (RenderNode+0x80), ``receive_shadows`` (RenderNode+0xe8 bit 0x100, set by
    Renderer.receiveShadows 0x97107b), ``subshader_resolved`` (false skips the
    object at 0x6c92db), ``forward_main_light`` (ForwardLightsBlock+0x78 from
    FindForwardLightsForObject, id or None), ``forward_additional_lights``
    (ids, block+0x98), ``additional_light_shadow_pass`` (SubShader+0x93), and
    for terrain batched patches (RenderNode+0xf4 == 1, set by
    QuadTreeBatchedNodeRenderer::PrepareNode 0x3ccd05) ``terrain_receiver_bounds``
    (the AABB at *(RenderNode+0x160)).
    ``lights``: id -> {'type', 'shadows' (SharedLightData+0xa4 != None),
    'active_flag_0x20' (ActiveLight+0x20)}.

    Returns (first_main_light, {light_id: MinMaxAABB}) before the 0.01
    expansion; the first main light's box is ForwardShaderRenderLoop+0x188,
    the others the ForwardShadowMap list at +0x1b8 (0x6cda00)."""
    first_main = None
    main_box = empty_minmax()
    others = {}

    def other(light_id, box):
        if light_id in others:                       # 0x6cda38 found -> 0x6cdab2
            encapsulate(others[light_id], box)
            return
        if lights[light_id]['type'] != LIGHT_DIRECTIONAL:
            raise ShadowBoundsGap('local_light_receiver_entry: creation for non-directional lights '
                                  'tests view depth with RenderLoop+0x748 (0x6cda58), not recovered')
        others[light_id] = minmax_from_aabb(box)      # 0x6cdb7c..0x6cdc64

    for obj in objects:
        if not obj['subshader_resolved']:
            continue
        if not receiver_gate:
            continue
        if not obj['receive_shadows']:
            continue
        node_box = _aabb(obj['world_bounds'])
        if not is_object_within_shadow_range(shadow_cull_data, node_box):
            continue
        box = node_box
        if obj.get('terrain_receiver_bounds') is not None:
            box = _aabb(obj['terrain_receiver_bounds'])
        main = obj['forward_main_light']
        if main is not None and lights[main]['shadows']:
            if first_main is None:
                first_main = main
            if main == first_main:
                encapsulate(main_box, box)
            else:
                other(main, box)
        if additional_light_gate and obj['additional_light_shadow_pass']:
            for light_id in obj['forward_additional_lights']:
                light = lights[light_id]
                if not light['shadows'] or not light['active_flag_0x20']:
                    continue
                if light_id == first_main:
                    continue
                other(light_id, box)
    out = dict(others)
    if first_main is not None:
        out[first_main] = main_box
    return first_main, out


def receiver_bounds(objects, light_id, lights, shadow_cull_data, *, receiver_gate,
                    additional_light_gate=False, expand=True):
    """Receiver MinMaxAABB of ``light_id`` as passed to PrepareShadowMaps.

    With ``expand`` the 0x6cff90 widening by 0.01 is applied (it runs only
    when FindShadowMapInCache misses; a cached shadow map skips
    PrepareShadowMaps).  Returns (min3, max3), or None when the box is not
    valid -- PrepareShadowMaps then returns null (0x69f514..0x69f52b)."""
    _, boxes = forward_receiver_bounds(objects, lights, shadow_cull_data,
                                       receiver_gate=receiver_gate,
                                       additional_light_gate=additional_light_gate)
    if light_id not in boxes:
        return None
    mm = ([F(v) for v in boxes[light_id][0]], [F(v) for v in boxes[light_id][1]])
    if expand:
        expand_receiver_bounds(mm)
    return _result(mm)


__all__ = [
    'ShadowBoundsGap', 'empty_minmax', 'is_valid', 'encapsulate', 'minmax_from_aabb',
    'expand_receiver_bounds', 'intersect_aabb_plane_bounds', 'intersect_aabb_frustum_full',
    'intersect_aabb_sphere', 'extract_projection_planes', 'calculate_frustum_planes',
    'setup_shadow_cull_data', 'is_object_within_shadow_range', 'directional_caster_post_cull',
    'caster_bounds_minmax', 'caster_bounds', 'should_use_shadows', 'forward_receiver_gate',
    'forward_receiver_bounds', 'receiver_bounds',
]

