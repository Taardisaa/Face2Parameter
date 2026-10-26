"""Shadow-caster culling of a directional light (Unity 2018.4.11f1 built-in, non-Umbra).

Transcribed from the installed ``E:/HoneySelect2_ArcticFox/UnityPlayer.dll`` (SHA-256
``5034cdc2...``, PDB GUID ``BFA91A8C44734BB4AABC5A975B304CFE`` age 1); disassembly dumps
under ``E:/HS2_offline_renderer_data/external_closure_20261025/disasm/caster_cull/``.

Call chain (per shadowed directional light, ``CullDirectionalShadowCasters`` 0x67af80):

1. ``PrepareShadowedLight`` 0x67eb30: copies the SceneCullingParameters into the light's
   CullingParameters (ShadowedLight+0xa0), calls ``CalculateShadowCasterCull`` 0x679140
   on the camera frustum planes ShadowCullData+0x2ec with ShadowCullData+0x1c0
   (clip->world), +0x244 (eye), 1.0, +0x364/+0x380, the light type (ActiveLight+0xa0) and
   SharedLightData+0x160 (directional: L2W column 2) and then
   ``CalculateCustomCullingParameters`` 0x6403c0: planes, plane count (+0xc0),
   cullingMask = ShadowCullData+0x240 (camera cullingMask), sceneMask (+0xc8),
   layer-cull mode +0x150 = 0.
2. Job ``CullDirectionalShadowCastersJob`` 0x67b410 -> 0x67c290 ->
   ``ProcessShadowCasterNodeVisibilityAndCullWithoutUmbra`` 0x6ac320 -> kernel
   <mode 0, 4/8/12 planes> (0x6a7bc0 / 0x6a7dd0 / 0x6a80a0; planes via
   ``PrepareOptimizedPlanes`` 0x6ec970, padded with the last plane).
3. Post job 0x67bba0: layer distances combined with the light's (0x683400 when
   Light+0xc0 is set), then ``CullDirectionalShadows`` 0x6a9fc0 -> per-node predicate
   0x6a9eb0 (``unity_shadow_bounds.directional_caster_post_cull``).

ShadowCullData inputs come from ``GetScriptableCullingParameters`` 0x7b8b60 (cull matrix
selection 0x7b9201..0x7b93b4, implicit flag +0x58c = Camera+0x4ec != 0 at 0x7b9411) and
``SetupShadowCullData`` 0x6815f0 (clip->world +0x1c0 at 0x6818ff..0x681a3a).
"""
from __future__ import annotations

import struct

import numpy as np

from . import unity_math as um
from .unity_shadows import get_frustum_points, perspective_multiply_point3

F = np.float32
ZERO = F(0.0)
ONE = F(1.0)
NEG_ONE = F(-1.0)
EPS_V3 = np.frombuffer(struct.pack('<I', 0x3727C5AC), np.float32)[0]       # Vector3f::epsilon .rdata 0x1257060
PARALLEL_EPS = np.frombuffer(struct.pack('<I', 0x3C23D70A), np.float32)[0]  # __real@3c23d70a (0.01) 0x679199
NEG_PARALLEL_EPS = np.frombuffer(struct.pack('<I', 0xBC23D70A), np.float32)[0]  # __real@bc23d70a 0x6791be
CROSS_EPS = np.frombuffer(struct.pack('<I', 0x3A83126F), np.float32)[0]     # __real@3a83126f (0.001) 0x678015
CULL_MATRIX_EPS = PARALLEL_EPS                                                # 0x7b926b
# NDC offsets of the six frustum planes (left, right, bottom, top, near, far), .rdata 0x14ae790
PLANE_NDC = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
SIDE_CYCLE = (0, 2, 1, 3)        # .rdata 0x14ae7d8 (left, bottom, right, top)
NEAR_FAR = (4, 5)                # .rdata 0x14ae7e8
PLANES_EXCLUDED = (False,) * 6   # PrepareShadowedLight passes .rdata 0x1244ec0 (all zero bytes)
MAX_PLANES = 10                  # ShadowCasterCull+0xa0 count limit (cmp 0xa)
LIGHT_SPOT, LIGHT_DIRECTIONAL, LIGHT_POINT = 0, 1, 2
SCENE_MASK_DEFAULT = 0x8000000000000000   # UnityScene ctor 0x820e56 (+0xc0); camera 0x7b8d84
NODE_PENDING_REMOVAL = 0x20000000          # SceneNode+0x1c bit 29: only RemoveRenderer 0x69101a sets it
NODE_LOD_INDEX = 0x0FFFFFFF


class ShadowCullGap(RuntimeError):
    """A branch of the shadow-caster cull that is not reproducible from the inputs."""


def _f(x):
    return F(x)


# --------------------------------------------------------------- camera helpers
def projection_near(implicit, near_clip, projection):
    """``Camera::GetProjectionNear`` 0x6639a0."""
    if implicit:
        return F(near_clip)
    m = [F(v) for v in projection]
    for r in range(3):
        for c in range(3):
            if r != c and not (m[r + 4 * c] == ZERO):
                return F(near_clip)
    a = m[7] + m[6]
    b = m[3] + m[2]
    c = m[11] + m[10]
    d = (a * a + b * b) + c * c
    return -((m[15] + m[14]) / F(np.sqrt(d)))


def projection_far(implicit, far_clip, projection):
    """``Camera::GetProjectionFar`` 0x663750 (as ``frame_orchestrate_paired.uf_projection_far``)."""
    if implicit:
        return F(far_clip)
    m = [F(v) for v in projection]
    for r in range(3):
        for c in range(3):
            if r != c and not (m[r + 4 * c] == ZERO):
                return F(far_clip)
    a = m[7] - m[6]
    b = m[3] - m[2]
    c = m[11] - m[10]
    d = (a * a + b * b) + c * c
    return (m[15] - m[14]) / F(np.sqrt(d))


def implicit_projection(orthographic, near, far, fov, aspect, orthographic_size, camera_aspect):
    """``Camera::GetImplicitProjectionMatrix(near, far, fov, aspect)`` 0x663420:
    perspective -> SetPerspective(fov, aspect, near, far); orthographic ->
    SetOrtho(-(size*Camera+0x4c8), size*Camera+0x4c8, -size, size, near, far)."""
    if not orthographic:
        return um.set_perspective(fov, aspect, near, far)
    size = F(orthographic_size)
    w = size * F(camera_aspect)
    return um.set_ortho(-w, w, -size, size, near, far)


def scriptable_cull_matrices(cam):
    """ScriptableCullingParameters +0x25c (clip->world) and +0x29c (cull world->clip)
    of ``GetScriptableCullingParameters`` (non-stereo):

    * +0x21c = GetWorldToClipMatrix (0x664840: projection x worldToCamera, 0x7b8f22),
      +0x25c = InvertMatrix4x4_Full(+0x21c) (0x7b8f5e);
    * 0x7b9201..0x7b9320: ratio = projFar / projNear; for the four frustum edges
      (GetFrustumPoints of +0x25c) a = |((n - eye)*ratio + eye) - f|^2,
      b = 0.01*|f - n|^2; ``comiss b, a; jb`` (b < a or unordered) -> implicit;
    * all four pass -> +0x29c = +0x21c; else +0x29c = GetImplicitProjectionMatrix(near,
      far, fov, aspect) x worldToCamera and +0x25c = its inverse (0x7b935e..0x7b93b4).

    ``cam``: dict with projection, world_to_camera, position (Transform position, CP+0x194),
    implicit (Camera+0x4ec == 1), near, far, fov, aspect, orthographic, orthographic_size."""
    w2c = [F(v) for v in cam['world_to_camera']]
    proj = [F(v) for v in cam['projection']]
    w2clip = um.multiply(proj, w2c)
    ok, c2w = um.invert_full(w2clip)
    pn = projection_near(cam['implicit'], cam['near'], proj)
    pf = projection_far(cam['implicit'], cam['far'], proj)
    ratio = pf / pn
    e = [F(v) for v in cam['position']]
    pts = get_frustum_points(c2w)
    implicit_branch = False
    for k in range(4):
        n, f = pts[k], pts[k + 4]
        vx = ((n[0] - e[0]) * ratio + e[0]) - f[0]
        vy = ((n[1] - e[1]) * ratio + e[1]) - f[1]
        vz = ((n[2] - e[2]) * ratio + e[2]) - f[2]
        dx, dy, dz = f[0] - n[0], f[1] - n[1], f[2] - n[2]
        a = (vy * vy + vx * vx) + vz * vz
        b = ((dy * dy + dx * dx) + dz * dz) * CULL_MATRIX_EPS
        if not (b >= a):
            implicit_branch = True
            break
    if implicit_branch:
        P = implicit_projection(cam['orthographic'], cam['near'], cam['far'], cam['fov'], cam['aspect'],
                                cam.get('orthographic_size', 0.0), cam.get('camera_aspect', cam['aspect']))
        cull = um.multiply(P, w2c)
        ok, c2w = um.invert_full(cull)
    else:
        cull = w2clip
    return {'clip_to_world_0x25c': c2w, 'cull_world_to_clip_0x29c': cull, 'world_to_clip_0x21c': w2clip,
            'projection_near_0x1c8': pn, 'projection_far_0x1cc': pf, 'implicit_matrix_branch': implicit_branch}


def shadow_cull_clip_to_world(cam, scp, shadow_distance):
    """ShadowCullData+0x1c0 and +0x380 (``SetupShadowCullData`` 0x6818ff..0x681a3a).

    SCP+0x58c = (Camera+0x4ec != 0) (0x7b9411): perspective -> SetPerspective(fov
    (+0x14 = GetFov), aspect (+0x1d8), projNear (+0x1c8), shadowDistance), orthographic
    -> SetOrtho(-(size*aspect), size*aspect, -size, size, projNear, shadowDistance);
    times worldToCamera (+0x39c), InvertMatrix4x4_Full; +0x380 = shadowDistance.
    Otherwise +0x1c0 = SCP+0x25c and +0x380 = projFar (+0x1cc)."""
    sd = F(shadow_distance)
    if cam['implicit']:
        pn = scp['projection_near_0x1c8']
        if cam['orthographic']:
            size = F(cam['orthographic_size'])
            w = size * F(cam['aspect'])
            P = um.set_ortho(-w, w, -size, size, pn, sd)
        else:
            P = um.set_perspective(cam['fov'], cam['aspect'], pn, sd)
        ok, c2w = um.invert_full(um.multiply(P, [F(v) for v in cam['world_to_camera']]))
        return c2w, sd
    return scp['clip_to_world_0x25c'], scp['projection_far_0x1cc']


# ------------------------------------------------------- CalculateShadowCasterCull
def edge_plane(orthographic, planes, clip_to_world, eye, t_near_arg, t_far_arg, i, j, k, light_type, light_vec):
    """Fn 0x677cd0: plane through the frustum edge (planes i, j) containing the light
    direction (directional) or the light position; None when degenerate.

    Edge end points in NDC: A -/+ PLANE_NDC[k] with A = PLANE_NDC[i] + PLANE_NDC[j]; t =
    arg5 for z < 0 else arg6; world points = PerspectiveMultiplyPoint3(clip->world),
    then (perspective) eye + (w - eye) * t."""
    N = [F(planes[i][c]) + F(planes[j][c]) for c in range(3)]
    A = [F(PLANE_NDC[i][c]) + F(PLANE_NDC[j][c]) for c in range(3)]
    K = [F(v) for v in PLANE_NDC[k]]
    p0 = [A[0] - K[0], A[1] - K[1], A[2] - K[2]]
    p1 = [K[0] + A[0], K[1] + A[1], K[2] + A[2]]
    t0 = F(t_near_arg) if ZERO > p0[2] else F(t_far_arg)
    t1 = F(t_near_arg) if ZERO > p1[2] else F(t_far_arg)
    # orthographic (0x677e09..0x677e41): (z + 1) * t - 1 is stored to [rsp+0x28] / [rsp+0x38],
    # the z slots of the two PerspectiveMultiplyPoint3 outputs, which the calls at 0x677e78 /
    # 0x677e8a then overwrite (0x2a1895..0x2a18a0, zeros on failure 0x2a18bf): as compiled the
    # NDC points are not changed
    _, w0 = perspective_multiply_point3(clip_to_world, tuple(p0))
    _, w1 = perspective_multiply_point3(clip_to_world, tuple(p1))
    w0 = [F(v) for v in w0]
    w1 = [F(v) for v in w1]
    if not orthographic:
        e = [F(v) for v in eye]
        w0 = [(w0[c] - e[c]) * t0 + e[c] for c in range(3)]
        w1 = [(w1[c] - e[c]) * t1 + e[c] for c in range(3)]
    dx, dy, dz = w1[0] - w0[0], w1[1] - w0[1], w1[2] - w0[2]
    L = [F(v) for v in light_vec]
    if int(light_type) == LIGHT_DIRECTIONAL:
        ln = F(np.sqrt((dy * dy + dx * dx) + dz * dz))
        if ln > EPS_V3:
            dx, dy, dz = dx / ln, dy / ln, dz / ln
        else:
            dx = dy = dz = ZERO
        nx = dy * L[2] - dz * L[1]
        ny = dz * L[0] - dx * L[2]
        nz = dx * L[1] - dy * L[0]
        ln2 = F(np.sqrt((ny * ny + nx * nx) + nz * nz))
        if CROSS_EPS > ln2:
            return None
        nx, ny, nz = nx / ln2, ny / ln2, nz / ln2
        s = (N[0] * nx + ny * N[1]) + nz * N[2]
        if ZERO > s:
            nx, ny, nz = nx * NEG_ONE, ny * NEG_ONE, nz * NEG_ONE
        d = -((nx * w0[0] + ny * w0[1]) + nz * w0[2])
        return [nx, ny, nz, d]
    # point / spot (0x6780c0): normal = (L - w0) x (w1 - w0) ... as compiled
    ax, ay, az = L[0] - w0[0], L[1] - w0[1], L[2] - w0[2]
    nx = az * dy - ay * dz
    ny = ax * dz - az * dx
    nz = ay * dx - ax * dy
    ln = F(np.sqrt((nx * nx + ny * ny) + nz * nz))
    if EPS_V3 > ln:
        return None
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    d = -((ny * w0[1] + nx * w0[0]) + nz * w0[2])
    s = (N[1] * ny + nx * N[0]) + N[2] * nz
    if ZERO > s:
        nx, ny, nz, d = nx * NEG_ONE, ny * NEG_ONE, nz * NEG_ONE, d * NEG_ONE
    return [nx, ny, nz, d]


def calculate_shadow_caster_cull(orthographic, planes, clip_to_world, eye, t_near_arg, t_far_arg, light_type,
                                 light_vec, excluded=PLANES_EXCLUDED):
    """``CalculateShadowCasterCull`` 0x679140 -> ShadowCasterCull planes (at most 10).

    Per camera plane i (0..5): directional: d = (n.x L.x + n.y L.y) + n.z L.z,
    facing = 0 > d, usable = |d| > 0.01; local: d = n.L + n.w, facing = d > -0.01,
    usable = !(0.01 > d), and 0 > d moves the plane through the light.  Facing,
    non-excluded planes are kept.  Then silhouette edges: (SIDE_CYCLE[k],
    SIDE_CYCLE[k+1]) with third plane 5, and (near/far, side SIDE_CYCLE[k]) with
    third SIDE_CYCLE[k+1], whenever facing differs and both are usable (fn 0x677cd0)."""
    L = [F(v) for v in light_vec]
    out = []
    facing, usable = [False] * 6, [False] * 6
    work = []
    for i in range(6):
        p = [F(v) for v in planes[i]]
        if int(light_type) == LIGHT_DIRECTIONAL:
            d = (p[0] * L[0] + p[1] * L[1]) + p[2] * L[2]
            facing[i] = bool(ZERO > d)
            usable[i] = bool(abs(d) > PARALLEL_EPS)
        else:
            d = ((L[0] * p[0] + L[1] * p[1]) + L[2] * p[2]) + p[3]
            facing[i], usable[i] = False, True
            if d > NEG_PARALLEL_EPS:
                facing[i] = True
                if PARALLEL_EPS > d:
                    usable[i] = False
                if ZERO > d:
                    p = [p[0], p[1], p[2], p[3] - d]
        work.append(p)
        if facing[i] and not excluded[i]:
            out.append(p)
            if len(out) == MAX_PLANES:
                return out
    for k, i in enumerate(SIDE_CYCLE):
        if excluded[i]:
            continue
        j = SIDE_CYCLE[(k + 1) & 3]
        if facing[i] != facing[j] and usable[i] and usable[j]:
            pl = edge_plane(orthographic, planes, clip_to_world, eye, t_near_arg, t_far_arg, i, j, 5, light_type, L)
            if pl is not None:
                out.append(pl)
                if len(out) == MAX_PLANES:
                    return out
    for i in NEAR_FAR:
        if excluded[i]:
            continue
        for k, j in enumerate(SIDE_CYCLE):
            if facing[i] != facing[j] and usable[i] and usable[j]:
                third = SIDE_CYCLE[(k + 1) & 3]
                pl = edge_plane(orthographic, planes, clip_to_world, eye, t_near_arg, t_far_arg, i, j, third,
                                light_type, L)
                if pl is not None:
                    out.append(pl)
                    if len(out) == MAX_PLANES:
                        return out
    return out


# ----------------------------------------------------------------- kernel (mode 0)
def kernel_node_visible(node, planes, culling_mask, scene_mask, lod_masks=None):
    """``ProcessShadowCasterNodeVisibilityAndCullWithoutUmbra_Kernel<0, n>`` (0x6a7bc0 for
    n <= 4; 8 / 12 identical per plane): layer bit in cullingMask, node sceneMask &
    cull sceneMask, renderer non-null, SceneNode+0x1c bit 29 clear, LOD test when
    flags & 0x0fffffff, then for every plane with negated planes (xorps sign):
    rad = (|ny| ey + |nx| ex) + |nz| ez, dist = ((cy (-ny) + cx (-nx)) + cz (-nz)) + (-d);
    outside iff rad < dist (cmpltps)."""
    if not (int(culling_mask) >> (int(node['layer']) & 31)) & 1:
        return False
    if not (int(node.get('scene_mask', SCENE_MASK_DEFAULT)) & int(scene_mask)):
        return False
    if not node.get('has_renderer', True):
        return False
    flags = int(node['flags']) & 0xFFFFFFFF
    if flags & NODE_PENDING_REMOVAL:
        return False
    li = flags & NODE_LOD_INDEX
    if li:
        if lod_masks is None:
            raise ShadowCullGap('LOD group node: LODDataArray masks (LODGroupManager) not available')
        if not (int(lod_masks[int(node.get('lod_slot', 0))][li]) & int(node.get('lod_mask', 0)) & 0xFF):
            return False
    cx, cy, cz, ex, ey, ez = (F(v) for v in node['aabb'])
    for a, b, c, d in planes:
        na, nb, nc, nd = -F(a), -F(b), -F(c), -F(d)
        dist = ((cy * nb + cx * na) + cz * nc) + nd
        rad = (abs(nb) * ey + abs(na) * ex) + abs(nc) * ez
        if rad < dist:
            return False
    return True


def combine_layer_distances(camera_distances, light_distances):
    """Fn 0x683400 (when Light+0xc0 is set): per layer, zeros become FLT_MAX, minps, and
    a result not below FLT_MAX becomes 0."""
    big = F(np.finfo(np.float32).max)
    out = []
    for a, b in zip(camera_distances, light_distances):
        a, b = F(a), F(b)
        a2 = big if a == ZERO else a
        b2 = big if b == ZERO else b
        m = a2 if a2 < b2 else b2           # minps xmm3(light-derived), xmm2(camera-derived)
        out.append(m if m < big else ZERO)
    return out


# ------------------------------------------------------------ whole light cull
def scene_node(renderer, render_native=None):
    """SceneNode of an exported renderer (``Renderer::AddToScene`` 0x739540 /
    ``RendererScene::AddRendererInternal`` 0x68ae10 / ``SetShadowCastingMode`` 0x73fd60):

    * +0x00 sceneMask = GameObject::GetCullSceneMask 0x638930 = UnityScene+0xc0, written only
      by the UnityScene ctor (0x820e56: 0x8000000000000000), or ~0 without a scene;
    * +0x08 BaseRenderer (non-null), +0x10 layer;
    * +0x1c bits 30..31 = Renderer+0x128 bits 6..7 (<< 24), bit 29 only RemoveRenderer
      (0x69101a), bit 28 the GameObject+0x5c editor flag, bits 0..27 LOD group index from
      Renderer+0x1a0 (GetLODGroupIndexAndMask 0x773000; 0 without a LODGroup);
    * the AABB = the renderer world AABB of this render (scene bounding-box array).

    ``render_native``: the renderer's native block of this render record (flags_0x128)."""
    nat = render_native or (renderer['raw'].get('native') or {})
    if 'flags_0x128' not in nat:
        raise ShadowCullGap('renderer %s: Renderer+0x128 flags not exported' % renderer['name'])
    flags128 = int(str(nat['flags_0x128']), 16) if isinstance(nat['flags_0x128'], str) else int(nat['flags_0x128'])
    mode = (flags128 >> 6) & 3
    return {'id': renderer['id'], 'name': renderer['name'], 'layer': int(renderer['layer']),
            'scene_mask': SCENE_MASK_DEFAULT, 'has_renderer': True, 'flags': (mode << 30) & 0xC0000000,
            'aabb': [F(v) for v in renderer['center']] + [F(v) for v in renderer['extent']],
            'shadow_casting_mode': ('off', 'on', 'two_sided', 'shadows_only')[mode],
            'lightmap_index_is_none': (int(renderer.get('lightmap_index', -1)) & 0xFFFF) == 0xFFFF,
            'scene_handle': int(nat.get('scene_handle_0x160', -1))}


def directional_caster_set(nodes, cam, light, shadow_distance, scd):
    """Renderer ids of the light's shadow-caster cull output (scene-node order) plus
    evidence.  ``cam``: dict for :func:`scriptable_cull_matrices` plus culling_mask,
    layer_cull_distances, layer_cull_spherical; ``light``: light_vec (SharedLightData+0x160,
    L2W-no-scale column 2), culling_mask, layer_shadow_cull_distances or None
    (Light+0xc0 not set); ``scd``: ``unity_shadow_bounds.setup_shadow_cull_data`` result."""
    from .unity_shadow_bounds import directional_caster_post_cull
    scp = scriptable_cull_matrices(cam)
    c2w, far380 = shadow_cull_clip_to_world(cam, scp, shadow_distance)
    ratio = F(shadow_distance) / far380
    planes = calculate_shadow_caster_cull(bool(cam['orthographic']), scd['planes_b'], c2w, cam['position'],
                                          ONE, ratio, LIGHT_DIRECTIONAL, light['light_vec'])
    if not 1 <= len(planes) <= 12:
        raise ShadowCullGap('%d shadow caster cull planes (kernels take 1..12)' % len(planes))
    cull_mask = int(cam['culling_mask']) & 0xFFFFFFFF
    lcd = [F(v) for v in cam['layer_cull_distances']]
    if light.get('layer_shadow_cull_distances') is not None:
        lcd = combine_layer_distances(lcd, light['layer_shadow_cull_distances'])
    ctx = {'layer_mask': int(light['culling_mask']) & cull_mask, 'skip_baked': None,
           'layer_cull_distances': lcd, 'layer_cull_spherical': bool(cam['layer_cull_spherical']),
           'shadow_cull_data': scd}
    visible, rows = [], []
    for n in sorted(nodes, key=lambda x: x['scene_handle']):
        k = kernel_node_visible(n, planes, cull_mask, SCENE_MASK_DEFAULT)
        p = None
        if k:
            if not n['lightmap_index_is_none']:
                raise ShadowCullGap('lightmapped caster %s: skip-baked test 0x682580 inputs not exported' % n['name'])
            p = directional_caster_post_cull({'shadow_casting_mode': n['shadow_casting_mode'], 'layer': n['layer'],
                                              'world_bounds': n['aabb'], 'lightmap_index_is_none': True}, ctx)
            if p:
                visible.append(n['id'])
        rows.append({'id': n['id'], 'name': n['name'], 'kernel': bool(k), 'post_cull': p})
    return {'ids': visible, 'planes': [[float(v) for v in p] for p in planes], 'nodes': rows,
            'cull_matrix_implicit_branch': scp['implicit_matrix_branch'],
            'cull_world_to_clip_0x29c': [float(v) for v in scp['cull_world_to_clip_0x29c']],
            'clip_to_world_0x1c0': [float(v) for v in c2w], 'far_0x380': float(far380)}
