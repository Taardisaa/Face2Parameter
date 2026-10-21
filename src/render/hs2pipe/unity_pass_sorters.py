"""Unity 2018.4.11f1 built-in NON-forward draw lists recovered from UnityPlayer.dll.

Source: ``E:/HoneySelect2_ArcticFox/UnityPlayer.dll`` (SHA-256 ``5034cdc2bff7...``,
CodeView GUID ``BFA91A8C44734BB4AABC5A975B304CFE`` age 1), static disassembly with
``tools/offline_renderer/unityplayer_disasm.py`` against Unity's public PDB.
Dumps: ``outputs/offline_renderer_unity_forward_20261020/disasm/sorters``.
Specification (every RVA, offset and explicit gap):
``outputs/offline_renderer_unity_forward_20261020/pass_sorters_spec.md``.

Passes covered (one camera render):
  1. Camera depth texture        RenderDepthTexture 0x6672d0 -> RenderSceneDepthPass 0x6cef40
  2. Camera depth-normals        RenderDepthNormalsTexture 0x667000 -> RenderShaderReplacementPass
                                 0x6680f0 -> RenderSceneShaderReplacement 0x6cf340
  3. Shadow-caster draws of a shadow map split   0x69cfa0/0x69d310 (parts), 0x6a5880 (keys + QSort)
  4. Object motion vectors       DoMotionVectorRenderLoop 0x6c9b60

All four sorts use the same ``qsort_internal::QSort`` algorithm as the forward
loop (checked instruction by instruction for each instantiation; see spec), so
``unity_forward_rules.unity_qsort`` is reused with the pass-specific comparator.

Plain dicts in, plain lists out.  Float operations are binary32 (numpy float32)
in machine-code operand order.  Anything not transcribed raises
``NotImplementedError`` naming the gap.
"""
from __future__ import annotations

import numpy as np

from . import unity_forward_rules as fr

F = np.float32
_ZERO = F(0.0)
_ONE = F(1.0)

QUEUE_SPLIT = fr.QUEUE_SPLIT                     # 2500: `cmp eax,0x9c4; jg skip` (DepthPass 0x6cd600)
SHADOW_DEPTH_SCALE = fr._fbits(0x467FFC00)       # 16383.0, 0x6a58de
VECTOR3_EPSILON = fr._fbits(0x3727C5AC)          # Vector3f::epsilon (.rdata 0x1257060), 0x6ca0bf
MV_DEPTH_BIAS = fr._fbits(0xBA83126F)            # -0.001, unity_MotionVectorsParams.z (0x6cc463)

# kPassLightModeTagNameIDs (shadertag::Initialize 0x84ba90, table at .data 0x1553700).
PASS_LIGHT_MODE_NAMES = ('ALWAYS', 'VERTEX', 'VERTEXLM', 'VERTEXLMRGBM', 'FORWARDBASE', 'FORWARDADD',
                         'PREPASSBASE', 'PREPASSFINAL', 'SHADOWCASTER', 'SHADOWCOLLECTOR', 'DEFERRED',
                         'META', 'MOTIONVECTORS', 'NEVEREXECUTED', 'SRPDEFAULTUNLIT')
PASS_TYPE_SHADOW_CASTER = 8
PASS_TYPE_MOTION_VECTORS = 12                    # `cmp dword [pass+0x2f0], 0xc`

SHADOW_CASTING_OFF, SHADOW_CASTING_ON, SHADOW_CASTING_TWO_SIDED, SHADOW_CASTING_SHADOWS_ONLY = 0, 1, 2, 3
MV_CAMERA, MV_OBJECT, MV_FORCE_NO_MOTION = 0, 1, 2  # RenderNode+0xe8 bits 10-11 (setter 0x970f90)

RENDERING_PATH_VERTEX_LIT, RENDERING_PATH_FORWARD = 0, 1
RENDERING_PATH_DEFERRED_LIGHTING, RENDERING_PATH_DEFERRED_SHADING = 2, 3


# =========================================================================
# Shader tags (shadertag::GetShaderTagID 0x84ae50) and subshader selection
# =========================================================================
def _tag_id_key(name):
    """ShaderTagIDs are interned case-insensitively (map with compare_tstring_insensitive).

    Returns None for ID 0: the empty string (0x84ae58: size 0 -> id 0) and
    ``<invalid>``, the first name registered by shadertag::Initialize (0x84bb0d)
    and therefore given id = map size = 0.  Every other name has an id >= 1.
    """
    if name is None:
        return None
    s = str(name)
    if s == '' or s.lower() == '<invalid>':
        return None
    return s.lower()


def _tag_equal(a, b):
    ka, kb = _tag_id_key(a), _tag_id_key(b)
    return ka is not None and ka == kb


def active_subshader_index(shader):
    """Shader::GetActiveSubShaderIndex 0xbf9360: IntShader+0x48 (set by IntShader::PostLoad 0x85c4c0)."""
    return int(shader['active_subshader'])


def intshader_get_tag(shader, tag):
    """IntShader::GetTag(tag, true) 0x85b760: the ACTIVE subshader's tag map only; absent -> id 0."""
    sub = shader['subshaders'][active_subshader_index(shader)]
    for k, v in sub.get('tags', {}).items():
        if _tag_equal(k, tag):
            return v if _tag_id_key(v) is not None else None
    return None


def subshader_with_tag_value(shader, tag, value):
    """Shader::GetSubShaderWithTagValue 0xbf9710: first subshader (array order, no
    support/LOD test) whose tag map holds ``tag == value``; -1 if none."""
    for i, sub in enumerate(shader['subshaders']):
        for k, v in sub.get('tags', {}).items():
            if _tag_equal(k, tag):
                if _tag_equal(v, value):
                    return i
                break
    return -1


def _material_override_tag(material, tag):
    for k, v in (material.get('override_tags') or {}).items():
        if _tag_equal(k, tag):
            return v if _tag_id_key(v) is not None else None
    return None


def calculate_subshader_index_to_use(shader, material, replace=None):
    """CalculateSubShaderIndexToUse 0x69b270.

    ``replace`` = ShaderReplaceData ``{'shader': replacement shader dict, 'tag': tag name}``
    or None.  ``material['override_tags']`` = SharedMaterialData+0x108 sorted
    (tagID, valueID) vector (Material.SetOverrideTag).
    """
    if replace is None or replace.get('shader') is None:
        return active_subshader_index(shader)
    tag = replace.get('tag')
    if _tag_id_key(tag) is None:                     # tag id <= 0 -> subshader 0 (0x69b303 -> 0x69b33f)
        return 0
    value = _material_override_tag(material, tag)
    if value is None:
        value = intshader_get_tag(shader, tag)
        if value is None:                            # 0x69b32d: value <= 0 -> -1
            return -1
    return subshader_with_tag_value(replace['shader'], tag, value)


# ---- ShadowCaster pass ---------------------------------------------------
def subshader_shadow_flags(subshader):
    """SubShader::PostLoad 0x85c960 -> (SubShader+0x95, SubShader+0x98).

    +0x95 starts at 1 (ctor 0x859007 writes word 0x100 at +0x94) and is cleared
    when the subshader tag ForceNoShadowCasting == "True" (0x85cbb9).
    +0x98 = index of the first pass whose LightMode tag equals
    kPassLightModeTagNameIDs[8] = "SHADOWCASTER" (0x85cb10 loop), else -1.
    """
    allowed = True
    for k, v in subshader.get('tags', {}).items():
        if _tag_equal(k, 'FORCENOSHADOWCASTING'):
            if _tag_equal(v, 'true'):
                allowed = False
            break
    idx = -1
    for i, p in enumerate(subshader.get('passes', [])):
        if _tag_equal(p.get('light_mode'), PASS_LIGHT_MODE_NAMES[PASS_TYPE_SHADOW_CASTER]):
            idx = i
            break
    return allowed, idx


def shader_fallback_shadow_caster(shader):
    """Shader+0x40/+0x48 written by Shader::CreateAndPostLoad 0xbf8e90 via 0xbf9250(…, 8, …).

    Scans the IntShader subshader array in order (this array also holds the
    subshaders copied in from the Fallback shader by IntShader::ResolveFallback
    0x85d170) and returns ``(subshader_index, pass_index)`` of the first pass
    whose LightMode is SHADOWCASTER, or None.
    """
    for si, sub in enumerate(shader['subshaders']):
        for pi, p in enumerate(sub.get('passes', [])):
            if _tag_equal(p.get('light_mode'), PASS_LIGHT_MODE_NAMES[PASS_TYPE_SHADOW_CASTER]):
                return si, pi
    return None


def get_shadow_caster_pass_to_use(shader, subshader_index):
    """Shader::GetShadowCasterPassToUse 0xbf96c0 -> {'subshader', 'pass'} or None.

    ``subshader`` is where the returned Pass lives: the requested subshader when
    it has its own ShadowCaster pass, otherwise the shader-level pass of
    :func:`shader_fallback_shadow_caster`.
    """
    sub = shader['subshaders'][int(subshader_index)]
    allowed, idx = subshader_shadow_flags(sub)
    if not allowed:
        return None
    if idx >= 0:
        return {'subshader': int(subshader_index), 'pass': idx}
    fb = shader_fallback_shadow_caster(shader)
    if fb is None:
        return None
    return {'subshader': fb[0], 'pass': fb[1]}


# =========================================================================
# Shared: view-space z and the depth/replacement 64-bit key (0x6cd6ad..0x6cd75b,
# identical at 0x6cd84b..0x6cd90e)
# =========================================================================
def _view_z(center, world_to_camera):
    """RenderSceneDepthPass 0x6cf0dd / RenderSceneShaderReplacement 0x6cf48e:
    ``((y*m[6] + x*m[2]) + z*m[10]) + m[14]`` (worldToCamera of the current camera)."""
    x, y, z = (F(c) for c in center)
    m = [F(v) for v in world_to_camera]
    return F(F(F(y * m[6] + x * m[2]) + z * m[10]) + m[14])


def depth_sort_key(subshader, node, view_z, material_instance_id):
    """64-bit key of RODataDepthPass / RODataReplacement (DepthPass::PrepareSubset 0x6cd5c0,
    ReplacementPass::PrepareSubset 0x6cd790).

    bits 56..63 subshader index & 0xff
    bits 42..55 (static batched ? RenderNode+0x100 : 0xffff) & 0x3fff
    bits 32..41 IEEE bits(view_z) >> 22  (sign, exponent, top mantissa bit)
    bits 24..31 ((material instance id << 23) & 0xff000000) >> 24  (= id bits 1..8)
    bits 16..23 RenderNode+0xb8 (first static-batch subset) & 0xff
    bits  0..15 RenderNode+0x100 (Mesh+0x174) & 0xffff
    """
    mesh = int(node.get('mesh_id', 0))
    sb = int(node.get('static_batch_subset_count', 0)) != 0
    hi = ((int(subshader) << 24) & 0xFFFFFFFF) | (((mesh if sb else 0xFFFF) & 0x3FFF) << 10)
    zbits = fr.f32_bits(view_z) & 0xFFC00000
    k = (hi << 16) | (int(node.get('first_subset', 0)) & 0xFF)
    k = (k << 6) | zbits
    k = (k << 10) & 0xFFFFFFFFFFFFFFFF
    mat = ((int(material_instance_id) << 23) & 0xFF000000)
    return k | mat | (mesh & 0xFFFF)


def _material_queue(mat):
    q = int(mat.get('custom_render_queue', -1))
    if q < 0:
        q = int(mat['shader_render_queue'])           # IntShader+0x4c of the MATERIAL's shader
    return q


# =========================================================================
# 1. Camera depth texture
# =========================================================================
def depth_pass_objects(nodes, world_to_camera, replace=None):
    """RenderSceneDepthPass(SharedRendererScene) 0x6cef40 + DepthPass::PrepareSubset 0x6cd5c0.

    ``nodes``: the camera RenderNodeQueue in index order (same queue as the forward
    pass).  Node keys: center (+0x80), shadow_casting_mode (+0xe8 bits 6-7),
    mesh_id (+0x100), first_subset (+0xb8), static_batch_subset_count (+0xba),
    materials: [{shader (dict, see calculate_subshader_index_to_use), material_instance_id
    (SharedMaterialData+0xdc), custom_render_queue, shader_render_queue, override_tags}].
    ``replace`` = CullResults+0x7a8 (Camera.SetReplacementShader data) or None.
    Returns RODataDepthPass entries in append order.
    """
    out = []
    for ni, node in enumerate(nodes):
        if int(node.get('shadow_casting_mode', 0)) == SHADOW_CASTING_SHADOWS_ONLY:   # 0x6cf0d4
            continue
        vz = _view_z(node['center'], world_to_camera)
        for mi, mat in enumerate(node['materials']):
            if _material_queue(mat) > QUEUE_SPLIT:                                   # 0x6cd600
                continue
            sub = calculate_subshader_index_to_use(mat['shader'], mat, replace)
            if sub < 0:
                continue
            shader = replace['shader'] if (replace is not None and replace.get('shader') is not None) else mat['shader']
            p = get_shadow_caster_pass_to_use(shader, sub)
            if p is None:                                                            # 0x6cd657
                continue
            out.append({'key': depth_sort_key(sub, node, vz, mat['material_instance_id']),
                        'node': ni, 'subshader': sub, 'shader': shader, 'material_index': mi,
                        'pass': p})
    return out


def depth_pass_less(a, b):
    """ROSorterDepthPass (inlined in QSort 0x6c4f30 / MedianOfThreeSwap 0x6c3040 /
    Partition3Way 0x6c3820): key (u64) ascending, then material index (int) ascending."""
    if a['key'] != b['key']:
        return a['key'] < b['key']
    return fr._s32(a['material_index']) < fr._s32(b['material_index'])


def sort_depth_pass(entries):
    """QSortSingleJobData<RODataDepthPass>::SortJob 0x6d0530 (depth = count)."""
    arr = list(entries)
    fr.unity_qsort(arr, depth_pass_less)
    return arr


def depth_texture_draws(nodes, world_to_camera, replace=None):
    """Sorted depth-texture draws: one ShadowCaster-pass draw per entry (render job
    0x6c8840 calls GetShadowCasterPassToUse again per entry)."""
    return sort_depth_pass(depth_pass_objects(nodes, world_to_camera, replace))


# =========================================================================
# 2. Camera depth-normals texture (shader replacement, tag "RenderType")
# =========================================================================
def replacement_objects(nodes, world_to_camera, replace):
    """RenderSceneShaderReplacement(SharedRendererScene) 0x6cf340 + ReplacementPass::PrepareSubset 0x6cd790.

    No queue filter, no ShadowsOnly test, no pass test: every material whose
    CalculateSubShaderIndexToUse >= 0 contributes one entry.  Extra node keys:
    layering (+0xbc, used as is, no SortingGroup substitution), sorting_group
    (+0xc0), priority (+0x122), lod_fade (+0x104).
    """
    out = []
    for ni, node in enumerate(nodes):
        vz = _view_z(node['center'], world_to_camera)
        for mi, mat in enumerate(node['materials']):
            sub = calculate_subshader_index_to_use(mat['shader'], mat, replace)
            if sub < 0:
                continue
            out.append({'key': depth_sort_key(sub, node, vz, mat['material_instance_id']),
                        'node': ni, 'subshader': sub, 'shader': replace['shader'], 'material_index': mi,
                        'layering': fr._u32(node['layering']),
                        'sorting_group': fr._u32(node.get('sorting_group', fr.NO_SORTING_GROUP)),
                        'priority': fr._u16(node.get('priority', 0))})
    return out


def replacement_less(a, b):
    """ROSorterReplacement (inlined in QSort 0x6c5140 / Partition3Way 0x6c3aa0):
    layerAndOrder (u32) asc, key (u64) asc, RenderNode+0x122 (u16) DESC, else not less."""
    if a['layering'] != b['layering']:
        return a['layering'] < b['layering']
    if a['key'] != b['key']:
        return a['key'] < b['key']
    if a['priority'] != b['priority']:
        return a['priority'] > b['priority']
    return False


def sort_replacement(entries):
    """QSortSingleJobData<RODataReplacement>::SortJob 0x6d0560 (depth = count)."""
    arr = list(entries)
    fr.unity_qsort(arr, replacement_less)
    return arr


def replacement_draws(nodes, world_to_camera, replace):
    """Render job 0x6cf770: for each sorted entry, EVERY pass of the replacement
    subshader in pass order (RenderSingleWithPass 0x6cf550), consecutively."""
    draws = []
    for e in sort_replacement(replacement_objects(nodes, world_to_camera, replace)):
        npass = len(replace['shader']['subshaders'][e['subshader']].get('passes', []))
        for p in range(npass):
            draws.append({'node': e['node'], 'material_index': e['material_index'],
                          'subshader': e['subshader'], 'pass': p})
    return draws


def depth_normals_draws(nodes, world_to_camera, depth_normals_shader):
    """RenderDepthNormalsTexture 0x667000: replacement shader = GraphicsSettings
    builtin shader 4 (Internal-DepthNormalsTexture), tag "RenderType"."""
    return replacement_draws(nodes, world_to_camera, {'shader': depth_normals_shader, 'tag': 'RenderType'})


def update_depth_textures(*, depth_texture_mode, rendering_path, caps_0x39f=False,
                          screen_space_shadows_tier=False, replacement_intshader_0x5c_clear=False,
                          has_screen_space_shadow_shader=False, cull_results_0x2f0=False,
                          should_use_shadows=False, format_supported=True, camera_valid=True):
    """Camera::UpdateDepthTextures 0x66e200 -> ordered list of 'depth' / 'depth_normals'.

    flags = Camera+0x4e4 (DepthTextureMode: 1 Depth, 2 DepthNormals).
    ``screen_space_shadows_tier``: GraphicsSettings tier[GraphicsCaps+0xd4]+0xb4;
    ``replacement_intshader_0x5c_clear``: a camera replacement shader is set and its
    IntShader+0x5c is 0; ``should_use_shadows``: ShouldUseShadows(true) 0x6a57f0;
    ``cull_results_0x2f0``: CullResults+0x2f0.  Depth is always rendered first.
    """
    flags = int(depth_texture_mode)
    deferred = int(rendering_path) in (RENDERING_PATH_DEFERRED_LIGHTING, RENDERING_PATH_DEFERRED_SHADING)
    if not caps_0x39f and deferred:
        flags |= 1
    if not deferred:
        if (screen_space_shadows_tier and not replacement_intshader_0x5c_clear and has_screen_space_shadow_shader
                and cull_results_0x2f0 and should_use_shadows):
            flags |= 1
    if (flags & 1) and deferred and caps_0x39f:
        flags &= ~1
    if (flags & 2) and deferred:
        flags &= ~2
    if flags == 0 or not format_supported or not camera_valid:
        return []
    out = []
    if flags & 1:
        out.append('depth')
    if flags & 2:
        out.append('depth_normals')
    return out


# =========================================================================
# 3. Shadow-caster draws of one shadow-map split
# =========================================================================
def shadow_caster_parts(shadow_nodes, replace=None, start_parts=None, start_casters=None):
    """fn 0x69cfa0 (loop over the light's shadow RenderNodeQueue, header+0x5d8) ->
    fn 0x69d310 per node.

    ``shadow_nodes``: shadow queue in index order; node keys: materials (as in
    :func:`depth_pass_objects`), first_subset, static_batch_subset_count.
    No queue filter and no ShadowCastingMode test here (Off/ShadowsOnly were
    decided by shadow culling).  ``replace`` = header+0x4d8.
    Returns (casters [{node, first, end}], parts [{subset, subshader, shader, material,
    node, material_index}]).
    """
    parts = list(start_parts or [])
    casters = list(start_casters or [])
    for ni, node in enumerate(shadow_nodes):
        first = len(parts)
        count = int(node.get('static_batch_subset_count', 0))
        for mi, mat in enumerate(node['materials']):
            shader = mat.get('shader')
            if shader is None:                                   # 0x69d384
                continue
            sub = calculate_subshader_index_to_use(shader, mat, replace)
            if sub < 0:
                continue
            use = replace['shader'] if (replace is not None and replace.get('shader') is not None) else shader
            if get_shadow_caster_pass_to_use(use, sub) is None:  # 0x69d488
                continue
            si = mi
            if count != 0 and mi >= count:
                si = count - 1
            parts.append({'subset': (int(node.get('first_subset', 0)) + si) & 0xFFFFFFFF,
                          'subshader': sub, 'shader': use, 'material': mat,
                          'node': ni, 'material_index': mi})
        if len(parts) != first:
            casters.append({'node': ni, 'first': first, 'end': len(parts)})
    return casters, parts


def directional_split_masks(caster_count, *, shadow_projection, cascade_count):
    """header+0x510: one byte per caster, new bytes set to 1 (0x6a0056 rep stosb),
    then CullDirectionalCascades 0x67aaf0 (call 0x6a03c4).  It returns without
    writing when projection == CloseFit (0) and cascadeCount == 1 (0x67ab1c)."""
    if int(shadow_projection) == 0 and int(cascade_count) == 1:
        return [1] * int(caster_count)
    raise NotImplementedError('CullDirectionalCascades 0x67aaf0 body: CalculateShadowCasterCull 0x679140 + '
                              'ComputeVisibleSplitMask 0x67a100 not transcribed (stable fit or >1 cascade)')


def shadow_caster_depth16(center, cull_matrix):
    """fn 0x6a5880 (0x6a5960..0x6a5a00): camera-clip depth of the node AABB centre.

    ``cull_matrix`` = ShadowJobData+0x80 (= ShadowCullData+0x80, copied by
    SetupShadowCullData 0x6817e2 from ScriptableCullingParameters+0x29c: the
    camera culling world-to-clip matrix).
    """
    x, y, z = (F(c) for c in center)
    m = [F(v) for v in cull_matrix]
    zc = F(F(F(y * m[6] + x * m[2]) + z * m[10]) + m[14])
    wc = F(F(F(y * m[7] + x * m[3]) + z * m[11]) + m[15])
    with np.errstate(all='ignore'):
        d = F(zc / wc)
    if _ZERO > d:                       # comiss 0, d; jbe -> NaN goes to the min branch
        v = F(_ZERO * SHADOW_DEPTH_SCALE)
    else:
        with np.errstate(all='ignore'):
            v = F(fr._min_ss(_ONE, d) * SHADOW_DEPTH_SCALE)
    if np.isnan(v) or abs(float(v)) >= 2147483648.0:
        return 0                        # cvttss2si -> 0x80000000, low 16 bits 0
    return int(np.trunc(float(v))) & 0xFFFF


def shadow_caster_sort_key(node, part, cull_matrix):
    """ShadowCasterSortData key, fn 0x6a5880 (0x6a5a26..0x6a5aaa).

    bit 63      ShadowCastingMode == TwoSided  ((RenderNode+0xe8 & 0xc0) == 0x80)
    bits 40..61 SharedMaterialData+0xd0 & 0x3fffff (shadow-caster pass values hash,
                Material::UpdateHashes 0x84e3b0 = Pass::ComputePassValuesHash 0x855f70)
    bit 39      odd negative scale ((RenderNode+0xb0 >> 2) & 1)
    bits 24..38 RenderNode+0x100 & 0x7fff
    bits 16..23 static batched ? 0 : part subset & 0xff
    bits  0..15 static batched ? RenderNode+0xb8 : depth16 (:func:`shadow_caster_depth16`)
    """
    h = part['material'].get('shadow_caster_hash')
    if h is None:
        raise NotImplementedError('SharedMaterialData+0xd0 (Pass::ComputePassValuesHash 0x855f70 of the '
                                  'shader ShadowCaster pass, Material::UpdateHashes 0x84e3b0) not transcribed: '
                                  'supply material["shadow_caster_hash"]')
    two = 0x800000 if int(node.get('shadow_casting_mode', 1)) == SHADOW_CASTING_TWO_SIDED else 0
    odd = (int(node.get('transform_type', 0)) >> 2) & 1
    sb = int(node.get('static_batch_subset_count', 0)) != 0
    low = (int(node.get('first_subset', 0)) & 0xFFFF) if sb else shadow_caster_depth16(node['center'], cull_matrix)
    byte = 0 if sb else (int(part['subset']) & 0xFF)
    k = (int(h) & 0x3FFFFF) | two
    k = (k + k) | odd
    k = (k << 15) | (int(node.get('mesh_id', 0)) & 0x7FFF)
    k = (k << 8) | byte
    k = (k << 16) | low
    return k & 0xFFFFFFFFFFFFFFFF


def shadow_caster_less(a, b):
    """ShadowCasterSortDataSorter (inlined in QSort 0x696c00 / pivot 0x696a50): key (u64) ascending only."""
    return a['key'] < b['key']


def shadow_caster_draws(shadow_nodes, casters, parts, *, split, split_masks, cull_matrix, custom_casters=()):
    """fn 0x6a5880: build ShadowCasterSortData for one split and QSort them (depth = count).

    casters with ``split_masks[i] & (1 << split)``, then every caster of
    ``custom_casters`` (second list, no mask test: terrain custom-cull casters,
    RenderShadowMaps 0x6a1f50).  Returns sorted entries {key, node, part}; each is
    drawn with GetShadowCasterPassToUse(part.shader, part.subshader) (0x6a102c);
    BatchRenderer::Add may merge consecutive entries into one instanced draw.
    """
    bit = 1 << int(split)
    out = []
    for ci, c in enumerate(casters):
        if not (int(split_masks[ci]) & bit & 0xFF):
            continue
        node = shadow_nodes[c['node']]
        for pi in range(c['first'], c['end']):
            out.append({'key': shadow_caster_sort_key(node, parts[pi], cull_matrix), 'node': c['node'], 'part': pi})
    for c in custom_casters:
        node = shadow_nodes[c['node']]
        for pi in range(c['first'], c['end']):
            out.append({'key': shadow_caster_sort_key(node, parts[pi], cull_matrix), 'node': c['node'], 'part': pi})
    fr.unity_qsort(out, shadow_caster_less)
    return out


# =========================================================================
# 4. Object motion vectors
# =========================================================================
def compare_approximately_matrix(a, b, eps=VECTOR3_EPSILON):
    """CompareApproximately(Matrix4x4f, Matrix4x4f, float) 0x7cb900:
    all 16 |a[i]-b[i]| <= eps (comiss eps, d; jb -> false; NaN -> false)."""
    e = F(eps)
    for i in range(16):
        d = F(abs(F(F(a[i]) - F(b[i]))))
        if not (e >= d):
            return False
    return True


def first_pass_of_type(subshader, pass_type):
    for i, p in enumerate(subshader.get('passes', [])):
        if int(p.get('type', 0)) == int(pass_type):
            return i
    return -1


def motion_vectors_enabled(*, depth_texture_mode, mv_shader_supported, rg_half_supported, mv_subshader_pass_count):
    """DoMotionVectorRenderLoop prologue 0x6c9b7e..0x6c9c57: (Camera+0x4e4 & 5) == 5,
    builtin shader 5 present and supported, IsFormatSupported(0x2e), and its active
    subshader has >= 3 passes."""
    return ((int(depth_texture_mode) & 5) == 5 and bool(mv_shader_supported) and bool(rg_half_supported)
            and int(mv_subshader_pass_count) >= 3)


def motion_vector_objects(opaque_ros, nodes, *, renderer_update_frame, internal_mv_shader):
    """DoMotionVectorRenderLoop 0x6ca08f..0x6ca363 over the OPAQUE RenderObjectData list
    (ctx+0x7e0, from unity_forward_rules.extract_render_objects) in RO order.

    Node keys: motion_vector_mode (+0xe8 bits 10-11), skinned_last_position
    (+0xe8 bit 9 = SkinnedMeshRenderer+0x255, written by SkinnedMeshRenderer::AddAsRenderNode
    0x77ae22), transform_update_frame (+0xb4 = BaseRenderer+0xbc), local_to_world
    (+0x00), prev_local_to_world (+0x40).  ``renderer_update_frame`` =
    RendererUpdateManager+0x388 (ctx+0x7bc, DoRenderLoop 0x6b99d5).  RO material
    shader dict needed for the MotionVectors pass test: nodes[n]['materials'][i]['shader'].
    Returns RPDataMotionVector [{ro, object_pass (bool), pass, force_no_motion,
    skinned}] in append order.
    """
    out = []
    for ri, ro in enumerate(opaque_ros):
        node = nodes[ro['node']]
        mode = int(node.get('motion_vector_mode', MV_CAMERA)) & 3
        if mode == 0:                                                    # test eax, 0xc00
            continue
        shader = node['materials'][ro['material_index']]['shader']
        sub = shader['subshaders'][active_subshader_index(shader)]
        own = first_pass_of_type(sub, PASS_TYPE_MOTION_VECTORS)
        skinned = bool(node.get('skinned_last_position', False))
        include = skinned
        if not include:
            if int(node.get('transform_update_frame', -1)) == int(renderer_update_frame) and \
                    not compare_approximately_matrix(node['local_to_world'], node['prev_local_to_world']):
                include = True
            elif own >= 0:
                include = True
            elif mode == MV_FORCE_NO_MOTION:
                include = True
        if not include:
            continue
        if own >= 0:
            entry = {'ro': ri, 'object_pass': True, 'pass': own}
        else:
            isub = internal_mv_shader['subshaders'][active_subshader_index(internal_mv_shader)]
            entry = {'ro': ri, 'object_pass': False, 'pass': first_pass_of_type(isub, PASS_TYPE_MOTION_VECTORS)}
        entry['force_no_motion'] = mode == MV_FORCE_NO_MOTION        # unity_MotionVectorsParams.y (0x6cc50d)
        entry['skinned'] = skinned                                    # unity_MotionVectorsParams.x (0x6cc4e7)
        out.append(entry)
    return out


def motion_vector_less(a, b, ros, nodes):
    """MotionVectorRenderLoop::RenderObjectSorter::operator() 0x6c6f20 (also inlined in
    InsertionSort 0x6c2cf0).  Fields of the RO at RPData+0x1c."""
    ra, rb = ros[a['ro']], ros[b['ro']]
    if ra['layering'] != rb['layering']:
        return ra['layering'] < rb['layering']
    if ra['queue'] != rb['queue']:
        return fr._s16(ra['queue']) < fr._s16(rb['queue'])
    ba, bb = fr.f32_bits(ra['distance']) >> 24, fr.f32_bits(rb['distance']) >> 24
    if ba != bb:
        return ba < bb
    if ra['instancing_key'] != rb['instancing_key']:
        return ra['instancing_key'] > rb['instancing_key']
    da, db = F(ra['distance']), F(rb['distance'])
    if not (da == db):
        return fr._gt(da, db)
    pa = fr._u16(nodes[ra['node']].get('priority', 0))
    pb = fr._u16(nodes[rb['node']].get('priority', 0))
    if pa != pb:
        return pa > pb
    return fr._s32(a['ro']) < fr._s32(b['ro'])


def sort_motion_vectors(entries, ros, nodes):
    """QSortSingleJobData<RPDataMotionVector>::SortJob 0x6d0590 (depth = count)."""
    arr = list(entries)
    fr.unity_qsort(arr, lambda a, b: motion_vector_less(a, b, ros, nodes))
    return arr


def motion_vector_draws(opaque_ros, nodes, *, renderer_update_frame, internal_mv_shader):
    """Object motion-vector draws after the full-screen camera-motion quad
    (0x6c9f47/0x6c9f72): one RenderSingleWithPass per sorted RPData (job 0x6cc380)."""
    return sort_motion_vectors(motion_vector_objects(opaque_ros, nodes, renderer_update_frame=renderer_update_frame,
                                                     internal_mv_shader=internal_mv_shader), opaque_ros, nodes)
