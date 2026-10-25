"""Non-forward camera passes of one Unity 2018.4.11f1 camera render, bound to the scene export.

Binds the installed shader library and the frame-pair export to the sibling
transcription ``unity_pass_sorters`` (UnityPlayer.dll RVAs there):

* camera depth texture (``RenderSceneDepthPass`` 0x6cef40): ShadowCaster pass
  per opaque-queue material, ``ROSorterDepthPass`` order;
* camera depth-normals (``RenderShaderReplacementPass`` 0x6680f0 with the
  GraphicsSettings depth-normals shader, tag ``RenderType``): every pass of the
  matched replacement subshader, ``ROSorterReplacement`` order.

Inputs the export does not carry are explicit gaps (``mesh_id`` / static-batch
fields -> the low key bits; recorded in ``gaps``).  Material override tags:
the export's ``Material.GetTag`` value is the override when one exists and the
shader tag otherwise, which is exactly the value ``CalculateSubShaderIndexToUse``
(0x69b270) looks up, so it is passed as the override when it differs from the
active subshader's tag.
"""
from __future__ import annotations

from . import unity_forward_rules as rules
from . import unity_pass_sorters as ps
from .unity_forward_plan import renderer_pass_keywords


# Legacy shader renames: (old, new) pairs at .rdata 0x14b6e90 (78 entries, read from the
# installed UnityPlayer.dll), used by the shader lookup at RVA 0x83d6e2 (no public symbol).
SHADER_RENAMES = {
    'Diffuse': 'Legacy Shaders/Diffuse',
    'Bumped Specular': 'Legacy Shaders/Bumped Specular',
    'Bumped Diffuse': 'Legacy Shaders/Bumped Diffuse',
    'VertexLit': 'Legacy Shaders/VertexLit',
    'Parallax Specular': 'Legacy Shaders/Parallax Specular',
    'Parallax Diffuse': 'Legacy Shaders/Parallax Diffuse',
    'Specular': 'Legacy Shaders/Specular',
    'Diffuse Detail': 'Legacy Shaders/Diffuse Detail',
    'Decal': 'Legacy Shaders/Decal',
    'Reflective/Bumped Diffuse': 'Legacy Shaders/Reflective/Bumped Diffuse',
    'Reflective/VertexLit': 'Legacy Shaders/Reflective/VertexLit',
    'Reflective/Parallax Specular': 'Legacy Shaders/Reflective/Parallax Specular',
    'Reflective/Parallax Diffuse': 'Legacy Shaders/Reflective/Parallax Diffuse',
    'Reflective/Specular': 'Legacy Shaders/Reflective/Specular',
    'Reflective/Diffuse': 'Legacy Shaders/Reflective/Diffuse',
    'Reflective/Bumped VertexLit': 'Legacy Shaders/Reflective/Bumped VertexLit',
    'Reflective/Bumped Specular': 'Legacy Shaders/Reflective/Bumped Specular',
    'Reflective/Bumped Unlit': 'Legacy Shaders/Reflective/Bumped Unlit',
    'Self-Illumin/Bumped Diffuse': 'Legacy Shaders/Self-Illumin/Bumped Diffuse',
    'Self-Illumin/Parallax Specular': 'Legacy Shaders/Self-Illumin/Parallax Specular',
    'Self-Illumin/Parallax Diffuse': 'Legacy Shaders/Self-Illumin/Parallax Diffuse',
    'Self-Illumin/Specular': 'Legacy Shaders/Self-Illumin/Specular',
    'Self-Illumin/Diffuse': 'Legacy Shaders/Self-Illumin/Diffuse',
    'Self-Illumin/Bumped Specular': 'Legacy Shaders/Self-Illumin/Bumped Specular',
    'Self-Illumin/VertexLit': 'Legacy Shaders/Self-Illumin/VertexLit',
    'Transparent/Bumped Diffuse': 'Legacy Shaders/Transparent/Bumped Diffuse',
    'Transparent/VertexLit': 'Legacy Shaders/Transparent/VertexLit',
    'Transparent/Parallax Specular': 'Legacy Shaders/Transparent/Parallax Specular',
    'Transparent/Parallax Diffuse': 'Legacy Shaders/Transparent/Parallax Diffuse',
    'Transparent/Specular': 'Legacy Shaders/Transparent/Specular',
    'Transparent/Diffuse': 'Legacy Shaders/Transparent/Diffuse',
    'Transparent/Bumped Specular': 'Legacy Shaders/Transparent/Bumped Specular',
    'Transparent/Cutout/VertexLit': 'Legacy Shaders/Transparent/Cutout/VertexLit',
    'Transparent/Cutout/Soft Edge Unlit': 'Legacy Shaders/Transparent/Cutout/Soft Edge Unlit',
    'Transparent/Cutout/Specular': 'Legacy Shaders/Transparent/Cutout/Specular',
    'Transparent/Cutout/Diffuse': 'Legacy Shaders/Transparent/Cutout/Diffuse',
    'Transparent/Cutout/Bumped Specular': 'Legacy Shaders/Transparent/Cutout/Bumped Specular',
    'Transparent/Cutout/Bumped Diffuse': 'Legacy Shaders/Transparent/Cutout/Bumped Diffuse',
    'RenderFX/Skybox Cubed': 'Skybox/Cubemap',
    'RenderFX/Skybox': 'Skybox/6 Sided',
    ' VertexLit': 'Legacy Shaders/VertexLit',
    ' Diffuse': 'Legacy Shaders/Diffuse',
    ' Glossy': 'Legacy Shaders/Specular',
    ' Bumped': 'Legacy Shaders/Bumped Diffuse',
    ' BumpedSpecular': 'Legacy Shaders/Bumped Specular',
    'ParallaxBump/Diffuse': 'Legacy Shaders/Parallax Diffuse',
    'ParallaxBump/Specular': 'Legacy Shaders/Parallax Specular',
    'Alpha/VertexLit': 'Legacy Shaders/Transparent/VertexLit',
    'Alpha/Diffuse': 'Legacy Shaders/Transparent/Diffuse',
    'Alpha/Glossy': 'Legacy Shaders/Transparent/Specular',
    'Alpha/Bumped': 'Legacy Shaders/Transparent/Bumped Diffuse',
    'Alpha/BumpedSpecular': 'Legacy Shaders/Transparent/Bumped Specular',
    'ParallaxBump/AlphaDiffuse': 'Legacy Shaders/Transparent/Parallax Diffuse',
    'ParallaxBump/AlphaSpecular': 'Legacy Shaders/Transparent/Parallax Specular',
    'Reflective/Glossy': 'Legacy Shaders/Reflective/Specular',
    'Reflective/Bumped': 'Legacy Shaders/Reflective/Bumped Diffuse',
    'Reflective/BumpedSpecular': 'Legacy Shaders/Reflective/Bumped Specular',
    'ParallaxBump/ReflectDiffuse': 'Legacy Shaders/Reflective/Parallax Diffuse',
    'ParallaxBump/ReflectSpecular': 'Legacy Shaders/Reflective/Parallax Specular',
    'Lightmapped/Glossy': 'Legacy Shaders/Lightmapped/Specular',
    'Lightmapped/Bumped': 'Legacy Shaders/Lightmapped/Bumped Diffuse',
    'Lightmapped/BumpedSpecular': 'Legacy Shaders/Lightmapped/Bumped Specular',
    'Lightmapped/VertexLit': 'Legacy Shaders/Lightmapped/VertexLit',
    'Lightmapped/Diffuse': 'Legacy Shaders/Lightmapped/Diffuse',
    'Lightmapped/Specular': 'Legacy Shaders/Lightmapped/Specular',
    'Lightmapped/Bumped Diffuse': 'Legacy Shaders/Lightmapped/Bumped Diffuse',
    'Lightmapped/Bumped Specular': 'Legacy Shaders/Lightmapped/Bumped Specular',
    'Diffuse Fast': 'Legacy Shaders/Diffuse Fast',
    'Self-Illumin/Glossy': 'Legacy Shaders/Self-Illumin/Specular',
    'Self-Illumin/Bumped': 'Legacy Shaders/Self-Illumin/Bumped Diffuse',
    'Self-Illumin/BumpedSpecular': 'Legacy Shaders/Self-Illumin/Bumped Specular',
    'ParallaxBump/IlluminDiffuse': 'Legacy Shaders/Self-Illumin/Parallax Diffuse',
    'ParallaxBump/IlluminSpecular': 'Legacy Shaders/Self-Illumin/Parallax Specular',
    ' DiffuseDetail': 'Legacy Shaders/Diffuse Detail',
    ' Diffuse (fast)': 'Legacy Shaders/Diffuse Fast',
    ' Decal': 'Legacy Shaders/Decal',
    'Hidden/TerrainEngine/Splatmap/Lightmap-FirstPass': 'Nature/Terrain/Diffuse',
    'Hidden/TerrainEngine/Splatmap/Lightmap-AddPass': 'Hidden/TerrainEngine/Splatmap/Diffuse-AddPass',
}


def _resolved_pass(shader, p):
    """(Pass, light mode) of a serialized pass (UsePass resolved)."""
    if p.type == 'use':
        rows = shader.use_pass(p, tier=None)
        if len(rows) != 1:
            raise ps.fr_gap('UsePass %s resolved to %d passes' % (p.record['use_name'], len(rows))) \
                if hasattr(ps, 'fr_gap') else RuntimeError('UsePass ambiguity')
        return rows[0], rows[0].light_mode
    return p, p.light_mode


def sorter_shader(lib, shader):
    """IntShader view for the sorters: own subshaders followed by the Fallback
    chain's (IntShader::ResolveFallback 0x85d170 appends them); active index =
    first supported subshader of that array (IntShader::PostLoad 0x85c4c0)."""
    subs = []
    seen = set()
    cur = shader
    while cur is not None and cur.name not in seen:
        seen.add(cur.name)
        for s in cur.subshaders:
            passes = []
            for p in s.passes:
                rp, lm = _resolved_pass(cur, p)
                passes.append({'light_mode': lm, 'lib_pass': rp})
            subs.append({'tags': s.tags, 'passes': passes, 'lib': s, 'supported': s.supported(None)})
        fb = cur.fallback
        cur = _fallback_copy(lib, fb) if fb and fb.lower() != 'off' else None
    active = next((i for i, s in enumerate(subs) if s['supported']), None)
    if active is None:
        raise RuntimeError('%s: no supported subshader' % shader.name)
    return {'name': shader.name, 'active_subshader': active, 'subshaders': subs}


def _structure(shader):
    return tuple((tuple(sorted(s.tags.items())), tuple(p.light_mode for p in s.passes)) for s in shader.subshaders)


def _fallback_copy(lib, name):
    """Installed copy of a Fallback shader.  Copies may differ in programs (per
    bundle); for pass routing only the subshader tags / pass light modes matter,
    which must agree across copies."""
    copies = lib.copies(name) or lib.copies(SHADER_RENAMES.get(name, name))
    if not copies:
        raise RuntimeError('Fallback shader %s not installed' % name)
    if len({_structure(c) for c in copies}) != 1:
        raise RuntimeError('Fallback %s: installed copies differ in subshader tags/passes' % name)
    return copies[0]


def _nodes(adapter, cam, lib, gaps, cache):
    return _node_rows(adapter.node_queue(cam), lib, gaps, cache)


def _node_rows(rens, lib, gaps, cache):
    nodes = []
    for r in rens:
        mats = []
        for m in r['materials']:
            if m is None:
                raise RuntimeError('null material slot on %s' % r['name'])
            key = m['shader']
            if key not in cache:
                cache[key] = sorter_shader(lib, lib.copies(m['shader'])[0])
            sh = cache[key]
            over = {}
            rt = (m.get('tags') or {}).get('RenderType', '')
            if rt != '' and rt != ps.intshader_get_tag(sh, 'RenderType'):
                over['RenderType'] = rt
            mats.append({'material_instance_id': int(m['id']), 'shader': sh, 'custom_render_queue': int(m['render_queue']),
                         'shader_render_queue': int(m['render_queue']), 'override_tags': over, 'raw': m})
        rp = r['raw']['properties']
        nodes.append({'center': r['center'], 'shadow_casting_mode': r['shadow_casting_mode'], 'materials': mats,
                      'layering': rules.layering_key(0, r['sorting_order']),
                      'priority': int(rp.get('rendererPriority', 0)) & 0xFFFF, 'renderer': r})
    gaps.append({'gap': 'depth_key_mesh_fields', 'detail': 'RenderNode+0x100 mesh id, +0xb8 first subset and '
                 'static-batch fields are not exported; taken as 0 (only the lowest key bits; ties)'})
    return nodes


def depth_texture_plan(adapter, cam, lib, global_keywords, quality, gaps, cache=None):
    cache = {} if cache is None else cache
    nodes = _nodes(adapter, cam, lib, gaps, cache)
    w2c = cam.m('world_to_camera_matrix')
    out = []
    for e in ps.depth_texture_draws(nodes, w2c):
        node = nodes[e['node']]
        mat = node['materials'][e['material_index']]
        sub = e['shader']['subshaders'][e['pass']['subshader']]
        p = sub['passes'][e['pass']['pass']]['lib_pass']
        r = node['renderer']
        kw = renderer_pass_keywords('SHADOWCASTER', r, mat['raw'], {'lights': [], 'has_main_light': False},
                                    global_keywords, quality)
        out.append({'pass_kind': 'depth_texture', 'renderer': r, 'material': mat['raw'], 'shader': e['shader']['name'],
                    'pass': p, 'light_mode': 'SHADOWCASTER', 'keywords': kw, 'key': e['key']})
    return out


def shadow_caster_plan(adapter, lib, global_keywords, quality, gaps, caster_ids, cache=None):
    """Shadow-caster draws of the main directional light (``RenderShadowMaps``
    0x6a18b0 -> caster job 0x6a56c0 -> loop 0x6a0b00).

    Per caster node, every material whose active subshader index
    (CalculateSubShaderIndexToUse 0x69b270) yields a pass from
    ``Shader::GetShadowCasterPassToUse`` (0xbf96c0) contributes one draw with the
    ``SHADOWS_DEPTH`` keyword (SetShadowsKeywords for a directional light).

    ``caster_ids``: renderer ids of the light's shadow RenderNodeQueue in draw
    order.  Both the membership (shadow-caster culling 0x6ac320, gap
    ``caster_cull_set``) and the order (the sort key includes
    ``SharedMaterialData+0xd0``, an XXH32 over a ``ShaderLab::Pass*`` heap
    address: gap ``shadow_caster_order``) are ``external:capture`` inputs."""
    cache = {} if cache is None else cache
    by_id = {r['id']: r for r in adapter.renderers()}
    rens = [by_id[i] for i in caster_ids]
    nodes = _node_rows(rens, lib, gaps, cache)
    out = []
    for node in nodes:
        r = node['renderer']
        for mi, mat in enumerate(node['materials']):
            sub = ps.calculate_subshader_index_to_use(mat['shader'], mat, None)
            if sub < 0:
                continue
            pp = ps.get_shadow_caster_pass_to_use(mat['shader'], sub)
            if pp is None:
                continue
            p = mat['shader']['subshaders'][pp['subshader']]['passes'][pp['pass']]['lib_pass']
            kw = renderer_pass_keywords('SHADOWCASTER', r, mat['raw'], {'lights': [], 'has_main_light': False},
                                        global_keywords, quality)
            out.append({'pass_kind': 'shadow_caster', 'renderer': r, 'material': mat['raw'],
                        'material_index': mi, 'shader': mat['shader']['name'], 'pass': p,
                        'light_mode': 'SHADOWCASTER', 'keywords': kw})
    gaps.append({'gap': 'caster_cull_set', 'detail': 'shadow caster culling (0x6ac320) not transcribed: '
                 'caster membership external:capture'})
    gaps.append({'gap': 'shadow_caster_order', 'detail': 'caster sort key uses SharedMaterialData+0xd0 '
                 '(XXH32 seed 0x8f37154b over a ShaderLab::Pass* address, Material::UpdateHashes 0x84e3b0): '
                 'order external:capture; v6 field request: +0xd0 per material or the caster queue order'})
    return out


def depth_normals_plan(adapter, cam, lib, global_keywords, gaps, cache=None):
    cache = {} if cache is None else cache
    nodes = _nodes(adapter, cam, lib, gaps, cache)
    rep = sorter_shader(lib, lib.shader('Hidden/Internal-DepthNormalsTexture'))
    w2c = cam.m('world_to_camera_matrix')
    out = []
    for d in ps.depth_normals_draws(nodes, w2c, rep):
        node = nodes[d['node']]
        mat = node['materials'][d['material_index']]
        sub = rep['subshaders'][d['subshader']]
        p = sub['passes'][d['pass']]['lib_pass']
        r = node['renderer']
        kw = frozenset(set(global_keywords) | set(mat['raw'].get('shader_keywords') or []))
        out.append({'pass_kind': 'depth_normals', 'renderer': r, 'material': mat['raw'], 'shader': rep['name'],
                    'pass': p, 'light_mode': p.light_mode, 'keywords': kw, 'subshader': d['subshader']})
    gaps.append({'gap': 'depth_normals_shader_source', 'detail': 'GraphicsSettings builtin shader 4 taken as the '
                 'installed Hidden/Internal-DepthNormalsTexture (library copy)'})
    return out


MV_MODES = {'Camera': ps.MV_CAMERA, 'Object': ps.MV_OBJECT, 'ForceNoMotion': ps.MV_FORCE_NO_MOTION}


def motion_vector_plan(adapter, cam, lib, global_keywords, gaps, included_renderer_ids, cache=None):
    """Object motion-vector draws (DoMotionVectorRenderLoop 0x6c9b60) over the
    opaque RenderObjectData list.

    Candidate set (generated): opaque ROs whose MotionVectorGenerationMode is not
    Camera.  Inclusion of a candidate also depends on SkinnedMeshRenderer+0x255
    and the transform-update frame stamp (sorter gaps M1, M2), which the export
    does not carry: the included renderer ids are an ``external:capture`` input
    and are checked against the candidates.  Pass choice (own MotionVectors pass
    or the internal shader's) and the order (MotionVectorRenderLoop
    RenderObjectSorter 0x6c6f20 with Unity's qsort) are generated."""
    from .unity_forward_plan import render_objects
    cache = {} if cache is None else cache
    ro = render_objects(adapter, cam)
    internal = sorter_shader(lib, lib.shader('Hidden/Internal-MotionVectors'))
    nodes = []
    for r in ro['renderers']:
        mats = []
        for m in r['materials']:
            if m['shader'] not in cache:
                cache[m['shader']] = sorter_shader(lib, lib.copies(m['shader'])[0])
            mats.append({'shader': cache[m['shader']], 'raw': m})
        nodes.append({'materials': mats, 'renderer': r, 'motion_vector_mode': MV_MODES[r['motion_vector_mode']],
                      'priority': int(r['raw']['properties'].get('rendererPriority', 0)) & 0xFFFF})
    candidates, entries = [], []
    for ri, x in enumerate(ro['opaque_ros']):
        node = nodes[x['node']]
        if node['motion_vector_mode'] == ps.MV_CAMERA:
            continue
        candidates.append(node['renderer']['id'])
        if node['renderer']['id'] not in included_renderer_ids:
            continue
        shader = node['materials'][x['material_index']]['shader']
        sub = shader['subshaders'][shader['active_subshader']]
        own = next((i for i, p in enumerate(sub['passes']) if p['light_mode'] == 'MOTIONVECTORS'), -1)
        if own >= 0:
            entries.append({'ro': ri, 'object_pass': True, 'pass': own, 'shader': shader})
        else:
            isub = internal['subshaders'][internal['active_subshader']]
            ip = next(i for i, p in enumerate(isub['passes']) if p['light_mode'] == 'MOTIONVECTORS')
            entries.append({'ro': ri, 'object_pass': False, 'pass': ip, 'shader': internal})
    gaps.append({'gap': 'motion_vector_inclusion', 'detail': 'SkinnedMeshRenderer+0x255 and BaseRenderer+0xbc not '
                 'exported: included renderers taken from the capture (external:capture)',
                 'candidates_not_included': sorted(set(candidates) - set(included_renderer_ids)),
                 'included_not_candidates': sorted(set(included_renderer_ids) - set(candidates))})
    out = []
    for e in ps.sort_motion_vectors(entries, ro['opaque_ros'], nodes):
        x = ro['opaque_ros'][e['ro']]
        node = nodes[x['node']]
        mat = node['materials'][x['material_index']]['raw']
        sh = e['shader']
        p = sh['subshaders'][sh['active_subshader']]['passes'][e['pass']]['lib_pass']
        kw = frozenset(set(global_keywords) | set(mat.get('shader_keywords') or []))
        out.append({'pass_kind': 'motion_vectors', 'renderer': node['renderer'], 'material': mat, 'shader': sh['name'],
                    'pass': p, 'light_mode': p.light_mode, 'keywords': kw, 'object_pass': e['object_pass']})
    return out, candidates
