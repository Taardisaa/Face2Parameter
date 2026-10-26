"""Draw plan of one Unity 2018.4.11f1 built-in forward camera render.

Binds the same-frame scene export (frame-pair probe ``scene_NNN.json`` /
``frame_NNN.json`` / ``static.json``) to the transcribed engine rules:

* ``unity_forward_rules`` - camera culling, RenderObjectData extraction, the
  forward RenderObjectSorter comparators and Unity's qsort, per-object light
  classification (UnityPlayer RVAs in that module);
* ``unity_pass_sorters`` - depth texture, depth-normals, shadow-caster and
  motion-vector object lists (separate transcription; used when present);
* ``shader_library`` - subshader/pass routing and variant selection;
* ``unity_forward`` - built-in constant values; ``draw_cbuffers`` writes them
  into the shared per-frame cbuffer pool with Unity's leftover-byte rule.

Every input the export does not carry is a named gap (``PlanGap``), never a
default.  Inputs taken from a capture instead of the scene state must be
passed explicitly and are labelled ``external:capture`` by the caller.
"""
from __future__ import annotations

import base64
import struct
from dataclasses import dataclass, field

import numpy as np

from . import unity_forward as uf
from . import unity_forward_rules as rules
from .scene_state import CameraRender

F = np.float32
OPAQUE_MAX_QUEUE = rules.QUEUE_SPLIT

# Unity ShaderLab PassType values for LightMode tags (UnityEngine.Rendering.PassType).
PASS_TYPE = {'ALWAYS': 0, 'VERTEX': 1, 'VERTEXLM': 2, 'FORWARDBASE': 4, 'FORWARDADD': 5,
             'PREPASSBASE': 6, 'PREPASSFINAL': 7, 'SHADOWCASTER': 8, 'DEFERRED': 10, 'META': 11,
             'MOTIONVECTORS': 12}
SHADOW_CASTING = {'Off': 0, 'On': 1, 'TwoSided': 2, 'ShadowsOnly': 3}
RENDER_MODE = {'Auto': 0, 'ForcePixel': 1, 'ForceVertex': 2}
LIGHT_TYPE = {'Spot': 0, 'Directional': 1, 'Point': 2, 'Area': 3, 'Rectangle': 3, 'Disc': 4}


class PlanGap(RuntimeError):
    pass


def enum_value(text):
    """'ShadowCastingMode.On=1' -> 'On'."""
    if isinstance(text, str) and '.' in text:
        return text.split('.', 1)[1].split('=')[0]
    return text


def row_major_to_memory(values):
    v = [F(x) for x in values]
    return [v[r * 4 + c] for c in range(4) for r in range(4)]


def _b64f(s):
    return [F(x) for x in np.frombuffer(base64.b64decode(s), np.float32)]


@dataclass
class PlannedDraw:
    pass_kind: str
    renderer: dict
    submesh: int
    material: dict
    shader: str
    pass_index: int
    light_mode: str
    keywords: tuple
    programs: dict = field(default_factory=dict)      # stage -> dxbc sha256
    builtin: dict = field(default_factory=dict)
    sort_data: int = 0
    notes: list = field(default_factory=list)

    def key(self):
        return {'pass': self.pass_kind, 'renderer': self.renderer['name'], 'mesh': self.renderer.get('mesh'),
                'submesh': self.submesh, 'material': self.material['name'], 'shader': self.shader,
                'pass_index': self.pass_index, 'vs': self.programs.get('vs'), 'ps': self.programs.get('ps')}


class SceneAdapter:
    """Scene export -> rule inputs for one camera."""

    def __init__(self, scene, frame=None, static=None, library=None):
        self.scene = scene
        self.frame = frame
        self.static = static
        self.lib = library
        self.materials = {m['id']: m for m in scene['materials']}
        self.frame_renderers = {r['id']: r for r in (frame or {}).get('renderers', [])}
        self.gaps = []

    # ------------------------------------------------------------- camera
    def camera(self, name):
        cams = [c for c in self.scene['cameras'] if c['name'] == name and c['active_in_hierarchy']
                and c['properties']['enabled']]
        if len(cams) != 1:
            raise PlanGap('expected one enabled camera named %r' % name)
        c = cams[0]
        p = c['properties']
        m = c.get('matrices') or {}
        proj = row_major_to_memory(m.get('projectionMatrix', p['projectionMatrix']))
        nj = row_major_to_memory(m.get('nonJitteredProjectionMatrix', p['nonJitteredProjectionMatrix']))
        view = row_major_to_memory(m.get('worldToCameraMatrix', p['worldToCameraMatrix']))
        c2w = row_major_to_memory(m.get('cameraToWorldMatrix', p['cameraToWorldMatrix']))
        cull = row_major_to_memory(p['cullingMatrix'])
        from . import unity_math as um
        native = c.get('_native') or {}
        if 'implicit_projection_flag_0x4ec' in native:
            # Camera+0x4ec: 1 after ResetProjectionMatrix (0x6690f0), 0 after SetProjectionMatrix (0x66bc84)
            implicit = int(native['implicit_projection_flag_0x4ec']) == 1
        else:
            implicit = um.to_bytes(proj) == um.to_bytes(um.set_perspective(p['fieldOfView'], p['aspect'],
                                                                          p['nearClipPlane'], p['farClipPlane']))
        rect = p['pixelRect']
        mode = enum_value(p['depthTextureMode'])
        flags = 0
        for part, bit in (('Depth', 1), ('DepthNormals', 2), ('MotionVectors', 4)):
            if part in str(p['depthTextureMode']).split('.', 1)[1].split('=')[0].replace(' ', '').split(','):
                flags |= bit
        cam = CameraRender({
            'name': name, 'implicit_projection': implicit, 'projection_matrix': proj,
            'non_jittered_projection_matrix': nj, 'world_to_camera_matrix': view,
            'render_into_texture': bool(p.get('forceIntoRenderTexture')) or p.get('targetTexture') is not None,
            'near_clip': p['nearClipPlane'], 'far_clip': p['farClipPlane'], 'field_of_view': p['fieldOfView'],
            'aspect': p['aspect'], 'orthographic': p['orthographic'], 'orthographic_size': p['orthographicSize'],
            'position': [c2w[12], c2w[13], c2w[14]],
            'target': {'pixel_width': int(rect[2]), 'pixel_height': int(rect[3])},
            'depth_texture_mode': flags, 'clear_flags': enum_value(p['clearFlags']),
            'culling_mask': int(p['cullingMask']) & 0xFFFFFFFF,
            'previous_view_projection': row_major_to_memory(p['previousViewProjectionMatrix']),
            'opaque_sort_mode': enum_value(p['opaqueSortMode']),
            'transparency_sort_mode': enum_value(p['transparencySortMode']),
            'layer_cull_distances': p['layerCullDistances'], 'layer_cull_spherical': p['layerCullSpherical'],
            'culling_matrix': cull})
        self.gaps.append({'gap': 'camera_transform_position', 'detail': 'position from cameraToWorldMatrix column 3'})
        self.gaps.append({'gap': 'camera_implicit_flags', 'detail': ('implicit projection from Camera+0x4ec (probe v5 '
                          'native field)' if 'implicit_projection_flag_0x4ec' in native else 'implicit projection '
                          'inferred by comparing the exported projection with SetPerspective') + '; culling matrix '
                          'taken as explicit (Camera.cullingMatrix export)'})
        return cam

    def rules_camera(self, cam):
        return {'world_to_camera': cam.m('world_to_camera_matrix'), 'projection': cam.m('projection_matrix'),
                'position': cam.get('position'), 'near': cam.f('near_clip'), 'far': cam.f('far_clip'),
                'culling_mask': cam.get('culling_mask'), 'layer_cull_distances': cam.get('layer_cull_distances'),
                'layer_cull_spherical': cam.get('layer_cull_spherical'), 'implicit_culling_matrix': False,
                'culling_matrix': cam.get('culling_matrix'), 'implicit_world_to_camera': True,
                'opaque_sort_mode': {'Default': 0, 'FrontToBack': 1, 'NoDistanceSort': 2}[cam.get('opaque_sort_mode')],
                'implicit_transparency_sort': cam.get('transparency_sort_mode') == 'Default',
                'transparency_sort_mode': {'Default': 0, 'Perspective': 1, 'Orthographic': 2,
                                           'CustomAxis': 3}[cam.get('transparency_sort_mode')],
                'transparency_sort_axis': (0.0, 0.0, 1.0), 'orthographic': cam.get('orthographic')}

    # ------------------------------------------------------------- renderers
    def renderers(self):
        out = []
        for r in self.scene['renderers']:
            p = r['properties']
            if not (r['active_in_hierarchy'] and p.get('enabled')):
                continue
            kind = r['type'].split('.')[-1]
            fr = self.frame_renderers.get(int(r['id']))
            rr = (self.scene.get('_render_renderers') or {}).get(int(r['id']))
            if rr is not None and rr.get('bounds') is not None:
                b = [F(v) for v in rr['bounds']]
                bounds_src = 'render record bounds (OnPreRender of this render)'
            elif fr is not None and fr.get('bounds_f32'):
                b = _b64f(fr['bounds_f32'])
                bounds_src = 'frame bounds_f32'
            else:
                b = [F(v) for v in p['bounds']]
                bounds_src = 'scene properties.bounds (decimal)'
            mats = []
            for i, ref in enumerate(p.get('sharedMaterials') or []):
                if ref is None and kind == 'ParticleSystemRenderer' and i == 1:
                    continue      # ParticleSystemRenderer slot 1 = trailMaterial; none -> no trail draw
                mats.append(self.materials.get(ref['ref_id']) if ref else None)
            out.append({'id': int(r['id']), 'name': r['name'], 'kind': kind, 'layer': int(r['layer']),
                        'mesh': (p.get('sharedMesh') or {}).get('ref_name'),
                        # STAND-IN (labelled): the managed sharedMesh instance id; the engine key is
                        # RenderNode+0x100 = Mesh+0x174 (not exported, v7 field request). Sorts whose
                        # outcome depends on it are reported as mesh_id_tiebreak where detected.
                        'mesh_id': (self.scene['_render_nodes'][int(r['id'])]
                                    if int(r['id']) in (self.scene.get('_render_nodes') or {})
                                    else (p.get('sharedMesh') or {}).get('ref_id')),
                        'mesh_id_source': ('RenderNode+0x100 (probe v7 render_nodes)'
                                           if int(r['id']) in (self.scene.get('_render_nodes') or {})
                                           else 'managed sharedMesh instance id (stand-in for Mesh+0x174)'),
                        'center': b[0:3], 'extent': b[3:6], 'bounds_source': bounds_src,
                        'shadow_casting_mode': SHADOW_CASTING[enum_value(p['shadowCastingMode'])],
                        'receive_shadows': bool(p['receiveShadows']),
                        'motion_vector_mode': enum_value(p.get('motionVectorGenerationMode')),
                        'sorting_layer_id': int(p.get('sortingLayerID', 0)), 'sorting_order': int(p.get('sortingOrder', 0)),
                        'lightmap_index': int(p.get('lightmapIndex', -1)),
                        'materials': mats, 'raw': r})
        return out

    def node_queue(self, cam):
        """Visible renderers in RenderNodeQueue order.

        Scene node index: ``Renderer+0x160`` is the renderer's index in the
        RendererScene node array -- ``AddRendererInternal`` (0x68ae10) appends a
        SceneNode and returns the previous count, and ``RemoveRenderer``
        (0x690f80) swap-removes, writing the moved node's new index back to its
        renderer (+0x38 back pointer, +0x128 -> +0x160, 0x691142).  Static nodes are
        the indices below the static count, dynamic ones above, so list 0 then
        list 1 in node order is ascending index.  ``cull`` keeps node order
        within a list; ``render_node_queue_order`` (BeginRenderQueueExtraction
        0x68bc00 ...) then appends main-thread-prepared nodes (SkinnedMeshRenderer
        has no PrepareRenderNodes job callback) after the job-prepared ones.
        Probe v5 exports the index as ``native.scene_handle_0x160``."""
        rens = self.renderers()
        handles = [r['raw'].get('native', {}).get('scene_handle_0x160') for r in rens]
        if any(h is None for h in handles):
            vis = self.visible(cam, rens)
            return vis
        if any(h < 0 for h in handles):
            raise PlanGap('enabled renderer without a scene node')
        rens = [r for _, r in sorted(zip(handles, rens), key=lambda x: x[0])]
        rc = self.rules_camera(cam)
        rr = [{'list': 1, 'center': r['center'], 'extent': r['extent'], 'layer': r['layer'],
               'flags': rules.shadow_casting_flags(r['shadow_casting_mode']), 'has_renderer': True} for r in rens]
        vis = rules.cull(rr, rc)
        flags = [{'job_prepared': r['kind'] != 'SkinnedMeshRenderer', 'produces_node': True} for r in rens]
        order = rules.render_node_queue_order(vis, flags, 1)
        self.gaps.append({'gap': 'static_scene_node_count', 'detail': 'static/dynamic split of the node array not '
                          'exported; it does not change the order (static then dynamic = ascending index); '
                          'job block ranges use one job (node count < 128 per job)'})
        return [rens[i] for i in order]

    def visible(self, cam, renderers):
        rc = self.rules_camera(cam)
        rr = []
        for r in renderers:
            rr.append({'list': 1, 'center': r['center'], 'extent': r['extent'], 'layer': r['layer'],
                       'flags': rules.shadow_casting_flags(r['shadow_casting_mode']), 'has_renderer': True})
        vis = rules.cull(rr, rc)
        self.gaps.append({'gap': 'scene_node_lists', 'detail': 'every renderer placed in the dynamic list (1) in '
                          'export order; static/intermediate lists and scene-node indices are not exported '
                          '(affects only the final RO-index tie-break)'})
        return [renderers[i] for i in vis[1]]

    # ------------------------------------------------------------- materials
    def shader_copies(self, material):
        """All installed copies of the material's shader.  Copies that differ
        (AIT/Clothes True, per clothes bundle) are resolved per draw by requiring
        the selected programs to agree across copies; the item's own bundle is
        not in the export (gap ``clothes_shader_bundle``)."""
        copies = self.lib.copies(material['shader'])
        if not copies:
            raise PlanGap('shader %s not in the installed library' % material['shader'])
        return copies


def select_programs(copies, light_mode_pass_index, keywords, gaps, label):
    """Selected DXBC per stage, required identical across installed copies."""
    result = None
    for sh in copies:
        shader, sub = sh.resolve(tier=None)
        pz = [x for x in sub.passes if x.index == light_mode_pass_index][0]
        progs = {}
        for st in pz.stages():
            sel = pz.select(st, keywords, tier=None)
            progs[st] = sel.dxbc_sha256 if sel else None
        if result is None:
            result = progs
        elif progs != result:
            gaps.append({'gap': 'clothes_shader_bundle', 'detail': '%s: installed copies select different '
                         'programs; item bundle not exported' % label})
            return {st: None for st in result}
    return result


def light_records(adapter, cam):
    """ActiveLights for the forward loop (FindActiveLights RVA 0x640e00 +
    AddDirectionalLights 0x640170): skips baked lights, camera-mask misses and
    intensity < 0.001 (``comiss 0.001, intensity; ja skip``).  Only directional
    lights are modelled; local lights raise (CullLocalLights not transcribed)."""
    out = []
    mask = int(cam.get('culling_mask'))
    for l in adapter.scene['lights']:
        p = l['properties']
        if not (l['active_in_hierarchy'] and p['enabled']):
            continue
        if not (int(p['cullingMask']) & mask):
            continue
        if F(0.001) > F(p['intensity']):
            continue
        t = LIGHT_TYPE[enum_value(p['type'])]
        if t != rules.LIGHT_DIRECTIONAL:
            raise PlanGap('local light %s: CullLocalLights not transcribed' % l['name'])
        out.append({'id': l['id'], 'name': l['name'], 'type': t, 'color': p['color'], 'intensity': p['intensity'],
                    'shadows': 0 if enum_value(p['shadows']) == 'None' else (1 if enum_value(p['shadows']) == 'Hard' else 2),
                    'render_mode': RENDER_MODE[enum_value(p['renderMode'])], 'culling_mask': int(p['cullingMask']),
                    'has_cookie': p.get('cookie') is not None, 'lightmap_mode_for_render': 0,
                    'luminance_intensity': rules.light_luminance_intensity(p['color'], p['intensity']), 'raw': l})
    ordered, has_main = rules.order_directional_lights(out)
    if len(out) > 1:
        adapter.gaps.append({'gap': 'light_manager_order', 'detail': 'LightManager registration order of '
                             'non-main directional lights not exported; export order used'})
    return {'lights': ordered, 'has_main_light': has_main, 'counts': (len(ordered), 0, 0, 0, 0)}


def renderer_pass_keywords(pass_light_mode, renderer, material, active, global_keywords, quality):
    """Requested keyword set of one draw: pass-context keywords | material keywords
    (shader_library rule ``requested_keywords``).  Pass-context forward keywords
    are the documented renderer keywords of shader_library §3 (DIRECTIONAL,
    LIGHTPROBE_SH, SHADOWS_SCREEN when the main light casts shadows, the object
    receives and the queue is opaque); not disassembled here (gap)."""
    kw = set(global_keywords) | set(material.get('shader_keywords') or [])
    if pass_light_mode == 'FORWARDBASE':
        kw |= {'DIRECTIONAL', 'LIGHTPROBE_SH'}
        main = active['lights'][0] if active['has_main_light'] else None
        if (main is not None and main['shadows'] and renderer['receive_shadows']
                and int(material['render_queue']) <= OPAQUE_MAX_QUEUE and quality.get('shadows') != 'Disable'):
            kw.add('SHADOWS_SCREEN')
    elif pass_light_mode == 'SHADOWCASTER':
        kw.add('SHADOWS_DEPTH')
    return frozenset(kw)


def render_objects(adapter, cam):
    """Camera render-node queue and RenderObjectData lists (CullResults ->
    ``extract_render_objects``): shared by the forward, motion-vector and
    depth passes."""
    rc = adapter.rules_camera(cam)
    rens = adapter.node_queue(cam)
    mode, axis = rules.resolve_transparency_sort(rc, {'transparency_sort_mode': 0, 'transparency_sort_axis': (0, 0, 1)})
    ctx = {'world_to_camera': cam.m('world_to_camera_matrix'), 'position': cam.get('position'),
           'sort_mode': mode, 'sort_axis': axis}
    nodes = []
    shaders = []
    for r in rens:
        mats = []
        sh_row = []
        for m in r['materials']:
            if m is None:
                raise PlanGap('null material slot on %s' % r['name'])
            copies = adapter.shader_copies(m)
            shader, sub = copies[0].resolve(tier=None)
            mats.append({'shader_id': m['shader_id'], 'material_instance_id': m['id'],
                         'enable_instancing': m['enable_instancing'], 'custom_render_queue': m['render_queue'],
                         'shader_render_queue': m['render_queue']})
            sh_row.append((shader, sub, copies))
        nodes.append({'center': r['center'], 'layering': rules.layering_key(0, r['sorting_order']),
                      'shadow_casting_mode': r['shadow_casting_mode'], 'mesh_id': r['mesh_id'] or 0,
                      'materials': mats, 'lightmap_static': 0xFFFF, 'lightmap_realtime': 0xFFFF})
        shaders.append(sh_row)
        if r['sorting_layer_id'] != 0:
            raise PlanGap('sorting layer value for id %d not exported' % r['sorting_layer_id'])
    opaque_ros, transparent_ros = rules.extract_render_objects(nodes, ctx)
    return {'rc': rc, 'renderers': rens, 'nodes': nodes, 'shaders': shaders, 'opaque_ros': opaque_ros,
            'transparent_ros': transparent_ros}


def forward_plan(adapter, cam, *, transparent, global_keywords, quality, has_hsr_on_gpu=False):
    """Sorted ForwardBase/ForwardAdd draws (DoForwardShaderRenderLoop RVA 0x6c8d60)."""
    ro = render_objects(adapter, cam)
    rc, rens, nodes, shaders = ro['rc'], ro['renderers'], ro['nodes'], ro['shaders']
    ros = ro['transparent_ros'] if transparent else ro['opaque_ros']
    active = light_records(adapter, cam)
    entries = []
    per_ro = []
    for ri, ro in enumerate(ros):
        node = nodes[ro['node']]
        r = rens[ro['node']]
        shader, sub, _copies = shaders[ro['node']][ro['material_index']]
        passes = []
        for pz in sub.passes:
            lm = pz.light_mode
            passes.append({'type': PASS_TYPE.get(lm, 99), 'needs_render_option': 0, 'disabled': False,
                           'is_grab': pz.type == 'grab', 'pass': pz})
        culled, offscreen = rules.cull_lights_for_object(active, {'layer': r['layer'], 'center': r['center'],
                                                                  'extent': r['extent']})
        block = rules.classify_lights({'culled': culled, 'has_offscreen': offscreen, 'layer': r['layer'],
                                       'center': r['center'], 'extent': r['extent'],
                                       'flags': {'lightmapped': False, 'vertex_lit_subshader': False,
                                                 'coefficient_type': 1, 'no_add_pass': False, 'no_lights': False},
                                       'ambient_sh': [0.0] * 27}, active,
                                      {'pixel_light_count': quality['pixel_light_count'], 'rendering_path': 1})
        es = rules.forward_pass_entries(ri, passes, vertex_lit_subshader=False, lightmapped=False,
                                        add_light_count=len(block['add']), render_options=0, multi_pass=False)
        per_ro.append((block, passes))
        entries.extend(es)
    ordered = rules.sort_draws(entries, rc, ros, nodes, opaque=not transparent,
                               has_hidden_surface_removal_on_gpu=has_hsr_on_gpu)
    draws = []
    for e in ordered:
        ro = ros[e['ro']]
        r = rens[ro['node']]
        m = r['materials'][ro['material_index']]
        block, passes = per_ro[e['ro']]
        pz = passes[e['data'] & 0xFF]['pass']
        shader, _, copies = shaders[ro['node']][ro['material_index']]
        kw = renderer_pass_keywords(pz.light_mode, r, m, active, global_keywords, quality)
        progs = select_programs(copies, pz.index, kw, adapter.gaps, r['name'])
        draws.append(PlannedDraw('transparent' if transparent else 'opaque', r, ro['material_index'], m,
                                 shader.name, pz.index, pz.light_mode, tuple(sorted(kw)), progs,
                                 sort_data=e['data']))
    return draws, active


def draw_cbuffers(draw, pass_values, layouts, pool, pack_cbuffer):
    """Exact cbuffer bytes for one draw with Unity's shared-buffer rule
    (GfxDeviceD3D11Base::BeforeDrawCall RVA 0x542770): for each Unity layout of
    the bound program, the produced variables are written into the (name, size)
    shadow copy of the per-frame pool and the whole copy is uploaded; bytes of
    variables the generator did not produce keep their previous content."""
    values = dict(pass_values)
    values.update(draw.builtin)
    out, missing = {}, {}
    for layout in layouts:
        key = (layout.name, layout.size)
        base = pool.get(key, bytes(layout.size))
        names = {v.name for v in layout.variables}
        have = {k: (v[1] if isinstance(v, tuple) else v) for k, v in values.items() if k in names}
        lacking = sorted(names - set(have))
        if lacking:
            missing[layout.name] = lacking
        data = pack_cbuffer(layout, have, base=base, allow_missing=True)
        pool[key] = data
        out[layout.slot] = {'name': layout.name, 'bytes': data}
    return out, missing
