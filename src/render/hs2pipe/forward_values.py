"""Constant-buffer values of Unity 2018.4.11f1 built-in forward draws from scene state.

One place that turns the frame-pair probe export (``scene_NNN.json``,
``frame_NNN.json``, ``static.json``) into the uniform values a forward-family
draw uploads, for both the structural diff and the frame orchestrator:

* camera values: ``unity_forward.camera_setup`` / ``camera_shader_props``
  (``Camera::SetCameraShaderProps`` RVA 0x66ab30);
* main directional light: ``_LightColor0`` (PrecalcLightColor 0x69eab0),
  ``_LightShadowData`` (0x679090 chain), ``unity_ShadowFadeCenterAndType``
  (CloseFit), ``unity_OcclusionMaskSelector``
  (SetupObjectBakedOcclusionInForward 0x6a5240);
* per draw: ObjectToWorld / WorldToObject (skinned renderers through the
  sibling ``paired_skin.unity_object_matrices``; mesh renderers from the
  exported ``localToWorldMatrix`` and InvertMatrix4x4_General3D),
  ``unity_WorldTransformParams`` (SetWorldMatrixAndType 0x8cf3b0), SH
  (interpolated probe export), reflection sample (GetReflectionSample 0x6512d0
  -> GenerateDefaultSample 0x67d5d0 -> SetReflections 0x6c11e0), probe volume;
* material values: typed manifest properties (GammaToLinearSpace for colours
  without the Gamma flag; ``_ST``; ``_TexelSize``).

Values the export does not carry are not produced here; callers label them
``external:capture`` (``EXTERNAL_CAPTURE``) or ``not_generated``.
"""
from __future__ import annotations

import numpy as np

from . import unity_forward as uf
from . import unity_forward_plan as up

F = np.float32
PROPERTY_FLAG_GAMMA = 32          # UnityEngine.Rendering.ShaderPropertyFlags.Gamma
GRAPHICS_D3D11 = {'reversed_z': True, 'uses_opengl_texture_coords': False}

EXTERNAL_CAPTURE = {
    '_WorldSpaceLightPos0': 'light localToWorldMatrix not exported (field request)',
    'unity_WorldToShadow': 'needs light localToWorldMatrix and shadow-caster cull set',
    'unity_LightShadowBias': 'needs the cascade texel size (light matrix)',
}


def material_values(material, shader, textures):
    """Material property values as GPU encodings (typed manifest properties only).

    Colours are converted with GammaToLinearSpace (RVA 0x27c970) in Linear
    colour space unless the property carries the Gamma flag; ``<tex>_ST`` is
    (scale, offset); ``<tex>_TexelSize`` is (1/w, 1/h, w, h) of the bound
    texture.  Missing typed properties -> the variable is not generated."""
    typed = material.get('typed_properties_from_manifest')
    if not typed:
        return {}, 'no typed properties (shader not in the probe manifest)'
    flags = {p['name']: p['flags'] for p in shader.properties}
    out = {}
    for name, v in typed.items():
        if not v.get('has'):
            continue
        t, val = v['type'], v['value']
        if t == 'color':
            c = [F(x) for x in val]
            out[name] = c if flags.get(name, 0) & PROPERTY_FLAG_GAMMA else uf.gamma_to_linear_color(c)
            if not flags.get(name, 0) & PROPERTY_FLAG_GAMMA:
                out.setdefault('_inputs', {})[name] = c
        elif t in ('float', 'range'):
            out[name] = [F(val)]
        elif t == 'int':
            out[name] = [F(val)]
        elif t == 'vector':
            out[name] = [F(x) for x in val]
        elif t == 'texture':
            tp = (material.get('texture_properties') or {}).get(name) or {}
            s, o = tp.get('scale', [1, 1]), tp.get('offset', [0, 0])
            out[name + '_ST'] = [F(s[0]), F(s[1]), F(o[0]), F(o[1])]
            ref = (val or {}).get('ref_id')
            tex = textures.get(ref)
            if tex is not None:
                w, h = F(tex['properties']['width']), F(tex['properties']['height'])
                out[name + '_TexelSize'] = [F(1.0) / w, F(1.0) / h, w, h]
    return out, None


def renderer_matrices(renderers, win, emu, frame_count):
    """unity_ObjectToWorld / unity_WorldToObject per renderer id."""
    from src.render import paired_skin as ps
    from . import unity_math as um
    fr = win.by_frame[frame_count]
    static = {r['id']: r for r in win.static['renderers']}
    out = {}
    for r in renderers:
        if r['id'] in static:
            o, w = ps.unity_object_matrices(emu, win, fr, static[r['id']])
            out[r['id']] = {'unity_ObjectToWorld': um.from_bytes(o), 'unity_WorldToObject': um.from_bytes(w),
                            '_source': 'paired_skin.unity_object_matrices'}
        else:
            o2w = up.row_major_to_memory(r['raw']['properties']['localToWorldMatrix'])
            ok, w2o = um.invert_general_3d(o2w)
            out[r['id']] = {'unity_ObjectToWorld': o2w, 'unity_WorldToObject': w2o,
                            '_source': 'Renderer.localToWorldMatrix export (decimal)'}
    return out


class ForwardValues:
    """Per-camera and per-draw built-in + material values of one exported frame."""

    def __init__(self, scene, adapter, library, window, emulator, cam, active, prev_scene=None):
        self.scene = scene
        self.prev_scene = prev_scene
        self.window = window
        self.emu = emulator
        self.adapter = adapter
        self.lib = library
        self.cam = cam
        self.active = active
        self.setup = uf.camera_setup(cam, GRAPHICS_D3D11)
        self.camera = uf.camera_shader_props(cam, self.setup, GRAPHICS_D3D11)
        self.mats = renderer_matrices(adapter.renderers(), window, emulator, scene['frame_count'])
        self.textures = {t['id']: t for t in scene['textures']}
        st = scene['statics']
        q = st['UnityEngine.QualitySettings']
        rs = st['UnityEngine.RenderSettings']
        self.linear = str(q.get('activeColorSpace', '')).endswith('=1')
        self.light = {}
        main = active['lights'][0] if active['has_main_light'] else None
        if main is not None:
            lp = main['raw']['properties']
            gs = st['UnityEngine.Rendering.GraphicsSettings']
            self.light['_LightColor0'] = uf.light_final_color({'color': lp['color'], 'intensity': lp['intensity']},
                                                              gs['lightsUseLinearIntensity'], gs['lightsUseColorTemperature'])
            sd = uf.shadow_distance_for_camera({'shadow_distance': q['shadowDistance']}, cam)
            self.light['_LightShadowData'] = uf.light_shadow_data({'shadow_strength': lp['shadowStrength']}, sd,
                                                                  cam.f('far_clip'))
            if q['shadowProjection'].startswith('ShadowProjection.CloseFit'):
                self.light['unity_ShadowFadeCenterAndType'] = [F(0.0)] * 4
            ch = int(lp['bakingOutput']['LightBakingOutput.occlusionMaskChannel'])
            sel = [F(0.0)] * 4
            sel[ch if ch != -1 else 0] = F(1.0)
            self.light['unity_OcclusionMaskSelector'] = sel
        # No ReflectionProbe is exported: every renderer's per-object list is empty and it
        # receives the context default sample.
        self.reflection = uf.reflection_probe_constants(
            uf.renderer_reflection_sample(rs, None, scene['reflection_probes'], self.linear))

    def previous_object_to_world(self, renderer):
        """The renderer matrix of frame N-1 (RenderNode+0x40, ``_PreviousM``):
        skinned head surfaces through ``paired_skin.unity_object_matrices`` on the
        previous window frame, other renderers from the previous scene export."""
        from src.render import paired_skin as ps
        from . import unity_math as um
        static = {r['id']: r for r in self.window.static['renderers']}
        prev = self.scene['frame_count'] - 1
        if renderer['id'] in static:
            if prev not in self.window.by_frame:
                return None, 'previous frame not in the probe window'
            o, _ = ps.unity_object_matrices(self.emu, self.window, self.window.by_frame[prev], static[renderer['id']])
            return um.from_bytes(o), 'paired_skin.unity_object_matrices (frame N-1)'
        if self.prev_scene is None or self.prev_scene['frame_count'] != prev:
            return None, 'previous scene export missing'
        rows = [r for r in self.prev_scene['renderers'] if int(r['id']) == renderer['id']]
        if len(rows) != 1:
            return None, 'renderer not in the previous scene export'
        return up.row_major_to_memory(rows[0]['properties']['localToWorldMatrix']),             'Renderer.localToWorldMatrix export of frame N-1 (decimal)'

    def motion_vector_values(self, renderer):
        """Object motion-vector pass values: ``_NonJitteredVP`` / ``_PreviousVP``
        (SetupMotionMatrices 0x6d0100), ``_PreviousM``, ``_ForceNoMotion`` = 1 when
        MotionVectorGenerationMode is ForceNoMotion (RenderNode+0xe8 & 0xc00 ==
        0x800, 0x6cc50d).  ``_HasLastPositionData`` (SkinnedMeshRenderer+0x255,
        0x6cc4e7) and ``_MotionVectorDepthBias`` (-0.001 when BuildSettings+0xfa,
        0x6cc539) are not produced: their inputs are not exported."""
        out = {'_NonJitteredVP': list(self.setup.non_jittered_vp),
               '_PreviousVP': self.cam.m('previous_view_projection'),
               '_ForceNoMotion': [F(1.0) if renderer['motion_vector_mode'] == 'ForceNoMotion' else F(0.0)]}
        pm, src = self.previous_object_to_world(renderer)
        if pm is not None:
            out['_PreviousM'] = pm
        return out, src

    def draw_values(self, renderer, material):
        """(values, notes) of one forward-family draw."""
        vals = dict(self.camera)
        vals.update(self.light)
        vals.update(uf.probe_volume_disabled())
        vals.update(self.reflection)
        notes = []
        m = self.mats.get(renderer['id'])
        if m is not None and renderer['kind'] != 'ParticleSystemRenderer':
            vals.update({k: v for k, v in m.items() if not k.startswith('_')})
            notes.append(m['_source'])
            o = m['unity_ObjectToWorld']
            det = (o[0] * (o[5] * o[10] - o[9] * o[6]) - o[4] * (o[1] * o[10] - o[9] * o[2])
                   + o[8] * (o[1] * o[6] - o[5] * o[2]))
            vals['unity_WorldTransformParams'] = [F(0.0), F(0.0), F(0.0), F(-1.0) if det < 0 else F(1.0)]
        elif renderer['kind'] == 'ParticleSystemRenderer':
            notes.append('particle_render_matrix: ParticleSystemRenderer draw matrix rule not transcribed')
        ish = renderer['raw'].get('interpolated_sh')
        if ish:
            vals.update(uf.sh_shader_constants([F(x) for x in ish]))
        sh = self.lib.copies(material['shader'])[0]
        mv, why = material_values(material, sh, self.textures)
        vals.update(mv)
        vals['_inputs'] = mv.get('_inputs', {})
        if why:
            notes.append(why)
        return vals, notes
