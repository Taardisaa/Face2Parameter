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


# Built-in shader parameters (GfxDevice built-in state, not ShaderPropertySheet lookups):
# vector names .rdata 0x14b89c0 (121), matrix names 0x14b8988 (6), read from the installed
# UnityPlayer.dll.  Every other uniform is resolved by shaderprops::Get* (material sheet,
# then global sheet, then the default: Vector4f::zero 0x85bc57, float 0.0 0x85aeb7,
# Matrix4x4f::identity 0x85b3a7).
BUILTIN_VECTORS = ('unity_LightColor0', 'unity_LightColor1', 'unity_LightColor2', 'unity_LightColor3', 'unity_LightColor4', 'unity_LightColor5', 'unity_LightColor6', 'unity_LightColor7', 'unity_LightPosition0', 'unity_LightPosition1', 'unity_LightPosition2', 'unity_LightPosition3', 'unity_LightPosition4', 'unity_LightPosition5', 'unity_LightPosition6', 'unity_LightPosition7', 'unity_SpotDirection0', 'unity_SpotDirection1', 'unity_SpotDirection2', 'unity_SpotDirection3', 'unity_SpotDirection4', 'unity_SpotDirection5', 'unity_SpotDirection6', 'unity_SpotDirection7', 'unity_LightAtten0', 'unity_LightAtten1', 'unity_LightAtten2', 'unity_LightAtten3', 'unity_LightAtten4', 'unity_LightAtten5', 'unity_LightAtten6', 'unity_LightAtten7', 'unity_VertexLightParams', 'glstate_lightmodel_ambient', '_WorldSpaceLightPos0', '_LightColor0', '_WorldSpaceCameraPos', 'unity_WorldTransformParams', '_ProjectionParams', '_ScreenParams', '_ZBufferParams', 'unity_OrthoParams', '_LightPositionRange', '_LightProjectionParams', 'unity_Ambient', 'unity_LightmapFade', 'unity_OcclusionMaskSelector', 'unity_ProbesOcclusion', '_ShadowOffsets0', '_ShadowOffsets1', '_ShadowOffsets2', '_ShadowOffsets3', '_LightShadowData', 'unity_LightShadowBias', '_LightSplitsNear', '_LightSplitsFar', 'unity_ShadowSplitSpheres0', 'unity_ShadowSplitSpheres1', 'unity_ShadowSplitSpheres2', 'unity_ShadowSplitSpheres3', 'unity_ShadowSplitSqRadii', 'unity_ShadowCascadeScales', 'unity_ShadowFadeCenterAndType', 'unity_ShadowColor', 'unity_LODFade', 'unity_RenderingLayer', 'unity_LightmapST', 'unity_DynamicLightmapST', 'unity_SHAr', 'unity_SHAg', 'unity_SHAb', 'unity_SHBr', 'unity_SHBg', 'unity_SHBb', 'unity_SHC', '_Time', '_LastTime', '_SinTime', '_CosTime', '_PiTime', 'unity_DeltaTime', 'unity_4LightPosX0', 'unity_4LightPosY0', 'unity_4LightPosZ0', 'unity_4LightAtten0', 'unity_FogStart', 'unity_FogEnd', 'unity_FogDensity', 'unity_FogColor', 'unity_FogParams', 'unity_CameraWorldClipPlanes0', 'unity_CameraWorldClipPlanes1', 'unity_CameraWorldClipPlanes2', 'unity_CameraWorldClipPlanes3', 'unity_CameraWorldClipPlanes4', 'unity_CameraWorldClipPlanes5', 'unity_AmbientSky', 'unity_AmbientEquator', 'unity_AmbientGround', 'unity_IndirectSpecColor', 'unity_SpecCube0_HDR', 'unity_SpecCube0_BoxMax', 'unity_SpecCube0_BoxMin', 'unity_SpecCube0_ProbePosition', 'unity_SpecCube1_HDR', 'unity_SpecCube1_BoxMax', 'unity_SpecCube1_BoxMin', 'unity_SpecCube1_ProbePosition', 'unity_BillboardNormal', 'unity_BillboardTangent', 'unity_BillboardCameraParams', 'unity_ProbeVolumeMin', 'unity_ProbeVolumeSizeInv', 'unity_ProbeVolumeParams', 'unity_StereoEyeIndex', 'unity_LightIndicesOffsetAndCount', 'unity_4LightIndices0', 'unity_4LightIndices1', 'unity_ReflectionProbeIndicesOffsetAndCount', 'unity_MotionVectorsParams', 'unity_HalfStereoSeparation')
BUILTIN_MATRICES = ('unity_ObjectToWorld', 'unity_WorldToObject', 'unity_MatrixV', 'unity_MatrixInvV', 'glstate_matrix_projection', 'unity_MatrixVP')



def material_values(material, shader, textures):
    """Material property values as GPU encodings (typed manifest properties only).

    Colours are converted with GammaToLinearSpace (RVA 0x27c970) in Linear
    colour space unless the property carries the Gamma flag; ``<tex>_ST`` is
    (scale, offset); ``<tex>_TexelSize`` is (1/w, 1/h, w, h) of the bound
    texture.  Missing typed properties -> the variable is not generated."""
    if '_native_values' in material:
        # probe v5: the material ShaderPropertySheet's stored bytes are the GPU values
        return dict(material['_native_values']), None
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


def renderer_matrices(renderers, win, emu, frame_count, render_renderers=None):
    """unity_ObjectToWorld / unity_WorldToObject per renderer id.

    With per-render renderer state (probe v5 render record): a SkinnedMeshRenderer's
    matrix is TRS(rootBone.position, rootBone.rotation, 1) as of that render
    (paired_skin.unity_object_matrices rule; the bridge's pose freeze moves the bones
    between the frame export and the renders), WorldToObject its General3D inverse."""
    from src.render import paired_skin as ps
    from src.render import gpu_skin_inputs as gsi
    from src.render.unity_skin_emu import invert_general3d
    from . import unity_math as um
    fr = win.by_frame[frame_count]
    static = {r['id']: r for r in win.static['renderers']}
    rr_all = render_renderers or {}
    out = {}
    for r in renderers:
        rr = rr_all.get(r['id'])
        if rr is not None and rr.get('root_bone') and r['kind'] == 'SkinnedMeshRenderer':
            rb = rr['root_bone']
            o2w = np.eye(4, dtype=F)
            o2w[:3, :3] = gsi.unity_rotation_matrix(np.asarray(rb['rotation'], F))
            o2w[:3, 3] = np.asarray(rb['position'], F)
            o, w = gsi.matrix_bytes(o2w), gsi.matrix_bytes(invert_general3d(emu, o2w))
            out[r['id']] = {'unity_ObjectToWorld': um.from_bytes(o), 'unity_WorldToObject': um.from_bytes(w),
                            '_source': 'render record rootBone TRS (paired_skin.unity_object_matrices rule)'}
        elif r['id'] in static:
            o, w = ps.unity_object_matrices(emu, win, fr, static[r['id']])
            out[r['id']] = {'unity_ObjectToWorld': um.from_bytes(o), 'unity_WorldToObject': um.from_bytes(w),
                            '_source': 'paired_skin.unity_object_matrices'}
        elif rr is not None and rr.get('local_to_world_rowmajor'):
            o2w = up.row_major_to_memory(rr['local_to_world_rowmajor'])
            ok, w2o = um.invert_general_3d(o2w)
            out[r['id']] = {'unity_ObjectToWorld': o2w, 'unity_WorldToObject': w2o,
                            '_source': 'render record localToWorld (float32 round-trip decimal)'}
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
        self.mats = renderer_matrices(adapter.renderers(), window, emulator, scene['frame_count'],
                                      scene.get('_render_renderers'))
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
            if 'world_rotation_xyzw' in lp:
                # SharedLightData::UpdateTransform (0x6a6610) stores
                # Transform::GetLocalToWorldMatrixNoScale at +0x140; SetupForwardAddLight /
                # the base-light path (0x680be5) writes (-m[8], -m[9], -m[10], 0) of it
                # (column 2 at +0x160).  Rotation part = the native quaternion->matrix
                # expression (gpu_skin_inputs.unity_rotation_matrix, bit-exact on the
                # skinned ObjectToWorld) of the world rotation.
                from src.render import gpu_skin_inputs as gsi
                R = gsi.unity_rotation_matrix(np.asarray(lp['world_rotation_xyzw'], np.float32))
                self.light['_WorldSpaceLightPos0'] = [-R[0, 2], -R[1, 2], -R[2, 2], F(0.0)]
            ch = int(lp['bakingOutput']['LightBakingOutput.occlusionMaskChannel'])
            sel = [F(0.0)] * 4
            sel[ch if ch != -1 else 0] = F(1.0)
            self.light['unity_OcclusionMaskSelector'] = sel
        # No ReflectionProbe is exported: every renderer's per-object list is empty and it
        # receives the context default sample.
        self.global_sheet = None
        if '_v5' in scene:
            from .scene_v5 import sheet_values
            self.global_sheet = sheet_values(scene['_v5']['render_record']['data']['global_sheet'])
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
        st = (self.scene.get('native') or {}).get('settings') or {}
        if 'build_settings_0xfa' in st:
            # probe v6: BuildSettings+0xfa; the MV loop sets -0.001 when it is set (0x6cc539)
            if not int(st['build_settings_0xfa']):
                raise up.PlanGap('_MotionVectorDepthBias branch for BuildSettings+0xfa == 0 not transcribed')
            out['_MotionVectorDepthBias'] = [F(-0.001)]
        rr = (self.scene.get('_render_renderers') or {}).get(renderer['id'])
        nat = (rr or {}).get('native') or {}
        if 'skin_buffer_holds_frame' in nat and 'smr_has_last_position_data_0x28d' in nat:
            # probe v6.1 (field base corrected, confirmed natively): SkinnedMeshRenderer+0x255 is
            # copied into RenderNode+0xe8 bit 9 by SkinnedMeshRenderer::AddAsRenderNode
            # (0x77ae22..0x77ae37); the MV loop sets _HasLastPositionData = bit 9 ? 1.0 : 0.0
            # (0x6cc4e7..0x6cc508, xmm6 = 1.0 from 0x6cc453)
            out['_HasLastPositionData'] = [F(1.0) if int(nat['smr_has_last_position_data_0x28d']) else F(0.0)]
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
            # Every ParticleSystemRenderer render callback (single/multiple x 3 modes,
            # 0xb02e30 / 0xb03530 / 0xb03c10 / 0xb042e0 / 0xb04780 / 0xb04c30) starts with
            # GfxDevice::SetWorldMatrixAndType(Matrix4x4f::identity, 0) (vtable +0xd8 =
            # 0x8cf3b0 -> SetWorldMatrix +0xe0); particle vertices are generated in world
            # space (PrepareForRender 0xb0b72e passes the simulation-space matrix to
            # ParticleSystemTempData::SetMatrices).  WorldToObject = General3D inverse.
            from . import unity_math as um
            ident = um.identity()
            ok_, inv = um.invert_general_3d(ident)
            vals['unity_ObjectToWorld'] = ident
            vals['unity_WorldToObject'] = inv
            notes.append('particle draw matrix: SetWorldMatrixAndType(identity) (0xb02e71)')
        ish = renderer['raw'].get('interpolated_sh')
        if ish:
            vals.update(uf.sh_shader_constants([F(x) for x in ish]))
        sh = self.lib.copies(material['shader'])[0]
        mv, why = material_values(material, sh, self.textures)
        vals.update(mv)
        vals['_inputs'] = mv.get('_inputs', {})
        if self.global_sheet is not None:
            vals['_property_fallback'] = {'global': self.global_sheet, 'builtin': BUILTIN_SET}
        if why:
            notes.append(why)
        return vals, notes


# Per-object values the motion-vector loop sets with ShaderPropertySheet::SetFloat on its
# own sheet (DoMotionVectorRenderLoop job 0x6cc4e7..0x6cc557); never a material/global lookup.
MOTION_VECTOR_SHEET = frozenset(('_HasLastPositionData', '_ForceNoMotion', '_MotionVectorDepthBias'))
BUILTIN_SET = frozenset(BUILTIN_VECTORS) | frozenset(BUILTIN_MATRICES) | MOTION_VECTOR_SHEET
