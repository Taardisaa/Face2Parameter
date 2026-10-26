"""Unity 2018.4.11f1 built-in forward pipeline generator (per camera render).

Produces, from ``scene_state.SceneState``, the values Unity writes into the
built-in constant buffers for one camera render and the ordered pass plan of
that render.  Every rule cites the UnityPlayer.dll routine it is transcribed
from (RVA, installed DLL SHA-256 ``5034cdc2…``; see ``unity_math`` for the
float32 conventions) or the Unity built-in shader source
(``builtin_shaders-2018.4.11f1``).  Rules taken from capture observation are
labelled ``capture_observed``.  Nothing is fitted.

Value objects are float32 lists; ``pack`` turns a variable into its exact
cbuffer bytes.  Which variables a draw actually receives is decided by the
bound program's Unity cbuffer layout (the installed export's
``m_ConstantBuffers``), exactly like Unity: ``GfxDeviceD3D11Base::BeforeDrawCall``
(RVA 0x542770) writes each built-in matrix only into the constant buffers of
the bound stages whose layout contains it, compares with ``memcmp`` and marks
the buffer dirty; dirty buffers are uploaded whole with ``Map(WRITE_DISCARD)``.
Bytes outside the bound layout keep the shared shadow copy's previous content
and are never read by that program (``CBufferShadowCopies`` models them).
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

from . import unity_math as um
from .scene_state import SceneStateError, f32_list, f32_value

F = np.float32
GENERATOR_VERSION = 1

# Unity CameraEvent / pass order of the forward path (Unity 2018.4 manual
# "Rendering order"; confirmed per pass label in the frame graph of the capture).
FORWARD_PASS_ORDER = (
    'depth_texture',           # DepthTextureMode.Depth: ShadowCaster passes into "Camera DepthTexture"
    'depth_normals',           # DepthTextureMode.DepthNormals: Internal-DepthNormalsTexture replacement
    'shadow_map',              # per shadowed directional light: ShadowCaster into "Shadowmap"
    'screenspace_shadows',     # Internal-ScreenSpaceShadows collector into "Screenspace ShadowMap"
    'opaque',                  # ForwardBase (+ForwardAdd) for queue <= 2500
    'skybox',
    'motion_vectors',          # DepthTextureMode.MotionVectors: camera pass 1 + object pass 0
    'transparent',             # queue > 2500 back to front
)


class GeneratorGap(RuntimeError):
    """A rule the generator has not recovered; never silently substituted."""


def pack(values):
    return struct.pack('<%df' % len(values), *[float(v) for v in values])


def pack_int(values):
    return struct.pack('<%di' % len(values), *[int(v) for v in values])


# --------------------------------------------------------------- utilities
def gamma_to_linear_space(v):
    """``GammaToLinearSpace(float)`` (RVA 0x27cae0); the same arithmetic is used
    per channel by GammaToLinearSpace(Color) 0x27c970 and the material colour
    conversion 0x83cde0.  ``powf`` is the installed CRT routine executed by
    ``unity_crt`` (not correctly rounded: 1-ulp differences in the capture)."""
    from .unity_crt import powf
    v = F(v)
    if F(0.04045) >= v:
        return v / F(12.92)
    if F(1.0) > v:
        return powf((v + F(0.055)) / F(1.055), F(2.4))
    if v == F(1.0):
        return F(1.0)
    return powf(v, F(2.2))


def sh_add_weighted(dst, src, w):
    """``SphericalHarmonicsL2::AddWeighted`` (RVA 0x7d3f90): dst[i] = src[i]*w + dst[i]."""
    w = F(w)
    return [src[i] * w + dst[i] for i in range(27)]


def final_ambient_probe(rs, color_space_linear=True):
    """``RenderSettings::UpdateFinalAmbientProbe`` (RVA 0x6a5e10), Skybox branch:
    mode 0 (Skybox) with a skybox material -> zero + ambientProbe*GammaToLinearSpace(intensity).
    Other ambient modes (Trilight/Flat via AddAmbientLight/AddDirectionalLight)
    are not transcribed yet (gap ``ambient_trilight_flat``)."""
    mode = rs['ambient_mode']
    raw = f32_list(rs['ambient_probe'], 27)
    if mode == 'Skybox' and rs.get('has_skybox_material'):
        w = gamma_to_linear_space(f32_value(rs['ambient_intensity']))
        return sh_add_weighted([F(0.0)] * 27, raw, w)
    if mode == 'Custom':
        return raw
    raise GeneratorGap('ambient mode %s (UpdateFinalAmbientProbe non-skybox branches) not transcribed' % mode)


def sh_shader_constants(sh):
    """``SphericalHarmonicsL2::GetShaderConstantsFromNormalizedSH`` (RVA 0x7d7fd0).
    sh is [r0..r8, g0..g8, b0..b8]; returns unity_SHAr..unity_SHC."""
    r, g, b = sh[0:9], sh[9:18], sh[18:27]
    three = F(3.0)
    return {
        'unity_SHAr': [r[3], r[1], r[2], r[0] - r[6]],
        'unity_SHAg': [g[3], g[1], g[2], g[0] - g[6]],
        'unity_SHAb': [b[3], b[1], b[2], b[0] - b[6]],
        'unity_SHBr': [r[4], r[5], r[6] * three, r[7]],
        'unity_SHBg': [g[4], g[5], g[6] * three, g[7]],
        'unity_SHBb': [b[4], b[5], b[6] * three, b[7]],
        'unity_SHC': [r[8], g[8], b[8], F(1.0)],
    }


# ------------------------------------------------------------ camera setup
@dataclass
class CameraSetup:
    projection_gl: list            # Camera.projectionMatrix (jittered when CTAA set it)
    non_jittered_gl: list
    device_projection: list        # glstate_matrix_projection
    view: list                     # unity_MatrixV
    view_projection: list          # unity_MatrixVP
    inverse_view: list             # unity_MatrixInvV
    world_to_camera: list          # unity_WorldToCamera
    camera_to_world: list          # unity_CameraToWorld
    camera_inverse_projection: list
    non_jittered_vp: list
    invert_y: bool
    notes: list = field(default_factory=list)


def implicit_projection(cam):
    """``Camera::GetImplicitProjectionMatrix`` (RVA 0x663420)."""
    if cam.get('orthographic'):
        size = cam.f('orthographic_size')
        aspect = cam.f('aspect')
        w = size * aspect
        return um.set_ortho(-w, w, -size, size, cam.f('near_clip'), cam.f('far_clip'))
    return um.set_perspective(cam.f('field_of_view'), cam.f('aspect'), cam.f('near_clip'), cam.f('far_clip'))


def camera_setup(cam, graphics, jitter_hook=None):
    """Camera matrices for one render.

    * Projection: implicit SetPerspective/SetOrtho unless a script set
      ``Camera.projectionMatrix`` (then the exported matrix is used verbatim).
      ``jitter_hook(cam, non_jittered_gl) -> jittered_gl`` is CTAA's interface;
      it is applied only when the exported projection is not provided.
    * Device projection: ``GfxDevice::SetProjectionMatrix`` (RVA 0x8cef30) ->
      ``CalculateDeviceProjectionMatrix`` (0x8ca400) with invertY = device
      invertProjection flag (render into texture on D3D11).
    * VP: ``GfxDevice::SetViewMatrix`` (0x8cf290): VP = devProj·view
      (MultiplyMatrices4x4), InvV = InvertMatrix4x4_General3D(view).
    * unity_WorldToCamera = Scale(1,1,-1)·view, unity_CameraToWorld = its
      General3D inverse, unity_CameraInvProjection = InvertMatrix4x4_Full(proj):
      ``Camera::SetCameraShaderProps`` (0x66ab30).
    """
    reversed_z = bool(graphics['reversed_z'])
    gl_tc = bool(graphics['uses_opengl_texture_coords'])
    invert_y = (not gl_tc) and bool(cam.get('render_into_texture'))
    notes = []
    non_jittered = cam.m('non_jittered_projection_matrix') if 'non_jittered_projection_matrix' in cam.raw else None
    if cam.get('implicit_projection'):
        proj = implicit_projection(cam)
        if non_jittered is None:
            non_jittered = proj
    elif 'projection_matrix' in cam.raw:
        proj = cam.m('projection_matrix')
        notes.append('explicit Camera.projectionMatrix used verbatim')
    elif jitter_hook is not None:
        base = implicit_projection(cam)
        proj = [F(v) for v in jitter_hook(cam, base)]
        non_jittered = base if non_jittered is None else non_jittered
        notes.append('projection = jitter_hook(implicit projection)')
    else:
        raise SceneStateError('explicit projection without projection_matrix or jitter hook')
    if non_jittered is None:
        raise SceneStateError('non_jittered_projection_matrix required for an explicit projection')
    view = cam.m('world_to_camera_matrix')
    dev = um.calculate_device_projection(proj, gl_tc, invert_y, reversed_z)
    vp = um.multiply(dev, view)
    ok, inv_view = um.invert_general_3d(view)
    w2c = um.multiply(um.set_scale(1.0, 1.0, -1.0), view)
    ok2, c2w = um.invert_general_3d(w2c)
    ok3, inv_proj = um.invert_full(proj)
    dev_nj = um.calculate_device_projection(non_jittered, gl_tc, invert_y, reversed_z)
    nj_vp = um.multiply(dev_nj, view)
    if not (ok and ok2):
        notes.append('singular view matrix: Unity writes zero inverse')
    if not ok3:
        notes.append('singular projection: Unity writes zero inverse projection')
    return CameraSetup(proj, non_jittered, dev, view, vp, inv_view, w2c, c2w, inv_proj, nj_vp, invert_y, notes)


def adjust_depth_range(m, near, far):
    """``Matrix4x4f::AdjustDepthRange`` (RVA 0x7cb800)."""
    m = list(m)
    n, f = F(near), F(far)
    d = f - n
    s = n + f
    perspective = not (m[3] == F(0.0) and m[7] == F(0.0) and m[11] == F(0.0) and m[15] == F(1.0))
    if perspective:
        m[10] = -(s / d)
        m[14] = ((f * F(-2.0)) * n) / d
    else:
        m[10] = F(-2.0) / d
        m[14] = -(s / d)
    return m


def skybox_matrices(cam, setup, graphics, caps_404=1.0):
    """``Skybox::RenderSkybox`` (RVA 0x643680, non-stereo path 0x6437bc..0x6439bd).

    O2W = Scale(far*10) at the camera position; projection =
    ``Camera::GetSkyboxProjectionMatrix(near*0.01)`` (0x664040: the camera's
    current projection then AdjustDepthRange(near*0.01, far)), then
    m22 = eps-1, m32 = -1, m23 = (eps-2)*near*0.01 with eps = GraphicsCaps+0x404 * 1e-6.
    The meaning of GraphicsCaps+0x404 is not identified; 1.0 is the
    capture_observed value (gap ``skybox_caps_404``).  VP = devProj(proj) * view."""
    far = cam.f('far_clip')
    near = cam.f('near_clip') * F(0.01)
    # Camera::GetPosition (0x3a09f0, called at 0x64378b / 0x6437fd) = Transform::GetPosition
    pos = f32_list(cam.get('transform_position') if 'transform_position' in cam.raw else cam.get('position'), 3)
    s = far * F(10.0)
    o2w = um.set_scale(s, s, s)
    o2w[12], o2w[13], o2w[14] = pos[0], pos[1], pos[2]
    proj = adjust_depth_range(setup.projection_gl, near, far)
    eps = F(caps_404) * F(1e-6)
    proj[10] = eps - F(1.0)
    proj[11] = F(-1.0)
    proj[14] = (eps - F(2.0)) * near
    dev = um.calculate_device_projection(proj, bool(graphics['uses_opengl_texture_coords']), setup.invert_y,
                                         bool(graphics['reversed_z']))
    return {'unity_ObjectToWorld': o2w, 'projection': proj, 'unity_MatrixVP': um.multiply(dev, setup.view)}


def projection_near_far(cam, proj):
    """Near/far as ``SetCameraShaderProps`` derives them (RVA 0x66ab30).

    Implicit projection (Camera+0x4ec == 1): the clip planes.  Otherwise, if
    any off-diagonal element of the upper 3x3 is non-zero the clip planes are
    used; else near = -(row3+row2).w/|(row3+row2).xyz| and far = (row3-row2).w/|...|.
    """
    if cam.get('implicit_projection'):
        return cam.f('near_clip'), cam.f('far_clip')
    off_diag = any(proj[c * 4 + r] != F(0.0) for r in range(3) for c in range(3) if r != c)
    if off_diag:
        return cam.f('near_clip'), cam.f('far_clip')
    a = proj[3] + proj[2]
    b = proj[7] + proj[6]
    c = proj[11] + proj[10]
    d = proj[15] + proj[14]
    near = -(d / F(math.sqrt(float((b * b + a * a) + c * c))))
    a = proj[3] - proj[2]
    b = proj[7] - proj[6]
    c = proj[11] - proj[10]
    d = proj[15] - proj[14]
    far = d / F(math.sqrt(float((b * b + a * a) + c * c)))
    return near, far


def camera_shader_props(cam, setup, graphics):
    """UnityPerCamera / UnityPerCameraRare / UnityPerFrame camera values
    (``Camera::SetCameraShaderProps`` RVA 0x66ab30, ``GfxDevice::SetViewMatrix``)."""
    out = {}
    pos = f32_list(cam.get('position'), 3)
    out['_WorldSpaceCameraPos'] = [pos[0], pos[1], pos[2]]       # float3 slot; w written 0
    near, far = projection_near_far(cam, setup.projection_gl)
    inv_near = F(1.0) if near == F(0.0) else F(1.0) / near
    inv_far = F(1.0) if far == F(0.0) else F(1.0) / far
    out['_ProjectionParams'] = [F(-1.0) if setup.invert_y else F(1.0), near, far, inv_far]
    tgt = cam.get('target')
    w, h = F(tgt['pixel_width']), F(tgt['pixel_height'])
    out['_ScreenParams'] = [w, h, F(1.0) / w + F(1.0), F(1.0) / h + F(1.0)]
    fn = inv_near * far
    d3 = float(fn)
    d2 = 1.0 - d3
    x = float(inv_far)
    zb = [F(d2), F(d3), F(x * d2), F(d3 * x)]
    if graphics['reversed_z']:
        zb = [-zb[0], zb[1] + zb[0], -zb[2], zb[3] + zb[2]]
    out['_ZBufferParams'] = zb
    proj = setup.projection_gl
    is_ortho = proj[3] == F(0.0) and proj[7] == F(0.0) and proj[11] == F(0.0) and proj[15] == F(1.0)
    size = cam.f('orthographic_size')
    out['unity_OrthoParams'] = [size * cam.f('aspect'), size, F(0.0), F(1.0) if is_ortho else F(0.0)]
    out['unity_CameraProjection'] = list(proj)
    out['unity_CameraInvProjection'] = list(setup.camera_inverse_projection)
    out['unity_WorldToCamera'] = list(setup.world_to_camera)
    out['unity_CameraToWorld'] = list(setup.camera_to_world)
    out['glstate_matrix_projection'] = list(setup.device_projection)
    out['unity_MatrixV'] = list(setup.view)
    out['unity_MatrixInvV'] = list(setup.inverse_view)
    out['unity_MatrixVP'] = list(setup.view_projection)
    out['unity_StereoEyeIndex'] = ('int', [0])
    return out


def time_shader_props(time):
    """``ShaderLab::UpdateGlobalShaderProperties`` (RVA 0x85d680).

    _Time = (t*0.05, t, t+t, t*3); _SinTime/_CosTime = sinf/cosf(t/8, t/4, t/2, t)
    (CRT trig: gap ``crt_trig``); unity_DeltaTime = (dt, 1/dt, sdt, 1/sdt) with
    dt clamped: 0.005 if 0.005 > dt else min(0.2, dt) (minss semantics)."""
    from .unity_crt import sinf as crt_sinf, cosf as crt_cosf
    t = f32_value(time['shader_time'])

    def clamp(v):
        v = F(v)
        if F(0.005) > v:
            return F(0.005)
        return F(0.2) if F(0.2) < v else v
    dt = clamp(f32_value(time['delta_time']))
    sdt = clamp(f32_value(time['smooth_delta_time']))
    q = (t * F(0.125), t * F(0.25), t * F(0.5), t)
    return {
        '_Time': [t * F(0.05), t, t + t, t * F(3.0)],
        '_SinTime': [crt_sinf(v) for v in q],
        '_CosTime': [crt_cosf(v) for v in q],
        'unity_DeltaTime': [dt, F(1.0) / dt, sdt, F(1.0) / sdt],
    }


# ---------------------------------------------------------------- per draw
def per_draw_values(renderer):
    """UnityPerDraw: unity_ObjectToWorld is the renderer matrix bound with
    SetWorldMatrix; unity_WorldToObject = InvertMatrix4x4_General3D(world)
    computed in ``GfxDeviceD3D11Base::BeforeDrawCall`` (RVA 0x542770, call at
    0x542af7).  unity_WorldTransformParams.w = -1 for odd-negative scale
    (``SetWorldMatrixAndType`` RVA 0x8cf3b0 forwards TransformType bit 2)."""
    o2w = f32_list(renderer['object_to_world'], 16)
    ok, w2o = um.invert_general_3d(o2w)
    out = {'unity_ObjectToWorld': o2w, 'unity_WorldToObject': w2o,
           'unity_WorldTransformParams': [F(0.0), F(0.0), F(0.0),
                                          F(-1.0) if renderer.get('transform_odd_negative_scale') else F(1.0)]}
    if 'lod_fade' in renderer and renderer['lod_fade'] is not None:
        lf = f32_list(renderer['lod_fade'], 2)
        out['unity_LODFade'] = [lf[0], lf[1], F(0.0), F(0.0)]
    if 'rendering_layer_mask' in renderer:
        # float4 slot whose x carries the mask bits (asuint in shaders).
        bits = struct.unpack('<f', struct.pack('<I', renderer['rendering_layer_mask'] & 0xffffffff))[0]
        out['unity_RenderingLayer'] = [F(bits), F(0.0), F(0.0), F(0.0)]
    return out


# ------------------------------------------------------------ lighting CB
def gamma_to_linear_color(c):
    """``GammaToLinearSpace(ColorRGBAf)`` (RVA 0x27c970): the float rule on r, g, b; alpha unchanged."""
    return [gamma_to_linear_space(c[0]), gamma_to_linear_space(c[1]), gamma_to_linear_space(c[2]), F(c[3])]


def light_final_color(light, lights_use_linear_intensity=False, lights_use_color_temperature=False):
    """``SharedLightData::PrecalcLightColor`` (RVA 0x69eab0) -> _LightColor0 / unity_LightColor.

    Gamma intensity (GraphicsSettings.lightsUseLinearIntensity false):
    GammaToLinearSpace(color * intensity) with alpha*intensity.  Linear
    intensity: (temperatureRGB or 1) * GammaToLinearSpace(color) * intensity,
    per channel ((t*c)*i); CorrelatedColorTemperatureToRGB is not transcribed
    (gap ``color_temperature``)."""
    c = f32_list(light['color'], 4)
    i = f32_value(light['intensity'])
    if not lights_use_linear_intensity:
        return gamma_to_linear_color([i * c[0], i * c[1], i * c[2], i * c[3]])
    if lights_use_color_temperature:
        raise GeneratorGap('CorrelatedColorTemperatureToRGB (RVA 0x7d5730) not transcribed')
    lin = gamma_to_linear_color(c)
    return [(F(1.0) * lin[k]) * i for k in range(4)]


def directional_light_pos(light):
    """_WorldSpaceLightPos0 for a directional light: (-forward, 0) where
    forward is column 2 of the light localToWorld (capture_observed: equals
    the negated light Z axis bit-exactly; float negation is exact)."""
    m = f32_list(light['local_to_world'], 16)
    return [-m[8], -m[9], -m[10], F(0.0)]


def object_sh_values(state, renderer):
    """SH constants for one renderer: ``ApplySH`` (RVA 0x6c7860) ->
    ``CalculateLightProbeCoefficients`` (0x6b6e10) -> GetShaderConstantsFromNormalizedSH.
    Without light probes the context's ambient entry is the final ambient probe."""
    usage = renderer.get('light_probe_usage', 'BlendProbes')
    if usage == 'Off':
        raise GeneratorGap('lightProbeUsage Off SH path (s_BlackCoeffs/property sheet) not transcribed')
    if state.value('light_probes.present'):
        if 'interpolated_sh' not in renderer:
            raise SceneStateError('renderers[].interpolated_sh required with light probes')
        sh = f32_list(renderer['interpolated_sh'], 27)
    else:
        sh = final_ambient_probe(state.section('render_settings'))
    return sh_shader_constants(sh)


def texture_decode_values(usage, texture_linear):
    """``GetTextureDecodeValues(TextureUsageMode, TextureColorSpace)`` (RVA 0x7569d0,
    jump table 0x756a8c indexed by usage-1)."""
    x, y, z, w = F(1.0), F(1.0), F(0.0), F(1.0)
    u = int(usage)
    if u in (1, 7):
        x = F(2.0)
        if texture_linear:
            x = gamma_to_linear_space(x)
        w = F(0.0)
    elif u in (2, 5, 9):
        x = F(5.0)
        if texture_linear:
            from .unity_crt import powf
            y = F(2.2)
            x = powf(F(5.0), y)
    elif u == 4:
        x, z, w = F(2.0), F(-1.0), F(0.0)
    else:
        w = F(0.0)
    return [x, y, z, w]


def texture_decode_values_with_intensity(usage, texture_linear, active_linear, intensity):
    """``GetTextureDecodeValuesWithIntensity`` (RVA 0x756ab0, jump table 0x756c4c)."""
    from .unity_crt import powf
    i = F(intensity)
    x0 = gamma_to_linear_space(i) if active_linear else i
    x, y, z, w = F(1.0), F(1.0), F(0.0), F(0.0)
    u = int(usage)
    if u == 0:
        x = x0
    elif u in (1, 7):
        x = i + i
        if texture_linear:
            x = gamma_to_linear_space(x)
    elif u in (2, 5, 9):
        w = F(1.0)
        if not texture_linear:
            x = x0 * F(5.0)
        else:
            y = F(2.2)
            # inline GammaToLinearSpace(intensity) (0x756b74..0x756bdf) times powf(5, 2.2)
            x = powf(F(5.0), y) * gamma_to_linear_space(i)
    elif u == 4:
        x, z = F(2.0), F(-1.0)
    return [x, y, z, w]


def default_reflection_texture(render_settings):
    """``RenderSettings::UpdateDefaultReflection`` (RVA 0x6a5db0): mode != Skybox
    (RenderSettings+0x198) hands ``customReflection`` (+0x1a4) to
    ``ReflectionProbes::SetDefaultTexture`` (0x6802c0), Skybox mode the generated
    cubemap (+0x1a8); the intensity is ``reflectionIntensity`` (+0x194).
    Returns ``None`` for a null texture, otherwise the texture's export record."""
    mode = str(render_settings['defaultReflectionMode'])
    if mode.endswith('=1') or mode.endswith('Custom'):
        return render_settings.get('customReflection')
    raise GeneratorGap('default_reflection_skybox: generated skybox cubemap (RenderSettings+0x1a8) '
                       'usage/colour space not exported')


def default_reflection_sample(render_settings, active_linear=True):
    """``ReflectionProbes::GenerateDefaultSample`` (RVA 0x67d5d0): memset 0x7c, then
    for both probes (stride 0x3c) centre = Vector3f::zero, extents =
    Vector3f::infinityVec, texture = default (black cube when null), HDR decode =
    ``GetTextureDecodeValues(0, 0)`` for a null texture else
    ``GetTextureDecodeValuesWithIntensity(usage, texCS if linear else 0, activeCS,
    intensity)``, probe position = 0; the blend weight (+0x78) stays 0."""
    tex = default_reflection_texture(render_settings)
    if tex is None:
        hdr = texture_decode_values(0, False)
    else:
        if 'usage_mode' not in tex or 'color_space_linear' not in tex:
            raise GeneratorGap('default_reflection_texture_usage: Texture+0x60/+0x64 not exported')
        hdr = texture_decode_values_with_intensity(tex['usage_mode'], bool(tex['color_space_linear']) and active_linear,
                                                   active_linear, render_settings['reflectionIntensity'])
    probe = {'center': [F(0.0)] * 3, 'extents': [F(np.inf)] * 3, 'hdr': hdr, 'position': [F(0.0)] * 4}
    return {'probes': [dict(probe), dict(probe)], 'blend': F(0.0), 'texture_null': tex is None}


def reflection_probe_constants(sample):
    """``SetReflections`` (RVA 0x6c11e0): builtin vector params 0x64..0x6b (name
    table 0x14b89c0): HDR = sample+0x1c, BoxMax = (c+e, 1), BoxMin = (c-e, blend)
    for probe 0 and (c-e, 1) for probe 1, ProbePosition = sample+0x2c."""
    out = {}
    for k, p in enumerate(sample['probes']):
        c, e = p['center'], p['extents']
        out['unity_SpecCube%d_HDR' % k] = list(p['hdr'])
        out['unity_SpecCube%d_BoxMax' % k] = [c[0] + e[0], c[1] + e[1], c[2] + e[2], F(1.0)]
        out['unity_SpecCube%d_BoxMin' % k] = [c[0] - e[0], c[1] - e[1], c[2] - e[2],
                                              sample['blend'] if k == 0 else F(1.0)]
        out['unity_SpecCube%d_ProbePosition' % k] = list(p['position'])
    return out


def renderer_reflection_sample(render_settings, renderer, active_probes, active_linear=True):
    """``GetReflectionSample`` (RVA 0x6512d0): a renderer without probe usage flags
    (RenderNode+0xe8 & 0x6000), or with no per-object probes
    (``SamplePerObjectReflectionProbes`` 0x6696d0, count 0), gets the context's
    default sample.  Per-object probe blending is not transcribed."""
    if active_probes:
        raise GeneratorGap('reflection_probe_blend: SamplePerObjectReflectionProbes with probes not transcribed')
    return default_reflection_sample(render_settings, active_linear)


def probe_volume_disabled():
    """UnityProbeVolume when the renderer uses no LightProbeProxyVolume
    (capture_observed; ``SetLightProbeProxyVolumeParams`` RVA 0x6c0fb0 not
    transcribed: gap ``lppv``)."""
    return {
        'unity_ProbeVolumeParams': [F(0.0), F(1.0), F(1.0), F(0.0)],
        'unity_ProbeVolumeWorldToObject': um.identity(),
        'unity_ProbeVolumeSizeInv': [F(1.0), F(1.0), F(1.0)],
        'unity_ProbeVolumeMin': [F(-np.inf)] * 3,
    }


# ------------------------------------------------------- motion vectors
def motion_vector_globals(setup, cam, renderer):
    """Object motion-vector pass $Globals (Internal-MotionVectors pass 0).

    _NonJitteredVP = devProj(nonJittered)·V and _PreviousVP = the camera's
    stored previous VP (exported); _PreviousM = renderer previous matrix;
    _HasLastPositionData is a bool (int 1); _MotionVectorDepthBias value is
    capture_observed (-0.001, source gap ``mv_depth_bias``)."""
    out = {'_NonJitteredVP': list(setup.non_jittered_vp),
           '_PreviousVP': cam.m('previous_view_projection'),
           '_PreviousM': f32_list(renderer['previous_object_to_world'], 16),
           '_HasLastPositionData': ('int', [1 if renderer.get('has_last_position_data') else 0]),
           '_ForceNoMotion': [F(1.0) if renderer.get('motion_vector_mode') == 'ForceNoMotion' else F(0.0)]}
    return out


# ------------------------------------------------------- shadow constants
def light_shadow_data(light, shadow_distance, camera_far):
    """``CalculateLightShadowFade`` (RVA 0x679090) for _LightShadowData:
    x = 1-strength; with d = fade distance > 0: a = d - d*0.2, den = d - a,
    z = 1/den, w = -(z*a), y = far/d.  Else y=inf, z=0, w=1."""
    s = f32_value(light['shadow_strength'])
    d = F(shadow_distance)
    x = F(1.0) - s
    if d > F(0.0):
        a = d - d * F(0.2)
        den = d - a
        z = F(1.0) / den
        w = -(z * a)
        y = F(camera_far) / d
        return [x, y, z, w]
    return [x, F(np.inf), F(0.0), F(1.0)]


def shadow_distance_for_camera(quality, cam):
    """``Camera::CustomCull`` (RVA 0x660f70 @0x66138b): minss(far, QualitySettings.shadowDistance)."""
    far = cam.f('far_clip')
    sd = f32_value(quality['shadow_distance'])
    return far if far < sd else sd


def shadow_map_size(graphics, quality, light, viewport_w, viewport_h):
    """``CalculateShadowMapSize`` (RVA 0x69ad80), directional branch; shift =
    3 - resolution (helper 0x69f180)."""
    res_names = ('Low', 'Medium', 'High', 'VeryHigh')
    lr = light.get('shadow_resolution', -1)
    res = res_names.index(quality['shadow_resolution']) if lr == -1 else int(lr)
    shift = 3 - res
    custom = int(light.get('shadow_custom_resolution', 0))

    def next_pow2(v):
        v = (v - 1) & 0xffffffff
        v |= v >> 16
        v |= v >> 8
        v |= v >> 4
        v |= v >> 2
        v |= v >> 1
        return (v + 1) & 0xffffffff
    if light['type'] != 'Directional':
        raise GeneratorGap('spot/point CalculateShadowMapSize branches not transcribed')
    if custom > 0:
        size = next_pow2(custom)
    else:
        m = F(viewport_h) if F(viewport_h) > F(viewport_w) else F(viewport_w)
        k = int(F(float(int(m))) * F(-3.8))          # cvttss2si(cvtdq2ps(cvttss2si(max)) * -3.8)
        size = next_pow2(-k) >> shift                 # smear(-1 - k) + 1, then sar shift
    limit = min(int(graphics['max_render_texture_size']), 0x4000)
    if custom <= 0:
        vram_ok = F(graphics['video_memory_mb']) >= F(480.0)
        if vram_ok:
            limit = min(limit, 0x1000)
        else:
            limit = min(limit, 0x1000 if shift == 0 else 0x800)
    size = max(size, 0x20)
    return min(size, limit)


def cascade_split_distances(near, shadow_distance, cascades, splits):
    """Split distances/percentages (RVA 0x69ab10 == 0x7aeed0 transcription).

    dist[0]=near, pct[0]=0, pct[1..3]=splits, dist[i]=pct[i]*(far-near)+near for
    0<i<C, dist[C]=far, pct[C]=1, then dist/pct[i]=prev*1.01 up to index 4."""
    n, f = F(near), F(shadow_distance)
    dist = [F(0.0)] * 5
    pct = [F(0.0)] * 5
    dist[0] = n
    for i in range(3):
        pct[i + 1] = F(splits[i])
    for i in range(1, cascades):
        dist[i] = pct[i] * (f - n) + n
    dist[cascades] = f
    pct[cascades] = F(1.0)
    for i in range(cascades + 1, 5):
        dist[i] = dist[i - 1] * F(1.01)
        pct[i] = pct[i - 1] * F(1.01)
    return dist, pct


def quality_split_ratios(quality):
    """Split ratio input of the split function: 2 cascades -> shadowCascade2Split,
    4 -> shadowCascade4Split (QualitySettings level fields)."""
    c = int(quality['shadow_cascades'])
    if c == 2:
        return [f32_value(quality['shadow_cascade_2_split']), F(0.0), F(0.0)]
    if c == 4:
        return f32_list(quality['shadow_cascade_4_split'], 3)
    return [F(0.0), F(0.0), F(0.0)]


def light_shadow_bias(light, device_shadow_projection, texel_x, texel_y, soft_normal_bias_branch=False):
    """unity_LightShadowBias for a directional cascade (``RenderShadowMaps``
    RVA 0x6a18b0, block 0x6a29c9..0x6a2ae5): x = -(devProj.m22*shadowBias),
    y = 1, z = normalBias*max(texelX, texelY) (normalBias*3.65 first when the
    soft-shadow tier branch holds), w = 0."""
    m22 = device_shadow_projection[10]
    x = -(m22 * f32_value(light['shadow_bias']))
    nb = f32_value(light['shadow_normal_bias'])
    if soft_normal_bias_branch:
        nb = nb * F(3.65)
    tx, ty = F(texel_x), F(texel_y)
    t = ty if ty > tx else tx
    return [x, F(1.0), nb * t, F(0.0)]


def soft_normal_bias_branch(*, light_shadows, caps_0xd0_bit7, tier_0xb4, build_0x101, build_0x102, caps_0x3fc,
                            quality_shadows, shadow_distance):
    """Condition of the normalBias*3.65 multiply (``RenderShadowMaps`` 0x6a29db..0x6a2a60).

    header+0x5b0 (= SharedLightData+0xa4, light shadow type, written 0x69fba8) == 2
    (Soft); GraphicsCaps+0xd0 bit 7; header+0x11140 (GraphicsSettings tier
    [GraphicsCaps+0xd4] +0xb4 byte, 0x69f48d); header+0x11141 / +0x11142
    (BuildSettings+0x101 / +0x102, 0x69f4a0..0x69f4b4); GraphicsCaps+0x3fc == 0;
    header+0x11144 (QualitySettings level +0x2c = shadows) >= 2; header+0x11148
    (level +0x3c = shadowDistance) > 0 (``comiss xmm8, xmm11``; xmm11 = 0 from
    0x6a1acf).  Header fields are filled by the helper 0x69f180."""
    if int(light_shadows) != 2 or not caps_0xd0_bit7 or not tier_0xb4:
        return False
    if not build_0x101 or not build_0x102 or caps_0x3fc:
        return False
    if int(quality_shadows) < 2:
        return False
    return bool(F(shadow_distance) > F(0.0))


def cascade_shader_params(world_to_shadow, cascades, split_dist, spheres):
    """``SetCascadedShadowShaderParams`` (RVA 0x6a3540)."""
    def scale(m):
        return F(math.sqrt(float((m[1] * m[1] + m[0] * m[0]) + m[2] * m[2]))) + F(1e-6)
    s = [scale(world_to_shadow[i]) for i in range(4)]
    out = {
        'unity_WorldToShadow': [v for m in world_to_shadow for v in m],
        '_LightSplitsNear': [split_dist[0], split_dist[1], split_dist[2], split_dist[3]],
        '_LightSplitsFar': [split_dist[1], split_dist[2], split_dist[3], split_dist[4]],
        'unity_ShadowSplitSpheres': [v for sp in spheres for v in sp],
        'unity_ShadowSplitSqRadii': [spheres[0][3], spheres[1][3], spheres[2][3], spheres[3][3]],
        'unity_ShadowCascadeScales': [F(1.0), s[0] / s[1], s[0] / s[2], s[0] / s[3]],
        'keyword_SHADOWS_SINGLE_CASCADE': cascades <= 1,
    }
    return out


def shadow_fade_center_and_type(quality, cam, shadow_distance):
    """``SetupShadowCullData`` (RVA 0x6815f0) / ``CalculateLightShadowCenterAndType``
    (0x679010): CloseFit -> (0,0,0, 0); StableFit -> sphere centre/flag 1
    (StableFit centre arithmetic at 0x681b3c..0x681cc4 not yet transcribed: gap)."""
    if quality['shadow_projection'] == 'CloseFit':
        return [F(0.0), F(0.0), F(0.0), F(0.0)]
    raise GeneratorGap('StableFit shadow fade centre (SetupShadowCullData 0x681b3c) not transcribed')


# ---------------------------------------------------- shadow copy model
class CBufferShadowCopies:
    """Unity ConstantBuffersD3D11 shadow copies keyed by (cbuffer name, size).

    Bytes written by a draw are exactly the variables of the bound program's
    layout; other bytes persist from earlier writers (zero at creation)."""

    def __init__(self):
        self.copies = {}

    def write(self, name, size, layout, values):
        buf = self.copies.setdefault((name, size), bytearray(size))
        before = bytes(buf)
        written = []
        for var in layout['variables']:
            if var['name'] not in values:
                continue
            data = encode(values[var['name']])
            if len(data) < var['size']:
                raise GeneratorGap('value for %s shorter than layout slot' % var['name'])
            buf[var['offset']:var['offset'] + var['size']] = data[:var['size']]
            written.append(var['name'])
        return bytes(buf), bytes(buf) != before, written


def encode(value):
    if isinstance(value, tuple) and value[0] == 'int':
        return pack_int(value[1])
    return pack(value)


def layout_values(layout, values):
    """{variable: hex} for the variables of one Unity cbuffer layout that the
    generator produced, and the list it could not produce."""
    have, missing = {}, []
    for var in layout['variables']:
        if var['name'] in values:
            data = encode(values[var['name']])
            have[var['name']] = data[:var['size']].hex()
        else:
            missing.append(var['name'])
    return have, missing
