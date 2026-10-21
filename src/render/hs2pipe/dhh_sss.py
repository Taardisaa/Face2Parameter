"""DHH Graphics Enhancer screen-space subsurface scattering (installed DHH_HS2.dll).

Source oracle: ``BepInEx/Plugins/HS2_Plugins/DHH_HS2.dll`` 1.15.30
(SHA-256 ``44137712ecaa9465340e2dc6abc123bed437fc94bfecea0f181c19253206ca3c``),
decompiled with ilspycmd 8.2 into the ignored
``outputs/offline_renderer_plugins_20261020/decompiled/DHH``.  Shaders come from
the plugin's embedded ``sss`` bundle (``Hidden/LightingPass``,
``Hidden/SSS_Profile``, ``Hidden/SeparableSSS``; bundle SHA-256
``ad7c8f1a…``).  Methods reproduced:

* ``DHH.SSS.OnPreRender`` - per render of the camera carrying ``SSS``
  (``Camera.main``; ``SSSManager.Initialize``): global keywords, profile and
  lighting camera configuration (``CreateCameras``/``UpdateCameraModes``),
  render-texture requests (``GetRT``/``GetProfileRT``), the replacement-shader
  renders (``Util.RenderToTarget`` -> ``Camera.RenderWithShader(shader,
  "RenderType")``), blur-material properties and keywords, globals.
* ``DHH.SSS.OnPostRender`` - re-enables ``SCENE_VIEW``.
* ``DHH.SSS_convolution.OnRenderImage`` (``[ImageEffectOpaque]`` on the lighting
  camera) - vertical/horizontal separable blur ping-pong and the final copies.

Float arithmetic follows the Mono rule in ``plugin_common``.  Camera renders are
*requests* to ``unity_forward`` carrying every camera field DHH sets; their
depth/depth-normals/shadow/replacement passes are that module's lowering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .plugin_common import (F32, Op, PluginStateError, UnimplementedBranch, UnityRenderTexture,
                            d, mathf_max, mathf_sqrt, mono_f32, pack_variables)

SOURCE = 'DHH_HS2.dll 1.15.30 sha256 44137712ecaa9465340e2dc6abc123bed437fc94bfecea0f181c19253206ca3c'
SSS_BUNDLE_SHA256 = 'ad7c8f1a5140875b8314091d9583fcfaa095df6fe161c665546224bb7b926c8e'
PITAGORAS_DIAGONAL_REFERENCE = F32(2202.9072)  # SSS_convolution.OnRenderImage literal
PROFILE_CAMERA_DEPTH = F32(-254.0)
LIGHTING_CAMERA_DEPTH = F32(-846.0)
CLEAR_FLAGS_SOLID_COLOR = 'SolidColor'   # CameraClearFlags.Color == SolidColor (2)

# Member paths of every field read from the game (see docs, "required scene state").
SSS_MEMBER = 'Camera.main.GetComponent<DHH.SSS>().'
CONV_MEMBER = 'GameObject("SSS Lighting Pass Camera").GetComponent<DHH.SSS_convolution>().'


@dataclass(frozen=True)
class SSSState:
    """``DHH.SSS`` instance fields (all private/internal members read by reflection)."""
    enabled: bool                    # <Enabled>k__BackingField : bool
    downsampling: float              # <Downsampling>k__BackingField : float
    scattering_radius: float         # <ScatteringRadius>k__BackingField : float
    scattering_iterations: int       # <ScatteringIterations>k__BackingField : int
    shader_iterations: int           # <ShaderIterations>k__BackingField : int
    dither: bool                     # <Dither>k__BackingField : bool
    dither_intensity: float          # <DitherIntensity>k__BackingField : float
    dither_scale: float              # <DitherScale>k__BackingField : float
    depth_test: float                # DepthTest : float
    normal_test: float               # NormalTest : float
    profile_color_test: float        # ProfileColorTest : float
    profile_radius_test: float       # ProfileRadiusTest : float
    dither_edge_test: bool           # DitherEdgeTest : bool
    edge_offset: float               # EdgeOffset : float
    fix_pixel_leaks: bool            # FixPixelLeaks : bool
    max_distance: int                # maxDistance : int
    profile_per_object: bool         # ProfilePerObject : bool
    use_profile_test: bool           # UseProfileTest : bool
    debug_distance: bool             # DEBUG_DISTANCE : bool
    sss_color: Tuple[float, float, float, float]  # sssColor : UnityEngine.Color
    sss_layer: int                   # SSS_Layer : LayerMask (m_Mask int)
    toggle_texture: str              # toggleTexture : DHH.SSS.ToggleTexture
    noise_texture: Optional[dict]    # NoiseTexture : Texture (instance id, size, format, filter, wrap, mips)
    lighting_pass_shader: str = 'Hidden/LightingPass'   # LightingPassShader : Shader
    profile_shader: str = 'Hidden/SSS_Profile'          # ProfileShader : Shader

    @classmethod
    def from_audit_fields(cls, fields: List[dict]) -> 'SSSState':
        """Build from a ``RenderAudit`` component field list (runtime_fields_*.json)."""
        v = {f['field']: f.get('value') for f in fields if f.get('read_succeeded')}
        def need(name):
            if name not in v:
                raise PluginStateError('DHH.SSS field not captured: ' + name)
            return v[name]
        noise = need('NoiseTexture')
        return cls(enabled=bool(need('<Enabled>k__BackingField')), downsampling=float(need('<Downsampling>k__BackingField')),
                   scattering_radius=float(need('<ScatteringRadius>k__BackingField')),
                   scattering_iterations=int(need('<ScatteringIterations>k__BackingField')),
                   shader_iterations=int(need('<ShaderIterations>k__BackingField')),
                   dither=bool(need('<Dither>k__BackingField')), dither_intensity=float(need('<DitherIntensity>k__BackingField')),
                   dither_scale=float(need('<DitherScale>k__BackingField')), depth_test=float(need('DepthTest')),
                   normal_test=float(need('NormalTest')), profile_color_test=float(need('ProfileColorTest')),
                   profile_radius_test=float(need('ProfileRadiusTest')), dither_edge_test=bool(need('DitherEdgeTest')),
                   edge_offset=float(need('EdgeOffset')), fix_pixel_leaks=bool(need('FixPixelLeaks')),
                   max_distance=int(need('maxDistance')), profile_per_object=bool(need('ProfilePerObject')),
                   use_profile_test=bool(need('UseProfileTest')), debug_distance=bool(need('DEBUG_DISTANCE')),
                   sss_color=tuple(float(x) for x in need('sssColor')), sss_layer=int(need('SSS_Layer')),
                   toggle_texture=str(need('toggleTexture')), noise_texture=noise)


@dataclass(frozen=True)
class ConvolutionState:
    """``DHH.SSS_convolution`` on the lighting camera."""
    init_fov: float                  # initFOV : float (main camera FOV at component OnEnable)
    allow_msaa: bool                 # AllowMSAA : bool
    blur_material_exists: bool       # _BlurMaterial != null (created lazily by the first OnRenderImage)

    @classmethod
    def from_audit_fields(cls, fields: List[dict]) -> 'ConvolutionState':
        v = {f['field']: f.get('value') for f in fields if f.get('read_succeeded')}
        for k in ('initFOV', 'AllowMSAA', '_BlurMaterial'):
            if k not in v:
                raise PluginStateError('DHH.SSS_convolution field not captured: ' + k)
        return cls(init_fov=float(v['initFOV']), allow_msaa=bool(v['AllowMSAA']), blur_material_exists=v['_BlurMaterial'] is not None)


@dataclass(frozen=True)
class CameraState:
    """The camera carrying ``DHH.SSS`` at the moment of ``OnPreRender``."""
    pixel_width: int                 # Camera.pixelWidth (target texture width when targetTexture set)
    pixel_height: int                # Camera.pixelHeight
    field_of_view: float             # Camera.fieldOfView
    aspect: float                    # Camera.aspect
    near_clip: float                 # Camera.nearClipPlane
    far_clip: float                  # Camera.farClipPlane
    orthographic: bool               # Camera.orthographic
    orthographic_size: float         # Camera.orthographicSize
    allow_hdr: bool                  # Camera.allowHDR
    allow_msaa: bool                 # Camera.allowMSAA
    clear_flags: str                 # Camera.clearFlags
    background_color: Tuple[float, float, float, float]   # Camera.backgroundColor
    culling_mask: int                # Camera.cullingMask
    stereo_enabled: bool = False     # Camera.stereoEnabled
    stereo_target_eye: str = 'Both'  # Camera.stereoTargetEye
    use_physical_properties: bool = False   # Camera.usePhysicalProperties
    projection_matrix: Optional[Tuple[float, ...]] = None  # Camera.projectionMatrix (physical branch only)
    world_to_camera: Optional[Tuple[float, ...]] = None    # Camera.worldToCameraMatrix (transform-derived)


@dataclass(frozen=True)
class QualityState:
    anti_aliasing: int               # QualitySettings.antiAliasing
    pixel_light_count: int           # QualitySettings.pixelLightCount
    shadows: str                     # QualitySettings.shadows
    color_space: str                 # QualitySettings.activeColorSpace


@dataclass
class SSSRuntime:
    """Render textures DHH keeps between renders (``SSS`` fields) plus the pool."""
    sss_profile_tex: Optional[dict] = None
    lighting_tex: Optional[dict] = None
    lighting_tex_blurred: Optional[dict] = None
    profile_pool_key_after_setup: Optional[tuple] = None


def texture_size(sss: SSSState, cam: CameraState) -> Tuple[np.float32, np.float32]:
    """``m_TextureSize = (float)cam.pixelWidth / Downsampling`` (non-stereo branch)."""
    if cam.stereo_enabled:
        raise UnimplementedBranch('DHH SSS stereo (XRSettings.eyeTextureWidth) branch is not lowered')
    return (mono_f32(float(cam.pixel_width) / d(sss.downsampling)),
            mono_f32(float(cam.pixel_height) / d(sss.downsampling)))


def update_camera_modes(src: CameraState) -> Dict[str, object]:
    """``SSS.UpdateCameraModes(src, dest)``: the fields written on ``dest``."""
    out = {'farClipPlane': F32(src.far_clip), 'nearClipPlane': F32(src.near_clip),
           'stereoTargetEye': src.stereo_target_eye, 'orthographic': src.orthographic,
           'aspect': F32(src.aspect), 'renderingPath': 'Forward', 'orthographicSize': F32(src.orthographic_size)}
    if not src.stereo_enabled:
        if not src.use_physical_properties:
            out['fieldOfView'] = F32(src.field_of_view)
        else:
            if src.projection_matrix is None:
                raise PluginStateError('Physical camera requires Camera.projectionMatrix')
            out['usePhysicalProperties'] = True
            out['projectionMatrix'] = np.asarray(src.projection_matrix, np.float32)
    else:
        raise UnimplementedBranch('Stereo UpdateCameraModes branch')
    return out


def blur_material(sss: SSSState) -> Tuple[Dict[str, object], Dict[str, bool]]:
    """Blur material properties and local keywords set in ``SSS.OnPreRender``.

    Returns ``(properties, keywords)``; ``properties`` maps names to binary32
    values as stored by ``Material.SetFloat/SetInt/SetColor/SetTexture``.
    """
    props: Dict[str, object] = {
        'DepthTest': mathf_max(F32(1e-05), mono_f32(d(sss.depth_test) / 20.0)),
        # maxDistance = Mathf.Max(0, maxDistance) (int); SetFloat(int -> float)
        'maxDistance': F32(max(0, int(sss.max_distance))),
        'NormalTest': mathf_max(F32(0.001), sss.normal_test),
        'ProfileColorTest': mathf_max(F32(0.001), sss.profile_color_test),
        'ProfileRadiusTest': mathf_max(F32(0.001), sss.profile_radius_test),
        'EdgeOffset': F32(sss.edge_offset),
        '_SSS_NUM_SAMPLES': F32(sss.shader_iterations + 1),   # SetInt -> stored as float, int at upload
        'sssColor': np.asarray(sss.sss_color, np.float32),
    }
    keywords = {'RANDOMIZED_ROTATION': bool(sss.dither),
                'PROFILE_TEST': bool(sss.use_profile_test and sss.profile_per_object),
                'DEBUG_DISTANCE': bool(sss.debug_distance),
                'OFFSET_EDGE_TEST': bool(sss.fix_pixel_leaks),
                'DITHER_EDGE_TEST': bool(sss.dither_edge_test)}
    if sss.dither:
        props['DitherScale'] = F32(sss.dither_scale)
        props['DitherIntensity'] = F32(sss.dither_intensity)
        if sss.noise_texture:
            props['NoiseTexture'] = sss.noise_texture
    return props, keywords


def lighting_blur_radius(sss: SSSState, cam: CameraState) -> np.float32:
    """``num = ScatteringRadius / Downsampling`` (``/2`` in stereo) -> ``BlurRadius``."""
    if cam.stereo_enabled:
        raise UnimplementedBranch('Stereo blur radius')
    return mono_f32(d(sss.scattering_radius) / d(sss.downsampling))


def convolution_radius(blur_radius, init_fov, current_fov, width: int, height: int) -> np.float32:
    """``SSS_convolution.OnRenderImage`` radius: field ``BlurRadius`` scaled in place.

    ``num = initFOV / fieldOfView``; ``num2 = Pitagoras(height, width) / 2202.9072f``
    with ``Pitagoras(x, y) = Mathf.Sqrt(x*x + y*y)``.
    """
    num = mono_f32(d(init_fov) / d(current_fov))
    p = mathf_sqrt(mono_f32(float(height) * float(height) + float(width) * float(width)))
    num2 = mono_f32(d(p) / d(PITAGORAS_DIAGONAL_REFERENCE))
    r = mono_f32(d(blur_radius) * d(num))
    r = mono_f32(d(r) * d(num2))
    return r


def convolution_on_render_image(*, iterations: int, blur_radius, init_fov, lighting_fov, source: UnityRenderTexture,
                                rt_format: str, allow_msaa_field: bool, lighting_cam_allow_msaa: bool,
                                quality_aa: int) -> List[Op]:
    """Ordered ops of ``SSS_convolution.OnRenderImage(source, destination)``.

    ``source``/``destination`` are the Unity image-effect buffers handed in by the
    lighting camera (``unity_forward`` resolves their identities); the plugin's
    own temporaries are named ``T1``/``T2``; ``blurred`` is ``LightingTexBlurred``.
    """
    src = 'SSS_convolution.OnRenderImage'
    ops: List[Op] = []
    if iterations > 0:
        radius = convolution_radius(blur_radius, init_fov, lighting_fov, source.width, source.height)
        vertical = np.array([0, radius, 0, 0], np.float32)
        horizontal = np.array([radius, 0, 0, 0], np.float32)
        msaa = quality_aa
        if not lighting_cam_allow_msaa or msaa <= 0 or not allow_msaa_field:
            msaa = 1
        temp = UnityRenderTexture('TempBuffer', source.width, source.height, 0, rt_format, 'Linear', msaa=msaa,
                                  filter_mode='Bilinear', wrap_mode='Clamp', source=src + ' GetTemporary')
        # RenderTexture.GetTemporary defaults: Bilinear filter, Clamp wrap (Unity 2018 temporary RT descriptor).
        ops.append(Op('get_temporary', 'plugin', src, {'handle': 'T1', 'rt': temp}))
        ops.append(Op('blit', 'plugin', src, {'source': 'source', 'dest': 'T1', 'material': 'Hidden/SeparableSSS',
                                               'pass': -1, 'set_vector': {'_TexelOffsetScale': vertical}, 'axis': 'vertical'}))
        n = iterations - 1
        if n != 0:
            ops.append(Op('get_temporary', 'plugin', src, {'handle': 'T2', 'rt': temp}))
            while True:
                ops.append(Op('blit', 'plugin', src, {'source': 'T1', 'dest': 'T2', 'material': 'Hidden/SeparableSSS',
                                                       'pass': -1, 'set_vector': {'_TexelOffsetScale': horizontal}, 'axis': 'horizontal'}))
                ops.append(Op('blit', 'plugin', src, {'source': 'T2', 'dest': 'T1', 'material': 'Hidden/SeparableSSS',
                                                       'pass': -1, 'set_vector': {'_TexelOffsetScale': vertical}, 'axis': 'vertical'}))
                n -= 1
                if n == 0:
                    break
            ops.append(Op('release_temporary', 'plugin', src, {'handle': 'T2'}))
        ops.append(Op('blit', 'plugin', src, {'source': 'T1', 'dest': 'blurred', 'material': 'Hidden/SeparableSSS',
                                               'pass': -1, 'set_vector': {'_TexelOffsetScale': horizontal}, 'axis': 'horizontal'}))
        ops.append(Op('release_temporary', 'plugin', src, {'handle': 'T1'}))
    else:
        ops.append(Op('blit', 'plugin', src, {'source': 'source', 'dest': 'blurred', 'material': None, 'pass': -1}))
    ops.append(Op('blit', 'plugin', src, {'source': 'source', 'dest': 'destination', 'material': None, 'pass': -1}))
    return ops


def lighting_rt(name: str, w: int, h: int, cam: CameraState, quality: QualityState) -> UnityRenderTexture:
    """``SSS.GetRT``: depth 24, ``GetRTFormat``/``GetRTReadWrite`` from ``cam.allowHDR``."""
    fmt = 'ARGBFloat' if cam.allow_hdr else 'Default'
    rw = 'Default' if cam.allow_hdr else 'Linear'
    msaa = quality.anti_aliasing if (cam.allow_msaa and quality.anti_aliasing > 0) else 1
    return UnityRenderTexture(name, w, h, 24, fmt, rw, msaa=msaa, filter_mode='Bilinear', wrap_mode='Clamp',
                              source='DHH.SSS.GetRT')


def profile_rt(w: int, h: int) -> UnityRenderTexture:
    """``SSS.GetProfileRT``: 16-bit depth, ARGB32 Linear, Point, Clamp, autoGenerateMips=false."""
    return UnityRenderTexture('SSS_ProfileTex', w, h, 16, 'ARGB32', 'Linear', filter_mode='Point', wrap_mode='Clamp',
                              source='DHH.SSS.GetProfileRT')


def on_pre_render(sss: SSSState, conv: ConvolutionState, cam: CameraState, quality: QualityState,
                  *, profile_camera_exists: bool = True, lighting_camera_exists: bool = True) -> List[Op]:
    """Ordered ops of one ``DHH.SSS.OnPreRender`` (followed by ``OnPostRender``).

    Profile/lighting cameras already exist after the first frame
    (``GameObject.Find``); creation-time defaults are applied only when the caller
    says they do not exist yet.
    """
    src = 'DHH.SSS.OnPreRender'
    ops: List[Op] = []
    if not sss.enabled:
        ops.append(Op('camera_field', 'unity_forward', src, {'camera': 'SSS Lighting Pass Camera',
                                                              'fields': {'depthTextureMode': 'None'}}))
        ops.append(Op('buffers_viewer', 'plugin', src, {'enabled': False}))
        return ops
    ops.append(Op('global_keyword', 'unity_forward', src, {'keyword': 'SCENE_VIEW', 'enabled': False}))
    tw, th = texture_size(sss, cam)
    w, h = int(tw), int(th)            # (int)m_TextureSize.x truncation
    # CreateCameras
    if sss.profile_per_object and not profile_camera_exists:
        ops.append(Op('create_camera', 'unity_forward', 'DHH.SSS.CreateCameras', {
            'camera': 'SSS profile Camera', 'parent': 'carrier camera transform', 'local_position': (0, 0, 0),
            'local_euler': (0, 0, 0), 'fields': {'backgroundColor': (0, 0, 0, 1), 'enabled': False,
                                                 'depth': PROFILE_CAMERA_DEPTH, 'allowMSAA': False}}))
    if not lighting_camera_exists:
        ops.append(Op('create_camera', 'unity_forward', 'DHH.SSS.CreateCameras', {
            'camera': 'SSS Lighting Pass Camera', 'parent': 'carrier camera transform', 'local_position': (0, 0, 0),
            'local_euler': (0, 0, 0), 'fields': {'enabled': False, 'depth': LIGHTING_CAMERA_DEPTH},
            'components': ['DHH.SSS_convolution (BlurShader Hidden/SeparableSSS)']}))
    ops.append(Op('camera_field', 'unity_forward', 'DHH.SSS.CreateCameras', {
        'camera': 'SSS Lighting Pass Camera', 'fields': {
            'allowMSAA': cam.allow_msaa, 'backgroundColor': cam.background_color,
            'clearFlags': cam.clear_flags, 'cullingMask': cam.culling_mask}}))
    if sss.profile_per_object:
        fields = dict(update_camera_modes(cam))
        fields.update({'depth': PROFILE_CAMERA_DEPTH, 'allowMSAA': False, 'cullingMask': sss.sss_layer,
                       'backgroundColor': (0.0, 0.0, 0.0, 1.0), 'clearFlags': CLEAR_FLAGS_SOLID_COLOR})
        ops.append(Op('camera_field', 'unity_forward', src, {'camera': 'SSS profile Camera', 'fields': fields}))
        ops.append(Op('quality', 'unity_forward', src, {'pixelLightCount': 0, 'shadows': 'Disable',
                                                         'restore': {'pixelLightCount': quality.pixel_light_count,
                                                                     'shadows': quality.shadows}}))
        ops.append(Op('global_keyword', 'unity_forward', src, {'keyword': 'SSS_PROFILES', 'enabled': True}))
        prof = profile_rt(w, h)
        ops.append(Op('release_temporary', 'plugin', 'DHH.SSS.GetProfileRT', {'handle': 'SSS_ProfileTex'}))
        ops.append(Op('get_temporary', 'plugin', 'DHH.SSS.GetProfileRT', {
            'handle': 'SSS_ProfileTex', 'rt': prof, 'post_create': {'autoGenerateMips': False}}))
        ops.append(Op('render_camera', 'unity_forward', 'DHH.Util.RenderToTarget', {
            'camera': 'SSS profile Camera', 'target': 'SSS_ProfileTex', 'replacement_shader': sss.profile_shader,
            'replacement_tag': 'RenderType', 'quality_override': {'pixelLightCount': 0, 'shadows': 'Disable'},
            'view': 'carrier camera transform (child at local identity)', 'image_effects_opaque': []}))
        ops.append(Op('set_global_texture', 'unity_forward', src, {'name': 'SSS_ProfileTex', 'handle': 'SSS_ProfileTex'}))
        ops.append(Op('set_global_texture', 'unity_forward', src, {'name': 'SSS_ProfileTexR', 'handle': 'SSS_ProfileTex'}))
        ops.append(Op('quality', 'unity_forward', src, {'pixelLightCount': quality.pixel_light_count,
                                                         'shadows': quality.shadows}))
    else:
        ops.append(Op('global_keyword', 'unity_forward', src, {'keyword': 'SSS_PROFILES', 'enabled': False}))
        ops.append(Op('destroy', 'plugin', src, {'handles': ['SSS_ProfileTex', 'SSS_ProfileTexR']}))
    lighting_fields = dict(update_camera_modes(cam))
    lighting_fields['allowHDR'] = cam.allow_hdr
    props, keywords = blur_material(sss)
    if conv.blur_material_exists:
        ops.append(Op('material', 'plugin', src, {'material': 'SSS_convolution._BlurMaterial',
                                                   'shader': 'Hidden/SeparableSSS', 'properties': props,
                                                   'keywords': keywords}))
    else:
        # First frame: _BlurMaterial is created inside the following OnRenderImage
        # with shader defaults only; properties are first set on the next render.
        ops.append(Op('material', 'plugin', src, {'material': 'SSS_convolution._BlurMaterial',
                                                   'shader': 'Hidden/SeparableSSS', 'properties': {},
                                                   'keywords': {}, 'note': 'created lazily; no properties this render'}))
    radius = lighting_blur_radius(sss, cam)
    lighting_fields.update({'backgroundColor': cam.background_color, 'clearFlags': cam.clear_flags,
                            'cullingMask': sss.sss_layer, 'depth': LIGHTING_CAMERA_DEPTH,
                            'depthTextureMode': 'DepthNormals'})
    ops.append(Op('camera_field', 'unity_forward', src, {'camera': 'SSS Lighting Pass Camera', 'fields': lighting_fields}))
    ops.append(Op('component_field', 'plugin', src, {'component': 'SSS_convolution', 'fields': {
        'iterations': sss.scattering_iterations, 'BlurRadius': radius}}))
    lt = lighting_rt('LightingTexture', w, h, cam, quality)
    lb = lighting_rt('SSSLightingTextureBlurred', w, h, cam, quality)
    ops.append(Op('release_temporary', 'plugin', 'DHH.SSS.GetRT', {'handle': 'LightingTexture'}))
    ops.append(Op('get_temporary', 'plugin', 'DHH.SSS.GetRT', {'handle': 'LightingTexture', 'rt': lt}))
    ops.append(Op('release_temporary', 'plugin', 'DHH.SSS.GetRT', {'handle': 'SSSLightingTextureBlurred'}))
    ops.append(Op('get_temporary', 'plugin', 'DHH.SSS.GetRT', {'handle': 'SSSLightingTextureBlurred', 'rt': lb}))
    lighting_fov = F32(cam.field_of_view)   # LightingCamera.fieldOfView after UpdateCameraModes
    conv_ops = convolution_on_render_image(
        iterations=sss.scattering_iterations, blur_radius=radius, init_fov=conv.init_fov, lighting_fov=lighting_fov,
        source=lt, rt_format=lt.resolved_format(), allow_msaa_field=conv.allow_msaa,
        lighting_cam_allow_msaa=cam.allow_msaa, quality_aa=quality.anti_aliasing)
    ops.append(Op('render_camera', 'unity_forward', 'DHH.Util.RenderToTarget', {
        'camera': 'SSS Lighting Pass Camera', 'target': 'LightingTexture', 'replacement_shader': sss.lighting_pass_shader,
        'replacement_tag': 'RenderType', 'depth_texture_mode': 'DepthNormals',
        'view': 'carrier camera transform (child at local identity); projection from fieldOfView/aspect/clip (never jittered)',
        'image_effects_opaque': [o.to_json() for o in conv_ops], 'blurred_handle': 'SSSLightingTextureBlurred'}))
    for name, handle in (('LightingTexBlurred', 'SSSLightingTextureBlurred'), ('LightingTex', 'LightingTexture'),
                         ('LightingTexBlurredR', 'SSSLightingTextureBlurred'), ('LightingTexR', 'LightingTexture')):
        ops.append(Op('set_global_texture', 'unity_forward', src, {'name': name, 'handle': handle}))
    viewer = sss.toggle_texture != 'None'
    ops.append(Op('buffers_viewer', 'plugin', src, {'enabled': viewer, 'input': {
        'LightingTex': 'LightingTexture', 'LightingTexBlurred': 'SSSLightingTextureBlurred',
        'ProfileTex': 'SSS_ProfileTex'}.get(sss.toggle_texture)}))
    return ops


def on_post_render() -> List[Op]:
    return [Op('global_keyword', 'unity_forward', 'DHH.SSS.OnPostRender', {'keyword': 'SCENE_VIEW', 'enabled': True})]


def blur_cbuffer(manifests, sss: SSSState, *, source: UnityRenderTexture, texel_offset_scale, global_keywords) -> Dict[str, dict]:
    """``$Globals`` variable bytes of one SeparableSSS blur draw.

    Variant = material keywords + enabled global keywords (``SSS_PROFILES`` is
    global, set in ``OnPreRender``).  ``_MainTex_TexelSize`` is the engine value
    of the Blit source.
    """
    from .plugin_common import keyword_variant
    props, keywords = blur_material(sss)
    enabled = {k for k, on in keywords.items() if on} | set(global_keywords)
    shader = manifests.shader('Hidden/SeparableSSS')
    kw = keyword_variant(shader, 0, 'progFragment', enabled)
    program = manifests.program('Hidden/SeparableSSS', 0, 'progFragment', kw)
    values = dict(props)
    values['_TexelOffsetScale'] = np.asarray(texel_offset_scale, np.float32)
    values['_MainTex_TexelSize'] = source.texel_size()
    return {'keywords': kw, 'variables': pack_variables(program, values)}
