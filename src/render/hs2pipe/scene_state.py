"""Scene-state input schema for the Unity built-in forward generator.

One JSON document describes exactly the state Unity 2018.4.11f1 reads for
the camera renders of one frame.  Every field names the Unity API (or
installed-plugin field) the game-side exporter must read, *at the time the
engine reads it* (``when``).  The generator never fills a missing field with
a default: absent required fields raise ``SceneStateError`` and are reported
as gaps.  Float fields are carried as their exact float32 bit pattern
(``"0x3f800000"`` strings) or as JSON numbers that round-trip to the same
float32; matrices are 16 floats in Unity memory order (column-major,
``Matrix4x4.this[i]`` index order m00,m10,m20,m30,m01,...), which is the same
order as ``Matrix4x4f`` and as the HLSL constant-buffer layout.

``FIELDS`` is the authoritative exporter specification (also rendered into
``docs/hs2_offline_unity_forward.md``).  ``validate`` checks a document
against it.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = 'hs2pipe.scene_state'
VERSION = 1


class SceneStateError(ValueError):
    pass


# --------------------------------------------------------------- field spec
# (path, type, unity_api_source, when, required_for)
# type: f32, f32[n], m44 (16 f32), i32, u32, bool, str, enum:<names>, list, obj
# 'when': 'per_render' = read in Camera.onPreRender for that exact render
#         (CTAA and the bridge change camera state between the 16 renders);
#         'per_frame' = once per player-loop frame before rendering;
#         'per_draw' = the value the renderer has when that camera renders it.
FIELDS = [
    # ---- device / graphics
    ('graphics.device_type', 'str', 'SystemInfo.graphicsDeviceType (must be Direct3D11)', 'per_frame', 'all'),
    ('graphics.reversed_z', 'bool', 'SystemInfo.usesReversedZBuffer', 'per_frame', 'projection'),
    ('graphics.uses_opengl_texture_coords', 'bool', 'derived: false for Direct3D11 (GraphicsCaps.usesOpenGLTextureCoords)', 'per_frame', 'projection'),
    ('graphics.video_memory_mb', 'i32', 'SystemInfo.graphicsMemorySize (GraphicsCaps+0xd8 compared to 480 in CalculateShadowMapSize)', 'per_frame', 'shadows'),
    ('graphics.max_render_texture_size', 'i32', 'SystemInfo.maxTextureSize / GraphicsCaps.maxRenderTextureSize (+0xf8)', 'per_frame', 'shadows'),
    ('graphics.tier_cascaded_shadow_maps', 'bool', 'EditorGraphicsSettings tier: Graphics tier setting cascadedShadowMaps for Graphics.activeTier (GraphicsSettings tier+0xb4)', 'per_frame', 'shadows'),
    ('graphics.active_color_space', 'enum:Gamma,Linear', 'QualitySettings.activeColorSpace', 'per_frame', 'lighting'),
    # ---- time (UpdateGlobalShaderProperties, RVA 0x85d680)
    ('time.shader_time', 'f32', 'Shader global time t = (float)(TimeManager curTime + offset); exporter: Shader.GetGlobalVector("_Time").y read inside OnPreRender, else Time.timeSinceLevelLoad', 'per_frame', 'UnityPerCamera'),
    ('time.delta_time', 'f32', 'Time.deltaTime', 'per_frame', 'UnityPerCamera'),
    ('time.smooth_delta_time', 'f32', 'Time.smoothDeltaTime', 'per_frame', 'UnityPerCamera'),
    # ---- quality settings (current level struct, 0xa0 bytes)
    ('quality.shadows', 'enum:Disable,HardOnly,All', 'QualitySettings.shadows', 'per_frame', 'shadows'),
    ('quality.shadow_resolution', 'enum:Low,Medium,High,VeryHigh', 'QualitySettings.shadowResolution', 'per_frame', 'shadows'),
    ('quality.shadow_projection', 'enum:CloseFit,StableFit', 'QualitySettings.shadowProjection', 'per_frame', 'shadows'),
    ('quality.shadow_cascades', 'i32', 'QualitySettings.shadowCascades', 'per_frame', 'shadows'),
    ('quality.shadow_distance', 'f32', 'QualitySettings.shadowDistance', 'per_frame', 'shadows'),
    ('quality.shadow_near_plane_offset', 'f32', 'QualitySettings.shadowNearPlaneOffset', 'per_frame', 'shadows'),
    ('quality.shadow_cascade_2_split', 'f32', 'QualitySettings.shadowCascade2Split', 'per_frame', 'shadows'),
    ('quality.shadow_cascade_4_split', 'f32[3]', 'QualitySettings.shadowCascade4Split', 'per_frame', 'shadows'),
    ('quality.pixel_light_count', 'i32', 'QualitySettings.pixelLightCount', 'per_frame', 'lights'),
    ('quality.lod_bias', 'f32', 'QualitySettings.lodBias', 'per_frame', 'culling'),
    ('quality.maximum_lod_level', 'i32', 'QualitySettings.maximumLODLevel', 'per_frame', 'culling'),
    ('quality.soft_particles', 'bool', 'QualitySettings.softParticles', 'per_frame', 'keywords'),
    ('build.flag_101', 'bool', 'BuildSettings+0x101 (copied to ShadowMapJobHeader+0x11141; normal-bias 3.65 branch)', 'per_frame', 'shadows'),
    ('build.flag_102', 'bool', 'BuildSettings+0x102 (ShadowMapJobHeader+0x11142)', 'per_frame', 'shadows'),
    # ---- render settings
    ('render_settings.ambient_mode', 'enum:Skybox,Trilight,Flat,Custom', 'RenderSettings.ambientMode', 'per_frame', 'lighting'),
    ('render_settings.ambient_intensity', 'f32', 'RenderSettings.ambientIntensity', 'per_frame', 'lighting'),
    ('render_settings.ambient_sky_color', 'f32[4]', 'RenderSettings.ambientSkyColor (gamma value as stored)', 'per_frame', 'lighting'),
    ('render_settings.ambient_equator_color', 'f32[4]', 'RenderSettings.ambientEquatorColor', 'per_frame', 'lighting'),
    ('render_settings.ambient_ground_color', 'f32[4]', 'RenderSettings.ambientGroundColor', 'per_frame', 'lighting'),
    ('render_settings.ambient_probe', 'f32[27]', 'RenderSettings.ambientProbe (SphericalHarmonicsL2, [rgb][9] order, RenderSettings+0x60)', 'per_frame', 'lighting'),
    ('render_settings.has_skybox_material', 'bool', 'RenderSettings.skybox != null', 'per_frame', 'lighting'),
    ('render_settings.subtractive_shadow_color', 'f32[4]', 'RenderSettings.subtractiveShadowColor', 'per_frame', 'UnityPerFrame'),
    ('render_settings.fog', 'obj', 'RenderSettings.fog/fogMode/fogColor/fogDensity/fogStartDistance/fogEndDistance', 'per_frame', 'UnityFog'),
    ('render_settings.default_reflection', 'obj', 'RenderSettings.defaultReflectionMode, customReflection (texture identity), defaultReflectionResolution, reflectionIntensity, reflectionBounces; plus ReflectionProbe.defaultTexture and ReflectionProbe.defaultTextureHDRDecodeValues', 'per_frame', 'UnityReflectionProbes'),
    ('light_probes.present', 'bool', 'LightmapSettings.lightProbes != null && count > 0', 'per_frame', 'lighting'),
    # ---- cameras: one entry per Camera.Render / player-loop render, in submission order
    ('cameras[].name', 'str', 'Camera.name + GetInstanceID()', 'per_render', 'all'),
    ('cameras[].render_index', 'i32', 'ordinal of this Camera.Render call inside the frame', 'per_render', 'all'),
    ('cameras[].target', 'obj', 'Camera.targetTexture (null|{width,height,format,depth,antiAliasing}) and RenderTexture.active; pixelRect / Camera.pixelWidth/pixelHeight', 'per_render', 'all'),
    ('cameras[].render_into_texture', 'bool', 'true when the render targets a RenderTexture (Camera.targetTexture or an intermediate image-effect texture); drives GfxDevice invertProjection', 'per_render', 'projection'),
    ('cameras[].field_of_view', 'f32', 'Camera.fieldOfView', 'per_render', 'projection'),
    ('cameras[].aspect', 'f32', 'Camera.aspect', 'per_render', 'projection'),
    ('cameras[].near_clip', 'f32', 'Camera.nearClipPlane', 'per_render', 'projection'),
    ('cameras[].far_clip', 'f32', 'Camera.farClipPlane', 'per_render', 'projection'),
    ('cameras[].orthographic', 'bool', 'Camera.orthographic', 'per_render', 'projection'),
    ('cameras[].orthographic_size', 'f32', 'Camera.orthographicSize', 'per_render', 'projection'),
    ('cameras[].implicit_projection', 'bool', 'true unless a script assigned Camera.projectionMatrix (CTAA does); Camera+0x4ec==1 test in SetCameraShaderProps', 'per_render', 'projection'),
    ('cameras[].projection_matrix', 'm44', 'Camera.projectionMatrix (GL convention, CTAA-jittered) read in OnPreRender', 'per_render', 'projection'),
    ('cameras[].non_jittered_projection_matrix', 'm44', 'Camera.nonJitteredProjectionMatrix', 'per_render', 'motion_vectors'),
    ('cameras[].world_to_camera_matrix', 'm44', 'Camera.worldToCameraMatrix', 'per_render', 'projection'),
    ('cameras[].position', 'f32[3]', 'Camera.transform.position', 'per_render', 'UnityPerCamera'),
    ('cameras[].previous_view_projection', 'm44', 'Camera previous VP used as _PreviousVP (Camera.previousViewProjectionMatrix, 2018.4 reflection: Camera::m_PreviousViewProjectionMatrix via UpdatePreviousViewProjectionMatrix)', 'per_render', 'motion_vectors'),
    ('cameras[].culling_mask', 'u32', 'Camera.cullingMask', 'per_render', 'culling'),
    ('cameras[].layer_cull_distances', 'f32[32]', 'Camera.layerCullDistances', 'per_render', 'culling'),
    ('cameras[].layer_cull_spherical', 'bool', 'Camera.layerCullSpherical', 'per_render', 'culling'),
    ('cameras[].use_occlusion_culling', 'bool', 'Camera.useOcclusionCulling (and whether occlusion data exists)', 'per_render', 'culling'),
    ('cameras[].depth_texture_mode', 'u32', 'Camera.depthTextureMode (Depth=1, DepthNormals=2, MotionVectors=4)', 'per_render', 'passes'),
    ('cameras[].clear_flags', 'enum:Skybox,SolidColor,Depth,Nothing', 'Camera.clearFlags', 'per_render', 'passes'),
    ('cameras[].background_color', 'f32[4]', 'Camera.backgroundColor', 'per_render', 'passes'),
    ('cameras[].allow_hdr', 'bool', 'Camera.allowHDR', 'per_render', 'passes'),
    ('cameras[].allow_msaa', 'bool', 'Camera.allowMSAA (+QualitySettings.antiAliasing)', 'per_render', 'passes'),
    ('cameras[].rendering_path', 'enum:Forward,Deferred,VertexLit', 'Camera.actualRenderingPath', 'per_render', 'passes'),
    ('cameras[].opaque_sort_mode', 'enum:Default,FrontToBack,NoDistanceSort', 'Camera.opaqueSortMode', 'per_render', 'sorting'),
    ('cameras[].transparency_sort_mode', 'enum:Default,Perspective,Orthographic,CustomAxis', 'Camera.transparencySortMode/transparencySortAxis', 'per_render', 'sorting'),
    ('cameras[].replacement_shader', 'obj', 'Camera.SetReplacementShader shader+tag (DHH SSS camera)', 'per_render', 'passes'),
    ('cameras[].command_buffers', 'list', 'Camera.GetCommandBuffers(evt) for every CameraEvent (owned by plugin generators; listed for ordering)', 'per_render', 'passes'),
    ('cameras[].stereo_enabled', 'bool', 'Camera.stereoEnabled (must be false)', 'per_render', 'all'),
    # ---- lights (all Light components that are enabled and active)
    ('lights[].instance_id', 'i32', 'Light.GetInstanceID()', 'per_render', 'lights'),
    ('lights[].type', 'enum:Spot,Directional,Point,Area', 'Light.type', 'per_render', 'lights'),
    ('lights[].local_to_world', 'm44', 'Light.transform.localToWorldMatrix (axes: shadow basis; -column2 = _WorldSpaceLightPos0 for directional)', 'per_render', 'lights'),
    ('lights[].color', 'f32[4]', 'Light.color', 'per_render', 'lights'),
    ('lights[].intensity', 'f32', 'Light.intensity', 'per_render', 'lights'),
    ('lights[].bounce_intensity', 'f32', 'Light.bounceIntensity', 'per_render', 'lights'),
    ('lights[].color_temperature', 'f32', 'Light.colorTemperature + GraphicsSettings.lightsUseColorTemperature/lightsUseLinearIntensity', 'per_render', 'lights'),
    ('lights[].range', 'f32', 'Light.range', 'per_render', 'lights'),
    ('lights[].spot_angle', 'f32', 'Light.spotAngle', 'per_render', 'lights'),
    ('lights[].render_mode', 'enum:Auto,ForcePixel,ForceVertex', 'Light.renderMode', 'per_render', 'lights'),
    ('lights[].culling_mask', 'u32', 'Light.cullingMask', 'per_render', 'lights'),
    ('lights[].shadows', 'enum:None,Hard,Soft', 'Light.shadows', 'per_render', 'shadows'),
    ('lights[].shadow_strength', 'f32', 'Light.shadowStrength', 'per_render', 'shadows'),
    ('lights[].shadow_bias', 'f32', 'Light.shadowBias', 'per_render', 'shadows'),
    ('lights[].shadow_normal_bias', 'f32', 'Light.shadowNormalBias', 'per_render', 'shadows'),
    ('lights[].shadow_near_plane', 'f32', 'Light.shadowNearPlane', 'per_render', 'shadows'),
    ('lights[].shadow_resolution', 'i32', 'Light.shadowResolution (-1 = FromQualitySettings)', 'per_render', 'shadows'),
    ('lights[].shadow_custom_resolution', 'i32', 'Light.shadowCustomResolution', 'per_render', 'shadows'),
    ('lights[].cookie', 'obj', 'Light.cookie identity / cookieSize', 'per_render', 'lights'),
    ('lights[].baking_output', 'obj', 'Light.bakingOutput (lightmapBakeType, mixedLightingMode, occlusionMaskChannel)', 'per_render', 'lights'),
    ('lights[].is_main_light_candidate', 'bool', 'RenderSettings.sun == this', 'per_render', 'lights'),
    # ---- renderers (every Renderer the camera can consider, before culling)
    ('renderers[].instance_id', 'i32', 'Renderer.GetInstanceID()', 'per_render', 'all'),
    ('renderers[].kind', 'enum:MeshRenderer,SkinnedMeshRenderer,ParticleSystemRenderer,Other', 'Renderer type', 'per_render', 'all'),
    ('renderers[].enabled', 'bool', 'Renderer.enabled && gameObject.activeInHierarchy', 'per_render', 'culling'),
    ('renderers[].layer', 'i32', 'Renderer.gameObject.layer', 'per_render', 'culling'),
    ('renderers[].world_bounds', 'f32[6]', 'Renderer.bounds (center xyz, extents xyz)', 'per_render', 'culling'),
    ('renderers[].object_to_world', 'm44', 'matrix Unity binds as unity_ObjectToWorld: MeshRenderer -> transform.localToWorldMatrix; SkinnedMeshRenderer -> rootBone TRS without scale (see head-skinning doc)', 'per_render', 'UnityPerDraw'),
    ('renderers[].transform_odd_negative_scale', 'bool', 'TransformType odd-negative-scale flag (det of object_to_world 3x3 < 0)', 'per_render', 'UnityPerDraw'),
    ('renderers[].lod_fade', 'f32[2]', 'LODGroup fade (unity_LODFade.xy) or absent when no LODGroup crossfade', 'per_render', 'UnityPerDraw'),
    ('renderers[].rendering_layer_mask', 'u32', 'Renderer.renderingLayerMask', 'per_render', 'UnityPerDraw'),
    ('renderers[].light_probe_usage', 'enum:Off,BlendProbes,UseProxyVolume,CustomProvided', 'Renderer.lightProbeUsage', 'per_render', 'lighting'),
    ('renderers[].reflection_probe_usage', 'enum:Off,BlendProbes,BlendProbesAndSkybox,Simple', 'Renderer.reflectionProbeUsage', 'per_render', 'lighting'),
    ('renderers[].probe_anchor_position', 'f32[3]', 'Renderer.probeAnchor ? anchor.position : bounds.center', 'per_render', 'lighting'),
    ('renderers[].interpolated_sh', 'f32[27]', 'LightProbes.GetInterpolatedProbe(probe_anchor_position, renderer, out sh) (only when light_probes.present)', 'per_render', 'lighting'),
    ('renderers[].reflection_probes', 'list', 'Renderer.GetClosestReflectionProbes(list): probe identity, weight, box, HDR decode', 'per_render', 'lighting'),
    ('renderers[].shadow_casting_mode', 'enum:Off,On,TwoSided,ShadowsOnly', 'Renderer.shadowCastingMode', 'per_render', 'shadows'),
    ('renderers[].receive_shadows', 'bool', 'Renderer.receiveShadows', 'per_render', 'shadows'),
    ('renderers[].motion_vector_mode', 'enum:Camera,Object,ForceNoMotion', 'Renderer.motionVectorGenerationMode', 'per_render', 'motion_vectors'),
    ('renderers[].previous_object_to_world', 'm44', 'previous-frame matrix used as _PreviousM (Unity keeps the last rendered L2W per renderer)', 'per_render', 'motion_vectors'),
    ('renderers[].has_last_position_data', 'bool', 'SkinnedMeshRenderer has previous skinned VB (_HasLastPositionData)', 'per_render', 'motion_vectors'),
    ('renderers[].sorting_layer_id', 'i32', 'Renderer.sortingLayerID (+SortingLayer.GetLayerValueFromID)', 'per_render', 'sorting'),
    ('renderers[].sorting_order', 'i32', 'Renderer.sortingOrder', 'per_render', 'sorting'),
    ('renderers[].sorting_fudge', 'f32', 'BaseRenderer sorting fudge (particle sortingFudge / renderer-specific)', 'per_render', 'sorting'),
    ('renderers[].lightmap_index', 'i32', 'Renderer.lightmapIndex / realtimeLightmapIndex', 'per_render', 'lighting'),
    ('renderers[].materials', 'list', 'Renderer.sharedMaterials: per submesh {instance_id, shader name, shader instance id, renderQueue, shaderKeywords, passes enabled (Material.GetShaderPassEnabled), SetPass state}', 'per_render', 'passes'),
    ('renderers[].submeshes', 'list', 'mesh submesh {index_start, index_count, base_vertex, topology} and mesh instance id', 'per_render', 'draws'),
    ('renderers[].material_property_block', 'obj', 'Renderer.GetPropertyBlock contents (if any)', 'per_render', 'shader_library'),
]

REQUIRED_TOP = ('graphics', 'time', 'quality', 'render_settings', 'cameras', 'lights', 'renderers')


def f32_value(v):
    """JSON value -> exact float32.  Accepts numbers or '0x%08x' bit strings."""
    if isinstance(v, str):
        if v.startswith('0x'):
            return np.frombuffer(struct.pack('<I', int(v, 16)), dtype=np.float32)[0]
        if v in ('inf', 'Infinity'):
            return np.float32(np.inf)
        if v in ('-inf', '-Infinity'):
            return np.float32(-np.inf)
        raise SceneStateError('Unknown float literal %r' % v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SceneStateError('Float expected, got %r' % (v,))
    return np.float32(v)


def f32_list(v, n=None):
    if not isinstance(v, list):
        raise SceneStateError('Float list expected')
    if n is not None and len(v) != n:
        raise SceneStateError('Expected %d floats, got %d' % (n, len(v)))
    return [f32_value(x) for x in v]


def _get(doc, dotted):
    cur = doc
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def validate(doc, needs=None):
    """Return the list of missing/invalid field paths.  ``needs`` limits the
    check to fields whose ``required_for`` is in that set (e.g. {'projection'})."""
    problems = []
    if doc.get('schema') != SCHEMA or doc.get('version') != VERSION:
        problems.append('schema/version must be %s/%d' % (SCHEMA, VERSION))
    for key in REQUIRED_TOP:
        if key not in doc:
            problems.append(key)
    for path, typ, _src, _when, req in FIELDS:
        if needs is not None and req not in needs and req != 'all':
            continue
        if '[]' in path:
            head, tail = path.split('[].', 1)
            items = doc.get(head)
            if not isinstance(items, list):
                continue
            for i, it in enumerate(items):
                val, ok = _get(it, tail)
                if not ok:
                    problems.append('%s[%d].%s' % (head, i, tail))
                else:
                    err = _check_type(val, typ)
                    if err:
                        problems.append('%s[%d].%s: %s' % (head, i, tail, err))
        else:
            val, ok = _get(doc, path)
            if not ok:
                problems.append(path)
            else:
                err = _check_type(val, typ)
                if err:
                    problems.append('%s: %s' % (path, err))
    return problems


def _check_type(val, typ):
    try:
        if typ == 'f32':
            f32_value(val)
        elif typ.startswith('f32['):
            f32_list(val, int(typ[4:-1]))
        elif typ == 'm44':
            f32_list(val, 16)
        elif typ in ('i32', 'u32'):
            if isinstance(val, bool) or not isinstance(val, int):
                return 'integer expected'
        elif typ == 'bool':
            if not isinstance(val, bool):
                return 'bool expected'
        elif typ == 'str':
            if not isinstance(val, str):
                return 'string expected'
        elif typ.startswith('enum:'):
            if val not in typ[5:].split(','):
                return 'one of %s expected' % typ[5:]
        elif typ == 'list':
            if not isinstance(val, list):
                return 'list expected'
        elif typ == 'obj':
            if val is not None and not isinstance(val, dict):
                return 'object or null expected'
    except SceneStateError as e:
        return str(e)
    return None


# ---------------------------------------------------------- typed accessors
@dataclass
class CameraRender:
    raw: dict

    def f(self, key):
        if key not in self.raw:
            raise SceneStateError('camera field %s missing' % key)
        return f32_value(self.raw[key])

    def m(self, key):
        if key not in self.raw:
            raise SceneStateError('camera field %s missing' % key)
        return f32_list(self.raw[key], 16)

    def get(self, key):
        if key not in self.raw:
            raise SceneStateError('camera field %s missing' % key)
        return self.raw[key]


@dataclass
class SceneState:
    doc: dict
    path: str = None
    cameras: list = field(default_factory=list)

    @classmethod
    def load(cls, path):
        doc = json.loads(Path(path).read_text(encoding='utf-8'))
        s = cls(doc, str(path))
        s.cameras = [CameraRender(c) for c in doc.get('cameras', [])]
        return s

    @classmethod
    def from_dict(cls, doc):
        s = cls(doc)
        s.cameras = [CameraRender(c) for c in doc.get('cameras', [])]
        return s

    def section(self, name):
        if name not in self.doc:
            raise SceneStateError('scene-state section %s missing' % name)
        return self.doc[name]

    def value(self, dotted):
        val, ok = _get(self.doc, dotted)
        if not ok:
            raise SceneStateError('scene-state field %s missing' % dotted)
        return val


def field_table_markdown():
    rows = ['| field | type | Unity API source | read when | used for |', '|---|---|---|---|---|']
    for path, typ, src, when, req in FIELDS:
        rows.append('| `%s` | %s | %s | %s | %s |' % (path, typ.replace('|', '/'), src.replace('|', '/'), when, req))
    return '\n'.join(rows)
