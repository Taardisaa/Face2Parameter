"""The HS2 MCP bridge's main-camera capture loop (``POST /maker/renderstyle/capture``).

Source: HS2Mod ``plugins/HS2_McpBridge/MakerRenderStyleService.cs``
``MakerRenderStyleService.Capture`` (route in ``HttpRoutes.cs``), with
``MakerRenderService.PoseFreeze``.  The 16-render loop is unchanged between
HEAD ``b6d170a`` and the working tree (the uncommitted diff only adds paired
geometry export after the readback).  Note: ``/maker/render`` (``RenderHead``)
uses a fresh ``McpBridgeCaptureCam`` with **one** ``Camera.Render`` and no
plugin components; the retry capture's 16 renders come from this route.

Why 16 renders: when the main camera carries an enabled ``CTAA_PC`` the bridge
calls ``ResetTemporal`` (``firstFrame = swap = true``, ``frameCounter = 0``,
the ``OnEnable`` seed) and renders ``16`` times, because the installed
``jitterCam`` cycles through its 16-entry table.  Render *k* uses jitter
index *k*; render 0 seeds the history from its own pre-enhanced colour; the
final target therefore depends on all 16 jittered renders.  Without CTAA the
bridge renders once.

Per render, the camera's component callbacks run in ``GetComponents`` order
(Camera, CameraControl_Ver2, PostProcessLayer, GameScreenShot,
PhysicsRaycaster, CharaCustom.CustomRender, CTAA_PC, GUILayer, DHH.SSS,
DHH.SSS_buffers_viewer): ``OnPreCull`` (PPv2, CTAA jitter), culling,
``OnPreRender`` (DHH SSS profile + lighting renders), the forward camera passes,
the ``OnRenderImage`` chain (``CustomRender`` then ``CTAA_PC``; the buffers
viewer only when enabled), ``OnPostRender`` (SSS).  PPv2 runs from its
command buffers (sibling ``ppv2``).  The last chain element writes the bridge's
``RenderTexture(width, height, 24, ARGB32)`` (sRGB view in Linear colour space).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import ctaa as ctaa_mod
from . import dhh_sss
from . import liquid as liquid_mod
from .plugin_common import (F32, Op, PluginStateError, UnimplementedBranch, UnityRenderTexture, d, mono_f32)

SOURCE = 'HS2Mod plugins/HS2_McpBridge/MakerRenderStyleService.cs Capture (HEAD b6d170a loop)'
DEG2RAD = F32(0.0174532924)          # Mathf.Deg2Rad = PI * 2 / 360 (float literal)
BACKGROUND = (0.5, 0.5, 0.5, 1.0)
TEMPORAL_RENDER_COUNT = 16

# Component order of Camera.main in the Maker (runtime_audit_hook_v3 camera_effects).
MAKER_CAMERA_COMPONENTS = ('UnityEngine.Camera', 'CameraControl_Ver2',
                           'UnityEngine.Rendering.PostProcessing.PostProcessLayer', 'GameScreenShot',
                           'UnityEngine.EventSystems.PhysicsRaycaster', 'CharaCustom.CustomRender', 'CTAA_PC',
                           'UnityEngine.GUILayer', 'DHH.SSS', 'DHH.SSS_buffers_viewer')
# Components of that list with an OnRenderImage (decompiled sources).  PPv2's
# PostProcessLayer is handled by the ppv2 generator from its command buffers.
ON_RENDER_IMAGE = ('CharaCustom.CustomRender', 'CTAA_PC', 'DHH.SSS_buffers_viewer')


@dataclass(frozen=True)
class CaptureRequest:
    path: str
    target: Tuple[float, float, float]
    ortho_size: float
    yaw: float = 0.0
    width: int = 768
    height: int = 1024
    freeze_pose: bool = True

    def validate(self):
        """Request checks of ``Capture`` (all before any mutation)."""
        if not self.path or not self.path.lower().endswith('.png'):
            raise PluginStateError('invalid_path: fresh absolute PNG path required')
        vals = list(self.target) + [self.ortho_size, self.yaw]
        if len(self.target) != 3 or any(math.isnan(v) or math.isinf(v) for v in vals):
            raise PluginStateError('invalid_camera')
        if self.ortho_size <= 0 or self.ortho_size > 20:
            raise PluginStateError('invalid_camera: 0 < ortho_size <= 20')
        if not (64 <= self.width <= 4096 and 64 <= self.height <= 4096):
            raise PluginStateError('invalid_camera: 64..4096 dimensions')


@dataclass(frozen=True)
class MainCamera:
    """``Camera.main`` fields read/changed by the capture (member paths in docs)."""
    field_of_view: float
    near_clip: float
    far_clip: float
    allow_hdr: bool
    allow_msaa: bool
    components: Tuple[Tuple[str, bool], ...]      # GetComponents<Behaviour>() (type, enabled)
    screen_width: int                             # Screen.width
    screen_height: int                            # Screen.height
    custom_render_update: bool = True             # CharaCustom.CustomRender.update
    custom_render_rt: Optional[Tuple[int, int]] = None   # CustomRender.rtCamera (w, h) or None


def capture_distance(ortho_size, field_of_view) -> np.float32:
    """``distance = size / Mathf.Tan(cam.fieldOfView * Mathf.Deg2Rad * .5f)``."""
    arg = mono_f32(d(field_of_view) * d(DEG2RAD) * 0.5)        # argument rounded on the float call
    tan = mono_f32(math.tan(d(arg)))                            # (float)Math.Tan((double)f)
    return mono_f32(d(ortho_size) / d(tan))


def capture_aspect(width: int, height: int) -> np.float32:
    """``cam.aspect = (float)w / h``."""
    return mono_f32(float(F32(width)) / float(F32(height)))


def capture_position(target, distance, yaw) -> np.ndarray:
    """``target + Quaternion.Euler(0, yaw, 0) * new Vector3(0, 0, distance)``.

    ``Quaternion.Euler`` is native (``Internal_FromEulerRad``); for ``yaw == 0``
    it is the identity quaternion exactly and the managed ``operator *``
    returns the vector unchanged.  Other yaws need the transcribed native
    quaternion routine (``unity_math``), not provided yet.
    """
    if float(yaw) != 0.0:
        raise UnimplementedBranch('Quaternion.Euler for yaw != 0 requires the transcribed native routine')
    t = [F32(v) for v in target]
    return np.array([mono_f32(d(t[0]) + 0.0), mono_f32(d(t[1]) + 0.0), mono_f32(d(t[2]) + d(distance))], np.float32)


def bridge_target(request: CaptureRequest) -> UnityRenderTexture:
    """``new RenderTexture(w, h, 24, RenderTextureFormat.ARGB32)``; ``Create()``."""
    return UnityRenderTexture('capture rt', request.width, request.height, 24, 'ARGB32', 'Default',
                              filter_mode='Bilinear', wrap_mode='Repeat', lifetime='persistent',
                              source='MakerRenderStyleService.Capture')


def temporal_component(cam: MainCamera) -> bool:
    """``FirstOrDefault(c => c.GetType().FullName == "CTAA_PC" && c.enabled)``."""
    return any(t == 'CTAA_PC' and en for t, en in cam.components)


@dataclass
class RenderRecord:
    index: int
    pre_cull: Optional[dict]
    sss_ops: List[Op]
    transparent_ops: List[Op]
    image_effect_chain: List[dict]
    post_render: List[Op]
    ctaa_state_after: dict

    def to_json(self):
        from .plugin_common import jsonable
        return {'index': self.index, 'pre_cull': jsonable(self.pre_cull),
                'sss_ops': [o.to_json() for o in self.sss_ops],
                'transparent_ops': [o.to_json() for o in self.transparent_ops],
                'image_effect_chain': jsonable(self.image_effect_chain),
                'post_render': [o.to_json() for o in self.post_render], 'ctaa_state_after': self.ctaa_state_after}


@dataclass
class CapturePlan:
    request: CaptureRequest
    setup: List[Op]
    renders: List[RenderRecord]
    readback: List[Op]
    restore: List[Op]

    def to_json(self):
        from .plugin_common import jsonable
        return {'source': SOURCE, 'request': self.request.__dict__, 'setup': [o.to_json() for o in self.setup],
                'renders': [r.to_json() for r in self.renders], 'readback': [o.to_json() for o in self.readback],
                'restore': [o.to_json() for o in self.restore]}


def plan_capture(request: CaptureRequest, cam: MainCamera, *, sss: Optional[dhh_sss.SSSState],
                 conv: Optional[dhh_sss.ConvolutionState], quality: dhh_sss.QualityState,
                 ctaa_state: Optional[ctaa_mod.CTAAState], character_layer_mask: int,
                 base_projection=None, transparent_objects: Sequence[liquid_mod.ForwardObject] = (),
                 camera_source_rt: Optional[UnityRenderTexture] = None,
                 dof_focus_present: bool = True) -> CapturePlan:
    """Full ordered plan of one ``Capture`` call.

    ``base_projection``: the main camera projection after
    ``ResetProjectionMatrix`` (column-major binary32, from ``unity_math``);
    required when CTAA is enabled.  ``camera_source_rt``: the Unity buffer the
    camera hands to ``OnRenderImage`` (an HDR intermediate when ``allowHDR``);
    its descriptor is ``unity_forward``'s, passed in for the chain sizes/formats.
    """
    request.validate()
    src = SOURCE
    temporal = temporal_component(cam) and ctaa_state is not None
    distance = capture_distance(request.ortho_size, cam.field_of_view)
    rt = bridge_target(request)
    setup = [
        Op('save_camera_state', 'plugin', src, {'fields': ['position', 'rotation', 'targetTexture', 'RenderTexture.active',
                                                         'orthographic', 'orthographicSize', 'aspect', 'cullingMask',
                                                         'clearFlags', 'backgroundColor', 'projectionMatrix',
                                                         'nonJitteredProjectionMatrix', 'worldToCameraMatrix',
                                                         'DHH depthOfFieldLayer.focusDistance.value',
                                                         'all character Transform local TRS', 'blink flag']}),
    ]
    if request.freeze_pose:
        setup.append(Op('pose_freeze', 'plugin', 'MakerRenderService.PoseFreeze.Apply', {
            'animators': 'every Animator with a controller: speed=0; Play(fullPathHash, 0, 0f); Update(0f)',
            'blink': 'ChaControl.ChangeEyesBlinkFlag(false)'}))
    setup += [
        Op('face_surfaces_bake', 'plugin', 'MakerRenderStyleService.FaceSurfaces', {'note': 'BakeMesh hash only; no render state change'}),
        Op('create_persistent', 'plugin', src, {'handle': 'capture rt', 'rt': rt, 'create': True}),
        Op('camera_field', 'unity_forward', src, {'camera': 'Main Camera', 'fields': {
            'position': capture_position(request.target, distance, request.yaw),
            'look_at': np.asarray(request.target, np.float32), 'orthographic': False,
            'aspect': capture_aspect(request.width, request.height),
            'cullingMask': character_layer_mask, 'clearFlags': 'SolidColor', 'backgroundColor': BACKGROUND,
            'targetTexture': 'capture rt'}, 'distance': distance}),
    ]
    if dof_focus_present:
        setup.append(Op('ppv2_parameter', 'ppv2', src, {'path': 'DHH_Main.depthOfFieldLayer.focusDistance.value',
                                                         'value': distance}))
    if temporal:
        ctaa_state.reset_temporal()
        setup.append(Op('reset_temporal', 'plugin', 'MakerRenderStyleService.ResetTemporal',
                        {'firstFrame': True, 'swap': True, 'frameCounter': 0}))
    count = TEMPORAL_RENDER_COUNT if temporal else 1
    renders = []
    sss_on = sss is not None and any(t == 'DHH.SSS' and en for t, en in cam.components)
    carrier = dhh_sss.CameraState(pixel_width=request.width, pixel_height=request.height,
                                  field_of_view=cam.field_of_view, aspect=float(capture_aspect(request.width, request.height)),
                                  near_clip=cam.near_clip, far_clip=cam.far_clip, orthographic=False, orthographic_size=0.0,
                                  allow_hdr=cam.allow_hdr, allow_msaa=cam.allow_msaa, clear_flags='SolidColor',
                                  background_color=BACKGROUND, culling_mask=character_layer_mask)
    for k in range(count):
        pre_cull = None
        if temporal:
            if base_projection is None:
                raise PluginStateError('CTAA jitter requires the reset main-camera projection (unity_math)')
            pre_cull = ctaa_mod.on_pre_cull(ctaa_state, camera_orthographic=False, base_projection=base_projection,
                                            pixel_rect_width=request.width, pixel_rect_height=request.height)
        sss_ops = dhh_sss.on_pre_render(sss, conv, carrier, quality) if sss_on else []
        transparent = liquid_mod.schedule_forward_loop(transparent_objects, camera_source_rt) if transparent_objects else []
        chain = []
        source_desc = camera_source_rt
        effects = [t for t, en in cam.components if en and t in ON_RENDER_IMAGE]
        if sss_on and sss.toggle_texture != 'None':
            effects.append('DHH.SSS_buffers_viewer')
        for i, eff in enumerate(effects):
            last = i == len(effects) - 1
            dest = 'capture rt' if last else 'ImageEffects Temp (unity_forward)'
            if eff == 'CharaCustom.CustomRender':
                if cam.custom_render_update:
                    ops = ([Op('blit', 'plugin', 'CharaCustom.CustomRender.OnRenderImage', {'source': 'source', 'dest': 'rtCamera', 'material': None, 'pass': -1})]
                           if cam.custom_render_rt else [])
                    ops.append(Op('blit', 'plugin', 'CharaCustom.CustomRender.OnRenderImage', {'source': 'source', 'dest': 'destination', 'material': None, 'pass': -1}))
                else:
                    ops = [Op('blit', 'plugin', 'CharaCustom.CustomRender.OnRenderImage', {'source': 'rtCamera', 'dest': 'destination', 'material': None, 'pass': -1})]
            elif eff == 'CTAA_PC':
                if source_desc is None:
                    raise PluginStateError('CTAA OnRenderImage needs the source descriptor (unity_forward image-effect buffer)')
                ops = ctaa_mod.on_render_image(ctaa_state, camera_orthographic=False, source=source_desc,
                                               screen_width=cam.screen_width, screen_height=cam.screen_height)
            elif eff == 'DHH.SSS_buffers_viewer':
                ops = [Op('blit', 'plugin', 'DHH.SSS_buffers_viewer.OnRenderImage', {'source': 'InputBuffer ?? source', 'dest': 'destination', 'material': None, 'pass': -1})]
            else:
                raise UnimplementedBranch(eff)
            chain.append({'component': eff, 'destination': dest, 'ops': ops})
        post = dhh_sss.on_post_render() if sss_on else []
        state_after = ({'first_frame': ctaa_state.first_frame, 'swap': ctaa_state.swap, 'frame_counter': ctaa_state.frame_counter}
                       if temporal else {})
        renders.append(RenderRecord(k, pre_cull, sss_ops, transparent, chain, post, state_after))
    readback = [Op('read_pixels', 'plugin', src, {'source': 'capture rt', 'texture': 'Texture2D(w, h, RGBA32, false)',
                                                   'rect': (0, 0, request.width, request.height),
                                                   'note': 'RenderTexture.active = rt; ReadPixels; Apply; EncodeToPNG'})]
    restore = [Op('restore_camera_state', 'plugin', src, {'note': 'targetTexture, active, transform, orthographic, size, aspect, mask, clear, background, focus, projectionMatrix/nonJitteredProjectionMatrix/worldToCameraMatrix set explicitly'})]
    if temporal:
        ctaa_state.reset_temporal()
        restore.append(Op('reset_temporal', 'plugin', 'MakerRenderStyleService.ResetTemporal', {'firstFrame': True, 'swap': True, 'frameCounter': 0}))
    restore += [Op('destroy', 'plugin', src, {'handles': ['capture rt', 'readback texture']}),
                Op('pose_restore', 'plugin', 'PoseFreeze.Dispose + local TRS restore', {})]
    return CapturePlan(request, setup, renders, readback, restore)


def plugin_pass_sequence(record: RenderRecord) -> List[Tuple[str, str]]:
    """Plugin-owned GPU work of one render, in order, as (label, detail).

    Labels match ``frame_graph.py`` pass labels so a capture segment can be
    compared directly; Unity-internal blits (HDR resolve of the profile
    camera, opaque image-effect copy-back) are listed with ``unity:`` details.
    """
    seq = []
    for op in record.sss_ops:
        if op.kind == 'render_camera' and op.args['camera'] == 'SSS profile Camera':
            seq.append(('sss_profile', 'profile camera RenderWithShader'))
            seq.append(('blit_copy', 'unity: HDR camera intermediate -> SSS_ProfileTex'))
        elif op.kind == 'render_camera' and op.args['camera'] == 'SSS Lighting Pass Camera':
            seq.append(('depth_texture_prepass', 'unity: lighting camera depth'))
            seq.append(('depth_normals', 'unity: lighting camera depth normals'))
            seq.append(('shadow_map', 'unity: lighting camera shadow map'))
            seq.append(('screenspace_shadows', 'unity: lighting camera screen-space shadows'))
            seq.append(('sss_lighting', 'LightingPass replacement'))
            for c in op.args['image_effects_opaque']:
                if c['kind'] == 'blit':
                    a = c['args']
                    if a['material'] == 'Hidden/SeparableSSS':
                        seq.append(('sss_separable_blur', '%s %s -> %s' % (a['axis'], a['source'], a['dest'])))
                    else:
                        seq.append(('blit_copy', '%s -> %s' % (a['source'], a['dest'])))
            seq.append(('blit_copy', 'unity: ImageEffects Temp -> camera target (opaque effect copy-back)'))
    for op in record.transparent_ops:
        if op.kind == 'grab_copy':
            seq.append(('grabpass_copy', op.args['object']))
        elif op.kind == 'draw_pass':
            seq.append(('forward_scene', op.args['object']))
    for eff in record.image_effect_chain:
        for op in eff['ops']:
            if op.kind != 'blit':
                continue
            a = op.args
            if a['material'] in ('Hidden/CTAA_Enhance_PC', 'Hidden/CTAA_PC'):
                seq.append(('temporal_aa_ctaa', '%s %s -> %s' % (a['material'], a['source'], a['dest'])))
            else:
                seq.append(('blit_copy', '%s: %s -> %s' % (eff['component'], a['source'], a['dest'])))
    return seq
