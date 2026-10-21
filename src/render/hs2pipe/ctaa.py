"""HS2_BetterAA (ManlyMarco) CTAA temporal anti-aliasing (installed HS2_BetterAA.dll).

Source oracle: ``BepInEx/Plugins/HS2_Plugins/ManlyMarco/HS2_BetterAA.dll`` 1.6
(SHA-256 ``f5ecfeea7a361f30733a1448dee42d5751690178a07796488e9e71a857f5a464``),
decompiled into the ignored ``outputs/offline_renderer_plugins_20261020/decompiled/BetterAA``.
Shaders ``Hidden/CTAA_PC`` and ``Hidden/CTAA_Enhance_PC`` come from the
embedded ``ctaa`` bundle (``outputs/offline_renderer_frame_graph_20261020/embedded``).

Methods reproduced:

* ``AI_stuff.BetterAA``: a ``CTAA_PC`` is added to every camera whose
  ``PostProcessLayer.OnEnable`` runs (Harmony postfix) and to ``Camera.main``
  after each scene load; config (``BepInEx/config/BetterAA.cfg``) is copied to
  the instance fields by ``UpdateSettings``.
* ``CTAA_PC.OnEnable``/``Awake``: ``firstFrame = swap = true``,
  ``frameCounter = 0``, ``depthTextureMode |= DepthNormals|Depth|MotionVectors``.
* ``CTAA_PC.OnPreCull`` -> ``jitterCam``: projection jitter from the fixed
  16-entry ``x_jit``/``y_jit`` tables.
* ``CTAA_PC.OnRenderImage``: ``SetCTAA_Parameters``, persistent history
  textures, pre-enhance (pass 1 of ``CTAA_Enhance_PC``), first-frame history
  init, ``CTAA_PC`` resolve with the ``rtAccum0``/``rtAccum1`` ping-pong, copy
  to ``txaaOut`` and the final blit to the destination.

The jitter arithmetic is byte-exact against all 16 captured iterations under
the Mono double-intermediate rule (``plugin_common``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .plugin_common import (F32, Op, PluginStateError, UnimplementedBranch, UnityRenderTexture, d,
                            mathf_lerp, mono_f32, pack_variables)

SOURCE = 'HS2_BetterAA.dll 1.6 sha256 f5ecfeea7a361f30733a1448dee42d5751690178a07796488e9e71a857f5a464'

X_JIT = np.array([0.5, -0.25, 0.75, -0.125, 0.625, 0.575, -0.875, 0.0625, -0.3, 0.75,
                  -0.25, -0.625, 0.325, 0.975, -0.075, 0.625], np.float32)
Y_JIT = np.array([0.33, -0.66, 0.51, 0.44, -0.77, 0.12, -0.55, 0.88, -0.83, 0.14,
                  0.71, -0.34, 0.87, -0.12, 0.75, 0.08], np.float32)
ADAPTIVE_RESOLVE = F32(3000.0)
ENHANCE_PASS = 1


@dataclass(frozen=True)
class CTAAConfig:
    """``CTAA_PC`` public tweak fields (copied from BetterAA.cfg by ``UpdateSettings``)."""
    temporal_stability: int          # TemporalStability : int
    hdr_response: float              # HdrResponse : float
    adaptive_enhance: float          # AdaptiveEnhance : float
    temporal_jitter_scale: float     # TemporalJitterScale : float
    micro_shimmer_reduction: float   # MicroShimmerReduction : float
    static_stability_power: float    # StaticStabilityPower : float


@dataclass
class CTAAState:
    """Mutable ``CTAA_PC`` instance state carried between camera renders."""
    enabled_field: bool              # _enabled : bool
    config: CTAAConfig
    first_frame: bool                # firstFrame : bool
    swap: bool                       # swap : bool
    frame_counter: int               # frameCounter : int
    jitter_scale: np.float32         # jitterScale : float (set by SetCTAA_Parameters / disabled branch)
    pre_enhance_enabled: bool = True
    pre_enhance_strength: np.float32 = F32(1.0)
    pre_enhance_clamp: np.float32 = F32(0.005)
    adaptive_resolve: np.float32 = F32(3000.0)
    # Persistent textures: Unity descriptors or None (fields rtAccum0, rtAccum1, txaaOut, afterPreEnhace)
    rt_accum0: Optional[UnityRenderTexture] = None
    rt_accum1: Optional[UnityRenderTexture] = None
    txaa_out: Optional[UnityRenderTexture] = None
    after_pre_enhance: Optional[UnityRenderTexture] = None
    generation: Dict[str, int] = field(default_factory=lambda: {'rtAccum0': 0, 'rtAccum1': 0, 'txaaOut': 0, 'afterPreEnhace': 0})

    @classmethod
    def from_audit_fields(cls, fields: List[dict]) -> 'CTAAState':
        v = {f['field']: f.get('value') for f in fields if f.get('read_succeeded')}
        need = ['_enabled', 'TemporalStability', 'HdrResponse', 'AdaptiveEnhance', 'TemporalJitterScale',
                'MicroShimmerReduction', 'StaticStabilityPower', 'firstFrame', 'swap', 'frameCounter', 'jitterScale',
                'PreEnhanceEnabled', 'preEnhanceStrength', 'preEnhanceClamp', 'AdaptiveResolve']
        missing = [k for k in need if k not in v]
        if missing:
            raise PluginStateError('CTAA_PC fields not captured: %s' % missing)
        cfg = CTAAConfig(int(v['TemporalStability']), float(v['HdrResponse']), float(v['AdaptiveEnhance']),
                         float(v['TemporalJitterScale']), float(v['MicroShimmerReduction']), float(v['StaticStabilityPower']))
        return cls(bool(v['_enabled']), cfg, bool(v['firstFrame']), bool(v['swap']), int(v['frameCounter']),
                   F32(v['jitterScale']), bool(v['PreEnhanceEnabled']), F32(v['preEnhanceStrength']),
                   F32(v['preEnhanceClamp']), F32(v['AdaptiveResolve']))

    def reset_temporal(self):
        """Bridge ``MakerRenderStyleService.ResetTemporal`` (= the OnEnable seed)."""
        self.first_frame = True
        self.swap = True
        self.frame_counter = 0


def ctaa_enabled(state: CTAAState, camera_orthographic: bool) -> bool:
    """``CTAA_Enabled`` getter: ``_enabled && !camera.orthographic``."""
    return bool(state.enabled_field) and not camera_orthographic


def set_ctaa_parameters(state: CTAAState):
    """``CTAA_PC.SetCTAA_Parameters``."""
    c = state.config
    state.pre_enhance_enabled = d(c.adaptive_enhance) > 0.01
    state.pre_enhance_strength = mathf_lerp(0.2, 2.0, c.adaptive_enhance)
    state.pre_enhance_clamp = mathf_lerp(0.005, 0.12, c.adaptive_enhance)
    state.jitter_scale = F32(c.temporal_jitter_scale)
    state.adaptive_resolve = ADAPTIVE_RESOLVE


def jitter_offsets(index: int, jitter_scale, pixel_width, pixel_height) -> Tuple[np.float32, np.float32]:
    """``jitterCam``: ``(x_jit[i]*jitterScale*2 - 1) / pixelRect.width`` (and y/height).

    ``num = x_jit[i] * jitterScale`` is a float local; the added term is evaluated
    in double and rounded when stored into ``projectionMatrix.m02``/``m12``
    (base value 0 for a symmetric perspective frustum).
    """
    num = mono_f32(d(X_JIT[index]) * d(jitter_scale))
    num2 = mono_f32(d(Y_JIT[index]) * d(jitter_scale))
    dx = (d(num) * 2.0 - 1.0) / d(pixel_width)
    dy = (d(num2) * 2.0 - 1.0) / d(pixel_height)
    return dx, dy


def jittered_projection(base_projection, dx: float, dy: float):
    """``projectionMatrix.m02 += dx; m12 += dy`` on Unity's column-major matrix."""
    m = [F32(v) for v in base_projection]
    m[8] = mono_f32(d(m[8]) + dx)    # m02: column 2, row 0
    m[9] = mono_f32(d(m[9]) + dy)    # m12: column 2, row 1
    return m


def on_pre_cull(state: CTAAState, *, camera_orthographic: bool, base_projection, pixel_rect_width, pixel_rect_height) -> Optional[dict]:
    """``CTAA_PC.OnPreCull``; returns the projection set on the camera.

    ``base_projection`` is the camera projection after ``ResetProjectionMatrix``
    (``Matrix4x4f::SetPerspective`` from fov/aspect/clip; ``unity_math``).
    ``ResetWorldToCameraMatrix`` makes the view follow the transform.
    """
    if not ctaa_enabled(state, camera_orthographic):
        return None
    index = state.frame_counter
    dx, dy = jitter_offsets(index, state.jitter_scale, pixel_rect_width, pixel_rect_height)
    proj = jittered_projection(base_projection, dx, dy)
    state.frame_counter = (state.frame_counter + 1) % 16
    return {'jitter_index': index, 'jitter_scale': F32(state.jitter_scale), 'm02_add': F32(dx), 'm12_add': F32(dy),
            'non_jittered_projection': [F32(v) for v in base_projection], 'projection': proj,
            'reset_world_to_camera': True, 'reset_projection': True}


def jittered_gpu_view_projection(non_jittered_vp, dx: float, dy: float, render_into_texture: bool = True):
    """Jittered ``unity_MatrixVP`` from the same render's ``_NonJitteredVP``.

    With Unity's ``MultiplyMatrices4x4`` order (``unity_math.multiply``) and a
    projection whose only non-zero row-0/1 entries are m00/m11 plus the jitter
    in column 2, ``VP[0][j] = NJ[0][j] + V2j*dx`` and ``VP[1][j] = NJ[1][j] +
    V2j*(-dy)`` (D3D render-texture Y flip), where ``V2j = -NJ[3][j]`` exactly.
    Used to byte-check jitter matrices against a capture independent of the view.
    """
    nj = np.asarray(non_jittered_vp, np.float32).reshape(-1)
    out = nj.copy()
    sy = F32(-1.0) if render_into_texture else F32(1.0)
    for c in range(4):
        v2 = F32(-nj[c * 4 + 3])
        out[c * 4 + 0] = F32(nj[c * 4 + 0] + F32(v2 * F32(dx)))
        out[c * 4 + 1] = F32(nj[c * 4 + 1] + F32(v2 * F32(F32(dy) * sy)))
    return out


def history_rt(name: str, source: UnityRenderTexture, filter_mode: str) -> UnityRenderTexture:
    """``new RenderTexture(source.width, source.height, 0, source.format)``; Clamp."""
    return UnityRenderTexture(name, source.width, source.height, 0, source.format, 'Default',
                              filter_mode=filter_mode, wrap_mode='Clamp', lifetime='persistent',
                              source='CTAA_PC.OnRenderImage')


def on_render_image(state: CTAAState, *, camera_orthographic: bool, source: UnityRenderTexture,
                    screen_width: int, screen_height: int) -> List[Op]:
    """Ordered ops of ``CTAA_PC.OnRenderImage(source, destination)``."""
    src = 'CTAA_PC.OnRenderImage'
    ops: List[Op] = []
    if not ctaa_enabled(state, camera_orthographic):
        state.jitter_scale = F32(0.0)
        ops.append(Op('blit', 'plugin', src, {'source': 'source', 'dest': 'destination', 'material': None, 'pass': -1}))
        return ops
    set_ctaa_parameters(state)
    for attr, name in (('rt_accum0', 'rtAccum0'), ('rt_accum1', 'rtAccum1'), ('txaa_out', 'txaaOut'),
                       ('after_pre_enhance', 'afterPreEnhace')):
        cur = getattr(state, attr)
        if cur is None or cur.width != source.width or cur.height != source.height:
            filt = source.filter_mode if name == 'afterPreEnhace' else 'Bilinear'
            new = history_rt(name, source, filt)
            if cur is not None:
                ops.append(Op('destroy', 'plugin', src, {'handles': [name]}))
            ops.append(Op('create_persistent', 'plugin', src, {'handle': name, 'rt': new}))
            setattr(state, attr, new)
            state.generation[name] += 1
    if state.pre_enhance_enabled:
        aex = mono_f32(1.0 / float(screen_width))
        aey = mono_f32(1.0 / float(screen_height))
        props = {'_AEXCTAA': aex, '_AEYCTAA': aey, '_AESCTAA': F32(state.pre_enhance_strength),
                 '_AEMAXCTAA': F32(state.pre_enhance_clamp)}
        ops.append(Op('material', 'plugin', src, {'material': 'mat_enhance', 'shader': 'Hidden/CTAA_Enhance_PC',
                                                   'properties': props, 'keywords': {}}))
        ops.append(Op('blit', 'plugin', src, {'source': 'source', 'dest': 'afterPreEnhace', 'material': 'Hidden/CTAA_Enhance_PC',
                                               'pass': ENHANCE_PASS}))
    else:
        ops.append(Op('blit', 'plugin', src, {'source': 'source', 'dest': 'afterPreEnhace', 'material': None, 'pass': -1}))
    if state.first_frame:
        ops.append(Op('blit', 'plugin', src, {'source': 'afterPreEnhace', 'dest': 'rtAccum0', 'material': None, 'pass': -1,
                                               'note': 'first-frame history initialisation'}))
        state.first_frame = False
    c = state.config
    control = np.array([F32(c.static_stability_power), F32(c.temporal_stability), F32(c.hdr_response),
                        F32(c.micro_shimmer_reduction)], np.float32)
    ctaa_props = {'_AdaptiveResolve': F32(state.adaptive_resolve), '_ControlParams': control}
    if state.swap:
        accum, write = 'rtAccum0', 'rtAccum1'
    else:
        accum, write = 'rtAccum1', 'rtAccum0'
    ctaa_props['_Accum'] = accum
    ops.append(Op('material', 'plugin', src, {'material': 'ctaaMat', 'shader': 'Hidden/CTAA_PC',
                                               'properties': ctaa_props, 'keywords': {}}))
    ops.append(Op('blit', 'plugin', src, {'source': 'afterPreEnhace', 'dest': write, 'material': 'Hidden/CTAA_PC', 'pass': -1,
                                           'textures': {'_Accum': accum, '_CameraMotionVectorsTexture': 'camera motion vectors (global)'}}))
    ops.append(Op('blit', 'plugin', src, {'source': write, 'dest': 'txaaOut', 'material': None, 'pass': -1}))
    ops.append(Op('blit', 'plugin', src, {'source': 'txaaOut', 'dest': 'destination', 'material': None, 'pass': -1}))
    state.swap = not state.swap
    return ops


def enhance_cbuffer(manifests, props: Dict[str, object], source: UnityRenderTexture) -> Dict[str, dict]:
    program = manifests.program('Hidden/CTAA_Enhance_PC', ENHANCE_PASS, 'progFragment', [])
    values = dict(props)
    values['_MainTex_TexelSize'] = source.texel_size()
    return pack_variables(program, values)


def resolve_cbuffer(manifests, props: Dict[str, object], source: UnityRenderTexture) -> Dict[str, dict]:
    program = manifests.program('Hidden/CTAA_PC', 0, 'progFragment', [])
    values = {k: v for k, v in props.items() if k != '_Accum'}
    values['_MainTex_TexelSize'] = source.texel_size()
    return pack_variables(program, values)


def depth_texture_mode_request() -> List[str]:
    """``OnEnable``: ``depthTextureMode |= DepthNormals | Depth | MotionVectors``."""
    return ['DepthNormals', 'Depth', 'MotionVectors']
