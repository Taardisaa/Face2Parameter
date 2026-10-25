"""Unity PostProcessing Stack v2 (package 2.1.7) as installed in HS2, as a frame-program generator.

Reimplements, from the installed ``Unity.Postprocessing.Runtime.dll`` (SHA-256
``f18eac429f656764c3abb1ab905db18fd374c4bbacd475ac76d9bc2718e11a35``; identical
identifier set to tag 2.1.7 / commit 283c8187), the legacy (built-in pipeline) path
``PostProcessLayer.OnPreCull -> BuildCommandBuffers`` and the two camera command
buffers it fills:

* ``BeforeImageEffectsOpaque`` ("Opaque Only Post-processing"):
  ``MultiScaleVO.RenderAfterOpaque`` -- depth copy (``Hidden/PostProcessing/MultiScaleVO``
  pass 0), 10 compute dispatches (Downsample1, Downsample2, Render x4 interleaved,
  Upsample x4) and the forward composite (pass 2) into the camera target.
* ``BeforeImageEffects`` ("Post-processing"): initial ``BuiltinBlit`` with
  ``CopyStd`` (stopNaNPropagation pass 1), ``RenderBuiltins`` (DoF: CoC, prefilter,
  bokeh kernel, postfilter, combine; ColorGrading HDR 3D LUT bake with
  ``Lut3DBaker``; Uber), ``RenderFinalPass`` (SMAA edge/weights/blend, Dithering,
  FinalPass) into the camera target.

and the immediate work done inside ``OnPreCull``: ``Shader.SetGlobalFloat(_RenderViewportScaleFactor)``,
volume blending (``ppv2_settings.VolumeBlender``) and the ``Texture2D.Apply`` upload of
the ColorGrading curves texture (staging Map + CopySubresourceRegion).

Commands are emitted through ``src/render/frame_program.ProgramBuilder``.  Inputs owned
by other generators (camera target, camera depth, ``_CameraDepthTexture``, built-in
camera shader globals such as ``_ZBufferParams``, shared constant-buffer shadows and
the temporary-RT pool) are passed in ``PPv2FrameInputs``.  Effects that are not part of
the HS2 Maker chain raise ``PPv2Unimplemented`` when they become active, instead of
being skipped.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.render.frame_program import DispatchState, DrawState, ProgramBuilder
from .ppv2_settings import VolumeBlender
from .ppv2_unity import (F, RTDescriptor, SystemRandom, TemporaryRTPool, TextureRef, UnityConstantBufferPool,
                         blend_desc, depth_desc, f32_bytes, float_to_half_bytes, mathf_ceil_to_int, mathf_exp,
                         mathf_max, mathf_min, mathf_pow, mathf_sqrt, raster_desc, resolve_rt_format, sampler_desc,
                         dv, r4)

GENERATOR = 'hs2pipe.ppv2/1'
PPV2_ASSEMBLY_SHA256 = 'f18eac429f656764c3abb1ab905db18fd374c4bbacd475ac76d9bc2718e11a35'

SHADER = {
    'copyStd': 'Hidden/PostProcessing/CopyStd', 'depthOfField': 'Hidden/PostProcessing/DepthOfField',
    'finalPass': 'Hidden/PostProcessing/FinalPass', 'uber': 'Hidden/PostProcessing/Uber',
    'smaa': 'Hidden/PostProcessing/SubpixelMorphologicalAntialiasing', 'multiScaleAO': 'Hidden/PostProcessing/MultiScaleVO',
}
ANTIALIASING = {'None': 0, 'FastApproximateAntialiasing': 1, 'SubpixelMorphologicalAntialiasing': 2,
                'TemporalAntialiasing': 3}


class PPv2Unimplemented(NotImplementedError):
    """A PPv2 branch that the HS2 chain does not use and this generator does not reproduce."""


# =============================================================== shader source
class ManifestShaderSource:
    """Installed shader programs from ``extract_shader_programs.py`` manifests and
    compute kernels from ``extract_compute_shaders.py``.

    Interface expected from ``hs2pipe.shader_library`` (sibling) -- this class is
    the adapter used until it exists:
      ``program(shader, subshader, pass_index, stage, keywords) -> dict``
      ``pass_state(shader, subshader, pass_index) -> dict``
      ``kernel(compute_name, kernel_name) -> dict``
    Variant selection: Unity picks the variant whose keyword set equals the enabled
    keywords restricted to the keywords the pass declares (exact match required)."""

    def __init__(self, manifests, compute_manifests):
        self.shaders = {}
        for m in manifests:
            path = Path(m)
            doc = json.loads(path.read_text(encoding='utf-8'))
            for s in doc['shaders']:
                self.shaders.setdefault(s['name'], []).append((path.parent, doc, s))
        self.compute = {}
        for m in compute_manifests:
            path = Path(m)
            doc = json.loads(path.read_text(encoding='utf-8'))
            for c in doc['compute_shaders']:
                self.compute[c['name']] = (path.parent, doc, c)

    def _pass(self, shader, subshader, pass_index):
        for root, doc, s in self.shaders[shader]:
            for p in s['passes']:
                if p['subshader'] == subshader and p['pass'] == pass_index:
                    return root, doc, s, p
        raise KeyError('%s subshader %d pass %d not exported' % (shader, subshader, pass_index))

    def pass_state(self, shader, subshader, pass_index):
        return self._pass(shader, subshader, pass_index)[3]['state']

    def program(self, shader, subshader, pass_index, stage, keywords):
        root, doc, s, p = self._pass(shader, subshader, pass_index)
        progs = [g for g in p['programs'] if g['stage'] == stage]
        space = set()
        for g in progs:
            space.update(g['keywords'])
        want = sorted(set(keywords) & space)
        hits = [g for g in progs if sorted(g['keywords']) == want]
        if not hits:
            raise KeyError('%s pass %d %s: no variant for keywords %r' % (shader, pass_index, stage, want))
        shas = {g['dxbc_sha256'] for g in hits}
        if len(shas) != 1:
            raise KeyError('%s pass %d %s: ambiguous variants %r' % (shader, pass_index, stage, shas))
        g = hits[0]
        data = (root / g['dxbc']).read_bytes()
        if hashlib.sha256(data).hexdigest() != g['dxbc_sha256']:
            raise ValueError('exported DXBC hash mismatch for ' + g['dxbc'])
        return {'dxbc': data, 'sha256': g['dxbc_sha256'], 'layout': unity_layout(g),
                'source': {'shader': shader, 'subshader': subshader, 'pass': pass_index, 'keywords': g['keywords'],
                           'asset': doc.get('bundle_path'), 'asset_sha256': doc.get('bundle_sha256'),
                           'path_id': s.get('asset_path_id'), 'blob_index': g['blob_index']}}

    def kernel(self, compute, kernel_name):
        root, doc, c = self.compute[compute]
        variants = [v for v in c['variants'] if v['targetRenderer'] == 2]
        if len(variants) != 1:
            raise KeyError('%s: expected exactly one D3D11 variant' % compute)
        v = variants[0]
        k = next(k for k in v['kernels'] if k['name'] == kernel_name)
        data = (root / k['dxbc']).read_bytes()
        if hashlib.sha256(data).hexdigest() != k['dxbc_sha256']:
            raise ValueError('exported compute DXBC hash mismatch for ' + k['dxbc'])
        return {'dxbc': data, 'sha256': k['dxbc_sha256'], 'kernel': k, 'cbuffers': {cb['name']: cb for cb in v['constant_buffers']},
                'source': {'compute': compute, 'kernel': kernel_name, 'asset': doc['asset_path'],
                           'asset_sha256': doc['asset_sha256'], 'path_id': c['path_id']}}


def unity_layout(program):
    """Unity per-program binding metadata: cbuffers (name, size, variables with type/
    offset/dims), textures (name, slot, sampler slot), cbuffer slots."""
    b = program['bindings']
    names = {i: n for n, i in program['name_indices']}
    cbs = {}
    for c in b['m_ConstantBuffers']:
        if c['m_MatrixParams'] or c.get('m_StructParams'):
            raise PPv2Unimplemented('matrix/struct cbuffer parameters in a PPv2 program')
        cbs[names[c['m_NameIndex']]] = {'size': c['m_Size'], 'variables': [
            {'name': names[v['m_NameIndex']], 'offset': v['m_Index'], 'type': v['m_Type'], 'dim': v['m_Dim'],
             'array': v['m_ArraySize']} for v in c['m_VectorParams']]}
    if b['m_VectorParams'] or b['m_MatrixParams'] or b.get('m_BufferParams') or b.get('m_UAVParams'):
        raise PPv2Unimplemented('global (non-cbuffer) parameters / buffers in a PPv2 program')
    return {'cbuffers': cbs,
            'cb_slots': {names[c['m_NameIndex']]: c['m_Index'] for c in b['m_ConstantBufferBindings']},
            'textures': [{'name': names[t['m_NameIndex']], 'slot': t['m_Index'], 'sampler': t['m_SamplerIndex'],
                          'dim': t['m_Dim']} for t in b['m_TextureParams']],
            'inline_samplers': b['m_Samplers']}


# =============================================================== asset textures
class PPv2Assets:
    """PostProcessResources textures (blueNoise64, SMAA LUTs) read from the
    installed asset file; sampler attributes from each Texture2D's TextureSettings."""

    TEXFMT = {1: ('A8_UNORM', 1), 3: ('R8G8B8A8_UNORM', 3), 4: ('R8G8B8A8_UNORM', 4)}

    def __init__(self, scene_state, data_dir=None):
        self.scene_state = scene_state
        self.res = scene_state['resources']
        self.data_dir = Path(data_dir) if data_dir else Path(r'E:\HoneySelect2_ArcticFox\HoneySelect2_Data')
        self._env = None
        self._objs = None

    def _load(self):
        if self._objs is None:
            import UnityPy
            path = self.data_dir / self.res['file']
            self._env = UnityPy.load(str(path))
            self._objs = {o.path_id: o for o in self._env.objects}
            self.asset_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._objs

    def texture(self, entry):
        """(resource desc, initial bytes, row pitch, attributes) of a Texture2D."""
        obj = self._load()[entry['path_id']]
        t = obj.read_typetree()
        fmt = t['m_TextureFormat']
        if fmt not in self.TEXFMT:
            raise PPv2Unimplemented('TextureFormat %d' % fmt)
        if t['m_MipCount'] != 1:
            raise PPv2Unimplemented('mipmapped PPv2 resource textures')
        data = bytes(t['image data']) if t.get('image data') else None
        if not data:
            # m_StreamData: raw image bytes stored in the .resS companion file
            sd = t['m_StreamData']
            with open(self.data_dir / Path(sd['path']).name, 'rb') as fh:
                fh.seek(sd['offset'])
                data = fh.read(sd['size'])
            if len(data) != sd['size']:
                raise ValueError('short stream data for %s' % t['m_Name'])
        w, h = t['m_Width'], t['m_Height']
        dxgi, src_bpp = self.TEXFMT[fmt]
        if fmt == 3:   # RGB24 -> RGBA32 with alpha 0xFF (D3D11 has no 24-bit format)
            px = np.frombuffer(data, np.uint8)[:w * h * 3].reshape(-1, 3)
            data = np.concatenate([px, np.full((len(px), 1), 255, np.uint8)], 1).tobytes()
            bpp = 4
        else:
            data = data[:w * h * src_bpp]
            bpp = src_bpp
        from .ppv2_unity import DXGI
        srgb = t.get('m_ColorSpace', 0) == 1 and dxgi == 'R8G8B8A8_UNORM'
        fmt_id = DXGI['R8G8B8A8_UNORM_SRGB'] if srgb else DXGI[dxgi]
        desc = {'Width': w, 'Height': h, 'MipLevels': 1, 'ArraySize': 1, 'Format': fmt_id,
                'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 8, 'CPUAccessFlags': 0, 'MiscFlags': 0}
        ts = t['m_TextureSettings']
        attrs = {'filter_mode': {0: 'Point', 1: 'Bilinear', 2: 'Trilinear'}[ts['m_FilterMode']],
                 'wrap_mode': {0: 'Repeat', 1: 'Clamp', 2: 'Mirror', 3: 'MirrorOnce'}[ts['m_WrapU']],
                 'aniso_level': ts['m_Aniso'], 'name': t['m_Name']}
        if ts['m_WrapU'] != ts['m_WrapV']:
            raise PPv2Unimplemented('per-axis wrap modes')
        return desc, data, w * bpp, attrs


# =============================================================== inputs / state
@dataclass
class CameraState:
    """Camera fields read by PPv2 (``Camera`` component state at OnPreCull)."""
    pixel_width: int
    pixel_height: int
    near_clip: float
    far_clip: float
    projection_m00: float          # camera.projectionMatrix[0,0] (as used by MultiScaleVO)
    allow_hdr: bool = True
    rendering_path: str = 'Forward'     # actualRenderingPath
    uses_reversed_z: bool = True        # SystemInfo.usesReversedZBuffer (D3D11)
    camera_type: str = 'Game'
    stereo: bool = False
    msaa: bool = False


@dataclass
class LayerConfig:
    """PostProcessLayer serialized fields (+ DHH overrides) of the rendering camera."""
    volume_layer: int = 256
    stop_nan_propagation: bool = True
    antialiasing_mode: int = 2
    smaa_quality: int = 2
    fxaa_fast_mode: bool = False
    fxaa_keep_alpha: bool = True
    break_before_color_grading: bool = False
    fog_enabled: bool = True
    custom_bundles: tuple = ()
    debug_overlay: int = 0

    @classmethod
    def from_scene(cls, layer_row, overrides=None):
        c = cls(volume_layer=layer_row['volumeLayer'], stop_nan_propagation=layer_row['stopNaNPropagation'],
                antialiasing_mode=layer_row['antialiasingMode'],
                smaa_quality=layer_row['subpixelMorphologicalAntialiasing']['quality'],
                fxaa_fast_mode=layer_row['fastApproximateAntialiasing']['fastMode'],
                fxaa_keep_alpha=layer_row['fastApproximateAntialiasing']['keepAlpha'],
                break_before_color_grading=layer_row['breakBeforeColorGrading'],
                fog_enabled=layer_row['fog']['enabled'],
                custom_bundles=tuple(layer_row['m_BeforeTransparentBundles'] + layer_row['m_BeforeStackBundles']
                                     + layer_row['m_AfterStackBundles']))
        for k, v in (overrides or {}).items():
            if k == 'antialiasingMode':
                c.antialiasing_mode = v
            elif k == 'subpixelMorphologicalAntialiasing.quality':
                c.smaa_quality = v
            elif k == 'fastApproximateAntialiasing.fastMode':
                c.fxaa_fast_mode = v
            elif k.startswith('temporalAntialiasing.'):
                pass   # only read when TAA is the active mode (raises below)
            else:
                raise KeyError(k)
        return c


@dataclass
class RenderSettingsState:
    fog: bool = False
    fog_density: float = 0.0
    fog_start: float = 0.0
    fog_end: float = 0.0


@dataclass
class PPv2FrameInputs:
    """Per-render resources and state owned by other generators."""
    camera_target: TextureRef                # BuiltinRenderTextureType.CameraTarget (color)
    camera_depth: TextureRef                 # depth buffer bound with CameraTarget (dsv)
    camera_depth_texture: TextureRef         # _CameraDepthTexture (srv)
    globals: dict                            # Unity global shader properties (name -> value | TextureRef)
    global_keywords: tuple = ()              # enabled global shader keywords
    front_ccw: object = None                 # callable(buffer, draw_index) -> bool (gap raster_front_ccw)


def default_front_ccw(buffer, draw_index):
    """Observed in every capture segment: the first fullscreen draw of the
    BeforeImageEffectsOpaque buffer (MSVO depth copy) has FrontCounterClockwise
    false, every other PPv2 draw true.  UnityPlayer's flip/invert-culling rule is
    not transcribed (gap ``raster_front_ccw``); CullMode is NONE for all of them."""
    return not (buffer == 'opaque' and draw_index == 0)


class Dithering:
    """``Dithering`` (private, created once per PostProcessLayer by ``Init``)."""

    def __init__(self, prior_calls=0, state=None):
        self.index = 0
        self.random = SystemRandom(1234)
        self.calls = 0
        if state is not None:            # (m_NoiseTextureIndex, System.Random) from scene state
            self.index, self.random = state
            return
        for _ in range(prior_calls):
            self.step(64)

    def step(self, count):
        self.index += 1
        if self.index >= count:
            self.index = 0
        z = F(self.random.next_double())
        w = F(self.random.next_double())
        self.calls += 1
        return self.index, z, w


# =============================================================== recorded commands
@dataclass
class Sheet:
    """PropertySheet: shader + enabled keywords + MaterialPropertyBlock snapshot."""
    shader: str
    keywords: tuple = ()
    floats: dict = field(default_factory=dict)
    textures: dict = field(default_factory=dict)

    def snapshot(self):
        return Sheet(self.shader, tuple(self.keywords), dict(self.floats), dict(self.textures))


class CommandBufferRec:
    """A Unity CommandBuffer recorded at OnPreCull; executed later by ``Executor``."""

    def __init__(self, name):
        self.name = name
        self.ops = []

    def get_temporary_rt(self, name_id, desc: RTDescriptor, filter_mode='Bilinear'):
        self.ops.append(('get_temp', name_id, desc, filter_mode))

    def release_temporary_rt(self, name_id):
        self.ops.append(('release_temp', name_id))

    def set_global_texture(self, name, target):
        self.ops.append(('set_global_texture', name, target))

    def blit_triangle(self, source, destination, sheet: Sheet, pass_index, clear=False, load=False):
        self.ops.append(('blit', source, destination, sheet.snapshot(), pass_index, clear, load))

    def builtin_blit(self, source, destination, shader, pass_index):
        self.ops.append(('builtin_blit', source, destination, shader, pass_index))

    def set_compute_param(self, compute, name, values):
        self.ops.append(('cs_param', compute, name, tuple(F(v) for v in values)))

    def dispatch(self, compute, kernel, textures, groups):
        self.ops.append(('dispatch', compute, kernel, dict(textures), tuple(groups)))


# Target references inside recorded commands
CAMERA_TARGET = ('camera_target',)
NONE_TARGET = ('none',)


def temp(name):
    return ('temp', name)


def persistent(key):
    return ('persistent', key)


# =============================================================== the layer
class PostProcessLayerGen:
    """One camera's PostProcessLayer + its bundles' renderer state, persisting across renders."""

    def __init__(self, builder: ProgramBuilder, shaders: ManifestShaderSource, assets: PPv2Assets,
                 layer: LayerConfig, *, cb_pool: UnityConstantBufferPool | None = None,
                 rt_pool: TemporaryRTPool | None = None, dithering_prior_calls=0, dithering_state=None,
                 curve_pixels=None, anisotropic_filtering='ForceEnable'):
        self.b = builder
        self.shaders = shaders
        self.assets = assets
        self.cfg = layer
        self.cb_pool = cb_pool or UnityConstantBufferPool()
        self.rt_pool = rt_pool or TemporaryRTPool(builder)
        self.aniso = anisotropic_filtering
        self.blender = VolumeBlender()
        self.dithering = Dithering(dithering_prior_calls, dithering_state)
        self.nan_killed = False
        self.persistent = {}
        self.cs_shadows = {}
        # ColorGradingRenderer.m_Pixels (persistent; new Color[256] at construction)
        self.curve_pixels = list(curve_pixels) if curve_pixels is not None else [(F(0), F(0), F(0), F(0))] * 256
        self._objects = {}
        self.log = []
        if layer.custom_bundles:
            raise PPv2Unimplemented('custom PPv2 bundles: %r' % (layer.custom_bundles,))
        if layer.debug_overlay:
            raise PPv2Unimplemented('PostProcessDebugLayer overlays')

    # ------------------------------------------------------------ object cache
    def _obj(self, key, make):
        if key not in self._objects:
            self._objects[key] = make()
        return self._objects[key]

    def _shader(self, stage, prog):
        return self._obj(('shader', stage, prog['sha256']),
                         lambda: self.b.shader(stage, prog['dxbc'], source=prog['source']))

    def _sampler(self, desc):
        return self._obj(('sampler', json.dumps(desc, sort_keys=True)), lambda: self.b.sampler(desc))

    def _state(self, table, desc):
        fn = {'blend': self.b.blend_state, 'depth': self.b.depth_state, 'raster': self.b.raster_state}[table]
        return self._obj((table, json.dumps(desc, sort_keys=True)), lambda: fn(desc))

    # ------------------------------------------------------- persistent objects
    def fullscreen_triangle(self):
        """``RuntimeUtilities.fullscreenTriangle`` Mesh: 3 float3 positions, u16 indices 0,1,2."""
        def make():
            vb = struct.pack('<9f', -1, -1, 0, -1, 3, 0, 3, -1, 0)
            ib = struct.pack('<3H', 0, 1, 2)
            v = self.b.buffer({'ByteWidth': 36, 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0, 'MiscFlags': 0,
                               'StructureByteStride': 0}, initial=vb, name='Fullscreen Triangle')
            i = self.b.buffer({'ByteWidth': 6, 'Usage': 0, 'BindFlags': 2, 'CPUAccessFlags': 0, 'MiscFlags': 0,
                               'StructureByteStride': 0}, initial=ib, name='Fullscreen Triangle')
            return v, i
        return self._obj('fullscreen_triangle', make)

    def blit_quad(self):
        """Unity built-in ``CommandBuffer.Blit`` quad (POSITION float3 + TEXCOORD0 float2,
        indices 0,1,2,0,2,3), bytes as captured.  Unity streams it through its dynamic
        VB/IB ring; ring placement is not reproduced (gap ``blit_dynamic_ring``)."""
        def make():
            verts = [(0, 0, 0.1, 0, 0), (0, 1, 0.1, 0, 1), (1, 1, 0.1, 1, 1), (1, 0, 0.1, 1, 0)]
            vb = b''.join(struct.pack('<5f', *[float(F(x)) for x in v]) for v in verts)
            ib = struct.pack('<6H', 0, 1, 2, 0, 2, 3)
            v = self.b.buffer({'ByteWidth': len(vb), 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0, 'MiscFlags': 0,
                               'StructureByteStride': 0}, initial=vb, name='Unity Blit quad')
            i = self.b.buffer({'ByteWidth': len(ib), 'Usage': 0, 'BindFlags': 2, 'CPUAccessFlags': 0, 'MiscFlags': 0,
                               'StructureByteStride': 0}, initial=ib, name='Unity Blit quad')
            return v, i
        return self._obj('blit_quad', make)

    def white_texture(self):
        """``RuntimeUtilities.whiteTexture``: Texture2D(1,1,ARGB32,false), linear
        color space project -> sRGB storage; default Bilinear/Repeat/aniso 1."""
        def make():
            desc = {'Width': 1, 'Height': 1, 'MipLevels': 1, 'ArraySize': 1, 'Format': 29,
                    'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 8, 'CPUAccessFlags': 0, 'MiscFlags': 0}
            r = self.b.texture('tex2d', desc, initial=[(0, b'\xff\xff\xff\xff', 4, 4)], name='White Texture')
            srv = self.b.view('srv', r, {'Format': 29, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}})
            return TextureRef(r, srv, 1, 1, filter_mode='Bilinear', wrap_mode='Repeat', aniso_level=1,
                              name='White Texture', resource_desc=desc)
        return self._obj('white', make)

    def asset_texture(self, entry):
        def make():
            desc, data, pitch, attrs = self.assets.texture(entry)
            r = self.b.texture('tex2d', desc, initial=[(0, data, pitch, pitch * desc['Height'])], name=attrs['name'])
            srv = self.b.view('srv', r, {'Format': desc['Format'], 'ViewDimension': 4,
                                         'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}})
            return TextureRef(r, srv, desc['Width'], desc['Height'], filter_mode=attrs['filter_mode'],
                              wrap_mode=attrs['wrap_mode'], aniso_level=attrs['aniso_level'], name=attrs['name'],
                              resource_desc=desc)
        return self._obj(('asset', entry['path_id']), make)

    def curves_texture(self):
        """``ColorGradingRenderer.m_GradingCurves``: Texture2D(128,2,RGBAHalf,linear),
        Clamp/Bilinear/aniso 0, plus the staging texture used by Texture2D.Apply."""
        def make():
            desc = {'Width': 128, 'Height': 2, 'MipLevels': 1, 'ArraySize': 1, 'Format': 10,
                    'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 8, 'CPUAccessFlags': 0, 'MiscFlags': 0}
            r = self.b.texture('tex2d', desc, name='Internal Curves Texture')
            srv = self.b.view('srv', r, {'Format': 10, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}})
            st = dict(desc, Usage=3, BindFlags=0, CPUAccessFlags=0x10000)
            s = self.b.texture('tex2d', st, name='Internal Curves Texture staging')
            return TextureRef(r, srv, 128, 2, filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0,
                              name='Internal Curves Texture', resource_desc=desc), s
        return self._obj('curves', make)

    def log_lut(self):
        """``ColorGradingRenderer.CheckInternalLogLut``: 33^3 ARGBHalf, Linear, Tex3D,
        randomWrite, Bilinear/Clamp/aniso 0."""
        def make():
            d = RTDescriptor(33, 33, 'ARGBHalf', srgb=False, dimension='Tex3D', volume_depth=33, enable_random_write=True)
            kind, res, srv, rtv, uav = d.d3d()
            r = self.b.texture(kind, res, name='Color Grading Log Lut')
            return TextureRef(r, self.b.view('srv', r, srv), 33, 33, depth=33, uav=self.b.view('uav', r, uav),
                              filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0, name='Color Grading Log Lut',
                              descriptor=d, resource_desc=res)
        return self._obj('log_lut', make)

    def ambient_only_ao(self, w, h):
        """``MultiScaleVO.CheckAOTexture``: new RenderTexture(w,h,0,R8,Linear) Point,
        randomWrite.  Wrap Clamp and anisoLevel are the RenderTexture defaults
        (observed sampler 230 = point/clamp)."""
        key = ('ao', w, h)

        def make():
            d = RTDescriptor(w, h, 'R8', srgb=False, enable_random_write=True)
            kind, res, srv, rtv, uav = d.d3d()
            r = self.b.texture(kind, res, name='MSVO AmbientOnlyAO')
            return TextureRef(r, self.b.view('srv', r, srv), w, h, rtv=self.b.view('rtv', r, rtv),
                              uav=self.b.view('uav', r, uav), filter_mode='Point', wrap_mode='Clamp', aniso_level=1,
                              name='MSVO AmbientOnlyAO', descriptor=d, resource_desc=res)
        return self._obj(key, make)

    # ------------------------------------------------------------ OnPreCull
    def on_pre_cull(self, camera: CameraState, volumes, frame, inputs: PPv2FrameInputs,
                    render_settings: RenderSettingsState):
        """``PostProcessLayer.OnPreCull`` (legacy path).  Emits the immediate uploads
        and returns the two recorded command buffers."""
        if camera.stereo:
            raise PPv2Unimplemented('stereo rendering')
        inputs.globals['_RenderViewportScaleFactor'] = F(1)       # Shader.SetGlobalFloat (non-stereo)
        self.camera = camera
        self.inputs = inputs
        self.render_settings = render_settings
        ctx = {'width': camera.pixel_width, 'height': camera.pixel_height, 'screenWidth': camera.pixel_width,
               'screenHeight': camera.pixel_height, 'source_format':
               'DefaultHDR' if camera.allow_hdr else 'Default'}
        self.ctx = ctx
        opaque = CommandBufferRec('Opaque Only Post-processing')
        post = CommandBufferRec('Post-processing')
        # SetupContext -> SetLegacyCameraFlags is evaluated with the previous blend
        # state; it only adds depthTextureMode flags (owned by the camera generator).
        applied = self.blender.update(self.cfg.volume_layer, volumes, frame)
        self.log.append({'volume_blend': applied, 'frame': frame})
        S = self.blender.layer
        self.settings = S
        self.target_pool = 0
        self.temps = {}
        if self.cfg.antialiasing_mode == ANTIALIASING['TemporalAntialiasing']:
            raise PPv2Unimplemented('PPv2 TemporalAntialiasing')
        # ---- ambient occlusion
        ao = S['AmbientOcclusion']
        ao_on = self._ao_enabled(ao)
        ambient_only = bool(ao['ambientOnly']) and camera.rendering_path == 'DeferredShading' and camera.allow_hdr
        if ao_on and ambient_only:
            raise PPv2Unimplemented('ambient-only (deferred) AO')
        if ao_on:
            if ao['mode'] != 1:
                raise PPv2Unimplemented('ScalableAO')
            self._msvo_render_after_opaque(opaque, ao)
        # ---- opaque-only effects
        ssr = S['ScreenSpaceReflections']
        if ssr['enabled'] and camera.rendering_path == 'DeferredShading':
            raise PPv2Unimplemented('ScreenSpaceReflections')
        if self.cfg.fog_enabled and render_settings.fog and camera.rendering_path == 'DeferredShading':
            raise PPv2Unimplemented('deferred fog')
        # no BeforeTransparent bundles (custom bundles rejected in __init__)
        # ---- main buffer
        fmt = ctx['source_format']
        if fmt not in ('DefaultHDR', 'ARGBHalf'):
            self.nan_killed = True     # !isFloatingPointFormat
        num2 = self._pool_get()
        self._get_screen_temp(post, num2, fmt, srgb=True)
        post.builtin_blit(CAMERA_TARGET, temp(num2), SHADER['copyStd'], 1 if self.cfg.stop_nan_propagation else 0)
        if not self.nan_killed:
            self.nan_killed = self.cfg.stop_nan_propagation
        self.source = temp(num2)
        self.destination = CAMERA_TARGET
        self._render(post)
        post.release_temporary_rt(num2)
        return opaque, post

    # ------------------------------------------------------------ helpers
    def _pool_get(self):
        """``TargetPool.Get`` after ``Reset`` (names are only identifiers here)."""
        n = '_TargetPool%d' % self.target_pool
        self.target_pool += 1
        return n

    def _desc(self, fmt, srgb, width=None, height=None, random_write=False):
        return RTDescriptor(width or self.ctx['width'], height or self.ctx['height'], resolve_rt_format(fmt), srgb=srgb,
                            enable_random_write=random_write)

    def _get_screen_temp(self, cmd, name, fmt, srgb=True, filter_mode='Bilinear', width=0, height=0):
        """``PostProcessRenderContext.GetScreenSpaceTemporaryRT`` (descriptor from the
        source descriptor; sRGB = readWrite != Linear)."""
        cmd.get_temporary_rt(name, self._desc(fmt, srgb, width or None, height or None), filter_mode)

    def _ao_enabled(self, ao):
        return bool(ao['enabled']) and ao['intensity'] > F(0)

    # ------------------------------------------------------------ MultiScaleVO
    def _msvo_render_after_opaque(self, cmd, ao):
        cam = self.camera
        sheet = Sheet(SHADER['multiScaleAO'])
        c = ao['color']
        sheet.floats['_AOColor'] = tuple(r4(1.0 - dv(x)) for x in c)   # Color.white - color (operator-)
        if cam.rendering_path == 'Forward' and self.render_settings.fog:
            sheet.keywords = ('APPLY_FORWARD_FOG',)
            rs = self.render_settings
            sheet.floats['_FogParams'] = (F(rs.fog_density), F(rs.fog_start), F(rs.fog_end))
        ao_tex = persistent(('ao', self.ctx['width'], self.ctx['height']))
        self.ambient_only_ao(self.ctx['width'], self.ctx['height'])
        self._generate_ao_map(cmd, ao, sheet, ao_tex)
        cmd.set_global_texture('_MSVOcclusionTexture', ao_tex)
        cmd.blit_triangle(NONE_TARGET, CAMERA_TARGET, sheet, 2, load=True)

    def _generate_ao_map(self, cmd, ao, sheet, destination):
        cam = self.camera
        if cam.msaa:
            raise PPv2Unimplemented('MSAA MultiScaleVO kernels')
        widths = [cam.pixel_width]
        heights = [cam.pixel_height]
        for i in range(1, 7):
            n = 1 << i
            widths.append((widths[0] + (n - 1)) // n)
            heights.append((heights[0] + (n - 1)) // n)
        W, H = widths, heights

        def alloc(name, level, fmt):
            cmd.get_temporary_rt(name, RTDescriptor(W[level], H[level], fmt, srgb=False, enable_random_write=True), 'Point')

        def alloc_array(name, level, fmt):
            cmd.get_temporary_rt(name, RTDescriptor(W[level], H[level], fmt, srgb=False, dimension='Tex2DArray',
                                                    volume_depth=16, enable_random_write=True), 'Point')
        alloc('_LinearDepth', 0, 'RHalf')
        for i in range(1, 5):
            alloc('_LowDepth%d' % i, i, 'RFloat')
        for i in range(1, 5):
            alloc_array('_TiledDepth%d' % i, i + 2, 'RHalf')
        for i in range(1, 5):
            alloc('_Occlusion%d' % i, i, 'R8')
        for i in range(1, 4):
            alloc('_Combined%d' % i, i, 'R8')
        # PushDownsampleCommands: forward path has no resolved depth -> DepthCopy pass 0
        if cam.rendering_path == 'DeferredShading':
            raise PPv2Unimplemented('ResolvedDepth MSVO source')
        cmd.get_temporary_rt('_DepthCopy', RTDescriptor(W[0], H[0], 'RFloat', srgb=False), 'Point')
        cmd.blit_triangle(NONE_TARGET, temp('_DepthCopy'), sheet, 0)
        far_over_near = r4(dv(cam.far_clip) / dv(cam.near_clip))
        zbp = (r4(dv(far_over_near) - 1.0), F(1), F(0), F(0)) if cam.uses_reversed_z else \
            (r4(1.0 - dv(far_over_near)), far_over_near, F(0), F(0))
        cmd.set_compute_param('MultiScaleVODownsample1', 'ZBufferParams', zbp)
        cmd.dispatch('MultiScaleVODownsample1', 'MultiScaleVODownsample1',
                     {'LinearZ': temp('_LinearDepth'), 'DS2x': temp('_LowDepth1'), 'DS4x': temp('_LowDepth2'),
                      'DS2xAtlas': temp('_TiledDepth1'), 'DS4xAtlas': temp('_TiledDepth2'), 'Depth': temp('_DepthCopy')},
                     (W[4], H[4], 1))
        cmd.release_temporary_rt('_DepthCopy')
        cmd.dispatch('MultiScaleVODownsample2', 'MultiScaleVODownsample2',
                     {'DS4x': temp('_LowDepth2'), 'DS8x': temp('_LowDepth3'), 'DS16x': temp('_LowDepth4'),
                      'DS8xAtlas': temp('_TiledDepth3'), 'DS16xAtlas': temp('_TiledDepth4')}, (W[6], H[6], 1))
        tan_half = r4(1.0 / dv(cam.projection_m00))
        for i in range(1, 5):
            self._msvo_render(cmd, ao, '_TiledDepth%d' % i, '_Occlusion%d' % i, (W[i + 2], H[i + 2], 16), tan_half)
        size = lambda lv: (W[lv], H[lv])   # noqa: E731
        self._msvo_upsample(cmd, ao, '_LowDepth4', '_Occlusion4', '_LowDepth3', '_Occlusion3', temp('_Combined3'), size(4), size(3))
        self._msvo_upsample(cmd, ao, '_LowDepth3', '_Combined3', '_LowDepth2', '_Occlusion2', temp('_Combined2'), size(3), size(2))
        self._msvo_upsample(cmd, ao, '_LowDepth2', '_Combined2', '_LowDepth1', '_Occlusion1', temp('_Combined1'), size(2), size(1))
        self._msvo_upsample(cmd, ao, '_LowDepth1', '_Combined1', '_LinearDepth', None, destination, size(1), size(0))
        for name in (['_LinearDepth'] + ['_LowDepth%d' % i for i in range(1, 5)] + ['_TiledDepth%d' % i for i in range(1, 5)]
                     + ['_Occlusion%d' % i for i in range(1, 5)] + ['_Combined%d' % i for i in range(1, 4)]):
            cmd.release_temporary_rt(name)

    SAMPLE_THICKNESS = None

    @classmethod
    def sample_thickness(cls):
        if cls.SAMPLE_THICKNESS is None:
            cls.SAMPLE_THICKNESS = [mathf_sqrt(F(x)) for x in (0.96, 0.84, 0.64, 0.35999995, 0.91999996, 0.79999995,
                                                               0.59999996, 0.31999993, 0.67999995, 0.47999996,
                                                               0.19999993, 0.27999997)]
        return cls.SAMPLE_THICKNESS

    def _msvo_render(self, cmd, ao, source, destination, size3, tan_half):
        th = self.sample_thickness()
        num = r4(2.0 * dv(tan_half) * 10.0 / dv(size3[0]))
        num2 = r4(1.0 / dv(num))
        inv = [r4(dv(num2) / dv(th[i])) for i in range(12)]
        w = [r4((4.0 if i in (0, 1, 2, 3, 4, 8, 11) else 8.0) * dv(th[i])) for i in range(12)]
        for i in (0, 2, 5, 7, 9):
            w[i] = F(0)
        total = F(0)
        for x in w:
            total = r4(dv(total) + dv(x))
        w = [r4(dv(x) / dv(total)) for x in w]
        cmd.set_compute_param('MultiScaleVORender', 'gInvThicknessTable', inv)
        cmd.set_compute_param('MultiScaleVORender', 'gSampleWeightTable', w)
        cmd.set_compute_param('MultiScaleVORender', 'gInvSliceDimension',
                              (r4(1.0 / dv(size3[0])), r4(1.0 / dv(size3[1])), F(0), F(0)))
        cmd.set_compute_param('MultiScaleVORender', 'AdditionalParams',
                              (r4(-1.0 / dv(ao['thicknessModifier'])), ao['intensity'], F(0), F(0)))
        tg = self.shaders.kernel('MultiScaleVORender', 'MultiScaleVORender_interleaved')['kernel']['thread_group_size']
        groups = tuple((int(size3[i]) + tg[i] - 1) // tg[i] for i in range(3))
        cmd.dispatch('MultiScaleVORender', 'MultiScaleVORender_interleaved',
                     {'DepthTex': temp(source), 'Occlusion': temp(destination)}, groups)

    def _msvo_upsample(self, cmd, ao, low_depth, interleaved_ao, high_depth, high_ao, dest, low_size, high_size):
        kernel = 'MultiScaleVOUpSample_blendout' if high_ao is not None else 'MultiScaleVOUpSample'
        num2 = r4(1920.0 / dv(low_size[0]))
        num3 = r4(1.0 - dv(mathf_pow(10.0, ao['blurTolerance'])) * dv(num2))
        num3 = r4(dv(num3) * dv(num3))
        num4 = mathf_pow(10.0, ao['upsampleTolerance'])
        x = r4(1.0 / (dv(mathf_pow(10.0, ao['noiseFilterTolerance'])) + dv(num4)))
        cmd.set_compute_param('MultiScaleVOUpsample', 'InvLowResolution',
                              (r4(1.0 / dv(low_size[0])), r4(1.0 / dv(low_size[1])), F(0), F(0)))
        cmd.set_compute_param('MultiScaleVOUpsample', 'InvHighResolution',
                              (r4(1.0 / dv(high_size[0])), r4(1.0 / dv(high_size[1])), F(0), F(0)))
        cmd.set_compute_param('MultiScaleVOUpsample', 'AdditionalParams', (x, num2, num3, num4))
        tex = {'LoResDB': temp(low_depth), 'HiResDB': temp(high_depth), 'LoResAO1': temp(interleaved_ao), 'AoResult': dest}
        if high_ao is not None:
            tex['HiResAO'] = temp(high_ao)
        cmd.dispatch('MultiScaleVOUpsample', kernel, tex, ((high_size[0] + 17) // 16, (high_size[1] + 17) // 16, 1))

    # ------------------------------------------------------------ Render()
    def _render(self, cmd):
        """``PostProcessLayer.Render`` (mono, non-SRP, no TAA)."""
        num = -1
        if self.cfg.stop_nan_propagation and not self.nan_killed:
            num = self._pool_get()
            self._get_screen_temp(cmd, num, self.ctx['source_format'])
            raise PPv2Unimplemented('Render() NaN-kill copy (initial blit always kills NaN in 2.1.7 with RequiresInitialBlit)')
        aa = self.cfg.antialiasing_mode
        flag3 = (aa == ANTIALIASING['FastApproximateAntialiasing'] or aa == ANTIALIASING['SubpixelMorphologicalAntialiasing']) \
            and not self.cfg.break_before_color_grading
        num = self._render_builtins(cmd, not flag3, num)
        if flag3:
            self._render_final_pass(cmd, num)
        self.nan_killed = False

    def _render_builtins(self, cmd, is_final_pass, release_after=-1):
        S = self.settings
        uber = Sheet(SHADER['uber'])
        self.uber = uber
        uber.textures['_AutoExposureTex'] = persistent('white')
        num = -1
        destination = self.destination
        if not is_final_pass:
            num = self._pool_get()
            self._get_screen_temp(cmd, num, self.ctx['source_format'])
            self.destination = temp(num)
            if self.cfg.antialiasing_mode == ANTIALIASING['FastApproximateAntialiasing'] and not self.cfg.fxaa_keep_alpha:
                uber.floats['_LumaInAlpha'] = F(1)
        num2 = -1
        if self._enabled('DepthOfField'):
            num2 = self._pool_get()
            dest = self.destination
            self._get_screen_temp(cmd, num2, self.ctx['source_format'])
            self.destination = temp(num2)
            self._depth_of_field(cmd, S['DepthOfField'])
            self.source = temp(num2)
            self.destination = dest
        for cls in ('MotionBlur', 'AutoExposure', 'LensDistortion', 'ChromaticAberration', 'Bloom', 'Vignette', 'Grain'):
            if self._enabled(cls):
                raise PPv2Unimplemented('%s is active' % cls)
        if not self.cfg.break_before_color_grading and self._enabled('ColorGrading'):
            self._color_grading(cmd, S['ColorGrading'])
        if is_final_pass:
            raise PPv2Unimplemented('Uber as final pass (no SMAA/FXAA)')
        uber.floats['_UVTransform'] = (F(1), F(-1), F(0), F(1))   # ApplyDefaultFlip, graphicsUVStartsAtTop
        cmd.blit_triangle(self.source, self.destination, uber, 0)
        self.source = self.destination
        self.destination = destination
        if release_after != -1:
            cmd.release_temporary_rt(release_after)
        if num2 != -1:
            cmd.release_temporary_rt(num2)
        return num

    def _enabled(self, cls):
        s = self.settings[cls]
        en = bool(s['enabled'])
        if cls == 'DepthOfField':
            return en
        if cls == 'ColorGrading':
            return en
        if cls == 'MotionBlur':
            return en and s['shutterAngle'] > F(0)
        if cls == 'AutoExposure':
            return en
        if cls == 'LensDistortion':
            return en and s['intensity'] != F(0) and (s['intensityX'] > F(0) or s['intensityY'] > F(0))
        if cls in ('ChromaticAberration', 'Bloom', 'Grain'):
            return en and s['intensity'] > F(0)
        if cls == 'Vignette':
            if not en:
                return False
            if s['mode'] == 0 and s['intensity'] > F(0):
                return True
            return s['mode'] == 1 and s['opacity'] > F(0) and s['mask'] is not None
        raise KeyError(cls)

    # ------------------------------------------------------------ DepthOfField
    def _depth_of_field(self, cmd, s):
        ctx = self.ctx
        color_fmt = 'ARGBHalf' if self.camera.allow_hdr else 'ARGB32'
        coc_fmt = 'R8'      # SelectFormat(R8, RHalf): R8 supported
        num = r4(dv(0.024) * (dv(ctx['height']) / 1080.0))
        num2 = r4(dv(s['focalLength']) / 1000.0)
        num3 = mathf_max(s['focusDistance'], num2)
        num4 = r4(dv(ctx['screenWidth']) / dv(ctx['screenHeight']))
        value = r4(dv(num2) * dv(num2) / (dv(s['aperture']) * (dv(num3) - dv(num2)) * dv(num) * 2.0))
        # CalculateMaxCoCRadius: float num = (float)kernelSize * 4f + 6f; Mathf.Min(0.05f, num / screenHeight)
        kr = r4(dv(int(s['kernelSize'])) * 4.0 + 6.0)
        num5 = mathf_min(F(0.05), r4(dv(kr) / dv(ctx['screenHeight'])))
        sheet = Sheet(SHADER['depthOfField'])
        sheet.floats.update({'_Distance': num3, '_LensCoeff': value, '_MaxCoC': num5, '_RcpMaxCoC': r4(1.0 / dv(num5)),
                             '_RcpAspect': r4(1.0 / dv(num4))})
        self._get_screen_temp(cmd, '_CoCTex', coc_fmt, srgb=False)
        cmd.blit_triangle(NONE_TARGET, temp('_CoCTex'), sheet, 0)
        half_w, half_h = ctx['width'] // 2, ctx['height'] // 2
        self._get_screen_temp(cmd, '_DepthOfFieldTex', color_fmt, srgb=True, width=half_w, height=half_h)
        cmd.blit_triangle(self.source, temp('_DepthOfFieldTex'), sheet, 2)
        self._get_screen_temp(cmd, '_DepthOfFieldTemp', color_fmt, srgb=True, width=half_w, height=half_h)
        cmd.blit_triangle(temp('_DepthOfFieldTex'), temp('_DepthOfFieldTemp'), sheet, 3 + int(s['kernelSize']))
        cmd.blit_triangle(temp('_DepthOfFieldTemp'), temp('_DepthOfFieldTex'), sheet, 7)
        cmd.release_temporary_rt('_DepthOfFieldTemp')
        cmd.blit_triangle(self.source, self.destination, sheet, 8)
        cmd.release_temporary_rt('_DepthOfFieldTex')
        cmd.release_temporary_rt('_CoCTex')

    # ------------------------------------------------------------ ColorGrading
    def _color_grading(self, cmd, s):
        mode = s['gradingMode']
        if mode == 2:
            raise PPv2Unimplemented('ColorGrading External LUT')
        if mode != 1:
            raise PPv2Unimplemented('ColorGrading LDR 2D pipeline (Lut2DBaker)')
        self._hdr_3d(cmd, s)

    def curves_pixels(self, s, hdr):
        """``GetCurveTexture``: row 0 = (hueVsHue, hueVsSat, satVsSat, lumVsSat); row 1
        (master as alpha, red, green, blue) is written only in LDR mode, otherwise it
        keeps the persistent ``m_Pixels`` contents of the last LDR render."""
        hh, hs, ss, ls = (s[n] for n in ('hueVsHueCurve', 'hueVsSatCurve', 'satVsSatCurve', 'lumVsSatCurve'))
        px = list(self.curve_pixels)
        for i in range(128):
            px[i] = (hh.cached[i], hs.cached[i], ss.cached[i], ls.cached[i])
            if not hdr:
                px[i + 128] = (s['redCurve'].cached[i], s['greenCurve'].cached[i], s['blueCurve'].cached[i],
                               s['masterCurve'].cached[i])
        self.curve_pixels = px
        return px

    def seed_ldr_curve_history(self, volumes, frame):
        """Fill ``m_Pixels`` row 1 as the last LDR-mode ``GetCurveTexture(false)`` left
        it: a blend over ``volumes`` (the scene volumes before DHH switched the
        grading mode to HDR).  History input, see docs (``curves_ldr_history``)."""
        blender = VolumeBlender()
        blender.update(self.cfg.volume_layer, volumes, frame)
        s = blender.layer['ColorGrading']
        if s['gradingMode'] != 0:
            raise ValueError('seed volumes do not select the LDR pipeline')
        px = list(self.curve_pixels)
        for i in range(128):
            px[i] = (s['hueVsHueCurve'].cached[i], s['hueVsSatCurve'].cached[i], s['satVsSatCurve'].cached[i],
                     s['lumVsSatCurve'].cached[i])
            px[i + 128] = (s['redCurve'].cached[i], s['greenCurve'].cached[i], s['blueCurve'].cached[i],
                           s['masterCurve'].cached[i])
        self.curve_pixels = px

    def _upload_curves(self, s):
        tex, staging = self.curves_texture()
        px = self.curves_pixels(s, hdr=True)
        data = float_to_half_bytes([c for p in px for c in p])
        # Texture2D.Apply: Map(staging, WRITE) of the whole 128x2 RGBAHalf image, then
        # CopySubresourceRegion(dst=curves, src=staging) -- executed immediately in OnPreCull.
        self.b.map_write(staging, 0, 2, 0, 0, len(data), data, generator=GENERATOR, unity='Texture2D.Apply (staging)')
        self.b.copy_subresource_region(tex.resource, 0, (0, 0, 0), staging, 0, None, generator=GENERATOR,
                                       unity='Texture2D.Apply (copy)')
        return tex

    def _hdr_3d(self, cmd, s):
        lut = self.log_lut()
        tonemapper = s['tonemapper']
        kernel = {0: 'KGenLut3D_NoTonemap', 1: 'KGenLut3D_NeutralTonemap', 2: 'KGenLut3D_AcesTonemap',
                  3: 'KGenLut3D_CustomTonemap'}[tonemapper]
        if tonemapper == 3:
            raise PPv2Unimplemented('Custom tonemapper (HableCurve uniforms)')
        C = 'Lut3DBaker'
        cmd.set_compute_param(C, '_Size', (F(33), F(1.0 / 32.0), F(0), F(0)))
        cb = color_balance(s['temperature'], s['tint'])
        cmd.set_compute_param(C, '_ColorBalance', (cb[0], cb[1], cb[2], F(0)))
        cmd.set_compute_param(C, '_ColorFilter', s['colorFilter'])
        x = r4(dv(s['hueShift']) / 360.0)
        y = r4(dv(s['saturation']) / 100.0 + 1.0)
        z = r4(dv(s['contrast']) / 100.0 + 1.0)
        cmd.set_compute_param(C, '_HueSatCon', (x, y, z, F(0)))
        red = (s['mixerRedOutRedIn'], s['mixerRedOutGreenIn'], s['mixerRedOutBlueIn'], F(0))
        green = (s['mixerGreenOutRedIn'], s['mixerGreenOutGreenIn'], s['mixerGreenOutBlueIn'], F(0))
        blue = (s['mixerBlueOutRedIn'], s['mixerBlueOutGreenIn'], s['mixerBlueOutBlueIn'], F(0))
        cmd.set_compute_param(C, '_ChannelMixerRed', tuple(r4(dv(v) / 100.0) for v in red))
        cmd.set_compute_param(C, '_ChannelMixerGreen', tuple(r4(dv(v) / 100.0) for v in green))
        cmd.set_compute_param(C, '_ChannelMixerBlue', tuple(r4(dv(v) / 100.0) for v in blue))
        lift = color_to_lift(tuple(r4(dv(v) * dv(0.2)) for v in s['lift']))
        gain = color_to_gain(tuple(r4(dv(v) * dv(0.8)) for v in s['gain']))
        inv_gamma = color_to_inverse_gamma(tuple(r4(dv(v) * dv(0.8)) for v in s['gamma']))
        cmd.set_compute_param(C, '_Lift', (lift[0], lift[1], lift[2], F(0)))
        cmd.set_compute_param(C, '_InvGamma', (inv_gamma[0], inv_gamma[1], inv_gamma[2], F(0)))
        cmd.set_compute_param(C, '_Gain', (gain[0], gain[1], gain[2], F(0)))
        self._upload_curves(s)
        n = mathf_ceil_to_int(F(8.25))
        cmd.dispatch(C, kernel, {'_Output': persistent('log_lut'), '_Curves': persistent('curves')}, (n, n, n))
        self.uber.keywords = tuple(self.uber.keywords) + ('COLOR_GRADING_HDR_3D',)
        self.uber.textures['_Lut3D'] = persistent('log_lut')
        self.uber.floats['_Lut3D_Params'] = (r4(1.0 / dv(lut.width)), r4(dv(lut.width) - 1.0))
        self.uber.floats['_PostExposure'] = exp2(s['postExposure'])

    # ------------------------------------------------------------ final pass
    def _render_final_pass(self, cmd, release_after):
        if self.cfg.break_before_color_grading:
            raise PPv2Unimplemented('breakBeforeColorGrading DiscardAlpha pass')
        sheet = Sheet(SHADER['finalPass'])
        num = -1
        aa = self.cfg.antialiasing_mode
        if aa == ANTIALIASING['FastApproximateAntialiasing']:
            raise PPv2Unimplemented('FXAA final pass')
        if aa == ANTIALIASING['SubpixelMorphologicalAntialiasing']:
            num = self._pool_get()
            destination = self.destination
            self._get_screen_temp(cmd, num, self.ctx['source_format'])
            self.destination = temp(num)
            self._smaa(cmd)
            self.source = temp(num)
            self.destination = destination
        # Dithering.Render
        noise = self.assets.res['blueNoise64']
        idx, z, w = self.dithering.step(len(noise))
        entry = noise[idx]
        tex = self.asset_texture(entry)
        sheet.textures['_DitheringTex'] = persistent(('asset', entry['path_id']))
        sheet.floats['_Dithering_Coords'] = (r4(dv(self.ctx['screenWidth']) / dv(tex.width)),
                                             r4(dv(self.ctx['screenHeight']) / dv(tex.height)), z, w)
        self.log.append({'dithering': {'index': idx, 'texture': entry['name'], 'z': float(z), 'w': float(w),
                                       'calls': self.dithering.calls}})
        sheet.floats['_UVTransform'] = (F(1), F(-1), F(0), F(1))   # ApplyFlip: context.flip false -> default flip
        cmd.blit_triangle(self.source, self.destination, sheet, 0)
        if num != -1:
            cmd.release_temporary_rt(num)
        if release_after != -1:
            cmd.release_temporary_rt(release_after)

    def _smaa(self, cmd):
        sheet = Sheet(SHADER['smaa'])
        luts = self.assets.res['smaaLuts']
        self.asset_texture(luts['area'])
        self.asset_texture(luts['search'])
        sheet.textures['_AreaTex'] = persistent(('asset', luts['area']['path_id']))
        sheet.textures['_SearchTex'] = persistent(('asset', luts['search']['path_id']))
        fmt = self.ctx['source_format']
        cmd.get_temporary_rt('_SMAA_Flip', self._desc(fmt, False), 'Bilinear')
        cmd.get_temporary_rt('_SMAA_Flop', self._desc(fmt, False), 'Bilinear')
        q = self.cfg.smaa_quality
        cmd.blit_triangle(self.source, temp('_SMAA_Flip'), sheet, q, clear=True)
        cmd.blit_triangle(temp('_SMAA_Flip'), temp('_SMAA_Flop'), sheet, 3 + q)
        cmd.set_global_texture('_BlendTex', temp('_SMAA_Flop'))
        cmd.blit_triangle(self.source, self.destination, sheet, 6)
        cmd.release_temporary_rt('_SMAA_Flip')
        cmd.release_temporary_rt('_SMAA_Flop')


# =============================================================== math helpers
def exp2(x):
    """``RuntimeUtilities.Exp2``: Mathf.Exp(x * 0.6931472f) (argument stored as float)."""
    return mathf_exp(r4(dv(x) * dv(0.6931472)))


def standard_illuminant_y(x):
    x = dv(x)
    return r4(dv(2.87) * x - 3.0 * x * x - dv(0.27509508))


def cie_xy_to_lms(x, y):
    x, y = dv(x), dv(y)
    num = 1.0
    num2 = dv(r4(num * x / y))
    num3 = dv(r4(num * (1.0 - x - y) / y))
    return (r4(dv(0.7328) * num2 + dv(0.4296) * num - dv(0.1624) * num3),
            r4(dv(-0.7036) * num2 + dv(1.6975) * num + dv(0.0061) * num3),
            r4(dv(0.003) * num2 + dv(0.0136) * num + dv(0.9834) * num3))


def color_balance(temperature, tint):
    """``ColorUtilities.ComputeColorBalance``."""
    num = r4(dv(temperature) / 60.0)
    num2 = r4(dv(tint) / 60.0)
    x = r4(dv(0.31271) - dv(num) * (dv(0.1) if num < F(0) else dv(0.05)))
    y = r4(dv(standard_illuminant_y(x)) + dv(num2) * dv(0.05))
    v = (dv(0.949237), dv(1.03542), dv(1.08728))
    l = cie_xy_to_lms(x, y)
    return tuple(r4(v[i] / dv(l[i])) for i in range(3))


def _luma(c):
    return r4(dv(c[0]) * dv(0.2126) + dv(c[1]) * dv(0.7152) + dv(c[2]) * dv(0.0722))


def color_to_lift(c):
    n = dv(_luma(c))
    v = [r4(dv(c[i]) - n) for i in range(3)]
    w = dv(c[3])
    return tuple(r4(dv(v[i]) + w) for i in range(3))


def color_to_inverse_gamma(c):
    n = dv(_luma(c))
    v = [r4(dv(c[i]) - n) for i in range(3)]
    n2 = r4(dv(c[3]) + 1.0)
    return tuple(r4(1.0 / dv(mathf_max(r4(dv(v[i]) + dv(n2)), F(0.001)))) for i in range(3))


def color_to_gain(c):
    n = dv(_luma(c))
    v = [r4(dv(c[i]) - n) for i in range(3)]
    n2 = r4(dv(c[3]) + 1.0)
    return tuple(r4(dv(v[i]) + dv(n2)) for i in range(3))


# =============================================================== executor
class Executor:
    """Executes recorded command buffers into the ProgramBuilder at the camera event."""

    def __init__(self, layer: PostProcessLayerGen):
        self.L = layer
        self.b = layer.b
        self.temps = {}
        self.records = []

    def resolve(self, ref):
        L = self.L
        kind = ref[0]
        if kind == 'camera_target':
            return L.inputs.camera_target
        if kind == 'none':
            return None
        if kind == 'temp':
            return self.temps[ref[1]]
        if kind == 'persistent':
            key = ref[1]
            if key == 'white':
                return L.white_texture()
            if key == 'log_lut':
                return L.log_lut()
            if key == 'curves':
                return L.curves_texture()[0]
            if isinstance(key, tuple) and key[0] == 'ao':
                return L.ambient_only_ao(key[1], key[2])
            if isinstance(key, tuple) and key[0] == 'asset':
                return L._objects[('asset', key[1])]
        raise KeyError(ref)

    def run(self, cmd: CommandBufferRec, buffer_kind):
        L = self.L
        draw_index = 0
        for op in cmd.ops:
            kind = op[0]
            if kind == 'get_temp':
                _, name, desc, filt = op
                if name in self.temps:
                    raise PPv2Unimplemented('GetTemporaryRT on a live name %s' % name)
                self.temps[name] = L.rt_pool.get(desc, filt, name=name)
                L.inputs.globals[name] = self.temps[name]   # temporary RTs are global texture properties
            elif kind == 'release_temp':
                tex = self.temps.pop(op[1])
                L.rt_pool.release(tex)
                L.inputs.globals.pop(op[1], None)
            elif kind == 'set_global_texture':
                L.inputs.globals[op[1]] = self.resolve(op[2])
            elif kind == 'cs_param':
                self.cs_param(*op[1:])
            elif kind == 'dispatch':
                self.dispatch(*op[1:])
            elif kind == 'blit':
                self.blit(op, buffer_kind, draw_index)
                draw_index += 1
            elif kind == 'builtin_blit':
                self.builtin_blit(op, buffer_kind, draw_index)
                draw_index += 1
            else:
                raise KeyError(kind)
        if self.temps and buffer_kind == 'post':
            raise ValueError('temporary RTs leaked: %r' % sorted(self.temps))

    # --------------------------------------------------------- compute
    def cs_param(self, compute, name, values):
        """ComputeShader parameter write into the per-ComputeShader cbuffer shadow:
        float4 arrays and vectors write ``len(values)`` floats, vectors are limited
        to the parameter's colCount (``AdditionalParams`` float2 in MultiScaleVORender)."""
        L = self.L
        k = L.shaders.kernel(compute, _any_kernel(L.shaders, compute))
        for cbname, cb in k['cbuffers'].items():
            for p in cb['params']:
                if p['name'] != name:
                    continue
                shadow = L.cs_shadows.setdefault((compute, cbname), bytearray(cb['byte_size']))
                if p['arraySize'] > 0:
                    count = min(len(values), p['arraySize'] * p['colCount'] * p['rowCount'])
                    if p['colCount'] != 4 or p['rowCount'] != 1:
                        raise PPv2Unimplemented('non-float4 compute array %s' % name)
                else:
                    count = min(len(values), p['colCount'])
                shadow[p['offset']:p['offset'] + 4 * count] = f32_bytes(values[:count])
                return
        raise KeyError('%s has no parameter %s' % (compute, name))

    def dispatch(self, compute, kernel, textures, groups):
        L = self.L
        k = L.shaders.kernel(compute, kernel)
        st = DispatchState()
        st.cs.shader = L._shader('cs', k)
        info = k['kernel']
        for cb in info['cbs']:
            shadow = L.cs_shadows.setdefault((compute, cb['name']), bytearray(k['cbuffers'][cb['name']]['byte_size']))
            st.cs.constant_buffers.append({'slot': cb['bindPoint'], 'blob': self.b.blob(bytes(shadow))})
        used = set()
        for t in info['textures']:
            tex = self.resolve(textures[t['name']])
            used.add(t['name'])
            st.cs.srvs.append({'slot': t['bindPoint'], 'view': tex.srv})
            if t['samplerBindPoint'] is not None and t['samplerBindPoint'] >= 0:
                st.cs.samplers.append({'slot': t['samplerBindPoint'], 'sampler': L._sampler(sampler_desc(tex, L.aniso))})
        for o in info['out_buffers']:
            tex = self.resolve(textures[o['name']])
            used.add(o['name'])
            st.cs.uavs.append({'slot': o['bindPoint'], 'view': tex.uav, 'initial_count': 0xFFFFFFFF})
        if info['in_buffers'] or info['builtin_samplers']:
            raise PPv2Unimplemented('compute buffers / inline samplers')
        if set(textures) - used:
            raise ValueError('unused compute bindings %r' % (set(textures) - used))
        prov = {'generator': GENERATOR, 'unity': 'CommandBuffer.DispatchCompute', 'compute': compute, 'kernel': kernel}
        idx = self.b.dispatch(st, prov, ThreadGroupCountX=groups[0], ThreadGroupCountY=groups[1], ThreadGroupCountZ=groups[2])
        self.records.append({'command': idx, 'kind': 'dispatch', 'compute': compute, 'kernel': kernel,
                             'dxbc_sha256': k['sha256'], 'groups': list(groups),
                             'cbuffers': {cb['name']: bytes(L.cs_shadows[(compute, cb['name'])]).hex() for cb in info['cbs']}})

    # --------------------------------------------------------- draws
    def _bind_stage(self, stage_binding, prog, values, textures, cb_slots_seen):
        L = self.L
        layout = prog['layout']
        for t in layout['textures']:
            tex = textures.get(t['name'])
            if tex is None:
                raise KeyError('no texture for %s in %s' % (t['name'], prog['source']['shader']))
            stage_binding.srvs.append({'slot': t['slot'], 'view': tex.srv})
            if t['sampler'] >= 0:
                stage_binding.samplers.append({'slot': t['sampler'], 'sampler': L._sampler(sampler_desc(tex, L.aniso))})
        if layout['inline_samplers']:
            raise PPv2Unimplemented('inline sampler states in %s' % prog['source']['shader'])
        for cbname, cb in layout['cbuffers'].items():
            for v in cb['variables']:
                val = _lookup(values, textures, v['name'])
                L.cb_pool.write(cbname, cb['size'], v['offset'], pack_variable(v, val))
            cb_slots_seen.append((stage_binding, layout['cb_slots'][cbname], cbname, cb['size']))

    def _draw(self, sheet_shader, pass_index, keywords, values, textures, dest_ref, buffer_kind, draw_index,
              geometry, unity_call, clear=False):
        L = self.L
        shaders = L.shaders
        vs = shaders.program(sheet_shader, 0, pass_index, 'progVertex', list(keywords) + list(L.inputs.global_keywords))
        ps = shaders.program(sheet_shader, 0, pass_index, 'progFragment', list(keywords) + list(L.inputs.global_keywords))
        pstate = shaders.pass_state(sheet_shader, 0, pass_index)
        st = DrawState()
        st.vs.shader = L._shader('vs', vs)
        st.ps.shader = L._shader('ps', ps)
        seen = []
        self._bind_stage(st.vs, vs, values, textures, seen)
        self._bind_stage(st.ps, ps, values, textures, seen)
        for binding, slot, cbname, size in seen:
            binding.constant_buffers.append({'slot': slot, 'blob': self.b.blob(L.cb_pool.snapshot(cbname, size))})
        dest = self.resolve(dest_ref)
        st.om.render_targets = [{'slot': 0, 'view': dest.rtv}]
        if dest_ref == CAMERA_TARGET:
            st.om.depth_stencil = L.inputs.camera_depth.dsv     # CameraTarget binds the camera depth buffer
        st.om.blend_state = L._state('blend', blend_desc(pstate))
        st.om.depth_stencil_state = L._state('depth', depth_desc(pstate, L.camera.uses_reversed_z))
        front = (L.inputs.front_ccw or default_front_ccw)(buffer_kind, draw_index)
        st.rs.state = L._state('raster', raster_desc(pstate, front))
        st.rs.viewports = [{'TopLeftX': 0.0, 'TopLeftY': 0.0, 'Width': float(dest.width), 'Height': float(dest.height),
                            'MinDepth': 0.0, 'MaxDepth': 1.0}]
        # Scissor test is disabled in every PPv2 raster state; Unity leaves a stale
        # scissor rect bound (captured: the previous full-size rect) -- not emitted.
        vb, ib, layout_elems, stride, count = geometry
        st.ia.input_layout = self.b.input_layout(layout_elems, vs['dxbc']) if False else \
            L._obj(('il', json.dumps(layout_elems), vs['sha256']), lambda: self.b.input_layout(layout_elems, vs['dxbc']))
        st.ia.topology = 4
        st.ia.vertex_buffers = [{'slot': 0, 'buffer': vb, 'stride': stride, 'offset': 0}]
        st.ia.index_buffer = {'buffer': ib, 'format': 57, 'offset': 0}
        if clear:
            # BlitFullscreenTriangle(clear: true): load DontCare + ClearRenderTarget(depth, color, Color.clear);
            # only the color target is bound (no DSV), so only the RTV is cleared.
            self.b.clear_rtv(dest.rtv, [0.0, 0.0, 0.0, 0.0], generator=GENERATOR, unity='ClearRenderTarget')
            self.records.append({'command': len(self.b.program['commands']) - 1, 'kind': 'clear_rtv',
                                 'target': _tex_record(dest)})
        prov = {'generator': GENERATOR, 'unity': unity_call, 'shader': sheet_shader, 'pass': pass_index,
                'keywords': list(vs['source']['keywords'])}
        idx = self.b.draw_indexed(st, prov, IndexCount=count, StartIndexLocation=0, BaseVertexLocation=0)
        self.records.append({'command': idx, 'kind': 'draw', 'shader': sheet_shader, 'pass': pass_index,
                             'vs_sha256': vs['sha256'], 'ps_sha256': ps['sha256'],
                             'cbuffers': [{'stage': 'VS' if b is st.vs else 'PS', 'slot': slot, 'name': n, 'size': size,
                                           'hex': L.cb_pool.snapshot(n, size).hex(),
                                           'variables': [v['name'] for v in (vs if b is st.vs else ps)['layout']['cbuffers'][n]['variables']]}
                                          for b, slot, n, size in seen],
                             'target': _tex_record(dest), 'depth': dest_ref == CAMERA_TARGET,
                             'srvs': {'VS': [], 'PS': [{'slot': s['slot'], 'view': s['view']} for s in st.ps.srvs]},
                             'textures': {t['name']: _tex_record(textures[t['name']]) for t in ps['layout']['textures']},
                             'samplers': [sampler_desc(textures[t['name']], L.aniso) for t in ps['layout']['textures'] if t['sampler'] >= 0],
                             'blend': blend_desc(pstate), 'depth_state': depth_desc(pstate, L.camera.uses_reversed_z),
                             'raster': raster_desc(pstate, front), 'viewport': st.rs.viewports[0],
                             'index_count': count})
        return idx

    def _textures_for(self, sheet_textures, material_defaults=None):
        L = self.L
        out = {}
        for k, v in L.inputs.globals.items():
            if isinstance(v, TextureRef):  # None (BuiltinRenderTextureType.None) binds nothing
                out[k] = v
        for k, v in (material_defaults or {}).items():
            out[k] = v
        for k, ref in sheet_textures.items():
            out[k] = self.resolve(ref)
        return out

    def _values_for(self, sheet_floats, textures):
        values = {k: v for k, v in self.L.inputs.globals.items() if not isinstance(v, TextureRef)}
        values.update(sheet_floats)
        return values

    def blit(self, op, buffer_kind, draw_index):
        """``RuntimeUtilities.BlitFullscreenTriangle``: SetGlobalTexture(_MainTex, source),
        SetRenderTarget(destination), optional clear, DrawMesh(fullscreenTriangle,
        identity, sheet.material, 0, pass, sheet.properties)."""
        _, source, dest, sheet, pass_index, clear, load = op
        L = self.L
        L.inputs.globals['_MainTex'] = self.resolve(source)     # None for BuiltinRenderTextureType.None
        textures = self._textures_for(sheet.textures)
        values = self._values_for(sheet.floats, textures)
        vb, ib = L.fullscreen_triangle()
        geom = (vb, ib, [{'SemanticName': 'POSITION', 'SemanticIndex': 0, 'Format': 6, 'InputSlot': 0, 'AlignedByteOffset': 0,
                          'InputSlotClass': 0, 'InstanceDataStepRate': 0}], 12, 3)
        self._draw(sheet.shader, pass_index, sheet.keywords, values, textures, dest, buffer_kind, draw_index, geom,
                   'BlitFullscreenTriangle', clear=clear)

    def builtin_blit(self, op, buffer_kind, draw_index):
        """``RuntimeUtilities.BuiltinBlit``: SetRenderTarget(dest, DontCare) then
        ``CommandBuffer.Blit(source, CurrentActive, mat, pass)`` -- Unity binds source
        as the material's ``_MainTex`` (texture scale/offset (1,1,0,0))."""
        _, source, dest, shader, pass_index = op
        src = self.resolve(source)
        textures = self._textures_for({}, {'_MainTex': src})
        values = self._values_for({'_MainTex_ST': (F(1), F(1), F(0), F(0))}, textures)
        vb, ib = self.L.blit_quad()
        geom = (vb, ib, [{'SemanticName': 'POSITION', 'SemanticIndex': 0, 'Format': 6, 'InputSlot': 0, 'AlignedByteOffset': 0,
                          'InputSlotClass': 0, 'InstanceDataStepRate': 0},
                         {'SemanticName': 'TEXCOORD', 'SemanticIndex': 0, 'Format': 16, 'InputSlot': 0, 'AlignedByteOffset': 12,
                          'InputSlotClass': 0, 'InstanceDataStepRate': 0}], 20, 6)
        if dest == CAMERA_TARGET:
            raise PPv2Unimplemented('BuiltinBlit into CameraTarget')
        self._draw(shader, pass_index, (), values, textures, dest, buffer_kind, draw_index, geom, 'CommandBuffer.Blit')


def _any_kernel(shaders, compute):
    """First kernel of the D3D11 variant: ComputeShader parameters belong to the
    ComputeShader (one cbuffer shadow shared by its kernels), so any kernel's
    cbuffer list serves ``SetComputeFloatParams``.  Shader sources other than the
    manifest adapter (``hs2pipe.frame.LibraryShaderSource``) provide ``any_kernel``."""
    if hasattr(shaders, 'any_kernel'):
        return shaders.any_kernel(compute)
    root, doc, c = shaders.compute[compute]
    v = [v for v in c['variants'] if v['targetRenderer'] == 2][0]
    return v['kernels'][0]['name']


def _tex_record(t: TextureRef):
    return {'resource': t.resource, 'name': t.name, 'width': t.width, 'height': t.height,
            'descriptor': t.resource_desc, 'filter': t.filter_mode, 'wrap': t.wrap_mode, 'aniso': t.aniso_level}


def _lookup(values, textures, name):
    """Property value for one program variable: property block / material / global;
    ``<tex>_TexelSize`` from the bound texture.  Variables set nowhere are written
    as zero (gap ``unset_uniform_policy``)."""
    if name in values:
        return values[name]
    if name.endswith('_TexelSize'):
        tex = textures.get(name[:-len('_TexelSize')])
        if tex is not None:
            return tex.texel_size
    return None


def pack_variable(v, val):
    n = v['dim'] * max(1, v['array'])
    if v['type'] != 0:
        raise PPv2Unimplemented('non-float cbuffer variable %s' % v['name'])
    if val is None:
        vals = [F(0)] * n
    else:
        vals = list(val) if isinstance(val, (tuple, list)) else [val]
        vals = (vals + [F(0)] * n)[:n]
    return f32_bytes(vals)
