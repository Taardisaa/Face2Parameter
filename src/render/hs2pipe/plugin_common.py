"""Shared, source-derived helpers for the plugin effect generators.

Used by ``dhh_sss``, ``ctaa``, ``liquid`` and ``capture_loop``.  Each rule cites
where it comes from; nothing here is fitted.

* **Mono float arithmetic.** The plugins run on Unity 2018.4's Mono JIT.  Their
  ``float`` expressions are evaluated in double precision and rounded to
  binary32 when stored to a field/local or returned (ECMA-335 permits the wider
  internal type).  This was established from the capture, not assumed: the
  CTAA projection jitter of all 16 iterations and ``Mathf.Lerp`` results of
  ``SetCTAA_Parameters`` match byte-for-byte only under this rule; pure
  binary32 evaluation mismatches 6 of the 16 jitter matrices and
  ``preEnhanceClamp`` (see ``docs/hs2_offline_plugin_effects.md``).
* **Unity RenderTexture -> D3D11 resources.**  The Unity-level descriptor a
  plugin requests (``RenderTexture.GetTemporary``/``new RenderTexture``) is
  recorded verbatim.  Its D3D11 lowering is the D3D11 GfxDevice convention of
  this install for the formats the plugins use; every row is checked against the
  resource descriptors of the retry capture (``validate_plugin_effects``).
* **Material constant packing.**  Compiled ``$Globals`` layouts come from the
  installed shader export (``m_ConstantBuffers``); a ``m_Type`` 1 (int)
  parameter written through ``Material.SetFloat``/``SetInt`` is uploaded as an
  int32, as Unity's property sheet converts at upload time (captured
  ``maxDistance``/``_SSS_NUM_SAMPLES`` bytes are integers).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

F32 = np.float32


class PluginStateError(ValueError):
    """Required scene-state field missing or outside the implemented branch."""


class UnimplementedBranch(NotImplementedError):
    """A source branch that is recovered but deliberately not lowered (explicit gap)."""


# --------------------------------------------------------------- Mono floats
def mono_f32(value: float) -> np.float32:
    """Round a double-precision intermediate to the binary32 store (Mono JIT)."""
    return F32(value)


def d(value) -> float:
    """Widen a binary32 operand to the double used by Mono float arithmetic."""
    return float(F32(value))


def mathf_lerp(a, b, t) -> np.float32:
    """``UnityEngine.Mathf.Lerp``: ``a + (b - a) * Clamp01(t)`` (CoreModule IL)."""
    t = d(t)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    t = d(mono_f32(t))  # Clamp01 returns float
    return mono_f32(d(a) + (d(b) - d(a)) * t)


def mathf_max(a, b) -> np.float32:
    """``Mathf.Max(float, float)``: ``a > b ? a : b``."""
    return F32(a) if d(a) > d(b) else F32(b)


def mathf_sqrt(x) -> np.float32:
    """``Mathf.Sqrt(float)``: ``(float)Math.Sqrt((double)f)``."""
    import math
    return mono_f32(math.sqrt(d(x)))


def f32_hex(values) -> str:
    return np.asarray(values, dtype=np.float32).reshape(-1).tobytes().hex()


def i32_hex(values) -> str:
    return np.asarray(values, dtype=np.int32).reshape(-1).tobytes().hex()


# --------------------------------------------------------- Unity RT records
# RenderTextureFormat names used by the plugins.  ``Default`` resolves to ARGB32
# on this D3D11 desktop device (Unity: "Default color render texture format").
RT_FORMAT_ALIASES = {'Default': 'ARGB32', 'DefaultHDR': 'ARGBHalf'}

# D3D11 typeless resource format and shader/target view formats per Unity
# RenderTextureFormat.  Values are DXGI_FORMAT integers; rows verified against
# the retry capture descriptors (59344 ARGB32 Linear, 59322 ARGB32 sRGB,
# 59360 ARGBFloat, 59668 ARGBHalf, 59420 RGHalf).
COLOR_FORMATS = {
    #           typeless, linear view, sRGB view
    'ARGB32': (27, 28, 29),       # R8G8B8A8_{TYPELESS,UNORM,UNORM_SRGB}
    'ARGBHalf': (9, 10, 10),      # R16G16B16A16_{TYPELESS,FLOAT}
    'ARGBFloat': (1, 2, 2),       # R32G32B32A32_{TYPELESS,FLOAT}
    'RGHalf': (33, 34, 34),       # R16G16_{TYPELESS,FLOAT}
}
# depthBufferBits -> (typeless resource, DSV format, SRV format)
DEPTH_FORMATS = {
    16: (53, 55, 56),             # R16_TYPELESS / D16_UNORM / R16_UNORM
    24: (19, 20, 21),             # R32G8X24_TYPELESS / D32_FLOAT_S8X24_UINT / R32_FLOAT_X8X24_TYPELESS
    32: (19, 20, 21),
}
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_RENDER_TARGET = 0x20
D3D11_BIND_DEPTH_STENCIL = 0x40
D3D11_BIND_UNORDERED_ACCESS = 0x80

FILTER_MODES = ('Point', 'Bilinear', 'Trilinear')
WRAP_MODES = ('Repeat', 'Clamp', 'Mirror', 'MirrorOnce')


@dataclass(frozen=True)
class UnityRenderTexture:
    """One Unity RenderTexture as a plugin creates it (Unity-level descriptor)."""
    name: str
    width: int
    height: int
    depth_bits: int
    format: str
    read_write: str            # 'Linear' | 'sRGB' | 'Default'
    msaa: int = 1
    filter_mode: str = 'Bilinear'
    wrap_mode: str = 'Repeat'
    aniso_level: int = 1
    auto_generate_mips: bool = True
    use_mipmap: bool = False
    random_write: bool = False
    lifetime: str = 'temporary'   # 'temporary' (GetTemporary pool) | 'persistent' (new RenderTexture)
    source: str = ''              # plugin method that created it

    def resolved_format(self) -> str:
        return RT_FORMAT_ALIASES.get(self.format, self.format)

    def is_srgb(self, color_space: str) -> bool:
        """``RenderTextureReadWrite``: Default follows the project colour space."""
        if self.read_write == 'sRGB':
            return True
        if self.read_write == 'Linear':
            return False
        if self.read_write == 'Default':
            return color_space == 'Linear'
        raise PluginStateError('Unknown RenderTextureReadWrite ' + self.read_write)

    def pool_key(self) -> tuple:
        """Fields that make up Unity's RenderTextureDesc for temporary-pool matching.

        ``autoGenerateMips`` is part of the descriptor flags; setting it on a pooled
        texture makes the next ``GetTemporary`` miss (DHH ``GetProfileRT``).
        """
        return (self.width, self.height, self.depth_bits, self.resolved_format(), self.read_write,
                self.msaa, self.auto_generate_mips, self.use_mipmap, self.random_write)

    def d3d11(self, color_space: str) -> dict:
        fmt = self.resolved_format()
        if fmt not in COLOR_FORMATS:
            raise UnimplementedBranch('D3D11 lowering of RenderTextureFormat %s not recovered' % fmt)
        typeless, linear_view, srgb_view = COLOR_FORMATS[fmt]
        view = srgb_view if self.is_srgb(color_space) else linear_view
        mips = 1 if not self.use_mipmap else None
        if mips is None:
            raise UnimplementedBranch('Mipmapped plugin render textures are not used by the recovered plugins')
        color = {'Width': self.width, 'Height': self.height, 'MipLevels': 1, 'ArraySize': 1, 'Format': typeless,
                 'SampleDesc': {'Count': self.msaa, 'Quality': 0}, 'Usage': 0,
                 'BindFlags': D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET |
                              (D3D11_BIND_UNORDERED_ACCESS if self.random_write else 0),
                 'CPUAccessFlags': 0, 'MiscFlags': 0}
        out = {'color': color, 'color_view_format': view, 'depth': None, 'dsv_format': None, 'depth_srv_format': None}
        if self.depth_bits:
            if self.depth_bits not in DEPTH_FORMATS:
                raise UnimplementedBranch('Depth bits %d' % self.depth_bits)
            dt, dsv, dsrv = DEPTH_FORMATS[self.depth_bits]
            out['depth'] = dict(color, Format=dt, BindFlags=D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL)
            out['dsv_format'] = dsv
            out['depth_srv_format'] = dsrv
        return out

    def texel_size(self) -> np.ndarray:
        """``<tex>_TexelSize`` = (1/w, 1/h, w, h), computed by the engine in binary32."""
        w, h = F32(self.width), F32(self.height)
        return np.array([F32(1) / w, F32(1) / h, w, h], np.float32)

    def to_json(self) -> dict:
        return asdict(self)


# ------------------------------------------------------ Unity-level operations
@dataclass
class Op:
    """One ordered Unity-level operation of a plugin generator.

    ``owner`` names the module that must lower it to frame-program commands:
    ``plugin`` ops are lowered here; ``unity_forward``/``ppv2`` ops are
    requests to the sibling generators with all parameters the plugin sets.
    """
    kind: str
    owner: str
    source: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {'kind': self.kind, 'owner': self.owner, 'source': self.source, 'args': jsonable(self.args)}


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {'f32_hex' if obj.dtype == np.float32 else 'i32_hex': obj.reshape(-1).tobytes().hex(),
                'values': [float(v) if obj.dtype == np.float32 else int(v) for v in obj.reshape(-1)]}
    if isinstance(obj, np.floating):
        return {'f32_hex': np.asarray([obj], np.float32).tobytes().hex(), 'value': float(obj)}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, UnityRenderTexture):
        return obj.to_json()
    if isinstance(obj, Op):
        return obj.to_json()
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    return obj


# --------------------------------------------------------- shader programs
class ShaderManifests:
    """Installed shader exports (``extract_shader_programs.py`` manifests)."""

    def __init__(self, paths: Sequence[Path]):
        self.shaders: Dict[str, dict] = {}
        self.sources: Dict[str, str] = {}
        for p in paths:
            m = json.loads(Path(p).read_text(encoding='utf-8'))
            for s in m['shaders']:
                self.shaders.setdefault(s['name'], s)
                self.sources.setdefault(s['name'], '%s (%s)' % (m.get('bundle_path'), m.get('bundle_sha256')))

    def shader(self, name: str) -> dict:
        if name not in self.shaders:
            raise PluginStateError('Installed shader export missing: ' + name)
        return self.shaders[name]

    def program(self, name: str, pass_index: int, stage: str, keywords) -> dict:
        """Exact keyword branch only (``shader_bindings.select_program``)."""
        from ..shader_bindings import select_program
        return select_program(self.shader(name), pass_index, stage, keywords)

    def pass_state(self, name: str, pass_index: int) -> dict:
        rows = [p for p in self.shader(name)['passes'] if p['pass'] == pass_index]
        if len(rows) != 1:
            raise PluginStateError('%s pass %d not unique' % (name, pass_index))
        return rows[0]

    def property_default(self, name: str, prop: str):
        for p in self.shader(name)['properties']:
            if p['name'] == prop:
                return p
        return None


def keyword_variant(shader: dict, pass_index: int, stage: str, enabled: set) -> List[str]:
    """Unity variant selection: the compiled branch whose keyword set is the
    subset of enabled keywords that this pass declares (``m_KeywordIndices`` set
    over all branches of the stage)."""
    rows = [r for p in shader['passes'] if p['pass'] == pass_index for r in p['programs'] if r['stage'] == stage]
    declared = set()
    for r in rows:
        declared.update(r['keywords'])
    chosen = sorted(set(enabled) & declared)
    match = [r for r in rows if set(r['keywords']) == set(chosen)]
    if len(match) != 1:
        raise PluginStateError('%s pass %d %s: %d branches for %s' % (shader['name'], pass_index, stage, len(match), chosen))
    return list(match[0]['keywords'])


def pack_variables(program: dict, values: Dict[str, Any], *, buffer: str = '$Globals') -> Dict[str, dict]:
    """Per-variable GPU bytes of one compiled cbuffer, as Unity uploads them.

    Only parameters of the compiled layout are produced (Unity writes per
    parameter; other bytes of a shared constant buffer are not determined by the
    draw).  Values must be supplied for every parameter: a missing value raises,
    it is never defaulted here.  Returns ``{name: {offset, size, hex, type}}``.
    """
    names = {i: n for n, i in program['name_indices']}
    out = {}
    for cb in program['bindings']['m_ConstantBuffers']:
        if names[cb['m_NameIndex']] != buffer:
            continue
        for fld in cb['m_VectorParams'] + cb['m_MatrixParams']:
            name = names[fld['m_NameIndex']]
            if name not in values:
                raise PluginStateError('No source value for compiled parameter %s.%s' % (buffer, name))
            count = fld['m_ArraySize'] or 1
            width = fld.get('m_Dim', fld.get('m_RowCount', 0) * 4)
            v = np.asarray(values[name])
            if v.size != count * width:
                raise PluginStateError('%s: expected %d components, got %d' % (name, count * width, v.size))
            typ = fld['m_Type']
            if typ == 0:
                raw = v.astype(np.float32).reshape(-1).tobytes()
            elif typ == 1:
                # Material float -> int32 at upload (truncation toward zero).
                raw = np.trunc(v.astype(np.float64)).astype(np.int32).reshape(-1).tobytes()
            elif typ == 2:
                raw = v.astype(np.bool_).astype(np.uint32).reshape(-1).tobytes()
            else:
                raise UnimplementedBranch('%s: constant type %d' % (name, typ))
            if count > 1:
                raise UnimplementedBranch('%s: array parameters are not used by the plugin materials' % name)
            out[name] = {'offset': fld['m_Index'], 'size': len(raw), 'hex': raw.hex(), 'type': typ}
        return out
    raise PluginStateError('Program has no %s cbuffer' % buffer)


def texture_slots(program: dict) -> Dict[str, dict]:
    """Texture name -> {srv slot, sampler slot, dim} from the compiled program."""
    names = {i: n for n, i in program['name_indices']}
    return {names[t['m_NameIndex']]: {'srv': t['m_Index'], 'sampler': t['m_SamplerIndex'], 'dim': t['m_Dim']}
            for t in program['bindings']['m_TextureParams']}


def unity_sampler_key(filter_mode: str, wrap_mode: str, aniso_level: int) -> str:
    """Unity's sampler-state cache debug name ``SamplerState-<filter>-<uvw>-<aniso>``.

    filter: Point 0 / Bilinear 1 / Trilinear 2; wrap per axis: Repeat 0 / Clamp 1.
    The D3D11 descriptor (including any QualitySettings anisotropic forcing) is
    created by the GfxDevice lowering, not here.
    """
    f = FILTER_MODES.index(filter_mode)
    w = WRAP_MODES.index(wrap_mode)
    return 'SamplerState-%d-%d%d%d-%d' % (f, w, w, w, aniso_level)


# --------------------------------------------------------- program builder
class ProgramRecorder:
    """In-memory stand-in with the object/command methods of ``frame_program.ProgramBuilder``.

    The plugin lowering only calls ``texture``, ``view`` and
    ``copy_subresource_region`` (same signatures as ``ProgramBuilder``), so a
    real builder can be passed instead.  This recorder lets generated output be
    inspected and tested without writing a program directory.
    """

    def __init__(self):
        self.resources: List[dict] = []
        self.views: List[dict] = []
        self.commands: List[dict] = []
        self._next = 1

    def _id(self):
        i = self._next
        self._next += 1
        return i

    def texture(self, kind, desc, initial=None, *, name='', ident=None):
        rid = self._id()
        self.resources.append({'id': rid, 'kind': kind, 'desc': dict(desc), 'initial': initial, 'name': name})
        return rid

    def view(self, kind, resource, desc, *, ident=None, name=''):
        vid = self._id()
        self.views.append({'id': vid, 'kind': kind, 'resource': resource, 'desc': dict(desc), 'name': name})
        return vid

    def copy_subresource_region(self, dst, dst_subresource, dst_xyz, src, src_subresource, src_box=None,
                                copy_flags=None, **p):
        x, y, z = dst_xyz
        cmd = {'op': 'copy_subresource_region', 'dst': dst, 'dst_subresource': dst_subresource, 'dst_x': x,
               'dst_y': y, 'dst_z': z, 'src': src, 'src_subresource': src_subresource, 'src_box': src_box,
               'copy_flags': copy_flags}
        if p:
            cmd['provenance'] = p
        self.commands.append(cmd)
        return len(self.commands) - 1

    def to_json(self) -> dict:
        return {'resources': self.resources, 'views': self.views, 'commands': self.commands}


def texture2d_view_desc(kind: str, fmt: int, msaa: int) -> dict:
    """Verbatim D3D11 view descriptors for a single-mip, single-slice 2D texture."""
    if kind == 'srv':                  # D3D11_SRV_DIMENSION_TEXTURE2D 4 / TEXTURE2DMS 6
        if msaa == 1:
            return {'Format': fmt, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}}
        return {'Format': fmt, 'ViewDimension': 6, 'Texture2DMS': {}}
    if kind == 'rtv':                  # D3D11_RTV_DIMENSION_TEXTURE2D 4 / TEXTURE2DMS 6
        if msaa == 1:
            return {'Format': fmt, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}}
        return {'Format': fmt, 'ViewDimension': 6, 'Texture2DMS': {}}
    if kind == 'dsv':                  # D3D11_DSV_DIMENSION_TEXTURE2D 3 / TEXTURE2DMS 5
        if msaa == 1:
            return {'Format': fmt, 'ViewDimension': 3, 'Flags': 0, 'Texture2D': {'MipSlice': 0}}
        return {'Format': fmt, 'ViewDimension': 5, 'Flags': 0, 'Texture2DMS': {}}
    raise PluginStateError(kind)


class RenderTextureAllocator:
    """Unity temporary-RT pool and persistent RTs, lowered to contract resources.

    ``GetTemporary`` returns the most recently released pooled texture whose
    descriptor key matches (LIFO per key, UnityPlayer ``ReleaseTempBuffer``
    0x75a610 links it at the list head), else creates one; ``ReleaseTemporary``
    files it under the texture's current descriptor.
    Persistent textures are created once and destroyed explicitly.  Identity is
    what matters for the frame program: a reused texture keeps its contents.
    """

    def __init__(self, builder, color_space: str):
        self.builder = builder
        self.color_space = color_space
        self.free: List[Tuple[tuple, dict]] = []
        self.live: Dict[str, dict] = {}
        self.created: List[dict] = []
        self._n = 0

    def _create(self, rt: UnityRenderTexture) -> dict:
        lowered = rt.d3d11(self.color_space)
        handle = 'rt%d' % self._n
        self._n += 1
        rec = {'handle': handle, 'rt': rt, 'lowered': lowered}
        if self.builder is not None:
            b = self.builder
            vf, ms = lowered['color_view_format'], rt.msaa
            rec['resource'] = b.texture('tex2d', lowered['color'], None, name=rt.name)
            rec['srv'] = b.view('srv', rec['resource'], texture2d_view_desc('srv', vf, ms), name=rt.name)
            rec['rtv'] = b.view('rtv', rec['resource'], texture2d_view_desc('rtv', vf, ms), name=rt.name)
            if lowered['depth']:
                rec['depth_resource'] = b.texture('tex2d', lowered['depth'], None, name=rt.name + ' depth')
                rec['dsv'] = b.view('dsv', rec['depth_resource'], texture2d_view_desc('dsv', lowered['dsv_format'], ms), name=rt.name)
                rec['depth_srv'] = b.view('srv', rec['depth_resource'], texture2d_view_desc('srv', lowered['depth_srv_format'], ms), name=rt.name)
        self.created.append(rec)
        return rec

    def get_temporary(self, rt: UnityRenderTexture) -> dict:
        key = rt.pool_key()
        # RenderBufferManager::Textures::ReleaseTempBuffer (UnityPlayer 0x75a610) files a
        # released texture under its *current* RenderTextureDesc and links it at the
        # head of that key's list: most recently released matching texture first.
        for i in range(len(self.free) - 1, -1, -1):
            k, rec = self.free[i]
            if k == key:
                self.free.pop(i)
                rec = dict(rec, rt=rt, reused=True)
                self.live[rec['handle']] = rec
                return rec
        rec = dict(self._create(rt), reused=False)
        self.live[rec['handle']] = rec
        return rec

    def release_temporary(self, rec: dict, *, key_override: Optional[tuple] = None):
        if rec is None:
            return
        self.live.pop(rec['handle'], None)
        self.free.append((key_override or rec['rt'].pool_key(), rec))

    def create_persistent(self, rt: UnityRenderTexture) -> dict:
        rec = dict(self._create(rt), reused=False)
        self.live[rec['handle']] = rec
        return rec

    def destroy(self, rec: dict):
        if rec is not None:
            self.live.pop(rec['handle'], None)


# ------------------------------------------------------------ op lowering
FORWARD_INTERFACE = {
    'blit': 'blit(builder, *, source, dest, shader, pass_index, keywords, ps_globals, textures): '
            'Graphics.Blit lowering (GL.LoadOrtho quad, unity_ObjectToWorld identity, material pass state); '
            'source/dest are allocator records or engine buffers',
    'render_camera': 'render_camera(builder, args, on_image_effects_opaque): Camera.Render/RenderWithShader of a '
                     'DHH camera; calls on_image_effects_opaque(source, destination) at the opaque image-effect point',
    'engine_state': 'engine_state(op_json): global keyword/texture, camera/quality fields',
    'draw_pass': 'draw_pass(builder, op_json): one forward pass of a sorted object',
    'active_color_target': 'active_color_target(): record of the bound camera colour target (has "resource")',
    'engine_buffer': 'engine_buffer(name): record for Unity-owned buffers (image-effect source/destination, rtCamera)',
}

PLUGIN_MATERIALS = {'Hidden/SeparableSSS': 'SSS_convolution._BlurMaterial', 'Hidden/CTAA_Enhance_PC': 'mat_enhance',
                    'Hidden/CTAA_PC': 'ctaaMat'}


class PluginLowering:
    """Lowers plugin ``Op`` lists to frame-program commands.

    Plugin-owned work (render-texture lifetime, material constants, GrabPass
    copies, blit parameters) is produced here; Unity-internal work is delegated
    to ``forward`` through the methods in ``FORWARD_INTERFACE``.  A missing
    method raises ``UnimplementedBranch``; nothing is skipped or substituted.
    """

    def __init__(self, builder, manifests: ShaderManifests, forward, color_space: str):
        self.builder = builder
        self.manifests = manifests
        self.forward = forward
        self.alloc = RenderTextureAllocator(builder, color_space)
        self.handles: Dict[str, dict] = {}
        self.materials: Dict[str, dict] = {}
        self.global_keywords: set = set()
        self.log: List[dict] = []

    def _need(self, name):
        fn = getattr(self.forward, name, None)
        if fn is None:
            raise UnimplementedBranch('unity_forward interface missing: ' + FORWARD_INTERFACE[name])
        return fn

    def _resolve(self, name, scope):
        if name in scope:
            return scope[name]
        if name in self.handles:
            return self.handles[name]
        return self._need('engine_buffer')(name)

    @staticmethod
    def _rt(value):
        return value if isinstance(value, UnityRenderTexture) else UnityRenderTexture(**value)

    def emit(self, ops: Sequence, scope: Optional[Dict[str, dict]] = None):
        scope = dict(scope or {})
        for op in ops:
            if isinstance(op, dict):
                op = Op(op['kind'], op['owner'], op['source'], op['args'])
            k, a = op.kind, op.args
            if k == 'get_temporary':
                rt = self._rt(a['rt'])
                rec = self.alloc.get_temporary(rt)
                post = a.get('post_create') or {}
                key = rt.pool_key()
                rec['release_key'] = key[:6] + (False,) + key[7:] if post.get('autoGenerateMips') is False else None
                scope[a['handle']] = rec
                self.handles[a['handle']] = rec
            elif k == 'release_temporary':
                rec = scope.pop(a['handle'], None) or self.handles.get(a['handle'])
                if rec is not None:
                    self.handles.pop(a['handle'], None)
                    self.alloc.release_temporary(rec, key_override=rec.get('release_key'))
            elif k == 'create_persistent':
                rec = self.alloc.create_persistent(self._rt(a['rt']))
                self.handles[a['handle']] = rec
                scope[a['handle']] = rec
            elif k == 'destroy':
                for h in a['handles']:
                    self.alloc.destroy(self.handles.pop(h, None))
            elif k == 'material':
                self.materials[a['material']] = {'shader': a['shader'], 'properties': dict(a['properties']),
                                                 'keywords': dict(a['keywords'])}
            elif k == 'blit':
                self._blit(a, scope)
            elif k == 'grab_copy':
                dst = self._resolve(a['dest'], scope)
                src = self._need('active_color_target')()
                self.builder.copy_subresource_region(dst['resource'], 0, (0, 0, 0), src['resource'], 0, src_box=a['box'],
                                                     source='GrabPasses::ApplyGrabPass -> GrabIntoRenderTexture',
                                                     object=a['object'])
            elif k == 'render_camera':
                def on_opaque(source, destination, _a=a, _scope=scope):
                    inner = dict(_scope, source=source, destination=destination,
                                 blurred=self._resolve(_a.get('blurred_handle', 'SSSLightingTextureBlurred'), _scope))
                    self.emit(_a['image_effects_opaque'], inner)
                self._need('render_camera')(self.builder, jsonable(a), on_opaque)
            elif k == 'draw_pass':
                self._need('draw_pass')(self.builder, op.to_json())
            elif k == 'global_keyword':
                (self.global_keywords.add if a['enabled'] else self.global_keywords.discard)(a['keyword'])
                self._need('engine_state')(op.to_json())
            elif k in ('set_global_texture', 'camera_field', 'quality', 'create_camera', 'ppv2_parameter'):
                self._need('engine_state')(op.to_json())
            elif k in ('component_field', 'buffers_viewer', 'reset_temporal', 'save_camera_state', 'pose_freeze',
                       'face_surfaces_bake', 'read_pixels', 'restore_camera_state', 'pose_restore',
                       'release_grab_frame_state'):
                self.log.append(op.to_json())   # CPU-side plugin/bridge state, no GPU command
            else:
                raise UnimplementedBranch('Unknown plugin op ' + k)

    def _blit(self, a, scope):
        source = self._resolve(a['source'], scope)
        dest = self._resolve(a['dest'], scope)
        shader = a['material']
        ps_globals, keywords, textures = None, [], {}
        pass_index = 0 if a['pass'] == -1 else a['pass']
        if shader is not None:
            mat = self.materials.get(PLUGIN_MATERIALS[shader])
            if mat is None:
                raise PluginStateError('Blit with %s before its material properties were set' % PLUGIN_MATERIALS[shader])
            props = dict(mat['properties'])
            for name, vec in (a.get('set_vector') or {}).items():
                props[name] = np.frombuffer(bytes.fromhex(vec['f32_hex']), np.float32) if isinstance(vec, dict) else vec
            enabled = {k for k, on in mat['keywords'].items() if on} | self.global_keywords
            keywords = keyword_variant(self.manifests.shader(shader), pass_index, 'progFragment', enabled)
            program = self.manifests.program(shader, pass_index, 'progFragment', keywords)
            props['_MainTex_TexelSize'] = source['rt'].texel_size()
            for tex in texture_slots(program):
                if isinstance(props.get(tex), str):          # plugin texture handle (e.g. _Accum)
                    textures[tex] = self._resolve(props[tex], scope)
                elif isinstance(props.get(tex), dict):       # material texture from scene state (NoiseTexture)
                    textures[tex] = props[tex]
            values = {n: v for n, v in props.items() if not isinstance(v, (str, dict))}
            ps_globals = pack_variables(program, values)
        self._need('blit')(self.builder, source=source, dest=dest, shader=shader or 'Hidden/BlitCopy',
                           pass_index=pass_index, keywords=keywords, ps_globals=ps_globals, textures=textures)
