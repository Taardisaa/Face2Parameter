"""Unity 2018.4 runtime mechanisms used by the PostProcessing v2 generator.

Everything here is something PPv2's C# calls into (UnityEngine / Mono / the D3D11
GfxDevice) rather than PPv2 logic itself.  Each item states its source:

* float32 scalar arithmetic: Mono (Unity 2018.4 MonoBleedingEdge, x64 JIT) single
  precision ops.  ``Mathf.*`` wrappers are ``(float)System.Math.*((double)x)``
  as in ``UnityEngine.Mathf`` (UnityEngine.CoreModule).  Python's ``math`` calls
  the same MSVC UCRT functions on Windows.
* ``SystemRandom``: .NET/Mono ``System.Random`` (Knuth subtractive generator,
  reference source ``Random.cs``) -- PPv2 ``Dithering.m_Random = new Random(1234)``.
* Render-texture/texture -> D3D11 descriptors, sampler states, pass render states:
  mapped from Unity enums using the D3D11 enum values; where the UnityPlayer
  native rule is not transcribed the mapping is labelled ``observed`` with its
  capture evidence (see ``docs/hs2_offline_ppv2.md`` gaps).
* ``UnityConstantBufferPool``: Unity's per-(cbuffer name, size) constant buffer
  shadow; only the variables of the executing programs are written, so bytes
  outside them keep earlier contents (observed as residue in the capture).
* ``TemporaryRTPool``: ``RenderTexture.GetTemporary`` pool keyed by descriptor.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import numpy as np

F = np.float32


def f32(x):
    return F(x)


# Mono (Unity 2018.4, no float32 JIT mode) evaluates float expressions in double
# precision and rounds to float32 only where a value is stored (local, field, array
# element, float parameter, float return) -- ``conv.r4``.  Established on the
# capture: MultiScaleVO gInvThicknessTable equals the double-intermediate result
# bit-exactly and differs by 1 ulp under pure float32 evaluation.
def r4(x):
    """conv.r4: round a double intermediate to float32."""
    return F(float(x))


def dv(x):
    """A float32 value (local/field/literal ``1.5f``) promoted to double."""
    return float(F(x))


# ------------------------------------------------------------------ Mathf (CoreModule)
def mathf_sqrt(x):
    return F(math.sqrt(float(F(x))))


def mathf_tan(x):
    return F(math.tan(float(F(x))))


def mathf_pow(x, y):
    return F(math.pow(float(F(x)), float(F(y))))


def mathf_exp(x):
    return F(math.exp(float(F(x))))


def mathf_log10(x):
    return F(math.log10(float(F(x))))


def mathf_max(a, b):
    a, b = F(a), F(b)
    return a if a > b else b


def mathf_min(a, b):
    a, b = F(a), F(b)
    return a if a < b else b


def mathf_clamp01(x):
    x = F(x)
    if x < F(0):
        return F(0)
    if x > F(1):
        return F(1)
    return x


def mathf_ceil_to_int(x):
    return int(math.ceil(float(F(x))))


def mathf_floor_to_int(x):
    return int(math.floor(float(F(x))))


def f32_bytes(values):
    return struct.pack('<%df' % len(values), *[float(F(v)) for v in values])


# ------------------------------------------------------------- System.Random (Mono)
class SystemRandom:
    """``System.Random`` (reference source Random.cs, Knuth subtractive, seed ctor)."""
    MBIG = 2147483647
    MSEED = 161803398

    def __init__(self, seed):
        self.seed_array = [0] * 56
        subtraction = self.MBIG if seed == -2147483648 else abs(seed)
        mj = self.MSEED - subtraction
        self.seed_array[55] = mj
        mk = 1
        for i in range(1, 55):
            ii = (21 * i) % 55
            self.seed_array[ii] = mk
            mk = mj - mk
            if mk < 0:
                mk += self.MBIG
            mj = self.seed_array[ii]
        for _ in range(1, 5):
            for i in range(1, 56):
                v = self.seed_array[i] - self.seed_array[1 + (i + 30) % 55]
                v = (v + 2 ** 31) % 2 ** 32 - 2 ** 31  # int32 wrap
                if v < 0:
                    v += self.MBIG
                self.seed_array[i] = v
        self.inext = 0
        self.inextp = 21

    def internal_sample(self):
        loc_inext = self.inext + 1
        if loc_inext >= 56:
            loc_inext = 1
        loc_inextp = self.inextp + 1
        if loc_inextp >= 56:
            loc_inextp = 1
        ret = self.seed_array[loc_inext] - self.seed_array[loc_inextp]
        if ret == self.MBIG:
            ret -= 1
        if ret < 0:
            ret += self.MBIG
        self.seed_array[loc_inext] = ret
        self.inext = loc_inext
        self.inextp = loc_inextp
        return ret

    def next_double(self):
        return self.internal_sample() * (1.0 / self.MBIG)


# ------------------------------------------------------------------ half floats
def float_to_half_bytes(values):
    """Texture2D.SetPixels -> RGBAHalf storage.  Round-to-nearest-even float32->half
    (numpy).  Accepted bit-exactly against the captured curves upload."""
    return np.asarray(values, dtype=np.float32).astype(np.float16).tobytes()


# --------------------------------------------------------- texture formats (D3D11)
# DXGI enum values (dxgiformat.h)
DXGI = {
    'R32G32B32A32_TYPELESS': 1, 'R32G32B32A32_FLOAT': 2, 'R16G16B16A16_TYPELESS': 9, 'R16G16B16A16_FLOAT': 10,
    'R32G32_TYPELESS': 15, 'R32G32_FLOAT': 16, 'R8G8B8A8_TYPELESS': 27, 'R8G8B8A8_UNORM': 28,
    'R8G8B8A8_UNORM_SRGB': 29, 'R16G16_TYPELESS': 33, 'R16G16_FLOAT': 34, 'R32_TYPELESS': 39, 'R32_FLOAT': 41,
    'R8G8_TYPELESS': 48, 'R8G8_UNORM': 49, 'R16_TYPELESS': 53, 'R16_FLOAT': 54, 'R8_TYPELESS': 60,
    'R8_UNORM': 61, 'A8_UNORM': 65, 'R24G8_TYPELESS': 44, 'R32G8X24_TYPELESS': 19,
}

# UnityEngine.RenderTextureFormat values used by PPv2 2.1.7.
RTF = {'ARGB32': 0, 'Depth': 1, 'ARGBHalf': 2, 'Shadowmap': 3, 'RGB565': 4, 'ARGB4444': 5, 'ARGB1555': 6,
       'Default': 7, 'ARGB2101010': 8, 'DefaultHDR': 9, 'ARGB64': 10, 'ARGBFloat': 11, 'RGFloat': 12,
       'RGHalf': 13, 'RFloat': 14, 'RHalf': 15, 'R8': 16, 'ARGBInt': 17, 'RGInt': 18, 'RInt': 19,
       'BGRA32': 20, 'RGB111110Float': 22, 'RG32': 23, 'RGBAUShort': 24, 'RG16': 25}
RTF_NAME = {v: k for k, v in RTF.items()}

# (resource typeless, linear view, sRGB view).  Observed on this device in the
# retry capture (frame graph resources): ARGBHalf 59330/59640 -> 9/10, RFloat
# 59632 -> 39/41, R8 59338/59652 -> 60/61.  DefaultHDR resolves to ARGBHalf on
# D3D11 (GfxDeviceD3D11Base::GetDefaultHDRRTFormat; RenderTextureFormat.DefaultHDR
# -> ARGBHalf as documented for desktop).  ARGB32/RG16 from the D3D11 format
# family (R8G8B8A8 / R8G8) used by GetGraphicsFormat.
RT_DXGI = {
    'ARGBHalf': ('R16G16B16A16_TYPELESS', 'R16G16B16A16_FLOAT', 'R16G16B16A16_FLOAT'),
    'ARGBFloat': ('R32G32B32A32_TYPELESS', 'R32G32B32A32_FLOAT', 'R32G32B32A32_FLOAT'),
    'RFloat': ('R32_TYPELESS', 'R32_FLOAT', 'R32_FLOAT'),
    'RGFloat': ('R32G32_TYPELESS', 'R32G32_FLOAT', 'R32G32_FLOAT'),
    'RHalf': ('R16_TYPELESS', 'R16_FLOAT', 'R16_FLOAT'),
    'RGHalf': ('R16G16_TYPELESS', 'R16G16_FLOAT', 'R16G16_FLOAT'),
    'R8': ('R8_TYPELESS', 'R8_UNORM', 'R8_UNORM'),
    'RG16': ('R8G8_TYPELESS', 'R8G8_UNORM', 'R8G8_UNORM'),
    'ARGB32': ('R8G8B8A8_TYPELESS', 'R8G8B8A8_UNORM', 'R8G8B8A8_UNORM_SRGB'),
}
# Observed random-write promotion: MultiScaleVO allocates LinearDepth and the
# TiledDepth arrays as RHalf with enableRandomWrite (MultiScaleVO.PushAllocCommands)
# and the device created R32_TYPELESS/R32_FLOAT (resources 59424, 59449...).  The
# UnityPlayer rule (GraphicsCaps::IsFormatSupported(..., LoadStore) fallback in
# RenderTexture creation) is not transcribed: gap ``rt_random_write_format_fallback``.
RANDOM_WRITE_FALLBACK = {'RHalf': 'RFloat', 'RGHalf': 'RGFloat'}


def resolve_rt_format(fmt_name):
    if fmt_name == 'DefaultHDR':
        return 'ARGBHalf'
    if fmt_name == 'Default':
        return 'ARGB32'
    return fmt_name


@dataclass(frozen=True)
class RTDescriptor:
    """Subset of ``UnityEngine.RenderTextureDescriptor`` that PPv2 2.1.7 sets."""
    width: int
    height: int
    color_format: str            # RenderTextureFormat name
    srgb: bool
    dimension: str = 'Tex2D'     # Tex2D | Tex2DArray | Tex3D
    volume_depth: int = 1
    enable_random_write: bool = False
    msaa_samples: int = 1
    depth_buffer_bits: int = 0
    use_mip_map: bool = False
    auto_generate_mips: bool = False

    def d3d(self):
        """(kind, resource desc, srv desc, rtv desc, uav desc) of the created texture."""
        fmt = resolve_rt_format(self.color_format)
        if self.enable_random_write and fmt in RANDOM_WRITE_FALLBACK:
            fmt = RANDOM_WRITE_FALLBACK[fmt]
        if self.depth_buffer_bits or self.msaa_samples != 1 or self.use_mip_map:
            raise NotImplementedError('PPv2 temporaries with depth/MSAA/mips are not used by this pipeline: %r' % (self,))
        typeless, linear, srgb = (DXGI[x] for x in RT_DXGI[fmt])
        view_fmt = srgb if self.srgb else linear
        bind = 8 | 32 | (128 if self.enable_random_write else 0)  # SRV | RTV | UAV
        if self.dimension == 'Tex3D':
            res = {'Width': self.width, 'Height': self.height, 'Depth': self.volume_depth, 'MipLevels': 1,
                   'Format': typeless, 'Usage': 0, 'BindFlags': bind, 'CPUAccessFlags': 0, 'MiscFlags': 0}
            srv = {'Format': view_fmt, 'ViewDimension': 8, 'Texture3D': {'MostDetailedMip': 0, 'MipLevels': 1}}
            uav = {'Format': view_fmt, 'ViewDimension': 8,
                   'Texture3D': {'MipSlice': 0, 'FirstWSlice': 0, 'WSize': 0xFFFFFFFF}}
            return 'tex3d', res, srv, None, uav
        arrays = self.volume_depth if self.dimension == 'Tex2DArray' else 1
        res = {'Width': self.width, 'Height': self.height, 'MipLevels': 1, 'ArraySize': arrays, 'Format': typeless,
               'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': bind, 'CPUAccessFlags': 0,
               'MiscFlags': 0}
        if self.dimension == 'Tex2DArray':
            srv = {'Format': view_fmt, 'ViewDimension': 5,
                   'Texture2DArray': {'MostDetailedMip': 0, 'MipLevels': 1, 'FirstArraySlice': 0, 'ArraySize': 0xFFFFFFFF}}
            rtv = {'Format': view_fmt, 'ViewDimension': 5,
                   'Texture2DArray': {'MipSlice': 0, 'FirstArraySlice': 0, 'ArraySize': 0xFFFFFFFF}}
            uav = {'Format': view_fmt, 'ViewDimension': 5,
                   'Texture2DArray': {'MipSlice': 0, 'FirstArraySlice': 0, 'ArraySize': 0xFFFFFFFF}}
        else:
            srv = {'Format': view_fmt, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}}
            rtv = {'Format': view_fmt, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}}
            uav = {'Format': view_fmt, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}}
        return 'tex2d', res, srv, rtv, (uav if self.enable_random_write else None)


# ----------------------------------------------------------------- texture objects
FILTER = {'Point': 0, 'Bilinear': 1, 'Trilinear': 2}
WRAP = {'Repeat': 0, 'Clamp': 1, 'Mirror': 2, 'MirrorOnce': 3}
D3D_ADDRESS = {0: 1, 1: 3, 2: 2, 3: 5}   # WRAP, CLAMP, MIRROR, MIRROR_ONCE
FLT_MAX = 3.4028234663852886e+38


@dataclass
class TextureRef:
    """A bindable texture: program resource + views + Unity sampler attributes."""
    resource: int
    srv: int | None
    width: int
    height: int
    depth: int = 1
    rtv: int | None = None
    uav: int | None = None
    dsv: int | None = None
    filter_mode: str = 'Bilinear'
    wrap_mode: str = 'Clamp'
    aniso_level: int = 1
    has_mips: bool = False
    name: str = ''
    descriptor: RTDescriptor | None = None
    resource_desc: dict | None = None

    @property
    def texel_size(self):
        """Unity ``<tex>_TexelSize`` = (1/w, 1/h, w, h) (float32)."""
        return (F(1) / F(self.width), F(1) / F(self.height), F(self.width), F(self.height))


def sampler_desc(tex: TextureRef, anisotropic_filtering='Enable'):
    """D3D11 sampler of a Unity texture (TextureSettings -> D3D11 sampler).

    Filter: Point -> MIN_MAG_MIP_POINT(0), Bilinear -> MIN_MAG_LINEAR_MIP_POINT(20),
    Trilinear -> MIN_MAG_MIP_LINEAR(21), effective anisotropy > 1 -> ANISOTROPIC(85).
    ``QualitySettings.anisotropicFiltering = ForceEnable`` (set by DHH
    ``ApplyGraphicSetting``) raises anisoLevel 1..8 to 9; anisoLevel 0 stays off;
    Point filtering never uses anisotropy.  All observed samplers in the capture
    (229, 230, 2721, 3001, 1940) follow this rule (``ComparisonFunc`` GREATER,
    LOD range +-FLT_MAX, zero border); the UnityPlayer routine itself is not
    transcribed (gap ``unity_sampler_mapping``)."""
    filt = FILTER[tex.filter_mode]
    aniso = tex.aniso_level
    if anisotropic_filtering == 'ForceEnable' and 0 < aniso < 9:
        aniso = 9
    elif anisotropic_filtering == 'Disable':
        aniso = 1
    if filt == 0 or aniso <= 1:
        aniso = 1
    if aniso > 1:
        d3d_filter = 85
    else:
        d3d_filter = {0: 0, 1: 20, 2: 21}[filt]
    addr = D3D_ADDRESS[WRAP[tex.wrap_mode]]
    return {'Filter': d3d_filter, 'AddressU': addr, 'AddressV': addr, 'AddressW': addr, 'MipLODBias': 0.0,
            'MaxAnisotropy': aniso, 'ComparisonFunc': 5, 'BorderColor': [0.0, 0.0, 0.0, 0.0],
            'MinLOD': -FLT_MAX, 'MaxLOD': FLT_MAX}


# ----------------------------------------------------------- ShaderLab pass states
UNITY_BLEND_TO_D3D = {0: 1, 1: 2, 2: 9, 3: 3, 4: 10, 5: 5, 6: 4, 7: 7, 8: 8, 9: 11, 10: 6}
UNITY_BLENDOP_TO_D3D = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
UNITY_CMP_TO_D3D = {0: 8, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
UNITY_CULL_TO_D3D = {0: 1, 1: 2, 2: 3}


def _state_val(state, key):
    v = state[key]
    if isinstance(v, dict) and 'val' in v:
        if v.get('name') not in (None, '<noninit>'):
            raise NotImplementedError('pass state %s bound to a material property %r' % (key, v['name']))
        return int(v['val'])
    return v


def blend_desc(pass_state):
    """D3D11 blend state of a ShaderLab pass (rtBlend0 only; IndependentBlend off).
    Matches captured objects 92 (One Zero) and 1764 (Zero OneMinusSrcColor)."""
    b = pass_state['rtBlend0']
    src, dst = _state_val(b, 'srcBlend'), _state_val(b, 'destBlend')
    srca, dsta = _state_val(b, 'srcBlendAlpha'), _state_val(b, 'destBlendAlpha')
    op, opa = _state_val(b, 'blendOp'), _state_val(b, 'blendOpAlpha')
    enable = not (src == 1 and dst == 0 and srca == 1 and dsta == 0 and op == 0 and opa == 0)
    rt0 = {'BlendEnable': enable, 'SrcBlend': UNITY_BLEND_TO_D3D[src], 'DestBlend': UNITY_BLEND_TO_D3D[dst],
           'BlendOp': UNITY_BLENDOP_TO_D3D[op], 'SrcBlendAlpha': UNITY_BLEND_TO_D3D[srca],
           'DestBlendAlpha': UNITY_BLEND_TO_D3D[dsta], 'BlendOpAlpha': UNITY_BLENDOP_TO_D3D[opa],
           'RenderTargetWriteMask': _state_val(b, 'colMask')}
    off = {'BlendEnable': False, 'SrcBlend': 0, 'DestBlend': 0, 'BlendOp': 0, 'SrcBlendAlpha': 0,
           'DestBlendAlpha': 0, 'BlendOpAlpha': 0, 'RenderTargetWriteMask': 0}
    return {'AlphaToCoverageEnable': False, 'IndependentBlendEnable': False, 'RenderTarget': [rt0] + [dict(off) for _ in range(7)]}


REVERSED_Z_CMP = {2: 5, 4: 7, 5: 2, 7: 4}


def depth_desc(pass_state, reversed_z=True):
    """ZTest/ZWrite of a ShaderLab pass; with ``SystemInfo.usesReversedZBuffer`` the
    ordered comparisons are mirrored (Unity reversed-Z convention, D3D11)."""
    ztest = _state_val(pass_state, 'zTest')
    if reversed_z:
        ztest = REVERSED_Z_CMP.get(ztest, ztest)
    zwrite = _state_val(pass_state, 'zWrite')
    keep = {'StencilFailOp': 1, 'StencilDepthFailOp': 1, 'StencilPassOp': 1, 'StencilFunc': 8}
    return {'DepthEnable': True, 'DepthWriteMask': 1 if zwrite else 0, 'DepthFunc': UNITY_CMP_TO_D3D[ztest],
            'StencilEnable': False, 'StencilReadMask': 255, 'StencilWriteMask': 255,
            'FrontFace': dict(keep), 'BackFace': dict(keep)}


def raster_desc(pass_state, front_counter_clockwise):
    """``FrontCounterClockwise`` is Unity's render-target-flip/invert-culling state at
    the draw (GfxDevice); supplied by the caller (see ``PPv2FrameInputs``)."""
    return {'FillMode': 3, 'CullMode': UNITY_CULL_TO_D3D[_state_val(pass_state, 'culling')],
            'FrontCounterClockwise': bool(front_counter_clockwise), 'DepthBias': 0, 'DepthBiasClamp': 0.0,
            'SlopeScaledDepthBias': -0.0, 'DepthClipEnable': True, 'ScissorEnable': False,
            'MultisampleEnable': True, 'AntialiasedLineEnable': False}


# ------------------------------------------------------- constant buffer shadows
class UnityConstantBufferPool:
    """Unity D3D11 constant buffers, one per (cbuffer name, Unity layout size).

    A draw writes only the variables listed in the executing programs' layouts
    (VS then PS, same shadow), then uploads the whole shadow (Map DISCARD).  Bytes
    outside those variables keep what earlier programs wrote -- the residue
    observed in the capture (e.g. ``ConstantBuffer-186-464`` shared by
    ``AIT/Skin True Face`` and SMAA).  ``seed`` lets the frame orchestrator (or the
    capture test) supply contents written by earlier, non-PPv2 draws."""

    def __init__(self):
        self.shadows = {}

    def seed(self, name, size, data):
        if len(data) != size:
            raise ValueError('seed size mismatch')
        self.shadows[(name, size)] = bytearray(data)

    def shadow(self, name, size):
        return self.shadows.setdefault((name, size), bytearray(size))

    def write(self, name, size, offset, data):
        buf = self.shadow(name, size)
        if offset + len(data) > size:
            raise ValueError('cbuffer write outside %s/%d' % (name, size))
        buf[offset:offset + len(data)] = data

    def snapshot(self, name, size):
        return bytes(self.shadow(name, size))


# --------------------------------------------------------- temporary RT pool
@dataclass
class PoolEntry:
    descriptor: RTDescriptor
    texture: TextureRef
    free: bool = True


class TemporaryRTPool:
    """``RenderTexture.GetTemporary`` / ``CommandBuffer.GetTemporaryRT`` pool.

    A request is served by a free entry with an equal descriptor (first in pool
    order), otherwise a new texture is created.  The created RenderTexture keeps
    its own object attributes (anisoLevel, wrap); GetTemporaryRT sets the
    requested filter mode.  Entries released by earlier (non-PPv2) users can be
    registered with ``adopt`` (e.g. DHH SSS's 543x724 "ImageEffects Temp" with
    anisoLevel 0, reused by DoF in the capture).  Which free entry Unity picks when
    several match is not transcribed (gap ``temporary_rt_pool_order``); it does not
    change contents because every PPv2 temporary is fully rewritten before use."""

    def __init__(self, builder, default_aniso=1, default_wrap='Clamp'):
        self.b = builder
        self.entries = []
        self.default_aniso = default_aniso
        self.default_wrap = default_wrap
        self.created = []

    def adopt(self, descriptor, texture, free=True):
        self.entries.append(PoolEntry(descriptor, texture, free))

    def get(self, descriptor: RTDescriptor, filter_mode='Bilinear', name=''):
        for e in self.entries:
            if e.free and e.descriptor == descriptor:
                e.free = False
                e.texture.filter_mode = filter_mode
                # Observed: an RT handed out again by the pool samples without the
                # forced anisotropy that a freshly created one gets (capture segment 0
                # vs 1..15, e.g. resource 59648 sampler 2721 -> 229).  Modelled as the
                # reused object's anisoLevel being 0 (gap ``temporary_rt_reuse_aniso``).
                e.texture.aniso_level = 0
                return e.texture
        kind, res, srv, rtv, uav = descriptor.d3d()
        rid = self.b.texture(kind, res, name=name or 'TempRT %dx%d' % (descriptor.width, descriptor.height))
        tex = TextureRef(resource=rid, srv=self.b.view('srv', rid, srv), width=descriptor.width,
                         height=descriptor.height, depth=descriptor.volume_depth,
                         rtv=self.b.view('rtv', rid, rtv) if rtv else None,
                         uav=self.b.view('uav', rid, uav) if uav else None,
                         filter_mode=filter_mode, wrap_mode=self.default_wrap, aniso_level=self.default_aniso,
                         name=name, descriptor=descriptor, resource_desc=res)
        self.entries.append(PoolEntry(descriptor, tex, False))
        self.created.append(tex)
        return tex

    def release(self, tex: TextureRef):
        for e in self.entries:
            if e.texture is tex:
                if e.free:
                    raise ValueError('temporary RT released twice')
                e.free = True
                return
        raise ValueError('release of a texture not owned by the pool')


# ---------------------------------------------------------- AnimationCurve
# Transcribed from UnityPlayer.dll 2018.4.11f1 (PDB-matched, see
# outputs/offline_renderer_ppv2_20261020/disasm/AnimationCurve_*.txt):
#   AnimationCurveTpl<float>::Evaluate              RVA 0x7d5900
#   AnimationCurveTpl<float>::FindIndexForSampling  RVA 0x7d7a80
#   AnimationCurveTpl<float>::CalculateCacheData    RVA 0x7d44b0
# Keyframe = {time, value, inSlope, outSlope, weightedMode, inWeight, outWeight}.
# All operations are SSE scalar float32.
WRAP_CLAMP = 2   # InternalWrapMode written by the serializer (m_PreInfinity/m_PostInfinity)
_INF = F(np.inf)


def curve_cache_coeffs(lhs, rhs, time_offset=F(0)):
    """CalculateCacheData: (time, timeEnd, c0, c1, c2, c3)."""
    t_lhs, t_rhs = F(lhs['time']), F(rhs['time'])
    time = F(time_offset) + t_lhs
    time_end = F(time_offset) + t_rhs
    dx = t_rhs - t_lhs
    dy = F(rhs['value']) - F(lhs['value'])
    dxm = dx if dx > F(0.0001) else F(0.0001)         # maxss(1e-4, dx)
    m1 = F(lhs['outSlope'])
    d2 = dxm * F(rhs['inSlope'])
    length = F(1) / (dxm * dxm)
    d1 = dxm * m1
    c0 = ((((d2 + d1) - dy) - dy) * length) / dxm
    c1 = (((((dy + dy) + dy) - d1) - d1) - d2) * length
    c2 = m1
    c3 = F(lhs['value'])
    if F(lhs['outSlope']) == _INF or F(rhs['inSlope']) == _INF:   # SetupStepped
        c0 = c1 = c2 = F(0)
    return time, time_end, c0, c1, c2, c3


def _poly(t, cache):
    time, _, c0, c1, c2, c3 = cache
    x = F(t) - time
    return ((x * c0 + c1) * x + c2) * x + c3


def evaluate_curve(keys, t, pre_infinity=WRAP_CLAMP, post_infinity=WRAP_CLAMP):
    """``AnimationCurve.Evaluate`` (native, cached-polynomial path) for float curves.
    Weighted keys and non-clamp wrap modes go through EvaluateWithoutCache / repeat
    logic, which is not transcribed and raises."""
    t = F(t)
    n = len(keys)
    if n == 1:
        return F(keys[0]['value'])
    if n < 1:
        return F(0)
    first, last = F(keys[0]['time']), F(keys[-1]['time'])
    if not np.isfinite(first) or not np.isfinite(last):
        return F(0)
    if t >= last:
        if post_infinity != WRAP_CLAMP:
            raise NotImplementedError('AnimationCurve post-infinity mode %r' % post_infinity)
        return _poly(t, (last, _INF, F(0), F(0), F(0), F(keys[-1]['value'])))
    if t < first:
        if pre_infinity != WRAP_CLAMP:
            raise NotImplementedError('AnimationCurve pre-infinity mode %r' % pre_infinity)
        return _poly(t, (t - F(1000), first, F(0), F(0), F(0), F(keys[0]['value'])))
    hi = sum(1 for k in keys if F(k['time']) <= t)       # FindIndexForSampling (upper bound)
    lhs, rhs = hi - 1, min(hi, n - 1)
    if keys[lhs]['weightedMode'] & 2 or keys[rhs]['weightedMode'] & 1:
        raise NotImplementedError('weighted AnimationCurve keys (EvaluateWithoutCache)')
    return _poly(t, curve_cache_coeffs(keys[lhs], keys[rhs]))
