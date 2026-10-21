"""Unity 2018.4.11f1 ShaderLab pass state -> D3D11 state descriptors.

Transcribed from the installed UnityPlayer.dll (SHA-256 ``5034cdc2…``):

* ``ShaderLab::ShaderState::FromSerializedShaderState`` (RVA 0x83c880): raster
  fields culling (+0x40), offsetUnits (+0x50 -> depth bias), offsetFactor
  (+0x48 -> slope bias), zClip (+0x28); a ``LightMode = ShadowCaster`` pass
  whose constant ``Offset`` is exactly (1, 1) gets both zeroed (0x83c962..0x83ca18).
* ``ShaderState::BuildDeviceStates`` (0x855d10): GfxDepthState {write, func}
  defaults (1, LEqual) then zWrite != 0 and zTest clamped to 0..8; GfxBlendState
  per-RT default 0x0100010f; GfxStencilState default {0, 0xff, 0xff} with
  front/back (Always, Keep, Keep, Keep).
* ``ShaderBlendStateDesc::ToGfxBlendState`` (0x857820): factors clamped 0..10,
  ops 0..0x23, mask 0..15; 8 RTs only when separateMRTBlend; alphaToMask != 0.
* ``ShaderRasterStateDesc::ToGfxRasterState`` (0x857af0): cull clamp 0..2,
  bias = (int)units, slope = factor, depthClip clamp 0..1.
* ``ShaderStencilStateDesc::ToGfxStencilState`` (0x857c10): the generic
  ``stencilOp`` replaces the face(s) the raster cull mode keeps; a face whose
  values are all constant defaults (Always/Keep/Keep/Keep) leaves the state
  disabled; otherwise comp clamp 0..8, ops 0..7 and enabled = 1.
* ``GfxDeviceD3D11Base::CreateBlendState`` (0x545680): D3D11 blend desc is
  zeroed, AlphaToCoverage = alphaToMask, IndependentBlend = separate && caps,
  only the used RTs are filled; an RT with One/Zero/One/Zero/Add/Add is
  BlendEnable 0 with the canonical factor fields; tables 0x1227098 (colour),
  0x12270c8 (alpha), 0x1227100 (op).
* ``GfxDeviceD3D11Base::SetupDeferredDepthStencilState`` (0x54d3c0): reversed-Z
  remaps Less<->Greater, LEqual<->GEqual; DepthEnable = 1 whenever a depth
  state is set; compare table 0x1227050, stencil-op table 0x1227078.
* ``GfxDeviceD3D11Base::SetupDeferredRasterState`` (0x54d800): FillMode
  2/3 (wireframe +0x589a), CullMode table 0x1227270 unless forced (+0x21f4),
  FrontCounterClockwise = backface (+0x5899) ^ userBackface (+0x21f1) ^
  invertProjection (+0x21f0), bias = (int)((float)bias + global +0x21f8),
  slope = slope + global +0x21fc, both negated with reversed Z
  (GraphicsCaps+0xdd), ScissorEnable = +0x589b, Multisample 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

F = np.float32
NONINIT = '<noninit>'

D3D11_BLEND_COLOR = (1, 2, 9, 3, 10, 5, 4, 7, 8, 11, 6, 0)     # 0x1227098
D3D11_BLEND_ALPHA = (1, 2, 7, 5, 8, 5, 6, 7, 8, 11, 6, 0)      # 0x12270c8
D3D11_BLEND_OP = (1, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1)          # 0x1227100 (first 12)
D3D11_COMPARE = (8, 1, 2, 3, 4, 5, 6, 7, 8)                    # 0x1227050
D3D11_STENCIL_OP = (1, 2, 3, 4, 5, 6, 7, 8)                    # 0x1227078
D3D11_CULL = (1, 2, 3)                                          # 0x1227270
REVERSED_Z_FUNC = {2: 5, 4: 7, 5: 2, 7: 4}                      # 0x54d446..0x54d494


class StateGap(RuntimeError):
    pass


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _val(entry, material_floats, what):
    """SerializedShaderFloatValue: constant, or the material float of its property."""
    name = entry.get('name')
    if name in (None, NONINIT):
        return float(entry['val']), None
    if name not in material_floats:
        raise StateGap('%s bound to property %s: material float required' % (what, name))
    return float(material_floats[name]), name


def _trunc(x):
    return int(F(x))        # cvttss2si


@dataclass
class GfxStates:
    blend: dict
    depth: dict
    stencil: dict
    raster: dict


def gfx_states(state, *, light_mode, material_floats=None):
    """GfxBlend/Depth/Stencil/RasterState of one ShaderLab pass (serialized state
    record as in the shader library: ``rtBlend0..7``, ``zTest`` ... with ``val``/``name``)."""
    mf = material_floats or {}
    # raster (FromSerializedShaderState + ToGfxRasterState)
    cull, _ = _val(state['culling'], mf, 'culling')
    units_e, factor_e = state['offsetUnits'], state['offsetFactor']
    units, units_p = _val(units_e, mf, 'offsetUnits')
    factor, factor_p = _val(factor_e, mf, 'offsetFactor')
    if (units_p is None and F(units) == F(1.0) and factor_p is None and F(factor) == F(1.0)
            and light_mode == 'SHADOWCASTER'):
        units, factor = 0.0, 0.0
    zclip, _ = _val(state['zClip'], mf, 'zClip')
    raster = {'cull': _clamp(_trunc(cull), 0, 2), 'depth_bias': _trunc(units), 'slope_bias': F(factor),
              'depth_clip': _clamp(_trunc(zclip), 0, 1)}
    # depth (BuildDeviceStates)
    zw, _ = _val(state['zWrite'], mf, 'zWrite')
    zt, _ = _val(state['zTest'], mf, 'zTest')
    depth = {'write': 1 if F(zw) != F(0.0) else 0, 'func': _clamp(_trunc(zt), 0, 8)}
    # blend (ToGfxBlendState)
    separate = bool(state['rtSeparateBlend'])
    rts = [{'mask': 15, 'src': 1, 'dst': 0, 'src_a': 1, 'dst_a': 0, 'op': 0, 'op_a': 0} for _ in range(8)]
    for i in range(8 if separate else 1):
        b = state['rtBlend%d' % i]
        rts[i] = {'src': _clamp(_trunc(_val(b['srcBlend'], mf, 'srcBlend')[0]), 0, 10),
                  'dst': _clamp(_trunc(_val(b['destBlend'], mf, 'destBlend')[0]), 0, 10),
                  'src_a': _clamp(_trunc(_val(b['srcBlendAlpha'], mf, 'srcBlendAlpha')[0]), 0, 10),
                  'dst_a': _clamp(_trunc(_val(b['destBlendAlpha'], mf, 'destBlendAlpha')[0]), 0, 10),
                  'op': _clamp(_trunc(_val(b['blendOp'], mf, 'blendOp')[0]), 0, 0x23),
                  'op_a': _clamp(_trunc(_val(b['blendOpAlpha'], mf, 'blendOpAlpha')[0]), 0, 0x23),
                  'mask': _clamp(_trunc(_val(b['colMask'], mf, 'colMask')[0]), 0, 15)}
    a2m, _ = _val(state['alphaToMask'], mf, 'alphaToMask')
    blend = {'rts': rts, 'separate': separate, 'alpha_to_mask': F(a2m) != F(0.0)}
    # stencil (ToGfxStencilState)
    rm, _ = _val(state['stencilReadMask'], mf, 'stencilReadMask')
    wm, _ = _val(state['stencilWriteMask'], mf, 'stencilWriteMask')
    stencil = {'enabled': 0, 'read_mask': _trunc(rm) & 0xFF, 'write_mask': _trunc(wm) & 0xFF,
               'front': (8, 0, 0, 0), 'back': (8, 0, 0, 0)}

    def is_default(op):
        return all(op[k].get('name') in (None, NONINIT) for k in ('comp', 'pass', 'fail', 'zFail')) and \
            F(op['pass']['val']) == 0 and F(op['fail']['val']) == 0 and F(op['zFail']['val']) == 0 and \
            F(op['comp']['val']) == F(8.0)
    front_src, back_src = state['stencilOpFront'], state['stencilOpBack']
    generic = state['stencilOp']
    if not is_default(generic):
        c = raster['cull']
        if c != 1:
            front_src = generic
        if c in (0, 1):
            back_src = generic
    for face, src in (('front', front_src), ('back', back_src)):
        if is_default(src):
            continue
        stencil[face] = (_clamp(_trunc(_val(src['comp'], mf, 'stencil comp')[0]), 0, 8),
                         _clamp(_trunc(_val(src['pass'], mf, 'stencil pass')[0]), 0, 7),
                         _clamp(_trunc(_val(src['fail'], mf, 'stencil fail')[0]), 0, 7),
                         _clamp(_trunc(_val(src['zFail'], mf, 'stencil zFail')[0]), 0, 7))
        stencil['enabled'] = 1
    stencil_ref, _ = _val(state['stencilRef'], mf, 'stencilRef')
    stencil['ref'] = _trunc(stencil_ref)
    return GfxStates(blend, depth, stencil, raster)


# ----------------------------------------------------------------- D3D11
def d3d11_blend_desc(blend, *, caps_independent_blend=True):
    if 5 <= blend['rts'][0]['op'] <= 20 and not blend['separate']:
        # 0x5458c9: logical ops take the D3D11.1 CreateBlendState1 branch when caps+0x115
        raise StateGap('logic-op blend (GfxDeviceD3D11Base::CreateBlendState D3D11.1 branch) not transcribed')
    independent = blend['separate'] and caps_independent_blend
    zero = {'BlendEnable': False, 'SrcBlend': 0, 'DestBlend': 0, 'BlendOp': 0, 'SrcBlendAlpha': 0,
            'DestBlendAlpha': 0, 'BlendOpAlpha': 0, 'RenderTargetWriteMask': 0}
    out = [dict(zero) for _ in range(8)]
    for i in range(8 if independent else 1):
        rt = blend['rts'][i]
        if (rt['src'], rt['dst'], rt['src_a'], rt['dst_a'], rt['op'], rt['op_a']) == (1, 0, 1, 0, 0, 0):
            enable, src, dst, op, src_a, dst_a, op_a = False, rt['src'], 0, 0, rt['src'], 0, 0
        else:
            enable, src, dst, op, src_a, dst_a, op_a = (True, rt['src'], rt['dst'], rt['op'], rt['src_a'],
                                                         rt['dst_a'], rt['op_a'])
        m = rt['mask']
        mask = ((m >> 3) & 1) | (((m >> 2) & 1) << 1) | (((m >> 1) & 1) << 2) | ((m & 1) << 3)
        out[i] = {'BlendEnable': enable, 'SrcBlend': D3D11_BLEND_COLOR[src], 'DestBlend': D3D11_BLEND_COLOR[dst],
                  'BlendOp': D3D11_BLEND_OP[op], 'SrcBlendAlpha': D3D11_BLEND_ALPHA[src_a],
                  'DestBlendAlpha': D3D11_BLEND_ALPHA[dst_a], 'BlendOpAlpha': D3D11_BLEND_OP[op_a],
                  'RenderTargetWriteMask': mask}
    return {'AlphaToCoverageEnable': bool(blend['alpha_to_mask']), 'IndependentBlendEnable': bool(independent),
            'RenderTarget': out}


def d3d11_depth_stencil_desc(depth, stencil, *, reversed_z=True):
    func = depth['func']
    if reversed_z:
        func = REVERSED_Z_FUNC.get(func, func)
    fc, fp, ff, fz = stencil['front']
    bc, bp, bf, bz = stencil['back']
    return {'DepthEnable': True, 'DepthWriteMask': 1 if depth['write'] else 0, 'DepthFunc': D3D11_COMPARE[func],
            'StencilEnable': bool(stencil['enabled']), 'StencilReadMask': stencil['read_mask'],
            'StencilWriteMask': stencil['write_mask'],
            'FrontFace': {'StencilFailOp': D3D11_STENCIL_OP[ff], 'StencilDepthFailOp': D3D11_STENCIL_OP[fz],
                          'StencilPassOp': D3D11_STENCIL_OP[fp], 'StencilFunc': D3D11_COMPARE[fc]},
            'BackFace': {'StencilFailOp': D3D11_STENCIL_OP[bf], 'StencilDepthFailOp': D3D11_STENCIL_OP[bz],
                         'StencilPassOp': D3D11_STENCIL_OP[bp], 'StencilFunc': D3D11_COMPARE[bc]}}


@dataclass
class DeviceRaster:
    """GfxDeviceD3D11Base raster-relevant device state."""
    backface_mode: bool = False        # +0x5899 SetBackfaceMode
    user_backface: bool = False        # +0x21f1 SetUserBackfaceMode
    invert_projection: bool = False    # +0x21f0 SetInvertProjectionMatrix
    wireframe: bool = False            # +0x589a SetWireframe
    scissor: bool = False              # +0x589b
    forced_cull: int = -1              # +0x21f4
    global_bias: float = 0.0           # +0x21f8 SetGlobalDepthBias
    global_slope: float = 0.0          # +0x21fc
    reversed_z: bool = True            # GraphicsCaps+0xdd


def d3d11_raster_desc(raster, dev: DeviceRaster):
    cull = dev.forced_cull if dev.forced_cull != -1 else raster['cull']
    slope = F(raster['slope_bias']) + F(dev.global_slope)
    bias = _trunc(F(raster['depth_bias']) + F(dev.global_bias))
    if dev.reversed_z:
        slope = slope * F(-1.0)
        bias = -bias
    return {'FillMode': 2 if dev.wireframe else 3, 'CullMode': D3D11_CULL[cull],
            'FrontCounterClockwise': bool(dev.backface_mode ^ dev.user_backface ^ dev.invert_projection),
            'DepthBias': bias, 'DepthBiasClamp': 0.0, 'SlopeScaledDepthBias': float(slope),
            'DepthClipEnable': bool(raster['depth_clip']), 'ScissorEnable': bool(dev.scissor),
            'MultisampleEnable': True, 'AntialiasedLineEnable': False}
