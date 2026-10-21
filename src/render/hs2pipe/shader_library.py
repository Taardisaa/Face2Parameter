"""Installed shader library and Unity 2018.4.11f1 shader variant selection.

Data comes from ``tools/offline_renderer/extract_shader_library.py`` runs on
installed assets (bundles, ``.assets``, zipmod members, plugin-embedded
bundles).  Every program is the exact installed DXBC, content-addressed by
SHA-256.

Engine behaviour reproduced here was recovered from the installed
``UnityPlayer.dll`` (CodeView GUID ``BFA91A8C44734BB4AABC5A975B304CFE`` age 1,
public PDB from Unity's symbol server); RVAs are given per rule.  Rules that
are only *documented* (Unity manual) and not disassembled are marked
``documented`` in :data:`RULES`; a rule may also be ``capture_verified``
(see ``tools/offline_renderer/shader_library_coverage_20261020.json``).

Variant selection (``Program::GetMatchingSubProgram`` RVA 0x85aff0,
``keywords::ComputeKeywordMatch`` RVA 0x85a400,
``ApplyMaterialPassAndKeywordsWithCache`` RVA 0x837340,
``ShaderState::FindSubProgramsToUse`` RVA 0x856570,
``SubProgram::CreateFromSerializedSubProgram`` RVA 0x8393e0):

1. At load each stage's ``Program`` keeps, in serialized order, only the
   SerializedSubPrograms whose ``m_ShaderHardwareTier >= 3`` (tier-independent)
   or equal to the device's active tier, whose requirement mask is satisfied by
   the device, and whose blob's GPU program type the device supports (DX11
   types on D3D11).
2. The requested keyword set is ``ShaderPassContext.keywords`` (global
   ``Shader.EnableKeyword`` state plus keywords the renderer sets for the pass,
   e.g. ``DIRECTIONAL``/``SHADOWS_SCREEN``) OR-ed with the material's keyword
   set.  It is **not** masked by the shader's keyword space; 2018.4 has only
   global keywords (256-bit ``ShaderKeywordSet``; no local keywords).
3. Each stage is selected independently with the same requested set.  An
   exact hit in the per-Program cache returns that variant; otherwise every
   variant is scored ``|req & var| - 16 * |var - req|`` and the first variant
   with the strictly greatest score wins (initial best ``INT_MIN``; variants
   recorded as unsupported are skipped).  The result is cached under the
   requested set.  An exact variant always has the maximal score, so the cache
   does not change the result.  Requested keywords absent from a variant cost
   nothing; keywords of a variant that are not requested cost 16 each.
"""
from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UNITY_VERSION = '2018.4.11f1'
UNITYPLAYER = {'guid': 'BFA91A8C44734BB4AABC5A975B304CFE', 'age': 1,
               'pdb': 'UnityPlayer_Win64_mono_x64.pdb'}
DX11_GPU_PROGRAM_TYPES = frozenset(range(15, 23))
TIER_INDEPENDENT = 3   # m_ShaderHardwareTier >= 3 is accepted for any active tier (RVA 0x839403)
MISMATCH_PENALTY = 16  # ComputeKeywordMatch: shl ecx, 4 (RVA 0x85a66a)
INT_MIN = -2 ** 31

RULES = {
    'variant_score': {'status': 'disassembled', 'rva': '0x85a400',
                      'rule': 'score = popcount(req & var) - 16 * popcount(var & ~req)'},
    'variant_scan': {'status': 'disassembled', 'rva': '0x85aff0',
                     'rule': 'serialized order, strictly greater score wins, cache keyed by requested set'},
    'requested_keywords': {'status': 'disassembled', 'rva': '0x837340/0x856570',
                           'rule': 'pass-context (global + renderer) keywords OR material keywords, unmasked'},
    'tier_filter': {'status': 'disassembled', 'rva': '0x839403',
                    'rule': 'tier >= 3 or tier == GraphicsCaps active tier; tier=None admits only tier >= 3 and '
                            'fails if a stage has tier-specific subprograms'},
    'requirements_filter': {'status': 'disassembled', 'rva': '0x83944f',
                            'rule': '(requirements & caps) == requirements; not evaluated without a caps mask'},
    'light_mode_first_pass': {'status': 'disassembled', 'rva': '0x841800',
                              'rule': 'LightMode -> pass index table keeps the first pass of each LightMode'},
    'light_mode_default_always': {'status': 'documented',
                                  'rule': 'a pass without a LightMode tag is LightMode Always'},
    'shadow_caster_pass': {'status': 'disassembled', 'rva': '0xbf96c0',
                           'rule': "active subshader's own ShadowCaster pass, else the shader-level "
                                   'fallback-resolved ShadowCaster pass'},
    'shadow_caster_fallback_resolution': {'status': 'documented',
                                          'rule': 'the shader-level ShadowCaster is taken from the Fallback chain'},
    'active_subshader': {'status': 'documented',
                         'rule': 'first supported subshader with LOD <= min(shader.maximumLOD, globalMaximumLOD); '
                                 'none -> Fallback shader'},
    'replacement_shader': {'status': 'documented',
                           'rule': 'Camera.RenderWithShader(shader, tag): empty tag renders every object with the '
                                   'replacement active subshader; otherwise the object shader tag value selects the '
                                   'first replacement subshader carrying an equal value; no match -> not rendered'},
    'use_pass': {'status': 'documented', 'rule': 'UsePass "Shader/PASS" -> passes named PASS in that shader'},
    'state_property_binding': {'status': 'documented',
                               'rule': 'a pass state value with a property name takes the material float'},
}


class ShaderLibraryError(ValueError):
    pass


# --------------------------------------------------------------- keywords
def keyword_score(requested, variant):
    """``keywords::ComputeKeywordMatch(requested, variant)``."""
    requested, variant = frozenset(requested), frozenset(variant)
    return len(requested & variant) - MISMATCH_PENALTY * len(variant - requested)


def select_index(variant_keyword_sets, requested, unsupported=()):
    """Index chosen by ``Program::GetMatchingSubProgram`` among loaded variants."""
    requested = frozenset(requested)
    unsupported = {frozenset(u) for u in unsupported}
    best, best_score = -1, INT_MIN
    for i, kw in enumerate(variant_keyword_sets):
        score = keyword_score(requested, kw)
        if score > best_score and frozenset(kw) not in unsupported:
            best, best_score = i, score
    return best, (best_score if best >= 0 else None)


# ----------------------------------------------------------- render state
UNITY_BLEND = ['Zero', 'One', 'DstColor', 'SrcColor', 'OneMinusDstColor', 'SrcAlpha', 'OneMinusSrcColor',
               'DstAlpha', 'OneMinusDstAlpha', 'SrcAlphaSaturate', 'OneMinusSrcAlpha']
D3D11_BLEND = {0: 1, 1: 2, 2: 9, 3: 3, 4: 10, 5: 5, 6: 4, 7: 7, 8: 8, 9: 11, 10: 6}
D3D11_BLEND_OP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}          # Add, Sub, RevSub, Min, Max
# D3D11 forbids colour factors in the alpha slots; the colour factor's alpha
# counterpart is used (captured: eyeshadow ``Blend DstColor Zero`` -> SrcBlendAlpha DEST_ALPHA).
ALPHA_SLOT = {3: 5, 4: 6, 9: 7, 10: 8}
UNITY_COMPARE = ['Disabled', 'Never', 'Less', 'Equal', 'LessEqual', 'Greater', 'NotEqual', 'GreaterEqual',
                 'Always']
D3D11_COMPARE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
REVERSED_Z = {2: 5, 4: 7, 5: 2, 7: 4}                      # Less<->Greater, LEqual<->GEqual
UNITY_CULL = ['Off', 'Front', 'Back']
D3D11_CULL = {0: 1, 1: 2, 2: 3}
UNITY_STENCIL_OP = ['Keep', 'Zero', 'Replace', 'IncrSat', 'DecrSat', 'Invert', 'IncrWrap', 'DecrWrap']
D3D11_STENCIL_OP = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8}
NONINIT = '<noninit>'


def _value(entry, material_floats, used):
    name = entry['name']
    if name == NONINIT:
        return entry['val'], None
    if name not in material_floats:
        raise ShaderLibraryError(f'Pass state is bound to property {name}; material value required')
    used[name] = material_floats[name]
    return float(material_floats[name]), name


def resolve_state(state, material_floats=None):
    """Unity-level pass render state with property-bound values substituted.

    ``material_floats`` maps property names to the material's float values (the
    native ``Material.GetFloat``).  Every returned field carries the bound
    property name (or ``None`` for a constant).
    """
    material_floats = material_floats or {}
    used = {}

    def v(e):
        return _value(e, material_floats, used)

    def blend(rt):
        out = {}
        for key in ('srcBlend', 'destBlend', 'srcBlendAlpha', 'destBlendAlpha', 'blendOp', 'blendOpAlpha', 'colMask'):
            val, prop = v(rt[key])
            out[key] = {'value': int(val), 'property': prop}
        return out

    def stencil(s):
        return {k: dict(zip(('value', 'property'), (int(x[0]), x[1]))) for k in ('pass', 'fail', 'zFail', 'comp')
                for x in [v(s[k])]}

    res = {'blend': [blend(state[f'rtBlend{i}']) for i in range(8)], 'separate_blend': bool(state['rtSeparateBlend'])}
    for key in ('zClip', 'zTest', 'zWrite', 'culling', 'alphaToMask', 'stencilReadMask', 'stencilWriteMask', 'stencilRef'):
        val, prop = v(state[key])
        res[key] = {'value': int(val), 'property': prop}
    for key in ('offsetFactor', 'offsetUnits'):
        val, prop = v(state[key])
        res[key] = {'value': float(val), 'property': prop}
    res['stencil'] = {k: stencil(state[k]) for k in ('stencilOp', 'stencilOpFront', 'stencilOpBack')}
    res['fog_mode'] = state.get('fogMode')
    res['lighting'] = state.get('lighting')
    res['material_floats_used'] = used
    return res


def unity_color_mask_to_d3d11(mask):
    """Unity ColorWriteMask (A=1,B=2,G=4,R=8) -> D3D11 (R=1,G=2,B=4,A=8)."""
    return ((mask >> 3) & 1) | (((mask >> 2) & 1) << 1) | (((mask >> 1) & 1) << 2) | ((mask & 1) << 3)


def blend_is_disabled(rt):
    return rt['srcBlend']['value'] == 1 and rt['destBlend']['value'] == 0 and \
        rt['srcBlendAlpha']['value'] == 1 and rt['destBlendAlpha']['value'] == 0


def to_d3d11(resolved, *, reversed_z=True):
    """Map resolved Unity state to D3D11 descriptor fields.

    ``reversed_z``: Unity uses a reversed depth buffer on D3D11 and flips the
    depth comparison.  Culling is returned as authored; render-target/projection
    flips that invert winding are a renderer decision, not shader state.  Depth
    bias conversion is not derived here (``offset_*`` is returned untranslated).
    """
    def bd(rt):
        return {'BlendEnable': not blend_is_disabled(rt), 'SrcBlend': D3D11_BLEND[rt['srcBlend']['value']],
                'DestBlend': D3D11_BLEND[rt['destBlend']['value']], 'BlendOp': D3D11_BLEND_OP[rt['blendOp']['value']],
                'SrcBlendAlpha': ALPHA_SLOT.get(D3D11_BLEND[rt['srcBlendAlpha']['value']],
                                                D3D11_BLEND[rt['srcBlendAlpha']['value']]),
                'DestBlendAlpha': ALPHA_SLOT.get(D3D11_BLEND[rt['destBlendAlpha']['value']],
                                                 D3D11_BLEND[rt['destBlendAlpha']['value']]),
                'BlendOpAlpha': D3D11_BLEND_OP[rt['blendOpAlpha']['value']],
                'RenderTargetWriteMask': unity_color_mask_to_d3d11(rt['colMask']['value'])}
    z = resolved['zTest']['value']
    if z == 0:
        depth_enable, func = False, 8
    else:
        depth_enable, func = True, D3D11_COMPARE[REVERSED_Z.get(z, z) if reversed_z else z]
    front, back = resolved['stencil']['stencilOpFront'], resolved['stencil']['stencilOpBack']

    def face(s):
        return {'StencilFailOp': D3D11_STENCIL_OP[s['fail']['value']],
                'StencilDepthFailOp': D3D11_STENCIL_OP[s['zFail']['value']],
                'StencilPassOp': D3D11_STENCIL_OP[s['pass']['value']],
                'StencilFunc': D3D11_COMPARE.get(s['comp']['value'], 8)}
    return {'blend': {'AlphaToCoverageEnable': bool(resolved['alphaToMask']['value']),
                      'IndependentBlendEnable': resolved['separate_blend'],
                      'RenderTarget': [bd(rt) for rt in resolved['blend']]},
            'depth_stencil': {'DepthEnable': depth_enable, 'DepthWriteMask': 1 if resolved['zWrite']['value'] else 0,
                              'DepthFunc': func, 'StencilReadMask': resolved['stencilReadMask']['value'],
                              'StencilWriteMask': resolved['stencilWriteMask']['value'],
                              'FrontFace': face(front), 'BackFace': face(back)},
            'stencil_ref': resolved['stencilRef']['value'],
            'cull_unity': UNITY_CULL[resolved['culling']['value']],
            'cull_d3d11_as_authored': D3D11_CULL[resolved['culling']['value']],
            'depth_clip': bool(resolved['zClip']['value']),
            'offset_factor': resolved['offsetFactor']['value'], 'offset_units': resolved['offsetUnits']['value']}


# ------------------------------------------------------------ reflection
@dataclass(frozen=True)
class Variable:
    name: str
    offset: int        # bytes
    kind: str          # float | int | bool
    rows: int          # 1 for vectors; registers for matrices
    columns: int       # vector width (matrices: 4)
    array_size: int    # 0 = not an array
    matrix: bool

    @property
    def components(self):
        return (self.array_size or 1) * self.rows * self.columns

    @property
    def byte_size(self):
        """HLSL footprint: array elements and matrix registers start on 16 bytes."""
        n = self.array_size or 1
        if self.matrix:
            return 16 * self.rows * n
        return 16 * (n - 1) + 4 * self.columns if n > 1 else 4 * self.columns


@dataclass
class CBufferLayout:
    name: str
    size: int
    variables: list
    slot: int | None = None
    declared_float4: int | None = None
    origin: str = 'serialized_subprogram'

    def variable(self, name):
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def coverage_mask(self):
        """Bytes covered by declared variables (False = padding/undeclared)."""
        mask = np.zeros(self.size, bool)
        for v in self.variables:
            for e in range(v.array_size or 1):
                if v.matrix:
                    base = v.offset + 64 * e if v.rows == 4 else v.offset + 16 * v.rows * e
                    mask[base:base + 16 * v.rows] = True
                else:
                    base = v.offset + 16 * e
                    mask[base:base + 4 * v.columns] = True
        return mask


KINDS = {0: 'float', 1: 'int', 2: 'bool'}


def layouts_from_bindings(bindings, names):
    """CBuffer layouts from one SerializedSubProgram (exact Unity names/offsets)."""
    if bindings.get('m_VectorParams') or bindings.get('m_MatrixParams'):
        raise ShaderLibraryError('Loose (non-cbuffer) uniforms are not a D3D11 layout')
    slots = {b['m_NameIndex']: b['m_Index'] for b in bindings['m_ConstantBufferBindings']}
    out = []
    for cb in bindings['m_ConstantBuffers']:
        variables = []
        # Struct arrays (UnityInstancing_* buffers): one flattened variable per
        # element member, ``struct[e].member``, at m_Index + e*m_StructSize + member.
        for st in cb.get('m_StructParams', []):
            sname = names[st['m_NameIndex']]
            for e in range(st['m_ArraySize'] or 1):
                prefix = f'{sname}[{e}].' if st['m_ArraySize'] else sname + '.'
                base = st['m_Index'] + e * st['m_StructSize']
                for p in st['m_VectorMembers']:
                    variables.append(Variable(prefix + names[p['m_NameIndex']], base + p['m_Index'],
                                              KINDS[p['m_Type']], 1, p['m_Dim'], p['m_ArraySize'], False))
                for p in st['m_MatrixMembers']:
                    if p['m_RowCount'] != 4:
                        raise ShaderLibraryError('Struct matrix member without 4 registers')
                    variables.append(Variable(prefix + names[p['m_NameIndex']], base + p['m_Index'],
                                              KINDS[p['m_Type']], 4, 4, p['m_ArraySize'], True))
        for p in cb['m_VectorParams']:
            variables.append(Variable(names[p['m_NameIndex']], p['m_Index'], KINDS[p['m_Type']], 1, p['m_Dim'],
                                      p['m_ArraySize'], False))
        for p in cb['m_MatrixParams']:
            if p['m_RowCount'] != 4:
                raise ShaderLibraryError(f"Matrix {names[p['m_NameIndex']]} with {p['m_RowCount']} rows "
                                         'requires its column layout')
            variables.append(Variable(names[p['m_NameIndex']], p['m_Index'], KINDS[p['m_Type']], 4, 4,
                                      p['m_ArraySize'], True))
        variables.sort(key=lambda v: v.offset)
        out.append(CBufferLayout(names[cb['m_NameIndex']], cb['m_Size'], variables,
                                 slot=slots.get(cb['m_NameIndex'])))
    return out


def pack_cbuffer(layout, values, *, size=None, base=None, allow_missing=False):
    """Pack uniform values into exact constant-buffer bytes.

    Same encoding contract as ``src/render/shader_bindings.constant_buffers``:
    values are already GPU encodings (linear colors where Unity converts,
    matrices flattened in GPU constant-memory order, i.e. Unity's column-major
    ``Matrix4x4f``), float -> IEEE binary32, int -> int32, bool -> 0/1 uint32;
    array elements and matrix registers are float4-aligned.  Undeclared bytes
    come from ``base`` (e.g. a captured buffer's prior bytes) or are zero; use
    :meth:`CBufferLayout.coverage_mask` to compare only declared bytes.
    Missing variables raise unless ``allow_missing``.
    """
    size = layout.size if size is None else size
    if base is not None:
        buf = bytearray(base)
        if len(buf) < size:
            raise ValueError('Base bytes shorter than the constant buffer')
    else:
        buf = bytearray(size)
    for v in layout.variables:
        if v.name not in values:
            if allow_missing:
                continue
            raise ShaderLibraryError(f'Missing value for {layout.name}.{v.name}')
        arr = np.asarray(values[v.name])
        if arr.size != v.components:
            raise ValueError(f'{v.name}: expected {v.components} components, got {arr.size}')
        flat = arr.reshape(-1)
        if v.kind == 'float':
            words = flat.astype(np.float32).view(np.uint32)
        elif v.kind == 'int':
            words = flat.astype(np.int64).astype(np.int32).view(np.uint32)
        else:
            words = flat.astype(bool).astype(np.uint32)
        if v.offset % 4:
            raise ShaderLibraryError(f'{v.name}: non-aligned offset')
        width = v.rows * v.columns
        for e in range(v.array_size or 1):
            chunk = words[e * width:(e + 1) * width]
            if v.matrix:
                start = v.offset + 64 * e
            else:
                start = v.offset + 16 * e
            end = start + 4 * len(chunk)
            if end > size:
                raise ValueError(f'{v.name} extends beyond the {size}-byte buffer')
            buf[start:end] = struct.pack('<%dI' % len(chunk), *map(int, chunk))
    return bytes(buf)


def unpack_cbuffer(layout, data):
    """Inverse of :func:`pack_cbuffer` for declared variables (used for checks)."""
    out = {}
    for v in layout.variables:
        width = v.rows * v.columns
        vals = []
        for e in range(v.array_size or 1):
            start = v.offset + (64 if v.matrix else 16) * e
            if start + 4 * width > len(data):
                break
            raw = np.frombuffer(data[start:start + 4 * width], np.uint32)
            vals.append(raw)
        raw = np.concatenate(vals) if vals else np.zeros(0, np.uint32)
        out[v.name] = raw.view(np.float32) if v.kind == 'float' else raw.view(np.int32) if v.kind == 'int' \
            else raw.astype(bool)
    return out


# ------------------------------------------- Unity built-in cbuffers (source)
HLSL_SIZES = {'float': 1, 'half': 1, 'fixed': 1, 'int': 1, 'uint': 1, 'bool': 1}


def _eval_condition(expr, defines):
    expr = re.sub(r'defined\s*\(\s*(\w+)\s*\)', lambda m: str(m.group(1) in defines), expr)
    expr = expr.replace('&&', ' and ').replace('||', ' or ')
    expr = re.sub(r'!(?!=)', ' not ', expr)
    expr = re.sub(r'\b([A-Z_][A-Z0-9_]*)\b', lambda m: repr(defines.get(m.group(1), 0)), expr)
    return bool(eval(expr, {'__builtins__': {}}, {'True': True, 'False': False}))  # noqa: S307


def parse_builtin_cbuffers(path, defines):
    """Unity built-in cbuffer layouts from the 2018.4.11f1 editor's
    ``CGIncludes/UnityShaderVariables.cginc`` for one define set.

    Packing: HLSL cbuffer rules (a variable never straddles a 16-byte row;
    array elements and matrix columns start rows; column_major float4x4).
    ``half``/``fixed`` are 4 bytes on D3D11 (checked against installed program
    layouts by :meth:`ShaderLibrary.check_builtin_layouts`).
    """
    defines = dict(defines)
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    stack, buffers, current = [], {}, None
    active = True
    for raw in lines:
        line = raw.split('//')[0].strip()
        if not line:
            continue
        m = re.match(r'#\s*(ifdef|ifndef|if|elif|else|endif|define|undef)\b\s*(.*)', line)
        if m:
            d, arg = m.group(1), m.group(2).strip()
            if d in ('ifdef', 'ifndef', 'if'):
                if d == 'ifdef':
                    cond = arg in defines
                elif d == 'ifndef':
                    cond = arg not in defines
                else:
                    cond = _eval_condition(arg, defines)
                stack.append([active, cond and active, cond])
                active = cond and active
            elif d == 'elif':
                parent, _, taken = stack[-1]
                cond = (not taken) and _eval_condition(arg, defines)
                stack[-1] = [parent, parent and cond, taken or cond]
                active = parent and cond
            elif d == 'else':
                parent, _, taken = stack[-1]
                stack[-1] = [parent, parent and not taken, True]
                active = parent and not taken
            elif d == 'endif':
                active = stack.pop()[0]
            elif active and d == 'define':
                parts = arg.split(None, 1)
                if re.fullmatch(r'\w+', parts[0]):
                    val = parts[1].strip() if len(parts) > 1 else '1'
                    defines[parts[0]] = int(val) if re.fullmatch(r'-?\d+', val) else val
            elif active and d == 'undef':
                defines.pop(arg, None)
            continue
        if not active:
            continue
        m = re.match(r'(?:GLOBAL_)?CBUFFER_START\((\w+)\)', line)
        if m:
            current = {'name': m.group(1), 'variables': [], 'offset': 0}
            continue
        if re.match(r'(?:GLOBAL_)?CBUFFER_END', line):
            size = (current['offset'] + 15) // 16 * 16
            buffers[current['name']] = CBufferLayout(current['name'], size, current['variables'],
                                                     origin='UnityShaderVariables.cginc')
            current = None
            continue
        if current is None:
            continue
        m = re.match(r'(float|half|fixed|int|uint|bool)(\d)?(?:x(\d))?\s+(.+);', line)
        if not m:
            raise ShaderLibraryError('Unparsed cbuffer declaration: ' + line)
        base, a, b, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        kind = 'int' if base in ('int', 'uint') else 'bool' if base == 'bool' else 'float'
        for decl in [s.strip() for s in rest.split(',')]:
            dm = re.fullmatch(r'(\w+)(?:\[(\d+)\])?', decl)
            name, count = dm.group(1), int(dm.group(2) or 0)
            if b:   # matrix: column_major, one register per column
                rows_, cols = int(a), int(b)
                if (rows_, cols) != (4, 4):
                    raise ShaderLibraryError('Only float4x4 built-in matrices are expected')
                off = (current['offset'] + 15) // 16 * 16
                var = Variable(name, off, kind, 4, 4, count, True)
                current['offset'] = off + 64 * (count or 1)
            else:
                width = int(a or 1)
                off = current['offset']
                if count or (off % 16) + 4 * width > 16:
                    off = (off + 15) // 16 * 16
                var = Variable(name, off, kind, 1, width, count, False)
                current['offset'] = off + (16 * (count - 1) + 4 * width if count else 4 * width)
            current['variables'].append(var)
    return buffers


D3D11_STANDALONE_DEFINES = {'SHADER_API_D3D11': 1, 'UNITY_LIGHT_PROBE_PROXY_VOLUME': 1}


# ------------------------------------------------------------- objects
@dataclass
class Subprogram:
    shader: 'Shader'
    subshader: int
    pass_index: int
    stage: str
    record: dict
    names: dict = field(repr=False, default=None)

    @property
    def order(self):
        return self.record['order']

    @property
    def keywords(self):
        return frozenset(self.record['keywords'])

    @property
    def tier(self):
        return self.record['tier']

    @property
    def dxbc_sha256(self):
        return self.record['dxbc_sha256']

    def dxbc(self):
        return self.shader.library.dxbc(self.dxbc_sha256)

    def reflection(self):
        return self.shader.library.reflection(self.dxbc_sha256)

    def constant_buffers(self):
        layouts = layouts_from_bindings(self.record['bindings'], self.names)
        decl = self.reflection()['cb']
        for cb in layouts:
            if cb.slot is not None and str(cb.slot) in decl:
                cb.declared_float4 = decl[str(cb.slot)]['float4_count']
        return layouts

    def uniform_map(self):
        """name -> (cbuffer name, slot, Variable)."""
        out = {}
        for cb in self.constant_buffers():
            for v in cb.variables:
                if v.name in out:
                    raise ShaderLibraryError('Uniform declared in two cbuffers: ' + v.name)
                out[v.name] = (cb.name, cb.slot, v)
        return out

    def resources(self):
        b, n = self.record['bindings'], self.names
        return {'textures': [{'name': n[t['m_NameIndex']], 'srv': t['m_Index'], 'sampler': t['m_SamplerIndex'],
                              'multisampled': t.get('m_MultiSampled'), 'dim': t['m_Dim']}
                             for t in b['m_TextureParams']],
                'buffers': [{'name': n[t['m_NameIndex']], 'srv': t['m_Index']} for t in b.get('m_BufferParams', [])],
                'uavs': [{'name': n[t['m_NameIndex']], 'uav': t['m_Index'], 'origin': t.get('m_OriginalIndex')}
                         for t in b.get('m_UAVParams', [])],
                'inline_samplers': b.get('m_Samplers', []),
                'constant_buffer_slots': {n[c['m_NameIndex']]: c['m_Index'] for c in b['m_ConstantBufferBindings']}}

    def identity(self):
        return {'shader': self.shader.name, 'subshader': self.subshader, 'pass': self.pass_index, 'stage': self.stage,
                'order': self.order, 'keywords': sorted(self.keywords), 'tier': self.tier,
                'dxbc_sha256': self.dxbc_sha256, 'source': self.shader.source_label}


@dataclass
class Pass:
    shader: 'Shader'
    subshader: int
    record: dict

    @property
    def index(self):
        return self.record['index']

    @property
    def type(self):
        return {0: 'normal', 1: 'use', 2: 'grab'}[self.record['type']]

    @property
    def name(self):
        return self.record['name'] or self.record['state_name']

    @property
    def tags(self):
        """Pass tags (serialized pass tags and state tags; keys upper-cased)."""
        out = {}
        for k, v in self.record['tags'] + self.record['state_tags']:
            out[k.upper()] = v
        return out

    @property
    def light_mode(self):
        return self.tags.get('LIGHTMODE', 'ALWAYS').upper()

    def subprograms(self, stage):
        names = {i: n for n, i in self.record['name_indices']}
        return [Subprogram(self.shader, self.subshader, self.index, stage, r, names)
                for r in self.record['stages'].get(stage, [])]

    def stages(self):
        return list(self.record['stages'])

    def loaded_subprograms(self, stage, *, tier=None, device_requirements=None):
        """``SubProgram::CreateFromSerializedSubProgram`` admission, in serialized order."""
        out = []
        for sp in self.subprograms(stage):
            if tier is None and sp.tier < TIER_INDEPENDENT:
                raise ShaderLibraryError(f'{self.shader.name} pass {self.index} {stage} has tier-specific '
                                         'subprograms; the device active tier is required')
            if not (sp.tier >= TIER_INDEPENDENT or sp.tier == tier):
                continue
            if sp.record['gpu_program_type'] not in DX11_GPU_PROGRAM_TYPES or sp.dxbc_sha256 is None:
                continue
            req = sp.record.get('shader_requirements')
            if device_requirements is not None and req is not None and (req & device_requirements) != req:
                continue
            out.append(sp)
        return out

    def select(self, stage, keywords, *, tier=None, device_requirements=None, unsupported=()):
        loaded = self.loaded_subprograms(stage, tier=tier, device_requirements=device_requirements)
        if not loaded:
            return None
        i, score = select_index([sp.keywords for sp in loaded], keywords, unsupported)
        if i < 0:
            return None
        return Selection(loaded[i], score, frozenset(keywords), exact=loaded[i].keywords == frozenset(keywords),
                         requirements_checked=device_requirements is not None)

    def select_all(self, keywords, *, tier=None, device_requirements=None):
        return {stage: self.select(stage, keywords, tier=tier, device_requirements=device_requirements)
                for stage in self.stages()}

    def render_state(self, material_floats=None):
        return resolve_state(self.record['state'], material_floats)


@dataclass(frozen=True)
class Selection:
    subprogram: Subprogram
    score: int
    requested: frozenset
    exact: bool
    requirements_checked: bool

    @property
    def dxbc_sha256(self):
        return self.subprogram.dxbc_sha256


@dataclass
class SubShader:
    shader: 'Shader'
    record: dict

    @property
    def index(self):
        return self.record['index']

    @property
    def lod(self):
        return self.record['lod']

    @property
    def tags(self):
        return {k: v for k, v in self.record['tags']}

    def tag(self, key):
        for k, v in self.record['tags']:
            if k.upper() == key.upper():
                return v
        return None

    @property
    def passes(self):
        return [Pass(self.shader, self.index, p) for p in self.record['passes']]

    def passes_with_light_mode(self, light_mode):
        return [p for p in self.passes if p.light_mode == light_mode.upper()]

    def first_pass_with_light_mode(self, light_mode):
        rows = self.passes_with_light_mode(light_mode)
        return rows[0] if rows else None

    def passes_named(self, name):
        return [p for p in self.passes if p.name.upper() == name.upper()]

    def supported(self, tier=None):
        """Every normal pass has a loadable DX11 subprogram for each exported stage."""
        for p in self.passes:
            if p.type != 'normal':
                continue
            for stage in p.stages():
                if not p.loaded_subprograms(stage, tier=tier):
                    return False
        return True


class Shader:
    def __init__(self, library, record, source):
        self.library, self.record, self.source = library, record, source

    name = property(lambda self: self.record['name'])
    fallback = property(lambda self: self.record['fallback'] or None)
    properties = property(lambda self: self.record['properties'])

    @property
    def source_label(self):
        s = self.source
        return s.get('member') and f"{s['archive']}::{s['member']}" or s.get('path')

    @property
    def subshaders(self):
        return [SubShader(self, s) for s in self.record['subshaders']]

    def active_subshader(self, *, tier=None, maximum_lod=None):
        for s in self.subshaders:
            if maximum_lod is not None and s.lod > maximum_lod:
                continue
            if s.supported(tier):
                return s
        return None

    def resolve(self, *, tier=None, maximum_lod=None, _seen=None):
        """(shader, subshader) actually used: own active subshader or Fallback chain."""
        _seen = set() if _seen is None else _seen
        if self.name in _seen:
            raise ShaderLibraryError('Fallback cycle at ' + self.name)
        _seen.add(self.name)
        sub = self.active_subshader(tier=tier, maximum_lod=maximum_lod)
        if sub is not None:
            return self, sub
        if not self.fallback or self.fallback.lower() == 'off':
            return self, None
        return self.library.shader(self.fallback).resolve(tier=tier, maximum_lod=maximum_lod, _seen=_seen)

    def shadow_caster_pass(self, *, tier=None, maximum_lod=None):
        """``Shader::GetShadowCasterPassToUse`` with the fallback-chain shader-level pass."""
        shader, sub = self.resolve(tier=tier, maximum_lod=maximum_lod)
        if sub is not None:
            own = sub.first_pass_with_light_mode('SHADOWCASTER')
            if own is not None:
                return own
        seen = {shader.name}
        fb = shader.fallback
        while fb and fb.lower() != 'off':
            target = self.library.shader(fb)
            if target.name in seen:
                raise ShaderLibraryError('Fallback cycle at ' + fb)
            seen.add(target.name)
            s = target.active_subshader(tier=tier, maximum_lod=maximum_lod)
            if s is not None and s.first_pass_with_light_mode('SHADOWCASTER') is not None:
                return s.first_pass_with_light_mode('SHADOWCASTER')
            fb = target.fallback
        return None

    def use_pass(self, pass_record, *, tier=None):
        """Resolve a serialized ``UsePass`` (type 1) by its ``SHADER/PASS`` name."""
        use = pass_record.record['use_name']
        shader_name, _, pass_name = use.rpartition('/')
        target, sub = self.library.shader(shader_name).resolve(tier=tier)
        if sub is None:
            raise ShaderLibraryError('UsePass target has no supported subshader: ' + use)
        rows = sub.passes_named(pass_name)
        if not rows:
            raise ShaderLibraryError('UsePass pass not found: ' + use)
        return rows

    def tag(self, key, *, tier=None):
        _, sub = self.resolve(tier=tier)
        return None if sub is None else sub.tag(key)


def replacement_subshader(replacement, tag, object_shader, *, tier=None):
    """``Camera.RenderWithShader(replacement, tag)`` subshader choice for one object."""
    if not tag:
        return replacement.active_subshader(tier=tier)
    value = object_shader.tag(tag, tier=tier)
    if value is None:
        return None
    for s in replacement.subshaders:
        if s.tag(tag) == value and s.supported(tier):
            return s
    return None


# ------------------------------------------------------------- library
class ShaderLibrary:
    """All installed shader copies from one or more extraction directories."""

    def __init__(self, roots):
        self.roots = [Path(r) for r in roots]
        self._shaders = {}
        self._computes = {}
        self._dxbc_root = {}
        self._reflection = {}
        for root in self.roots:
            lib = json.loads((root / 'library.json').read_text(encoding='utf-8'))
            refl = json.loads((root / 'reflection.json').read_text(encoding='utf-8'))
            self._reflection.update(refl)
            for sha in refl:
                self._dxbc_root.setdefault(sha, root)
            for src in lib['sources']:
                manifest = json.loads((root / src['manifest']).read_text(encoding='utf-8'))
                for rec in manifest['shaders']:
                    self._shaders.setdefault(rec['name'], []).append(Shader(self, rec, manifest['source']))
                for rec in manifest['compute_shaders']:
                    self._computes.setdefault(rec['name'], []).append({'record': rec, 'source': manifest['source']})

    # -- lookup
    def shader_names(self):
        return sorted(self._shaders)

    def copies(self, name):
        return list(self._shaders.get(name, []))

    def shader(self, name, source=None):
        """One shader by name.  Several installed copies must be program-identical
        unless ``source`` (substring of the source path) disambiguates."""
        rows = self.copies(name)
        if source is not None:
            rows = [s for s in rows if source in (s.source_label or '')]
        if not rows:
            raise ShaderLibraryError('Shader not in library: ' + name)
        sig = {_program_signature(s.record) for s in rows}
        if len(sig) > 1:
            raise ShaderLibraryError(f'{name}: {len(rows)} installed copies differ; pass source=')
        return rows[0]

    def compute(self, name, source=None):
        rows = [r for r in self._computes.get(name, [])
                if source is None or source in (r['source'].get('path') or '')]
        if not rows:
            raise ShaderLibraryError('ComputeShader not in library: ' + name)
        return rows[0]

    def compute_kernel(self, name, kernel, *, target_renderer=2):
        rec = self.compute(name)['record']
        for v in rec['variants']:
            if v['target_renderer'] != target_renderer:
                continue
            for k in v['kernels']:
                if k['name'] == kernel:
                    return {'kernel': k, 'constant_buffers': v['constant_buffers'], 'variant': v}
        raise ShaderLibraryError(f'{name}.{kernel} has no renderer {target_renderer} kernel')

    def dxbc(self, sha):
        return (self._dxbc_root[sha] / 'dxbc' / (sha + '.dxbc')).read_bytes()

    def assembly(self, sha):
        return (self._dxbc_root[sha] / 'asm' / (sha + '.asm')).read_text(encoding='utf-8')

    def reflection(self, sha):
        return self._reflection[sha]

    # -- reverse identity
    def match(self, sha):
        """Every installed (shader, pass, stage, variant) identity of one DXBC hash."""
        out = []
        for name, copies in self._shaders.items():
            for shader in copies:
                for sub in shader.subshaders:
                    for p in sub.passes:
                        for stage in p.stages():
                            for sp in p.subprograms(stage):
                                if sp.dxbc_sha256 == sha:
                                    row = sp.identity()
                                    row['pass_name'] = p.name
                                    row['light_mode'] = p.light_mode
                                    out.append(row)
        for name, copies in self._computes.items():
            for c in copies:
                for v in c['record']['variants']:
                    for k in v['kernels']:
                        if k.get('dxbc_sha256') == sha:
                            out.append({'compute_shader': name, 'kernel': k['name'], 'stage': 'cs',
                                        'target_renderer': v['target_renderer'], 'target_level': v['target_level'],
                                        'dxbc_sha256': sha, 'source': c['source'].get('path')
                                        or f"{c['source'].get('archive')}::{c['source'].get('member')}"})
        return out

    def hash_index(self):
        index = {}
        for name, copies in self._shaders.items():
            for shader in copies:
                for sub in shader.subshaders:
                    for p in sub.passes:
                        for stage in p.stages():
                            for sp in p.subprograms(stage):
                                if sp.dxbc_sha256:
                                    index.setdefault(sp.dxbc_sha256, []).append(sp.identity())
        for name, copies in self._computes.items():
            for c in copies:
                for v in c['record']['variants']:
                    for k in v['kernels']:
                        if k.get('dxbc_sha256'):
                            index.setdefault(k['dxbc_sha256'], []).append(
                                {'compute_shader': name, 'kernel': k['name'], 'stage': 'cs',
                                 'target_renderer': v['target_renderer'], 'dxbc_sha256': k['dxbc_sha256'],
                                 'source': c['source'].get('path')})
        return index

    # -- built-in cbuffer catalog cross-check
    def check_builtin_layouts(self, builtin):
        """Compare source-derived built-in layouts with every installed program.

        Returns {cbuffer: {'programs': n, 'mismatches': [...]}}; a mismatch is a
        variable whose serialized offset/kind/width differs from the source.
        """
        report = {}
        for copies in self._shaders.values():
            for shader in copies:
                for sub in shader.subshaders:
                    for p in sub.passes:
                        for stage in p.stages():
                            for sp in p.subprograms(stage):
                                if not sp.dxbc_sha256:
                                    continue
                                for cb in sp.constant_buffers():
                                    if cb.name not in builtin:
                                        continue
                                    ref = builtin[cb.name]
                                    row = report.setdefault(cb.name, {'programs': 0, 'mismatches': [],
                                                                      'source_size': ref.size, 'sizes': set()})
                                    row['programs'] += 1
                                    row['sizes'].add(cb.size)
                                    for v in cb.variables:
                                        try:
                                            r = ref.variable(v.name)
                                        except KeyError:
                                            row['mismatches'].append({'variable': v.name, 'reason': 'absent in source',
                                                                      'shader': shader.name})
                                            continue
                                        if (r.offset, r.kind, r.columns, r.array_size, r.matrix) != \
                                                (v.offset, v.kind, v.columns, v.array_size, v.matrix):
                                            row['mismatches'].append({'variable': v.name, 'shader': shader.name,
                                                                      'serialized': v.__dict__, 'source': r.__dict__})
        for row in report.values():
            row['sizes'] = sorted(row['sizes'])
        return report


def _program_signature(record):
    rows = []
    for s in record['subshaders']:
        for p in s['passes']:
            for stage, subs in sorted(p['stages'].items()):
                rows.append((s['index'], p['index'], stage,
                             tuple((sp['order'], sp['tier'], tuple(sp['keywords']), sp['dxbc_sha256']) for sp in subs)))
    return hash(tuple(rows))
