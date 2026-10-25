"""Head skinning data path of an actual HS2 D3D11 RenderDoc capture.

Source/evidence basis (see docs/hs2_offline_head_skinning.md):

* The captured frame contains six compute shaders, all PostProcessing v2
  (MultiScaleVO*, Lut3DBaker); every one reads/writes typed textures only and
  no buffer UAV, stream-out target or skinning compute exists. Unity 2018.4
  therefore skins the head on the CPU. Each active SkinnedMeshRenderer owns a
  DYNAMIC vertex buffer (debug name ``Buffer-1-<bytes>``) holding 40-byte
  vertices: POSITION float3 @0, NORMAL float3 @12, TANGENT float4 @24. Blend
  shapes are applied on the CPU before skinning; no GPU blendshape stream exists.
* Static mesh streams (UV/colour, plus an unreferenced 32-byte stream) and the
  index buffer are DEFAULT-usage buffers named after the mesh and never change.
* Skinned positions/normals/tangents are expressed in the renderer's actual
  root bone frame **without scale**; scale is part of the skinning matrices.
  ``unity_ObjectToWorld`` in the per-draw cbuffer is TRS(root.position,
  root.rotation, 1). For these head surfaces the root bone is the ABMX-modified
  ``cf_J_FaceUp_ty``, so root modifiers change both the VB and per-draw CBs.
* RenderDoc 1.46 D3D11 stores every buffer's capture-start contents in the
  ``InitialData`` blob of its CreateBuffer chunk (``Serialise_InitialState``
  writes the staging copy into ``record->GetDataPtr()``), so buffers have no
  separate Initial Contents chunk. Draws before the in-frame skinning Unmap use
  those initial bytes.

This module is CPU-only. It indexes the official zip.xml export, tracks buffer
writes and draw bindings, binds source surfaces to their real GPU resources by
content, and generates source-derived replacement bytes plus a byte patch plan
compatible with :func:`src.render.capture_mutation.mutate_capture`. Nothing here
fits, offsets or thresholds captured data into agreement; unresolved inputs are
reported as explicit gaps.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SKIN_STRIDE = 40
SKIN_ELEMENTS = (('POSITION', 0, 6, 0), ('NORMAL', 0, 6, 12), ('TANGENT', 0, 2, 24))
DXGI_R16_UINT = 57
DXGI_R32_UINT = 42
DXGI_R32G32B32_FLOAT = 6
DXGI_R32G32_FLOAT = 16
D3D11_USAGE_DEFAULT = 0
D3D11_USAGE_DYNAMIC = 2
D3D11_BIND_VERTEX_BUFFER = 1
D3D11_BIND_INDEX_BUFFER = 2
D3D11_BIND_CONSTANT_BUFFER = 4
MAP_WRITE_DISCARD = 4
UAV_DIMENSION_BUFFER = 1
# Buffer-writing context methods that must be accounted for explicitly.
BUFFER_WRITERS = {'Unmap', 'UpdateSubresource', 'UpdateSubresource1', 'CopyResource',
                  'CopySubresourceRegion', 'CopySubresourceRegion1', 'CopyStructureCount',
                  'DiscardResource', 'DiscardView', 'DiscardView1', 'SOSetTargets',
                  'ClearUnorderedAccessViewUint', 'ClearUnorderedAccessViewFloat'}


class SkinPathError(RuntimeError):
    """A source or capture branch is not implemented; never silently skipped."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Capture index
# --------------------------------------------------------------------------

@dataclass
class BufferRef:
    ordinal: int          # document order among <buffer> nodes of the chunk
    path: str             # structured field path inside the chunk
    blob: int             # external ZIP member index
    length: int


@dataclass
class Chunk:
    index: int
    name: str
    fields: dict
    buffers: list = field(default_factory=list)

    @property
    def method(self):
        return self.name.split('::')[-1]


def _decode(node, path, refs):
    tag = node.tag
    text = (node.text or '').strip()
    if tag in ('uint', 'int', 'enum', 'ResourceId'):
        return int(text)
    if tag in ('float', 'double'):
        return float(text)
    if tag == 'bool':
        if text not in ('true', 'false', '1', '0'):
            raise SkinPathError('Unknown serialized boolean')
        return text in ('true', '1')
    if tag == 'string':
        return text
    if tag == 'null':
        return None
    if tag == 'buffer':
        ref = BufferRef(len(refs), path, int(text), int(node.attrib['byteLength']))
        refs.append(ref)
        return ref
    if tag == 'array':
        return [_decode(c, f'{path}[{i}]', refs) for i, c in enumerate(node)]
    result = {}
    for child in node:
        name = child.attrib.get('name')
        if name is None:
            raise SkinPathError('Unnamed structured field in ' + path)
        value = _decode(child, f'{path}.{name}' if path else name, refs)
        if name in result:
            old = result[name]
            if isinstance(old, _Repeated):
                old.append(value)
            else:
                result[name] = _Repeated([old, value])
        else:
            result[name] = value
    return result


class _Repeated(list):
    _repeated = True


class CaptureIndex:
    """All chunks of an official RenderDoc zip.xml export with exact blob refs."""

    def __init__(self, xml_path, zip_path=None):
        self.xml_path = Path(xml_path)
        self.zip_path = Path(zip_path) if zip_path else self.xml_path.with_suffix('')
        raw = self.xml_path.read_bytes()
        self.xml_sha256 = sha256(raw)
        self.zip_sha256 = sha256(self.zip_path.read_bytes())
        root = ET.fromstring(raw)
        if root.tag != 'rdc' or root.findtext('header/driver') != 'D3D11':
            raise SkinPathError('Official D3D11 RenderDoc zip.xml export required')
        self.chunks = []
        for node in root.find('chunks'):
            refs = []
            fields = _decode(node, '', refs)
            self.chunks.append(Chunk(int(node.attrib['chunkIndex']), node.attrib['name'], fields, refs))
        self.by_index = {c.index: c for c in self.chunks}
        if len(self.by_index) != len(self.chunks):
            raise SkinPathError('Duplicate chunk index')
        self._zip = zipfile.ZipFile(self.zip_path)
        self.names = {}
        self.buffers = {}
        self.shaders = {}
        self.layouts = {}
        self.uavs = {}
        for c in self.chunks:
            f = c.fields
            if c.name == 'ID3D11Resource::SetDebugName':
                self.names.setdefault(f['pResource'], []).append(f['Name'])
            elif c.name == 'ID3D11Device::CreateBuffer':
                self.buffers[f['pBuffer']] = {'chunk': c.index, 'desc': f['pDesc'],
                                              'initial': f.get('InitialData')}
            elif c.name in ('ID3D11Device::CreateVertexShader', 'ID3D11Device::CreatePixelShader',
                            'ID3D11Device::CreateComputeShader', 'ID3D11Device::CreateGeometryShader',
                            'ID3D11Device::CreateHullShader', 'ID3D11Device::CreateDomainShader'):
                self.shaders[f['pShader']] = {'chunk': c.index, 'stage': c.name.split('Create')[1],
                                              'bytecode': f['pShaderBytecode']}
            elif c.name == 'ID3D11Device::CreateGeometryShaderWithStreamOutput':
                raise SkinPathError('Stream-output shader present; GPU-written geometry path must be implemented')
            elif c.name == 'ID3D11Device::CreateInputLayout':
                self.layouts[f['pInputLayout']] = f['pInputElementDescs']
            elif c.name == 'ID3D11Device::CreateUnorderedAccessView':
                self.uavs[f['pView']] = f
        init = [c for c in self.chunks if c.name == 'Internal::DeviceInitialisation']
        if len(init) != 1:
            raise SkinPathError('Exactly one DeviceInitialisation chunk required')
        self.immediate_context = init[0].fields['ImmediateContext']
        begin = [c for c in self.chunks if c.name == 'Internal::Beginning of Capture']
        if len(begin) != 1:
            raise SkinPathError('Exactly one Beginning of Capture chunk required')
        self.begin = begin[0]

    def close(self):
        self._zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def blob(self, ref: BufferRef) -> bytes:
        data = self._zip.read('%06d' % ref.blob)
        if len(data) != ref.length:
            raise SkinPathError('ZIP member length disagrees with XML byteLength')
        return data

    def name(self, rid):
        return (self.names.get(rid) or [''])[0]

    def shader_bytes(self, sid):
        return self.blob(self.shaders[sid]['bytecode'])


# --------------------------------------------------------------------------
# Command simulation: buffer write provenance and draw bindings
# --------------------------------------------------------------------------

def simulate(index: CaptureIndex):
    """Ordered immediate-context replay of binding state and buffer writes.

    Returns (writes, draws). ``writes`` maps a write id to its chunk/blob range;
    each draw lists the write ids current for every bound VB/IB/CB.
    """
    state = index.begin.fields['state']
    ia = state['IA']
    vbs, strides, offsets = list(ia['VBs']), list(ia['Strides']), list(ia['Offsets'])
    ib, ib_fmt, ib_off, layout = ia['IndexBuffer'], ia['IndexFormat'], ia['IndexOffset'], ia['Layout']
    shader = {s: state[s]['Object'] for s in ('VS', 'PS', 'GS', 'HS', 'DS', 'CS')}
    cbs = {s: list(state[s]['ConstantBuffers']) for s in ('VS', 'PS', 'GS', 'HS', 'DS', 'CS')}
    cb_off = {s: list(state[s]['CBOffsets']) for s in cbs}
    if any(any(state[s]['CBOffsets']) for s in cbs):
        raise SkinPathError('D3D11.1 partial constant-buffer offsets at capture start require implementation')
    if any(state['SO']['Buffers']):
        raise SkinPathError('Stream-out targets bound at capture start')
    for view in index.uavs.values():
        if view['pDesc']['ViewDimension'] == UAV_DIMENSION_BUFFER:
            raise SkinPathError('Buffer UAV exists; GPU-written buffer path must be implemented')
    writes = {}
    current = {}
    for rid, info in index.buffers.items():
        if info['initial'] is not None:
            wid = f'create:{info["chunk"]}'
            writes[wid] = {'id': wid, 'kind': 'capture_start_contents', 'resource_id': rid,
                           'chunk_index': info['chunk'], 'chunk_name': 'ID3D11Device::CreateBuffer',
                           'buffer': info['initial'], 'start': 0, 'end': info['desc']['ByteWidth']}
            current[rid] = wid
    draws = []
    open_maps = {}
    for c in index.chunks:
        if c.index <= index.begin.index or not c.name.startswith('ID3D11DeviceContext::'):
            continue
        f, m = c.fields, c.method
        if f.get('Context') != index.immediate_context:
            raise SkinPathError('Deferred/foreign context command requires implementation')
        if m == 'IASetVertexBuffers':
            s = f['StartSlot']
            for k, v in enumerate(f['ppVertexBuffers']):
                vbs[s + k], strides[s + k], offsets[s + k] = v, f['pStrides'][k], f['pOffsets'][k]
        elif m == 'IASetIndexBuffer':
            ib, ib_fmt, ib_off = f['pIndexBuffer'], f['Format'], f['Offset']
        elif m == 'IASetInputLayout':
            layout = f['pInputLayout']
        elif m[2:] == 'SetShader' and m[:2] in shader:
            shader[m[:2]] = f['pShader']
        elif m[2:] == 'SetConstantBuffers' and m[:2] in cbs:
            s = f['StartSlot']
            for k, v in enumerate(f['ppConstantBuffers']):
                cbs[m[:2]][s + k] = v
                cb_off[m[:2]][s + k] = 0
        elif m[2:] == 'SetConstantBuffers1':
            raise SkinPathError('D3D11.1 constant-buffer ranges require implementation')
        elif m == 'Map':
            key = (f['pResource'], f['Subresource'])
            if key in open_maps:
                raise SkinPathError('Resource mapped twice without Unmap')
            open_maps[key] = c.index
        elif m == 'Unmap':
            rid = f['pResource']
            open_maps.pop((rid, f['Subresource']), None)
            if rid in index.buffers:
                start = f['Byte offset to start of written data']
                end = f['Byte offset to end of written data']
                ref = f.get('MapWrittenData')
                wid = f'unmap:{c.index}'
                writes[wid] = {'id': wid, 'kind': 'unmap', 'map_type': f['MapType'], 'resource_id': rid,
                               'chunk_index': c.index, 'chunk_name': c.name, 'buffer': ref,
                               'start': start, 'end': end,
                               'previous': current.get(rid) if f['MapType'] != MAP_WRITE_DISCARD else None}
                current[rid] = wid
        elif m in BUFFER_WRITERS:
            dst = f.get('pDstResource', f.get('pResource'))
            if dst in index.buffers:
                wid = f'{m}:{c.index}'
                writes[wid] = {'id': wid, 'kind': m, 'resource_id': dst, 'chunk_index': c.index,
                               'chunk_name': c.name, 'buffer': None, 'unsupported': True}
                current[dst] = wid
        elif m in ('Draw', 'DrawIndexed', 'DrawInstanced', 'DrawIndexedInstanced', 'DrawAuto',
                   'DrawIndexedInstancedIndirect', 'DrawInstancedIndirect'):
            bound = {}
            for slot, v in enumerate(vbs):
                if v:
                    bound[('VB', slot)] = (v, current.get(v))
            if ib:
                bound[('IB', 0)] = (ib, current.get(ib))
            for stage in ('VS', 'PS', 'GS', 'HS', 'DS'):
                if shader[stage] or stage in ('VS', 'PS'):
                    for slot, v in enumerate(cbs[stage]):
                        if v:
                            bound[(stage + 'CB', slot)] = (v, current.get(v))
            draws.append({'chunk_index': c.index, 'method': m,
                          'args': {k: v for k, v in f.items() if k not in ('Context', 'DebugMessages')},
                          'vbs': list(vbs), 'strides': list(strides), 'offsets': list(offsets),
                          'ib': ib, 'ib_format': ib_fmt, 'ib_offset': ib_off, 'layout': layout,
                          'shaders': dict(shader), 'bound': bound})
    return writes, draws


# --------------------------------------------------------------------------
# Vertex-shader matrix usage (stripped DXBC; disassembly via fxc /dumpbin)
# --------------------------------------------------------------------------

def disassemble(dxbc: bytes, fxc: Path, work: Path, tag: str) -> str:
    work.mkdir(parents=True, exist_ok=True)
    src = work / f'{tag}.dxbc'
    out = work / f'{tag}.asm'
    src.write_bytes(dxbc)
    result = subprocess.run([str(fxc), '/nologo', '/dumpbin', '/Fc', str(out), str(src)],
                            capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise SkinPathError('fxc disassembly failed: ' + result.stderr[:400])
    return out.read_text(errors='replace')


_SIG = re.compile(r'^//\s+(\w+)\s+(\d+)\s+(\w+)\s+(\d+)\s+(\w+)\s+(\w+)')
_MAT = re.compile(r'^(mul|mad|add)\s+(r\d+)\.xyzw,\s*(.*)$')


def input_registers(asm: str):
    """Input signature register -> (semantic, index)."""
    regs = {}
    section = None
    for line in asm.splitlines():
        if line.startswith('// Input signature'):
            section = 'in'
        elif line.startswith('// Output signature'):
            section = None
        elif section == 'in':
            m = _SIG.match(line)
            if m:
                regs[int(m.group(4))] = (m.group(1), int(m.group(2)))
    return regs


def object_matrix_uses(asm: str):
    """Find 4-column CB matrix products applied to vertex inputs.

    Returns rows {'cb_slot','first_register','operand','input'} where the
    operand is a vertex input register or a temp produced by mov/movc from
    inputs. Matrices applied to other CB products (view/projection) are not
    object matrices and are excluded.
    """
    regs = input_registers(asm)
    lines = [l.strip() for l in asm.splitlines() if l.strip() and not l.startswith('//')]
    origins = {}   # temp register -> set of input semantics feeding it
    results = []
    i = 0
    while i < len(lines):
        line = lines[i]
        mv = re.match(r'^(mov|movc)\s+(r\d+)\.(\w+),\s*(.*)$', line)
        if mv:
            ins = set()
            for vr in re.findall(r'\bv(\d+)\.', mv.group(4)):
                ins.add(regs.get(int(vr), ('?', -1)))
            reg, mask, sources = mv.group(2), mv.group(3), mv.group(4)
            if ins:
                origins.setdefault(reg, set()).update(ins)
            elif mask == 'xyzw' and not re.search(rf'\b{reg}\.', sources):
                origins.pop(reg, None)
            # Partial-component constants (e.g. w=1) keep the input origin.
        m = re.match(r'^mul\s+(r\d+)\.xyzw,\s*(\w+)\.yyyy,\s*cb(\d+)\[(\d+)\]\.xyzw$', line)
        if m and i + 3 < len(lines):
            dst, operand, slot, row = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            nxt = lines[i + 1:i + 4]
            p0 = re.match(rf'^mad\s+{dst}\.xyzw,\s*cb{slot}\[{row - 1}\]\.xyzw,\s*{operand}\.xxxx,\s*{dst}\.xyzw$', nxt[0])
            p2 = re.match(rf'^mad\s+{dst}\.xyzw,\s*cb{slot}\[{row + 1}\]\.xyzw,\s*{operand}\.zzzz,\s*{dst}\.xyzw$', nxt[1])
            p3 = (re.match(rf'^mad\s+r\d+\.xyzw,\s*cb{slot}\[{row + 2}\]\.xyzw,\s*{operand}\.wwww,\s*{dst}\.xyzw$', nxt[2]) or
                  re.match(rf'^add\s+r\d+\.xyzw,\s*{dst}\.xyzw,\s*cb{slot}\[{row + 2}\]\.xyzw$', nxt[2]))
            if p0 and p2 and p3:
                if operand.startswith('v'):
                    feed = {regs.get(int(operand[1:]), ('?', -1))}
                else:
                    feed = origins.get(operand, set())
                if feed:
                    results.append({'cb_slot': slot, 'first_register': row - 1, 'operand': operand,
                                    'inputs': sorted(f'{s}{n}' for s, n in feed)})
        d = re.match(r'^(\w+)\s+(r\d+)\.', line)
        if d and not mv and d.group(2) in origins and not line.startswith('mul ' + d.group(2) + '.xyzw, ' + d.group(2)):
            # A temp overwritten by arithmetic no longer carries a raw input.
            if not re.search(r'\bv\d+\.', line):
                origins.pop(d.group(2), None)
        i += 1
    normal_rows = []
    nreg = [r for r, (s, n) in regs.items() if s == 'NORMAL']
    for line in lines:
        for r in nreg:
            m = re.match(rf'^dp3\s+r\d+\.\w,\s*v{r}\.xyzx,\s*cb(\d+)\[(\d+)\]\.xyzx$', line)
            if m:
                normal_rows.append((int(m.group(1)), int(m.group(2))))
    return {'inputs': {str(k): f'{s}{n}' for k, (s, n) in regs.items()},
            'object_matrices': results, 'normal_rows': sorted(set(normal_rows))}


# --------------------------------------------------------------------------
# Surface binding
# --------------------------------------------------------------------------

def _triangles(surface):
    tri = np.asarray(surface['source']['triangles'], np.int64).reshape(-1)
    return tri


def bind_surfaces(index: CaptureIndex, draws, surfaces):
    """Bind each source surface to its real GPU resources by exact content.

    ``surfaces`` rows: ``mesh_name``, ``renderer_path``, ``vertex_count``,
    ``active_in_hierarchy``, ``source`` (triangles, uv, uv2, bone_weights,
    bone_indices). Debug names only nominate candidates; IB indices and UV
    streams must equal the installed source arrays bit-exactly (float32).
    """
    groups = {}
    for d in draws:
        ib = d['ib']
        sk = d['vbs'][0]
        if not ib or not sk or sk not in index.buffers:
            continue
        desc = index.buffers[sk]['desc']
        if desc['Usage'] != D3D11_USAGE_DYNAMIC or desc['BindFlags'] != D3D11_BIND_VERTEX_BUFFER:
            continue
        if d['strides'][0] != SKIN_STRIDE or not index.name(ib).endswith('(Clone)'):
            continue
        key = (sk, d['vbs'][1], d['vbs'][2], ib)
        groups.setdefault(key, []).append(d)
    bindings = []
    used_keys = set()
    for s_index, surface in enumerate(surfaces):
        name = surface['mesh_name']
        candidates = []
        for key, ds in groups.items():
            sk, s1, s2, ib = key
            if index.name(ib) != name + '(Clone)' or index.name(s1) != name:
                continue
            if index.buffers[sk]['desc']['ByteWidth'] != SKIN_STRIDE * surface['vertex_count']:
                continue
            evidence = _static_evidence(index, ds[0], surface)
            if evidence['index_buffer_equal'] and evidence['uv0_equal']:
                candidates.append((key, ds, evidence))
        row = {'surface_index': s_index, 'mesh_name': name, 'renderer_path': surface['renderer_path'],
               'active_in_hierarchy': surface.get('active_in_hierarchy')}
        if not candidates:
            if surface.get('active_in_hierarchy') is False:
                row.update(status='inactive_no_gpu_upload',
                           reason='renderer inactive in hierarchy; no draw binds a resource set with this mesh content')
            else:
                row.update(status='unbound', reason='no exact-content GPU resource set found')
        else:
            row['candidates'] = [{'skinned_vb': k[0], 'static_streams': [k[1], k[2]], 'index_buffer': k[3]}
                                 for k, _, _ in candidates]
            row['candidate_keys'] = [k for k, _, _ in candidates]
            row['evidence'] = [e for _, _, e in candidates]
        bindings.append(row)
    # Duplicate mesh content (three identical o_tang meshes): resolve only by
    # recorded renderer activity; ambiguity among active renderers is an error.
    by_key = {}
    for row in bindings:
        for key in row.get('candidate_keys', []):
            by_key.setdefault(key, []).append(row)
    for key, rows in by_key.items():
        active = [r for r in rows if r.get('active_in_hierarchy') is not False]
        if len(active) != 1:
            raise SkinPathError(f'GPU resource set {key} matches {len(active)} active source surfaces')
        for r in rows:
            if r is not active[0]:
                r['candidate_keys'].remove(key)
    for row in bindings:
        keys = row.pop('candidate_keys', None)
        if keys is None:
            continue
        if len(keys) == 1:
            key = keys[0]
            if key in used_keys:
                raise SkinPathError('One GPU resource set bound to two surfaces')
            used_keys.add(key)
            ev = row['evidence'][[c['skinned_vb'] for c in row['candidates']].index(key[0])]
            row.update(status='bound', skinned_vb=key[0], static_streams=[key[1], key[2]], index_buffer=key[3],
                       skinned_vb_name=index.name(key[0]), static_evidence=ev,
                       draw_chunks=[d['chunk_index'] for d in groups[key]])
        elif not keys and row.get('active_in_hierarchy') is False:
            row.update(status='inactive_no_gpu_upload',
                       reason='identical mesh content is drawn only through the active renderer; this renderer is inactive')
        else:
            raise SkinPathError('Surface binding ambiguous: ' + row['mesh_name'])
        row.pop('candidates', None)
        row.pop('evidence', None)
    return bindings


def _static_evidence(index, draw, surface):
    ib = draw['ib']
    ibdata = index.blob(index.buffers[ib]['initial'])
    if draw['ib_format'] == DXGI_R16_UINT:
        indices = np.frombuffer(ibdata, np.uint16)
    elif draw['ib_format'] == DXGI_R32_UINT:
        indices = np.frombuffer(ibdata, np.uint32)
    else:
        raise SkinPathError('Unknown index format')
    tri = _triangles(surface)
    ev = {'index_count': int(len(indices)), 'source_index_count': int(len(tri)),
          'index_buffer_equal': bool(len(indices) == len(tri) and np.array_equal(indices.astype(np.int64), tri))}
    layout = index.layouts[draw['layout']]
    stream1 = index.blob(index.buffers[draw['vbs'][1]]['initial'])
    stride1 = draw['strides'][1]
    n = surface['vertex_count']
    if len(stream1) != stride1 * n:
        raise SkinPathError('Static stream byte width disagrees with vertex count')
    s1 = np.frombuffer(stream1, np.uint8).reshape(n, stride1)
    ev['uv0_equal'] = False
    for e in layout:
        if e['InputSlot'] == 1 and e['SemanticName'] == 'TEXCOORD' and e['Format'] == 16:
            data = s1[:, e['AlignedByteOffset']:e['AlignedByteOffset'] + 8].copy().view(np.float32).reshape(n, 2)
            key = {0: 'uv', 1: 'uv2'}.get(e['SemanticIndex'])
            if key and key in surface['source'] and len(surface['source'][key]):
                src = np.asarray(surface['source'][key], np.float32).reshape(n, -1)[:, :2]
                ev[f'uv{e["SemanticIndex"]}_equal'] = bool(np.array_equal(data, src))
    stream2 = index.blob(index.buffers[draw['vbs'][2]]['initial'])
    stride2 = draw['strides'][2]
    ev['skin_stream_stride'] = stride2
    if stride2 == 32 and len(stream2) == 32 * n:
        s2 = np.frombuffer(stream2, np.uint8).reshape(n, 32)
        weights = s2[:, :16].copy().view(np.float32)
        bones = s2[:, 16:].copy().view(np.int32)
        sw = np.asarray(surface['source']['bone_weights'], np.float32)
        sb = np.asarray(surface['source']['bone_indices'], np.int64)
        ev['skin_stream_weights_equal'] = bool(np.array_equal(weights, sw))
        used = sw > 0
        ev['skin_stream_indices_equal_where_weighted'] = bool(np.array_equal(bones[used], sb[used]))
    ev['skin_stream_referenced_by_input_layout'] = any(e['InputSlot'] == 2 for e in layout)
    return ev


# --------------------------------------------------------------------------
# Upload points per surface
# --------------------------------------------------------------------------

def upload_points(index, writes, draws, bindings, vs_usage, target_chunk=None):
    """Every write consumed by a bound head draw, with semantic roles.

    Roles:
      skinned_vertices_current  - slot0 skinned VB write used by a draw
      skinned_vertices_previous - slot1 TEXCOORD4 previous-position VB write
      per_draw_object_matrices  - CB write providing unity_ObjectToWorld (+W2O)
      motion_vector_previous_matrix - CB write providing _PreviousM (TEXCOORD4/POSITION path)
    Consumers outside the bound head draws are recorded; a write shared with a
    non-head draw cannot be patched by in-place byte replacement.
    """
    head_draws = {}
    for b in bindings:
        if b.get('status') != 'bound':
            continue
        for d in draws:
            if d['vbs'][0] == b['skinned_vb'] and d['ib'] == b['index_buffer']:
                head_draws[d['chunk_index']] = b['surface_index']
    consumers = {}
    for d in draws:
        for key, (rid, wid) in d['bound'].items():
            if wid:
                consumers.setdefault(wid, set()).add((d['chunk_index'], key))
    points = {}

    def add(wid, role, surface_index, draw, slot_key, extra=None):
        if wid is None:
            raise SkinPathError(f'Draw {draw["chunk_index"]} reads a buffer without captured bytes')
        w = writes[wid]
        if w.get('unsupported'):
            raise SkinPathError(f'{w["kind"]} writes a head input at chunk {w["chunk_index"]}')
        p = points.setdefault((wid, role), {'write_id': wid, 'role': role, 'resource_id': w['resource_id'],
                                            'resource_name': index.name(w['resource_id']),
                                            'chunk_index': w['chunk_index'], 'chunk_name': w['chunk_name'],
                                            'write_kind': w['kind'], 'byte_range': [w['start'], w['end']],
                                            'buffer_ordinal': w['buffer'].ordinal if w['buffer'] else None,
                                            'buffer_index': w['buffer'].blob if w['buffer'] else None,
                                            'surfaces': set(), 'head_draws': set(), 'extra': {}})
        p['surfaces'].add(surface_index)
        p['head_draws'].add(draw['chunk_index'])
        if extra:
            for k, v in extra.items():
                p['extra'].setdefault(k, set()).add(v)

    for d in draws:
        si = head_draws.get(d['chunk_index'])
        if si is None:
            # Motion-vector draws bind the skinned VB with a different IB? They
            # reuse the mesh IB; handled above. Nothing else is a head draw.
            continue
        add(d['bound'][('VB', 0)][1], 'skinned_vertices_current', si, d, ('VB', 0))
        layout = index.layouts[d['layout']]
        tc4 = [e for e in layout if e['SemanticName'] == 'TEXCOORD' and e['SemanticIndex'] == 4 and e['InputSlot'] == 1]
        if tc4:
            # The skinned motion-vector pass binds the previous skin buffer as
            # TEXCOORD4 R32G32B32_FLOAT at offset 0 of a DYNAMIC stride-40 slot-1
            # VB. A mesh with a fifth UV set (o_top_camisole2_a) also has a
            # slot-1 TEXCOORD4, but as R32G32_FLOAT inside its DEFAULT static
            # stream (stride 44): a static input, not previous skin.
            prev_rid = d['vbs'][1]
            desc = index.buffers[prev_rid]['desc'] if prev_rid in index.buffers else None
            is_skin = (tc4[0]['Format'] == DXGI_R32G32B32_FLOAT and tc4[0]['AlignedByteOffset'] == 0
                       and desc is not None and desc['Usage'] == D3D11_USAGE_DYNAMIC and d['strides'][1] == SKIN_STRIDE)
            is_static_uv = (tc4[0]['Format'] == DXGI_R32G32_FLOAT and desc is not None
                            and desc['Usage'] != D3D11_USAGE_DYNAMIC)
            if is_skin:
                add(d['bound'][('VB', 1)][1], 'skinned_vertices_previous', si, d, ('VB', 1))
            elif not is_static_uv:
                raise SkinPathError(f'Draw {d["chunk_index"]}: slot-1 TEXCOORD4 is neither previous skin nor a static UV')
        usage = vs_usage[d['shaders']['VS']]
        for mat in usage['object_matrices']:
            rid, wid = d['bound'].get(('VSCB', mat['cb_slot']), (None, None))
            if rid is None:
                raise SkinPathError('Object matrix CB slot unbound at head draw')
            role = ('motion_vector_previous_matrix' if 'TEXCOORD4' in mat['inputs']
                    else 'per_draw_object_matrices')
            add(wid, role, si, d, ('VSCB', mat['cb_slot']),
                {'matrix_byte_offset': mat['first_register'] * 16, 'vs': d['shaders']['VS']})
        for slot, row in usage['normal_rows']:
            rid, wid = d['bound'].get(('VSCB', slot), (None, None))
            add(wid, 'per_draw_object_matrices', si, d, ('VSCB', slot),
                {'world_to_object_row_offset': row * 16})
    result = []
    for p in points.values():
        all_consumers = consumers.get(p['write_id'], set())
        foreign = sorted({c for c, _ in all_consumers if c not in head_draws})
        p['non_head_consumer_draws'] = foreign
        p['in_place_patchable'] = not foreign
        p['surfaces'] = sorted(p['surfaces'])
        p['head_draws'] = sorted(p['head_draws'])
        p['extra'] = {k: sorted(v) for k, v in p['extra'].items()}
        if target_chunk is not None:
            p['head_draws_before_target'] = [c for c in p['head_draws'] if c < target_chunk]
        result.append(p)
    result.sort(key=lambda r: (r['chunk_index'], r['role']))
    return result, head_draws


# --------------------------------------------------------------------------
# Source-derived payload generation
# --------------------------------------------------------------------------

def unity_rotation_matrix(q_xyzw):
    """UnityCsReference Matrix4x4.Rotate float32 expression order."""
    x, y, z, w = (np.float32(v) for v in q_xyzw)
    two = np.float32(2.0)
    one = np.float32(1.0)
    X, Y, Z = x * two, y * two, z * two
    xx, yy, zz = x * X, y * Y, z * Z
    xy, xz, yz = x * Y, x * Z, y * Z
    wx, wy, wz = w * X, w * Y, w * Z
    m = np.zeros((3, 3), np.float32)
    m[0, 0] = one - (yy + zz); m[1, 0] = xy + wz; m[2, 0] = xz - wy
    m[0, 1] = xy - wz; m[1, 1] = one - (xx + zz); m[2, 1] = yz + wx
    m[0, 2] = xz + wy; m[1, 2] = yz - wx; m[2, 2] = one - (xx + yy)
    return m


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by + ay * bw + az * bx - ax * bz,
                     aw * bz + az * bw + ax * by - ay * bx,
                     aw * bw - ax * bx - ay * by - az * bz])


def world_rotation(transforms, local, tid):
    """Unity Transform.rotation: product of local rotations up the hierarchy."""
    chain = []
    while tid is not None:
        chain.append(tid)
        tid = transforms[tid]['parent_id']
    q = np.array([0., 0., 0., 1.])
    for t in reversed(chain):
        q = quat_mul(q, np.asarray(local[t]['local_rotation_xyzw'], float))
    return q


def root_frame(world_matrix, world_rotation_xyzw):
    """Scale-free root TRS used by Unity for skinned renderers.

    Position is the root bone's world translation; rotation is the Transform
    rotation (product of local rotations), not a polar decomposition.
    """
    M = np.asarray(world_matrix, float).reshape(4, 4)
    R = unity_rotation_matrix(world_rotation_xyzw).astype(np.float64)
    t = M[:3, 3].astype(np.float32).astype(np.float64)
    O = np.eye(4)
    O[:3, :3] = R
    O[:3, 3] = t
    return O


def skinned_vb_bytes(world_vertices, world_normals, world_tangents, tangent_handedness, root):
    """Root-relative 40-byte vertices: p, n, t in root frame without scale."""
    root = np.asarray(root, float)
    R = root[:3, :3]
    t = root[:3, 3]
    inv_r = R.T  # exact orthonormal inverse of the float32-rounded rotation within float64
    p = (np.asarray(world_vertices, float) - t) @ inv_r.T
    n = np.asarray(world_normals, float) @ inv_r.T
    tan = np.asarray(world_tangents, float) @ inv_r.T
    w = np.asarray(tangent_handedness, float)
    if len({len(p), len(n), len(tan), len(w)}) != 1:
        raise SkinPathError('Attribute counts differ')
    out = np.concatenate([p, n, tan, w[:, None]], axis=1).astype(np.float32)
    return out.tobytes()


def matrix_bytes(M):
    """Unity cbuffer float4x4: column-major, register k = column k."""
    return np.asarray(M, float).T.astype(np.float32).tobytes()


def _matrix_from(data: bytes, offset: int):
    return np.frombuffer(data[offset:offset + 64], np.float32).reshape(4, 4).astype(float).T


def label_poses(index, writes, points):
    """Assign capture-derived root pose labels to every upload point.

    Each distinct recorded root matrix (UnityPerDraw O2W or motion-vector
    _PreviousM) becomes ``pose<k>`` in first-occurrence order. Skinned VB writes
    inherit the pose of the O2W (current) or _PreviousM (previous) that their
    consuming draws read; W2O entries are classified against the inverses of
    the recorded poses. Nothing is adjusted: labels only route source states.
    """
    poses = {}

    def label(matrix_bytes_):
        if matrix_bytes_ not in poses:
            poses[matrix_bytes_] = f'pose{len(poses)}'
        return poses[matrix_bytes_]

    by_write = {}
    draw_pose = {}
    mv_pose = {}
    for p in points:
        if p['role'] not in ('per_draw_object_matrices', 'motion_vector_previous_matrix'):
            continue
        data = index.blob(writes[p['write_id']]['buffer'])
        if p['role'] == 'per_draw_object_matrices':
            if p['extra'].get('matrix_byte_offset', [0]) != [0]:
                raise SkinPathError('UnityPerDraw object matrix not at offset 0')
            p['object_pose'] = label(data[0:64])
            for c in p['head_draws']:
                draw_pose[c] = p['object_pose']
        else:
            offs = p['extra'].get('matrix_byte_offset', [])
            if len(offs) != 1:
                raise SkinPathError('Motion-vector previous matrix offset ambiguous')
            p['previous_pose'] = label(data[offs[0]:offs[0] + 64])
            for c in p['head_draws']:
                mv_pose[c] = p['previous_pose']
        by_write[p['write_id']] = data
    inverses = {lab: np.linalg.inv(np.frombuffer(m, np.float32).reshape(4, 4).astype(float).T)
                for m, lab in poses.items()}
    for p in points:
        if p['role'] == 'per_draw_object_matrices':
            data = by_write[p['write_id']]
            w2o = _matrix_from(data, 64)
            if np.array_equal(w2o, np.eye(4)):
                p['world_to_object'] = 'identity'
            else:
                errors = {lab: float(np.abs(w2o - inv).max()) for lab, inv in inverses.items()}
                best = sorted(errors.items(), key=lambda kv: kv[1])
                margin = best[1][1] / max(best[0][1], 1e-30) if len(best) > 1 else float('inf')
                p['world_to_object_inverse_errors'] = errors
                # A float32 inverse differs from the float64 inverse by rounding
                # only (~1e-6 here); a different pose differs by >1e-3.
                if best[0][1] < 1e-4 and (len(best) == 1 or best[1][1] > 1e-3):
                    p['world_to_object'] = 'inverse:' + best[0][0]
                else:
                    p['world_to_object'] = 'unclassified'
                p['world_to_object_margin'] = margin
        elif p['role'] in ('skinned_vertices_current', 'skinned_vertices_previous'):
            table = draw_pose if p['role'] == 'skinned_vertices_current' else mv_pose
            labs = {table.get(c) for c in p['head_draws']}
            labs.discard(None)
            if len(labs) != 1:
                p['skin_pose'] = 'ambiguous:' + ','.join(sorted(labs))
            else:
                p['skin_pose'] = labs.pop()
    return {lab: np.frombuffer(m, np.float32).reshape(4, 4).T.astype(float).tolist() for m, lab in poses.items()}


@dataclass
class SurfaceState:
    """Source-derived skin state of one renderer for one root pose."""
    surface_index: int
    world_vertices: np.ndarray
    world_normals: np.ndarray
    world_tangents: np.ndarray
    tangent_handedness: np.ndarray
    root_world_matrix: np.ndarray
    root_world_rotation_xyzw: np.ndarray
    # Unity-exact payloads from src.render.unity_skin_emu.skin_from_locals (original
    # UnityPlayer instructions); when present they replace the float64 path.
    vertex_bytes: bytes | None = None
    object_to_world: bytes | None = None
    world_to_object: bytes | None = None


def generate_patches(index, writes, points, states, out_dir: Path):
    """Produce byte payloads and a mutate_capture-compatible plan.

    ``states`` maps a capture pose label (from :func:`label_poses`) to
    ``{surface_index: SurfaceState}``. A pose state must be the source state of
    the frame that produced that recorded pose. Missing pose/surface states
    leave the original bytes and are listed as gaps; they are never filled
    with another pose's state.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    patches, gaps = [], []

    def emit(point, offset, payload, role_note, witness):
        w = writes[point['write_id']]
        original = index.blob(w['buffer'])
        if offset + len(payload) > len(original):
            raise SkinPathError('Payload exceeds original write')
        name = f'patch_{point["chunk_index"]}_{w["buffer"].ordinal}_{offset}.bin'
        (out_dir / name).write_bytes(payload)
        patches.append({'chunk_index': point['chunk_index'], 'chunk_name': point['chunk_name'],
                        'buffer_ordinal': w['buffer'].ordinal, 'expected_buffer_index': w['buffer'].blob,
                        'expected_blob_sha256': sha256(original), 'byte_offset': offset,
                        'replacement_file': name, 'replacement_sha256': sha256(payload),
                        'expected_range_sha256': sha256(original[offset:offset + len(payload)]),
                        'resource_id': point['resource_id'], 'semantic_role': role_note,
                        'source_witness': witness, 'head_draws': point['head_draws'],
                        'unchanged_original_range': payload == original[offset:offset + len(payload)]})

    def exact_for(pose, surfaces, field):
        frame = states.get(pose) or {}
        vals = [getattr(frame.get(si), field, None) if frame.get(si) is not None else None for si in surfaces]
        if any(v is None for v in vals):
            return None
        if any(v != vals[0] for v in vals[1:]):
            raise SkinPathError('Surfaces sharing one per-draw write disagree on ' + field)
        return vals[0]

    def roots_for(pose, surfaces):
        frame = states.get(pose) or {}
        mats = []
        for si in surfaces:
            st = frame.get(si)
            if st is None:
                return None
            if st.object_to_world is not None:
                mats.append(np.frombuffer(st.object_to_world, np.float32).reshape(4, 4).T.astype(float))
            else:
                mats.append(root_frame(st.root_world_matrix, st.root_world_rotation_xyzw))
        if any(not np.array_equal(m, mats[0]) for m in mats[1:]):
            raise SkinPathError('Surfaces sharing one per-draw write disagree on their root frame')
        return mats[0]

    for point in points:
        if not point['in_place_patchable']:
            gaps.append({'write': point['write_id'], 'role': point['role'],
                         'reason': 'write also consumed by non-head draws',
                         'non_head_draws': point['non_head_consumer_draws'][:16]})
            continue
        if point['role'] in ('skinned_vertices_current', 'skinned_vertices_previous'):
            pose = point.get('skin_pose', '')
            if len(point['surfaces']) != 1 or pose.startswith('ambiguous'):
                gaps.append({'write': point['write_id'], 'role': point['role'], 'reason': 'pose/surface ambiguous: ' + pose})
                continue
            si = point['surfaces'][0]
            st = (states.get(pose) or {}).get(si)
            if st is None:
                gaps.append({'write': point['write_id'], 'role': point['role'], 'pose': pose, 'surface': si,
                             'reason': 'source state for this recorded pose/surface not supplied'})
                continue
            if st.vertex_bytes is not None:
                payload = st.vertex_bytes
            else:
                root = root_frame(st.root_world_matrix, st.root_world_rotation_xyzw)
                payload = skinned_vb_bytes(st.world_vertices, st.world_normals, st.world_tangents,
                                           st.tangent_handedness, root)
            if len(payload) != point['byte_range'][1] - point['byte_range'][0] or point['byte_range'][0] != 0:
                raise SkinPathError('Generated VB size differs from captured write')
            emit(point, 0, payload, f'{point["role"]}:{pose}', f'source skin state {pose}, surface {si}')
        elif point['role'] == 'motion_vector_previous_matrix':
            pose = point['previous_pose']
            root = roots_for(pose, point['surfaces'])
            if root is None:
                gaps.append({'write': point['write_id'], 'role': point['role'], 'pose': pose,
                             'reason': 'source root for recorded previous pose not supplied'})
                continue
            exact = exact_for(pose, point['surfaces'], 'object_to_world')
            emit(point, point['extra']['matrix_byte_offset'][0], exact if exact is not None else matrix_bytes(root),
                 f'_PreviousM:{pose}', f'root TRS without scale of {pose}')
        elif point['role'] == 'per_draw_object_matrices':
            pose = point['object_pose']
            root = roots_for(pose, point['surfaces'])
            if root is None:
                gaps.append({'write': point['write_id'], 'role': point['role'], 'pose': pose,
                             'reason': 'source root for recorded object pose not supplied'})
                continue
            exact = exact_for(pose, point['surfaces'], 'object_to_world')
            emit(point, 0, exact if exact is not None else matrix_bytes(root), f'unity_ObjectToWorld:{pose}',
                 f'root TRS without scale of {pose}')
            cls = point['world_to_object']
            if cls == 'identity':
                continue
            if not cls.startswith('inverse:'):
                gaps.append({'write': point['write_id'], 'role': 'unity_WorldToObject', 'reason': cls})
                continue
            inv_pose = cls.split(':', 1)[1]
            inv_root = roots_for(inv_pose, point['surfaces'])
            if inv_root is None:
                gaps.append({'write': point['write_id'], 'role': 'unity_WorldToObject', 'pose': inv_pose,
                             'reason': 'recorded W2O is the inverse of a pose whose source root is not supplied'})
                continue
            exact = exact_for(inv_pose, point['surfaces'], 'world_to_object')
            emit(point, 64, exact if exact is not None else matrix_bytes(np.linalg.inv(inv_root)),
                 f'unity_WorldToObject:inverse:{inv_pose}',
                 f'inverse root TRS of {inv_pose} (recorded engine W2O source)')
        else:
            raise SkinPathError('Unknown upload role ' + point['role'])
    plan = {'schema_version': 1, 'source_xml_sha256': index.xml_sha256, 'source_zip_sha256': index.zip_sha256,
            'patches': [p for p in patches if not p['unchanged_original_range']],
            'unchanged_ranges_omitted': [p for p in patches if p['unchanged_original_range']],
            'gaps': gaps,
            'scope': ('Head skinning inputs only. Bounds/culling invariance, material $Globals, '
                      'shadow-caster/SSS/postprocess and temporal history dependencies are outside this plan.')}
    (out_dir / 'plan.json').write_text(json.dumps(plan, indent=1, default=_json), encoding='utf-8')
    return plan


def _json(value):
    if isinstance(value, (set, tuple)):
        return sorted(value) if isinstance(value, set) else list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    raise TypeError(type(value))


def surface_state_from_forward(result, surface_index, geometry):
    """Build a SurfaceState from ``CapturedHeadForward.evaluate`` output.

    Root position is the evaluated root bone world translation. Root rotation
    is Unity's Transform.rotation, recomputed as the product of the evaluated
    local rotations (``result['bone_local']``), because ABMX scale/rotation on
    head bones makes the world matrix non-orthogonal (polar decomposition is
    not Unity's rotation).
    """
    surface = result['surfaces'][surface_index]
    mesh = geometry['meshes'][surface_index]
    if 'world_normals' not in surface:
        raise SkinPathError('Loaded API normal/tangent frames required for GPU vertex payloads')
    if 'bone_local' not in result:
        raise SkinPathError('Evaluated local rotations required for the Unity root rotation')
    root_id = mesh['root_bone_transform_id']
    world = result['bone_world'].get(root_id)
    if world is None:
        raise SkinPathError('Root bone world matrix not evaluated')
    transforms = {t['id']: t for t in geometry['transforms']}
    q = world_rotation(transforms, result['bone_local'], root_id)
    return SurfaceState(surface_index, surface['world_vertices'], surface['world_normals'],
                        surface['world_tangents'], surface['tangent_handedness'], world, q)


def surface_state_from_export(geometry, surface_index):
    """Diagnostic state from a native live export (BakeMesh-frame bones).

    Uses the exported bone local-to-world matrices, Unity world rotation and
    loaded API blendshape frames of *that* export's frame. It is only
    the source state of the frame it was exported in.
    """
    from tools.unity_parity.geometry import skin_world
    from src.render.skin_attributes import deform_attributes
    transforms = {t['id']: t for t in geometry['transforms']}
    mesh = geometry['meshes'][surface_index]
    world, _ = skin_world(mesh, transforms, influences=4)
    attrs = deform_attributes(mesh, transforms, loaded_api_frames=True)
    root = transforms[mesh['root_bone_transform_id']]
    return SurfaceState(surface_index, world, attrs['world_normals'], attrs['world_tangents'],
                        attrs['tangent_handedness'], np.asarray(root['local_to_world'], float).reshape(4, 4),
                        np.asarray(root['world_rotation_xyzw'], float))
