"""Ordered D3D11 frame graph from an official RenderDoc XML+ZIP export.

CPU only: no game, RenderDoc DLL, GPU or capture is touched. Every action of
the recorded frame is interpreted from the actual serialized chunks; the
official driver action listing (``renderdoc_replay_msvc.exe inspect``) supplies
event IDs, and is checked against the chunk order instead of being guessed.

Mechanisms that are taken from the pinned RenderDoc 1.46 source rather than
assumed:

* Buffer capture-start contents. ``WrappedID3D11Device::Serialise_InitialState``
  (d3d11_initstate.cpp) serializes no separate chunk for ``Resource_Buffer``;
  it copies the staged capture-start bytes into the resource record data, so
  the ``CreateBuffer`` ``InitialData`` blob *is* the capture-start content.
* Texture capture-start contents come from ``Internal::Initial Contents``
  chunks when present, otherwise from the creation record (creation data and
  pre-capture updates); a texture without either is reported, not filled.
* Map/Unmap: written bytes are attached to Unmap (``Serialise_Unmap``);
  WRITE_DISCARD invalidates the whole buffer.

Used shader slots come from each executed shader's own DXBC RDEF/ISGN chunks.
Shader identities are bound by exact SHA-256 of the executed DXBC against
installed shader exports; every matching keyword variant is retained.
Pass labels are derived from recorded debug names with the rule recorded;
nothing is classified from rendered appearance.  Unknown commands, view
dimensions or overlapping states raise ``UnresolvedFrame``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

STAGES = ('VS', 'HS', 'DS', 'GS', 'PS', 'CS')
SCHEMA_VERSION = 1


class UnresolvedFrame(RuntimeError):
    pass


class Repeated(list):
    """A structured field name that occurs several times in one struct."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


# ----------------------------------------------------------------- XML decode
def decode_element(el):
    kind = el.tag
    text = (el.text or '').strip()
    if kind in ('uint', 'int', 'enum', 'ResourceId'):
        return int(text)
    if kind in ('float', 'double'):
        return float(text)
    if kind == 'bool':
        if text not in ('true', 'false', '1', '0'):
            raise UnresolvedFrame('Unknown serialized boolean ' + repr(text))
        return text in ('true', '1')
    if kind == 'string':
        return el.text or ''
    if kind == 'null':
        return None
    if kind == 'buffer':
        return {'buffer_index': int(text), 'byte_length': int(el.attrib['byteLength'])}
    if kind == 'array':
        return [decode_element(c) for c in el]
    if kind in ('chunk', 'struct'):
        result = {}
        for c in el:
            name = c.attrib.get('name')
            if name is None:
                raise UnresolvedFrame('Unnamed structured field in ' + el.attrib.get('name', kind))
            value = decode_element(c)
            if name in result:
                old = result[name]
                result[name] = Repeated([*old, value]) if isinstance(old, Repeated) else Repeated([old, value])
            else:
                result[name] = value
        return result
    raise UnresolvedFrame('Uninterpreted XML kind ' + kind)


def load_chunks(xml_path):
    chunks = []
    header = None
    for _, el in ET.iterparse(str(xml_path), events=('end',)):
        if el.tag == 'header':
            driver = el.find('driver')
            header = {'driver': driver.text if driver is not None else None}
        if el.tag != 'chunk':
            continue
        attrs = dict(el.attrib)
        chunks.append({'index': int(attrs['chunkIndex']), 'name': attrs['name'],
                       'fields': decode_element(el), 'attributes': attrs})
        el.clear()
    if header is None or header['driver'] != 'D3D11':
        raise UnresolvedFrame('Official D3D11 RenderDoc XML export required')
    return chunks


def load_driver_listing(path):
    """Official driver action listing: chunk index -> event id, actions by event."""
    chunk_event = {}
    actions = {}
    opened = None
    for line in Path(path).read_bytes().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['event'] == 'opened':
            opened = row
        elif row['event'] == 'action':
            actions[row['event_id']] = row
            for api in row['api_events']:
                if api['chunk_index'] in chunk_event and chunk_event[api['chunk_index']] != api['event_id']:
                    raise UnresolvedFrame('Driver listing assigns two events to one chunk')
                chunk_event[api['chunk_index']] = api['event_id']
    return opened, chunk_event, actions


# ------------------------------------------------------------ DXBC reflection
# Unity strips RDEF from its compiled programs, so used slots are taken from
# the mandatory SHDR/SHEX declaration tokens (d3d11TokenizedProgramFormat.hpp).
OP_CUSTOMDATA = 53
DECL_OPCODES = {88: ('srv', 'dcl_resource'), 89: ('cb', 'dcl_constantbuffer'), 90: ('sampler', 'dcl_sampler'),
                156: ('uav', 'dcl_uav_typed'), 157: ('uav', 'dcl_uav_raw'), 158: ('uav', 'dcl_uav_structured'),
                161: ('srv', 'dcl_resource_raw'), 162: ('srv', 'dcl_resource_structured')}
OPERAND_TYPES = {'srv': 7, 'sampler': 6, 'cb': 8, 'uav': 30}
RESOURCE_DIMENSIONS = {1: 'buffer', 2: 'texture1d', 3: 'texture2d', 4: 'texture2dms', 5: 'texture3d',
                       6: 'texturecube', 7: 'texture1darray', 8: 'texture2darray', 9: 'texture2dmsarray',
                       10: 'texturecubearray', 11: 'raw_buffer', 12: 'structured_buffer'}


def _cstr(data, offset):
    end = data.index(b'\0', offset)
    return data[offset:end].decode('ascii', 'replace')


def dxbc_chunks(data):
    if data[:4] != b'DXBC':
        raise UnresolvedFrame('Executed shader bytecode is not a DXBC container')
    total, count = struct.unpack_from('<II', data, 24)
    if total != len(data):
        raise UnresolvedFrame('DXBC container size disagrees with serialized bytecode length')
    result = {}
    for i in range(count):
        offset, = struct.unpack_from('<I', data, 32 + 4 * i)
        fourcc = data[offset:offset + 4].decode('ascii', 'replace')
        size, = struct.unpack_from('<I', data, offset + 4)
        if fourcc in result:
            raise UnresolvedFrame('Duplicate DXBC chunk ' + fourcc)
        result[fourcc] = data[offset + 8:offset + 8 + size]
    return result


def parse_declarations(code):
    """Resource/CB/sampler/UAV declarations of a SM4/SM5 token stream."""
    words = struct.unpack('<%dI' % (len(code) // 4), code[:len(code) // 4 * 4])
    version, length = words[0], words[1]
    if length > len(words):
        raise UnresolvedFrame('Shader token stream length exceeds chunk')
    major, minor = (version >> 4) & 0xf, version & 0xf
    if (major, minor) > (5, 0):
        raise UnresolvedFrame('SM5.1 register spaces require interpretation')
    result = {'program_type': version >> 16, 'shader_model': f'{major}.{minor}', 'cb': {}, 'srv': {}, 'sampler': {}, 'uav': {}}
    pos = 2
    while pos < length:
        token = words[pos]
        opcode = token & 0x7ff
        size = words[pos + 1] if opcode == OP_CUSTOMDATA else (token >> 24) & 0x7f
        if size == 0:
            raise UnresolvedFrame('Zero-length shader instruction')
        if opcode in DECL_OPCODES:
            kind, label = DECL_OPCODES[opcode]
            i = pos + 1
            if token >> 31:
                while words[i] >> 31:
                    i += 1
                i += 1
            operand = words[i]
            i += 1
            if operand >> 31:
                while words[i] >> 31:
                    i += 1
                i += 1
            if (operand >> 12) & 0xff != OPERAND_TYPES[kind]:
                raise UnresolvedFrame('Unexpected operand type in ' + label)
            dims = (operand >> 20) & 3
            if (operand >> 22) & 7 != 0:
                raise UnresolvedFrame('Non-immediate declaration index requires interpretation')
            slot = words[i]
            row = {'declaration': label}
            if kind == 'cb':
                if dims != 2:
                    raise UnresolvedFrame('Constant buffer declaration without size index')
                row['float4_count'] = words[i + 1]
            if opcode in (88, 156):
                row['dimension'] = RESOURCE_DIMENSIONS.get((token >> 11) & 0x1f, (token >> 11) & 0x1f)
            if slot in result[kind]:
                raise UnresolvedFrame('Shader declares one slot twice')
            result[kind][slot] = row
        pos += size
    return result


def parse_signature(chunk):
    count, _ = struct.unpack_from('<II', chunk, 0)
    rows = []
    for i in range(count):
        name_o, index, system_value, component, register = struct.unpack_from('<5I', chunk, 8 + 24 * i)
        mask, rw_mask = struct.unpack_from('<BB', chunk, 8 + 24 * i + 20)
        rows.append({'semantic': _cstr(chunk, name_o), 'index': index, 'system_value': system_value,
                     'register': register, 'mask': mask, 'used_mask': rw_mask})
    return rows


def reflect(data):
    chunks = dxbc_chunks(data)
    code = chunks.get('SHEX', chunks.get('SHDR'))
    if code is None:
        raise UnresolvedFrame('Executed shader has no SHDR/SHEX token stream')
    result = parse_declarations(code)
    result['container_chunks'] = sorted(chunks)
    if 'ISGN' in chunks:
        result['input_signature'] = parse_signature(chunks['ISGN'])
    return result


def asm_declarations(text):
    """Independent cross-check from an installed export's disassembly text."""
    import re
    result = {'cb': {}, 'srv': {}, 'sampler': {}, 'uav': {}}
    for line in text.splitlines():
        m = re.match(r'dcl_constantbuffer [Cc][Bb](\d+)\[(\d+)\]', line)
        if m:
            result['cb'][int(m.group(1))] = int(m.group(2))
            continue
        m = re.match(r'dcl_sampler s(\d+)', line)
        if m:
            result['sampler'][int(m.group(1))] = True
            continue
        m = re.match(r'dcl_resource\S*.* t(\d+)', line)
        if m:
            result['srv'][int(m.group(1))] = True
            continue
        m = re.match(r'dcl_uav\S*.* u(\d+)', line)
        if m:
            result['uav'][int(m.group(1))] = True
    return result


def export_names(program):
    """Slot names and Unity cbuffer layouts recorded by an installed shader export."""
    b = program['bindings']
    names = {i: n for n, i in program['name_indices']}
    cbs = {}
    for c in b['m_ConstantBuffers']:
        variables = [{'name': names[v['m_NameIndex']], 'offset': v['m_Index'],
                      'size': 4 * v['m_Dim'] * max(1, v['m_ArraySize'])} for v in c['m_VectorParams']]
        variables += [{'name': names[v['m_NameIndex']], 'offset': v['m_Index'],
                       'size': 16 * v['m_RowCount'] * max(1, v['m_ArraySize'])} for v in c['m_MatrixParams']]
        cbs[names[c['m_NameIndex']]] = {'size': c['m_Size'], 'variables': sorted(variables, key=lambda r: r['offset'])}
    return {'cb': {c['m_Index']: names[c['m_NameIndex']] for c in b['m_ConstantBufferBindings']},
            'srv': {**{t['m_Index']: names[t['m_NameIndex']] for t in b['m_TextureParams']},
                    **{t['m_Index']: names[t['m_NameIndex']] for t in b.get('m_BufferParams', [])}},
            'uav': {t['m_Index']: names[t['m_NameIndex']] for t in b.get('m_UAVParams', [])},
            'cb_layouts': cbs}


# ----------------------------------------------------------- shader exports
class ShaderLibrary:
    def __init__(self, manifests=()):
        self.by_hash = {}
        self.sources = []
        for path in map(Path, manifests):
            source = json.loads(path.read_text(encoding='utf-8'))
            self.sources.append({'manifest': str(path.resolve()), 'bundle_path': source.get('bundle_path'),
                                 'bundle_sha256': source.get('bundle_sha256')})
            for shader in source['shaders']:
                for p in shader['passes']:
                    for program in p['programs']:
                        data = (path.parent / program['dxbc']).read_bytes()
                        h = digest(data)
                        asm = path.parent / program['assembly'] if program.get('assembly') else None
                        self.by_hash.setdefault(h, []).append({
                            'bundle_sha256': source.get('bundle_sha256'), 'bundle_path': source.get('bundle_path'),
                            'shader': shader['name'], 'subshader': p['subshader'], 'pass': p['pass'],
                            'pass_name': p.get('name'), 'stage': program['stage'], 'blob_index': program['blob_index'],
                            'keywords': program['keywords'], 'dxbc': program['dxbc'],
                            '_names': export_names(program),
                            '_asm': str(asm) if asm and asm.exists() else None})

    def match(self, sha):
        return [{k: v for k, v in m.items() if not k.startswith('_')} for m in self.by_hash.get(sha, [])]

    def names(self, sha):
        """Slot names/layouts for one executed program; conflicting variants are reported."""
        rows = self.by_hash.get(sha, [])
        if not rows:
            return None, []
        first = json.dumps(rows[0]['_names'], sort_keys=True)
        conflicts = [r['dxbc'] for r in rows[1:] if json.dumps(r['_names'], sort_keys=True) != first]
        return rows[0]['_names'], conflicts

    def asm_check(self, sha, declarations):
        rows = self.by_hash.get(sha, [])
        if not rows or not rows[0]['_asm']:
            return None
        text = Path(rows[0]['_asm']).read_text(encoding='utf-8', errors='replace')
        expected = asm_declarations(text)
        parsed = {'cb': {k: v['float4_count'] for k, v in declarations['cb'].items()},
                  'srv': {k: True for k in declarations['srv']}, 'sampler': {k: True for k in declarations['sampler']},
                  'uav': {k: True for k in declarations['uav']}}
        return {'assembly': rows[0]['_asm'], 'agrees': expected == parsed,
                'assembly_declarations': expected if expected != parsed else None}


# --------------------------------------------------------------- blob access
class Blobs:
    def __init__(self, zip_path):
        self.zip = zipfile.ZipFile(zip_path) if zip_path else None
        self.cache = {}

    def read(self, ref):
        if ref is None:
            return None
        index = ref['buffer_index']
        if index not in self.cache:
            if self.zip is None:
                raise UnresolvedFrame('Capture ZIP is required to read blob %d' % index)
            data = self.zip.read('%06d' % index)
            if len(data) != ref['byte_length']:
                raise UnresolvedFrame('Capture blob %d length mismatch' % index)
            self.cache[index] = data
        return self.cache[index]


# ------------------------------------------------------- resource modelling
TEXTURE_KINDS = ('CreateTexture1D', 'CreateTexture2D', 'CreateTexture3D', 'SwapchainBuffer')
VIEW_KINDS = ('CreateShaderResourceView', 'CreateRenderTargetView', 'CreateDepthStencilView', 'CreateUnorderedAccessView')
STATE_KINDS = ('CreateSamplerState', 'CreateRasterizerState', 'CreateDepthStencilState', 'CreateBlendState')
SHADER_KINDS = ('CreateVertexShader', 'CreateHullShader', 'CreateDomainShader', 'CreateGeometryShader',
                'CreatePixelShader', 'CreateComputeShader')
STENCIL_FORMATS = {19, 20, 21, 22, 44, 45, 46, 47}


def subresource_count(res):
    d = res['descriptor']
    mips = d.get('MipLevels') or max(d.get('Width', 1), d.get('Height', 1), d.get('Depth', 1)).bit_length()
    arrays = 1 if res['kind'] == 'CreateTexture3D' else d.get('ArraySize', 1)
    return mips, arrays


def view_subresources(resources, view_id):
    """Exact (base resource, subresources or byte range) addressed by a view."""
    view = resources[view_id]
    base_id = view['base']
    base = resources[base_id]
    desc = view['descriptor']
    if base['kind'] == 'CreateBuffer':
        if desc is None:
            return base_id, {'bytes': (0, base['descriptor']['ByteWidth'])}
        member = [k for k in desc if k.startswith('Buffer')]
        if len(member) != 1:
            raise UnresolvedFrame('Buffer view union member required')
        f = desc[member[0]]
        stride = base['descriptor'].get('StructureByteStride') or 0
        if not stride:
            stride = FORMAT_BYTES.get(desc.get('Format'))
        if member[0] == 'BufferEx' and f.get('Flags', 0) & 1:
            stride = 4
        if not stride:
            raise UnresolvedFrame('Buffer view element size not derivable')
        first = f.get('FirstElement', f.get('ElementOffset', 0))
        count = f.get('NumElements', f.get('ElementWidth', 0))
        return base_id, {'bytes': (first * stride, (first + count) * stride)}
    if base['kind'] not in TEXTURE_KINDS:
        raise UnresolvedFrame('View base is neither buffer nor texture')
    mips, arrays = subresource_count(base)
    reading = view['kind'] == 'CreateShaderResourceView'
    mip, nmip, first, narr = 0, (mips if reading else 1), 0, arrays
    if desc is not None:
        members = [k for k in desc if k.startswith('Texture')]
        if len(members) != 1:
            raise UnresolvedFrame('Texture view union member required')
        name = members[0]
        f = desc[name]
        if reading:
            mip = f.get('MostDetailedMip', 0)
            nmip = f.get('MipLevels', 1)
            if nmip == 0xffffffff:
                nmip = mips - mip
        else:
            mip = f.get('MipSlice', 0)
            nmip = 1
        if 'FirstArraySlice' in f:
            first, narr = f['FirstArraySlice'], f['ArraySize']
            if narr == 0xffffffff:  # D3D11: -1 addresses all remaining slices
                narr = arrays - first
        elif name == 'TextureCubeArray':
            first, narr = f['First2DArrayFace'], f['NumCubes'] * 6
        elif name == 'TextureCube':
            first, narr = 0, 6
        elif name in ('Texture2DMS', 'Texture2D', 'Texture1D', 'Texture3D'):
            first, narr = 0, arrays if name == 'Texture3D' else 1
        if name == 'Texture3D' and not reading:
            if f.get('FirstWSlice', 0) != 0 or f.get('WSize', 0xffffffff) not in (0xffffffff, base['descriptor'].get('Depth', 1) >> mip):
                raise UnresolvedFrame('Partial Texture3D output slices require volume-range tracking')
    if nmip < 1 or mip + nmip > mips or narr < 1 or first + narr > arrays:
        raise UnresolvedFrame('View %d exceeds recorded mip/array storage: %r %r' % (view_id, view, base['descriptor']))
    return base_id, {'subresources': [m + a * mips for a in range(first, first + narr) for m in range(mip, mip + nmip)]}


FORMAT_BYTES = {1: 16, 2: 16, 3: 16, 4: 16, 6: 12, 7: 12, 8: 12, 10: 8, 11: 8, 13: 8, 16: 8, 17: 8,
                24: 4, 26: 4, 28: 4, 29: 4, 41: 4, 42: 4, 43: 4, 34: 4, 35: 4, 36: 4, 54: 2, 56: 2,
                57: 2, 58: 2, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1, 39: 4, 40: 4, 31: 4, 32: 4, 33: 4, 9: 8,
                27: 4, 37: 4, 38: 4, 30: 4, 49: 2, 50: 2, 51: 2, 52: 2, 53: 2, 55: 2, 60: 1, 34: 4}


# --------------------------------------------------- buffer byte provenance
class BufferContents:
    """Ordered writes to one buffer; bytes are resolved lazily from blobs."""

    def __init__(self, width, initial_ref, initial_kind):
        self.width = width
        # Segments: (start, end, source) with source a dict; None = unknown.
        self.segments = [(0, width, {'producer': 'capture_start', 'kind': initial_kind,
                                     'blob': initial_ref, 'blob_offset': 0})] if initial_ref else \
            [(0, width, {'producer': 'capture_start', 'kind': 'no_capture_start_bytes', 'blob': None})]

    def write(self, start, end, source, discard=False):
        if start < 0 or end > self.width or end < start:
            raise UnresolvedFrame('Buffer write outside resource')
        old = [(0, self.width, {'producer': source['producer'], 'kind': 'discarded_undefined', 'blob': None})] if discard else self.segments
        new = []
        for s, e, src in old:
            if e <= start or s >= end:
                new.append((s, e, src))
                continue
            if s < start:
                new.append((s, start, src))
            if e > end:
                shift = end - s
                new.append((end, e, dict(src, blob_offset=src.get('blob_offset', 0) + shift) if src.get('blob') else src))
        if end > start:
            new.append((start, end, source))
        self.segments = sorted(new, key=lambda r: r[0])

    def copy_from(self, other, dst_offset, src_start, src_end, producer):
        pieces = []
        for s, e, src in other.segments:
            a, b = max(s, src_start), min(e, src_end)
            if a >= b:
                continue
            moved = dict(src, blob_offset=src.get('blob_offset', 0) + (a - s)) if src.get('blob') else dict(src)
            moved['copied_by'] = producer
            pieces.append((a - src_start + dst_offset, b - src_start + dst_offset, moved))
        for s, e, src in pieces:
            self.write(s, e, src)

    def producers(self, start, end):
        result = []
        for s, e, src in self.segments:
            if e <= start or s >= end:
                continue
            p = src['producer']
            entry = {'producer': p, 'kind': src['kind'], 'bytes': [max(s, start), min(e, end)]}
            if 'copied_by' in src:
                entry['copied_by'] = src['copied_by']
            if not result or result[-1]['producer'] != p or result[-1]['kind'] != entry['kind']:
                result.append(entry)
            else:
                result[-1]['bytes'][1] = entry['bytes'][1]
        return result

    def read(self, blobs, start, end):
        out = bytearray()
        for s, e, src in self.segments:
            if e <= start or s >= end:
                continue
            a, b = max(s, start), min(e, end)
            if src.get('blob') is None:
                return None
            data = blobs.read(src['blob'])
            o = src.get('blob_offset', 0) + (a - s)
            out += data[o:o + (b - a)]
        return bytes(out) if len(out) == end - start else None


# --------------------------------------------------------- frame builder
class FrameGraph:
    def __init__(self, chunks, driver_listing=None, blobs=None, library=None):
        self.chunks = chunks
        self.blobs = blobs or Blobs(None)
        self.library = library or ShaderLibrary()
        self.opened, self.chunk_event, self.driver_actions = driver_listing or (None, {}, {})
        self.resources = {}
        self.reflections = {}
        self.state = None
        self.context = None
        self.actions = []
        self.notes = []
        self.buffers = {}
        self.tex_writers = {}     # (res, sub) -> {'base': producer, 'since': [events]}
        self.initial_list = []
        self.initial_contents = {}
        self.open_maps = {}
        self.unmatched_maps = []
        self.pending_queries = []
        self.capture_started = False
        self._cb_snapshots = {}
        self._pending_names = {}
        self.cb_resource_names = {}

    # ---------------- resources
    def _add(self, ident, record):
        if not ident:
            raise UnresolvedFrame('Null created resource')
        if ident in self.resources:
            raise UnresolvedFrame('Resource ID %d created twice' % ident)
        record.setdefault('debug_names', [])
        if ident in self._pending_names:
            record['debug_names'] = self._pending_names.pop(ident) + record['debug_names']
            record['debug_names_recorded_before_creation'] = True
        record['id'] = ident
        self.resources[ident] = record

    def _create(self, chunk):
        method = chunk['name'].split('::')[1]
        f = chunk['fields']
        idx = chunk['index']
        if method == 'CreateBuffer':
            self._add(f['pBuffer'], {'kind': method, 'chunk': idx, 'descriptor': f['pDesc'],
                                     'initial_data': f.get('InitialData'), 'in_frame': self.capture_started})
        elif method in ('CreateTexture1D', 'CreateTexture2D', 'CreateTexture3D'):
            contents = f.get('SubresourceContents')
            refs = list(contents) if isinstance(contents, Repeated) else ([] if contents is None else [contents])
            self._add(f['pTexture'], {'kind': method, 'chunk': idx, 'descriptor': f['Descriptor'],
                                      'creation_subresource_blobs': refs, 'in_frame': self.capture_started})
        elif method in VIEW_KINDS:
            self._add(f['pView'], {'kind': method, 'chunk': idx, 'base': f['pResource'], 'descriptor': f['pDesc']})
        elif method in SHADER_KINDS:
            data = self.blobs.read(f['pShaderBytecode'])
            sha = digest(data)
            refl = reflect(data)
            names, conflicts = self.library.names(sha)
            refl['export_names'] = names
            self.reflections[f['pShader']] = refl
            self._add(f['pShader'], {'kind': method, 'chunk': idx, 'dxbc_sha256': sha, 'dxbc_bytes': len(data),
                                     'source_matches': self.library.match(sha),
                                     'declarations': {k: refl[k] for k in ('shader_model', 'program_type', 'cb', 'srv', 'sampler', 'uav')},
                                     'input_signature': refl.get('input_signature'),
                                     'export_slot_names': None if names is None else {k: names[k] for k in ('cb', 'srv', 'uav')},
                                     'export_name_conflicts': conflicts,
                                     'declaration_crosscheck_against_export_assembly': self.library.asm_check(sha, refl)})
        elif method == 'CreateInputLayout':
            data = self.blobs.read(f['pShaderBytecodeWithInputSignature'])
            self._add(f['pInputLayout'], {'kind': method, 'chunk': idx, 'elements': f['pInputElementDescs'],
                                          'signature_sha256': digest(data)})
        elif method in STATE_KINDS:
            self._add(f['pState'], {'kind': method, 'chunk': idx, 'descriptor': f['Descriptor']})
        elif method == 'CreateQuery':
            self._add(f['pQuery'], {'kind': method, 'chunk': idx, 'descriptor': f['Descriptor']})
        else:
            raise UnresolvedFrame('Uninterpreted resource creation ' + method)

    def names(self, ident):
        r = self.resources.get(ident)
        return list(r['debug_names']) if r else []

    def base_of(self, view):
        r = self.resources.get(view)
        if r is None:
            raise UnresolvedFrame('Bound view %d has no creation record' % view)
        return r['base']

    def buffer(self, ident):
        if ident not in self.buffers:
            r = self.resources[ident]
            ref = r.get('initial_data')
            kind = ('capture_start_bytes_in_CreateBuffer_record' if not r['in_frame'] else 'in_frame_creation_data') if ref else 'none'
            self.buffers[ident] = BufferContents(r['descriptor']['ByteWidth'], ref, kind)
        return self.buffers[ident]

    def texture_initial(self, ident):
        r = self.resources[ident]
        if r['in_frame']:
            return {'kind': 'in_frame_creation', 'creation_blobs': len(r.get('creation_subresource_blobs', []))}
        if ident in self.initial_contents:
            ic = self.initial_contents[ident]
            return {'kind': 'initial_contents_chunk', 'chunk': ic['chunk'], 'omitted': ic.get('omitted')}
        if r.get('creation_subresource_blobs'):
            return {'kind': 'creation_record_data', 'creation_blobs': len(r['creation_subresource_blobs']),
                    'pre_capture_updates': r.get('pre_capture_updates', [])}
        listed = [e for e in self.initial_list if e['id'] == ident]
        return {'kind': 'no_serialized_bytes', 'needed_initials_entries': listed,
                'pre_capture_updates': r.get('pre_capture_updates', [])}

    # ---------------- texture versions
    def _tex_state(self, ident, sub):
        key = (ident, sub)
        if key not in self.tex_writers:
            self.tex_writers[key] = {'base': 'capture_start', 'since': []}
        return self.tex_writers[key]

    def tex_producers(self, ident, subs):
        rows = []
        for sub in subs:
            st = self._tex_state(ident, sub)
            rows.append({'subresource': sub, 'base': st['base'], 'partial_writers_since_base': list(st['since'])})
        return rows

    def tex_write(self, ident, subs, event, full):
        for sub in subs:
            st = self._tex_state(ident, sub)
            if full:
                st['base'] = event
                st['since'] = []
            else:
                st['since'].append(event)

    # ---------------- state helpers
    def _slots(self, stage, key, values, start):
        arr = self.state[stage][key]
        if start + len(values) > len(arr):
            raise UnresolvedFrame('Binding exceeds native slot array')
        arr[start:start + len(values)] = values

    def build(self):
        names = [c['name'] for c in self.chunks]
        for c in self.chunks:
            name, f, idx = c['name'], c['fields'], c['index']
            if name == 'Internal::DeviceInitialisation':
                self.context = f['ImmediateContext']
            elif name in ('Internal::Driver Initialisation Parameters', 'Internal::Frame Metadata'):
                if name == 'Internal::Frame Metadata':
                    self.frame_metadata = f
            elif name == 'Internal::Initial Contents':
                ident = f['id']
                self.initial_contents[ident] = {'chunk': idx, 'type': f.get('type'), 'omitted': f.get('OmittedContents')}
            elif name == 'Internal::List of Initial Contents Resources':
                self.initial_list = f['NeededInitials']
            elif name == 'Internal::Beginning of Capture':
                import copy
                self.state = copy.deepcopy(f['state'])
                self.capture_started = True
                self.begin_chunk = idx
                if any(f['HiddenStreamOutCounters']) or f['numAnnotations']:
                    raise UnresolvedFrame('Initial stream-out counters/annotations require interpretation')
            elif name == 'Internal::End of Capture':
                self.end_chunk = idx
            elif name.startswith('ID3D11Device::Create'):
                self._create(c)
            elif name == 'IDXGISwapChain::GetBuffer':
                self._add(f['SwapbufferID'], {'kind': 'SwapchainBuffer', 'chunk': idx, 'descriptor': f['BackbufferDescriptor'],
                                              'creation_subresource_blobs': [], 'in_frame': False})
            elif name == 'ID3D11Resource::SetDebugName':
                if f['pResource'] in self.resources:
                    self.resources[f['pResource']]['debug_names'].append(f['Name'])
                else:
                    self._pending_names.setdefault(f['pResource'], []).append(f['Name'])
            elif name.startswith('ID3D11DeviceContext::') or name == 'IDXGISwapChain::Present':
                if f.get('Context') != self.context:
                    raise UnresolvedFrame('Deferred/foreign context command requires interpretation')
                method = name.split('::')[1]
                if self.state is None:
                    self._pre_capture(method, f, idx)
                else:
                    self._context(method, f, idx)
            else:
                raise UnresolvedFrame('Uninterpreted capture command ' + name)
        if self.state is None:
            raise UnresolvedFrame('Capture has no Beginning of Capture state')
        for ident, names in self._pending_names.items():
            self.notes.append({'note': 'debug name for a resource never created in this capture', 'resource': ident, 'names': names})
        self.resolve_cbuffer_names()
        return self

    def _pre_capture(self, method, f, idx):
        if method in ('UpdateSubresource', 'UpdateSubresource1'):
            r = self.resources.get(f['pDstResource'])
            if r is None:
                raise UnresolvedFrame('Pre-capture update of unrecorded resource')
            r.setdefault('pre_capture_updates', []).append({'chunk': idx, 'subresource': f['DstSubresource']})
        else:
            raise UnresolvedFrame('Pre-capture context command requires interpretation: ' + method)

    def event_of(self, idx):
        return self.chunk_event.get(idx)

    # ---------------- context commands
    def _context(self, method, f, idx):
        st = self.state
        ia, om, rs = st['IA'], st['OM'], st['RS']
        ev = self.event_of(idx)
        if method == 'IASetInputLayout':
            ia['Layout'] = f['pInputLayout']
        elif method == 'IASetVertexBuffers':
            s = f['StartSlot']
            self._slots('IA', 'VBs', f['ppVertexBuffers'], s)
            self._slots('IA', 'Strides', f['pStrides'], s)
            self._slots('IA', 'Offsets', f['pOffsets'], s)
        elif method == 'IASetIndexBuffer':
            ia.update(IndexBuffer=f['pIndexBuffer'], IndexFormat=f['Format'], IndexOffset=f['Offset'])
        elif method == 'IASetPrimitiveTopology':
            ia['Topo'] = f['Topology']
        elif method[2:] == 'SetShader' and method[:2] in STAGES:
            st[method[:2]].update(Object=f['pShader'], Instances=f['ppClassInstances'], NumInstances=f['NumClassInstances'])
            if f['NumClassInstances']:
                raise UnresolvedFrame('Class instances require interpretation')
        elif method[:2] in STAGES and method[2:] in ('SetConstantBuffers', 'SetShaderResources', 'SetSamplers'):
            key, arr = {'SetConstantBuffers': ('ConstantBuffers', 'ppConstantBuffers'),
                        'SetShaderResources': ('SRVs', 'ppShaderResourceViews'),
                        'SetSamplers': ('Samplers', 'ppSamplers')}[method[2:]]
            self._slots(method[:2], key, f[arr], f['StartSlot'])
            if key == 'ConstantBuffers':
                n = len(f[arr])
                st[method[:2]]['CBOffsets'][f['StartSlot']:f['StartSlot'] + n] = [0] * n
                st[method[:2]]['CBCounts'][f['StartSlot']:f['StartSlot'] + n] = [4096] * n
        elif method == 'CSSetUnorderedAccessViews':
            s = f['StartSlot']
            vals = f['ppUnorderedAccessViews']
            if s + len(vals) > len(st['CSUAVs']):
                raise UnresolvedFrame('CS UAV binding exceeds slot array')
            st['CSUAVs'][s:s + len(vals)] = vals
            if any(c != 0xffffffff for c in f.get('pUAVInitialCounts', [])):
                self.notes.append({'event': ev, 'note': 'UAV hidden counter initial values set', 'values': f['pUAVInitialCounts']})
        elif method == 'OMSetRenderTargets':
            n = f['NumViews']
            if n != 0xffffffff:
                om['RenderTargets'] = list(f['ppRenderTargetViews']) + [0] * (len(om['RenderTargets']) - len(f['ppRenderTargetViews']))
            om['DepthView'] = f['pDepthStencilView']
        elif method == 'OMSetRenderTargetsAndUnorderedAccessViews':
            raise UnresolvedFrame('OM UAV binding requires interpretation')
        elif method == 'OMSetDepthStencilState':
            om.update(DepthStencilState=f['pDepthStencilState'], StencRef=f['StencilRef'])
        elif method == 'OMSetBlendState':
            om.update(BlendState=f['pBlendState'], BlendFactor=f['BlendFactor'], SampleMask=f['SampleMask'])
        elif method == 'RSSetState':
            rs['State'] = f['pRasterizerState']
        elif method == 'RSSetViewports':
            rs.update(NumViews=f['NumViewports'], Viewports=f['pViewports'])
        elif method in ('RSSetScissors', 'RSSetScissorRects'):
            rs.update(NumScissors=f['NumRects'], Scissors=f['pRects'])
        elif method == 'Map':
            key = (f['pResource'], f['Subresource'])
            if key in self.open_maps:
                raise UnresolvedFrame('Resource mapped twice without Unmap')
            self.open_maps[key] = {'chunk': idx, 'event': ev, 'map_type': f['MapType']}
        elif method == 'Unmap':
            self._unmap(f, idx, ev)
        elif method in ('UpdateSubresource', 'UpdateSubresource1'):
            self._update(f, idx, ev)
        elif method == 'CopyResource':
            self._copy(f, idx, ev, region=False)
        elif method == 'CopySubresourceRegion':
            self._copy(f, idx, ev, region=True)
        elif method == 'ClearRenderTargetView':
            self._clear(f['pRenderTargetView'], f, idx, ev, 'ClearRenderTargetView')
        elif method == 'ClearDepthStencilView':
            self._clear(f['pDepthStencilView'], f, idx, ev, 'ClearDepthStencilView')
        elif method == 'ClearUnorderedAccessViewFloat' or method == 'ClearUnorderedAccessViewUint':
            self._clear(f['pUnorderedAccessView'], f, idx, ev, method)
        elif method in ('Draw', 'DrawIndexed', 'DrawInstanced', 'DrawIndexedInstanced'):
            self._draw(method, f, idx, ev)
        elif method == 'Dispatch':
            self._dispatch(f, idx, ev)
        elif method in ('Begin', 'End'):
            self.pending_queries.append({'event': ev, 'chunk': idx, 'method': method, 'query': f['pAsync']})
        elif method == 'Present':
            self._present(f, idx, ev)
        else:
            raise UnresolvedFrame('Uninterpreted context command ' + method)

    def _resource_kind(self, ident):
        r = self.resources.get(ident)
        return r['kind'] if r else None

    def _unmap(self, f, idx, ev):
        ident, sub = f['pResource'], f['Subresource']
        opened = self.open_maps.pop((ident, sub), None)
        if opened is None:
            # The Map was issued before Beginning of Capture; pinned Serialise_Unmap
            # still records MapType/range/MapWrittenData from the open-map intercept,
            # and replay applies those bytes.  Retained explicitly, not dropped.
            if f.get('MapWrittenData') is None and f['Byte offset to start of written data'] != f['Byte offset to end of written data']:
                raise UnresolvedFrame('Unmap of pre-capture Map without written bytes')
            opened = {'chunk': None, 'event': None, 'map_type': f['MapType'], 'map_issued_before_capture': True}
            self.notes.append({'event': ev, 'chunk': idx, 'note': 'Unmap of a Map issued before Beginning of Capture; '
                               'serialized MapWrittenData applied', 'resource': ident})
        if opened['map_type'] != f['MapType']:
            raise UnresolvedFrame('Map/Unmap type mismatch')
        if ident not in self.resources:
            raise UnresolvedFrame('Unmap of unrecorded resource %d' % ident)
        start = f['Byte offset to start of written data']
        end = f['Byte offset to end of written data']
        ref = f['MapWrittenData']
        mt = f['MapType']
        action = {'event_id': ev, 'chunk_index': idx, 'kind': 'Unmap', 'map_chunk': opened['chunk'], 'map_event': opened['event'],
                  'map_issued_before_capture': opened.get('map_issued_before_capture', False),
                  'map_type': mt, 'resource': ident, 'subresource': sub, 'written_bytes': [start, end],
                  'blob': ref}
        if self._resource_kind(ident) == 'CreateBuffer':
            if mt in (2, 3, 4, 5):
                self.buffer(ident).write(start, end, {'producer': ev, 'kind': 'unmap', 'blob': ref, 'blob_offset': 0}, discard=(mt == 4))
        else:
            if mt in (2, 3, 4, 5):
                full = mt == 4
                self.tex_write(ident, [sub], ev, full)
        self.actions.append(action)

    def _update(self, f, idx, ev):
        ident, sub = f['pDstResource'], f['DstSubresource']
        r = self.resources[ident]
        if f['IsUpdate']:
            box = f['pDstBox']
            ref = f['pSrcData']
        else:
            box = None
            ref = f['Contents']
        action = {'event_id': ev, 'chunk_index': idx, 'kind': 'UpdateSubresource', 'resource': ident, 'subresource': sub, 'box': box}
        if r['kind'] == 'CreateBuffer':
            s = box['left'] if box else 0
            e = box['right'] if box else r['descriptor']['ByteWidth']
            self.buffer(ident).write(s, e, {'producer': ev, 'kind': 'update_subresource', 'blob': ref, 'blob_offset': 0},
                                     discard=f.get('CopyFlags') == 2)
        else:
            self.tex_write(ident, [sub], ev, full=box is None)
        self.actions.append(action)

    def _full_box(self, ident, sub, box, dx=0, dy=0, dz=0):
        if box is None:
            return dx == dy == dz == 0
        r = self.resources[ident]
        mips, _ = subresource_count(r)
        mip = sub % mips
        d = r['descriptor']
        w, h = max(1, d['Width'] >> mip), max(1, d.get('Height', 1) >> mip)
        depth = max(1, d.get('Depth', 1) >> mip)
        return (dx, dy, dz) == (0, 0, 0) and box['left'] == 0 and box['top'] == 0 and box['front'] == 0 and \
            box['right'] >= w and box['bottom'] >= h and box['back'] >= depth

    def _copy(self, f, idx, ev, region):
        dst, src = f['pDstResource'], f['pSrcResource']
        dk, sk = self._resource_kind(dst), self._resource_kind(src)
        if dk is None or sk is None:
            raise UnresolvedFrame('Copy references unrecorded resource')
        action = {'event_id': ev, 'chunk_index': idx, 'kind': 'CopySubresourceRegion' if region else 'CopyResource',
                  'source': src, 'destination': dst}
        if dk == 'CreateBuffer' and sk == 'CreateBuffer':
            sb, db = self.buffer(src), self.buffer(dst)
            if region:
                box = f['pSrcBox']
                s0, s1 = (box['left'], box['right']) if box else (0, sb.width)
                action['reads'] = [{'via': 'copy_source', 'resource': src, 'bytes': [s0, s1], 'producers': sb.producers(s0, s1)}]
                db.copy_from(sb, f['DstX'], s0, s1, ev)
                action['writes'] = [{'resource': dst, 'bytes': [f['DstX'], f['DstX'] + s1 - s0]}]
            else:
                action['reads'] = [{'via': 'copy_source', 'resource': src, 'bytes': [0, sb.width], 'producers': sb.producers(0, sb.width)}]
                db.copy_from(sb, 0, 0, sb.width, ev)
                action['writes'] = [{'resource': dst, 'bytes': [0, db.width]}]
        elif dk in TEXTURE_KINDS and sk in TEXTURE_KINDS:
            if region:
                ssub, dsub = f['SrcSubresource'], f['DstSubresource']
                full = self._full_box(dst, dsub, f['pSrcBox'], f['DstX'], f['DstY'], f['DstZ'])
                if f['pSrcBox'] is not None and full:
                    # Exact cover also requires matching destination extent.
                    full = self._full_box(dst, dsub, {'left': 0, 'top': 0, 'front': 0,
                                                      'right': f['pSrcBox']['right'] - f['pSrcBox']['left'],
                                                      'bottom': f['pSrcBox']['bottom'] - f['pSrcBox']['top'],
                                                      'back': f['pSrcBox']['back'] - f['pSrcBox']['front']})
                elif f['pSrcBox'] is None:
                    sd, dd = self.resources[src]['descriptor'], self.resources[dst]['descriptor']
                    full = full and sd['Width'] >= dd['Width'] and sd.get('Height', 1) >= dd.get('Height', 1)
                action['reads'] = [{'via': 'copy_source', 'resource': src, 'subresources': [ssub],
                                    'producers': self.tex_producers(src, [ssub])}]
                action['box'] = f['pSrcBox']
                action['destination_offset'] = [f['DstX'], f['DstY'], f['DstZ']]
                self.tex_write(dst, [dsub], ev, full)
                action['writes'] = [{'resource': dst, 'subresources': [dsub], 'full_overwrite': full}]
            else:
                mips, arrays = subresource_count(self.resources[src])
                subs = list(range(mips * arrays))
                action['reads'] = [{'via': 'copy_source', 'resource': src, 'subresources': subs, 'producers': self.tex_producers(src, subs)}]
                self.tex_write(dst, subs, ev, True)
                action['writes'] = [{'resource': dst, 'subresources': subs, 'full_overwrite': True}]
        else:
            raise UnresolvedFrame('Buffer/texture cross copy requires interpretation')
        self.actions.append(action)

    def _clear(self, view, f, idx, ev, kind):
        base, sel = view_subresources(self.resources, view)
        full = True
        if kind == 'ClearDepthStencilView':
            fmt = (self.resources[view]['descriptor'] or {}).get('Format')
            full = f['ClearFlags'] == 3 or fmt not in STENCIL_FORMATS
        action = {'event_id': ev, 'chunk_index': idx, 'kind': kind, 'view': view, 'resource': base,
                  'values': {k: v for k, v in f.items() if k in ('ColorRGBA', 'ClearFlags', 'Depth', 'Stencil', 'Values')}}
        if 'subresources' in sel:
            self.tex_write(base, sel['subresources'], ev, full)
            action['writes'] = [{'resource': base, 'subresources': sel['subresources'], 'full_overwrite': full}]
        else:
            s, e = sel['bytes']
            self.buffer(base).write(s, e, {'producer': ev, 'kind': 'clear_uav', 'blob': None})
            action['writes'] = [{'resource': base, 'bytes': [s, e]}]
        self.actions.append(action)

    # ---------------- shader-resource reads
    def _stage_bindings(self, stage, action, event=None):
        st = self.state[stage]
        shader = st['Object']
        if not shader:
            return None
        res = self.resources[shader]
        refl = self.reflections[shader]
        names = refl.get('export_names') or {'cb': {}, 'srv': {}, 'uav': {}, 'cb_layouts': {}}
        info = {'resource': shader, 'debug_names': res['debug_names'], 'dxbc_sha256': res['dxbc_sha256'],
                'source_matches': res['source_matches'], 'constant_buffers': [], 'srvs': [], 'samplers': [],
                'bound_but_unused': {'cb': [], 'srv': [], 'sampler': []}}
        for slot, ident in enumerate(st['ConstantBuffers']):
            if slot not in refl['cb']:
                if ident:
                    info['bound_but_unused']['cb'].append([slot, ident])
                continue
            entry = {'slot': slot, 'declared_float4': refl['cb'][slot]['float4_count']}
            if slot in names['cb']:
                entry['name'] = names['cb'][slot]
                entry['name_source'] = 'installed_export'
            if not ident:
                entry.update(resource=0, unbound=True)
                info['constant_buffers'].append(entry)
                continue
            width = self.resources[ident]['descriptor']['ByteWidth']
            start = st['CBOffsets'][slot] * 16
            end = min(width, start + min(st['CBCounts'][slot], refl['cb'][slot]['float4_count']) * 16)
            buf = self.buffer(ident)
            entry.update(resource=ident, bytes=[start, end], debug_names=self.resources[ident]['debug_names'],
                         producers=buf.producers(start, end))
            self._cb_snapshots[(event, stage, slot)] = list(buf.segments)
            info['constant_buffers'].append(entry)
        for slot, view in enumerate(st['SRVs']):
            if slot not in refl['srv']:
                if view:
                    info['bound_but_unused']['srv'].append([slot, view])
                continue
            entry = {'slot': slot, 'declared_dimension': refl['srv'][slot].get('dimension', refl['srv'][slot]['declaration']),
                     'view': view}
            if slot in names['srv']:
                entry['name'] = names['srv'][slot]
            if not view:
                entry['unbound'] = True
            else:
                entry.update(self._view_read(view))
            info['srvs'].append(entry)
        for slot, sampler in enumerate(st['Samplers']):
            if slot in refl['sampler']:
                info['samplers'].append({'slot': slot, 'sampler': sampler})
            elif sampler:
                info['bound_but_unused']['sampler'].append([slot, sampler])
        return info

    def _view_read(self, view):
        base, sel = view_subresources(self.resources, view)
        entry = {'resource': base, 'debug_names': self.resources[base]['debug_names'],
                 'view_format': (self.resources[view]['descriptor'] or {}).get('Format')}
        if 'subresources' in sel:
            entry['subresources'] = sel['subresources']
            entry['producers'] = self.tex_producers(base, sel['subresources'])
        else:
            s, e = sel['bytes']
            entry['bytes'] = [s, e]
            entry['producers'] = self.buffer(base).producers(s, e)
        return entry

    def resolve_cbuffer_names(self):
        """Name unmatched CB reads only via the same CB resource bound under that
        name by an installed-export-matched program, then read camera variables."""
        learned = {}
        for a in self.actions:
            for stage, info in a.get('shaders', {}).items():
                names = self.reflections[info['resource']].get('export_names')
                if not names:
                    continue
                for cb in info['constant_buffers']:
                    if cb.get('unbound') or 'name' not in cb:
                        continue
                    layout = names['cb_layouts'].get(cb['name'])
                    learned.setdefault(cb['resource'], {}).setdefault(cb['name'], (layout, info['dxbc_sha256'], a['event_id']))
        self.cb_resource_names = {str(k): {n: {'witness_dxbc_sha256': v[1], 'witness_event': v[2], 'layout': v[0]}
                                           for n, v in d.items()} for k, d in learned.items()}
        for a in self.actions:
            for stage, info in a.get('shaders', {}).items():
                for cb in info['constant_buffers']:
                    if cb.get('unbound') or 'name' in cb:
                        continue
                    options = learned.get(cb['resource'], {})
                    if len(options) == 1:
                        name, (layout, sha, ev) = next(iter(options.items()))
                        cb['name'] = name
                        cb['name_source'] = 'same_cb_resource_named_by_matched_program'
                        cb['name_witness'] = {'dxbc_sha256': sha, 'event_id': ev}
                    elif options:
                        cb['name_candidates'] = sorted(options)
            if 'VS' in a.get('shaders', {}):
                a['camera_values'] = self._camera_values(a, learned)

    def _camera_values(self, a, learned):
        out = {}
        info = a['shaders']['VS']
        export = self.reflections[info['resource']].get('export_names')
        for cb in info['constant_buffers']:
            name = cb.get('name')
            if cb.get('unbound') or name not in ('UnityPerCamera', 'UnityPerFrame', 'UnityPerCameraRare'):
                continue
            layout = (export or {}).get('cb_layouts', {}).get(name) or learned.get(cb['resource'], {}).get(name, (None,))[0]
            if layout is None:
                continue
            snap = BufferContents(self.resources[cb['resource']]['descriptor']['ByteWidth'], None, 'none')
            snap.segments = self._cb_snapshots[(a['event_id'], 'VS', cb['slot'])]
            for var in layout['variables']:
                if var['name'] not in CAMERA_VARIABLES:
                    continue
                s0 = cb['bytes'][0] + var['offset']
                data = snap.read(self.blobs, s0, s0 + var['size'])
                out[var['name']] = list(struct.unpack('<%df' % (var['size'] // 4), data)) if data else None
        return out

    # ---------------- draws
    def _output_targets(self):
        om = self.state['OM']
        blend = self.resources[om['BlendState']]['descriptor'] if om['BlendState'] else None
        targets = []
        for slot, view in enumerate(om['RenderTargets']):
            if not view:
                continue
            base, sel = view_subresources(self.resources, view)
            if blend is not None:
                rt = blend['RenderTarget'][slot if blend['IndependentBlendEnable'] else 0]
                mask, blending = rt['RenderTargetWriteMask'], rt['BlendEnable']
                bdesc = rt
            else:
                mask, blending, bdesc = 15, False, None
            targets.append({'slot': slot, 'view': view, 'resource': base, 'subresources': sel['subresources'],
                            'debug_names': self.resources[base]['debug_names'], 'write_mask': mask,
                            'blend_enable': blending, 'blend': bdesc,
                            'view_format': (self.resources[view]['descriptor'] or {}).get('Format')})
        depth = None
        if om['DepthView']:
            base, sel = view_subresources(self.resources, om['DepthView'])
            ds = self.resources[om['DepthStencilState']]['descriptor'] if om['DepthStencilState'] else \
                {'DepthEnable': True, 'DepthWriteMask': 1, 'DepthFunc': 2, 'StencilEnable': False}
            flags = (self.resources[om['DepthView']]['descriptor'] or {}).get('Flags', 0)
            write = bool(ds['DepthEnable'] and ds['DepthWriteMask'] and not flags & 1)
            stencil_write = False
            if ds.get('StencilEnable') and ds.get('StencilWriteMask') and not flags & 2:
                stencil_write = any(ds[face][op] != 1 for face in ('FrontFace', 'BackFace')
                                    for op in ('StencilFailOp', 'StencilDepthFailOp', 'StencilPassOp'))
            depth = {'view': om['DepthView'], 'resource': base, 'subresources': sel['subresources'],
                     'debug_names': self.resources[base]['debug_names'], 'depth_enable': ds['DepthEnable'],
                     'depth_write': write, 'depth_func': ds.get('DepthFunc'), 'stencil_enable': ds.get('StencilEnable'),
                     'stencil_write': stencil_write, 'stencil_ref': om['StencRef'], 'state': om['DepthStencilState'],
                     'read_only_flags': flags}
        return targets, depth

    def _draw(self, method, f, idx, ev):
        st = self.state
        if st['Predicate']:
            raise UnresolvedFrame('Predicated draw requires captured query result')
        if any(st['SO']['Buffers']):
            raise UnresolvedFrame('Stream-out requires interpretation')
        if any(st['OM']['UAVs']):
            raise UnresolvedFrame('OM UAV writes require interpretation')
        ia = st['IA']
        action = {'event_id': ev, 'chunk_index': idx, 'kind': method, 'arguments': {k: v for k, v in f.items() if k not in ('Context', 'DebugMessages')},
                  'topology': ia['Topo'], 'shaders': {}, 'reads': [], 'writes': []}
        layout = self.resources.get(ia['Layout']) if ia['Layout'] else None
        vs = st['VS']['Object']
        consumed = None
        if vs and 'input_signature' in self.reflections[vs]:
            consumed = {(r['semantic'].upper(), r['index']) for r in self.reflections[vs]['input_signature'] if r['system_value'] == 0}
        elements = []
        if layout:
            for e in layout['elements']:
                row = dict(e)
                row['consumed_by_vs'] = None if consumed is None else (e['SemanticName'].upper(), e['SemanticIndex']) in consumed
                elements.append(row)
        action['input_layout'] = {'resource': ia['Layout'], 'debug_names': layout['debug_names'] if layout else [], 'elements': elements}
        used_slots = sorted({e['InputSlot'] for e in elements if e['consumed_by_vs'] is not False})
        action['vertex_buffers'] = []
        for slot in used_slots:
            ident = ia['VBs'][slot]
            entry = {'slot': slot, 'resource': ident, 'stride': ia['Strides'][slot], 'offset': ia['Offsets'][slot],
                     'semantics': [f"{e['SemanticName']}{e['SemanticIndex']}" for e in elements if e['InputSlot'] == slot and e['consumed_by_vs'] is not False]}
            if ident:
                r = self.resources[ident]
                entry.update(debug_names=r['debug_names'], byte_width=r['descriptor']['ByteWidth'],
                             usage=r['descriptor']['Usage'], cpu_access=r['descriptor']['CPUAccessFlags'],
                             producers=self.buffer(ident).producers(ia['Offsets'][slot], r['descriptor']['ByteWidth']))
            action['vertex_buffers'].append(entry)
        action['bound_unreferenced_vertex_slots'] = [[s, v] for s, v in enumerate(ia['VBs']) if v and s not in used_slots]
        if method.startswith('DrawIndexed'):
            ib = ia['IndexBuffer']
            r = self.resources[ib]
            size = 2 if ia['IndexFormat'] == 57 else 4 if ia['IndexFormat'] == 42 else None
            if size is None:
                raise UnresolvedFrame('Unknown index format')
            s = ia['IndexOffset'] + f['StartIndexLocation'] * size
            e = s + f['IndexCount'] * size
            action['index_buffer'] = {'resource': ib, 'format': ia['IndexFormat'], 'offset': ia['IndexOffset'],
                                      'bytes': [s, e], 'debug_names': r['debug_names'],
                                      'byte_width': r['descriptor']['ByteWidth'], 'usage': r['descriptor']['Usage'],
                                      'producers': self.buffer(ib).producers(s, e)}
        for stage in ('VS', 'HS', 'DS', 'GS', 'PS'):
            info = self._stage_bindings(stage, action, ev)
            if info:
                action['shaders'][stage] = info
        targets, depth = self._output_targets()
        action['render_targets'] = []
        for t in targets:
            t['prior_content'] = self.tex_producers(t['resource'], t['subresources'])
            action['render_targets'].append(t)
        action['depth'] = depth
        if depth:
            depth['prior_content'] = self.tex_producers(depth['resource'], depth['subresources'])
        rs = st['RS']
        action['viewports'] = rs['Viewports'][:rs['NumViews']] if rs.get('NumViews') else rs['Viewports'][:1]
        action['scissors'] = rs['Scissors'][:rs['NumScissors']] if rs.get('NumScissors') else []
        action['rasterizer'] = {'state': rs['State'], 'descriptor': self.resources[rs['State']]['descriptor'] if rs['State'] else None}
        action['blend_state'] = st['OM']['BlendState']
        for t in targets:
            if t['write_mask']:
                self.tex_write(t['resource'], t['subresources'], ev, False)
                action['writes'].append({'resource': t['resource'], 'subresources': t['subresources'], 'via': 'render_target'})
        if depth and (depth['depth_write'] or depth['stencil_write']):
            self.tex_write(depth['resource'], depth['subresources'], ev, False)
            action['writes'].append({'resource': depth['resource'], 'subresources': depth['subresources'], 'via': 'depth_stencil'})
        self.actions.append(action)

    def _dispatch(self, f, idx, ev):
        info = self._stage_bindings('CS', None, ev)
        if info is None:
            raise UnresolvedFrame('Dispatch without compute shader')
        refl = self.reflections[self.state['CS']['Object']]
        uav_names = (refl.get('export_names') or {}).get('uav', {})
        action = {'event_id': ev, 'chunk_index': idx, 'kind': 'Dispatch',
                  'arguments': {k: v for k, v in f.items() if k not in ('Context', 'DebugMessages')},
                  'shaders': {'CS': info}, 'uavs': [], 'writes': []}
        for slot, view in enumerate(self.state['CSUAVs']):
            if slot not in refl['uav']:
                if view:
                    info['bound_but_unused'].setdefault('uav', []).append([slot, view])
                continue
            entry = {'slot': slot, 'declaration': refl['uav'][slot]['declaration'], 'view': view}
            if slot in uav_names:
                entry['name'] = uav_names[slot]
            if view:
                entry.update(self._view_read(view))
            action['uavs'].append(entry)
        for u in action['uavs']:
            if not u.get('view'):
                continue
            if 'subresources' in u:
                self.tex_write(u['resource'], u['subresources'], ev, False)
                action['writes'].append({'resource': u['resource'], 'subresources': u['subresources'], 'via': 'uav'})
            else:
                s, e = u['bytes']
                self.buffer(u['resource']).write(s, e, {'producer': ev, 'kind': 'gpu_uav_write', 'blob': None})
                action['writes'].append({'resource': u['resource'], 'bytes': [s, e], 'via': 'uav'})
        self.actions.append(action)

    def _present(self, f, idx, ev):
        swap = [r for r in self.resources.values() if r['kind'] == 'SwapchainBuffer']
        action = {'event_id': ev, 'chunk_index': idx, 'kind': 'Present', 'arguments': {k: v for k, v in f.items() if k != 'Context'},
                  'reads': [{'via': 'present', 'resource': r['id'], 'producers': self.tex_producers(r['id'], [0])} for r in swap]}
        self.actions.append(action)

    # ---------------- validation against the official driver listing
    def validate_events(self):
        if not self.chunk_event:
            return {'driver_listing': False}
        expected = {}
        for ev, a in self.driver_actions.items():
            expected[a['api_events'][-1]['chunk_index']] = ev
        mismatched = []
        recorded = {a['chunk_index']: a for a in self.actions}
        for chunk, ev in expected.items():
            a = recorded.get(chunk)
            if a is None or a['event_id'] != ev:
                mismatched.append({'chunk': chunk, 'driver_event': ev, 'interpreted': a and a['kind']})
        action_kinds = {'Draw', 'DrawIndexed', 'DrawInstanced', 'DrawIndexedInstanced', 'Dispatch', 'CopyResource',
                        'CopySubresourceRegion', 'ClearRenderTargetView', 'ClearDepthStencilView', 'Present'}
        missing = [a['event_id'] for a in self.actions if a['kind'] in action_kinds and a['event_id'] not in self.driver_actions]
        return {'driver_listing': True, 'driver_actions': len(self.driver_actions),
                'interpreted_actions': sum(1 for a in self.actions if a['kind'] in action_kinds),
                'mismatched_driver_actions': mismatched, 'interpreted_actions_not_in_driver_listing': missing}


CAMERA_VARIABLES = ('unity_MatrixVP', 'unity_MatrixV', 'glstate_matrix_projection', '_WorldSpaceCameraPos',
                    '_ProjectionParams', '_ScreenParams', 'unity_CameraProjection', '_ZBufferParams')


# ----------------------------------------------------------- pass grouping
LABEL_RULES = [
    # (label, test on evidence dict) -- evidence: ps/vs/cs debug names, target names, depth names, kind
    ('readback_copy', lambda e: e['kind'] in ('CopySubresourceRegion', 'CopyResource') and any(n.startswith('Readback-') for n in e['dst_names'])),
    ('grabpass_copy', lambda e: e['kind'] == 'CopySubresourceRegion' and 'GrabPass Temp' in e['dst_names']),
    ('present', lambda e: e['kind'] == 'Present'),
    ('post_curves_texture_copy', lambda e: e['kind'] == 'CopySubresourceRegion' and 'Internal Curves Texture' in e['dst_names']),
    ('deferred_depth_copy', lambda e: e['kind'] == 'CopyResource' and 'Deferred Depth' in e['dst_names']),
    ('staging_upload_copy', lambda e: e['kind'] in ('CopySubresourceRegion', 'CopyResource') and e.get('src_usage') == 3),
    ('ui', lambda e: e['shader'] <= {'UI/Default'} and e['shader']),
    ('sss_profile', lambda e: e['shader'] == {'Hidden/SSS_Profile'}),
    ('sss_separable_blur', lambda e: e['shader'] == {'Hidden/SeparableSSS'}),
    ('shadow_map', lambda e: 'Shadowmap' in e['depth_names'] and not e['rt_names']),
    ('depth_texture_prepass', lambda e: 'Camera DepthTexture' in e['depth_names'] and not e['rt_names']),
    ('depth_normals', lambda e: 'Camera DepthNormalsTexture' in e['rt_names']),
    ('screenspace_shadows', lambda e: e['shader'] == {'Hidden/Internal-ScreenSpaceShadows'}),
    ('sss_lighting', lambda e: 'LightingTexture' in e['rt_names'] and 'Hidden/LightingPass' in e['shader']),
    ('motion_vectors', lambda e: e['shader'] == {'Hidden/Internal-MotionVectors'}),
    ('ambient_occlusion_msvo', lambda e: any('MultiScaleVO' in n for n in e['shader'])),
    ('color_grading_lut_bake', lambda e: any(n.startswith('Lut3DBaker') for n in e['shader'])),
    ('temporal_aa_ctaa', lambda e: any(n.startswith('Hidden/CTAA') for n in e['shader'])),
    ('post_depth_of_field', lambda e: e['shader'] == {'Hidden/PostProcessing/DepthOfField'}),
    ('post_uber', lambda e: e['shader'] == {'Hidden/PostProcessing/Uber'}),
    ('post_smaa', lambda e: e['shader'] == {'Hidden/PostProcessing/SubpixelMorphologicalAntialiasing'}),
    ('post_final_pass', lambda e: e['shader'] == {'Hidden/PostProcessing/FinalPass'}),
    ('post_copy_std', lambda e: e['shader'] == {'Hidden/PostProcessing/CopyStd'}),
    ('deferred_reflections', lambda e: e['shader'] == {'Hidden/Internal-DeferredReflections'}),
    ('deferred_shading', lambda e: e['shader'] == {'Hidden/Internal-DeferredShading'}),
    ('blit_copy', lambda e: e['shader'] == {'Hidden/BlitCopy'}),
    ('forward_scene', lambda e: e['kind'] in ('DrawIndexed', 'Draw') and e['shader'] and
     not any(n.startswith('Hidden/') for n in e['shader'])),
]


def action_outputs(a):
    if a['kind'] in ('Draw', 'DrawIndexed', 'DrawInstanced', 'DrawIndexedInstanced'):
        vp = a['viewports'][0] if a['viewports'] else None
        return ('draw', tuple((t['resource'], tuple(t['subresources'])) for t in a['render_targets']),
                (a['depth']['resource'], tuple(a['depth']['subresources'])) if a['depth'] else None,
                (vp['TopLeftX'], vp['TopLeftY'], vp['Width'], vp['Height']) if vp else None)
    if a['kind'] == 'Dispatch':
        return ('dispatch', a['event_id'])
    return ('transfer', a['event_id'])


def camera_signature(values):
    if not values or 'unity_MatrixVP' not in values or values['unity_MatrixVP'] is None:
        return None
    # The view is identified by the actual unity_MatrixVP bytes alone: shaders
    # declare different subsets of the other camera variables.
    return digest(struct.pack('<16f', *values['unity_MatrixVP']))[:16]


def camera_views(graph, passes):
    views = {}
    by_event = {a['event_id']: a for a in graph.actions if a['event_id'] is not None}
    for p in passes:
        for e in p['events']:
            a = by_event[e]
            sig = camera_signature(a.get('camera_values'))
            if not sig:
                continue
            v = views.setdefault(sig, {'unity_MatrixVP': a['camera_values']['unity_MatrixVP'], 'other_values': {}, 'first_event': e,
                                       'last_event': e, 'draws': 0, 'pass_labels': {}})
            v['last_event'] = e
            for name, value in a['camera_values'].items():
                if name != 'unity_MatrixVP' and value is not None and value not in v['other_values'].setdefault(name, []):
                    v['other_values'][name].append(value)
            v['draws'] += 1
            v['pass_labels'][p['label']] = v['pass_labels'].get(p['label'], 0) + 1
            vp = a['viewports'][0] if a.get('viewports') else None
            v.setdefault('viewports', [])
            key = vp and [vp['Width'], vp['Height']]
            if key not in v['viewports']:
                v['viewports'].append(key)
    return views


def group_passes(graph):
    actions = [a for a in graph.actions if a['kind'] not in ('Unmap', 'UpdateSubresource')]
    passes = []
    pending_clears = []
    current = None
    for a in actions:
        if a['kind'].startswith('Clear'):
            pending_clears.append(a)
            current = None
            continue
        key = action_outputs(a)
        if current is None or current['key'] != key or key[0] != 'draw':
            outputs = set()
            if key[0] == 'draw':
                outputs = {t['resource'] for t in a['render_targets']} | ({a['depth']['resource']} if a['depth'] else set())
            attached = [c for c in pending_clears if c['resource'] in outputs]
            for c in pending_clears:
                if c not in attached:
                    passes.append({'key': ('clear', c['event_id']), 'events': [c['event_id']], 'clears': [], 'kind': 'clear'})
            pending_clears = []
            current = {'key': key, 'events': [], 'clears': [c['event_id'] for c in attached], 'kind': key[0]}
            passes.append(current)
        current['events'].append(a['event_id'])
    for c in pending_clears:
        passes.append({'key': ('clear', c['event_id']), 'events': [c['event_id']], 'clears': [], 'kind': 'clear'})
    by_event = {a['event_id']: a for a in graph.actions if a['event_id'] is not None}
    result = []
    for i, p in enumerate(passes):
        acts = [by_event[e] for e in p['events']]
        shader_names = set()
        for a in acts:
            for stage, s in a.get('shaders', {}).items():
                if stage in ('PS', 'CS'):
                    shader_names.update(s['debug_names'][-1:] if s['debug_names'] else ['<unnamed>'])
        first = acts[0]
        rt_names, depth_names, dst_names = set(), set(), set()
        targets = []
        if first['kind'] in ('Draw', 'DrawIndexed'):
            rt_names = {n for t in first['render_targets'] for n in t['debug_names'][-1:]}
            depth_names = set(first['depth']['debug_names'][-1:]) if first['depth'] else set()
            targets = [{'resource': t['resource'], 'debug_names': t['debug_names'], 'subresources': t['subresources']} for t in first['render_targets']]
            if first['depth']:
                targets.append({'resource': first['depth']['resource'], 'debug_names': first['depth']['debug_names'],
                                'subresources': first['depth']['subresources'], 'depth': True})
        elif first['kind'] in ('CopySubresourceRegion', 'CopyResource'):
            dst_names = set(graph.names(first['destination']))
            src_usage = graph.resources[first['source']]['descriptor'].get('Usage')
            targets = [{'resource': first['destination'], 'debug_names': graph.names(first['destination'])}]
        elif first['kind'] == 'Dispatch':
            targets = [{'resource': u['resource'], 'debug_names': u.get('debug_names', [])} for u in first['uavs'] if u.get('view')]
        elif first['kind'].startswith('Clear'):
            targets = [{'resource': first['resource'], 'debug_names': graph.names(first['resource'])}]
        evidence = {'kind': first['kind'], 'shader': shader_names, 'rt_names': rt_names,
                    'depth_names': depth_names, 'dst_names': dst_names,
                    'src_usage': src_usage if first['kind'] in ('CopySubresourceRegion', 'CopyResource') else None}
        label = 'clear_only' if p['kind'] == 'clear' else next((name for name, rule in LABEL_RULES if rule(evidence)), 'unclassified')
        vp = first.get('viewports', [None])[0] if first.get('viewports') else None
        views = []
        for a in acts:
            sig = camera_signature(a.get('camera_values'))
            if sig and sig not in views:
                views.append(sig)
        result.append({'camera_view_signatures': views,'pass_index': i, 'label': label, 'label_rule': 'debug-name evidence: ' + label,
                       'first_event': p['events'][0], 'last_event': p['events'][-1], 'events': p['events'],
                       'clear_events': p['clears'], 'action_count': len(p['events']),
                       'pixel_and_compute_shader_debug_names': sorted(shader_names), 'targets': targets,
                       'viewport': vp})
    return result


def segment_passes(passes, graph, boundary_resource=None):
    """Split the ordered pass list after each write into ``boundary_resource``."""
    segments, current = [], []
    by_event = {a['event_id']: a for a in graph.actions if a['event_id'] is not None}
    for p in passes:
        current.append(p['pass_index'])
        last = by_event[p['last_event']]
        writes_boundary = boundary_resource is not None and any(w['resource'] == boundary_resource for w in last.get('writes', []))
        if writes_boundary or last['kind'] == 'Present':
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


# -------------------------------------------------- head surface binding
def _index_slice(graph, a):
    ib = a['index_buffer']
    data = graph.buffer(ib['resource']).read(graph.blobs, ib['bytes'][0], ib['bytes'][1])
    if data is None:
        return None
    fmt = '<%dH' if ib['format'] == 57 else '<%dI'
    count = (ib['bytes'][1] - ib['bytes'][0]) // (2 if ib['format'] == 57 else 4)
    base = a['arguments'].get('BaseVertexLocation', 0)
    return [i + base for i in struct.unpack(fmt % count, data)]


ATTRIBUTE_SOURCES = {('TEXCOORD', 0): 'uv', ('TEXCOORD', 1): 'uv2', ('COLOR', 0): 'source_vertex_colors_rgba',
                     ('NORMAL', 0): 'normals', ('TANGENT', 0): 'tangents', ('POSITION', 0): 'vertices'}
FORMAT_UNPACK = {2: ('<4f', 4), 6: ('<3f', 3), 16: ('<2f', 2), 41: ('<f', 1), 28: ('<4B', 4), 10: ('<4e', 4), 34: ('<2e', 2)}


def _attribute_compare(graph, a, mesh, vertex_count):
    rows = []
    elements = a['input_layout']['elements']
    vbs = {v['slot']: v for v in a['vertex_buffers']}
    for e in elements:
        if e['consumed_by_vs'] is False:
            continue
        key = (e['SemanticName'].upper(), e['SemanticIndex'])
        field = ATTRIBUTE_SOURCES.get(key)
        vb = vbs.get(e['InputSlot'])
        row = {'semantic': f"{e['SemanticName']}{e['SemanticIndex']}", 'slot': e['InputSlot'], 'format': e['Format'],
               'source_field': field}
        if vb is None or not vb['resource']:
            row['result'] = 'unbound'
            rows.append(row)
            continue
        row['vertex_buffer'] = vb['resource']
        row['vertex_buffer_usage'] = vb.get('usage')
        if field is None or e['Format'] not in FORMAT_UNPACK:
            row['result'] = 'no_source_comparison_defined'
            rows.append(row)
            continue
        if vb.get('usage') == 2:
            row['result'] = 'dynamic_cpu_written_stream_not_compared_to_source_bind_geometry'
            rows.append(row)
            continue
        source = mesh['source'].get(field) if field != 'source_vertex_colors_rgba' else mesh.get(field)
        if not source:
            row['result'] = 'source_field_absent'
            rows.append(row)
            continue
        fmt, n = FORMAT_UNPACK[e['Format']]
        size = struct.calcsize(fmt)
        stride, offset = vb['stride'], vb['offset'] + e['AlignedByteOffset']
        need = offset + stride * (vertex_count - 1) + size
        data = graph.buffer(vb['resource']).read(graph.blobs, 0, need) if need <= vb['byte_width'] else None
        if data is None:
            row['result'] = 'bytes_unavailable'
            rows.append(row)
            continue
        mismatches = 0
        maxdiff = 0.0
        for i in range(vertex_count):
            vals = struct.unpack_from(fmt, data, offset + i * stride)
            src = source[i]
            if e['Format'] == 28:
                expect = [round(float(c) * 255) for c in src]
                if list(vals) != expect:
                    mismatches += 1
                continue
            for x, y in zip(vals, src[:n]):
                d = abs(x - struct.unpack('<f', struct.pack('<f', y))[0])
                maxdiff = max(maxdiff, d)
                if d != 0:
                    mismatches += 1
                    break
        row.update(result='exact' if mismatches == 0 else 'mismatch', mismatching_vertices=mismatches, max_abs_difference=maxdiff)
        rows.append(row)
    return rows


def bind_surfaces(graph, geometry, passes, segments):
    meshes = geometry['meshes']
    pass_of = {}
    for p in passes:
        for e in p['events']:
            pass_of[e] = p
    segment_of = {}
    for si, seg in enumerate(segments):
        for pi in seg:
            segment_of[pi] = si
    surfaces = []
    for mi, m in enumerate(meshes):
        surfaces.append({'surface_index': mi, 'mesh_name': m['mesh_name'], 'renderer_path': m['renderer_path'],
                         'active_in_hierarchy_in_source_export': m.get('active_in_hierarchy'),
                         'enabled_in_source_export': m.get('enabled'),
                         'vertex_count': m['vertex_count'], 'triangle_count': m['triangle_count'],
                         'submesh_index_counts': [len(s['indices']) for s in m['source']['submeshes']],
                         'draws': [], 'buffers': {}})
    submesh_lists = [[(si, s['indices']) for si, s in enumerate(m['source']['submeshes'])] for m in meshes]
    unmatched_named = []
    cache = {}
    head_names = {m['mesh_name'] for m in meshes}
    for a in graph.actions:
        if not a['kind'].startswith('DrawIndexed'):
            continue
        ib = a['index_buffer']
        key = (ib['resource'], tuple(ib['bytes']), a['arguments'].get('BaseVertexLocation', 0))
        if key not in cache:
            cache[key] = _index_slice(graph, a)
        indices = cache[key]
        matches = [(mi, si) for mi, lists in enumerate(submesh_lists) for si, lst in lists if indices is not None and lst == indices]
        named = any(n.split('(')[0] in head_names for n in ib['debug_names'])
        if not matches:
            if named:
                unmatched_named.append({'event_id': a['event_id'], 'index_buffer': ib['resource'], 'debug_names': ib['debug_names']})
            continue
        p = pass_of.get(a['event_id'])
        vertex_count = max(indices) + 1
        for mi, si in matches:
            mesh = meshes[mi]
            row = {'event_id': a['event_id'], 'chunk_index': a['chunk_index'], 'submesh': si,
                   'pass_index': p and p['pass_index'], 'pass_label': p and p['label'],
                   'segment': p and segment_of.get(p['pass_index']),
                   'vs': a['shaders'].get('VS', {}).get('debug_names'), 'ps': a['shaders'].get('PS', {}).get('debug_names'),
                   'vs_dxbc_sha256': a['shaders'].get('VS', {}).get('dxbc_sha256'),
                   'ps_dxbc_sha256': a['shaders'].get('PS', {}).get('dxbc_sha256'),
                   'ps_source_variants': len(a['shaders'].get('PS', {}).get('source_matches', [])),
                   'index_buffer': {'resource': ib['resource'], 'debug_names': ib['debug_names'], 'bytes': ib['bytes'],
                                    'binding': 'exact index list equality with source submesh (BaseVertex applied)'},
                   'ambiguous_with_surfaces': [x for x, _ in matches if x != mi],
                   'vertex_buffers': [{'slot': v['slot'], 'resource': v['resource'], 'stride': v['stride'],
                                       'semantics': v['semantics'], 'debug_names': v.get('debug_names'),
                                       'usage': v.get('usage'), 'byte_width': v.get('byte_width'),
                                       'byte_width_equals_vertex_count_times_stride': v.get('byte_width') == mesh['vertex_count'] * v['stride'] if v['resource'] else None,
                                       'producers': v.get('producers')} for v in a['vertex_buffers']],
                   'render_targets': [t['resource'] for t in a['render_targets']],
                   'depth': a['depth'] and a['depth']['resource'],
                   'depth_write': a['depth'] and a['depth']['depth_write']}
            if vertex_count != mesh['vertex_count']:
                row['vertex_count_note'] = 'max index + 1 = %d differs from source vertex count' % vertex_count
            surfaces[mi]['draws'].append(row)
            for v in a['vertex_buffers']:
                if v['resource']:
                    surfaces[mi]['buffers'].setdefault(str(v['resource']), {'slot': v['slot'], 'semantics': v['semantics'],
                                                                            'debug_names': v.get('debug_names'), 'usage': v.get('usage')})
            if 'attribute_comparison' not in surfaces[mi]:
                surfaces[mi]['attribute_comparison'] = {'event_id': a['event_id'], 'rows': _attribute_compare(graph, a, mesh, mesh['vertex_count'])}
    for s in surfaces:
        s['draw_count'] = len(s['draws'])
        s['passes'] = sorted({d['pass_label'] for d in s['draws'] if d['pass_label']})
        provenance = {}
        for d in s['draws']:
            for v in d['vertex_buffers']:
                if not v['resource']:
                    continue
                key = '%s|%s' % (v['resource'], ','.join(v['semantics']))
                row = provenance.setdefault(key, {'resource': v['resource'], 'semantics': v['semantics'],
                                                  'debug_names': v['debug_names'], 'usage': v['usage'],
                                                  'producers_by_segment': {}})
                prods = sorted({str(p['producer']) + ':' + p['kind'] for p in (v['producers'] or [])})
                seg = str(d['segment'])
                if prods not in row['producers_by_segment'].setdefault(seg, []):
                    row['producers_by_segment'][seg].append(prods)
        s['vertex_stream_provenance'] = list(provenance.values())
    return {'surfaces': surfaces, 'named_head_index_buffers_without_source_match': unmatched_named,
            'binding_rule': 'A draw binds to a source surface only by exact equality of its executed index list with the '
                            'source submesh index list (BaseVertexLocation applied). Debug names, appearance and counts '
                            'alone are not used.'}


def buffer_writers(graph, resources):
    """Every in-frame write and the capture-start representation for listed buffers."""
    out = {}
    for ident in resources:
        r = graph.resources.get(ident)
        if r is None or r['kind'] != 'CreateBuffer':
            continue
        writes = [a for a in graph.actions if (a['kind'] in ('Unmap', 'UpdateSubresource') and a.get('resource') == ident) or
                  any(w.get('resource') == ident for w in a.get('writes', []) if a['kind'] in ('CopyResource', 'CopySubresourceRegion', 'Dispatch'))]
        out[str(ident)] = {'debug_names': r['debug_names'], 'descriptor': r['descriptor'], 'creation_chunk': r['chunk'],
                           'capture_start_representation': 'CreateBuffer InitialData blob %s (pinned RenderDoc 1.46 '
                                                           'Serialise_InitialState copies capture-start bytes into the '
                                                           'buffer record)' % (r['initial_data'] and r['initial_data']['buffer_index'])
                           if r.get('initial_data') else 'none',
                           'in_frame_writes': [{'event_id': a['event_id'], 'chunk_index': a['chunk_index'], 'kind': a['kind'],
                                                'map_type': a.get('map_type'), 'map_event': a.get('map_event'),
                                                'written_bytes': a.get('written_bytes'), 'blob': a.get('blob')} for a in writes]}
    return out


# ------------------------------------------------------------- edges
def _producer_edges(consumer, resource, via, producers):
    """Texture producers are per subresource (base + later partial writers);
    buffer producers are byte intervals."""
    out = []
    for p in producers or []:
        if 'base' in p:
            writers = p['partial_writers_since_base']
            out.append({'consumer': consumer, 'producer': p['base'], 'resource': resource, 'via': via,
                        'subresource': p['subresource'], 'latest_writer': not writers,
                        'relation': 'last_full_overwrite' if p['base'] != 'capture_start' else 'capture_start'})
            for i, w in enumerate(writers):
                out.append({'consumer': consumer, 'producer': w, 'resource': resource, 'via': via,
                            'subresource': p['subresource'], 'latest_writer': i == len(writers) - 1,
                            'relation': 'partial_writer_since_base'})
        else:
            out.append({'consumer': consumer, 'producer': p['producer'], 'resource': resource, 'via': via,
                        'bytes': p['bytes'], 'relation': p['kind']})
    return out


def resource_edges(graph):
    """Every producer->consumer edge of every recorded read, in action order."""
    edges = []
    for a in graph.actions:
        ev = a['event_id']
        for v in a.get('vertex_buffers', []):
            if v['resource']:
                edges += _producer_edges(ev, v['resource'], 'vertex_buffer_slot_%d' % v['slot'], v.get('producers'))
        if a.get('index_buffer'):
            edges += _producer_edges(ev, a['index_buffer']['resource'], 'index_buffer', a['index_buffer']['producers'])
        for stage, info in a.get('shaders', {}).items():
            for cb in info['constant_buffers']:
                if not cb.get('unbound'):
                    edges += _producer_edges(ev, cb['resource'], '%s_cb%d' % (stage, cb['slot']), cb['producers'])
            for srv in info['srvs']:
                if not srv.get('unbound'):
                    edges += _producer_edges(ev, srv['resource'], '%s_t%d' % (stage, srv['slot']), srv['producers'])
        for u in a.get('uavs', []):
            if u.get('view'):
                edges += _producer_edges(ev, u['resource'], 'cs_u%d' % u['slot'], u['producers'])
        for t in a.get('render_targets', []):
            edges += _producer_edges(ev, t['resource'], 'render_target_prior_content', t['prior_content'])
        if a.get('depth'):
            edges += _producer_edges(ev, a['depth']['resource'], 'depth_stencil_prior_content', a['depth']['prior_content'])
        for r in a.get('reads', []):
            edges += _producer_edges(ev, r['resource'], r['via'], r.get('producers'))
    return edges


def resource_timelines(graph):
    timeline = {}
    for a in graph.actions:
        ev = a['event_id']

        def add(resource, op, via):
            timeline.setdefault(str(resource), []).append([ev, op, via])
        for v in a.get('vertex_buffers', []):
            if v['resource']:
                add(v['resource'], 'read', 'vb%d' % v['slot'])
        if a.get('index_buffer'):
            add(a['index_buffer']['resource'], 'read', 'ib')
        for stage, info in a.get('shaders', {}).items():
            for cb in info['constant_buffers']:
                if not cb.get('unbound'):
                    add(cb['resource'], 'read', '%s_cb%d' % (stage, cb['slot']))
            for srv in info['srvs']:
                if not srv.get('unbound'):
                    add(srv['resource'], 'read', '%s_t%d' % (stage, srv['slot']))
        for r in a.get('reads', []):
            add(r['resource'], 'read', r['via'])
        for w in a.get('writes', []):
            add(w['resource'], 'write', w.get('via', a['kind']))
        if a['kind'] in ('Unmap', 'UpdateSubresource'):
            add(a['resource'], 'write', a['kind'])
    return timeline


def temporal_dependencies(graph, passes, segments, edges):
    """Motion-vector inputs, cross-segment edges and definite capture-start
    reads of GPU-writable resources (the previous-frame dependencies)."""
    seg_of, label_of = {}, {}
    for si, seg in enumerate(segments):
        for pi in seg:
            for e in passes[pi]['events']:
                seg_of[e] = si
                label_of[e] = passes[pi]['label']
    by_event = {a['event_id']: a for a in graph.actions if a['event_id'] is not None}
    mv_inputs, mv_targets = [], set()
    for p in passes:
        if p['label'] != 'motion_vectors':
            continue
        for e in p['events']:
            a = by_event[e]
            mv_targets.update(t['resource'] for t in a['render_targets'])
            prev = [v for v in a['vertex_buffers'] if any(x.startswith('TEXCOORD4') for x in v['semantics'])]
            mv_inputs.append({'event_id': e, 'segment': seg_of.get(e),
                              'index_buffer': a.get('index_buffer', {}).get('resource'),
                              'current_position_stream': [v['resource'] for v in a['vertex_buffers'] if 'POSITION0' in v['semantics']],
                              'previous_position_stream': [{'resource': v['resource'], 'debug_names': v.get('debug_names'),
                                                            'producers': v.get('producers')} for v in prev]})
    mv_consumers = sorted({(ed['consumer'], ed['resource'], ed['via']) for ed in edges
                           if ed['resource'] in mv_targets and not ed['via'].endswith('prior_content')})
    cross = {}
    for ed in edges:
        c, p = ed['consumer'], ed['producer']
        if not isinstance(p, int) or c not in seg_of or p not in seg_of or seg_of[p] >= seg_of[c]:
            continue
        if ed.get('latest_writer') is False:
            continue
        if ed['via'].endswith('prior_content'):
            continue
        key = (ed['resource'], ed['via'], label_of.get(c), label_of.get(p))
        row = cross.setdefault(key, {'resource': ed['resource'], 'debug_names': graph.names(ed['resource']), 'via': ed['via'],
                                     'consumer_pass': label_of.get(c), 'producer_pass': label_of.get(p), 'examples': []})
        if len(row['examples']) < 4:
            row['examples'].append({'consumer': c, 'consumer_segment': seg_of[c], 'producer': p,
                                    'producer_segment': seg_of[p], 'relation': ed['relation']})
    writable = {i for i, r in graph.resources.items() if r['kind'] in TEXTURE_KINDS + ('CreateBuffer',)
                and (r['descriptor'].get('BindFlags', 0) & 0xE0 or r['descriptor'].get('Usage') == 2)}
    definite = {}
    for a in graph.actions:
        for stage, info in a.get('shaders', {}).items():
            for srv in info['srvs']:
                if srv.get('unbound') or srv['resource'] not in writable or 'subresources' not in srv:
                    continue
                if all(pr['base'] == 'capture_start' and not pr['partial_writers_since_base'] for pr in srv['producers']):
                    row = definite.setdefault(srv['resource'], {'resource': srv['resource'], 'debug_names': graph.names(srv['resource']),
                                                                'capture_start': graph.texture_initial(srv['resource']),
                                                                'segments': set(), 'reads': 0, 'first_event': a['event_id']})
                    row['segments'].add(seg_of.get(a['event_id']))
                    row['reads'] += 1
    for r in definite.values():
        r['segments'] = sorted(x for x in r['segments'] if x is not None)
    return {'motion_vector_draws': mv_inputs, 'motion_vector_targets': sorted(mv_targets),
            'motion_vector_target_consumers': [{'event_id': c, 'resource': r, 'via': v, 'pass': label_of.get(c),
                                                'segment': seg_of.get(c)} for c, r, v in mv_consumers],
            'cross_segment_reads': list(cross.values()),
            'definite_capture_start_reads_of_gpu_writable_textures': list(definite.values()),
            'rule': 'cross-segment: a read whose most recent recorded writer is an action of an earlier segment '
                    '(nothing in the reading segment wrote it first); definite '
                    'capture-start: no in-frame writer before the read. Texture writes by draws are partial '
                    '(coverage needs execution), so other capture-start bases remain potential dependencies.'}


# ------------------------------------------------------------------- driver
def jsonable(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj).__name__)


def run(xml, zip_path, inspect, manifests, geometry_path, out, boundary_resource=None, final_event=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    chunks = load_chunks(xml)
    listing = load_driver_listing(inspect) if inspect else None
    graph = FrameGraph(chunks, listing, Blobs(zip_path), ShaderLibrary(manifests)).build()
    validation = graph.validate_events()
    passes = group_passes(graph)
    segments = segment_passes(passes, graph, boundary_resource)
    geometry = json.loads(Path(geometry_path).read_text(encoding='utf-8')) if geometry_path else None
    binding = bind_surfaces(graph, geometry, passes, segments) if geometry else None
    out.mkdir(parents=True)
    resources = {}
    for ident, r in graph.resources.items():
        row = dict(r)
        if r['kind'] in TEXTURE_KINDS:
            row['capture_start'] = graph.texture_initial(ident)
        resources[str(ident)] = row
    head_buffers = set()
    if binding:
        for s in binding['surfaces']:
            for d in s['draws']:
                for v in d['vertex_buffers']:
                    if v['resource']:
                        head_buffers.add(v['resource'])
                head_buffers.add(d['index_buffer']['resource'])
    provenance = {'xml': str(Path(xml).resolve()), 'xml_sha256': digest(Path(xml).read_bytes()),
                  'zip': str(Path(zip_path).resolve()) if zip_path else None,
                  'inspect_stdout': str(Path(inspect).resolve()) if inspect else None,
                  'inspect_sha256': digest(Path(inspect).read_bytes()) if inspect else None,
                  'driver_opened': graph.opened, 'shader_sources': graph.library.sources,
                  'geometry': str(Path(geometry_path).resolve()) if geometry_path else None,
                  'tool_sha256': digest(Path(__file__).read_bytes())}
    summary = {'schema_version': SCHEMA_VERSION, 'provenance': provenance, 'event_validation': validation,
               'frame_metadata': getattr(graph, 'frame_metadata', None),
               'counts': {'actions': len(graph.actions), 'passes': len(passes), 'segments': len(segments),
                          'resources': len(graph.resources)},
               'open_maps_at_end_of_capture': [{'resource': k[0], 'subresource': k[1], **v} for k, v in graph.open_maps.items()],
               'queries': graph.pending_queries, 'notes': graph.notes,
               'camera_views': camera_views(graph, passes),
               'cb_resource_names': graph.cb_resource_names,
               'segments': [{'segment': i, 'passes': seg, 'first_event': passes[seg[0]]['first_event'],
                             'last_event': passes[seg[-1]]['last_event'],
                             'labels': [passes[p]['label'] for p in seg]} for i, seg in enumerate(segments)]}
    (out / 'passes.json').write_text(json.dumps(passes, indent=1, default=jsonable), encoding='utf-8')
    (out / 'actions.json').write_text(json.dumps(graph.actions, default=jsonable), encoding='utf-8')
    timelines = resource_timelines(graph)
    for ident, row in resources.items():
        row['timeline'] = timelines.get(ident, [])
    (out / 'resources.json').write_text(json.dumps(resources, default=jsonable), encoding='utf-8')
    edges = resource_edges(graph)
    summary['counts']['edges'] = len(edges)
    summary['temporal_dependencies'] = temporal_dependencies(graph, passes, segments, edges)
    (out / 'edges.json').write_text(json.dumps(edges, default=jsonable), encoding='utf-8')
    if binding:
        binding['buffer_writers'] = buffer_writers(graph, sorted(head_buffers))
        (out / 'head_surfaces.json').write_text(json.dumps(binding, indent=1, default=jsonable), encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=jsonable), encoding='utf-8')
    return graph, passes, segments, binding, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', type=Path, required=True, help='official renderdoc zip.xml export')
    ap.add_argument('--zip', type=Path, help='companion ZIP (default: xml path without .xml)')
    ap.add_argument('--inspect', type=Path, help='renderdoc_replay_msvc inspect stdout (JSON lines)')
    ap.add_argument('--shader-manifest', type=Path, action='append', default=[])
    ap.add_argument('--geometry', type=Path, help='source surface geometry export for head binding')
    ap.add_argument('--segment-boundary-resource', type=int)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    zip_path = args.zip or args.xml.with_suffix('')
    graph, passes, segments, binding, summary = run(args.xml, zip_path, args.inspect, args.shader_manifest,
                                                    args.geometry, args.out, args.segment_boundary_resource)
    print(json.dumps({'actions': summary['counts'], 'event_validation': {k: (v if not isinstance(v, list) else len(v))
                                                                          for k, v in summary['event_validation'].items()}}))


if __name__ == '__main__':
    main()
