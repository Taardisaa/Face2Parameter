"""HS2 offline renderer frame program: schema, builder, validator, blob store.

A frame program is the executor input defined by the architecture contract
(docs/hs2_offline_renderer_architecture.md, "Frame program"). It is a directory:

    program.json   ordered, fully explicit D3D11 command stream (this schema)
    blobs/<sha256> content-addressed exact bytes (shaders, initial data, uploads)

Nothing is decoded or converted implicitly: descriptors are verbatim D3D11 struct
fields (RenderDoc XML / d3d11.h member names), blobs are the exact bytes handed to
D3D11, and every draw/dispatch carries its complete binding snapshot (absent
slots are explicitly unbound; the executor starts every action from ClearState).

Usage (generators)::

    from src.render.frame_program import ProgramBuilder, DrawState
    b = ProgramBuilder('outputs/.../my_program', provenance={'generator': 'unity_forward/0.1'})
    vb = b.buffer({'ByteWidth': len(vbytes), 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0,
                   'MiscFlags': 0, 'StructureByteStride': 0}, initial=vbytes, name='quad vb')
    rt = b.texture('tex2d', {'Width': 64, 'Height': 64, 'MipLevels': 1, 'ArraySize': 1, 'Format': 28,
                             'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 40,
                             'CPUAccessFlags': 0, 'MiscFlags': 0}, name='target')
    rtv = b.view('rtv', rt, {'Format': 28, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}})
    vs = b.shader('vs', vs_dxbc, source={'shader': 'Hidden/X', 'pass': 0, 'keywords': []})
    ps = b.shader('ps', ps_dxbc)
    il = b.input_layout([{'SemanticName': 'POSITION', 'SemanticIndex': 0, 'Format': 6, 'InputSlot': 0,
                          'AlignedByteOffset': 0, 'InputSlotClass': 0, 'InstanceDataStepRate': 0}], vs_dxbc)
    b.clear_rtv(rtv, [0.0, 0.0, 0.0, 1.0])
    s = DrawState()
    s.ia.input_layout = il; s.ia.topology = 4; s.ia.vertex_buffers = [{'slot': 0, 'buffer': vb, 'stride': 12, 'offset': 0}]
    s.vs.shader = vs; s.ps.shader = ps
    s.vs.constant_buffers = [{'slot': 0, 'blob': b.blob(cb_bytes)}]      # or {'slot','buffer','first_constant','num_constants'}
    s.om.render_targets = [{'slot': 0, 'view': rtv}]
    s.rs.viewports = [{'TopLeftX': 0.0, 'TopLeftY': 0.0, 'Width': 64.0, 'Height': 64.0, 'MinDepth': 0.0, 'MaxDepth': 1.0}]
    b.draw(s, VertexCount=3, StartVertexLocation=0)
    b.readback('target', rt, subresource=0, orientation='unity_texture_rows_bottom_up')
    b.write()                        # validates, then writes program.json (blobs are written on put)

Run ``tools/offline_renderer/frame_executor.exe PROGRAM_DIR OUT_DIR [--warp]`` to execute.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from math import ceil
from pathlib import Path

SCHEMA = 'hs2_frame_program/1'
RESOURCE_KINDS = ('buffer', 'tex1d', 'tex2d', 'tex3d', 'texcube')
VIEW_KINDS = ('srv', 'rtv', 'dsv', 'uav')
STAGES = ('vs', 'hs', 'ds', 'gs', 'ps', 'cs')
GRAPHICS_STAGES = ('vs', 'hs', 'ds', 'gs', 'ps')
ORIENTATIONS = ('unity_texture_rows_bottom_up', 'd3d_rows_top_down')
DRAW_OPS = {'draw': ('VertexCount', 'StartVertexLocation'),
            'draw_indexed': ('IndexCount', 'StartIndexLocation', 'BaseVertexLocation'),
            'draw_instanced': ('VertexCountPerInstance', 'InstanceCount', 'StartVertexLocation', 'StartInstanceLocation'),
            'draw_indexed_instanced': ('IndexCountPerInstance', 'InstanceCount', 'StartIndexLocation',
                                       'BaseVertexLocation', 'StartInstanceLocation')}
DRAW_OPS_API = {'Draw': 'draw', 'DrawIndexed': 'draw_indexed', 'DrawInstanced': 'draw_instanced',
                'DrawIndexedInstanced': 'draw_indexed_instanced'}
DISPATCH_ARGS = ('ThreadGroupCountX', 'ThreadGroupCountY', 'ThreadGroupCountZ')
OPS = ('clear_rtv', 'clear_dsv', 'clear_uav_float', 'clear_uav_uint', 'update_subresource', 'map_write',
       'copy_resource', 'copy_subresource_region', 'resolve', 'generate_mips', *DRAW_OPS, 'dispatch')
BUFFER_DESC = ('ByteWidth', 'Usage', 'BindFlags', 'CPUAccessFlags', 'MiscFlags', 'StructureByteStride')
TEX_DESC = {'tex1d': ('Width', 'MipLevels', 'ArraySize', 'Format', 'Usage', 'BindFlags', 'CPUAccessFlags', 'MiscFlags'),
            'tex2d': ('Width', 'Height', 'MipLevels', 'ArraySize', 'Format', 'SampleDesc', 'Usage', 'BindFlags',
                      'CPUAccessFlags', 'MiscFlags'),
            'tex3d': ('Width', 'Height', 'Depth', 'MipLevels', 'Format', 'Usage', 'BindFlags', 'CPUAccessFlags', 'MiscFlags')}
TEX_DESC['texcube'] = TEX_DESC['tex2d']
MISC_TEXTURECUBE = 4
BIND_DEPTH_STENCIL = 64
USAGE_IMMUTABLE, USAGE_DYNAMIC, USAGE_STAGING = 1, 2, 3
MAP_WRITE, MAP_WRITE_DISCARD, MAP_WRITE_NO_OVERWRITE = 2, 4, 5
CB_SLOTS, SRV_SLOTS, SAMPLER_SLOTS, UAV_SLOTS, RT_SLOTS, VB_SLOTS, VIEWPORTS = 14, 128, 16, 64, 8, 32, 16
CB_FULL_RANGE = 4096  # D3D11 legacy *SetConstantBuffers: first_constant 0, num_constants 4096
COMMAND_KEYS = {
    'clear_rtv': {'view', 'color'}, 'clear_dsv': {'view', 'flags', 'depth', 'stencil'},
    'clear_uav_float': {'view', 'values'}, 'clear_uav_uint': {'view', 'values'},
    'update_subresource': {'resource', 'subresource', 'box', 'blob', 'row_pitch', 'depth_pitch', 'copy_flags'},
    'map_write': {'resource', 'subresource', 'map_type', 'map_flags', 'start', 'end', 'blob'},
    'copy_resource': {'dst', 'src'},
    'copy_subresource_region': {'dst', 'dst_subresource', 'dst_x', 'dst_y', 'dst_z', 'src', 'src_subresource', 'src_box',
                                'copy_flags'},
    'resolve': {'dst', 'dst_subresource', 'src', 'src_subresource', 'format'}, 'generate_mips': {'view'},
    **{op: {'args', 'state'} for op in (*DRAW_OPS, 'dispatch')}}
ENTRY_KEYS = {'resources': ({'id', 'kind', 'desc', 'initial'}, {'name', 'provenance'}),
              'views': ({'id', 'kind', 'resource', 'desc'}, {'name', 'provenance'}),
              'shaders': ({'id', 'stage', 'blob'}, {'source', 'provenance'}),
              'input_layouts': ({'id', 'elements', 'signature_blob'}, {'provenance'}),
              'blend_states': ({'id', 'desc'}, {'provenance'}), 'depth_states': ({'id', 'desc'}, {'provenance'}),
              'raster_states': ({'id', 'desc'}, {'provenance'}), 'samplers': ({'id', 'desc'}, {'provenance'})}


class ProgramInvalid(ValueError):
    """The program violates the contract. Never silently repaired."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- DXGI storage
def block_layout(fmt):
    """(block width, block height, bytes per block) for DXGI storage; unknown formats fail."""
    if 70 <= fmt <= 84 or 94 <= fmt <= 99:  # BC1..BC5, BC6H, BC7 (typeless/unorm/srgb/...)
        return 4, 4, 8 if fmt in (70, 71, 72, 79, 80, 81) else 16
    if fmt == 66:  # R1_UNORM: 8 texels per byte
        return 8, 1, 1
    if fmt in (68, 69):  # R8G8_B8G8_UNORM, G8R8_G8B8_UNORM: 2x1 texel blocks of 4 bytes
        return 2, 1, 4
    if 1 <= fmt <= 4:
        n = 16
    elif 5 <= fmt <= 8:
        n = 12
    elif 9 <= fmt <= 22:
        n = 8
    elif 23 <= fmt <= 47 or fmt == 67 or 87 <= fmt <= 93:
        n = 4
    elif 48 <= fmt <= 59 or fmt in (85, 86, 115):
        n = 2
    elif 60 <= fmt <= 65:
        n = 1
    else:
        raise ProgramInvalid('DXGI storage format not implemented: %r' % (fmt,))
    return 1, 1, n


def mip_count(kind, desc):
    if kind == 'buffer':
        return 1
    if desc['MipLevels']:
        return desc['MipLevels']
    return max(desc['Width'], desc.get('Height', 1), desc.get('Depth', 1)).bit_length()


def array_count(kind, desc):
    return 1 if kind in ('buffer', 'tex3d') else desc['ArraySize']


def subresource_layouts(kind, desc):
    """Tightly packed layout of every subresource (D3D11 order: mip + array * mips)."""
    if kind == 'buffer':
        return [{'subresource': 0, 'mip': 0, 'array': 0, 'width': desc['ByteWidth'], 'height': 1, 'depth': 1,
                 'rows': 1, 'row_bytes': desc['ByteWidth'], 'slice_bytes': desc['ByteWidth'], 'bytes': desc['ByteWidth']}]
    mips = mip_count(kind, desc)
    bw, bh, size = block_layout(desc['Format'])
    out = []
    for a in range(array_count(kind, desc)):
        for m in range(mips):
            w = max(1, desc['Width'] >> m)
            h = max(1, desc.get('Height', 1) >> m)
            d = max(1, desc.get('Depth', 1) >> m)
            rows = ceil(h / bh)
            row_bytes = ceil(w / bw) * size
            out.append({'subresource': m + a * mips, 'mip': m, 'array': a, 'width': w, 'height': h, 'depth': d,
                        'rows': rows, 'row_bytes': row_bytes, 'slice_bytes': row_bytes * rows,
                        'bytes': row_bytes * rows * d})
    return out


# ------------------------------------------------------------------ blob store
class BlobStore:
    """Content-addressed exact bytes: blobs/<sha256>. Existing files are re-verified, never trusted.

    ``link_sources``: other program directories whose verified blobs/<sha256> files are hard-linked
    instead of written again (same bytes, no extra disk; each program stays self-contained).
    """

    def __init__(self, root, link_sources=()):
        self.root = Path(root)
        self.dir = self.root / 'blobs'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sizes = {}
        self.link_sources = [Path(s) / 'blobs' for s in link_sources]
        self.linked = 0

    def put(self, data: bytes) -> str:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ProgramInvalid('Blob must be bytes')
        data = bytes(data)
        h = sha256(data)
        if h not in self.sizes:
            path = self.dir / h
            if path.exists():
                if sha256(path.read_bytes()) != h:
                    raise ProgramInvalid('Existing blob file content does not match its name: ' + h)
            elif not self._link(h, path):
                tmp = path.with_suffix('.tmp')
                tmp.write_bytes(data)
                os.replace(tmp, path)
            self.sizes[h] = len(data)
        return h

    def _link(self, h, path):
        for src in self.link_sources:
            cand = src / h
            if cand.is_file() and sha256(cand.read_bytes()) == h:
                os.link(cand, path)
                self.linked += 1
                return True
        return False

    def path(self, h):
        return self.dir / h

    def size(self, h):
        if h not in self.sizes:
            p = self.dir / h
            if not p.is_file():
                raise ProgramInvalid('Missing blob ' + h)
            self.sizes[h] = p.stat().st_size
        return self.sizes[h]

    def read(self, h):
        data = (self.dir / h).read_bytes()
        if sha256(data) != h:
            raise ProgramInvalid('Blob content hash mismatch ' + h)
        return data


# ------------------------------------------------------------ binding snapshot
@dataclass
class StageBinding:
    shader: int | None = None
    constant_buffers: list = field(default_factory=list)  # {'slot','buffer','first_constant','num_constants'} | {'slot','blob'}
    srvs: list = field(default_factory=list)              # {'slot','view'}
    samplers: list = field(default_factory=list)          # {'slot','sampler'}


@dataclass
class ComputeBinding(StageBinding):
    uavs: list = field(default_factory=list)              # {'slot','view','initial_count'}


@dataclass
class InputAssembler:
    input_layout: int | None = None
    topology: int = 0
    vertex_buffers: list = field(default_factory=list)    # {'slot','buffer','stride','offset'}
    index_buffer: dict | None = None                       # {'buffer','format','offset'}


@dataclass
class OutputMerger:
    render_targets: list = field(default_factory=list)    # {'slot','view'}
    depth_stencil: int | None = None
    uavs: list = field(default_factory=list)              # OM UAVs: unsupported by executor v1 (must be empty)
    blend_state: int | None = None
    blend_factor: list = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
    sample_mask: int = 0xFFFFFFFF
    depth_stencil_state: int | None = None
    stencil_ref: int = 0


@dataclass
class Rasterizer:
    state: int | None = None
    viewports: list = field(default_factory=list)
    scissors: list = field(default_factory=list)


@dataclass
class DrawState:
    ia: InputAssembler = field(default_factory=InputAssembler)
    vs: StageBinding = field(default_factory=StageBinding)
    hs: StageBinding = field(default_factory=StageBinding)
    ds: StageBinding = field(default_factory=StageBinding)
    gs: StageBinding = field(default_factory=StageBinding)
    ps: StageBinding = field(default_factory=StageBinding)
    om: OutputMerger = field(default_factory=OutputMerger)
    rs: Rasterizer = field(default_factory=Rasterizer)
    predication: None = None
    stream_out: list = field(default_factory=list)


@dataclass
class DispatchState:
    cs: ComputeBinding = field(default_factory=ComputeBinding)


def _state_dict(state):
    return asdict(state) if not isinstance(state, dict) else copy.deepcopy(state)


# --------------------------------------------------------------------- builder
class ProgramBuilder:
    """Accumulates objects and ordered commands; ``write`` validates before writing program.json."""

    def __init__(self, root, provenance=None, *, first_id=1, blob_link_sources=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / 'program.json').exists():
            raise ProgramInvalid('Fresh program directory required (program.json exists)')
        self.blobs = BlobStore(self.root, blob_link_sources)
        self.program = {'schema': SCHEMA, 'provenance': dict(provenance or {}), 'resources': [], 'views': [],
                        'shaders': [], 'input_layouts': [], 'blend_states': [], 'depth_states': [],
                        'raster_states': [], 'samplers': [], 'commands': [], 'readbacks': []}
        self._next = first_id
        self._ids = set()

    def _id(self, ident):
        if ident is None:
            ident = self._next
        if not isinstance(ident, int) or isinstance(ident, bool) or ident <= 0:
            raise ProgramInvalid('Object ids must be positive integers')
        if ident in self._ids:
            raise ProgramInvalid('Duplicate object id %d' % ident)
        self._ids.add(ident)
        self._next = max(self._next, ident + 1)
        return ident

    def blob(self, data: bytes) -> str:
        return self.blobs.put(data)

    # objects
    def buffer(self, desc, initial=None, *, name='', ident=None):
        init = None if initial is None else [{'subresource': 0, 'blob': self.blob(initial),
                                              'row_pitch': desc['ByteWidth'], 'slice_pitch': desc['ByteWidth']}]
        return self.resource('buffer', desc, init, name=name, ident=ident)

    def texture(self, kind, desc, initial=None, *, name='', ident=None):
        """initial: None or list of (subresource, bytes, row_pitch, slice_pitch) covering every subresource."""
        init = None
        if initial is not None:
            init = [{'subresource': s, 'blob': self.blob(data), 'row_pitch': rp, 'slice_pitch': sp}
                    for s, data, rp, sp in initial]
        return self.resource(kind, desc, init, name=name, ident=ident)

    def resource(self, kind, desc, initial, *, name='', ident=None):
        ident = self._id(ident)
        self.program['resources'].append({'id': ident, 'kind': kind, 'name': name, 'desc': copy.deepcopy(desc),
                                          'initial': initial})
        return ident

    def view(self, kind, resource, desc, *, ident=None, name=''):
        ident = self._id(ident)
        self.program['views'].append({'id': ident, 'kind': kind, 'resource': resource,
                                      'desc': copy.deepcopy(desc), 'name': name})
        return ident

    def shader(self, stage, dxbc: bytes, source=None, *, ident=None):
        ident = self._id(ident)
        self.program['shaders'].append({'id': ident, 'stage': stage, 'blob': self.blob(dxbc), 'source': source or {}})
        return ident

    def input_layout(self, elements, signature_dxbc: bytes, *, ident=None):
        ident = self._id(ident)
        self.program['input_layouts'].append({'id': ident, 'elements': copy.deepcopy(elements),
                                              'signature_blob': self.blob(signature_dxbc)})
        return ident

    def _state_object(self, table, desc, ident):
        ident = self._id(ident)
        self.program[table].append({'id': ident, 'desc': copy.deepcopy(desc)})
        return ident

    def blend_state(self, desc, *, ident=None):
        return self._state_object('blend_states', desc, ident)

    def depth_state(self, desc, *, ident=None):
        return self._state_object('depth_states', desc, ident)

    def raster_state(self, desc, *, ident=None):
        return self._state_object('raster_states', desc, ident)

    def sampler(self, desc, *, ident=None):
        return self._state_object('samplers', desc, ident)

    # commands
    def command(self, op, provenance=None, **fields):
        cmd = {'op': op, **fields}
        if provenance:
            cmd['provenance'] = provenance
        self.program['commands'].append(cmd)
        return len(self.program['commands']) - 1

    def clear_rtv(self, view, color, **p):
        return self.command('clear_rtv', p or None, view=view, color=list(color))

    def clear_dsv(self, view, flags, depth, stencil, **p):
        return self.command('clear_dsv', p or None, view=view, flags=flags, depth=depth, stencil=stencil)

    def clear_uav_float(self, view, values, **p):
        return self.command('clear_uav_float', p or None, view=view, values=list(values))

    def clear_uav_uint(self, view, values, **p):
        return self.command('clear_uav_uint', p or None, view=view, values=list(values))

    def update_subresource(self, resource, subresource, data, row_pitch, depth_pitch, box=None, copy_flags=None, **p):
        return self.command('update_subresource', p or None, resource=resource, subresource=subresource, box=box,
                            blob=self.blob(data), row_pitch=row_pitch, depth_pitch=depth_pitch, copy_flags=copy_flags)

    def map_write(self, resource, subresource, map_type, map_flags, start, end, data, **p):
        return self.command('map_write', p or None, resource=resource, subresource=subresource, map_type=map_type,
                            map_flags=map_flags, start=start, end=end, blob=self.blob(data))

    def copy_resource(self, dst, src, **p):
        return self.command('copy_resource', p or None, dst=dst, src=src)

    def copy_subresource_region(self, dst, dst_subresource, dst_xyz, src, src_subresource, src_box=None,
                                copy_flags=None, **p):
        x, y, z = dst_xyz
        return self.command('copy_subresource_region', p or None, dst=dst, dst_subresource=dst_subresource,
                            dst_x=x, dst_y=y, dst_z=z, src=src, src_subresource=src_subresource, src_box=src_box,
                            copy_flags=copy_flags)

    def resolve(self, dst, dst_subresource, src, src_subresource, fmt, **p):
        return self.command('resolve', p or None, dst=dst, dst_subresource=dst_subresource, src=src,
                            src_subresource=src_subresource, format=fmt)

    def generate_mips(self, view, **p):
        return self.command('generate_mips', p or None, view=view)

    def _draw(self, op, state, provenance, args):
        if set(args) != set(DRAW_OPS[op]):
            raise ProgramInvalid('%s arguments must be exactly %s' % (op, DRAW_OPS[op]))
        return self.command(op, provenance, args=dict(args), state=_state_dict(state))

    def draw(self, state, provenance=None, **args):
        return self._draw('draw', state, provenance, args)

    def draw_indexed(self, state, provenance=None, **args):
        return self._draw('draw_indexed', state, provenance, args)

    def draw_instanced(self, state, provenance=None, **args):
        return self._draw('draw_instanced', state, provenance, args)

    def draw_indexed_instanced(self, state, provenance=None, **args):
        return self._draw('draw_indexed_instanced', state, provenance, args)

    def dispatch(self, state, provenance=None, **args):
        if set(args) != set(DISPATCH_ARGS):
            raise ProgramInvalid('dispatch arguments must be exactly %s' % (DISPATCH_ARGS,))
        return self.command('dispatch', provenance, args=dict(args), state=_state_dict(state))

    def readback(self, name, resource, *, subresource=0, after_command=None, orientation='d3d_rows_top_down',
                 byte_offset=None, byte_length=None, provenance=None):
        if after_command is None:
            after_command = len(self.program['commands']) - 1
        rb = {'name': name, 'after_command': after_command, 'resource': resource, 'subresource': subresource,
              'orientation': orientation}
        if byte_offset is not None or byte_length is not None:
            rb.update(byte_offset=byte_offset or 0, byte_length=byte_length or 0)
        if provenance:
            rb['provenance'] = provenance
        self.program['readbacks'].append(rb)
        return rb

    def write(self, *, indent=None):
        validate(self.program, self.blobs)
        path = self.root / 'program.json'
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.program, indent=indent, separators=(',', ':') if indent is None else None,
                                  allow_nan=False), encoding='utf-8')
        os.replace(tmp, path)
        return path


# ------------------------------------------------------------------- validator
def _int(v, what, lo=0, hi=(1 << 64) - 1):
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ProgramInvalid('%s must be an integer in [%d, %d], got %r' % (what, lo, hi, v))
    return v


def _num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProgramInvalid('%s must be a number, got %r' % (what, v))
    return v


def _view_dimension_ok(kind, view_kind, desc, resource):
    if desc is None:
        return
    dim = _int(desc.get('ViewDimension'), 'view ViewDimension')
    members = [k for k in desc if k not in ('Format', 'ViewDimension', 'Flags')]
    if len(members) != 1:
        raise ProgramInvalid('View desc must have exactly one union member, got %r' % members)
    allowed = {
        'srv': {1: 'Buffer', 2: 'Texture1D', 3: 'Texture1DArray', 4: 'Texture2D', 5: 'Texture2DArray', 6: 'Texture2DMS',
                7: 'Texture2DMSArray', 8: 'Texture3D', 9: 'TextureCube', 10: 'TextureCubeArray', 11: 'BufferEx'},
        'rtv': {1: 'Buffer', 2: 'Texture1D', 3: 'Texture1DArray', 4: 'Texture2D', 5: 'Texture2DArray', 6: 'Texture2DMS',
                7: 'Texture2DMSArray', 8: 'Texture3D'},
        'dsv': {1: 'Texture1D', 2: 'Texture1DArray', 3: 'Texture2D', 4: 'Texture2DArray', 5: 'Texture2DMS',
                6: 'Texture2DMSArray'},
        'uav': {1: 'Buffer', 2: 'Texture1D', 3: 'Texture1DArray', 4: 'Texture2D', 5: 'Texture2DArray', 8: 'Texture3D'}}[view_kind]
    if allowed.get(dim) != members[0]:
        raise ProgramInvalid('View dimension %d does not match union member %s' % (dim, members[0]))
    compatible = {'buffer': ('Buffer', 'BufferEx'), 'tex1d': ('Texture1D', 'Texture1DArray'),
                  'tex2d': ('Texture2D', 'Texture2DArray', 'Texture2DMS', 'Texture2DMSArray'),
                  'texcube': ('Texture2D', 'Texture2DArray', 'TextureCube', 'TextureCubeArray'),
                  'tex3d': ('Texture3D',)}[resource['kind']]
    if members[0] not in compatible:
        raise ProgramInvalid('View union %s does not match resource kind %s' % (members[0], resource['kind']))


def validate(program, blobs):
    """Structural + reference + byte-size validation. Raises ProgramInvalid on any violation."""
    if program.get('schema') != SCHEMA:
        raise ProgramInvalid('schema must be ' + SCHEMA)
    for key in ('provenance', 'resources', 'views', 'shaders', 'input_layouts', 'blend_states', 'depth_states',
                'raster_states', 'samplers', 'commands', 'readbacks'):
        if key not in program:
            raise ProgramInvalid('Missing top-level key ' + key)
    objects = {}

    def register(table, entry):
        need, optional = ENTRY_KEYS[table]
        keys = set(entry)
        if not need <= keys or keys - need - optional:
            raise ProgramInvalid('%s entry keys must be %s (+ optional %s), got %s' % (table, sorted(need), sorted(optional), sorted(keys)))
        ident = _int(entry.get('id'), table + ' id', 1)
        if ident in objects:
            raise ProgramInvalid('Duplicate object id %d' % ident)
        objects[ident] = (table, entry)

    def blob_size(h, what):
        if not isinstance(h, str) or len(h) != 64:
            raise ProgramInvalid(what + ' blob must be a sha256 hex string')
        return blobs.size(h)

    layouts = {}
    for r in program['resources']:
        register('resources', r)
        kind = r.get('kind')
        if kind not in RESOURCE_KINDS:
            raise ProgramInvalid('Unknown resource kind %r' % kind)
        d = r.get('desc') or {}
        fields = BUFFER_DESC if kind == 'buffer' else TEX_DESC[kind]
        if set(d) != set(fields):
            raise ProgramInvalid('Resource %d desc fields must be exactly %s, got %s' % (r['id'], fields, sorted(d)))
        if kind == 'texcube' and not d['MiscFlags'] & MISC_TEXTURECUBE:
            raise ProgramInvalid('texcube resource requires D3D11_RESOURCE_MISC_TEXTURECUBE')
        if kind == 'tex2d' and d['MiscFlags'] & MISC_TEXTURECUBE:
            raise ProgramInvalid('cube-flagged texture must use kind texcube')
        ls = subresource_layouts(kind, d)
        layouts[r['id']] = (kind, d, ls)
        init = r.get('initial')
        if init is None:
            if d['Usage'] == USAGE_IMMUTABLE:
                raise ProgramInvalid('IMMUTABLE resource %d requires initial data' % r['id'])
            continue
        if sorted(i['subresource'] for i in init) != [l['subresource'] for l in sorted(ls, key=lambda l: l['subresource'])]:
            raise ProgramInvalid('Resource %d initial data must cover every subresource exactly once' % r['id'])
        by_sub = {l['subresource']: l for l in ls}
        for i in init:
            l = by_sub[i['subresource']]
            n = blob_size(i['blob'], 'initial')
            rp = _int(i['row_pitch'], 'row_pitch')
            sp = _int(i['slice_pitch'], 'slice_pitch')
            if kind == 'buffer':
                if n != d['ByteWidth']:
                    raise ProgramInvalid('Buffer %d initial blob must be ByteWidth bytes' % r['id'])
                continue
            if rp < l['row_bytes'] or (l['depth'] > 1 and sp < rp * l['rows']):
                raise ProgramInvalid('Resource %d subresource %d pitches too small' % (r['id'], l['subresource']))
            need = (l['depth'] - 1) * sp + (l['rows'] - 1) * rp + l['row_bytes']
            limit = max(need, l['depth'] * sp if l['depth'] > 1 else l['rows'] * rp)
            if not need <= n <= limit:
                raise ProgramInvalid('Resource %d subresource %d initial blob size %d outside [%d, %d]'
                                     % (r['id'], l['subresource'], n, need, limit))
    for v in program['views']:
        register('views', v)
        if v.get('kind') not in VIEW_KINDS:
            raise ProgramInvalid('Unknown view kind %r' % v.get('kind'))
        base = objects.get(v.get('resource'))
        if not base or base[0] != 'resources':
            raise ProgramInvalid('View %d references no resource' % v['id'])
        _view_dimension_ok(base[1]['kind'], v['kind'], v.get('desc'), base[1])
    for s in program['shaders']:
        register('shaders', s)
        if s.get('stage') not in STAGES:
            raise ProgramInvalid('Unknown shader stage %r' % s.get('stage'))
        if blob_size(s['blob'], 'shader') < 32:
            raise ProgramInvalid('Shader blob too small to be DXBC')
    for il in program['input_layouts']:
        register('input_layouts', il)
        blob_size(il['signature_blob'], 'input layout signature')
        for e in il['elements']:
            for k in ('SemanticName', 'SemanticIndex', 'Format', 'InputSlot', 'AlignedByteOffset', 'InputSlotClass',
                      'InstanceDataStepRate'):
                if k not in e:
                    raise ProgramInvalid('Input element misses ' + k)
    for table in ('blend_states', 'depth_states', 'raster_states', 'samplers'):
        for s in program[table]:
            register(table, s)
            if not isinstance(s.get('desc'), dict):
                raise ProgramInvalid(table + ' desc required')

    def ref(ident, table, what, kinds=None):
        if ident is None:
            return None
        o = objects.get(ident)
        if o is None or o[0] != table or (kinds and o[1].get('kind', o[1].get('stage')) not in kinds):
            raise ProgramInvalid('%s: %r is not a %s%s' % (what, ident, table, '' if not kinds else ' of kind %s' % (kinds,)))
        return o[1]

    def slots(items, limit, what, key):
        seen = set()
        for it in items:
            s = _int(it.get('slot'), what + ' slot', 0, limit - 1)
            if s in seen:
                raise ProgramInvalid(what + ' slot %d bound twice' % s)
            seen.add(s)
            if it.get(key) is None and key != 'blob':
                raise ProgramInvalid(what + ' entries list bound objects only (absent slot = unbound)')

    def stage(b, name, expect_stage, compute=False):
        allowed = {'shader', 'constant_buffers', 'srvs', 'samplers'} | ({'uavs'} if compute else set())
        if set(b) - allowed:
            raise ProgramInvalid('%s binding has unknown keys %s' % (name, set(b) - allowed))
        ref(b.get('shader'), 'shaders', name + ' shader', (expect_stage,))
        cbs = b.get('constant_buffers', [])
        slots(cbs, CB_SLOTS, name + ' CB', 'slot')
        for cb in cbs:
            if 'blob' in cb:
                if set(cb) != {'slot', 'blob'}:
                    raise ProgramInvalid(name + ' blob CB must be {slot, blob}')
                n = blob_size(cb['blob'], 'CB')
                if n == 0 or n % 16 or n > 4096 * 16:
                    raise ProgramInvalid(name + ' CB blob size must be a positive multiple of 16 up to 65536')
            else:
                buf = ref(cb.get('buffer'), 'resources', name + ' CB buffer', ('buffer',))
                fc = _int(cb.get('first_constant'), 'first_constant')
                nc = _int(cb.get('num_constants'), 'num_constants', 1, 4096)
                if fc % 16 or nc % 16:
                    raise ProgramInvalid(name + ' CB range must be a multiple of 16 constants (D3D11.1)')
                if (fc, nc) != (0, CB_FULL_RANGE) and fc * 16 >= buf['desc']['ByteWidth']:
                    raise ProgramInvalid(name + ' CB range begins outside the buffer')
        slots(b.get('srvs', []), SRV_SLOTS, name + ' SRV', 'view')
        for it in b.get('srvs', []):
            ref(it['view'], 'views', name + ' SRV', ('srv',))
        slots(b.get('samplers', []), SAMPLER_SLOTS, name + ' sampler', 'sampler')
        for it in b.get('samplers', []):
            ref(it['sampler'], 'samplers', name + ' sampler')
        if compute:
            slots(b.get('uavs', []), UAV_SLOTS, name + ' UAV', 'view')
            for it in b.get('uavs', []):
                ref(it['view'], 'views', name + ' UAV', ('uav',))
                _int(it.get('initial_count'), 'UAV initial_count', 0, 0xFFFFFFFF)

    ncmd = len(program['commands'])
    for n, c in enumerate(program['commands']):
        op = c.get('op')
        where = 'command %d (%s)' % (n, op)
        if op not in OPS:
            raise ProgramInvalid('%s: unknown op' % where)
        if set(c) - {'op', 'provenance'} != COMMAND_KEYS[op]:
            raise ProgramInvalid('%s: fields must be exactly %s' % (where, sorted(COMMAND_KEYS[op])))
        if op == 'clear_rtv':
            ref(c['view'], 'views', where, ('rtv',))
            if len(c['color']) != 4:
                raise ProgramInvalid(where + ': color needs 4 floats')
            [_num(x, where) for x in c['color']]
        elif op == 'clear_dsv':
            ref(c['view'], 'views', where, ('dsv',))
            _int(c['flags'], where + ' flags', 1, 3)
            _num(c['depth'], where)
            _int(c['stencil'], where + ' stencil', 0, 255)
        elif op in ('clear_uav_float', 'clear_uav_uint'):
            ref(c['view'], 'views', where, ('uav',))
            if len(c['values']) != 4:
                raise ProgramInvalid(where + ': 4 values required')
            for x in c['values']:
                (_num(x, where) if op == 'clear_uav_float' else _int(x, where, 0, 0xFFFFFFFF))
        elif op == 'update_subresource':
            r = ref(c['resource'], 'resources', where)
            ls = layouts[r['id']][2]
            _int(c['subresource'], where + ' subresource', 0, len(ls) - 1)
            blob_size(c['blob'], where)
            _int(c['row_pitch'], where + ' row_pitch')
            _int(c['depth_pitch'], where + ' depth_pitch')
            if c['copy_flags'] is not None:
                _int(c['copy_flags'], where + ' copy_flags', 0, 3)
            if r['desc']['Usage'] in (USAGE_IMMUTABLE, USAGE_DYNAMIC) or (r['kind'] != 'buffer' and r['desc']['BindFlags'] & BIND_DEPTH_STENCIL):
                raise ProgramInvalid(where + ': UpdateSubresource is invalid on IMMUTABLE/DYNAMIC/depth-stencil resources')
        elif op == 'map_write':
            r = ref(c['resource'], 'resources', where)
            ls = layouts[r['id']][2]
            sub = _int(c['subresource'], where + ' subresource', 0, len(ls) - 1)
            mt = _int(c['map_type'], where + ' map_type', 2, 5)
            if mt == 3:
                raise ProgramInvalid(where + ': MAP_READ_WRITE writes are not implemented')
            if r['desc']['Usage'] not in (USAGE_DYNAMIC, USAGE_STAGING):
                raise ProgramInvalid(where + ': only DYNAMIC/STAGING resources can be mapped')
            _int(c['map_flags'], where + ' map_flags', 0, 0xFFFFFFFF)
            s0, s1 = _int(c['start'], where + ' start'), _int(c['end'], where + ' end')
            if not s0 < s1 or (r['kind'] == 'buffer' and s1 > ls[sub]['bytes']) or \
                    (r['kind'] != 'buffer' and s1 - s0 > ls[sub]['bytes']):
                raise ProgramInvalid(where + ': written range must be non-empty and inside the subresource')
            if blob_size(c['blob'], where) != s1 - s0:
                raise ProgramInvalid(where + ': blob length must equal end-start')
            if mt == MAP_WRITE_NO_OVERWRITE and r['kind'] != 'buffer':
                raise ProgramInvalid(where + ': NO_OVERWRITE texture writes are not implemented')
        elif op == 'copy_resource':
            ref(c['dst'], 'resources', where)
            ref(c['src'], 'resources', where)
        elif op == 'copy_subresource_region':
            for k in ('dst', 'src'):
                ref(c[k], 'resources', where)
            for k in ('dst_subresource', 'src_subresource', 'dst_x', 'dst_y', 'dst_z'):
                _int(c[k], where + ' ' + k)
            if c['copy_flags'] is not None:
                _int(c['copy_flags'], where + ' copy_flags', 0, 3)
        elif op == 'resolve':
            ref(c['dst'], 'resources', where)
            ref(c['src'], 'resources', where)
            _int(c['format'], where + ' format', 1, 200)
        elif op == 'generate_mips':
            ref(c['view'], 'views', where, ('srv',))
        elif op in DRAW_OPS:
            if set(c.get('args', {})) != set(DRAW_OPS[op]):
                raise ProgramInvalid(where + ': arguments must be exactly %s' % (DRAW_OPS[op],))
            st = c.get('state') or {}
            if set(st) != {'ia', 'vs', 'hs', 'ds', 'gs', 'ps', 'om', 'rs', 'predication', 'stream_out'}:
                raise ProgramInvalid(where + ': draw state must be the complete graphics snapshot')
            if st['predication'] is not None or st['stream_out']:
                raise ProgramInvalid(where + ': predication/stream-out are not implemented by the executor')
            ia = st['ia']
            ref(ia.get('input_layout'), 'input_layouts', where + ' input layout')
            _int(ia.get('topology'), where + ' topology', 0, 64)
            slots(ia.get('vertex_buffers', []), VB_SLOTS, where + ' VB', 'buffer')
            for vb in ia.get('vertex_buffers', []):
                ref(vb['buffer'], 'resources', where + ' VB', ('buffer',))
                _int(vb['stride'], where + ' stride')
                _int(vb['offset'], where + ' offset')
            if ia.get('index_buffer') is not None:
                ref(ia['index_buffer']['buffer'], 'resources', where + ' IB', ('buffer',))
                if ia['index_buffer']['format'] not in (42, 57):
                    raise ProgramInvalid(where + ': index format must be R32_UINT or R16_UINT')
            elif op.startswith('draw_indexed'):
                raise ProgramInvalid(where + ': indexed draw without index buffer')
            for s in GRAPHICS_STAGES:
                stage(st[s], where + ' ' + s, s)
            if st['vs'].get('shader') is None:
                raise ProgramInvalid(where + ': draw without vertex shader')
            om = st['om']
            slots(om.get('render_targets', []), RT_SLOTS, where + ' RTV', 'view')
            for rt in om.get('render_targets', []):
                ref(rt['view'], 'views', where + ' RTV', ('rtv',))
            ref(om.get('depth_stencil'), 'views', where + ' DSV', ('dsv',))
            if om.get('uavs'):
                raise ProgramInvalid(where + ': OM UAVs are not implemented by the executor')
            ref(om.get('blend_state'), 'blend_states', where + ' blend')
            ref(om.get('depth_stencil_state'), 'depth_states', where + ' depth state')
            if len(om['blend_factor']) != 4:
                raise ProgramInvalid(where + ': blend_factor needs 4 floats')
            _int(om['sample_mask'], where + ' sample_mask', 0, 0xFFFFFFFF)
            _int(om['stencil_ref'], where + ' stencil_ref', 0, 0xFFFFFFFF)
            rs = st['rs']
            ref(rs.get('state'), 'raster_states', where + ' raster')
            if len(rs['viewports']) > VIEWPORTS or len(rs['scissors']) > VIEWPORTS:
                raise ProgramInvalid(where + ': too many viewports/scissors')
            for vp in rs['viewports']:
                [_num(vp[k], where) for k in ('TopLeftX', 'TopLeftY', 'Width', 'Height', 'MinDepth', 'MaxDepth')]
            for sc in rs['scissors']:
                [_int(sc[k], where, -(1 << 31), (1 << 31) - 1) for k in ('left', 'top', 'right', 'bottom')]
        elif op == 'dispatch':
            if set(c.get('args', {})) != set(DISPATCH_ARGS):
                raise ProgramInvalid(where + ': arguments must be exactly %s' % (DISPATCH_ARGS,))
            st = c.get('state') or {}
            if set(st) != {'cs'}:
                raise ProgramInvalid(where + ': dispatch state must be {cs}')
            stage(st['cs'], where + ' cs', 'cs', compute=True)
            if st['cs'].get('shader') is None:
                raise ProgramInvalid(where + ': dispatch without compute shader')
    names = set()
    for rb in program['readbacks']:
        if not isinstance(rb.get('name'), str) or not rb['name'] or rb['name'] in names or \
                any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for ch in rb['name']):
            raise ProgramInvalid('Readback names must be unique [A-Za-z0-9_.-]+: %r' % rb.get('name'))
        names.add(rb['name'])
        _int(rb.get('after_command'), 'readback after_command', 0, max(0, ncmd - 1))
        if not ncmd:
            raise ProgramInvalid('Readback requires at least one command')
        r = ref(rb.get('resource'), 'resources', 'readback ' + rb['name'])
        ls = layouts[r['id']][2]
        _int(rb.get('subresource'), 'readback subresource', 0, len(ls) - 1)
        if rb.get('orientation') not in ORIENTATIONS:
            raise ProgramInvalid('readback orientation must be one of %s' % (ORIENTATIONS,))
        if r['kind'] != 'buffer' and r['desc'].get('SampleDesc', {}).get('Count', 1) > 1:
            raise ProgramInvalid('MSAA readback is not implemented (no implicit resolve)')
        if set(rb) - {'name', 'after_command', 'resource', 'subresource', 'orientation', 'provenance'} - (
                {'byte_offset', 'byte_length'} if 'byte_offset' in rb else set()):
            raise ProgramInvalid('readback %s has unknown fields' % rb['name'])
        if 'byte_offset' in rb:
            if r['kind'] != 'buffer':
                raise ProgramInvalid('byte ranges apply to buffer readbacks only')
            o = _int(rb['byte_offset'], 'byte_offset')
            n = _int(rb['byte_length'], 'byte_length')
            if o + (n or (r['desc']['ByteWidth'] - o)) > r['desc']['ByteWidth']:
                raise ProgramInvalid('buffer readback range outside buffer')
    return True


def load(root):
    """Load and validate an existing program directory."""
    root = Path(root)
    program = json.loads((root / 'program.json').read_text(encoding='utf-8'))
    blobs = BlobStore(root)
    validate(program, blobs)
    return program, blobs
