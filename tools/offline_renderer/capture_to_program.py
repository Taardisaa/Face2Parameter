"""Mechanical converter: official RenderDoc D3D11 zip.xml capture -> HS2 frame program.

CPU only. Produces a frame program (src/render/frame_program.py contract) that
covers the capture from Beginning of Capture up to and including an explicit
event, plus readbacks at explicit (event, resource) checkpoints.

Mechanisms are the pinned RenderDoc v1.46 replay source, not assumptions:

* Capture-start content (``d3d11_initstate.cpp`` + ``resource_manager.h``):
  buffers start from their CreateBuffer InitialData record (Serialise_InitialState
  copies the capture-start bytes into that record).  A texture listed in
  ``NeededInitials`` with an ``Internal::Initial Contents`` chunk starts from that
  chunk (row/depth pitches as serialized).  A listed texture without a chunk gets
  ``Create_InitialState(id, live, written)``: written=false with MipLevels==1 and
  RENDER_TARGET bind -> ClearRenderTargetView(0,0,0,0) through a view of
  ``GetTypedFormat``; with DEPTH_STENCIL bind -> ClearDepthStencilView(DEPTH|STENCIL,
  1.0, 0) through ``GetDepthTypedFormat``; otherwise (non-IMMUTABLE) a copy of
  the live resource as loaded (creation data + pre-capture updates).  Only the
  first NeededInitials record of a resource takes effect (later ones see existing
  initial contents).  Initial-content chunks of unlisted resources are freed.
* Creation data (``Serialise_CreateTextureData``): tightly packed per subresource,
  SysMemPitch = GetRowPitch(w, fmt, mip), SysMemSlicePitch = GetByteSize(w, h, 1, fmt, mip).
* Map/Unmap (``Serialise_Unmap``): an Unmap with an empty written range does
  nothing on replay; otherwise it becomes ``map_write`` with the exact recorded
  range and bytes (NO_OVERWRITE via temp-buffer copy, other types via a real Map
  and ``MapIntercept::CopyToD3D`` - implemented identically in the executor).
* UpdateSubresource (``Serialise_UpdateSubresource1``): IsUpdate=true replays the
  recorded box/pitches; IsUpdate=false replays the full contents with no box and
  pitches GetRowPitch(subWidth)/GetByteSize(subWidth, subHeight, 1).
* Binding state follows ``D3D11RenderState`` (``ChangeRefRead`` /
  ``ChangeRefWrite`` / ``ValidOutputMerger`` / ``ResourceRange``), which is how
  RenderDoc keeps its replay state consistent with the runtime's hazard rules:
  an SRV whose range is bound for write is forced to NULL, binding an output
  unbinds overlapping reads and other writes, invalid OM combinations are ignored.
  Every hazard resolution is recorded in the report.

Unsupported constructs (deferred contexts, predication, stream-out, OM UAVs,
class instances, non-event queries, MAP_READ_WRITE, MSAA initial contents, ...)
raise ``Unconvertible``.  No state is guessed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.render import frame_program as fp  # noqa: E402
import frame_graph as fg  # noqa: E402

CONVERTER_VERSION = 'capture_to_program/1'
STAGE_KEYS = ('VS', 'HS', 'DS', 'GS', 'PS', 'CS')
SYNTHETIC_BASE = 4_000_000_000  # ids of converter-made initial-state views (beyond capture ResourceIds)
ALL_MIP, ALL_SLICE = 0xF, 0x7FF
BIND_SRV, BIND_RT, BIND_DS, BIND_UAV, BIND_SO = 8, 32, 64, 128, 16
TYPED = {1: 2, 5: 6, 9: 10, 15: 16, 19: 19, 23: 24, 27: 28, 33: 34, 39: 41, 44: 46, 90: 87, 92: 88,
         48: 49, 53: 56, 60: 61, 70: 71, 79: 80, 73: 74, 76: 77, 82: 83, 94: 95, 97: 98}
# GetTypedFormat (dxgi_common.cpp): B8G8R8A8_TYPELESS(90)->B8G8R8A8_UNORM(87), B8G8R8X8_TYPELESS(92)->B8G8R8X8_UNORM(88),
# R24G8_TYPELESS(44)->R24_UNORM_X8_TYPELESS(46), R32G8X24_TYPELESS(19) stays typeless.
DEPTH_TYPED = {41: 40, 39: 40, 19: 20, 21: 20, 22: 20, 44: 45, 46: 45, 47: 45, 54: 55, 56: 55, 53: 55}


class Unconvertible(RuntimeError):
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get_typed_format(f):
    return TYPED.get(f, f)


def get_depth_typed_format(f):
    return DEPTH_TYPED.get(f, f)


# ------------------------------------------------------------- resource model
class Res:
    __slots__ = ('id', 'method', 'kind', 'desc', 'chunk', 'names', 'creation', 'buffer_initial', 'pre_updates',
                 'mips', 'arrays')

    def __init__(self, ident, method, desc, chunk):
        self.id, self.method, self.desc, self.chunk = ident, method, desc, chunk
        self.names, self.creation, self.buffer_initial, self.pre_updates = [], None, None, []
        if method == 'CreateBuffer':
            self.kind = 'buffer'
        elif method == 'CreateTexture1D':
            self.kind = 'tex1d'
        elif method == 'CreateTexture3D':
            self.kind = 'tex3d'
        else:
            self.kind = 'texcube' if desc['MiscFlags'] & fp.MISC_TEXTURECUBE else 'tex2d'
        self.mips = fp.mip_count(self.kind, desc)
        self.arrays = fp.array_count(self.kind, desc)


class View:
    __slots__ = ('id', 'kind', 'resource', 'desc', 'chunk', 'range')

    def __init__(self, ident, kind, resource, desc, chunk):
        self.id, self.kind, self.resource, self.desc, self.chunk = ident, kind, resource, desc, chunk
        self.range = None


class Range:
    """RenderDoc ResourceRange (d3d11_common.cpp), including its bitfield widths."""
    __slots__ = ('resource', 'min_mip', 'max_mip', 'min_slice', 'max_slice', 'full', 'depth_ro', 'stencil_ro')

    def __init__(self, resource, min_mip=0, num_mips=ALL_MIP, min_slice=0, num_slices=ALL_SLICE, *, full=None,
                 depth_ro=False, stencil_ro=False, exact=None):
        self.resource = resource
        self.depth_ro, self.stencil_ro = depth_ro, stencil_ro
        if exact is not None:  # buffer: minMip=minSlice=maxMip=maxSlice=0, fullRange=true
            self.min_mip = self.max_mip = self.min_slice = self.max_slice = 0
            self.full = exact
            return
        self.min_mip = min_mip & 0xF
        self.min_slice = min_slice & 0xFFF
        self.max_mip = ALL_MIP if num_mips == ALL_MIP else (self.min_mip + num_mips - 1) & 0xF
        self.max_slice = ALL_SLICE if num_slices == ALL_SLICE else (self.min_slice + num_slices - 1) & 0xFFF
        self.full = (self.min_mip == 0 and self.min_slice == 0 and self.max_mip == ALL_MIP and self.max_slice == ALL_SLICE) \
            if full is None else full

    def intersects(self, other):
        if self.resource != other.resource:
            return False
        if self.full or other.full:
            return True
        return self.min_mip <= other.max_mip and other.min_mip <= self.max_mip and \
            self.min_slice <= other.max_slice and other.min_slice <= self.max_slice


def typeless_of(f):
    groups = [(1, 4), (5, 8), (9, 14), (15, 18), (19, 22), (23, 26), (27, 32), (33, 38), (39, 43), (44, 47),
              (48, 52), (53, 59), (60, 65)]
    for lo, hi in groups:
        if lo <= f <= hi:
            return lo
    return f


def view_range(view, res):
    """ResourceRange of a view (null desc resolved like the runtime GetDesc)."""
    d = view.desc
    if res.kind == 'buffer':
        if view.kind == 'srv' or view.kind == 'uav' or view.kind == 'rtv':
            return Range(res.id, 0, 1, 0, 1)
        raise Unconvertible('DSV of a buffer')
    if d is None:
        d = default_view_desc(view.kind, res)
    dims = {k: v for k, v in d.items() if k not in ('Format', 'ViewDimension', 'Flags')}
    name, f = next(iter(dims.items()))
    if view.kind == 'srv':
        fmt = d.get('Format') or res.desc['Format']
        # ResourceRange(SRV): X32_TYPELESS_G8X24_UINT(22)/X24_TYPELESS_G8_UINT(47) read stencil only;
        # R32_FLOAT_X8X24_TYPELESS(21)/R24_UNORM_X8_TYPELESS(46) and any R32/R16 typeless-group format read depth only.
        stencil_ro = fmt in (22, 47)
        depth_ro = fmt in (21, 46)
        if not stencil_ro and not depth_ro and typeless_of(fmt) in (39, 53):
            depth_ro = True
        mm, nm, ms, ns = {
            'Texture1D': lambda: (f['MostDetailedMip'], f['MipLevels'], 0, 1),
            'Texture1DArray': lambda: (f['MostDetailedMip'], f['MipLevels'], f['FirstArraySlice'], f['ArraySize']),
            'Texture2D': lambda: (f['MostDetailedMip'], f['MipLevels'], 0, 1),
            'Texture2DArray': lambda: (f['MostDetailedMip'], f['MipLevels'], f['FirstArraySlice'], f['ArraySize']),
            'Texture2DMS': lambda: (0, 1, 0, 1),
            'Texture2DMSArray': lambda: (0, 1, f['FirstArraySlice'], f['ArraySize']),
            'Texture3D': lambda: (f['MostDetailedMip'], f['MipLevels'], 0, 1),
            'TextureCube': lambda: (f['MostDetailedMip'], f['MipLevels'], 0, 1),
            'TextureCubeArray': lambda: (f['MostDetailedMip'], f['MipLevels'], f['First2DArrayFace'], f['NumCubes'] * 6),
        }[name]()
        return Range(res.id, mm, nm, ms, ns, depth_ro=depth_ro, stencil_ro=stencil_ro)
    flags = d.get('Flags', 0) if view.kind == 'dsv' else 0
    mm, ms, ns = {
        'Texture1D': lambda: (f['MipSlice'], 0, 1),
        'Texture1DArray': lambda: (f['MipSlice'], f['FirstArraySlice'], f['ArraySize']),
        'Texture2D': lambda: (f.get('MipSlice', 0), 0, 1),
        'Texture2DArray': lambda: (f['MipSlice'], f['FirstArraySlice'], f['ArraySize']),
        'Texture2DMS': lambda: (0, 0, 1),
        'Texture2DMSArray': lambda: (0, f['FirstArraySlice'], f['ArraySize']),
        'Texture3D': lambda: (f['MipSlice'], f['FirstWSlice'], f['WSize']) if view.kind != 'dsv' else None,
    }[name]()
    return Range(res.id, mm, 1, ms, ns, depth_ro=bool(flags & 1), stencil_ro=bool(flags & 2))


def default_view_desc(kind, res):
    """The view desc the runtime fills in for a NULL pDesc (whole resource, resource format)."""
    d = res.desc
    ms = res.kind == 'tex2d' and d['SampleDesc']['Count'] > 1
    if kind == 'srv':
        if res.kind == 'tex1d':
            return {'Format': d['Format'], 'ViewDimension': 3 if res.arrays > 1 else 2,
                    ('Texture1DArray' if res.arrays > 1 else 'Texture1D'): dict(MostDetailedMip=0, MipLevels=res.mips, **(
                        {'FirstArraySlice': 0, 'ArraySize': res.arrays} if res.arrays > 1 else {}))}
        if res.kind == 'tex3d':
            return {'Format': d['Format'], 'ViewDimension': 8, 'Texture3D': {'MostDetailedMip': 0, 'MipLevels': res.mips}}
        if res.kind == 'texcube':
            if res.arrays > 6:
                return {'Format': d['Format'], 'ViewDimension': 10, 'TextureCubeArray': {
                    'MostDetailedMip': 0, 'MipLevels': res.mips, 'First2DArrayFace': 0, 'NumCubes': res.arrays // 6}}
            return {'Format': d['Format'], 'ViewDimension': 9, 'TextureCube': {'MostDetailedMip': 0, 'MipLevels': res.mips}}
        if ms:
            return {'Format': d['Format'], 'ViewDimension': 7, 'Texture2DMSArray': {'FirstArraySlice': 0, 'ArraySize': res.arrays}} \
                if res.arrays > 1 else {'Format': d['Format'], 'ViewDimension': 6, 'Texture2DMS': {}}
        if res.arrays > 1:
            return {'Format': d['Format'], 'ViewDimension': 5, 'Texture2DArray': {
                'MostDetailedMip': 0, 'MipLevels': res.mips, 'FirstArraySlice': 0, 'ArraySize': res.arrays}}
        return {'Format': d['Format'], 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': res.mips}}
    two_d = res.kind in ('tex2d', 'texcube')
    if res.kind == 'tex3d':
        if kind == 'dsv':
            raise Unconvertible('DSV of a Texture3D')
        return {'Format': d['Format'], 'ViewDimension': 8, 'Texture3D': {'MipSlice': 0, 'FirstWSlice': 0, 'WSize': d['Depth']}}
    if res.kind == 'tex1d':
        return {'Format': d['Format'], 'ViewDimension': 2 if res.arrays == 1 else 3, **(
            {'Texture1D': {'MipSlice': 0}} if res.arrays == 1 else
            {'Texture1DArray': {'MipSlice': 0, 'FirstArraySlice': 0, 'ArraySize': res.arrays}})}
    assert two_d
    base = {'rtv': (4, 5, 6, 7), 'uav': (4, 5, None, None), 'dsv': (3, 4, 5, 6)}[kind]
    if ms:
        dim = base[2] if res.arrays == 1 else base[3]
        member = {'Texture2DMS': {}} if res.arrays == 1 else {'Texture2DMSArray': {'FirstArraySlice': 0, 'ArraySize': res.arrays}}
    elif res.arrays == 1:
        dim, member = base[0], {'Texture2D': {'MipSlice': 0}}
    else:
        dim, member = base[1], {'Texture2DArray': {'MipSlice': 0, 'FirstArraySlice': 0, 'ArraySize': res.arrays}}
    out = {'Format': d['Format'], 'ViewDimension': dim, **member}
    if kind == 'dsv':
        out['Flags'] = 0
    return out


# ------------------------------------------------------------- state tracker
class State:
    """D3D11RenderState hazard semantics on capture ResourceIds."""

    def __init__(self, boc, conv):
        self.s = copy.deepcopy(boc)
        self.conv = conv
        if self.s['Predicate']:
            raise Unconvertible('Predication bound at capture start')
        if len(self.s['OM']['BlendFactor']) != 4:
            raise Unconvertible('Beginning-of-capture OM BlendFactor must hold 4 floats')
        self.pending_uav_counts = {}

    # ranges
    def rng(self, view):
        v = self.conv.views.get(view)
        if v is None:
            raise Unconvertible('Bound view %d has no creation record' % view)
        if v.range is None:
            v.range = view_range(v, self.conv.res[v.resource])
        return v.range

    def buf_range(self, ident):
        if ident not in self.conv.res:
            raise Unconvertible('Bound buffer %d has no creation record' % ident)
        return Range(ident, exact=True)

    def bound_for_write(self, r):
        s = self.s
        for v in s['CSUAVs']:
            if v and r.intersects(self.rng(v)):
                return True
        for b in s['SO']['Buffers']:
            if b and r.intersects(self.buf_range(b)):
                return True
        for v in s['OM']['RenderTargets']:
            if v and r.intersects(self.rng(v)):
                return True
        for v in s['OM']['UAVs']:
            if v and r.intersects(self.rng(v)):
                return True
        dv = s['OM']['DepthView']
        if dv:
            d = self.rng(dv)
            if r.intersects(d):
                if d.depth_ro and d.stencil_ro:
                    pass
                elif d.depth_ro and r.depth_ro:
                    pass
                elif d.stencil_ro and r.stencil_ro:
                    pass
                else:
                    return True
        return False

    def unbind_for_read(self, r, event):
        s = self.s
        for i, b in enumerate(s['IA']['VBs']):
            if b and r.intersects(self.buf_range(b)):
                s['IA']['VBs'][i] = 0
                self.conv.hazard(event, 'unbind_vb', i, b)
        if s['IA']['IndexBuffer'] and r.intersects(self.buf_range(s['IA']['IndexBuffer'])):
            self.conv.hazard(event, 'unbind_ib', 0, s['IA']['IndexBuffer'])
            s['IA']['IndexBuffer'] = 0
        for st in STAGE_KEYS:
            sh = s[st]
            for i, b in enumerate(sh['ConstantBuffers']):
                if b and r.intersects(self.buf_range(b)):
                    sh['ConstantBuffers'][i] = 0
                    self.conv.hazard(event, 'unbind_cb_' + st, i, b)
            for i, v in enumerate(sh['SRVs']):
                if not v:
                    continue
                sr = self.rng(v)
                if r.intersects(sr):
                    if r.depth_ro and sr.depth_ro:
                        continue
                    if r.stencil_ro and sr.stencil_ro:
                        continue
                    sh['SRVs'][i] = 0
                    self.conv.hazard(event, 'unbind_srv_' + st, i, v)

    def unbind_for_write(self, r, event):
        s = self.s
        for i, v in enumerate(s['CSUAVs']):
            if v and r.intersects(self.rng(v)):
                s['CSUAVs'][i] = 0
                self.conv.hazard(event, 'unbind_cs_uav', i, v)
        for i, b in enumerate(s['SO']['Buffers']):
            if b and r.intersects(self.buf_range(b)):
                s['SO']['Buffers'][i] = 0
                self.conv.hazard(event, 'unbind_so', i, b)
        for i, v in enumerate(s['OM']['RenderTargets']):
            if v and r.intersects(self.rng(v)):
                s['OM']['RenderTargets'][i] = 0
                self.conv.hazard(event, 'unbind_rtv', i, v)
        for i, v in enumerate(s['OM']['UAVs']):
            if v and r.intersects(self.rng(v)):
                s['OM']['UAVs'][i] = 0
                self.conv.hazard(event, 'unbind_om_uav', i, v)
        dv = s['OM']['DepthView']
        if dv and r.intersects(self.rng(dv)):
            s['OM']['DepthView'] = 0
            self.conv.hazard(event, 'unbind_dsv', 0, dv)

    def change_read(self, arr, idx, new, event, what, kind):
        if arr[idx] == new:
            return
        arr[idx] = new
        if not new:
            return
        if kind == 'srv':
            forced = self.bound_for_write(self.rng(new))
        elif kind == 'buffer':
            r = self.conv.res.get(new)
            if r is None:
                raise Unconvertible('Bound buffer %d has no creation record' % new)
            read_only = (r.desc['BindFlags'] & (BIND_UAV | BIND_SO | BIND_RT | BIND_DS)) == 0
            forced = (not read_only) and self.bound_for_write(self.buf_range(new))
        else:
            forced = False
        if forced:
            arr[idx] = 0
            self.conv.hazard(event, 'forced_null_' + what, idx, new)

    def change_write(self, arr, idx, new, event, kind):
        if arr[idx] == new:
            return
        arr[idx] = 0
        if new:
            r = self.rng(new)
            if kind != 'dsv' or not (r.depth_ro and r.stencil_ro):
                self.unbind_for_read(r, event)
            self.unbind_for_write(r, event)
        arr[idx] = new

    def valid_output_merger(self, rtvs, dsv):
        """ValidOutputMerger (d3d11_renderstate.cpp) for OMSetRenderTargets (no UAVs)."""
        ranges = []
        for v in rtvs:
            if not v:
                break
            ranges.append(self.rng(v))
        depth = self.rng(dsv) if dsv else None
        for i, a in enumerate(ranges):
            for b in ranges[i + 1:]:
                if a.intersects(b):
                    return False
            if depth is not None and a.intersects(depth):
                return False
        dims, sizes = [], []
        for v in rtvs:
            if not v:
                continue
            view = self.conv.views[v]
            res = self.conv.res[view.resource]
            dims.append(res.kind if res.kind != 'texcube' else 'tex2d')
            sizes.append(self.conv.view_extent(view, res))
        if len(set(dims)) > 1:
            return False
        if dsv:
            view = self.conv.views[dsv]
            res = self.conv.res[view.resource]
            kind = res.kind if res.kind != 'texcube' else 'tex2d'
            if dims and kind != dims[0]:
                return False
        if any(s != sizes[0] for s in sizes[1:]):
            return False
        if dsv and sizes:
            w, h, d, sc = self.conv.view_extent(self.conv.views[dsv], self.conv.res[self.conv.views[dsv].resource])
            w0, h0, d0, sc0 = sizes[0]
            if (w, h, d, sc) != (w0, h0, d0, sc0) and not (w0 <= w and h0 <= h and d0 <= d and sc0 == sc):
                return False
        return True


# ---------------------------------------------------------------- converter
class Converter:
    def __init__(self, xml, zip_path, inspect, manifests=(), actions_json=None):
        self.xml, self.zip_path, self.inspect = Path(xml), Path(zip_path), Path(inspect)
        t0 = time.perf_counter()
        self.chunks = fg.load_chunks(self.xml)
        self.blobs = fg.Blobs(self.zip_path)
        self.opened, self.chunk_event, self.driver_actions = fg.load_driver_listing(self.inspect)
        self.library = fg.ShaderLibrary(manifests)
        self.manifests = [str(Path(m).resolve()) for m in manifests]
        self.actions_json = Path(actions_json) if actions_json else None
        self.load_seconds = time.perf_counter() - t0
        self.res, self.views, self.objects, self.names = {}, {}, {}, {}
        self.initial_chunks, self.needed = {}, []
        self.hazards, self.notes, self.skipped = [], [], {}
        self.queries = {}
        self.driver_init = None

    def hazard(self, event, what, slot, ident):
        self.hazards.append({'event': event, 'resolution': what, 'slot': slot, 'object': ident})

    def view_extent(self, view, res):
        d = view.desc or default_view_desc(view.kind, res)
        dims = {k: v for k, v in d.items() if k not in ('Format', 'ViewDimension', 'Flags')}
        name, f = next(iter(dims.items()))
        mip = f.get('MipSlice', 0)
        w = max(1, res.desc['Width'] >> mip)
        h = max(1, res.desc.get('Height', 1) >> mip)
        if name.endswith('Array'):
            depth = min(res.arrays, f['ArraySize'])
        elif name == 'Texture3D':
            depth = min(max(1, res.desc['Depth'] >> mip), f['WSize'])
        else:
            depth = 1
        samples = res.desc.get('SampleDesc', {}).get('Count', 1)
        if name.startswith('Texture2DMS'):
            w, h = res.desc['Width'], res.desc['Height']
        return w, h, depth, samples

    # ------------------------------------------------------- pre-capture records
    def _create(self, c):
        method = c['name'].split('::')[1]
        f, idx = c['fields'], c['index']
        if method == 'CreateBuffer':
            r = Res(f['pBuffer'], method, f['pDesc'], idx)
            if 'InitialData' not in f or f['InitialData'] is None:
                raise Unconvertible('CreateBuffer %d has no InitialData record' % f['pBuffer'])
            if f['InitialData']['byte_length'] != f['pDesc']['ByteWidth']:
                raise Unconvertible('CreateBuffer %d InitialData length differs from ByteWidth' % f['pBuffer'])
            r.buffer_initial = f['InitialData']
            self._add_res(r)
        elif method in ('CreateTexture1D', 'CreateTexture2D', 'CreateTexture3D'):
            r = Res(f['pTexture'], method, f['Descriptor'], idx)
            contents = f.get('SubresourceContents')
            refs = list(contents) if isinstance(contents, fg.Repeated) else ([] if contents is None else [contents])
            if refs and len(refs) != r.mips * r.arrays:
                raise Unconvertible('Texture %d creation data count differs from subresource count' % r.id)
            if refs and r.kind == 'tex2d' and r.desc['SampleDesc']['Count'] > 1:
                raise Unconvertible('MSAA creation data')
            r.creation = refs or None
            self._add_res(r)
        elif method in ('CreateShaderResourceView', 'CreateRenderTargetView', 'CreateDepthStencilView',
                        'CreateUnorderedAccessView'):
            kind = {'CreateShaderResourceView': 'srv', 'CreateRenderTargetView': 'rtv', 'CreateDepthStencilView': 'dsv',
                    'CreateUnorderedAccessView': 'uav'}[method]
            v = View(f['pView'], kind, f['pResource'], f['pDesc'], idx)
            if v.id in self.views or v.id in self.res or v.id in self.objects:
                raise Unconvertible('Object id reused %d' % v.id)
            self.views[v.id] = v
        elif method.endswith('Shader') and method.startswith('Create'):
            stage = {'CreateVertexShader': 'vs', 'CreateHullShader': 'hs', 'CreateDomainShader': 'ds',
                     'CreateGeometryShader': 'gs', 'CreatePixelShader': 'ps', 'CreateComputeShader': 'cs'}.get(method)
            if stage is None:
                raise Unconvertible('Uninterpreted shader creation ' + method)
            if f.get('pClassLinkage'):
                raise Unconvertible('Class linkage shaders')
            self._add_obj(f['pShader'], 'shader', {'stage': stage, 'ref': f['pShaderBytecode'], 'chunk': idx})
        elif method == 'CreateInputLayout':
            self._add_obj(f['pInputLayout'], 'input_layout', {'elements': f['pInputElementDescs'],
                                                              'ref': f['pShaderBytecodeWithInputSignature'], 'chunk': idx})
        elif method in ('CreateBlendState', 'CreateDepthStencilState', 'CreateRasterizerState', 'CreateSamplerState'):
            table = {'CreateBlendState': 'blend_states', 'CreateDepthStencilState': 'depth_states',
                     'CreateRasterizerState': 'raster_states', 'CreateSamplerState': 'samplers'}[method]
            self._add_obj(f['pState'], table, {'desc': f['Descriptor'], 'chunk': idx})
        elif method == 'CreateQuery':
            self.queries[f['pQuery']] = f['Descriptor']
        else:
            raise Unconvertible('Uninterpreted resource creation ' + method)

    def _add_res(self, r):
        if r.id in self.res or r.id in self.views or r.id in self.objects or not r.id:
            raise Unconvertible('Invalid/reused resource id %d' % r.id)
        self.res[r.id] = r

    def _add_obj(self, ident, table, data):
        if not ident or ident in self.res or ident in self.views or ident in self.objects:
            raise Unconvertible('Invalid/reused object id %r' % ident)
        self.objects[ident] = (table, data)

    # ----------------------------------------------------------- conversion
    def convert(self, out, through_event, readbacks, label='', blob_link_sources=()):
        t0 = time.perf_counter()
        out = Path(out)
        b = fp.ProgramBuilder(out, provenance={}, blob_link_sources=blob_link_sources)
        cmds = b.program['commands']
        self.builder = b
        state = None
        boc_seen = False
        pre_updates = []
        last_event = 0
        events_seen = 0
        for c in self.chunks:
            name, f, idx = c['name'], c['fields'], c['index']
            if not boc_seen:
                if name == 'Internal::Driver Initialisation Parameters':
                    self.driver_init = f['InitParams']
                elif name in ('Internal::DeviceInitialisation', 'Internal::Frame Metadata'):
                    pass
                elif name == 'Internal::Initial Contents':
                    if f['id'] in self.initial_chunks:
                        raise Unconvertible('Two initial-content chunks for %d' % f['id'])
                    self.initial_chunks[f['id']] = c
                elif name == 'Internal::List of Initial Contents Resources':
                    self.needed = f['NeededInitials']
                elif name == 'Internal::Beginning of Capture':
                    if any(f['HiddenStreamOutCounters']) or f['numAnnotations']:
                        raise Unconvertible('Initial stream-out counters/annotations')
                    boc_seen = True
                    state = State(f['state'], self)
                    self._emit_initial_state(pre_updates)
                    self.initial_command_count = len(cmds)
                elif name.startswith('ID3D11Device::Create'):
                    self._create(c)
                elif name == 'IDXGISwapChain::GetBuffer':
                    r = Res(f['SwapbufferID'], 'CreateTexture2D', f['BackbufferDescriptor'], idx)
                    r.names.append('swapchain buffer %d' % f['Buffer'])
                    self._add_res(r)
                elif name == 'ID3D11Resource::SetDebugName':
                    self.names.setdefault(f['pResource'], []).append(f['Name'])
                elif name.startswith('ID3D11DeviceContext::'):
                    if f.get('Context') != self.immediate_context():
                        raise Unconvertible('Deferred context command')
                    method = name.split('::')[1]
                    if method not in ('UpdateSubresource', 'UpdateSubresource1'):
                        raise Unconvertible('Pre-capture context command ' + method)
                    pre_updates.append(c)
                else:
                    raise Unconvertible('Uninterpreted pre-capture chunk ' + name)
                continue
            if name == 'Internal::End of Capture':
                break
            ev = self.chunk_event.get(idx)
            if ev is None:
                if name == 'ID3D11Resource::SetDebugName':
                    self.names.setdefault(f['pResource'], []).append(f['Name'])
                    continue
                raise Unconvertible('Frame chunk %d (%s) has no official event id' % (idx, name))
            if ev > through_event:
                break
            if ev <= last_event:
                raise Unconvertible('Event ids are not increasing at chunk %d' % idx)
            last_event = ev
            events_seen += 1
            if not (name.startswith('ID3D11DeviceContext::') or name == 'IDXGISwapChain::Present'):
                raise Unconvertible('Uninterpreted frame chunk ' + name)
            if f.get('Context') != self.immediate_context():
                raise Unconvertible('Deferred/foreign context command at event %d' % ev)
            self._context(name.split('::')[1], f, idx, ev, state)
        if not boc_seen:
            raise Unconvertible('No Beginning of Capture')
        if last_event != through_event:
            raise Unconvertible('Requested end event %d is not an API event of this capture (last %d)'
                                % (through_event, last_event))
        # readbacks
        rb_rows = []
        for rb in readbacks:
            after = None
            for n in range(len(cmds) - 1, -1, -1):
                e = cmds[n].get('provenance', {}).get('event')
                if e is not None and e <= rb['event']:
                    after = n
                    break
            if after is None:
                raise Unconvertible('Readback %s precedes every frame command' % rb['name'])
            if rb['event'] > through_event:
                raise Unconvertible('Readback %s is after the program end event' % rb['name'])
            if rb['resource'] not in self.res:
                raise Unconvertible('Readback resource %d was not created' % rb['resource'])
            row = b.readback(rb['name'], rb['resource'], subresource=rb.get('subresource', 0), after_command=after,
                             orientation=rb.get('orientation', 'd3d_rows_top_down'),
                             provenance={'event': rb['event']})
            rb_rows.append(row)
        self._emit_objects(b)
        b.program['provenance'] = self._provenance(through_event, label, time.perf_counter() - t0)
        b.write()
        report = self._report(out, b, through_event, events_seen)
        (out / 'conversion_report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
        return b.program, report

    def immediate_context(self):
        if getattr(self, '_ctx', None) is not None:
            return self._ctx
        for c in self.chunks:
            if c['name'] == 'Internal::DeviceInitialisation':
                self._ctx = c['fields']['ImmediateContext']
                return self._ctx
        raise Unconvertible('No DeviceInitialisation chunk')

    # ---------------------------------------------------------- initial state
    def _emit_initial_state(self, pre_updates):
        """RenderDoc CreateInitialContents/ApplyInitialContents, recorded per resource."""
        self.initial_class = {}
        first_record = {}
        for wr in self.needed:
            first_record.setdefault(wr['id'], wr['written'])
        self.clear_views = {}
        self.pre_updates = pre_updates
        for ident, r in self.res.items():
            listed = ident in first_record
            ic = self.initial_chunks.get(ident) if listed else None
            if ic is not None:
                if ic['fields'].get('OmittedContents'):
                    raise Unconvertible('Omitted initial contents for %d' % ident)
                self.initial_class[ident] = 'initial_contents_chunk'
                continue
            if r.kind == 'buffer':
                self.initial_class[ident] = 'capture_start_createbuffer_record'
                continue
            written = first_record.get(ident)
            if listed and not written and r.mips == 1 and r.desc['BindFlags'] & BIND_RT:
                self.initial_class[ident] = 'renderdoc_clear_rtv_zero'
                if r.kind in ('tex2d', 'texcube') and r.arrays > 1:
                    self.notes.append({'resource': ident, 'note': 'Create_InitialState clear RTV uses TEXTURE2D '
                                       'dimension on an array resource (as RenderDoc does)'})
            elif listed and not written and r.mips == 1 and r.desc['BindFlags'] & BIND_DS and r.kind in ('tex1d', 'tex2d', 'texcube'):
                self.initial_class[ident] = 'renderdoc_clear_dsv_one_zero'
            elif r.creation is not None:
                self.initial_class[ident] = 'creation_data' + ('_listed_copy_of_live' if listed else '')
            else:
                self.initial_class[ident] = 'undefined_at_creation' + ('_listed_copy_of_live' if listed else '')
        for ident in self.initial_chunks:
            if ident not in first_record:
                self.notes.append({'resource': ident, 'note': 'initial-content chunk of an unlisted resource is freed on replay; ignored'})
        # pre-capture updates (executed at load, before the copy-of-live snapshot)
        full_updates = {}
        for c in pre_updates:
            f = c['fields']
            if self.initial_class.get(f['pDstResource']) == 'initial_contents_chunk':
                self.notes.append({'chunk': c['index'], 'note': 'pre-capture update superseded by initial-contents chunk'})
                continue
            self._update_subresource(f, c['index'], None)
            if not f['IsUpdate']:
                full_updates.setdefault(f['pDstResource'], set()).add(f['DstSubresource'])
        for ident, subs in full_updates.items():
            r = self.res[ident]
            if self.initial_class[ident].startswith('undefined_at_creation') and subs == set(range(r.mips * r.arrays)):
                # Serialise_UpdateSubresource1 IsUpdate=false carries the whole subresource: fully defined at load.
                self.initial_class[ident] = 'pre_capture_full_update' + self.initial_class[ident][len('undefined_at_creation'):]
        # ApplyInitialContents clears
        for ident, cls in sorted(self.initial_class.items()):
            r = self.res[ident]
            if cls == 'renderdoc_clear_rtv_zero':
                vid = SYNTHETIC_BASE + len(self.clear_views)
                fmt = get_typed_format(r.desc['Format'])
                if r.kind == 'tex1d':
                    desc = {'Format': fmt, 'ViewDimension': 2, 'Texture1D': {'MipSlice': 0}}
                elif r.kind == 'tex3d':
                    desc = {'Format': fmt, 'ViewDimension': 8, 'Texture3D': {'MipSlice': 0, 'FirstWSlice': 0, 'WSize': r.desc['Depth']}}
                elif r.desc['SampleDesc']['Count'] > 1:
                    desc = {'Format': fmt, 'ViewDimension': 6, 'Texture2DMS': {}}
                else:
                    desc = {'Format': fmt, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}}
                self.clear_views[vid] = View(vid, 'rtv', ident, desc, None)
                self.builder.clear_rtv(vid, [0.0, 0.0, 0.0, 0.0], initial_state=cls, resource=ident)
            elif cls == 'renderdoc_clear_dsv_one_zero':
                vid = SYNTHETIC_BASE + len(self.clear_views)
                fmt = get_depth_typed_format(r.desc['Format'])
                if r.kind == 'tex1d':
                    desc = {'Format': fmt, 'ViewDimension': 1, 'Flags': 0, 'Texture1D': {'MipSlice': 0}}
                elif r.desc['SampleDesc']['Count'] > 1:
                    desc = {'Format': fmt, 'ViewDimension': 5, 'Flags': 0, 'Texture2DMS': {}}
                else:
                    desc = {'Format': fmt, 'ViewDimension': 3, 'Flags': 0, 'Texture2D': {'MipSlice': 0}}
                self.clear_views[vid] = View(vid, 'dsv', ident, desc, None)
                self.builder.clear_dsv(vid, 3, 1.0, 0, initial_state=cls, resource=ident)

    def _initial_entries(self, r):
        cls = self.initial_class[r.id]
        b = self.builder
        if cls == 'capture_start_createbuffer_record':
            data = self.blobs.read(r.buffer_initial)
            return [{'subresource': 0, 'blob': b.blob(data), 'row_pitch': len(data), 'slice_pitch': len(data)}]
        if cls == 'initial_contents_chunk':
            f = self.initial_chunks[r.id]['fields']
            if r.kind == 'tex2d' or r.kind == 'texcube':
                if r.desc['SampleDesc']['Count'] > 1:
                    raise Unconvertible('MSAA initial contents')
                n = f['NumSubresources']
                if n != r.mips * r.arrays:
                    raise Unconvertible('Initial contents subresource count mismatch for %d' % r.id)
                pitches = list(f['RowPitch']) if isinstance(f['RowPitch'], fg.Repeated) else [f['RowPitch']]
                refs = list(f['SubresourceContents']) if isinstance(f['SubresourceContents'], fg.Repeated) else [f['SubresourceContents']]
                if len(pitches) != n or len(refs) != n:
                    raise Unconvertible('Initial contents field count mismatch for %d' % r.id)
                ls = fp.subresource_layouts(r.kind, r.desc)
                out = []
                for sub, (rp, ref) in enumerate(zip(pitches, refs)):
                    data = self.blobs.read(ref)
                    rows = ls[sub]['rows']
                    if len(data) != rp * (rows - 1) + ls[sub]['row_bytes']:
                        raise Unconvertible('Initial contents length differs from RowPitch*(rows-1)+rowLength for %d/%d' % (r.id, sub))
                    out.append({'subresource': sub, 'blob': b.blob(data), 'row_pitch': rp, 'slice_pitch': rp * rows})
                return out
            if r.kind == 'tex3d':
                n = f['NumSubresources']
                rps = list(f['RowPitch']) if isinstance(f['RowPitch'], fg.Repeated) else [f['RowPitch']]
                dps = list(f['DepthPitch']) if isinstance(f['DepthPitch'], fg.Repeated) else [f['DepthPitch']]
                refs = list(f['SubresourceContents']) if isinstance(f['SubresourceContents'], fg.Repeated) else [f['SubresourceContents']]
                if not (n == r.mips == len(rps) == len(dps) == len(refs)):
                    raise Unconvertible('Texture3D initial contents count mismatch for %d' % r.id)
                ls = fp.subresource_layouts(r.kind, r.desc)
                out = []
                for sub in range(n):
                    data = self.blobs.read(refs[sub])
                    l = ls[sub]
                    if len(data) != dps[sub] * (l['depth'] - 1) + rps[sub] * (l['rows'] - 1) + l['row_bytes']:
                        raise Unconvertible('Texture3D initial contents length mismatch for %d/%d' % (r.id, sub))
                    out.append({'subresource': sub, 'blob': b.blob(data), 'row_pitch': rps[sub], 'slice_pitch': dps[sub]})
                return out
            raise Unconvertible('Texture1D initial contents not implemented')
        if r.creation is None:
            return None
        ls = fp.subresource_layouts(r.kind, r.desc)
        out = []
        for sub, ref in enumerate(r.creation):
            data = self.blobs.read(ref)
            if len(data) != ls[sub]['bytes']:
                raise Unconvertible('Creation data length mismatch for %d/%d' % (r.id, sub))
            out.append({'subresource': sub, 'blob': b.blob(data), 'row_pitch': ls[sub]['row_bytes'],
                        'slice_pitch': ls[sub]['slice_bytes']})
        return out

    # -------------------------------------------------------------- commands
    def _prov(self, ev, idx, method):
        return {'event': ev, 'chunk': idx, 'api': method}

    def _update_subresource(self, f, idx, ev):
        r = self.res.get(f['pDstResource'])
        if r is None:
            raise Unconvertible('UpdateSubresource of an unrecorded resource')
        sub = f['DstSubresource']
        flags = f['CopyFlags']
        copy_flags = None if flags == 0xFFFFFFFF else flags
        if f['IsUpdate']:
            ref = f['pSrcData']
            data = b'' if ref is None else self.blobs.read(ref)
            if len(data) != f['SourceDataLength']:
                raise Unconvertible('UpdateSubresource source length mismatch')
            if not data:
                self.skipped['UpdateSubresource_empty'] = self.skipped.get('UpdateSubresource_empty', 0) + 1
                return
            box = f['pDstBox']
            rp, dp = f['SrcRowPitch'], f['SrcDepthPitch']
        else:
            data = self.blobs.read(f['Contents'])
            if len(data) != f['ContentsLength']:
                raise Unconvertible('UpdateSubresource contents length mismatch')
            box = None
            if r.kind == 'buffer':
                rp = dp = 0
                # GetRowPitch(1, UNKNOWN)/GetByteSize(...UNKNOWN) evaluate to 0 for buffers; pitches are unused.
            else:
                mip = sub % r.mips
                w = max(1, r.desc['Width'] >> mip)
                h = max(1, r.desc.get('Height', 1) >> mip)
                bw, bh, size = fp.block_layout(r.desc['Format'])
                rp = -(-w // bw) * size
                dp = rp * -(-h // bh)
        self.builder.update_subresource(r.id, sub, data, rp, dp, box=box, copy_flags=copy_flags,
                                        **({} if ev is None else {'event': ev}), chunk=idx, api='UpdateSubresource')

    def _context(self, method, f, idx, ev, state):
        s = state.s
        ia, om, rs = s['IA'], s['OM'], s['RS']
        b = self.builder
        if method == 'IASetInputLayout':
            state.change_read(ia, 'Layout', f['pInputLayout'], ev, 'layout', 'object')
        elif method == 'IASetVertexBuffers':
            st = f['StartSlot']
            n = f['NumBuffers']
            for i in range(n):
                state.change_read(ia['VBs'], st + i, f['ppVertexBuffers'][i], ev, 'vb', 'buffer')
                ia['Strides'][st + i] = f['pStrides'][i]
                ia['Offsets'][st + i] = f['pOffsets'][i]
        elif method == 'IASetIndexBuffer':
            state.change_read(ia, 'IndexBuffer', f['pIndexBuffer'], ev, 'ib', 'buffer')
            ia['IndexFormat'] = f['Format']
            ia['IndexOffset'] = f['Offset']
        elif method == 'IASetPrimitiveTopology':
            ia['Topo'] = f['Topology']
        elif method[2:] == 'SetShader' and method[:2] in STAGE_KEYS:
            if f['NumClassInstances']:
                raise Unconvertible('Class instances')
            s[method[:2]]['Object'] = f['pShader']
        elif method[:2] in STAGE_KEYS and method[2:] == 'SetConstantBuffers':
            st, sh = f['StartSlot'], s[method[:2]]
            for i, v in enumerate(f['ppConstantBuffers']):
                state.change_read(sh['ConstantBuffers'], st + i, v, ev, 'cb_' + method[:2], 'buffer')
                sh['CBOffsets'][st + i] = 0
                sh['CBCounts'][st + i] = 4096
        elif method[:2] in STAGE_KEYS and method[2:] == 'SetShaderResources':
            st, sh = f['StartSlot'], s[method[:2]]
            for i, v in enumerate(f['ppShaderResourceViews']):
                state.change_read(sh['SRVs'], st + i, v, ev, 'srv_' + method[:2], 'srv')
        elif method[:2] in STAGE_KEYS and method[2:] == 'SetSamplers':
            st, sh = f['StartSlot'], s[method[:2]]
            for i, v in enumerate(f['ppSamplers']):
                sh['Samplers'][st + i] = v
        elif method == 'CSSetUnorderedAccessViews':
            st = f['StartSlot']
            counts = f.get('pUAVInitialCounts') or [0xFFFFFFFF] * f['NumUAVs']
            for i, v in enumerate(f['ppUnorderedAccessViews']):
                state.change_write(s['CSUAVs'], st + i, v, ev, 'uav')
                if counts[i] != 0xFFFFFFFF:
                    state.pending_uav_counts[st + i] = counts[i]
                else:
                    state.pending_uav_counts.pop(st + i, None)
        elif method == 'OMSetRenderTargets':
            n = f['NumViews']
            rtvs = list(f['ppRenderTargetViews']) + [0] * (fp.RT_SLOTS - len(f['ppRenderTargetViews']))
            dsv = f['pDepthStencilView']
            if state.valid_output_merger(rtvs[:n], dsv):
                for i in range(fp.RT_SLOTS):
                    state.change_write(om['RenderTargets'], i, rtvs[i], ev, 'rtv')
                holder = [om['DepthView']]
                state.change_write(holder, 0, dsv, ev, 'dsv')
                om['DepthView'] = holder[0]
            else:
                self.hazard(ev, 'invalid_output_merger_ignored', 0, 0)
            for i in range(len(om['UAVs'])):
                om['UAVs'][i] = 0
            om['UAVStartSlot'] = n
        elif method == 'OMSetRenderTargetsAndUnorderedAccessViews':
            raise Unconvertible('OM UAV binding')
        elif method == 'OMSetDepthStencilState':
            om['DepthStencilState'] = f['pDepthStencilState']
            om['StencRef'] = f['StencilRef']
        elif method == 'OMSetBlendState':
            om['BlendState'] = f['pBlendState']
            bf = f['BlendFactor']
            om['BlendFactor'] = list(bf) if bf else [1.0, 1.0, 1.0, 1.0]  # NULL -> {1,1,1,1} (D3D11 spec)
            om['SampleMask'] = f['SampleMask']
        elif method == 'RSSetState':
            rs['State'] = f['pRasterizerState']
        elif method == 'RSSetViewports':
            rs['NumViews'] = f['NumViewports']
            rs['Viewports'][:f['NumViewports']] = f['pViewports']
        elif method in ('RSSetScissorRects', 'RSSetScissors'):
            rs['NumScissors'] = f['NumRects']
            rs['Scissors'][:f['NumRects']] = f['pRects']
        elif method == 'Map':
            if f['MapType'] == 3:
                raise Unconvertible('MAP_READ_WRITE at event %d' % ev)
        elif method == 'Unmap':
            self._unmap(f, idx, ev)
        elif method in ('UpdateSubresource', 'UpdateSubresource1'):
            self._update_subresource(f, idx, ev)
        elif method == 'CopyResource':
            self._need(f['pDstResource'], f['pSrcResource'])
            b.copy_resource(f['pDstResource'], f['pSrcResource'], **self._prov(ev, idx, method))
        elif method == 'CopySubresourceRegion':
            self._need(f['pDstResource'], f['pSrcResource'])
            b.copy_subresource_region(f['pDstResource'], f['DstSubresource'], (f['DstX'], f['DstY'], f['DstZ']),
                                      f['pSrcResource'], f['SrcSubresource'], f['pSrcBox'], None,
                                      **self._prov(ev, idx, method))
        elif method == 'ClearRenderTargetView':
            self._need_view(f['pRenderTargetView'], 'rtv')
            b.clear_rtv(f['pRenderTargetView'], f['ColorRGBA'], **self._prov(ev, idx, method))
        elif method == 'ClearDepthStencilView':
            self._need_view(f['pDepthStencilView'], 'dsv')
            b.clear_dsv(f['pDepthStencilView'], f['ClearFlags'], f['Depth'], f['Stencil'], **self._prov(ev, idx, method))
        elif method == 'ClearUnorderedAccessViewFloat':
            self._need_view(f['pUnorderedAccessView'], 'uav')
            b.clear_uav_float(f['pUnorderedAccessView'], f['Values'], **self._prov(ev, idx, method))
        elif method == 'ClearUnorderedAccessViewUint':
            self._need_view(f['pUnorderedAccessView'], 'uav')
            b.clear_uav_uint(f['pUnorderedAccessView'], f['Values'], **self._prov(ev, idx, method))
        elif method == 'GenerateMips':
            self._need_view(f['pShaderResourceView'], 'srv')
            b.generate_mips(f['pShaderResourceView'], **self._prov(ev, idx, method))
        elif method == 'ResolveSubresource':
            self._need(f['pDstResource'], f['pSrcResource'])
            b.resolve(f['pDstResource'], f['DstSubresource'], f['pSrcResource'], f['SrcSubresource'], f['Format'],
                      **self._prov(ev, idx, method))
        elif method in fp.DRAW_OPS_API:
            self._draw(method, f, idx, ev, state)
        elif method == 'Dispatch':
            self._dispatch(f, idx, ev, state)
        elif method in ('Begin', 'End'):
            q = self.queries.get(f['pAsync'])
            if q is None or q['Query'] != 0:
                raise Unconvertible('Non-event query %s at event %d' % (q, ev))
            self.skipped['event_query_' + method] = self.skipped.get('event_query_' + method, 0) + 1
        elif method == 'Flush':
            # ID3D11DeviceContext::Flush only submits the queued command buffer to the GPU; it
            # changes no pipeline state and no resource contents (D3D11 API contract)
            self.skipped['context_Flush'] = self.skipped.get('context_Flush', 0) + 1
        elif method == 'Present':
            raise Unconvertible('Present inside the program range is not a frame-program command')
        else:
            raise Unconvertible('Uninterpreted context command %s at event %d' % (method, ev))

    def _need(self, *ids):
        for i in ids:
            if i not in self.res:
                raise Unconvertible('Command references unrecorded resource %d' % i)

    def _need_view(self, v, kind):
        if v not in self.views or self.views[v].kind != kind:
            raise Unconvertible('Command references unrecorded %s %d' % (kind, v))

    def _unmap(self, f, idx, ev):
        ident, sub = f['pResource'], f['Subresource']
        if ident not in self.res:
            raise Unconvertible('Unmap of unrecorded resource %d at event %d' % (ident, ev))
        mt = f['MapType']
        start, end = f['Byte offset to start of written data'], f['Byte offset to end of written data']
        if mt == 1:
            self.skipped['Unmap_read'] = self.skipped.get('Unmap_read', 0) + 1
            return
        if start >= end:
            # Serialise_Unmap replay: diffStart >= diffEnd -> nothing (no Map of the real resource).
            self.skipped['Unmap_empty_range'] = self.skipped.get('Unmap_empty_range', 0) + 1
            return
        data = self.blobs.read(f['MapWrittenData'])
        if len(data) != end - start:
            raise Unconvertible('MapWrittenData length differs from the recorded range at event %d' % ev)
        self.builder.map_write(ident, sub, mt, f['MapFlags'], start, end, data, **self._prov(ev, idx, 'Unmap'))

    def _stage(self, sh, key, compute=False, state=None):
        out = {'shader': sh['Object'] or None, 'constant_buffers': [], 'srvs': [], 'samplers': []}
        for i, cb in enumerate(sh['ConstantBuffers']):
            if cb:
                out['constant_buffers'].append({'slot': i, 'buffer': cb, 'first_constant': sh['CBOffsets'][i],
                                                'num_constants': sh['CBCounts'][i]})
        out['srvs'] = [{'slot': i, 'view': v} for i, v in enumerate(sh['SRVs']) if v]
        out['samplers'] = [{'slot': i, 'sampler': v} for i, v in enumerate(sh['Samplers']) if v]
        if compute:
            out['uavs'] = [{'slot': i, 'view': v, 'initial_count': state.pending_uav_counts.get(i, 0xFFFFFFFF)}
                           for i, v in enumerate(state.s['CSUAVs']) if v]
        return out

    def _draw(self, method, f, idx, ev, state):
        s = state.s
        if s['Predicate']:
            raise Unconvertible('Predicated draw')
        if any(s['SO']['Buffers']):
            raise Unconvertible('Stream-out draw')
        if any(s['OM']['UAVs']):
            raise Unconvertible('OM UAV draw')
        ia, om, rs = s['IA'], s['OM'], s['RS']
        st = {'ia': {'input_layout': ia['Layout'] or None, 'topology': ia['Topo'],
                     'vertex_buffers': [{'slot': i, 'buffer': v, 'stride': ia['Strides'][i], 'offset': ia['Offsets'][i]}
                                        for i, v in enumerate(ia['VBs']) if v],
                     'index_buffer': {'buffer': ia['IndexBuffer'], 'format': ia['IndexFormat'], 'offset': ia['IndexOffset']}
                     if ia['IndexBuffer'] else None},
              'om': {'render_targets': [{'slot': i, 'view': v} for i, v in enumerate(om['RenderTargets']) if v],
                     'depth_stencil': om['DepthView'] or None, 'uavs': [], 'blend_state': om['BlendState'] or None,
                     'blend_factor': list(om['BlendFactor']), 'sample_mask': om['SampleMask'],
                     'depth_stencil_state': om['DepthStencilState'] or None, 'stencil_ref': om['StencRef']},
              'rs': {'state': rs['State'] or None, 'viewports': copy.deepcopy(rs['Viewports'][:rs['NumViews']]),
                     'scissors': copy.deepcopy(rs['Scissors'][:rs['NumScissors']])},
              'predication': None, 'stream_out': []}
        for k in ('VS', 'HS', 'DS', 'GS', 'PS'):
            st[k.lower()] = self._stage(s[k], k)
        op = fp.DRAW_OPS_API[method]
        args = {k: f[k] for k in fp.DRAW_OPS[op]}
        self.builder._draw(op, st, self._prov(ev, idx, method), args)

    def _dispatch(self, f, idx, ev, state):
        st = {'cs': self._stage(state.s['CS'], 'CS', compute=True, state=state)}
        state.pending_uav_counts.clear()
        self.builder.dispatch(st, self._prov(ev, idx, 'Dispatch'), **{k: f[k] for k in fp.DISPATCH_ARGS})

    # --------------------------------------------------------------- objects
    def _referenced(self, program):
        refs = set()

        def stage(bd):
            if bd.get('shader'):
                refs.add(bd['shader'])
            for cb in bd.get('constant_buffers', []):
                if 'buffer' in cb:
                    refs.add(cb['buffer'])
            refs.update(x['view'] for x in bd.get('srvs', []))
            refs.update(x['sampler'] for x in bd.get('samplers', []))
            refs.update(x['view'] for x in bd.get('uavs', []))
        for c in program['commands']:
            for k in ('view', 'resource', 'dst', 'src'):
                if k in c:
                    refs.add(c[k])
            st = c.get('state')
            if not st:
                continue
            if 'cs' in st:
                stage(st['cs'])
                continue
            ia = st['ia']
            if ia['input_layout']:
                refs.add(ia['input_layout'])
            refs.update(v['buffer'] for v in ia['vertex_buffers'])
            if ia['index_buffer']:
                refs.add(ia['index_buffer']['buffer'])
            for k in fp.GRAPHICS_STAGES:
                stage(st[k])
            om = st['om']
            refs.update(v['view'] for v in om['render_targets'])
            for k in ('depth_stencil', 'blend_state', 'depth_stencil_state'):
                if om[k]:
                    refs.add(om[k])
            if st['rs']['state']:
                refs.add(st['rs']['state'])
        for rb in program['readbacks']:
            refs.add(rb['resource'])
        for v in list(refs):
            view = self.views.get(v) or self.clear_views.get(v)
            if view:
                refs.add(view.resource)
        return refs

    def _emit_objects(self, b):
        p = b.program
        refs = self._referenced(p)
        self.referenced = refs
        for ident in sorted(refs):
            if ident in self.res:
                r = self.res[ident]
                b._ids.add(ident)
                p['resources'].append({'id': ident, 'kind': r.kind, 'name': ' | '.join(self.names.get(ident, []) + r.names),
                                       'desc': r.desc, 'initial': self._initial_entries(r),
                                       'provenance': {'chunk': r.chunk, 'initial_state': self.initial_class[ident]}})
            elif ident in self.views or ident in self.clear_views:
                v = self.views.get(ident) or self.clear_views[ident]
                if v.resource not in self.res:
                    raise Unconvertible('View %d base resource %d was not created' % (ident, v.resource))
                b._ids.add(ident)
                p['views'].append({'id': ident, 'kind': v.kind, 'resource': v.resource, 'desc': v.desc,
                                   'name': ' | '.join(self.names.get(ident, [])),
                                   'provenance': {'chunk': v.chunk} if v.chunk is not None else
                                   {'synthetic': 'RenderDoc Create_InitialState fast-clear view'}})
            elif ident in self.objects:
                table, data = self.objects[ident]
                b._ids.add(ident)
                if table == 'shader':
                    dxbc = self.blobs.read(data['ref'])
                    h = sha(dxbc)
                    p['shaders'].append({'id': ident, 'stage': data['stage'], 'blob': b.blob(dxbc),
                                         'source': {'capture_resource_id': ident, 'dxbc_sha256': h,
                                                    'installed_matches': self.library.match(h)},
                                         'provenance': {'chunk': data['chunk']}})
                elif table == 'input_layout':
                    p['input_layouts'].append({'id': ident, 'elements': data['elements'],
                                               'signature_blob': b.blob(self.blobs.read(data['ref'])),
                                               'provenance': {'chunk': data['chunk']}})
                else:
                    p[table].append({'id': ident, 'desc': data['desc'], 'provenance': {'chunk': data['chunk']}})
            else:
                raise Unconvertible('Program references object %d with no creation record' % ident)

    def _provenance(self, through_event, label, seconds):
        def fsha(path):
            h = hashlib.sha256()
            with open(path, 'rb') as fh:
                for blk in iter(lambda: fh.read(1 << 24), b''):
                    h.update(blk)
            return h.hexdigest()
        return {'converter': CONVERTER_VERSION, 'converter_sha256': fsha(__file__),
                'frame_program_module_sha256': fsha(fp.__file__),
                'label': label, 'capture_xml': str(self.xml.resolve()), 'capture_xml_sha256': fsha(self.xml),
                'capture_zip': str(self.zip_path.resolve()), 'capture_zip_sha256': fsha(self.zip_path),
                'driver_listing': str(self.inspect.resolve()), 'driver_listing_sha256': fsha(self.inspect),
                'driver_opened': self.opened, 'through_event': through_event,
                'capture_adapter': (self.driver_init or {}).get('AdapterDesc'),
                'capture_driver_init': self.driver_init, 'shader_manifests': self.manifests,
                'conversion_seconds': seconds, 'xml_load_seconds': self.load_seconds,
                'initial_state_commands': self.initial_command_count}

    def _report(self, out, b, through_event, events_seen):
        p = b.program
        classes = {}
        for r in p['resources']:
            cls = r['provenance']['initial_state']
            classes[cls] = classes.get(cls, 0) + 1
        undefined = [r['id'] for r in p['resources'] if r['provenance']['initial_state'].startswith('undefined_at_creation')]
        ops = {}
        for c in p['commands']:
            ops[c['op']] = ops.get(c['op'], 0) + 1
        crosscheck = self._crosscheck(p) if self.actions_json else None
        dependencies = []
        for ident in undefined:
            r = next(x for x in p['resources'] if x['id'] == ident)
            size = fp.subresource_layouts(r['kind'], r['desc'])
            first = None
            for n, c in enumerate(p['commands']):
                touched = c.get('resource') == ident or c.get('dst') == ident or c.get('src') == ident or                     any(v['resource'] == ident and v['id'] == c.get('view') for v in p['views'])
                if c.get('state'):
                    touched = touched or ident in self._referenced({'commands': [c], 'readbacks': []})
                if touched:
                    first = (n, c)
                    break
            row = {'resource': ident, 'name': r['name'], 'first_command': None if first is None else first[0]}
            if first is not None:
                n, c = first
                row['first_op'] = c['op']
                row['first_event'] = (c.get('provenance') or {}).get('event')
                if c['op'] == 'map_write':
                    total = size[c['subresource']]['bytes']
                    kept = []
                    if c['start'] > 0:
                        kept.append([0, c['start']])
                    if c['end'] < total:
                        kept.append([c['end'], total])
                    row['undefined_bytes_kept'] = kept
                    row['fully_defined_by_first_write'] = not kept
                elif c['op'] == 'copy_subresource_region' and c['dst'] == ident:
                    row['fully_defined_by_first_write'] = c['src_box'] is None
                elif c['op'] == 'copy_resource' and c['dst'] == ident:
                    row['fully_defined_by_first_write'] = True
                else:
                    row['fully_defined_by_first_write'] = False
            dependencies.append(row)
        return {'schema': 'hs2_capture_to_program_report/1', 'through_event': through_event,
                'api_events_converted': events_seen, 'commands': len(p['commands']), 'ops': ops,
                'objects': {k: len(p[k]) for k in ('resources', 'views', 'shaders', 'input_layouts', 'blend_states',
                                                   'depth_states', 'raster_states', 'samplers')},
                'created_but_unreferenced_resources': len([i for i in self.res if i not in self.referenced]),
                'initial_state_classes': classes, 'undefined_at_creation_resources': undefined,
                'undefined_content_dependencies': dependencies,
                'hazard_resolutions': self.hazards, 'skipped_without_gpu_effect': self.skipped, 'notes': self.notes,
                'readbacks': p['readbacks'], 'frame_graph_crosscheck': crosscheck,
                'blob_count': len(b.blobs.sizes), 'blob_bytes': sum(b.blobs.sizes.values()), 'blobs_hard_linked': b.blobs.linked}

    def _crosscheck(self, p):
        """Every converted GPU action must be the frame graph's action at the same event, same kind."""
        actions = {a['event_id']: a for a in json.loads(self.actions_json.read_text(encoding='utf-8'))}
        kinds = {'Unmap': 'Unmap', 'UpdateSubresource': 'UpdateSubresource', 'CopyResource': 'CopyResource',
                 'CopySubresourceRegion': 'CopySubresourceRegion', 'ClearRenderTargetView': 'ClearRenderTargetView',
                 'ClearDepthStencilView': 'ClearDepthStencilView', 'DrawIndexed': 'DrawIndexed', 'Draw': 'Draw',
                 'Dispatch': 'Dispatch', 'DrawInstanced': 'DrawInstanced', 'DrawIndexedInstanced': 'DrawIndexedInstanced'}
        checked, mismatched = 0, []
        for c in p['commands']:
            pr = c.get('provenance') or {}
            if 'event' not in pr or 'api' not in pr:
                continue
            a = actions.get(pr['event'])
            want = kinds.get(pr['api'])
            if a is None or a['kind'] != want:
                mismatched.append({'event': pr['event'], 'program_api': pr['api'], 'frame_graph': a and a['kind']})
            checked += 1
        return {'actions_json': str(self.actions_json.resolve()), 'checked': checked, 'mismatched': mismatched}


def parse_readback(text):
    """NAME:EVENT:RESOURCE[:SUBRESOURCE[:ORIENTATION]]"""
    parts = text.split(':')
    if not 3 <= len(parts) <= 5:
        raise argparse.ArgumentTypeError('readback must be NAME:EVENT:RESOURCE[:SUB[:ORIENTATION]]')
    row = {'name': parts[0], 'event': int(parts[1]), 'resource': int(parts[2])}
    if len(parts) > 3:
        row['subresource'] = int(parts[3])
    if len(parts) > 4:
        row['orientation'] = parts[4]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', type=Path, required=True, help='official RenderDoc zip.xml export')
    ap.add_argument('--zip', type=Path, help='companion ZIP (default: xml path without .xml)')
    ap.add_argument('--inspect', type=Path, required=True, help='official driver action listing (inspect stdout)')
    ap.add_argument('--shader-manifest', type=Path, action='append', default=[])
    ap.add_argument('--frame-graph-actions', type=Path, help='frame graph actions.json for an event/kind cross-check')
    ap.add_argument('--through-event', type=int, required=True, help='last API event included (inclusive)')
    ap.add_argument('--readback', type=parse_readback, action='append', default=[])
    ap.add_argument('--label', default='')
    ap.add_argument('--link-blobs-from', type=Path, action='append', default=[],
                    help='existing program directory whose identical blobs are hard-linked instead of rewritten')
    ap.add_argument('--out', type=Path, required=True, help='fresh program directory')
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError('Fresh program directory required')
    conv = Converter(a.xml, a.zip or a.xml.with_suffix(''), a.inspect, a.shader_manifest, a.frame_graph_actions)
    program, report = conv.convert(a.out, a.through_event, a.readback, a.label, a.link_blobs_from)
    print(json.dumps({k: report[k] for k in ('through_event', 'commands', 'ops', 'objects', 'initial_state_classes',
                                             'skipped_without_gpu_effect')} |
                     {'hazards': len(report['hazard_resolutions']),
                      'crosscheck_mismatches': None if not report['frame_graph_crosscheck'] else
                      len(report['frame_graph_crosscheck']['mismatched'])}))


if __name__ == '__main__':
    main()
