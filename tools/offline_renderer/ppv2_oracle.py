"""Exact PostProcessing-v2 oracle records from an official RenderDoc XML+ZIP export.

CPU only.  Builds the existing source-derived ``frame_graph.FrameGraph`` (same
chunk interpretation, Map/Unmap/initial-content rules) and writes, for every
PPv2-owned action of every capture-camera segment, the complete bound state the
generator in ``src/render/hs2pipe/ppv2.py`` must reproduce:

* shader identity (executed DXBC SHA-256 + installed-export matches),
* every declared constant buffer's exact bytes at the action (hex) and the
  CB resource descriptor,
* SRV / UAV / RTV views (verbatim view descriptors) and their base resource
  descriptors, sampler descriptors, viewport, rasterizer/blend/depth states,
* IA (fullscreen triangle mesh buffers, strides, topology, input layout),
* dispatch group counts / draw arguments,
* clears and copies inside the PPv2 range, and the curves-texture upload.

Nothing is decoded into settings here; it is the byte oracle for the
generator diff (``ppv2_diff.py``).  Output must be a fresh directory.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frame_graph as fg  # noqa: E402

class StateRecordingGraph(fg.FrameGraph):
    """FrameGraph that also keeps the raw pipeline state at every draw/dispatch.

    The base graph keeps only the states it needs for dependency tracking; the
    generator diff needs every bound state object (depth-stencil state even with
    no DSV, blend factor, sample mask, samplers per stage)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.raw_states = {}

    def _snapshot(self, ev):
        st = self.state
        self.raw_states[ev] = {
            'OM': {k: st['OM'][k] for k in ('BlendState', 'BlendFactor', 'SampleMask', 'DepthStencilState', 'StencRef')
                   if k in st['OM']},
            'RS': {k: st['RS'][k] for k in ('State', 'NumViews', 'NumScissors') if k in st['RS']},
            'samplers': {stage: list(st[stage]['Samplers']) for stage in ('VS', 'PS', 'CS')}}

    def _geometry(self, method, f, ev):
        """Exact IB bytes of the draw's index range and VB bytes of the addressed vertices."""
        ia = self.state['IA']
        if not method.startswith('DrawIndexed'):
            return None
        size = 2 if ia['IndexFormat'] == 57 else 4
        s = ia['IndexOffset'] + f['StartIndexLocation'] * size
        e = s + f['IndexCount'] * size
        ib = self.buffer(ia['IndexBuffer']).read(self.blobs, s, e)
        out = {'index_bytes': [s, e], 'index_hex': ib.hex() if ib is not None else None, 'vertex': []}
        if ib is None:
            return out
        idx = struct.unpack('<%d%s' % (f['IndexCount'], 'H' if size == 2 else 'I'), ib)
        lo, hi = min(idx) + f['BaseVertexLocation'], max(idx) + f['BaseVertexLocation']
        for slot, vb in enumerate(ia['VBs']):
            if not vb:
                continue
            stride, off = ia['Strides'][slot], ia['Offsets'][slot]
            vs, ve = off + lo * stride, off + (hi + 1) * stride
            width = self.resources[vb]['descriptor']['ByteWidth']
            if ve > width:
                continue
            data = self.buffer(vb).read(self.blobs, vs, ve)
            out['vertex'].append({'slot': slot, 'resource': vb, 'bytes': [vs, ve], 'hex': data.hex() if data is not None else None,
                                  'producers': self.buffer(vb).producers(vs, ve)})
        return out

    def _draw(self, method, f, idx, ev):
        self._snapshot(ev)
        self.raw_states[ev]['geometry'] = self._geometry(method, f, ev)
        return super()._draw(method, f, idx, ev)

    def _dispatch(self, f, idx, ev):
        self._snapshot(ev)
        return super()._dispatch(f, idx, ev)


PPV2_LABELS = ('ambient_occlusion_msvo', 'post_copy_std', 'post_depth_of_field', 'color_grading_lut_bake',
               'post_uber', 'post_smaa', 'post_final_pass')
CURVES_LABEL = 'post_curves_texture_copy'


def _desc(graph, ident):
    if not ident:
        return None
    r = graph.resources.get(ident)
    if r is None:
        return {'id': ident, 'missing': True}
    row = {'id': ident, 'kind': r['kind'], 'descriptor': r.get('descriptor'), 'debug_names': r.get('debug_names', [])}
    if 'base' in r:
        row['base'] = r['base']
    return row


def _cb_bytes(graph, event, stage, cb):
    if cb.get('unbound'):
        return None
    snap = fg.BufferContents(graph.resources[cb['resource']]['descriptor']['ByteWidth'], None, 'none')
    snap.segments = graph._cb_snapshots[(event, stage, cb['slot'])]
    data = snap.read(graph.blobs, cb['bytes'][0], cb['bytes'][1])
    return data.hex() if data is not None else None


def _stage_state(graph, stage):
    st = graph.state[stage]
    return {'samplers': list(st['Samplers'])}


def action_record(graph, a, referenced):
    rec = {'event_id': a['event_id'], 'kind': a['kind'], 'chunk_index': a['chunk_index']}
    if a['kind'] in ('Draw', 'DrawIndexed', 'Dispatch'):
        rec['arguments'] = a['arguments']
        rec['shaders'] = {}
        for stage, s in a['shaders'].items():
            srow = {'resource': s['resource'], 'debug_names': s['debug_names'], 'dxbc_sha256': s['dxbc_sha256'],
                    'source_matches': [{k: m[k] for k in ('shader', 'subshader', 'pass', 'stage', 'blob_index', 'keywords')}
                                       for m in s['source_matches']],
                    'constant_buffers': [], 'srvs': [], 'samplers': []}
            for cb in s['constant_buffers']:
                srow['constant_buffers'].append({'slot': cb['slot'], 'name': cb.get('name'), 'resource': cb.get('resource'),
                                                 'bytes_range': cb.get('bytes'), 'declared_float4': cb['declared_float4'],
                                                 'data_hex': _cb_bytes(graph, a['event_id'], stage, cb),
                                                 'producers': cb.get('producers')})
                referenced.add(cb.get('resource'))
            for srv in s['srvs']:
                srow['srvs'].append({'slot': srv['slot'], 'name': srv.get('name'), 'view': srv.get('view'),
                                     'resource': srv.get('resource'), 'subresources': srv.get('subresources')})
                referenced.update([srv.get('view'), srv.get('resource')])
            for smp in s['samplers']:
                srow['samplers'].append(smp)
                referenced.add(smp['sampler'])
            rec['shaders'][stage] = srow
    if a['event_id'] in getattr(graph, 'raw_states', {}):
        raw = graph.raw_states[a['event_id']]
        rec['raw_state'] = raw
        referenced.update([raw['OM'].get('BlendState'), raw['OM'].get('DepthStencilState'), raw['RS'].get('State')])
        for smps in raw['samplers'].values():
            referenced.update(smps)
    if a['kind'] == 'Dispatch':
        rec['uavs'] = [{'slot': u['slot'], 'name': u.get('name'), 'view': u.get('view'), 'resource': u.get('resource'),
                        'subresources': u.get('subresources')} for u in a['uavs']]
        for u in a['uavs']:
            referenced.update([u.get('view'), u.get('resource')])
    if a['kind'] in ('Draw', 'DrawIndexed'):
        rec['topology'] = a['topology']
        rec['input_layout'] = a['input_layout']
        rec['vertex_buffers'] = [{k: v[k] for k in ('slot', 'resource', 'stride', 'offset', 'semantics')} for v in a['vertex_buffers']]
        for v in a['vertex_buffers']:
            referenced.add(v['resource'])
        if 'index_buffer' in a:
            rec['index_buffer'] = {k: a['index_buffer'][k] for k in ('resource', 'format', 'offset', 'bytes')}
            referenced.add(a['index_buffer']['resource'])
        rec['render_targets'] = [{k: t[k] for k in ('slot', 'view', 'resource', 'subresources', 'write_mask', 'blend_enable', 'blend', 'view_format')}
                                 for t in a['render_targets']]
        for t in a['render_targets']:
            referenced.update([t['view'], t['resource']])
        rec['depth'] = a['depth'] and {k: a['depth'][k] for k in a['depth'] if k != 'prior_content'}
        if a['depth']:
            referenced.update([a['depth']['view'], a['depth']['resource'], a['depth']['state']])
        rec['viewports'] = a['viewports']
        rec['scissors'] = a['scissors']
        rec['rasterizer'] = a['rasterizer']
        rec['blend_state'] = a['blend_state']
        referenced.add(a['blend_state'])
    if a['kind'].startswith('Clear'):
        rec.update(view=a['view'], resource=a['resource'], values=a['values'])
        referenced.update([a['view'], a['resource']])
    if a['kind'] in ('CopySubresourceRegion', 'CopyResource'):
        rec.update(source=a['source'], destination=a['destination'], box=a.get('box'),
                   destination_offset=a.get('destination_offset'))
        referenced.update([a['source'], a['destination']])
    if a['kind'] == 'Unmap':
        rec.update(resource=a['resource'], subresource=a['subresource'], map_type=a['map_type'],
                   written_bytes=a['written_bytes'])
        data = graph.blobs.read(a['blob']) if a.get('blob') else None
        rec['data_hex'] = data.hex() if data is not None else None
    return rec


def run(xml, zip_path, inspect, manifests, out, boundary_resource=59322):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    graph = StateRecordingGraph(fg.load_chunks(xml), fg.load_driver_listing(inspect) if inspect else None,
                          fg.Blobs(zip_path), fg.ShaderLibrary(manifests)).build()
    passes = fg.group_passes(graph)
    segments = fg.segment_passes(passes, graph, boundary_resource)
    by_event = {a['event_id']: a for a in graph.actions if a['event_id'] is not None}
    ordered = [a for a in graph.actions if a['event_id'] is not None]
    rows = []
    referenced = set()
    for si, seg in enumerate(segments):
        sp = [passes[i] for i in seg]
        pp = [p for p in sp if p['label'] in PPV2_LABELS]
        if not pp:
            continue
        lo, hi = pp[0]['first_event'], pp[-1]['last_event']
        curves = [p for p in sp if p['label'] == CURVES_LABEL]
        events = []
        for p in curves:
            # The staging Unmap feeding the copy is part of Texture2D.Apply.
            copy = by_event[p['first_event']]
            src = copy['source']
            maps = [a for a in ordered if a['kind'] == 'Unmap' and a['resource'] == src and a['event_id'] < copy['event_id']]
            if maps:
                events.append(maps[-1]['event_id'])
            events.append(copy['event_id'])
        # Every action inside the PPv2 range belongs to the PostProcessLayer command
        # buffers except the transparent/GrabPass forward work between MSVO composite
        # and CopyStd, which is excluded by label.
        foreign = {e for p in sp if lo <= p['first_event'] <= hi and p['label'] not in PPV2_LABELS for e in p['events']}
        for a in ordered:
            if lo <= a['event_id'] <= hi and a['event_id'] not in foreign:
                if a['kind'] == 'Unmap':
                    continue  # CB contents are captured per action through the snapshots
                events.append(a['event_id'])
        records = [action_record(graph, by_event[e], referenced) for e in events]
        rows.append({'segment': si, 'first_event': lo, 'last_event': hi,
                     'labels': [(p['label'], p['first_event'], p['last_event']) for p in sp if lo <= p['first_event'] <= hi],
                     'actions': records})
    referenced.discard(None)
    referenced.discard(0)
    resources = {str(i): _desc(graph, i) for i in sorted(referenced)}
    # views reference base resources
    for i in list(resources):
        b = resources[i].get('base') if resources[i] else None
        if b and str(b) not in resources:
            resources[str(b)] = _desc(graph, b)
    out.mkdir(parents=True)
    provenance = {'xml': str(Path(xml).resolve()), 'xml_sha256': fg.digest(Path(xml).read_bytes()),
                  'zip': str(Path(zip_path).resolve()), 'inspect': str(Path(inspect).resolve()) if inspect else None,
                  'shader_manifests': [str(Path(m).resolve()) for m in manifests],
                  'tool_sha256': fg.digest(Path(__file__).read_bytes()),
                  'frame_graph_tool_sha256': fg.digest(Path(fg.__file__).read_bytes())}
    (out / 'oracle.json').write_text(json.dumps({'schema': 'hs2_ppv2_oracle/1', 'provenance': provenance,
                                                 'segments': rows, 'resources': resources}, default=fg.jsonable),
                                     encoding='utf-8')
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--zip', type=Path)
    ap.add_argument('--inspect', type=Path)
    ap.add_argument('--shader-manifest', type=Path, action='append', default=[])
    ap.add_argument('--boundary-resource', type=int, default=59322)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    rows = run(args.xml, args.zip or args.xml.with_suffix(''), args.inspect, args.shader_manifest, args.out,
               args.boundary_resource)
    print(json.dumps({'segments': len(rows), 'actions': sum(len(r['actions']) for r in rows)}))


if __name__ == '__main__':
    main()
