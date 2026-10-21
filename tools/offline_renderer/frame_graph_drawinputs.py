"""Cross-check frame-graph CPU buffer provenance against official replayed bytes.

For explicitly selected existing draw events of an existing RDC, runs the
already-built MSVC ``renderdoc_replay_msvc.exe drawinputs`` (official RenderDoc
1.46 local D3D11 replay, NoOptimisation) once per event through the binary
logger, then compares every dumped vertex/index buffer byte-for-byte with the
bytes the CPU frame graph reconstructs at that event (CreateBuffer record =
capture-start contents, then ordered Unmap/Update/Copy writes).  No game,
capture, injection, deployment or retry is performed; a failed run is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .frame_graph import FrameGraph, ShaderLibrary, Blobs, load_chunks, load_driver_listing, digest
from .logged_process import run_logged


class SnapshotGraph(FrameGraph):
    def __init__(self, *args, events=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.wanted = set(events)
        self.snapshots = {}

    def _draw(self, method, f, idx, ev):
        if ev in self.wanted:
            ia = self.state['IA']
            ids = {v for v in ia['VBs'] if v}
            if ia['IndexBuffer']:
                ids.add(ia['IndexBuffer'])
            self.snapshots[ev] = {i: (self.buffer(i).width, list(self.buffer(i).segments)) for i in ids}
        super()._draw(method, f, idx, ev)


def cpu_bytes(graph, event, ident):
    width, segments = graph.snapshots[event][ident]
    from .frame_graph import BufferContents
    b = BufferContents(width, None, 'none')
    b.segments = segments
    return b.read(graph.blobs, 0, width), [{'producer': p['producer'], 'kind': p['kind'], 'bytes': p['bytes']} for p in b.producers(0, width)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rdc', type=Path, required=True)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--inspect', type=Path, required=True)
    ap.add_argument('--tool', type=Path, required=True, help='built MSVC renderdoc_replay_msvc.exe')
    ap.add_argument('--renderdoc-dir', type=Path, required=True, help='pinned portable RenderDoc 1.46 directory (renderdoc.dll)')
    ap.add_argument('--event', type=int, action='append', required=True)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    graph = SnapshotGraph(load_chunks(args.xml), load_driver_listing(args.inspect), Blobs(args.xml.with_suffix('')),
                          ShaderLibrary(), events=args.event).build()
    env = dict(os.environ)
    env['PATH'] = str(args.renderdoc_dir.resolve()) + os.pathsep + env.get('PATH', '')
    receipt = {'scope': 'existing RDC official drawinputs replay vs CPU frame-graph buffer bytes; no game/capture',
               'rdc': str(args.rdc.resolve()), 'rdc_sha256': digest(args.rdc.read_bytes()),
               'tool': str(args.tool.resolve()), 'tool_sha256': digest(args.tool.read_bytes()),
               'renderdoc_dll_sha256': digest((args.renderdoc_dir / 'renderdoc.dll').read_bytes()),
               'events': []}
    for event in args.event:
        if event not in graph.snapshots:
            raise ValueError('Event %d is not an interpreted draw' % event)
        run_dir = args.out / ('event_%d' % event)
        dump = run_dir / 'inputs'
        run_dir.mkdir()
        result = run_logged([args.tool.resolve(), 'drawinputs', args.rdc.resolve(), event, dump.resolve()], run_dir / 'process', timeout=args.timeout, env=env)
        row = {'event_id': event, 'process': result, 'buffers': []}
        if result['status'] == 'pass':
            lines = [json.loads(l) for l in (dump / 'draw_inputs.jsonl').read_bytes().splitlines() if l.strip()]
            row['replay_records'] = [l for l in lines if l['event'] != 'buffer_data']
            for l in lines:
                if l['event'] != 'buffer_data':
                    continue
                ident = l['resource_id']
                replayed = (dump / l['path']).read_bytes()
                if ident not in graph.snapshots[event]:
                    row['buffers'].append({'resource': ident, 'result': 'not_bound_in_cpu_state'})
                    continue
                cpu, producers = cpu_bytes(graph, event, ident)
                entry = {'resource': ident, 'debug_names': graph.resources[ident]['debug_names'],
                         'replayed_bytes': len(replayed), 'replayed_sha256': hashlib.sha256(replayed).hexdigest(),
                         'cpu_producers': producers}
                if cpu is None:
                    entry['result'] = 'cpu_bytes_undefined'
                else:
                    entry['cpu_sha256'] = hashlib.sha256(cpu).hexdigest()
                    entry['result'] = 'byte_identical' if cpu == replayed else 'different'
                    if cpu != replayed:
                        n = min(len(cpu), len(replayed))
                        entry['first_difference'] = next((i for i in range(n) if cpu[i] != replayed[i]), n)
                row['buffers'].append(entry)
        receipt['events'].append(row)
    receipt['all_byte_identical'] = all(b.get('result') == 'byte_identical' for e in receipt['events'] for b in e['buffers']) and \
        all(e['process']['status'] == 'pass' for e in receipt['events'])
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print(json.dumps({'all_byte_identical': receipt['all_byte_identical'],
                      'events': [(e['event_id'], e['process']['status'], [b.get('result') for b in e['buffers']]) for e in receipt['events']]}))


if __name__ == '__main__':
    main()
