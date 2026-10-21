"""Bind a new capture to its own FramePairProbe export and compare skin bytes.

CPU only (no game, no GPU). Inputs: the official zip.xml export of the capture
saved by the probe's TriggerCapture, the probe window directory and its nonce.

    python tools/offline_renderer/pair_skin_capture.py --xml CAP.zip.xml \
      --window SESSION/window_<nonce> --nonce <hex32> --fxc FXC.exe --out FRESH_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import gpu_skin_inputs as g  # noqa: E402
from src.render import paired_skin as ps  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--window', type=Path, required=True)
    ap.add_argument('--nonce', required=True)
    ap.add_argument('--fxc', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    nonce = bytes.fromhex(args.nonce)
    index = g.CaptureIndex(args.xml)
    writes, draws = g.simulate(index)
    win = ps.ProbeWindow(args.window)
    markers = ps.find_markers(index, nonce)
    own = [m for m in markers if m['nonce_ok']]
    frames_in_window = sorted(win.by_frame)
    result = {'xml_sha256': index.xml_sha256, 'zip_sha256': index.zip_sha256, 'markers': markers,
              'window_frames': frames_in_window, 'pairing': None}
    if len(own) != 1 or own[0]['frame_count'] not in win.by_frame:
        result['pairing'] = {'accepted': False, 'reason': f'{len(own)} in-frame markers with this nonce'}
        (args.out / 'pairing.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
        raise g.SkinPathError(result['pairing']['reason'])
    frame = own[0]['frame_count']
    surfaces = win.surfaces()
    bindings = g.bind_surfaces(index, draws, surfaces)
    pairs = {(b['skinned_vb'], b['index_buffer']) for b in bindings if b['status'] == 'bound'}
    head = [d for d in draws if (d['vbs'][0], d['ib']) in pairs]
    usage = {}
    for vs in sorted({d['shaders']['VS'] for d in head}):
        usage[vs] = g.object_matrix_uses(g.disassemble(index.shader_bytes(vs), args.fxc, args.out / 'shaders', f'vs_{vs}'))
    points, _ = g.upload_points(index, writes, draws, bindings, usage)
    poses = g.label_poses(index, writes, points)
    renderers = {r['id']: r for r in win.static['renderers']}
    skin_rows = []
    for p in points:
        if not p['role'].startswith('skinned'):
            continue
        b = bindings[p['surfaces'][0]]
        rid = surfaces[p['surfaces'][0]]['renderer_id']
        mesh = win.meshes[rid]
        original = index.blob(writes[p['write_id']]['buffer'])
        per_frame = {}
        for fc, fr in win.by_frame.items():
            mats = ps.frame_matrices(win, fr, renderers[rid])
            variants = ps.skin_variants(mesh, mats, fr['W'][rid])
            per_frame[fc] = {k: ps.compare_words(original, v) for k, v in variants.items()}
        best = max(((fc, k, v) for fc, rows in per_frame.items() for k, v in rows.items()),
                   key=lambda x: (x[2]['equal_words'], -x[2]['max_abs_diff']))
        skin_rows.append({'surface': b['mesh_name'], 'role': p['role'], 'write': p['write_id'], 'chunk_index': p['chunk_index'],
                          'pose': p.get('skin_pose'), 'best': {'frame_count': best[0], 'variant': best[1], **best[2]},
                          'marker_frame': frame, 'marker_frame_results': per_frame[frame],
                          'previous_frame_results': per_frame.get(frame - 1)})
    matrix_rows = []
    for p in points:
        if p['role'] != 'per_draw_object_matrices':
            continue
        data = index.blob(writes[p['write_id']]['buffer'])
        rid = surfaces[p['surfaces'][0]]['renderer_id']
        rows = {}
        for fc, fr in win.by_frame.items():
            mats = ps.frame_matrices(win, fr, renderers[rid])
            o2w = ps._trs(mats['root_pos'], mats['root_rot'])
            rows[fc] = ps.compare_words(data[0:64], g.matrix_bytes(o2w))
        best = max(rows.items(), key=lambda kv: kv[1]['equal_words'])
        matrix_rows.append({'chunk_index': p['chunk_index'], 'pose': p['object_pose'], 'best_frame': best[0], **best[1]})
    result['pairing'] = {'accepted_marker_frame': frame, 'marker': own[0], 'poses': poses,
                         'bindings': [{k: b.get(k) for k in ('mesh_name', 'status', 'skinned_vb', 'index_buffer')} for b in bindings],
                         'skin': skin_rows, 'object_to_world': matrix_rows}
    (args.out / 'pairing.json').write_text(json.dumps(result, indent=1, default=g._json), encoding='utf-8')
    print(json.dumps({'marker_frame': frame, 'skin': [(r['surface'], r['role'], r['best']['frame_count'], r['best']['variant'],
                                                       r['best']['bit_exact'], r['best']['max_ulp']) for r in skin_rows]}, default=g._json))


if __name__ == '__main__':
    main()
