"""Bind a new capture to its own FramePairProbe export and compare skin bytes.

CPU only (no game, no GPU). Inputs: the official zip.xml export of the capture
saved by the probe's TriggerCapture, the probe window directory and its nonce.

    python tools/offline_renderer/pair_skin_capture.py --xml CAP.zip.xml \
      --window SESSION/window_<nonce> --nonce <hex32> --fxc FXC.exe --out FRESH_DIR

Every exported ChaControl skinned mesh (probe Meshes=ALL) is paired; the role labels and
the skinning mechanism live in src/render/gpu_skin_inputs.py and paired_skin.py.

Object matrices are compared against every exported frame and, for probe >= 6.1.0
renderstyle windows, also against the character transforms each bridge render's
pre_cull record carries (the PoseFreeze pose that render culls and draws with,
head_skinning.md §10.1). Frame and bridge-render hits are reported separately.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import gpu_skin_inputs as g  # noqa: E402
from src.render import paired_skin as ps  # noqa: E402


def bridge_pre_cull_poses(window: Path, frames_by_index, captured_frame, transform_count):
    """{ordinal: {'T': (n, 33) float32}} from the captured frame's bridge pre_cull records.

    Returns (poses, status). Fails closed on any malformed or partial export; a window
    without bridge renders or from a probe that did not export them gives ({}, reason).
    """
    receipt = json.loads((window / 'capture_receipt.json').read_text(encoding='utf-8'))
    if receipt.get('mode') != 'renderstyle':
        return {}, 'not a renderstyle window'
    if captured_frame not in frames_by_index:
        raise g.SkinPathError('captured frame %s has no probe export index' % captured_frame)
    path = window / 'frames' / ('renders_%03d.json' % frames_by_index.index(captured_frame))
    records = json.loads(path.read_text(encoding='utf-8'))['records']
    pre = [r for r in records if r.get('phase') == 'pre_cull' and r.get('bridge_render')]
    with_t = [r for r in pre if 'character_transforms' in (r.get('data') or {})]
    if not with_t:
        return {}, 'bridge pre_cull records carry no character_transforms (probe %s)' % receipt.get('probe')
    if len(with_t) != len(pre):
        raise g.SkinPathError('character_transforms missing from %d of %d bridge pre_cull records' % (len(pre) - len(with_t), len(pre)))
    poses = {}
    for r in pre:
        o = r['bridge_render']['ordinal']
        ct = r['data']['character_transforms']
        if o in poses:
            raise g.SkinPathError('duplicate bridge ordinal %d' % o)
        if r['frame_count'] != captured_frame:
            raise g.SkinPathError('bridge render %d is in frame %s, not %s' % (o, r['frame_count'], captured_frame))
        if ct.get('order') != 'static.json transforms' or ct.get('count') != transform_count:
            raise g.SkinPathError('bridge render %d transforms are not in static.json order/count' % o)
        t = np.frombuffer(base64.b64decode(ct['transforms_f32']), np.float32)
        if t.size != transform_count * 33:
            raise g.SkinPathError('bridge render %d has %d floats, expected %d' % (o, t.size, transform_count * 33))
        poses[o] = {'T': t.reshape(transform_count, 33), 'frame_count': captured_frame, 'bridge_ordinal': o}
    if sorted(poses) != list(range(len(poses))):
        raise g.SkinPathError('bridge ordinals are not 0..n-1: %s' % sorted(poses))
    return poses, 'exported'


def object_matrix_summary(rows):
    """Counts only (no acceptance rule): rows with a bit-exact O2W / W2O from an exported frame,
    from a bridge render's pre_cull pose, and rows whose O2W matches neither source."""
    out = {'rows': len(rows)}
    for m in ('o2w', 'w2o'):
        for src in ('frames', 'bridge_renders'):
            key = '%s_bit_exact_%s' % (m, src)
            out['%s_rows_matching_%s' % (m, src)] = sum(1 for r in rows if r.get(key))
    out['o2w_rows_matching_neither'] = sum(1 for r in rows if not r.get('o2w_bit_exact_frames') and not r.get('o2w_bit_exact_bridge_renders'))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--window', type=Path, required=True)
    ap.add_argument('--nonce', required=True)
    ap.add_argument('--fxc', type=Path, required=True)
    ap.add_argument('--ancestors-geometry', type=Path, required=True,
                    help='same-session bridge geometry export with actor ancestors (local TRS above ChaControl)')
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
    receipt = json.loads((args.window / 'capture_receipt.json').read_text(encoding='utf-8'))
    bridge_poses, bridge_status = bridge_pre_cull_poses(args.window, receipt.get('frame_counts_by_index', []), frame,
                                                        len(win.static['transforms']))
    result['bridge_pre_cull_poses'] = {'status': bridge_status, 'ordinals': sorted(bridge_poses)}
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
    from src.render.unity_skin_emu import UnityPlayerEmu
    emu = UnityPlayerEmu()
    geo = json.loads(args.ancestors_geometry.read_text(encoding='utf-8'))
    gt = {t['id']: t for t in geo['transforms']}
    top_id = win.static['chara_transform_id']
    if top_id not in gt:
        raise g.SkinPathError('Ancestor export is not from this process (ChaControl id missing)')
    chain, cur = [], gt[top_id]['parent_id']
    while cur is not None:
        t = gt[cur]
        chain.append((t['local_position'], t['local_rotation_xyzw'], t['local_scale'], t['name']))
        cur = t['parent_id']
    chain.reverse()
    ancestors = [(np.asarray(a, np.float32), np.asarray(b, np.float32), np.asarray(c, np.float32)) for a, b, c, _ in chain]
    result['ancestors'] = [{'name': n, 'local': [list(map(float, a)), list(map(float, b)), list(map(float, c))]} for a, b, c, n in chain]
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
            variants['unity_emulated_prepared'] = ps.unity_exact_vertices(emu, win, fr, renderers[rid], mesh, ancestors)
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
        o2w_hits, w2o_hits = [], []
        for fc, fr in win.by_frame.items():
            o2w, w2o = ps.unity_object_matrices(emu, win, fr, renderers[rid])
            if data[0:64] == o2w:
                o2w_hits.append(fc)
            if data[64:128] == w2o:
                w2o_hits.append(fc)
        b_o2w, b_w2o = [], []
        for o, pose in bridge_poses.items():
            o2w, w2o = ps.unity_object_matrices(emu, win, pose, renderers[rid])
            if data[0:64] == o2w:
                b_o2w.append(o)
            if data[64:128] == w2o:
                b_w2o.append(o)
        identity = data[64:128] == g.matrix_bytes(np.eye(4))
        matrix_rows.append({'chunk_index': p['chunk_index'], 'pose': p['object_pose'], 'o2w_bit_exact_frames': o2w_hits,
                            'w2o_bit_exact_frames': w2o_hits, 'w2o_identity': identity,
                            'o2w_bit_exact_bridge_renders': b_o2w, 'w2o_bit_exact_bridge_renders': b_w2o})
    for p in points:
        if p['role'] != 'motion_vector_previous_matrix':
            continue
        data = index.blob(writes[p['write_id']]['buffer'])
        off = p['extra']['matrix_byte_offset'][0]
        rid = surfaces[p['surfaces'][0]]['renderer_id']
        hits = [fc for fc, fr in win.by_frame.items() if data[off:off + 64] == ps.unity_object_matrices(emu, win, fr, renderers[rid])[0]]
        b_hits = [o for o, pose in bridge_poses.items() if data[off:off + 64] == ps.unity_object_matrices(emu, win, pose, renderers[rid])[0]]
        matrix_rows.append({'chunk_index': p['chunk_index'], 'role': '_PreviousM', 'o2w_bit_exact_frames': hits,
                            'o2w_bit_exact_bridge_renders': b_hits})
    result['pairing'] = {'accepted_marker_frame': frame, 'marker': own[0], 'poses': poses,
                         'bindings': [{k: b.get(k) for k in ('mesh_name', 'status', 'skinned_vb', 'index_buffer')} for b in bindings],
                         'skin': skin_rows, 'object_to_world': matrix_rows,
                         'object_matrix_summary': object_matrix_summary(matrix_rows)}
    (args.out / 'pairing.json').write_text(json.dumps(result, indent=1, default=g._json), encoding='utf-8')
    print(json.dumps({'marker_frame': frame, 'skin': [(r['surface'], r['role'], r['best']['frame_count'], r['best']['variant'],
                                                       r['best']['bit_exact'], r['best']['max_ulp']) for r in skin_rows]}, default=g._json))


if __name__ == '__main__':
    main()
