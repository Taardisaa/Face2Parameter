"""End-to-end source chain acceptance against a paired RDC (CPU only, no game).

card + head prefab + native59/body (exact ShapeInfo ports) -> CollectBaselineCo
-> ABMX Apply (steady state) -> bone local TRS -> UnityPlayer skinning
(emulated) -> skinned VB / ObjectToWorld / WorldToObject bytes, compared with
the capture's own uploads for the marker frame (current) and marker-1 (previous).

    python tools/offline_renderer/forward_chain_acceptance.py --session SESSION \
      --nonce N --parse parse_dir --effect-context CTX.json --fxc FXC --out FRESH
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
from src.render.unity_skin_emu import UnityPlayerEmu, skin_from_locals  # noqa: E402
from src.hs2_offline_state import CapturedHeadForward  # noqa: E402
from tools.offline_renderer.paired_forward_context import build  # noqa: E402

F = np.float32


def frame_locals(model_result, win, geo):
    """Forward bone locals keyed by probe transform index (+ ancestors)."""
    tf = win.static['transforms']
    local = model_result['bone_local']
    by_id = {t['id']: t for t in geo['transforms']}
    parent, loc = {}, {}
    for i, t in enumerate(tf):
        L = local[t['id']]
        loc[i] = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F), np.asarray(L['local_scale'], F))
        parent[i] = t['parent'] if t['parent'] >= 0 else ('anc', by_id[t['id']]['parent_id'])
    # ancestors above the probe root, from the forward model's own transform table
    for key in list(parent.values()):
        cur = key
        while isinstance(cur, tuple) and cur not in loc:
            tid = cur[1]
            L = local[tid]
            loc[cur] = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F), np.asarray(L['local_scale'], F))
            pid = by_id[tid]['parent_id']
            parent[cur] = ('anc', pid) if pid is not None else None
            cur = parent[cur]
    return parent, loc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--session', type=Path, required=True)
    ap.add_argument('--nonce', required=True)
    ap.add_argument('--parse', required=True)
    ap.add_argument('--effect-context', type=Path, required=True)
    ap.add_argument('--fxc', type=Path, required=True)
    ap.add_argument('--apply-count', type=int, default=2)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    S = args.session
    win = ps.ProbeWindow(S / f'window_{args.nonce}')
    index = g.CaptureIndex(S / args.parse / 'capture.zip.xml')
    writes, draws = g.simulate(index)
    marker = [m for m in ps.find_markers(index, bytes.fromhex(args.nonce)) if m['nonce_ok']]
    if len(marker) != 1:
        raise g.SkinPathError('Exactly one in-frame marker required')
    F0 = marker[0]['frame_count']
    surfaces = win.surfaces()
    bindings = g.bind_surfaces(index, draws, surfaces)
    pairs = {(b['skinned_vb'], b['index_buffer']) for b in bindings if b['status'] == 'bound'}
    usage = {vs: g.object_matrix_uses(g.disassemble(index.shader_bytes(vs), args.fxc, args.out / 'shaders', f'vs_{vs}'))
             for vs in sorted({d['shaders']['VS'] for d in draws if (d['vbs'][0], d['ib']) in pairs})}
    points, _ = g.upload_points(index, writes, draws, bindings, usage)
    emu = UnityPlayerEmu()
    renderers = {r['id']: r for r in win.static['renderers']}
    tf = win.static['transforms']
    chain = set()
    for r in renderers.values():
        if r['id'] in win.meshes:
            for b in r['bones']:
                j = b
                while j >= 0 and j not in chain:
                    chain.add(j)
                    j = tf[j]['parent']
    report = {'marker_frame': F0, 'apply_count': args.apply_count, 'frames': {}, 'uploads': []}
    produced = {}
    for fc in (F0 - 1, F0):
        geo, ctx, info = build(S / 'prepare/geometry_frames.json', S / 'prepare/loaded_state.json', args.effect_context, win, fc)
        model = CapturedHeadForward(geo, ctx)
        result = model.evaluate(apply_count=args.apply_count, baseline_mode='card_load')
        T = win.by_frame[fc]['T']
        diff = []
        for i in sorted(chain):
            L = result['bone_local'][tf[i]['id']]
            for k, (a, b) in (('local_position', (0, 3)), ('local_rotation_xyzw', (3, 7)), ('local_scale', (7, 10))):
                if not np.array_equal(np.asarray(L[k], F).view(np.uint32), T[i, a:b].view(np.uint32)):
                    diff.append([tf[i]['name'], k])
        parent, loc = frame_locals(result, win, geo)
        out = {}
        for si, (b, s) in enumerate(zip(bindings, surfaces)):
            if b['status'] != 'bound':
                continue
            r = renderers[s['renderer_id']]
            m = dict(win.meshes[r['id']])
            blended = ps.unity_blend(win.meshes[r['id']], win.by_frame[fc]['W'][r['id']])
            m['v'], m['nrm'], m['t'] = blended[:, 0:3], blended[:, 3:6], blended[:, 6:10]
            out[si] = skin_from_locals(emu, parent, loc, r['bones'], r['root_bone'], m)
        produced[fc] = out
        report['frames'][fc] = {'context': info, 'chain_bones': len(chain), 'local_fields_differing': diff,
                                'effect_decisions': result['effect_decisions']}
    # compare every recorded upload consumed by head draws
    poses = g.label_poses(index, writes, points)
    o2w_frame = {}
    for fc, out in produced.items():
        o2w_frame[next(iter(out.values()))[1]] = fc
    for p in points:
        data = index.blob(writes[p['write_id']]['buffer'])
        row = {'role': p['role'], 'chunk_index': p['chunk_index'], 'surfaces': p['surfaces']}
        if p['role'] == 'skinned_vertices_current':
            row['bit_exact'] = data == produced[F0][p['surfaces'][0]][0]
        elif p['role'] == 'skinned_vertices_previous':
            row['bit_exact'] = data == produced[F0 - 1][p['surfaces'][0]][0]
        elif p['role'] == 'per_draw_object_matrices':
            si = p['surfaces'][0]
            row['o2w_bit_exact'] = data[0:64] == produced[F0][si][1]
            ident = np.array_equal(np.frombuffer(data[64:128], F).reshape(4, 4), np.eye(4, dtype=F))
            row['w2o'] = 'identity' if ident else bool(data[64:128] == produced[F0][si][2])
        elif p['role'] == 'motion_vector_previous_matrix':
            off = p['extra']['matrix_byte_offset'][0]
            row['bit_exact'] = data[off:off + 64] == produced[F0 - 1][p['surfaces'][0]][1]
        report['uploads'].append(row)
    summary = {}
    for row in report['uploads']:
        key = row['role']
        ok = row.get('bit_exact', row.get('o2w_bit_exact'))
        if key == 'per_draw_object_matrices':
            ok = ok and row['w2o'] in (True, 'identity')
        summary.setdefault(key, [0, 0])
        summary[key][0] += bool(ok)
        summary[key][1] += 1
    report['summary'] = summary
    (args.out / 'acceptance.json').write_text(json.dumps(report, indent=1, default=g._json), encoding='utf-8')
    print(json.dumps({'marker_frame': F0, 'summary': summary,
                      'local_fields_differing': {k: len(v['local_fields_differing']) for k, v in report['frames'].items()}}))


if __name__ == '__main__':
    main()
