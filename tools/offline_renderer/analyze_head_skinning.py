"""Static head-skinning upload analysis and baseline byte correspondence.

CPU only: reads an official RenderDoc zip.xml export (no OpenCapture/GPU), a
native live geometry export and optionally a CapturedHeadForward context.
Launches no game and modifies no capture. Writes into a fresh directory.

    python tools/offline_renderer/analyze_head_skinning.py \
      --xml .../retry_parser_v1/retry.zip.xml \
      --geometry .../native_resume_v1/live_full_frames_materials.json \
      --target-chunk 37429 --fxc "C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64/fxc.exe" \
      [--forward-geometry G.json --forward-context C.json] --out outputs/.../fresh
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import gpu_skin_inputs as g  # noqa: E402

BUFFER_DCL = re.compile(r'^dcl_(resource_buffer|resource_raw|resource_structured|uav_raw|uav_structured|uav_typed_buffer)')


def compute_inventory(index, fxc, work):
    rows = []
    dispatch = collections.Counter()
    current = None
    for c in index.chunks:
        if c.method == 'CSSetShader':
            current = c.fields['pShader']
        elif c.method == 'Dispatch':
            dispatch[current] += 1
    for sid, info in index.shaders.items():
        if info['stage'] != 'ComputeShader':
            continue
        asm = g.disassemble(index.shader_bytes(sid), fxc, work, f'cs_{sid}')
        dcls = [l.strip() for l in asm.splitlines() if l.startswith('dcl_') and not l.startswith('dcl_temps')]
        rows.append({'shader_id': sid, 'debug_name': index.name(sid), 'create_chunk': info['chunk'],
                     'bytecode_sha256': g.sha256(index.shader_bytes(sid)), 'declarations': dcls,
                     'buffer_resources': [d for d in dcls if BUFFER_DCL.match(d)],
                     'dispatch_count': dispatch.get(sid, 0)})
    geometry_shaders = [c.fields['pShader'] for c in index.chunks if c.method == 'GSSetShader']
    return {'compute_shaders': rows, 'total_dispatches': sum(dispatch.values()),
            'gs_set_calls': geometry_shaders,
            'buffer_uav_count': sum(1 for v in index.uavs.values() if v['pDesc']['ViewDimension'] == g.UAV_DIMENSION_BUFFER),
            'uav_view_dimensions': sorted({v['pDesc']['ViewDimension'] for v in index.uavs.values()}),
            'stream_out_used': any(c.method == 'SOSetTargets' for c in index.chunks)}


def compare_bytes(original: bytes, generated: bytes):
    a = np.frombuffer(original, np.float32)
    b = np.frombuffer(generated, np.float32)
    diff = np.abs(a.astype(float) - b.astype(float))
    return {'bytes': len(original), 'bit_exact': original == generated,
            'float_words': int(a.size), 'equal_words': int((a.view(np.uint32) == b.view(np.uint32)).sum()),
            'max_abs_diff': float(diff.max()) if diff.size else 0.0}


def vb_compare(original: bytes, generated: bytes):
    a = np.frombuffer(original, np.float32).reshape(-1, 10).astype(float)
    b = np.frombuffer(generated, np.float32).reshape(-1, 10).astype(float)
    pos = np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
    nrm = np.abs(a[:, 3:6] - b[:, 3:6]).max(1)
    tan = np.abs(a[:, 6:9] - b[:, 6:9]).max(1)
    out = compare_bytes(original, generated)
    out.update({'vertices': len(a), 'bit_exact_vertices': int((np.frombuffer(original, np.uint32).reshape(-1, 10) ==
                                                                np.frombuffer(generated, np.uint32).reshape(-1, 10)).all(1).sum()),
                'position_max': float(pos.max()), 'position_median': float(np.median(pos)),
                'vertices_position_gt_1e-5': int((pos > 1e-5).sum()),
                'normal_max_abs': float(nrm.max()), 'tangent_xyz_max_abs': float(tan.max()),
                'tangent_w_equal': bool(np.array_equal(a[:, 9], b[:, 9]))})
    return out, pos


def bone_attribution(mesh, pos_err, threshold=1e-5):
    idx = np.asarray(mesh['source']['bone_indices'])
    w = np.asarray(mesh['source']['bone_weights'])
    bad = np.where(pos_err > threshold)[0]
    count = collections.Counter()
    for v in bad:
        for k in range(4):
            if w[v, k] > 0:
                count[mesh['bone_names'][idx[v, k]]] += 1
    return [{'bone': b, 'vertices_over_threshold': n} for b, n in count.most_common(12)]


def mouth_bone_diagnosis(mesh, state, original, root):
    """Diagnostic least squares of per-bone affine deltas (never used to patch)."""
    a = np.frombuffer(original, np.float32).reshape(-1, 10).astype(float)
    inv = np.linalg.inv(root)
    pv = (np.c_[state.world_vertices, np.ones(len(state.world_vertices))] @ inv.T)[:, :3]
    r = (a[:, :3] - pv)
    idx = np.asarray(mesh['source']['bone_indices'])
    w = np.asarray(mesh['source']['bone_weights'])
    bad = np.linalg.norm(r, axis=1) > 1e-5
    cand = sorted({int(idx[v, k]) for v in np.where(bad)[0] for k in range(4) if w[v, k] > 0})
    if not cand:
        return []
    col = {b: i for i, b in enumerate(cand)}
    A = np.zeros((len(pv) * 3, len(cand) * 12))
    for v in range(len(pv)):
        for k in range(4):
            b = int(idx[v, k])
            if w[v, k] > 0 and b in col:
                ph = np.r_[pv[v], 1.0] * w[v, k]
                for ax in range(3):
                    A[v * 3 + ax, col[b] * 12 + ax * 4: col[b] * 12 + ax * 4 + 4] += ph
    sol = np.linalg.lstsq(A, r.ravel(), rcond=None)[0]
    rows = []
    for b in cand:
        D = sol[col[b] * 12: col[b] * 12 + 12].reshape(3, 4)
        rows.append({'bone': mesh['bone_names'][b], 'linear_max': float(np.abs(D[:, :3]).max()),
                     'translation': D[:, 3].tolist()})
    rows.sort(key=lambda r_: -max(r_['linear_max'], max(abs(x) for x in r_['translation'])))
    return {'residual_before': float(np.abs(r).max()), 'residual_after_per_bone_affine': float(np.abs(A @ sol - r.ravel()).max()),
            'bones': rows[:10], 'scope': 'diagnostic attribution of a cross-frame bone-state difference only'}


def merge_loaded_frames(geometry, frames_geometry):
    """Transfer loaded API blendshape frames between exports of one sharedMesh.

    Allowed only when renderer path, source geometry hash, blendshape names and
    frame weights agree; position deltas present in both must be identical.
    """
    out = copy.deepcopy(geometry)
    for mesh, other in zip(out['meshes'], frames_geometry['meshes']):
        if (mesh['renderer_path'], mesh['source_geometry_sha256']) != (other['renderer_path'], other['source_geometry_sha256']):
            raise g.SkinPathError('Different sharedMesh/renderer; frames not transferable')
        if [s['name'] for s in mesh['blendshapes']] != [s['name'] for s in other['blendshapes']]:
            raise g.SkinPathError('Blendshape list differs')
        for s, t in zip(mesh['blendshapes'], other['blendshapes']):
            if [f['weight'] for f in s['frames']] != [f['weight'] for f in t['frames']]:
                raise g.SkinPathError('Blendshape frame weights differ')
            for f, h in zip(s['frames'], t['frames']):
                if 'delta_vertices' in f and f['delta_vertices'] != h['delta_vertices']:
                    raise g.SkinPathError('Recorded position deltas differ')
                for key in ('delta_vertices', 'delta_normals', 'delta_tangents'):
                    f[key] = h[key]
        for key in ('normals', 'tangents'):
            if mesh['source'][key] != other['source'][key]:
                raise g.SkinPathError('Source attribute differs: ' + key)
    out['include_blendshape_frames'] = True
    out['loaded_frames_provenance'] = {'frames_source_frame': frames_geometry['frame_count'],
                                       'frames_source_utc': frames_geometry['captured_utc'],
                                       'rule': 'identical sharedMesh hash/renderer path/blendshape list'}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--target-chunk', type=int, required=True)
    ap.add_argument('--fxc', type=Path, required=True)
    ap.add_argument('--forward-geometry', type=Path)
    ap.add_argument('--forward-context', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    index = g.CaptureIndex(args.xml)
    writes, draws = g.simulate(index)
    inventory = compute_inventory(index, args.fxc, args.out / 'shaders')
    geometry = json.loads(args.geometry.read_text(encoding='utf-8'))
    surfaces = [{'mesh_name': m['mesh_name'], 'renderer_path': m['renderer_path'], 'vertex_count': m['vertex_count'],
                 'active_in_hierarchy': m['active_in_hierarchy'], 'source': m['source']} for m in geometry['meshes']]
    bindings = g.bind_surfaces(index, draws, surfaces)
    pairs = {(b['skinned_vb'], b['index_buffer']) for b in bindings if b['status'] == 'bound'}
    head = [d for d in draws if (d['vbs'][0], d['ib']) in pairs]
    usage = {}
    for vs in sorted({d['shaders']['VS'] for d in head}):
        asm = g.disassemble(index.shader_bytes(vs), args.fxc, args.out / 'shaders', f'vs_{vs}')
        usage[vs] = g.object_matrix_uses(asm)
        usage[vs]['debug_name'] = index.name(vs)
        if not usage[vs]['object_matrices']:
            raise g.SkinPathError(f'Head VS {vs} has no recognised object matrix use')
    points, head_map = g.upload_points(index, writes, draws, bindings, usage, target_chunk=args.target_chunk)
    poses = g.label_poses(index, writes, points)
    skin_unmaps = sorted(p['chunk_index'] for p in points if p['role'] == 'skinned_vertices_current' and p['write_kind'] == 'unmap')

    # ---- internal capture consistency (static, bit level) ----
    consistency = {}
    by_surface = collections.defaultdict(dict)
    for p in points:
        if p['role'].startswith('skinned'):
            by_surface[p['surfaces'][0]][(p['role'], p['write_kind'])] = p
    for si, rows in by_surface.items():
        cur0 = rows.get(('skinned_vertices_current', 'capture_start_contents'))
        prev0 = rows.get(('skinned_vertices_previous', 'capture_start_contents'))
        upd = rows.get(('skinned_vertices_current', 'unmap'))
        entry = {'mesh_name': surfaces[si]['mesh_name']}
        blob = lambda p: index.blob(writes[p['write_id']]['buffer'])  # noqa: E731
        if prev0 and upd:
            entry['previous_start_vs_in_frame_skin'] = compare_bytes(blob(prev0), blob(upd))
        if cur0 and upd:
            entry['current_start_vs_in_frame_skin'] = compare_bytes(blob(cur0), blob(upd))
        consistency[si] = entry
    w2o_checks = []
    for p in points:
        if p['role'] == 'per_draw_object_matrices' and p['world_to_object'].startswith('inverse:'):
            data = index.blob(writes[p['write_id']]['buffer'])
            inv_pose = p['world_to_object'].split(':')[1]
            M = np.asarray(poses[inv_pose])
            f32 = np.linalg.inv(M.astype(np.float32)).astype(np.float32)
            w2o_checks.append(data[64:128] == g.matrix_bytes(np.linalg.inv(M)) or data[64:128] == f32.T.tobytes())
    consistency['w2o_bit_exact_vs_numpy_inverse'] = {'checked': len(w2o_checks), 'bit_exact': int(sum(w2o_checks))}

    # ---- baseline correspondence with the live export state (other frame) ----
    baseline = {'state_source': str(args.geometry), 'state_frame': geometry['frame_count'],
                'capture_card_state_note': 'export and capture were produced from the same card state hash; different process/frame',
                'surfaces': {}, 'matrices': {}}
    export_states = {}
    for row in bindings:
        if row['status'] != 'bound':
            continue
        si = row['surface_index']
        st = g.surface_state_from_export(geometry, si)
        export_states[si] = st
        root = g.root_frame(st.root_world_matrix, st.root_world_rotation_xyzw)
        payload = g.skinned_vb_bytes(st.world_vertices, st.world_normals, st.world_tangents, st.tangent_handedness, root)
        mesh = geometry['meshes'][si]
        res = {}
        for p in points:
            if p['role'].startswith('skinned') and p['surfaces'] == [si]:
                original = index.blob(writes[p['write_id']]['buffer'])
                cmp_, pos = vb_compare(original, payload)
                cmp_['skin_pose'] = p['skin_pose']
                cmp_['bones_over_1e-5'] = bone_attribution(mesh, pos)
                if mesh['mesh_name'] == 'o_head':
                    cmp_['bone_state_diagnosis'] = mouth_bone_diagnosis(mesh, st, original, root)
                res[f'{p["role"]}@{p["chunk_index"]}'] = cmp_
        # Counterfactual: blendshape deltas omitted (shows CPU blend inclusion).
        if any(s['current_weight'] for s in mesh['blendshapes']):
            m0 = copy.deepcopy(mesh)
            for s in m0['blendshapes']:
                s['current_weight'] = 0.0
            from tools.unity_parity.geometry import skin_world
            w0, _ = skin_world(m0, {t['id']: t for t in geometry['transforms']}, influences=4)
            p0 = g.skinned_vb_bytes(w0, st.world_normals, st.world_tangents, st.tangent_handedness, root)
            upd = [p for p in points if p['role'] == 'skinned_vertices_current' and p['surfaces'] == [si] and p['write_kind'] == 'unmap']
            if upd:
                _, pos0 = vb_compare(index.blob(writes[upd[0]['write_id']]['buffer']), p0)
                res['counterfactual_without_blendshapes_position_max'] = float(pos0.max())
                res['active_blendshapes'] = [(s['name'], s['current_weight']) for s in mesh['blendshapes'] if s['current_weight']]
        baseline['surfaces'][mesh['mesh_name'] + f'#{si}'] = res
    roots = {tuple(g.root_frame(s.root_world_matrix, s.root_world_rotation_xyzw).ravel()) for s in export_states.values()}
    if len(roots) != 1:
        raise g.SkinPathError('Bound head surfaces do not share one root frame in the export')
    export_root = np.asarray(next(iter(roots))).reshape(4, 4)
    rid = geometry['meshes'][next(iter(export_states))]['root_bone_transform_id']
    tr = {t['id']: t for t in geometry['transforms']}[rid]
    L2W = np.asarray(tr['local_to_world'], float).reshape(4, 4)
    polar = L2W[:3, :3] / np.linalg.norm(L2W[:3, :3], axis=0)
    baseline['matrices'] = {
        'export_root_bone': tr['name'],
        'export_root_noscale': export_root.tolist(),
        'unity_rotation_vs_normalized_world_columns_max': float(np.abs(export_root[:3, :3] - polar).max()),
        'capture_poses': poses,
        'export_root_vs_capture_pose_max_abs': {k: float(np.abs(np.asarray(v) - export_root).max()) for k, v in poses.items()}}

    # ---- forward-model integration (zero patch), optional ----
    forward = None
    if args.forward_geometry and args.forward_context:
        from src.hs2_offline_state import CapturedHeadForward
        fg = merge_loaded_frames(json.loads(args.forward_geometry.read_text(encoding='utf-8')), geometry)
        ctx = json.loads(args.forward_context.read_text(encoding='utf-8'))
        model = CapturedHeadForward(fg, ctx)
        result = model.evaluate()
        forward = {'geometry': str(args.forward_geometry), 'context': str(args.forward_context),
                   'geometry_frame': fg['frame_count'], 'surfaces': {}}
        for row in bindings:
            if row['status'] != 'bound':
                continue
            si = row['surface_index']
            st = g.surface_state_from_forward(result, si, fg)
            root = g.root_frame(st.root_world_matrix, st.root_world_rotation_xyzw)
            payload = g.skinned_vb_bytes(st.world_vertices, st.world_normals, st.world_tangents, st.tangent_handedness, root)
            upd = [p for p in points if p['role'] == 'skinned_vertices_current' and p['surfaces'] == [si] and p['write_kind'] == 'unmap']
            cmp_, pos = vb_compare(index.blob(writes[upd[0]['write_id']]['buffer']), payload)
            cmp_['bones_over_1e-5'] = bone_attribution(fg['meshes'][si], pos)
            forward['surfaces'][fg['meshes'][si]['mesh_name'] + f'#{si}'] = cmp_
        forward['root_noscale'] = root.tolist()

    # ---- demonstration plan: export state routed only to its matching pose ----
    # The export frame is not any recorded pose; it is NOT routed. The plan is
    # generated with no states to enumerate every upload and its gap.
    plan = g.generate_patches(index, writes, points, {}, args.out / 'empty_state_plan')

    summary_points = collections.Counter()
    for p in points:
        summary_points[(p['role'], p['write_kind'], p.get('skin_pose') or p.get('object_pose') or p.get('previous_pose'),
                        p.get('world_to_object', ''))] += 1
    evidence = {
        'schema_version': 1,
        'scope': 'Static CPU analysis of one existing RDC export; no game, capture, injection or GPU replay.',
        'source_xml': str(args.xml), 'source_xml_sha256': index.xml_sha256, 'source_zip_sha256': index.zip_sha256,
        'immediate_context': index.immediate_context, 'beginning_of_capture_chunk': index.begin.index,
        'target_readback_chunk': args.target_chunk, 'in_frame_skin_unmap_chunks': skin_unmaps,
        'compute_inventory': inventory,
        'surface_bindings': bindings,
        'head_draw_count': len(head),
        'head_draws_before_target': sum(1 for d in head if d['chunk_index'] < args.target_chunk),
        'vertex_shader_matrix_usage': {str(k): v for k, v in usage.items()},
        'poses': poses,
        'upload_point_summary': [{'role': k[0], 'write_kind': k[1], 'pose': k[2], 'world_to_object': k[3], 'count': v}
                                 for k, v in sorted(summary_points.items(), key=str)],
        'upload_points': points,
        'internal_consistency': consistency,
        'baseline_vs_live_export': baseline,
        'forward_integration_zero_patch': forward,
        'empty_state_plan_gap_count': len(plan['gaps']),
        'accepted': {'bit_exact_baseline': False, 'candidate_render': False, 'full_renderer': False},
    }
    (args.out / 'evidence.json').write_text(json.dumps(evidence, indent=1, default=g._json), encoding='utf-8')
    print(json.dumps({'bound': [b['mesh_name'] + ':' + b['status'] for b in bindings],
                      'points': len(points), 'poses': list(poses), 'head_draws': len(head)}, default=g._json))


if __name__ == '__main__':
    main()
