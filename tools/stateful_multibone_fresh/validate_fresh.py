"""Validate fresh predeclared native candidates from earlier identity history."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig
from tools.stateful_multibone_fresh.adapter import EarlierHistoryProtocol, FreshObservedCountTorchRig
from tools.stateful_multibone.adapter import DEFAULT_BONES, read, sha
from tools.abmx_replay.validate_trace import trs_errors, cache_errors
from tools.abmx_replay.validate_geometry import cache_correspondence, verify_pairs
from tools.abmx_multibone.geometry import verify_nonoverridden_ancestors
from tools.abmx_multibone.run import restored_flags, verify_capture_pair_signatures
from tools.unity_parity.geometry import analyze_snapshot, blendshape_delta, recorded_uniform_renderer_scale, rigid_alignment


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def audit(case, window, manifest_path, contract_path, device):
    protocol = EarlierHistoryProtocol.from_manifest(manifest_path, contract_path, case['head_id'], window['window'])
    snapshot = read(window['geometry']['path'])
    pairs = verify_pairs(window, snapshot, window['geometry']['sha256'])
    signatures = verify_capture_pair_signatures(window, snapshot)
    source_snapshot = read(case['source_history']['geometry']['path'])
    source_pairs = verify_pairs(case['source_history'], source_snapshot, case['source_history']['geometry']['sha256'])
    source_signatures = verify_capture_pair_signatures(case['source_history'], source_snapshot)
    rig = HeadRig(case['head_id'], sampling_profile='slider_unlocker_18_2')
    model = FreshObservedCountTorchRig(rig, protocol, device=device, dtype=torch.float64)
    native = torch.as_tensor(protocol.candidate_native[None], device=device, dtype=model.dtype)
    patches = {name: {'scale': value[:3], 'length': value[3], 'position': value[4:7], 'rotation': value[7:]}
               for name, value in protocol.patches.items()}
    ab = model.ab_tensor(patches)
    errors, predicted = {}, {}
    with torch.no_grad():
        _, states, caches = model.local_states(native, ab)
        for name in DEFAULT_BONES:
            predicted[name] = {key: value[0].cpu().tolist() for key, value in
                               zip(('local_position', 'local_rotation_xyzw', 'local_scale'), states[name])}
            bone_id = protocol.identities[name]['bone_transform_id']
            matches = [row for row in snapshot['transforms'] if row['id'] == bone_id and row['name'] == name]
            if len(matches) != 1:
                raise ValueError('Unique actual bone required')
            actual_cache = next(row for row in snapshot['abmx_runtime']['bones'] if row['name'] == name)['runtime_baseline']['fields']
            predicted_cache = {key: value[0].cpu().tolist() for key, value in caches[name].items()}
            errors[name] = {'local': trs_errors(predicted[name], {key: matches[0][key] for key in predicted[name]}),
                            'cache': cache_errors(predicted_cache, actual_cache)}
        certificate = analyze_snapshot(snapshot, normalized_tolerance=1e-5)
        heads = [mesh for mesh in snapshot['meshes'] if mesh['mesh_name'] == 'o_head' and mesh['enabled'] and mesh['active_in_hierarchy']]
        if len(heads) != 1:
            raise ValueError('Exactly one active head required')
        mesh = heads[0]
        cert = next(row for row in certificate['meshes'] if row['renderer_path'] == mesh['renderer_path'])
        if not cert['certified'] or cert['declared_influences'] != 4 or cert['influences_overridden'] or len(cert['matching_candidates']) != 1:
            raise ValueError('Unique actual four-influence head convention required')
        correspondence = cache_correspondence(rig, mesh)
        ancestors = verify_nonoverridden_ancestors(snapshot, rig, predicted)
        actual = np.asarray(mesh['baked']['world_candidates'][cert['selected_candidate']]['vertices'])
        delta, active = blendshape_delta(mesh)
        world = model.bone_world(native, ab)[0]
        skin = world[model.skin_bone] @ model.bindpose
        homogeneous = torch.cat([torch.as_tensor(rig.verts+delta, device=device, dtype=model.dtype),
                                 torch.ones(len(rig.verts), 1, device=device, dtype=model.dtype)], 1)
        vertices = (torch.einsum('vkij,vj->vki', skin[model.bone_idx], homogeneous)[..., :3]*model.bone_w[..., None]).sum(1).cpu().numpy()
        factor, evidence = recorded_uniform_renderer_scale(mesh, {row['id']: row for row in snapshot['transforms']})
        _, comparison = rigid_alignment(vertices, actual, unit_scale=factor)
    paths = [Path(rig.data_dir)/name for name in ('o_head_mesh.npz', 'skeleton.json', 'anmShapeHead.json')]
    paths += [Path(rig.root_dir)/name for name in ('enums.json', 'customhead.json', 'update_eqns.json')]
    return {'name': window['name'], 'head_id': case['head_id'], 'window': window['window'],
            'passed': all(row['local']['passed'] and row['cache']['passed'] for row in errors.values())
                      and comparison['rigid_errors']['max_normalized'] <= 1e-5,
            'per_bone_errors': errors, 'full_head_errors': comparison['rigid_errors'], 'protocol': model.protocol_metadata,
            'paired_views': pairs, 'pair_signature_binding': signatures,
            'earlier_history_paired_views': source_pairs, 'earlier_history_pair_signature_binding': source_signatures,
            'cached_rig_source_sha256': {str(path.resolve()): sha(path) for path in paths},
            'actual_head_certificate': cert, 'cache_correspondence': correspondence,
            'native_nonoverridden_ancestors': ancestors, 'recorded_ancestor_uniform_scale': evidence,
            'active_actual_expression': active, 'fitted_scale': False, 'fitted_affine': False,
            'proper_rigid_pose_only': True, 'actual_before_used_as_prediction_input': False,
            'actual_after_used_as_prediction_input': False, 'common_N_runtime_certified': False,
            'arbitrary_candidate_runtime_certified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('New evidence directory required')
    torch.set_num_threads(4)
    manifest = read(args.manifest)
    restored_flags(manifest)
    if len(manifest['cases']) != 4 or {row['head_id'] for row in manifest['cases']} != {0, 1, 2, 3}:
        raise ValueError('Expected four independently source-bound fresh heads')
    args.out.mkdir(parents=True)
    sources = [Path(__file__), Path(__file__).with_name('adapter.py'), ROOT/'src/hs2_abmx_torch.py',
               ROOT/'src/hs2_deform_torch.py', ROOT/'src/hs2_mesh_deform.py', ROOT/'tools/abmx_multibone/geometry.py',
               ROOT/'tools/abmx_replay/validate_trace.py', ROOT/'tools/stateful_multibone/adapter.py']
    report = {'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
              'contract_sha256': sha(args.contract), 'device': args.device, 'windows': [],
              'normalized_tolerance': 1e-5, 'sampling_profile': 'slider_unlocker_18_2',
              'source_hashes': {str(path.resolve()): sha(path) for path in sources},
              'common_N_runtime_certified': False, 'arbitrary_candidate_runtime_certified': False, 'character_ready': False,
              'scope': 'Fresh changed predeclared native59; earlier identity measured length/history only, independently predicted candidate initial/cache TRS, observed per-bone counts'}
    for case in manifest['cases']:
        if len(case['windows']) != 2 or {w['window'] for w in case['windows']} != {'early', 'late'}:
            raise ValueError('Expected fresh early/late windows')
        for window in case['windows']:
            try:
                row = audit(case, window, args.manifest, args.contract, args.device)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                row = {'name': window['name'], 'head_id': case['head_id'], 'window': window['window'],
                       'passed': False, 'rejection': str(exc)}
            target = args.out/(window['name']+'.json')
            if target.resolve().parent != args.out.resolve():
                raise ValueError('Unsafe output name')
            save(target, row)
            compact = {key: row[key] for key in ('name', 'head_id', 'window', 'passed')}
            compact.update(rejection=row.get('rejection'), report_path=str(target.resolve()))
            if 'full_head_errors' in row:
                compact.update(full_head_errors=row['full_head_errors'], per_bone_errors=row['per_bone_errors'],
                               observed_counts=row['protocol']['observed_candidate_apply_counts'])
            report['windows'].append(compact)
            save(args.out/'progress.json', report)
            print(json.dumps(compact), flush=True)
    report['passed_count'] = sum(row['passed'] for row in report['windows'])
    report['passed'] = report['passed_count'] == 8
    report['sources_unchanged'] = (all(sha(path) == digest for path, digest in report['source_hashes'].items())
                                   and sha(args.manifest) == report['manifest_sha256'])
    if not report['sources_unchanged']:
        raise ValueError('Sources changed during audit')
    save(args.out/'summary.json', report)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
