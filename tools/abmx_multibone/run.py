"""Audit restored four-head early/late multi-bone capture manifests offline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.abmx_multibone.geometry import DEFAULT_BONES, cache_correspondence, reconstruct, select_cursor_predictions, verify_nonoverridden_ancestors
from tools.abmx_replay.model import require
from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_replay.validate_geometry import verify_pairs
from tools.unity_parity.geometry import analyze_snapshot, recorded_uniform_renderer_scale, rigid_alignment
from tools.geometry_quality.cross_mesh import load_certified_head
from src.hs2_mesh_deform import HeadRig

TOLERANCE = 1e-5


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def restored_flags(manifest):
    categories = (('state_restored', 'snapshot_restored'), ('expression_restored',),
                  ('bone_restored', 'bones_restored', 'modifiers_restored'))
    flags = {}
    for names in categories:
        found = [name for name in names if name in manifest]
        require(found and all(manifest[name] is True for name in found), 'Manifest is not completed/restored: '+str(names))
        flags.update({name: manifest[name] for name in found})
    return flags


def verify_protocol(protocol, snapshot, cursor_binding):
    native = np.asarray(snapshot['character']['shape_value_face'], float)
    require(native.shape == (59,) and np.isfinite(native).all() and np.max(np.abs(native-np.asarray(protocol['native59'], float))) < 2e-6, 'Actual native59 differs from predeclared protocol')
    patches = {patch['name']: patch for patch in protocol['patches']}
    require(set(patches) == set(DEFAULT_BONES) and len(patches) == len(protocol['patches']), 'Predeclared patch group missing/duplicated')
    for binding in cursor_binding['selected_bones']:
        effective = binding['prediction']['effective_modifier']
        require(effective is not None, 'Null effective modifier at snapshot')
        for key in ('scale', 'length', 'position', 'rotation'):
            require(np.max(np.abs(np.asarray(effective[key])-np.asarray(patches[binding['bone_name']][key]))) < 2e-6, 'Actual effective patch differs from predeclared protocol: '+binding['bone_name']+'/'+key)
    return native.tolist()


def verify_capture_pair_signatures(window, snapshot):
    """Cross-bind actual metadata values, in addition to producer pair booleans."""
    pose, visibility = snapshot.get('pose_signature'), snapshot.get('visibility_sample_signature')
    require(isinstance(pose, str) and bool(pose) and isinstance(visibility, str) and bool(visibility), 'Missing snapshot pose/visibility signatures')
    views = window['capture']['views']
    require(len(views) == 3, 'Expected actual three paired views')
    for view in views:
        pair = view['paired_geometry']
        require(pair.get('pose_signature') == pose and pair.get('visibility_sample_signature') == visibility, 'Paired geometry pose/visibility signature mismatch')
        require(view.get('visibility_sample_signature_before_render') == view.get('visibility_sample_signature_after_render') == visibility, 'View visibility signatures differ from geometry despite producer flags')
        require(pair.get('frame_count') == snapshot['frame_count'] == view['frame_count'] == view['frame_count_before_render'], 'Pair/view/export actual frame mismatch')
    return {'actual_signature_values_crosschecked': True, 'pose_signature': pose, 'visibility_sample_signature': visibility,
            'scope': 'Exported sampled signatures plus immutable geometry/PNG identities and numerical bone/FK/LBS checks; not full shader visibility proof'}


def audit_window(case, window, trace, contract, protocol, sampling_profile):
    geometry = window['geometry']
    require(sha(geometry['path']) == geometry['sha256'], 'Geometry producer SHA mismatch')
    snapshot = read(geometry['path'])
    require(case['head_id'] == window['head_id'] == snapshot['character']['head_id'], 'Head ID source mismatch')
    require(set(case['names']) == set(DEFAULT_BONES), 'Case bone group differs from protocol')
    result = {'name': window['name'], 'head_id': case['head_id'], 'window': window['window'], 'passed': False,
              'trace_path': case['trace'], 'trace_sha256': sha(case['trace']), 'geometry_path': geometry['path'], 'geometry_sha256': sha(geometry['path'])}
    try:
        predicted, binding, call_report = select_cursor_predictions(snapshot, trace, contract)
        result.update(call_replay=call_report, exact_cursor_binding=binding)
        native = verify_protocol(protocol, snapshot, binding)
        result['actual_native59'] = native
        result['paired_views'] = verify_pairs(window, snapshot, geometry['sha256'])
        result['capture_pair_signature_binding'] = verify_capture_pair_signatures(window, snapshot)
        certificate = analyze_snapshot(snapshot, normalized_tolerance=TOLERANCE)
        certificate.update(snapshot_path=geometry['path'], snapshot_sha256=geometry['sha256'])
        result['actual_bone_lbs_certificate'] = certificate
        certified, skipped = load_certified_head(snapshot, geometry['sha256'], certificate)
        heads = [value for value in certified if value.name == 'o_head']
        require(len(heads) == 1, 'Exactly one certified visible source-bound o_head required')
        head = heads[0]
        mesh = next(value for value in snapshot['meshes'] if value['renderer_path'] == head.path)
        entry = next(row for row in certificate['meshes'] if row['renderer_path'] == head.path)
        require(entry['declared_influences'] == 4 and entry['influences_overridden'] is False, 'Offline four-weight contract differs from actual skin quality')
        result.update(selected_head_renderer={'path': head.path, 'source_geometry_sha256': head.source_hash, 'candidate': head.candidate}, skipped_renderers=skipped)
        rig = HeadRig(case['head_id'], sampling_profile=sampling_profile)
        paths = [Path(rig.data_dir)/name for name in ('o_head_mesh.npz', 'skeleton.json', 'anmShapeHead.json')]
        paths += [Path(rig.root_dir)/name for name in ('enums.json', 'customhead.json', 'update_eqns.json')]
        result['cached_rig_source_sha256'] = {str(path.resolve()): sha(path) for path in paths}
        result['cache_correspondence'] = cache_correspondence(rig, mesh)
        result['native_nonoverridden_skin_ancestors'] = verify_nonoverridden_ancestors(snapshot, rig, predicted)
        vertices, expression = reconstruct(rig, native, predicted, mesh)
        transform_ids = {value['id']: value for value in snapshot['transforms']}
        require(len(transform_ids) == len(snapshot['transforms']), 'Duplicate snapshot transform IDs')
        factor, evidence = recorded_uniform_renderer_scale(mesh, transform_ids)
        _, comparison = rigid_alignment(vertices, head.vertices, unit_scale=factor)
        result.update(active_actual_blendshapes=expression, recorded_ancestor_uniform_scale=evidence,
                      unit_scale=1, fitted_scale=False, fitted_affine=False, proper_rigid_pose_only=True,
                      state_conditioned_full_o_head_comparison=comparison)
        error = comparison['rigid_errors']['max_normalized']
        result['passed'] = error is not None and np.isfinite(error) and error <= TOLERANCE
        result['status'] = 'passed_specific_multibone_state_conditioned_snapshot' if result['passed'] else 'failed_specific_multibone_state_conditioned_snapshot'
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result['rejection'] = str(exc)
        result['status'] = 'rejected_incomplete_or_unsupported_snapshot'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sampling-profile', default='slider_unlocker_18_2', choices=('vanilla', 'slider_unlocker_18_2'))
    args = parser.parse_args()
    require(not args.out.exists(), 'New audit output directory required')
    manifest, contract = read(args.manifest), read(args.contract)
    flags = restored_flags(manifest)
    require(sha(contract['assembly_path']) == contract['assembly_sha256'] and sha(contract['unity_core_path']) == contract['unity_core_sha256'], 'Installed assembly contract changed')
    require(all(sha(source['path']) == source['sha256'] for source in contract['sources']), 'Installed decompiled source changed')
    require(sha(manifest['protocol_path']) == manifest['protocol_sha256'], 'Predeclared protocol SHA changed')
    protocol = read(manifest['protocol_path'])
    require(set(protocol['names']) == set(DEFAULT_BONES) and set(protocol['heads']) == {0, 1, 2, 3}, 'Different predeclared group/heads')
    require(len(manifest['cases']) == 4 and {case['head_id'] for case in manifest['cases']} == {0, 1, 2, 3}, 'Expected four independently observed head cases')
    args.out.mkdir(parents=True)
    summary = {'schema_version': 1, 'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest), 'restored_flags': flags,
               'contract_sha256': sha(args.contract), 'protocol_path': manifest['protocol_path'], 'protocol_sha256': manifest['protocol_sha256'],
               'sampling_profile': args.sampling_profile, 'required_names': list(DEFAULT_BONES), 'normalized_tolerance': TOLERANCE,
               'candidate_predictor_certified': False, 'static_whole_head_model_fixed': False, 'likeness_validated': False,
               'scope': 'Observed-before/cache-conditioned four bone replay, parent-first cached FK/LBS, specific actual o_head snapshots',
               'implementation_sha256': {str(path.resolve()): sha(path) for path in [Path(__file__), Path(__file__).with_name('geometry.py'),
                   ROOT/'tools/abmx_replay/model.py', ROOT/'tools/abmx_replay/validate_trace.py', ROOT/'tools/abmx_replay/validate_geometry.py',
                   ROOT/'src/hs2_mesh_deform.py', ROOT/'tools/unity_parity/geometry.py']}, 'windows': []}
    for case in manifest['cases']:
        require(sha(case['trace']) == case['trace_sha256'], 'Trace producer SHA mismatch')
        trace = read(case['trace'])
        require(len(case['windows']) == 2 and {window['window'] for window in case['windows']} == {'early', 'late'}, 'Missing early/late paired snapshots')
        for window in case['windows']:
            result = audit_window(case, window, trace, contract, protocol, args.sampling_profile)
            path = args.out/(window['name']+'.json')
            require(path.resolve().parent == args.out.resolve(), 'Unsafe output case name')
            save(path, result)
            compact = {'name': window['name'], 'head_id': case['head_id'], 'window': window['window'], 'passed': result['passed'],
                       'status': result['status'], 'rejection': result.get('rejection'), 'report_path': str(path.resolve())}
            if 'exact_cursor_binding' in result:
                binding = result['exact_cursor_binding']
                compact.update(cursor_sequence=binding['cursor']['last_completed_sequence'],
                               selected_sequences={row['bone_name']: row['selected_observed_sequence'] for row in binding['selected_bones']},
                               call_count=result['call_replay']['observed_call_count'], later_calls_excluded=binding['later_calls_excluded'],
                               external_boundaries=len(binding['external_boundaries_at_or_before_cursor']))
            if 'state_conditioned_full_o_head_comparison' in result:
                compact['full_o_head_errors'] = result['state_conditioned_full_o_head_comparison']['rigid_errors']
            summary['windows'].append(compact)
            save(args.out/'progress.json', summary)
            print(json.dumps(compact), flush=True)
    summary['passed_count'] = sum(row['passed'] for row in summary['windows'])
    summary['all_source_files_unchanged'] = (sha(args.manifest) == summary['manifest_sha256'] and sha(manifest['protocol_path']) == manifest['protocol_sha256']
        and all(sha(case['trace']) == case['trace_sha256'] and all(sha(window['geometry']['path']) == window['geometry']['sha256'] for window in case['windows']) for case in manifest['cases']))
    require(summary['all_source_files_unchanged'], 'Source files changed during read-only audit')
    summary['status'] = 'passed_specific_multibone_state_conditioned_snapshots' if summary['passed_count'] == 8 else 'failed_specific_multibone_state_conditioned_snapshots'
    summary['limitations'] = ['Trace before/cache are observed conditioning inputs, not candidate-native predictor validation.',
                              'External boundaries remain unknown writers; segments are reanchored for observed-call replay only.',
                              'Only these four named bones and snapshots; complete ABMX generalization unproven.',
                              'Actual expressions/recorded ancestors are nuisance inputs; no fitted scale/affine.',
                              'Runtime parity is separate from deformation quality and likeness.']
    save(args.out/'summary.json', summary)
    return 0 if summary['passed_count'] == 8 else 2


if __name__ == '__main__':
    raise SystemExit(main())
