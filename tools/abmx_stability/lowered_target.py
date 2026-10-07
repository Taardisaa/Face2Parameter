"""Independent source-only FIRST logical target; candidate after is never input."""
from __future__ import annotations

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from src.hs2_mesh_deform import HeadRig, _fk_world
from tools.abmx_multibone.geometry import (
    DEFAULT_BONES,
    cache_correspondence,
    reconstruct,
    select_cursor_predictions,
    verify_nonoverridden_ancestors,
)
from tools.abmx_multibone_quality.run import identity_bound, load_snapshot
from tools.abmx_replay.model import FLAG_FIELDS, IDENTITY, require
from tools.abmx_replay.validate_trace import read, replay_event, sha, trs_errors
from tools.abmx_stability.classify import classify, source_contract
from tools.geometry_quality.cross_mesh import load_certified_head
from tools.unity_parity.geometry import (
    analyze_snapshot,
    recorded_uniform_renderer_scale,
    rigid_alignment,
)


def numpy_native_locals(rig, native):
    """Independent NumPy native FK then TRS decomposition; no compiler target data."""
    world = _fk_world(rig, native, None)
    result = {}
    for name in DEFAULT_BONES:
        pid = rig.name2pid[name]
        parent = rig.bones[pid]['parent']
        local = np.linalg.inv(world[parent]) @ world[pid] if parent in world else world[pid]
        scale = np.linalg.norm(local[:3, :3], axis=0)
        require(np.isfinite(local).all() and np.all(scale > 0), 'Invalid cached native local')
        rotation = local[:3, :3] / scale
        require(np.max(np.abs(rotation.T @ rotation-np.eye(3))) < 1e-10 and np.linalg.det(rotation) > 0,
                'Native TRS has unsupported shear/reflection')
        result[name] = {'local_position': local[:3, 3].astype(np.float32).tolist(),
                        'local_scale': scale.astype(np.float32).tolist(),
                        'local_rotation_xyzw': Rotation.from_matrix(rotation).as_quat().astype(np.float32).tolist()}
    return result


def certified_head(snapshot, descriptor):
    certificate = analyze_snapshot(snapshot, normalized_tolerance=1e-5)
    certificate.update(snapshot_sha256=descriptor['sha256'], snapshot_path=descriptor['path'])
    certified, _ = load_certified_head(snapshot, descriptor['sha256'], certificate)
    heads = [head for head in certified if head.name == 'o_head']
    require(len(heads) == 1, 'Exactly one certified actual head surface required')
    return heads[0], certificate


def prepare_target(artifact, contract):
    """Accept only frozen earlier source + predeclared native/logical controls.

    Compiler numerical predictions are comparison targets only. Actual candidate
    traces, snapshots, blendshapes, counts and after states are not accepted here.
    """
    source_contract(contract)
    inputs = artifact['compiler_inputs']
    history, declared = inputs['history'], inputs['declared']
    require(artifact['actual_candidate_after_used_for_target'] is False, 'Compiler admits candidate-after target leakage')
    require(all(sha(path) == digest for path, digest in artifact['provenance']['source_files'].items()),
            'Frozen compiler/source history changed')
    require(sha(history['trace']) == history['trace_sha256'], 'Source trace changed')
    snapshot, mesh, _ = load_snapshot(history['geometry'], declared['head_id'])
    trace = read(history['trace'])
    _, cursor, replay = select_cursor_predictions(snapshot, trace, contract)
    require(not replay['inter_call_boundaries'], 'History contains external boundary')
    require(cursor['cursor'] == artifact['provenance']['source_history_cursor'], 'Source cursor changed')
    native = np.asarray(declared['native59'], dtype=np.float64)
    require(native.shape == (59,) and np.isfinite(native).all(), 'Complete finite native59 required')
    require(np.array_equal(native.astype(np.float32), np.asarray(declared['source_history_native59'], dtype=np.float32)),
            'This source-only target audit requires same-native source')
    require(np.array_equal(native.astype(np.float32), np.asarray(snapshot['character']['shape_value_face'], dtype=np.float32)),
            'Source actual native differs')
    rig = HeadRig(declared['head_id'], sampling_profile='slider_unlocker_18_2')
    baseline = numpy_native_locals(rig, native)
    patches = {patch['name']: patch for patch in declared['logical_patches']}
    require(set(patches) == set(DEFAULT_BONES) and len(declared['logical_patches']) == 4, 'Logical four-bone declaration incomplete')
    predictions, checks = {}, {}
    for name in DEFAULT_BONES:
        event = next(event for event in trace['events'] if event['bone_name'] == name)
        fields = copy.deepcopy(event['before']['cache']['fields'])
        require(fields['_hasBaseline'] and not any(fields[key] for key in FLAG_FIELDS if key != '_hasBaseline'),
                'Earlier source flags not clean')
        require(event['resolved_modifier'] == IDENTITY, 'Earlier first call is not identity')
        require(not event['additional_modifiers'] and not event['no_rotation_excluded'] and not event['is_during_h_scene'],
                'Unsupported logical target branch')
        source_local = {key: event['before'][key] for key in baseline[name]}
        baseline_error = trs_errors(baseline[name], source_local)
        require(baseline_error['passed'], 'Independent native fails measured clean source boundary')
        for key, field in (('local_position', '_posBaseline'), ('local_rotation_xyzw', '_rotBaseline'), ('local_scale', '_sclBaseline')):
            fields[field] = copy.deepcopy(baseline[name][key])
        logical_event = copy.deepcopy(event)
        logical_event['resolved_modifier'] = {key: value for key, value in patches[name].items() if key != 'name'}
        logical_event['before'] = {**baseline[name], 'cache': {'fields': fields}}
        branch = classify(logical_event, long_count=2)
        require(branch['supported_selected_branch'] and branch['regime'] == 'combined', 'Unsupported FIRST clean logical branch')
        prediction = replay_event(logical_event, baseline[name], fields)
        compiler_error = trs_errors(prediction['after'], artifact['bone_diagnostics'][name]['logical_first_prediction']['after'])
        require(compiler_error['passed'], 'Compiler logical target differs from independent source-only prediction')
        predictions[name] = prediction['after']
        checks[name] = {'independent_native_source_boundary': baseline_error, 'compiler_target_comparison': compiler_error,
                        'source_first_sequence': event['sequence'], 'before': baseline[name], 'cache_before': fields,
                        'prediction': prediction, 'branch_classification': branch}
    correspondence = cache_correspondence(rig, mesh)
    ancestors = verify_nonoverridden_ancestors(snapshot, rig, baseline)
    source_head, certificate = certified_head(snapshot, history['geometry'])
    factor, factor_evidence = recorded_uniform_renderer_scale(mesh, {row['id']: row for row in snapshot['transforms']})
    native_vertices, _ = reconstruct(rig, native, baseline, mesh)
    _, source_error = rigid_alignment(native_vertices, source_head.vertices, unit_scale=factor)
    require(source_error['rigid_errors']['max_normalized'] <= 1e-5, 'Source-only native fullhead guard failed')
    target, active = reconstruct(rig, native, predictions, mesh)
    return {'vertices': target, 'source_snapshot': snapshot, 'source_mesh': mesh, 'rig': rig,
            'evidence': {'native59': native.tolist(), 'source_trace_sha256': history['trace_sha256'],
                         'source_geometry_sha256': history['geometry']['sha256'], 'source_cursor': cursor['cursor'],
                         'independent_first_logical_predictions': checks, 'source_fullhead_guard': source_error,
                         'source_lbs_certificate': certificate, 'cache_correspondence': correspondence,
                         'native_nonoverridden_ancestors': ancestors, 'source_active_blendshapes': active,
                         'source_uniform_ancestor_scale': factor_evidence,
                         'target_input_candidate_after_or_counts': False, 'target_runtime_certified': False}}


def compare_window_to_target(target, window):
    """Actual geometry is a comparison object, never a generator of target."""
    snapshot, mesh, _ = load_snapshot(window['geometry'], target['source_snapshot']['character']['head_id'])
    identity_bound(mesh, target['source_mesh'])
    require(snapshot['character']['expression'] == target['source_snapshot']['character']['expression'], 'Target/actual expression config differs')
    require([(b.get('name'), b['current_weight']) for b in mesh.get('blendshapes', [])]
            == [(b.get('name'), b['current_weight']) for b in target['source_mesh'].get('blendshapes', [])],
            'Actual expression weights differ from source-only target nuisance')
    require(np.array_equal(np.asarray(snapshot['character']['shape_value_face'], dtype=np.float32),
                           np.asarray(target['evidence']['native59'], dtype=np.float32)), 'Actual target native differs')
    head, certificate = certified_head(snapshot, window['geometry'])
    factor, scale_evidence = recorded_uniform_renderer_scale(mesh, {row['id']: row for row in snapshot['transforms']})
    _, comparison = rigid_alignment(target['vertices'], head.vertices, unit_scale=factor)
    error = comparison['rigid_errors']['max_normalized']
    return {'window': window['window'], 'geometry_sha256': window['geometry']['sha256'], 'comparison': comparison,
            'passed': bool(np.isfinite(error) and error <= 1e-5), 'normalized_max_tolerance': 1e-5,
            'actual_lbs_certificate': certificate, 'actual_recorded_uniform_ancestor_scale': scale_evidence,
            'target_uses_actual_candidate_after_or_expression': False, 'proper_rigid_pose_only': True,
            'fitted_scale': False, 'fitted_affine': False}
