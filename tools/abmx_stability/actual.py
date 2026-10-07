"""Executable actual-case audit; orchestration schema is bound by the caller."""
from __future__ import annotations
import copy
import numpy as np
from tools.abmx_stability.classify import classify, private_difference, source_contract
from tools.abmx_multibone.geometry import DEFAULT_BONES, select_cursor_predictions
from tools.abmx_multibone.run import audit_window, verify_capture_pair_signatures
from tools.abmx_replay.validate_trace import read, sha, trs_errors
from tools.abmx_replay.validate_geometry import verify_pairs
from tools.abmx_replay.model import FLAG_FIELDS, IDENTITY, require
from tools.abmx_multibone_quality.run import (load_snapshot, identity_bound, certify, DEFAULT_ACCEPTANCE,
                                            evaluate_surface, surface_gate, nuisance)
from tools.stateful_multibone_fresh_quality.run import state


def vector(value):
    return np.asarray([*value['scale'], value['length'], *value['position'], *value['rotation']], dtype=np.float32)


def persistent_equal(a, b):
    return all(np.array_equal(np.asarray(a[key], dtype=np.float32), np.asarray(b[key], dtype=np.float32))
               for key in ('_lenBaseline', '_positionBaseline'))


def audit_case(case, declared, contract, out_dir, *, label, long_count=512):
    """Recompute actual replay, whole head and temporal metrics, without old green verdicts."""
    source_contract(contract)
    source = case['source_history']
    require(sha(source['trace']) == source['trace_sha256'] and sha(case['trace']) == case['trace_sha256'], 'Actual trace source SHA mismatch')
    source_trace, trace = read(source['trace']), read(case['trace'])
    source_snapshot, source_mesh, _ = load_snapshot(source['geometry'], case['head_id'])
    _, source_binding, source_replay = select_cursor_predictions(source_snapshot, source_trace, contract)
    require(not source_replay['inter_call_boundaries'], 'History trace has external local/cache boundary')
    verify_pairs(source, source_snapshot, source['geometry']['sha256'])
    verify_capture_pair_signatures(source, source_snapshot)
    require(source_trace['metadata']['stopped_frame'] < trace['metadata']['started_frame'], 'Source observer did not finish before candidate')
    require(source_trace['metadata']['character_transform_id'] == trace['metadata']['character_transform_id'], 'Source/candidate character identity changed')
    classifications, history, first_active = {}, {}, {}
    for name in DEFAULT_BONES:
        source_events = [event for event in source_trace['events'] if event['bone_name'] == name]
        events = [event for event in trace['events'] if event['bone_name'] == name]
        first = source_events[0]
        initial = first['before']['cache']['fields']
        identity = (first['before']['bone_transform_id'], first['modifier_instance_identity'])
        for event in source_events:
            require(np.array_equal(vector(event['resolved_modifier']), vector(IDENTITY)), 'Source must be identity-only')
            for phase in ('before', 'after'):
                fields = event[phase]['cache']['fields']
                require(fields['_hasBaseline'] and not any(fields[key] for key in FLAG_FIELDS if key != '_hasBaseline'), 'Source flags not clean')
        expected = next(patch for patch in declared['patches'] if patch['name'] == name)
        for event in source_events+events:
            require((event['before']['bone_transform_id'], event['modifier_instance_identity']) == identity, 'Source/candidate bone or modifier identity changed')
            for phase in ('before', 'after'):
                require(persistent_equal(event[phase]['cache']['fields'], initial), 'Persistent history changed: '+name)
        starts = [i for i, event in enumerate(events) if not np.array_equal(vector(event['resolved_modifier']), vector(IDENTITY))]
        require(starts and starts[0] > 0, 'Candidate trace needs initial identity phase then active patch')
        require(all(np.array_equal(vector(event['resolved_modifier']), vector(expected)) for event in events[starts[0]:]), 'Candidate must retain one constant predeclared patch')
        first_active[name] = events[starts[0]]
        classifications[name] = classify(events[starts[0]], long_count=long_count)
        require(classifications[name]['regime'] == label, 'Declared regime differs from actual HasLength/HasPosition channels')
        history[name] = {'source_first_fields': copy.deepcopy(initial), 'source_last_fields': copy.deepcopy(source_events[-1]['after']['cache']['fields']),
                         'candidate_first_before_fields': copy.deepcopy(events[0]['before']['cache']['fields']),
                         'active_first_before': copy.deepcopy(events[starts[0]]['before']),
                         'active_first_sequence': events[starts[0]]['sequence'], 'active_total_observed_count': len(events)-starts[0]}
    windows, compact = {}, []
    for window in case['windows']:
        report = audit_window(case, window, trace, contract, declared, 'slider_unlocker_18_2')
        require(report['passed'], 'Actual full head/replay rejected: '+str(report.get('rejection')))
        binding = report['exact_cursor_binding']
        require(not binding['external_boundaries_at_or_before_cursor'] and not report['call_replay']['inter_call_boundaries'], 'Complete candidate trace has external boundary')
        snapshot, mesh, path = load_snapshot(window['geometry'], case['head_id'])
        require(snapshot['character']['transform_id'] == source_snapshot['character']['transform_id'], 'Snapshot belongs to another character')
        identity_bound(mesh, source_mesh)
        canonical, faces, _, certificate = certify(snapshot, mesh, window['geometry'], path, out_dir/(window['name']+'_lbs.json'))
        counts = {}
        for row in binding['selected_bones']:
            name = row['bone_name']
            counts[name] = sum(event['bone_name'] == name and first_active[name]['sequence'] <= event['sequence'] <= binding['cursor']['last_completed_sequence'] for event in trace['events'])
            fields = next(bone for bone in snapshot['abmx_runtime']['bones'] if bone['name'] == name)['runtime_baseline']['fields']
            require(persistent_equal(fields, history[name]['source_first_fields']), 'Snapshot persistent history changed')
        actual_state = state(snapshot, mesh)
        windows[window['window']] = {'vertices': canonical, 'faces': faces, 'raw': np.asarray(mesh['baked']['vertices']),
                                    'state': actual_state, 'snapshot': snapshot, 'mesh': mesh, 'observed_counts': counts}
        # The evidence includes all observed before/cache inputs used by replay.
        compact.append({'name': window['name'], 'window': window['window'], 'full_head_replay': report,
                        'world_coordinate_certificate': certificate, 'observed_candidate_counts': counts,
                        'source_sha256': window['geometry']['sha256'], 'actual_state': actual_state})
    require(set(windows) == {'early', 'late', 'far60'}, 'Actual early/late/far60 windows required')
    temporal = []
    for earlier, later in (('early', 'late'), ('late', 'far60'), ('early', 'far60')):
        a, b = windows[earlier], windows[later]
        require(np.array_equal(a['faces'], b['faces']), 'Actual temporal topology changed')
        surface = evaluate_surface(b['vertices'], b['faces'], a['vertices'], a['faces'], count=4096, seed=7381)
        raw_delta, canonical_delta = b['raw']-a['raw'], b['vertices']-a['vertices']
        local, private = {}, {}
        for name in DEFAULT_BONES:
            bones = []
            for snapshot in (a['snapshot'], b['snapshot']):
                rows = [t for t in snapshot['transforms'] if t['name'] == name]
                require(len(rows) == 1, 'Ambiguous actual selected bone')
                bones.append({key: rows[0][key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')})
            local[name] = {'early': bones[0], 'late': bones[1], 'difference': trs_errors(*bones)}
            wrappers = [next(row for row in snapshot['abmx_runtime']['bones'] if row['name'] == name)['runtime_baseline']
                        for snapshot in (a['snapshot'], b['snapshot'])]
            private[name] = private_difference(*wrappers)
        temporal.append({'earlier': earlier, 'later': later, 'surface': surface, 'gate': surface_gate(surface, DEFAULT_ACCEPTANCE),
                         'raw_corresponding_vertices_rms': float(np.sqrt(np.mean(np.sum(raw_delta**2, axis=1)))),
                         'raw_corresponding_vertices_max': float(np.linalg.norm(raw_delta, axis=1).max()),
                         'recorded_rigid_frame_corresponding_vertices_rms': float(np.sqrt(np.mean(np.sum(canonical_delta**2, axis=1)))),
                         'recorded_rigid_frame_corresponding_vertices_max': float(np.linalg.norm(canonical_delta, axis=1).max()),
                         'local_selected_bone_drift': local, 'private_fields_vs_runtime_metadata': private,
                         'nuisance': nuisance(a['state'], b['state']), 'actual_per_bone_counts': {earlier: a['observed_counts'], later: b['observed_counts']},
                         'fitted_scale': False, 'fitted_affine': False, 'common_N_certified': False})
    return {'label': label, 'head_id': case['head_id'], 'history': history, 'source_cursor': source_binding,
            'source_history_trace_sha256': source['trace_sha256'], 'candidate_trace_sha256': case['trace_sha256'],
            'classifications': classifications, 'windows': compact, 'temporal': temporal,
            'full_head_passed_count': len(compact), 'temporal_pairs_passed_count': sum(pair['gate']['within_existing_surface_tolerances'] for pair in temporal),
            'actual_temporal_stable_in_observed_windows': all(pair['gate']['within_existing_surface_tolerances'] for pair in temporal),
            'unsupported_selected_branches': [name for name, row in classifications.items() if not row['supported_selected_branch']],
            'common_N_runtime_certified': False, 'global_or_infinite_time_runtime_stability_certified': False,
            'arbitrary_candidate_or_likeness_certified': False}
