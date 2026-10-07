"""Exact global-cursor selection per actual bone and parent-first FK/LBS."""
from __future__ import annotations
import copy
import numpy as np
from tools.abmx_replay.model import require
from tools.abmx_replay.validate_trace import validate, local_only, trs_errors, cache_errors
from tools.abmx_replay.validate_geometry import cache_correspondence
from tools.unity_parity.geometry import blendshape_delta
from tools.unity_parity.bone_locals import compare_bone_locals
from src.hs2_mesh_deform import _fk_world, _trs

DEFAULT_BONES = ('cf_J_Chin_rs', 'cf_J_ChinTip_s', 'cf_J_CheekUp_L', 'cf_J_CheekUp_R')
MIN_PREFIX_CALLS_PER_BONE = 2
MAX_LAST_CALL_FRAME_AGE = 1


def select_cursor_predictions(snapshot, trace, contract, required_names=DEFAULT_BONES):
    """Recompute independent replay; select last completed call for EACH bone.

    Observable external boundaries are explicit conditioned input anchors, never
    identified writers or candidate prediction across an unsupported boundary.
    """
    names = tuple(required_names)
    require(names and len(set(names)) == len(names) and all(isinstance(n, str) and n for n in names), 'Unique required bone names needed')
    report = validate(trace, contract)
    require(report['passed'], 'Trace did not pass independent actual-call replay')
    require(set(trace['metadata']['requested_names']) == set(names), 'Observer filter differs from required complete bone group')
    cursor = snapshot.get('abmx_trace_cursor')
    require(isinstance(cursor, dict), 'Missing exact snapshot trace cursor')
    require(cursor.get('session_id') == trace['metadata']['session_id'], 'Snapshot references another trace session')
    require(type(cursor.get('frame')) is int and cursor['frame'] == snapshot['frame_count'] == snapshot['frame_count_end'], 'Snapshot cursor/frame mismatch')
    for key in ('pending_calls', 'dropped_events', 'observer_error_count'):
        require(type(cursor.get(key)) is int and cursor[key] == 0, 'Incomplete snapshot cursor: '+key)
    sequence = cursor.get('last_completed_sequence')
    require(type(sequence) is int and 1 <= sequence <= len(trace['events']), 'Invalid global completed-call cursor')
    require(cursor.get('observed_calls') == cursor.get('completed_calls') == sequence, 'Cursor global call coverage mismatch')
    require(trace['metadata']['character_transform_id'] == snapshot['character']['transform_id'], 'Snapshot belongs to another character')
    prefix = [row for row in report['rows'] if row['sequence'] <= sequence]
    require(len(prefix) == sequence and all(row['frame'] <= cursor['frame'] for row in prefix), 'Cursor includes future/missing calls')
    predicted, bindings = {}, []
    selected_ids = set()
    for name in names:
        rows = [row for row in prefix if row['bone_name'] == name]
        require(len(rows) >= MIN_PREFIX_CALLS_PER_BONE, 'Sparse observed prefix coverage for '+name)
        identities = {(row['modifier_instance_identity'], row['bone_transform_id']) for row in rows}
        require(len(identities) == 1, 'Bone/modifier identity changed within observed prefix: '+name)
        row = rows[-1]
        age = cursor['frame'] - row['frame']
        require(0 <= age <= MAX_LAST_CALL_FRAME_AGE, 'Stale/sparse latest observed call for '+name)
        require(row['bone_transform_id'] not in selected_ids, 'Two required names resolve to the same transform')
        selected_ids.add(row['bone_transform_id'])
        matches = [bone for bone in snapshot['transforms'] if bone['id'] == row['bone_transform_id']]
        require(len(matches) == 1 and matches[0]['name'] == name, 'Exact snapshot bone ID/name missing or ambiguous: '+name)
        bone = matches[0]
        prediction = row.get('chain_prediction', row['prediction'])
        local_error = trs_errors(prediction['after'], local_only(bone))
        require(local_error['passed'], 'Replay predicted local differs from actual snapshot: '+name)
        modifiers = [value for value in snapshot['abmx_runtime']['bones'] if value['name'] == name]
        require(len(modifiers) == 1, 'Exact snapshot private state missing or ambiguous: '+name)
        wrapper = modifiers[0]['runtime_baseline']
        require(wrapper.get('missing_fields') == [] and wrapper.get('bone_transform_id') == bone['id'] and wrapper.get('frame_count') == cursor['frame'], 'Snapshot cache bone/frame/missing-field mismatch: '+name)
        require(wrapper.get('assembly_mvid') == trace['metadata']['plugin_mvid'] and wrapper.get('modifier_type') == 'KKABMX.Core.BoneModifier', 'Snapshot private cache source mismatch: '+name)
        private_error = cache_errors(prediction['cache_after'], wrapper['fields'])
        require(private_error['passed'], 'Replay predicted private state differs from snapshot: '+name)
        predicted[name] = copy.deepcopy(prediction['after'])
        bindings.append({'bone_name': name, 'bone_transform_id': bone['id'], 'bone_path': bone['path'],
                         'selected_observed_sequence': row['sequence'], 'selected_call_frame': row['frame'],
                         'frames_since_selected_call': age, 'observed_prefix_call_count': len(rows),
                         'chain_segment_start_sequence': row['chain_segment'], 'prediction': prediction,
                         'snapshot_local_error': local_error, 'snapshot_cache_error': private_error})
    return predicted, {'cursor': cursor, 'selected_bones': bindings, 'trace_total_calls': len(trace['events']),
                       'later_calls_excluded': len(trace['events'])-sequence,
                       'coverage_policy': {'min_prefix_calls_per_bone': MIN_PREFIX_CALLS_PER_BONE, 'max_latest_call_frame_age': MAX_LAST_CALL_FRAME_AGE},
                       'override_source': 'independently predicted local states, never copied actual after',
                       'external_boundaries_at_or_before_cursor': [b for b in report['inter_call_boundaries'] if b['sequence'] <= sequence],
                       'external_writers_identified': False, 'application_counts_fitted': False}, report


def parent_first_world(rig, local, overrides):
    """Apply overrides to LOCAL matrices once, then recompute all descendants."""
    require(len(rig._topo) == len(set(rig._topo)) and set(rig._topo) == set(rig.bones), 'Incomplete/duplicate cached topology order')
    require(set(local) == set(rig.bones), 'Incomplete cached native local matrices')
    matrices = {pid: value.copy() for pid, value in local.items()}
    for name, state in overrides.items():
        matches = [pid for pid, bone in rig.bones.items() if bone['name'] == name]
        require(len(matches) == 1, 'Unknown/ambiguous cached override bone: '+name)
        require(np.isfinite(np.r_[state['local_position'], state['local_rotation_xyzw'], state['local_scale']]).all(), 'Nonfinite predicted override')
        matrices[matches[0]] = _trs(state['local_position'], state['local_rotation_xyzw'], state['local_scale'])
    world = {}
    for pid in rig._topo:
        parent = rig.bones[pid]['parent']
        require(parent not in rig.bones or parent in world, 'Cached topology is not parent-first or has a cycle')
        world[pid] = world[parent] @ matrices[pid] if parent in world else matrices[pid]
    return world


def skin_vertices(rig, world, delta):
    require(np.asarray(delta).shape == rig.verts.shape and np.isfinite(delta).all(), 'Expression delta does not match source mesh')
    homogeneous = np.column_stack([rig.verts+delta, np.ones(len(rig.verts))])
    matrices = np.stack([world[rig.name2pid[name]] @ rig.bindpose[i] for i, name in enumerate(rig.skin_bone_names)])
    return (np.einsum('nkij,nj->nki', matrices[rig.bone_idx], homogeneous)[..., :3]*rig.bone_w[..., None]).sum(axis=1)


def reconstruct(rig, native59, overrides, mesh):
    require(np.asarray(native59).shape == (59,) and np.isfinite(native59).all(), 'Actual complete native59 required')
    native_world = _fk_world(rig, native59, None)
    local = {}
    for pid in rig._topo:
        parent = rig.bones[pid]['parent']
        local[pid] = np.linalg.inv(native_world[parent]) @ native_world[pid] if parent in native_world else native_world[pid].copy()
    world = parent_first_world(rig, local, overrides)
    delta, active = blendshape_delta(mesh)
    return skin_vertices(rig, world, delta), active


def verify_nonoverridden_ancestors(snapshot, rig, overrides):
    """Native locals for every skin-palette ancestor, not only skin endpoints."""
    result = compare_bone_locals(snapshot, rig, None)
    required_pids = set()
    for name in rig.skin_bone_names:
        pid = rig.name2pid[name]
        while pid in rig.bones and pid not in required_pids:
            required_pids.add(pid)
            pid = rig.bones[pid]['parent']
    roots = {pid for pid in required_pids if rig.bones[pid]['parent'] not in rig.bones}
    names = {rig.bones[pid]['name'] for pid in required_pids-roots}
    rows = [row for row in result['bones'] if row['name'] in names]
    require({row['name'] for row in rows} == names and len(rows) == len(names), 'Skin ancestor local coverage incomplete/ambiguous')
    boundary = []
    for row in rows:
        parent = rig.bones[rig.name2pid[row['name']]]['parent']
        if parent in roots:
            boundary.append({'bone_name': row['name'], 'cached_parent': row['cached_parent'], 'live_parent': row['live_parent'],
                             'scope': 'asset root naming/frame boundary; actual ancestor uniform and whole-head rigid comparison independently checked'})
        else:
            require(row['cached_parent'] == row['live_parent'], 'Cached/live internal parent mapping differs: '+row['name'])
    other = [row for row in rows if row['name'] not in overrides]
    require(other and all(row['position_error'] <= 1e-5 and row['scale_error'] <= 1e-5 and row['rotation_angle_deg'] <= .001 for row in other), 'Unmodeled skin ancestor local deformation')
    return {'nonoverridden_bones': other, 'overridden_bones_parent_checked': [row for row in rows if row['name'] in overrides],
            'asset_root_frame_boundaries': boundary, 'thresholds': {'position': 1e-5, 'scale': 1e-5, 'rotation_deg': .001}}
