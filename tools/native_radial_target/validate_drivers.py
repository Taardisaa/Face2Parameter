"""Additional source-bound native-driver gate; never generates a target from a candidate.

This revision leaves the frozen compiler and mathematical validators unchanged.
PE/MVID and enum decoding are supplied by the hash-bound installation contract;
this module verifies those bindings, rather than claiming to decode installed IL.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.hs2_mesh_deform import HeadRig
from tools.abmx_replay.model import IDENTITY, modifier, replay_apply, serialized
from tools.abmx_replay.validate_trace import local_only, read, sha, trs_errors
from tools.abmx_stable_lowering.compiler import check_files, digest_json
from tools.native_radial_target.adapter import (
    MODE,
    NAMES,
    PROFILE,
    native_numpy,
    patch_array,
)
from tools.native_radial_target.compiler import prepare_execution_guard, require

POLICY = 'native_face_drivers_v1'
ENUM = {'NO_LOOK': 0, 'TARGET': 1, 'AWAY': 2, 'FORWARD': 3, 'CONTROL': 4}
GAZE = ('cf_J_look_L', 'cf_J_look_R')
ROLES = {'bridge_pe', 'game_pe', 'driver_snapshot_source', 'eye_look_source', 'enum_source'}
_CAPABILITY = object()


def _integer(value, label):
    require(type(value) is int and value >= 0, 'Exact nonnegative integer required: ' + label)
    return value


def _boolean(value, label):
    require(type(value) is bool, 'Exact boolean required: ' + label)
    return value


def _array(value, shape, label):
    a = np.asarray(value, dtype=np.float32)
    require(a.shape == shape and np.isfinite(a).all(), 'Finite shaped array required: ' + label)
    return a


def _ref(state):
    require(isinstance(state, dict), 'Actual transform reference missing')
    require(type(state.get('transform_id')) is int, 'Exact transform ID required')
    require(type(state.get('name')) is str and state['name'], 'Transform name missing')
    require(type(state.get('path')) is str and state['path'].startswith('/'), 'Transform path missing')
    for k, n in (('local_position', 3), ('local_scale', 3), ('local_rotation_xyzw', 4)):
        _array(state.get(k), (n,), k)
    require(np.linalg.norm(_array(state['local_rotation_xyzw'], (4,), 'quaternion')) > 0,
            'Zero local quaternion')
    return {k: state[k] for k in ('transform_id', 'name', 'path')}


def _mapped(state, snapshot):
    ref = _ref(state)
    matches = [t for t in snapshot['transforms'] if t.get('id') == ref['transform_id']]
    require(len(matches) == 1, 'Missing or duplicated geometry transform ID')
    t = matches[0]
    require((t['name'], t['path']) == (ref['name'], ref['path']), 'Driver/geometry name or path mismatch')
    error = trs_errors(local_only(t), local_only(state))
    require(error['passed'], 'Driver/geometry local TRS mismatch ' + ref['name'])
    return ref, error


def validate_contract(contract, artifact):
    require(contract.get('schema_version') == 1 and type(contract.get('schema_version')) is int
            and contract.get('policy') == POLICY, 'Unsupported driver contract')
    require(contract.get('eye_look_enum') == ENUM and
            all(type(v) is int for v in contract['eye_look_enum'].values()), 'Source-bound NO_LOOK enum mapping required')
    require(set(contract.get('source_roles', {})) == ROLES, 'Explicit installed driver source roles required')
    files = contract.get('source_files')
    require(type(files) is dict and files, 'Driver source hashes required')
    for path in contract['source_roles'].values():
        require(type(path) is str and Path(path).is_absolute() and path in files, 'Unbound source role path')
    check_files(files)
    p = artifact['source_bindings']
    require(contract.get('bridge_mvid') == p['source_bridge_mvid'] and type(contract.get('bridge_mvid')) is str,
            'Compiler/driver bridge MVID mismatch')
    require(type(contract.get('game_mvid')) is str and contract['game_mvid'], 'Game MVID required')
    require(sha(contract['source_roles']['game_pe']) == p['game_assembly_sha256'], 'Driver game PE differs from source rig experiment')


def inspect_snapshot(snapshot, contract):
    """Typed actual state plus raw transform bindings. No target generation."""
    d = snapshot.get('native_face_drivers')
    require(type(d) is dict and type(d.get('schema_version')) is int and d['schema_version'] == 1,
            'Missing actual native_face_drivers schema; legacy evidence refused')
    frame = _integer(d.get('frame'), 'driver frame')
    require(type(snapshot.get('frame_count')) is int and snapshot['frame_count'] == frame
            and type(snapshot.get('frame_count_end')) is int and snapshot['frame_count_end'] == frame,
            'Driver is not same-frame geometry')
    actor = snapshot['character']['transform_id']
    require(type(actor) is int and type(d.get('actor_transform_id')) is int and d['actor_transform_id'] == actor,
            'Driver/geometry actor mismatch')
    require(d.get('bridge_mvid') == contract['bridge_mvid'] and d.get('game_mvid') == contract['game_mvid'],
            'Actual driver assembly identity mismatch')
    configured, eye = d.get('configured'), d.get('eye_controller')
    require(type(configured) is dict and type(eye) is dict, 'Configured driver/controller missing')
    require(_boolean(configured.get('mouth_adjust_width'), 'mouth_adjust_width') is False,
            'Mouth width native driver must be disabled')
    index = _integer(configured.get('eyes_look_pattern'), 'configured eye pattern')
    require(_integer(eye.get('pattern'), 'actual eye pattern') == index, 'Configured and actual eye pattern disagree')
    patterns = eye.get('patterns')
    require(type(patterns) is list and patterns, 'Actual pattern table missing')
    for i, row in enumerate(patterns):
        require(type(row) is dict and _integer(row.get('index'), 'pattern index') == i,
                'Pattern table must have actual unique ordered indices')
        require(type(row.get('name')) is str and row['name'] in contract['eye_look_enum'] and
                type(row.get('value')) is int and row['value'] == contract['eye_look_enum'][row['name']],
                'Actual pattern differs from source-bound enum')
    require(index < len(patterns), 'Configured eye pattern out of actual table')
    selected = patterns[index]
    resolved = eye.get('resolved_type')
    require(type(resolved) is dict and type(resolved.get('value')) is int and
            resolved == {'name': selected['name'], 'value': selected['value']} and
            selected['name'] == 'NO_LOOK' and selected['value'] == 0,
            'Actual selected eye mode must resolve to NO_LOOK=0; pattern index is not assumed')
    flags = {k: _boolean(eye.get(k), k) for k in
             ('enabled', 'active_and_enabled', 'script_enabled', 'global_enabled')}
    dt = eye.get('delta_time')
    require(type(dt) in (int, float) and np.isfinite(dt) and dt >= 0, 'Invalid actual delta_time')
    expression = snapshot['character'].get('expression')
    require(type(expression) is dict and expression.get('mouth_adjust_width') is False
            and type(expression.get('eyes_look_pattern')) is int and expression['eyes_look_pattern'] == index,
            'Expression and actual driver readbacks disagree')
    mouth_ref, mouth_error = _mapped(d.get('mouth_adjust_target'), snapshot)
    require(mouth_ref['name'] == 'cf_J_MouthMove', 'Unsupported actual mouth driver target')
    gaze = d.get('gaze_bones')
    require(type(gaze) is list and len(gaze) == 2 and sorted(g['name'] for g in gaze) == sorted(GAZE),
            'Complete unique gaze bones required')
    refs = {}; errors = {}
    for g in gaze:
        refs[g['name']], errors[g['name']] = _mapped(g, snapshot)
    require(len({r['transform_id'] for r in refs.values()}) == 2, 'Gaze bone IDs collide')
    objects = eye.get('eye_objects')
    require(type(objects) is list and len(objects) == 2, 'Actual eye-script controlled objects missing')
    ordered = []
    angles = _array(eye.get('fixed_angles_xyzw'), (2, 4), 'actual fixAngle')
    for i, obj in enumerate(objects):
        ref, _ = _mapped(obj, snapshot)
        require(ref['name'] in refs and ref == refs[ref['name']], 'Eye script pointer is not the declared gaze bone')
        require(np.linalg.norm(angles[i]) > 0, 'Zero fixed eye quaternion')
        fixed = local_only(obj); fixed['local_rotation_xyzw'] = angles[i].tolist()
        require(trs_errors(fixed, local_only(obj))['passed'], 'NO_LOOK cached angle differs from actual eye object')
        ordered.append(ref)
    require(len({r['transform_id'] for r in ordered}) == 2, 'Duplicated controlled eye object')
    target = eye.get('target')
    target_ref = None if target is None else _ref(target)
    return {'frame': frame, 'actor_id': actor, 'pattern_index': index, 'patterns': patterns,
            'eye_flags': flags, 'fixed_angles_xyzw': angles.tolist(), 'mouth_ref': mouth_ref,
            'gaze_refs': refs, 'eye_object_order': ordered, 'target_ref': target_ref,
            'gaze_locals': {g['name']: local_only(g) for g in gaze},
            'geometry_binding_errors': {'mouth': mouth_error, **errors}}


def _source_targets(artifact):
    """Independent NumPy native baseline and one installed physical replay; source only."""
    d = artifact['compiler_inputs']['declared']
    require(d['semantic_mode'] == MODE and d['sampling_profile'] == PROFILE, 'Wrong target semantics/profile')
    native = native_numpy(HeadRig(d['head_id'], sampling_profile=PROFILE), d['native59'])
    trace = read(artifact['compiler_inputs']['history']['trace'])
    logical = patch_array(d['logical_patches']); targets = {}
    for name, row in zip(NAMES, logical):
        n = {k: np.asarray(v, dtype=np.float32) for k, v in native[name].items()}
        event = next(e for e in trace['events'] if e['bone_name'] == name)
        require(trs_errors(serialized(n), local_only(event['before']))['passed'], 'NumPy native/source mismatch ' + name)
        fields = copy.deepcopy(event['before']['cache']['fields'])
        for k, field in (('local_position', '_posBaseline'), ('local_scale', '_sclBaseline'), ('local_rotation_xyzw', '_rotBaseline')):
            fields[field] = n[k].tolist()
        desired = n['local_position'] * row[3] + row[4:7]
        physical = {'scale': row[:3].tolist(), 'length': 1., 'position': (desired - n['local_position']).tolist(),
                    'rotation': row[7:].tolist()}
        prediction = replay_apply(before=serialized(n), cache=fields, coordinate_modifiers=[physical], coordinate=0,
                                  additional_modifiers=[], bone_exists=True, rotation_excluded=False, is_during_h_scene=False)
        targets[name] = prediction['after']
        require(trs_errors(targets[name], artifact['desired_target_locals'][name])['passed'], 'Independent target/compiled disagreement')
    return targets


class DriverGuardContext:
    """Prepared before observer start. Extra gate, not a trace/LBS/quality certificate."""
    def __init__(self, old, compiled, contract, contract_path, source, targets, *, _capability,
                 artifact_path=None, artifact_sha256=None):
        require(_capability is _CAPABILITY, 'Use prepare_driver_guard; JSON trust tokens refused')
        self._old = old
        self._compiled_ref = compiled
        self._artifact = copy.deepcopy(compiled)
        self._digest = digest_json(compiled)
        self._contract = copy.deepcopy(contract)
        self._contract_path = str(Path(contract_path).resolve())
        self._contract_sha = sha(contract_path)
        self._source = copy.deepcopy(source)
        self._targets = copy.deepcopy(targets)
        self._module_sha = sha(__file__)
        self._prepared_digest = digest_json({'source': self._source, 'targets': self._targets, 'contract': self._contract})
        self._artifact_path = artifact_path
        self._artifact_sha = artifact_sha256
        self._helper_files = {str(Path(__file__).resolve().parents[1] / 'abmx_replay' / n):
                              sha(Path(__file__).resolve().parents[1] / 'abmx_replay' / n)
                              for n in ('model.py', 'validate_trace.py')}

    def __reduce__(self):
        raise TypeError('Prepare a new nonserializable DriverGuardContext from frozen sources')

    def _bound(self, snapshot, metadata):
        require(sha(__file__) == self._module_sha, 'Driver gate implementation changed after preparation')
        require(sha(self._contract_path) == self._contract_sha, 'Driver contract changed after preparation')
        require(digest_json(self._compiled_ref) == self._digest and digest_json(self._artifact) == self._digest,
                'Compiled artifact mutated after preparation')
        require(digest_json({'source': self._source, 'targets': self._targets, 'contract': self._contract}) == self._prepared_digest,
                'Prepared driver targets/anchors/contract mutated')
        if self._artifact_path:
            require(sha(self._artifact_path) == self._artifact_sha, 'Frozen artifact file changed')
        check_files(self._helper_files)
        check_files(self._artifact['source_bindings']['source_files'])
        check_files(self._contract['source_files'])
        p = self._artifact['source_bindings']; cursor = snapshot['abmx_trace_cursor']
        for k in ('frame', 'observed_calls', 'completed_calls', 'last_completed_sequence', 'pending_calls', 'dropped_events', 'observer_error_count'):
            _integer(cursor.get(k), k)
        require(cursor.get('active') is True and cursor['observed_calls'] == cursor['completed_calls'] == cursor['last_completed_sequence']
                and cursor['pending_calls'] == cursor['dropped_events'] == cursor['observer_error_count'] == 0,
                'Complete active serial prefix required')
        require(type(metadata.get('started_frame')) is int and 0 <= metadata['started_frame'] <= cursor['frame']
                and type(metadata.get('session_id')) is str and metadata['session_id'] and cursor['session_id'] == metadata['session_id'],
                'Actual session/frame binding differs')
        require(metadata.get('bridge_mvid') == p['source_bridge_mvid'] and metadata.get('plugin_mvid') == p['expected_abmx_mvid']
                and metadata.get('apply_method_il_sha256') == p['expected_apply_il_sha256'] and metadata.get('pre_existing_patch_owners') == [],
                'Actual source/patch identity differs')
        require(type(metadata.get('requested_names')) is list and len(metadata['requested_names']) == 30
                and set(metadata['requested_names']) == set(NAMES), 'Actual ALL30 observer filter required')
        require(snapshot['character']['transform_id'] == metadata.get('character_transform_id') == p['source_actor_id']
                and type(snapshot['character']['transform_id']) is int, 'Source actor changed')
        require(type(snapshot['character']['head_id']) is int and snapshot['character']['head_id'] == p['head_id']
                and np.array_equal(_array(snapshot['character']['shape_value_face'], (59,), 'native59'), np.asarray(p['native59'], np.float32)),
                'Source head/native59 changed')
        require(snapshot['game']['game_assembly_sha256'] == p['game_assembly_sha256'], 'Actual game assembly differs')
        current = inspect_snapshot(snapshot, self._contract)
        require(cursor['frame'] == current['frame'], 'Driver/cursor frame differs')
        for k in ('actor_id', 'pattern_index', 'patterns', 'eye_flags', 'mouth_ref', 'gaze_refs', 'eye_object_order', 'target_ref'):
            require(current[k] == self._source[k], 'Frozen source driver binding changed: ' + k)
        require(np.array_equal(np.asarray(current['fixed_angles_xyzw'], np.float32), np.asarray(self._source['fixed_angles_xyzw'], np.float32)),
                'Source fixed eye angles changed')
        gaze_errors = {n: trs_errors(self._source['gaze_locals'][n], current['gaze_locals'][n]) for n in GAZE}
        require(all(e['passed'] for e in gaze_errors.values()), 'Actual gaze differs from frozen source nuisance')
        return current, gaze_errors

    def verify_identity(self, snapshot, current_trace_metadata):
        original = self._old.verify_snapshot(snapshot, current_trace_metadata)
        current, gaze = self._bound(snapshot, current_trace_metadata)
        return self._report(current, gaze, original_identity_guard=original)

    def verify_candidate(self, snapshot, current_trace_metadata):
        current, gaze = self._bound(snapshot, current_trace_metadata)
        expected = {r['name']: r for r in self._artifact['executed_patches']}
        source = read(self._artifact['compiler_inputs']['history']['geometry']['path'])
        checks = {}
        bones = snapshot['abmx_runtime']['bones']
        for bone in bones:
            if bone['name'] not in NAMES:
                require(serialized(modifier({k: bone[k] for k in IDENTITY})) == serialized(modifier(IDENTITY)),
                        'Extra active modifier unsupported')
        for name in NAMES:
            rows = [b for b in bones if b['name'] == name]
            require(len(rows) == 1 and rows[0]['exists'] is True, 'Missing/duplicate candidate bone ' + name)
            row = rows[0]
            require(serialized(modifier({k: row[k] for k in IDENTITY})) == serialized(modifier({k: expected[name][k] for k in IDENTITY})), 'Executed physical patch differs ' + name)
            original = [t for t in source['transforms'] if t['name'] == name]
            require(len(original) == 1, 'Ambiguous source transform ' + name)
            matches = [t for t in snapshot['transforms'] if t['id'] == original[0]['id']]
            require(len(matches) == 1 and matches[0]['name'] == name and matches[0]['path'] == original[0]['path'], 'Candidate/source transform binding differs ' + name)
            w = row['runtime_baseline']
            require(w.get('modifier_type') == 'KKABMX.Core.BoneModifier' and w.get('assembly_mvid') == self._artifact['source_bindings']['expected_abmx_mvid']
                    and w.get('missing_fields') == [] and type(w.get('frame_count')) is int and w['frame_count'] == current['frame']
                    and type(w.get('bone_transform_id')) is int and w['bone_transform_id'] == matches[0]['id'], 'Actual candidate cache/frame/ID differs')
            checks[name] = trs_errors(self._targets[name], local_only(matches[0]))
        require(all(e['passed'] for e in checks.values()), 'Candidate actual ALL30 locals differ from source-only desired target: ' + ','.join(n for n,e in checks.items() if not e['passed']))
        return self._report(current, gaze, all30_target_locals=checks)

    def verify_view(self, view, snapshot):
        """Recompute exact paired driver equality; producer green flag alone is insufficient."""
        driver = snapshot.get('native_face_drivers')
        require(type(driver) is dict, 'Paired geometry driver state missing')
        for key in ('native_face_drivers_before_render', 'native_face_drivers_after_render'):
            require(type(view.get(key)) is dict and view[key] == driver, 'Actual paired view driver mismatch: ' + key)
        require(view.get('paired_native_face_drivers_unchanged') is True, 'Paired driver producer status absent/false')
        return {'passed': True, 'full_driver_objects_independently_equal': True, 'policy': POLICY}

    def _report(self, current, gaze, **extras):
        return {'passed': True, 'policy': POLICY, 'gate_implementation_sha256': self._module_sha,
                'driver_contract_sha256': self._contract_sha, 'compiled_binding_sha256': self._digest,
                'actual_pattern_index': current['pattern_index'], 'resolved_name': 'NO_LOOK', 'resolved_value': 0,
                'source_gaze_policy': 'frozen_source_locals', 'gaze_checks': gaze,
                'candidate_used_to_generate_target': False, 'future_writer_absence_certified': False,
                'whole_surface_trace_quality_or_character_ready_certified': False, **extras}


def prepare_driver_guard(compiled, abmx_contract_path, driver_contract_path, *, artifact_path=None, artifact_sha256=None):
    """Expensive validation before observer start; preserves original guard semantics."""
    old = prepare_execution_guard(compiled, abmx_contract_path, artifact_path=artifact_path, artifact_sha256=artifact_sha256)
    contract = read(driver_contract_path)
    validate_contract(contract, compiled)
    source = read(compiled['compiler_inputs']['history']['geometry']['path'])
    inspected = inspect_snapshot(source, contract)
    targets = _source_targets(compiled)
    return DriverGuardContext(old, compiled, contract, driver_contract_path, inspected, targets, _capability=_CAPABILITY,
                              artifact_path=artifact_path, artifact_sha256=artifact_sha256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', required=True)
    parser.add_argument('--artifact-sha256', required=True)
    parser.add_argument('--abmx-contract', required=True)
    parser.add_argument('--driver-contract', required=True)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--metadata', required=True, help='Actual trace export or trace-start metadata object')
    parser.add_argument('--phase', choices=('identity', 'candidate'), required=True)
    parser.add_argument('--out', required=True, help='NEW report file; never overwrites existing evidence')
    args = parser.parse_args()
    paths = {k: getattr(args, k) for k in ('compiled', 'abmx_contract', 'driver_contract', 'snapshot', 'metadata')}
    report = {'policy': POLICY, 'phase': args.phase, 'inputs': {k: {'path': str(Path(v).resolve()), 'sha256': sha(v)} for k,v in paths.items()},
              'gate_implementation_sha256': sha(__file__), 'read_only_posthoc': True,
              'whole_trace_surface_quality_or_live_certified': False}
    try:
        ctx = prepare_driver_guard(read(args.compiled), args.abmx_contract, args.driver_contract,
                                   artifact_path=args.compiled, artifact_sha256=args.artifact_sha256)
        metadata = read(args.metadata)
        if 'metadata' in metadata: metadata = metadata['metadata']
        result = getattr(ctx, 'verify_' + args.phase)(read(args.snapshot), current_trace_metadata=metadata)
        report.update(result=result, passed=True)
    except (ValueError, KeyError, TypeError, OSError, StopIteration) as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(serialized(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__': main()
