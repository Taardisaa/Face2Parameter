"""Source-bound ALL30 consumer lifecycle. No implicit resets or anatomy claims.

Only absent facial modifiers are owned. Native writes precede their creation;
their removal precedes restoration of the original native values. Every run
stages an immutable authored-asset provider from a NEW stopped identity source.
"""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

from src.face_data_utils.utils import BONE_NAME_LIST
from src.hs2_ingame_semantics import (
    FLAGS,
    digest,
    file_sha,
    finite_vector,
    require,
    validate_write,
)

MODE = 'native_radial_target_v1'
PROFILE = 'slider_unlocker_18_2'
NAMES = tuple(BONE_NAME_LIST)
IDENTITY = {'scale': [1., 1., 1.], 'length': 1.,
            'position': [0., 0., 0.], 'rotation': [0., 0., 0.]}
POLICIES = {'strict', 'source_bound_identity_nuisance_v1'}
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
    return {'path': str(path.resolve()), 'sha256': file_sha(path)}


def bound_read(descriptor):
    path = Path(descriptor['path'])
    require(path.is_absolute() and path.is_file() and file_sha(path) == descriptor['sha256'],
            'Immutable source/capture bytes changed')
    return read(path)


def logical_map(controls):
    require(type(controls) is dict and set(controls) == set(NAMES), 'Explicit raw ALL30 logical dictionary required')
    out = {}
    for name in NAMES:
        row = controls[name]
        require(type(row) is dict and set(row) == set(IDENTITY), 'Exact raw patch fields required: '+name)
        patch = {}
        for key, size in (('scale', 3), ('position', 3), ('rotation', 3), ('length', 1)):
            value = [row[key]] if key == 'length' else row[key]
            require(isinstance(value, (list, tuple)) and all(type(v) in (int, float) for v in value),
                    'Raw numeric patch, not bool or normalized ML vector, required')
            vec = finite_vector(value, size, name+'/'+key)
            if key in ('scale', 'length'):
                require(np.all(vec > 0), 'Positive raw scale/length required')
            patch[key] = float(vec[0]) if key == 'length' else vec.tolist()
        out[name] = patch
    return out


def identity_row(row):
    require(type(row) is dict, 'Explicit modifier object required')
    return all(row.get(k) == v for k, v in IDENTITY.items())


def transform_table(snapshot):
    rows = snapshot['transforms']
    require(type(rows) is list and rows and all(type(t.get('id')) is int for t in rows),
            'Explicit actual skeleton transform IDs required')
    result = {t['id']: t for t in rows}
    require(len(result) == len(rows), 'Duplicate actual transform IDs are ambiguous')
    return result


def extras_anchor(snapshot, policy):
    """Private history is observed independently; it is never current local TRS."""
    require(policy in POLICIES, 'Unknown explicit outside-state policy')
    transforms = transform_table(snapshot)
    result, missing = {}, []
    bones = snapshot['abmx_runtime']['bones']
    require(len({b['name'] for b in bones}) == len(bones), 'Ambiguous full modifier inventory')
    for row in bones:
        if row['name'] in NAMES or row.get('exists') is not True:
            continue
        require(identity_row(row) and all(identity_row(c) for c in row['coordinate_modifiers']),
                'Outside ALL30 modifier is not public/coordinate identity: '+row['name'])
        wrapper = copy.deepcopy(row['runtime_baseline'])
        wrapper.pop('frame_count', None)
        require(wrapper.get('missing_fields') == [] and type(wrapper.get('bone_transform_id')) is int,
                'Outside runtime identity/private fields unavailable')
        fields = wrapper['fields']
        require(all(type(fields.get(k)) is bool for k in FLAGS), 'Outside flags must be actual bools')
        if policy == 'strict':
            require(not any(fields[k] for k in FLAGS if k != '_hasBaseline'),
                    'Strict provider refuses inherited pending outside modifier: '+row['name'])
        transform = transforms.get(wrapper['bone_transform_id'])
        pending = any(fields[k] for k in FLAGS if k != '_hasBaseline')
        require(not pending or transform is not None,
                'Pending outside modifier lacks actual local/ancestor observation: '+row['name'])
        if transform is not None:
            ancestor, visited = transform, set()
            while ancestor['parent_id'] is not None:
                require(ancestor['id'] not in visited and ancestor['parent_id'] in transforms,
                        'Outside modifier ancestor coverage incomplete: '+row['name'])
                visited.add(ancestor['id'])
                ancestor = transforms[ancestor['parent_id']]
        if transform is None:
            missing.append({'name': row['name'], 'bone_transform_id': wrapper['bone_transform_id']})
        result[row['name']] = {
            'public': {k: copy.deepcopy(row[k]) for k in (*IDENTITY, 'coordinate_modifiers', 'exists')},
            'runtime': wrapper,
            'actual_local': local_record(transform) if transform is not None else None,
        }
    return {'policy': policy, 'modifiers': result, 'unexported_body_local_ids': missing,
            'actual_local_scope': 'actual exported actor hierarchy/ancestors; private baselines are not current local observations',
            'full_actor_pose_certified': False}


def local_record(row):
    return {k: copy.deepcopy(row[k]) for k in ('id', 'name', 'path', 'parent_id',
            'local_position', 'local_rotation_xyzw', 'local_scale')}


def local_comparison(before, after, *, exclude=()):
    from tools.abmx_replay.validate_trace import local_only, trs_errors
    a = {identity: t for identity, t in transform_table(before).items() if t['name'] not in exclude}
    b = {identity: t for identity, t in transform_table(after).items() if t['name'] not in exclude}
    require(set(a) == set(b), 'Actual exported local skeleton ID coverage changed')
    checks = {}
    for identity, row in a.items():
        other = b[identity]
        require(all(row[k] == other[k] for k in ('id', 'name', 'path', 'parent_id')), 'Actual skeleton provenance changed')
        checks[str(identity)] = {'name': row['name'], 'path': row['path'],
                                **trs_errors(local_only(row), local_only(other))}
    return {'passed': all(c['passed'] for c in checks.values()), 'checks': checks,
            'scope': 'every actual exported actor hierarchy/ancestor local by ID/path/TRS; not future dynamic pose or rendered RGB',
            'full_actor_pose_certified': False}


class StrictBackend:
    """Concrete immutable V2 asset + source-only compiler + independent NumPy audit."""

    def __init__(self, contract, driver_contract):
        from tools.native_radial_target import (
            asset_provider_strict,
            compiler,
            validate_drivers,
        )
        self.assets = asset_provider_strict
        self.compiler = compiler
        self.drivers = validate_drivers
        self.contract, self.driver_contract = Path(contract), Path(driver_contract)

    def build(self, history, declared, overrides, out):
        staged = self.assets.stage_strict_from_identity(history, self.contract, overrides, out/'assets')
        provider = self.assets.prepare_strict_asset_provider(staged['path'], staged['sha256'])
        declared = copy.deepcopy(declared)
        declared['source_files'][staged['path']] = staged['sha256']
        compiled = self.compiler.compile_native_target(history, declared, self.contract)
        artifact = save(out/'compiled_execution.json', compiled)
        target = provider.independent_target(compiled, read(self.contract), source_gaze_policy='frozen_source_locals')
        checks = target['evidence']['source_mesh_checks']
        require(checks and all(c['native_source_model_passed'] and c['target_scope_supported'] for c in checks.values()),
                'Fresh source-only full visible-mesh preflight failed')
        source = bound_read(history['geometry'])
        provider.verify_snapshot_assets(source)
        save(out/'source_only_target.json', target['evidence'])
        guard = self.drivers.prepare_driver_guard(compiled, self.contract, self.driver_contract,
                artifact_path=artifact['path'], artifact_sha256=artifact['sha256'])
        self.provider, self.guard, self.compiled, self.staged = provider, guard, compiled, staged
        return {'compiler_artifact': artifact, 'asset_provider': staged, 'patches': compiled['executed_patches']}

    def identity(self, snapshot, metadata):
        self.provider.verify_snapshot_assets(snapshot)
        return self.guard.verify_identity(snapshot, current_trace_metadata=metadata)

    def candidate(self, capture, metadata):
        snapshot = bound_read(capture['paired_geometry'])
        self.provider.verify_snapshot_assets(snapshot)
        result = self.guard.verify_candidate(snapshot, current_trace_metadata=metadata)
        result['view_checks'] = [self.guard.verify_view(v, snapshot) for v in capture['views']]
        return result

    def audit(self, manifest, out):
        return self.provider.runtime_audit(manifest, self.contract, out, source_gaze_policy='frozen_source_locals')


class ControlledRadialProvider:
    """One candidate at a time; successful validation includes terminal restoration.

    ``outside_state_policy`` defaults to strict refusal. The explicit nuisance
    policy observes exact public/private outside state and exported local scope;
    no named Neck exemption and no body/ALL30 private writes are performed.
    """

    requires_actor_transform_coverage = True

    def __init__(self, *, work_dir, contract_path, driver_contract_path, asset_plan,
                 strict_backend='identity_source_assets_strict_v2',
                 outside_state_policy='strict', resolution=512):
        require(strict_backend == 'identity_source_assets_strict_v2', 'Explicit trusted strict V2 backend required')
        require(outside_state_policy in POLICIES, 'Explicit supported outside-state policy required')
        require(type(resolution) is int and 16 <= resolution <= 4096, 'Resolution 16..4096 required')
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.contract = Path(contract_path).resolve()
        self.driver_contract = Path(driver_contract_path).resolve()
        self.asset_plan = Path(asset_plan).resolve()
        for p in (self.contract, self.driver_contract, self.asset_plan):
            require(p.is_file(), 'Explicit local contract/asset plan required: '+str(p))
        plan = read(self.asset_plan)
        require(type(plan.get('head_id')) is int and plan['head_id'] == 3
                and type(plan.get('overrides')) is dict and plan['overrides'],
                'Initial consumer supports explicit known-source head3 asset plan only')
        local_sources = (Path(__file__).resolve(), ROOT/'src/hs2_ingame_semantics.py',
                         ROOT/'src/face_data_utils/utils.py')
        self.bindings = {str(p): file_sha(p) for p in (self.contract, self.driver_contract, self.asset_plan, *local_sources)}
        self.policy, self.resolution = outside_state_policy, resolution
        self._active = self._prepared = None

    def _sources(self):
        require(all(Path(p).is_file() and file_sha(p) == sha for p, sha in self.bindings.items()),
                'Provider source/contracts/asset plan changed; construct a new provider')

    def prepare(self, context, candidate_native59, *, logical_controls):
        self._sources()
        require(self._active is None, 'Terminally restore previous candidate before preparing another')
        require(context['mode'] == MODE and context['profile'] == PROFILE
                and context['binding']['head_id'] == 3, 'Explicit head3/installed sampler context required')
        require(context['binding_sha256'] == digest(context['binding']), 'Input context binding is stale')
        require(context['geometry'].get('include_actor_transforms') is True
                and context['geometry'].get('transform_scope') == 'actor_hierarchy_and_ancestors',
                'Actual entire actor transform observation is required before prepare')
        require(isinstance(candidate_native59, (list, tuple, np.ndarray))
                and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
                        for v in candidate_native59), 'Raw finite numeric native59, not bool, required')
        values, _ = validate_write(candidate_native59, None, context['capabilities'], context['binding']['range_mode'])
        logical = logical_map(logical_controls)
        require(not any(b.get('exists') is True and b['name'] in NAMES for b in context['abmx']['bones']),
                'Existing ALL30 modifiers have historical-radius refresh risk; provider refuses to remove/reset them')
        outside = extras_anchor(context['geometry'], self.policy)
        plan = {'schema_version': 1, 'mode': MODE, 'semantic_mode': MODE, 'head_id': 3,
                'sampler_profile': PROFILE, 'sampling_profile': PROFILE,
                'input_context_binding_sha256': context['binding_sha256'], 'request_id': uuid.uuid4().hex,
                'native59': values.tolist(), 'source_history_native59': values.tolist(),
                'logical_patches': logical, 'outside_state_policy': self.policy, 'outside_original': outside,
                'range_mode': context['binding']['range_mode'], 'source_files': self.bindings,
                'candidate_actual_after_input': False, 'runtime_certified': False, 'anatomy_certified': False}
        self._prepared = {'plan': copy.deepcopy(plan), 'context': copy.deepcopy(context)}
        return plan

    def _capture(self, name, settle, camera=None):
        state = self._active
        query = {'out': str(state['out']/(name+'.png')), 'w': self.resolution, 'h': self.resolution,
                 'aa': 1, 'yaws': '-30,0,30', 'hide_meshes': 'o_body_cf', 'freeze_pose': 'true',
                 'settle': settle, 'geometry_out': str(state['out']/(name+'.json')),
                 'geometry_blendshape_frames': 'true', 'geometry_include_actor_transforms': 'true'}
        if camera:
            query.update(ortho_size=camera['ortho_size'], dist=camera['distance'],
                         target=','.join(map(str, camera['target'])))
        result = state['boundary']('/maker/render?'+urlencode(query), 'POST', timeout=180)
        save(state['out']/(name+'_response.json'), result)
        self._paired(result)
        return result

    @staticmethod
    def _paired(capture):
        from tools.abmx_multibone.run import verify_capture_pair_signatures
        from tools.abmx_replay.validate_geometry import verify_pairs
        snapshot = bound_read(capture['paired_geometry'])
        require(snapshot.get('include_actor_transforms') is True
                and snapshot.get('transform_scope') == 'actor_hierarchy_and_ancestors'
                and capture['paired_geometry'].get('include_actor_transforms') is True
                and capture['paired_geometry'].get('transform_scope') == 'actor_hierarchy_and_ancestors',
                'Paired native export did not confirm actual entire actor transform scope')
        verify_pairs({'capture': capture, 'geometry': capture['paired_geometry']}, snapshot, capture['paired_geometry']['sha256'])
        verify_capture_pair_signatures({'capture': capture}, snapshot)
        require(capture.get('capture_state_restored') is True and capture.get('views'), 'Actual paired capture restoration/views missing')
        for view in capture['views']:
            require(all(view.get(k) is True for k in ('paired_pose_unchanged', 'paired_capture_state_unchanged',
                        'paired_visibility_sampled_unchanged', 'paired_native_face_drivers_unchanged', 'paired_lights_unchanged')),
                    'Actual paired pose/visibility/driver/lights/capture state changed')
            p = Path(view['path'])
            require(p.is_file() and p.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n', 'Actual PNG unavailable')
        return snapshot

    def _supplied_images(self, response, evidence):
        rows = evidence.get('images')
        require(type(rows) is list and len(rows) == len(response['views']) and rows,
                'Consumer supplied PNG evidence coverage missing')
        for view, observed in zip(response['views'], rows):
            path = Path(view['path']).resolve()
            require(path == Path(observed['path']).resolve() and file_sha(path) == observed['sha256'],
                    'Actual PNG bytes/path changed since consumer evidence; no scoring')
            data = path.read_bytes()
            require(data[:8] == b'\x89PNG\r\n\x1a\n' and len(data) >= 24
                    and int.from_bytes(data[16:20], 'big') == view['width'] == self.resolution
                    and int.from_bytes(data[20:24], 'big') == view['height'] == self.resolution,
                    'Supplied actual PNG dimensions/signature differ')

    def _start(self):
        s = self._active
        require(s['boundary']('/maker/abmx/trace').get('active') is False,
                'Another observer became active; do not replace/stop it')
        # After explicit no-observer readback, a transport failure can still mean
        # the server accepted START. Recovery must attempt this owned start.
        s['trace_owned'] = True
        value = s['boundary']('/maker/abmx/trace', 'POST', {'action': 'start', 'names': list(NAMES), 'max_events': 10000})
        return value['metadata']

    def _stop(self, filename):
        s = self._active
        require(s['trace_owned'], 'No owned trace to stop')
        value = s['boundary']('/maker/abmx/trace', 'POST', {'action': 'stop'})
        s['trace_owned'] = value.get('active') is not False
        descriptor = save(s['out']/filename, value)
        require(value.get('trace_complete') is True and value.get('active') is False and value.get('events'),
                'Stopped trace incomplete; preserve evidence and refuse acceptance')
        return descriptor

    def _outside_check(self, snapshot, anchor, *, original=False):
        current = extras_anchor(snapshot, self.policy)
        # Local numeric equality uses the existing installed replay tolerance;
        # wrapper IDs, public fields and every cache flag/field remain exact.
        a, b = copy.deepcopy(anchor), copy.deepcopy(current)
        for value in (a, b):
            for row in value['modifiers'].values():
                row.pop('actual_local')
        require(a == b, 'Outside ALL30 observed identity/private state changed')
        source = self._active['original_snapshot'] if original else self._active['source_snapshot']
        local = local_comparison(source, snapshot, exclude=() if original else NAMES)
        require(local['passed'], 'Noncontrolled actual head/ancestor local state changed')
        return {'outside_binding_sha256': digest(current), 'locals': local}

    def apply(self, plan, boundary):
        self._sources()
        require(self._prepared is not None and digest(plan) == digest(self._prepared['plan']),
                'Unknown/tampered prepared request (including JSON numeric/bool types)')
        require(boundary('/maker/abmx/trace').get('active') is False, 'Another observer is active; no owned writes')
        context = self._prepared['context']
        # Re-read full boundary before first owned write. Root consumer preflight
        # cannot replace this independent stale-context check.
        from src.hs2_ingame_semantics import snapshot_context
        fresh = snapshot_context(boundary, MODE, plan['range_mode'], PROFILE, include_actor_transforms=True)
        require(fresh['binding_sha256'] == context['binding_sha256'], 'Prewrite context changed')
        out = self.work_dir/plan['request_id']
        out.mkdir()
        self._active = {'out': out, 'boundary': boundary, 'plan': copy.deepcopy(plan), 'context': context,
                        'trace_owned': False, 'owned_writes': False, 'created': False, 'record': {'cases': []}}
        s, record = self._active, self._active['record']
        save(out/'predeclared_protocol.json', plan)
        try:
            record['before'] = boundary('/maker/snapshot?regions=all')
            record['expression_before'] = boundary('/maker/face/express')
            record['modifiers_before'] = boundary('/maker/abmx')
            record['original_pose'] = self._capture('original_pose', 2)
            s['original_snapshot'] = bound_read(record['original_pose']['paired_geometry'])
            require(extras_anchor(s['original_snapshot'], self.policy) == plan['outside_original'], 'Original boundary drift before first write')
            original_locals = local_comparison(context['geometry'], s['original_snapshot'])
            require(original_locals['passed'], 'Original unmodified local skeleton moving/changed; explicit unsupported source')
            s['owned_writes'] = True
            boundary('/maker/face/express', 'POST', {'eyes_ptn': 0, 'eyebrow_ptn': 0, 'mouth_ptn': 0,
                'eyes_blink': False, 'eyes_open_max': 1., 'mouth_open_max': 0., 'mouth_open_min': 0., 'mouth_fixed': True})
            boundary('/maker/face/shapes/batch', 'POST', {'values': plan['native59'], 'range_mode': plan['range_mode']})
            selection = self._capture('native_driver_selection', 2)
            drivers = bound_read(selection['paired_geometry'])['native_face_drivers']
            entries = [p for p in drivers['eye_controller']['patterns'] if p['name'] == 'NO_LOOK' and p['value'] == 0]
            require(len(entries) == 1 and type(entries[0]['index']) is int, 'Unique actual runtime NO_LOOK mapping required')
            driver_patch = {'mouth_adjust_width': False, 'eyes_look_pattern': entries[0]['index']}
            driver_readback = boundary('/maker/face/express', 'POST', driver_patch)
            require(all(type(driver_readback.get(k)) is type(v) and driver_readback[k] == v for k, v in driver_patch.items()), 'Actual driver readback rejected')
            record['actual_driver_selection'] = {'capture': selection, 'patch': driver_patch, 'readback': driver_readback}
            s['created'] = True  # A partial transport error still requires removal attempts.
            boundary('/maker/abmx/batch', 'POST', {'bones': [{'name': n, **IDENTITY} for n in NAMES]})
            baseline = self._capture('settled_identity', 5)
            source_start = self._start()
            source_capture = self._capture('source_history', 5, baseline)
            trace = self._stop('source_history_trace.json')
            history = {'trace': trace['path'], 'trace_sha256': trace['sha256'], 'trace_start': source_start,
                       'geometry': source_capture['paired_geometry'], 'capture': source_capture}
            record['source_history'] = history
            s['source_snapshot'] = bound_read(history['geometry'])
            s['outside_source'] = extras_anchor(s['source_snapshot'], self.policy)
            record['outside_source'] = s['outside_source']
            declared = {k: copy.deepcopy(plan[k]) for k in ('semantic_mode', 'head_id', 'native59', 'source_history_native59', 'sampling_profile', 'logical_patches')}
            declared['source_files'] = {**self.bindings, str(out/'predeclared_protocol.json'): file_sha(out/'predeclared_protocol.json')}
            s['backend'] = StrictBackend(self.contract, self.driver_contract)
            built = s['backend'].build(history, declared, read(self.asset_plan)['overrides'], out)
            record.update(compiler_artifact=built['compiler_artifact'], asset_provider=built['asset_provider'])
            s['metadata'] = self._start()
            s['identity_capture'] = self._capture('candidate_identity', 2, baseline)
            identity_snapshot = bound_read(s['identity_capture']['paired_geometry'])
            guard = s['backend'].identity(identity_snapshot, s['metadata'])
            guard['outside_state'] = self._outside_check(identity_snapshot, s['outside_source'])
            descriptor = save(out/'execution_guard.json', guard)
            s['patch'] = boundary('/maker/abmx/batch', 'POST', {'bones': built['patches']})
            s['baseline'], s['windows'] = baseline, []
            for window, settle in (('early', 5), ('late', 18), ('far60', 60)):
                capture = self._capture(window, settle, baseline)
                # Cheap state checks only during observer; heavy independent LBS
                # is intentionally deferred until the trace has stopped.
                self._outside_check(bound_read(capture['paired_geometry']), s['outside_source'])
                s['windows'].append({'window': window, 'head_id': plan['head_id'], 'capture': capture, 'geometry': capture['paired_geometry']})
            execution = {'accepted': True, 'mode': MODE, 'plan_binding_sha256': digest(plan),
                         'request_id': plan['request_id'], 'model_source': built['compiler_artifact'],
                         'requires_actor_transform_coverage': True,
                         'guard_report': descriptor, 'owned_driver_fields': sorted(driver_patch),
                         'owned_expression_patch_fields': ['eyes_ptn', 'eyebrow_ptn', 'mouth_ptn', 'eyes_blink',
                            'eyes_open_max', 'mouth_open_max', 'mouth_open_min', 'mouth_fixed', *sorted(driver_patch)],
                         'owned_expression_values': boundary('/maker/face/express'),
                         'fixed_framing': {k: baseline[k] for k in ('ortho_size', 'distance', 'target')},
                         'runtime_certified': False, 'candidate_proof_pending': True}
            s['execution'] = execution
            return copy.deepcopy(execution)
        except BaseException as error:
            save(out/'apply_failure.json', {'error': repr(error), 'scoring_started': False})
            self.restore(context, boundary)
            raise

    def validate_capture(self, execution, evidence):
        s = self._active
        require(s is not None and execution['request_id'] == s['plan']['request_id']
                and execution['plan_binding_sha256'] == digest(s['plan']), 'Unknown/stale active execution')
        self._sources()
        # Consumer explicitly supplies its saved full response descriptor. Reopen
        # files independently; the evidence booleans are not backend authority.
        supplied = bound_read(evidence['render_response'])
        require(supplied['paired_geometry'] == evidence['geometry'], 'Supplied actual capture geometry differs')
        self._supplied_images(supplied, evidence)
        self._paired(supplied)
        self._outside_check(bound_read(supplied['paired_geometry']), s['outside_source'])
        trace = self._stop('candidate_trace.json')
        case = {'head_id': s['plan']['head_id'], 'names': list(NAMES), 'baseline': s['baseline'],
                'source_history': s['record']['source_history'], 'trace': trace['path'], 'trace_sha256': trace['sha256'],
                'trace_start': s['metadata'], 'identity_capture': s['identity_capture'], 'patch': s['patch'], 'windows': s['windows']}
        s['record']['cases'] = [case]
        # Expensive guards/audits run only after observer STOP and terminal cleanup.
        restoration = self.restore(s['context'], s['boundary'])
        require(restoration['restored'] is True, 'Terminal original public/local/cache restoration failed; no scoring')
        for row in [*s['windows'], {'window': 'supplied', 'capture': supplied}]:
            save(s['out']/(row['window']+'_candidate_driver_asset_check.json'), s['backend'].candidate(row['capture'], s['metadata']))
        manifest = save(s['out']/'live_cases.json', s['record'])
        result = s['backend'].audit(manifest['path'], s['out']/'runtime_internal')
        supplied_manifest = copy.deepcopy(s['record'])
        supplied_manifest['source_manifest'] = manifest
        supplied_manifest['substitution'] = {'old_window': 'early', 'new_capture': evidence['render_response'],
                                              'counts_policy': 'actual recorded cursor; never infer from settle'}
        supplied_manifest['cases'][0]['windows'][0] = {'window': 'early', 'head_id': s['plan']['head_id'],
                                                    'capture': supplied, 'geometry': supplied['paired_geometry']}
        supplied_path = save(s['out']/'supplied_capture_manifest.json', supplied_manifest)
        supplied_result = s['backend'].audit(supplied_path['path'], s['out']/'runtime_supplied')
        require(result.get('passed') is True and supplied_result.get('passed') is True,
                'Independent whole visible-head/trace/temporal supplied/internal proof failed; no scoring')
        reports = [{'path': str(s['out']/p/'summary.json'), 'sha256': file_sha(s['out']/p/'summary.json')}
                   for p in ('runtime_internal', 'runtime_supplied')]
        accepted = {'accepted': True, 'source_bound': True, 'numerical_runtime_accepted': True,
                    'restored_before_acceptance': True, 'restoration_report': restoration['report'],
                    'report_bindings': [*reports, manifest, supplied_path, restoration['report']],
                    'deformation_quality': 'not_certified_by_numerical_gate', 'anatomy_certified': False,
                    'likeness_certified': False, 'runtime_certified': False, 'full_actor_pose_certified': False,
                    'full_actor_restoration_certified': False,
                    'outside_state_policy': self.policy, 'observed_count_policy': 'trace cursors, not commonN or settle count'}
        save(s['out']/'acceptance.json', accepted)
        return accepted

    def restore(self, original_context, boundary):
        """Attempt each cleanup independently; public equality cannot hide local failure."""
        s = self._active
        if s is None:
            return {'restored': True, 'owned_writes': False, 'scope': 'no active candidate'}
        if s.get('restoration') is not None:
            return copy.deepcopy(s['restoration'])
        require(original_context['binding_sha256'] == s['context']['binding_sha256'], 'Wrong original restoration context')
        errors, record = [], s['record']
        def attempt(label, operation):
            try:
                return operation()
            except (Exception, SystemExit, KeyboardInterrupt) as error:  # noqa: BLE001 -- independent recovery attempts must continue
                errors.append({'step': label, 'error': repr(error)})
                return None
        if s['trace_owned']:
            attempt('stop_owned_trace', lambda: self._stop('recovery_trace.json'))
        if s['created']:
            attempt('owned_identity', lambda: boundary('/maker/abmx/batch', 'POST', {'bones': [{'name': n, **IDENTITY} for n in NAMES]}))
            attempt('remove_created_before_native', lambda: boundary('/maker/abmx/batch', 'POST', {'bones': [{'name': n, 'remove': True} for n in NAMES]}))
        if s['owned_writes']:
            def restore_native():
                validate_write(s['context']['native59'], None, s['context']['capabilities'], 'installed')
                return boundary('/maker/face/shapes/batch', 'POST', {'values': s['context']['native59'], 'range_mode': 'installed'})
            attempt('native_original', restore_native)
            attempt('expression_original', lambda: boundary('/maker/face/express', 'POST', {k: v for k, v in record['expression_before'].items() if k != 'sex'}))
        for key, route in (('after', '/maker/snapshot?regions=all'), ('expression_after', '/maker/face/express'), ('modifiers_after', '/maker/abmx')):
            record[key] = attempt(key, lambda route=route: boundary(route))
        record['state_restored'] = record.get('before') == record.get('after') and 'before' in record
        record['expression_restored'] = record.get('expression_before') == record.get('expression_after') and 'expression_before' in record
        record['bone_restored'] = record.get('modifiers_before') == record.get('modifiers_after') and 'modifiers_before' in record
        if record.get('original_pose'):
            record['restored_pose'] = attempt('restored_pose', lambda: self._capture('restored_pose', 2, record['original_pose']))
            if record['restored_pose']:
                def verify():
                    snapshot = bound_read(record['restored_pose']['paired_geometry'])
                    return self._outside_check(snapshot, s['plan']['outside_original'], original=True)
                record['full_exported_local_restoration'] = attempt('actual_original_local_cache', verify)
        record['restore_failures'] = errors
        restored = all(record.get(k) is True for k in ('state_restored', 'expression_restored', 'bone_restored')) and not errors
        restored = restored and record.get('full_exported_local_restoration') is not None
        report = save(s['out']/'restoration_report.json', {
            'restored': restored, 'errors': errors, 'state_restored': record['state_restored'],
            'expression_restored': record['expression_restored'], 'bone_restored': record['bone_restored'],
            'full_exported_local_restoration': record.get('full_exported_local_restoration'),
            'full_actor_pose_certified': False, 'full_actor_restoration_certified': False,
            'unexported_body_locals': s['plan']['outside_original']['unexported_body_local_ids']})
        result = {'restored': restored, 'report': report, 'owned_writes': s['owned_writes'],
                  'full_actor_pose_certified': False, 'full_actor_restoration_certified': False,
                  'scope': 'public configuration + exact outside private state + every exported actor hierarchy/ancestor local'}
        s['restoration'] = result
        if restored:
            self._active = self._prepared = None
        return copy.deepcopy(result)
