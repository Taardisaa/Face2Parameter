"""Explicit V3: full consumer views inside a bounded observer lifetime."""
from __future__ import annotations

import copy
from pathlib import Path
from urllib.parse import urlencode

from src import hs2_controlled_radial_v2 as v2
from src.hs2_controlled_radial import (
    IDENTITY,
    MODE,
    NAMES,
    PROFILE,
    bound_read,
    extras_anchor,
    read,
    save,
)
from src.hs2_ingame_semantics import digest, file_sha, require
from tools.native_radial_target.capture_groups_v3 import (
    make_schedule,
    prepare_capture_groups,
)
from tools.native_radial_target.runtime_backend_v3 import (
    REVISION,
    StrictBackendV3,
    dependencies,
)


class ControlledRadialProvider(v2.ControlledRadialProvider):
    def __init__(self, *, capture_backend_revision, **kwargs):
        require(capture_backend_revision == REVISION, 'Explicit multigroup V3 backend required')
        super().__init__(**kwargs)
        self.bindings.update(dependencies())
        self.consumer_plan = None
        self.capture_contracts = {}

    def declare_consumer_capture(self, groups, *, settle, hide_hair, options):
        require(self._active is None and self._prepared is None, 'Cannot change an active capture plan')
        require(type(settle) is int and 0 <= settle <= 60, 'Strict declared consumer settle required')
        require(type(hide_hair) is int and hide_hair == 1, 'Initial explicit profile hides hair')
        require(type(options) is dict and options == {'settle': settle, 'aa': 1},
                'Initial V3 capture profile requires explicit AA1 and no extra camera options')
        schedule = make_schedule(groups=groups, width=self.resolution, height=self.resolution, phase='early')
        self.consumer_settle = settle
        self.consumer_plan = {'schedule': schedule, 'settle': settle, 'hide_hair': hide_hair,
                              'options': copy.deepcopy(options)}

    def prepare(self, context, candidate_native59, *, logical_controls):
        require(self.consumer_plan is not None, 'Consumer must predeclare its entire view request')
        plan = super().prepare(context, candidate_native59, logical_controls=logical_controls)
        self.capture_contracts = {}
        plan.update(provider_revision='controlled_radial_multigroup_phase_v3',
                    capture_backend_revision=REVISION, consumer_capture_plan=copy.deepcopy(self.consumer_plan),
                    observer_schedule={'candidate_identity': 2, 'early': self.consumer_settle, 'late': 18, 'far60': 60},
                    observer_stop='immediately_after_far60_before_file_checks_or_return')
        phases = ('original_pose', 'original_confirm', 'restored_pose', 'native_driver_selection',
                  'settled_identity', 'source_history', 'candidate_identity', 'early', 'late', 'far60')
        plan['view_schedules'] = {
            phase: make_schedule(groups=
                [{'id': g['id'], 'yaws': g['yaws']} for g in self.consumer_plan['schedule']['groups']]
                if phase == 'early' else [{'id': 'a', 'yaws': [-30, 0, 30]}],
                width=self.resolution, height=self.resolution, phase=phase)
            for phase in phases}
        source = context['geometry']
        anchor = {'actor_transform_id': source['character']['transform_id'], 'head_id': plan['head_id'],
                  'bridge_mvid': read(self.driver_contract)['bridge_mvid'],
                  'head_meshes': [{'renderer_path': m['renderer_path'], 'mesh_name': m['mesh_name'],
                                   'source_geometry_sha256': m['source_geometry_sha256']} for m in source['meshes']
                                 if m['enabled'] and m['active_in_hierarchy']],
                  'transform_scope': 'actor_hierarchy_and_ancestors'}
        plan['phase_sources'] = {
            phase: {**copy.deepcopy(anchor), 'native59': copy.deepcopy(context['native59']
                if phase in ('original_pose', 'original_confirm', 'restored_pose') else plan['native59'])}
            for phase in phases}
        self._prepared['plan'] = copy.deepcopy(plan)
        return plan

    def _capture(self, name, settle, camera=None):
        state = self._active
        if name in ('original_pose', 'original_confirm', 'restored_pose'):
            settle = self.canonical_settle
        require(name in state['plan']['view_schedules'], 'Capture phase was not predeclared')
        schedule = state['plan']['view_schedules'][name]
        yaws = schedule['flat_yaws']
        query = {'out': str(state['out']/(name+'.png')), 'w': self.resolution, 'h': self.resolution,
                 'aa': 1, 'yaws': ','.join(map(str, yaws)), 'hide_meshes': 'o_body_cf',
                 'hide_hair': 1, 'freeze_pose': 'true', 'settle': settle,
                 'geometry_out': str(state['out']/(name+'.json')),
                 'geometry_blendshape_frames': 'true', 'geometry_include_actor_transforms': 'true'}
        if camera:
            query.update(ortho_size=camera['ortho_size'], dist=camera['distance'],
                         target=','.join(map(str, camera['target'])))
        result = state['boundary']('/maker/render?'+urlencode(query), 'POST', timeout=180)
        response = save(state['out']/(name+'_response.json'), result)
        expected = state['plan']['phase_sources'][name]
        self.capture_contracts[str(Path(result['paired_geometry']['path']).resolve())] = {
            'schedule': schedule, 'expected_source': expected, 'response': response,
            'response_digest': digest(result),
            'cursor_policy': 'complete_active' if state['trace_owned'] else 'complete_or_inactive'}
        # Do not reopen/parse whole geometry while observation is active.
        if not state['trace_owned']:
            self._paired(result)
        return result

    def _paired(self, capture):
        key = str(Path(capture['paired_geometry']['path']).resolve())
        require(key in self.capture_contracts, 'Unregistered actual capture')
        entry = self.capture_contracts[key]
        require(digest(capture) == entry['response_digest'], 'Complete native capture response changed')
        context = prepare_capture_groups(entry['schedule'], entry['expected_source'], cursor_policy=entry['cursor_policy'])
        context.verify_response(entry['response']['path'], entry['response']['sha256'], capture['paired_geometry'])
        require(type(capture.get('froze_animators')) is int and capture['froze_animators'] > 0,
                'Native canonical rewind not confirmed')
        return bound_read(capture['paired_geometry'])

    def validate_capture(self, execution, evidence):
        s = self._active
        require(s is not None and not s['trace_owned'] and execution.get('observer_stopped_before_return') is True,
                'Explicit V3 stopped observer required before consumer validation')
        require(execution['request_id'] == s['plan']['request_id']
                and execution['plan_binding_sha256'] == digest(s['plan']), 'Stale active execution')
        self._sources()
        supplied = bound_read(evidence['render_response'])
        early = s['windows'][0]['capture']
        require(digest(supplied) == digest(early) and supplied['paired_geometry'] == evidence['geometry'],
                'Consumer must supply the unchanged predeclared actual six-view early capture')
        self._supplied_images(supplied, evidence)
        self._paired(supplied)
        self._outside_check(bound_read(supplied['paired_geometry']), s['outside_source'])
        case = {'head_id': s['plan']['head_id'], 'names': list(NAMES), 'baseline': s['baseline'],
                'source_history': s['record']['source_history'], 'trace': s['candidate_trace']['path'],
                'trace_sha256': s['candidate_trace']['sha256'], 'trace_start': s['metadata'],
                'identity_capture': s['identity_capture'], 'patch': s['patch'], 'windows': s['windows']}
        s['record']['cases'] = [case]
        restoration = self.restore(s['context'], s['boundary'])
        require(restoration['restored'] is True, 'Terminal public/local/cache restoration failed; no scoring')
        # Restoration is another explicitly predeclared three-view observation.
        s['record']['capture_contracts'] = copy.deepcopy(self.capture_contracts)
        s['backend'].capture_contracts = copy.deepcopy(self.capture_contracts)
        for row in s['windows']:
            save(s['out']/(row['window']+'_candidate_driver_asset_check.json'),
                 s['backend'].candidate(row['capture'], s['metadata']))
        manifest = save(s['out']/'live_cases.json', s['record'])
        result = s['backend'].audit(manifest['path'], s['out']/'runtime_internal')
        supplied_manifest = copy.deepcopy(s['record'])
        supplied_manifest['consumer_response'] = evidence['render_response']
        supplied_path = save(s['out']/'supplied_capture_manifest.json', supplied_manifest)
        supplied_result = s['backend'].audit(supplied_path['path'], s['out']/'runtime_supplied')
        require(result.get('passed') is True and supplied_result.get('passed') is True,
                'Whole visible-head/trace/temporal independent proof failed; no scoring')
        reports = [{'path': str(s['out']/p/'summary_v3.json'), 'sha256': file_sha(s['out']/p/'summary_v3.json')}
                   for p in ('runtime_internal', 'runtime_supplied')]
        accepted = {'accepted': True, 'source_bound': True, 'numerical_runtime_accepted': True,
                    'capture_backend_revision': REVISION, 'restored_before_acceptance': True,
                    'restoration_report': restoration['report'],
                    'report_bindings': [*reports, manifest, supplied_path, restoration['report']],
                    'deformation_quality': 'not_certified_by_numerical_gate', 'anatomy_certified': False,
                    'likeness_certified': False, 'runtime_certified': False, 'full_actor_pose_certified': False,
                    'full_actor_restoration_certified': False, 'outside_state_policy': self.policy,
                    'observed_count_policy': 'actual trace cursors; no dropped events or inferred settle counts'}
        save(s['out']/'acceptance.json', accepted)
        return accepted

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
            self._establish_canonical_original(plan, context, record)
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
            self._paired(source_capture)
            history = {'trace': trace['path'], 'trace_sha256': trace['sha256'], 'trace_start': source_start,
                       'geometry': source_capture['paired_geometry'], 'capture': source_capture}
            record['source_history'] = history
            s['source_snapshot'] = bound_read(history['geometry'])
            s['outside_source'] = extras_anchor(s['source_snapshot'], self.policy)
            record['outside_source'] = s['outside_source']
            declared = {k: copy.deepcopy(plan[k]) for k in ('semantic_mode', 'head_id', 'native59', 'source_history_native59', 'sampling_profile', 'logical_patches')}
            declared['source_files'] = {**self.bindings, str(out/'predeclared_protocol.json'): file_sha(out/'predeclared_protocol.json')}
            s['backend'] = StrictBackendV3(self.contract, self.driver_contract, self.capture_contracts)
            built = s['backend'].build(history, declared, read(self.asset_plan)['overrides'], out)
            record.update(compiler_artifact=built['compiler_artifact'], asset_provider=built['asset_provider'])
            s['metadata'] = self._start()
            s['identity_capture'] = self._capture('candidate_identity', 2, baseline)
            # Actual identity is a prerequisite for the physical candidate;
            # malformed/stale identity must refuse before nonidentity writes.
            identity_snapshot = self._paired(s['identity_capture'])
            guard = s['backend'].identity(identity_snapshot, s['metadata'])
            guard['outside_state'] = self._outside_check(identity_snapshot, s['outside_source'])
            descriptor = save(out/'execution_guard.json', guard)
            s['patch'] = boundary('/maker/abmx/batch', 'POST', {'bones': built['patches']})
            s['baseline'], s['windows'] = baseline, []
            for window, settle in (('early', self.consumer_settle), ('late', 18), ('far60', 60)):
                capture = self._capture(window, settle, baseline)
                s['windows'].append({'window': window, 'head_id': plan['head_id'],
                    'capture': capture, 'geometry': capture['paired_geometry']})
            # STOP immediately after the last actual capture. Candidate file,
            # local, asset, PNG and mathematical guards run after observation.
            s['candidate_trace'] = self._stop('candidate_trace.json')
            s['backend'].capture_contracts = copy.deepcopy(self.capture_contracts)
            for row in s['windows']:
                snapshot = self._paired(row['capture'])
                self._outside_check(snapshot, s['outside_source'])
            record['capture_backend_revision'] = REVISION
            record['capture_contracts'] = copy.deepcopy(self.capture_contracts)
            execution = {'accepted': True, 'mode': MODE, 'plan_binding_sha256': digest(plan),
                         'request_id': plan['request_id'], 'model_source': built['compiler_artifact'],
                         'original_pose_policy': self.pose_policy,
                         'canonical_original_guard': record['canonical_original_guard'],
                         'original_live_actor_pose_restoration_certified': False,
                         'requires_actor_transform_coverage': True,
                         'guard_report': descriptor, 'owned_driver_fields': sorted(driver_patch),
                         'owned_expression_patch_fields': ['eyes_ptn', 'eyebrow_ptn', 'mouth_ptn', 'eyes_blink',
                            'eyes_open_max', 'mouth_open_max', 'mouth_open_min', 'mouth_fixed', *sorted(driver_patch)],
                         'owned_expression_values': boundary('/maker/face/express'),
                         'fixed_framing': {k: baseline[k] for k in ('ortho_size', 'distance', 'target')},
                         'runtime_certified': False, 'candidate_proof_pending': True,
                         'capture_backend_revision': REVISION, 'observer_stopped_before_return': True,
                         'predeclared_consumer_capture': self.capture_contracts[str(Path(s['windows'][0]['geometry']['path']).resolve())]['response'],
                         'consumer_capture_plan': copy.deepcopy(self.consumer_plan)}
            s['execution'] = execution
            return copy.deepcopy(execution)
        except BaseException as error:
            save(out/'apply_failure.json', {'error': repr(error), 'scoring_started': False})
            self.restore(context, boundary)
            raise
