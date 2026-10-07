"""Explicit canonical-rewind capture phase; frozen V1 remains unchanged."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from src import hs2_controlled_radial as v1
from src.hs2_controlled_radial import (
    IDENTITY,
    MODE,
    NAMES,
    PROFILE,
    ROOT,
    StrictBackend,
    bound_read,
    extras_anchor,
    local_comparison,
    read,
    save,
    transform_table,
)
from src.hs2_ingame_semantics import digest, file_sha, require

POSE_POLICY = 'canonical_rewind_snapshot_v1'


class ControlledRadialProvider(v1.ControlledRadialProvider):
    """Restore the first declared canonical capture, never claim live pose recovery."""

    def __init__(self, *, original_pose_policy, canonical_settle=60, **kwargs):
        require(original_pose_policy == POSE_POLICY and type(canonical_settle) is int and canonical_settle == 60,
                'Explicit canonical_rewind_snapshot_v1 with declared settle60 required')
        super().__init__(**kwargs)
        self.pose_policy, self.canonical_settle = original_pose_policy, canonical_settle
        self.render_source = ROOT.parent/'HS2Mod/plugins/HS2_McpBridge/MakerRenderService.cs'
        require(self.render_source.is_file(), 'Local native PoseFreeze source unavailable')
        self.bindings.update({str(Path(__file__).resolve()): file_sha(__file__),
                              str(self.render_source): file_sha(self.render_source)})

    def prepare(self, context, candidate_native59, *, logical_controls):
        plan = super().prepare(context, candidate_native59, logical_controls=logical_controls)
        plan.update(provider_revision='controlled_radial_canonical_phase_v2', original_pose_policy=self.pose_policy,
                    original_pose_schedule={'original': 60, 'original_confirm': 60, 'restored': 60},
                    original_live_actor_pose_restoration_certified=False,
                    capture_phase_definition='native freeze_pose=true: Animator.Play(current_state,0,0); Update(0); explicit settle60')
        self._prepared['plan'] = copy.deepcopy(plan)
        return plan

    def _capture(self, name, settle, camera=None):
        if name in ('original_pose', 'original_confirm', 'restored_pose'):
            settle = self.canonical_settle
        result = super()._capture(name, settle, camera)
        result['declared_pose_phase'] = {'policy': self.pose_policy, 'freeze_pose': True,
                                         'settle_frames': settle, 'phase': name}
        return result

    @staticmethod
    def _paired(capture):
        snapshot = v1.ControlledRadialProvider._paired(capture)
        require(type(capture.get('froze_animators')) is int and capture['froze_animators'] > 0,
                'Native export did not confirm canonical animation rewind')
        return snapshot

    @staticmethod
    def _phase_identity(before, after):
        """Bind actor/source/actual local references without equating two phases."""
        require(before['game'] == after['game'], 'Live/canonical installed source changed')
        a, b = before['character'], after['character']
        require(all(type(a[k]) is type(b[k]) and a[k] == b[k] for k in
                ('transform_id', 'head_id', 'head_root_transform_id', 'native_count')),
                'Live/canonical actor/head identity changed')
        require(np.array_equal(np.asarray(a['shape_value_face'], dtype=np.float32),
                               np.asarray(b['shape_value_face'], dtype=np.float32)), 'Live/canonical native59 changed')
        left, right = transform_table(before), transform_table(after)
        require(set(left) == set(right) and all(all(left[i][k] == right[i][k] for k in
                ('id', 'name', 'path', 'parent_id')) for i in left), 'Live/canonical skeleton provenance changed')
        def source(snapshot):
            rows = []
            for mesh in snapshot['meshes']:
                fields = ('renderer_path', 'source_geometry_sha256', 'renderer_id', 'object_id',
                          'renderer_transform_id', 'mesh_instance_id', 'root_bone_transform_id', 'bone_transform_ids')
                require(all(k in mesh for k in fields), 'Actual renderer/component/palette provenance missing')
                require(all(type(mesh[k]) is int for k in ('renderer_id', 'object_id', 'renderer_transform_id', 'mesh_instance_id')),
                        'Actual renderer/component IDs must be integers')
                rows.append({k: copy.deepcopy(mesh[k]) for k in fields})
            require(len({m['renderer_path'] for m in rows}) == len(rows), 'Ambiguous actual renderer paths')
            return sorted(rows, key=lambda m: m['renderer_path'])
        require(digest(source(before)) == digest(source(after)), 'Live/canonical mesh component/asset/palette source changed')
        return {'actor_transform_id': a['transform_id'], 'head_id': a['head_id'],
                'actual_transform_reference_count': len(left), 'mesh_sources': source(before),
                'live_pose_equal_to_canonical_claimed': False}

    @staticmethod
    def _cache_public_only(anchor):
        anchor = copy.deepcopy(anchor)
        for row in anchor['modifiers'].values():
            local = row.pop('actual_local')
            row['actual_transform_reference'] = None if local is None else {
                k: local[k] for k in ('id', 'name', 'path', 'parent_id')}
        return anchor

    def _original_public_check(self, record):
        boundary = self._active['boundary']
        current = {'before': boundary('/maker/snapshot?regions=all'),
                   'expression_before': boundary('/maker/face/express'),
                   'modifiers_before': boundary('/maker/abmx')}
        require(all(digest(current[k]) == digest(record[k]) for k in current),
                'Original public expression/configuration/ABMX changed across canonical captures')
        return {'exact_public_configuration_preserved': True,
                'bindings': {k: digest(v) for k, v in current.items()}}

    def _establish_canonical_original(self, plan, context, record):
        s = self._active
        record['original_pose_policy'] = self.pose_policy
        record['original_pose_schedule'] = copy.deepcopy(plan['original_pose_schedule'])
        record['original_pose'] = self._capture('original_pose', self.canonical_settle)
        first = bound_read(record['original_pose']['paired_geometry'])
        s['original_snapshot'] = first
        identity = self._phase_identity(context['geometry'], first)
        original_anchor = self._cache_public_only(plan['outside_original'])
        require(digest(self._cache_public_only(extras_anchor(first, self.policy))) == digest(original_anchor),
                'Original public/private cache drift across live/canonical phase')
        differences = local_comparison(context['geometry'], first)
        live_report = {'phase_before': 'live_animated_preflight', 'phase_after': self.pose_policy,
                       'native_posefreeze_source': {'path': str(self.render_source), 'sha256': self.bindings[str(self.render_source)]},
                       'source_live_json_sha256': digest(context['geometry']),
                       'canonical_geometry': record['original_pose']['paired_geometry'],
                       'provenance': identity, 'actual_local_differences': differences,
                       'local_differences_not_individually_attributed_to_rewind': True,
                       'original_live_actor_pose_restoration_certified': False}
        record['live_to_canonical_observation'] = save(s['out']/'live_to_canonical_observation.json', live_report)
        self._original_public_check(record)
        # Independent second observation occurs before expression/native/ABMX
        # ownership. Unknown drift WITHIN the canonical phase still refuses.
        record['original_confirm'] = self._capture('original_confirm', self.canonical_settle, record['original_pose'])
        second = bound_read(record['original_confirm']['paired_geometry'])
        self._phase_identity(first, second)
        require(digest(self._cache_public_only(extras_anchor(second, self.policy))) == digest(original_anchor),
                'Original private/public cache drift within canonical phase')
        repeat = local_comparison(first, second)
        report = {'policy': self.pose_policy, 'schedule': plan['original_pose_schedule'],
                  'first': record['original_pose']['paired_geometry'], 'second': record['original_confirm']['paired_geometry'],
                  'actual_local_comparison': repeat, 'public_check': self._original_public_check(record),
                  'before_owned_writes': True, 'source_or_candidate_assets_used': False,
                  'original_live_actor_pose_restoration_certified': False}
        record['canonical_original_guard'] = save(s['out']/'canonical_original_guard.json', report)
        require(repeat['passed'], 'Original actual skeleton drift within declared canonical phase; no owned writes')

    def restore(self, original_context, boundary):
        s = self._active
        result = super().restore(original_context, boundary)
        if s is None or 'report' not in result:
            return result
        if s.get('canonical_restoration') is not None:
            return copy.deepcopy(s['canonical_restoration'])
        annotated = {'restored': result['restored'], 'base_restoration_report': result['report'],
                     'original_pose_policy': self.pose_policy, 'original_pose_schedule': s['plan']['original_pose_schedule'],
                     'canonical_original_guard': s['record'].get('canonical_original_guard'),
                     'live_to_canonical_observation': s['record'].get('live_to_canonical_observation'),
                     'canonical_reference': s['record'].get('original_pose', {}).get('paired_geometry'),
                     'original_live_actor_pose_restoration_certified': False,
                     'full_actor_restoration_certified': False,
                     'interpretation': 'Declared first canonical snapshot is the local restoration reference; original live animated TRS is not certified'}
        result['report'] = save(s['out']/'canonical_restoration_report.json', annotated)
        result['original_pose_policy'] = self.pose_policy
        result['original_live_actor_pose_restoration_certified'] = False
        s['canonical_restoration'] = copy.deepcopy(result)
        return result

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
                         'original_pose_policy': self.pose_policy,
                         'canonical_original_guard': record['canonical_original_guard'],
                         'original_live_actor_pose_restoration_certified': False,
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
