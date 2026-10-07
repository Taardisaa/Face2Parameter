"""Evidence corruptions and differentiable four-bone candidate boundaries."""
import copy
from pathlib import Path
import unittest
import torch
import numpy as np
from tools.stateful_multibone.adapter import (DEFAULT_BONES, IDENTITY, MultiNativeBaselineProtocol,
                                              MultiStatefulTorchHeadRig, read, sha)
from src.hs2_mesh_deform import HeadRig
from tools.abmx_replay.model import replay_apply


def analytic_group(actual_trace, actual_snapshot):
    """Synthetic uneven schedule with independently generated NumPy transitions."""
    trace = copy.deepcopy(actual_trace)
    trace['events'] = []
    trace['metadata'].update(started_frame=1, stopped_frame=3, session_id='synthetic-uneven')
    initial = {name: copy.deepcopy(next(e for e in actual_trace['events'] if e['bone_name'] == name)) for name in DEFAULT_BONES}
    last = {name: copy.deepcopy(event['before']) for name, event in initial.items()}
    candidate = {name: copy.deepcopy(next(e['resolved_modifier'] for e in actual_trace['events']
                  if e['bone_name'] == name and e['resolved_modifier']['length'] != 1)) for name in DEFAULT_BONES}
    for frame, names in ((1, DEFAULT_BONES), (2, DEFAULT_BONES), (3, DEFAULT_BONES[:1])):
        for name in names:
            event = copy.deepcopy(initial[name])
            event.update(sequence=len(trace['events'])+1, frame=frame, completed_frame=frame)
            event['before'] = copy.deepcopy(last[name])
            event['before']['cache']['frame_count'] = frame
            if frame > 1:
                event['resolved_modifier'] = candidate[name]
            prediction = replay_apply(before={key: event['before'][key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')},
                cache=event['before']['cache']['fields'], coordinate_modifiers=[event['resolved_modifier']], coordinate=0,
                additional_modifiers=[], bone_exists=True, rotation_excluded=False, is_during_h_scene=False)
            event['after'] = {**copy.deepcopy(event['before']), **prediction['after']}
            event['after']['cache']['fields'] = prediction['cache_after']
            trace['events'].append(event)
            last[name] = event['after']
    trace['observed_calls'] = len(trace['events'])
    snapshot = {'character': copy.deepcopy(actual_snapshot['character']), 'frame_count': 3, 'frame_count_end': 3,
                'abmx_trace_cursor': {'session_id': 'synthetic-uneven', 'frame': 3, 'last_completed_sequence': 9,
                                     'observed_calls': 9, 'completed_calls': 9, 'pending_calls': 0,
                                     'dropped_events': 0, 'observer_error_count': 0}, 'transforms': [], 'abmx_runtime': {'bones': []}}
    for name, state in last.items():
        snapshot['transforms'].append({'id': state['bone_transform_id'], 'name': name, 'path': '/synthetic/'+name,
                                      **{key: state[key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')}})
        wrapper = copy.deepcopy(state['cache'])
        wrapper['frame_count'] = 3
        snapshot['abmx_runtime']['bones'].append({'name': name, 'runtime_baseline': wrapper})
    return trace, snapshot

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_v1/live_cases.json'
CONTRACT = ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json'


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        manifest = read(MANIFEST)
        cls.case = manifest['cases'][0]
        cls.trace = read(cls.case['trace'])
        cls.snapshot_path = cls.case['windows'][0]['geometry']['path']
        cls.snapshot = read(cls.snapshot_path)
        cls.contract = read(CONTRACT)
        cls.rig = HeadRig(0, sampling_profile='slider_unlocker_18_2')

    def protocol(self, trace=None, snapshot=None, contract=None, count=None):
        return MultiNativeBaselineProtocol(self.trace if trace is None else trace,
            self.contract if contract is None else contract, self.snapshot if snapshot is None else snapshot,
            trace_path=self.case['trace'], snapshot_path=self.snapshot_path, contract_path=CONTRACT,
            common_apply_count=count)

    def model_inputs(self, count=None):
        protocol = self.protocol(count=count)
        model = MultiStatefulTorchHeadRig(self.rig, protocol, device='cpu', dtype=torch.float64)
        native = torch.as_tensor(protocol.reference_native[None], dtype=torch.float64)
        patches = {name: {'scale': v[:3], 'length': v[3], 'position': v[4:7], 'rotation': v[7:]}
                   for name, v in protocol.reference_modifiers.items()}
        return protocol, model, native, model.ab_tensor(patches)

    def test_recorded_prefix_independent_counts_and_persistent_history(self):
        p = self.protocol()
        self.assertEqual(set(p.observed_counts), set(DEFAULT_BONES))
        for name in DEFAULT_BONES:
            event = next(e for e in self.trace['events'] if e['bone_name'] == name)
            self.assertEqual(p.fields[name]['_lenBaseline'], event['before']['cache']['fields']['_lenBaseline'])
            self.assertEqual(p.fields[name]['_positionBaseline'], event['before']['cache']['fields']['_positionBaseline'])
        self.assertFalse(p.metadata['new_candidate_runtime_certified'])

    def test_entire_trace_later_corruption_rejected_even_before_cursor(self):
        t = copy.deepcopy(self.trace)
        t['events'][-1]['after']['local_scale'][0] += .1
        with self.assertRaises(ValueError):
            self.protocol(trace=t)

    def test_source_assembly_and_decompiled_contract_rejected(self):
        for source in ('assembly_sha256', 'sources'):
            contract = copy.deepcopy(self.contract)
            if source == 'sources':
                contract['sources'][0]['sha256'] = '0'*64
            else:
                contract[source] = '0'*64
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'source'):
                self.protocol(contract=contract)

    def test_wrong_source_mvid_and_external_patch_rejected(self):
        for key, value in (('plugin_mvid', 'foreign'), ('pre_existing_patch_owners', ['unknown.plugin'])):
            t = copy.deepcopy(self.trace)
            t['metadata'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.protocol(trace=t)

    def test_cursor_session_counts_frame_pending_future_rejected(self):
        for key, value in (('session_id', 'foreign'), ('observed_calls', 1), ('frame', -1),
                           ('pending_calls', 1), ('last_completed_sequence', len(self.trace['events']))):
            snapshot = copy.deepcopy(self.snapshot)
            snapshot['abmx_trace_cursor'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.protocol(snapshot=snapshot)

    def test_missing_bone_filter_and_history_rejected(self):
        for change in ('filter', 'history'):
            t = copy.deepcopy(self.trace)
            if change == 'filter':
                t['metadata']['requested_names'].pop()
            else:
                t['events'][3]['before']['cache']['fields'].pop('_lenBaseline')
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.protocol(trace=t)

    def test_dirty_flags_on_any_first_bone_rejected(self):
        for name in DEFAULT_BONES:
            t = copy.deepcopy(self.trace)
            event = next(e for e in t['events'] if e['bone_name'] == name)
            for phase in ('before', 'after'):
                event[phase]['cache']['fields']['_changedScale'] = True
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.protocol(trace=t)

    def test_native_cache_guard_each_bone(self):
        for name in DEFAULT_BONES:
            p = self.protocol()
            p.fields[name]['_posBaseline'][0] += .1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'baseline'):
                MultiStatefulTorchHeadRig(self.rig, p, device='cpu', dtype=torch.float64)

    def test_new_counts_must_be_declared_common_positive_integer(self):
        for count in (True, 0, -1, 1.5, 10001, {DEFAULT_BONES[0]: 4}):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'common Apply'):
                self.protocol(count=count)
        p = self.protocol(count=5)
        self.assertEqual(set(p.counts.values()), {5})

    def test_recorded_count_cannot_silently_generalize_native_or_abmx(self):
        _, model, native, ab = self.model_inputs()
        for change in ('native', 'patch'):
            x, a = native.clone(), ab.clone()
            if change == 'native':
                x[0, 0] += .01
            else:
                a[0, model.slots[DEFAULT_BONES[0]], 0] += .01
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'declared common'):
                model.local_transforms(x, a)

    def test_other_bone_static_fallback_and_nonfinite_rejected(self):
        _, model, native, ab = self.model_inputs(count=3)
        other = next(i for i in range(len(model.ab_names)) if i not in model.slots.values())
        ab[0, other, 0] = 1.01
        with self.assertRaisesRegex(ValueError, 'static fallback'):
            model.local_transforms(native, ab)
        ab[0, other] = torch.as_tensor(IDENTITY)
        ab[0, model.slots[DEFAULT_BONES[0]], 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            model.local_transforms(native, ab)

    def test_float64_and_explicit_profile_required(self):
        p = self.protocol()
        with self.assertRaisesRegex(ValueError, 'float64'):
            MultiStatefulTorchHeadRig(self.rig, p, device='cpu')
        with self.assertRaisesRegex(ValueError, 'sampling profile'):
            MultiStatefulTorchHeadRig(HeadRig(0, sampling_profile='vanilla'), p, device='cpu', dtype=torch.float64)

    def test_candidate_states_preserve_history_and_support_gradients(self):
        p, model, native, ab = self.model_inputs(count=4)
        native = native.clone().requires_grad_()
        ab = ab.clone().requires_grad_()
        (_, _, _), _, caches = model.local_states(native, ab)
        for name, cache in caches.items():
            self.assertEqual(cache['_lenBaseline'].item(), float(np.float32(p.fields[name]['_lenBaseline'])))
            torch.testing.assert_close(cache['_positionBaseline'][0], torch.tensor(p.fields[name]['_positionBaseline'], dtype=torch.float32))
        vertices = model(native, ab)
        vertices.square().mean().backward()
        self.assertTrue(torch.isfinite(native.grad).all())
        self.assertTrue(torch.isfinite(ab.grad).all())
        for slot in model.slots.values():
            self.assertGreater(ab.grad[0, slot].abs().sum().item(), 0)

    def test_actual_after_in_metadata_not_candidate_input(self):
        p, model, native, ab = self.model_inputs(count=3)
        with torch.no_grad():
            before = model(native, ab)
            for row in p.metadata['cursor_binding']['selected_bones']:
                row['prediction']['after']['local_position'] = [100, 100, 100]
            after = model(native, ab)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_uneven_observed_counts_are_per_bone_not_global_or_settle(self):
        trace, snapshot = analytic_group(self.trace, self.snapshot)
        p = self.protocol(trace=trace, snapshot=snapshot)
        self.assertEqual(p.observed_counts, {name: 2 if name == DEFAULT_BONES[0] else 1 for name in DEFAULT_BONES})
        self.assertEqual(p.metadata['cursor_sequence'], 9)

    def test_valid_but_external_boundary_cannot_anchor_candidate(self):
        trace, snapshot = analytic_group(self.trace, self.snapshot)
        event = trace['events'][-1]
        event['before']['local_position'][0] += .01
        prediction = replay_apply(before={key: event['before'][key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')},
            cache=event['before']['cache']['fields'], coordinate_modifiers=[event['resolved_modifier']], coordinate=0,
            additional_modifiers=[], bone_exists=True, rotation_excluded=False, is_during_h_scene=False)
        event['after'].update(prediction['after'])
        event['after']['cache']['fields'] = prediction['cache_after']
        bone = snapshot['transforms'][0]
        bone.update(prediction['after'])
        snapshot['abmx_runtime']['bones'][0]['runtime_baseline']['fields'] = prediction['cache_after']
        with self.assertRaisesRegex(ValueError, 'external'):
            self.protocol(trace=trace, snapshot=snapshot)

    def test_dictionary_unknown_or_missing_bones_rejected(self):
        _, model, _, _ = self.model_inputs(count=3)
        for data in ({}, {'unknown': {}}, {name: {} for name in DEFAULT_BONES}):
            with self.assertRaises(ValueError):
                model.ab_tensor(data)


if __name__ == '__main__':
    unittest.main()
