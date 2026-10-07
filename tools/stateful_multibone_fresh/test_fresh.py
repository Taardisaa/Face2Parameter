"""Synthetic lifecycle guards, using earlier actual source contract/history."""
import copy
from pathlib import Path
import unittest
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig
from src.hs2_deform_torch import TorchHeadRig
from tools.stateful_multibone.adapter import DEFAULT_BONES, read
from tools.stateful_multibone_fresh.adapter import EarlierHistoryProtocol, FreshObservedCountTorchRig
from tools.abmx_replay.model import replay_apply

ROOT = Path(__file__).resolve().parents[2]


def synthetic_capture(metadata, first, candidate, frame_start, native, char):
    trace = {'metadata': copy.deepcopy(metadata), 'active': False, 'max_events': 1000,
             'trace_complete': True, 'dropped_events': 0, 'pending_calls': 0, 'observer_errors': [], 'events': []}
    final = {name: copy.deepcopy(row['before']) for name, row in first.items()}
    phases = ((frame_start, DEFAULT_BONES), (frame_start+1, DEFAULT_BONES))
    if candidate is not None:
        phases += ((frame_start+2, DEFAULT_BONES[:1]),)
    for frame, names in phases:
        for name in names:
            event = copy.deepcopy(first[name])
            event.update(sequence=len(trace['events'])+1, frame=frame, completed_frame=frame)
            event['before'] = copy.deepcopy(final[name])
            event['before']['cache']['frame_count'] = frame
            if candidate is not None and frame > frame_start:
                event['resolved_modifier'] = copy.deepcopy(candidate[name])
            prediction = replay_apply(before={key: event['before'][key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')},
                cache=event['before']['cache']['fields'], coordinate_modifiers=[event['resolved_modifier']], coordinate=0,
                additional_modifiers=[], bone_exists=True, rotation_excluded=False, is_during_h_scene=False)
            event['after'] = {**copy.deepcopy(event['before']), **prediction['after']}
            event['after']['cache']['fields'] = prediction['cache_after']
            final[name] = event['after']
            trace['events'].append(event)
    frame = phases[-1][0]
    trace['metadata'].update(started_frame=frame_start, stopped_frame=frame,
                              session_id='synthetic-neutral' if candidate is None else 'synthetic-candidate')
    trace['observed_calls'] = len(trace['events'])
    snapshot = {'character': {**copy.deepcopy(char), 'shape_value_face': list(native)}, 'frame_count': frame,
                'frame_count_end': frame, 'transforms': [], 'abmx_runtime': {'bones': []},
                'abmx_trace_cursor': {'session_id': trace['metadata']['session_id'], 'frame': frame,
                    'last_completed_sequence': len(trace['events']), 'observed_calls': len(trace['events']),
                    'completed_calls': len(trace['events']), 'pending_calls': 0, 'dropped_events': 0, 'observer_error_count': 0}}
    for name, state in final.items():
        snapshot['transforms'].append({'id': state['bone_transform_id'], 'name': name, 'path': '/synthetic/'+name,
                                      **{key: state[key] for key in ('local_position', 'local_rotation_xyzw', 'local_scale')}})
        wrapper = copy.deepcopy(state['cache'])
        wrapper['frame_count'] = frame
        snapshot['abmx_runtime']['bones'].append({'name': name, 'runtime_baseline': wrapper})
    return trace, snapshot


class FreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        manifest = read(ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261005/abmx_multibone_v1/live_cases.json')
        case = manifest['cases'][0]
        actual = read(case['trace'])
        snapshot = read(case['windows'][0]['geometry']['path'])
        cls.contract = read(ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json')
        cls.rig = HeadRig(0, sampling_profile='slider_unlocker_18_2')
        cls.declared = read(manifest['protocol_path'])
        cls.declared['native59'] = [[.48, .52, .47, .53][i % 4] for i in range(59)]
        cls.declared.update(sampling_profile='slider_unlocker_18_2', normalized_whole_head_gate=1e-5,
                            common_apply_count=None, common_count_runtime_certified=False,
                            source_history_expected_native59=[.5]*59,
                            candidate_expected_native59=cls.declared['native59'], baseline_expected_native59=cls.declared['native59'])
        first = {name: copy.deepcopy(next(e for e in actual['events'] if e['bone_name'] == name)) for name in DEFAULT_BONES}
        source_trace, source_snapshot = synthetic_capture(actual['metadata'], first, None, 1, [.5]*59, snapshot['character'])
        native_model = TorchHeadRig(cls.rig, device='cpu', dtype=torch.float64)
        with torch.no_grad():
            pos, quat, scale = native_model.local_transforms(torch.tensor([cls.declared['native59']], dtype=torch.float64))
        candidate_first = copy.deepcopy(first)
        for name, event in candidate_first.items():
            index = native_model.name2bone[name]
            for key, field, value in (('local_position', '_posBaseline', pos), ('local_rotation_xyzw', '_rotBaseline', quat), ('local_scale', '_sclBaseline', scale)):
                vector = value[0, index].float().tolist()
                event['before'][key] = vector
                event['before']['cache']['fields'][field] = vector
        patches = {row['name']: {key: value for key, value in row.items() if key != 'name'} for row in cls.declared['patches']}
        candidate_trace, candidate_snapshot = synthetic_capture(actual['metadata'], candidate_first, patches, 4,
                                                                cls.declared['native59'], snapshot['character'])
        cls.fixture = (source_trace, source_snapshot, candidate_trace, candidate_snapshot)
        cls.patches = patches

    def protocol(self, fixture=None):
        return EarlierHistoryProtocol(*(self.fixture if fixture is None else fixture), self.contract, self.declared)

    def test_changed_candidate_initial_trs_derived_from_predeclaration(self):
        p = self.protocol()
        m = FreshObservedCountTorchRig(self.rig, p, device='cpu', dtype=torch.float64)
        self.assertTrue(all(check['before']['passed'] and check['baseline']['passed'] for check in m.protocol_metadata['native_boundary_checks'].values()))
        self.assertFalse(p.metadata['common_N_runtime_certified'])
        self.assertEqual(p.counts, {name: 2 if name == DEFAULT_BONES[0] else 1 for name in DEFAULT_BONES})

    def test_candidate_before_target_not_injected_into_prediction(self):
        p = self.protocol()
        m = FreshObservedCountTorchRig(self.rig, p, device='cpu', dtype=torch.float64)
        native, ab = torch.tensor([p.candidate_native.tolist()], dtype=torch.float64), m.ab_tensor(self.patches)
        with torch.no_grad():
            before = m(native, ab)
            for name in DEFAULT_BONES:
                p.candidate_before[name]['local_position'] = [99, 99, 99]
                p.candidate_baseline[name]['local_position'] = [99, 99, 99]
            after = m(native, ab)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_history_change_rejected_for_every_bone(self):
        for name in DEFAULT_BONES:
            fixture = copy.deepcopy(self.fixture)
            event = next(e for e in fixture[2]['events'] if e['bone_name'] == name)
            event['before']['cache']['fields']['_lenBaseline'] += .01
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.protocol(fixture)

    def test_same_native_not_a_fresh_varied_diagnostic(self):
        fixture = copy.deepcopy(self.fixture)
        fixture[1]['character']['shape_value_face'] = self.declared['native59']
        with self.assertRaisesRegex(ValueError, 'predeclared source history'):
            self.protocol(fixture)

    def test_history_measured_later_or_another_character_rejected(self):
        for change in ('frame', 'character', 'session'):
            fixture = copy.deepcopy(self.fixture)
            if change == 'frame':
                fixture[0]['metadata']['stopped_frame'] = 5
            elif change == 'character':
                fixture[1]['character']['head_id'] = 1
            else:
                fixture[2]['metadata']['session_id'] = fixture[0]['metadata']['session_id']
                fixture[3]['abmx_trace_cursor']['session_id'] = fixture[0]['metadata']['session_id']
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.protocol(fixture)

    def test_native_before_guard_each_bone(self):
        for name in DEFAULT_BONES:
            p = self.protocol()
            p.candidate_before[name]['local_position'][0] += .01
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'candidate actual before'):
                FreshObservedCountTorchRig(self.rig, p, device='cpu', dtype=torch.float64)

    def test_observed_counts_cannot_generalize_native_or_patch(self):
        p = self.protocol()
        m = FreshObservedCountTorchRig(self.rig, p, device='cpu', dtype=torch.float64)
        native, ab = torch.tensor([p.candidate_native.tolist()], dtype=torch.float64), m.ab_tensor(self.patches)
        for change in ('native', 'patch', 'other'):
            x, a = native.clone(), ab.clone()
            if change == 'native':
                x[0, 0] += .01
            elif change == 'patch':
                a[0, m.slots[DEFAULT_BONES[0]], 3] += .01
            else:
                other = next(i for i in range(len(m.ab_names)) if i not in m.slots.values())
                a[0, other, 0] += .01
            with self.subTest(change=change), self.assertRaises(ValueError):
                m(x, a)

    def test_dirty_candidate_first_flags_rejected(self):
        fixture = copy.deepcopy(self.fixture)
        fixture[2]['events'][0]['before']['cache']['fields']['_forceApply'] = True
        with self.assertRaises(ValueError):
            self.protocol(fixture)

    def test_complete_trace_later_corruption_rejected(self):
        fixture = copy.deepcopy(self.fixture)
        fixture[2]['events'][-1]['after']['local_position'][0] += .1
        with self.assertRaises(ValueError):
            self.protocol(fixture)

    def test_coherent_new_history_or_identity_still_rejected(self):
        for change in ('length', 'direction', 'identity', 'flags'):
            fixture = list(copy.deepcopy(self.fixture))
            first = {name: copy.deepcopy(next(e for e in fixture[2]['events'] if e['bone_name'] == name)) for name in DEFAULT_BONES}
            name = DEFAULT_BONES[1]
            if change == 'length':
                first[name]['before']['cache']['fields']['_lenBaseline'] += .01
            elif change == 'direction':
                first[name]['before']['cache']['fields']['_positionBaseline'][0] += .01
            elif change == 'identity':
                first[name]['modifier_instance_identity'] += 1
            else:
                first[name]['before']['cache']['fields']['_changedScale'] = True
            fixture[2], fixture[3] = synthetic_capture(fixture[2]['metadata'], first, self.patches, 4,
                                                       self.declared['native59'], fixture[3]['character'])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.protocol(fixture)

    def test_predeclared_sampler_gate_and_common_count_scope(self):
        for key, value in (('sampling_profile', 'vanilla'), ('normalized_whole_head_gate', .1),
                           ('common_apply_count', 7), ('common_count_runtime_certified', True)):
            declared = copy.deepcopy(self.declared)
            declared[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                EarlierHistoryProtocol(*self.fixture, self.contract, declared)


if __name__ == '__main__':
    unittest.main()
