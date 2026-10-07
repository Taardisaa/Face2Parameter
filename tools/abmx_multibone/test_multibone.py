"""Hand-specified two-bone actual-call analogs and analytic parent/child LBS."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from tools.abmx_multibone.geometry import parent_first_world, select_cursor_predictions, skin_vertices
from tools.abmx_replay.model import FLAG_FIELDS, IDENTITY
from tools.abmx_multibone.run import verify_capture_pair_signatures
from src.hs2_mesh_deform import _trs

MVID = '5442c72a-f463-4bf9-831a-247be87146c8'


def fixture():
    names = ['P', 'C']
    contract = {'schema_version': 1, 'plugin_mvid': MVID, 'plugin_version': '4.4.6.0', 'apply_method_il_sha256': 'a'*64,
                'allowed_pre_existing_patch_owners': [], 'no_rotation_bones': [], 'is_coordinate_specific': False}
    metadata = {'schema_version': 1, 'session_id': 'analytic-group', 'plugin_mvid': MVID, 'plugin_version': '4.4.6.0',
                'apply_method_il_sha256': 'a'*64, 'pre_existing_patch_owners': [], 'started_frame': 1, 'stopped_frame': 3,
                'requested_names': names, 'character_transform_id': 44,
                'observation_policy': 'void Prefix last / void Postfix first; no argument, baseline, transform or return writes'}
    initial = {}
    for name, position in [('P', [1., 0., 0.]), ('C', [0., 1., 0.])]:
        state = {key: False for key in FLAG_FIELDS}
        state.update(_hasBaseline=True, _sclBaseline=[1., 1., 1.], _rotBaseline=[0., 0., 0., 1.],
                     _lenBaseline=1., _posBaseline=position, _positionBaseline=position)
        initial[name] = state

    def raw(name, position, cache, frame):
        bone_id = 101 if name == 'P' else 102
        return {'bone_transform_id': bone_id, 'local_position': position, 'local_rotation_xyzw': [0., 0., 0., 1.], 'local_scale': [1., 1., 1.],
                'cache': {'fields': copy.deepcopy(cache), 'missing_fields': [], 'assembly_mvid': MVID,
                          'frame_count': frame, 'bone_transform_id': bone_id, 'modifier_type': 'KKABMX.Core.BoneModifier'}}
    events = []
    for frame in (1, 2, 3):
        for name in names:
            final = copy.deepcopy(initial[name])
            final.update(_forceApply=True, _changedPosition=True)
            before = initial[name] if frame == 1 else final
            previous_position = initial[name]['_posBaseline'] if frame == 1 else ([1.2, 0., 0.] if name == 'P' else [0., 1.3, 0.])
            mod = copy.deepcopy(IDENTITY)
            mod['position'] = [9., 0., 0.] if frame == 3 else ([.2, 0., 0.] if name == 'P' else [0., .3, 0.])
            after_position = ([10., 0., 0.] if name == 'P' else [9., 1., 0.]) if frame == 3 else ([1.2, 0., 0.] if name == 'P' else [0., 1.3, 0.])
            events.append({'sequence': len(events)+1, 'frame': frame, 'completed_frame': frame, 'bone_name': name,
                           'modifier_instance_identity': 111 if name == 'P' else 112, 'coordinate': 0, 'coordinate_specific': False,
                           'resolved_modifier': mod, 'additional_modifiers': [], 'no_rotation_excluded': False, 'is_during_h_scene': False,
                           'before': raw(name, previous_position, before, frame), 'after': raw(name, after_position, final, frame)})
    trace = {'metadata': metadata, 'trace_complete': True, 'active': False, 'observed_calls': len(events),
             'dropped_events': 0, 'pending_calls': 0, 'observer_errors': [], 'events': events}
    snapshot = {'frame_count': 2, 'frame_count_end': 2, 'character': {'transform_id': 44},
                'abmx_trace_cursor': {'session_id': 'analytic-group', 'frame': 2, 'last_completed_sequence': 4,
                                      'observed_calls': 4, 'completed_calls': 4, 'pending_calls': 0, 'dropped_events': 0, 'observer_error_count': 0},
                'transforms': [], 'abmx_runtime': {'bones': []}}
    for event in events[2:4]:
        after = event['after']
        snapshot['transforms'].append({'id': after['bone_transform_id'], 'name': event['bone_name'], 'path': '/group/'+event['bone_name'],
                                       **{key: copy.deepcopy(after[key]) for key in ('local_position', 'local_rotation_xyzw', 'local_scale')}})
        snapshot['abmx_runtime']['bones'].append({'name': event['bone_name'], 'runtime_baseline': copy.deepcopy(after['cache'])})
    return snapshot, trace, contract


class CursorTests(unittest.TestCase):
    def select(self, snapshot, trace, contract):
        return select_cursor_predictions(snapshot, trace, contract, ('P', 'C'))

    def test_each_bone_uses_its_own_last_completed_call_not_global_final(self):
        predicted, binding, _ = self.select(*fixture())
        self.assertAlmostEqual(predicted['P']['local_position'][0], 1.2, places=6)
        self.assertAlmostEqual(predicted['C']['local_position'][1], 1.3, places=6)
        self.assertEqual([b['selected_observed_sequence'] for b in binding['selected_bones']], [3, 4])
        self.assertEqual(binding['later_calls_excluded'], 2)

    def test_capture_before_next_lateupdate_allows_one_frame_age(self):
        snapshot, trace, contract = fixture()
        snapshot['frame_count'] = snapshot['frame_count_end'] = snapshot['abmx_trace_cursor']['frame'] = 3
        for bone in snapshot['abmx_runtime']['bones']:
            bone['runtime_baseline']['frame_count'] = 3
        _, binding, _ = self.select(snapshot, trace, contract)
        self.assertTrue(all(row['frames_since_selected_call'] == 1 for row in binding['selected_bones']))

    def test_future_global_cursor_refused(self):
        snapshot, trace, contract = fixture()
        for key in ('last_completed_sequence', 'observed_calls', 'completed_calls'):
            snapshot['abmx_trace_cursor'][key] = 6
        with self.assertRaisesRegex(ValueError, 'future'):
            self.select(snapshot, trace, contract)

    def test_sparse_one_call_per_bone_refused(self):
        snapshot, trace, contract = fixture()
        for key in ('last_completed_sequence', 'observed_calls', 'completed_calls'):
            snapshot['abmx_trace_cursor'][key] = 2
        with self.assertRaisesRegex(ValueError, 'Sparse'):
            self.select(snapshot, trace, contract)

    def test_stale_last_bone_call_refused(self):
        snapshot, trace, contract = fixture()
        snapshot['frame_count'] = snapshot['frame_count_end'] = snapshot['abmx_trace_cursor']['frame'] = 4
        with self.assertRaisesRegex(ValueError, 'Stale'):
            self.select(snapshot, trace, contract)

    def test_dropped_or_missing_required_filter_not_certified(self):
        for change in ('dropped', 'missing_filter'):
            snapshot, trace, contract = fixture()
            if change == 'dropped':
                trace['dropped_events'] = 1
            else:
                trace['metadata']['requested_names'] = ['P']
            with self.assertRaises(ValueError):
                self.select(snapshot, trace, contract)

    def test_actual_after_corruption_is_not_used_as_prediction_input(self):
        snapshot, trace, contract = fixture()
        trace['events'][2]['after']['local_position'][0] = 99
        with self.assertRaisesRegex(ValueError, 'independent actual-call'):
            self.select(snapshot, trace, contract)

    def test_wrong_snapshot_local_private_flags_or_id_refused(self):
        for change in ('local', 'flag', 'id'):
            snapshot, trace, contract = fixture()
            if change == 'local':
                snapshot['transforms'][0]['local_position'][0] = 99
            elif change == 'flag':
                snapshot['abmx_runtime']['bones'][0]['runtime_baseline']['fields']['_forceApply'] = False
            else:
                snapshot['transforms'][0]['id'] = 999
            with self.assertRaises(ValueError):
                self.select(snapshot, trace, contract)

    def test_external_boundary_remains_explicit_unknown_writer(self):
        snapshot, trace, contract = fixture()
        event = trace['events'][2]
        event['before']['local_position'] = [1.2, 2., 0.]
        for phase in ('before', 'after'):
            event[phase]['cache']['fields']['_posBaseline'] = [1., 2., 0.]
        event['after']['local_position'] = [1.2, 2., 0.]
        # Update later inputs/outputs analytically so the whole stopped trace is valid.
        trace['events'][4]['before'] = copy.deepcopy(event['after'])
        trace['events'][4]['before']['cache']['frame_count'] = 3
        trace['events'][4]['after']['local_position'] = [10., 2., 0.]
        trace['events'][4]['after']['cache']['fields']['_posBaseline'] = [1., 2., 0.]
        snapshot['transforms'][0]['local_position'] = [1.2, 2., 0.]
        snapshot['abmx_runtime']['bones'][0]['runtime_baseline']['fields']['_posBaseline'] = [1., 2., 0.]
        _, binding, _ = self.select(snapshot, trace, contract)
        self.assertEqual(len(binding['external_boundaries_at_or_before_cursor']), 1)
        self.assertFalse(binding['external_writers_identified'])


class ParentChildTests(unittest.TestCase):
    def rig(self):
        return SimpleNamespace(_topo=['p', 'c'], bones={'p': {'name': 'P', 'parent': None}, 'c': {'name': 'C', 'parent': 'p'}},
                               verts=np.array([[1., 0., 0.]]), skin_bone_names=['C'], name2pid={'P': 'p', 'C': 'c'},
                               bindpose=np.array([np.eye(4)]), bone_idx=np.array([[0]]), bone_w=np.array([[1.]]))

    def test_parent_rotation_and_scale_propagate_to_child_then_skin_vertex(self):
        rig = self.rig()
        local = {'p': np.eye(4), 'c': _trs([0., 1., 0.], [0., 0., 0., 1.], [1., 1., 1.])}
        # 90deg Z, x scale2; child local x1 overridden. Child origin => (1,2,0),
        # skinned source x1 adds another rotated/scaled x => (1,4,0).
        override = {'P': {'local_position': [1., 0., 0.], 'local_rotation_xyzw': [0., 0., 2**-.5, 2**-.5], 'local_scale': [2., 1., 1.]},
                    'C': {'local_position': [1., 0., 0.], 'local_rotation_xyzw': [0., 0., 0., 1.], 'local_scale': [1., 1., 1.]}}
        world = parent_first_world(rig, local, override)
        np.testing.assert_allclose(world['c'][:3, 3], [1., 2., 0.], atol=1e-12)
        np.testing.assert_allclose(skin_vertices(rig, world, np.zeros((1, 3))), [[1., 4., 0.]], atol=1e-12)

    def test_child_before_parent_and_unknown_override_refused(self):
        rig = self.rig()
        local = {'p': np.eye(4), 'c': np.eye(4)}
        rig._topo.reverse()
        with self.assertRaisesRegex(ValueError, 'parent-first'):
            parent_first_world(rig, local, {})
        rig._topo.reverse()
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            parent_first_world(rig, local, {'missing': {}})

    def test_reflected_parent_scale_is_preserved_not_fitted_away(self):
        rig = self.rig()
        local = {'p': np.eye(4), 'c': np.eye(4)}
        override = {'P': {'local_position': [1., 0., 0.], 'local_rotation_xyzw': [0., 0., 2**-.5, 2**-.5], 'local_scale': [-2., 1., 1.]},
                    'C': {'local_position': [1., 0., 0.], 'local_rotation_xyzw': [0., 0., 0., 1.], 'local_scale': [1., 1., 1.]}}
        world = parent_first_world(rig, local, override)
        self.assertLess(np.linalg.det(world['c'][:3, :3]), 0)
        np.testing.assert_allclose(skin_vertices(rig, world, np.zeros((1, 3))), [[1., -4., 0.]], atol=1e-12)


class PairMetadataTests(unittest.TestCase):
    def test_green_producer_flag_cannot_hide_different_visibility_signature(self):
        snapshot = {'pose_signature': 'pose', 'visibility_sample_signature': 'actual', 'frame_count': 12}
        view = {'paired_geometry': {'pose_signature': 'pose', 'visibility_sample_signature': 'actual', 'frame_count': 12},
                'frame_count': 12, 'frame_count_before_render': 12, 'paired_visibility_sampled_unchanged': True,
                'visibility_sample_signature_before_render': 'other', 'visibility_sample_signature_after_render': 'other'}
        window = {'capture': {'views': [copy.deepcopy(view) for _ in range(3)]}}
        with self.assertRaisesRegex(ValueError, 'despite producer flags'):
            verify_capture_pair_signatures(window, snapshot)


if __name__ == '__main__':
    unittest.main()
