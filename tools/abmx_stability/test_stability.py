"""Analytic fixed-branch dynamics and metadata/field separation."""
import copy
import unittest
import numpy as np
from tools.abmx_stability.classify import classify, private_difference


def event(length=1., offset=(0., 0., 0.)):
    fields = {'_hasBaseline': True, '_lenBaseline': 5., '_posBaseline': [0., 3., 4.],
              '_positionBaseline': [0., 3., 4.], '_sclBaseline': [1., 1., 1.], '_rotBaseline': [0., 0., 0., 1.],
              '_lenModForceUpdate': False, '_lenModNeedsPositionRestore': False, '_forceApply': False,
              '_changedPosition': False, '_changedRotation': False, '_changedScale': False}
    return {'coordinate': 0, 'additional_modifiers': [], 'is_during_h_scene': False, 'no_rotation_excluded': False,
            'resolved_modifier': {'scale': [1.02, .99, 1.01], 'length': length, 'position': list(offset), 'rotation': [.6, -.4, .3]},
            'before': {'local_position': [0., 3., 4.], 'local_rotation_xyzw': [0., 0., 0., 1.],
                       'local_scale': [1., 1., 1.], 'cache': {'fields': fields, 'frame_count': 1}}}


class StabilityTests(unittest.TestCase):
    def test_length_only_position_only_and_scale_rotation_are_fixed_branch_idempotent(self):
        for args, regime in (((2., (0, 0, 0)), 'length_only'), ((1., (1, 0, 0)), 'position_only'), ((1., (0, 0, 0)), 'scale_rotation_only')):
            report = classify(event(*args), long_count=8)
            self.assertEqual(report['regime'], regime)
            self.assertEqual(report['mathematical_classification'], 'idempotent_after_one_apply_under_fixed_selected_branch')
            self.assertTrue(report['simulation']['first_vs_final_trs']['passed'])
            self.assertTrue(report['simulation']['first_vs_final_private_fields']['passed'])
            self.assertFalse(report['actual_runtime_stability_proven'])

    def test_combined_two_step_analytic_current_direction(self):
        report = classify(event(2., (1, 0, 0)), long_count=8)
        rows = report['simulation']['sparse_steps']
        np.testing.assert_allclose(rows[0]['local_position'], [1, 6, 8], atol=1e-6)
        np.testing.assert_allclose(rows[1]['local_position'], [1+10/np.sqrt(101), 60/np.sqrt(101), 80/np.sqrt(101)], atol=1e-6)
        self.assertFalse(report['simulation']['first_vs_final_trs']['passed'])
        self.assertEqual(report['positive_fixed_point'], [11, 0, 0])
        self.assertEqual(report['negative_fixed_point'], [-9, 0, 0])
        self.assertAlmostEqual(report['local_angular_contraction_at_positive_fixed_point'], 10/11)
        self.assertFalse(report['numeric_long_run_is_runtime_evidence'])

    def test_combined_collinear_fixed_point_exception(self):
        x = event(2., (1, 0, 0))
        x['before']['local_position'] = [3, 0, 0]
        report = classify(x, long_count=8)
        self.assertTrue(report['simulation']['first_vs_final_trs']['passed'])
        self.assertEqual(report['mathematical_classification'], 'generally_nonidempotent_direction_iteration')
        self.assertFalse(report['actual_runtime_stability_proven'])

    def test_negative_fixed_point_is_not_global_stability(self):
        x = event(2., (1, 0, 0))
        x['before']['local_position'] = [-9, 0, 0]
        report = classify(x, long_count=8)
        self.assertTrue(report['simulation']['first_vs_final_trs']['passed'])
        self.assertGreater(report['negative_fixed_point_angular_multiplier'], 1)

    def test_short_length_zero_direction_hscene_and_flags_unsupported(self):
        for change in ('short', 'zero_current', 'zero_history', 'scene', 'flags', 'length_baseline', 'additional'):
            x = event(2., (1, 0, 0))
            if change == 'short':
                x['resolved_modifier']['length'] = .05
            elif change == 'zero_current':
                x['before']['local_position'] = [0, 0, 0]
            elif change == 'zero_history':
                x['before']['cache']['fields']['_positionBaseline'] = [0, 0, 0]
            elif change == 'scene':
                x['is_during_h_scene'] = True
            elif change == 'flags':
                x['before']['cache']['fields']['_changedPosition'] = True
            elif change == 'length_baseline':
                x['before']['cache']['fields']['_lenBaseline'] = 0
            else:
                x['additional_modifiers'] = [x['resolved_modifier']]
            with self.subTest(change=change):
                report = classify(x)
                self.assertFalse(report['supported_selected_branch'])
                self.assertEqual(report['mathematical_classification'], 'unsupported_branch_no_generalization')

    def test_simulated_path_falling_to_zero_is_not_current_direction_proof(self):
        x = event(2., (10, 0, 0))
        x['before']['local_position'] = [-3, 0, 0]
        report = classify(x, long_count=8)
        self.assertFalse(report['supported_selected_branch'])
        self.assertEqual(report['simulation']['completed_count'], 1)

    def test_frame_only_change_not_private_field_change(self):
        a = event()['before']['cache']
        b = copy.deepcopy(a)
        b['frame_count'] = 90
        report = private_difference(a, b)
        self.assertTrue(report['private_fields_unchanged'])
        self.assertEqual(report['changed_runtime_metadata_keys'], ['frame_count'])
        self.assertFalse(report['metadata_change_implies_private_change'])

    def test_actual_private_field_change_retained_even_with_frame_change(self):
        a = event()['before']['cache']
        b = copy.deepcopy(a)
        b['frame_count'] = 90
        b['fields']['_posBaseline'][0] = .01
        report = private_difference(a, b)
        self.assertFalse(report['private_fields_unchanged'])
        self.assertEqual(set(report['changed_private_fields']), {'_posBaseline'})

    def test_long_prediction_not_runtime_proof_and_count_not_fitted(self):
        report = classify(event(2., (.001, 0, 0)), long_count=512)
        self.assertFalse(report['actual_runtime_stability_proven'])
        self.assertFalse(report['controlled_common_count_proven'])
        self.assertEqual(report['simulation']['requested_count'], 512)
        with self.assertRaises(ValueError):
            classify(event(), long_count=True)

    def test_actual_after_never_used_to_generate_long_prediction(self):
        a = event(2., (1, 0, 0))
        b = copy.deepcopy(a)
        b['after'] = {'local_position': [100, 100, 100], 'cache': {'fields': 'corrupt target'}}
        self.assertEqual(classify(a, long_count=8), classify(b, long_count=8))


if __name__ == '__main__':
    unittest.main()
