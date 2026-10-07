"""Held-out evidence must not reuse fitted gains, altered requests or input states."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from review_guidance import (
    THRESHOLDS,
    Uncertifiable,
    analyze,
    load_payload,
    prediction_metrics,
)
from test_live_response import fixture


class GuidanceEvidenceTests(unittest.TestCase):
    def setUp(self):
        from explore_live_response import calibration_identity, canonical_hash
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.training = {'complete': True, 'coordinate_transform_name': 'cf_J_Head',
            'baselines': {'card': {'native59': [.5]*59,
                                  'geometry': self.save('old', fixture())}}, 'cases': []}
        identity = calibration_identity(fixture(), self.training)
        self.catalog = {'complete': True, 'baselines': {'card': {
            'identity': identity, 'identity_sha256': canonical_hash(identity),
            'head_diagonal': float(np.sqrt(3)), 'repeat_drift_normalized': 0.,
            'baseline_receipt': self.training['baselines']['card']['geometry']}},
            'entries': []}
        self.plan = {'schema_version': 1, 'baselines': copy.deepcopy(self.catalog['baselines']),
                     'thresholds': THRESHOLDS, 'entries': [], 'skipped': []}
        self.manifest = {'complete': True, 'public_configuration_restored': True,
            'before': {'snapshot': {}, 'expression': {}, 'abmx': {}},
            'after': {'snapshot': {}, 'expression': {}, 'abmx': {}, 'physics': {'active': False}},
            'coordinate_transform_name': 'cf_J_Head',
            'baselines': {'card': {'native59': [.5]*59, 'geometry': self.save('fresh', fixture())}},
            'cases': [], 'repeats': [{'baseline_name': 'card', 'native59': [.5]*59,
                                    'geometry': self.save('fresh_repeat', fixture())}]}
        self.training['cases'].append({'name': 'repeat', 'kind': 'baseline',
            'baseline_name': 'card', 'native59': [.5]*59, 'geometry': self.save('old_repeat', fixture())})
        for control in range(59):
            entry = {'id': f'c{control}', 'kind': 'native', 'control': control,
                     'baseline_name': 'card', 'file': f'c{control}.js'}
            payload = {**entry, 'mesh': 'o_head', 'identity': identity,
                       'levels': [.45, .55], 'roles': ['local_minus', 'local_plus'],
                       'verification': {'case_receipts': []}}
            for sign in (-1, 1):
                native = [.5]*59
                native[control] += sign*.05
                case = {'name': f'train{control}_{sign}', 'kind': 'native',
                    'baseline_name': 'card', 'control': control, 'native59': native,
                    'probe_role': 'local_minus' if sign == -1 else 'local_plus',
                    'value': native[control], 'geometry': self.save(f'train{control}_{sign}', fixture(native))}
                self.training['cases'].append(case)
                payload['verification']['case_receipts'].append(case['geometry'])
                measured_gain = abs(sum(native)-29.5)/abs(native[control]-.5)
                target = float(np.sqrt(3)*.0001)
                step = target/measured_gain
                heldout = [.5]*59
                heldout[control] += sign*step
                request = {'name': f'heldout{control}_{sign}', 'entry_id': entry['id'],
                    'baseline_name': 'card', 'kind': 'native', 'sign': sign,
                    'control': control, 'value': heldout[control], 'step': step,
                    'signed_step': sign*step, 'target_units': target,
                    'target_percent': .01, 'requested_target_percent': .01,
                    'max_probe_fraction': .4, 'training_step': abs(native[control]-.5),
                    'native59': heldout}
                self.plan['entries'].append(request)
                self.manifest['cases'].append({**request, 'geometry': self.save(request['name'], fixture(heldout))})
            path = self.root / entry['file']
            path.write_text('window.HS2_RESPONSE('+json.dumps(payload)+');\n')
            entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.catalog['entries'].append(entry)
        self.bind()

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, value):
        raw = json.dumps(value).encode()
        path = self.root / (name+'.json.gz')
        path.write_bytes(gzip.compress(raw))
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}

    def bind(self):
        self.plan['training_manifest_receipt'] = self.save('training', self.training)
        self.catalog['provenance'] = {'source_manifest_sha256': self.plan['training_manifest_receipt']['sha256']}
        self.plan['catalog_receipt'] = self.save('catalog', self.catalog)
        self.manifest['plan_receipt'] = self.save('plan', self.plan)
        self.path = self.root/'manifest.json'
        self.path.write_text(json.dumps(self.manifest))

    def test_linear_analytic_fixture_certifies_all59_both_signs(self):
        report = analyze(self.path)
        self.assertTrue(report['all59_native_guidance_verified'])
        self.assertEqual(118, report['coverage']['card']['verified_native_signs'])
        self.assertTrue(report['all_planned_predictions_verified'])
        self.assertEqual(['o_head'], report['analysis_mesh_scope'])

    def test_changed_request_cannot_relabel_a_new_step(self):
        self.manifest['cases'][0]['signed_step'] *= 2
        self.bind()
        with self.assertRaisesRegex(Uncertifiable, 'predeclared plan'):
            analyze(self.path)

    def test_missing_case_or_duplicate_sign_is_rejected(self):
        self.manifest['cases'].pop()
        self.bind()
        with self.assertRaisesRegex(Uncertifiable, 'bijection'):
            analyze(self.path)

    def test_changed_input_or_vector_prediction_keeps_failed_witness(self):
        original = self.manifest['cases'][0]
        native = original['native59']
        self.manifest['cases'][0]['geometry'] = self.save('wrong_prediction', fixture(native, drift=.005))
        self.bind()
        report = analyze(self.path)
        first = report['entries'][0]
        self.assertTrue(first['input_trusted'])
        self.assertFalse(first['verified_prediction'])
        self.assertGreater(first['max_vector_error'], first['error_budget'])
        self.assertFalse(report['all59_native_guidance_verified'])
        bad = copy.deepcopy(native)
        bad[3] = .6
        self.manifest['cases'][0]['geometry'] = self.save('wrong_input', fixture(bad))
        self.bind()
        row = analyze(self.path)['entries'][0]
        self.assertFalse(row['input_trusted'])
        self.assertFalse(row['verified_prediction'])

    def test_restoration_receipt_and_arbitrary_skip_cannot_be_waived(self):
        self.manifest['after']['expression'] = {'mouth': 1}
        self.bind()
        with self.assertRaisesRegex(Uncertifiable, 'Before/after'):
            analyze(self.path)
        self.manifest['after']['expression'] = {}
        entry = self.plan['entries'].pop(0)
        self.plan['skipped'].append({'entry_id': entry['entry_id'], 'sign': entry['sign'], 'reason': 'unwanted'})
        self.manifest['cases'].pop(0)
        self.bind()
        with self.assertRaisesRegex(Uncertifiable, 'noise floor'):
            analyze(self.path)

    def test_catalog_js_tampering_and_rounded_step_are_rejected(self):
        entry = self.catalog['entries'][0]
        path = self.root/entry['file']
        original = path.read_text()
        path.write_text(original.replace('local_minus', 'native_min'))
        with self.assertRaisesRegex(Uncertifiable, 'SHA-256'):
            load_payload(entry, self.root)
        path.write_text(original)
        self.plan['entries'][0]['step'] = .001
        self.manifest['cases'][0]['step'] = .001
        self.bind()
        with self.assertRaisesRegex(Uncertifiable, 'Recommendation step'):
            analyze(self.path)

    def test_fresh_baseline_cannot_be_fitted_back_to_training(self):
        self.manifest['baselines']['card']['geometry'] = self.save('moved_baseline', fixture(drift=.001))
        self.bind()
        report = analyze(self.path)
        self.assertFalse(report['baselines']['card']['baseline_reuse_pass'])
        self.assertGreater(report['baselines']['card']['baseline_reuse_normalized'], 1e-5)
        self.assertFalse(report['all_planned_predictions_verified'])

    def test_abmx_channel_isolation_checks_unrequested_axis(self):
        from review_guidance import isolation
        patch = {'name': 'face_bone', 'scale': [1.02, 1, 1], 'length': 1,
                 'position': [0, 0, 0], 'rotation': [0, 0, 0]}
        case = {'kind': 'abmx', 'bone': 'face_bone', 'channel': 'scale',
                'axis': 0, 'value': 1.02, 'native59': [.5]*59, 'patch': patch}
        isolation(case, fixture())
        patch['scale'][1] = 1.1
        with self.assertRaisesRegex(Uncertifiable, 'another channel'):
            isolation(case, fixture())

    def test_opposite_direction_does_not_pass_matching_target_magnitude(self):
        delta = np.array([[1., 0, 0], [0, 0, 0]])
        report = prediction_metrics(delta, .1, .01, -delta*.1,
            target=.1, diagonal=1, training_noise=0, fresh_noise=0)
        self.assertEqual(0., report['relative_target_error'])
        self.assertFalse(report['verified_prediction'])
        self.assertAlmostEqual(.2, report['max_vector_error'])


if __name__ == '__main__':
    unittest.main()
