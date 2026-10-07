"""Analytical fixtures check evidence gates and measured local response."""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.parameter_atlas.live_response import ROLES, Uncertifiable, analyze_manifest


def fixture(native=None, *, drift=0, shapes=True):
    native = [.5] * 59 if native is None else native
    displacement = sum(native) - 59 * .5 + drift
    points = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[0, 3] = displacement
    transforms = [{'id': 1, 'name': 'cf_J_Head', 'path': '/root/head', 'parent_id': None,
                   'local_position': [0, 0, 0], 'local_rotation_xyzw': [0, 0, 0, 1],
                   'local_scale': [1, 1, 1], 'local_to_world': np.eye(4).reshape(-1).tolist()},
                  {'id': 2, 'name': 'face_bone', 'path': '/root/head/bone', 'parent_id': 1,
                   'local_to_world': transform.reshape(-1).tolist()}]
    meshes = []
    for index, name in enumerate(['o_head', 'o_eyebase_L', 'o_eyebase_R', 'o_eyelashes',
                                   'o_eyeshadow', 'o_namida', 'o_tooth', 'o_tang', 'o_tang', 'o_tang']):
        world = points + [displacement, 0, 0]
        source = {'vertices': points.tolist(), 'triangles': [0, 1, 2, 0, 2, 3],
                  'bone_indices': [[0, 0, 0, 0]] * 4, 'bone_weights': [[1, 0, 0, 0]] * 4,
                  'bindposes': [np.eye(4).reshape(-1).tolist()]}
        meshes.append({'renderer_id': 10 + index, 'renderer_path': f'/root/head/renderer{index}',
                       'mesh_instance_id': 20 + index, 'mesh_name': name, 'source_geometry_sha256': 'sourcehash',
                       'bone_names': ['face_bone'], 'bone_transform_ids': [2], 'root_bone_transform_id': 2,
                       'vertex_count': 4, 'triangle_count': 2, 'source': source, 'skin_quality': 'Bone4',
                       'blendshapes': [{'name': 'expression', 'shape_index': 0, 'current_weight': 0.,
                                        'frames': [{'weight': 100., 'delta_vertices': [[0, 0, 0]] * 4}]}] if shapes else [],
                       'baked': {'vertices': world.tolist(), 'triangles': source['triangles'],
                                 'world_candidates': {'renderer_matrix': {'matrix': np.eye(4).reshape(-1).tolist(), 'vertices': world.tolist()}}}})
    return {'schema_version': 1, 'frame_count': 1, 'frame_count_end': 1,
            'character': {'head_id': 2, 'shape_value_face': native, 'expression': {'mouth_ptn': 0}},
            'transforms': transforms, 'meshes': meshes, 'abmx_runtime': {'bones': []}}


class LiveResponseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        baseline = fixture()
        self.manifest = {'baselines': {'card': {'geometry': self.save('source', baseline), 'native59': [.5] * 59}},
                         'cases': [], 'complete': True, 'required_probe_roles': list(ROLES)}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, snapshot):
        raw = json.dumps(snapshot).encode()
        path = self.directory / (name + '.json.gz')
        path.write_bytes(gzip.compress(raw))
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}

    def add(self, control=0, role='local_plus', value=.55, *, snapshot=None, kind='native'):
        native = [.5] * 59
        if kind == 'native':
            native[control] = value
        snapshot = fixture(native) if snapshot is None else snapshot
        for mesh in snapshot['meshes']:
            for shape in mesh['blendshapes']:
                for frame in shape['frames']:
                    frame.pop('delta_vertices', None)
        name = f'case{len(self.manifest["cases"])}'
        row = {'name': name, 'kind': kind, 'baseline_name': 'card', 'control': control,
               'probe_role': role, 'value': value, 'native59': native, 'geometry': self.save(name, snapshot)}
        self.manifest['cases'].append(row)
        return row

    def analyze(self, *, head_only=False):
        return analyze_manifest(self.manifest, self.directory, offline=False, head_only=head_only)

    def test_all_59_controls_and_duplicate_mesh_names_are_distinct(self):
        for control in range(59):
            for role, value in zip(ROLES, [.45, .55, 0, 1, -.25, 1.25]):
                self.add(control, role, value)
        self.add(kind='baseline')
        report = self.analyze()
        self.assertTrue(report['all_captured_native_configurations_verified'])
        self.assertEqual(10, len(report['cases'][0]['meshes']))
        self.assertEqual(59, len(report['local_slopes']['card']))
        for metrics in report['local_slopes']['card'][0]['meshes'].values():
            self.assertAlmostEqual(1., metrics['max_surface_units_per_parameter_unit'])
            self.assertAlmostEqual(.01, metrics['predicted_max_for_plus_0_01'])

    def test_missing_one_control_cannot_pass(self):
        self.add()
        report = self.analyze()
        self.assertFalse(report['all_captured_native_configurations_verified'])
        self.assertEqual(1, report['baselines']['card']['coverage']['unique_native_controls'])

    def test_endpoint_label_cannot_stand_in_for_endpoint_value(self):
        self.add(role='native_min', value=.55)
        self.assertIn('sampling protocol', self.analyze()['cases'][0]['untrusted_reason'])

    def test_changed_receipt_is_not_trusted(self):
        row = self.add()
        row['geometry']['sha256'] = '0' * 64
        self.assertIn('SHA-256', self.analyze()['cases'][0]['untrusted_reason'])

    def test_changed_source_and_palette_are_rejected(self):
        for field in ('vertices', 'triangles', 'bindposes', 'bone_weights'):
            with self.subTest(field=field):
                snap = fixture([.55] + [.5] * 58)
                snap['meshes'][0]['source'][field][0] = None
                self.manifest['cases'] = []
                self.add(snapshot=snap)
                self.assertIn('changed source', self.analyze()['cases'][0]['untrusted_reason'])
        self.manifest['cases'] = []
        snap = fixture([.55] + [.5] * 58)
        snap['meshes'][0]['bone_transform_ids'] = [3]
        self.add(snapshot=snap)
        self.assertIn('bone_transform_ids', self.analyze()['cases'][0]['untrusted_reason'])

    def test_cloned_mesh_is_rejected_even_when_arrays_equal(self):
        snap = fixture([.55] + [.5] * 58)
        snap['meshes'][0]['mesh_instance_id'] += 100
        self.add(snapshot=snap)
        self.assertIn('mesh_instance_id', self.analyze()['cases'][0]['untrusted_reason'])

    def test_missing_source_expression_frame_is_rejected_even_at_zero_weight(self):
        snap = fixture()
        del snap['meshes'][0]['blendshapes'][0]['frames'][0]['delta_vertices']
        self.manifest['baselines']['card']['geometry'] = self.save('missing', snap)
        with self.assertRaises(KeyError):
            self.analyze()

    def test_expression_change_cannot_be_labeled_isolated_response(self):
        snap = fixture([.55] + [.5] * 58)
        snap['meshes'][0]['blendshapes'][0]['current_weight'] = 10
        self.add(snapshot=snap)
        case = self.analyze()['cases'][0]
        self.assertFalse(case['trusted_isolated_response'])
        self.assertIn('Expression', case['untrusted_reason'])

    def test_native_readback_and_other_controls_are_checked(self):
        row = self.add()
        row['native59'][3] = .4
        self.assertIn('readback', self.analyze()['cases'][0]['untrusted_reason'])
        self.manifest['cases'] = []
        snap = fixture([.55, .6] + [.5] * 57)
        row = self.add(snapshot=snap)
        row['native59'][1] = .6
        self.assertIn('Other native', self.analyze()['cases'][0]['untrusted_reason'])

    def test_repeat_drift_cannot_certify_complete_response(self):
        self.add(kind='baseline', snapshot=fixture(drift=.01))
        coverage = self.analyze()['baselines']['card']['coverage']
        self.assertFalse(coverage['repeat_drift_pass'])
        self.assertGreater(coverage['max_repeat_drift_normalized'], 1e-5)

    def test_anchor_scale_not_erased_by_coordinate_conversion(self):
        snap = fixture([.55] + [.5] * 58)
        snap['transforms'][0]['local_scale'] = [1.1, 1, 1]
        self.add(snapshot=snap)
        self.assertIn('local_scale', self.analyze()['cases'][0]['untrusted_reason'])

    def test_plan_receipt_and_incomplete_pilot_override_coverage(self):
        self.manifest['predeclared_plan'] = str(self.directory / 'plan.json')
        (self.directory / 'plan.json').write_text('{}')
        self.manifest['predeclared_plan_sha256'] = '0' * 64
        with self.assertRaises(Uncertifiable):
            self.analyze()

    def test_abmx_readback_and_rotation_exclusion_are_explicit(self):
        self.add(kind='baseline')
        patch = {'name': 'face_bone', 'scale': [1, 1, 1], 'length': 1,
                 'position': [0, 0, 0], 'rotation': [1, 0, 0]}
        snap = fixture()
        snap['abmx_runtime'] = {'bones': [patch], 'rotation_excluded_bones': ['face_bone']}
        row = self.add(kind='abmx', snapshot=snap)
        row.update({'patch': patch, 'bone': 'face_bone', 'channel': 'rotation', 'axis': 0})
        case = self.analyze()['cases'][-1]
        self.assertTrue(case['trusted_isolated_response'])
        self.assertTrue(case['interpretable_isolated_response_all_renderers'])
        self.assertTrue(case['requested_rotation_excluded'])
        self.assertIn('Not predicted', case['abmx_offline_replay_status'])
        row['patch'] = dict(patch, rotation=[2, 0, 0])
        self.assertIn('ABMX modifiers', self.analyze()['cases'][-1]['untrusted_reason'])

    def test_abmx_without_repeat_cannot_be_interpretable(self):
        patch = {'name': 'face_bone', 'scale': [1, 1, 1], 'length': 1,
                 'position': [.01, 0, 0], 'rotation': [0, 0, 0]}
        snap = fixture(drift=.01)
        snap['abmx_runtime'] = {'bones': [patch]}
        row = self.add(kind='abmx', snapshot=snap)
        row.update({'patch': patch, 'bone': 'face_bone', 'channel': 'position', 'axis': 0})
        report = self.analyze()
        self.assertEqual(1, report['abmx_scope']['trusted_sample_count'])
        self.assertEqual(0, report['abmx_scope']['interpretable_all_renderer_sample_count'])

    def test_explicit_head_component_does_not_waive_full_mesh_failure(self):
        self.add(kind='baseline')
        snap = fixture([.55] + [.5] * 58)
        snap['meshes'][1]['baked']['vertices'][0][0] += .1
        self.add(snapshot=snap)
        full = self.analyze()
        self.assertFalse(full['cases'][-1]['trusted_isolated_response'])
        component = self.analyze(head_only=True)
        self.assertEqual(['o_head'], component['analysis_mesh_scope'])
        self.assertFalse(component['full_captured_mesh_set_certified'])
        self.assertTrue(component['cases'][-1]['trusted_isolated_response'])
        self.assertTrue(component['cases'][-1]['interpretable_isolated_response_analysis_scope'])
        self.assertFalse(component['cases'][-1]['interpretable_isolated_response_all_renderers'])
        self.assertEqual(1, len(component['cases'][-1]['meshes']))

    def test_head_component_keeps_selected_lbs_gate(self):
        snap = fixture([.55] + [.5] * 58)
        snap['meshes'][0]['baked']['vertices'][0][0] += .1
        self.add(snapshot=snap)
        self.assertFalse(self.analyze(head_only=True)['cases'][0]['trusted_isolated_response'])

    def test_head_component_still_requires_all_raw_source_identity(self):
        snap = fixture([.55] + [.5] * 58)
        snap['meshes'][1]['source']['vertices'][0][0] = .1
        self.add(snapshot=snap)
        self.assertIn('changed source', self.analyze(head_only=True)['cases'][0]['untrusted_reason'])

    def test_missing_or_duplicate_o_head_cannot_be_component_certified(self):
        for count in (0, 2):
            with self.subTest(count=count):
                snap = fixture()
                snap['meshes'][0]['mesh_name'] = 'other' if count == 0 else 'o_head'
                if count == 2:
                    snap['meshes'][1]['mesh_name'] = 'o_head'
                self.manifest['baselines']['card']['geometry'] = self.save('bad_head', snap)
                with self.assertRaises(Uncertifiable):
                    self.analyze(head_only=True)


if __name__ == '__main__':
    unittest.main()
