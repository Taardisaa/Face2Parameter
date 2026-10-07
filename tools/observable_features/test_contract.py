"""Synthetic verifier tests; none certify real game geometry or semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from pydantic import ValidationError

from .contract import Manifest, digest, json_digest
from .measure import compare_spans, execute


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.snapshot = {'schema_version': 1, 'snapshot_kind': 'maker_live_skinned_geometry',
            'character': {'head_id': 2, 'native_count': 59, 'shape_value_face': [.5]*59, 'expression': {'mouth_ptn': 0}},
            'frame_count': 1, 'frame_count_end': 1, 'pose_signature': 'b'*64,
            'meshes': [{'renderer_path': '/head', 'source_geometry_sha256': 'c'*64,
                'source': {'triangles': [0, 1, 2]},
                'baked': {'vertices': [[0., 0., 0.], [2., 0., 0.], [0., 3., 0.]], 'triangles': [0, 1, 2],
                    'submeshes': [{'submesh_index': 0, 'topology': 'Triangles', 'indices_apply_base_vertex': True, 'indices': [0, 1, 2]}]}}]}
        self.path = self.root/'geometry.json'
        self.path.write_text(json.dumps(self.snapshot))
        self.data = {'schema_version': 1, 'purpose': 'measurement_diagnostic_only',
            'geometry': {'path': str(self.path), 'sha256': digest(self.path), 'head_id': 2,
                         'renderer_path': '/head', 'source_geometry_sha256': 'c'*64, 'frame_count': 1,
                         'pose_signature': 'b'*64, 'coordinate_space': 'renderer_baked_raw',
                         'native_face_values_sha256': json_digest(self.snapshot['character']['shape_value_face']),
                         'expression_configuration_sha256': json_digest(self.snapshot['character']['expression']),
                         'abmx_parameter_values_status': 'unverified_not_bound_by_this_contract',
                         'units': 'game_units_unscaled', 'frame_policy_id': 'raw-test-frame'},
            'features': [self.formal('x_span', 'directional_span', [1., 0., 0.])]}

    @staticmethod
    def formal(id, operation, axis=None):
        return {'id': id, 'kind': 'formal_surface', 'operation': operation,
            'region': {'kind': 'full_asset_submesh', 'submesh_id': 0}, 'axis': axis,
            'numerical_tolerance_game_units': 1e-10, 'recomputation_rule': 'recompute_on_each_actual_surface'}

    def parse(self, data=None):
        return Manifest.model_validate(self.data if data is None else data)

    def test_span_area_support_are_literal(self):
        self.data['features'] += [self.formal('area', 'surface_area'), self.formal('support', 'directional_support_set', [0., 0., 1.])]
        rows = execute(self.parse())['results']
        self.assertEqual(rows[0]['result']['span_game_units'], 2.)
        self.assertEqual(rows[1]['result']['area_game_units_squared'], 3.)
        self.assertFalse(rows[2]['result']['unique_maximum_under_declared_tolerance'])
        self.assertEqual(rows[2]['result']['maximizing_source_vertex_ids'], [0, 1, 2])
        self.assertFalse(rows[2]['result']['fixed_material_identity_created'])

    def test_geometric_feature_cannot_claim_anatomy_or_visibility(self):
        for claim in ['anatomical_correspondence_validated', 'image_visibility_validated', 'cross_view_fixed_material_point_validated', 'cross_base_semantic_equivalence_validated', 'likeness_validated']:
            data = copy.deepcopy(self.data)
            data['features'][0]['claims'] = {claim: True}
            with self.assertRaises(ValidationError): self.parse(data)
        self.data['features'][0]['certified_anatomical_label'] = 'nose_tip'
        with self.assertRaises(ValidationError): self.parse()

    def test_extra_certification_fields_rejected(self):
        self.data['features'][0]['shader_depth_validated'] = True
        with self.assertRaises(ValidationError): self.parse()

    def test_gate_not_redefinable(self):
        self.data['fixed_image_gate_max_error_px'] = 4.
        with self.assertRaises(ValidationError): self.parse()

    def test_axis_must_be_finite_unit_and_not_bool(self):
        for axis in ([2., 0., 0.], [float('nan'), 0., 0.], [True, 0., 0.]):
            self.data['features'][0]['axis'] = axis
            with self.assertRaises(ValidationError): self.parse()

    def test_plane_keeps_coplanar_ambiguity(self):
        self.data['features'] = [self.formal('plane', 'plane_intersection_segments', [0., 0., 1.]) | {'plane_offset': 0.}]
        result = execute(self.parse())['results'][0]['result']
        self.assertEqual(result['status'], 'ambiguous_coplanar_surface')
        self.assertEqual(result['coplanar_region_triangle_ids'], [0])
        self.assertFalse(result['unique_ordered_curve_claimed'])

    def test_plane_segment_interpolates_actual_triangle(self):
        self.data['features'] = [self.formal('plane', 'plane_intersection_segments', [1., 0., 0.]) | {'plane_offset': 1.}]
        result = execute(self.parse())['results'][0]['result']
        self.assertEqual(result['status'], 'measured_segments')
        self.assertEqual(result['segments'][0]['endpoints'], [[1., 0., 0.], [1., 1.5, 0.]])

    def test_artifact_tamper_rejected(self):
        manifest = self.parse()
        self.path.write_text('{}')
        with self.assertRaises(ValueError): execute(manifest)

    def test_wrong_pose_base_source_and_frame_rejected(self):
        for key, value in [('head_id', 0), ('frame_count', 2), ('pose_signature', 'd'*64), ('source_geometry_sha256', 'e'*64),
                           ('native_face_values_sha256', 'f'*64), ('expression_configuration_sha256', '1'*64)]:
            data = copy.deepcopy(self.data); data['geometry'][key] = value
            with self.assertRaises(ValueError): execute(self.parse(data))

    def test_null_is_preserved_not_zero_filled(self):
        self.data['features'] = [{'id': 'N', 'kind': 'unobservable', 'intended_feature': 'nasal skin dome', 'reason': 'Smooth nonunique RGB region'}]
        row = execute(self.parse())['results'][0]
        self.assertIsNone(row['result']['value'])
        self.assertFalse(row['claims']['anatomical_correspondence_validated'])

    def test_material_order_and_source_bound_point(self):
        feature = {'id': 'sample', 'kind': 'fixed_material_candidate', 'triangle_id': 0,
            'ordered_vertex_ids': [0, 1, 2], 'barycentric': [.5, .25, .25], 'transport_rule': 'fixed_source_triangle_barycentric'}
        self.data['features'] = [feature]
        self.assertEqual(execute(self.parse())['results'][0]['result']['point_renderer_baked_raw'], [.5, .75, 0.])
        feature['ordered_vertex_ids'] = [1, 0, 2]
        with self.assertRaises(ValueError): execute(self.parse())

    def test_cross_base_and_frame_compare_rejected(self):
        first = self.parse()
        for key, value in [('head_id', 3), ('source_geometry_sha256', 'a'*64), ('frame_policy_id', 'other-frame')]:
            data = copy.deepcopy(self.data); data['geometry'][key] = value
            with self.assertRaises(ValueError): compare_spans(first, self.parse(data))
        report = compare_spans(first, first)
        self.assertEqual(report['differences_game_units'], {'x_span': 0.})
        self.assertFalse(report['facial_width_length_depth_anatomically_certified'])

    def test_triangle_stream_fractional_indices_rejected(self):
        self.snapshot['meshes'][0]['baked']['triangles'] = [0., 1., 2.]
        self.path.write_text(json.dumps(self.snapshot));self.data['geometry']['sha256'] = digest(self.path)
        with self.assertRaises(ValueError): execute(self.parse())

    def test_appearance_ambiguity_requires_null_and_no_skin_certification(self):
        feature = {'id': 'lip_texture', 'kind': 'appearance_pixel',
            'image': {'path': 'not-opened.png', 'sha256': 'a'*64, 'width': 10, 'height': 10, 'view_id': 'view',
                      'camera_payload_path': 'not-opened.json', 'camera_payload_sha256': 'b'*64,
                      'coordinate_convention': 'PNG pixel centers top_left'},
            'visibility': 'ambiguous', 'xy': None, 'uncertainty_radius_px': None,
            'appearance_definition': 'Dark lip seam endpoint', 'skin_semantic_decision': 'ambiguous',
            'annotation_method': 'original_png_before_overlay'}
        self.data['features'] = [feature]; self.parse()
        feature['xy'] = [2., 2.]
        with self.assertRaises(ValidationError): self.parse()
        feature['xy'] = None;feature['skin_semantic_decision'] = 'accepted_skin'
        with self.assertRaises(ValidationError): self.parse()

    def test_actual_image_camera_pose_bytes_bound(self):
        from PIL import Image
        image_path = self.root/'original.png'
        Image.new('RGB', (10, 10)).save(image_path)
        camera_path = self.root/'capture.json'
        capture = {'path': str(image_path), 'width': 10, 'height': 10,
                   'paired_geometry': {'sha256': digest(self.path), 'pose_signature': 'b'*64, 'frame_count': 1},
                   'frame_count': 1, 'frame_count_before_render': 1, 'paired_pose_unchanged': True,
                   'pose_signature_before_render': 'b'*64, 'pose_signature_after_render': 'b'*64,
                   'capture_camera': {'world_to_camera': [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.],
                     'projection': [1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.],
                     'matrix_layout': 'row_major_16; column_vectors', 'viewport_origin': 'bottom_left',
                     'image_coordinate_convention': 'PNG pixel centers top_left', 'pixel_rect': [0.,0.,10.,10.],
                     'cpu_ndc_depth_range': [-1.,1.]}}
        camera_path.write_text(json.dumps(capture))
        self.data['features'] = [{'id': 'appearance', 'kind': 'appearance_pixel',
            'image': {'path': str(image_path), 'sha256': digest(image_path), 'width': 10, 'height': 10,
                      'view_id': 'view', 'camera_payload_path': str(camera_path), 'camera_payload_sha256': digest(camera_path),
                      'coordinate_convention': 'PNG pixel centers top_left'},
            'visibility': 'visible', 'xy': [2., 2.], 'uncertainty_radius_px': 3.,
            'appearance_definition': 'Synthetic arbitrary appearance pixel', 'skin_semantic_decision': 'texture',
            'annotation_method': 'original_png_before_overlay'}]
        report = execute(self.parse())
        self.assertFalse(report['camera_pixel_lbs_certified_by_this_tool'])
        self.assertEqual(report['results'][0]['result']['uncertainty_radius_px'], 3.)
        camera_path.write_text('{}')
        with self.assertRaises(ValueError): execute(self.parse())

    def test_jpeg_disguised_as_png_is_rejected(self):
        from PIL import Image
        image_path = self.root/'renamed.png'
        Image.new('RGB', (10, 10)).save(image_path, format='JPEG')
        camera_path = self.root/'capture.json'
        camera_path.write_text('{}')
        self.data['features'] = [{'id': 'appearance', 'kind': 'appearance_pixel',
            'image': {'path': str(image_path), 'sha256': digest(image_path), 'width': 10, 'height': 10,
                      'view_id': 'view', 'camera_payload_path': str(camera_path), 'camera_payload_sha256': digest(camera_path),
                      'coordinate_convention': 'PNG pixel centers top_left'},
            'visibility': 'ambiguous', 'xy': None, 'uncertainty_radius_px': None,
            'appearance_definition': 'Ambiguous diagnostic only', 'skin_semantic_decision': 'unknown',
            'annotation_method': 'original_png_before_overlay'}]
        with self.assertRaisesRegex(ValueError, 'Actual PNG encoding required'):
            execute(self.parse())


if __name__ == '__main__':
    unittest.main()
