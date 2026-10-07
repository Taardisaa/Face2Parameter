"""Semantic guards prevent geometry-only/FAN failures becoming named truth."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tools.anatomical_candidates.run import validate_seed
from tools.anatomical_candidates.validate_review import evaluate_point, review_identity
from tools.surface_calibration.core import Camera, ContractError, SurfaceMesh, follow
from tools.surface_calibration.pixel_certificate import digest


class IndependentReviewGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report_path = Path(self.tmp.name)/'report.json'
        self.report_path.write_text('{}')
        self.report = {'points': [{'id': 'N', 'seed_author': 'seed_author'}]}
        self.review = {'schema_version': 1, 'candidate_report_path': str(self.report_path),
                       'candidate_report_sha256': digest(self.report_path), 'reviewer_identity': 'independent_reviewer',
                       'reviewer_is_seed_author': False, 'review_original_png_before_candidate_overlay': True,
                       'points': [{'id': 'N'}]}

    def test_independent_identity_guard_accepts_specific_bound_review(self):
        self.assertEqual(review_identity(self.report, self.review, self.report_path), 'independent_reviewer')

    def test_seed_author_cannot_validate_self_even_if_checkbox_claims_independence(self):
        self.review['reviewer_identity'] = 'seed_author'
        with self.assertRaisesRegex(ContractError, 'seed author'):
            review_identity(self.report, self.review, self.report_path)

    def test_overlay_first_is_not_independent_annotation(self):
        self.review['review_original_png_before_candidate_overlay'] = False
        with self.assertRaisesRegex(ContractError, 'original-image-first'):
            review_identity(self.report, self.review, self.report_path)

    def test_report_mutation_cannot_reuse_previous_review(self):
        self.report_path.write_text('{"points": "changed"}')
        with self.assertRaisesRegex(ContractError, 'changed'):
            review_identity(self.report, self.review, self.report_path)

    def test_no_point_omission(self):
        self.review['points'] = []
        with self.assertRaisesRegex(ContractError, 'incomplete'):
            review_identity(self.report, self.review, self.report_path)

    def test_seed_from_other_capture_rejected_before_ray(self):
        evidence = {'sources': {'capture': {'sha256': 'actual'}, 'geometry': {'sha256': 'geometry'}}}
        seed = {'schema_version': 1, 'semantic_validated': False, 'capture_sha256': 'different'}
        with self.assertRaisesRegex(ContractError, 'different capture'):
            validate_seed(seed, evidence, [])

    def analytic_review(self):
        projection = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, -10.1/9.9, -.2*10/9.9], [0, 0, -1., 0]])
        mesh = SurfaceMesh('head', np.array([[-1., -1., -3.], [1., -1., -3.], [0., 1., -3.]]),
                           np.array([[0, 1, 2]]), 'source', world_policy_certified=True)
        surface = follow({'renderer_path': 'head', 'source_geometry_sha256': 'source',
                          'triangle_id': 0, 'vertex_ids': [0, 1, 2], 'barycentric': [.25, .25, .5]}, [mesh])
        cameras, views, records = [], [], []
        for i, shift in enumerate((-1., 0., 1.)):
            view = np.eye(4)
            view[0, 3] = -shift
            camera = Camera(32, 32, view, projection, np.array([0., 0., 32., 32.]))
            png = Path(self.tmp.name)/f'{i}.png'
            png.write_bytes(b'original-image-identity')
            cameras.append(camera)
            views.append({'path': str(png), 'yaw': (i-1)*30})
            records.append({'view_index': i, 'raw_png_sha256': digest(png), 'raw_png': str(png),
                            'visibility': 'visible', 'annotation_method': 'original_png_before_overlay',
                            'independent_xy': camera.project([0., 0., -3.])['xy']})
        candidate = {'id': 'A', 'definition': 'Explicit alar hypothesis', 'hypothesis': 'alar_margin_front_image_left',
                     'status': 'awaiting_independent_semantic_review', 'material': surface}
        annotation = {'definition': candidate['definition'], 'semantic_decision': 'pending',
                      'skin_vs_texture_anchor': 'skin', 'views': records}
        return candidate, annotation, views, cameras, [mesh]

    def test_exact_three_view_geometry_does_not_promote_pending_semantics(self):
        result = evaluate_point(*self.analytic_review())
        self.assertTrue(result['fixed_material_geometry_test_passed'])
        self.assertFalse(result['accepted_for_named_material_candidate'])

    def test_failed_geometric_seed_cannot_be_promoted_by_review_checkbox(self):
        inputs = self.analytic_review()
        inputs[0]['status'] = 'seed_geometric_hit_rejected_or_ambiguous'
        inputs[1]['semantic_decision'] = 'accepted'
        result = evaluate_point(*inputs)
        self.assertTrue(result['fixed_material_geometry_test_passed'])
        self.assertFalse(result['accepted_for_named_material_candidate'])


if __name__ == '__main__':
    unittest.main()
