"""New identity may vary; original definition, topology and placement may not.

Small synthetic data checks rejection contracts, not model or game accuracy.
The existing original-cut packager and source decoder are separate oracles.
"""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .artifact import sha
from .attachment_rebase import rebase, SHARED_STATE


class RebaseContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.old_path = Path(self.directory.name)/'old.json'
        self.new_path = Path(self.directory.name)/'new.json'
        self.old_path.write_text('old'); self.new_path.write_text('new')
        self.old = {'vertices': [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], 'triangles': [0, 1, 2]}
        self.new = copy.deepcopy(self.old); self.new['vertices'][1][0] = 1.25
        old_state = {key: np.zeros((3,), np.float32) for key in SHARED_STATE}
        self.old_model = SimpleNamespace(state=old_state)
        self.model = SimpleNamespace(state=copy.deepcopy(old_state), image_index=1,
            image={'input_sha256': 'photo', 'sha256': 'raw'}, path=self.new_path)
        self.reference = {'artifact_sha256': sha(self.old_path),
            'render_vertices': self.old['vertices'], 'render_triangles': self.old['triangles']}
        self.current = {'active': True, 'artifact_sha256': sha(self.new_path),
            'canonical_vertices': self.new['vertices'], 'canonical_triangles': self.new['triangles'],
            'local_scale': [2., 2., 2.], 'local_position': [0., 1., 0.], 'attachment': {'status': 'absent'}}
        for key in ('source_vertices_unchanged', 'source_triangle_indices_unchanged',
                    'source_world_similarity_frame', 'source_render_mesh_bound',
                    'source_render_material_bound', 'source_display_enabled'):
            self.current[key] = True
        self.descriptor = {'native': {'original_signatures': 'kept'}, 'edge_interpolation': [[0, 1, .5]],
            'triangles': [0, 1, 2], 'source_ring_indices': [0, 1, 2], 'native_ring_cut_indices': [3, 4, 5],
            'collar_triangles': [1, 0, 3, 1, 3, 4, 2, 1, 4, 2, 4, 5, 0, 2, 5, 0, 5, 3],
            'design': {'normals_hash_exclusion': 'native mutable normals'}}

    def run_rebase(self):
        with patch('tools.model_bridge.attachment_rebase.package', return_value=copy.deepcopy(self.descriptor)), patch(
                'tools.model_bridge.attachment_rebase.original_artifact', side_effect=[(self.old, self.old_model), (self.new, self.model)]):
            return rebase({}, {}, self.reference, self.current, self.old_path, self.new_path)

    def test_distinct_identity_preserves_native_cut_and_binds_actual_placement(self):
        before = copy.deepcopy(self.new)
        result = self.run_rebase()
        self.assertEqual(result['source_artifact_sha256'], sha(self.new_path))
        self.assertEqual(result['source_placement'], {'scale': 2., 'translation': [0., 1., 0.]})
        for key in ('native', 'triangles', 'edge_interpolation', 'collar_triangles'):
            self.assertEqual(result[key], self.descriptor[key])
        self.assertEqual(self.new, before)
        self.assertFalse(result['design']['source_accuracy_certified'])

    def test_different_skin_definition_rejected(self):
        self.model.state['lbs_weights'][0] = .5
        with self.assertRaisesRegex(ValueError, 'lbs_weights'):
            self.run_rebase()

    def test_changed_face_order_rejected(self):
        self.new['triangles'] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, 'topology/order'):
            self.run_rebase()

    def test_posed_or_modified_current_mesh_rejected(self):
        self.current['canonical_vertices'] = copy.deepcopy(self.new['vertices'])
        self.current['canonical_vertices'][0][2] = .1
        with self.assertRaisesRegex(ValueError, 'Actual current canonical mesh'):
            self.run_rebase()

    def test_false_similarity_or_nonuniform_local_scale_rejected(self):
        for flag, scale in [(False, [2., 2., 2.]), (True, [2., 2.1, 2.])]:
            with self.subTest(flag=flag):
                self.current['source_world_similarity_frame'] = flag
                self.current['local_scale'] = scale
                with self.assertRaises(ValueError):
                    self.run_rebase()

    def test_second_attachment_or_unbound_renderer_rejected(self):
        self.current['attachment']['status'] = 'active'
        with self.assertRaisesRegex(ValueError, 'Detach'):
            self.run_rebase()
        self.current['attachment']['status'] = 'absent'
        self.current['source_render_mesh_bound'] = False
        with self.assertRaisesRegex(ValueError, 'Current exact original source'):
            self.run_rebase()


if __name__ == '__main__':
    unittest.main()
