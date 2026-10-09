"""Local repair must not propagate changes into the accepted exterior."""
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from tools.native_head.mother_local_repair import local_solve, similarity
from tools.native_head.mother_local_contacts import separate_pair, dirty_crossings, correct
from tools.geometry_quality.mesh_quality import triangle_intersection
from tools.model_bridge.artifact import sha


class LocalRepairTests(unittest.TestCase):
    def test_damaged_patch_recovers_without_moving_exterior(self):
        x, y = np.meshgrid(np.arange(7.), np.arange(7.))
        original = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        faces = []
        for row in range(6):
            for col in range(6):
                i = row*7+col
                faces.extend([[i, i+1, i+7], [i+1, i+8, i+7]])
        faces = np.asarray(faces)
        accepted = original+np.array([5., -2., 3.])
        active = (x.ravel() >= 2) & (x.ravel() <= 4) & (y.ravel() >= 2) & (y.ravel() <= 4)
        damaged = accepted.copy()
        damaged[24] += [0., 0., 1.]
        fixed = np.zeros(len(original), bool)
        fixed[16] = True  # An explicit landmark inside the patch.
        repaired, _ = local_solve(original, damaged, faces, active, fixed)
        np.testing.assert_array_equal(repaired[~active | fixed], damaged[~active | fixed])
        self.assertLess(np.linalg.norm(repaired[24]-accepted[24]), .1)
        self.assertGreater(np.linalg.norm(damaged[24]-accepted[24]), .9)

    def test_boundary_similarity_is_proper_and_uniform(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        target = source@rotation*1.4+np.array([2., 3., 4.])
        scale, fitted, translation = similarity(source, target)
        np.testing.assert_allclose(source@fitted*scale+translation, target, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(fitted), 1.)

    def test_shared_edge_fold_is_not_repaired_by_collapsing_triangle(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., .2]])
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        damaged = source.copy(); damaged[3] = [.2, .3, 0.]
        movable = np.array([False, False, False, True])
        self.assertEqual(triangle_intersection(*damaged[faces], 1e-8)['kind'], 'coplanar_overlap')
        self.assertTrue(separate_pair(source, damaged, faces, movable, np.eye(3)))
        np.testing.assert_array_equal(damaged[:3], source[:3])
        self.assertGreater(np.linalg.norm(np.cross(damaged[0]-damaged[1], damaged[3]-damaged[1])), .01)
        result = triangle_intersection(*damaged[faces], 1e-8)
        self.assertTrue(result is None or result['kind'] not in ('proper_crossing', 'coplanar_overlap'))

    def test_dirty_triangle_is_checked_against_untouched_triangle(self):
        positions = np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],
            [.5,.5,-1.],[.5,.5,1.],[1.,.5,1.]])
        faces = np.array([[0,1,2],[3,4,5]])
        self.assertEqual(dirty_crossings(positions,faces,np.array([True,False]),set()),{(0,1)})

    def test_default_contact_mode_does_not_restore_source_implicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); candidate=root/'candidate'; candidate.mkdir()
            quality=root/'quality'; quality.mkdir(); inputs=root/'inputs'; inputs.mkdir()
            original=np.array([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.],
                [.5,.5,-1.],[.5,.5,1.],[1.,.5,1.]])
            accepted=original+np.array([.1,.2,.3])
            arrays=candidate/'o_head_candidate.npz'
            np.savez_compressed(arrays,verts=accepted,original_vertices=original,
                faces=np.array([[0,1,2],[3,4,5]]),bone_names=np.array(['cf_J_MouthBase_s']),
                bone_idx=np.zeros((6,4),int),bone_w=np.tile([1.,0.,0.,0.],(6,1)))
            (inputs/'native_regions.json').write_text(json.dumps({'graph_aliases':{},'physical_eye_boundaries':{}}))
            receipt={'inputs':{'path':str(inputs/'receipt.json')},'eye_constraint_residual':0.,
                'local_repair':{'base_candidate':{'path':str(candidate/'receipt.json')},
                               'eye_boundary_unchanged_exactly':True}}
            (candidate/'receipt.json').write_text(json.dumps(receipt))
            mask={'moved_render_vertex_ids':list(range(6)), 'patches':[{
                'free_logical_vertex_ids':list(range(6)), 'fixed_logical_boundary_ids':[],
                'boundary_similarity':{'scale':1.,'rotation':np.eye(3).tolist(),'translation':[0.,0.,0.]}}]}
            (candidate/'local_patch_regions.json').write_text(json.dumps(mask))
            (quality/'receipt.json').write_text(json.dumps({'arrays':{'sha256':sha(arrays)},
                'added_crossings':[[0,1]], 'source_crossings':[]}))
            out=root/'result'; correct(candidate,quality,out,passes=0)
            np.testing.assert_array_equal(np.load(out/'o_head_candidate.npz')['verts'],accepted)


if __name__ == '__main__':
    unittest.main()
