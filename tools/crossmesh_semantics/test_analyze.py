import unittest
import numpy as np
from analyze import segment, barycentric, submesh_for_triangle, pairing


class SemanticsTests(unittest.TestCase):
    def test_known_crossing_and_translation(self):
        a=np.array([[-1,-1,0],[1,-1,0],[0,1,0]],float)
        b=np.array([[0,-.5,-1],[0,-.5,1],[0,.5,0]],float)
        for shift in [np.zeros(3),np.array([14.2,-9.7,41.3])]:
            result=segment(a+shift,b+shift,1e-7)
            self.assertEqual(result['classification']['kind'],'proper_crossing')
            np.testing.assert_allclose(sorted(np.asarray(result['endpoints_world'])[:,1]-shift[1]),[-.5,.5],atol=1e-13)
            self.assertAlmostEqual(result['length'],1)
            self.assertLess(result['plane_residual_max'],1e-13)
            self.assertGreaterEqual(result['left_barycentric']['min_weight'],-1e-13)

    def test_no_intersection_not_fabricated(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        self.assertIsNone(segment(a,a+[0,0,1],1e-7)['classification'])

    def test_bary_uv_preserves_affine_correspondence(self):
        triangle=np.array([[0,0,0],[2,0,0],[0,3,0]],float)
        weights=np.array([[.2,.3,.5],[1,0,0]])
        actual=np.array(barycentric(triangle,weights@triangle)['weights'])
        uv=np.array([[.1,.2],[.7,.3],[.2,.9]])
        np.testing.assert_allclose(actual@uv,weights@uv,atol=1e-15)

    def test_submesh_ordered_ids_no_guess(self):
        mesh={'source':{'submeshes':[{'submesh_index':0,'topology':0,'base_vertex':2,'indices_apply_base_vertex':False,'indices':[0,1,2]}]}}
        self.assertEqual(submesh_for_triangle(mesh,np.array([2,3,4])),[{'submesh_index':0,'local_triangle_id':0}])
        self.assertEqual(submesh_for_triangle(mesh,np.array([2,4,3])),[])

    def test_camera_pair_source_frame_and_state_tamper(self):
        from copy import deepcopy
        snapshot={'pose_signature':'abc','frame_count':7,'frame_count_end':7,'capture_state':{'x':1}}
        view={'paired_geometry':{'path':__file__,'pose_signature':'abc','frame_count':7,'capture_state':{'x':1}},
              'paired_pose_unchanged':True,'pose_signature_before_render':'abc','pose_signature_after_render':'abc',
              'frame_count':7,'frame_count_before_render':7}
        pairing(view,snapshot,__file__)
        for mutation in ['path','pose_signature','frame_count','capture_state']:
            bad=deepcopy(view); bad['paired_geometry'][mutation]={'x':2} if mutation=='capture_state' else 8 if mutation=='frame_count' else 'bad'
            with self.assertRaises(ValueError): pairing(bad,snapshot,__file__)


if __name__=='__main__': unittest.main()
