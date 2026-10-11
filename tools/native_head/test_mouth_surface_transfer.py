"""Expression/identity separation and source-bound chart/closure contracts."""
import unittest
import copy
import numpy as np
from scipy import sparse
from tools.native_head.mother_component_adaptation import adapt_frames
from tools.native_head.mother_default_pose import dense_frame
from tools.native_head.mouth_chart_solver import oriented_chart,areas
from tools.native_head.mouth_close_pose import close_adjustment
from tools.native_head.mouth_contact_contract import segment_points,barycentric
from tools.native_head.mother_bundle_candidate import validate_native_frames

def shapes():
    def row(index,v):return dict(index=index,vertex=dict(zip('xyz',v)),normal=dict(x=0.,y=0.,z=0.),tangent=dict(x=0.,y=0.,z=0.))
    return dict(channels=[dict(name='native.close',frameIndex=0,frameCount=1),dict(name='native.open',frameIndex=1,frameCount=1)],
        fullWeights=[100.,100.],shapes=[dict(firstVertex=i,vertexCount=1,hasNormals=True,hasTangents=True) for i in range(2)],
        vertices=[row(0,[.1,0,0]),row(1,[0,-.2,0])])

class MouthSurfaceContracts(unittest.TestCase):
    def test_close_delta_is_not_baked_into_bind_or_open(self):
        a=dict(verts=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float),
            normals=np.tile([0.,0.,1.],(4,1)),tangents=np.tile([1.,0.,0.,1.],(4,1)))
        s=shapes();jac=np.tile(np.eye(3),(4,1,1));ref=a['verts']+[.05,0,0]
        before,old_frames,_,_=adapt_frames(a,s,{'0':100},ref,jac)
        d=np.zeros((4,3));d[3]=[0,-.05,0]
        action=dict(channel=0,vertex=d,normal=np.zeros_like(d),tangent=np.zeros_like(d))
        after,frames,closed,error=adapt_frames(a,s,{'0':100},ref,jac,action)
        np.testing.assert_allclose(after['verts'],before['verts'],atol=1e-12,rtol=0)
        self.assertEqual(frames['channels'],s['channels']);self.assertEqual(frames['fullWeights'],s['fullWeights'])
        self.assertEqual(frames['shapes'][0]['vertexCount'],2)
        np.testing.assert_allclose(after['verts']+dense_frame(frames,0,4)['vertex'],ref+d,atol=1e-12)
        np.testing.assert_allclose(after['verts']+dense_frame(frames,1,4)['vertex'],before['verts']+dense_frame(old_frames,1,4)['vertex'],atol=1e-12)
        self.assertLess(max(error.values()),1e-12)
        contract=validate_native_frames(s,frames,4)
        self.assertTrue(contract['all_original_sparse_rows_retained'])
        self.assertEqual(contract['extra_sparse_rows'],1)
        broken=copy.deepcopy(frames);broken['vertices'][0]['index']=2
        with self.assertRaisesRegex(ValueError,'Original sparse rows'):
            validate_native_frames(s,broken,4)
        broken=copy.deepcopy(frames);broken['fullWeights'][0]=50
        with self.assertRaisesRegex(ValueError,'weight identity'):
            validate_native_frames(s,broken,4)
        with self.assertRaisesRegex(ValueError,'actual default'):
            adapt_frames(a,s,{'0':50},ref,jac,action)

    def test_source_contact_virtual_points_live_on_both_actual_faces(self):
        a=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        b=np.array([[.2,-.1,-1],[.2,.8,1],[.2,.8,-1]])
        for point in segment_points(a,b):
            np.testing.assert_allclose(barycentric(point,a)@a,point,atol=1e-12)
            np.testing.assert_allclose(barycentric(point,b)@b,point,atol=1e-12)
        self.assertIsNone(segment_points(a,a+[0,0,1]))

    def test_close_constraints_and_exterior_are_exact(self):
        reference=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        identity=reference.copy();identity[0,2]=.1;identity[1,2]=.2
        C=sparse.csr_matrix([[1.,-1,0,0]]);A=sparse.csr_matrix([[0.,0,1,0]])
        d,report=close_adjustment(reference,identity,C,[0,1],A)
        np.testing.assert_allclose(C@(identity+d-reference),0,atol=1e-12)
        np.testing.assert_array_equal(d[2:],np.zeros((2,3)))
        np.testing.assert_array_equal(identity[:,0],reference[:,0])
        with self.assertRaisesRegex(ValueError,'conflict'):
            close_adjustment(reference,identity,C,[],A)

    def test_chart_keeps_anatomical_anchors_and_rejects_dependent_data(self):
        p=np.array([[0.,0],[1,0],[1,1],[0,1],[1.3,.5]])
        f=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
        A=np.eye(5)[:4];fixed=p[:4].copy()
        mapped,report=oriented_chart(p,f,A,fixed)
        np.testing.assert_allclose(mapped[:4],fixed,atol=1e-12)
        self.assertTrue(np.all(areas(mapped,f)>0))
        self.assertFalse(report['identity_coefficients_used'])
        native=p.copy();native[4]=[.5,.5];native[:,0]*=-1
        regular,quality=oriented_chart(p,f,A,fixed,native)
        np.testing.assert_allclose(regular[:4],fixed,atol=1e-12)
        self.assertTrue(np.all(areas(regular,f)>0))
        self.assertFalse(quality['numeric_trial_continuation_used_in_result'])
        self.assertGreater(quality['minimum_relative_area'],.5)
        with self.assertRaisesRegex(ValueError,'Dependent'):
            oriented_chart(p,f,np.vstack([A,A[0]]),np.vstack([fixed,fixed[0]]))

if __name__=='__main__':unittest.main()
