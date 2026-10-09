"""Contracts for the new identity authoring field, without game sweeps."""
import unittest

import numpy as np

from tools.native_head.mother_identity import barycentric, coherent_mouth, neck_pin_basis


class IdentityContracts(unittest.TestCase):
    def test_surface_correspondence_preserves_affine_field(self):
        triangles=np.array([[[0.,0,0],[1.,0,0],[0,1.,0]]])
        points=np.array([[.3,.2,0.]])
        weights=barycentric(points,triangles)
        np.testing.assert_allclose(weights@triangles[0],points)
        values=triangles[0]@np.diag([2.,3.,4.])+[1.,2.,3.]
        np.testing.assert_allclose(weights@values,points@np.diag([2.,3.,4.])+[1.,2.,3.])

    def test_neck_pin_is_local_and_linear_in_all_coefficients(self):
        vertices=np.array([[i,j,0.] for i in range(7) for j in range(2)],float)
        faces=np.array([[i*2,i*2+1,i*2+2] for i in range(6)]+
                       [[i*2+1,i*2+3,i*2+2] for i in range(6)])
        basis=np.arange(len(vertices)*3*2,dtype=float).reshape(len(vertices),3,2)/100
        result,record=neck_pin_basis(vertices,faces,basis,np.array([0,1]),3)
        np.testing.assert_array_equal(result[:2],0)
        np.testing.assert_array_equal(result[8:],basis[8:])
        beta=np.array([.4,-.2])
        direct,_=neck_pin_basis(vertices,faces,(basis@beta)[:,:,None],np.array([0,1]),3)
        np.testing.assert_allclose(result@beta,direct[:,:,0],atol=1e-14)

    def test_explicit_mouth_preview_reproduces_affine_controls(self):
        vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
        faces=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
        control=vertices*np.array([.2,.3,.4])+[.1,-.2,.3]
        result,record=coherent_mouth(vertices,faces,np.zeros((4,3,1)),np.arange(4),vertices,control[:,:,None])
        np.testing.assert_allclose(result[:,:,0],control,atol=1e-14)
        self.assertFalse(record['exact_flame_lip_reconstruction'])


if __name__=='__main__':
    unittest.main()
