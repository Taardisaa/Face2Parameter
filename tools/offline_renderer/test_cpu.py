"""Analytic CPU clipping/depth/interpolation and installed cache tests."""
import unittest
import numpy as np
from src.render.cpu import rasterize, interpolate, project
from src.hs2_offline_state import reset, collect
from tools.abmx_replay.model import FLAG_FIELDS

def cache():
    d={k:False for k in FLAG_FIELDS}
    d.update({'_hasBaseline':True,'_sclBaseline':[2,3,4], '_rotBaseline':[0,0,0,1],
              '_posBaseline':[0,3,4], '_positionBaseline':[3,4,0], '_lenBaseline':10})
    return d

class CPU(unittest.TestCase):
    def setUp(self):
        self.camera={'projection':np.eye(4).ravel().tolist(),
                     'world_to_camera':np.eye(4).ravel().tolist()}

    def test_ortho_pixels_and_nearer_surface(self):
        v=np.array([[-1,-1,.7],[1,-1,.7],[0,1,.7],[-1,-1,-.5],[1,-1,-.5],[0,1,-.5]])
        f=np.array([[0,1,2],[3,4,5]])
        r=rasterize(v,f,self.camera,4,4)
        self.assertTrue(np.all(r.triangle[r.mask]==1))
        np.testing.assert_allclose(r.depth[r.mask],-.5)
        np.testing.assert_allclose(r.barycentric[r.mask].sum(1),1)
        pixels,_=project(v,self.camera,4,4)
        np.testing.assert_allclose(pixels[:3],[[0,4],[4,4],[2,0]])

    def test_perspective_interpolation_reprojects_pixel_centers(self):
        camera={'projection':np.array([[1,0,0,0],[0,1,0,0],[0,0,-11/9,-20/9],[0,0,-1,0]]).ravel().tolist(),
                'world_to_camera':np.eye(4).ravel().tolist()}
        v=np.array([[-1,-1,-2],[1,-1,-2],[0,2,-4]])
        f=np.array([[0,1,2]])
        r=rasterize(v,f,camera,32,32)
        interpolated=interpolate(r,f,v)[r.mask]
        pixels,_=project(interpolated,camera,32,32)
        y,x=np.nonzero(r.mask)
        np.testing.assert_allclose(pixels,np.c_[x+.5,y+.5],atol=1e-12)

    def test_near_clipping_retains_original_triangle_coordinates(self):
        v=np.array([[-.8,-.8,-2],[.8,-.8,0],[0,.8,0]])
        f=np.array([[0,1,2]])
        r=rasterize(v,f,self.camera,16,16)
        self.assertGreater(r.mask.sum(),0)
        xyz=interpolate(r,f,v)[r.mask]
        self.assertTrue(np.all(xyz[:,2]>=-1-1e-12))
        pixels,_=project(xyz,self.camera,16,16)
        y,x=np.nonzero(r.mask)
        np.testing.assert_allclose(pixels,np.c_[x+.5,y+.5],atol=1e-12)

    def test_reset_preserves_historical_length(self):
        d=cache(); local={}
        reset(local,d)
        np.testing.assert_allclose(local['local_position'],[0,6,8])
        collect(local,d)
        self.assertEqual(d['_lenBaseline'],10)
        self.assertEqual(d['_positionBaseline'],[3,4,0])
        self.assertEqual(d['_posBaseline'],[0,6,8])

    def test_reset_historical_direction_branch(self):
        d=cache(); d['_lenModNeedsPositionRestore']=True
        local={}; reset(local,d)
        np.testing.assert_allclose(local['local_position'],[3,4,0])
        self.assertFalse(d['_lenModNeedsPositionRestore'])

if __name__=='__main__':
    unittest.main()
