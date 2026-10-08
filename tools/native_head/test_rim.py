"""Neck-authoring regression: face/chin and retained height are hard locks."""
import unittest
from unittest.mock import patch
import numpy as np
from tools.native_head.rim import conform,transition


class NeckLocks(unittest.TestCase):
    def test_front_face_height_and_shared_transition(self):
        count=12;angles=np.arange(count)*2*np.pi/count
        ring=np.c_[np.sin(angles),np.zeros(count),np.cos(angles)]
        vertices=np.r_[ring,ring+[0,.2,0],ring+[0,.5,0]]
        faces=[]
        for row in range(2):
            for i in range(count):
                j=(i+1)%count;a=row*count;b=(row+1)*count
                faces.extend([[a+i,a+j,b+i],[a+j,b+j,b+i]])
        faces=np.array(faces);target=ring*.4;target[:,1]=-.15
        normals=np.c_[np.sin(angles),np.zeros(count),np.cos(angles)]
        protected=vertices[:,2]>.5
        with patch('tools.native_head.rim.native_contour',return_value=(target,normals)):
            result,_,design=conform(vertices,faces,np.arange(count),{}, {},.4,protected)
        self.assertTrue(np.array_equal(result[protected],vertices[protected]))
        self.assertTrue(np.array_equal(result[:,1],vertices[:,1]))
        self.assertTrue(np.array_equal(result[vertices[:,2]>=.4],vertices[vertices[:,2]>=.4]))
        self.assertTrue(np.any(result[~protected]!=vertices[~protected]))
        all_vertices,all_faces,_,inner=transition(result,faces,np.arange(count),
            design['target_ring'],normals,design['target_normals'])
        self.assertTrue(np.array_equal(all_vertices[:len(vertices)],result))
        self.assertTrue(np.array_equal(all_faces[:len(faces)],faces))
        self.assertTrue(np.array_equal(all_vertices[inner],design['target_ring']))
        self.assertEqual(len(all_vertices)-len(vertices),count*4)
        directed=np.concatenate([all_faces[:,[0,1]],all_faces[:,[1,2]],all_faces[:,[2,0]]])
        for i in range(count):
            j=(i+1)%count
            self.assertEqual(np.sum(np.all(directed==[i,j],axis=1)),1)
            self.assertEqual(np.sum(np.all(directed==[j,i],axis=1)),1)


if __name__=='__main__':unittest.main()
