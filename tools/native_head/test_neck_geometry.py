"""Regressions for preserving topology and native corners during neck fitting."""
import unittest

import numpy as np

from tools.native_head.neck_geometry import corresponding_lower_row, ordered_loops


class NeckGeometryTests(unittest.TestCase):
    def test_shifted_upper_polygon_keeps_original_corner_identity(self):
        lower=np.array([[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],float)
        upper=lower+[.7,.2,-.3]
        points=np.vstack([upper,(upper[0]+upper[1])/2])
        wanted=np.vstack([lower,(lower[0]+lower[1])/2])
        np.testing.assert_allclose(corresponding_lower_row(points,upper,lower),wanted,atol=1e-14)

    def test_off_polygon_mapping_is_rejected(self):
        p=np.array([[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]],float)
        with self.assertRaises(ValueError):corresponding_lower_row(np.array([[0,1,0]]),p,p)

    def test_boundary_is_oriented_and_rejects_nonmanifold_faces(self):
        self.assertEqual(ordered_loops([[0,1,2],[0,2,3]]),[[0,1,2,3]])
        with self.assertRaises(ValueError):ordered_loops([[0,1,2],[0,1,3]])


if __name__=='__main__':unittest.main()
