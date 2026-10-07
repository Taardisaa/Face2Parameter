import unittest

import numpy as np

from tools.surface_calibration.core import Camera, ContractError
from tools.surface_calibration.silhouette import compare_masks, observed_foreground, raster_union


class SilhouetteTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(8, 8, np.eye(4), np.eye(4), np.array([0, 0, 8, 8]))
        self.vertices = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]])
        self.triangles = np.array([[0, 1, 2], [0, 2, 3]])

    def test_known_rectangle_both_windings(self):
        expected = np.zeros((8, 8), bool)
        expected[2:6, 2:6] = True
        np.testing.assert_array_equal(raster_union(self.camera, self.vertices, self.triangles), expected)
        np.testing.assert_array_equal(raster_union(self.camera, self.vertices, self.triangles[:, ::-1]), expected)

    def test_shift_is_measured_without_alignment(self):
        mask = raster_union(self.camera, self.vertices, self.triangles)
        shifted = np.roll(mask, 2, axis=1)
        report = compare_masks(mask, shifted)
        self.assertAlmostEqual(report["iou"], 1 / 3)
        self.assertEqual(report["symmetric_boundary_max_px"], 2)

    def test_identical_masks(self):
        mask = raster_union(self.camera, self.vertices, self.triangles)
        report = compare_masks(mask, mask)
        self.assertEqual(report["iou"], 1)
        self.assertEqual(report["symmetric_boundary_p95_px"], 0)

    def test_background_and_contamination(self):
        rgb = np.full((8, 8, 3), 128, dtype=np.uint8)
        rgb[2:6, 2:6] = 30
        mask, color = observed_foreground(rgb)
        self.assertEqual(mask.sum(), 16)
        self.assertEqual(color, [128, 128, 128])
        rgb[0, 0] = 30
        with self.assertRaises(ContractError):
            observed_foreground(rgb)

    def test_clipped_depth_and_bad_indices_fail(self):
        vertices = self.vertices.copy()
        vertices[0, 2] = 2
        with self.assertRaises(ContractError):
            raster_union(self.camera, vertices, self.triangles)
        with self.assertRaises(ContractError):
            raster_union(self.camera, self.vertices, np.array([[0, 1, 7]]))


if __name__ == "__main__":
    unittest.main()
