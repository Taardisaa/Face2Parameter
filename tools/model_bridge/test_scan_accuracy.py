"""Geometry checks independent of model inference or licensed scan fixtures.

Run: python -m unittest tools.model_bridge.test_scan_accuracy -v
"""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.model_bridge.scan_accuracy import (
    TriangleSurface, camera_rays, closest_on_triangles, first_ray_hits, read_scan, similarity,
)


class ScanAccuracyTests(unittest.TestCase):
    def test_surface_interior_edge_and_degenerate(self):
        triangle = np.array([[[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]])
        np.testing.assert_allclose(closest_on_triangles([.5, .5, 3], triangle), [[.5, .5, 0]])
        np.testing.assert_allclose(closest_on_triangles([2., 2., 1], triangle), [[1., 1., 0]])
        np.testing.assert_allclose(closest_on_triangles([-2., -1., 0], triangle), [[0., 0., 0]])
        degenerate = np.array([[[0., 0., 0.], [2., 0., 0.], [1., 0., 0.]]])
        np.testing.assert_allclose(closest_on_triangles([1., 3., 0], degenerate), [[1., 0., 0]])

    def test_far_centroid_can_have_nearest_surface(self):
        triangles = np.array([[[0., 0., .5], [1., 0., .5], [0., 1., .5]],
                              [[0., 0., 0.], [100., 0., 0.], [0., 100., 0.]]])
        point = [.1, .1, .01]
        nearest, ids, _ = TriangleSurface(triangles).closest([point])
        self.assertEqual(ids[0], 1)
        np.testing.assert_allclose(nearest, [[.1, .1, 0]], atol=1e-14)

    def test_acceleration_matches_exhaustive_surface(self):
        rng = np.random.default_rng(42)
        triangles = rng.normal(size=(150, 3, 3))
        points = rng.normal(size=(25, 3))
        nearest, _, _ = TriangleSurface(triangles).closest(points)
        for p, actual in zip(points, nearest):
            all_candidates = closest_on_triangles(p, triangles)
            expected = all_candidates[np.argmin(np.linalg.norm(all_candidates - p, axis=1))]
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_first_visible_intersection_and_miss(self):
        triangles = np.array([[[-2., -2., 5.], [2., -2., 5.], [0., 2., 5.]],
                              [[-2., -2., 2.], [2., -2., 2.], [0., 2., 2.]]])
        hits, ids = first_ray_hits(np.zeros(3), [[0., 0., 1.]], triangles, chunk=1)
        np.testing.assert_allclose(hits, [[0., 0., 2.]])
        self.assertEqual(ids[0], 1)
        with self.assertRaises(ValueError):
            first_ray_hits(np.zeros(3), [[0., 0., -1.]], triangles)

    def test_calibrated_ray_convention(self):
        k = np.array([[100., 0., 30.], [0., 100., 40.], [0., 0., 1.]])
        rt = np.c_[np.eye(3), [0., 0., -2.]]
        origin, rays = camera_rays([[30., 40.], [130., 40.]], k, rt, np.zeros(5))
        np.testing.assert_allclose(origin, [0., 0., 2.])
        np.testing.assert_allclose(rays, [[0., 0., 1.], [2**-.5, 0., 2**-.5]])

    def test_known_positive_similarity(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        target = 3 * source @ rotation.T + [4., -2., 7.]
        scale, result_rotation, translation = similarity(source, target)
        self.assertAlmostEqual(scale, 3.)
        self.assertGreater(np.linalg.det(result_rotation), 0)
        np.testing.assert_allclose(scale * source @ result_rotation.T + translation, target, atol=1e-14)

    def test_binary_ply_layout(self):
        header = ("ply\nformat binary_little_endian 1.0\ncomment fixture\nelement vertex 3\n"
                  "property float x\nproperty float y\nproperty float z\nelement face 1\n"
                  "property list uchar int vertex_indices\nend_header\n").encode()
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype="<f4")
        records = np.array([(3, [0, 1, 2])], dtype=[("n", "u1"), ("ids", "<i4", (3,))])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "triangle.ply"
            path.write_bytes(header + vertices.tobytes() + records.tobytes())
            actual, faces = read_scan(path)
            np.testing.assert_array_equal(actual, vertices)
            np.testing.assert_array_equal(faces, [[0, 1, 2]])
            path.write_bytes(path.read_bytes() + b"unexpected")
            with self.assertRaises(ValueError):
                read_scan(path)


if __name__ == "__main__":
    unittest.main()
