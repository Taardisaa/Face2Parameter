"""Analytic checks for the CalculateFocusedLightHull transcription (RVA 0x6deb40)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from src.render.hs2pipe import unity_light_hull as hull

F = np.float32


def cube(a, b):
    """8 corners in GetFrustumPoints order: (-,-,-) (+,-,-) (+,+,-) (-,+,-) then z = +."""
    return [(a, a, a), (b, a, a), (b, b, a), (a, b, a),
            (a, a, b), (b, a, b), (b, b, b), (a, b, b)]


def as_floats(points):
    return [tuple(float(c) for c in p) for p in points]


class PrimitiveTests(unittest.TestCase):
    def test_constants_are_exact_bit_patterns(self):
        self.assertEqual(hull.EPS_SEGMENT_PLANE, F(1e-6))
        self.assertEqual(hull.RAY_TMAX_INIT, F(1e30))
        self.assertEqual(hull.RAY_EPS, F(1e-10))
        self.assertEqual(hull.RAY_NEG_EPS, F(-1e-10))

    def test_minmax_aabb_is_valid(self):
        inf = float('inf')
        self.assertFalse(hull.minmax_aabb_is_valid((inf, inf, inf), (-inf, -inf, -inf)))
        self.assertFalse(hull.minmax_aabb_is_valid((inf, inf, inf), (1, 1, 1)))
        self.assertFalse(hull.minmax_aabb_is_valid((0, 0, 0), (-inf, -inf, -inf)))
        self.assertTrue(hull.minmax_aabb_is_valid((0, 0, 0), (1, 1, 1)))
        self.assertTrue(hull.minmax_aabb_is_valid((inf, inf, float('nan')), (1, 1, 1)))
        self.assertFalse(hull.minmax_aabb_is_valid((inf, inf, float('nan')), (-inf, -inf, -inf)))

    def test_planes_layout(self):
        planes = hull.minmax_aabb_planes((1, 2, 3), (4, 5, 6))
        self.assertEqual([tuple(float(c) for c in p) for p in planes], [
            (0.0, 1.0, 0.0, -2.0), (0.0, -1.0, 0.0, 5.0),
            (1.0, 0.0, 0.0, -1.0), (-1.0, 0.0, 0.0, 4.0),
            (0.0, 0.0, 1.0, -3.0), (0.0, 0.0, -1.0, 6.0)])

    def test_intersect_segment_plane(self):
        plane = (F(0), F(1), F(0), F(0))
        p0 = (F(0), F(-1), F(0))
        p1 = (F(2), F(1), F(4))
        ok, q = hull.intersect_segment_plane(p0, p1, plane)
        self.assertTrue(ok)
        self.assertEqual(as_floats([q]), [(1.0, 0.0, 2.0)])
        # parallel segment: |denom| <= 1e-6 is rejected
        ok, _ = hull.intersect_segment_plane((F(0), F(1), F(0)), (F(5), F(1), F(0)), plane)
        self.assertFalse(ok)
        # crossing beyond the segment end: t > 1 rejected
        ok, _ = hull.intersect_segment_plane((F(0), F(-3), F(0)), (F(0), F(-1), F(0)), plane)
        self.assertFalse(ok)

    def test_clip_polygon_rotates_inside_vertices(self):
        plane = (F(0), F(1), F(0), F(10))   # y >= -10, everything inside
        poly = [(F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(1), F(1), F(0))]
        cap = []
        kept, out = hull.clip_polys_by_plane(poly, plane, cap)
        self.assertTrue(kept)
        self.assertEqual(out, [poly[1], poly[2], poly[0]])
        self.assertEqual(cap, [])

    def test_clip_polygon_with_nan_vertex_counts_inside(self):
        plane = (F(0), F(1), F(0), F(0))
        nan = F('nan')
        poly = [(F(0), nan, F(0)), (F(1), F(1), F(0)), (F(0), F(1), F(0))]
        kept, out = hull.clip_polys_by_plane(poly, plane, [])
        self.assertTrue(kept)
        self.assertEqual(len(out), 3)

    def test_degenerate_polygon_rejected(self):
        kept, out = hull.clip_polys_by_plane([(F(0), F(0), F(0))] * 2, (F(0), F(1), F(0), F(0)), [])
        self.assertFalse(kept)
        self.assertEqual(out, [])

    def test_ray_far_point(self):
        mn, mx = (F(0), F(0), F(0)), (F(2), F(3), F(4))
        ok, q = hull.ray_minmax_aabb_far_point((F(1), F(1), F(1)), (F(1), F(0), F(0)), mn, mx)
        self.assertTrue(ok)
        self.assertEqual(as_floats([q]), [(2.0, 1.0, 1.0)])
        # outside origin entering the box: far exit, not entry
        ok, q = hull.ray_minmax_aabb_far_point((F(-5), F(1), F(1)), (F(1), F(0), F(0)), mn, mx)
        self.assertTrue(ok)
        self.assertEqual(as_floats([q]), [(2.0, 1.0, 1.0)])
        # pointing away from the box
        ok, q = hull.ray_minmax_aabb_far_point((F(-5), F(1), F(1)), (F(-1), F(0), F(0)), mn, mx, 'kept')
        self.assertFalse(ok)
        self.assertEqual(q, 'kept')
        # parallel and outside the slab
        ok, _ = hull.ray_minmax_aabb_far_point((F(1), F(9), F(1)), (F(1), F(0), F(0)), mn, mx)
        self.assertFalse(ok)


class HullTests(unittest.TestCase):
    def test_empty_intersection(self):
        self.assertEqual(hull.calculate_focused_light_hull(cube(-1, 1), (0, -1, 0), ((5, 5, 5), (6, 6, 6))), [])

    def test_output_is_float32(self):
        out = hull.calculate_focused_light_hull(cube(-1, 1), (0, -1, 0), ((0, 0, 0), (2, 2, 2)))
        self.assertTrue(out)
        for p in out:
            self.assertEqual(len(p), 3)
            for c in p:
                self.assertIsInstance(c, np.float32)

    def test_frustum_inside_aabb_exact_sequence(self):
        # Perspective-like frustum fully inside the AABB: no plane cuts, each of
        # the six stages rotates every quad by one vertex (6 mod 4 = 2).
        near = [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
        far = [(-4, -4, 8), (4, -4, 8), (4, 4, 8), (-4, 4, 8)]
        pts = near + far
        out = hull.calculate_focused_light_hull(pts, (0, 0, 1), [-100, -100, -100, 100, 100, 100])
        expected = []
        for quad in hull._FRUSTUM_QUADS:
            for k in (2, 3, 0, 1):
                x, y, z = pts[quad[k]]
                expected.append((float(x), float(y), float(z)))
                expected.append((float(x), float(y), -100.0))   # far exit along -light_dir
        self.assertEqual(as_floats(out), expected)

    def test_aabb_inside_frustum(self):
        box = ((0, 0, 0), (1, 1, 1))
        out = as_floats(hull.calculate_focused_light_hull(cube(-10, 10), (0, -1, 0), box))
        corners = {(float(x), float(y), float(z)) for x in (0, 1) for y in (0, 1) for z in (0, 1)}
        self.assertTrue(corners.issubset(set(out)))
        for p in out:
            for c in p:
                self.assertTrue(0.0 <= c <= 1.0, p)
        # vertices alternate with their exits at y = max.y (extrusion along +y)
        exits = [p for p in out if p[1] == 1.0]
        self.assertTrue(exits)

    def test_partial_overlap(self):
        # frustum cube [-1,1]^3 against AABB [0,2]^3: intersection [0,1]^3,
        # extruded along +y (light travelling -y) up to y = 2.
        out = as_floats(hull.calculate_focused_light_hull(cube(-1, 1), (0, -1, 0), ((0, 0, 0), (2, 2, 2))))
        expected = {(float(x), float(y), float(z)) for x in (0, 1) for y in (0, 1, 2) for z in (0, 1)}
        self.assertEqual(set(out), expected)

    def test_diagonal_light_exits_on_aabb_boundary(self):
        box = ((-3, -3, -3), (3, 3, 3))
        d = (F(0.6), F(-0.8), F(0.0))
        out = hull.calculate_focused_light_hull(cube(-1, 1), d, box)
        pts = as_floats(out)
        self.assertEqual(len(pts), 48)
        for v, e in zip(pts[0::2], pts[1::2]):
            self.assertTrue(any(math.isclose(abs(c), 3.0, rel_tol=0, abs_tol=1e-5) for c in e), e)
            self.assertLessEqual(e[0], v[0])
            self.assertGreaterEqual(e[1], v[1])

    def test_cap_omitted_for_last_plane(self):
        # Only plane 5 (z <= max.z) cuts: the far quad is clipped away and no
        # cap polygon is appended.
        polys = hull.clip_frustum_by_aabb(cube(-1, 1), (-2, -2, -2), (2, 2, 0))
        self.assertEqual(len(polys), 5)
        # Only plane 4 (z >= min.z) cuts: the near quad is removed and the cap
        # survives the final stage.
        polys = hull.clip_frustum_by_aabb(cube(-1, 1), (-2, -2, 0), (2, 2, 2))
        self.assertEqual(len(polys), 6)
        self.assertEqual(len(polys[-1]), 8)   # 4 crossing faces x 2 points, encounter order


if __name__ == '__main__':
    unittest.main()
