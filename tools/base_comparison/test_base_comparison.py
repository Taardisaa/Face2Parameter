"""Analytic metric/contract/gate cases; cached-rig fitting is a separate bench."""
import copy
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "geometry_quality"))

import numpy as np
from mesh_quality import analyze_snapshot
from contracts import Mesh, SCOPE, full_mesh_config, load_target, proper_rigid
from search import accepted, quality_gate
from evaluate import evaluate_pair
from surface import TriangleSurface, content_hash, evaluate_surface, quadrature, topology_hash


def square():
    return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float), np.array([[0, 1, 2], [0, 2, 3]])


def snapshot(vertices, faces):
    return {"schema_version": 1, "snapshot_kind": "maker_live_skinned_geometry", "transforms": [],
            "meshes": [{"mesh_name": "o_head", "renderer_path": "/head", "source_geometry_sha256": "a" * 64,
                        "baked": {"vertices": vertices.tolist(), "triangles": faces.reshape(-1).tolist()}}]}


class CommonSurfaceTests(unittest.TestCase):
    def test_parallel_complete_surfaces_distance_and_area(self):
        vertices, faces = square()
        other = vertices + [0, 0, .25]
        report = evaluate_surface(vertices, faces, other, faces, count=64)
        self.assertAlmostEqual(report["symmetric"]["rms"], .25)
        self.assertAlmostEqual(report["symmetric"]["p95"], .25)
        self.assertAlmostEqual(report["oriented_normal_degrees"]["rms"], 0)
        self.assertAlmostEqual(report["source_area"], 1)

    def test_query_distance_reaches_triangle_interior_not_vertex(self):
        vertices, faces = square()
        distances, _, bary = TriangleSurface(vertices, faces).closest([[.25, .2, .3]])
        self.assertAlmostEqual(distances[0], .3)
        self.assertTrue((bary[0] > 0).all())

    def test_kdtree_bound_matches_exhaustive_including_long_triangle(self):
        from surface import closest_on_triangles
        rng = np.random.default_rng(172)
        vertices = rng.normal(size=(36, 3))
        vertices[0] = [-100, 0, 0]
        faces = np.arange(36).reshape(-1, 3)
        surface = TriangleSurface(vertices, faces)
        points = rng.normal(size=(20, 3))
        distances, _, _ = surface.closest(points)
        expected = [np.sqrt(closest_on_triangles(point, surface.triangles)[0].min()) for point in points]
        np.testing.assert_allclose(distances, expected, atol=1e-10)

    def test_area_quadrature_not_vertex_density(self):
        vertices, faces = square()
        coarse = quadrature(vertices, faces, count=100)
        self.assertAlmostEqual(coarse.weights.sum(), 1)
        # Tiny densely tessellated patch must retain only its actual 0.01 area.
        small = vertices * .1 + [2, 0, 0]
        all_vertices = np.r_[vertices, small]
        all_faces = np.r_[faces, faces + 4]
        q = quadrature(all_vertices, all_faces, count=4)
        self.assertAlmostEqual(q.weights[q.face_ids >= 2].sum(), .01)
        self.assertEqual(set(q.face_ids), set(range(4)))

    def test_scale_is_preserved_not_fitted_away(self):
        vertices, faces = square()
        report = evaluate_surface(vertices, faces, vertices * 2, faces, count=512)
        self.assertGreater(report["symmetric"]["rms"], .1)
        with self.assertRaisesRegex(ValueError, "scale"):
            proper_rigid(np.diag([2, 2, 2, 1]))

    def test_surface_winding_normal_difference(self):
        vertices, faces = square()
        report = evaluate_surface(vertices, faces, vertices, faces[:, ::-1], count=20)
        self.assertLess(report["symmetric"]["max_sampled"], 1e-12)
        self.assertAlmostEqual(report["oriented_normal_degrees"]["p95"], 180)

    def test_unknown_space_scope_asset_hash_rejected(self):
        vertices, faces = square()
        mesh = Mesh(vertices, faces, {"units": "hs2_cache_units", "coordinate_frame": "hs2_cached_head_fk", "scope": dict(SCOPE),
                    "asset": {"provenance": "analytic fixture", "sha256": "a" * 64},
                    "content_sha256": content_hash(vertices, faces), "topology_sha256": topology_hash(faces)})
        config = full_mesh_config(mesh)
        load_target(config)
        for field, value in (("units", "unknown"), ("coordinate_frame", "world_unspecified"),
                             ("scope", {}), ("asset", {}), ("topology_sha256", "b" * 64)):
            changed = copy.deepcopy(config)
            changed["metadata"][field] = value
            with self.assertRaises(ValueError):
                load_target(changed)

    def test_declared_rigid_only_and_region_provenance(self):
        vertices, faces = square()
        matrix = np.eye(4)
        matrix[:3, 3] = [1, 2, 3]
        np.testing.assert_array_equal(proper_rigid(matrix), matrix)
        with self.assertRaisesRegex(ValueError, "provenance"):
            evaluate_surface(vertices, faces, vertices, faces, regions=[{"name": "nose", "side": "source", "face_ids": [0]}])

    def test_distance_zero_cannot_override_new_degenerate_quality_failure(self):
        vertices, faces = square()
        base = snapshot(vertices, faces)
        damaged = vertices.copy()
        damaged[1] = damaged[0]
        report = analyze_snapshot(snapshot(damaged, faces), base)
        quality = quality_gate(report, analyze_snapshot(base))
        self.assertFalse(quality["quality_valid"])
        self.assertIn("new degenerate triangles", quality["reasons"])
        surface = evaluate_surface(damaged, faces, damaged, faces, count=32)
        self.assertLess(surface["symmetric"]["max_sampled"], 1e-12)
        self.assertFalse(accepted(surface, quality, {"rms": 1, "p95": 1, "max_sampled": 1, "normal_p95_degrees": 180}))

    def test_existing_baseline_crossing_not_new_failure(self):
        vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [0, 2, 0]], float)
        faces = np.array([[0, 1, 2], [3, 4, 5]])
        base = snapshot(vertices, faces)
        gate = quality_gate(analyze_snapshot(base, base), analyze_snapshot(base))
        self.assertEqual(gate["existing_baseline_crossing_count"], 1)
        self.assertEqual(gate["new_crossing_pairs"], [])
        self.assertTrue(gate["quality_valid"])

    def test_full_mesh_pair_enforces_same_head_quality_baseline(self):
        vertices, faces = square()
        mesh = Mesh(vertices, faces, {"head_id": 2, "units": "hs2_cache_units", "coordinate_frame": "hs2_cached_head_fk", "scope": dict(SCOPE),
                    "asset": {"provenance": "analytic fixture", "sha256": "a" * 64},
                    "content_sha256": content_hash(vertices, faces), "topology_sha256": topology_hash(faces)})
        config = full_mesh_config(mesh)
        report = evaluate_pair(config, config, config, count=32)
        self.assertEqual(report["status"], "found_quality_valid_approximation")
        head3 = copy.deepcopy(config)
        head3["metadata"]["head_id"] = 3
        self.assertEqual(evaluate_pair(head3, head3, head3, count=32)["status"], "found_quality_valid_approximation")
        wrong = copy.deepcopy(config)
        wrong["metadata"]["head_id"] = 1
        with self.assertRaisesRegex(ValueError, "same declared"):
            evaluate_pair(config, config, wrong, count=32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
