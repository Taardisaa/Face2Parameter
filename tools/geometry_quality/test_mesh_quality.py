"""Analytic cases for geometry-quality predicates; no game connection or assets."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from mesh_quality import Thresholds, analyze_snapshot, triangle_intersection, self_intersections


def snapshot(vertices, faces, *, normals=None, matrix=None, mesh_name="o_head", path="/head[0]", enabled=True):
    identity = np.eye(4).reshape(-1).tolist()
    return {
        "schema_version": 1, "snapshot_kind": "maker_live_skinned_geometry",
        "transforms": [{"id": 1, "local_to_world": matrix if matrix is not None else identity}],
        "meshes": [{"mesh_name": mesh_name, "renderer_path": path, "source_geometry_sha256": "analytic_source",
                    "enabled": enabled, "active_in_hierarchy": True,
                    "bone_names": ["root"], "bone_transform_ids": [1],
                    "baked": {"vertices": np.asarray(vertices).tolist(), "triangles": np.asarray(faces).reshape(-1).tolist(),
                              "normals": np.asarray(normals).tolist() if normals is not None else []}}]
    }


def first_report(snap, baseline=None):
    return analyze_snapshot(snap, baseline)["meshes"][0]


class TriangleIntersectionTests(unittest.TestCase):
    def test_actual_noncoplanar_crossing(self):
        a = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=float)
        b = np.array([[0, 0, -1], [0, 0, 1], [0, 2, 0]], dtype=float)
        result = triangle_intersection(a, b, 1e-9)
        self.assertEqual(result["kind"], "proper_crossing")
        self.assertAlmostEqual(result["intersection_length"], 1)

    def test_aabb_overlap_is_not_intersection(self):
        a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
        b = np.array([[2, 2, 0], [0.9, 2, 0], [2, 0.9, 0]], dtype=float)
        self.assertTrue(np.all(a.min(0) <= b.max(0)) and np.all(b.min(0) <= a.max(0)))
        self.assertIsNone(triangle_intersection(a, b, 1e-9))

    def test_coplanar_positive_area_overlap(self):
        a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float)
        b = np.array([[0.25, 0.25, 0], [1, 0.25, 0], [0.25, 1, 0]], dtype=float)
        result = triangle_intersection(a, b, 1e-9)
        self.assertEqual(result["kind"], "coplanar_overlap")
        self.assertAlmostEqual(result["overlap_area"], 0.75 ** 2 / 2)

    def test_coplanar_edge_contact_has_zero_area(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        b = np.array([[0, 0, 0], [1, 0, 0], [1, -1, 0]], dtype=float)
        result = triangle_intersection(a, b, 1e-9)
        self.assertEqual(result["kind"], "coplanar_contact")

    def test_nonintersecting_parallel_planes(self):
        a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        self.assertIsNone(triangle_intersection(a, a + [0, 0, 0.1], 1e-9))

    def test_coplanar_overlap_under_large_translation(self):
        a = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=float) + [1e6, 1e6, 1e6]
        b = np.array([[0.25, 0.25, 0], [1, 0.25, 0], [0.25, 1, 0]], dtype=float) + [1e6, 1e6, 1e6]
        result = triangle_intersection(a, b, 1e-9)
        self.assertEqual(result["kind"], "coplanar_overlap")
        self.assertAlmostEqual(result["overlap_area"], 0.75 ** 2 / 2)

    def test_crossing_rotation_translation_and_scale(self):
        a = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=float)
        b = np.array([[0, 0, -1], [0, 0, 1], [0, 2, 0]], dtype=float)
        q, _ = np.linalg.qr(np.array([[1, 2, 3], [4, 0, 2], [1, 1, 0]], dtype=float))
        a, b = (a @ q) * 7 + [20, -10, 5], (b @ q) * 7 + [20, -10, 5]
        self.assertEqual(triangle_intersection(a, b, 1e-8)["kind"], "proper_crossing")


class MeshQualityTests(unittest.TestCase):
    def test_shared_edge_does_not_excuse_positive_area_fold(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [.5, .5, 0]])
        faces = np.array([[0, 1, 2], [1, 0, 3]])
        excluded = self_intersections(vertices, faces, vertices[faces], np.ones(2, bool), 1e-9)
        self.assertEqual(excluded['narrow_phase_pairs'], 0)
        checked = self_intersections(vertices, faces, vertices[faces], np.ones(2, bool), 1e-9,
                                     exclude_shared_vertex=False)
        self.assertEqual(checked['counts']['coplanar_overlap'], 1)
        self.assertEqual(checked['excluded_shared_vertex_pairs'], 0)

    def test_legal_shared_edge_excluded(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        report = first_report(snapshot(vertices, [[0, 1, 2], [1, 3, 2]]))
        self.assertEqual(report["self_intersections"]["excluded_shared_vertex_pairs"], 1)
        self.assertEqual(report["self_intersections"]["narrow_phase_pairs"], 0)
        self.assertEqual(report["topology"]["boundary_edge_count"], 4)
        self.assertEqual(report["topology"]["inconsistent_winding_edge_count"], 0)

    def test_disjoint_index_triangles_really_cross(self):
        vertices = [[-1, -1, 0], [1, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [0, 2, 0]]
        report = first_report(snapshot(vertices, [[0, 1, 2], [3, 4, 5]]))
        self.assertEqual(report["self_intersections"]["counts"]["proper_crossing"], 1)
        self.assertEqual(report["topology"]["vertex_connected_components"], 2)

    def test_equilateral_aspect_and_area(self):
        report = first_report(snapshot([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]], [[0, 1, 2]]))
        self.assertAlmostEqual(report["aspect_distribution"]["p50"], 1)
        self.assertAlmostEqual(report["total_area"], np.sqrt(3) / 4)

    def test_degenerate_triangle_reported_and_not_intersection_tested(self):
        report = first_report(snapshot([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]))
        self.assertEqual(report["degenerate_triangle_ids"], [0])
        self.assertEqual(report["infinite_aspect_count"], 1)
        self.assertEqual(report["self_intersections"]["excluded_degenerate_triangles"], 1)

    def test_nonmanifold_edge_and_normal_direction(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0.5, 0, 1]]
        report = first_report(snapshot(vertices, [[0, 1, 2], [1, 0, 3], [0, 1, 4]], normals=[[0, 0, -1]] * 5))
        self.assertEqual(report["topology"]["nonmanifold_edge_count"], 1)
        self.assertIn(0, report["normal_direction"]["opposed_triangle_ids"])

    def test_bowtie_vertex_detected_even_without_nonmanifold_edges(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]
        report = first_report(snapshot(vertices, [[0, 1, 2], [0, 3, 4]]))
        self.assertEqual(report["topology"]["nonmanifold_edge_count"], 0)
        self.assertEqual(report["topology"]["nonmanifold_vertex_count"], 1)
        self.assertEqual(report["topology"]["nonmanifold_vertices"][0]["vertex"], 0)

    def test_negative_bone_world_determinant(self):
        matrix = np.diag([-2, 3, 4, 1]).reshape(-1).tolist()
        report = first_report(snapshot([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], matrix=matrix))
        self.assertEqual(report["bone_world_determinants"]["negative_count"], 1)
        self.assertAlmostEqual(report["bone_world_determinants"]["bones"][0]["world_determinant"], -24)

    def test_baseline_stretch_preserves_scale_and_removes_rigid_pose(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        baseline = snapshot(vertices, faces)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        rigid = first_report(snapshot(vertices @ rotation + [10, 20, 30], faces), baseline)["baseline"]
        self.assertEqual(rigid["normal_reversal_triangle_ids"], [])
        self.assertLess(rigid["displacement_after_alignment_distribution"]["max"], 1e-12)
        stretched = first_report(snapshot(vertices * 3, faces), baseline)["baseline"]
        self.assertAlmostEqual(stretched["edge_ratio_distribution"]["p50"], 3)
        self.assertAlmostEqual(stretched["area_ratio_distribution"]["p50"], 9)
        self.assertTrue(stretched["unusual_edges"])

    def test_local_face_fold_reversal(self):
        vertices = np.array([[x, y, 0] for y in range(6) for x in range(6)], dtype=float)
        faces = []
        for y in range(5):
            for x in range(5):
                a = y * 6 + x
                faces.extend([[a, a + 1, a + 6], [a + 1, a + 7, a + 6]])
        baseline = snapshot(vertices, faces)
        vertices[2 * 6 + 2] += [0, 2.5, 0]
        report = first_report(snapshot(vertices, faces), baseline)["baseline"]
        self.assertTrue(report["normal_reversal_triangle_ids"])

    def test_topology_mismatch_does_not_compare_deformation(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        report = first_report(snapshot(vertices, [[0, 2, 1]]), snapshot(vertices, [[0, 1, 2]]))["baseline"]
        self.assertEqual(report["status"], "topology_mismatch")
        self.assertFalse(report["deformation_comparison_performed"])

    def test_duplicate_mesh_names_preserve_renderer_and_visibility(self):
        base = snapshot([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], mesh_name="o_tang", path="/visible[0]")
        disabled = copy.deepcopy(base["meshes"][0])
        disabled["renderer_path"], disabled["enabled"] = "/silhouette[1]", False
        base["meshes"].append(disabled)
        self.assertEqual(analyze_snapshot(base)["mesh_count"], 2)
        self.assertEqual(analyze_snapshot(base, visible_only=True)["mesh_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
