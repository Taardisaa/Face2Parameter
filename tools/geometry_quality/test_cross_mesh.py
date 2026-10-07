"""Analytic cross-renderer cases and strict certificate binding rejection tests."""
import copy
import hashlib
import json
import unittest

import numpy as np

from cross_mesh import CertificationError, analyze_certified, load_certified_head


def digest(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()


def fixture(*, second_x=0):
    matrix_a = np.eye(4)
    matrix_a[0, 3] = -10
    matrix_b = np.array([[2, 0, 0, second_x], [0, 0, -4, 0.5], [0, 3, 0, 0], [0, 0, 0, 1]], dtype=float)
    a = [[9, -1, 0], [11, -1, 0], [10, 1, 0]]
    b = [[0, -1 / 3, 0.125], [0, 1 / 3, 0.125], [0, 0, -0.375]]
    meshes, cert_rows = [], []
    for number, (raw, matrix) in enumerate(((a, matrix_a), (b, matrix_b))):
        path, source_hash = f"/head/renderer_{number}", ("a" if number == 0 else "b") * 64
        mesh = {"mesh_name": "duplicate_name", "renderer_path": path, "source_geometry_sha256": source_hash,
                "renderer_transform_id": 11 + number, "enabled": True, "active_in_hierarchy": True,
                "baked": {"vertices": raw, "triangles": [0, 1, 2],
                          "world_candidates": {"renderer_matrix": {"matrix": matrix.reshape(-1).tolist()}}}}
        meshes.append(mesh)
        cert_rows.append({"mesh_name": mesh["mesh_name"], "renderer_path": path, "source_geometry_sha256": source_hash,
                          "certified": True, "influences_overridden": False, "influences": 4, "declared_influences": 4,
                          "renderer_visibility": {"enabled": True, "active_in_hierarchy": True},
                          "selected_candidate": "renderer_matrix", "matching_candidates": ["renderer_matrix"],
                          "candidate_errors": {"renderer_matrix": {"max_normalized": 0, "max_l2": 0, "vertex_count": 3,
                                                                     "exported_world_consistency": {"max_normalized": 0, "max_l2": 0, "vertex_count": 3}}}})
    snapshot = {"schema_version": 1, "snapshot_kind": "maker_live_skinned_geometry", "frame_count": 3, "frame_count_end": 3,
                "character": {"head_root_transform_id": 10}, "transforms": [
                    {"id": 1, "parent_id": None}, {"id": 10, "parent_id": 1},
                    {"id": 11, "parent_id": 10}, {"id": 12, "parent_id": 10}], "meshes": meshes}
    certificate = {"schema_version": 1, "scope": "snapshot-local LBS versus BakeMesh; no universal engine convention inferred",
                   "snapshot_sha256": digest(snapshot), "snapshot_frame_stable": True, "normalized_tolerance": 1e-5,
                   "meshes": cert_rows}
    return snapshot, certificate


class CrossMeshTests(unittest.TestCase):
    def test_different_raw_spaces_rotation_scale_and_duplicate_names(self):
        snapshot, certificate = fixture()
        a, b = (np.asarray(mesh["baked"]["vertices"]) for mesh in snapshot["meshes"])
        self.assertFalse(np.all(a.min(0) <= b.max(0)) and np.all(b.min(0) <= a.max(0)))
        report = analyze_certified(snapshot, digest(snapshot), certificate)
        self.assertEqual(len(report["meshes"]), 2)
        self.assertEqual(report["cross_mesh"]["counts"]["proper_crossing"], 1)

    def test_exact_file_hash_binding_rejects_stale_certificate(self):
        snapshot, certificate = fixture()
        snapshot["frame_count"] = snapshot["frame_count_end"] = 9
        with self.assertRaisesRegex(CertificationError, "SHA-256"):
            analyze_certified(snapshot, digest(snapshot), certificate)

    def test_source_hash_binding_rejected(self):
        snapshot, certificate = fixture()
        certificate["meshes"][0]["source_geometry_sha256"] = "c" * 64
        with self.assertRaisesRegex(CertificationError, "source hash"):
            analyze_certified(snapshot, digest(snapshot), certificate)

    def test_renderer_path_binding_rejected(self):
        snapshot, certificate = fixture()
        certificate["meshes"][0]["renderer_path"] = "/wrong"
        with self.assertRaisesRegex(CertificationError, "Missing certificate"):
            analyze_certified(snapshot, digest(snapshot), certificate)

    def test_uncertified_and_ambiguous_rows_rejected(self):
        snapshot, certificate = fixture()
        certificate["meshes"][0]["certified"] = False
        with self.assertRaisesRegex(CertificationError, "Uncertified"):
            analyze_certified(snapshot, digest(snapshot), certificate)
        certificate["meshes"][0]["certified"] = True
        certificate["meshes"][0]["selected_candidate"] = None
        certificate["meshes"][0]["matching_candidates"] = ["renderer_matrix", "scale_free_trs"]
        with self.assertRaisesRegex(CertificationError, "ambiguous"):
            analyze_certified(snapshot, digest(snapshot), certificate)

    def test_loose_tolerance_and_influence_override_rejected(self):
        snapshot, certificate = fixture()
        certificate["normalized_tolerance"] = 0.1
        with self.assertRaisesRegex(CertificationError, "overly loose"):
            analyze_certified(snapshot, digest(snapshot), certificate)
        certificate["normalized_tolerance"] = 1e-5
        certificate["meshes"][0]["influences_overridden"] = True
        with self.assertRaisesRegex(CertificationError, "Overridden"):
            analyze_certified(snapshot, digest(snapshot), certificate)

    def test_outside_head_and_inactive_renderers_excluded_before_certification(self):
        snapshot, certificate = fixture()
        outside = copy.deepcopy(snapshot["meshes"][0])
        outside["renderer_path"], outside["renderer_transform_id"] = "/body/same_name", 13
        inactive = copy.deepcopy(snapshot["meshes"][0])
        inactive["renderer_path"], inactive["renderer_transform_id"], inactive["active_in_hierarchy"] = "/head/inactive", 14, False
        snapshot["meshes"].extend([outside, inactive])
        snapshot["transforms"].extend([{"id": 13, "parent_id": 1}, {"id": 14, "parent_id": 10}])
        certificate["snapshot_sha256"] = digest(snapshot)
        meshes, skipped = load_certified_head(snapshot, digest(snapshot), certificate)
        self.assertEqual(len(meshes), 2)
        self.assertEqual({row["reason"] for row in skipped}, {"outside_character_head_subtree", "disabled_or_inactive"})

    def test_new_pair_against_its_separately_certified_baseline(self):
        snapshot, certificate = fixture()
        baseline, base_certificate = fixture(second_x=5)
        report = analyze_certified(snapshot, digest(snapshot), certificate,
                                   baseline, digest(baseline), base_certificate)
        self.assertEqual(report["baseline_cross_mesh"]["crossing_count"], 0)
        self.assertEqual(len(report["baseline_comparison"]["new_crossing_pairs"]), 1)

    def test_cross_renderer_shared_edge_is_contact_not_crossing(self):
        snapshot, certificate = fixture()
        mesh = snapshot["meshes"][1]
        mesh["baked"]["vertices"] = [[-1, -1, 0], [1, -1, 0], [0, -2, 0]]
        mesh["baked"]["world_candidates"]["renderer_matrix"]["matrix"] = np.eye(4).reshape(-1).tolist()
        certificate["snapshot_sha256"] = digest(snapshot)
        report = analyze_certified(snapshot, digest(snapshot), certificate)
        self.assertEqual(report["cross_mesh"]["crossing_count"], 0)
        self.assertEqual(report["cross_mesh"]["counts"]["coplanar_contact"], 1)

    def test_baseline_source_change_stops_indexed_comparison(self):
        snapshot, certificate = fixture()
        baseline, base_certificate = fixture(second_x=5)
        baseline["meshes"][0]["source_geometry_sha256"] = "c" * 64
        base_certificate["meshes"][0]["source_geometry_sha256"] = "c" * 64
        base_certificate["snapshot_sha256"] = digest(baseline)
        report = analyze_certified(snapshot, digest(snapshot), certificate,
                                   baseline, digest(baseline), base_certificate)
        self.assertFalse(report["baseline_comparison"]["comparison_performed"])
        self.assertEqual(report["baseline_comparison"]["status"], "source_or_topology_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
