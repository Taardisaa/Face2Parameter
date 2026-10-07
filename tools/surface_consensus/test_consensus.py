"""Exact synthetic adversarial contracts; no detector, game or HTTP."""
import copy
import unittest

import numpy as np

from tools.surface_calibration.core import Camera, ContractError, Ray, SurfaceMesh, follow, observe
from .consensus import Policy, definitions, evaluate_candidate, fit, material, triangle_minima, triangulate
from .geometric import geometric_definitions


def scene():
    mesh = SurfaceMesh("head", np.array([[-.8, -.8, 0], [.8, -.8, 0], [0, .8, 0]]),
                       np.array([[0, 1, 2]]), "fixture_source", cull_mode="off", world_policy_certified=True)
    cameras = []
    for angle in [-20, 0, 20]:
        radians = np.radians(angle)
        c, s = np.cos(radians), np.sin(radians)
        view = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
        cameras.append(Camera(512, 512, view, np.eye(4), np.array([0, 0, 512, 512]), view_id=str(angle),
                              pixel_contract_validated=True, pose_pairing_validated=True))
    observations = []
    for i, camera in enumerate(cameras):
        xy = camera.project([0, 0, 0])["xy"]
        observations.append({"view_index": i, "yaw": [-20, 0, 20][i], "raw_png_sha256": f"fixture_{i}",
                             "landmark68": [xy[:] for _ in range(68)],
                             "heatmap_grid_decode": {"raw_interpolation_footprint_fully_in_image": [True] * 68}})
    return [mesh], cameras, observations


class ConsensusContracts(unittest.TestCase):
    def test_parallel_and_antiparallel_rejected(self):
        for direction in ([0, 0, 1], [0, 0, -1]):
            with self.assertRaisesRegex(ContractError, "parallel"):
                triangulate([Ray(np.array([0, 0, -1]), np.array([0, 0, 1]), 5),
                             Ray(np.array([1, 0, -1]), np.array(direction), 5)])

    def test_ray_near_far_is_enforced(self):
        with self.assertRaisesRegex(ContractError, "near/far"):
            triangulate([Ray(np.array([-1, 0, 0]), np.array([1, 0, 0]), .5),
                         Ray(np.array([0, -1, 0]), np.array([0, 1, 0]), .5)])

    def test_outside_camera_viewport_rejected(self):
        _, cameras, _ = scene()
        with self.assertRaisesRegex(ContractError, "viewport"):
            cameras[0].ray([-5, 100])

    def test_triangle_interior_and_edge_exact_simplex(self):
        meshes, _, _ = scene()
        for target in ([0, 0, 0], [2, 0, 0]):
            _, weights, worlds, _ = triangle_minima(meshes[0], np.eye(3), np.array(target))
            self.assertAlmostEqual(weights[0].sum(), 1)
            self.assertGreaterEqual(weights.min(), -1e-10)
            if target[0] == 0:
                np.testing.assert_allclose(worlds[0], [0, 0, 0], atol=1e-12)
            else:
                self.assertAlmostEqual(weights[0].min(), 0)

    def test_unoccluded_independent_views_pass_numeric_only(self):
        meshes, cameras, observations = scene()
        result = evaluate_candidate(definitions()[0], cameras, observations, meshes, ["head"])
        self.assertTrue(result["accepted"])
        self.assertFalse(result["anatomical_correspondence_validated"])
        self.assertEqual(len(result["leave_one_view_out"]), 3)
        for holdout in result["leave_one_view_out"]:
            self.assertNotIn(holdout["heldout_view_index"], holdout["training_view_indices"])
            self.assertLess(holdout["heldout"]["error_px"], 1e-8)

    def test_occluded_observations_cannot_participate(self):
        meshes, cameras, observations = scene()
        blocker = SurfaceMesh("blocker", meshes[0].vertices + [0, 0, -.3], meshes[0].triangles.copy(),
                              "blocker_source", cull_mode="off", world_policy_certified=True)
        screened = observe(cameras[1], meshes + [blocker], {"id": "nose", "xy": observations[1]["landmark68"][30],
                                                         "allowed_renderer_paths": ["head"]})
        self.assertEqual(screened["status"], "occluded")
        result = evaluate_candidate(definitions()[0], cameras, observations, meshes + [blocker], ["head"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["eligible_view_indices"], [])

    def test_shifted_landmark_independently_rejected(self):
        meshes, cameras, observations = scene()
        observations[2]["landmark68"][30][0] += 15
        result = evaluate_candidate(definitions()[0], cameras, observations, meshes, ["head"])
        self.assertFalse(result["accepted"])
        holdout = next(r for r in result["leave_one_view_out"] if r["heldout_view_index"] == 2)
        self.assertFalse(holdout["accepted"])
        self.assertGreater(holdout["heldout"]["error_px"], 14.9)
        np.testing.assert_allclose(holdout["fitted_surface"]["world"], [0, 0, 0], atol=1e-10)

    def test_heldout_not_used_to_pick_fit(self):
        meshes, cameras, observations = scene()
        xy = [o["landmark68"][30] for o in observations]
        before = fit(cameras[:2], xy[:2], meshes, ["head"])
        xy[2] = [300, 320]
        after = fit(cameras[:2], xy[:2], meshes, ["head"])
        self.assertEqual(before["surface"], after["surface"])

    def test_model_source_change_requires_recalibration(self):
        meshes, _, _ = scene()
        anchor = material(meshes[0], 0, np.array([.25, .25, .5]))
        changed = copy.deepcopy(meshes)
        changed[0].source_hash = "another_head_or_source_mesh"
        with self.assertRaisesRegex(ContractError, "identity changed"):
            follow(anchor, changed)

    def test_ordered_topology_change_requires_recalibration(self):
        meshes, _, _ = scene()
        anchor = material(meshes[0], 0, np.array([.25, .25, .5]))
        changed = copy.deepcopy(meshes)
        changed[0].triangles[0] = [0, 2, 1]
        with self.assertRaisesRegex(ContractError, "topology/order changed"):
            follow(anchor, changed)

    def test_profile_predictions_excluded_even_when_projectable(self):
        meshes, cameras, observations = scene()
        observations[2]["yaw"] = 60
        result = evaluate_candidate(definitions()[0], cameras, observations, meshes, ["head"])
        self.assertFalse(result["accepted"])
        self.assertNotIn(2, result["eligible_view_indices"])
        self.assertEqual(result["observations"][2]["status"], "excluded_profile_or_hidden_side_risk")

    def test_jaw_is_never_fixed_material_point(self):
        meshes, cameras, observations = scene()
        result = evaluate_candidate(definitions()[7], cameras, observations, meshes, ["head"])
        self.assertFalse(result["accepted"])
        self.assertNotIn("all_view_fit", result)
        self.assertFalse(result["fixed_material_point_allowed"])

    def test_literal_geometric_support_is_not_semantic_anatomy(self):
        meshes, cameras, _ = scene()
        meshes[0].submesh_ids[:] = 0
        meshes[0].vertices[2, 2] = -.2
        result = geometric_definitions(meshes[0], cameras[1], cameras, meshes)
        support = result["support_definition"]
        self.assertEqual(support["maximizing_vertex_ids"], [2])
        self.assertFalse(support["anatomical_correspondence_validated"])
        self.assertFalse(support["is_named_nose_apex"])
        self.assertFalse(support["independent_image_semantic_observation_present"])
        self.assertEqual(result["indexed_topology_boundary_inventory"]["boundary_edge_count"], 3)

    def test_fixed_support_follows_material_without_reselecting_maximum(self):
        meshes, cameras, _ = scene()
        meshes[0].submesh_ids[:] = 0
        meshes[0].vertices[2, 2] = -.2
        original = geometric_definitions(meshes[0], cameras[1], cameras, meshes)["support_definition"]["material"]
        changed = copy.deepcopy(meshes)
        changed[0].vertices[0, 2] = -.8  # New maximum, deliberately a different vertex.
        changed[0].vertices[2, 2] = .1
        followed = follow(original, changed)
        np.testing.assert_array_equal(followed["world"], changed[0].vertices[2])
        self.assertEqual(followed["barycentric"], original["barycentric"])
        reselected = geometric_definitions(changed[0], cameras[1], cameras, changed)["support_definition"]
        self.assertEqual(reselected["maximizing_vertex_ids"], [0])
        self.assertNotEqual(followed["world"], reselected["material"]["world"])

    def test_transport_rejects_changed_source_before_reconstruction(self):
        from .transport import geometry_transport
        anchor = {"renderer_path": "head", "source_geometry_sha256": "fixed_source"}
        snapshot = {"character": {"head_id": 2}, "meshes": [{"mesh_name": "o_head", "renderer_path": "head", "source_geometry_sha256": "changed_source"}]}
        with self.assertRaisesRegex(ContractError, "identity changed"):
            geometry_transport(snapshot, anchor)


if __name__ == "__main__":
    unittest.main()
