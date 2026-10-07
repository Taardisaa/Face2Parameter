"""Synthetic fixtures only; no detector initialization, game IO or downloads."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

from core import Camera, ContractError, SurfaceMesh, follow, meshes_from_geometry, observe, validate_reprojection
from calibrate import digest, run


def ortho():
    return np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, -2 / 9, -11 / 9], [0, 0, 0, 1]])


def perspective():
    return np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, -11 / 9, -20 / 9], [0, 0, -1, 0]])


def camera(projection=None, view=None, rect=None, **kwargs):
    return Camera(100, 100, np.eye(4) if view is None else view,
                  ortho() if projection is None else projection,
                  [0, 0, 100, 100] if rect is None else rect,
                  pixel_contract_validated=True, pose_pairing_validated=True, evidence="synthetic", **kwargs)


def mesh(path="head", z=-2, x=0, visible=True, cull="off"):
    return SurfaceMesh(path, np.array([[-1+x, -1, z], [1+x, -1, z], [x, 1, z]]),
                       np.array([[0, 1, 2]]), "synthetic-source-hash", visible=visible,
                       world_policy_certified=True, cull_mode=cull)


def capture_dict():
    return {
        "width": 100, "height": 100, "path": "raw.png", "frame_count": 17,
        "frame_count_before_render": 17, "pose_signature_before_render": "same",
        "pose_signature_after_render": "same", "paired_pose_unchanged": True,
        "paired_geometry": {"pose_signature": "same", "frame_count": 17},
        "capture_camera": {
            "world_to_camera": np.eye(4).reshape(-1).tolist(),
            "camera_to_world": np.eye(4).reshape(-1).tolist(),
            "projection": ortho().reshape(-1).tolist(),
            "matrix_layout": "row_major_16; column_vectors", "cpu_ndc_depth_range": [-1, 1],
            "viewport_origin": "bottom_left", "pixel_rect": [0, 0, 100, 100],
            "image_coordinate_convention": "PNG pixel centers top_left; viewport y must be inverted",
            "image_y_flip_validated": False,
        },
    }


def geometry_dict():
    return {"meshes": [{
        "renderer_path": "head", "source_geometry_sha256": "synthetic-source-hash",
        "enabled": True, "active_in_hierarchy": True, "cull_mode": "off",
        "baked": {
            "triangles": [0, 1, 2], "submeshes": [{"topology": "Triangles", "submesh_index": 4, "indices": [0, 1, 2]}],
            "world_candidates": {"scale_free_trs": {"vertices": mesh().vertices.tolist(), "validated": False}},
        },
    }]}


class ProjectionTests(unittest.TestCase):
    def test_pixel_centers_and_y_axis(self):
        ray = camera().ray([0, 0])
        np.testing.assert_allclose(ray.origin, [-.99, .99, -1])
        np.testing.assert_allclose(ray.direction, [0, 0, -1])
        np.testing.assert_allclose(camera().project([-.99, .99, -2])["xy"], [0, 0], atol=1e-10)
        np.testing.assert_allclose(camera().project([0, 0, -2])["xy"], [49.5, 49.5])

    def test_orthographic_rays_have_distinct_origins_and_parallel_directions(self):
        left, right = camera().ray([20, 49]), camera().ray([70, 49])
        self.assertNotEqual(left.origin[0], right.origin[0])
        np.testing.assert_allclose(left.direction, right.direction)

    def test_perspective_rays_converge_at_camera_and_roundtrip(self):
        c = camera(perspective())
        for xy in ([20, 10], [49.5, 49.5], [75, 88]):
            ray = c.ray(xy)
            np.testing.assert_allclose(np.cross(ray.origin, ray.direction), [0, 0, 0], atol=1e-10)
            world = ray.origin + ray.direction * 2
            np.testing.assert_allclose(c.project(world)["xy"], xy, atol=1e-10)

    def test_offset_viewport_and_roll(self):
        c = camera(rect=[10, 20, 40, 60])
        np.testing.assert_allclose(c.project([0, 0, -2])["xy"], [29.5, 49.5])
        np.testing.assert_allclose(c.ray([29.5, 49.5]).origin, [0, 0, -1], atol=1e-10)
        with self.assertRaises(ContractError): c.ray([80, 50])
        view = np.eye(4)
        view[:2, :2] = [[0, -1], [1, 0]]
        c = camera(view=view)
        np.testing.assert_allclose(c.project([.2, 0, -2])["xy"], [49.5, 39.5])
        np.testing.assert_allclose(c.ray([49.5, 39.5]).origin[:2], [.2, 0], atol=1e-10)

    def test_clip_and_behind_camera(self):
        self.assertEqual(camera().project([0, 0, -.5])["status"], "outside_clip")
        self.assertEqual(camera(perspective()).project([0, 0, 2])["status"], "behind_camera")
        self.assertEqual(camera().project([3, 0, -2])["status"], "outside_viewport")

    def test_capture_contract_is_not_certified_from_text(self):
        data = capture_dict()
        with self.assertRaises(ContractError): Camera.from_capture(data)
        c = Camera.from_capture(data, diagnostic=True)
        self.assertFalse(c.pixel_contract_validated)
        self.assertTrue(c.pose_pairing_validated)
        cert = {"pixel_contract_validated": True, "evidence_id": "synthetic"}
        with self.assertRaises(ContractError): Camera.from_capture(data, certification=cert)
        self.assertFalse(Camera.from_capture(data, certification=cert, diagnostic=True).pixel_contract_validated)
        data["paired_pose_unchanged"] = False
        with self.assertRaises(ContractError): Camera.from_capture(data, certification={**cert, "pose_pairing_validated": True})

    def test_singular_nonfinite_and_bad_inverse(self):
        with self.assertRaises(ContractError): camera(projection=np.zeros((4, 4)))
        with self.assertRaises(ContractError): camera().ray([np.nan, 20])
        data = capture_dict()
        data["capture_camera"]["camera_to_world"] = np.zeros(16).tolist()
        with self.assertRaises(ContractError): Camera.from_capture(data, diagnostic=True)

    def test_missing_frames_do_not_validate_pose_pairing(self):
        data = capture_dict()
        del data["frame_count"]
        del data["frame_count_before_render"]
        del data["paired_geometry"]["frame_count"]
        self.assertFalse(Camera.from_capture(data, diagnostic=True).pose_pairing_validated)


class SurfaceTests(unittest.TestCase):
    def test_hit_triangle_barycentric_and_normal(self):
        result = observe(camera(), [mesh()], {"id": "candidate", "xy": [49.5, 49.5]})
        self.assertEqual(result["status"], "surface_candidate")
        hit = result["surface"]
        np.testing.assert_allclose(hit["barycentric"], [.25, .25, .5])
        np.testing.assert_allclose(hit["world"], [0, 0, -2])
        np.testing.assert_allclose(hit["geometric_normal"], [0, 0, 1])
        self.assertEqual(hit["triangle_id"], 0)
        self.assertFalse(result["anatomical_correspondence_validated"])

    def test_occlusion_with_target_mesh_and_disabled_occluder(self):
        point = {"id": "head-point", "xy": [49.5, 49.5], "allowed_renderer_paths": ["head"]}
        result = observe(camera(), [mesh("eye", -1.5), mesh()], point)
        self.assertEqual(result["status"], "occluded")
        self.assertEqual(result["surface"]["renderer_path"], "head")
        self.assertEqual(result["blocker"]["renderer_path"], "eye")
        result = observe(camera(), [mesh("eye", -1.5, visible=False), mesh()], point)
        self.assertEqual(result["status"], "surface_candidate")

    def test_backface_and_clip_interval(self):
        back = mesh(cull="back")
        back.triangles = np.array([[0, 2, 1]])
        self.assertEqual(observe(camera(), [back], {"id": "x", "xy": [49.5, 49.5]})["status"], "no_target_intersection")
        far = mesh(z=-20)
        self.assertEqual(observe(camera(), [far], {"id": "x", "xy": [49.5, 49.5]})["status"], "no_target_intersection")

    def test_silhouette_never_binds_triangle(self):
        result = observe(camera(), [mesh()], {"id": "outline", "xy": [49.5, 49.5], "kind": "silhouette"})
        self.assertEqual(result["status"], "view_observation_only")
        self.assertNotIn("surface", result)
        self.assertNotIn("barycentric", result)

    def test_coincident_surfaces_are_ambiguous(self):
        result = observe(camera(), [mesh(), mesh("overlap")], {"id": "x", "xy": [49.5, 49.5]})
        self.assertTrue(result["intersection_ambiguous"])
        self.assertEqual(result["equal_distance_hit_count"], 2)

    def test_followed_inactive_mesh_is_not_a_visible_observation(self):
        surface = observe(camera(), [mesh()], {"id": "x", "xy": [49.5, 49.5]})["surface"]
        result = validate_reprojection(surface, camera(), [mesh(visible=False)], observed_xy=[49.5, 49.5])
        self.assertEqual(result["status"], "target_renderer_not_visible")
        self.assertIsNone(result["error_px"])

    def test_heldout_reprojection_and_deformation_follow(self):
        original = mesh()
        surface = observe(camera(), [original], {"id": "x", "xy": [49.5, 49.5]})["surface"]
        view = np.eye(4)
        view[0, 3] = -.2
        heldout = camera(view=view, view_id="heldout")
        result = validate_reprojection(surface, heldout, [original], observed_xy=[39.5, 49.5])
        self.assertEqual(result["status"], "within_pixel_tolerance")
        self.assertLess(result["error_px"], 1e-10)
        self.assertFalse(result["anatomical_correspondence_validated"])
        deformed = mesh(x=.1)
        np.testing.assert_allclose(follow(surface, [deformed])["world"], [.1, 0, -2])
        self.assertEqual(validate_reprojection(surface, heldout, [deformed], observed_xy=[44.5, 49.5])["status"], "within_pixel_tolerance")
        self.assertEqual(validate_reprojection(surface, heldout, [deformed], observed_xy=[10, 10])["status"], "outside_pixel_tolerance")
        self.assertEqual(validate_reprojection(surface, heldout, [deformed])["status"], "projected_without_independent_observation")

    def test_heldout_occlusion_does_not_invent_error(self):
        surface = observe(camera(), [mesh()], {"id": "x", "xy": [49.5, 49.5]})["surface"]
        result = validate_reprojection(surface, camera(), [mesh(), mesh("blocker", -1.5)], observed_xy=[0, 0])
        self.assertEqual(result["status"], "occluded")
        self.assertIsNone(result["error_px"])

    def test_follow_rejects_identity_and_topology_changes(self):
        surface = observe(camera(), [mesh()], {"id": "x", "xy": [49.5, 49.5]})["surface"]
        changed = mesh()
        changed.source_hash = "different-head"
        with self.assertRaises(ContractError): follow(surface, [changed])
        changed = mesh()
        changed.triangles = np.array([[0, 2, 1]])
        with self.assertRaises(ContractError): follow(surface, [changed])

    def test_world_candidate_requires_explicit_evidence(self):
        data = geometry_dict()
        with self.assertRaises(ContractError): meshes_from_geometry(data)
        result = meshes_from_geometry(data, diagnostic=True)
        self.assertFalse(result[0].world_policy_certified)
        self.assertEqual(result[0].submesh_ids.tolist(), [4])
        cert = {"world_policy_validated": True, "world_candidate": "scale_free_trs", "mesh_source_hashes": {"head": "synthetic-source-hash"}}
        self.assertTrue(meshes_from_geometry(data, certification=cert)[0].world_policy_certified)
        cert["mesh_source_hashes"]["head"] = "wrong"
        with self.assertRaises(ContractError): meshes_from_geometry(data, certification=cert)

    def test_cli_json_parsing_and_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / name for name in ("capture.json", "geometry.json", "points.json")]
            payloads = [capture_dict(), geometry_dict(), {"coordinate_convention": "top_left_pixel_centers_integer", "points": [{"id": "candidate", "xy": [49.5, 49.5]}]}]
            for path, payload in zip(paths, payloads): path.write_text(json.dumps(payload), encoding="utf-8")
            args = SimpleNamespace(capture=paths[0], geometry=paths[1], points=paths[2], out=root / "out.json", certification=None, view_index=0, world_candidate="scale_free_trs", diagnostic_unvalidated=True)
            output = run(args)
            self.assertFalse(output["pixel_contract_validated"])
            self.assertFalse(output["anatomical_correspondence_validated"])
            cert = {"evidence_id": "synthetic-attestation", "capture_file_sha256": digest(paths[0]), "geometry_file_sha256": digest(paths[1]), "pixel_contract_validated": True, "world_policy_validated": True, "world_candidate": "scale_free_trs", "mesh_source_hashes": {"head": "synthetic-source-hash"}}
            args.certification = root / "cert.json"
            args.certification.write_text(json.dumps(cert), encoding="utf-8")
            args.diagnostic_unvalidated = False
            with self.assertRaises(ContractError): run(args)
            args.diagnostic_unvalidated = True
            output = run(args)
            self.assertTrue(output["world_policy_certified"])
            self.assertFalse(output["pixel_contract_validated"])
            cert["geometry_file_sha256"] = "wrong"
            args.certification.write_text(json.dumps(cert), encoding="utf-8")
            with self.assertRaises(ContractError): run(args)


if __name__ == "__main__":
    unittest.main()
