"""Analytical synthetic tests without Unity, game calls or real asset dependencies."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from bone_locals import compare_bone_locals
from geometry import (
    Uncertifiable,
    analyze_snapshot,
    blendshape_delta,
    recorded_uniform_renderer_scale,
    rigid_alignment,
    skin_world,
    transform_points,
)


def affine(scale=(1, 1, 1), translation=(0, 0, 0), rotation=None):
    result = np.eye(4)
    result[:3, :3] = (np.eye(3) if rotation is None else rotation) @ np.diag(scale)
    result[:3, 3] = translation
    return result


def snapshot_fixture(*, convention="renderer_matrix", unit_scale=False):
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    renderer = affine((1, 1, 1) if unit_scale else (2, 3, 4), (5, -1, 2), rotation)
    no_scale = affine(translation=(5, -1, 2), rotation=rotation)
    bones = [affine((2, 3, 4), (1, 0, 0)), affine(translation=(0, 2, 0))]
    transforms = {
        i + 1: {"id": i + 1, "local_to_world": mat.flatten().tolist()}
        for i, mat in enumerate(bones)
    }
    mesh = {
        "mesh_name": "o_head",
        "skin_quality": "Bone4",
        "renderer_path": "synthetic/head",
        "bone_transform_ids": [1, 2],
        "blendshapes": [],
        "source": {
            "vertices": vertices.tolist(),
            "triangles": [0, 1, 2, 0, 2, 3],
            "bone_indices": [[0, 1, 0, 0]] * 4,
            "bone_weights": [[0.25, 0.75, 0, 0]] * 4,
            "bindposes": [np.eye(4).flatten().tolist()] * 2,
        },
    }
    world, _ = skin_world(mesh, transforms)
    conversion = renderer if convention == "renderer_matrix" else no_scale
    raw = transform_points(world, np.linalg.inv(conversion))
    mesh["baked"] = {
        "vertices": raw.tolist(),
        "triangles": mesh["source"]["triangles"],
        "world_candidates": {
            name: {
                "matrix": mat.flatten().tolist(),
                "vertices": transform_points(raw, mat).tolist(),
            }
            for name, mat in [
                ("renderer_matrix", renderer),
                ("scale_free_trs", no_scale),
            ]
        },
    }
    return {
        "schema_version": 1,
        "frame_count": 100,
        "frame_count_end": 100,
        "transforms": list(transforms.values()),
        "meshes": [mesh],
    }


class GeometryParity(unittest.TestCase):
    def test_analytic_two_bone_nonuniform_scale_mixed_weights(self):
        snapshot = snapshot_fixture()
        mesh = snapshot["meshes"][0]
        transforms = {t["id"]: t for t in snapshot["transforms"]}
        world, _ = skin_world(mesh, transforms)
        expected = np.array(
            [[0.25, 1.5, 0], [1.5, 1.5, 0], [0.25, 3, 0], [0.25, 1.5, 1.75]]
        )
        np.testing.assert_allclose(world, expected, atol=1e-14)

    def test_bindpose_is_not_omitted(self):
        snapshot = snapshot_fixture()
        mesh = snapshot["meshes"][0]
        mesh["source"]["bindposes"][0] = (
            affine(translation=(-1, 0, 0)).flatten().tolist()
        )
        world, _ = skin_world(mesh, {t["id"]: t for t in snapshot["transforms"]})
        np.testing.assert_allclose(world[0], [-0.25, 1.5, 0])

    def test_each_bake_convention_is_identified_and_identity_is_ambiguous(self):
        for convention in ["renderer_matrix", "scale_free_trs"]:
            row = analyze_snapshot(snapshot_fixture(convention=convention))["meshes"][0]
            self.assertTrue(row["certified"])
            self.assertTrue(row["convention_distinguished"])
            self.assertEqual(convention, row["selected_candidate"])
            self.assertLess(row["candidate_errors"][convention]["max_l2"], 1e-12)
        row = analyze_snapshot(snapshot_fixture(unit_scale=True))["meshes"][0]
        self.assertTrue(row["certified"])
        self.assertFalse(row["convention_distinguished"])
        self.assertIsNone(row["selected_candidate"])

    def test_active_weight_without_deltas_cannot_certify(self):
        snapshot = snapshot_fixture()
        snapshot["meshes"][0]["blendshapes"] = [
            {"name": "blink", "current_weight": 25, "frames": [{"weight": 100}]}
        ]
        row = analyze_snapshot(snapshot)["meshes"][0]
        self.assertFalse(row["certified"])
        self.assertIn("lacks", row["uncertifiable_reason"])
        snapshot["meshes"][0]["blendshapes"][0]["current_weight"] = 0
        self.assertTrue(analyze_snapshot(snapshot)["meshes"][0]["certified"])

    def test_single_multiframe_and_multiple_active_blendshapes_before_skinning(self):
        snapshot = snapshot_fixture()
        mesh = snapshot["meshes"][0]
        mesh["blendshapes"] = [
            {
                "name": "single",
                "current_weight": 25,
                "frames": [{"weight": 100, "delta_vertices": [[4, 0, 0]] * 4}],
            },
            {
                "name": "multi",
                "current_weight": 75,
                "frames": [
                    {"weight": 50, "delta_vertices": [[0, 2, 0]] * 4},
                    {"weight": 100, "delta_vertices": [[0, 6, 0]] * 4},
                ],
            },
        ]
        delta, active = blendshape_delta(mesh)
        np.testing.assert_allclose(delta, [[1, 4, 0]] * 4)
        self.assertEqual(2, len(active))
        transforms = {t["id"]: t for t in snapshot["transforms"]}
        world, _ = skin_world(mesh, transforms)
        np.testing.assert_allclose(world[0], [1.5, 7.5, 0])
        mesh["blendshapes"][0]["current_weight"] = -25
        np.testing.assert_allclose(blendshape_delta(mesh)[0], [[-1, 4, 0]] * 4)

    def test_quality_hypothesis_and_zero_weight_missing_bone(self):
        snapshot = snapshot_fixture()
        mesh = snapshot["meshes"][0]
        mesh["source"]["bone_indices"] = [[0, 1, 999, -1]] * 4
        transforms = {t["id"]: t for t in snapshot["transforms"]}
        world, _ = skin_world(mesh, transforms, influences=1)
        np.testing.assert_allclose(world[0], [0, 2, 0])
        mesh["source"]["bone_weights"][0] = [0.25, 0.5, 0.25, 0]
        with self.assertRaises(Uncertifiable):
            skin_world(mesh, transforms)

    def test_bad_weights_nonfinite_and_unstable_frame_uncertified(self):
        snapshot = snapshot_fixture()
        snapshot["frame_count_end"] = 101
        self.assertFalse(analyze_snapshot(snapshot)["meshes"][0]["certified"])
        snapshot = snapshot_fixture()
        snapshot["meshes"][0]["source"]["bone_weights"][0][0] = 0.1
        self.assertFalse(analyze_snapshot(snapshot)["meshes"][0]["certified"])
        snapshot = snapshot_fixture()
        snapshot["meshes"][0]["source"]["vertices"][0][0] = float("nan")
        self.assertFalse(analyze_snapshot(snapshot)["meshes"][0]["certified"])

    def test_rigid_alignment_recovers_pose_but_not_anisotropic_shape(self):
        source = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
        target = source @ rotation + [1, 2, 3]
        aligned, result = rigid_alignment(source, target)
        np.testing.assert_allclose(aligned, target, atol=1e-12)
        self.assertAlmostEqual(result["rotation_determinant"], 1)
        distorted = source * [2, 1, 1]
        _, result = rigid_alignment(source, distorted)
        self.assertGreater(result["rigid_errors"]["max_l2"], 0.1)
        self.assertEqual(1, result["unit_scale_applied"])
        self.assertNotEqual(1, result["suggested_additional_uniform_scale_NOT_applied"])

    def test_unit_scale_is_explicit_and_reflection_never_fitted(self):
        source = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        target = source * 1000 + [1, 2, 3]
        _, result = rigid_alignment(source, target, unit_scale=1000)
        self.assertLess(result["rigid_errors"]["max_l2"], 1e-10)
        self.assertEqual(1000, result["unit_scale_applied"])
        _, reflected = rigid_alignment(source, source * [-1, 1, 1])
        self.assertAlmostEqual(reflected["rotation_determinant"], 1)
        self.assertGreater(reflected["rigid_errors"]["max_l2"], 0.1)

    def test_recorded_scale_uses_matrix_evidence_and_rejects_shear(self):
        mesh = {
            "renderer_local_to_world": affine((0.9, 0.9, 0.9)).flatten().tolist(),
            "renderer_lossy_scale": [0.9, 0.9, 0.9],
            "renderer_transform_id": 1,
        }
        transforms = {
            1: {
                "id": 1,
                "name": "height",
                "parent_id": None,
                "local_scale": [0.9, 0.9, 0.9],
                "lossy_scale": [0.9, 0.9, 0.9],
            }
        }
        scale, evidence = recorded_uniform_renderer_scale(mesh, transforms)
        self.assertAlmostEqual(scale, 0.9)
        self.assertEqual("height", evidence["ancestor_scale_contributors"][0]["name"])
        for bad in [affine((0.9, 1, 0.9)), affine((-0.9, 0.9, 0.9))]:
            mesh["renderer_local_to_world"] = bad.flatten().tolist()
            with self.assertRaises(Uncertifiable):
                recorded_uniform_renderer_scale(mesh, transforms)
        bad = affine((0.9, 0.9, 0.9))
        bad[0, 1] = 0.1
        mesh["renderer_local_to_world"] = bad.flatten().tolist()
        with self.assertRaises(Uncertifiable):
            recorded_uniform_renderer_scale(mesh, transforms)

    def test_bone_local_comparison_preserves_control_differences(self):
        rig = SimpleNamespace(
            enums={"src": [], "dst": []},
            customhead=[],
            eqns=[],
            bones={
                "root": {
                    "name": "bone",
                    "parent": None,
                    "pos": [1, 2, 3],
                    "rot": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                }
            },
            _topo=["root"],
            skin_bone_names=["bone"],
        )
        live = {
            "id": 1,
            "name": "bone",
            "parent_id": None,
            "local_position": [1.1, 2, 3],
            "local_rotation_xyzw": [0, 0, 0, 1],
            "local_scale": [1, 1.2, 1],
        }
        snapshot = {"transforms": [live], "character": {"shape_value_face": None}}
        result = compare_bone_locals(snapshot, rig)
        self.assertAlmostEqual(result["max_head_skin_position_error"], 0.1)
        self.assertAlmostEqual(result["max_head_skin_scale_error"], 0.2)


if __name__ == "__main__":
    unittest.main()
