"""Analytical atlas tests; no cached assets, game, training or downloads needed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from core import (
    analyze_jacobian,
    finite_difference_jacobian,
    summarize_displacement,
    summarize_verified_regions,
    validate_regions,
)
from inspect_atlas import exterior_effects


class AtlasDiagnostics(unittest.TestCase):
    def setUp(self):
        self.baseline = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_uniform_translation_is_measured_without_shape_alignment(self):
        stats, delta, mask = summarize_displacement(
            self.baseline, self.baseline + [0.1, 0.2, 0.3]
        )
        np.testing.assert_allclose(
            stats["bbox"]["extent_change"], [0, 0, 0], atol=1e-15
        )
        np.testing.assert_allclose(stats["centroid_shift"], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(stats["displacement"]["rms"], np.sqrt(0.14))
        self.assertAlmostEqual(
            stats["normalized_displacement"]["max"], np.sqrt(0.14 / 3)
        )
        self.assertEqual(4, stats["affected_vertex_count"])
        self.assertTrue(mask.all())
        np.testing.assert_allclose(delta, [[0.1, 0.2, 0.3]] * 4)

    def test_spatial_effects_preserve_vertex_indices_without_semantic_guess(self):
        candidate = self.baseline.copy()
        candidate[2, 0] += 0.4
        stats, _, mask = summarize_displacement(self.baseline, candidate)
        self.assertEqual([False, False, True, False], mask.tolist())
        self.assertEqual(2, stats["top_vertices"][0]["vertex_index"])
        self.assertEqual(
            "none unless supplied by source-matched verified vertex masks",
            stats["semantic_regions"],
        )
        stats, _, mask = summarize_displacement(self.baseline, self.baseline)
        self.assertEqual(0, stats["affected_vertex_count"])

    def test_linear_jacobian_coupling_and_rank(self):
        response = np.zeros((3, 4, 3))
        response[0, :, 0] = 1
        response[2, :, 0] = 2
        minus = self.baseline[None] - response * 0.001
        plus = self.baseline[None] + response * 0.001
        center, left, right = finite_difference_jacobian(
            self.baseline, minus, plus, 0.001
        )
        np.testing.assert_allclose(center, response.reshape(3, -1).T, atol=1e-12)
        np.testing.assert_allclose(left, right, atol=1e-12)
        report, coupling, directions = analyze_jacobian(center)
        self.assertEqual(1, report["rank"])
        self.assertEqual([1], report["zero_columns_at_this_baseline_and_mesh_set"])
        self.assertAlmostEqual(coupling[0, 2], 1)
        self.assertEqual((3, 3), directions.shape)
        self.assertIn("not a global", report["scope"])

    def test_keyframe_knot_averages_slopes_but_keeps_both(self):
        minus = self.baseline[None].copy()
        minus[0, :, 0] -= 0.001
        plus = self.baseline[None].copy()
        plus[0, :, 0] += 0.003
        center, left, right = finite_difference_jacobian(
            self.baseline, minus, plus, 0.001
        )
        np.testing.assert_allclose(center[::3, 0], 2, atol=1e-12)
        np.testing.assert_allclose(left[::3, 0], 1, atol=1e-12)
        np.testing.assert_allclose(right[::3, 0], 3, atol=1e-12)

    def test_semantics_require_matching_source_and_verified_mask(self):
        spec = {
            "source_file_sha256": "cache-hash",
            "provenance": {
                "verified": True,
                "source": "reviewed correspondence",
                "kind": "verified_vertex_mask",
            },
            "regions": {"reviewed_label": [1, 2]},
        }
        regions = validate_regions(
            spec, source_file_sha256="cache-hash", vertex_count=4
        )
        stats, delta, mask = summarize_displacement(
            self.baseline, self.baseline + [0.1, 0, 0]
        )
        values = summarize_verified_regions(delta, mask, regions)
        self.assertAlmostEqual(values["reviewed_label"]["max_displacement"], 0.1)
        with self.assertRaises(ValueError):
            validate_regions(spec, source_file_sha256="other-head", vertex_count=4)
        spec["provenance"]["verified"] = False
        with self.assertRaises(ValueError):
            validate_regions(spec, source_file_sha256="cache-hash", vertex_count=4)
        spec["provenance"]["verified"] = True
        spec["regions"] = {"label": [4]}
        with self.assertRaises(ValueError):
            validate_regions(spec, source_file_sha256="cache-hash", vertex_count=4)

    def test_invalid_dimensions_and_nonfinite_values_rejected(self):
        with self.assertRaises(ValueError):
            summarize_displacement(self.baseline, self.baseline[:3])
        with self.assertRaises(ValueError):
            summarize_displacement(self.baseline, self.baseline * float("nan"))
        with self.assertRaises(ValueError):
            finite_difference_jacobian(
                self.baseline, self.baseline[None], self.baseline[None], 0
            )

    def test_external_probe_compares_to_endpoint_not_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "effects.npz"
            # Each outer point differs strongly fromneutral but equals its bounded endpoint.
            displacements = np.array(
                [[-1.0, 0, 0], [-1.0, 0, 0], [0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]]
            )[:, None]
            np.savez_compressed(path, o_head__displacement=displacements)
            atlas = {
                "head_id": 2,
                "sampling_profile": "vanilla",
                "levels": [-0.25, 0, 0.5, 1, 1.25],
                "mesh_layout": {"o_head": {}},
                "controls": [
                    {
                        "index": 0,
                        "vertex_effects_path": str(path),
                        "samples": [
                            {
                                "meshes": {
                                    "o_head": {
                                        "baseline_bbox_diagonal": 1,
                                        "effect_threshold_absolute": 1e-6,
                                    }
                                }
                            }
                            for _ in range(5)
                        ],
                    }
                ],
            }
            row = exterior_effects(atlas)["controls"][0]
            self.assertFalse(row["edges"]["below_zero"]["any_mesh_extends"])
            self.assertFalse(row["edges"]["above_one"]["any_mesh_extends"])
            displacements[0, 0, 0] = -1.2
            np.savez_compressed(path, o_head__displacement=displacements)
            row = exterior_effects(atlas)["controls"][0]
            self.assertTrue(row["edges"]["below_zero"]["any_mesh_extends"])


if __name__ == "__main__":
    unittest.main()
