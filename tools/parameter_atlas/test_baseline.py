"""Baseline input, boundary response and all-control preservation tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from baseline import load_baseline, native_values, probe_inputs, probe_jacobian
from view_displacement import viewer_data


class NativeBaselineTests(unittest.TestCase):
    def test_default_and_source_provenance(self):
        default, provenance = load_baseline()
        np.testing.assert_array_equal(default, np.full(59, 0.5))
        values = np.linspace(-0.1, 1.1, 59).tolist()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text(
                json.dumps({"head_id": 2, "native_input": values}), encoding="utf-8"
            )
            actual, source = load_baseline(path)
            np.testing.assert_array_equal(actual, values)
            self.assertEqual(2, source["head_id"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                source["source_file_sha256"],
            )
            path.write_text(json.dumps(values, indent=2), encoding="utf-8")
            _, formatted = load_baseline(path)
            self.assertEqual(
                source["native_input_sha256"], formatted["native_input_sha256"]
            )
            self.assertNotEqual(
                source["source_file_sha256"], formatted["source_file_sha256"]
            )
            self.assertNotEqual(
                provenance["native_input_sha256"], source["native_input_sha256"]
            )

    def test_geometry_snapshot_consumes_native_values_only(self):
        document = {
            "character": {"head_id": 1, "shape_value_face": [0.3] * 59},
            "abmx": ["not consumed"],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            values, source = load_baseline(path)
            np.testing.assert_array_equal(values, [0.3] * 59)
            self.assertEqual("character.shape_value_face", source["source_field"])
            self.assertIn("excludes ABMX", source["consumed_state"])

    def test_partial_invalid_or_ambiguous_values_fail(self):
        for values in [
            [0.5] * 54,
            [0.5] * 60,
            [[0.5]] * 59,
            [True] * 59,
            ["0.5"] * 59,
            [float("nan")] * 59,
            [float("inf")] * 59,
        ]:
            with self.subTest(values=str(values)[:30]), self.assertRaises(ValueError):
                native_values(values)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for document in [
                {"native_input": [0.5] * 59, "head_id": True},
                {"native_input": [0.5] * 59, "head_id": -1},
                {"shape_values": [0.5] * 59},
            ]:
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_baseline(path)

    def test_every_probe_keeps_other_58_controls_and_unclipped_boundaries(self):
        baseline = np.linspace(-0.25, 1.25, 59)
        probes, left, right = probe_inputs(baseline, 0.02)
        for control in range(59):
            for row, delta in [(control, -0.02), (59 + control, 0.02)]:
                expected = baseline.copy()
                expected[control] += delta
                np.testing.assert_array_equal(probes[row], expected)
        self.assertLess(probes[0, 0], -0.25)
        self.assertGreater(probes[-1, -1], 1.25)
        np.testing.assert_allclose(left, 0.02)
        np.testing.assert_allclose(right, 0.02)
        with self.assertRaises(ValueError):
            probe_inputs(np.full(59, 1e20), 0.001)

    def test_boundary_keeps_zero_outward_and_nonzero_inward_surface_slope(self):
        baseline = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        # Profile clamps surface response below zero; inputs themselves are still -0.02.
        minus = baseline[None].copy()
        plus = baseline[None] + [0.02, 0, 0]
        center, left, right = probe_jacobian(baseline, minus, plus, [0.02], [0.02])
        np.testing.assert_allclose(left, 0)
        np.testing.assert_allclose(right[::3, 0], 1)
        np.testing.assert_allclose(center[::3, 0], 0.5)

    def test_actual_unequal_intervals_are_used(self):
        baseline = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        center, left, right = probe_jacobian(
            baseline,
            baseline[None] - [0.01, 0, 0],
            baseline[None] + [0.03, 0, 0],
            [0.01],
            [0.03],
        )
        for matrix in [center, left, right]:
            np.testing.assert_allclose(matrix[::3, 0], 1)

    def test_all_59_generation_archives_keep_arbitrary_baseline(self):
        import generate

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in [
                "o_head_mesh.npz",
                "skeleton.json",
                "anmShapeHead.json",
                "customhead.json",
                "enums.json",
                "update_eqns.json",
            ]:
                (root / name).write_text("synthetic source", encoding="utf-8")
            rig = SimpleNamespace(data_dir=root, root_dir=root, customhead=[])
            baseline = np.linspace(0.1, 0.9, 59)
            _, provenance = load_baseline()
            provenance["kind"] = "synthetic_test_baseline"
            args = SimpleNamespace(
                out=root / "out",
                device="cpu",
                levels=[0.0, 0.5, 1.0],
                baseline_input=baseline,
                baseline_provenance=provenance,
                batch_size=8,
                effect_threshold=1e-6,
                step=1e-3,
            )
            base_vertices = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )

            def evaluate(trig, values, subs, *, batch_size):
                vertices = np.repeat(base_vertices[None], len(values), axis=0)
                vertices[:, :, 0] += (values**2).sum(axis=1)[:, None]
                return {"o_head": vertices}

            with (
                patch.object(generate, "HeadRig", return_value=rig),
                patch.object(
                    generate, "TorchHeadRig", return_value=SimpleNamespace(n_slider=59)
                ),
                patch.object(generate, "render_geometry", side_effect=evaluate),
            ):
                result = generate.generate_head(2, "vanilla", args, {})
            self.assertEqual(59, result["controls"])
            atlas = json.loads(Path(result["atlas_path"]).read_text(encoding="utf-8"))
            self.assertEqual(59, len(atlas["controls"]))
            view = viewer_data(result["atlas_path"], 58)
            self.assertEqual(4, len(view["baseline"]))
            self.assertEqual(3, len(view["deltas"]))
            expected_candidates = np.asarray(view["baseline"])[None] + np.asarray(
                view["deltas"]
            )
            all_vertices = np.concatenate(
                [np.asarray(view["baseline"])[None], expected_candidates]
            )
            np.testing.assert_array_equal(
                view["fixed_min"], all_vertices.min(axis=(0, 1))
            )
            np.testing.assert_array_equal(
                view["fixed_max"], all_vertices.max(axis=(0, 1))
            )
            for control in atlas["controls"]:
                index = control["index"]
                with np.load(control["vertex_effects_path"]) as arrays:
                    for row, level in zip(arrays["native_inputs"], args.levels):
                        expected = baseline.copy()
                        expected[index] = level
                        np.testing.assert_array_equal(row, expected)
                self.assertAlmostEqual(
                    control["samples"][0]["native_delta"], -baseline[index]
                )
            with np.load(root / "out/head_2/vanilla/jacobian.npz") as jacobian:
                np.testing.assert_allclose(
                    jacobian["raw_jacobian"][::3],
                    np.tile(2 * baseline, (4, 1)),
                    atol=1e-10,
                )
                np.testing.assert_array_equal(
                    jacobian["baseline_native_input"], baseline
                )

    def test_recorded_head_mismatch_rejected_before_loading_cache(self):
        import generate

        args = SimpleNamespace(baseline_provenance={"head_id": 2})
        with patch.object(generate, "HeadRig") as loader:
            with self.assertRaisesRegex(ValueError, "refusing head 1"):
                generate.generate_head(1, "vanilla", args, {})
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
