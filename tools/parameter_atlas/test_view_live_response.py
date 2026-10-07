"""Guard verified game viewers against mixed reports and changed geometry receipts."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from view_live_response import (
    UnverifiedResponse,
    bound_inputs,
    live_viewer_data,
    scoped_measurement,
    verified_selection,
)


def declarations():
    cases = [
        {
            "name": role,
            "kind": "native",
            "baseline_name": "card_input",
            "control": 30,
            "probe_role": role,
            "value": value,
        }
        for role, value in [("local_minus", 0.45), ("local_plus", 0.55)]
    ]
    cases.append({"name": "repeat", "kind": "baseline", "baseline_name": "card_input"})
    manifest = {
        "complete": True,
        "baselines": {"card_input": {"native59": [0.5] * 59}},
        "cases": cases,
    }
    report = {
        "normalized_tolerance": 1e-5,
        "baselines": {
            "card_input": {
                "native59": [0.5] * 59,
                "coverage": {
                    "required_probe_roles": ["local_minus", "local_plus"],
                    "per_renderer": {
                        "/head": {
                            "repeat_drift_pass": True,
                            "all_native_controls_verified_at_this_configuration": True,
                            "max_repeat_drift_normalized": 0.0,
                        }
                    },
                },
            }
        },
        "cases": [{**row, "trusted_isolated_response": True} for row in cases],
    }
    return manifest, report


class GameViewerGuards(unittest.TestCase):
    def test_component_scope_rejects_eye_request_without_global_waiver(self):
        self.assertTrue(scoped_measurement({"analysis_mesh_scope": ["o_head"]}, "o_head"))
        with self.assertRaisesRegex(UnverifiedResponse, "cannot verify another mesh"):
            scoped_measurement({"analysis_mesh_scope": ["o_head"]}, "o_eyebase_L")
        self.assertFalse(scoped_measurement({"analysis_mesh_scope": ["all_captured_renderers"]}, "o_eyebase_L"))
        with self.assertRaisesRegex(UnverifiedResponse, "Unsupported"):
            scoped_measurement({"analysis_mesh_scope": ["o_head", "o_eyebase_L"]}, "o_head")

    def test_requires_matching_manifest_hash_and_complete_collection(self):
        manifest, report = declarations()
        with tempfile.TemporaryDirectory() as folder:
            manifest_path, report_path = (
                Path(folder) / "manifest.json",
                Path(folder) / "report.json",
            )
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(UnverifiedResponse, "SHA-256-bound"):
                bound_inputs(manifest_path, report_path)
            report["source_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            bound_inputs(manifest_path, report_path)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            with self.assertRaisesRegex(UnverifiedResponse, "SHA-256-bound"):
                bound_inputs(manifest_path, report_path)
            manifest["complete"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report["source_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(UnverifiedResponse, "Incomplete"):
                bound_inputs(manifest_path, report_path)

    def test_missing_renderer_or_unverified_repeat_gate_rejected(self):
        for mutation in ["missing", "repeat", "coverage", "drift"]:
            manifest, report = declarations()
            gate = report["baselines"]["card_input"]["coverage"]["per_renderer"][
                "/head"
            ]
            if mutation == "missing":
                report["baselines"]["card_input"]["coverage"].pop("per_renderer")
            elif mutation == "repeat":
                gate["repeat_drift_pass"] = False
            elif mutation == "coverage":
                gate["all_native_controls_verified_at_this_configuration"] = False
            else:
                gate["max_repeat_drift_normalized"] = 0.01
            with self.subTest(mutation=mutation), self.assertRaises(UnverifiedResponse):
                verified_selection(manifest, report, "card_input", 30, "/head")

    def test_untrusted_mixed_or_missing_selected_case_rejected(self):
        for mutation in [
            "untrusted",
            "mixed",
            "missing",
            "duplicate",
            "role",
            "baseline",
        ]:
            manifest, report = declarations()
            if mutation == "untrusted":
                report["cases"][0]["trusted_isolated_response"] = False
            elif mutation == "mixed":
                report["cases"][0]["value"] = 0.4
            elif mutation == "missing":
                report["cases"].pop(0)
            elif mutation == "duplicate":
                report["cases"].append(copy.deepcopy(report["cases"][0]))
            elif mutation == "role":
                manifest["cases"].pop(0)
            else:
                report["baselines"]["card_input"]["native59"][3] = 0.4
            with self.subTest(mutation=mutation), self.assertRaises(UnverifiedResponse):
                verified_selection(manifest, report, "card_input", 30, "/head")

    def test_all_control_receipts_reconstructed_and_later_tampering_rejected(self):
        from live_response import Uncertifiable, analyze_manifest
        from test_live_response import fixture

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)

            def receipt(name, snapshot):
                raw = json.dumps(snapshot).encode("utf-8")
                path = root / (name + ".json.gz")
                path.write_bytes(gzip.compress(raw))
                return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

            manifest = {
                "complete": True,
                "baselines": {
                    "card_input": {
                        "native59": [0.5] * 59,
                        "geometry": receipt("baseline", fixture()),
                    }
                },
                "required_probe_roles": ["local_minus", "local_plus"],
                "cases": [],
            }
            for control in range(59):
                for role, value in [("local_minus", 0.45), ("local_plus", 0.55)]:
                    native = [0.5] * 59
                    native[control] = value
                    name = f"control_{control}_{role}"
                    manifest["cases"].append(
                        {
                            "name": name,
                            "kind": "native",
                            "baseline_name": "card_input",
                            "control": control,
                            "probe_role": role,
                            "value": value,
                            "native59": native,
                            "geometry": receipt(name, fixture(native)),
                        }
                    )
            manifest["cases"].append(
                {
                    "name": "repeat",
                    "kind": "baseline",
                    "baseline_name": "card_input",
                    "native59": [0.5] * 59,
                    "geometry": receipt("repeat", fixture()),
                }
            )
            report = analyze_manifest(manifest, root, offline=False)
            manifest_path, report_path = root / "manifest.json", root / "report.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report["source_manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            view = live_viewer_data(manifest_path, report_path, control=30)
            self.assertEqual(4, len(view["baseline"]))
            np.testing.assert_allclose(
                np.asarray(view["deltas"])[:, :, 0], [[-0.05] * 4, [0.05] * 4]
            )
            np.testing.assert_allclose(np.asarray(view["deltas"])[:, :, 1:], 0)
            self.assertEqual(
                0.0, view["verification"]["recomputed_repeat_drift_normalized"]
            )
            import live_response

            component = analyze_manifest(manifest, root, offline=False, head_only=True)
            component["source_manifest_sha256"] = report["source_manifest_sha256"]
            report_path.write_text(json.dumps(component), encoding="utf-8")
            with patch.object(live_response, "measure_snapshot", wraps=live_response.measure_snapshot) as measured:
                head_view = live_viewer_data(manifest_path, report_path, control=30)
                self.assertTrue(measured.call_args_list)
                self.assertTrue(all(call.kwargs.get("head_only") is True for call in measured.call_args_list))
            self.assertEqual(["o_head"], head_view["verification"]["analysis_mesh_scope"])
            with self.assertRaisesRegex(UnverifiedResponse, "cannot verify another mesh"):
                live_viewer_data(manifest_path, report_path, control=30, mesh="o_eyebase_L")
            first = Path(manifest["cases"][60]["geometry"]["path"])
            first.write_bytes(gzip.compress(b"{}"))
            with self.assertRaisesRegex(Uncertifiable, "SHA-256 mismatch"):
                live_viewer_data(manifest_path, report_path, control=30)


if __name__ == "__main__":
    unittest.main()
