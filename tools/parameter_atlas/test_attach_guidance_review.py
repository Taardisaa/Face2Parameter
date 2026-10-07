"""Reject changed provenance and distinguish measured points from interval claims."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from attach_guidance_review import attach_review
from explore_live_response import canonical_hash
from view_live_response import UnverifiedResponse


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class GuidanceAttachmentTests(unittest.TestCase):
    def fixture(self, root):
        package = root / "original"
        (package / "data").mkdir(parents=True)
        identity = {"head_id": 2, "native59": [0.5] * 59, "expression": {"fixed": True}}
        identity_sha = canonical_hash(identity)
        training_receipt = write_json(root / "training.json", {"complete": True})
        case_receipt = {"path": "training-case.json", "sha256": "a" * 64}
        data = {
            "id": "card_input_native_30",
            "kind": "native",
            "baseline_name": "card_input",
            "control": 30,
            "mesh": "o_head",
            "identity": identity,
            "baseline_level": 0.5,
            "baseline": [[0, 0, 0], [1, 0, 0]],
            "deltas": [[[-0.01, 0, 0]] * 2, [[0.01, 0, 0]] * 2],
            "levels": [0.45, 0.55],
            "roles": ["local_minus", "local_plus"],
            "verification": {"case_receipts": [case_receipt, case_receipt]},
            "local_response": {
                "left_gain": 0.2,
                "right_gain": 0.2,
                "minus_step": 0.05,
                "plus_step": 0.05,
                "valid_interval": [0.45, 0.55],
                "estimate_verified": False,
            },
        }
        js = package / "data/card_input_native_30.js"
        js.write_text(
            "window.HS2_RESPONSE(" + json.dumps(data) + ");\n", encoding="utf-8"
        )
        catalog = {
            "schema_version": 1,
            "head_id": 2,
            "complete": True,
            "provenance": {"source_manifest_sha256": training_receipt["sha256"]},
            "baselines": {
                "card_input": {
                    "identity": identity,
                    "identity_sha256": identity_sha,
                    "head_diagonal": 1,
                    "repeat_drift_normalized": 0,
                }
            },
            "entries": [
                {
                    "id": data["id"],
                    "kind": "native",
                    "baseline_name": "card_input",
                    "control": 30,
                    "file": "data/card_input_native_30.js",
                    "sha256": hashlib.sha256(js.read_bytes()).hexdigest(),
                }
            ],
            "failures": [
                {"name": f"eye-{i}", "reason": "original gate failure"}
                for i in range(3)
            ],
        }
        catalog_path = package / "catalog.json"
        catalog_receipt = write_json(catalog_path, catalog)
        thresholds = {
            "geometry_normalized": 1e-5,
            "repeat_normalized": 1e-5,
            "relative_prediction_error": 0.05,
            "noise_multiplier": 3,
            "baseline_reuse_normalized": 1e-5,
        }
        planned, rows = [], []
        for sign in (-1, 1):
            point = {
                "entry_id": data["id"],
                "sign": sign,
                "kind": "native",
                "baseline_name": "card_input",
                "signed_step": sign * 0.0005,
                "target_units": 0.0001,
                "value": 0.5 + sign * 0.0005,
                "requested_target_percent": 0.01,
                "max_probe_fraction": 0.4,
            }
            planned.append(point)
            rows.append(
                {
                    "entry_id": data["id"],
                    "sign": sign,
                    "kind": "native",
                    "baseline_name": "card_input",
                    "signed_step": point["signed_step"],
                    "heldout_value": point["value"],
                    "input_trusted": True,
                    "baseline_reuse_pass": True,
                    "verified_prediction": True,
                    "baseline_identity_sha256": identity_sha,
                    "training_gain": 0.2,
                    "training_case_receipt": case_receipt,
                    "target_units": 0.0001,
                    "actual_max": 0.0001,
                    "max_vector_error": 0,
                    "relative_target_error": 0,
                    "error_budget": 0.000005,
                }
            )
        plan_receipt = write_json(
            root / "plan.json",
            {
                "schema_version": 1,
                "thresholds": thresholds,
                "catalog_receipt": catalog_receipt,
                "training_manifest_receipt": training_receipt,
                "entries": planned,
                "skipped": [],
                "baselines": catalog["baselines"],
            },
        )
        manifest_path = root / "heldout.json"
        manifest_receipt = write_json(
            manifest_path,
            {
                "complete": True,
                "public_configuration_restored": True,
                "restoration_errors": [],
                "plan_receipt": plan_receipt,
            },
        )
        source = root / "executed_review.py"
        source.write_text("# fixture reviewer source\n", encoding="utf-8")
        review = {
            "schema_version": 1,
            "analysis_mesh_scope": ["o_head"],
            "public_configuration_restored": True,
            "source_manifest_sha256": manifest_receipt["sha256"],
            "catalog_receipt": catalog_receipt,
            "plan_receipt": plan_receipt,
            "training_manifest_sha256": training_receipt["sha256"],
            "thresholds": thresholds,
            "entries": rows,
            "skipped": [],
            "source_receipts": [
                {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            ],
            "baselines": {
                "card_input": {
                    "identity_sha256": identity_sha,
                    "baseline_reuse_pass": True,
                    "repeat_drift_pass": True,
                    "training_repeat_noise": 0,
                    "fresh_repeat_noise": 0,
                    "baseline_reuse_normalized": 0,
                    "repeat_count": 2,
                    "head_diagonal": 1,
                }
            },
            "coverage": {
                "card_input": {
                    "native_controls": 1,
                    "verified_native_signs": 2,
                    "all59_both_signs_verified": False,
                }
            },
        }
        review_path = root / "review.json"
        write_json(review_path, review)
        return catalog_path, review_path, manifest_path, review, data

    def test_points_attach_without_promoting_estimates_or_modifying_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog, review, manifest, _, original = self.fixture(root)
            before = catalog.read_bytes()
            result = attach_review(catalog, review, manifest, root / "overlay")
            self.assertEqual(before, catalog.read_bytes())
            self.assertEqual(3, len(result["failures"]))
            output = root / "overlay" / result["entries"][0]["file"]
            data = json.loads(output.read_text()[len("window.HS2_RESPONSE(") : -3])
            self.assertFalse(data["local_response"]["estimate_verified"])
            self.assertEqual(
                {"-1", "1"}, set(data["local_response"]["heldout_validation_by_sign"])
            )
            for key in ("baseline", "deltas", "identity", "levels", "roles"):
                self.assertEqual(original[key], data[key])
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(),
                result["entries"][0]["sha256"],
            )
            self.assertNotEqual(
                result["entries"][0]["sha256"],
                result["entries"][0]["training_data_sha256"],
            )
            self.assertIn(
                "独立采样对照",
                (root / "overlay/index.html").read_text(encoding="utf-8"),
            )

    def test_failed_prediction_is_preserved_without_interval_certification(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog, report_path, manifest, review, _ = self.fixture(root)
            review["entries"][0].update(
                verified_prediction=False, max_vector_error=0.001
            )
            review["coverage"]["card_input"]["verified_native_signs"] = 1
            write_json(report_path, review)
            result = attach_review(catalog, report_path, manifest, root / "overlay")
            raw = (root / "overlay" / result["entries"][0]["file"]).read_text()
            data = json.loads(raw[len("window.HS2_RESPONSE(") : -3])
            self.assertFalse(
                data["local_response"]["heldout_validation_by_sign"]["-1"][
                    "verified_prediction"
                ]
            )
            self.assertFalse(data["local_response"]["estimate_verified"])

    def test_untrusted_or_incomplete_source_review_rejects_whole_overlay(self):
        for change in (
            "input",
            "missing_sign",
            "duplicate",
            "identity",
            "budget",
            "point",
            "global_status",
            "reuse",
            "source",
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                catalog, report_path, manifest, review, _ = self.fixture(root)
                if change == "input":
                    review["entries"][0]["input_trusted"] = False
                elif change == "missing_sign":
                    review["entries"].pop()
                elif change == "duplicate":
                    review["entries"].append(copy.deepcopy(review["entries"][0]))
                elif change == "identity":
                    review["entries"][0]["baseline_identity_sha256"] = "wrong"
                elif change == "budget":
                    review["entries"][0]["error_budget"] = 1
                elif change == "point":
                    review["entries"][0]["heldout_value"] += 0.01
                elif change == "global_status":
                    review["entries"][0]["verified_prediction"] = False
                elif change == "reuse":
                    review["baselines"]["card_input"]["baseline_reuse_pass"] = False
                elif change == "source":
                    Path(review["source_receipts"][0]["path"]).write_text(
                        "changed source"
                    )
                write_json(report_path, review)
                with self.assertRaises(UnverifiedResponse):
                    attach_review(catalog, report_path, manifest, root / "overlay")
                self.assertFalse((root / "overlay").exists())

    def test_modified_data_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            catalog, report, manifest, _, _ = self.fixture(root)
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "fresh"):
                attach_review(catalog, report, manifest, output)
            (catalog.parent / "data/card_input_native_30.js").write_text("changed")
            with self.assertRaisesRegex(UnverifiedResponse, "SHA-256"):
                attach_review(catalog, report, manifest, root / "overlay")
            self.assertFalse((root / "overlay").exists())


if __name__ == "__main__":
    unittest.main()
