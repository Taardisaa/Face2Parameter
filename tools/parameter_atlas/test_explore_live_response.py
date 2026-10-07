"""Exercise the explorer with reconstructable surfaces and changed-state negatives."""

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from check_response_profile import check_profile
from explore_live_response import (
    calibration_identity,
    export_explorer,
    identity_mismatches,
    local_response,
    validate_abmx_case,
)
from live_response import analyze_manifest
from test_live_response import fixture
from view_live_response import UnverifiedResponse


class ExplorerTests(unittest.TestCase):
    def test_directional_estimates_do_not_average_nonlinear_sides(self):
        cases = [
            {"kind": "native", "probe_role": "local_minus", "value": 0.45},
            {"kind": "native", "probe_role": "local_plus", "value": 0.55},
        ]
        response = local_response(0.5, cases, np.array([[[-0.1, 0, 0]], [[0.2, 0, 0]]]))
        self.assertAlmostEqual(2, response["left_gain"])
        self.assertAlmostEqual(4, response["right_gain"])
        self.assertFalse(response["estimate_verified"])
        with self.assertRaisesRegex(UnverifiedResponse, "Exactly one"):
            local_response(0.5, cases[:1], np.array([[[-0.1, 0, 0]]]))

    def test_identity_rejects_head_other_coefficients_abmx_and_expression(self):
        manifest = {"coordinate_transform_name": "cf_J_Head"}
        expected = calibration_identity(fixture(), manifest)
        actual = calibration_identity(fixture([0.6] + [0.5] * 58), manifest)
        self.assertEqual(["native59"], identity_mismatches(expected, actual))
        for key, value in (
            ("head_id", 3),
            ("abmx", {"x": 1}),
            ("expression", {"mouth_ptn": 1}),
            ("source_meshes", []),
            ("anchor_local_scale", [2, 2, 2]),
        ):
            with self.subTest(key=key):
                changed = dict(expected, **{key: value})
                self.assertEqual([key], identity_mismatches(expected, changed))
        tiny = dict(expected, native59=[0.500001] * 59)
        self.assertFalse(identity_mismatches(expected, tiny))
        self.assertIn(
            "native59",
            identity_mismatches(expected, dict(expected, native59=[True] * 59)),
        )

    def test_abmx_metadata_and_unreviewed_channels_rejected(self):
        snap = fixture()
        case = {
            "name": "a",
            "kind": "abmx",
            "baseline_name": "card_input",
            "bone": "face_bone",
            "channel": "scale",
            "axis": 0,
            "value": 1.02,
            "step": 0.02,
            "native59": [0.5] * 59,
            "patch": {
                "name": "face_bone",
                "scale": [1.02, 1, 1],
                "length": 1,
                "position": [0, 0, 0],
                "rotation": [0, 0, 0],
            },
        }
        checked = {
            **case,
            "trusted_isolated_response": True,
            "interpretable_isolated_response_all_renderers": True,
        }
        plan = {"abmx_cases": [case]}
        validate_abmx_case(case, checked, plan, snap)
        with self.assertRaisesRegex(UnverifiedResponse, "interpretation"):
            validate_abmx_case(
                case,
                dict(checked, interpretable_isolated_response_all_renderers=False),
                plan,
                snap,
            )
        with self.assertRaisesRegex(UnverifiedResponse, "metadata"):
            validate_abmx_case(case, dict(checked, axis=1), plan, snap)
        changed = json.loads(json.dumps(case))
        changed["patch"]["rotation"][1] = 1
        with self.assertRaisesRegex(UnverifiedResponse, "more than one"):
            validate_abmx_case(changed, checked, {"abmx_cases": [changed]}, snap)

    def test_all_controls_two_baselines_and_abmx_have_verified_data_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def receipt(name, snapshot):
                raw = json.dumps(snapshot).encode()
                path = root / (name + ".json.gz")
                path.write_bytes(gzip.compress(raw))
                return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

            manifest = {
                "head_id": 2,
                "complete": True,
                "public_configuration_restored": True,
                "coordinate_transform_name": "cf_J_Head",
                "baselines": {},
                "cases": [],
                "before": {
                    "shapes": {
                        "shapes": [
                            {"index": i, "name": f"control{i}"} for i in range(59)
                        ]
                    }
                },
            }
            for name, value in (("card_input", 0.5), ("mixed_input", 0.6)):
                native = [value] * 59
                manifest["baselines"][name] = {
                    "native59": native,
                    "geometry": receipt(name, fixture(native)),
                }
                for control in range(59):
                    for role, step in (("local_minus", -0.05), ("local_plus", 0.05)):
                        vector = native.copy()
                        vector[control] += step
                        case = {
                            "name": f"{name}{control}{role}",
                            "kind": "native",
                            "baseline_name": name,
                            "control": control,
                            "probe_role": role,
                            "value": vector[control],
                            "native59": vector,
                        }
                        case["geometry"] = receipt(case["name"], fixture(vector))
                        manifest["cases"].append(case)
                manifest["cases"].append(
                    {
                        "name": name + "repeat",
                        "kind": "baseline",
                        "baseline_name": name,
                        "native59": native,
                        "geometry": receipt(name + "repeat", fixture(native)),
                    }
                )
            for sign in (-1, 1):
                patch = {
                    "name": "face_bone",
                    "scale": [1 + sign * 0.02, 1, 1],
                    "length": 1,
                    "position": [0, 0, 0],
                    "rotation": [0, 0, 0],
                }
                snap = fixture()
                snap["abmx_runtime"]["bones"] = [patch]
                case = {
                    "name": f"abmx{sign}",
                    "kind": "abmx",
                    "baseline_name": "card_input",
                    "bone": "face_bone",
                    "channel": "scale",
                    "axis": 0,
                    "value": patch["scale"][0],
                    "step": 0.02,
                    "native59": [0.5] * 59,
                    "patch": patch,
                    "geometry": receipt(f"abmx{sign}", snap),
                }
                manifest["cases"].append(case)
            plan = {
                "baselines": {
                    k: r["native59"] for k, r in manifest["baselines"].items()
                },
                "native_cases": [r for r in manifest["cases"] if r["kind"] == "native"],
                "abmx_cases": [r for r in manifest["cases"] if r["kind"] == "abmx"],
                "local_step": 0.05,
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            manifest.update(
                predeclared_plan=str(plan_path),
                predeclared_plan_sha256=hashlib.sha256(
                    plan_path.read_bytes()
                ).hexdigest(),
            )
            mp = root / "manifest.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            digest = hashlib.sha256(mp.read_bytes()).hexdigest()
            hp, fp = root / "head.json", root / "full.json"
            for path, head_only in ((hp, True), (fp, False)):
                report = analyze_manifest(
                    manifest, root, offline=False, head_only=head_only
                )
                report["source_manifest_sha256"] = digest
                path.write_text(json.dumps(report), encoding="utf-8")
            catalog = export_explorer(mp, hp, fp, root / "explorer")
            self.assertTrue(catalog["complete"])
            self.assertEqual(119, len(catalog["entries"]))
            self.assertEqual(
                1, len([e for e in catalog["entries"] if e["kind"] == "abmx"])
            )
            catalog_path = root / "explorer" / "catalog.json"
            original_snapshot = Path(
                manifest["baselines"]["card_input"]["geometry"]["path"]
            )
            self.assertTrue(
                check_profile(catalog_path, original_snapshot, "card_input")[
                    "compatible_public_configuration"
                ]
            )
            shifted_snapshot = Path(
                manifest["baselines"]["mixed_input"]["geometry"]["path"]
            )
            mismatch = check_profile(catalog_path, shifted_snapshot, "card_input")
            self.assertFalse(mismatch["compatible_public_configuration"])
            self.assertEqual(["native59"], mismatch["changed_fields"])
            modified = json.loads(catalog_path.read_text(encoding="utf-8"))
            modified["baselines"]["card_input"]["identity"]["head_id"] = 3
            forged = root / "forged_catalog.json"
            forged.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity SHA-256 mismatch"):
                check_profile(forged, original_snapshot, "card_input")
            for entry in catalog["entries"]:
                file = root / "explorer" / entry["file"]
                raw = file.read_bytes()
                self.assertEqual(entry["sha256"], hashlib.sha256(raw).hexdigest())
                data = json.loads(
                    raw.decode()
                    .strip()
                    .removeprefix("window.HS2_RESPONSE(")
                    .removesuffix(");")
                )
                self.assertEqual(entry["id"], data["id"])
                self.assertEqual(2, len(data["levels"]))
                self.assertEqual(4, len(data["baseline"]))
                self.assertFalse(data["local_response"]["estimate_verified"])
            broken = Path(manifest["cases"][0]["geometry"]["path"])
            broken.write_bytes(b"bad")
            with self.assertRaises((OSError, ValueError)):
                export_explorer(mp, hp, fp, root / "corrupt")
            self.assertFalse((root / "corrupt" / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
