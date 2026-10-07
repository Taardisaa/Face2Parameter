"""Adversarial acceptance tests for runtime evidence binding, no game needed."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime_validation import IDENTITY, CHIN, validate_inputs, offline_gate


def baseline():
    case = {"name": "head_3_baseline", "head_id": 3, "native59_requested": [.5] * 59,
            "native59_actual": [.5] * 59, "abmx_requested": {},
            "abmx_actual": {**copy.deepcopy(IDENTITY), "name": CHIN, "exists": True,
                            "coordinate_modifiers": [copy.deepcopy(IDENTITY)]}}
    snapshot = {"snapshot_kind": "maker_live_skinned_geometry", "character": {"head_id": 3, "shape_value_face": [.5] * 59}}
    return case, snapshot


class RuntimeEvidenceTests(unittest.TestCase):
    def test_actual_baseline_requires_complete_inputs(self):
        case, snapshot = baseline()
        self.assertEqual(validate_inputs(case, snapshot)["profile"], "vanilla")
        case["native59_actual"] = [.5] * 54
        with self.assertRaisesRegex(ValueError, "native59"):
            validate_inputs(case, snapshot)

    def test_actual_geometry_native_mismatch_is_not_hidden(self):
        case, snapshot = baseline()
        snapshot["character"]["shape_value_face"][58] = .50001
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_inputs(case, snapshot)

    def test_coordinate_specific_modifier_ambiguity_refused(self):
        case, snapshot = baseline()
        case["abmx_actual"]["coordinate_modifiers"].append(copy.deepcopy(IDENTITY))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            validate_inputs(case, snapshot)

    def test_original_checkpoint_tamper_refused(self):
        case, snapshot = baseline()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            path.write_text("{}", encoding="utf-8")
            case.update({"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(b"original").hexdigest()})
            with self.assertRaisesRegex(ValueError, "SHA"):
                validate_inputs(case, snapshot)

    def test_good_surface_does_not_hide_bone_position_failure(self):
        row = {"renderer_path": "head", "comparison": {"unit_scale_applied": 1, "rigid_errors": {"max_normalized": 1e-6}},
               "bone_local_comparison": {"max_head_skin_position_error": .05, "max_head_skin_rotation_angle_deg": 0,
                                         "max_head_skin_scale_error": 0, "bones": []}}
        gate = offline_gate(row, {"head"})
        self.assertFalse(gate["passed"])
        self.assertTrue(any("position" in reason for reason in gate["reasons"]))

    def test_duplicate_name_outside_certified_renderer_does_not_pass(self):
        row = {"renderer_path": "inactive_head", "comparison": {"unit_scale_applied": 1, "rigid_errors": {"max_normalized": 0}},
               "bone_local_comparison": {"max_head_skin_position_error": 0, "max_head_skin_rotation_angle_deg": 0,
                                         "max_head_skin_scale_error": 0, "bones": []}}
        self.assertFalse(offline_gate(row, {"visible_head"})["passed"])

    def test_missing_skin_palette_locals_cannot_certify_geometry_only(self):
        row = {"renderer_path": "head", "comparison": {"unit_scale_applied": 1, "rigid_errors": {"max_normalized": 0}},
               "bone_local_comparison": {"max_head_skin_position_error": 0, "max_head_skin_rotation_angle_deg": 0,
                                         "max_head_skin_scale_error": 0, "bones": []}}
        self.assertFalse(offline_gate(row, {"head"}, [CHIN])["passed"])


if __name__ == "__main__":
    unittest.main()
