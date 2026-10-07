"""Adversarial trace coverage/binding and analytically specified call chains."""
import copy
import unittest
from model import IDENTITY
from test_replay import state
from validate_trace import validate, verify_trace_header

MVID = "5442c72a-f463-4bf9-831a-247be87146c8"


def fixture():
    contract = {"schema_version": 1, "plugin_mvid": MVID, "plugin_version": "4.4.6.0", "apply_method_il_sha256": "a" * 64,
                "allowed_pre_existing_patch_owners": [], "no_rotation_bones": [], "is_coordinate_specific": False}
    metadata = {"schema_version": 1, "session_id": "analytic", "plugin_mvid": MVID, "plugin_version": "4.4.6.0", "apply_method_il_sha256": "a" * 64,
                "pre_existing_patch_owners": [], "started_frame": 1, "stopped_frame": 3, "requested_names": ["cf_J_ChinTip_s"],
                "observation_policy": "void Prefix last / void Postfix first; no argument, baseline, transform or return writes"}

    def raw(position, cache, frame):
        return {"bone_transform_id": 123, "local_position": position, "local_rotation_xyzw": [0, 0, 0, 1], "local_scale": [1, 1, 1],
                "cache": {"fields": copy.deepcopy(cache), "missing_fields": [], "assembly_mvid": MVID,
                          "frame_count": frame, "bone_transform_id": 123, "modifier_type": "KKABMX.Core.BoneModifier"}}

    initial, final = state(), state()
    final.update({"_forceApply": True, "_changedPosition": True})
    modifier = copy.deepcopy(IDENTITY)
    modifier["position"] = [1, 0, 0]
    events = []
    for sequence in (1, 2):
        events.append({"sequence": sequence, "frame": sequence, "completed_frame": sequence, "bone_name": "cf_J_ChinTip_s",
                       "modifier_instance_identity": 444, "coordinate": 0, "coordinate_specific": False,
                       "resolved_modifier": copy.deepcopy(modifier), "additional_modifiers": [], "no_rotation_excluded": False,
                       "is_during_h_scene": False,
                       "before": raw([0, 0, 10] if sequence == 1 else [1, 0, 10], initial if sequence == 1 else final, sequence),
                       "after": raw([1, 0, 10], final, sequence)})
    trace = {"metadata": metadata, "trace_complete": True, "active": False, "observed_calls": 2,
             "dropped_events": 0, "pending_calls": 0, "observer_errors": [], "events": events}
    return trace, contract


class TraceRejections(unittest.TestCase):
    def test_real_order_propagates_without_estimating_counts(self):
        trace, contract = fixture()
        report = validate(trace, contract)
        self.assertTrue(report["passed"])
        self.assertEqual(report["observed_call_count"], 2)
        self.assertEqual(report["rows"][1]["previous_same_modifier_call_sequence"], 1)
        self.assertTrue(report["rows"][1]["chain_contiguous"])
        self.assertFalse(report["application_count_fitted"])
        self.assertFalse(report["static_whole_head_model_fixed"])

    def test_native_overwrite_is_explicit_external_boundary(self):
        trace, contract = fixture()
        second = trace["events"][1]
        second["before"]["local_position"] = [0, 2, 10]
        second["before"]["cache"]["fields"]["_posBaseline"] = [0, 2, 10]
        second["after"]["local_position"] = [1, 2, 10]
        second["after"]["cache"]["fields"]["_posBaseline"] = [0, 2, 10]
        report = validate(trace, contract)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["inter_call_boundaries"]), 1)
        self.assertFalse(report["rows"][1]["chain_contiguous"])

    def test_drop_pending_observer_error_are_rejected(self):
        for field, value in (("dropped_events", 1), ("pending_calls", 1), ("observer_errors", ["prefix failed"]), ("active", True)):
            with self.subTest(field=field):
                trace, contract = fixture()
                trace[field] = value
                with self.assertRaises(ValueError):
                    verify_trace_header(trace, contract)

    def test_external_patches_and_wrong_il_refused(self):
        for field, value in (("pre_existing_patch_owners", ["unknown.plugin"]), ("apply_method_il_sha256", "b" * 64)):
            with self.subTest(field=field):
                trace, contract = fixture()
                trace["metadata"][field] = value
                with self.assertRaises(ValueError):
                    verify_trace_header(trace, contract)

    def test_missing_call_out_of_order_sequence_refused(self):
        trace, contract = fixture()
        trace["events"].reverse()
        self.assertFalse(validate(trace, contract)["passed"])
        trace, contract = fixture()
        trace["observed_calls"] = 3
        with self.assertRaises(ValueError):
            verify_trace_header(trace, contract)

    def test_after_transform_flag_and_private_identity_corruption_refused(self):
        trace, contract = fixture()
        trace["events"][0]["after"]["local_position"][0] = 2
        self.assertFalse(validate(trace, contract)["passed"])
        trace, contract = fixture()
        trace["events"][0]["after"]["cache"]["fields"]["_changedPosition"] = False
        self.assertFalse(validate(trace, contract)["passed"])
        trace, contract = fixture()
        trace["events"][0]["before"]["cache"]["bone_transform_id"] = 999
        self.assertFalse(validate(trace, contract)["passed"])

    def test_missing_additional_input_and_exclusion_mismatch_refused(self):
        trace, contract = fixture()
        del trace["events"][0]["additional_modifiers"]
        self.assertFalse(validate(trace, contract)["passed"])
        trace, contract = fixture()
        trace["events"][0]["no_rotation_excluded"] = True
        self.assertFalse(validate(trace, contract)["passed"])


if __name__ == "__main__":
    unittest.main()
