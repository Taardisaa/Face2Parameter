"""Snapshot cursor rejection tests, no game or pose/count fitting."""
import copy
import unittest
from test_trace import fixture
from validate_trace import validate
from validate_geometry import select_cursor_prediction


def cursor_fixture():
    trace, contract = fixture()
    trace["metadata"]["character_transform_id"] = 44
    final = copy.deepcopy(trace["events"][1])
    final.update({"sequence": 3, "frame": 3, "completed_frame": 3})
    for key in ("before", "after"):
        final[key]["cache"]["frame_count"] = 3
    final["resolved_modifier"]["position"] = [9, 0, 0]
    final["after"]["local_position"] = [9, 0, 10]
    trace["events"].append(final)
    trace["observed_calls"] = 3
    report = validate(trace, contract)
    observed = copy.deepcopy(trace["events"][1]["after"])
    snapshot = {"frame_count": 2, "frame_count_end": 2, "character": {"transform_id": 44},
                "abmx_trace_cursor": {"session_id": "analytic", "frame": 2, "active": True,
                        "last_completed_sequence": 2, "observed_calls": 2, "completed_calls": 2,
                        "pending_calls": 0, "dropped_events": 0, "observer_error_count": 0},
                "transforms": [{"id": 123, "name": "cf_J_ChinTip_s", "path": "/head/chin",
                                **{key: observed[key] for key in ("local_position", "local_scale", "local_rotation_xyzw")}}],
                "abmx_runtime": {"bones": [{"name": "cf_J_ChinTip_s", "runtime_baseline": observed["cache"]}]}}
    return snapshot, trace, report


class ExactCursorTests(unittest.TestCase):
    def test_cursor_selects_earlier_prediction_not_later_call(self):
        snapshot, trace, report = cursor_fixture()
        predicted, binding = select_cursor_prediction(snapshot, trace, report)
        self.assertEqual(predicted["local_position"], [1, 0, 10])
        self.assertEqual(binding["selected_observed_call_sequence"], 2)
        self.assertEqual(binding["later_calls_excluded"], 1)
        self.assertEqual(trace["events"][-1]["after"]["local_position"], [9, 0, 10])

    def test_capture_before_current_frame_lateupdate_uses_exact_sequence(self):
        snapshot, trace, report = cursor_fixture()
        snapshot["frame_count"] = snapshot["frame_count_end"] = snapshot["abmx_trace_cursor"]["frame"] = 3
        snapshot["abmx_runtime"]["bones"][0]["runtime_baseline"]["frame_count"] = 3
        predicted, binding = select_cursor_prediction(snapshot, trace, report)
        self.assertEqual(predicted["local_position"], [1, 0, 10])
        self.assertEqual(binding["frames_since_selected_call"], 1)

    def test_missing_wrong_session_future_cursor_rejected(self):
        snapshot, trace, report = cursor_fixture()
        del snapshot["abmx_trace_cursor"]
        with self.assertRaises(ValueError):
            select_cursor_prediction(snapshot, trace, report)
        snapshot, trace, report = cursor_fixture()
        snapshot["abmx_trace_cursor"]["session_id"] = "other-session"
        with self.assertRaises(ValueError):
            select_cursor_prediction(snapshot, trace, report)
        snapshot, trace, report = cursor_fixture()
        for key in ("last_completed_sequence", "completed_calls", "observed_calls"):
            snapshot["abmx_trace_cursor"][key] = 3
        with self.assertRaisesRegex(ValueError, "after snapshot"):
            select_cursor_prediction(snapshot, trace, report)

    def test_actual_snapshot_local_or_private_mismatch_refused(self):
        snapshot, trace, report = cursor_fixture()
        snapshot["transforms"][0]["local_position"][0] = 3
        with self.assertRaisesRegex(ValueError, "predicted local"):
            select_cursor_prediction(snapshot, trace, report)
        snapshot, trace, report = cursor_fixture()
        snapshot["abmx_runtime"]["bones"][0]["runtime_baseline"]["fields"]["_forceApply"] = False
        with self.assertRaisesRegex(ValueError, "private state"):
            select_cursor_prediction(snapshot, trace, report)

    def test_foreign_bone_and_incomplete_cursor_refused(self):
        snapshot, trace, report = cursor_fixture()
        snapshot["transforms"][0]["id"] = 555
        with self.assertRaisesRegex(ValueError, "bone ID"):
            select_cursor_prediction(snapshot, trace, report)
        snapshot, trace, report = cursor_fixture()
        snapshot["abmx_trace_cursor"]["pending_calls"] = 1
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_cursor_prediction(snapshot, trace, report)


if __name__ == "__main__":
    unittest.main()
