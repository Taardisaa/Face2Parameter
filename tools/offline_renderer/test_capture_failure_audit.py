import unittest

from audit_renderdoc_capture_failure import classify


class FailureAuditTests(unittest.TestCase):
    def test_selected_last_reason_preserves_both_causes(self):
        result = classify("Starting capture\nDon't have command list ResourceId::42 captured! This frame is unsuccessful.\n"
                          "Saw an Unmap that we didn't capture the corresponding Map for - this frame is unsuccessful\n"
                          "Failed to capture, frame 4294967295: Uncapped Map()/Unmap()\n")
        attempt = result["attempts"][0]
        self.assertEqual(attempt["selected_reason"], "uncaptured_map_unmap")
        self.assertEqual([event["branch"] for event in attempt["events"]],
                         ["uncaptured_command_list", "uncaptured_map_unmap", "uncaptured_map_unmap"])
        self.assertFalse(result["native_capture_accepted"])

    def test_multiple_attempts_never_merge(self):
        result = classify("Starting capture\nFailed to capture, frame 1: Uncapped command list\n"
                          "Starting capture\nFinished capture, Frame 2\n")
        self.assertEqual([attempt["status"] for attempt in result["attempts"]],
                         ["driver_rejected", "driver_finished_not_file_verified"])

    def test_missing_start_and_buffered_summary_not_inferred(self):
        result = classify("Failed to capture, frame 1: Uncapped command list\nStarting capture\n"
                          "Don't have command list ResourceId::7 captured! This frame is unsuccessful.\n")
        self.assertEqual(len(result["orphan_events"]), 1)
        self.assertEqual(result["attempts"][0]["status"], "incomplete_log")
        self.assertNotIn("selected_reason", result["attempts"][0])

    def test_unknown_summary_remains_unknown(self):
        result = classify("Starting capture\nFailed to capture, frame 1: Unknown reason\n")
        self.assertEqual(result["attempts"][0]["selected_reason"], "unknown")


if __name__ == "__main__":
    unittest.main()
