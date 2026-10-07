"""Analytical direction, clipping and noise guards for independent witnesses."""

import unittest

from guidance_plan import interior_suggestion


class GuidancePlanTests(unittest.TestCase):
    def setUp(self):
        self.payload = {
            "baseline_level": 0.5,
            "local_response": {
                "left_gain": 2.0,
                "right_gain": 4.0,
                "minus_step": 0.05,
                "plus_step": 0.05,
                "valid_interval": [0.45, 0.55],
            },
        }
        self.baseline = {"head_diagonal": 1.0, "repeat_drift_normalized": 1e-6}

    def test_direction_uses_matching_one_sided_gain(self):
        negative = interior_suggestion(self.payload, self.baseline, -1)
        positive = interior_suggestion(self.payload, self.baseline, 1)
        self.assertAlmostEqual(-0.00005, negative["signed_step"])
        self.assertAlmostEqual(0.000025, positive["signed_step"])
        self.assertAlmostEqual(0.0001, positive["target_units"])
        self.assertFalse(negative["request_clipped"])

    def test_large_request_becomes_explicit_interior_target(self):
        row = interior_suggestion(self.payload, self.baseline, -1, target_percent=100)
        self.assertTrue(row["request_clipped"])
        self.assertAlmostEqual(0.02, row["step"])
        self.assertAlmostEqual(0.04, row["target_units"])
        self.assertAlmostEqual(4, row["target_percent"])
        self.assertEqual(100, row["requested_target_percent"])

    def test_weak_response_is_noise_guarded_and_invalid_inputs_rejected(self):
        row = interior_suggestion(self.payload, self.baseline, 1, target_percent=0.0001)
        self.assertFalse(row["supported"])
        self.assertIn("noise", row["reason"])
        for arguments in (
            {"max_probe_fraction": 1},
            {"target_percent": float("nan")},
            {"target_percent": 0},
        ):
            with self.assertRaises(ValueError):
                interior_suggestion(self.payload, self.baseline, 1, **arguments)
        with self.assertRaises(ValueError):
            interior_suggestion(self.payload, self.baseline, True)


if __name__ == "__main__":
    unittest.main()
