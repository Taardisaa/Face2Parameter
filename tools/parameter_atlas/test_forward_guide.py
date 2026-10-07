"""Necessary solver/identity contracts using fake evaluations, never the game."""
import copy
import unittest

from forward_guide import guide


class ForwardGuideTests(unittest.TestCase):
    def setUp(self):
        self.context = {"context_id": "a" * 32, "identity": {"native59": [.5] * 59},
                        "selected_abmx_bones": ["cf_J_Nose_t"],
                        "surfaces": [{"renderer_path": "head"}]}
        self.body = {"request": {"context_id": "a" * 32, "native_values": [.5] * 59},
                     "control": {"kind": "native", "index": 58}, "surface_path": "head",
                     "target_max": .01, "bound": .8}
        self.calls = []

    def evaluate(self, request):
        self.calls.append(copy.deepcopy(request))
        values = request.get("native_values")
        value = values[58] if values else request["bones"][0]["position"][1] + .5
        # A synthetic nonlinear scalar fixture checks full-candidate iteration.
        # It is not an HS2 formula or a replacement for its implementation.
        displacement = (value - .5) ** 2
        return {"context_id": "a" * 32, "live_state_unchanged": True,
                "surfaces": [{"renderer_path": "head", "delta": [[displacement, 0., 0.]]}]}

    def test_full_candidate_calls_solve_nonlinear_target_without_mutating_input(self):
        original = copy.deepcopy(self.body)
        result = guide(self.context, self.body, self.evaluate)
        self.assertEqual("target_bracketed", result["guidance"]["status"])
        self.assertLessEqual(result["guidance"]["absolute_residual"], .00001)
        self.assertEqual(original, self.body)
        self.assertTrue(all(call["native_values"][:58] == [.5] * 58 for call in self.calls))
        self.assertLessEqual(len(self.calls), 14)

    def test_unbracketed_endpoint_is_explicit_and_does_not_launch_sweep(self):
        self.body["target_max"] = .2
        result = guide(self.context, self.body, self.evaluate)
        self.assertEqual("target_not_bracketed", result["guidance"]["status"])
        self.assertEqual(2, len(self.calls))

    def test_context_other_parameters_and_bone_axes_are_bound(self):
        wrong = copy.deepcopy(self.body)
        wrong["request"]["native_values"][10] = .6
        with self.assertRaises(ValueError):
            guide(self.context, wrong, self.evaluate)
        wrong = copy.deepcopy(self.body)
        wrong["request"]["context_id"] = "b" * 32
        with self.assertRaises(ValueError):
            guide(self.context, wrong, self.evaluate)
        bone = copy.deepcopy(self.body)
        bone["control"] = {"kind": "abmx", "name": "cf_J_Nose_t", "channel": "position", "axis": 1}
        bone["request"].pop("native_values")
        bone["request"]["bones"] = [{"name": "cf_J_Nose_t", "position": [0., 0., 0.]}]
        bone["bound"] = .3
        result = guide(self.context, bone, self.evaluate)
        self.assertEqual("target_bracketed", result["guidance"]["status"])
        self.assertTrue(all(call["bones"][0]["position"][::2] == [0., 0.] for call in self.calls))
        bone["request"]["bones"][0]["position"][2] = .01
        with self.assertRaises(ValueError):
            guide(self.context, bone, self.evaluate)


if __name__ == "__main__":
    unittest.main()
