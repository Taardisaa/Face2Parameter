"""Analytic branch and malformed-evidence tests independent of live game calls."""
import copy
import unittest
import numpy as np
from model import IDENTITY, ReplayRejected, replay_apply, unity_zero


def state():
    return {"_hasBaseline": True, "_lenBaseline": 10., "_sclBaseline": [2., 3., 4.],
            "_posBaseline": [0., 0., 10.], "_positionBaseline": [0., 0., 2.],
            "_rotBaseline": [0., 0., 0., 1.], "_lenModForceUpdate": False,
            "_lenModNeedsPositionRestore": False, "_changedScale": False,
            "_changedRotation": False, "_changedPosition": False, "_forceApply": False}


def call(**overrides):
    parameters = {"before": {"local_position": [3., 4., 0.], "local_rotation_xyzw": [0., 0., 0., 1.], "local_scale": [1., 1., 1.]},
                  "cache": state(), "coordinate_modifiers": [copy.deepcopy(IDENTITY)], "coordinate": 0,
                  "additional_modifiers": [], "bone_exists": True, "rotation_excluded": False,
                  "is_during_h_scene": False, "coordinate_specific": False}
    parameters.update(overrides)
    return replay_apply(**parameters)


def mod(**overrides):
    value = copy.deepcopy(IDENTITY)
    value.update(overrides)
    return value


class ReplayBranches(unittest.TestCase):
    def test_normalized_length_and_offset(self):
        out = call(coordinate_modifiers=[mod(length=2, position=[1, 0, 0])])
        np.testing.assert_allclose(out["after"]["local_position"], [13, 16, 0], atol=1e-6)
        self.assertTrue(out["cache_after"]["_changedPosition"])
        self.assertTrue(out["cache_after"]["_forceApply"])

    def test_second_real_call_uses_changed_direction(self):
        initial = {"local_position": [0, 1, 0], "local_rotation_xyzw": [0, 0, 0, 1], "local_scale": [1, 1, 1]}
        cache = state()
        cache["_lenBaseline"] = 1
        first = call(before=initial, cache=cache, coordinate_modifiers=[mod(length=2, position=[1, 0, 0])])
        second = call(before=first["after"], cache=first["cache_after"], coordinate_modifiers=[mod(length=2, position=[1, 0, 0])])
        np.testing.assert_allclose(second["after"]["local_position"], [1 + 2/np.sqrt(5), 4/np.sqrt(5), 0], atol=1e-6)

    def test_small_length_switches_historical_direction(self):
        out = call(coordinate_modifiers=[mod(length=.05)])
        np.testing.assert_allclose(out["after"]["local_position"], [0, 0, .5], atol=1e-7)
        self.assertTrue(out["cache_after"]["_lenModNeedsPositionRestore"])

    def test_h_scene_switches_historical_direction(self):
        out = call(coordinate_modifiers=[mod(length=2)], is_during_h_scene=True)
        self.assertEqual(out["after"]["local_position"], [0, 0, 20])

    def test_installed_unity_approximate_zero(self):
        self.assertTrue(unity_zero(np.array([1e-6, 0, 0], np.float32)))
        self.assertFalse(unity_zero(np.array([1e-4, 0, 0], np.float32)))
        before = {"local_position": [1e-6, 0, 0], "local_rotation_xyzw": [0, 0, 0, 1], "local_scale": [1, 1, 1]}
        self.assertEqual(call(before=before, coordinate_modifiers=[mod(length=2)])["after"]["local_position"], [0, 0, 20])

    def test_missing_length_baseline_does_not_clear_force_length_flag(self):
        cache = state()
        cache.update({"_positionBaseline": [1e-6, 0, 0], "_lenModForceUpdate": True})
        out = call(cache=cache)
        self.assertEqual(out["after"]["local_position"], [3, 4, 0])
        self.assertTrue(out["cache_after"]["_lenModForceUpdate"])

    def test_empty_canapply_skip_does_not_restore_changed_scale(self):
        cache = state()
        cache["_changedScale"] = True
        out = call(cache=cache)
        self.assertEqual(out["after"]["local_scale"], [1, 1, 1])
        self.assertTrue(out["cache_after"]["_changedScale"])
        self.assertEqual(out["branches"], ["return_CanApply_false"])

    def test_force_apply_empty_restores_flags_and_position(self):
        cache = state()
        cache.update({"_forceApply": True, "_changedScale": True, "_changedRotation": True, "_changedPosition": True})
        out = call(cache=cache)
        self.assertEqual(out["after"]["local_scale"], [2, 3, 4])
        # Length computation is explicitly overwritten when restoring position.
        self.assertEqual(out["after"]["local_position"], [0, 0, 10])
        for key in ("_forceApply", "_lenModForceUpdate", "_changedScale", "_changedRotation", "_changedPosition"):
            self.assertFalse(out["cache_after"][key])

    def test_excluded_rotation_restores_if_previously_changed(self):
        cache = state()
        cache["_changedRotation"] = True
        out = call(cache=cache, coordinate_modifiers=[mod(rotation=[0, 90, 0])], rotation_excluded=True)
        self.assertEqual(out["after"]["local_rotation_xyzw"], [0, 0, 0, 1])
        self.assertFalse(out["cache_after"]["_changedRotation"])

    def test_analytic_quarter_turn(self):
        out = call(coordinate_modifiers=[mod(rotation=[0, 90, 0])])
        np.testing.assert_allclose(out["after"]["local_rotation_xyzw"], [0, np.sqrt(.5), 0, np.sqrt(.5)], atol=1e-7)

    def test_additional_modifier_combines_products_and_sums(self):
        out = call(coordinate_modifiers=[mod(scale=[2, 2, 2], length=1.5, position=[1, 0, 0])],
                   additional_modifiers=[mod(scale=[3, 3, 3], length=2, position=[0, 1, 0])])
        self.assertEqual(out["after"]["local_scale"], [12, 18, 24])
        np.testing.assert_allclose(out["after"]["local_position"], [19, 25, 0], atol=1e-6)

    def test_installed_first_coordinate_always_selected(self):
        out = call(coordinate=6, coordinate_modifiers=[mod(position=[1, 0, 0]), mod(position=[9, 0, 0])])
        self.assertEqual(out["after"]["local_position"], [1, 0, 10])
        with self.assertRaisesRegex(ReplayRejected, "incompatible variant"):
            call(coordinate_specific=True)

    def test_null_base_with_additional_uses_identity(self):
        out = call(coordinate_modifiers=[None], additional_modifiers=[mod(position=[1, 0, 0])])
        self.assertEqual(out["after"]["local_position"], [1, 0, 10])

    def test_missing_transform_and_null_modifier_return_unchanged(self):
        self.assertEqual(call(bone_exists=False)["branches"], ["return_missing_transform"])
        self.assertEqual(call(coordinate_modifiers=[None])["branches"], ["return_null_modifier"])

    def test_hasbaseline_is_not_a_canapply_guard(self):
        cache = state()
        cache["_hasBaseline"] = False
        out = call(cache=cache, coordinate_modifiers=[mod(position=[1, 0, 0])])
        self.assertEqual(out["after"]["local_position"], [1, 0, 10])

    def test_incomplete_cache_nan_and_bool_integer_refused(self):
        cache = state()
        del cache["_positionBaseline"]
        with self.assertRaisesRegex(ReplayRejected, "private cache"):
            call(cache=cache)
        with self.assertRaisesRegex(ReplayRejected, "finite"):
            call(coordinate_modifiers=[mod(length=float("nan"))])
        with self.assertRaisesRegex(ReplayRejected, "bool"):
            call(rotation_excluded=0)


if __name__ == "__main__":
    unittest.main()
