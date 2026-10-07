"""Analytic state/gradient regressions, independent of observed after values."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from src.hs2_abmx_torch import apply_sequence, apply_transition, tensor_cache


def cache():
    return tensor_cache({"_hasBaseline": True, "_lenBaseline": 5.,
                         "_sclBaseline": [1., 1., 1.], "_posBaseline": [0., 3., 4.],
                         "_rotBaseline": [0., 0., 0., 1.], "_positionBaseline": [0., 3., 4.],
                         "_lenModForceUpdate": False, "_lenModNeedsPositionRestore": False,
                         "_changedScale": False, "_changedRotation": False,
                         "_changedPosition": False, "_forceApply": False})


def local(position=(0., 3., 4.)):
    return torch.tensor([position]), torch.tensor([[0., 0., 0., 1.]]), torch.ones(1, 3)


class TransitionTests(unittest.TestCase):
    def test_two_calls_change_direction_and_propagate_flags(self):
        modifier = torch.tensor([[1., 1., 1., 2., 1., 0., 0., 0., 0., 0.]])
        first, state = apply_transition(local(), cache(), modifier)
        torch.testing.assert_close(first[0], torch.tensor([[1., 6., 8.]]))
        second, state = apply_transition(first, state, modifier)
        expected = torch.tensor([[1 + 10 / math.sqrt(101), 60 / math.sqrt(101), 80 / math.sqrt(101)]])
        torch.testing.assert_close(second[0], expected)
        self.assertTrue(state["_changedPosition"].item())
        self.assertTrue(state["_forceApply"].item())
        self.assertFalse(state["_lenModForceUpdate"].item())

    def test_neutral_restore_overwrites_length_result(self):
        state = cache()
        state["_forceApply"][:] = True
        state["_changedPosition"][:] = True
        neutral = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]])
        after, state = apply_transition(local((9., 8., 7.)), state, neutral)
        torch.testing.assert_close(after[0], torch.tensor([[0., 3., 4.]]))
        self.assertFalse(state["_changedPosition"].item())
        self.assertFalse(state["_forceApply"].item())

    def test_canapply_false_does_not_restore_changed_flag(self):
        state = cache()
        state["_changedPosition"][:] = True
        after, state = apply_transition(local((9., 8., 7.)), state,
                                         torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]]))
        torch.testing.assert_close(after[0], torch.tensor([[9., 8., 7.]]))
        self.assertTrue(state["_changedPosition"].item())

    def test_short_length_fallback_uses_historical_direction(self):
        modifier = torch.tensor([[1., 1., 1., .05, 0., 0., 0., 0., 0., 0.]])
        after, state = apply_transition(local((1., 0., 0.)), cache(), modifier)
        torch.testing.assert_close(after[0], torch.tensor([[0., .15, .2]]))
        self.assertTrue(state["_lenModNeedsPositionRestore"].item())

    def test_inactive_zero_length_branch_has_finite_backward(self):
        state = cache()
        state["_positionBaseline"][:] = 0
        position = torch.zeros(1, 3, requires_grad=True)
        modifier = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
        after, _ = apply_transition((position, local()[1], local()[2]), state, modifier)
        sum(value.sum() for value in after).backward()
        self.assertTrue(torch.isfinite(position.grad).all())
        self.assertTrue(torch.isfinite(modifier.grad).all())
        torch.testing.assert_close(position.grad, torch.ones_like(position))

    def test_count_is_explicit_integer_and_cache_cannot_be_invented(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            apply_sequence(local(), cache(), torch.ones(1, 10), True)
        with self.assertRaisesRegex(ValueError, "cache"):
            tensor_cache({})

    def test_direct_nonfinite_cache_is_rejected_before_backward(self):
        state = cache()
        state["_lenBaseline"][:] = float("nan")
        with self.assertRaisesRegex(ValueError, "cache"):
            apply_transition(local(), state, torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]]))

    def test_large_finite_inactive_length_does_not_overflow_gradient(self):
        state = cache()
        state["_positionBaseline"][:] = 0
        state["_lenBaseline"][:] = 1e20
        position = torch.full((1, 3), 1e20, requires_grad=True)
        modifier = torch.tensor([[1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
        after, _ = apply_transition((position, local()[1], local()[2]), state, modifier)
        sum(value.sum() for value in after).backward()
        self.assertTrue(torch.isfinite(modifier.grad).all())
        self.assertTrue(torch.isfinite(position.grad).all())


if __name__ == "__main__":
    unittest.main()
