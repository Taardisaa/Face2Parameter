"""Native sampling contract tests, independent of the running game."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.hs2_deform_torch import TorchHeadRig
from src.hs2_sampling import (
    rotation_is_exempt,
    sample_keyframes,
    unlocker_rotation_delta,
)

FRAMES = [
    {"pos": [1, 2, 3], "rot": [350, 10, 20], "scl": [1, 1, 1]},
    {"pos": [11, -12, 7], "rot": [355, 5, 20], "scl": [3, 5, 2]},
    {"pos": [4, -1, 9], "rot": [5, 350, 10], "scl": [2, 0.5, 1.5]},
]


def synthetic_rig(*, one_key=False, all_exempt=False):
    names = ["cf_s_Mouth_L" if all_exempt else "cf_s_Test", "cf_s_Mouth_R"]
    return SimpleNamespace(
        sampling_profile="vanilla",
        _topo=["root"],
        bones={
            "root": {
                "name": "cf_J_Test",
                "parent": "",
                "pos": [0, 0, 0],
                "rot": [0, 0, 0, 1],
                "scale": [1, 1, 1],
            }
        },
        enums={"src": names, "dst": ["cf_J_Test"]},
        eqns=[],
        customhead=[
            {"bone": name, "category": i, "use": [1] * 9}
            for i, name in enumerate(names)
        ],
        anm={name: copy.deepcopy(FRAMES[:1] if one_key else FRAMES) for name in names},
        skin_bone_names=["cf_J_Test"],
        bindpose=np.eye(4)[None],
        verts=np.array([[1.0, 2.0, 3.0]]),
        faces=np.array([[0, 0, 0]]),
        bone_idx=np.zeros((1, 4), int),
        bone_w=np.array([[1.0, 0, 0, 0]]),
    )


def expected_sources(rig, rates, profile):
    fields = [[], [], []]
    for row in rig.customhead:
        values = sample_keyframes(
            rig.anm[row["bone"]],
            rates[row["category"]],
            profile=profile,
            bone_name=row["bone"],
        )
        for fi in range(3):
            fields[fi].extend(values[fi])
    return np.concatenate(fields)


def legacy_sample(frames, rate):
    """Frozen pre-change scalar implementation to catch default regressions."""
    if rate <= 0:
        f = frames[0]
        return f["pos"], f["rot"], f["scl"]
    if rate >= 1:
        f = frames[-1]
        return f["pos"], f["rot"], f["scl"]
    x = (len(frames) - 1) * rate
    i = int(np.floor(x))
    t = x - i
    a, b = frames[i], frames[i + 1]
    pos = [a["pos"][k] * (1 - t) + b["pos"][k] * t for k in range(3)]
    scl = [a["scl"][k] * (1 - t) + b["scl"][k] * t for k in range(3)]
    rot = []
    for k in range(3):
        da = a["rot"][k] % 360
        db = b["rot"][k] % 360
        rot.append(da + ((db - da + 540) % 360 - 180) * t)
    return pos, rot, scl


class SamplingContracts(unittest.TestCase):
    def test_default_matches_frozen_legacy_exactly(self):
        rng = np.random.default_rng(713)
        for rate in [-1.0, 0.0, 0.5, 1.0, 2.0, *rng.uniform(-0.5, 1.5, 30)]:
            np.testing.assert_array_equal(
                sample_keyframes(FRAMES, rate), legacy_sample(FRAMES, rate)
            )
            if 0 <= rate <= 1:
                np.testing.assert_array_equal(
                    sample_keyframes(FRAMES, rate, profile="slider_unlocker_18_2"),
                    legacy_sample(FRAMES, rate),
                )

    def test_unlocker_global_endpoint_not_adjacent_extension(self):
        for rate in [-0.5, 1.5]:
            pos, _, scl = sample_keyframes(FRAMES, rate, profile="slider_unlocker_18_2")
            np.testing.assert_allclose(
                pos, np.asarray(FRAMES[0]["pos"]) + np.array([3, -3, 6]) * rate
            )
            np.testing.assert_allclose(
                scl, np.ones(3) + np.array([1, -0.5, 0.5]) * rate
            )
        np.testing.assert_array_equal(unlocker_rotation_delta(FRAMES), [15, -20, 350])
        np.testing.assert_array_equal(
            sample_keyframes(FRAMES, -0.5, profile="slider_unlocker_18_2")[1],
            [342.5, 20, -155],
        )
        np.testing.assert_array_equal(
            sample_keyframes(FRAMES, 1.5, profile="slider_unlocker_18_2")[1],
            [12.5, 340, 185],
        )

    def test_all_six_exception_predicates_preserve_endpoint_rotation(self):
        names = [
            "cf_s_Mune_x",
            "cf_s_Mouth_L",
            "cf_s_LegLow_L",
            "cf_s_MayuTip_R",
            "xthigh01x",
            "cf_a_bust01_size",
        ]
        for name in names:
            self.assertTrue(rotation_is_exempt(name))
            for rate in [-0.5, 1.5]:
                p, r, s = sample_keyframes(
                    FRAMES, rate, profile="slider_unlocker_18_2", bone_name=name
                )
                np.testing.assert_array_equal(r, FRAMES[0 if rate < 0 else -1]["rot"])
                self.assertNotEqual(p, FRAMES[0 if rate < 0 else -1]["pos"])
        self.assertFalse(rotation_is_exempt("CF_S_Mouth_L"))
        self.assertFalse(rotation_is_exempt("cf_a_bust01_other"))

    def test_torch_numpy_batched_three_keyframes_and_exceptions(self):
        rig = synthetic_rig()
        rates = torch.tensor(
            [[-0.5, 1.5], [0.21, 0.79], [1.5, -0.5]], dtype=torch.float64
        )
        for profile in ["vanilla", "slider_unlocker_18_2"]:
            trig = TorchHeadRig(
                rig, device="cpu", dtype=torch.float64, sampling_profile=profile
            )
            output = trig.source_values(rates).numpy()
            for row, rate in zip(output, rates.numpy()):
                expected = expected_sources(rig, rate, profile)
                if profile == "vanilla":
                    # Preserve the old Torch endpoint Euler representation (e.g. 365 vs 5).
                    delta = row - expected
                    delta[6:12] = (delta[6:12] + 180) % 360 - 180
                    np.testing.assert_allclose(delta, 0, atol=1e-12)
                else:
                    np.testing.assert_allclose(row, expected, atol=1e-12)
            if profile == "slider_unlocker_18_2":
                self.assertTrue(
                    torch.autograd.gradcheck(
                        trig.source_values,
                        (rates.requires_grad_(True),),
                        eps=1e-6,
                        atol=1e-7,
                        rtol=1e-5,
                    )
                )

    def test_torch_profile_inheritance_and_override(self):
        rig = synthetic_rig()
        rig.sampling_profile = "slider_unlocker_18_2"
        self.assertEqual(
            "slider_unlocker_18_2", TorchHeadRig(rig, device="cpu").sampling_profile
        )
        self.assertEqual(
            "vanilla",
            TorchHeadRig(
                rig, device="cpu", sampling_profile="vanilla"
            ).sampling_profile,
        )

    def test_finite_guards_and_unknown_profile(self):
        for rate in [float("nan"), float("inf"), -float("inf")]:
            with self.assertRaises(ValueError):
                sample_keyframes(FRAMES, rate)
            with self.assertRaises(ValueError):
                TorchHeadRig(synthetic_rig(), device="cpu").source_values(
                    torch.tensor([[rate, 0.5]])
                )
        with self.assertRaises(ValueError):
            sample_keyframes(FRAMES, 0.5, profile="guess")
        with self.assertRaises(ValueError):
            sample_keyframes([], 0.5)
        frames = copy.deepcopy(FRAMES)
        frames[1]["pos"][0] = float("nan")
        with self.assertRaises(ValueError):
            sample_keyframes(frames, 0.5)
        rig = synthetic_rig()
        rig.anm["cf_s_Test"] = frames
        with self.assertRaises(ValueError):
            TorchHeadRig(rig, device="cpu")

    def test_single_key_safe_and_undefined_unlocker_rejected(self):
        for rate in [-0.5, 0.5, 1.5]:
            np.testing.assert_array_equal(
                sample_keyframes(FRAMES[:1], rate),
                [FRAMES[0][k] for k in ["pos", "rot", "scl"]],
            )
            sample_keyframes(
                FRAMES[:1],
                rate,
                profile="slider_unlocker_18_2",
                bone_name="cf_s_Mouth_L",
            )
        rig = synthetic_rig(one_key=True, all_exempt=True)
        for profile in ["vanilla", "slider_unlocker_18_2"]:
            trig = TorchHeadRig(
                rig, device="cpu", dtype=torch.float64, sampling_profile=profile
            )
            rates = torch.tensor([[-0.5, 1.5]], dtype=torch.float64, requires_grad=True)
            np.testing.assert_allclose(
                trig.source_values(rates).detach().numpy()[0],
                expected_sources(rig, rates.detach().numpy()[0], profile),
            )
            self.assertTrue(torch.autograd.gradcheck(trig.source_values, (rates,)))
        with self.assertRaises(ValueError):
            sample_keyframes(FRAMES[:1], -0.5, profile="slider_unlocker_18_2")
        with self.assertRaises(ValueError):
            TorchHeadRig(
                synthetic_rig(one_key=True),
                device="cpu",
                sampling_profile="slider_unlocker_18_2",
            ).source_values(torch.tensor([[-0.5, 1.5]]))


if __name__ == "__main__":
    unittest.main()
