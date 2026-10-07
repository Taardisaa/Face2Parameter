"""Explicit native-animation sampling contracts; no ML normalization changes."""

from __future__ import annotations

from typing import Literal

import numpy as np

SamplingProfile = Literal["vanilla", "slider_unlocker_18_2"]
SAMPLING_PROFILES = ("vanilla", "slider_unlocker_18_2")


def validate_keyframes(frames):
    if not frames:
        raise ValueError("Animation needs at least one keyframe")
    for frame in frames:
        for field in ("pos", "rot", "scl"):
            values = np.asarray(frame[field], dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Animation {field} keyframes must be finite 3-vectors"
                )


def validate_sampling_profile(profile: str) -> SamplingProfile:
    if profile not in SAMPLING_PROFILES:
        raise ValueError(
            f"Unknown sampling profile {profile!r}; choose {SAMPLING_PROFILES}"
        )
    return profile


def rotation_is_exempt(name: str) -> bool:
    """SliderUnlocker 18.2 SafeCalculateRotation's case-sensitive predicates."""
    return (
        any(
            part in name
            for part in ("cf_s_Mune", "cf_s_Mouth", "cf_s_LegLow", "cf_s_MayuTip")
        )
        or ("thigh" in name and "01" in name)
        or (name.startswith("cf_a_bust") and name.endswith("_size"))
    )


def unlocker_rotation_delta(frames) -> np.ndarray:
    """Global endpoint difference unwrapped in the raw first-segment direction.

    This deliberately does NOT use shortest-arc interpolation or modulo angles.
    Equal first/second angles imply a positive direction in the installed plugin.
    """
    if len(frames) < 2:
        raise ValueError(
            "SliderUnlocker rotation extrapolation requires at least two keyframes"
        )
    first = np.asarray(frames[0]["rot"], dtype=np.float64)
    direction = np.asarray(frames[1]["rot"], dtype=np.float64) - first >= 0
    delta = np.asarray(frames[-1]["rot"], dtype=np.float64) - first
    return np.where(
        (delta > 0) & ~direction,
        delta - 360,
        np.where((delta < 0) & direction, delta + 360, delta),
    )


def sample_keyframes(
    frames, rate, *, profile="vanilla", bone_name="", sample_rotation=True
):
    """Return (position, Euler rotation, scale) with an explicit runtime profile.

    Vanilla interpolation is unchanged, including endpoint clamping. The unlocked
    profile substitutes only rate<0 or rate>1, matching prefix clamp + postfix.
    Single-key unlocked rotation is rejected unless exempt/unused: the actual
    plugin indexes its second keyframe and cannot evaluate this case safely.
    """
    validate_sampling_profile(profile)
    rate = float(rate)
    if not np.isfinite(rate):
        raise ValueError("Animation sampling rate must be finite")
    validate_keyframes(frames)
    n = len(frames)
    if profile == "slider_unlocker_18_2" and (rate < 0 or rate > 1):
        first, last = frames[0], frames[-1]
        pos = np.asarray(first["pos"]) + (np.asarray(last["pos"]) - first["pos"]) * rate
        scl = np.asarray(first["scl"]) + (np.asarray(last["scl"]) - first["scl"]) * rate
        endpoint = first if rate < 0 else last
        if not sample_rotation or rotation_is_exempt(bone_name):
            rot = endpoint["rot"]
        else:
            delta = unlocker_rotation_delta(frames)
            rot = (
                np.asarray(first["rot"]) + delta * rate
                if rate < 0
                else np.asarray(last["rot"]) + delta * (rate - 1)
            )
        return list(pos), list(rot), list(scl)
    if rate <= 0 or n == 1:
        f = frames[0]
        return f["pos"], f["rot"], f["scl"]
    if rate >= 1:
        f = frames[-1]
        return f["pos"], f["rot"], f["scl"]
    x = (n - 1) * rate
    i = int(np.floor(x))
    t = x - i
    a, b = frames[i], frames[i + 1]
    pos = [a["pos"][k] * (1 - t) + b["pos"][k] * t for k in range(3)]
    scl = [a["scl"][k] * (1 - t) + b["scl"][k] * t for k in range(3)]
    rot = []
    for k in range(3):
        da = a["rot"][k] % 360
        db = b["rot"][k] % 360
        diff = (db - da + 540) % 360 - 180
        rot.append(da + diff * t)
    return pos, rot, scl
