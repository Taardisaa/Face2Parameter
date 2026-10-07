"""Differentiable, explicit-state HS2ABMX 4.4.6 Apply transition.

Cache and local TRS are inputs and outputs. This is never a timeless p*length
deformation. Runtime provenance/complete observation checks live in abmx_replay.
Tensor layout: scale xyz, length, position xyz, Euler rotation xyz.
"""
from __future__ import annotations

import torch

from .hs2_deform_torch import euler_zxy_quat, qmul

FLAGS = ("_hasBaseline", "_lenModForceUpdate", "_lenModNeedsPositionRestore",
         "_changedScale", "_changedRotation", "_changedPosition", "_forceApply")
VECTORS = {"_sclBaseline": 3, "_posBaseline": 3, "_positionBaseline": 3, "_rotBaseline": 4}


def tensor_cache(fields, *, device="cpu", dtype=torch.float32, batch=1):
    """Parse recorded private fields; never invent a length baseline or flags."""
    expected = set(FLAGS) | set(VECTORS) | {"_lenBaseline"}
    if not isinstance(fields, dict) or set(fields) != expected:
        raise ValueError("Complete explicit ABMX private cache required")
    result = {}
    for key in FLAGS:
        if type(fields[key]) is not bool:
            raise ValueError("ABMX flag must be bool: " + key)
        result[key] = torch.full((batch,), fields[key], device=device, dtype=torch.bool)
    for key, length in VECTORS.items():
        value = torch.as_tensor(fields[key], device=device, dtype=dtype)
        if value.shape != (length,) or not torch.isfinite(value).all():
            raise ValueError("Invalid cache vector: " + key)
        result[key] = value[None].expand(batch, -1)
    value = torch.as_tensor(fields["_lenBaseline"], device=device, dtype=dtype)
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("Invalid length baseline")
    result["_lenBaseline"] = value.expand(batch)
    return result


def _zero(vector):
    # Installed Unity Vector3 equality: ordered float32 sum, near-zero threshold.
    square = vector * vector
    return ((square[..., 0] + square[..., 1]) + square[..., 2]) < 9.9999994e-11


def apply_transition(local, cache, modifier, *, rotation_excluded=False,
                     is_during_h_scene=False):
    """One batched resolved Apply. Return new local TRS and private cache.

    Missing-transform/null-modifier early returns are handled by the caller.
    No additional modifiers are silently supplied; caller must resolve them.
    Branch selection follows installed exact IsEmpty/CanApply predicates.
    """
    pos, quat, scale = local
    if modifier.ndim != 2 or modifier.shape[1] != 10 or pos.shape != (len(modifier), 3):
        raise ValueError("Expected batched ABMX modifier (B,10) and local TRS")
    if quat.shape != (len(modifier), 4) or scale.shape != pos.shape:
        raise ValueError("Invalid local TRS shape")
    if set(cache) != set(FLAGS) | set(VECTORS) | {"_lenBaseline"}:
        raise ValueError("Incomplete ABMX cache")
    for key, value in cache.items():
        expected = (len(modifier), VECTORS[key]) if key in VECTORS else (len(modifier),)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected or value.device != pos.device:
            raise ValueError("Invalid explicit cache tensor: " + key)
        if key in FLAGS:
            if value.dtype != torch.bool:
                raise ValueError("Cache flags require bool tensors: " + key)
        elif not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError("Nonfinite/invalid explicit cache value: " + key)
    if any(not torch.isfinite(value).all() for value in (pos, quat, scale, modifier)):
        raise ValueError("Nonfinite ABMX input")
    state = dict(cache)
    has_scale = (modifier[:, :3] != 1).any(1)
    has_length = modifier[:, 3] != 1
    has_pos = (modifier[:, 4:7] != 0).any(1)
    has_rot = (modifier[:, 7:] != 0).any(1)
    nonempty = has_scale | has_length | has_pos | has_rot
    was_force = state["_forceApply"]
    apply = nonempty | was_force | state["_lenModForceUpdate"]
    len_force = state["_lenModForceUpdate"] | (~nonempty & was_force)
    state["_forceApply"] = nonempty

    def choose(condition, left, right):
        return torch.where(condition[:, None], left, right)

    scaled = choose(has_scale, state["_sclBaseline"] * modifier[:, :3],
                    choose(state["_changedScale"], state["_sclBaseline"], scale))
    state["_changedScale"] = torch.where(apply, has_scale, state["_changedScale"])
    rotation_allowed = has_rot & ~torch.as_tensor(rotation_excluded, device=pos.device, dtype=torch.bool)
    delta = euler_zxy_quat(modifier[:, 7], modifier[:, 8], modifier[:, 9])
    rotated = choose(rotation_allowed, qmul(state["_rotBaseline"], delta),
                     choose(state["_changedRotation"], state["_rotBaseline"], quat))
    state["_changedRotation"] = torch.where(apply, rotation_allowed, state["_changedRotation"])

    length_branch = (len_force | has_length) & ~_zero(state["_positionBaseline"])
    fallback = (modifier[:, 3] < .1) | _zero(pos) | torch.as_tensor(is_during_h_scene, device=pos.device, dtype=torch.bool)
    direction = choose(fallback, state["_positionBaseline"], pos)
    # Mask operands as well as the output. Eagerly evaluating an inactive
    # branch with large finite inputs can overflow and contaminate gradients.
    safe_direction = choose(length_branch, direction, torch.zeros_like(direction))
    squares = safe_direction * safe_direction
    square_magnitude = (squares[:, 0] + squares[:, 1]) + squares[:, 2]
    # Mask before sqrt: sqrt(0)'s infinite derivative would otherwise leak NaN
    # through a non-selected torch.where branch during backward.
    divisor = torch.sqrt(torch.where(length_branch, square_magnitude,
                                    torch.ones_like(square_magnitude)))
    if ((divisor <= 0) | ~torch.isfinite(divisor)).any():
        raise ValueError("Invalid ABMX length normalization divisor")
    safe_baseline = torch.where(length_branch, state["_lenBaseline"], torch.ones_like(state["_lenBaseline"]))
    safe_length = torch.where(length_branch, modifier[:, 3], torch.ones_like(modifier[:, 3]))
    length_pos = (safe_direction / divisor[:, None] * safe_baseline[:, None]) * safe_length[:, None]
    length_pos = choose(has_pos, length_pos + modifier[:, 4:7],
                        choose(state["_changedPosition"], state["_posBaseline"], length_pos))
    ordinary_pos = choose(has_pos, state["_posBaseline"] + modifier[:, 4:7],
                          choose(state["_changedPosition"], state["_posBaseline"], pos))
    positioned = choose(length_branch, length_pos, ordinary_pos)
    state["_changedPosition"] = torch.where(apply, has_pos, state["_changedPosition"])
    state["_lenModForceUpdate"] = torch.where(apply, len_force & ~length_branch,
                                              state["_lenModForceUpdate"])
    state["_lenModNeedsPositionRestore"] = state["_lenModNeedsPositionRestore"] | (apply & length_branch & fallback)
    result = (choose(apply, positioned, pos), choose(apply, rotated, quat), choose(apply, scaled, scale))
    if any(not torch.isfinite(value).all() for value in result):
        raise ValueError("Nonfinite predicted ABMX local TRS")
    return result, state


def apply_sequence(local, cache, modifier, count, **kwargs):
    """Declared call count, never optimized to match a target or snapshot."""
    if type(count) is not int or count < 0 or count > 10000:
        raise ValueError("Explicit integer Apply count 0..10000 required")
    for _ in range(count):
        local, cache = apply_transition(local, cache, modifier, **kwargs)
    return local, cache
