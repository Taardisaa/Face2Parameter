"""Independent installed HS2ABMX 4.4.6 BoneModifier.Apply state transition.

This models one observed call, not a timeless native59->mesh approximation.
Unity float32 arithmetic and Vector3 approximate-zero semantics are explicit.
"""
from __future__ import annotations

import copy
import numpy as np

F = np.float32
FLAG_FIELDS = ("_hasBaseline", "_lenModForceUpdate", "_lenModNeedsPositionRestore",
               "_changedScale", "_changedRotation", "_changedPosition", "_forceApply")
VECTOR_FIELDS = ("_sclBaseline", "_posBaseline", "_positionBaseline")
CACHE_FIELDS = set(FLAG_FIELDS + VECTOR_FIELDS + ("_rotBaseline", "_lenBaseline"))
IDENTITY = {"scale": [1., 1., 1.], "length": 1., "position": [0., 0., 0.], "rotation": [0., 0., 0.]}


class ReplayRejected(ValueError):
    """Incomplete/unsafe evidence cannot be called a replay certification."""


def require(value, message):
    if not value:
        raise ReplayRejected(message)


def finite(value, shape, label):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=F)
    require(result.shape == shape and np.isfinite(result).all(), "Invalid finite " + label)
    return result


def bool_value(value, label):
    require(type(value) is bool, "Missing or invalid bool " + label)
    return value


def cache_state(value):
    require(isinstance(value, dict) and set(value) == CACHE_FIELDS, "Incomplete/unknown private cache fields")
    result = {key: bool_value(value[key], key) for key in FLAG_FIELDS}
    result.update({key: finite(value[key], (3,), key).copy() for key in VECTOR_FIELDS})
    result["_rotBaseline"] = finite(value["_rotBaseline"], (4,), "_rotBaseline").copy()
    result["_lenBaseline"] = finite(value["_lenBaseline"], (), "_lenBaseline").item()
    return result


def transform(value):
    require(isinstance(value, dict) and set(value) == {"local_position", "local_rotation_xyzw", "local_scale"}, "Missing/unknown raw local TRS fields")
    return {"local_position": finite(value["local_position"], (3,), "position").copy(),
            "local_rotation_xyzw": finite(value["local_rotation_xyzw"], (4,), "rotation").copy(),
            "local_scale": finite(value["local_scale"], (3,), "scale").copy()}


def modifier(value):
    require(isinstance(value, dict) and set(value) == set(IDENTITY), "Incomplete/unknown modifier fields")
    return {"scale": finite(value["scale"], (3,), "modifier scale"),
            "length": F(finite(value["length"], (), "modifier length")),
            "position": finite(value["position"], (3,), "modifier position"),
            "rotation": finite(value["rotation"], (3,), "modifier rotation")}


def unity_sqrmag(vector):
    # Actual installed UnityEngine.Vector3.SqrMagnitude evaluation order.
    return F(F(F(vector[0] * vector[0]) + F(vector[1] * vector[1])) + F(vector[2] * vector[2]))


def unity_zero(vector):
    return bool(unity_sqrmag(vector) < F(9.9999994e-11))


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by + ay*bw + az*bx - ax*bz,
                     aw*bz + az*bw + ax*by - ay*bx,
                     aw*bw - ax*bx - ay*by - az*bz], dtype=F)


def euler_zxy(value):
    # Quaternion.Euler multiplies degree vector by float(pi/180) before native
    # FromEulerRad; native's equivalent Z-X-Y product is independently computed.
    radians = value * F(np.pi / 180)
    quaternions = []
    for axis, angle in enumerate(radians):
        half = F(angle * F(.5))
        q = np.array([0, 0, 0, np.cos(half)], dtype=F)
        q[axis] = F(np.sin(half))
        quaternions.append(q)
    return qmul(quaternions[1], qmul(quaternions[0], quaternions[2]))


def serialized(value):
    if isinstance(value, dict):
        return {key: serialized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialized(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def effective_modifier(coordinate_modifiers, coordinate, additional_modifiers, coordinate_specific):
    require(type(coordinate) is int and coordinate >= 0, "Invalid coordinate index")
    require(isinstance(coordinate_modifiers, list) and len(coordinate_modifiers) > 0, "Missing coordinate modifiers")
    require(isinstance(additional_modifiers, list), "Additional modifiers must be explicitly recorded, including []")
    bool_value(coordinate_specific, "coordinate_specific")
    require(coordinate_specific is False, "Installed HS2 IsCoordinateSpecific is false; incompatible variant refused")
    # Installed HS2 method returns false even when CoordinateModifiers has >1.
    selected = coordinate_modifiers[0]
    base = modifier(selected) if selected is not None else None
    if not additional_modifiers:
        return base
    base = copy.deepcopy(base) if base is not None else modifier(IDENTITY)
    for value in additional_modifiers:
        add = modifier(value)
        base["scale"] = base["scale"] * add["scale"]
        base["length"] = F(base["length"] * add["length"])
        base["position"] = base["position"] + add["position"]
        base["rotation"] = base["rotation"] + add["rotation"]
    require(all(np.isfinite(np.asarray(value)).all() for value in base.values()), "Combined modifier overflowed")
    return base


def replay_apply(*, before, cache, coordinate_modifiers, coordinate, additional_modifiers,
                 bone_exists, rotation_excluded, is_during_h_scene, coordinate_specific=False):
    """Return predicted after TRS/cache and exact branch decisions for one call."""
    local = transform(before)
    state = cache_state(cache)
    for key, value in (("bone_exists", bone_exists), ("rotation_excluded", rotation_excluded),
                       ("is_during_h_scene", is_during_h_scene)):
        bool_value(value, key)
    decisions = []
    effective = None

    def result():
        require(all(np.isfinite(value).all() for value in local.values()), "Predicted local transform is nonfinite")
        return serialized({"after": local, "cache_after": state, "effective_modifier": effective, "branches": decisions})

    if not bone_exists:
        decisions.append("return_missing_transform")
        return result()
    effective = effective_modifier(coordinate_modifiers, coordinate, additional_modifiers, coordinate_specific)
    if effective is None:
        decisions.append("return_null_modifier")
        return result()
    has_scale = bool(np.any(effective["scale"] != F(1)))
    has_length = bool(effective["length"] != F(1))
    has_position = bool(np.any(effective["position"] != F(0)))
    has_rotation = bool(np.any(effective["rotation"] != F(0)))
    nonempty = has_scale or has_length or has_position or has_rotation
    if nonempty:
        state["_forceApply"] = True
        decisions.append("CanApply_nonempty")
    elif state["_forceApply"]:
        state["_forceApply"] = False
        state["_lenModForceUpdate"] = True
        decisions.append("CanApply_force_to_length_restore")
    elif state["_lenModForceUpdate"]:
        decisions.append("CanApply_pending_length_restore")
    else:
        decisions.append("return_CanApply_false")
        return result()
    if has_scale:
        local["local_scale"] = state["_sclBaseline"] * effective["scale"]
        state["_changedScale"] = True
        decisions.append("scale_from_cached_baseline")
    elif state["_changedScale"]:
        local["local_scale"] = state["_sclBaseline"].copy()
        state["_changedScale"] = False
        decisions.append("restore_cached_scale")
    if has_rotation and not rotation_excluded:
        local["local_rotation_xyzw"] = qmul(state["_rotBaseline"], euler_zxy(effective["rotation"]))
        require(float(np.linalg.norm(local["local_rotation_xyzw"])) > 0, "Cannot certify Unity setter of zero quaternion")
        state["_changedRotation"] = True
        decisions.append("rotation_baseline_times_euler_zxy")
    elif state["_changedRotation"]:
        local["local_rotation_xyzw"] = state["_rotBaseline"].copy()
        state["_changedRotation"] = False
        decisions.append("restore_cached_rotation_including_excluded_bone")
    if (state["_lenModForceUpdate"] or has_length) and not unity_zero(state["_positionBaseline"]):
        direction = local["local_position"].copy()
        if effective["length"] < F(.1) or unity_zero(direction) or is_during_h_scene:
            direction = state["_positionBaseline"].copy()
            state["_lenModNeedsPositionRestore"] = True
            decisions.append("length_use_historical_direction")
        else:
            decisions.append("length_use_current_direction")
        magnitude = F(np.sqrt(unity_sqrmag(direction)))
        require(np.isfinite(magnitude) and magnitude > 0, "Cannot replay zero/overflow normalization divisor")
        local["local_position"] = ((direction / magnitude) * F(state["_lenBaseline"])) * effective["length"]
        state["_lenModForceUpdate"] = False
        if has_position:
            local["local_position"] = local["local_position"] + effective["position"]
            state["_changedPosition"] = True
            decisions.append("length_then_add_position")
        elif state["_changedPosition"]:
            # Installed code overwrites the length result, not merely subtracts offset.
            local["local_position"] = state["_posBaseline"].copy()
            state["_changedPosition"] = False
            decisions.append("length_result_overwritten_by_position_restore")
    elif has_position:
        local["local_position"] = state["_posBaseline"] + effective["position"]
        state["_changedPosition"] = True
        decisions.append("position_only_cached_baseline_plus_offset")
    elif state["_changedPosition"]:
        local["local_position"] = state["_posBaseline"].copy()
        state["_changedPosition"] = False
        decisions.append("restore_cached_position")
    else:
        decisions.append("position_unchanged")
    return result()
