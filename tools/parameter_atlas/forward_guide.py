"""Bounded parameter guidance using complete source-method evaluations.

This solves an explicit scalar target on a declared surface. It does not infer
deformation slopes, fit correction factors, or promise a globally optimal step.
"""
from __future__ import annotations

import copy
import math


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def guide(context, body, evaluate):
    if not isinstance(context, dict):
        raise TypeError("Capture a context in this viewer before guidance")
    if set(body) != {"request", "control", "surface_path", "target_max", "bound"}:
        raise ValueError("Guidance requires request, control, surface_path, target_max, bound")
    request, control = body["request"], body["control"]
    if not isinstance(request, dict) or not isinstance(control, dict):
        raise TypeError("request and control must be objects")
    if request.get("context_id") != context["context_id"]:
        raise ValueError("Guidance context expired; capture again")
    target = number(body["target_max"], "target_max")
    bound = number(body["bound"], "bound")
    if target <= 0:
        raise ValueError("target_max must be positive in returned game geometry units")
    surface_path = body["surface_path"]
    if surface_path not in {row["renderer_path"] for row in context["surfaces"]}:
        raise ValueError("Unknown captured surface")
    baseline = context["identity"]["native59"]
    native = control.get("kind") == "native"
    if native:
        index = control.get("index")
        if set(control) != {"kind", "index"} or type(index) is not int or not 0 <= index < 59:
            raise ValueError("Native control requires one index in 0..58")
        if request.get("bones") or request.get("native_values") is None:
            raise ValueError("Native guidance requires a complete native-only request")
        values = request["native_values"]
        if not isinstance(values, list) or len(values) != 59:
            raise ValueError("Complete native vector required")
        if any(number(v, "native value") != baseline[i] for i, v in enumerate(values) if i != index):
            raise ValueError("Guidance must preserve every other captured native parameter")
        current = baseline[index]
    else:
        name, channel = control.get("name"), control.get("channel")
        axis = control.get("axis")
        expected = {"kind", "name", "channel"} | ({"axis"} if channel != "length" else set())
        if control.get("kind") != "abmx" or set(control) != expected or channel not in {"scale", "length", "position", "rotation"}:
            raise ValueError("ABMX guidance requires one declared scalar channel")
        if channel != "length" and (type(axis) is not int or not 0 <= axis < 3):
            raise ValueError("ABMX vector axis must be 0..2")
        if name not in context["selected_abmx_bones"]:
            raise ValueError("Unsupported captured bone")
        if request.get("native_values") is not None:
            raise ValueError("ABMX guidance requires an ABMX-only request")
        if len(request.get("bones", [])) != 1 or request["bones"][0].get("name") != name:
            raise ValueError("ABMX guidance requires exactly the selected bone")
        patch = request["bones"][0]
        if set(patch) != {"name", channel}:
            raise ValueError("ABMX guidance changes only its selected channel")
        rows = (context["identity"].get("abmx_runtime") or {}).get("bones", [])
        data = next((row for row in rows if row["name"] == name), {})
        original = data.get(channel, 1.0 if channel == "length" else [1.0]*3 if channel == "scale" else [0.0]*3)
        if channel == "length":
            current = original
        else:
            value = patch[channel]
            if not isinstance(value, list) or len(value) != 3 or any(number(v, "bone axis") != original[i] for i, v in enumerate(value) if i != axis):
                raise ValueError("Guidance must preserve every other captured bone axis")
            current = original[axis]
    current = number(current, "captured value")
    if bound == current:
        raise ValueError("Search bound must differ from the captured value")

    def at(value):
        candidate = copy.deepcopy(request)
        if native:
            candidate["native_values"][index] = value
        elif channel == "length":
            candidate["bones"][0][channel] = value
        else:
            candidate["bones"][0][channel][axis] = value
        result = evaluate(candidate)
        if result.get("context_id") != context["context_id"] or result.get("live_state_unchanged") is not True:
            raise ValueError("Source evaluation/context isolation check failed")
        surface = next((row for row in result["surfaces"] if row["renderer_path"] == surface_path), None)
        if surface is None or not surface.get("delta"):
            raise ValueError("Source evaluation returned no selected surface")
        distances = [math.hypot(*(number(v, "surface delta") for v in point)) for point in surface["delta"]]
        return result, max(distances)

    _, origin_displacement = at(current)
    if origin_displacement != 0:
        raise ValueError("Same-protocol captured coefficients must give exactly zero delta")
    endpoint, endpoint_displacement = at(bound)
    best_value, best, best_displacement = bound, endpoint, endpoint_displacement
    status, iterations = "target_not_bracketed", 2
    if endpoint_displacement >= target:
        left, right = current, bound
        status = "residual_limit"
        for _ in range(12):
            if abs(best_displacement - target) <= target * .001:
                status = "target_bracketed"
                break
            middle = (left + right) / 2
            result, displacement = at(middle)
            iterations += 1
            if abs(displacement - target) < abs(best_displacement - target):
                best_value, best, best_displacement = middle, result, displacement
            if displacement < target:
                left = middle
            else:
                right = middle
        if abs(best_displacement - target) <= target * .001:
            status = "target_bracketed"
    return {
        "evaluation": best,
        "guidance": {
            "status": status, "candidate_value": best_value,
            "max_displacement": best_displacement, "target_max": target,
            "absolute_residual": abs(best_displacement - target), "iterations": iterations,
            "surface_path": surface_path, "captured_value": current, "bound": bound,
            "note": "Each candidate uses the full declared source-method protocol. Endpoint bracketing and at most 12 bisections; no global monotonicity, uniqueness, anatomical direction or optimal-step guarantee. Residual tolerance is 0.1% of the requested displacement, not a game-parity threshold.",
        },
    }
