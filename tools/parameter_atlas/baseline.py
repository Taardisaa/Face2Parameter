"""Strict native baselines and actual probe intervals; no game or torch required."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def native_values(values, *, count=59):
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"Baseline must contain exactly {count} native controls")
    if any(type(value) not in (int, float) for value in values):
        raise ValueError(
            "Baseline controls must be JSON numbers, not strings or booleans"
        )
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Baseline controls must all be finite")
    return result


def baseline_digest(values):
    payload = json.dumps(values.tolist(), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_baseline(path=None):
    """Accept native JSON or a bridge geometry snapshot's explicit character fields."""
    if path is None:
        values = np.full(59, 0.5, dtype=np.float64)
        return values, {
            "kind": "default_constant",
            "native_input_sha256": baseline_digest(values),
            "head_id": None,
        }
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"))
    head_id = None
    if isinstance(document, list):
        source = document
        field = "root array"
    elif isinstance(document, dict) and "native_input" in document:
        source = document["native_input"]
        head_id = document.get("head_id")
        field = "native_input"
    elif isinstance(document, dict) and isinstance(document.get("character"), dict):
        source = document["character"].get("shape_value_face")
        head_id = document["character"].get("head_id")
        field = "character.shape_value_face"
    else:
        raise ValueError(
            "Use a 59-number array, native_input object, or bridge geometry snapshot"
        )
    if head_id is not None and (type(head_id) is not int or head_id < 0):
        raise ValueError("Recorded head_id must be a nonnegative JSON integer")
    values = native_values(source)
    return values, {
        "kind": "source_json",
        "source_path": str(path),
        "source_file_sha256": hashlib.sha256(raw).hexdigest(),
        "source_field": field,
        "native_input_sha256": baseline_digest(values),
        "head_id": head_id,
        "consumed_state": "native face shape values only; excludes ABMX, live expression and pose",
    }


def probe_inputs(baseline, step):
    """Preserve supplied coefficients outside 0..1; the rig owns sampling semantics."""
    baseline = np.asarray(baseline, dtype=np.float64)
    if baseline.ndim != 1 or not np.isfinite(baseline).all():
        raise ValueError("Finite one-dimensional baseline required")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Positive finite step required")
    count = len(baseline)
    probes = np.repeat(baseline[None], count * 2, axis=0)
    indices = np.arange(count)
    probes[indices, indices] -= step
    probes[count + indices, indices] += step
    left_steps = baseline - probes[indices, indices]
    right_steps = probes[count + indices, indices] - baseline
    if (
        not np.isfinite(probes).all()
        or np.any(left_steps <= 0)
        or np.any(right_steps <= 0)
    ):
        raise ValueError(
            "Step is not representable at this baseline; choose a larger step"
        )
    return probes, left_steps, right_steps


def probe_jacobian(baseline, minus, plus, left_steps, right_steps):
    """Use the actual input intervals, including one-sided clamped surface responses."""
    baseline = np.asarray(baseline, dtype=np.float64)
    minus, plus = (
        np.asarray(minus, dtype=np.float64),
        np.asarray(plus, dtype=np.float64),
    )
    left_steps, right_steps = np.asarray(left_steps), np.asarray(right_steps)
    if (
        baseline.ndim != 2
        or baseline.shape[1] != 3
        or minus.ndim != 3
        or minus.shape != plus.shape
        or minus.shape[1:] != baseline.shape
        or left_steps.shape != (len(minus),)
        or right_steps.shape != (len(minus),)
        or not all(
            np.isfinite(value).all()
            for value in [baseline, minus, plus, left_steps, right_steps]
        )
        or np.any(left_steps <= 0)
        or np.any(right_steps <= 0)
    ):
        raise ValueError(
            "Matching finite geometry and positive per-control probe intervals required"
        )
    left = (baseline[None] - minus) / left_steps[:, None, None]
    right = (plus - baseline[None]) / right_steps[:, None, None]
    center = (plus - minus) / (left_steps + right_steps)[:, None, None]
    return tuple(value.reshape(len(minus), -1).T for value in (center, left, right))
