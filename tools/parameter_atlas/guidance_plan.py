"""Predeclare independent interior game probes for directional step suggestions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from explore_live_response import canonical_hash


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_payload(catalog_path, entry):
    directory = Path(catalog_path).resolve().parent
    path = (directory / entry["file"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Data path escapes the explorer package")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("Explorer data SHA-256 mismatch")
    text = raw.decode("utf-8").strip()
    prefix, suffix = "window.HS2_RESPONSE(", ");"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("Unexpected explorer data wrapper")
    payload = json.loads(text[len(prefix) : -len(suffix)])
    for key in ("id", "kind", "baseline_name"):
        if payload[key] != entry[key]:
            raise ValueError("Catalog and data selection differ")
    return payload


def interior_suggestion(
    payload, baseline, sign, *, target_percent=0.01, max_probe_fraction=0.4
):
    if sign not in (-1, 1) or type(sign) is not int:
        raise ValueError("Direction must be -1 or +1")
    if not math.isfinite(target_percent) or target_percent <= 0:
        raise ValueError("Requested displacement percentage must be positive")
    if not math.isfinite(max_probe_fraction) or not 0 < max_probe_fraction < 1:
        raise ValueError("Independent probe must be strictly inside the training step")
    response = payload["local_response"]
    gain = response["left_gain" if sign < 0 else "right_gain"]
    training_step = response["minus_step" if sign < 0 else "plus_step"]
    diagonal = baseline["head_diagonal"]
    noise = baseline["repeat_drift_normalized"] * diagonal
    if not all(
        type(x) in (float, int) and math.isfinite(x)
        for x in (gain, training_step, diagonal, noise)
    ):
        raise ValueError("Response gains, steps, units and noise must be finite")
    if training_step <= 0 or diagonal <= 0 or noise < 0:
        raise ValueError("Degenerate local response")
    if gain <= 0:
        return {
            "supported": False,
            "reason": "No measured head response in this direction",
        }
    requested = target_percent / 100 * diagonal
    target = min(requested, gain * training_step * max_probe_fraction)
    if target <= 3 * noise:
        return {
            "supported": False,
            "reason": "Interior target does not exceed three times training repeat noise",
            "target_units": target,
            "noise_units": noise,
        }
    signed_step = sign * target / gain
    value = payload["baseline_level"] + signed_step
    if not response["valid_interval"][0] < value < response["valid_interval"][1]:
        raise ValueError("Suggested value is outside the local training interval")
    return {
        "supported": True,
        "signed_step": signed_step,
        "step": abs(signed_step),
        "value": value,
        "target_units": target,
        "target_percent": 100 * target / diagonal,
        "requested_target_percent": target_percent,
        "request_clipped": target < requested,
        "training_step": training_step,
        "training_gain": gain,
        "max_probe_fraction": max_probe_fraction,
    }


def make_plan(catalog_path, *, target_percent=0.01, max_probe_fraction=0.4):
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("complete") is not True:
        raise ValueError("Completed explorer required")
    if len({r["id"] for r in catalog["entries"]}) != len(catalog["entries"]):
        raise ValueError("Duplicate explorer entries")
    for name, baseline in catalog["baselines"].items():
        if canonical_hash(baseline["identity"]) != baseline["identity_sha256"]:
            raise ValueError("Baseline identity SHA-256 mismatch")
        controls = {
            r["control"]
            for r in catalog["entries"]
            if r["kind"] == "native" and r["baseline_name"] == name
        }
        if controls != set(range(59)):
            raise ValueError("Every baseline must cover all59 native head controls")
    plan = {
        "schema_version": 1,
        "catalog_receipt": receipt(catalog_path),
        "training_manifest_receipt": {
            "path": catalog["provenance"]["manifest_path"],
            "sha256": catalog["provenance"]["source_manifest_sha256"],
        },
        "baselines": catalog["baselines"],
        "entries": [],
        "skipped": [],
        "thresholds": {
            "geometry_normalized": 1e-5,
            "repeat_normalized": 1e-5,
            "relative_prediction_error": 0.05,
            "noise_multiplier": 3,
            "baseline_reuse_normalized": 1e-5,
        },
        "target_percent": target_percent,
        "max_probe_fraction": max_probe_fraction,
        "generator_receipt": receipt(__file__),
        "scope": "Independent finite o_head witnesses, not certification of an entire continuous interval or other meshes",
    }
    training = Path(plan["training_manifest_receipt"]["path"])
    if receipt(training)["sha256"] != plan["training_manifest_receipt"]["sha256"]:
        raise ValueError("Original training manifest SHA-256 mismatch")
    for entry in catalog["entries"]:
        payload = load_payload(catalog_path, entry)
        baseline = catalog["baselines"][entry["baseline_name"]]
        if payload["identity"] != baseline["identity"]:
            raise ValueError("Data and baseline input identity differ")
        for sign in (-1, 1):
            suggestion = interior_suggestion(
                payload,
                baseline,
                sign,
                target_percent=target_percent,
                max_probe_fraction=max_probe_fraction,
            )
            if not suggestion.pop("supported"):
                plan["skipped"].append(
                    {"entry_id": entry["id"], "sign": sign, **suggestion}
                )
                continue
            vector = list(baseline["identity"]["native59"])
            metadata = {
                k: entry[k]
                for k in ("kind", "baseline_name", "control", "bone", "channel", "axis")
                if k in entry
            }
            row = {
                "name": f"{entry['id']}_heldout_{sign:+d}",
                "entry_id": entry["id"],
                "sign": sign,
                **metadata,
                **suggestion,
                "native59": vector,
            }
            if entry["kind"] == "native":
                vector[entry["control"]] = row["value"]
            else:
                default = {
                    "scale": [1.0, 1.0, 1.0],
                    "length": 1.0,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                }
                patch = json.loads(
                    json.dumps(baseline["identity"]["abmx"].get(entry["bone"], default))
                )
                patch["name"] = entry["bone"]
                if entry["channel"] == "length":
                    patch["length"] = row["value"]
                else:
                    patch[entry["channel"]][entry["axis"]] = row["value"]
                row["patch"] = patch
            plan["entries"].append(row)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--target-percent", type=float, default=0.01)
    parser.add_argument("--max-probe-fraction", type=float, default=0.4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a fresh predeclared plan path")
    plan = make_plan(
        args.catalog,
        target_percent=args.target_percent,
        max_probe_fraction=args.max_probe_fraction,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "plan": str(args.out.resolve()),
                "entries": len(plan["entries"]),
                "noise_guarded_directions": len(plan["skipped"]),
            }
        )
    )


if __name__ == "__main__":
    main()
