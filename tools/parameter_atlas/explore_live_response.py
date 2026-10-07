"""Build a receipt-verified explorer for all native head controls and measured ABMX.

This consumes saved game geometry, never drives the game. Each receipt is decoded
once per baseline. HTML loads local JavaScript data files without a web server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from core import summarize_displacement
from live_response import (
    anchor,
    attach_frames,
    expression_changes,
    load_receipt,
    measure_snapshot,
    public_abmx,
    require_abmx,
    require_native,
    require_same_anchor,
)
from view_live_response import UnverifiedResponse, bound_inputs, verified_selection


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def calibration_identity(snapshot, manifest):
    """Public state and source assets, independent of transient actor instance IDs."""
    origin, _ = anchor(snapshot, manifest)
    return {
        "head_id": snapshot["character"]["head_id"],
        "native59": snapshot["character"]["shape_value_face"],
        "abmx": public_abmx(snapshot),
        "abmx_coordinate": (snapshot.get("abmx_runtime") or {}).get("coordinate"),
        "expression": snapshot["character"].get("expression"),
        "source_meshes": [
            {
                "mesh_name": mesh["mesh_name"],
                "source_geometry_sha256": mesh["source_geometry_sha256"],
                "skin_quality": mesh.get("skin_quality"),
                "blendshape_weights": {
                    b["name"]: b["current_weight"] for b in mesh.get("blendshapes", [])
                },
            }
            for mesh in snapshot["meshes"]
        ],
        "coordinate_policy": manifest.get("coordinate_policy"),
        "coordinate_transform_name": manifest.get(
            "coordinate_transform_name", "cf_J_Head"
        ),
        "anchor_local_position": origin.get("local_position"),
        "anchor_local_scale": origin.get("local_scale"),
    }


def identity_mismatches(expected, actual):
    """Coefficient serialization tolerance only; changed state is never adapted."""
    changed = []
    for key in expected:
        a, b = expected[key], actual.get(key)
        if key == "native59":
            try:
                equal = (
                    len(a) == len(b) == 59
                    and all(type(v) in (int, float) for v in b)
                    and np.isfinite(b).all()
                    and np.max(np.abs(np.asarray(a) - b)) <= 2e-6
                )
            except (TypeError, ValueError):
                equal = False
        else:
            equal = a == b
        if not equal:
            changed.append(key)
    return changed


def local_response(baseline, cases, deltas):
    pairs = {"minus": [], "plus": []}
    for case, delta in zip(cases, deltas):
        step = float(case["value"]) - baseline
        role = case.get("probe_role")
        if case["kind"] == "native" and role not in ("local_minus", "local_plus"):
            continue
        if step == 0 or not np.isfinite(step):
            raise UnverifiedResponse("Local probes need distinct finite values")
        pairs["plus" if step > 0 else "minus"].append((step, delta))
    if any(len(rows) != 1 for rows in pairs.values()):
        raise UnverifiedResponse("Exactly one local probe per direction required")
    negative, left = pairs["minus"][0]
    positive, right = pairs["plus"][0]
    left_gain = float(np.linalg.norm(left / negative, axis=1).max())
    right_gain = float(np.linalg.norm(right / positive, axis=1).max())
    return {
        "minus_step": -negative,
        "plus_step": positive,
        "left_gain": left_gain,
        "right_gain": right_gain,
        "max_units_per_input_unit": max(left_gain, right_gain),
        "valid_interval": [baseline + negative, baseline + positive],
        "estimate_verified": False,
        "policy": "One-sided local secant; independent held-out validation still pending",
    }


def abmx_base_value(case, snapshot):
    default = {
        "scale": [1.0, 1.0, 1.0],
        "position": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "length": 1.0,
    }
    bone = public_abmx(snapshot).get(case["bone"], default)
    value = bone[case["channel"]]
    return float(value if case["channel"] == "length" else value[case["axis"]])


def validate_abmx_case(case, reviewed, plan, snapshot):
    if (
        reviewed.get("trusted_isolated_response") is not True
        or reviewed.get("interpretable_isolated_response_all_renderers") is not True
    ):
        raise UnverifiedResponse(
            "ABMX sample lacks verified all-renderer interpretation"
        )
    keys = ("name", "kind", "baseline_name", "bone", "channel", "axis", "value")
    if any(case.get(k) != reviewed.get(k) for k in keys):
        raise UnverifiedResponse("ABMX case/report metadata differs")
    declared = [r for r in plan.get("abmx_cases", []) if r["name"] == case["name"]]
    if len(declared) != 1 or any(
        declared[0].get(k) != case.get(k) for k in (*keys, "native59", "patch", "step")
    ):
        raise UnverifiedResponse("ABMX sample differs from predeclared plan")
    baseline = abmx_base_value(case, snapshot)
    patch = case["patch"]
    defaults = {
        "scale": [1.0, 1.0, 1.0],
        "position": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "length": 1.0,
    }
    expected = json.loads(json.dumps(public_abmx(snapshot).get(case["bone"], defaults)))
    if case["channel"] == "length":
        expected["length"] = case["value"]
    else:
        expected[case["channel"]][case["axis"]] = case["value"]
    if any(patch[k] != expected[k] for k in expected):
        raise UnverifiedResponse("ABMX probe changes more than one declared channel")
    if abs(abs(case["value"] - baseline) - case["step"]) > 2e-6:
        raise UnverifiedResponse("ABMX local step differs from requested value")


def export_explorer(manifest_path, head_report_path, full_report_path, out):
    from explorer_template import TEMPLATE

    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh explorer output directory")
    manifest, report, provenance = bound_inputs(manifest_path, head_report_path)
    other, full, full_provenance = bound_inputs(manifest_path, full_report_path)
    if other != manifest or report.get("analysis_mesh_scope") != ["o_head"]:
        raise UnverifiedResponse("Explicit head component report required")
    if full.get("analysis_mesh_scope", ["all_captured_renderers"]) != [
        "all_captured_renderers"
    ]:
        raise UnverifiedResponse("ABMX needs the full-renderer report")
    if manifest.get("public_configuration_restored") is not True:
        raise UnverifiedResponse("Public configuration restoration must pass")
    directory = Path(manifest_path).resolve().parent
    plan_path = Path(manifest["predeclared_plan"])
    if not plan_path.is_absolute():
        plan_path = directory / plan_path
    plan_raw = plan_path.read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != manifest["predeclared_plan_sha256"]:
        raise UnverifiedResponse("Predeclared plan SHA-256 mismatch")
    plan = json.loads(plan_raw)
    full_cases = {r["name"]: r for r in full["cases"]}
    if len(full_cases) != len(full["cases"]):
        raise UnverifiedResponse("Full report case names must be unique")
    labels = {r["index"]: r["name"] for r in manifest["before"]["shapes"]["shapes"]}
    if set(labels) != set(range(59)):
        raise UnverifiedResponse("All 59 game control labels required")
    catalog = {
        "schema_version": 1,
        "head_id": manifest["head_id"],
        "units": report["units"],
        "provenance": {
            **provenance,
            "full_report": full_provenance,
            "generator_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "baselines": {},
        "entries": [],
        "complete": False,
        "failures": [
            {"name": r["name"], "reason": r["untrusted_reason"]}
            for r in full["cases"]
            if not r["trusted_isolated_response"]
        ],
        "scope": "Verified captured o_head point surfaces; finite ABMX probe set. No anatomical, opaque-render or arbitrary-state certification.",
    }
    (out / "data").mkdir(parents=True)
    tolerance = report["normalized_tolerance"]
    allow_rotation = (
        manifest.get("allow_anchor_rotation_change") is True
        or manifest.get("coordinate_policy")
        == "remove recorded rigid head pose; anchor local position and scale must stay fixed"
    )
    for name, declaration in manifest["baselines"].items():
        source = load_receipt(declaration["geometry"], directory)
        source = attach_frames(source, source)
        require_native(source, declaration["native59"])
        surfaces, _, origin = measure_snapshot(
            source, manifest, tolerance=tolerance, head_only=True
        )
        head_path = next(iter(surfaces))
        first = surfaces[head_path]
        diagonal = float(np.linalg.norm(np.ptp(first, axis=0)))
        groups, rows_by_name = {}, {}
        for control in range(59):
            selected, repeats = verified_selection(
                manifest, report, name, control, head_path
            )
            group = f"{name}_native_{control:02d}"
            groups[group] = {
                "cases": selected,
                "kind": "native",
                "control": control,
                "label": f"{control:02d} · {labels[control]}",
            }
            rows_by_name.update({r["name"]: r for r in selected + repeats})
        abmx = [
            c
            for c in manifest["cases"]
            if c["kind"] == "abmx" and c["baseline_name"] == name
        ]
        for case in abmx:
            validate_abmx_case(case, full_cases.get(case["name"], {}), plan, source)
            group = f"{name}_abmx_{case['bone']}_{case['channel']}{case['axis']}"
            axis = "" if case["channel"] == "length" else " XYZ"[case["axis"] + 1]
            if group not in groups:
                groups[group] = {
                    "cases": [],
                    "kind": "abmx",
                    "bone": case["bone"],
                    "channel": case["channel"],
                    "axis": case["axis"],
                    "label": f"{case['bone']} · {case['channel']} {axis}",
                }
            groups[group]["cases"].append(case)
            rows_by_name[case["name"]] = case
        measured, receipts, repeat_max = {}, {}, 0.0
        for serial, case in enumerate(rows_by_name.values(), 1):
            snapshot = attach_frames(source, load_receipt(case["geometry"], directory))
            requested = require_native(snapshot, case["native59"])
            keep = np.ones(59, dtype=bool)
            if case["kind"] == "native":
                keep[case["control"]] = False
                if abs(requested[case["control"]] - case["value"]) > 2e-6:
                    raise UnverifiedResponse("Native probe coefficient mismatch")
                declared = [
                    r for r in plan["native_cases"] if r["name"] == case["name"]
                ]
                if len(declared) != 1 or any(
                    declared[0].get(k) != case.get(k)
                    for k in (
                        "control",
                        "probe_role",
                        "native59",
                        "value",
                        "baseline_name",
                    )
                ):
                    raise UnverifiedResponse(
                        "Native probe differs from predeclared plan"
                    )
            if (
                np.max(
                    np.abs(requested[keep] - np.asarray(declaration["native59"])[keep])
                )
                > 2e-6
            ):
                raise UnverifiedResponse("Other native coefficients changed")
            require_abmx(snapshot, source, case)
            if expression_changes(source, snapshot):
                raise UnverifiedResponse("Expression changed during isolated probe")
            current, _, current_origin = measure_snapshot(
                snapshot,
                manifest,
                tolerance=tolerance,
                head_only=case["kind"] != "abmx",
            )
            require_same_anchor(origin, current_origin, allow_rotation=allow_rotation)
            delta = current[head_path] - first
            if case["kind"] == "baseline":
                drift = float(np.linalg.norm(delta, axis=1).max() / diagonal)
                if drift > tolerance:
                    raise UnverifiedResponse("Recomputed baseline drift exceeds gate")
                repeat_max = max(repeat_max, drift)
            else:
                measured[case["name"]] = delta
            receipts[case["name"]] = case["geometry"]
            if serial % 30 == 0:
                print(
                    f"{name}: verified {serial}/{len(rows_by_name)} receipts",
                    flush=True,
                )
        identity = calibration_identity(source, manifest)
        catalog["baselines"][name] = {
            "identity": identity,
            "identity_sha256": canonical_hash(identity),
            "head_diagonal": diagonal,
            "repeat_drift_normalized": repeat_max,
            "baseline_receipt": declaration["geometry"],
        }
        for key, group in groups.items():
            cases = sorted(group["cases"], key=lambda c: c["value"])
            base_value = (
                declaration["native59"][group["control"]]
                if group["kind"] == "native"
                else abmx_base_value(cases[0], source)
            )
            deltas = np.asarray([measured[c["name"]] for c in cases])
            all_vertices = np.concatenate([first[None], first[None] + deltas])
            stats = [
                summarize_displacement(first, first + delta)[0] for delta in deltas
            ]
            plot_first, plot_deltas = np.round(first, 7), np.round(deltas, 8)
            rounding = float(
                np.linalg.norm(
                    (plot_first[None] + plot_deltas) - (first[None] + deltas), axis=2
                ).max()
            )
            if rounding > diagonal * 1e-7:
                raise UnverifiedResponse("Display rounding exceeds declared bound")
            metadata = {k: v for k, v in group.items() if k != "cases"}
            payload = {
                **metadata,
                "id": key,
                "head_id": manifest["head_id"],
                "mesh": "o_head",
                "baseline_name": name,
                "baseline_level": base_value,
                "levels": [c["value"] for c in cases],
                "roles": [c.get("probe_role", "abmx_measured") for c in cases],
                "baseline": plot_first.tolist(),
                "deltas": plot_deltas.tolist(),
                "stats": stats,
                "fixed_min": all_vertices.min(axis=(0, 1)).tolist(),
                "fixed_max": all_vertices.max(axis=(0, 1)).tolist(),
                "max_distance": float(np.linalg.norm(deltas, axis=2).max()),
                "local_response": local_response(base_value, cases, deltas),
                "identity": identity,
                "verification": {
                    "head_diagonal": diagonal,
                    "repeat_drift_normalized": repeat_max,
                    "display_rounding_max_l2": rounding,
                    "scope": "o_head measured point surface only",
                    "baseline_receipt": declaration["geometry"],
                    "case_receipts": [receipts[c["name"]] for c in cases],
                },
            }
            filename = f"data/{key}.js"
            raw = json.dumps(
                payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ).replace("<", "\\u003c")
            target = out / filename
            target.write_text("window.HS2_RESPONSE(" + raw + ");\n", encoding="utf-8")
            catalog["entries"].append(
                {
                    **metadata,
                    "id": key,
                    "baseline_name": name,
                    "file": filename,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
    if len([r for r in catalog["entries"] if r["kind"] == "abmx"]) * 2 != len(
        [c for c in manifest["cases"] if c["kind"] == "abmx"]
    ):
        raise UnverifiedResponse("ABMX channel groups need exactly two directions")
    catalog["complete"] = True
    (out / "catalog.json").write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    catalog_json = json.dumps(
        catalog, separators=(",", ":"), ensure_ascii=False
    ).replace("<", "\\u003c")
    (out / "index.html").write_text(
        TEMPLATE.replace("CATALOG_JSON", catalog_json), encoding="utf-8"
    )
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--head-report", required=True, type=Path)
    parser.add_argument("--full-report", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = export_explorer(
        args.manifest, args.head_report, args.full_report, args.out
    )
    print(
        json.dumps(
            {
                "html": str(args.out.resolve() / "index.html"),
                "entries": len(result["entries"]),
                "baselines": list(result["baselines"]),
            }
        )
    )


if __name__ == "__main__":
    main()
