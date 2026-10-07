"""View verified, receipt-bound game surface responses with fixed geometric axes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from view_displacement import TEMPLATE


class UnverifiedResponse(ValueError):
    pass


def scoped_measurement(report, mesh):
    scope = report.get("analysis_mesh_scope", ["all_captured_renderers"])
    if scope == ["o_head"]:
        if mesh != "o_head":
            raise UnverifiedResponse("Head-only component report cannot verify another mesh")
        return True
    if scope != ["all_captured_renderers"]:
        raise UnverifiedResponse("Unsupported independent report mesh scope")
    return False


def bound_inputs(manifest_path, report_path):
    manifest_path, report_path = (
        Path(manifest_path).resolve(),
        Path(report_path).resolve(),
    )
    raw_manifest, raw_report = manifest_path.read_bytes(), report_path.read_bytes()
    manifest = json.loads(raw_manifest.decode("utf-8-sig"))
    report = json.loads(raw_report.decode("utf-8-sig"))
    digest = hashlib.sha256(raw_manifest).hexdigest()
    if report.get("source_manifest_sha256") != digest:
        raise UnverifiedResponse(
            "Independent report is not SHA-256-bound to this manifest"
        )
    if manifest.get("complete") is not True:
        raise UnverifiedResponse(
            "Incomplete game collection cannot produce a verified tuning view"
        )
    tolerance = report.get("normalized_tolerance")
    if (
        type(tolerance) not in (int, float)
        or not np.isfinite(tolerance)
        or not 0 < tolerance <= 1e-5
    ):
        raise UnverifiedResponse(
            "Report needs a finite normalized tolerance in (0, 1e-5]"
        )
    return (
        manifest,
        report,
        {
            "manifest_path": str(manifest_path),
            "source_manifest_sha256": digest,
            "report_path": str(report_path),
            "report_sha256": hashlib.sha256(raw_report).hexdigest(),
        },
    )


def verified_selection(manifest, report, baseline_name, control, renderer_path):
    """Reject missing/mixed gates before touching any geometry receipts."""
    if type(control) is not int or not 0 <= control < 59:
        raise UnverifiedResponse("Control must be an integer 0..58")
    if baseline_name not in manifest.get(
        "baselines", {}
    ) or baseline_name not in report.get("baselines", {}):
        raise UnverifiedResponse(
            "Baseline is missing from the collection or independent report"
        )
    baseline_report = report["baselines"][baseline_name]
    baseline = manifest["baselines"][baseline_name]
    if baseline_report.get("native59") != baseline.get("native59"):
        raise UnverifiedResponse("Independent report baseline differs from collection")
    coverage = baseline_report.get("coverage", {})
    gate = coverage.get("per_renderer", {}).get(renderer_path, {})
    if (
        gate.get("repeat_drift_pass") is not True
        or gate.get("all_native_controls_verified_at_this_configuration") is not True
    ):
        raise UnverifiedResponse(
            "Selected renderer lacks full-control coverage and repeat-drift verification"
        )
    drift = gate.get("max_repeat_drift_normalized")
    if (
        type(drift) not in (int, float)
        or not np.isfinite(drift)
        or drift > report["normalized_tolerance"]
        or drift < 0
    ):
        raise UnverifiedResponse("Selected renderer repeat drift is invalid")
    cases = manifest.get("cases", [])
    report_cases = report.get("cases", [])
    names = [case.get("name") for case in cases]
    report_names = [case.get("name") for case in report_cases]
    if (
        not all(isinstance(name, str) and name for name in names + report_names)
        or len(set(names)) != len(names)
        or len(set(report_names)) != len(report_names)
    ):
        raise UnverifiedResponse("Case identities must be nonempty and unique")
    reviewed = {case["name"]: case for case in report_cases}
    selected = [
        case
        for case in cases
        if case.get("kind") == "native"
        and case.get("baseline_name") == baseline_name
        and case.get("control") == control
    ]
    repeats = [
        case
        for case in cases
        if case.get("kind") == "baseline" and case.get("baseline_name") == baseline_name
    ]
    if not selected or not repeats:
        raise UnverifiedResponse(
            "Selected control needs native probes and baseline repeats"
        )
    roles = [case.get("probe_role") for case in selected]
    required = coverage.get("required_probe_roles")
    if (
        not isinstance(required, list)
        or not required
        or set(roles) != set(required)
        or len(roles) != len(set(roles))
    ):
        raise UnverifiedResponse(
            "Selected control roles differ from independently verified coverage"
        )
    for case in selected + repeats:
        checked = reviewed.get(case["name"], {})
        if checked.get("trusted_isolated_response") is not True:
            raise UnverifiedResponse(
                f"Untrusted or unreviewed game case: {case['name']}"
            )
        keys = (
            ("kind", "baseline_name")
            if case["kind"] == "baseline"
            else ("kind", "baseline_name", "control", "probe_role", "value")
        )
        if any(checked.get(key) != case.get(key) for key in keys):
            raise UnverifiedResponse(
                f"Report and collection case metadata disagree: {case['name']}"
            )
    return selected, repeats


def live_viewer_data(
    manifest_path, report_path, *, control, baseline_name="card_input", mesh="o_head"
):
    # These imports load the independent reviewer only when real receipts are consumed.
    from core import summarize_displacement
    from live_response import (
        attach_frames,
        expression_changes,
        load_receipt,
        measure_snapshot,
        require_abmx,
        require_native,
        require_same_anchor,
    )

    manifest, report, provenance = bound_inputs(manifest_path, report_path)
    head_only = scoped_measurement(report, mesh)
    directory = Path(manifest_path).resolve().parent
    if baseline_name not in manifest.get("baselines", {}):
        raise UnverifiedResponse("Missing baseline")
    declaration = manifest["baselines"][baseline_name]
    baseline = load_receipt(declaration["geometry"], directory)
    if (
        report.get("baselines", {}).get(baseline_name, {}).get("head_id")
        != baseline["character"]["head_id"]
    ):
        raise UnverifiedResponse(
            "Independent report head ID differs from actual baseline receipt"
        )
    paths = [
        item["renderer_path"]
        for item in baseline["meshes"]
        if item["mesh_name"] == mesh
    ]
    if len(paths) != 1:
        raise UnverifiedResponse("Selected mesh must resolve to exactly one renderer")
    path = paths[0]
    selected, repeats = verified_selection(
        manifest, report, baseline_name, control, path
    )
    tolerance = report["normalized_tolerance"]
    baseline = attach_frames(baseline, baseline)
    require_native(baseline, declaration["native59"])
    surfaces, _, anchor = measure_snapshot(
        baseline, manifest, tolerance=tolerance, head_only=head_only
    )
    first = surfaces[path]
    head_paths = [
        item["renderer_path"]
        for item in baseline["meshes"]
        if item["mesh_name"] == "o_head"
    ]
    if len(head_paths) != 1:
        raise UnverifiedResponse("Baseline needs exactly one head renderer")
    head_diagonal = float(np.linalg.norm(np.ptp(surfaces[head_paths[0]], axis=0)))
    if not np.isfinite(head_diagonal) or head_diagonal <= 0:
        raise UnverifiedResponse("Degenerate baseline head")
    allow_rotation = (
        manifest.get("allow_anchor_rotation_change", False)
        or manifest.get("coordinate_policy")
        == "remove recorded rigid head pose; anchor local position and scale must stay fixed"
    )

    def measured_case(case):
        snapshot = attach_frames(baseline, load_receipt(case["geometry"], directory))
        requested = require_native(snapshot, case["native59"])
        base_values = np.asarray(declaration["native59"])
        keep = (
            np.arange(59) != control
            if case["kind"] == "native"
            else np.ones(59, dtype=bool)
        )
        if np.max(np.abs(requested[keep] - base_values[keep])) > 2e-6:
            raise UnverifiedResponse(
                "Other coefficients changed in the selected isolated game case"
            )
        if (
            case["kind"] == "native"
            and abs(requested[control] - float(case["value"])) > 2e-6
        ):
            raise UnverifiedResponse(
                "Selected coefficient differs from declared sample value"
            )
        require_abmx(snapshot, baseline, case)
        if expression_changes(baseline, snapshot):
            raise UnverifiedResponse(
                "Live expression changed during selected game case"
            )
        measured, _, current_anchor = measure_snapshot(
            snapshot, manifest, tolerance=tolerance, head_only=head_only
        )
        require_same_anchor(anchor, current_anchor, allow_rotation=allow_rotation)
        return measured[path]

    observed_repeat_max = 0.0
    for case in repeats:
        repeat = measured_case(case)
        drift = float(np.linalg.norm(repeat - first, axis=1).max() / head_diagonal)
        if drift > tolerance:
            raise UnverifiedResponse(
                "Reconstructed repeat drift fails selected renderer gate"
            )
        observed_repeat_max = max(observed_repeat_max, drift)
    deltas, stats, levels = [], [], []
    for case in selected:
        current = measured_case(case)
        statistic, delta, _ = summarize_displacement(first, current)
        deltas.append(delta)
        stats.append(statistic)
        levels.append(case["value"])
    deltas = np.asarray(deltas)
    all_vertices = np.concatenate([first[None], first[None] + deltas])
    labels = manifest.get("before", {}).get("shapes", {}).get("shapes", [])
    names = [item.get("name") for item in labels if item.get("index") == control]
    data = {
        "head_id": baseline["character"]["head_id"],
        "profile": "actual captured game surfaces",
        "mesh": mesh,
        "control": control,
        "control_name": names[0] if len(names) == 1 else None,
        "baseline_level": declaration["native59"][control],
        "baseline_input_sha256": declaration["geometry"]["sha256"],
        "levels": levels,
        "sample_roles": [case["probe_role"] for case in selected],
        "baseline": first.tolist(),
        "deltas": deltas.tolist(),
        "fixed_min": all_vertices.min(axis=(0, 1)).tolist(),
        "fixed_max": all_vertices.max(axis=(0, 1)).tolist(),
        "max_distance": float(np.linalg.norm(deltas, axis=2).max()),
        "stats": stats,
        "verification": {
            **provenance,
            "renderer_path": path,
            "baseline_name": baseline_name,
            "normalized_tolerance": tolerance,
            "recomputed_repeat_drift_normalized": observed_repeat_max,
            "geometry_receipts": [declaration["geometry"]]
            + [case["geometry"] for case in selected + repeats],
            "scope": "Selected captured game surface point cloud; no skin opacity, texture or arbitrary-character certification",
            "analysis_mesh_scope": report.get("analysis_mesh_scope", ["all_captured_renderers"]),
        },
    }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--control", type=int, required=True)
    parser.add_argument("--baseline", default="card_input")
    parser.add_argument("--mesh", default="o_head")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = live_viewer_data(
        args.manifest,
        args.report,
        control=args.control,
        baseline_name=args.baseline,
        mesh=args.mesh,
    )
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    html = TEMPLATE.replace("DATA_JSON", payload)
    html = html.replace(
        "· Control ${d.control} · ${d.mesh}",
        "· Control ${d.control} ${d.control_name||''} · ${d.mesh}",
    )
    html = html.replace(
        "o.textContent=v;",
        "o.textContent=`${v} (${d.sample_roles[i]})`;",
    )
    html = html.replace(
        "距离是未校准的资产单位", "距离是捕获的固定祖先局部坐标单位，未校准为毫米"
    )
    html = html.replace(
        "未包含 ABMX／实时表情", "包含本次捕获的固定 ABMX／表情状态；未认证其他状态"
    )
    html = html.replace("Baseline coefficient:", "Captured baseline coefficient:")
    html = html.replace("input SHA-256:", "geometry receipt SHA-256:")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(
        json.dumps(
            {"html": str(args.out.resolve()), "verification": data["verification"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
