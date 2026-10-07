"""Independently validate frozen full59 winners against saved Unity exports.

Read-only inputs; no HTTP/game calls. Original benchmark and hashed code are
never rewritten. A runtime match does not reverse an offline quality rejection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/unity_parity"))
sys.path.insert(0, str(ROOT / "tools/geometry_quality"))
from compare_export import compare_offline
from geometry import analyze_snapshot
from cross_mesh import load_certified_head

CHIN = "cf_J_ChinTip_s"
IDENTITY = {"scale": [1., 1., 1.], "length": 1., "position": [0., 0., 0.], "rotation": [0., 0., 0.]}
THRESHOLDS = {"readback_max_abs": 2e-6, "engine_max_normalized": 1e-5,
              "offline_vertex_max_normalized": 1e-5,
              "head_skin_local_position_max": 1e-5,
              "head_skin_local_rotation_deg_max": .001,
              "head_skin_local_scale_max": 1e-5}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finite(value, shape, label):
    result = np.asarray(value, dtype=float)
    require(result.shape == shape and np.isfinite(result).all(), "Invalid " + label)
    return result


def modifier_vector(value):
    require(set(value) == set(IDENTITY), "Expected explicit ChinTip scale/length/position/rotation")
    return np.r_[finite(value["scale"], (3,), "scale"), finite(value["length"], (), "length"),
                 finite(value["position"], (3,), "position"), finite(value["rotation"], (3,), "rotation")]


def compare_readback(expected, actual, label, shape=None):
    expected = np.asarray(expected, float)
    actual = finite(actual, expected.shape if shape is None else shape, label)
    require(expected.shape == actual.shape and np.isfinite(expected).all(), "Invalid expected " + label)
    error = float(np.abs(expected - actual).max())
    require(error < THRESHOLDS["readback_max_abs"], f"{label} mismatch: {error}")
    return error


def validate_inputs(case, snapshot):
    """Bind actual native/ABMX inputs to the unchanged bounded-search checkpoint."""
    require(snapshot.get("snapshot_kind") == "maker_live_skinned_geometry", "Expected actual Unity export")
    head = case["head_id"]
    require(head in (0, 1, 2, 3) and snapshot["character"]["head_id"] == head, "Wrong actual head ID")
    requested = finite(case["native59_requested"], (59,), "requested native59")
    actual = finite(case["native59_actual"], (59,), "actual native59")
    errors = {"native_requested_vs_actual": compare_readback(requested, actual, "native59", (59,)),
              "native_actual_vs_geometry": compare_readback(actual, snapshot["character"]["shape_value_face"], "geometry native59", (59,))}
    requested_ab = case["abmx_requested"]
    require(not (set(requested_ab) - {CHIN}), "Unsupported ABMX bone")
    expected_ab = requested_ab.get(CHIN, IDENTITY)
    live_ab = case["abmx_actual"]
    require(live_ab.get("name") == CHIN and live_ab.get("exists") is True, "Actual ChinTip configuration absent")
    actual_ab = {key: live_ab[key] for key in IDENTITY}
    errors["abmx_requested_vs_actual"] = compare_readback(modifier_vector(expected_ab), modifier_vector(actual_ab), "ChinTip")
    coordinate_modifiers = live_ab.get("coordinate_modifiers")
    require(isinstance(coordinate_modifiers, list) and len(coordinate_modifiers) == 1, "Ambiguous coordinate-specific ChinTip inputs")
    errors["abmx_coordinate_vs_actual"] = compare_readback(modifier_vector(actual_ab), modifier_vector(coordinate_modifiers[0]), "ChinTip coordinate")
    checkpoint_info = None
    if "checkpoint" in case:
        path = Path(case["checkpoint"])
        require(digest(path) == case["checkpoint_sha256"], "Checkpoint SHA mismatch")
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        require(checkpoint["head_id"] == head, "Checkpoint belongs to a different head")
        winners = [row for row in checkpoint["candidates"] if row["kind"] == "bounded_search_winner"]
        require(len(winners) == 1, "Expected exactly one original bounded search winner")
        winner = winners[0]
        errors["checkpoint_vs_requested_native"] = compare_readback(winner["native59"], requested, "checkpoint native59", (59,))
        require(winner["abmx"] == requested_ab, "Requested ABMX does not equal original checkpoint")
        require(winner["quality"]["quality_valid"] == case["offline_quality_valid"] and winner["accepted"] == case["offline_accepted"], "Original quality/acceptance flag mismatch")
        require(winner["search"]["active_native"] == list(range(59)), "Checkpoint was not full59")
        require(checkpoint["sampling_profile"] in ("vanilla", "slider_unlocker_18_2"), "Unknown native profile")
        profile = checkpoint["sampling_profile"]
        checkpoint_info = {"path": str(path.resolve()), "sha256": digest(path), "mode": checkpoint["mode"],
                           "selected_candidate_index_unchanged": checkpoint["selected_candidate_index"],
                           "tested_kind": winner["kind"], "offline_quality_valid": winner["quality"]["quality_valid"],
                           "offline_accepted": winner["accepted"], "status_unchanged": checkpoint["status"]}
    else:
        require(case["name"] == f"head_{head}_baseline", "Unbound non-baseline case")
        require(np.array_equal(requested, np.full(59, .5)) and not requested_ab, "Baseline is not same-head neutral native59/identity ABMX")
        profile = "vanilla"
    return {"errors": errors, "profile": profile, "actual_abmx": {CHIN: actual_ab}, "checkpoint": checkpoint_info}


def validate_pairs(case, snapshot, snapshot_sha):
    require(snapshot["frame_count"] == snapshot["frame_count_end"], "Geometry frame changed")
    require(case["geometry"]["pose_signature"] == snapshot["pose_signature"], "Geometry pose signature mismatch")
    views = case["capture"].get("views", [])
    require(len(views) == 3, "Expected three actual paired views")
    rows = []
    for view in views:
        pair = view["paired_geometry"]
        require(pair["sha256"] == snapshot_sha and Path(pair["path"]).resolve() == Path(case["geometry"]["path"]).resolve(), "View does not bind to exact geometry")
        require(view.get("paired_pose_unchanged") is True and view["pose_signature_before_render"] == view["pose_signature_after_render"] == snapshot["pose_signature"], "View pose was not stable")
        require(view.get("paired_visibility_sampled_unchanged") is True, "View sampled material/visibility changed")
        require(view["frame_count_before_render"] == view["frame_count"] == snapshot["frame_count"], "View and geometry are not same-frame")
        path = Path(view["path"])
        raw = path.read_bytes()
        require(raw[:8] == b"\x89PNG\r\n\x1a\n", "Paired image is not PNG")
        rows.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "yaw": view["yaw"], "frame": view["frame_count"]})
    return rows


def offline_gate(row, certified_head_paths, required_skin_names=None):
    reasons = []
    if row.get("unavailable_reason"):
        return {"passed": False, "reasons": [row["unavailable_reason"]]}
    if row.get("renderer_path") not in certified_head_paths:
        reasons.append("offline renderer is not a certified visible head renderer")
    errors = row.get("comparison", {}).get("rigid_errors", {})
    error = errors.get("max_normalized")
    if error is None or not np.isfinite(error) or error > THRESHOLDS["offline_vertex_max_normalized"]:
        reasons.append("complete o_head vertex residual exceeds normalized tolerance")
    bones = row.get("bone_local_comparison", {})
    checks = [("max_head_skin_position_error", "head_skin_local_position_max"),
              ("max_head_skin_rotation_angle_deg", "head_skin_local_rotation_deg_max"),
              ("max_head_skin_scale_error", "head_skin_local_scale_max")]
    for field, threshold in checks:
        value = bones.get(field)
        if value is None or not np.isfinite(value) or value > THRESHOLDS[threshold]:
            reasons.append(field + " exceeds declared tolerance")
    skin_rows = [bone for bone in bones.get("bones", []) if bone["is_head_skin_bone"]]
    if required_skin_names is not None and set(bone["name"] for bone in skin_rows) != set(required_skin_names):
        reasons.append("head-skin bone local coverage is incomplete or ambiguous")
    if any(bone["cached_parent"] != bone["live_parent"] for bone in skin_rows):
        reasons.append("head-skin bone hierarchy differs from cached hierarchy")
    if row.get("comparison", {}).get("unit_scale_applied") != 1:
        reasons.append("non-unit metric scaling is disallowed")
    return {"passed": not reasons, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "New report directory required; never overwrite old evidence")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    require(all(manifest.get(key) is True for key in ("state_restored", "expression_restored", "bone_restored")), "Live job has not completed/restored")
    benchmark = Path(manifest["benchmark"])
    implementations = json.loads((benchmark / "implementation_hashes.json").read_text(encoding="utf-8"))
    require(all(digest(item["path"]) == item["sha256"] for item in implementations), "Original benchmark implementation changed")
    original_files = [path for path in benchmark.iterdir() if path.is_file()]
    preserved = {str(path.resolve()): digest(path) for path in original_files}
    cases = manifest["cases"]
    names = [case["name"] for case in cases]
    require(len(names) == len(set(names)) == 16, "Expected 16 distinct baseline/winner cases")
    expected_names = {f"head_{head}_{suffix}" for head in range(4) for suffix in ("baseline", "native", "installed18_2", "native_ABMX")}
    require(set(names) == expected_names, "Unexpected or missing case coverage")
    args.out.mkdir(parents=True)
    dependencies = [ROOT / name for name in ("src/hs2_mesh_deform.py", "src/hs2_deform_torch.py", "src/hs2_sampling.py",
                    "tools/unity_parity/compare_export.py", "tools/unity_parity/bone_locals.py", "tools/unity_parity/geometry.py",
                    "tools/geometry_quality/cross_mesh.py")]
    summary = {"schema_version": 1, "manifest": str(args.manifest.resolve()), "manifest_sha256": digest(args.manifest),
               "scope": "16 specific frozen snapshots; full o_head offline native59+actual ChinTip vs actual-bone LBS vs Unity BakeMesh; no likeness/expressivity proof",
               "thresholds": THRESHOLDS, "unit_scale": 1, "scale_fitting": False,
               "pose_removal": "proper rigid rotation/translation only; separately recorded uniform renderer ancestor scale",
               "cases": [], "original_benchmark_file_hashes": preserved,
               "validator_implementation_sha256": digest(__file__),
               "runtime_comparison_dependency_hashes": [{"path": str(path.resolve()), "sha256": digest(path)} for path in dependencies]}
    for case in cases:
        name = case["name"]
        require(Path(name).name == name, "Invalid output case name")
        report = {"name": name, "head_id": case["head_id"], "thresholds": THRESHOLDS}
        compact = {"name": name, "head_id": case["head_id"], "passed": False}
        try:
            path = Path(case["geometry"]["path"])
            actual_sha = digest(path)
            require(actual_sha == case["geometry"]["sha256"], "Geometry SHA mismatch")
            snapshot = json.loads(path.read_text(encoding="utf-8"))
            inputs = validate_inputs(case, snapshot)
            report["input_binding"] = inputs
            report["paired_views"] = validate_pairs(case, snapshot, actual_sha)
            report["exported_abmx_private_runtime_state"] = snapshot.get("abmx_runtime")
            certificate = analyze_snapshot(snapshot, normalized_tolerance=THRESHOLDS["engine_max_normalized"])
            certificate.update({"snapshot_path": str(path.resolve()), "snapshot_sha256": actual_sha})
            report["engine_certificate"] = certificate
            require(all(mesh["certified"] for mesh in certificate["meshes"]), "An exported renderer failed actual-bone LBS certification")
            selected, skipped = load_certified_head(snapshot, actual_sha, certificate)
            report["visible_head_renderers"] = [{"name": mesh.name, "path": mesh.path, "source_sha256": mesh.source_hash, "candidate": mesh.candidate} for mesh in selected]
            report["skipped_renderers"] = skipped
            report["offline_head_comparison"] = compare_offline(snapshot, profile=inputs["profile"], abmx=inputs["actual_abmx"], apply_renderer_uniform_scale=True)
            head_rows = report["offline_head_comparison"]
            gates = [offline_gate(row, {mesh.path for mesh in selected}, next((mesh["bone_names"] for mesh in snapshot["meshes"]
                            if mesh["renderer_path"] == row.get("renderer_path")), [])) for row in head_rows]
            report["offline_gates"] = gates
            require(len(head_rows) == 1, "Expected one complete o_head renderer")
            head = head_rows[0]
            bones = head.get("bone_local_comparison", {})
            compact.update({"passed": all(gate["passed"] for gate in gates), "reasons": [reason for gate in gates for reason in gate["reasons"]],
                            "input_errors": inputs["errors"], "checkpoint": inputs["checkpoint"],
                            "engine_all_renderer_certified": all(mesh["certified"] for mesh in certificate["meshes"]),
                            "engine_visible_head_renderer_count": len(selected),
                            "offline_errors": head.get("comparison", {}).get("rigid_errors"),
                            "ancestor_scale": head.get("recorded_ancestor_scale", {}).get("factor"),
                            "head_skin_local_position_max": bones.get("max_head_skin_position_error"),
                            "head_skin_local_rotation_deg_max": bones.get("max_head_skin_rotation_angle_deg"),
                            "head_skin_local_scale_max": bones.get("max_head_skin_scale_error"),
                            "largest_skin_local_differences": sorted([row for row in bones.get("bones", []) if row["is_head_skin_bone"]], key=lambda row: row["matrix_max_abs"], reverse=True)[:5]})
        except (ValueError, KeyError, IndexError, OSError) as exc:
            report["validation_failure"] = str(exc)
            compact["reasons"] = [str(exc)]
        report["acceptance"] = compact
        destination = args.out / (name + ".json")
        save(destination, report)
        compact["report"] = str(destination.resolve())
        summary["cases"].append(compact)
        save(args.out / "progress.json", summary)
        print(json.dumps({"case": name, "passed": compact["passed"], "max_normalized": compact.get("offline_errors", {}).get("max_normalized"), "reasons": compact.get("reasons")}), flush=True)
    summary["original_benchmark_unchanged"] = all(digest(path) == sha for path, sha in preserved.items())
    summary["original_implementations_unchanged"] = all(digest(item["path"]) == item["sha256"] for item in implementations)
    summary["passed_count"] = sum(case["passed"] for case in summary["cases"])
    summary["limitations"] = ["Runtime parity applies only to these actual parameter combinations, expressions and sampled assets.",
                              "Actual-bone LBS validates exported inputs independently; cached native/ABMX mapping additionally needs offline gate.",
                              "Inactive extra body/silhouette tongue renderers cannot become visible head evidence through name matching.",
                              "Other ABMX bones/parameter combinations, cached eyes/lashes, skin appearance and character likeness remain outside this certification.",
                              "Offline benchmark quality-invalid winners remain diagnostic and never become official fits here."]
    save(args.out / "runtime_summary.json", summary)
    require(summary["original_benchmark_unchanged"] and summary["original_implementations_unchanged"], "Original evidence changed during analysis")
    print(f"complete={len(cases)}, passed={summary['passed_count']}, output={args.out.resolve()}", flush=True)
    return 0 if summary["passed_count"] == len(cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
