"""Source-bound full8/three-window diagnostic composition; no game or FK runs."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/geometry_quality"))  # Existing cross_mesh imports its standalone helper.
sys.path.insert(0, str(ROOT / "tools/base_comparison"))  # Existing search imports contracts/surface.

from tools.base_comparison.search import quality_gate
from tools.geometry_quality.cross_mesh import (
    compare_crossings,
    cross_intersections,
    load_certified_head,
)
from tools.geometry_quality.mesh_quality import (
    Thresholds,
    analyze_mesh,
    baseline_comparison,
)
from tools.native_radial_target.capture_groups_v3 import (
    _exact,
    prepare_capture_groups,
)
from tools.native_radial_target.runtime_backend_v3 import dependencies

REVISION = "source_bound_full8_quality_diagnostic_v1"


class QualityAuditRejected(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise QualityAuditRejected(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def save(path, value):
    path = Path(path)
    require(not path.exists(), "NEW report file required")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return binding(path)


def bound(descriptor):
    require(type(descriptor) is dict and type(descriptor.get("path")) is str
            and type(descriptor.get("sha256")) is str, "Malformed actual file descriptor")
    require(Path(descriptor["path"]).is_absolute() and sha(descriptor["path"]) == descriptor["sha256"], "Actual file SHA/path differs")
    return read(descriptor["path"])


def verify_runtime_binding(manifest_path, summary_path, current_dependencies):
    """Pure supplied-evidence preflight, before any mesh computation."""
    manifest, summary = read(manifest_path), read(summary_path)
    require(type(summary.get("manifest_path")) is str and Path(summary["manifest_path"]).resolve() == Path(manifest_path).resolve()
            and summary.get("manifest_sha256") == sha(manifest_path), "Supplied runtime report binds another manifest")
    require(summary.get("passed") is True and summary.get("full_visible_mesh_scope_supported") is True, "Full visible runtime proof unavailable")
    require(summary.get("normalization_tolerance") == 1e-5 and summary.get("temporal_rms_gate") == .002
            and summary.get("temporal_p95_gate") == .005, "Frozen runtime tolerance changed")
    _exact(summary["explicit_multigroup_backend"]["source_bindings"], current_dependencies, "independently reconstructed runtime helper closure")
    require(len(current_dependencies) == 40, "Expected current explicit40-helper V3 revision; another revision needs new contract")
    for path, expected in current_dependencies.items():
        require(sha(path) == expected, "Runtime dependency changed")
    require(type(manifest.get("cases")) is list and len(manifest["cases"]) == 1, "One actual case required")
    case = manifest["cases"][0]
    require(type(case.get("head_id")) is int, "Exact head ID required")
    windows = case.get("windows")
    require(type(windows) is list and len(windows) == 3 and [w.get("window") for w in windows] == ["early", "late", "far60"], "Complete ordered three windows required")
    require(all(manifest.get(k) is True for k in ("state_restored", "expression_restored", "bone_restored"))
            and manifest["before"] == manifest["after"] and manifest["expression_before"] == manifest["expression_after"]
            and manifest["modifiers_before"] == manifest["modifiers_after"] and manifest.get("restore_failures") == [], "Terminal public restoration evidence differs")
    proof_windows = summary.get("windows")
    require(type(proof_windows) is list and len(proof_windows) == 3, "Runtime window coverage missing")
    for window, proof in zip(windows, proof_windows):
        require(proof.get("window") == window["window"] and proof.get("geometry_sha256") == window["geometry"]["sha256"]
                and proof.get("passed") is True and len(proof.get("full_mesh_comparisons", {})) == 8, "Runtime/candidate full8 byte binding differs")
        require(proof.get("target_uses_candidate_after") is False and proof.get("candidate_pose_or_scale_fitted") is False,
                "Runtime target uses forbidden candidate inputs or fitting")
    return manifest, summary


def strict_identity(current, baseline):
    """No unique-name fallback; compare all authored arrays and palette bindings."""
    for key in ("renderer_path", "mesh_name", "source_geometry_sha256", "bone_names", "bone_transform_ids"):
        _exact(current.get(key), baseline.get(key), "source identity." + key)
    _exact(current["source"], baseline["source"], "all authored source arrays")
    _exact(current["baked"]["triangles"], baseline["baked"]["triangles"], "actual ordered topology")
    _exact(current.get("blendshapes"), baseline.get("blendshapes"), "authored blendshapes and current weights")


def selected_map(snapshot, certified):
    paths = [m["renderer_path"] for m in snapshot["meshes"]]
    require(len(paths) == len(set(paths)), "Duplicate renderer path cannot overwrite")
    all_meshes = {m["renderer_path"]: m for m in snapshot["meshes"]}
    require(len(certified) == 8 and len({m.path for m in certified}) == 8, "Exactly eight certified head renderers required")
    return {m.path: all_meshes[m.path] for m in certified}


def absolute_flags(report):
    return {"degenerate_triangle_ids": report["degenerate_triangle_ids"],
            "self_intersections": report["self_intersections"], "topology": report["topology"],
            "normal_direction": report["normal_direction"], "aspect_warning_triangle_ids": report["aspect_warning_triangle_ids"],
            "bone_world_determinants": report["bone_world_determinants"],
            "absolute_anatomical_acceptance": "unaccepted_requires_review"}


def consume_certificate(snapshot, descriptor, certificate):
    require(certificate.get("snapshot_sha256") == descriptor["sha256"] == sha(descriptor["path"]), "Certificate bound to wrong actual snapshot")
    return load_certified_head(snapshot, descriptor["sha256"], certificate, maximum_tolerance=1e-5)


def audit(manifest_path, summary_path, out):
    began = time.perf_counter()
    manifest_path, summary_path, out = Path(manifest_path).resolve(), Path(summary_path).resolve(), Path(out).resolve()
    require(not out.exists(), "NEW quality report directory required")
    closure = dependencies()
    manifest, runtime = verify_runtime_binding(manifest_path, summary_path, closure)
    out.mkdir(parents=True)
    artifact = bound(manifest["compiler_artifact"])
    # Protocol is an earlier frozen compiler dependency, not candidate content.
    protocol_paths = [p for p in artifact["compiler_inputs"]["declared"]["source_files"] if Path(p).name == "predeclared_protocol.json"]
    require(len(protocol_paths) == 1, "Unique frozen earlier predeclared protocol required")
    protocol_path = Path(protocol_paths[0])
    require(sha(protocol_path) == artifact["compiler_inputs"]["declared"]["source_files"][str(protocol_path)], "Earlier protocol changed")
    protocol = read(protocol_path)
    capture_receipts = []
    for registry_key, entry in manifest["capture_contracts"].items():
        phase = entry["schedule"]["phase"]
        _exact(entry["schedule"], protocol["view_schedules"][phase], "registered/earlier schedule")
        _exact(entry["expected_source"], protocol["phase_sources"][phase], "registered/earlier source")
        response = bound(entry["response"])
        require(str(Path(response["paired_geometry"]["path"]).resolve()) == registry_key, "Registry geometry path changed")
        context = prepare_capture_groups(entry["schedule"], entry["expected_source"], cursor_policy=entry["cursor_policy"])
        capture_receipts.append(context.verify_response(entry["response"]["path"], entry["response"]["sha256"], response["paired_geometry"]))
    require({r["schedule"]["phase"] for r in capture_receipts} == set(protocol["view_schedules"]), "Registered capture scope incomplete")
    save(out / "full_response_bindings.json", capture_receipts)
    source_descriptor = manifest["source_history"]["geometry"]
    _exact(source_descriptor, artifact["compiler_inputs"]["history"]["geometry"], "Earlier source-only compiler/quality baseline")
    source = bound(source_descriptor)
    source_evidence = runtime["source_only_target"]
    require(source_evidence.get("candidate_actual_after_or_expression_used_for_target") is False, "Execution target source scope changed")
    certificate = copy.deepcopy(source_evidence["source_lbs_certificate"])
    extracted = save(out / "extracted_source_lbs_certificate.json", certificate)
    derivation = {"parent_report": binding(summary_path), "object_pointer": "/source_only_target/source_lbs_certificate",
                  "source_snapshot": source_descriptor, "extracted_certificate": extracted, "extractor": binding(__file__),
                  "new_measurement": False, "source_only_execution_proof_is_person_target": False}
    save(out / "source_certificate_derivation.json", derivation)
    base_certified, base_skipped = consume_certificate(source, source_descriptor, certificate)
    baseline_meshes = selected_map(source, base_certified)
    thresholds = Thresholds()
    base_reports, base_arrays = {}, {}
    transforms = {t["id"]: t for t in source["transforms"]}
    for path, mesh in baseline_meshes.items():
        base_reports[path], base_arrays[path] = analyze_mesh(mesh, transforms, thresholds, space="baked_raw")
    base_cross = cross_intersections(base_certified, thresholds)
    base_report = {"snapshot": source_descriptor, "certificate_derivation": derivation,
                   "per_renderer": {p: {"report": q, "absolute_flags": absolute_flags(q)} for p, q in base_reports.items()},
                   "cross_mesh_report": base_cross, "skipped_renderers": base_skipped,
                   "absolute_baseline_acceptance": "unaccepted_requires_review"}
    base_output = save(out / "source_absolute_quality.json", base_report)
    print(json.dumps({"phase": "source_absolute_complete", "meshes": len(base_reports), "cross_pairs": base_cross["crossing_count"]}), flush=True)
    windows = []
    for window in manifest["cases"][0]["windows"]:
        name, descriptor = window["window"], window["geometry"]
        snapshot = bound(descriptor)
        require(type(snapshot["character"]["head_id"]) is int and snapshot["character"]["head_id"] == source["character"]["head_id"], "Head baseline changed")
        require(snapshot["character"]["transform_id"] == source["character"]["transform_id"], "Actor baseline changed")
        require(np.array_equal(np.asarray(snapshot["character"]["shape_value_face"], dtype=np.float32), np.asarray(source["character"]["shape_value_face"], dtype=np.float32)), "Native59 reference changed")
        _exact(snapshot["character"]["expression"], source["character"]["expression"], "expression nuisance")
        certificate_path = summary_path.parent / (name + "_lbs.json")
        candidate_certificate = read(certificate_path)
        certified, skipped = consume_certificate(snapshot, descriptor, candidate_certificate)
        candidate_meshes = selected_map(snapshot, certified)
        require(set(candidate_meshes) == set(baseline_meshes), "Renderer coverage/source correspondence differs")
        comparisons = {}
        transforms = {t["id"]: t for t in snapshot["transforms"]}
        for path, mesh in candidate_meshes.items():
            strict_identity(mesh, baseline_meshes[path])
            quality, arrays = analyze_mesh(mesh, transforms, thresholds, space="baked_raw")
            quality["baseline"] = baseline_comparison(quality, base_reports[path], arrays, base_arrays[path], thresholds)
            quality["baseline"]["match_policy"] = "exact_path_all_source_arrays_palette_topology_no_fallback"
            gate = quality_gate({"meshes": [quality]}, {"meshes": [base_reports[path]]})
            comparisons[path] = {"mesh_name": mesh["mesh_name"], "report": quality, "relative_gate": gate,
                                 "source_absolute_flags": absolute_flags(base_reports[path]), "candidate_absolute_flags": absolute_flags(quality)}
        current_cross = cross_intersections(certified, thresholds)
        crossed = compare_crossings(certified, base_certified, current_cross, base_cross)
        require(crossed["comparison_performed"] is True and crossed["status"] == "source_and_topology_matched", "Cross source/topology correspondence failed")
        report = {"window": name, "snapshot": descriptor, "baseline_snapshot": source_descriptor,
                  "certificate": binding(certificate_path), "certificate_parent_runtime_summary": binding(summary_path),
                  "per_renderer": comparisons, "cross_mesh_report": current_cross,
                  "cross_mesh_baseline_comparison": crossed, "cross_mesh_semantic_acceptance": "requires_review",
                  "skipped_renderers": skipped, "full_face_quality_certified": False}
        output = save(out / (name + "_full8_quality.json"), report)
        windows.append({"window": name, "report": output, "relative_pass_count": sum(q["relative_gate"]["quality_valid"] for q in comparisons.values()),
                        "relative_gates": {p: {"mesh_name": q["mesh_name"], **q["relative_gate"]} for p, q in comparisons.items()},
                        "absolute_self_crossings": {p: q["report"]["self_intersections"]["true_crossing_or_area_overlap_count"] for p, q in comparisons.items()},
                        "cross_count": current_cross["crossing_count"], "new_cross_pairs": len(crossed["new_crossing_pairs"]),
                        "resolved_cross_pairs": len(crossed["resolved_crossing_pairs"]), "cross_semantic_acceptance": "requires_review"})
        print(json.dumps({"phase": name, "relative_passes": windows[-1]["relative_pass_count"], "new_cross_pairs": windows[-1]["new_cross_pairs"]}), flush=True)
    quality_sources = [Path(__file__), ROOT / "tools/geometry_quality/mesh_quality.py", ROOT / "tools/geometry_quality/cross_mesh.py",
                       ROOT / "tools/base_comparison/search.py", ROOT / "tools/native_radial_target/capture_groups_v3.py"]
    summary = {"revision": REVISION, "manifest": binding(manifest_path), "consumed_runtime_summary": binding(summary_path),
               "protocol": binding(protocol_path), "runtime_helper_sources": closure, "runtime_helper_count": len(closure),
               "quality_sources": [binding(p) for p in quality_sources], "thresholds": asdict(thresholds),
               "baseline_report": base_output, "baseline_absolute_acceptance": "unaccepted_requires_review",
               "baseline_self_crossings": {p: q["self_intersections"]["true_crossing_or_area_overlap_count"] for p, q in base_reports.items()},
               "baseline_cross_count": base_cross["crossing_count"], "windows": windows,
               "relative_mesh_gate_pass_count": sum(w["relative_pass_count"] for w in windows), "relative_mesh_gate_total": 24,
               "full_face_quality_certified": False, "character_ready": False, "likeness_certified": False,
               "execution_target_is_trusted_person_surface": False, "existing_fk_or_target_solver_rerun": False,
               "existing_evidence_only": True, "original_reports_or_thresholds_modified": False, "new_game_operations": 0,
               "elapsed_seconds": time.perf_counter() - began}
    save(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.runtime_summary, args.out_dir)
    print(json.dumps({"out": str(args.out_dir.resolve()), "relative_passes": result["relative_mesh_gate_pass_count"], "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
