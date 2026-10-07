"""State-conditioned cached head FK/LBS using independently replayed Apply calls.

Only the replay-predicted ChinTip local TRS overrides native cached FK. Actual
after locals are acceptance targets, never prediction inputs for the override.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
try:
    from .model import ReplayRejected, require
    from .validate_trace import validate, trs_errors, cache_errors, local_only, sha, read
except ImportError:  # Direct script execution.
    from model import ReplayRejected, require
    from validate_trace import validate, trs_errors, cache_errors, local_only, sha, read

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/unity_parity"))
sys.path.insert(0, str(ROOT / "tools/geometry_quality"))
from geometry import analyze_snapshot, blendshape_delta, recorded_uniform_renderer_scale, rigid_alignment, vertex_errors
from cross_mesh import load_certified_head
from compare_export import compare_offline
from bone_locals import compare_bone_locals
from src.hs2_mesh_deform import HeadRig, _fk_world, _trs

CHIN = "cf_J_ChinTip_s"
TOLERANCE = 1e-5


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def select_cursor_prediction(snapshot, trace, report):
    cursor = snapshot.get("abmx_trace_cursor")
    require(isinstance(cursor, dict), "Missing exact snapshot trace cursor")
    require(cursor.get("session_id") == trace["metadata"]["session_id"], "Snapshot references another trace session")
    require(cursor.get("frame") == snapshot["frame_count"] == snapshot["frame_count_end"], "Snapshot cursor/frame mismatch")
    for field in ("pending_calls", "dropped_events", "observer_error_count"):
        require(type(cursor.get(field)) is int and cursor[field] == 0, "Incomplete snapshot cursor: " + field)
    sequence = cursor.get("last_completed_sequence")
    require(type(sequence) is int and sequence > 0 and sequence <= len(trace["events"]), "Invalid snapshot completed call cursor")
    require(cursor.get("observed_calls") == cursor.get("completed_calls") == sequence, "Cursor call coverage/count mismatch")
    require(report.get("passed") is True, "Trace was not independently replay certified")
    selected = [row for row in report["rows"] if row["sequence"] == sequence]
    require(len(selected) == 1 and selected[0]["bone_name"] == CHIN, "Exact cursor call is missing or not ChinTip")
    row = selected[0]
    require(row["frame"] <= cursor["frame"], "Exact selected Apply call occurs after snapshot cursor")
    require(trace["metadata"]["character_transform_id"] == snapshot["character"]["transform_id"], "Snapshot/trace belong to different characters")
    # Take propagated model output, not any actual `after` value.
    predicted = row.get("chain_prediction", row["prediction"])
    matches = [transform for transform in snapshot["transforms"] if transform["id"] == row["bone_transform_id"]]
    require(len(matches) == 1 and matches[0]["name"] == CHIN, "Exact cursor bone ID/name missing in geometry")
    bone = matches[0]
    local_error = trs_errors(predicted["after"], local_only(bone))
    require(local_error["passed"], "Replay predicted local does not match the snapshot ChinTip")
    modifiers = [value for value in snapshot["abmx_runtime"]["bones"] if value["name"] == CHIN]
    require(len(modifiers) == 1, "Exact snapshot ChinTip private state missing/ambiguous")
    wrapper = modifiers[0]["runtime_baseline"]
    require(wrapper["missing_fields"] == [] and wrapper["bone_transform_id"] == bone["id"] and wrapper["frame_count"] == cursor["frame"], "Snapshot cache does not bind to exact cursor bone/frame")
    require(wrapper["assembly_mvid"] == trace["metadata"]["plugin_mvid"], "Snapshot private cache source mismatch")
    private_error = cache_errors(predicted["cache_after"], wrapper["fields"])
    require(private_error["passed"], "Replay predicted private state differs from exact geometry snapshot")
    return predicted["after"], {"cursor": cursor, "selected_observed_call_sequence": sequence,
              "trace_total_calls": len(trace["events"]), "later_calls_excluded": len(trace["events"]) - sequence,
              "selected_call_frame": row["frame"], "snapshot_frame": cursor["frame"],
              "frames_since_selected_call": cursor["frame"] - row["frame"],
              "frame_binding_note": "Cursor selects completed call sequence; capture may precede current-frame LateUpdate. Snapshot current local/cache are separately required to match replay prediction.",
              "chain_segment_start_sequence": row["chain_segment"],
              "override_source": "independent replay transition/propagated chain prediction; never copied actual after",
              "prediction": predicted, "snapshot_local_error": local_error, "snapshot_cache_error": private_error,
              "bone_transform_id": bone["id"], "bone_path": bone["path"]}


def cache_correspondence(rig, mesh):
    source = np.asarray(mesh["source"]["vertices"], float)
    require(source.shape == rig.verts.shape and np.array_equal(np.asarray(mesh["source"]["triangles"]).reshape(-1, 3), rig.faces), "Cached/live topology correspondence differs")
    error = vertex_errors(source, rig.verts)
    require(error["max_abs_component"] <= 1e-6, "Cached source vertices differ")
    require(rig.skin_bone_names == mesh["bone_names"] and np.array_equal(rig.bone_idx, mesh["source"]["bone_indices"]), "Source palette/indices differ")
    require(np.allclose(rig.bone_w, mesh["source"]["bone_weights"], atol=1e-7, rtol=0), "Source weights differ")
    bindpose_error = float(np.max(np.abs(rig.bindpose - np.asarray(mesh["source"]["bindposes"]).reshape(-1, 4, 4))))
    require(bindpose_error <= 1e-6, "Source bindposes differ")
    return {"vertices": error, "bindpose_max_abs": bindpose_error, "topology_palette_indices_weights_matched": True}


def reconstruct(rig, native59, chin_local, mesh):
    """Cached native locals + one independently predicted local -> new FK/LBS."""
    native_world = _fk_world(rig, native59, None)
    local = {}
    for pid in rig._topo:
        parent = rig.bones[pid]["parent"]
        local[pid] = np.linalg.inv(native_world[parent]) @ native_world[pid] if parent in native_world else native_world[pid].copy()
    pid = rig.name2pid[CHIN]
    local[pid] = _trs(chin_local["local_position"], chin_local["local_rotation_xyzw"], chin_local["local_scale"])
    world = {}
    for key in rig._topo:
        parent = rig.bones[key]["parent"]
        world[key] = world[parent] @ local[key] if parent in world else local[key]
    # Source and cache asset correspondence is checked separately; expression
    # delta is an actual independent nuisance input, not an inferred face target.
    delta, active = blendshape_delta(mesh)
    homogeneous = np.column_stack([rig.verts + delta, np.ones(len(rig.verts))])
    matrices = np.stack([world[rig.name2pid[name]] @ rig.bindpose[index] for index, name in enumerate(rig.skin_bone_names)])
    influences = matrices[rig.bone_idx]
    vertices = (np.einsum("nkij,nj->nki", influences, homogeneous)[..., :3] * rig.bone_w[..., None]).sum(axis=1)
    return vertices, active


def verify_checkpoint(case, snapshot):
    require(sha(case["checkpoint"]) == case["checkpoint_sha256"], "Original checkpoint hash mismatch")
    checkpoint = read(case["checkpoint"])
    require(checkpoint["head_id"] == case["head_id"] == snapshot["character"]["head_id"], "Actual/cache/checkpoint head differs")
    winners = [candidate for candidate in checkpoint["candidates"] if candidate["kind"] == "bounded_search_winner"]
    require(len(winners) == 1 and winners[0]["search"]["active_native"] == list(range(59)), "Not the original full59 winner")
    native = np.asarray(snapshot["character"]["shape_value_face"], float)
    require(native.shape == (59,) and np.isfinite(native).all() and np.max(np.abs(native - winners[0]["native59"])) < 2e-6, "Actual geometry native59 differs from unchanged checkpoint")
    patch = case["patch"]
    require(patch["trace_cursor_at_patch"]["session_id"] == snapshot["abmx_trace_cursor"]["session_id"], "Patch references another trace")
    require(patch["patch_frame"] <= snapshot["frame_count"], "Geometry predates parameter patch")
    actual = [value for value in patch["current"]["bones"] if value["name"] == CHIN]
    require(len(actual) == 1 and set(winners[0]["abmx"]) == {CHIN}, "Actual ChinTip patch missing")
    for key in ("scale", "length", "position", "rotation"):
        require(np.max(np.abs(np.asarray(actual[0][key]) - winners[0]["abmx"][CHIN][key])) < 2e-6, "Actual ABMX patch differs from unchanged checkpoint")
    return checkpoint, winners[0], native.tolist(), actual[0]


def verify_pairs(case, snapshot, snapshot_sha):
    views = case["capture"]["views"]
    require(len(views) == 3, "Expected three actual paired views")
    rows = []
    for view in views:
        pair = view["paired_geometry"]
        require(pair["sha256"] == snapshot_sha and Path(pair["path"]).resolve() == Path(case["geometry"]["path"]).resolve(), "Paired view references another geometry export")
        require(view["paired_pose_unchanged"] is True and view["pose_signature_before_render"] == view["pose_signature_after_render"] == snapshot["pose_signature"], "Paired view pose changed")
        require(view["frame_count"] == view["frame_count_before_render"] == snapshot["frame_count"], "View not same-frame")
        require(view["paired_visibility_sampled_unchanged"] is True, "View material/visibility changed")
        path = Path(view["path"])
        require(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "Missing actual PNG")
        rows.append({"path": str(path.resolve()), "sha256": sha(path), "yaw": view["yaw"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "New report directory required")
    manifest, contract = read(args.manifest), read(args.contract)
    require(all(manifest.get(key) is True for key in ("state_restored", "expression_restored", "bone_restored")), "Capture job not completed/restored")
    require(sha(contract["assembly_path"]) == contract["assembly_sha256"] and sha(contract["unity_core_path"]) == contract["unity_core_sha256"], "Installed assemblies differ from independent source contract")
    require(all(sha(source["path"]) == source["sha256"] for source in contract["sources"]), "Installed decompiled source changed")
    require(len(manifest["cases"]) == 4 and {case["head_id"] for case in manifest["cases"]} == {0, 1, 2, 3}, "Expected four actual head cases")
    args.out.mkdir(parents=True)
    summary = {"schema_version": 1, "manifest_path": str(args.manifest.resolve()), "manifest_sha256": sha(args.manifest),
               "contract_path": str(args.contract.resolve()), "contract_sha256": sha(args.contract), "normalized_tolerance": TOLERANCE,
               "scope": "Specific four snapshots: state-conditioned replay-derived ChinTip + native cached rest of head FK + actual expression + recorded ancestor; full o_head versus BakeMesh",
               "static_whole_head_model_fixed": False, "likeness_validated": False,
               "implementation_hashes": [{"path": str(path.resolve()), "sha256": sha(path)} for path in (Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("validate_trace.py"), ROOT / "src/hs2_mesh_deform.py", ROOT / "tools/unity_parity/geometry.py")],
               "cases": []}
    for case in manifest["cases"]:
        report = {"name": case["name"], "head_id": case["head_id"], "passed": False}
        compact = {"name": case["name"], "head_id": case["head_id"], "passed": False}
        try:
            require(sha(case["trace"]) == case["trace_sha256"] and sha(case["geometry"]["path"]) == case["geometry"]["sha256"], "Trace/geometry SHA mismatch")
            trace, snapshot = read(case["trace"]), read(case["geometry"]["path"])
            trace_report = validate(trace, contract)
            report["call_replay"] = trace_report
            predicted, cursor = select_cursor_prediction(snapshot, trace, trace_report)
            report["exact_cursor_binding"] = cursor
            checkpoint, winner, native, actual_ab = verify_checkpoint(case, snapshot)
            report["unchanged_original_checkpoint"] = {"path": case["checkpoint"], "sha256": sha(case["checkpoint"]),
                        "kind": winner["kind"], "mode": checkpoint["mode"], "native59": native,
                        "sampling_profile": checkpoint["sampling_profile"], "original_quality_valid": winner["quality"]["quality_valid"],
                        "original_accepted": winner["accepted"], "official_selected_candidate_index_unchanged": checkpoint["selected_candidate_index"]}
            report["paired_views"] = verify_pairs(case, snapshot, case["geometry"]["sha256"])
            certificate = analyze_snapshot(snapshot, normalized_tolerance=TOLERANCE)
            certificate.update({"snapshot_path": case["geometry"]["path"], "snapshot_sha256": case["geometry"]["sha256"]})
            report["engine_certificate"] = certificate
            require(all(mesh["certified"] for mesh in certificate["meshes"]), "Actual-bone LBS cannot certify an exported renderer")
            certified, skipped = load_certified_head(snapshot, case["geometry"]["sha256"], certificate)
            heads = [value for value in certified if value.name == "o_head"]
            require(len(heads) == 1, "Expected one certified visible o_head renderer")
            certified_head = heads[0]
            meshes = [mesh for mesh in snapshot["meshes"] if mesh["renderer_path"] == certified_head.path]
            require(len(meshes) == 1, "Ambiguous source head renderer")
            mesh = meshes[0]
            head_cert = next(row for row in certificate["meshes"] if row["renderer_path"] == certified_head.path)
            require(head_cert["declared_influences"] == 4 and head_cert["influences_overridden"] is False, "Offline full-four-weight scope differs from actual skin quality")
            report["selected_head_renderer"] = {"path": certified_head.path, "source_sha256": certified_head.source_hash, "world_candidate": certified_head.candidate}
            report["skipped_renderers"] = skipped
            rig = HeadRig(case["head_id"], sampling_profile=checkpoint["sampling_profile"])
            cache_paths = [Path(rig.data_dir) / name for name in ("o_head_mesh.npz", "skeleton.json", "anmShapeHead.json")]
            cache_paths += [Path(rig.root_dir) / name for name in ("enums.json", "customhead.json", "update_eqns.json")]
            report["cached_rig_source_hashes"] = [{"path": str(path.resolve()), "sha256": sha(path)} for path in cache_paths]
            report["cache_correspondence"] = cache_correspondence(rig, mesh)
            locals_report = compare_bone_locals(snapshot, rig, None)
            others = [row for row in locals_report["bones"] if row["is_head_skin_bone"] and row["name"] != CHIN]
            require({row["name"] for row in others} == set(rig.skin_bone_names) - {CHIN}, "Other head skin palette locals incomplete")
            require(all(row["cached_parent"] == row["live_parent"] for row in others), "Other skin-bone cached/live parent differs")
            other_max = {"position": max(row["position_error"] for row in others), "rotation_deg": max(row["rotation_angle_deg"] for row in others), "scale": max(row["scale_error"] for row in others)}
            report["other_native_head_skin_bone_locals"] = {"maxima": other_max, "bones": others}
            require(other_max["position"] <= 1e-5 and other_max["scale"] <= 1e-5 and other_max["rotation_deg"] <= .001, "Other native skin locals differ; one-bone state-conditioned model insufficient")
            vertices, active = reconstruct(rig, native, predicted, mesh)
            transforms = {value["id"]: value for value in snapshot["transforms"]}
            factor, evidence = recorded_uniform_renderer_scale(mesh, transforms)
            _, uncompensated = rigid_alignment(vertices, certified_head.vertices, unit_scale=1)
            _, comparison = rigid_alignment(vertices, certified_head.vertices, unit_scale=factor)
            report.update({"active_actual_blendshapes": active, "recorded_ancestor_uniform_scale": evidence,
                           "unit_scale": 1, "scale_fitting": False, "proper_rigid_pose_only": True,
                           "uncompensated_comparison": uncompensated, "state_conditioned_comparison": comparison,
                           "static_model_comparison_retained": compare_offline(snapshot, profile=checkpoint["sampling_profile"],
                                                   abmx={CHIN: {key: actual_ab[key] for key in ("scale", "length", "position", "rotation")}}, apply_renderer_uniform_scale=True)})
            error = comparison["rigid_errors"]["max_normalized"]
            report["passed"] = error is not None and np.isfinite(error) and error <= TOLERANCE
            compact.update({"passed": report["passed"], "call_count": trace_report["observed_call_count"],
                            "snapshot_call_cursor": cursor["selected_observed_call_sequence"], "later_calls_excluded": cursor["later_calls_excluded"],
                            "chain_segment_start": cursor["chain_segment_start_sequence"], "snapshot_local_error": cursor["snapshot_local_error"],
                            "other_skin_local_maxima": other_max, "full_head_state_conditioned_errors": comparison["rigid_errors"],
                            "static_model_errors": report["static_model_comparison_retained"][0].get("comparison", {}).get("rigid_errors"),
                            "ancestor_scale": factor, "original_offline_quality_valid": winner["quality"]["quality_valid"]})
        except (ReplayRejected, ValueError, KeyError, TypeError, OSError) as exc:
            report["rejection"] = str(exc)
            compact["rejection"] = str(exc)
        destination = args.out / (case["name"] + ".json")
        require(destination.resolve().parent == args.out.resolve(), "Unsafe report name")
        save(destination, report)
        compact["report"] = str(destination.resolve())
        summary["cases"].append(compact)
        save(args.out / "progress.json", summary)
        print(json.dumps(compact), flush=True)
    summary["passed_count"] = sum(row["passed"] for row in summary["cases"])
    summary["original_checkpoints_unchanged"] = all(sha(case["checkpoint"]) == case["checkpoint_sha256"] for case in manifest["cases"])
    require(summary["original_checkpoints_unchanged"], "Original checkpoint changed during read-only analysis")
    summary["status"] = "passed_specific_state_conditioned_snapshots" if summary["passed_count"] == 4 else "failed_specific_state_conditioned_snapshots"
    summary["limitations"] = ["Current cached-rest stateless ABMX optimizer is not fixed; replay state is an explicit observed conditioning input.",
                              "Only one traced ChinTip override and four actual cases; other ABMX bones/conditions need separate runtime evidence.",
                              "Expressions and recorded ancestor scale are actual nuisance inputs, not fitted facial geometry or scale.",
                              "Original offline quality-invalid winners remain diagnostic; runtime parity does not make them safe fits.",
                              "No character likeness or global base expressivity proof."]
    save(args.out / "geometry_summary.json", summary)
    return 0 if summary["passed_count"] == 4 else 2


if __name__ == "__main__":
    raise SystemExit(main())
