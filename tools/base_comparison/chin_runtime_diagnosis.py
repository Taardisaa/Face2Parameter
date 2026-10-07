"""Diagnose cached ABMX length/offset history without changing a rig or game.

Recurrence counts are discrete retrospective diagnostics, not calibration or a
replacement for captured private plugin state/application timing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hs2_mesh_deform import HeadRig, _fk_world

CHIN = "cf_J_ChinTip_s"


def local_native(rig, values):
    world = _fk_world(rig, values)
    pid = rig.name2pid[CHIN]
    return (np.linalg.inv(world[rig.bones[pid]["parent"]]) @ world[pid])[:3, 3]


def actual_local(snapshot):
    matches = [row for row in snapshot["transforms"] if row["name"] == CHIN]
    if len(matches) != 1:
        raise ValueError("Ambiguous ChinTip transform")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in manifest["cases"]}
    before = manifest["before"]
    before_values = [item["value"] for item in sorted(before["face_shapes"]["items"], key=lambda item: item["index"])]
    before_rig = HeadRig(before["face_base"]["head_id"])
    before_position = local_native(before_rig, before_values)
    old_length = float(np.linalg.norm(before_position))
    rows = []
    for head in range(4):
        case = cases[f"head_{head}_native_ABMX"]
        baseline = cases[f"head_{head}_baseline"]
        snapshot = json.loads(Path(case["geometry"]["path"]).read_text(encoding="utf-8"))
        base_snapshot = json.loads(Path(baseline["geometry"]["path"]).read_text(encoding="utf-8"))
        rig = HeadRig(head)
        native = local_native(rig, case["native59_actual"])
        rest = np.asarray(rig.bones[rig.name2pid[CHIN]]["pos"], float)
        live = actual_local(snapshot)
        identity = actual_local(base_snapshot)
        live_pos = np.asarray(live["local_position"], float)
        base_pos = np.asarray(identity["local_position"], float)
        modifiers = case["abmx_actual"]
        offset, length = np.asarray(modifiers["position"], float), modifiers["length"]
        row = {"head_id": head, "snapshot_sha256": hashlib.sha256(Path(case["geometry"]["path"]).read_bytes()).hexdigest(),
               "renderer_parent_evidence": {"cached_parent": rig.bones[rig.bones[rig.name2pid[CHIN]]["parent"]]["name"],
                                            "live_parent_id": live["parent_id"], "actual_transform_path": live["path"]},
               "cached_rest_position": rest.tolist(), "cached_rest_length": float(np.linalg.norm(rest)),
               "native59_post_shape_position": native.tolist(), "native_length": float(np.linalg.norm(native)),
               "same_head_identity_live_position": base_pos.tolist(), "same_head_identity_live_length": float(np.linalg.norm(base_pos)),
               "same_head_identity_vs_cached_rest_l2": float(np.linalg.norm(base_pos - rest)),
               "actual_chin_position": live_pos.tolist(), "actual_length_modifier": length, "actual_offset": offset.tolist(),
               "existing_offline_prediction": (native * length + offset).tolist(),
               "existing_offline_position_error": float(np.linalg.norm(native * length + offset - live_pos)),
               "derived_length_baseline_from_actual_position_not_authoritative_cache": float(np.linalg.norm(live_pos - offset) / length),
               "single_length_application_using_before_history": (native / np.linalg.norm(native) * old_length * length + offset).tolist()}
        # Keep each discrete observation: this is not a continuous fitted parameter.
        trajectory = []
        position = native.copy()
        for step in range(21):
            trajectory.append({"applications": step, "position": position.tolist(), "l2_error_vs_live": float(np.linalg.norm(position - live_pos))})
            position = position / np.linalg.norm(position) * old_length * length + offset
        row["historical_length_plus_repeated_offset_diagnostic"] = {"length_from_independent_manifest_before": old_length,
                        "formula": "p[k+1] = p[k]/norm(p[k]) * length_before * actual_L + actual_offset",
                        "diagnostic_counts": trajectory, "smallest_discrete_residual": min(trajectory, key=lambda value: value["l2_error_vs_live"]),
                        "warning": "Application count is retrospectively compared, not observed. Do not silently fit it or treat it as a model patch."}
        if length == 1:
            row["branch_note"] = "HasLength false: Apply uses _posBaseline + offset; repeated normalization branch is not active unless force-update flag is set."
        else:
            row["branch_note"] = "HasLength true: persistent length normalization + additive offset on every Apply; previous offset changes next direction if native shape is not rewritten."
        row["exported_abmx_runtime_diagnostic_if_available"] = snapshot.get("abmx_runtime")
        if snapshot.get("abmx_runtime") is not None:
            matches = [bone for bone in snapshot["abmx_runtime"]["bones"] if bone["name"] == CHIN]
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous actual private ChinTip state")
            modifier = matches[0]
            runtime = modifier["runtime_baseline"]
            fields = runtime["fields"]
            if runtime["missing_fields"] or runtime["bone_transform_id"] != live["id"] or runtime["frame_count"] != snapshot["frame_count"]:
                raise ValueError("Private cache is incomplete or does not bind to the actual same-frame bone")
            if fields["_hasBaseline"] is not True:
                raise ValueError("Actual ChinTip private cache has no baseline")
            cached_length = float(fields["_lenBaseline"])
            if not np.isfinite(cached_length) or cached_length <= 0:
                raise ValueError("Invalid actual private length baseline")
            if any(not np.allclose(modifier[key], modifiers[key], atol=2e-6, rtol=0) for key in ("scale", "length", "position", "rotation")):
                raise ValueError("Same-frame exported ABMX modifier differs from readback")
            private_trajectory = []
            position = native.copy()
            for step in range(21):
                private_trajectory.append({"applications": step, "position": position.tolist(), "l2_error_vs_live": float(np.linalg.norm(position - live_pos))})
                position = position / np.linalg.norm(position) * cached_length * length + offset
            row["actual_private_cache_independent_evidence"] = {
                "runtime_baseline": runtime,
                "modifier_matches_actual_readback": True,
                "same_frame_and_exact_bone_id_matched": True,
                "captured_length_vs_independent_before_native_length_abs": abs(cached_length - old_length),
                "captured_position_baseline_vs_before_native_position_l2": float(np.linalg.norm(np.asarray(fields["_positionBaseline"]) - before_position)),
                "captured_posBaseline_vs_current_native_position_l2": float(np.linalg.norm(np.asarray(fields["_posBaseline"]) - native)),
                "captured_length_vs_same_head_identity_live_length_abs": abs(cached_length - np.linalg.norm(base_pos)),
                "length_normalization_private_cache_formula": "p[k+1] = normalize(p[k]) * captured_private_lenBaseline * actual_L + actual_offset",
                "one_apply_private_cache_prediction": private_trajectory[1],
                "diagnostic_sequence_private_cache": private_trajectory,
                "smallest_discrete_residual_private_cache": min(private_trajectory, key=lambda value: value["l2_error_vs_live"]),
                "length_and_baseline_are_observed_not_fitted": True,
                "application_count_is_observed": False,
                "normalization_branch_active_by_length": length != 1,
                "position_only_private_cache_prediction": (np.asarray(fields["_posBaseline"]) + offset).tolist() if length == 1 else None,
                "position_only_private_cache_l2_error_vs_actual": float(np.linalg.norm(np.asarray(fields["_posBaseline"]) + offset - live_pos)) if length == 1 else None,
                "warning": "Actual cache confirms persistent length history; discrete counts still need explicit Apply timing evidence before formal runtime prediction."}
            if length == 1:
                row["actual_private_cache_independent_evidence"]["diagnostic_sequence_private_cache"] = []
                row["actual_private_cache_independent_evidence"]["smallest_discrete_residual_private_cache"] = None
                row["actual_private_cache_independent_evidence"]["one_apply_private_cache_prediction"] = None
        rows.append(row)
    report = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "before_head_id": before_rig.head_id, "before_native_chin_position": before_position.tolist(), "before_native_chin_length": old_length,
              "cases": rows,
              "decompiled_installed_logic": {"Apply": "HS2Mod/tools/ABMX_BoneModifier.cs:163-186",
                        "CollectBaseline": "HS2Mod/tools/ABMX_BoneModifier.cs:243-267",
                        "LateUpdate": "HS2Mod/tools/ABMX_BoneController.cs:240-260",
                        "UpdateBaseline": "HS2Mod/tools/ABMX_BoneController.cs:374-390"},
              "minimum_fix_recommendation": [
                    "Make runtime state explicit: capture persistent _lenBaseline/_positionBaseline and current _posBaseline/_sclBaseline/_rotBaseline plus Apply flags/timing.",
                    "For an effective deterministic comparison contract, either freshly collect a known modifier baseline and state exactly when native FK/ABMX Apply occurs, or reconstruct the observed stateful Apply sequence from captured cache.",
                    "Update NumPy _fk_world and TorchHeadRig.local_transforms consistently for length normalization and cached baseline semantics only after real private-cache evidence confirms the contract.",
                    "A single replacement p*L with normalize(p)*rest_length*L does not explain these captures: stale length and repeated offsets both matter.",
                    "Preserve benchmark quality decisions and original target/parameters. Runtime-correct re-evaluation is separately versioned evidence, never overwrite the original report."],
              "scope": "Diagnoses ChinTip local positions for four specific frozen mixed snapshots; no geometry fitting, no similarity claim, no runtime mutation."}
    if args.out.exists():
        raise ValueError("Do not overwrite old diagnosis")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"report={args.out.resolve()}")


if __name__ == "__main__":
    main()
