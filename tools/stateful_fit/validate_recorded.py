"""Validate parameter-predicted initial locals and stateful full-head Torch FK.

Unlike recorded-call replay, no observed before/after TRS is substituted into
candidate transforms. Native59 supplies the clean boundary and cache TRS;
historical length fields/flags and measured call count remain explicit inputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig
from tools.stateful_fit.adapter import CHIN, NativeBaselineProtocol, StatefulTorchHeadRig, read, sha
from tools.abmx_replay.validate_trace import trs_errors
from tools.abmx_replay.validate_geometry import cache_correspondence, verify_checkpoint
from tools.unity_parity.geometry import analyze_snapshot, blendshape_delta, recorded_uniform_renderer_scale, rigid_alignment


def validate_case(case, manifest_path, contract_path, device, *, protocol=None):
    snapshot = read(case["geometry"]["path"])
    if protocol is None:
        protocol = NativeBaselineProtocol.from_manifest(manifest_path, contract_path, case["head_id"])
    checkpoint, winner, _, _ = verify_checkpoint(case, snapshot)
    rig = HeadRig(case["head_id"], sampling_profile="vanilla")
    model = StatefulTorchHeadRig(rig, protocol, device=device, dtype=torch.float64)
    native = torch.as_tensor(protocol.reference_native[None], device=device, dtype=model.dtype)
    ab = model.ab_tensor({CHIN: {"scale": protocol.reference_modifier[:3], "length": protocol.reference_modifier[3],
                               "position": protocol.reference_modifier[4:7], "rotation": protocol.reference_modifier[7:]}})
    with torch.no_grad():
        pos, quat, scale = model.local_transforms(native, ab)
        predicted = {"local_position": pos[0, model.chin_bone].cpu().tolist(),
                     "local_rotation_xyzw": quat[0, model.chin_bone].cpu().tolist(),
                     "local_scale": scale[0, model.chin_bone].cpu().tolist()}
        bone_id = read(case["trace"])["events"][0]["before"]["bone_transform_id"]
        actual_bones = [row for row in snapshot["transforms"] if row["id"] == bone_id and row["name"] == CHIN]
        if len(actual_bones) != 1:
            raise ValueError("Actual snapshot ChinTip identity missing")
        actual_local = {key: actual_bones[0][key] for key in predicted}
        local_error = trs_errors(predicted, actual_local)
        certificate = analyze_snapshot(snapshot)
        if not all(row["certified"] for row in certificate["meshes"]):
            raise ValueError("Actual renderer LBS/BakeMesh certificate failed")
        heads = [mesh for mesh in snapshot["meshes"] if mesh["mesh_name"] == "o_head" and mesh["enabled"] and mesh["active_in_hierarchy"]]
        if len(heads) != 1:
            raise ValueError("Unique visible head renderer required")
        mesh = heads[0]
        correspondence = cache_correspondence(rig, mesh)
        cert = next(row for row in certificate["meshes"] if row["renderer_path"] == mesh["renderer_path"])
        if cert["declared_influences"] != 4 or cert["influences_overridden"] or len(cert["matching_candidates"]) != 1:
            raise ValueError("Unique actual four-influence world convention required")
        actual = np.asarray(mesh["baked"]["world_candidates"][cert["selected_candidate"]]["vertices"])
        delta, active = blendshape_delta(mesh)
        world = model.bone_world(native, ab)[0]
        skin = world[model.skin_bone] @ model.bindpose
        homogeneous = torch.cat([torch.as_tensor(rig.verts + delta, device=device, dtype=model.dtype),
                                 torch.ones(len(rig.verts), 1, device=device, dtype=model.dtype)], 1)
        influences = skin[model.bone_idx]
        vertices = (torch.einsum("vkij,vj->vki", influences, homogeneous)[..., :3] * model.bone_w[..., None]).sum(1).cpu().numpy()
        ancestor, ancestor_evidence = recorded_uniform_renderer_scale(mesh, {row["id"]: row for row in snapshot["transforms"]})
        _, uncompensated = rigid_alignment(vertices, actual, unit_scale=1)
        _, comparison = rigid_alignment(vertices, actual, unit_scale=ancestor)
    return {"head_id": case["head_id"], "passed": local_error["passed"] and comparison["rigid_errors"]["max_normalized"] <= 1e-5,
            "local_error": local_error, "full_head_errors": comparison, "uncompensated_errors": uncompensated,
            "native59_derived_initial_and_baseline_trs": True, "actual_after_used_as_prediction_input": False,
            "historical_length_state_measured": True, "protocol": model.protocol_metadata,
            "cache_correspondence": correspondence, "active_expression_inputs": active,
            "checkpoint_sha256": sha(case["checkpoint"]), "original_offline_quality_valid": winner["quality"]["quality_valid"],
            "ancestor_scale_evidence": ancestor_evidence, "unit_scale": 1, "additional_fitted_scale_applied": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("New evidence path required")
    report = {"manifest_sha256": sha(args.manifest), "contract_sha256": sha(args.contract),
              "device": args.device, "normalized_tolerance": 1e-5, "cases": [],
              "source_hashes": [{"path": str(path.resolve()), "sha256": sha(path)} for path in
                                (Path(__file__), Path(__file__).with_name("adapter.py"), ROOT / "src/hs2_abmx_torch.py", ROOT / "src/hs2_deform_torch.py")],
              "new_candidate_runtime_certified": False, "character_ready": False}
    for case in read(args.manifest)["cases"]:
        row = validate_case(case, args.manifest, args.contract, args.device)
        report["cases"].append(row)
        print(json.dumps({"head": row["head_id"], "passed": row["passed"], "max_normalized": row["full_head_errors"]["rigid_errors"]["max_normalized"]}), flush=True)
    report["passed"] = len(report["cases"]) == 4 and all(row["passed"] for row in report["cases"])
    report["scope"] = "Recorded four complete native59/candidate cases; candidate native FK supplies initial/cache TRS, fixed measured history and exact observed Apply count, no unknown writer boundaries"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
