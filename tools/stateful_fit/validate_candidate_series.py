"""Actual new candidate early/late capture versus parameter-predicted head.

Temporal drift is measured on the actual world-space source-corresponding
surface, removing only proper rigid pose. Exact observed counts are reported
separately from the search's declared8-call protocol.
"""
from pathlib import Path
import argparse
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.stateful_fit.adapter import NativeBaselineProtocol, read, sha
from tools.stateful_fit.validate_recorded import validate_case
from tools.abmx_replay.validate_geometry import verify_pairs
from tools.unity_parity.geometry import analyze_snapshot, rigid_alignment


def actual_head(snapshot):
    certificate = analyze_snapshot(snapshot)
    meshes = [row for row in snapshot["meshes"] if row["mesh_name"] == "o_head" and row["enabled"] and row["active_in_hierarchy"]]
    if len(meshes) != 1:
        raise ValueError("Unique visible o_head required")
    mesh = meshes[0]
    cert = next(row for row in certificate["meshes"] if row["renderer_path"] == mesh["renderer_path"])
    if not cert["certified"] or len(cert["matching_candidates"]) != 1:
        raise ValueError("Actual world-space head convention not certified")
    return mesh, np.asarray(mesh["baked"]["world_candidates"][cert["selected_candidate"]]["vertices"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve previous evidence")
    manifest, contract = read(args.manifest), read(args.contract)
    if not all(manifest.get(key) is True for key in ("state_restored", "expression_restored", "bone_restored")):
        raise ValueError("Capture job did not complete/restore")
    rows = []
    for source_case in manifest["cases"]:
        case_rows, surfaces = [], []
        trace = read(source_case["trace"])
        if sha(source_case["trace"]) != source_case["trace_sha256"]:
            raise ValueError("Trace producer hash mismatch")
        for phase in ("early", "late"):
            case = copy.deepcopy(source_case)
            if phase == "late":
                if case.get("late_capture") is None:
                    raise ValueError("Actual later phase required")
                case["capture"] = case["late_capture"]
                case["geometry"] = case["capture"]["paired_geometry"]
            if sha(case["geometry"]["path"]) != case["geometry"]["sha256"]:
                raise ValueError("Geometry producer hash mismatch")
            snapshot = read(case["geometry"]["path"])
            paired = verify_pairs(case, snapshot, case["geometry"]["sha256"])
            protocol = NativeBaselineProtocol(trace, contract, snapshot, trace_path=case["trace"], contract_path=args.contract,
                                              snapshot_path=case["geometry"]["path"])
            result = validate_case(case, args.manifest, args.contract, "cuda", protocol=protocol)
            result.update(phase=phase, paired_views=paired)
            case_rows.append(result)
            surfaces.append(actual_head(snapshot))
        if surfaces[0][0]["source_geometry_sha256"] != surfaces[1][0]["source_geometry_sha256"] or not np.array_equal(surfaces[0][0]["baked"]["triangles"], surfaces[1][0]["baked"]["triangles"]):
            raise ValueError("Early/late source identity/topology differs")
        _, drift = rigid_alignment(surfaces[1][1], surfaces[0][1], unit_scale=1)
        stable = drift["rigid_errors"]["max_normalized"] <= 1e-5
        row = {"head_id": source_case["head_id"], "phases": case_rows, "actual_temporal_drift": drift,
               "temporal_tolerance": 1e-5, "temporally_stable_in_this_window": stable,
               "passed": all(result["passed"] for result in case_rows) and stable,
               "observed_counts": [result["protocol"]["observed_candidate_apply_count"] for result in case_rows],
               "search_declared_count": 8, "count_was_fitted": False,
               "original_offline_quality_valid": all(result["original_offline_quality_valid"] for result in case_rows),
               "candidate_accepted_for_fitting": all(result["passed"] and result["original_offline_quality_valid"] for result in case_rows) and stable,
               "scope": "Actual early/late stages of this candidate; different counts retained explicitly, not claimed as an exact8-call capture."}
        rows.append(row)
        print(json.dumps({"head": row["head_id"], "passed": row["passed"], "actual_counts": row["observed_counts"],
                          "actual_drift_normalized": drift["rigid_errors"]["max_normalized"]}), flush=True)
    report = {"passed": bool(rows) and all(row["passed"] for row in rows), "cases": rows,
              "manifest_sha256": sha(args.manifest), "contract_sha256": sha(args.contract),
              "source_hashes": [{"path": str(path.resolve()), "sha256": sha(path)} for path in
                                (Path(__file__), Path(__file__).with_name("adapter.py"), Path(__file__).with_name("validate_recorded.py"), ROOT / "src/hs2_abmx_torch.py")],
              "character_ready": False, "anatomical_correspondence_validated": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
