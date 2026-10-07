"""Current-state acceptance audit; scopes remain explicit and character deferred."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.stateful_fit.adapter import read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve old audit")
    output = ROOT / "outputs/stateful_fit_20261005"
    paths = [output / name for name in ("recorded_forward_final.json", "gradient_v2.json", "protocol_rejections_v1.json",
                                        "head3_candidate_series_final.json", "other_candidates_series_v1.json")]
    paths += [ROOT / "outputs/abmx_replay_20261005" / name for name in ("torch_review_cpu_fixed.json", "torch_review_cuda_fixed.json")]
    evidence = [{"path": str(path.resolve()), "sha256": sha(path)} for path in paths]
    reports = [read(path) for path in paths]
    for report in reports[:5]:
        if report.get("passed") is not True:
            raise ValueError("An actual acceptance report did not pass")
        for source in report.get("source_hashes", []):
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Report source changed: " + source["path"])
    for report in reports[5:]:
        if report["actual_call_count"] != 114 or report["forward_passed"] != 114 or report["finite_gradient_calls"] != 114:
            raise ValueError("Independent actual-call Torch review failed/incomplete")
        for source in report["implementation_hashes"]:
            if sha(source["path"]) != source["sha256"]:
                raise ValueError("Independently reviewed source changed")
    comparison_path = ROOT / "outputs/stateful_base_comparison_20261005/benchmark_summary.json"
    benchmark = read(comparison_path)
    if benchmark["completed_cases"] != 12 or benchmark["common_apply_count"] != 8:
        raise ValueError("Common full benchmark incomplete")
    if sha(benchmark["original_config"]) != benchmark["original_config_sha256"]:
        raise ValueError("Original benchmark config changed")
    for source in benchmark["source_hashes"]:
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("Benchmark source changed during/after the run")
    rows = []
    for case in benchmark["cases"]:
        if sha(case["checkpoint"]) != case["checkpoint_sha256"]:
            raise ValueError("New checkpoint changed")
        checkpoint = read(case["checkpoint"])
        winner = next(candidate for candidate in checkpoint["candidates"] if candidate["kind"] == "bounded_search_winner")
        if winner["search"]["active_native"] != list(range(59)) or winner["search"]["completed_iterations"] != 40:
            raise ValueError("Full59/40-step contract changed")
        if winner["accepted"] and not winner["quality"]["quality_valid"]:
            raise ValueError("Invalid quality candidate was accepted")
        if checkpoint["mode"] == "native+ABMX" and checkpoint["abmx_protocol"]["declared_apply_count"] != 8:
            raise ValueError("ABMX count differs between heads")
        rows.append({"head_id": case["head_id"], "mode": case["mode"], "winner_rms": case["winner_rms"],
                     "winner_quality_valid": winner["quality"]["quality_valid"], "winner_accepted": winner["accepted"]})
    runtime_cases = reports[3]["cases"] + reports[4]["cases"]
    if len(runtime_cases) != 4 or {case["head_id"] for case in runtime_cases} != {0, 1, 2, 3}:
        raise ValueError("Four-head new-candidate temporal verification incomplete")
    runtime = [{"head_id": case["head_id"], "observed_counts": case["observed_counts"],
                "early_max_normalized": case["phases"][0]["full_head_errors"]["rigid_errors"]["max_normalized"],
                "late_max_normalized": case["phases"][1]["full_head_errors"]["rigid_errors"]["max_normalized"],
                "actual_drift_normalized": case["actual_temporal_drift"]["rigid_errors"]["max_normalized"],
                "candidate_quality_valid": case["original_offline_quality_valid"],
                "candidate_accepted_in_this_contract": case["candidate_accepted_for_fitting"]} for case in runtime_cases]
    # Actual producer manifests must still match the referenced checkpoint,
    # geometry and trace files and report full restoration.
    for report in reports[3:5]:
        folders = {Path(phase["protocol"]["geometry_path"]).parent for case in report["cases"] for phase in case["phases"]}
        if len(folders) != 1:
            raise ValueError("Ambiguous producer manifest directory")
        manifest_path = next(iter(folders)) / "live_cases.json"
        producer = read(manifest_path)
        if sha(manifest_path) != report["manifest_sha256"] or not all(producer.get(key) is True for key in ("state_restored", "expression_restored", "bone_restored")):
            raise ValueError("Actual capture producer changed or failed restoration")
        for case in report["cases"]:
            for phase in case["phases"]:
                protocol = phase["protocol"]
                for name in ("trace", "contract", "geometry"):
                    if sha(protocol[name + "_path"]) != protocol[name + "_sha256"]:
                        raise ValueError("Runtime evidence source file changed")
    summary = {"passed": True, "benchmark_cases": rows, "new_runtime_cases": runtime, "evidence": evidence,
               "benchmark_summary": str(comparison_path.resolve()), "benchmark_sha256": sha(comparison_path),
               "gradient_all59_native_passed": reports[1]["all59_native_checked"],
               "explicit_stateful_forward_verified": True, "static_abmx_formal_search_disabled": True,
               "new_candidate_runtime_scope": "Four specific full59/ChinTip candidates at actual early/late cursors; no fitted counts, actual after not used as candidate input",
               "remaining": ["Other ABMX bones and explicit external-writer lifecycle require separate contracts.",
                             "Direct legacy/ingame/ML pipelines are not migrated.",
                             "Calibrated anatomical points, ocular surfaces and original-shader visibility remain incomplete.",
                             "Known synthetic target does not certify real-person base expressivity or likeness."],
               "character_ready": False, "full_infrastructure_goal_complete": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "benchmark_cases": len(rows), "runtime_cases": len(runtime),
                      "quality_valid_runtime_cases": sum(case["candidate_quality_valid"] for case in runtime),
                      "max_runtime_normalized": max(max(case["early_max_normalized"], case["late_max_normalized"]) for case in runtime),
                      "character_ready": False}), flush=True)


if __name__ == "__main__":
    main()
