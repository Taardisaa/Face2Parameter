"""Independently replay and summarize restored capture manifests, hash bound."""
import argparse
from collections import Counter
import json
from pathlib import Path

from model import require
from validate_trace import read, sha, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "New summary path required")
    contract = read(args.contract)
    require(sha(contract["assembly_path"]) == contract["assembly_sha256"] and sha(contract["unity_core_path"]) == contract["unity_core_sha256"], "Installed assemblies changed")
    require(all(sha(source["path"]) == source["sha256"] for source in contract["sources"]), "Installed source files changed")
    summary = {"schema_version": 1, "contract_path": str(args.contract.resolve()), "contract_sha256": sha(args.contract),
               "manifests": [], "cases": [], "static_whole_head_model_fixed": False, "application_count_fitted": False}
    for path in args.manifest:
        manifest = read(path)
        flags = ("state_restored", "expression_restored", "bone_restored") if "state_restored" in manifest else ("snapshot_restored", "bones_restored")
        require(all(manifest.get(flag) is True for flag in flags), "Manifest is not completed/restored")
        summary["manifests"].append({"path": str(path.resolve()), "sha256": sha(path), "restored_flags": {flag: manifest[flag] for flag in flags}})
        for case in manifest["cases"]:
            if "trace_sha256" in case:
                require(sha(case["trace"]) == case["trace_sha256"], "Manifest trace hash mismatch")
            trace = read(case["trace"])
            report = validate(trace, contract)
            branches = Counter(branch for row in report["rows"] for branch in row.get("prediction", {}).get("branches", []))
            rows = report["rows"]
            excluded_active_rotations = sum(event["no_rotation_excluded"] and row.get("prediction", {}).get("effective_modifier") is not None
                                           and any(value != 0 for value in row["prediction"]["effective_modifier"]["rotation"])
                                           for event, row in zip(trace["events"], rows))
            summary["cases"].append({"name": case["name"], "trace_path": case["trace"], "trace_sha256": sha(case["trace"]),
                         "passed": report["passed"], "observed_calls": report["observed_call_count"], "passed_calls": report["passed_call_count"],
                         "producer_manifest_trace_sha_present": "trace_sha256" in case,
                         "producer_hash_binding_note": "Producer SHA verified" if "trace_sha256" in case else "Producer manifest records path only; independently hashed exact file as read, no prior producer SHA assertion",
                         "branch_counts": dict(branches), "external_input_boundaries": report["inter_call_boundaries"],
                         "observed_excluded_bone_with_nonzero_effective_rotation_calls": excluded_active_rotations,
                         "max_position_component_error": max((row.get("local_error", {}).get("position_max_abs", 0) for row in rows), default=None),
                         "max_scale_component_error": max((row.get("local_error", {}).get("scale_max_abs", 0) for row in rows), default=None),
                         "max_quaternion_sign_equivalent_component_error": max((row.get("local_error", {}).get("quaternion_sign_equivalent_max_abs", 0) for row in rows), default=None),
                         "max_rotation_deg_error": max((row.get("local_error", {}).get("rotation_angle_deg", 0) for row in rows), default=None),
                         "max_cache_numeric_error": max((row.get("cache_error", {}).get("float_max_abs", 0) for row in rows), default=None),
                         "cache_flag_mismatches": sum(len(row.get("cache_error", {}).get("flag_mismatches", {})) for row in rows)})
    summary["observed_call_count"] = sum(case["observed_calls"] for case in summary["cases"])
    summary["passed_call_count"] = sum(case["passed_calls"] for case in summary["cases"])
    summary["passed"] = all(case["passed"] for case in summary["cases"])
    summary["scope"] = "Actual recorded Apply transitions only; branch coverage explicit, no whole-head/stateless model or likeness certification"
    summary["implementation_hashes"] = [{"path": str(path.resolve()), "sha256": sha(path)} for path in (Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("validate_trace.py"))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"calls={summary['observed_call_count']},passed={summary['passed_call_count']},report={args.out.resolve()}")
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
