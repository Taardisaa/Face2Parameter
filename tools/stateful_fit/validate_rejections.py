"""Actual-evidence mutations must fail candidate protocol validation."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.stateful_fit.adapter import NativeBaselineProtocol, read, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve prior evidence")
    case = read(args.manifest)["cases"][0]
    trace, contract, snapshot = read(case["trace"]), read(args.contract), read(case["geometry"]["path"])
    mutations = [
        ("dropped_call", lambda t, s: t.update(dropped_events=1)),
        ("external_apply_patch", lambda t, s: t["metadata"].update(pre_existing_patch_owners=["unknown.plugin"])),
        ("missing_length_history", lambda t, s: t["events"][0]["before"]["cache"]["fields"].pop("_lenBaseline")),
        ("foreign_cursor_session", lambda t, s: s["abmx_trace_cursor"].update(session_id="another-session")),
        ("later_call_cursor", lambda t, s: s["abmx_trace_cursor"].update(last_completed_sequence=s["abmx_trace_cursor"]["last_completed_sequence"] + 1)),
        ("snapshot_actual_local_mismatch", lambda t, s: next(row for row in s["transforms"] if row["id"] == t["events"][0]["before"]["bone_transform_id"])["local_position"].__setitem__(0, 1.)),
        ("invented_apply_count", lambda t, s: None),
    ]
    rows = []
    for name, mutation in mutations:
        changed_trace, changed_snapshot = copy.deepcopy(trace), copy.deepcopy(snapshot)
        mutation(changed_trace, changed_snapshot)
        try:
            NativeBaselineProtocol(changed_trace, contract, changed_snapshot,
                                   trace_path=case["trace"], contract_path=args.contract,
                                   snapshot_path=case["geometry"]["path"], apply_count=True if name == "invented_apply_count" else None)
            rows.append({"case": name, "rejected": False})
        except ValueError as exc:
            rows.append({"case": name, "rejected": True, "reason": str(exc)})
    report = {"passed": all(row["rejected"] for row in rows), "cases": rows,
              "actual_trace_sha256": sha(case["trace"]), "actual_geometry_sha256": sha(case["geometry"]["path"]),
              "contract_sha256": sha(args.contract), "scope": "Negative protocol checks using in-memory corruptions of actual exported evidence; source files unmodified"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
