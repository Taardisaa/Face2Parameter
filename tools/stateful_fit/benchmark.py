"""Repeat the original full59 target/search with an explicit common Apply count.

No target fallback, no old evidence overwrite, no candidate runtime claim.
Native and expanded modes use the same surface/quality contract as stateful ABMX.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/base_comparison"))
import torch
from search import compare
from tools.stateful_fit.adapter import read, sha


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--apply-count", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("New output directory required")
    config = copy.deepcopy(read(args.original_config))
    if config["heads"] != [3, 0, 1, 2] or config["modes"] != ["native", "installed18.2", "native+ABMX"]:
        raise ValueError("Original four-head/three-mode full benchmark required")
    if config["search"].get("active_native", list(range(59))) != list(range(59)) or config["search"]["iterations"] != 40:
        raise ValueError("Preserve all59 controls and the original40-step budget")
    preflight = read(config["target_quality_artifact"])
    if preflight["quality"]["quality_valid"] is not True or preflight["trial"]["name"] != "mixed_0_24_4":
        raise ValueError("Original quality-valid mixed target required; no fallback")
    config["abmx_protocol"] = {"manifest": str(args.manifest.resolve()), "contract": str(args.contract.resolve()),
                               "apply_count": args.apply_count}
    config["abmx_scope"] = "Explicit fixed common call count and measured persistent history; new candidates not runtime-certified"
    args.out_dir.mkdir(parents=True)
    save(args.out_dir / "comparison_config.json", config)
    checkpoints = []
    started = time.perf_counter()
    torch.set_num_threads(2)

    def progress(row, count):
        path = args.out_dir / f"head_{row['head_id']}_{row['mode'].replace('+', '_').replace('.', '_')}.json"
        save(path, row)
        winner = next(candidate for candidate in row["candidates"] if candidate["kind"] == "bounded_search_winner")
        selected = row["candidates"][row["selected_candidate_index"]]
        if winner["search"]["active_native"] != list(range(59)):
            raise ValueError("Active dimension changed")
        item = {"head_id": row["head_id"], "mode": row["mode"], "checkpoint": str(path.resolve()), "checkpoint_sha256": sha(path),
                "winner_quality_valid": winner["quality"]["quality_valid"], "winner_rms": winner["surface"]["symmetric"]["rms"],
                "status": row["status"], "selected_kind": selected["kind"], "selected_rms": selected["surface"]["symmetric"]["rms"],
                "iterations": winner["search"]["completed_iterations"], "active_native59": True}
        checkpoints.append(item)
        save(args.out_dir / "progress.json", {"completed_cases": count, "planned_cases": 12, "cases": checkpoints})
        print(json.dumps(item), flush=True)

    report = compare(config, device="cuda", progress=progress)
    save(args.out_dir / "comparison_report.json", report)
    summary = {"completed_cases": len(checkpoints), "cases": checkpoints, "elapsed_seconds": time.perf_counter() - started,
               "original_config": str(args.original_config.resolve()), "original_config_sha256": sha(args.original_config),
               "original_target_quality_sha256": sha(config["target_quality_artifact"]),
               "configuration_sha256": sha(args.out_dir / "comparison_config.json"), "common_apply_count": args.apply_count,
               "source_hashes": [{"path": str(path.resolve()), "sha256": sha(path)} for path in
                                 (Path(__file__), ROOT / "tools/base_comparison/search.py", ROOT / "src/hs2_abmx_torch.py", Path(__file__).with_name("adapter.py"))],
               "all59_each_case": True, "same_original_target": True, "new_candidate_runtime_certified": False,
               "scope": "Complete o_head 4 bases x3profiles, same original mixed target,40steps/full59, same surface+quality gates; stateful clean-boundary/call protocol"}
    save(args.out_dir / "benchmark_summary.json", summary)
    print(json.dumps({"complete": True, "cases": len(checkpoints), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
