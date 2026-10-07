"""One bounded all59 CUDA job on a separately quality-checked cached head3 target."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "geometry_quality"))

import numpy as np
import torch
from contracts import AB_IDENTITY, full_mesh_config, rig_mesh
from search import compare, quality_gate, synthetic_snapshot
from mesh_quality import analyze_snapshot

TARGET_TRIALS = [
    {"name": "mixed_0_24_4", "changes": {0: .65, 24: .60, 4: .55}},
    {"name": "mixed_0_24_smaller", "changes": {0: .60, 24: .55}},
    {"name": "mixed_0_24_minimal", "changes": {0: .55, 24: .53}},
    {"name": "one_control_fallback", "changes": {0: .80}},
]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/base_comparison_full59_20261005")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if args.iterations < 1 or args.iterations > 40:
        parser.error("This bounded bench admits 1..40 iterations")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Requested CUDA is unavailable; no silent resource/profile change")
    if args.out_dir.exists():
        parser.error("Output directory already exists; preserve earlier benchmarks and checkpoints")
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    resource = {"device": args.device, "torch": torch.__version__, "numpy": np.__version__, "cpu_threads": 2,
                "cuda_device": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                "cuda_free_total_before": torch.cuda.mem_get_info() if args.device == "cuda" else None}
    save(args.out_dir / "resources.json", resource)
    source_files = [Path(__file__), Path(__file__).with_name("search.py"), Path(__file__).with_name("surface.py"), Path(__file__).with_name("contracts.py")]
    save(args.out_dir / "implementation_hashes.json", [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_files])
    baseline, rig, trig = rig_mesh(3, [.5] * 59, device=args.device)
    ab = np.tile(AB_IDENTITY, (len(trig.ab_names), 1))
    baseline_snapshot = synthetic_snapshot(rig, trig, np.full(59, .5), ab, baseline.vertices)
    baseline_report = analyze_snapshot(baseline_snapshot)
    selected = None
    trials = []
    baseline_cache = {}
    for trial in TARGET_TRIALS:
        native = np.full(59, .5)
        for index, value in trial["changes"].items():
            native[index] = value
        target, target_rig, target_trig = rig_mesh(3, native.tolist(), device=args.device)
        target_snapshot = synthetic_snapshot(target_rig, target_trig, native, ab, target.vertices)
        quality_report = analyze_snapshot(target_snapshot, baseline_snapshot, baseline_cache=baseline_cache)
        gate = quality_gate(quality_report, baseline_report)
        destination = args.out_dir / ("target_" + trial["name"] + ".json")
        save(destination, {"trial": trial, "target": full_mesh_config(target), "quality": gate, "quality_report": quality_report})
        trials.append({"trial": trial, "artifact": str(destination), "native59": native.tolist(), "quality": gate})
        save(args.out_dir / "target_trials.json", trials)
        print(json.dumps({"phase": "target_quality", "trial": trial["name"], "quality_valid": gate["quality_valid"], "reasons": gate["reasons"]}), flush=True)
        if gate["quality_valid"]:
            selected = trial, native, target.metadata
            break
    if selected is None:
        save(args.out_dir / "target_selection_failure.json", {"status": "not_found_within_this_search", "scope": "No listed target trial passed the explicit same-head quality gate; no fitting job started", "trials": trials})
        print("No quality-valid target trial; preserved all failures and did not advertise a safe target", flush=True)
        return 2
    trial, native, metadata = selected
    config = {"target": {"kind": "rig_target", "head_id": 3, "native59": native.tolist(), "sampling_profile": "vanilla"},
              "heads": [3, 0, 1, 2], "modes": ["native", "installed18.2", "native+ABMX"],
              "seed": 10052026, "evaluation_seed": 7381,
              "search": {"iterations": args.iterations, "restarts": 1, "optimization_samples": 128,
                         "evaluation_samples": 4096, "learning_rate": .015, "normal_loss_weight": 0.0},
              "benchmark_scope": "Every native index 0..58 is optimized; complete o_head final quadrature + strict same-head quality gate",
              "target_quality_artifact": str(args.out_dir / ("target_" + trial["name"] + ".json")),
              "abmx_scope": "ChinTip probe envelope only; all optimized combinations and head0/1/3 behavior unverified in runtime"}
    assert "active_native" not in config["search"]
    save(args.out_dir / "comparison_config.json", config)
    save(args.out_dir / "selected_target_provenance.json", {"trial": trial, "metadata": metadata, "quality_valid": True,
                                                           "fallback_used": trial["name"] == "one_control_fallback"})
    completed = []
    started = time.perf_counter()

    def progress(row, count):
        winners = [candidate for candidate in row["candidates"] if candidate["search"] is not None]
        assert winners and all(candidate["search"]["active_native"] == list(range(59)) for candidate in winners)
        name = f"head_{row['head_id']}_{row['mode'].replace('+', '_').replace('.', '_')}"
        checkpoint = args.out_dir / (name + ".json")
        save(checkpoint, row)
        candidate = row["candidates"][row["selected_candidate_index"]]
        winner = winners[0]
        summary = {"head_id": row["head_id"], "mode": row["mode"], "status": row["status"], "checkpoint": str(checkpoint),
                   "selected_kind": candidate["kind"], "selected_surface": candidate["surface"], "selected_quality": candidate["quality"],
                   "winner_surface": winner["surface"], "winner_quality": winner["quality"],
                   "native_dimension": len(winner["search"]["active_native"]),
                   "bounded_parameter_dimension": len(winner["search"]["lower_bounds"]),
                   "nonconstant_bound_dimension": sum(a < b for a, b in zip(winner["search"]["lower_bounds"], winner["search"]["upper_bounds"])),
                   "iterations": winner["search"]["completed_iterations"], "seed": winner["search"]["seed"]}
        completed.append(summary)
        save(args.out_dir / "progress.json", {"completed_cases": count, "planned_cases": 12, "elapsed_seconds": time.perf_counter() - started, "cases": completed})
        print(json.dumps({"phase": "comparison", "completed": count, "head_id": row["head_id"], "mode": row["mode"], "status": row["status"],
                          "rms": candidate["surface"]["symmetric"]["rms"], "winner_quality_valid": winner["quality"]["quality_valid"]}), flush=True)

    report = compare(config, device=args.device, progress=progress)
    report["target_selection_trials"] = trials
    report["resource_before"] = resource
    report["configuration_file_sha256"] = hashlib.sha256((args.out_dir / "comparison_config.json").read_bytes()).hexdigest()
    report["implementation_hashes"] = json.loads((args.out_dir / "implementation_hashes.json").read_bytes())
    save(args.out_dir / "comparison_report.json", report)
    save(args.out_dir / "benchmark_summary.json", {"target": report["target"], "target_trial": trial, "target_quality_valid": True,
                                                  "elapsed_seconds": report["elapsed_seconds"], "cases": completed,
                                                  "limitations": report["limitations"], "acceptance": report["acceptance"]})
    print(json.dumps({"complete": True, "cases": len(completed), "out_dir": str(args.out_dir), "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
