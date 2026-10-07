"""Real cached-head bounded-fit benchmark and a deliberate unsafe-mesh case."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from search import compare, quality_gate, synthetic_snapshot
from contracts import AB_IDENTITY, rig_mesh
from surface import evaluate_surface
from mesh_quality import analyze_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "base_comparison_20261004")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    native = [.5] * 59
    native[0] = .8
    config = {"target": {"kind": "rig_target", "head_id": 2, "native59": native, "sampling_profile": "vanilla"},
              "heads": [0, 1, 2], "modes": ["native", "installed18.2", "native+ABMX"], "seed": 8245,
              "evaluation_seed": 7381,
              "search": {"active_native": [0], "iterations": args.iterations, "optimization_samples": 64,
                         "evaluation_samples": 4096, "restarts": 1, "learning_rate": .025},
              "benchmark_scope": "One known native control changed; a harness check, not exhaustive 59-control expressivity search"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "known_reachable_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    def progress(row, completed):
        checkpoint = args.out_dir / f"head_{row['head_id']}_{row['mode'].replace('+', '_').replace('.', '_')}.json"
        checkpoint.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"completed": completed, "head_id": row["head_id"], "mode": row["mode"], "status": row["status"]}), flush=True)

    report = compare(config, device=args.device, progress=progress)
    (args.out_dir / "known_reachable_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    mesh, rig, trig = rig_mesh(2, [.5] * 59, device=args.device)
    ab = np.tile(AB_IDENTITY, (len(trig.ab_names), 1))
    base = synthetic_snapshot(rig, trig, np.full(59, .5), ab, mesh.vertices)
    damaged = mesh.vertices.copy()
    face = rig.faces[0]
    damaged[face[1]] = damaged[face[0]]
    damage_snapshot = synthetic_snapshot(rig, trig, np.full(59, .5), ab, damaged)
    damage_quality = analyze_snapshot(damage_snapshot, base)
    gate = quality_gate(damage_quality, analyze_snapshot(base))
    surface = evaluate_surface(damaged, rig.faces, damaged, rig.faces, count=4096, seed=7381)
    unsafe = {"case": "deliberately degenerate cached head candidate equals its target",
              "scope": "same cached HeadRig asset, manually collapsed edge; not a native-rig reachable target or runtime capture",
              "collapsed_vertex_pair": [int(face[0]), int(face[1])], "surface": surface,
              "quality": gate, "quality_report": damage_quality,
              "status": "not_found_within_this_search",
              "interpretation": "Zero distance alone cannot pass quality; no claim that every safe approximation to this target is impossible"}
    (args.out_dir / "unsafe_exact_match_report.json").write_text(json.dumps(unsafe, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = {"real_cached_target": report["target"], "elapsed_seconds": report["elapsed_seconds"],
               "results": [{"head_id": row["head_id"], "mode": row["mode"], "status": row["status"],
                            "selected": {key: row["candidates"][row["selected_candidate_index"]][key]
                                         for key in ("native59", "abmx", "surface", "quality")}}
                           for row in report["results"]],
               "unsafe_exact_match": {"rms": surface["symmetric"]["rms"], "quality_valid": gate["quality_valid"], "reasons": gate["reasons"]}}
    (args.out_dir / "benchmark_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "elapsed_seconds": report["elapsed_seconds"],
                      "results": [{"head_id": row["head_id"], "mode": row["mode"], "status": row["status"]} for row in report["results"]],
                      "unsafe_quality_valid": gate["quality_valid"]}, indent=2))


if __name__ == "__main__":
    main()
