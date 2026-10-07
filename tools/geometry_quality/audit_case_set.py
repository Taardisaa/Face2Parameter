"""Audit every Unity geometry snapshot in a case directory against one baseline.

Does not access the game. Reuses an immutable baseline's triangle diagnostics,
preserves renderer identity, and writes individual reports plus a compact index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mesh_quality import analyze_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--baseline", type=Path, help="Defaults to case_dir/baseline.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--meshes", nargs="+", default=["o_head"])
    parser.add_argument("--case-prefix", default="", help="Only inspect matching filename prefixes; useful for separate head baselines")
    args = parser.parse_args()
    baseline_path = args.baseline or args.case_dir / "baseline.json"
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)
    baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()
    cache = {}
    summaries = []
    skipped = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for case_path in sorted(args.case_dir.glob(args.case_prefix + "*.json")):
        case_bytes = case_path.read_bytes()
        case = json.loads(case_bytes)
        if not isinstance(case, dict) or case.get("snapshot_kind") != "maker_live_skinned_geometry":
            skipped.append(case_path.name)
            continue
        report = analyze_snapshot(case, baseline, mesh_names=set(args.meshes), baseline_cache=cache)
        report["input_path"] = str(case_path.resolve())
        report["input_sha256"] = hashlib.sha256(case_bytes).hexdigest()
        report["baseline_path"] = str(baseline_path.resolve())
        report["baseline_sha256"] = baseline_sha
        report["expression_config_identical"] = case.get("character", {}).get("expression") == baseline.get("character", {}).get("expression")
        report_path = args.out_dir / (case_path.stem + "_quality.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for mesh in report["meshes"]:
            comparison = mesh.get("baseline", {})
            summary = {
                "case": case_path.stem, "mesh_name": mesh["mesh_name"], "renderer_path": mesh["renderer_path"],
                "report_path": str(report_path.resolve()), "baseline_status": comparison.get("status"),
                "source_hash_identical": comparison.get("source_hash_identical"),
                "expression_config_identical": report["expression_config_identical"],
                "degenerate_triangles": len(mesh["degenerate_triangle_ids"]),
                "total_crossings": mesh["self_intersections"]["true_crossing_or_area_overlap_count"],
                "contacts": mesh["self_intersections"]["contact_count"],
                "new_crossing_pairs": len(comparison.get("new_crossing_pairs", [])),
                "resolved_crossing_pairs": len(comparison.get("resolved_crossing_pairs", [])),
                "normal_reversals": len(comparison.get("normal_reversal_triangle_ids", [])),
                "unusual_edges": len(comparison.get("unusual_edges", [])),
                "unusual_area_triangles": len(comparison.get("unusual_area_triangles", [])),
                "edge_ratio_distribution": comparison.get("edge_ratio_distribution"),
                "area_ratio_distribution": comparison.get("area_ratio_distribution"),
                "bone_negative_determinants": mesh["bone_world_determinants"]["negative_count"],
                "nonmanifold_edges": mesh["topology"]["nonmanifold_edge_count"],
                "nonmanifold_vertices": mesh["topology"]["nonmanifold_vertex_count"],
            }
            summaries.append(summary)
            print(json.dumps({"case": summary["case"], "new_crossings": summary["new_crossing_pairs"],
                              "normal_reversals": summary["normal_reversals"], "unusual_edges": summary["unusual_edges"],
                              "baseline_status": summary["baseline_status"]}, ensure_ascii=True), flush=True)
    if not summaries:
        raise ValueError("No Unity geometry snapshots found in case directory")
    output = {"baseline_path": str(baseline_path.resolve()), "baseline_sha256": baseline_sha,
              "case_count": len({summary["case"] for summary in summaries}), "mesh_report_count": len(summaries),
              "skipped_non_geometry_files": skipped, "summaries": summaries,
              "scope": "Offline per-mesh quality vs baseline; no identity/likeness or automatic accept/reject decision"}
    index = args.out_dir / "case_index.json"
    index.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": output["case_count"], "index": str(index.resolve())}), flush=True)


if __name__ == "__main__":
    main()
