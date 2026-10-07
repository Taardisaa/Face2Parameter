"""Evaluate two declared full mesh/rig inputs with an explicit same-head baseline."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "geometry_quality"))

import numpy as np
from mesh_quality import analyze_snapshot
from contracts import load_target
from search import DEFAULT_ACCEPTANCE, accepted, quality_gate
from surface import evaluate_surface


def snapshot(mesh):
    """Surface-only adapter; no invented bone/world-matrix evidence."""
    return {"schema_version": 1, "snapshot_kind": "maker_live_skinned_geometry", "transforms": [],
            "origin": "offline declared surface only; no bone matrices supplied",
            "meshes": [{"mesh_name": "o_head", "renderer_path": f"/declared_head_{mesh.metadata['head_id']}/o_head",
                        "source_geometry_sha256": mesh.metadata["asset"]["sha256"],
                        "baked": {"vertices": mesh.vertices.tolist(), "triangles": mesh.faces.reshape(-1).tolist()}}]}


def evaluate_pair(source_config, target_config, baseline_config, *, count=4096, seed=7381, regions=None, acceptance=None):
    source, target, baseline = (load_target(config) for config in (source_config, target_config, baseline_config))
    if source.metadata.get("head_id") not in (0, 1, 2, 3) or source.metadata.get("head_id") != baseline.metadata.get("head_id"):
        raise ValueError("Quality baseline requires the same declared cached head ID")
    if source.metadata["asset"]["sha256"] != baseline.metadata["asset"]["sha256"] or source.vertices.shape != baseline.vertices.shape \
            or not np.array_equal(source.faces, baseline.faces):
        raise ValueError("Quality baseline source asset/topology must match candidate; target topology may differ")
    metrics = evaluate_surface(source.vertices, source.faces, target.vertices, target.faces, count=count, seed=seed, regions=regions)
    base_snapshot, current_snapshot = snapshot(baseline), snapshot(source)
    quality_report = analyze_snapshot(current_snapshot, base_snapshot)
    quality = quality_gate(quality_report, analyze_snapshot(base_snapshot))
    thresholds = {**DEFAULT_ACCEPTANCE, **(acceptance or {})}
    if any(not np.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError("Invalid acceptance thresholds")
    return {"schema_version": 1, "report_kind": "hs2_declared_surface_pair", "source": source.metadata,
            "target": target.metadata, "baseline": baseline.metadata, "surface": metrics, "quality": quality,
            "quality_report": quality_report, "quality_bone_scope": "not evaluated by this surface-only adapter; search rig adapter records bone matrices",
            "acceptance": thresholds, "status": "found_quality_valid_approximation" if accepted(metrics, quality, thresholds)
            else "not_found_within_this_search", "search_scope": "one supplied candidate only; no optimization or expressivity conclusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7381)
    parser.add_argument("--regions", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    paths = {"source": args.source, "target": args.target, "baseline": args.baseline}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    report = evaluate_pair(*(json.loads(raw[name]) for name in ("source", "target", "baseline")), count=args.samples,
                           seed=args.seed, regions=json.loads(args.regions.read_bytes()) if args.regions else None,
                           acceptance=json.loads(args.acceptance.read_bytes()) if args.acceptance else None)
    report["input_file_sha256"] = {name: hashlib.sha256(contents).hexdigest() for name, contents in raw.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": report["status"], "rms": report["surface"]["symmetric"]["rms"],
                      "quality_valid": report["quality"]["quality_valid"]}, indent=2))


if __name__ == "__main__":
    main()
