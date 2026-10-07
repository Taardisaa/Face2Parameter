"""Inspect saved vertex effects, including whether range extension adds a direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def exterior_effects(atlas):
    """Measure actual outer-probe versus corresponding endpoint, without labels."""
    levels = atlas["levels"]
    results = []
    for control in atlas["controls"]:
        edges = {}
        with np.load(control["vertex_effects_path"]) as data:
            for side, endpoint, probes in [
                ("below_zero", 0, [level for level in levels if level < 0]),
                ("above_one", 1, [level for level in levels if level > 1]),
            ]:
                if endpoint not in levels or not probes:
                    edges[side] = {"available": False}
                    continue
                probe = min(probes) if endpoint == 0 else max(probes)
                pi, ei = levels.index(probe), levels.index(endpoint)
                meshes = {}
                for name in atlas["mesh_layout"]:
                    delta = (
                        data[name + "__displacement"][pi]
                        - data[name + "__displacement"][ei]
                    )
                    distances = np.linalg.norm(delta, axis=1)
                    endpoint_stats = control["samples"][ei]["meshes"][name]
                    scale = endpoint_stats["baseline_bbox_diagonal"]
                    threshold = endpoint_stats["effect_threshold_absolute"]
                    meshes[name] = {
                        "max": float(distances.max()),
                        "rms": float(np.sqrt(np.mean(distances**2))),
                        "max_normalized": float(distances.max() / scale)
                        if scale
                        else None,
                        "extends_at_effect_threshold": bool(
                            (distances > threshold).any()
                        ),
                        "effect_threshold_absolute": threshold,
                    }
                edges[side] = {
                    "available": True,
                    "probe": probe,
                    "endpoint": endpoint,
                    "any_mesh_extends": any(
                        mesh["extends_at_effect_threshold"] for mesh in meshes.values()
                    ),
                    "meshes": meshes,
                }
        results.append({"index": control["index"], "edges": edges})
    return {
        "head_id": atlas["head_id"],
        "profile": atlas["sampling_profile"],
        "scope": "measured additional motion relative to a bounded endpoint; not an anatomical or feasibility classification",
        "controls": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {
        "schema_version": 1,
        "manifest_path": str(args.manifest.resolve()),
        "atlases": [],
    }
    for entry in manifest["atlases"]:
        atlas = json.loads(Path(entry["atlas_path"]).read_text(encoding="utf-8"))
        row = exterior_effects(atlas)
        row["controls_with_no_additional_external_effect"] = [
            control["index"]
            for control in row["controls"]
            if all(
                edge.get("available") and not edge.get("any_mesh_extends")
                for edge in control["edges"].values()
            )
        ]
        report["atlases"].append(row)
        print(
            row["head_id"],
            row["profile"],
            "noAdditionalOuterEffect",
            row["controls_with_no_additional_external_effect"],
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
