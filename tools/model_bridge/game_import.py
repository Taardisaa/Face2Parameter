"""Import exact source vertices through the native HS2 bridge preview route.

Placement uses one uniform scale and translation against a current head reference.
This positions the source mesh; it never fits source shape to the native head.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

from .artifact import sha


def request(base, method, body=None, route="/maker/face/model"):
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(base.rstrip("/") + route, data=encoded, method=method,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:43127")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    if args.clear:
        print(json.dumps(request(args.base, "DELETE")))
        return
    if args.mesh is None or args.receipt is None:
        parser.error("--mesh and --receipt are required for import")
    if args.receipt.exists():
        raise FileExistsError("Use a new receipt path")
    path = args.mesh.resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("format") not in {"hs2_source_head_mesh_v1", "hs2_source_head_mesh_v2"} or source.get("geometry_mode") != "head_local":
        raise ValueError("Explicit head-local source-model interchange required")
    vertices = np.asarray(source["vertices"], dtype=np.float32)
    triangles = np.asarray(source["triangles"], dtype=np.int32)
    current = request(args.base, "GET")
    center = np.asarray(current["reference_local_center"], dtype=np.float64)
    size = np.asarray(current["reference_local_size"], dtype=np.float64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    if high[1] <= low[1] or size[1] <= 0:
        raise ValueError("Nonzero source/reference head height required for placement")
    scale = float(size[1] / (high[1] - low[1]))
    translation = center - scale * ((low.astype(np.float64) + high) / 2)
    result = request(args.base, "POST", {"path": str(path), "sha256": sha(path),
                                         "scale": scale, "translation": translation.tolist()})
    actual_vertices = np.asarray(result.pop("render_vertices"), dtype=np.float32)
    actual_triangles = np.asarray(result.pop("render_triangles"), dtype=np.int32)
    passed = np.array_equal(vertices, actual_vertices) and np.array_equal(triangles, actual_triangles)
    report = {
        "source": {"path": str(path), "sha256": sha(path)},
        "game": result,
        "vertex_arrays_equal": bool(np.array_equal(vertices, actual_vertices)),
        "triangle_arrays_equal": bool(np.array_equal(triangles, actual_triangles)),
        "placement_policy": "Single reference head-height uniform scale, center translation; no per-axis scaling, remeshing, vertex fitting or parameter conversion",
        "native_slider_mapping": False, "card_persistence": result.get("card_persistence", False),
        "native_expression_retargeting": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        request(args.base, "DELETE")
        raise RuntimeError("Unity source arrays changed; preview released")
    print(json.dumps({"receipt": str(args.receipt.resolve()), "active": result["active"],
                      "source_geometry_unchanged": passed}))


if __name__ == "__main__":
    main()
