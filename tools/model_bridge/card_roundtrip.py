"""Save an imported source head in a new card and reload its embedded geometry.

Uses the native card writer/loader. Replaces the current Maker character with the
card just saved. No screenshots, parameter sweeps, fitting or external model IO
are performed during reload.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifact import sha
from .game_import import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:43127")
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.card.exists() or args.receipt.exists():
        raise FileExistsError("Use new card and receipt paths; existing files are retained")
    if not args.thumbnail.is_file():
        raise FileNotFoundError(args.thumbnail)
    before = request(args.base, "GET")
    if not before.get("active") or not before.get("card_persistence"):
        raise RuntimeError("An imported source head with card support is required")
    saved = request(args.base, "POST", {"path": str(args.card.resolve()),
                                        "thumbnail": str(args.thumbnail.resolve())},
                    route="/maker/face/model/save")
    cleared = request(args.base, "DELETE")
    loaded = request(args.base, "POST", {"path": str(args.card.resolve())}, route="/maker/card/load")
    after = request(args.base, "GET")
    equal_vertices = np.array_equal(np.asarray(before.pop("render_vertices"), dtype=np.float32),
                                   np.asarray(after.pop("render_vertices", []), dtype=np.float32))
    equal_triangles = np.array_equal(np.asarray(before.pop("render_triangles"), dtype=np.int32),
                                    np.asarray(after.pop("render_triangles", []), dtype=np.int32))
    same_source = before["artifact_sha256"] == after.get("artifact_sha256")
    same_placement = (before["local_scale"] == after.get("local_scale") and
                      before["local_position"] == after.get("local_position"))
    report = {"before": before, "saved": saved, "cleared": cleared, "loaded": loaded,
              "after": after, "vertex_arrays_equal": bool(equal_vertices),
              "triangle_arrays_equal": bool(equal_triangles), "source_digest_equal": same_source,
              "placement_equal": same_placement, "card_sha256": sha(args.card)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    passed = after.get("active") and equal_vertices and equal_triangles and same_source and same_placement
    if not passed:
        raise RuntimeError("Embedded source head roundtrip failed; card and receipt retained")
    print(json.dumps({"receipt": str(args.receipt.resolve()), "card": str(args.card.resolve()),
                      "source_geometry_and_placement_preserved": True}))


if __name__ == "__main__":
    main()
