"""Analyze saved live_cases.json exports without contacting the game."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from compare_export import compare_offline
from geometry import analyze_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["vanilla", "slider_unlocker_18_2"], required=True
    )
    parser.add_argument("--renderer-uniform-scale", action="store_true")
    parser.add_argument(
        "--abmx-cases",
        type=Path,
        help="explicit JSON mapping from case name to bone modifiers",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    modifiers = (
        json.loads(args.abmx_cases.read_text(encoding="utf-8"))
        if args.abmx_cases
        else {}
    )
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "profile": args.profile,
        "cases": [],
    }
    for case in manifest["cases"]:
        path = Path(case["geometry"]["path"])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if case["geometry"].get("sha256") and digest != case["geometry"]["sha256"]:
            raise ValueError(f"Export hash mismatch for {case['name']}")
        snapshot = json.loads(raw)
        report = analyze_snapshot(snapshot)
        report.update(
            {
                "snapshot_path": str(path.resolve()),
                "snapshot_sha256": digest,
                "case": case["name"],
                "game": snapshot.get("game"),
            }
        )
        report["offline_head_comparison"] = compare_offline(
            snapshot,
            profile=args.profile,
            apply_renderer_uniform_scale=args.renderer_uniform_scale,
            abmx=modifiers.get(case["name"]),
        )
        destination = args.out / (case["name"] + ".json")
        if destination.resolve().parent != args.out.resolve():
            raise ValueError("Case names must not contain path components")
        if destination.resolve() == path.resolve():
            raise ValueError("Report cannot overwrite the live geometry input")
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        head = (
            report["offline_head_comparison"][0]
            if report["offline_head_comparison"]
            else {}
        )
        bones = head.get("bone_local_comparison", {})
        top = sorted(
            bones.get("bones", []),
            key=lambda bone: bone["matrix_max_abs"],
            reverse=True,
        )[:5]
        summary["cases"].append(
            {
                "name": case["name"],
                "report": str(destination.resolve()),
                "engine_lbs_certified": all(
                    mesh["certified"] for mesh in report["meshes"]
                ),
                "offline_errors": head.get("comparison", {}).get("rigid_errors"),
                "ancestor_scale": head.get("recorded_ancestor_scale", {}).get("factor"),
                "head_skin_bone_local_max_position": bones.get(
                    "max_head_skin_position_error"
                ),
                "head_skin_bone_local_max_rotation_deg": bones.get(
                    "max_head_skin_rotation_angle_deg"
                ),
                "head_skin_bone_local_max_scale": bones.get(
                    "max_head_skin_scale_error"
                ),
                "largest_bone_differences": top,
                "offline_unavailable": head.get("unavailable_reason"),
                "explicit_abmx_supplied": case["name"] in modifiers,
            }
        )
        errors = head.get("comparison", {}).get("rigid_errors", {})
        print(
            f"{case['name']}: maxNormalized={errors.get('max_normalized')}; "
            f"skinBonePosition={bones.get('max_head_skin_position_error')}"
        )
    destination = args.out / "series.json"
    if destination.resolve() == args.manifest.resolve():
        raise ValueError("Summary cannot overwrite the input manifest")
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary={destination.resolve()}")


if __name__ == "__main__":
    main()
