"""Compare saved bridge snapshots to LBS and optionally an offline native rig."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from bone_locals import compare_bone_locals

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from geometry import (  # noqa: E402
    Uncertifiable,
    analyze_snapshot,
    blendshape_delta,
    recorded_uniform_renderer_scale,
    rigid_alignment,
    skin_world,
    vertex_errors,
)

from src.hs2_mesh_deform import HeadRig, build_mesh  # noqa: E402


def compare_offline(
    snapshot,
    *,
    profile,
    unit_scale=1,
    unit_scale_reason=None,
    abmx=None,
    apply_renderer_uniform_scale=False,
):
    """Compare the identical head topology after proper rigid alignment only."""
    if unit_scale != 1 and not unit_scale_reason:
        raise ValueError("A non-unit scale needs an explicit unit_scale_reason")
    transforms = {item["id"]: item for item in snapshot["transforms"]}
    character = snapshot["character"]
    try:
        rig = HeadRig(character["head_id"], sampling_profile=profile)
    except (FileNotFoundError, ValueError) as exc:
        return [{"head_id": character["head_id"], "unavailable_reason": str(exc)}]
    rows = []
    for mesh in snapshot["meshes"]:
        if mesh["mesh_name"] != "o_head":
            continue
        row = {
            "mesh_name": mesh["mesh_name"],
            "renderer_path": mesh.get("renderer_path"),
            "sampling_profile": profile,
            "unit_scale": unit_scale,
            "unit_scale_reason": unit_scale_reason,
            "abmx_input": "explicit supplied modifiers; offline ABMX runtime parity still unverified"
            if abmx
            else "none; live actual bone transforms may contain modifiers",
            "scope": "diagnostic residual; no likeness or base expressivity certification",
        }
        rows.append(row)
        try:
            source = np.asarray(mesh["source"]["vertices"], dtype=float)
            faces = np.asarray(mesh["source"]["triangles"], dtype=int).reshape(-1, 3)
            if rig.verts.shape != source.shape or not np.array_equal(rig.faces, faces):
                raise Uncertifiable(
                    "Cached head topology/vertex count differs; cannot assert correspondence"
                )
            row["source_cache_errors"] = vertex_errors(source, rig.verts)
            row["bone_local_comparison"] = compare_bone_locals(snapshot, rig, abmx)
            if row["source_cache_errors"]["max_abs_component"] > 1e-6:
                raise Uncertifiable(
                    "Cached source vertices differ; extract the exact live asset before fitting"
                )
            if (
                rig.skin_bone_names != mesh["bone_names"]
                or not np.array_equal(rig.bone_idx, mesh["source"]["bone_indices"])
                or not np.allclose(
                    rig.bone_w, mesh["source"]["bone_weights"], atol=1e-7, rtol=0
                )
            ):
                raise Uncertifiable(
                    "Cached source bone palette/weights differ; correspondence cannot be certified"
                )
            live_bindpose = np.asarray(mesh["source"]["bindposes"]).reshape(-1, 4, 4)
            row["bindpose_cache_max_abs"] = float(
                np.max(np.abs(rig.bindpose - live_bindpose))
            )
            if row["bindpose_cache_max_abs"] > 1e-6:
                raise Uncertifiable("Cached bindposes differ from live source geometry")
            # Actual bone-world LBS is independent of choosing either BakeMesh convention.
            world, active = skin_world(mesh, transforms, influences=4)
            delta, _ = blendshape_delta(mesh)
            posed_rig = copy.copy(rig)
            posed_rig.verts = rig.verts + delta
            offline, _ = build_mesh(posed_rig, character["shape_value_face"], abmx)
            # Always retain the uncompensated diagnostic, so parent scaling cannot disappear.
            _, uncompensated = rigid_alignment(offline, world, unit_scale=unit_scale)
            row["uncompensated_comparison"] = uncompensated
            ancestor_scale = 1
            if apply_renderer_uniform_scale:
                ancestor_scale, evidence = recorded_uniform_renderer_scale(
                    mesh, transforms
                )
                row["recorded_ancestor_scale"] = evidence
            _, aligned = rigid_alignment(
                offline, world, unit_scale=unit_scale * ancestor_scale
            )
            aligned["total_input_scale_applied"] = unit_scale * ancestor_scale
            aligned["unit_scale_applied"] = unit_scale
            aligned["recorded_ancestor_scale_factor_applied"] = ancestor_scale
            row.update(
                {
                    "active_blendshapes_applied": active,
                    "comparison": aligned,
                    "native_input": character["shape_value_face"],
                    "note": "Rigid alignment removes pose only; no affine/deformation fitting or fitted scale applied",
                }
            )
        except (Uncertifiable, ValueError, KeyError, IndexError) as exc:
            row["unavailable_reason"] = str(exc)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normalized-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--influences",
        type=int,
        choices=[1, 2, 4],
        help="explicit BakeMesh quality hypothesis",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--profile", choices=["vanilla", "slider_unlocker_18_2"])
    parser.add_argument(
        "--abmx-json",
        type=Path,
        help="bone-name to scale/length/position/rotation modifiers",
    )
    parser.add_argument("--unit-scale", type=float, default=1)
    parser.add_argument(
        "--unit-scale-reason", help="required for a non-unit explicit scale"
    )
    parser.add_argument(
        "--renderer-uniform-scale",
        action="store_true",
        help="apply independently recorded uniform ancestor scale; reject shear/reflection",
    )
    args = parser.parse_args()
    if args.offline and not args.profile:
        parser.error("--offline requires an explicit --profile")
    if not np.isfinite(args.normalized_tolerance) or args.normalized_tolerance <= 0:
        parser.error("--normalized-tolerance must be finite and positive")
    if args.unit_scale != 1 and not args.unit_scale_reason:
        parser.error("A non-unit --unit-scale requires --unit-scale-reason")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if args.out.resolve() == args.snapshot.resolve() or (
        args.abmx_json and args.out.resolve() == args.abmx_json.resolve()
    ):
        parser.error("Output report must not overwrite an input snapshot/modifier file")
    report = analyze_snapshot(
        snapshot,
        normalized_tolerance=args.normalized_tolerance,
        influences=args.influences,
    )
    report["snapshot_path"] = str(args.snapshot.resolve())
    report["snapshot_sha256"] = hashlib.sha256(args.snapshot.read_bytes()).hexdigest()
    report["game"] = snapshot.get("game")
    report["character_head_id"] = snapshot.get("character", {}).get("head_id")
    if args.offline:
        abmx = (
            json.loads(args.abmx_json.read_text(encoding="utf-8"))
            if args.abmx_json
            else None
        )
        report["offline_head_comparison"] = compare_offline(
            snapshot,
            profile=args.profile,
            unit_scale=args.unit_scale,
            unit_scale_reason=args.unit_scale_reason,
            abmx=abmx,
            apply_renderer_uniform_scale=args.renderer_uniform_scale,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for mesh in report["meshes"]:
        print(
            f"{mesh['mesh_name']}: certified={mesh['certified']}; selected={mesh.get('selected_candidate')}; "
            f"reason={mesh.get('uncertifiable_reason', mesh.get('note', 'numerical agreement'))}"
        )
    print(f"report={args.out.resolve()}")
    return (
        0
        if report["meshes"] and all(mesh["certified"] for mesh in report["meshes"])
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
