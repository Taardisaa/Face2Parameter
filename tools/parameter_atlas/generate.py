"""Generate native-control surface atlases from cached rigs, without game calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core import (  # noqa: E402
    analyze_jacobian,
    finite_difference_jacobian,
    summarize_displacement,
    summarize_verified_regions,
    validate_regions,
)

from src.hs2_deform_torch import TorchHeadRig  # noqa: E402
from src.hs2_mesh_deform import HeadRig, available_heads  # noqa: E402


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_geometry(trig, values, subs, *, batch_size):
    """No raster rendering: evaluate common FK and LBS for each source mesh."""
    buffers = {"o_head": [], **{name: [] for name in subs}}
    head_sub = {
        "skin_bone": trig.skin_bone,
        "bindpose": trig.bindpose,
        "bone_idx": trig.bone_idx,
        "bone_w": trig.bone_w,
        "verts_h": trig.verts_h,
    }
    meshes = {"o_head": head_sub, **subs}
    with torch.no_grad():
        for first in range(0, len(values), batch_size):
            sf = torch.as_tensor(
                values[first : first + batch_size], dtype=trig.dtype, device=trig.device
            )
            world = trig.bone_world(sf)
            for name, sub in meshes.items():
                buffers[name].append(trig.skin(sub, world).cpu().numpy())
    return {name: np.concatenate(batches) for name, batches in buffers.items()}


def generate_head(head_id, profile, args, region_specs):
    rig = HeadRig(head_id, sampling_profile=profile)
    trig = TorchHeadRig(rig, device=args.device, dtype=torch.float64)
    directory = args.out / f"head_{head_id}" / profile
    directory.mkdir(parents=True, exist_ok=True)
    mesh_files = {"o_head": Path(rig.data_dir) / "o_head_mesh.npz"}
    mesh_files.update(
        {
            path.stem: path
            for path in sorted((Path(rig.data_dir) / "submeshes").glob("*.npz"))
        }
    )
    hashes = {name: sha256(path) for name, path in mesh_files.items()}
    subs = {name: trig.load_submesh(name) for name in mesh_files if name != "o_head"}
    baseline_input = np.full(trig.n_slider, 0.5, dtype=np.float64)
    baseline = {
        name: mesh[0]
        for name, mesh in render_geometry(
            trig, baseline_input[None], subs, batch_size=1
        ).items()
    }
    mesh_layout = {}
    cursor = 0
    for name, verts in baseline.items():
        mesh_layout[name] = {
            "vertex_count": len(verts),
            "jacobian_row_start": cursor,
            "jacobian_row_stop": cursor + verts.size,
            "source_file": str(mesh_files[name]),
            "source_file_sha256": hashes[name],
        }
        cursor += verts.size
    regions = {}
    for name, specification in region_specs.get(str(head_id), {}).items():
        if name not in baseline:
            raise ValueError(f"Semantic mask references missing mesh {name}")
        regions[name] = validate_regions(
            specification,
            source_file_sha256=hashes[name],
            vertex_count=len(baseline[name]),
        )
    report = {
        "schema_version": 1,
        "head_id": head_id,
        "sampling_profile": profile,
        "coordinate_contract": "cached prefab-native FK world frame; neutral ancestor scale=1; no scene pose",
        "units": "Unity asset units; physical millimeter/centimeter scale not calibrated",
        "native_input_order": "all59 shapeValueFace controls including ears; index-based, not ML54 label order",
        "baseline_native_input": baseline_input.tolist(),
        "levels": args.levels,
        "abmx": "none",
        "expression": "no expression blendshape deltas; cached bind surface baseline",
        "mesh_layout": mesh_layout,
        "cache_hashes": {
            str(Path(rig.data_dir) / name): sha256(Path(rig.data_dir) / name)
            for name in ["skeleton.json", "anmShapeHead.json"]
        },
        "shared_table_hashes": {
            name: sha256(Path(rig.root_dir) / name)
            for name in ["customhead.json", "enums.json", "update_eqns.json"]
        },
        "semantic_region_provenance": region_specs.get(str(head_id), {}),
        "controls": [],
    }
    np.savez_compressed(
        directory / "baseline.npz", native_input=baseline_input, **baseline
    )
    for control in range(trig.n_slider):
        inputs = np.repeat(baseline_input[None], len(args.levels), axis=0)
        inputs[:, control] = args.levels
        meshes = render_geometry(trig, inputs, subs, batch_size=args.batch_size)
        archive = {"native_inputs": inputs, "levels": np.asarray(args.levels)}
        samples = []
        for sample, level in enumerate(args.levels):
            measurements = {}
            for name, candidates in meshes.items():
                stats, delta, mask = summarize_displacement(
                    baseline[name],
                    candidates[sample],
                    effect_threshold_relative=args.effect_threshold,
                )
                if name in regions:
                    stats["verified_regions"] = summarize_verified_regions(
                        delta, mask, regions[name]
                    )
                measurements[name] = stats
            samples.append(
                {
                    "level": level,
                    "native_input": inputs[sample].tolist(),
                    "meshes": measurements,
                }
            )
        for name, candidates in meshes.items():
            delta = candidates - baseline[name][None]
            threshold = max(
                np.linalg.norm(np.ptp(baseline[name], axis=0)) * args.effect_threshold,
                1e-12,
            )
            archive[name + "__displacement"] = delta
            archive[name + "__affected_vertex_mask"] = (
                np.linalg.norm(delta, axis=2) > threshold
            )
        path = directory / f"control_{control:02d}.npz"
        np.savez_compressed(path, **archive)
        report["controls"].append(
            {
                "index": control,
                "source_rows": [r for r in rig.customhead if r["category"] == control],
                "vertex_effects_path": str(path.resolve()),
                "samples": samples,
            }
        )
        if control % 10 == 0:
            print(
                f"head{head_id} {profile}: control{control}/{trig.n_slider}", flush=True
            )
    probes = np.repeat(baseline_input[None], trig.n_slider * 2, axis=0)
    for control in range(trig.n_slider):
        probes[control, control] -= args.step
        probes[trig.n_slider + control, control] += args.step
    probe_meshes = render_geometry(trig, probes, subs, batch_size=args.batch_size)
    raw_blocks, normalized_blocks, left_blocks, right_blocks = [], [], [], []
    for name, values in probe_meshes.items():
        center, left, right = finite_difference_jacobian(
            baseline[name], values[: trig.n_slider], values[trig.n_slider :], args.step
        )
        diagonal = float(np.linalg.norm(np.ptp(baseline[name], axis=0)))
        if diagonal <= 0:
            raise ValueError(
                f"Degenerate baseline mesh {name} cannot normalize geometric response"
            )
        metric_factor = diagonal * np.sqrt(len(baseline[name]))
        mesh_layout[name]["local_metric_divisor"] = metric_factor
        raw_blocks.append(center)
        normalized_blocks.append(center / metric_factor)
        left_blocks.append(left / metric_factor)
        right_blocks.append(right / metric_factor)
    raw_jacobian, normalized_jacobian = (
        np.concatenate(raw_blocks),
        np.concatenate(normalized_blocks),
    )
    left, right = np.concatenate(left_blocks), np.concatenate(right_blocks)
    jac_report, coupling, directions = analyze_jacobian(normalized_jacobian)
    head_jac_report, _, _ = analyze_jacobian(normalized_blocks[0])
    jac_report.update(
        {
            "step": args.step,
            "parameter_units": "native keyframe coefficients; no claim of distance units",
            "surface_metric": "permesh divide by baseline bbox diagonal and sqrt(vertexcount), then concatenate",
            "finite_difference_policy": "central symmetric response; at a keyframe knot averages left/right local slopes",
            "left_right_slope_difference_norms": np.linalg.norm(
                right - left, axis=0
            ).tolist(),
            "head_mesh_only": head_jac_report,
        }
    )
    archive = directory / "jacobian.npz"
    np.savez_compressed(
        archive,
        raw_jacobian=raw_jacobian,
        normalized_jacobian=normalized_jacobian,
        left_jacobian=left,
        right_jacobian=right,
        coupling=coupling,
        singular_control_directions=directions,
        baseline_native_input=baseline_input,
    )
    jac_report["array_path"] = str(archive.resolve())
    report["local_diagnostics"] = jac_report
    path = directory / "atlas.json"
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"head{head_id} {profile}: complete; localrank={jac_report['rank']}; atlas={path}",
        flush=True,
    )
    return {
        "head_id": head_id,
        "profile": profile,
        "atlas_path": str(path.resolve()),
        "controls": trig.n_slider,
        "levels": len(args.levels),
        "mesh_count": len(baseline),
        "local_rank": jac_report["rank"],
        "zero_columns": jac_report["zero_columns_at_this_baseline_and_mesh_set"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head-id", type=int, action="append")
    parser.add_argument(
        "--profile", choices=["vanilla", "slider_unlocker_18_2"], action="append"
    )
    parser.add_argument("--levels", default="-.25,0,.25,.5,.75,1,1.25")
    parser.add_argument("--step", type=float, default=1e-3)
    parser.add_argument("--effect-threshold", type=float, default=1e-6)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--regions", type=Path, help="source-matched verified vertex region JSON"
    )
    args = parser.parse_args()
    args.levels = [float(level) for level in args.levels.split(",")]
    if (
        not args.levels
        or not np.isfinite(args.levels).all()
        or len(set(args.levels)) != len(args.levels)
    ):
        parser.error("Levels must be distinct finite numbers")
    if not np.isfinite(args.step) or args.step <= 0 or args.step > 0.1:
        parser.error("Step must be finite and within (0,.1]")
    if args.batch_size < 1 or args.batch_size > 64:
        parser.error("Batch size must be within1..64")
    if not np.isfinite(args.effect_threshold) or args.effect_threshold < 0:
        parser.error("Effect threshold must be finite and nonnegative")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    heads = args.head_id if args.head_id is not None else available_heads()
    if not heads:
        parser.error("No cached heads available")
    regions = (
        json.loads(args.regions.read_text(encoding="utf-8"))["heads"]
        if args.regions
        else {}
    )
    summary = {
        "schema_version": 1,
        "purpose": "native surface effects and local conditioning; not a likeness score",
        "atlases": [],
    }
    for head in heads:
        for profile in args.profile or ["vanilla", "slider_unlocker_18_2"]:
            summary["atlases"].append(generate_head(head, profile, args, regions))
    path = args.out / "manifest.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={path}", flush=True)


if __name__ == "__main__":
    main()
