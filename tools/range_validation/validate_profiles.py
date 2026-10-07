"""Offline profile agreement report and mesh fixtures for later Unity comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from test_sampling_profiles import SamplingContracts  # noqa: E402

from src.hs2_deform_torch import TorchHeadRig  # noqa: E402
from src.hs2_mesh_deform import (  # noqa: E402
    HeadRig,
    _fk_world,
    available_heads,
    build_mesh,
)


def numpy_submesh(path, world):
    data = np.load(path, allow_pickle=True)
    matrices = (
        np.stack([world[str(pid)] for pid in data["skin_bone_pids"]]) @ data["bindpose"]
    )
    homogeneous = np.column_stack([data["verts"], np.ones(len(data["verts"]))])
    transforms = matrices[data["bone_idx"]]
    output = (
        np.einsum("vkij,vj->vki", transforms, homogeneous)[..., :3]
        * data["bone_w"][..., None]
    ).sum(axis=1)
    return output, data["faces"]


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head-id", type=int, action="append", help="default: all cached heads"
    )
    parser.add_argument("--sliders", default="0,4,24,27,47,53,54")
    parser.add_argument("--rates", default="-.25,.23,.77,1.25")
    parser.add_argument("--out", default="outputs/range_validation")
    parser.add_argument("--shape-file", help="JSON array of all59 native slider values")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SamplingContracts)
    )
    report = {
        "schema_version": 1,
        "purpose": "offline mathematical agreement; Unity/game parity still requires live exported meshes",
        "sampling_tests": {
            "run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
        },
        "abmx": "not applied; runtimeABMX length/cached-baseline and rotation-exclusion parity unverified",
        "expression": "neutral (expression blendshapes not applied)",
        "dtype": "float64",
        "device": args.device,
        "cases": [],
        "gradients": [],
        "cache_hashes": {},
    }
    heads = args.head_id if args.head_id is not None else available_heads()
    if not heads:
        raise ValueError(
            "No extracted heads; extract a card-driven rig before validation"
        )
    sliders = [int(x) for x in args.sliders.split(",")]
    rates = [float(x) for x in args.rates.split(",")]
    worst = 0.0
    for head_id in heads:
        profiles = {
            name: HeadRig(head_id, sampling_profile=name)
            for name in ["vanilla", "slider_unlocker_18_2"]
        }
        trigs = {
            name: TorchHeadRig(rig, device=args.device, dtype=torch.float64)
            for name, rig in profiles.items()
        }
        rig = profiles["vanilla"]
        baseline = (
            np.asarray(
                json.loads(Path(args.shape_file).read_text(encoding="utf-8")),
                dtype=float,
            )
            if args.shape_file
            else np.full(trigs["vanilla"].n_slider, 0.5)
        )
        if (
            baseline.ndim != 1
            or len(baseline) != trigs["vanilla"].n_slider
            or not np.isfinite(baseline).all()
        ):
            raise ValueError(
                "shape-file must be one finite native slider vector matching the cached rig"
            )
        cases = [("baseline", baseline)]
        for slider in sliders:
            if slider < 0 or slider >= len(baseline):
                raise ValueError(f"Invalid slider index {slider}")
            for rate in rates:
                values = baseline.copy()
                values[slider] = rate
                cases.append((f"slider_{slider:02d}_{rate:+.3f}", values))
        cache_paths = [
            Path(rig.data_dir) / name
            for name in ["o_head_mesh.npz", "skeleton.json", "anmShapeHead.json"]
        ]
        cache_paths += [
            Path(rig.root_dir) / name
            for name in ["customhead.json", "enums.json", "update_eqns.json"]
        ]
        subpaths = sorted((Path(rig.data_dir) / "submeshes").glob("*.npz"))
        cache_paths += subpaths
        report["cache_hashes"][str(head_id)] = {
            str(path): hash_file(path) for path in cache_paths
        }
        for profile, current in profiles.items():
            trig = trigs[profile]
            outdir = destination / f"head_{head_id}" / profile
            outdir.mkdir(parents=True, exist_ok=True)
            for name, values in cases:
                sf = torch.tensor(values[None], device=args.device, dtype=torch.float64)
                world = _fk_world(current, values)
                with torch.no_grad():
                    torch_world = trig.bone_world(sf)
                    head_torch = trig(sf)[0].cpu().numpy()
                head_numpy, faces = build_mesh(current, values)
                arrays = {
                    "shape_face": values,
                    "o_head": head_numpy,
                    "o_head__faces": faces,
                }
                errors = {"o_head": float(np.max(np.abs(head_numpy - head_torch)))}
                for subpath in subpaths:
                    verts, faces = numpy_submesh(subpath, world)
                    with torch.no_grad():
                        torch_verts = (
                            trig.skin(trig.load_submesh(subpath.stem), torch_world)[0]
                            .cpu()
                            .numpy()
                        )
                    arrays[subpath.stem], arrays[subpath.stem + "__faces"] = (
                        verts,
                        faces,
                    )
                    errors[subpath.stem] = float(np.max(np.abs(verts - torch_verts)))
                path = outdir / (name + ".npz")
                np.savez_compressed(path, **arrays)
                worst = max(worst, max(errors.values()))
                report["cases"].append(
                    {
                        "head_id": head_id,
                        "profile": profile,
                        "name": name,
                        "mesh_fixture": str(path),
                        "errors": errors,
                    }
                )
            # Outer-range head-geometry gradients, kept away from boundaries/keyframe knots.
            for slider in sliders[:4]:
                for rate in [-0.25, 1.25]:
                    values = baseline.copy()
                    values[slider] = rate
                    sf = torch.tensor(
                        values[None],
                        device=args.device,
                        dtype=torch.float64,
                        requires_grad=True,
                    )
                    trig(sf).square().sum().backward()
                    auto = float(sf.grad[0, slider])
                    losses = []
                    for sign in [1, -1]:
                        perturbed = sf.detach().clone()
                        perturbed[0, slider] += sign * 1e-5
                        losses.append(float(trig(perturbed).square().sum()))
                    fd = (losses[0] - losses[1]) / 2e-5
                    report["gradients"].append(
                        {
                            "head_id": head_id,
                            "profile": profile,
                            "slider": slider,
                            "rate": rate,
                            "autograd": auto,
                            "finite_difference": fd,
                            "passed": abs(auto - fd) <= 1e-7 + 1e-4 * abs(fd),
                        }
                    )
        print(
            f"head{head_id}: {len(cases) * 2} cases, head + {len(subpaths)} submeshes"
        )
    report["max_numpy_torch_error"] = worst
    report["passed"] = (
        result.wasSuccessful()
        and worst < 1e-9
        and all(row["passed"] for row in report["gradients"])
    )
    path = destination / "validation.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={path}; maxerror={worst:.3e}; passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
