"""Autograd versus independent central differences for all native/Chin controls."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig
from tools.stateful_fit.adapter import CHIN, NativeBaselineProtocol, StatefulTorchHeadRig, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Preserve prior evidence")
    protocol = NativeBaselineProtocol.from_manifest(args.manifest, args.contract, 3)
    rig = HeadRig(3)
    model = StatefulTorchHeadRig(rig, protocol, device="cuda", dtype=torch.float64)
    initial = np.r_[protocol.reference_native, protocol.reference_modifier]
    # Finite differences must stay on the same installed predicate branches.
    # Offsets with exact zero remain zero; their branch boundary is diagnosed,
    # not falsely certified as having a single smooth derivative.
    steps = np.r_[np.full(59, .001), np.full(3, .001), .001, np.full(3, .0001), np.full(3, .01)]
    knots = np.linspace(0, 1, model.n_key)
    native_knot_distances = np.min(np.abs(initial[:59, None] - knots[None]), axis=1)
    # The original full59 candidate has controls20/47 within1e-3 of .5.
    # Central differences crossing a sampling knot compare different slopes;
    # select the step from the declared table before evaluating either side.
    steps[:59] = np.minimum(steps[:59], native_knot_distances / 4)
    excluded = [59 + i for i in range(10) if i in (3,) and initial[59 + i] == 1]
    excluded += np.flatnonzero(native_knot_distances <= 1e-8).tolist()
    # has_position is a vector predicate; zero components are differentiable
    # while another component is nonzero. Rotation/scale predicates likewise.
    if np.all(initial[63:66] == 0):
        excluded += list(range(63, 66))
    if np.all(initial[66:69] == 0):
        excluded += list(range(66, 69))
    if np.all(initial[59:62] == 1):
        excluded += list(range(59, 62))
    weights = torch.as_tensor(np.random.default_rng(102026).normal(size=(4, len(rig.verts), 3)) / np.sqrt(len(rig.verts)),
                              device="cuda", dtype=torch.float64)
    identity = model.ab_tensor({})[0]

    def projections(parameter):
        ab = identity.index_copy(0, torch.tensor([model.chin_slot], device="cuda"), parameter[59:][None])
        vertices = model(parameter[:59][None], ab[None])[0]
        return (weights * vertices[None]).sum((1, 2))

    parameter = torch.tensor(initial, dtype=torch.float64, device="cuda", requires_grad=True)
    values = projections(parameter)
    analytic = torch.stack([torch.autograd.grad(value, parameter, retain_graph=True)[0] for value in values]).detach().cpu().numpy()
    rows = []
    with torch.no_grad():
        for index in range(69):
            if index in excluded:
                rows.append({"index": index, "status": "predicate_boundary_not_a_smooth_gradient_claim"})
                continue
            plus, minus = parameter.detach().clone(), parameter.detach().clone()
            plus[index] += steps[index]
            minus[index] -= steps[index]
            numeric = ((projections(plus) - projections(minus)) / (2 * steps[index])).cpu().numpy()
            error = float(np.max(np.abs(numeric - analytic[:, index])))
            denominator = float(max(np.max(np.abs(numeric)), np.max(np.abs(analytic[:, index]))))
            threshold = 1e-5 + .01 * denominator
            rows.append({"index": index, "analytic": analytic[:, index].tolist(), "central_difference": numeric.tolist(),
                         "step": float(steps[index]), "max_abs_error": error, "tolerance": threshold,
                         "passed": np.isfinite(numeric).all().item() and np.isfinite(analytic[:, index]).all().item() and error <= threshold})
    report = {"passed": all(row.get("passed", True) for row in rows), "rows": rows,
              "all59_native_checked": all(row.get("passed", False) for row in rows[:59]),
              "protocol": model.protocol_metadata, "parameter_point": initial.tolist(),
              "native_keyframe_knots": knots.tolist(), "native_distance_to_sampling_knot": native_knot_distances.tolist(),
              "tolerance_policy": "Four fixed seeded projections; fixed 1e-5 absolute +1% relative of maximum projected gradient. Native step=min(.001, nearest table-knot distance/4), selected before differences. Float32 transition uses declared finite steps; exact branch boundaries excluded explicitly.",
              "scope": "Differentiability at this actual nonempty head3 candidate; not global smoothness or new runtime certification",
              "source_hashes": [{"path": str(path.resolve()), "sha256": sha(path)} for path in (Path(__file__), ROOT / "src/hs2_abmx_torch.py", Path(__file__).with_name("adapter.py"))]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "native59": report["all59_native_checked"],
                      "failed": [row["index"] for row in rows if row.get("passed") is False],
                      "predicate_boundaries": excluded}), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
