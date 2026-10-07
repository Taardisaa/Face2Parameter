"""Small, bounded offline fitting harness using one common surface evaluator."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "geometry_quality"))

import numpy as np
import torch
from mesh_quality import Thresholds, analyze_snapshot
from contracts import AB_IDENTITY, CHIN, SCOPE, ab_dict, load_target, rig_mesh
from surface import TriangleSurface, content_hash, evaluate_surface, geometry, quadrature
from tools.stateful_fit.adapter import NativeBaselineProtocol, StatefulTorchHeadRig

PROFILES = {
    "native": {"sampling_profile": "vanilla", "native_bounds": [0, 1], "abmx": False},
    "installed18.2": {"sampling_profile": "slider_unlocker_18_2", "native_bounds": [-1, 2], "abmx": False},
    "native+ABMX": {"sampling_profile": "vanilla", "native_bounds": [0, 1], "abmx": True},
}
# Envelope between identity and the four actual recorded single-channel probes.
# Combinations/other amplitudes remain unverified in Unity even inside this box.
CHIN_LOWER = AB_IDENTITY.copy()
CHIN_UPPER = np.array([1.08, 1.02, 1.04, 1.05, .01, .02, 0, 2, 0, 1], float)
DEFAULT_ACCEPTANCE = {"rms": .002, "p95": .005, "max_sampled": .03, "normal_p95_degrees": 45}


def synthetic_snapshot(rig, trig, native, ab, vertices):
    """Geometry-quality adapter, explicitly an offline rig result, not Unity export."""
    with torch.no_grad():
        world = trig.bone_world(torch.as_tensor(native[None], dtype=trig.dtype, device=trig.device),
                                torch.as_tensor(ab[None], dtype=trig.dtype, device=trig.device))[0].cpu().numpy()
    path = f"/offline_cached_head_{rig.head_id}/o_head"
    mesh = {"mesh_name": "o_head", "renderer_path": path, "source_geometry_sha256": content_hash(rig.verts, rig.faces),
            "enabled": True, "active_in_hierarchy": True,
            "baked": {"vertices": vertices.tolist(), "triangles": rig.faces.reshape(-1).tolist()},
            "bone_names": [rig.bones[pid]["name"] for pid in trig.pids], "bone_transform_ids": list(range(trig.n_bones))}
    return {"schema_version": 1, "snapshot_kind": "maker_live_skinned_geometry",
            "origin": "offline quality adapter only; not a game capture or runtime certification",
            "character": {"head_id": rig.head_id}, "meshes": [mesh],
            "transforms": [{"id": i, "local_to_world": matrix.reshape(-1).tolist()} for i, matrix in enumerate(world)]}


def quality_gate(report, baseline_report):
    current, baseline = report["meshes"][0], baseline_report["meshes"][0]
    comparison = current["baseline"]
    new_degenerate = sorted(set(current["degenerate_triangle_ids"]) - set(baseline["degenerate_triangle_ids"]))
    reasons = []
    if comparison["status"] != "topology_matched" or comparison["source_hash_identical"] is not True:
        reasons.append("same-head/source/topology baseline correspondence failed")
    for value, reason in ((new_degenerate, "new degenerate triangles"),
                          (comparison.get("new_crossing_pairs", []), "new nonadjacent self-crossings"),
                          (comparison.get("normal_reversal_triangle_ids", []), "relative normal-direction reversals"),
                          (comparison.get("unusual_edges", []), "edge ratio outside [0.5,2]"),
                          (comparison.get("unusual_area_triangles", []), "area ratio outside [0.25,4]")):
        if value:
            reasons.append(reason)
    current_bones = current["bone_world_determinants"]["bones"]
    baseline_bones = baseline["bone_world_determinants"]["bones"]
    current_reflected = {row["name"] for row in current_bones if row.get("negative")}
    baseline_reflected = {row["name"] for row in baseline_bones if row.get("negative")}
    current_singular = {row["name"] for row in current_bones if row.get("near_singular")}
    baseline_singular = {row["name"] for row in baseline_bones if row.get("near_singular")}
    if current_reflected - baseline_reflected:
        reasons.append("new reflected bone-world frame")
    if current_singular - baseline_singular:
        reasons.append("new singular bone-world frame")
    return {"quality_valid": not reasons, "reasons": reasons, "new_degenerate_triangle_ids": new_degenerate,
            "existing_baseline_crossing_count": baseline["self_intersections"]["true_crossing_or_area_overlap_count"],
            "candidate_crossing_count": current["self_intersections"]["true_crossing_or_area_overlap_count"],
            "new_crossing_pairs": comparison.get("new_crossing_pairs", []),
            "new_reflected_bones": sorted(current_reflected - baseline_reflected),
            "new_singular_bones": sorted(current_singular - baseline_singular),
            "resolved_crossing_pairs": comparison.get("resolved_crossing_pairs", []),
            "thresholds": asdict(Thresholds()),
            "policy": "strict diagnostic gate; existing baseline flags retained, new crossings/reversals/degeneracy or abnormal ratios reject"}


def accepted(surface, quality, thresholds):
    return quality["quality_valid"] and all(surface["symmetric"][key] <= thresholds[key]
                                             for key in ("rms", "p95", "max_sampled")) \
        and surface["oriented_normal_degrees"]["p95"] <= thresholds["normal_p95_degrees"]


def torch_samples(vertices, faces, q, device, dtype):
    face_ids = torch.as_tensor(q.face_ids, dtype=torch.long, device=device)
    bary = torch.as_tensor(q.barycentric, dtype=dtype, device=device)
    return (vertices[faces[face_ids]] * bary[..., None]).sum(1)


def optimize(rig, trig, target, baseline_vertices, config, mode, seed):
    """Projected Adam; nearest-triangle assignments refreshed at each step.

    This is a finite local search, not an expressivity proof. The common final
    evaluator remains unchanged across heads/profiles/restarts.
    """
    device, dtype = trig.device, trig.dtype
    profile = PROFILES[mode]
    if profile["abmx"] and not isinstance(trig, StatefulTorchHeadRig) and config.get("legacy_static_abmx_diagnostic") is not True:
        raise ValueError("ABMX fitting requires an explicit stateful baseline/call protocol; static p*length+offset is not runtime-valid")
    active = np.asarray(config.get("active_native", list(range(59))))
    if active.ndim != 1 or (len(active) and not np.issubdtype(active.dtype, np.integer)) or len(set(active.tolist())) != len(active) or (len(active) and (active.min() < 0 or active.max() >= 59)):
        raise ValueError("active_native must be unique native indices 0..58")
    active = active.astype(np.int64)
    if not len(active) and not profile["abmx"]:
        raise ValueError("At least one enabled search parameter is required")
    native_lo, native_hi = profile["native_bounds"]
    lower = np.r_[np.full(len(active), native_lo), CHIN_LOWER if profile["abmx"] else []]
    upper = np.r_[np.full(len(active), native_hi), CHIN_UPPER if profile["abmx"] else []]
    initial_native = np.asarray(config.get("initial_native59", [.5] * 59), float)
    if initial_native.shape != (59,) or not np.isfinite(initial_native).all() or (initial_native < native_lo).any() or (initial_native > native_hi).any():
        raise ValueError("initial_native59 must fit the selected profile bounds")
    fixed_native = torch.as_tensor(initial_native, dtype=dtype, device=device)
    identity = torch.as_tensor(np.tile(AB_IDENTITY, (len(trig.ab_names), 1)), dtype=dtype, device=device)
    chin_slot = trig.ab_names.index(CHIN) if profile["abmx"] else None
    # Installed Apply predicates change exactly at identity. Start inside the
    # declared ABMX envelope so Adam does not confuse a masked identity-branch
    # gradient with a control having no effect. This is a declared seed, not a
    # smoothed replacement of the installed state transition.
    initial_ab = AB_IDENTITY
    if isinstance(trig, StatefulTorchHeadRig):
        initial_ab = CHIN_LOWER + .01 * (CHIN_UPPER - CHIN_LOWER)
    initial = np.r_[initial_native[active], initial_ab if profile["abmx"] else []]
    if config.get("random_start", False):
        initial = np.random.default_rng(seed).uniform(lower, upper)
    parameter = torch.nn.Parameter(torch.as_tensor(initial, dtype=dtype, device=device))
    optimizer = torch.optim.Adam([parameter], lr=config.get("learning_rate", .03))
    lo, hi = torch.as_tensor(lower, dtype=dtype, device=device), torch.as_tensor(upper, dtype=dtype, device=device)
    active_t = torch.as_tensor(active, dtype=torch.long, device=device)
    faces = torch.as_tensor(rig.faces, dtype=torch.long, device=device)
    sq = quadrature(baseline_vertices, rig.faces, config.get("optimization_samples", 128), seed, False)
    tq = quadrature(target.vertices, target.faces, config.get("optimization_samples", 128), seed, False)
    target_points = tq.points(target.vertices, target.faces)
    target_points_t = torch.as_tensor(target_points, dtype=dtype, device=device)
    target_surface = TriangleSurface(target.vertices, target.faces)
    reference_areas = geometry(baseline_vertices, rig.faces)[1]
    area_denominator = torch.as_tensor(reference_areas[sq.face_ids], dtype=dtype, device=device)
    source_face_ids = torch.as_tensor(sq.face_ids, dtype=torch.long, device=device)
    target_normal_t = torch.as_tensor(target_surface.normals[tq.face_ids], dtype=dtype, device=device)
    diagonal2 = float(np.linalg.norm(np.ptp(target.vertices, axis=0)) ** 2)
    if diagonal2 <= 0:
        raise ValueError("Target surface is collapsed")
    history, best, best_loss = [], None, float("inf")

    def unpack():
        native = fixed_native.index_copy(0, active_t, parameter[:len(active)])
        ab = identity.clone()
        if profile["abmx"]:
            ab[chin_slot] = parameter[len(active):]
        return native, ab

    for iteration in range(config.get("iterations", 30) + 1):
        optimizer.zero_grad()
        native, ab = unpack()
        vertices = trig(native[None], ab[None])[0]
        if not torch.isfinite(vertices).all():
            raise ValueError("Nonfinite candidate deformation")
        source_points = torch_samples(vertices, faces, sq, device, dtype)
        _, target_face_ids, target_bary = target_surface.closest(source_points.detach().cpu().numpy())
        nearest_target = (target_surface.triangles[target_face_ids] * target_bary[..., None]).sum(1)
        candidate_surface = TriangleSurface(vertices.detach().cpu().numpy(), rig.faces)
        _, candidate_face_ids, candidate_bary = candidate_surface.closest(target_points)
        candidate_face_t = torch.as_tensor(candidate_face_ids, dtype=torch.long, device=device)
        candidate_bary_t = torch.as_tensor(candidate_bary, dtype=dtype, device=device)
        nearest_candidate = (vertices[faces[candidate_face_t]] * candidate_bary_t[..., None]).sum(1)
        tri = vertices[faces]
        cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
        double_area = torch.linalg.vector_norm(cross, dim=-1)
        normals = cross / double_area.clamp_min(1e-15)[:, None]
        source_weights = double_area[source_face_ids] / 2 / area_denominator
        source_weights = source_weights / source_weights.sum().clamp_min(1e-15)
        source_d2 = ((source_points - torch.as_tensor(nearest_target, dtype=dtype, device=device)) ** 2).sum(1)
        target_d2 = ((nearest_candidate - target_points_t) ** 2).sum(1)
        distance_loss = (source_weights @ source_d2 + target_d2.mean()) / (2 * diagonal2)
        target_nearest_normals = torch.as_tensor(target_surface.normals[target_face_ids], dtype=dtype, device=device)
        normal_loss = ((source_weights * (1 - (normals[source_face_ids] * target_nearest_normals).sum(1))).sum()
                       + (1 - (normals[candidate_face_t] * target_normal_t).sum(1)).mean()) / 2
        loss = distance_loss + config.get("normal_loss_weight", 0.0) * normal_loss
        number = float(loss.detach())
        history.append({"iteration": iteration, "loss": number, "distance_loss": float(distance_loss.detach()),
                        "normal_loss": float(normal_loss.detach())})
        if number < best_loss:
            best_loss = number
            best = {"native59": native.detach().cpu().numpy(), "ab": ab.detach().cpu().numpy(),
                    "vertices": vertices.detach().cpu().numpy(), "iteration": iteration}
        if iteration == config.get("iterations", 30):
            break
        loss.backward()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError("Nonfinite parameter gradient")
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(lo, hi)
    return best, {"seed": seed, "initial_parameters": initial.tolist(), "active_native": active.tolist(),
                  "lower_bounds": lower.tolist(), "upper_bounds": upper.tolist(), "history": history,
                  "completed_iterations": config.get("iterations", 30), "winner_iteration": best["iteration"],
                  "objective": "symmetric area-weighted point-to-triangle squared distance / fixed target diagonal² + declared oriented normal term",
                  "sampling": "fixed seeded area-proportional source/target queries; candidate source importance weights track current areas",
                  "quality_during_optimization": "not enforced; final restart winners and same-head baseline are gated"}


def compare(config, *, device="cpu", progress=None):
    began = time.perf_counter()
    target = load_target(config["target"], device=device)
    acceptance = {**DEFAULT_ACCEPTANCE, **config.get("acceptance", {})}
    if any(not np.isfinite(v) or v < 0 for v in acceptance.values()):
        raise ValueError("Acceptance thresholds must be finite nonnegative")
    search = {"iterations": 30, "optimization_samples": 128, "learning_rate": .03, "restarts": 1,
              "evaluation_samples": 4096, "normal_loss_weight": 0.0, **config.get("search", {})}
    if search["iterations"] < 0 or search["restarts"] < 1 or search["optimization_samples"] < 1 or search["evaluation_samples"] < 1:
        raise ValueError("Invalid positive sample/restart counts or negative iterations")
    if not np.isfinite(search["learning_rate"]) or search["learning_rate"] <= 0 or not np.isfinite(search["normal_loss_weight"]) or search["normal_loss_weight"] < 0:
        raise ValueError("Learning rate/normal weight invalid")
    rows = []
    # All profiles share the neutral in-range baseline, but reuse only after
    # content equality including actual bone matrices, not merely head/name.
    neutral_quality_cache, surface_cache, quality_cache = {}, {}, {}
    for head in config.get("heads", [0, 1, 2]):
        for mode in config.get("modes", list(PROFILES)):
            if mode not in PROFILES:
                raise ValueError("Unknown bounded mode")
            baseline_mesh, rig, trig = rig_mesh(head, [.5] * 59, profile=PROFILES[mode]["sampling_profile"], device=device)
            baseline_ab = np.tile(AB_IDENTITY, (len(trig.ab_names), 1))
            legacy_abmx = False
            if PROFILES[mode]["abmx"]:
                protocol_config = config.get("abmx_protocol")
                if protocol_config is not None:
                    if set(protocol_config) != {"manifest", "contract", "apply_count"}:
                        raise ValueError("ABMX protocol requires manifest, independent contract and explicit common apply_count")
                    protocol = NativeBaselineProtocol.from_manifest(protocol_config["manifest"], protocol_config["contract"], head,
                                                                    apply_count=protocol_config["apply_count"])
                    trig = StatefulTorchHeadRig(rig, protocol, device=device, dtype=torch.float64)
                    with torch.no_grad():
                        baseline_mesh.vertices = trig(torch.full((1, 59), .5, dtype=trig.dtype, device=trig.device),
                                                      torch.as_tensor(baseline_ab[None], dtype=trig.dtype, device=trig.device))[0].cpu().numpy()
                    baseline_mesh.metadata = {**baseline_mesh.metadata,
                                              "content_sha256": content_hash(baseline_mesh.vertices, rig.faces),
                                              "abmx_protocol": trig.protocol_metadata}
                elif config.get("legacy_static_abmx_diagnostic") is True:
                    legacy_abmx = True
                else:
                    raise ValueError("ABMX fitting requires abmx_protocol; old stateless results are diagnostic evidence")
            baseline_snapshot = synthetic_snapshot(rig, trig, np.full(59, .5), baseline_ab, baseline_mesh.vertices)
            baseline_key = hashlib.sha256(json.dumps(baseline_snapshot, sort_keys=True).encode()).hexdigest()
            if baseline_key not in neutral_quality_cache:
                neutral_quality_cache[baseline_key] = analyze_snapshot(baseline_snapshot), {}
            baseline_quality, baseline_cache = neutral_quality_cache[baseline_key]
            candidates = [{"native59": np.full(59, .5), "ab": baseline_ab, "vertices": baseline_mesh.vertices,
                           "kind": "same_head_baseline", "search": None}]
            for restart in range(search["restarts"]):
                seed = int(config.get("seed", 0)) + restart
                restart_config = {**search, "random_start": restart > 0,
                                  "legacy_static_abmx_diagnostic": legacy_abmx}
                best, details = optimize(rig, trig, target, baseline_mesh.vertices, restart_config, mode, seed)
                candidates.append({**best, "kind": "bounded_search_winner", "search": details})
            evaluations = []
            for candidate in candidates:
                surface_key = (content_hash(candidate["vertices"], rig.faces),
                               search["evaluation_samples"], int(config.get("evaluation_seed", 7381)))
                if surface_key not in surface_cache:
                    surface_cache[surface_key] = evaluate_surface(candidate["vertices"], rig.faces, target.vertices, target.faces,
                                                                 count=search["evaluation_samples"], seed=int(config.get("evaluation_seed", 7381)),
                                                                 regions=config.get("regions"))
                surface = surface_cache[surface_key]
                snapshot = synthetic_snapshot(rig, trig, candidate["native59"], candidate["ab"], candidate["vertices"])
                snapshot_key = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
                quality_key = (snapshot_key, baseline_key)
                if quality_key not in quality_cache:
                    quality_cache[quality_key] = analyze_snapshot(snapshot, baseline_snapshot, baseline_cache=baseline_cache)
                quality_report = quality_cache[quality_key]
                quality = quality_gate(quality_report, baseline_quality)
                evaluations.append({"kind": candidate["kind"], "native59": candidate["native59"].tolist(),
                                    "abmx": ab_dict(candidate["ab"][trig.ab_names.index(CHIN)]) if mode == "native+ABMX" else {},
                                    "surface": surface, "quality": quality, "quality_report": quality_report,
                                    "accepted": accepted(surface, quality, acceptance) and not legacy_abmx,
                                    "legacy_static_abmx_diagnostic": legacy_abmx, "search": candidate["search"]})
            eligible = [item for item in evaluations if item["accepted"]]
            valid = [item for item in evaluations if item["quality"]["quality_valid"]]
            best_pool = eligible or valid or evaluations
            selected = min(range(len(evaluations)), key=lambda i: evaluations[i]["surface"]["symmetric"]["rms"]
                           if evaluations[i] in best_pool else float("inf"))
            rows.append({"head_id": head, "mode": mode, "sampling_profile": PROFILES[mode]["sampling_profile"],
                         "native_bounds": PROFILES[mode]["native_bounds"], "baseline": baseline_mesh.metadata,
                         "status": "found_quality_valid_approximation" if eligible else "not_found_within_this_search",
                         "selected_candidate_index": selected, "candidates": evaluations,
                         "abmx_runtime_scope": "Explicit native-baseline and measured persistent-history model; new search candidates require actual state/cursor validation"
                         if isinstance(trig, StatefulTorchHeadRig) else "Known-invalid static model; diagnostic only, acceptance disabled" if legacy_abmx else "none",
                         "abmx_protocol": trig.protocol_metadata if isinstance(trig, StatefulTorchHeadRig) else None})
            if progress is not None:
                progress(rows[-1], len(rows))
    evidence_path = ROOT / "outputs" / "unity_parity_20261004" / "recorded_abmx_cases.json"
    probe_evidence = {"path": str(evidence_path), "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                      "scope": "head 2 four recorded single-channel ChinTip probes only; head 0/1/3 and combined optimization points not certified"} \
        if evidence_path.exists() else {"scope": "recorded probe evidence unavailable in this checkout; no runtime claim"}
    return {"schema_version": 1, "report_kind": "hs2_common_surface_base_comparison", "target": target.metadata,
            "configuration": config, "resolved_search": search, "acceptance": acceptance, "scope": dict(SCOPE),
            "device": device, "dtype": "torch.float64", "software": {"numpy": np.__version__, "torch": torch.__version__},
            "abmx_probe_evidence": probe_evidence, "results": rows, "elapsed_seconds": time.perf_counter() - began,
            "limitations": ["Entire o_head only; eyes/lashes and multiview silhouettes remain future acceptance groups.",
                            "No real-person calibrated 3D truth is supplied by bone centers or candidate FAN landmarks.",
                            "Finite local bounded search failure does not prove a base cannot represent a target or justify a new base.",
                            "Surface maxima are sampled, not continuous exact Hausdorff bounds; predicates and nearest normals are tolerance-sensitive.",
                            "Same-head offline quality gate is strict diagnostic policy, not proof of complete anatomy or visual quality.",
                            "No fitted scale/affine or fitted rigid registration changes the metric; explicit proper rigid input transform only."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    torch.set_num_threads(int(config.get("cpu_threads", 2)))
    report = compare(config, device=args.device)
    report["configuration_file_sha256"] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "elapsed_seconds": report["elapsed_seconds"],
                      "results": [{"head_id": row["head_id"], "mode": row["mode"], "status": row["status"],
                                   "rms": row["candidates"][row["selected_candidate_index"]]["surface"]["symmetric"]["rms"]}
                                  for row in report["results"]]}, indent=2))


if __name__ == "__main__":
    main()
