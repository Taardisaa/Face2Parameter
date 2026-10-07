"""Calibrated ray consensus; a good fit never proves a detector's anatomy."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

from tools.surface_calibration.core import (
    Camera, ContractError, SurfaceMesh, finite_array, follow, observe,
    validate_reprojection,
)


@dataclass(frozen=True)
class Policy:
    # Fixed before examining residuals. These are operational tolerances, not
    # detector accuracy estimates. Tightening/loosening creates a new policy.
    max_error_px: float = 2.0
    min_ray_angle_degrees: float = 10.0
    max_ray_condition: float = 100.0
    max_normal_incidence_degrees: float = 75.0
    max_loo_normal_spread_degrees: float = 20.0
    max_abs_yaw_degrees: float = 30.0
    min_observations: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ContractError(f"Invalid policy {name}")
        if type(self.min_observations) is not int or self.min_observations < 3:
            raise ContractError("LOO requires at least three observations")
        if self.max_abs_yaw_degrees > 30:
            raise ContractError("Unverified detector hidden-side predictions require the conservative <=30 degree scope")


STABLE = {
    30: ("nose_apex_candidate", "FAN30 purported nose apex on the head skin"),
    48: ("mouth_corner_48_candidate", "FAN48 purported first outer lip commissure on the head skin"),
    54: ("mouth_corner_54_candidate", "FAN54 purported opposite outer lip commissure on the head skin"),
    36: ("eye_corner_36_candidate", "FAN36 purported first eye's outer lid junction on the head skin"),
    39: ("eye_corner_39_candidate", "FAN39 purported first eye's inner lid junction on the head skin"),
    42: ("eye_corner_42_candidate", "FAN42 purported opposite eye's inner lid junction on the head skin"),
    45: ("eye_corner_45_candidate", "FAN45 purported opposite eye's outer lid junction on the head skin"),
}


def definitions():
    stable = [{"detector_index": i, "id": name, "kind": "stable_surface_hypothesis",
               "semantic_definition": definition, "target_mesh_name": "o_head",
               "semantic_definition_validated": False,
               "material_definition": "source geometry SHA + renderer path + ordered triangle IDs + barycentric weights"}
              for i, (name, definition) in STABLE.items()]
    moving = [{"detector_index": i, "id": f"jaw_outline_{i}", "kind": "view_dependent_silhouette",
               "semantic_definition": f"FAN{i} image jaw outline sample; apparent contour moves with view",
               "semantic_definition_validated": False, "fixed_material_point_allowed": False}
              for i in range(17)]
    return stable + moving


def triangulate(rays, policy=Policy()):
    """Least perpendicular-distance solution with finite near/far checks."""
    if len(rays) < 2:
        raise ContractError("At least two independent camera rays required")
    origins = np.asarray([r.origin for r in rays])
    directions = np.asarray([r.direction for r in rays])
    if origins.shape != (len(rays), 3) or not np.isfinite(origins).all():
        raise ContractError("Invalid ray origins")
    if not np.isfinite(directions).all() or not np.allclose(np.linalg.norm(directions, axis=1), 1, atol=1e-8):
        raise ContractError("Ray directions must be finite unit vectors")
    distances = finite_array([r.max_distance for r in rays], (len(rays),))
    if distances.min() <= 0:
        raise ContractError("Positive near/far ray extents required")
    # Anti-parallel rays contain the same line-direction information.
    angles = np.degrees(np.arccos(np.clip(np.abs(directions @ directions.T), 0, 1)))
    best_angle = float(angles.max())
    projectors = np.eye(3) - directions[:, :, None] * directions[:, None, :]
    a = projectors.sum(axis=0)
    singular = np.linalg.eigvalsh(a)
    if singular[0] <= 1e-10:
        raise ContractError("Degenerate parallel/anti-parallel ray geometry")
    condition = float(singular[-1] / singular[0])
    if best_angle < policy.min_ray_angle_degrees or condition > policy.max_ray_condition:
        raise ContractError("Insufficient independent ray angle/conditioning")
    b = np.einsum("nij,nj->i", projectors, origins)
    point = np.linalg.solve(a, b)
    along = np.einsum("ij,ij->i", point - origins, directions)
    if np.any(along < -1e-7) or np.any(along > distances + 1e-7):
        raise ContractError("Triangulated point outside a ray's near/far segment")
    residuals = np.linalg.norm(np.einsum("nij,nj->ni", projectors, point - origins), axis=1)
    return {"world": point.tolist(), "ray_condition": condition, "best_ray_angle_degrees": best_angle,
            "ray_perpendicular_residuals_game_units": residuals.tolist(),
            "along_ray_distances": along.tolist(),
            "unit_noise_inverse_information": np.linalg.inv(a).tolist(),
            "uncertainty_is_calibrated_probability": False}, a, b


def triangle_minima(mesh, a, b):
    """Exact minimum of a convex quadratic on every triangle's simplex.

    Interior stationary point and all three clamped edge minima are enumerated.
    This minimizes ray distance, with no camera/scale/shape fitting.
    """
    points = mesh.vertices[mesh.triangles]
    bases = points[:, 0]
    edge = np.stack((points[:, 1] - bases, points[:, 2] - bases), axis=2)
    h = np.einsum("nki,kl,nlj->nij", edge, a, edge)
    rhs = np.einsum("nki,nk->ni", edge, b - bases @ a)
    valid = np.linalg.det(h) > 1e-20
    uv = np.zeros((len(points), 2))
    uv[valid] = np.linalg.solve(h[valid], rhs[valid, :, None])[:, :, 0]
    bary = np.column_stack((1 - uv.sum(axis=1), uv))
    valid &= bary.min(axis=1) >= -1e-10
    worlds = np.einsum("ni,nij->nj", bary, points)
    objective = np.einsum("ni,ij,nj->n", worlds, a, worlds) - 2 * worlds @ b
    objective[~valid] = np.inf
    for start, end in [(0, 1), (1, 2), (2, 0)]:
        origin, delta = points[:, start], points[:, end] - points[:, start]
        denom = np.einsum("ni,ij,nj->n", delta, a, delta)
        t = np.clip(np.divide(np.einsum("ni,ni->n", delta, b - origin @ a), denom,
                              out=np.zeros_like(denom), where=denom > 1e-20), 0, 1)
        proposed = origin + t[:, None] * delta
        cost = np.einsum("ni,ij,nj->n", proposed, a, proposed) - 2 * proposed @ b
        better = cost < objective
        worlds[better], objective[better] = proposed[better], cost[better]
        replacement = np.zeros_like(bary)
        replacement[:, start], replacement[:, end] = 1 - t, t
        bary[better] = replacement[better]
    degenerate = np.linalg.norm(np.cross(edge[:, :, 0], edge[:, :, 1]), axis=1) < 1e-12
    objective[degenerate] = np.inf
    return points, bary, worlds, objective


def material(mesh, triangle, barycentric):
    return follow({"renderer_path": mesh.renderer_path, "source_geometry_sha256": mesh.source_hash,
                   "triangle_id": int(triangle), "vertex_ids": mesh.triangles[triangle].tolist(),
                   "submesh_id": int(mesh.submesh_ids[triangle]), "barycentric": barycentric.tolist()}, [mesh])


def fit(cameras, xy, meshes, targets, policy=Policy()):
    rays = [camera.ray(point) for camera, point in zip(cameras, xy, strict=True)]
    tri, a, b = triangulate(rays, policy)
    options = []
    for mesh in meshes:
        if mesh.renderer_path not in targets or not mesh.visible:
            continue
        _, bary, _, objective = triangle_minima(mesh, a, b)
        options.extend((float(objective[t]), mesh, int(t), bary[t]) for t in np.flatnonzero(np.isfinite(objective)))
    options.sort(key=lambda item: (item[0], item[1].renderer_path, item[2]))
    rejected = {"visibility_or_clipping": 0, "grazing_normal": 0}
    # Exact triangle minima are examined in increasing objective order. The
    # first eligible minimum is retained; visibility-constrained edge optima
    # inside a triangle are NOT searched, so this may conservatively reject.
    for _, mesh, triangle, weights in options:
        surface = material(mesh, triangle, weights)
        projections = [validate_reprojection(surface, camera, meshes, observed_xy=point,
                                            max_error_px=policy.max_error_px)
                       for camera, point in zip(cameras, xy, strict=True)]
        if any(p["status"] not in ("within_pixel_tolerance", "outside_pixel_tolerance") for p in projections):
            rejected["visibility_or_clipping"] += 1
            continue
        normal = np.asarray(surface["geometric_normal"])
        incidence = [float(np.degrees(np.arccos(np.clip(abs(normal @ ray.direction), 0, 1)))) for ray in rays]
        if max(incidence) > policy.max_normal_incidence_degrees:
            rejected["grazing_normal"] += 1
            continue
        error = max(p["error_px"] for p in projections)
        return {"surface": surface, "triangulation": tri, "training": projections,
                "max_training_error_px": error, "normal_incidence_degrees": incidence,
                "geometric_screen_rejected_minima": rejected,
                "surface_constraint_displacement_game_units": float(np.linalg.norm(np.asarray(surface["world"]) - tri["world"]))}
    raise ContractError(f"No geometrically visible non-grazing surface minimum; rejected={rejected}")


def evaluate_candidate(definition, cameras, image_observations, meshes, targets, policy=Policy()):
    index = definition["detector_index"]
    output = {**definition, "anatomical_correspondence_validated": False,
              "detector_visibility_verified": False, "material_visibility_validated": False,
              "policy": asdict(policy), "observations": [], "accepted": False}
    eligible = []
    for camera, observation in zip(cameras, image_observations, strict=True):
        xy = observation["landmark68"][index]
        record = {"view_index": observation["view_index"], "view_id": camera.view_id,
                  "xy": xy, "png_sha256": observation["raw_png_sha256"], "yaw": observation["yaw"]}
        if definition["kind"] == "view_dependent_silhouette":
            record.update(observe(camera, meshes, {"id": definition["id"], "xy": xy, "kind": "silhouette"}))
        elif abs(float(observation["yaw"])) > policy.max_abs_yaw_degrees:
            record.update(status="excluded_profile_or_hidden_side_risk")
        elif not observation["heatmap_grid_decode"]["raw_interpolation_footprint_fully_in_image"][index]:
            record.update(status="excluded_crop_padding")
        else:
            screened = observe(camera, meshes, {"id": definition["id"], "xy": xy, "allowed_renderer_paths": targets})
            record.update(screened)
            if screened["status"] == "surface_candidate":
                eligible.append((camera, xy, record["view_index"]))
        output["observations"].append(record)
    if definition["kind"] == "view_dependent_silhouette":
        return {**output, "status": "silhouette_requires_view_specific_definition"}
    output["eligible_view_indices"] = [x[2] for x in eligible]
    if len(eligible) < policy.min_observations:
        return {**output, "status": "insufficient_geometrically_unoccluded_observations"}
    try:
        final = fit([e[0] for e in eligible], [e[1] for e in eligible], meshes, targets, policy)
    except ContractError as exc:
        return {**output, "status": "surface_fit_rejected", "reason": str(exc)}
    output["all_view_fit"] = final
    heldout = []
    for i, excluded in enumerate(eligible):
        training = eligible[:i] + eligible[i+1:]
        record = {"heldout_view_index": excluded[2], "training_view_indices": [e[2] for e in training],
                  "heldout_observation_used_in_fit": False}
        try:
            fitted = fit([e[0] for e in training], [e[1] for e in training], meshes, targets, policy)
            check = validate_reprojection(fitted["surface"], excluded[0], meshes,
                                          observed_xy=excluded[1], max_error_px=policy.max_error_px)
            heldout_ray = excluded[0].ray(excluded[1])
            heldout_incidence = float(np.degrees(np.arccos(np.clip(abs(np.asarray(fitted["surface"]["geometric_normal"]) @ heldout_ray.direction), 0, 1))))
            record.update(fitted_surface=fitted["surface"], training_max_error_px=fitted["max_training_error_px"],
                          triangulation=fitted["triangulation"], heldout=check,
                          heldout_normal_incidence_degrees=heldout_incidence,
                          accepted=check["status"] == "within_pixel_tolerance" and fitted["max_training_error_px"] <= policy.max_error_px and heldout_incidence <= policy.max_normal_incidence_degrees)
        except ContractError as exc:
            record.update(accepted=False, reason=str(exc))
        heldout.append(record)
    output["leave_one_view_out"] = heldout
    normals = [np.asarray(final["surface"]["geometric_normal"])] + [np.asarray(h["fitted_surface"]["geometric_normal"]) for h in heldout if "fitted_surface" in h]
    normal_spread = max(float(np.degrees(np.arccos(np.clip(n @ m, -1, 1)))) for n in normals for m in normals)
    output["loo_normal_spread_degrees"] = normal_spread
    errors = [h["heldout"]["error_px"] for h in heldout if h.get("heldout", {}).get("error_px") is not None]
    output["max_heldout_error_px"] = max(errors) if errors else None
    output["accepted"] = all(h["accepted"] for h in heldout) and final["max_training_error_px"] <= policy.max_error_px and normal_spread <= policy.max_loo_normal_spread_degrees
    output["status"] = "empirical_operational_material_point_candidate" if output["accepted"] else "independent_consensus_gate_failed"
    output["numeric_material_definition_valid"] = True
    output["numeric_material_definition_is_anatomical_ground_truth"] = False
    return output
