"""Read-only exported geometry reconstruction; no game or bridge calls."""

from __future__ import annotations

import numpy as np


class Uncertifiable(ValueError):
    """Incomplete or ambiguous inputs cannot certify an engine convention."""


def finite_array(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise Uncertifiable(
            f"Expected finite array with shape {shape}, got {array.shape}"
        )
    return array


def matrix(value):
    result = finite_array(value).reshape(4, 4)
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7, rtol=0):
        raise Uncertifiable("Expected an affine, row-major column-vector matrix")
    return result


def transform_points(points, transform):
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix(transform) @ homogeneous.T).T[:, :3]


def vertex_errors(reference, actual):
    reference = finite_array(reference)
    actual = finite_array(actual, reference.shape)
    difference = actual - reference
    lengths = np.linalg.norm(difference, axis=1)
    scale = float(np.linalg.norm(np.ptp(reference, axis=0)))
    return {
        "vertex_count": len(reference),
        "max_l2": float(lengths.max()),
        "rms_l2": float(np.sqrt(np.mean(lengths**2))),
        "max_abs_component": float(np.abs(difference).max()),
        "reference_bbox_diagonal": scale,
        "max_normalized": float(lengths.max() / scale) if scale > 0 else None,
        "rms_normalized": float(np.sqrt(np.mean(lengths**2)) / scale)
        if scale > 0
        else None,
    }


def blendshape_delta(mesh):
    """Piecewise linear frame deltas with an implicit zero-weight base pose.

    Extrapolation uses the outer segment; recorded as a hypothesis until agreement
    with BakeMesh verifies it for the sampled weights, not a universal Unity claim.
    """
    source = finite_array(mesh["source"]["vertices"])
    delta = np.zeros_like(source)
    active = []
    for shape in mesh.get("blendshapes", []):
        weight = float(shape["current_weight"])
        if not np.isfinite(weight):
            raise Uncertifiable("Blendshape weights must be finite")
        if weight == 0:
            continue
        frames = shape.get("frames", [])
        if not frames or any("delta_vertices" not in frame for frame in frames):
            raise Uncertifiable(
                f"Active blendshape {shape.get('name')} lacks vertex frame deltas"
            )
        keys = [
            (
                float(frame["weight"]),
                finite_array(frame["delta_vertices"], source.shape),
            )
            for frame in frames
        ]
        if not any(key == 0 for key, _ in keys):
            keys.append((0.0, np.zeros_like(source)))
        keys.sort(key=lambda pair: pair[0])
        keyweights = np.asarray([key for key, _ in keys])
        if not np.isfinite(keyweights).all() or np.any(np.diff(keyweights) <= 0):
            raise Uncertifiable("Blendshape frame weights must be finite and unique")
        if len(keys) == 1:
            raise Uncertifiable("Active blendshape has only a zero-weight frame")
        index = int(
            np.clip(
                np.searchsorted(keyweights, weight, side="right") - 1, 0, len(keys) - 2
            )
        )
        (a, va), (b, vb) = keys[index : index + 2]
        delta += va + (vb - va) * ((weight - a) / (b - a))
        active.append(
            {
                "name": shape.get("name"),
                "weight": weight,
                "frame_weights": keyweights.tolist(),
            }
        )
    return delta, active


def skin_world(mesh, transforms, *, influences=4):
    """Exact four-weight LBS, or explicit top-weight normalized quality hypothesis."""
    if influences not in [1, 2, 4]:
        raise Uncertifiable("influences must be 1,2 or4")
    source = mesh["source"]
    verts = finite_array(source["vertices"])
    if verts.ndim != 2 or verts.shape[1] != 3 or not len(verts):
        raise Uncertifiable("Source vertices must be nonempty N×3")
    delta, active = blendshape_delta(mesh)
    indices_raw = finite_array(source["bone_indices"], (len(verts), 4))
    if not np.array_equal(indices_raw, indices_raw.astype(np.int64)):
        raise Uncertifiable("Bone indices must be integers")
    indices = indices_raw.astype(np.int64)
    weights = finite_array(source["bone_weights"], indices.shape)
    if (weights < 0).any() or not np.allclose(
        weights.sum(axis=1), 1, atol=1e-5, rtol=0
    ):
        raise Uncertifiable("Bone weights must be nonnegative and sum to one")
    poses = [matrix(pose) for pose in source["bindposes"]]
    ids = mesh["bone_transform_ids"]
    if not poses or len(poses) != len(ids):
        raise Uncertifiable("Bindpose count must equal nonempty renderer bone count")
    used = weights > 0
    if (((indices < 0) | (indices >= len(ids))) & used).any():
        raise Uncertifiable("An active bone influence has an invalid index")
    # Zero-weight influences may legally refer to an absent bone. Never use them.
    safe_indices = np.where(used, indices, 0)
    needed = set(safe_indices[used].tolist())
    bone_matrices = []
    for index, (bone_id, bindpose) in enumerate(zip(ids, poses)):
        if index not in needed:
            bone_matrices.append(np.eye(4))
            continue
        if bone_id not in transforms:
            raise Uncertifiable(f"Missing active bone transform {bone_id}")
        bone_matrices.append(matrix(transforms[bone_id]["local_to_world"]) @ bindpose)
    if influences < 4:
        order = np.argsort(-weights, axis=1, kind="stable")[:, :influences]
        safe_indices = np.take_along_axis(safe_indices, order, axis=1)
        weights = np.take_along_axis(weights, order, axis=1)
        weights = weights / weights.sum(axis=1, keepdims=True)
    skin = np.stack(bone_matrices)[safe_indices]
    homogeneous = np.column_stack([verts + delta, np.ones(len(verts))])
    world = (
        np.einsum("nkij,nj->nki", skin, homogeneous)[..., :3] * weights[..., None]
    ).sum(axis=1)
    return world, active


def quality_influences(mesh, snapshot):
    quality = mesh.get("skin_quality", "Auto")
    if quality == "Auto":
        quality = snapshot.get("game", {}).get("skin_quality_setting")
    return {
        "Bone1": 1,
        "OneBone": 1,
        "Bone2": 2,
        "TwoBones": 2,
        "Bone4": 4,
        "FourBones": 4,
    }.get(quality)


def analyze_snapshot(snapshot, *, normalized_tolerance=1e-5, influences=None):
    """Certify only snapshot-local numerical agreement, including all inputs."""
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported geometry snapshot schema")
    transforms = {item["id"]: item for item in snapshot["transforms"]}
    report = {
        "schema_version": 1,
        "scope": "snapshot-local LBS versus BakeMesh; no universal engine convention inferred",
        "blendshape_interpolation": "linear framewise with implicit zero base and outer-segment extrapolation",
        "normalized_tolerance": normalized_tolerance,
        "snapshot_frame_stable": snapshot.get("frame_count")
        == snapshot.get("frame_count_end"),
        "meshes": [],
    }
    for mesh in snapshot["meshes"]:
        row = {
            "mesh_name": mesh["mesh_name"],
            "renderer_path": mesh.get("renderer_path"),
            "source_geometry_sha256": mesh.get("source_geometry_sha256"),
            "certified": False,
        }
        row["renderer_visibility"] = {
            key: mesh.get(key)
            for key in ("enabled", "active_self", "active_in_hierarchy")
        }
        report["meshes"].append(row)
        try:
            declared = quality_influences(mesh, snapshot)
            selected = influences if influences is not None else declared
            if selected is None:
                raise Uncertifiable(
                    "Unknown skin quality; supply explicit influence hypothesis"
                )
            world, active = skin_world(mesh, transforms, influences=selected)
            row.update(
                {
                    "influences": selected,
                    "declared_influences": declared,
                    "influences_overridden": influences is not None,
                    "active_blendshapes": active,
                }
            )
            if not np.array_equal(
                mesh["source"]["triangles"], mesh["baked"]["triangles"]
            ):
                raise Uncertifiable("Source and baked topology/order differs")
            candidates = mesh["baked"]["world_candidates"]
            raw = finite_array(mesh["baked"]["vertices"], world.shape)
            errors = {}
            for name, candidate in candidates.items():
                converted = transform_points(raw, candidate["matrix"])
                errors[name] = vertex_errors(world, converted)
                errors[name]["exported_world_consistency"] = vertex_errors(
                    converted, candidate["vertices"]
                )
            row["candidate_errors"] = errors
            matching = [
                name
                for name, err in errors.items()
                if err["max_normalized"] is not None
                and err["max_normalized"] <= normalized_tolerance
                and err["exported_world_consistency"]["max_normalized"] is not None
                and err["exported_world_consistency"]["max_normalized"]
                <= normalized_tolerance
            ]
            row["matching_candidates"] = matching
            row["selected_candidate"] = matching[0] if len(matching) == 1 else None
            row["certified"] = bool(matching) and report["snapshot_frame_stable"]
            row["convention_distinguished"] = len(matching) == 1
            if len(matching) != 1:
                row["note"] = (
                    "Both candidates agree: scale convention not distinguishable in this snapshot"
                    if matching
                    else "Neither candidate agrees; inspect skin quality, expression and input integrity"
                )
            row["quality_hypotheses"] = {}
            for count in [1, 2, 4]:
                hypothesis, _ = skin_world(mesh, transforms, influences=count)
                row["quality_hypotheses"][str(count)] = {
                    name: vertex_errors(hypothesis, candidate["vertices"])
                    for name, candidate in candidates.items()
                }
        except (Uncertifiable, ValueError, KeyError, IndexError) as exc:
            row["uncertifiable_reason"] = str(exc)
    return report


def rigid_alignment(source, target, *, unit_scale=1.0):
    """Proper rotation + translation only; any unit scale is explicit, not fitted."""
    source = finite_array(source)
    target = finite_array(target, source.shape)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise Uncertifiable("Rigid alignment requires corresponding N×3 points, N>=3")
    if not np.isfinite(unit_scale) or unit_scale <= 0:
        raise Uncertifiable("Explicit unit scale must be finite and positive")
    scaled = source * unit_scale
    source_center, target_center = scaled.mean(axis=0), target.mean(axis=0)
    a, b = scaled - source_center, target - target_center
    u, singular, vt = np.linalg.svd(a.T @ b)
    if np.linalg.matrix_rank(a) < 2 or np.linalg.matrix_rank(b) < 2:
        raise Uncertifiable(
            "Degenerate points cannot determine a proper rigid alignment"
        )
    correction = np.eye(3)
    correction[-1, -1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    rotation = u @ correction @ vt
    translation = target_center - source_center @ rotation
    aligned = scaled @ rotation + translation
    suggested_scale = float(np.sum(singular * np.diag(correction)) / np.square(a).sum())
    return aligned, {
        "unit_scale_applied": unit_scale,
        "rotation_row_vector": rotation.tolist(),
        "translation": translation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
        "suggested_additional_uniform_scale_NOT_applied": suggested_scale,
        "raw_errors": vertex_errors(target, scaled),
        "rigid_errors": vertex_errors(target, aligned),
    }


def recorded_uniform_renderer_scale(mesh, transforms):
    """Validate a recorded ancestor scale independently of vertex fitting."""
    linear = matrix(mesh["renderer_local_to_world"])[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(scales.mean())
    if scale <= 0 or np.linalg.det(linear) <= 0:
        raise Uncertifiable("Renderer scale must be positive and have no reflection")
    normalized = linear / scales
    gram = normalized.T @ normalized
    spread = float(np.max(np.abs(scales / scale - 1)))
    shear = float(np.max(np.abs(gram - np.eye(3))))
    if spread > 1e-5 or shear > 1e-5:
        raise Uncertifiable(
            "Recorded renderer transform is nonuniform or sheared; cannot remove it as a scalar"
        )
    lossy = finite_array(mesh["renderer_lossy_scale"], (3,))
    if not np.allclose(lossy, scales, rtol=1e-5, atol=1e-7):
        raise Uncertifiable("Renderer matrix and exported lossy scale disagree")
    contributors = []
    current = transforms.get(mesh["renderer_transform_id"])
    seen = set()
    while current:
        if current["id"] in seen:
            raise Uncertifiable("Cycle in exported ancestor chain")
        seen.add(current["id"])
        local = finite_array(current["local_scale"], (3,))
        if not np.allclose(local, [1, 1, 1], rtol=0, atol=1e-7):
            contributors.append(
                {
                    "id": current["id"],
                    "name": current["name"],
                    "path": current.get("path"),
                    "local_scale": local.tolist(),
                    "lossy_scale": current.get("lossy_scale"),
                }
            )
        current = transforms.get(current.get("parent_id"))
    return scale, {
        "factor": scale,
        "source": "exported renderer_local_to_world column lengths; NOT vertex-fitted",
        "renderer_transform_id": mesh["renderer_transform_id"],
        "renderer_path": mesh.get("renderer_path"),
        "axis_scales": scales.tolist(),
        "exported_lossy_scale": lossy.tolist(),
        "relative_axis_spread": spread,
        "orthogonality_error": shear,
        "ancestor_scale_contributors": contributors,
    }
