"""Geometric measurements and local sensitivity diagnostics, without anatomy guesses."""

from __future__ import annotations

import numpy as np


def vertices(value):
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or not len(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError("Expected nonempty finite N×3 vertices")
    return array


def summarize_displacement(baseline, candidate, *, effect_threshold_relative=1e-6):
    baseline, candidate = vertices(baseline), vertices(candidate)
    if baseline.shape != candidate.shape:
        raise ValueError("Baseline/candidate must preserve vertex correspondence")
    if not np.isfinite(effect_threshold_relative) or effect_threshold_relative < 0:
        raise ValueError("Effect threshold must be finite and nonnegative")
    displacement = candidate - baseline
    distances = np.linalg.norm(displacement, axis=1)
    base_extent = np.ptp(baseline, axis=0)
    base_diag = float(np.linalg.norm(base_extent))
    extent = np.ptp(candidate, axis=0)
    threshold = max(base_diag * effect_threshold_relative, 1e-12)
    mask = distances > threshold
    top = np.argsort(-distances, kind="stable")[:20]
    mean_shift = displacement.mean(axis=0)
    report = {
        "vertex_count": len(baseline),
        "bbox": {
            "min": candidate.min(axis=0).tolist(),
            "max": candidate.max(axis=0).tolist(),
            "extent": extent.tolist(),
            "diagonal": float(np.linalg.norm(extent)),
            "extent_change": (extent - base_extent).tolist(),
        },
        "centroid": candidate.mean(axis=0).tolist(),
        "centroid_shift": mean_shift.tolist(),
        "centroid_shift_l2": float(np.linalg.norm(mean_shift)),
        "displacement": {
            "max": float(distances.max()),
            "mean": float(distances.mean()),
            "rms": float(np.sqrt(np.mean(distances**2))),
            "p50": float(np.quantile(distances, 0.5)),
            "p95": float(np.quantile(distances, 0.95)),
            "p99": float(np.quantile(distances, 0.99)),
        },
        "baseline_bbox_diagonal": base_diag,
        "effect_threshold_absolute": threshold,
        "effect_threshold_relative": effect_threshold_relative,
        "affected_vertex_count": int(mask.sum()),
        "affected_vertex_fraction": float(mask.mean()),
        "top_vertices": [
            {"vertex_index": int(index), "distance": float(distances[index])}
            for index in top
        ],
        "semantic_regions": "none unless supplied by source-matched verified vertex masks",
    }
    report["normalized_displacement"] = {
        key: value / base_diag if base_diag > 0 else None
        for key, value in report["displacement"].items()
    }
    return report, displacement, mask


def finite_difference_jacobian(baseline, minus, plus, step):
    baseline = vertices(baseline)
    minus, plus = np.asarray(minus, dtype=float), np.asarray(plus, dtype=float)
    if minus.shape != plus.shape or minus.shape[1:] != baseline.shape:
        raise ValueError(
            "Minus/plus must be controls×vertices×3 with matching topology"
        )
    if (
        not np.isfinite(minus).all()
        or not np.isfinite(plus).all()
        or not np.isfinite(step)
        or step <= 0
    ):
        raise ValueError("Finite probes and positive finite step are required")
    left = (baseline[None] - minus) / step
    right = (plus - baseline[None]) / step
    center = (plus - minus) / (2 * step)
    return (
        center.reshape(len(center), -1).T,
        left.reshape(len(left), -1).T,
        right.reshape(len(right), -1).T,
    )


def analyze_jacobian(jacobian, *, rank_relative_tolerance=1e-7):
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.ndim != 2 or not np.isfinite(jacobian).all():
        raise ValueError("Jacobian must be finite 2D")
    norms = np.linalg.norm(jacobian, axis=0)
    positive = norms > 1e-12
    unit = np.zeros_like(jacobian)
    unit[:, positive] = jacobian[:, positive] / norms[positive]
    coupling = np.clip(unit.T @ unit, -1, 1)
    _, singular, directions = np.linalg.svd(jacobian, full_matrices=False)
    rank_threshold = (
        float(singular[0] * rank_relative_tolerance) if len(singular) else 0
    )
    rank = int(np.sum(singular > max(rank_threshold, 1e-12)))
    pairs = [
        {"controls": [i, j], "cosine": float(coupling[i, j])}
        for i in range(len(norms))
        for j in range(i + 1, len(norms))
        if positive[i] and positive[j]
    ]
    pairs.sort(key=lambda pair: abs(pair["cosine"]), reverse=True)
    report = {
        "scope": "local surface sensitivity at a fixed baseline; not a global base-expressivity certificate",
        "column_norms": norms.tolist(),
        "zero_columns_at_this_baseline_and_mesh_set": np.flatnonzero(
            ~positive
        ).tolist(),
        "singular_values": singular.tolist(),
        "rank": rank,
        "rank_relative_tolerance": rank_relative_tolerance,
        "rank_absolute_threshold": rank_threshold,
        "nonzero_subspace_condition_number": float(singular[0] / singular[rank - 1])
        if rank
        else None,
        "strongest_couplings": pairs[:50],
        "coupling_definition": "cosine of geometric response columns; zero columns assigned0 and separately listed",
    }
    return report, coupling, directions


def validate_regions(specification, *, source_file_sha256, vertex_count):
    """Accept only externally supplied masks with explicit correspondence provenance."""
    if specification["source_file_sha256"] != source_file_sha256:
        raise ValueError("Semantic mask source hash does not match this cached mesh")
    provenance = specification.get("provenance", {})
    if (
        provenance.get("verified") is not True
        or not provenance.get("source")
        or provenance.get("kind")
        not in ["verified_vertex_mask", "verified_surface_correspondence"]
    ):
        raise ValueError(
            "Semantic labels require explicit verified mask/correspondence provenance"
        )
    regions = {}
    for name, indices in specification["regions"].items():
        if not name or not indices or any(type(index) is not int for index in indices):
            raise ValueError(
                "Semantic region needs a name and nonempty integer vertex indices"
            )
        if (
            len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= vertex_count
        ):
            raise ValueError("Semantic mask indices must be unique and in range")
        regions[name] = np.asarray(indices, dtype=int)
    return regions


def summarize_verified_regions(displacement, mask, regions):
    distances = np.linalg.norm(displacement, axis=1)
    return {
        name: {
            "vertex_count": len(indices),
            "max_displacement": float(distances[indices].max()),
            "rms_displacement": float(np.sqrt(np.mean(distances[indices] ** 2))),
            "affected_fraction": float(mask[indices].mean()),
        }
        for name, indices in regions.items()
    }
