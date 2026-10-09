"""Offline mesh-quality diagnostics for Unity Maker geometry exports.

The AABB sweep only generates candidate triangle pairs. Narrow-phase tests use
triangle/plane intersection intervals or coplanar polygon clipping; boxes alone
never count as intersections. Shared vertex-index adjacency is excluded.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class Thresholds:
    relative_area_epsilon: float = 1e-12
    relative_intersection_epsilon: float = 1e-7
    aspect_warning: float = 10.0
    min_edge_ratio: float = 0.5
    max_edge_ratio: float = 2.0
    min_area_ratio: float = 0.25
    max_area_ratio: float = 4.0
    normal_reversal_cosine: float = 0.0


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index):
        root = index
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[index] != index:
            next_index = self.parent[index]
            self.parent[index] = root
            index = next_index
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"finite_count": 0}
    return dict(zip(("min", "p01", "p50", "p95", "p99", "max"),
                    map(float, np.quantile(values, [0, 0.01, 0.5, 0.95, 0.99, 1])))) | {"finite_count": int(values.size)}


def triangle_arrays(vertices, faces):
    triangles = vertices[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, area2[:, None], out=np.zeros_like(cross), where=area2[:, None] > 0)
    return triangles, area2 / 2, normals


def plane_slice(triangle, signed_distances, epsilon):
    points = [point for point, distance in zip(triangle, signed_distances) if abs(distance) <= epsilon]
    for start, end in ((0, 1), (1, 2), (2, 0)):
        ds, de = signed_distances[start], signed_distances[end]
        if (ds > epsilon and de < -epsilon) or (ds < -epsilon and de > epsilon):
            fraction = ds / (ds - de)
            points.append(triangle[start] + fraction * (triangle[end] - triangle[start]))
    return np.asarray(points)


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def coplanar_intersection(a, b, normal, epsilon):
    """Clip one projected convex triangle by the other's three half-planes."""
    drop = int(np.argmax(np.abs(normal)))
    keep = [axis for axis in range(3) if axis != drop]
    subject = [point.copy() for point in a[:, keep]]
    clip = b[:, keep]
    orientation = np.sign(cross2(clip[1] - clip[0], clip[2] - clip[0]))
    if orientation == 0:
        return None
    for start, end in ((0, 1), (1, 2), (2, 0)):
        if not subject:
            return None
        origin, edge = clip[start], clip[end] - clip[start]
        tolerance = epsilon * np.linalg.norm(edge)
        previous = subject[-1]
        d_previous = orientation * cross2(edge, previous - origin)
        output = []
        for current in subject:
            d_current = orientation * cross2(edge, current - origin)
            was_inside, inside = d_previous >= -tolerance, d_current >= -tolerance
            if inside != was_inside:
                denominator = d_previous - d_current
                if abs(denominator) > 0:
                    fraction = np.clip(d_previous / denominator, 0.0, 1.0)
                    output.append(previous + fraction * (current - previous))
            if inside:
                output.append(current)
            previous, d_previous = current, d_current
        subject = output
    if not subject:
        return None
    polygon = np.asarray(subject)
    shifted = polygon - polygon[0]
    area2d = abs(sum(cross2(shifted[i], shifted[(i + 1) % len(shifted)])
                     for i in range(len(polygon)))) / 2
    # Area corrected for the dominant-axis projection, not interpreted as anatomy.
    area3d = area2d / abs(normal[drop])
    perimeter_bound = sum(np.linalg.norm(polygon[(i + 1) % len(polygon)] - polygon[i])
                          for i in range(len(polygon))) / abs(normal[drop])
    # A sub-tolerance sliver at a seam is a contact, not a certified positive-area crossing.
    area_tolerance = epsilon * perimeter_bound / 2
    return {"kind": "coplanar_overlap" if area3d > area_tolerance else "coplanar_contact",
            "overlap_area": float(area3d), "overlap_area_epsilon": float(area_tolerance)}


def triangle_intersection(a, b, epsilon):
    """Narrow-phase classification for nondegenerate 3-D triangles.

    epsilon is a length tolerance. A positive-area coplanar overlap and a
    noncoplanar interior crossing are separated from edge/point contacts.
    """
    na = np.cross(a[1] - a[0], a[2] - a[0])
    nb = np.cross(b[1] - b[0], b[2] - b[0])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    if la == 0 or lb == 0:
        return None
    na, nb = na / la, nb / lb
    da, db = (a - b[0]) @ nb, (b - a[0]) @ na
    if np.all(da > epsilon) or np.all(da < -epsilon) or np.all(db > epsilon) or np.all(db < -epsilon):
        return None
    line = np.cross(na, nb)
    line_length = np.linalg.norm(line)
    if line_length <= 1e-10:
        if np.max(np.abs(da)) <= epsilon and np.max(np.abs(db)) <= epsilon:
            return coplanar_intersection(a, b, na, epsilon)
        return None
    if np.max(np.abs(da)) <= epsilon and np.max(np.abs(db)) <= epsilon:
        # Nearly coplanar, including tiny triangles whose plane separation is below tolerance.
        return coplanar_intersection(a, b, na, epsilon)
    pa, pb = plane_slice(a, da, epsilon), plane_slice(b, db, epsilon)
    if not len(pa) or not len(pb):
        return None
    direction = line / line_length
    # Shift the scalar origin to reduce cancellation for meshes far from world origin.
    origin = (a[0] + b[0]) / 2
    ta, tb = (pa - origin) @ direction, (pb - origin) @ direction
    low, high = max(ta.min(), tb.min()), min(ta.max(), tb.max())
    if high < low - epsilon:
        return None
    cuts_a = np.any(da > epsilon) and np.any(da < -epsilon)
    cuts_b = np.any(db > epsilon) and np.any(db < -epsilon)
    if high - low <= epsilon:
        kind = "point_contact"
    elif cuts_a and cuts_b:
        kind = "proper_crossing"
    else:
        kind = "edge_contact"
    return {"kind": kind, "intersection_length": float(max(0, high - low))}


def self_intersections(vertices, faces, triangles, valid, epsilon, *, exclude_shared_vertex=True):
    start_time = time.perf_counter()
    mins, maxs = triangles.min(axis=1), triangles.max(axis=1)
    axis = int(np.argmax(np.ptp(vertices, axis=0)))
    order = np.flatnonzero(valid)
    order = order[np.argsort(mins[order, axis], kind="stable")]
    sorted_mins = mins[order, axis]
    pairs = []
    candidates = tested = adjacent = 0
    for position, first in enumerate(order):
        stop = int(np.searchsorted(sorted_mins, maxs[first, axis] + epsilon, side="right"))
        remaining = order[position + 1:stop]
        overlap = np.all(mins[remaining] <= maxs[first] + epsilon, axis=1) & np.all(maxs[remaining] >= mins[first] - epsilon, axis=1)
        remaining = remaining[overlap]
        candidates += len(remaining)
        if not len(remaining):
            continue
        shared = np.any(faces[remaining, :, None] == faces[first][None, None, :], axis=(1, 2))
        adjacent += int(shared.sum()) if exclude_shared_vertex else 0
        # Mother-template authoring also checks adjacent triangles: sharing an
        # edge/vertex normally gives a contact, but does not excuse a fold that
        # overlaps a positive area or crosses beyond that shared boundary.
        selected = remaining[~shared] if exclude_shared_vertex else remaining
        for second in selected:
            tested += 1
            intersection = triangle_intersection(triangles[first], triangles[second], epsilon)
            if intersection is not None:
                pairs.append({"triangles": [int(first), int(second)], **intersection})
    counts = {kind: sum(entry["kind"] == kind for entry in pairs)
              for kind in ("proper_crossing", "coplanar_overlap", "edge_contact", "point_contact", "coplanar_contact")}
    return {
        "algorithm": "AABB sweep broad phase; triangle-plane interval intersection and coplanar convex clipping narrow phase",
        "length_epsilon": float(epsilon), "sweep_axis": axis, "aabb_candidate_pairs": int(candidates),
        "excluded_shared_vertex_pairs": adjacent, "narrow_phase_pairs": tested,
        "exclude_shared_vertex": bool(exclude_shared_vertex),
        "excluded_degenerate_triangles": int((~valid).sum()), "counts": counts,
        "true_crossing_or_area_overlap_count": counts["proper_crossing"] + counts["coplanar_overlap"],
        "contact_count": counts["edge_contact"] + counts["point_contact"] + counts["coplanar_contact"],
        "pairs": pairs, "elapsed_seconds": time.perf_counter() - start_time,
    }


def topology_report(faces, vertex_count):
    edges = defaultdict(list)
    vertices = UnionFind(vertex_count)
    face_components = UnionFind(len(faces))
    duplicate_groups = defaultdict(list)
    incident_faces = defaultdict(list)
    for number, face in enumerate(faces):
        duplicate_groups[tuple(sorted(map(int, face)))].append(number)
        for vertex in set(map(int, face)):
            incident_faces[vertex].append(number)
        for start, end in ((0, 1), (1, 2), (2, 0)):
            a, b = int(face[start]), int(face[end])
            vertices.union(a, b)
            edges[tuple(sorted((a, b)))].append((number, a, b))
    boundary, nonmanifold, winding = [], [], []
    for edge, uses in edges.items():
        if len(uses) == 1:
            boundary.append(edge)
        if len(uses) > 2:
            nonmanifold.append(edge)
        if len(uses) == 2 and uses[0][1:] == uses[1][1:]:
            winding.append(edge)
        for use in uses[1:]:
            face_components.union(uses[0][0], use[0])
    used = np.unique(faces)
    vertex_groups, face_groups = defaultdict(int), defaultdict(int)
    for index in used:
        vertex_groups[vertices.find(int(index))] += 1
    for index in range(len(faces)):
        face_groups[face_components.find(index)] += 1
    root_to_component = {root: number for number, root in enumerate(face_groups)}
    face_component_ids = [root_to_component[face_components.find(index)] for index in range(len(faces))]
    nonmanifold_vertices = []
    for vertex, incident in incident_faces.items():
        link = defaultdict(list)
        malformed = False
        for number in incident:
            other = [int(index) for index in faces[number] if index != vertex]
            if len(other) != 2 or other[0] == other[1]:
                malformed = True
                continue
            link[other[0]].append(other[1])
            link[other[1]].append(other[0])
        seen = set()
        components = 0
        for node in link:
            if node in seen:
                continue
            components += 1
            pending = [node]
            while pending:
                current = pending.pop()
                if current in seen:
                    continue
                seen.add(current)
                pending.extend(link[current])
        degrees = [len(neighbors) for neighbors in link.values()]
        valid_cycle = degrees and all(degree == 2 for degree in degrees)
        valid_path = degrees.count(1) == 2 and all(degree in (1, 2) for degree in degrees)
        if malformed or components != 1 or not (valid_cycle or valid_path):
            nonmanifold_vertices.append({"vertex": vertex, "link_components": components,
                                         "link_degrees": degrees, "malformed_incident_face": malformed})
    return {
        "unique_edge_count": len(edges), "boundary_edge_count": len(boundary),
        "nonmanifold_edge_count": len(nonmanifold), "inconsistent_winding_edge_count": len(winding),
        "boundary_edges": [list(edge) for edge in boundary],
        "nonmanifold_edges": [list(edge) for edge in nonmanifold],
        "nonmanifold_vertex_count": len(nonmanifold_vertices), "nonmanifold_vertices": nonmanifold_vertices,
        "inconsistent_winding_edges": [list(edge) for edge in winding],
        "vertex_connected_components": len(vertex_groups), "component_vertex_counts": sorted(vertex_groups.values(), reverse=True),
        "edge_connected_face_components": len(face_groups), "component_face_counts": sorted(face_groups.values(), reverse=True),
        "face_component_ids": face_component_ids,
        "unused_vertex_count": int(vertex_count - len(used)),
        "duplicate_face_groups": [group for group in duplicate_groups.values() if len(group) > 1],
        "closed_edge_manifold": not boundary and not nonmanifold,
        "vertex_manifold_check_performed": True,
    }, np.asarray(list(edges), dtype=int)


def mesh_arrays(mesh, space):
    if space == "baked_raw":
        geometry = mesh["baked"]
    elif space in ("world_renderer", "world_scale_free"):
        geometry = mesh["baked"]["world_candidates"]["renderer_matrix" if space == "world_renderer" else "scale_free_trs"]
    else:
        raise ValueError(f"Unknown space: {space}")
    vertices = np.asarray(geometry["vertices"], dtype=float)
    flat = np.asarray(mesh["baked"]["triangles"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f"{mesh['mesh_name']}: invalid finite Nx3 vertices")
    if flat.ndim != 1 or len(flat) % 3 or not len(flat) or not np.issubdtype(flat.dtype, np.integer):
        raise ValueError(f"{mesh['mesh_name']}: invalid nonempty triangle indices")
    faces = flat.astype(np.int64).reshape(-1, 3)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{mesh['mesh_name']}: triangle index outside vertex range")
    normals = np.asarray(geometry.get("normals", []), dtype=float)
    if normals.size and (normals.shape != vertices.shape or not np.isfinite(normals).all()):
        raise ValueError(f"{mesh['mesh_name']}: invalid vertex normals")
    return vertices, faces, normals


def bone_determinants(mesh, transforms):
    rows = []
    for name, key in zip(mesh.get("bone_names", []), mesh.get("bone_transform_ids", [])):
        if key is None:
            rows.append({"name": name, "transform_id": key, "status": "null_bone"})
            continue
        if key not in transforms:
            raise ValueError(f"Missing actual world transform for bone {key}")
        matrix = np.asarray(transforms[key]["local_to_world"], dtype=float).reshape(4, 4)
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite world matrix for bone {key}")
        determinant = float(np.linalg.det(matrix[:3, :3]))
        rows.append({"name": name, "transform_id": key, "world_determinant": determinant,
                     "negative": determinant < 0, "near_singular": abs(determinant) <= 1e-12})
    return {"negative_count": sum(row.get("negative", False) for row in rows),
            "near_singular_count": sum(row.get("near_singular", False) for row in rows), "bones": rows}


def analyze_mesh(mesh, transforms, thresholds, space="baked_raw"):
    vertices, faces, vertex_normals = mesh_arrays(mesh, space)
    triangles, areas, face_normals = triangle_arrays(vertices, faces)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if diagonal == 0:
        raise ValueError(f"{mesh['mesh_name']}: whole mesh is collapsed")
    area_epsilon = thresholds.relative_area_epsilon * diagonal * diagonal
    epsilon = thresholds.relative_intersection_epsilon * diagonal
    valid = areas > area_epsilon
    edge_lengths = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    aspect = np.divide(np.sqrt(3) * edge_lengths.max(axis=1) ** 2, 4 * areas,
                       out=np.full_like(areas, np.inf), where=valid)
    topology, unique_edges = topology_report(faces, len(vertices))
    normals_report = {"available": bool(vertex_normals.size), "convention": "np.cross(v1-v0,v2-v0); source winding/handedness not assumed outward"}
    if vertex_normals.size:
        averaged = vertex_normals[faces].mean(axis=1)
        lengths = np.linalg.norm(averaged, axis=1)
        direction = np.divide(averaged, lengths[:, None], out=np.zeros_like(averaged), where=lengths[:, None] > 0)
        dots = np.einsum("ij,ij->i", direction, face_normals)
        normals_report.update({"dot_distribution": quantiles(dots[valid & (lengths > 0)]),
                               "opposed_triangle_ids": np.flatnonzero(valid & (lengths > 0) & (dots < 0)).tolist(),
                               "zero_average_normal_triangle_ids": np.flatnonzero(lengths == 0).tolist()})
    intersections = self_intersections(vertices, faces, triangles, valid, epsilon)
    component_ids = topology["face_component_ids"]
    for pair in intersections["pairs"]:
        pair["component_ids"] = [component_ids[index] for index in pair["triangles"]]
    crossing_pairs = [pair for pair in intersections["pairs"] if pair["kind"] in ("proper_crossing", "coplanar_overlap")]
    intersections["within_component_crossing_count"] = sum(pair["component_ids"][0] == pair["component_ids"][1] for pair in crossing_pairs)
    intersections["between_component_crossing_count"] = sum(pair["component_ids"][0] != pair["component_ids"][1] for pair in crossing_pairs)
    topology["component_bounds"] = []
    for component in sorted(set(component_ids)):
        indices = np.unique(faces[np.asarray(component_ids) == component])
        topology["component_bounds"].append({"component_id": component,
                                              "min": vertices[indices].min(axis=0).tolist(),
                                              "max": vertices[indices].max(axis=0).tolist(),
                                              "mean_vertex_position": vertices[indices].mean(axis=0).tolist()})
    return {
        "mesh_name": mesh["mesh_name"], "renderer_path": mesh.get("renderer_path"),
        "source_geometry_sha256": mesh.get("source_geometry_sha256"), "space": space,
        "active_in_hierarchy": mesh.get("active_in_hierarchy"), "enabled": mesh.get("enabled"),
        "vertex_count": len(vertices), "triangle_count": len(faces), "bounds_diagonal": diagonal,
        "area_epsilon": area_epsilon, "area_distribution": quantiles(areas),
        "total_area": float(areas.sum()), "degenerate_triangle_ids": np.flatnonzero(~valid).tolist(),
        "aspect_definition": "sqrt(3)*longest_edge_squared/(4*triangle_area); equilateral=1",
        "aspect_distribution": quantiles(aspect), "infinite_aspect_count": int((~np.isfinite(aspect)).sum()),
        "aspect_warning_triangle_ids": np.flatnonzero(aspect > thresholds.aspect_warning).tolist(),
        "topology": topology, "normal_direction": normals_report,
        "bone_world_determinants": bone_determinants(mesh, transforms), "self_intersections": intersections,
    }, (vertices, faces, face_normals, areas, unique_edges)


def baseline_comparison(current, previous, arrays, base_arrays, thresholds):
    vertices, faces, _, areas, edges = arrays
    base_vertices, base_faces, base_normals, base_areas, _ = base_arrays
    if vertices.shape != base_vertices.shape or not np.array_equal(faces, base_faces):
        return {"status": "topology_mismatch", "candidate_vertices": len(vertices), "baseline_vertices": len(base_vertices),
                "candidate_triangles": len(faces), "baseline_triangles": len(base_faces),
                "deformation_comparison_performed": False}
    # Remove global rotation and translation, preserving any actual global scale change.
    centered, base_centered = vertices - vertices.mean(axis=0), base_vertices - base_vertices.mean(axis=0)
    u, _, vt = np.linalg.svd(centered.T @ base_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    aligned = centered @ rotation + base_vertices.mean(axis=0)
    _, _, aligned_normals = triangle_arrays(aligned, faces)
    cosine = np.einsum("ij,ij->i", aligned_normals, base_normals)
    diagonal = float(np.linalg.norm(np.ptp(base_vertices, axis=0)))
    area_valid = base_areas > thresholds.relative_area_epsilon * diagonal * diagonal
    current_valid = areas > thresholds.relative_area_epsilon * diagonal * diagonal
    base_lengths = np.linalg.norm(base_vertices[edges[:, 1]] - base_vertices[edges[:, 0]], axis=1)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    edge_valid = base_lengths > thresholds.relative_intersection_epsilon * diagonal
    edge_ratio = np.divide(lengths, base_lengths, out=np.full_like(lengths, np.nan), where=edge_valid)
    area_ratio = np.divide(areas, base_areas, out=np.full_like(areas, np.nan), where=area_valid)
    unusual_edges = np.flatnonzero(edge_valid & ((edge_ratio < thresholds.min_edge_ratio) | (edge_ratio > thresholds.max_edge_ratio)))
    unusual_areas = np.flatnonzero(area_valid & ((area_ratio < thresholds.min_area_ratio) | (area_ratio > thresholds.max_area_ratio)))
    reversals = np.flatnonzero(area_valid & current_valid & (cosine < thresholds.normal_reversal_cosine))
    current_crossings = {tuple(sorted(pair["triangles"])) for pair in current["self_intersections"]["pairs"]
                        if pair["kind"] in ("proper_crossing", "coplanar_overlap")}
    base_crossings = {tuple(sorted(pair["triangles"])) for pair in previous["self_intersections"]["pairs"]
                     if pair["kind"] in ("proper_crossing", "coplanar_overlap")}
    return {
        "status": "topology_matched", "deformation_comparison_performed": True,
        "source_hash_identical": (current["source_geometry_sha256"] == previous["source_geometry_sha256"]
                                  if current["source_geometry_sha256"] and previous["source_geometry_sha256"] else None),
        "correspondence_caveat": "Matching indices/topology is assumed; changed source asset hashes require separate vertex-semantic validation",
        "alignment": "proper rigid Kabsch, translation+rotation only; scale and reflection retained",
        "candidate_to_baseline_rotation_row_vectors": rotation.tolist(),
        "displacement_after_alignment_distribution": quantiles(np.linalg.norm(aligned - base_vertices, axis=1)),
        "normal_cosine_after_alignment_distribution": quantiles(cosine[area_valid & current_valid]),
        "normal_reversal_triangle_ids": reversals.tolist(),
        "normal_reversal_caveat": "Direction reversal relative to baseline after global alignment; not proof of a volumetric element inversion",
        "edge_ratio_distribution": quantiles(edge_ratio),
        "unusual_edges": [{"vertices": edges[index].tolist(), "ratio": float(edge_ratio[index])} for index in unusual_edges],
        "undefined_baseline_edge_count": int((~edge_valid).sum()),
        "area_ratio_distribution": quantiles(area_ratio),
        "unusual_area_triangles": [{"triangle": int(index), "ratio": float(area_ratio[index])} for index in unusual_areas],
        "undefined_baseline_area_count": int((~area_valid).sum()),
        "new_crossing_pairs": [list(pair) for pair in sorted(current_crossings - base_crossings)],
        "resolved_crossing_pairs": [list(pair) for pair in sorted(base_crossings - current_crossings)],
    }


def choose_baseline(mesh, baseline_meshes):
    exact = [candidate for candidate in baseline_meshes if candidate["mesh_name"] == mesh["mesh_name"]
             and candidate.get("renderer_path") == mesh.get("renderer_path")]
    if len(exact) == 1:
        return exact[0], "exact_renderer_path"
    same_name = [candidate for candidate in baseline_meshes if candidate["mesh_name"] == mesh["mesh_name"]]
    if len(same_name) == 1:
        return same_name[0], "unique_mesh_name_fallback"
    return None, "missing" if not same_name else "ambiguous"


def analyze_snapshot(snapshot, baseline=None, thresholds=None, space="baked_raw", mesh_names=None, visible_only=False,
                     baseline_cache=None):
    if snapshot.get("schema_version") != 1 or snapshot.get("snapshot_kind") != "maker_live_skinned_geometry":
        raise ValueError("Expected MakerGeometryService schema 1 JSON")
    if baseline is not None and (baseline.get("schema_version") != 1 or baseline.get("snapshot_kind") != "maker_live_skinned_geometry"):
        raise ValueError("Expected MakerGeometryService schema 1 baseline JSON")
    thresholds = thresholds or Thresholds()
    transforms = {entry["id"]: entry for entry in snapshot["transforms"]}
    base_transforms = {entry["id"]: entry for entry in baseline["transforms"]} if baseline else {}
    entries = [entry for entry in snapshot["meshes"] if (mesh_names is None or entry["mesh_name"] in mesh_names)
               and (not visible_only or (entry.get("enabled") and entry.get("active_in_hierarchy")))]
    if not entries:
        raise ValueError("No selected meshes in snapshot")
    reports = []
    baseline_cache = {} if baseline_cache is None else baseline_cache
    for entry in entries:
        report, arrays = analyze_mesh(entry, transforms, thresholds, space)
        if baseline:
            matched, match_policy = choose_baseline(entry, baseline["meshes"])
            if matched is None:
                report["baseline"] = {"status": "unmatched_" + match_policy, "deformation_comparison_performed": False}
            else:
                cache_key = (id(matched), thresholds, space)
                if cache_key not in baseline_cache:
                    baseline_cache[cache_key] = analyze_mesh(matched, base_transforms, thresholds, space)
                base_report, base_arrays = baseline_cache[cache_key]
                report["baseline"] = baseline_comparison(report, base_report, arrays, base_arrays, thresholds)
                report["baseline"]["match_policy"] = match_policy
        reports.append(report)
    return {
        "schema_version": 1, "report_kind": "hs2_mesh_quality", "space": space,
        "thresholds": asdict(thresholds), "mesh_count": len(reports), "meshes": reports,
        "selection_policy": "enabled and active_in_hierarchy only" if visible_only else "all selected renderers; visibility states preserved separately",
        "limitations": [
            "Geometry diagnostics only; no identity similarity, anatomical calibration or aesthetic score.",
            "World candidates are not certified by this tool; default uses raw BakeMesh per renderer.",
            "Intersections are per mesh, not between meshes; eye/head or clothing/body contact requires a shared certified world frame.",
            "Triangles sharing a vertex index are excluded from self-intersection tests, even if an adjacent fold intersects beyond that vertex.",
            "Degenerate triangles are reported separately and excluded from narrow-phase intersection predicates.",
            "Floating-point predicates use the reported length tolerance; extremely close or near-coplanar cases can be tolerance-sensitive.",
            "Open boundaries, UV seams, slender triangles and reflected bone frames can be intentional; flags require baseline and asset context.",
            "Topological link graphs are checked, but watertight volume, geometric seam equivalence and full skinning validity are not certified."
        ]
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--space", choices=("baked_raw", "world_renderer", "world_scale_free"), default="baked_raw")
    parser.add_argument("--meshes", nargs="+")
    parser.add_argument("--visible-only", action="store_true", help="Select enabled AND active renderers; this does not test camera occlusion")
    for name, default in asdict(Thresholds()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args()
    parameters = {name: getattr(args, name) for name in asdict(Thresholds())}
    if any(not np.isfinite(value) for value in parameters.values()):
        parser.error("All thresholds must be finite")
    if any(parameters[name] <= 0 for name in parameters if name != "normal_reversal_cosine"):
        parser.error("Length/area/aspect/stretch thresholds must be positive")
    if not -1 <= parameters["normal_reversal_cosine"] <= 1:
        parser.error("normal-reversal-cosine must be in [-1,1]")
    if parameters["min_edge_ratio"] >= parameters["max_edge_ratio"] or parameters["min_area_ratio"] >= parameters["max_area_ratio"]:
        parser.error("Minimum ratios must be below maximum ratios")
    contents = args.snapshot.read_bytes()
    baseline_contents = args.baseline.read_bytes() if args.baseline else None
    report = analyze_snapshot(json.loads(contents), json.loads(baseline_contents) if baseline_contents else None,
                              Thresholds(**parameters), args.space, set(args.meshes) if args.meshes else None, args.visible_only)
    report["input_sha256"] = hashlib.sha256(contents).hexdigest()
    report["baseline_sha256"] = hashlib.sha256(baseline_contents).hexdigest() if baseline_contents else None
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(json.dumps({"out": str(args.out), "mesh_count": report["mesh_count"],
                          "summary": [{"mesh": entry["mesh_name"], "degenerate": len(entry["degenerate_triangle_ids"]),
                                       "renderer_path": entry["renderer_path"], "enabled": entry["enabled"],
                                       "active_in_hierarchy": entry["active_in_hierarchy"],
                                       "true_selfintersections": entry["self_intersections"]["true_crossing_or_area_overlap_count"],
                                       "contacts": entry["self_intersections"]["contact_count"],
                                       "baseline_status": entry.get("baseline", {}).get("status")}
                                      for entry in report["meshes"]]}, ensure_ascii=False, indent=2))
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
