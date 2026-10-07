"""Certified visible-head cross-mesh triangle diagnostics, strictly offline.

Requires a separate snapshot-local LBS certificate for the exact file SHA.
Raw BakeMesh coordinates from different renderers are never mixed directly.
Intersections are measured facts, not automatic anatomical defects.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mesh_quality import Thresholds, triangle_arrays, triangle_intersection


class CertificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise CertificationError(message)


def identity(mesh):
    return mesh["mesh_name"], mesh["renderer_path"]


def has_ancestor(key, ancestor, transforms):
    visited = set()
    while key is not None:
        require(key not in visited, "Cycle in renderer ancestry")
        require(key in transforms, "Incomplete renderer ancestry")
        if key == ancestor:
            return True
        visited.add(key)
        key = transforms[key]["parent_id"]
    return False


def finite_nonnegative(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and np.isfinite(value) and value >= 0, "Invalid numerical certificate field: " + label)
    return float(value)


@dataclass
class CertifiedMesh:
    name: str
    path: str
    source_hash: str
    candidate: str
    vertices: np.ndarray
    faces: np.ndarray
    residual_floor: float


def load_certified_head(snapshot, snapshot_sha256, certificate, maximum_tolerance=1e-5):
    require(snapshot.get("schema_version") == 1 and snapshot.get("snapshot_kind") == "maker_live_skinned_geometry",
            "Expected Unity geometry schema 1")
    require(certificate.get("schema_version") == 1, "Expected LBS certificate schema 1")
    require(certificate.get("snapshot_sha256") == snapshot_sha256, "Certificate snapshot SHA-256 does not match the exact input file")
    require(certificate.get("scope") == "snapshot-local LBS versus BakeMesh; no universal engine convention inferred",
            "Expected independent snapshot-local LBS certificate scope")
    require(certificate.get("snapshot_frame_stable") is True and snapshot.get("frame_count") == snapshot.get("frame_count_end"),
            "Snapshot/certificate is not frame-stable")
    tolerance = finite_nonnegative(certificate.get("normalized_tolerance"), "normalized_tolerance")
    require(0 < tolerance <= maximum_tolerance, "Certificate uses an overly loose or zero normalized tolerance")
    transforms = {row["id"]: row for row in snapshot["transforms"]}
    require(len(transforms) == len(snapshot["transforms"]), "Duplicate transform IDs")
    head_root = snapshot["character"].get("head_root_transform_id")
    require(head_root is not None and head_root in transforms, "A recorded head-root subtree is required")
    certificate_rows = {}
    for row in certificate.get("meshes", []):
        key = identity(row)
        require(key not in certificate_rows, "Ambiguous duplicate renderer identity in certificate")
        certificate_rows[key] = row
    selected, skipped, seen = [], [], set()
    for mesh in snapshot["meshes"]:
        key = identity(mesh)
        require(key not in seen, "Duplicate renderer identity in snapshot; names cannot overwrite one another")
        seen.add(key)
        if not mesh.get("enabled") or not mesh.get("active_in_hierarchy"):
            skipped.append({"mesh_name": key[0], "renderer_path": key[1], "reason": "disabled_or_inactive"})
            continue
        if not has_ancestor(mesh["renderer_transform_id"], head_root, transforms):
            skipped.append({"mesh_name": key[0], "renderer_path": key[1], "reason": "outside_character_head_subtree"})
            continue
        require(key in certificate_rows, "Missing certificate for visible head renderer " + key[1])
        row = certificate_rows[key]
        source_hash = mesh.get("source_geometry_sha256")
        require(isinstance(source_hash, str) and len(source_hash) == 64 and row.get("source_geometry_sha256") == source_hash,
                "Certificate source hash mismatch for " + key[1])
        require(row.get("certified") is True, "Uncertified visible head renderer " + key[1])
        require(row.get("influences_overridden") is False and row.get("influences") == row.get("declared_influences"),
                "Overridden skin-quality hypotheses are not accepted as actual render certification")
        require(row.get("renderer_visibility", {}).get("enabled") == mesh.get("enabled")
                and row.get("renderer_visibility", {}).get("active_in_hierarchy") == mesh.get("active_in_hierarchy"),
                "Certificate renderer visibility mismatch")
        candidate = row.get("selected_candidate")
        require(candidate in ("renderer_matrix", "scale_free_trs")
                and row.get("matching_candidates") == [candidate], "World candidate is missing or ambiguous")
        errors = row.get("candidate_errors", {}).get(candidate)
        require(isinstance(errors, dict), "Missing selected-candidate numerical certificate")
        normalized = finite_nonnegative(errors.get("max_normalized"), "max_normalized")
        require(normalized <= tolerance, "Selected-candidate LBS error exceeds certificate tolerance")
        world_consistency = errors.get("exported_world_consistency")
        require(isinstance(world_consistency, dict), "Missing exported-world consistency certificate")
        consistency_normalized = finite_nonnegative(world_consistency.get("max_normalized"), "exported_world_consistency.max_normalized")
        require(consistency_normalized <= tolerance, "Recorded world candidate fails matrix consistency")
        raw = np.asarray(mesh["baked"]["vertices"], dtype=float)
        require(raw.ndim == 2 and raw.shape[1] == 3 and len(raw) and np.isfinite(raw).all(), "Invalid raw baked vertices")
        require(errors.get("vertex_count") == len(raw) and world_consistency.get("vertex_count") == len(raw), "Certificate vertex count mismatch")
        candidate_data = mesh["baked"]["world_candidates"].get(candidate)
        require(isinstance(candidate_data, dict), "Certified world candidate missing from geometry")
        matrix = np.asarray(candidate_data["matrix"], dtype=float)
        require(matrix.shape == (16,) and np.isfinite(matrix).all(), "Invalid candidate matrix")
        matrix = matrix.reshape(4, 4)
        require(abs(np.linalg.det(matrix[:3, :3])) > 1e-12, "Singular world candidate cannot define a reliable cross-mesh frame")
        world = raw @ matrix[:3, :3].T + matrix[:3, 3]
        flat = np.asarray(mesh["baked"]["triangles"])
        require(flat.ndim == 1 and flat.size and flat.size % 3 == 0 and np.issubdtype(flat.dtype, np.integer), "Invalid triangle indices")
        faces = flat.astype(np.int64).reshape(-1, 3)
        require(faces.min() >= 0 and faces.max() < len(world), "Triangle index outside mesh vertex range")
        numerical_floor = finite_nonnegative(errors.get("max_l2"), "max_l2")
        numerical_floor += finite_nonnegative(world_consistency.get("max_l2"), "exported_world_consistency.max_l2")
        selected.append(CertifiedMesh(key[0], key[1], source_hash, candidate, world, faces, numerical_floor))
    require(len(selected) >= 2, "At least two certified enabled head-subtree renderers are required")
    return selected, skipped


def cross_intersections(meshes, thresholds=None):
    thresholds = thresholds or Thresholds()
    began = time.perf_counter()
    triangles, owners, local_indices, valid = [], [], [], []
    for number, mesh in enumerate(meshes):
        tris, areas, _ = triangle_arrays(mesh.vertices, mesh.faces)
        diagonal = np.linalg.norm(np.ptp(mesh.vertices, axis=0))
        triangles.append(tris)
        owners.append(np.full(len(tris), number))
        local_indices.append(np.arange(len(tris)))
        valid.append(areas > thresholds.relative_area_epsilon * diagonal * diagonal)
    triangles = np.concatenate(triangles)
    owners, local_indices, valid = np.concatenate(owners), np.concatenate(local_indices), np.concatenate(valid)
    mins, maxs = triangles.min(axis=1), triangles.max(axis=1)
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    base_epsilon = thresholds.relative_intersection_epsilon * diagonal
    broad_epsilon = base_epsilon + 2 * max(mesh.residual_floor for mesh in meshes)
    axis = int(np.argmax(np.ptp(vertices, axis=0)))
    order = np.flatnonzero(valid)
    order = order[np.argsort(mins[order, axis], kind="stable")]
    sorted_mins = mins[order, axis]
    records, aabb_count, narrow_count = [], 0, 0
    for position, first in enumerate(order):
        stop = int(np.searchsorted(sorted_mins, maxs[first, axis] + broad_epsilon, side="right"))
        second_ids = order[position + 1:stop]
        keep = (owners[second_ids] != owners[first]) & np.all(mins[second_ids] <= maxs[first] + broad_epsilon, axis=1)
        keep &= np.all(maxs[second_ids] >= mins[first] - broad_epsilon, axis=1)
        second_ids = second_ids[keep]
        aabb_count += len(second_ids)
        for second in second_ids:
            mesh_a, mesh_b = meshes[owners[first]], meshes[owners[second]]
            epsilon = base_epsilon + mesh_a.residual_floor + mesh_b.residual_floor
            narrow_count += 1
            result = triangle_intersection(triangles[first], triangles[second], epsilon)
            if result is None:
                continue
            left = {"mesh_name": mesh_a.name, "renderer_path": mesh_a.path, "triangle": int(local_indices[first])}
            right = {"mesh_name": mesh_b.name, "renderer_path": mesh_b.path, "triangle": int(local_indices[second])}
            if (left["renderer_path"], left["mesh_name"]) > (right["renderer_path"], right["mesh_name"]):
                left, right = right, left
            records.append({"left": left, "right": right, "length_epsilon": epsilon, **result})
    counts = {kind: sum(record["kind"] == kind for record in records)
              for kind in ("proper_crossing", "coplanar_overlap", "edge_contact", "point_contact", "coplanar_contact")}
    by_pair = {}
    for record in records:
        key = (record["left"]["renderer_path"], record["right"]["renderer_path"])
        if key not in by_pair:
            by_pair[key] = {"left": {k: v for k, v in record["left"].items() if k != "triangle"},
                            "right": {k: v for k, v in record["right"].items() if k != "triangle"},
                            "crossings": 0, "contacts": 0}
        by_pair[key]["crossings" if record["kind"] in ("proper_crossing", "coplanar_overlap") else "contacts"] += 1
    return {"counts": counts, "crossing_count": counts["proper_crossing"] + counts["coplanar_overlap"],
            "contact_count": counts["edge_contact"] + counts["point_contact"] + counts["coplanar_contact"],
            "aabb_candidate_pairs": int(aabb_count), "narrow_phase_pairs": narrow_count,
            "excluded_degenerate_triangles": int((~valid).sum()), "base_length_epsilon": base_epsilon,
            "broad_phase_length_epsilon": broad_epsilon, "pairs": records, "renderer_pair_summary": list(by_pair.values()),
            "elapsed_seconds": time.perf_counter() - began,
            "interpretation": "Cross-mesh surface intersections and contacts; no automatic anatomical defect label"}


def pair_key(record):
    return tuple((record[side]["mesh_name"], record[side]["renderer_path"], record[side]["triangle"]) for side in ("left", "right"))


def compare_crossings(current_meshes, baseline_meshes, current, baseline):
    base_map = {(mesh.name, mesh.path): mesh for mesh in baseline_meshes}
    current_map = {(mesh.name, mesh.path): mesh for mesh in current_meshes}
    if set(base_map) != set(current_map):
        return {"status": "renderer_selection_mismatch", "comparison_performed": False,
                "candidate_only": [list(key) for key in sorted(set(current_map) - set(base_map))],
                "baseline_only": [list(key) for key in sorted(set(base_map) - set(current_map))]}
    mismatch = []
    for key, mesh in current_map.items():
        base = base_map[key]
        if mesh.source_hash != base.source_hash or mesh.vertices.shape != base.vertices.shape or not np.array_equal(mesh.faces, base.faces):
            mismatch.append(list(key))
    if mismatch:
        return {"status": "source_or_topology_mismatch", "comparison_performed": False, "mismatched_renderers": mismatch}
    current_pairs = {pair_key(record): record for record in current["pairs"] if record["kind"] in ("proper_crossing", "coplanar_overlap")}
    base_pairs = {pair_key(record): record for record in baseline["pairs"] if record["kind"] in ("proper_crossing", "coplanar_overlap")}
    return {"status": "source_and_topology_matched", "comparison_performed": True,
            "new_crossing_pairs": [current_pairs[key] for key in sorted(set(current_pairs) - set(base_pairs))],
            "resolved_crossing_pairs": [base_pairs[key] for key in sorted(set(base_pairs) - set(current_pairs))],
            "unchanged_crossing_pair_count": len(set(current_pairs) & set(base_pairs)),
            "interpretation": "New indexed crossing pairs under the stated tolerances, not an automatic defect or identity verdict"}


def analyze_certified(snapshot, digest, certificate, baseline=None, baseline_digest=None, baseline_certificate=None,
                      maximum_tolerance=1e-5):
    meshes, skipped = load_certified_head(snapshot, digest, certificate, maximum_tolerance)
    report = {"schema_version": 1, "report_kind": "hs2_certified_cross_mesh_quality", "snapshot_sha256": digest,
              "selection": "enabled + active_in_hierarchy + recorded head-root subtree", "skipped_renderers": skipped,
              "meshes": [{"mesh_name": mesh.name, "renderer_path": mesh.path, "source_geometry_sha256": mesh.source_hash,
                          "selected_candidate": mesh.candidate, "numerical_residual_floor": mesh.residual_floor,
                          "vertex_count": len(mesh.vertices), "triangle_count": len(mesh.faces)} for mesh in meshes]}
    report["cross_mesh"] = cross_intersections(meshes)
    if baseline is not None:
        require(baseline_digest is not None and baseline_certificate is not None, "Baseline needs its own exact-file certificate")
        base_meshes, _ = load_certified_head(baseline, baseline_digest, baseline_certificate, maximum_tolerance)
        base_report = cross_intersections(base_meshes)
        report["baseline_sha256"] = baseline_digest
        report["baseline_cross_mesh"] = base_report
        report["baseline_comparison"] = compare_crossings(meshes, base_meshes, report["cross_mesh"], base_report)
    report["limitations"] = [
        "Certificates are consumed as separate supplied LBS evidence, not signed security attestations.",
        "Only snapshot-local candidates are certified; conventions are not extrapolated to other files or renderers.",
        "Enabled/active head-subtree selection does not determine shader transparency, visual occlusion or anatomical regions.",
        "Eye/head, mouth/teeth and other component intersections can be intentional; crossing counts do not label defects.",
        "Per-pair length tolerance includes reported LBS and world serialization residuals, not an anatomical uncertainty interval.",
        "Nearly coplanar or sub-tolerance crossings may be classified as contacts; no identity/likeness metric is computed."
    ]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-certificate", type=Path)
    parser.add_argument("--max-certification-tolerance", type=float, default=1e-5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.baseline and not args.baseline_certificate:
        parser.error("--baseline requires --baseline-certificate")
    require(np.isfinite(args.max_certification_tolerance) and args.max_certification_tolerance > 0,
            "Maximum certificate tolerance must be finite and positive")
    contents = args.snapshot.read_bytes()
    certificate_bytes = args.certificate.read_bytes()
    base_contents = args.baseline.read_bytes() if args.baseline else None
    base_certificate_bytes = args.baseline_certificate.read_bytes() if args.baseline else None
    report = analyze_certified(json.loads(contents), hashlib.sha256(contents).hexdigest(), json.loads(certificate_bytes),
                               json.loads(base_contents) if base_contents else None,
                               hashlib.sha256(base_contents).hexdigest() if base_contents else None,
                               json.loads(base_certificate_bytes) if base_certificate_bytes else None,
                               args.max_certification_tolerance)
    report["certificate_sha256"] = hashlib.sha256(certificate_bytes).hexdigest()
    report["baseline_certificate_sha256"] = hashlib.sha256(base_certificate_bytes).hexdigest() if base_certificate_bytes else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    comparison = report.get("baseline_comparison", {})
    print(json.dumps({"out": str(args.out), "meshes": len(report["meshes"]),
                      "crossings": report["cross_mesh"]["crossing_count"], "contacts": report["cross_mesh"]["contact_count"],
                      "baseline_status": comparison.get("status"),
                      "new_crossings": len(comparison.get("new_crossing_pairs", [])),
                      "resolved_crossings": len(comparison.get("resolved_crossing_pairs", []))}, indent=2))


if __name__ == "__main__":
    main()
