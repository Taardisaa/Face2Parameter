"""Project actual exported mesh samples and ray-test availability diagnostically.

Samples are synthetic mesh centroids, not detector/anatomical landmarks. A
same-camera roundtrip cannot validate PNG orientation or surface semantics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .calibrate import digest, read_json
    from .core import Camera, ContractError, meshes_from_geometry, observe, validate_reprojection
else:
    from calibrate import digest, read_json
    from core import Camera, ContractError, meshes_from_geometry, observe, validate_reprojection


def run(capture_path, geometry_path, out_dir, *, sample_count=24, parity_report_path=None):
    capture = read_json(capture_path)
    capture = capture.get("bridge_result", capture)
    views = capture.get("views") or [capture]
    geometry = read_json(geometry_path)
    geometry_sha = digest(geometry_path)
    parity = read_json(parity_report_path) if parity_report_path else None
    if parity is not None and parity.get("snapshot_sha256") != geometry_sha:
        raise ContractError("Independent parity report refers to a different geometry snapshot")
    cameras = []
    for view in views:
        pair = view.get("paired_geometry") or {}
        if pair.get("sha256") and pair["sha256"].lower() != geometry_sha:
            raise ContractError("Paired geometry file mismatch")
        cameras.append(Camera.from_capture(view, diagnostic=True))
    report = {
        "schema_version": 1, "evidence_kind": "real_geometry_synthetic_mesh_projection_diagnostic",
        "capture_file_sha256": digest(capture_path), "geometry_file_sha256": geometry_sha,
        "pixel_contract_validated": False, "anatomical_correspondence_validated": False,
        "paired_pose_validated_views": sum(camera.pose_pairing_validated for camera in cameras),
        "candidate_reports": {},
        "independent_parity_report": str(parity_report_path.resolve()) if parity_report_path else None,
        "limits": ["Centroids and heldout projections are synthetic mesh-derived points, not independent image observations.",
                   "Roundtrip precision does not establish actual PNG orientation, shader visibility or anatomical truth.",
                   "World policy can be certified only by the separate snapshot-bound skinning report, never by these ray/projection roundtrips."],
    }
    for candidate in ("renderer_matrix", "scale_free_trs"):
        certificate = None
        if parity is not None:
            certified_hashes = {
                entry["renderer_path"]: entry["source_geometry_sha256"]
                for entry in parity["meshes"]
                if entry.get("certified") is True and candidate in entry.get("matching_candidates", [])
            }
            certificate = {
                "evidence_id": str(parity_report_path.resolve()) + "#sha256=" + digest(parity_report_path),
                "capture_file_sha256": digest(capture_path), "geometry_file_sha256": geometry_sha,
                "scope": "snapshot_local_world_skinning_policy_only",
                "pixel_contract_validated": False, "world_policy_validated": True,
                "world_candidate": candidate, "mesh_source_hashes": certified_hashes,
            }
        meshes = meshes_from_geometry(geometry, candidate=candidate, certification=certificate, diagnostic=True)
        selected = next((m for m in meshes if m.visible and any(entry["renderer_path"] == m.renderer_path and entry["mesh_name"] == "o_head" for entry in geometry["meshes"])), None)
        if selected is None or not len(selected.triangles):
            raise ContractError("No visible triangle head mesh")
        indices = np.linspace(0, len(selected.triangles) - 1, min(sample_count, len(selected.triangles)), dtype=int)
        candidate_report = {"world_policy_certified": selected.world_policy_certified, "renderer_path": selected.renderer_path, "views": []}
        if certificate is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            cert_path = out_dir / f"{candidate}_world_certificate.json"
            cert_path.write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            candidate_report["certificate_path"] = str(cert_path.resolve())
        anchors = []
        for view_index, camera in enumerate(cameras):
            records = []
            for triangle in indices:
                world = selected.vertices[selected.triangles[triangle]].mean(axis=0)
                projection = camera.project(world)
                record = {"synthetic_triangle_id": int(triangle), "synthetic_world": world.tolist(), "projection": projection}
                if projection["status"] == "in_view":
                    observation = observe(camera, meshes, {"id": f"mesh_centroid_{triangle}", "xy": projection["xy"], "allowed_renderer_paths": [selected.renderer_path]})
                    record["ray_observation"] = observation
                    surface = observation.get("surface")
                    if surface:
                        record["distance_to_synthetic_point"] = float(np.linalg.norm(np.asarray(surface["world"]) - world))
                        projected_hit = camera.project(surface["world"])
                        record["algebraic_roundtrip_px"] = float(np.linalg.norm(np.asarray(projected_hit["xy"]) - projection["xy"]))
                        if view_index == 0 and observation["status"] == "surface_candidate" and len(anchors) < 5:
                            anchors.append(surface)
                records.append(record)
            available = [r for r in records if "ray_observation" in r]
            candidate_report["views"].append({
                "view_id": camera.view_id, "yaw": views[view_index].get("yaw"),
                "pose_pairing_validated": camera.pose_pairing_validated,
                "pixel_contract_validated": camera.pixel_contract_validated,
                "projection_in_view": len(available),
                "target_intersections": sum("surface" in r["ray_observation"] for r in available),
                "geometric_occlusions": sum(r["ray_observation"]["status"] == "occluded" for r in available),
                "max_algebraic_roundtrip_px": max((r.get("algebraic_roundtrip_px", 0.0) for r in available), default=None),
                "records": records,
            })
        candidate_report["heldout_without_independent_observations"] = [
            {"anchor": index, "source_view": cameras[0].view_id,
             "heldout": [validate_reprojection(surface, camera, meshes) for camera in cameras[1:]]}
            for index, surface in enumerate(anchors)
        ]
        report["candidate_reports"][candidate] = candidate_report
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "geometry_projection_probe.json"
    if destination.resolve() in [p.resolve() for p in [capture_path, geometry_path, parity_report_path] if p is not None]:
        raise ContractError("Probe output must not overwrite an input")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(destination.resolve()), "views": len(views), "paired_pose_validated_views": report["paired_pose_validated_views"], "pixel_contract_validated": False, "anatomical_correspondence_validated": False}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--parity-report", type=Path)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    run(args.capture, args.geometry, args.out_dir, sample_count=args.samples, parity_report_path=args.parity_report)
