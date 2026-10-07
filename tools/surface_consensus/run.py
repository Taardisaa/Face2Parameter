"""Offline source-bound consensus CLI. No HTTP, game changes or model downloads."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.surface_calibration.core import Camera, ContractError, meshes_from_geometry
from tools.surface_calibration.pixel_certificate import certify_pixel_contract, digest, read_json
from tools.unity_parity.geometry import analyze_snapshot
from tools.detector_coordinates.mapping import heatmap_to_raw, trace_crop
from .consensus import Policy, definitions, evaluate_candidate
from .geometric import geometric_definitions
from .diagnostic import render_support_diagnostics


def load_scene(capture_path, geometry_path, lbs_report_path, pixel_report_path, points_path):
    capture = read_json(capture_path)
    capture = capture.get("bridge_result", capture)
    geometry, report, points = map(read_json, [geometry_path, lbs_report_path, points_path])
    geometry_sha, capture_sha = digest(geometry_path), digest(capture_path)
    if report.get("snapshot_sha256") != geometry_sha:
        raise ContractError("Independent LBS report geometry file SHA changed")
    if points.get("capture_file_sha256") != capture_sha or points.get("geometry_file_sha256") != geometry_sha:
        raise ContractError("Independent detector observations refer to different capture/geometry files")
    # Numeric LBS is independently recalculated, never an external certificate
    # bool. Comparison locks the original report's measured candidate residuals.
    fresh = analyze_snapshot(geometry, normalized_tolerance=1e-5)
    saved = {r["renderer_path"]: r for r in report["meshes"]}
    hashes = {}
    for row in fresh["meshes"]:
        old = saved.get(row["renderer_path"], {})
        if not row["certified"] or "scale_free_trs" not in row.get("matching_candidates", []):
            raise ContractError("World conversion does not pass independent LBS at fixed 1e-5 normalized tolerance")
        if old.get("source_geometry_sha256") != row["source_geometry_sha256"]:
            raise ContractError("LBS mesh source identity changed")
        for name, metrics in row["candidate_errors"].items():
            for key in ("max_normalized", "rms_normalized", "max_l2"):
                if not np.isclose(metrics[key], old["candidate_errors"][name][key], rtol=1e-8, atol=1e-12):
                    raise ContractError("LBS report residual changed on independent recomputation")
        hashes[row["renderer_path"]] = row["source_geometry_sha256"]
    world_cert = {"world_policy_validated": True, "world_candidate": "scale_free_trs", "mesh_source_hashes": hashes}
    meshes = meshes_from_geometry(geometry, candidate="scale_free_trs", certification=world_cert)
    targets = [m["renderer_path"] for m in geometry["meshes"] if m["mesh_name"] == "o_head"]
    if len(targets) != 1:
        raise ContractError("Exactly one source-bound o_head required")
    views = capture.get("views") or [capture]
    observed = points.get("views", [])
    if len(views) != len(observed) or points.get("detector_coordinate_mode") != "crop-grid":
        raise ContractError("One actual crop-grid observation record per captured view required")
    cameras, certificates = [], []
    for i, (view, obs) in enumerate(zip(views, observed, strict=True)):
        paired = view.get("paired_geometry", {})
        if paired.get("sha256") != geometry_sha or paired.get("pose_signature") != geometry.get("pose_signature"):
            raise ContractError("Captured view is not SHA/pose-bound to this geometry")
        if geometry.get("frame_count") != geometry.get("frame_count_end") or paired.get("frame_count") != geometry.get("frame_count"):
            raise ContractError("Geometry/view frame pairing failed")
        if obs.get("view_index") != i or Path(view["path"]).resolve() != Path(obs["path"]).resolve() or digest(view["path"]) != obs["raw_png_sha256"]:
            raise ContractError("Independent detector PNG identity changed")
        if obs.get("yaw") != view.get("yaw"):
            raise ContractError("Observation view label differs from capture")
        if obs.get("status") != "detected" or not obs.get("detector_crop_grid_traced"):
            raise ContractError("Actual traced independent FAN observation required")
        if np.asarray(obs.get("landmark68"), dtype=float).shape != (68, 2) or np.asarray(obs["heatmap_grid_decode"].get("heatmap_decoded_edge_coordinates"), dtype=float).shape != (68, 2):
            raise ContractError("FAN observations and decoded heatmap grids must both be exactly 68x2")
        certificate = certify_pixel_contract(pixel_report_path, view)
        cameras.append(Camera.from_capture(view, pixel_certificate=certificate))
        certificates.append(certificate.evidence())
        with Image.open(view["path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        trace = obs["crop_trace"]
        _, actual_trace = trace_crop(rgb, trace["center"], trace["scale"], trace["crop_resolution"])
        if actual_trace != trace:
            raise ContractError("Crop trace not reproduced from actual raw PNG and installed pipeline")
        mapped = heatmap_to_raw(obs["heatmap_grid_decode"]["heatmap_decoded_edge_coordinates"], trace)
        if not np.allclose(mapped["raw_pixel_center_indices"], obs["landmark68"], atol=1e-9, rtol=0):
            raise ContractError("Floating observation coordinates differ from independently recomputed crop grid")
        if mapped["raw_interpolation_footprint_fully_in_image"] != obs["heatmap_grid_decode"]["raw_interpolation_footprint_fully_in_image"]:
            raise ContractError("Detector crop padding flags differ")
    sources = {name: {"path": str(Path(path).resolve()), "sha256": digest(path)} for name, path in
               [("capture", capture_path), ("geometry", geometry_path), ("lbs_report", lbs_report_path),
                ("pixel_report", pixel_report_path), ("independent_observations", points_path)]}
    implementation = {str(Path(p).resolve()): digest(p) for p in [
        Path(__file__), Path(__file__).with_name("consensus.py"), Path(__file__).with_name("geometric.py"), Path(__file__).with_name("diagnostic.py"),
        Path("tools/surface_calibration/core.py"), Path("tools/surface_calibration/pixel_certificate.py"),
        Path("tools/unity_parity/geometry.py"), Path("tools/detector_coordinates/mapping.py")]}
    return cameras, meshes, targets, observed, {"sources": sources, "pixel_certificates": certificates,
            "implementation_sha256": implementation,
            "geometry_pose_signature": geometry["pose_signature"], "geometry_frame_count": geometry["frame_count"],
            "character_head_id": geometry["character"]["head_id"],
            "independent_lbs_recomputed": True, "fixed_lbs_normalized_tolerance": 1e-5,
            "pose_pairing_from_actual_signatures_and_frames": True}


def run(args):
    cameras, meshes, targets, observations, evidence = load_scene(args.capture, args.geometry, args.lbs_report, args.pixel_report, args.points)
    policy = Policy(max_error_px=args.max_error_px)
    results = [evaluate_candidate(definition, cameras, observations, meshes, targets, policy) for definition in definitions()]
    report = {"schema_version": 1, "evidence_kind": "independent_multiview_fixed_material_point_consensus",
              **evidence, "policy": asdict(policy), "camera_or_scale_fitted": False,
              "anatomical_correspondence_validated": False, "material_visibility_validated": False,
              "detector_feature_anchor_convention_validated": False,
              "results": results, "accepted_ids": [r["id"] for r in results if r["accepted"]],
              "limits": [
                  "FAN semantic names and feature-anchor coordinates remain unvalidated. Fits cannot certify anatomy.",
                  "Only independent visible-surface geometry candidates in <=30 degree yaw scope can participate; profile/hidden-side predictions are excluded, never accepted from a heatmap score.",
                  "All LOO fits exclude the heldout xy entirely, then test it with the source-bound camera. Ray-hit to same-image reprojection is only a visibility screen.",
                  "Unknown cull/alpha/depth and incomplete renderer graph make visibility geometric only. Skin candidate may correspond to a texture/lid edge rather than its named anatomy.",
                  "Units are Unity game units. Inverse ray information is numerical conditioning, not calibrated detector uncertainty.",
                  "Material definitions are snapshot-local and fixed by source/topology identity. This run does not establish transport across morphs, expressions, heads or new captures.",
                  "Triangle minima are exact convex ray-distance solutions. Screening checks these minima only, and may reject a valid visibility-constrained interior solution conservatively."]}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    anchors = {"schema_version": 1, "source_bindings": evidence["sources"],
               "implementation_sha256": evidence["implementation_sha256"],
               "geometry_pose_signature": evidence["geometry_pose_signature"],
               "character_head_id": evidence["character_head_id"], "policy": asdict(policy),
               "anchors": [{"id": r["id"], "detector_index": r["detector_index"],
                            "operational_accepted": r["accepted"], "semantic_validated": False,
                            "status": r["status"], "material": r["all_view_fit"]["surface"]}
                           for r in results if "all_view_fit" in r]}
    (args.out / "material_definitions.json").write_text(json.dumps(anchors, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    front_index = min(range(len(observations)), key=lambda i: abs(float(observations[i]["yaw"])))
    geometric = geometric_definitions(next(m for m in meshes if m.renderer_path == targets[0]), cameras[front_index], cameras, meshes)
    geometric.update(source_bindings=evidence["sources"], implementation_sha256=evidence["implementation_sha256"],
                     character_head_id=evidence["character_head_id"], geometry_pose_signature=evidence["geometry_pose_signature"])
    (args.out / "geometric_definitions.json").write_text(json.dumps(geometric, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if geometric["support_definition"]["unique_maximum"]:
        diagnostic = render_support_diagnostics(geometric["support_definition"], observations, args.out / "visual_review")
        (args.out / "visual_review_manifest.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for r in results[:7]:
        print(json.dumps({k: r.get(k) for k in ["id", "status", "eligible_view_indices", "max_heldout_error_px", "loo_normal_spread_degrees", "reason"]}))
    print(f"accepted={report['accepted_ids']}; out={args.out.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "geometry", "lbs-report", "pixel-report", "points", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--max-error-px", type=float, default=2.0)
    args = parser.parse_args()
    if args.out.resolve() in [getattr(args, key).resolve() for key in ("capture", "geometry", "lbs_report", "pixel_report", "points")]:
        parser.error("Output must not overwrite source inputs")
    try:
        run(args)
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Consensus contract rejected: {exc}\n")


if __name__ == "__main__":
    main()
