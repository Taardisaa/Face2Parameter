"""Run already-cached 2DFAN/S3FD on raw paired captures; downloads forbidden."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

if __package__:
    from .calibrate import digest, read_json
    from .core import Camera, ContractError, meshes_from_geometry, observe, validate_reprojection
else:
    from calibrate import digest, read_json
    from core import Camera, ContractError, meshes_from_geometry, observe, validate_reprojection


def run(args):
    import torch
    import face_alignment
    import face_alignment.api as api

    cache = Path(torch.hub.get_dir()) / "checkpoints"
    fan = cache / "2DFAN4-11f355bf06.pth.tar"
    sfd = cache / "s3fd-619a316812.pth"
    for path in (fan, sfd):
        if not path.is_file():
            raise ContractError(f"Required cached weight absent; no download permitted: {path}")

    def cached_only(url, **kwargs):
        if url.rsplit("/", 1)[-1] != fan.name:
            raise ContractError("Unexpected model requested; no download permitted")
        return str(fan)

    def no_download(*args, **kwargs):
        raise ContractError("Network model download forbidden")

    torch.set_num_threads(4)
    # Explicit local S3FD path, local FAN path loader and eager mode. Keep download
    # helpers blocked during both initialization and inference.
    with patch.object(api, "load_file_from_url", cached_only), patch.object(torch.hub, "download_url_to_file", no_download):
        detector = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device=args.device, flip_input=False,
            compile=False, face_detector_kwargs={"path_to_detector": str(sfd)},
        )
        capture = read_json(args.capture)
        capture = capture.get("bridge_result", capture)
        views = capture.get("views") or [capture]
        geometry = read_json(args.geometry)
        cert = read_json(args.certification) if args.certification else None
        if cert and (cert.get("geometry_file_sha256") != digest(args.geometry) or cert.get("capture_file_sha256") != digest(args.capture)):
            raise ContractError("World certificate file binding differs")
        meshes = meshes_from_geometry(geometry, candidate="scale_free_trs", certification=cert, diagnostic=True)
        head_paths = [entry["renderer_path"] for entry in geometry["meshes"] if entry["mesh_name"] == "o_head"]
        candidates = {"nose_tip_candidate": 30, "mouth_corner_48_candidate": 48, "mouth_corner_54_candidate": 54,
                      "eye_corner_36_candidate": 36, "eye_corner_39_candidate": 39,
                      "eye_corner_42_candidate": 42, "eye_corner_45_candidate": 45}
        results = []
        cameras = []
        from PIL import Image
        for index, view in enumerate(views):
            pixel_certificate = None
            if args.pixel_report:
                if __package__:
                    from .pixel_certificate import certify_pixel_contract
                else:
                    from pixel_certificate import certify_pixel_contract
                pixel_certificate = certify_pixel_contract(args.pixel_report, view)
            camera = Camera.from_capture(view, pixel_certificate=pixel_certificate, diagnostic=True)
            cameras.append(camera)
            path = Path(view["path"])
            with Image.open(path) as image:
                if image.size != (view["width"], view["height"]):
                    raise ContractError("Raw PNG dimensions differ from capture camera metadata")
                rgb = np.asarray(image.convert("RGB"))
            crop_traces, grid_decodes = [], []
            if args.coordinate_mode == "crop-grid":
                from tools.detector_coordinates.mapping import decode_heatmaps, trace_crop

                native_decode = api.get_preds_fromhm

                def traced_crop(image, center, scale):
                    cropped, trace = trace_crop(image, center, scale)
                    crop_traces.append(trace)
                    return cropped

                def traced_decode(heatmaps, center=None, scale=None):
                    trace = crop_traces[len(grid_decodes)]
                    if not np.allclose(center, trace["center"], atol=1e-7) or not np.isclose(scale, trace["scale"], atol=1e-7):
                        raise ContractError("Actual decoder and crop trace parameters differ")
                    result = native_decode(heatmaps, center, scale)
                    grid_decodes.append(decode_heatmaps(heatmaps, trace))
                    return result

                with patch.object(api, "crop", traced_crop), patch.object(api, "get_preds_fromhm", traced_decode):
                    landmarks, scores, boxes = detector.get_landmarks_from_image(rgb, return_landmark_score=True, return_bboxes=True)
            else:
                landmarks, scores, boxes = detector.get_landmarks_from_image(rgb, return_landmark_score=True, return_bboxes=True)
            result = {"view_index": index, "path": str(path), "raw_png_sha256": digest(path), "yaw": view.get("yaw"), "pixel_contract_validated": camera.pixel_contract_validated, "pose_pairing_validated": camera.pose_pairing_validated,
                      "pixel_certificate": pixel_certificate.evidence() if pixel_certificate else None}
            if landmarks is None or len(landmarks) != 1:
                result.update(status="no_face" if landmarks is None else "ambiguous_faces", face_count=0 if landmarks is None else len(landmarks))
            else:
                native_points = np.asarray(landmarks[0], dtype=float)
                if args.coordinate_mode == "crop-grid":
                    if len(grid_decodes) != 1 or len(crop_traces) != 1:
                        raise ContractError("Single face must have exactly one crop and decoder trace")
                    points = np.asarray(grid_decodes[0]["raw_pixel_center_indices"], dtype=float)
                    result.update(native_landmark68=native_points.tolist(), crop_trace=crop_traces[0],
                                  heatmap_grid_decode=grid_decodes[0], detector_crop_grid_traced=True)
                else:
                    points = native_points
                result.update(status="detected", landmark68=points.tolist(), heatmap_peak_scores=np.asarray(scores[0]).tolist(), bbox=np.asarray(boxes[0]).tolist())
                projected = []
                for name, point_index in candidates.items():
                    point = {"id": name, "xy": points[point_index].tolist(), "allowed_renderer_paths": head_paths}
                    try:
                        if grid_decodes and not grid_decodes[0]["raw_interpolation_footprint_fully_in_image"][point_index]:
                            raise ContractError("Detector point interpolation footprint intersects crop padding")
                        observation = observe(camera, meshes, point)
                    except ContractError as exc:
                        observation = {"id": name, "status": "projection_rejected", "reason": str(exc), "anatomical_correspondence_validated": False}
                    observation["detector_index"] = point_index
                    projected.append(observation)
                result["surface_candidates"] = projected
                result["view_dependent_indices"] = {"jaw_outline": list(range(17)), "brow_texture": list(range(17, 27))}
            results.append(result)
            print(json.dumps({"view": index, "yaw": view.get("yaw"), "status": result["status"]}), flush=True)
    # Front source; independent other-view detector observations are compared
    # without asserting the detector predicts visible/anatomically fixed points.
    front_index = min(range(len(views)), key=lambda i: abs(float(views[i].get("yaw", 0))))
    front = results[front_index]
    heldout = []
    if front["status"] == "detected":
        for observation in front.get("surface_candidates", []):
            if observation.get("status") != "surface_candidate":
                continue
            point_index = observation["detector_index"]
            comparisons = []
            for i, target in enumerate(results):
                if i == front_index or target["status"] != "detected":
                    continue
                projected = validate_reprojection(observation["surface"], cameras[i], meshes, observed_xy=target["landmark68"][point_index])
                target_hit = next((o.get("surface") for o in target.get("surface_candidates", []) if o.get("detector_index") == point_index), None)
                if target_hit:
                    projected["candidate_world_drift_game_units"] = float(np.linalg.norm(np.asarray(target_hit["world"]) - observation["surface"]["world"]))
                    same_triangle = target_hit["triangle_id"] == observation["surface"]["triangle_id"] and target_hit["renderer_path"] == observation["surface"]["renderer_path"]
                    projected["same_triangle_candidate"] = same_triangle
                    projected["barycentric_l2_drift"] = float(np.linalg.norm(np.asarray(target_hit["barycentric"]) - observation["surface"]["barycentric"])) if same_triangle else None
                projected["detector_visibility_verified"] = False
                comparisons.append(projected)
            heldout.append({"candidate_id": observation["id"], "front_surface": observation["surface"], "heldout": comparisons})
    output = {
        "schema_version": 1, "evidence_kind": "cached_FAN68_image_observations_and_geometry_candidates",
        "face_alignment_version": metadata.version("face-alignment"), "device": args.device, "compile": False,
        "weights": {str(fan): digest(fan), str(sfd): digest(sfd)},
        "capture_file_sha256": digest(args.capture), "geometry_file_sha256": digest(args.geometry),
        "pixel_contract_validated": all(camera.pixel_contract_validated for camera in cameras), "anatomical_correspondence_validated": False,
        "detector_pixel_offset_assumed": 0.0, "detector_coordinate_mapping_validated": False,
        "detector_coordinate_mode": args.coordinate_mode,
        "coordinate_assumption": (
            "actual integer crop and bilinear resize traced; float raw pixel-center indices preserved; heatmap feature-anchor convention and semantics remain unvalidated"
            if args.coordinate_mode == "crop-grid" else
            "legacy library integer xy treated as top-left integer pixel centers; actual crop-grid and integer truncation bias not corrected"
        ),
        "score_kind": "FAN heatmap peaks, not calibrated visibility probabilities",
        "views": results, "front_view_index": front_index, "heldout_candidate_comparisons": heldout,
        "limits": ["Profile/hidden-side detector outputs can be hallucinated; they are not verified observations of an anatomical point.",
                   "A supplied pixel report validates the scoped camera-to-PNG convention only; detector crop/heatmap mapping remains separately unvalidated.",
                   "Head-ray candidates may hit wrong semantic surface or be affected by unmodeled alpha/culling; no anatomy certification.",
                   "Cross-view drift and pixel error characterize candidates, not ground-truth fitting accuracy."],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--certification", type=Path)
    parser.add_argument("--pixel-report", type=Path)
    parser.add_argument("--coordinate-mode", choices=["legacy", "crop-grid"], default="legacy")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.out.resolve() in [p.resolve() for p in [args.capture, args.geometry, args.certification, args.pixel_report] if p is not None]:
        parser.error("Output must not overwrite an input")
    run(args)
