"""Local JSON-only calibration scaffold. Never starts a detector or contacts HS2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .core import Camera, ContractError, meshes_from_geometry, observe
    from .pixel_certificate import certify_pixel_contract
else:
    from core import Camera, ContractError, meshes_from_geometry, observe
    from pixel_certificate import certify_pixel_contract


def read_json(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    capture, geometry, points = read_json(args.capture), read_json(args.geometry), read_json(args.points)
    capture = capture.get("bridge_result", capture)
    views = capture.get("views") or [capture]
    if not 0 <= args.view_index < len(views):
        raise ContractError("view-index out of range")
    view = views[args.view_index]
    cert = read_json(args.certification) if args.certification else {}
    if cert:
        if not cert.get("evidence_id"):
            raise ContractError("Certification needs an evidence_id; it is an external attestation, not auto-certification")
        for name, path in (("capture_file_sha256", args.capture), ("geometry_file_sha256", args.geometry)):
            if cert.get(name) != digest(path):
                raise ContractError(f"Certification {name} does not match input")
    paired = view.get("paired_geometry") or {}
    paired_hash = paired.get("sha256")
    if paired_hash is not None and paired_hash.lower() != digest(args.geometry):
        raise ContractError("Capture references a different geometry file SHA256")
    if points.get("coordinate_convention") != "top_left_pixel_centers_integer":
        raise ContractError("Detector points must explicitly use top_left_pixel_centers_integer; invert crop transforms first")
    if "image_path" in points and Path(points["image_path"]).resolve() != Path(view["path"]).resolve():
        raise ContractError("Detector point image path differs from selected raw view")
    pixel_report = getattr(args, "pixel_report", None)
    pixel_certificate = certify_pixel_contract(pixel_report, view) if pixel_report else None
    camera = Camera.from_capture(view, certification=cert, pixel_certificate=pixel_certificate, diagnostic=args.diagnostic_unvalidated)
    meshes = meshes_from_geometry(geometry, candidate=args.world_candidate, certification=cert, diagnostic=args.diagnostic_unvalidated)
    observations = points.get("points")
    if not isinstance(observations, list) or not observations:
        raise ContractError("Nonempty points list required")
    ids = [point.get("id") for point in observations]
    if not all(isinstance(name, str) and name for name in ids) or len(set(ids)) != len(ids):
        raise ContractError("Point IDs must be unique nonempty strings")
    results = [observe(camera, meshes, point) for point in observations]
    output = {
        "schema_version": 1, "evidence_kind": "local_projection_and_geometry_candidates",
        "diagnostic_unvalidated": args.diagnostic_unvalidated,
        "capture_file_sha256": digest(args.capture), "geometry_file_sha256": digest(args.geometry),
        "points_file_sha256": digest(args.points), "world_candidate": args.world_candidate,
        "certification_evidence_id": cert.get("evidence_id"),
        "pixel_pipeline_certificate": pixel_certificate.evidence() if pixel_certificate else None,
        "pixel_contract_validated": camera.pixel_contract_validated,
        "pose_pairing_validated": camera.pose_pairing_validated,
        "world_policy_certified": all(mesh.world_policy_certified for mesh in meshes),
        "anatomical_correspondence_validated": False, "results": results,
        "limits": [
            "No detector was run; no image orientation or anatomical definition is inferred.",
            "Geometry-only occlusion omits shader alpha/depth and camera layer masks unless separately verified.",
            "Geometry certification is external file-bound evidence; pixel report source PNG measurements are independently rechecked.",
            "Pixel coordinate transfer uses a reviewed shared render/readback path within matching runtime/camera scope; it does not certify material visibility or effective MSAA.",
            "Ray-hit to same-pixel reprojection is not independent landmark validation.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "points": len(results), "diagnostic": args.diagnostic_unvalidated, "anatomical_correspondence_validated": False}))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--world-candidate", choices=["renderer_matrix", "scale_free_trs"], default="scale_free_trs")
    parser.add_argument("--certification", type=Path)
    parser.add_argument("--pixel-report", type=Path, help="Independent saved-marker PNG report; separate from geometry certification")
    parser.add_argument("--diagnostic-unvalidated", action="store_true", help="Produce labelled candidates despite uncertified pixel/pose/world policy")
    args = parser.parse_args()
    input_paths = [args.capture, args.geometry, args.points] + ([args.certification] if args.certification else []) + ([args.pixel_report] if args.pixel_report else [])
    if args.out.resolve() in [path.resolve() for path in input_paths]:
        parser.error("Output must not overwrite an input")
    try:
        run(args)
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Calibration contract rejected: {exc}\n")


if __name__ == "__main__":
    main()
