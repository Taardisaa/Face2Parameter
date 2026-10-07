"""Measure saved-PNG outer contours against actual paired triangle projections.

This measures a geometry-union hypothesis, not alpha/culling or anatomical truth.
No fitting, recropping, rescaling, or independent per-view alignment is allowed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt

if __package__:
    from .calibrate import digest, read_json
    from .core import Camera, ContractError, finite_array, meshes_from_geometry
else:
    from calibrate import digest, read_json
    from core import Camera, ContractError, finite_array, meshes_from_geometry


def raster_union(camera, vertices, triangles):
    """Two-sided opaque projected union sampled at integer pixel centers.

    Reject triangles crossing the near/far plane rather than silently dropping
    them; viewport clipping alone is handled by bounding the sample rectangle.
    Shared-edge coverage is inclusive, with tiny arithmetic tolerance only.
    """
    vertices = finite_array(vertices)
    indices = np.asarray(triangles)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ContractError("Vertices must be Nx3")
    if indices.dtype.kind not in "iu" or indices.ndim != 2 or indices.shape[1] != 3:
        raise ContractError("Triangles must be integer Mx3")
    if indices.size and (indices.min() < 0 or indices.max() >= len(vertices)):
        raise ContractError("Triangle indices outside vertex array")
    clip = (camera.projection @ camera.view @ np.column_stack((vertices, np.ones(len(vertices)))).T).T
    if np.any(clip[:, 3] <= 0):
        raise ContractError("Near/behind-camera triangles need explicit clipping")
    ndc = clip[:, :3] / clip[:, 3:4]
    if np.any(ndc[:, 2] < camera.ndc_depth[0]) or np.any(ndc[:, 2] > camera.ndc_depth[1]):
        raise ContractError("Depth-clipped geometry requires explicit polygon clipping")
    x, y, width, height = camera.rect
    xy = np.column_stack((x + (ndc[:, 0] + 1) * width / 2 - .5,
                          camera.height - (y + (ndc[:, 1] + 1) * height / 2) - .5))
    result = np.zeros((camera.height, camera.width), dtype=bool)
    for triangle in xy[indices]:
        lower = np.maximum(np.ceil(triangle.min(axis=0)).astype(int), [0, 0])
        upper = np.minimum(np.floor(triangle.max(axis=0)).astype(int), [camera.width - 1, camera.height - 1])
        if np.any(lower > upper):
            continue
        a, b, c = triangle
        denominator = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(denominator) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(lower[0], upper[0] + 1), np.arange(lower[1], upper[1] + 1))
        px, py = xs - a[0], ys - a[1]
        u = (px * (c[1] - a[1]) - py * (c[0] - a[0])) / denominator
        v = ((b[0] - a[0]) * py - (b[1] - a[1]) * px) / denominator
        result[lower[1]:upper[1] + 1, lower[0]:upper[0] + 1] |= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    # Respect subviewports; integer pixel centers are shifted by half a pixel.
    xs, ys = np.meshgrid(np.arange(camera.width) + .5, camera.height - np.arange(camera.height) - .5)
    return result & (xs >= x) & (xs < x + width) & (ys >= y) & (ys < y + height)


def observed_foreground(rgb, *, tolerance=2):
    if type(tolerance) is not int or not 0 <= tolerance <= 8:
        raise ContractError("Background tolerance must be an integer from 0 to 8")
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ContractError("RGB uint8 required")
    background = image[0, 0]
    if not np.array_equal(image[0], np.tile(background, (image.shape[1], 1))):
        raise ContractError("Entire top border must independently agree on uniform background")
    delta = np.abs(image.astype(np.int16) - background.astype(np.int16)).max(axis=2)
    foreground = delta > tolerance
    if foreground.all() or not foreground.any():
        raise ContractError("Nonempty foreground and background required")
    if np.mean(delta == 0) < .25:
        raise ContractError("At least one quarter of the image must independently match background RGB")
    return binary_fill_holes(foreground), background.tolist()


def boundary(mask):
    return mask & ~binary_erosion(mask, border_value=0)


def compare_masks(observed, projected):
    a, b = np.asarray(observed, bool), np.asarray(projected, bool)
    if a.shape != b.shape or a.ndim != 2 or not a.any() or not b.any():
        raise ContractError("Same-size nonempty 2D masks required")
    edge_a, edge_b = boundary(a), boundary(b)
    a_to_b = distance_transform_edt(~edge_b)[edge_a]
    b_to_a = distance_transform_edt(~edge_a)[edge_b]
    distances = np.r_[a_to_b, b_to_a]
    return {"observed_pixels": int(a.sum()), "projected_pixels": int(b.sum()),
            "observed_only_pixels": int((a & ~b).sum()), "projected_only_pixels": int((b & ~a).sum()),
            "iou": float((a & b).sum() / (a | b).sum()),
            "symmetric_boundary_rms_px": float(np.sqrt(np.mean(distances ** 2))),
            "symmetric_boundary_p95_px": float(np.percentile(distances, 95)),
            "symmetric_boundary_max_px": float(distances.max()),
            "observed_to_projected_p95_px": float(np.percentile(a_to_b, 95)),
            "projected_to_observed_p95_px": float(np.percentile(b_to_a, 95))}


def run(args):
    capture = read_json(args.capture)
    capture = capture.get("bridge_result", capture)
    geometry, parity = read_json(args.geometry), read_json(args.parity_report)
    geometry_sha = digest(args.geometry)
    if parity.get("snapshot_sha256") != geometry_sha:
        raise ContractError("Independent LBS parity report is bound to another snapshot")
    if not parity.get("meshes"):
        raise ContractError("Nonempty independent LBS parity report required")
    certified = {entry["renderer_path"]: entry["source_geometry_sha256"] for entry in parity["meshes"]
                 if entry.get("certified") is True and "scale_free_trs" in entry.get("matching_candidates", [])}
    meshes = meshes_from_geometry(geometry, certification={"world_policy_validated": True,
        "world_candidate": "scale_free_trs", "mesh_source_hashes": certified}, diagnostic=True)
    head_paths = {entry["renderer_path"] for entry in geometry["meshes"] if entry["mesh_name"] == "o_head"}
    active = [mesh for mesh in meshes if mesh.visible]
    if any(not mesh.world_policy_certified for mesh in active):
        raise ContractError("Every active face renderer must have snapshot-local world-policy evidence")
    views = capture.get("views") or [capture]
    report = {"schema_version": 1, "evidence_kind": "paired_geometry_saved_png_outer_contour_measurement",
        "capture_sha256": digest(args.capture), "geometry_sha256": geometry_sha,
        "parity_report_sha256": digest(args.parity_report), "anatomical_correspondence_validated": False,
        "material_visibility_validated": False, "views": [],
        "limits": ["Opaque two-sided geometry union does not reproduce alpha/culling/material visibility.",
                   "PNG foreground uses a uniformly observed top-border RGB and at least 25% exact background, with explicit tolerance.",
                   "Missing body meshes can affect neckline; full-image and upper-95-percent metrics remain separate.",
                   "No per-view fitted scale, camera adjustment, cropping, or translation is used."]}
    args.out.mkdir(parents=True, exist_ok=True)
    for index, view in enumerate(views):
        if (view.get("paired_geometry") or {}).get("sha256") != geometry_sha:
            raise ContractError("View does not reference this actual geometry SHA")
        certificate = None
        if args.pixel_report:
            if __package__:
                from .pixel_certificate import certify_pixel_contract
            else:
                from pixel_certificate import certify_pixel_contract
            certificate = certify_pixel_contract(args.pixel_report, view)
        camera = Camera.from_capture(view, pixel_certificate=certificate, diagnostic=args.diagnostic_unvalidated)
        with Image.open(view["path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        if rgb.shape[:2] != (camera.height, camera.width):
            raise ContractError("PNG dimensions differ from paired camera")
        observed, bg = observed_foreground(rgb, tolerance=args.background_tolerance)
        row = {"view_index": index, "yaw": view.get("yaw"), "png_sha256": digest(Path(view["path"])),
               "pixel_contract_validated": camera.pixel_contract_validated,
               "pose_pairing_validated": camera.pose_pairing_validated, "background_rgb": bg, "hypotheses": {}}
        for name, selection in (("head_only", [mesh for mesh in active if mesh.renderer_path in head_paths]),
                                ("face_renderer_union", active)):
            projected = np.zeros(observed.shape, bool)
            for mesh in selection:
                projected |= raster_union(camera, mesh.vertices, mesh.triangles)
            projected = binary_fill_holes(projected)
            row["hypotheses"][name] = {"full_image": compare_masks(observed, projected),
                "upper_95_percent": compare_masks(observed[:int(camera.height * .95)], projected[:int(camera.height * .95)])}
            overlay = rgb.copy()
            overlay[observed & ~projected] = [255, 0, 255]
            overlay[projected & ~observed] = [0, 255, 255]
            path = args.out / f"view_{index}_{name}.png"
            Image.fromarray(overlay).save(path)
            row["hypotheses"][name]["overlay"] = str(path.resolve())
        report["views"].append(row)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "views": len(views), "anatomical_correspondence_validated": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "geometry", "parity-report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--pixel-report", type=Path)
    parser.add_argument("--background-tolerance", type=int, default=2)
    parser.add_argument("--diagnostic-unvalidated", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
