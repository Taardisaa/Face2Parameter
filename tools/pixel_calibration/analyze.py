"""Independent saved-PNG calibration; never contacts HS2 or loads a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


class CalibrationError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_array(value, shape, name):
    a = np.asarray(value, dtype=float)
    if a.size != math.prod(shape) or not np.isfinite(a).all():
        raise CalibrationError(f"{name}: expected finite array of shape {shape}")
    return a.reshape(shape)


def unwrap(value):
    for key in ("response", "result", "data"):
        if "capture_camera" not in value and isinstance(value.get(key), dict):
            return unwrap(value[key])
    if "capture_camera" not in value or "pixel_calibration" not in value:
        raise CalibrationError("Missing capture_camera or pixel_calibration metadata")
    return value


def project(points, camera, width, height):
    """Unity CPU projection -> bottom-left edge coordinates, without PNG convention."""
    v = finite_array(camera["world_to_camera"], (4, 4), "world_to_camera")
    p = finite_array(camera["projection"], (4, 4), "projection")
    if abs(np.linalg.det(v)) < 1e-10 or abs(np.linalg.det(p)) < 1e-10:
        raise CalibrationError("Singular view/projection matrix")
    if camera.get("matrix_layout") != "row_major_16; column_vectors":
        raise CalibrationError("Unknown matrix layout")
    rect = finite_array(camera["pixel_rect"], (4,), "pixel_rect")
    if rect[2] <= 0 or rect[3] <= 0 or rect[0] < 0 or rect[1] < 0:
        raise CalibrationError("Invalid pixel_rect")
    if rect[0] + rect[2] > width + 1e-4 or rect[1] + rect[3] > height + 1e-4:
        raise CalibrationError("pixel_rect extends outside PNG")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise CalibrationError("Invalid world coordinates")
    clip = (p @ v @ np.column_stack((points, np.ones(len(points)))).T).T
    if np.any(clip[:, 3] <= 1e-8):
        raise CalibrationError("Marker behind camera or zero homogeneous w")
    ndc = clip[:, :3] / clip[:, 3:4]
    if np.any(np.abs(ndc[:, :2]) >= 1) or np.any(np.abs(ndc[:, 2]) > 1 + 1e-5):
        raise CalibrationError("Marker outside CPU clip volume")
    return rect[:2] + (ndc[:, :2] + 1) * 0.5 * rect[2:]


def component_count(mask):
    unseen = set(zip(*np.nonzero(mask)))
    count = 0
    while unseen:
        count += 1
        todo = [unseen.pop()]
        while todo:
            y, x = todo.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    todo.append(neighbor)
    return count


def decode_coverage(levels, aa):
    if aa == 1:
        if np.any(levels != 255):
            raise CalibrationError("AA1 marker has mixed/blurred edge pixels")
        return "binary", levels.astype(float) / 255, {"binary": 0.0}
    if aa not in (4, 8):
        raise CalibrationError("Only AA1, AA4 and AA8 have an implemented coverage contract")
    fractions = np.arange(1, aa + 1) / aa
    linear = np.rint(fractions * 255)
    srgb = np.rint(np.where(fractions <= .0031308, 12.92 * fractions,
                           1.055 * fractions ** (1 / 2.4) - .055) * 255)
    errors = {name: float(np.max(np.min(np.abs(levels[:, None] - values), axis=1)))
              for name, values in (("linear", linear), ("srgb", srgb))}
    candidates = [name for name, error in errors.items() if error <= 1]
    if not candidates:
        raise CalibrationError(f"Edge colors are not AA{aa} coverage levels: {errors}")
    if len(candidates) > 1:
        # Full coverage alone cannot certify a transfer curve or subpixel edges.
        return "undetermined", levels.astype(float) / 255, errors
    encoding = candidates[0]
    values = linear if encoding == "linear" else srgb
    weights = fractions[np.argmin(np.abs(levels[:, None] - values), axis=1)]
    return encoding, weights, errors


def measure_pixels(pixels, markers, aa):
    height, width = pixels.shape[:2]
    assigned = np.zeros((height, width), dtype=bool)
    measured = []
    if not np.all(pixels[0] == 0) or not np.all(pixels[-1] == 0) or not np.all(pixels[:, 0] == 0) or not np.all(pixels[:, -1] == 0):
        raise CalibrationError("Image boundary is not uniform black")
    for marker in markers:
        rgb = finite_array(marker["rgb"], (3,), "marker rgb")
        if not np.isin(rgb, [0, 1]).all() or not rgb.any():
            raise CalibrationError("Markers must have nonblack primary/secondary binary RGB")
        active = rgb.astype(bool)
        channels = pixels[:, :, active].astype(int)
        mask = np.all(pixels[:, :, ~active] == 0, axis=2) & np.all(channels > 0, axis=2)
        mask &= np.max(channels, axis=2) - np.min(channels, axis=2) <= 1
        if np.any(mask & assigned):
            raise CalibrationError("Duplicate/ambiguous marker RGB")
        if component_count(mask) != 1:
            raise CalibrationError(f"Marker {marker['id']} must form exactly one connected blob")
        assigned |= mask
        ys, xs = np.nonzero(mask)
        levels = np.rint(np.mean(channels[mask], axis=1)).astype(int)
        if np.count_nonzero(levels == 255) < 4:
            raise CalibrationError(f"Marker {marker['id']} has no usable solid-color interior")
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        # Every row and column of an axis-aligned quad is contiguous. Interior pixels
        # must be full intensity; otherwise a blur/color effect could mimic coverage.
        interior = pixels[y0 + 1:y1, x0 + 1:x1]
        if interior.size and not np.all(interior == np.rint(rgb * 255).astype(np.uint8)):
            raise CalibrationError(f"Marker {marker['id']} has nonuniform interior or holes")
        for y in range(y0, y1 + 1):
            row = np.flatnonzero(mask[y])
            if len(row) == 0 or np.any(np.diff(row) != 1):
                raise CalibrationError("Nonrectangular or fragmented marker support")
        measured.append({"id": marker["id"], "shader": marker.get("shader"),
                         "support_bbox_inclusive": [x0, y0, x1, y1],
                         "solid_pixel_count": int(np.count_nonzero(levels == 255)),
                         "coverage_pixel_count": int(len(xs)),
                         "observed_channel_levels": sorted(set(map(int, levels))),
                         "_levels": levels, "_xs": xs, "_ys": ys})
    nonblack = np.any(pixels != 0, axis=2)
    if np.any(nonblack & ~assigned):
        raise CalibrationError("Unassigned nonblack pixels: background contamination or invalid mixed colors")
    background = ~assigned
    if np.count_nonzero(background) < width * height * .5 or not np.all(pixels[background] == 0):
        raise CalibrationError("Insufficient independently verified uniform black background")
    # Only now may color levels be interpreted as coverage on uniform black.
    for obs in measured:
        encoding, weights, errors = decode_coverage(obs.pop("_levels"), aa)
        xs, ys = obs.pop("_xs"), obs.pop("_ys")
        obs.update(coverage_mass=float(weights.sum()), coverage_encoding=encoding,
                   encoding_level_error=errors,
                   centroid_pixel_index=[float(np.average(xs, weights=weights)),
                                         float(np.average(ys, weights=weights))])
    encodings = {obs["coverage_encoding"] for obs in measured} - {"undetermined"}
    if len(encodings) > 1:
        raise CalibrationError("Markers disagree on a single capture-graph coverage transfer encoding")
    return measured, int(background.sum())


def evaluate(response, png_path):
    response = unwrap(response)
    cal = response["pixel_calibration"]
    if not isinstance(cal, dict) or cal.get("kind") != "isolated_world_space_unlit_markers":
        raise CalibrationError("Unsupported calibration renderer graph")
    if cal.get("background_rgb") != [0, 0, 0] or cal.get("background_uniform_expected") is not True:
        raise CalibrationError("Metadata does not declare uniform black calibration background")
    markers = cal["markers"]
    if len(markers) < 6 or len({m["id"] for m in markers}) != len(markers):
        raise CalibrationError("At least six uniquely named markers are required")
    with Image.open(png_path) as image:
        if image.mode not in ("RGB", "RGBA"):
            raise CalibrationError("PNG must contain actual RGB or RGBA samples")
        raw = np.asarray(image)
        if image.mode == "RGBA" and np.any(raw[:, :, 3] != 255):
            raise CalibrationError("PNG alpha is not fully opaque")
        pixels = raw[:, :, :3]
    height, width = pixels.shape[:2]
    if (response["width"], response["height"]) != (width, height):
        raise CalibrationError("Metadata / PNG dimensions disagree")
    aa = cal["anti_aliasing"]
    measured, background_count = measure_pixels(pixels, markers, aa)
    centers, corners = [], []
    for marker in markers:
        center = project([marker["world_center"]], response["capture_camera"], width, height)[0]
        vertices = project(marker["world_corners"], response["capture_camera"], width, height)
        if len(vertices) != 4:
            raise CalibrationError("Expected four actual primitive quad vertices")
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        if np.max(np.abs((low + high) / 2 - center)) > .003:
            raise CalibrationError("Quad center / exported actual corners disagree")
        for vertex in vertices:
            if min(abs(vertex[0] - low[0]), abs(vertex[0] - high[0])) > .003 or min(abs(vertex[1] - low[1]), abs(vertex[1] - high[1])) > .003:
                raise CalibrationError("Marker is not an axis-aligned projected quad")
        centers.append(center)
        corners.append(vertices)
    centers = np.asarray(centers)
    observations = np.asarray([m["centroid_pixel_index"] for m in measured])
    candidates = []
    for flip in (True, False):
        for offset in (0.5, 0.0):
            expected = centers.copy()
            if flip:
                expected[:, 1] = height - expected[:, 1]
            expected -= offset
            residual = observations - expected
            candidates.append({"png_y_top_left": flip, "pixel_center_edge_offset": offset,
                               "rmse_pixels": float(np.sqrt(np.mean(residual ** 2))),
                               "max_abs_error_pixels": float(np.abs(residual).max()),
                               "marker_residual_xy": residual.tolist()})
    candidates.sort(key=lambda c: c["rmse_pixels"])
    best = candidates[0]
    flip = best["png_y_top_left"]
    direction_alt = min(c["rmse_pixels"] for c in candidates if c["png_y_top_left"] != flip)
    direction_status = "validated" if best["max_abs_error_pixels"] < 1 and direction_alt > 2 else "uncertain"
    offset_interval = None
    guarded_offset_interval = None
    edge_guard = .003
    offset_status = "uncertain"
    if aa == 1:
        lower, upper = -math.inf, math.inf
        for vertices, obs in zip(corners, measured):
            xy = vertices.copy()
            if flip:
                xy[:, 1] = height - xy[:, 1]
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            x0, y0, x1, y1 = obs["support_bbox_inclusive"]
            for edge_lo, edge_hi, first, last in zip(lo, hi, (x0, y0), (x1, y1)):
                lower = max(lower, edge_lo - first, edge_hi - last - 1)
                upper = min(upper, edge_lo - first + 1, edge_hi - last)
        offset_interval = [float(lower), float(upper)]
        guarded_offset_interval = [float(lower - edge_guard), float(upper + edge_guard)]
        accepted = [offset for offset in (0.5, 0.0) if lower - edge_guard <= offset <= upper + edge_guard]
        if lower > upper + edge_guard:
            raise CalibrationError("Observed AA1 edges have no common pixel-center sampling offset")
        if len(accepted) == 1 and upper - lower < .4:
            best_offset = accepted[0]
            best = next(c for c in candidates if c["png_y_top_left"] == flip and c["pixel_center_edge_offset"] == best_offset)
            offset_status = "validated"
    elif all(m["coverage_encoding"] != "undetermined" for m in measured):
        alternative = next(c for c in candidates if c["png_y_top_left"] == flip and c["pixel_center_edge_offset"] != best["pixel_center_edge_offset"])
        # Fixed acceptance bands, declared before seeing any raster. No fitted offset,
        # fitted gamma, or chosen per-marker correction is allowed.
        if best["max_abs_error_pixels"] <= .20 and alternative["rmse_pixels"] >= .32 and alternative["rmse_pixels"] - best["rmse_pixels"] >= .20:
            offset_status = "validated"
    return {"status": "validated" if direction_status == offset_status == "validated" else "uncertain",
            "image_y_direction": {"status": direction_status, "png_y_top_left": flip,
                                  "opposite_direction_rmse_pixels": direction_alt},
            "pixel_center_convention": {"status": offset_status,
                                        "edge_coordinate_of_index_zero_center": best["pixel_center_edge_offset"],
                                        "integer_index_projection_translation": -best["pixel_center_edge_offset"],
                                        "aa1_common_offset_interval": offset_interval,
                                        "aa1_guarded_offset_interval": guarded_offset_interval,
                                        "aa1_edge_projection_and_raster_guard_pixels": edge_guard if aa == 1 else None,
                                        "guard_basis": "Fixed 0.003-pixel measurement guard declared before live capture: serialized CPU/GPU arithmetic and D3D11 8-bit subpixel vertex snapping, not a fitted offset." if aa == 1 else None,
                                        "multisample_fixed_max_centroid_error_guard_pixels": .20 if aa in (4, 8) else None},
            "matches_expected_top_left_half_pixel_convention": flip and best["pixel_center_edge_offset"] == .5,
            "candidate_conventions": candidates, "selected_convention": best,
            "image_size": [width, height], "background_uniform_black_verified_pixels": background_count,
            "antialiasing_evidence": {"declared_samples": aa,
                                      "fractional_edge_levels_observed": any(any(level != 255 for level in m["observed_channel_levels"]) for m in measured),
                                      "effective_multisampling_verified": aa in (4, 8) and all(m["coverage_encoding"] not in ("binary", "undetermined") for m in measured)},
            "markers": measured, "cpu_projected_centers_bottom_left_edge": centers.tolist(),
            "renderer_graph_scope": {key: cal.get(key) for key in (
                "kind", "bridge_version", "unity_version", "graphics_api", "graphics_uv_starts_at_top",
                "uses_reversed_z_buffer", "color_space", "anti_aliasing", "layer", "bridge_module_version_id",
                "bridge_mvid", "allow_msaa", "actual_rendering_path")},
            "capture_camera_contract": {key: response["capture_camera"].get(key) for key in (
                "matrix_layout", "pixel_rect", "near_clip", "far_clip", "aspect", "roll",
                "allow_hdr", "allow_msaa", "rendering_path", "actual_rendering_path", "quality_antialiasing",
                "bridge_mvid", "graphics_api", "graphics_uv_starts_at_top", "uses_reversed_z_buffer",
                "render_texture_format", "anti_aliasing", "viewport_origin", "cpu_ndc_depth_range")},
            "capture_request_controls": {key: response.get(key) for key in (
                "capture_kind", "orthographic", "ortho_size", "fov", "yaw", "pitch", "distance")},
            "scope_limit": "Only this isolated unlit-marker saved PNG and capture camera. Does not certify character shaders, postprocessing, geometry skinning, or another capture graph.",
            "evidence": "Measured RGB components, actual support edges and decoded coverage centroids versus independently evaluated CPU P*V. Metadata viewport coordinates and hashes are not validation inputs."}


def analyze_file(metadata_path, png_override=None):
    metadata_path = Path(metadata_path).resolve()
    response = unwrap(json.loads(metadata_path.read_text(encoding="utf-8-sig")))
    png_path = Path(png_override or response.get("path", ""))
    if not png_path.is_absolute():
        png_path = metadata_path.parent / png_path
    png_path = png_path.resolve()
    result = {"metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
              "png_path": str(png_path)}
    if png_path.is_file():
        result["png_sha256"] = sha256(png_path)
        with Image.open(png_path) as image:
            raw = np.asarray(image.convert("RGB"))
        colors, counts = np.unique(raw.reshape(-1, 3), axis=0, return_counts=True)
        order = np.argsort(-counts)
        result["raw_png_audit"] = {"unique_rgb_count": len(colors),
                                   "most_frequent_rgb": [{"rgb": colors[i].tolist(), "pixels": int(counts[i])} for i in order[:40]],
                                   "boundary_uniform_black": bool(np.all(raw[0] == 0) and np.all(raw[-1] == 0) and np.all(raw[:, 0] == 0) and np.all(raw[:, -1] == 0))}
    try:
        result.update(evaluate(response, png_path))
    except (CalibrationError, KeyError, ValueError, TypeError, IndexError, OSError) as exc:
        result.update(status="failed", reason=str(exc),
                      image_y_direction={"status": "failed"}, pixel_center_convention={"status": "failed"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Capture response JSON, or manifest {cases:[{metadata,png?}]}")
    parser.add_argument("--png", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if "cases" in source:
        reports = []
        for case in source["cases"]:
            metadata_name = case.get("metadata", case.get("response_path"))
            metadata = args.input.parent / metadata_name
            png_name = case.get("png", case.get("png_path"))
            png = args.input.parent / png_name if png_name else None
            reports.append({"case": case.get("name", metadata_name), **analyze_file(metadata, png)})
        report = {"status": "validated" if reports and all(r["status"] == "validated" for r in reports) else "failed" if any(r["status"] == "failed" for r in reports) else "uncertain", "cases": reports}
        report["identical_png_different_declared_aa"] = [
            [a["case"], b["case"]] for i, a in enumerate(reports) for b in reports[i + 1:]
            if a.get("png_sha256") == b.get("png_sha256") and
            unwrap(json.loads(Path(a["metadata_path"]).read_text(encoding="utf-8-sig")))["pixel_calibration"]["anti_aliasing"] !=
            unwrap(json.loads(Path(b["metadata_path"]).read_text(encoding="utf-8-sig")))["pixel_calibration"]["anti_aliasing"]]
    else:
        report = analyze_file(args.input, args.png)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.out)}, ensure_ascii=False))
    return 0 if report["status"] == "validated" else 2


if __name__ == "__main__":
    raise SystemExit(main())
