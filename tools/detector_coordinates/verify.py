"""Read coordinate-encoded images after the installed crop/resize; no FAN model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mapping import bbox_parameters, decode_heatmaps, file_sha, heatmap_to_raw, trace_crop


CASES = [
    ("small_inside", (160, 128), [50.25, 46.6, 100.9, 96.7]),
    ("fractional_square", (320, 320), [102.35, 109.75, 206.8, 211.45]),
    ("large_rect_inside", (641, 479), [240.2, 172.8, 363.7, 294.4]),
    ("edge_left_top_padding", (256, 192), [-18.6, -11.2, 78.7, 80.4]),
    ("edge_right_bottom_padding", (513, 257), [426.25, 190.4, 536.8, 298.75]),
]


def encoded_ramp(width, height, axis):
    """Quarter-pixel readout precision; exclude discontinuous 64-pixel seams."""
    y, x = np.mgrid[:height, :width]
    coordinate = x if axis == 0 else y
    return np.stack(((coordinate * 4) % 256, coordinate // 64,
                     np.full_like(coordinate, 255)), axis=-1).astype(np.uint8)


def stats(error):
    error = np.asarray(error)
    return {"count": int(error.size), "mean_signed_pixels": float(error.mean()),
            "rmse_pixels": float(np.sqrt(np.mean(error ** 2))),
            "mean_abs_pixels": float(np.mean(np.abs(error))), "max_abs_pixels": float(np.abs(error).max()),
            "min_signed_pixels": float(error.min()), "max_signed_pixels": float(error.max())}


def synthetic_heatmaps():
    yy, xx = np.mgrid[:64, :64]
    peaks = [(8.3, 10.7), (32.2, 32.0), (48.7, 40.2), (2.3, 2.1), (60.2, 61.1)]
    return np.asarray([[np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 3)
                        for x, y in peaks]], dtype=np.float32)


def verify_case(directory, name, size, bbox):
    directory.mkdir(parents=True, exist_ok=True)
    width, height = size
    center, scale = bbox_parameters(bbox)
    axis_reports, trace = [], None
    for axis, label in enumerate(("x", "y")):
        original = encoded_ramp(width, height, axis)
        raw_path, crop_path = directory / f"{name}_{label}_raw.png", directory / f"{name}_{label}_crop.png"
        Image.fromarray(original).save(raw_path)
        # The image supplied to actual crop is independently reloaded from disk.
        source = np.array(Image.open(raw_path).convert("RGB"))
        cropped, current_trace = trace_crop(source, center, scale)
        Image.fromarray(cropped).save(crop_path)
        trace = current_trace
        resolution = trace["crop_resolution"]
        yy, xx = np.mgrid[:resolution, :resolution]
        # All output cell centers, not a selected/fitted subset of matches.
        q = np.column_stack(((xx.ravel() + .5) * 64 / resolution,
                             (yy.ravel() + .5) * 64 / resolution))
        mapped = heatmap_to_raw(q, trace)
        expected = np.asarray(mapped["raw_pixel_center_indices"]).reshape(resolution, resolution, 2)[:, :, axis]
        footprint_valid = np.asarray(mapped["raw_interpolation_footprint_fully_in_image"]).reshape(resolution, resolution)
        seam_safe = np.floor(expected).astype(int) // 64 == np.ceil(expected).astype(int) // 64
        mask = footprint_valid & seam_safe & (cropped[:, :, 2] == 255)
        observed = cropped[:, :, 0].astype(float) / 4 + cropped[:, :, 1].astype(float) * 64
        if mask.sum() < 200:
            raise ValueError("Insufficient independently observed ramp support")
        errors = observed[mask] - expected[mask]
        report = {"axis": label, "raw_png": str(raw_path.resolve()), "raw_sha256": file_sha(raw_path),
                  "crop_png": str(crop_path.resolve()), "crop_sha256": file_sha(crop_path),
                  "valid_pixels": int(mask.sum()), "padding_or_interpolation_invalid_pixels": int((~footprint_valid).sum()),
                  "coordinate_encoding_seam_excluded_pixels": int((footprint_valid & ~seam_safe).sum()),
                  "measured_error": stats(errors),
                  "wrong_plus_half_pixel_convention_error": stats(errors - .5),
                  "wrong_minus_half_pixel_convention_error": stats(errors + .5),
                  "fixed_ramp_measurement_guard_pixels": .25,
                  "status": "validated_crop_grid" if np.max(np.abs(errors)) <= .25 else "uncertain"}
        axis_reports.append(report)
    trace["bbox"] = bbox
    trace["bbox_reference_scale"] = 195.
    trace["bbox_center_y_offset"] = .12
    trace_path = directory / f"{name}_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    decoded = decode_heatmaps(synthetic_heatmaps(), trace)
    native_errors = np.asarray(decoded["native_vs_actual_crop_grid"])
    truncation = np.asarray(decoded["native_int32_truncation_vs_library_continuous"])
    valid = np.asarray(decoded["raw_interpolation_footprint_fully_in_image"], dtype=bool)
    return {"case": name, "status": "validated_crop_grid" if all(r["status"] == "validated_crop_grid" for r in axis_reports) else "uncertain",
            "trace_path": str(trace_path.resolve()), "trace_sha256": file_sha(trace_path),
            "raw_png_size": size, "bbox": bbox, "actual_ul_int": trace["actual_ul_int"],
            "actual_br_int": trace["actual_br_int"], "padding_stage_pixels_ltrb": trace["padding_stage_pixels_ltrb"],
            "ramp_measurements": axis_reports, "synthetic_heatmap_decode": decoded,
            "native_preds_orig_vs_actual_crop_grid_error": stats(native_errors),
            "native_vs_actual_crop_grid_valid_footprint_only": stats(native_errors[valid]) if valid.any() else None,
            "native_int32_precision_loss": stats(truncation),
            "heatmap_anchor_status": "declared_grid_convention_only; FAN anchor/anatomical accuracy unvalidated"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [verify_case(args.out_dir, name, size, bbox) for name, size, bbox in CASES]
    report = {"status": "validated_crop_grid" if all(c["status"] == "validated_crop_grid" for c in reports) else "uncertain",
              "tool_sources": {"mapping_sha256": file_sha(Path(__file__).with_name("mapping.py")),
                               "verify_sha256": file_sha(Path(__file__))},
              "cases": reports, "evidence": "Actual installed crop() and cv2.resize() were executed on saved/reloaded raw coordinate-encoded PNGs; output RGB independently decodes raw x/y. No fitted correction and no model inference.",
              "coordinate_convention": "Raw PNG top-left integer centers; heatmap q is edge coordinates; raw=ul+q*(br-ul)/64-.5, with resize border clamp and padding validity.",
              "limits": "Uint8 ramp quantization (fixed .25px guard), coordinate-code seams excluded explicitly, padding support uncertain. FAN receptive-field anchor, training target bias and anatomical accuracy are not validated."}
    path = args.out_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(path), "cases": len(reports)}))
    return 0 if report["status"] == "validated_crop_grid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
