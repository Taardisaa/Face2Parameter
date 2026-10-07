"""Actual face_alignment crop grid -> raw PNG integer pixel-center indices.

Imports utilities only. Never constructs FaceAlignment or loads/downloads weights.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import cv2
import numpy as np
import torch
from face_alignment import utils


class CoordinateError(ValueError):
    pass


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise CoordinateError(f"{name} must be finite with shape {shape}")
    return array


def installed_contract():
    util_path = Path(utils.__file__).resolve()
    api_path = util_path.with_name("api.py")
    return {"face_alignment_version": importlib.metadata.version("face-alignment"),
            "utils_path": str(util_path), "utils_sha256": file_sha(util_path),
            "api_path": str(api_path), "api_sha256": file_sha(api_path),
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "torch_version": torch.__version__}


def bbox_parameters(bbox, reference_scale=195., center_y_offset=.12):
    box = finite(bbox, (4,), "bbox")
    if box[2] <= box[0] or box[3] <= box[1] or not np.isfinite(reference_scale) or reference_scale <= 0:
        raise CoordinateError("bbox extent and detector reference_scale must be positive")
    if not np.isfinite(center_y_offset):
        raise CoordinateError("center_y_offset must be finite")
    center = (box[:2] + box[2:]) / 2
    center[1] -= (box[3] - box[1]) * center_y_offset
    scale = float(((box[2] - box[0]) + (box[3] - box[1])) / reference_scale)
    return center, scale


def trace_crop(image, center, scale, resolution=256):
    """Instrument actual library transform/cv2 calls, restoring them immediately.

    Sequential-only: spies are process-local, not thread-safe. The library's actual
    uint8 staging array and true integer transform outputs provide the contract.
    """
    image = np.asarray(image)
    center = finite(center, (2,), "center")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise CoordinateError("trace_crop requires an actual uint8 RGB image")
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(resolution) or resolution <= 1 or resolution != int(resolution):
        raise CoordinateError("Invalid crop scale/resolution")
    resolution = int(resolution)
    original_transform, original_resize = utils.transform, cv2.resize
    transformed, resize_calls = [], []

    def transform_spy(point, actual_center, actual_scale, actual_resolution, invert=False):
        result = original_transform(point, actual_center, actual_scale, actual_resolution, invert)
        transformed.append({"point": list(point), "invert": bool(invert), "result": result.tolist()})
        return result

    def resize_spy(source, *args, **kwargs):
        resize_calls.append({"stage_shape": list(source.shape), "stage_dtype": str(source.dtype),
                             "stage_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
                             "destination_size": list(kwargs.get("dsize", args[0] if args else [])),
                             "interpolation": int(kwargs.get("interpolation", cv2.INTER_LINEAR))})
        return original_resize(source, *args, **kwargs)

    try:
        utils.transform = transform_spy
        cv2.resize = resize_spy
        cropped = utils.crop(image, center, float(scale), resolution)
    finally:
        utils.transform = original_transform
        cv2.resize = original_resize
    if len(transformed) != 2 or len(resize_calls) != 1:
        raise CoordinateError("Installed crop call graph differs from the inspected contract")
    if transformed[0]["point"] != [1, 1] or transformed[1]["point"] != [resolution, resolution]:
        raise CoordinateError("Installed crop endpoints differ from inspected contract")
    ul, br = np.asarray(transformed[0]["result"], int), np.asarray(transformed[1]["result"], int)
    stage = resize_calls[0]
    if stage["interpolation"] != cv2.INTER_LINEAR or stage["stage_dtype"] != "uint8":
        raise CoordinateError("Unknown interpolation/staging dtype")
    if stage["stage_shape"] != [int(br[1] - ul[1]), int(br[0] - ul[0]), 3]:
        raise CoordinateError("Observed staging shape disagrees with observed bounds")
    height, width = image.shape[:2]
    record = {"schema": "face_alignment_actual_crop_grid_v1", "installed_source": installed_contract(),
              "input_rgb_array_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "cropped_rgb_array_sha256": hashlib.sha256(cropped.tobytes()).hexdigest(),
              "raw_png_size": [width, height], "center": center.tolist(), "scale": float(scale),
              "crop_resolution": resolution, "actual_ul_int": ul.tolist(), "actual_br_int": br.tolist(),
              "actual_transform_calls": transformed, "actual_resize": stage,
              "padding_stage_pixels_ltrb": [int(min(br[0] - ul[0], max(0, -ul[0]))),
                                             int(min(br[1] - ul[1], max(0, -ul[1]))),
                                             int(min(br[0] - ul[0], max(0, br[0] - width))),
                                             int(min(br[1] - ul[1], max(0, br[1] - height)))],
              "raw_copy_rect_exclusive": [int(np.clip(ul[0], 0, width)), int(np.clip(ul[1], 0, height)),
                                           int(np.clip(br[0], 0, width)), int(np.clip(br[1], 0, height))],
              "source_pixel_convention": "top_left_integer_pixel_center_indices",
              "resize_convention": "INTER_LINEAR destination center (j+.5)*stage_extent/resolution-.5; clamped stage border",
              "heatmap_convention": "decoded q = zero_based_argmax_index + .5 + quarter_sign_offset; edge coordinates",
              "scope_limit": "Actual crop/resize geometric grid only. Heatmap-to-crop edge alignment is an explicit grid convention, not proof of FAN feature anchor or anatomical accuracy."}
    return cropped, record


def check_trace(trace):
    if trace.get("schema") != "face_alignment_actual_crop_grid_v1":
        raise CoordinateError("Unknown crop trace schema")
    installed = installed_contract()
    for key in ("utils_sha256", "api_sha256", "opencv_version", "numpy_version", "torch_version", "face_alignment_version"):
        if trace["installed_source"][key] != installed[key]:
            raise CoordinateError(f"Installed source/runtime {key} changed since the trace")
    for key in ("actual_ul_int", "actual_br_int", "raw_png_size"):
        coordinates = finite(trace[key], (2,), key)
        if np.any(coordinates != np.trunc(coordinates)):
            raise CoordinateError(f"{key} must contain integer indices/extents")
    extent = np.asarray(trace["actual_br_int"]) - trace["actual_ul_int"]
    if np.any(extent <= 0) or np.any(np.asarray(trace["raw_png_size"]) <= 0):
        raise CoordinateError("Invalid trace extent/dimensions")
    if trace["actual_resize"]["interpolation"] != cv2.INTER_LINEAR:
        raise CoordinateError("Trace interpolation is not INTER_LINEAR")
    if trace["actual_resize"]["stage_shape"] != [int(extent[1]), int(extent[0]), 3]:
        raise CoordinateError("Trace stage shape contradicts integer bounds")
    if trace["actual_resize"]["destination_size"] != [trace["crop_resolution"]] * 2:
        raise CoordinateError("Trace resize destination contradicts crop resolution")
    return extent


def heatmap_to_raw(points, trace, heatmap_resolution=64):
    """Map declared heatmap edge coordinates; retain floats and padding validity."""
    extent = check_trace(trace)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise CoordinateError("points must be a finite Nx2 array")
    if heatmap_resolution != 64:
        raise CoordinateError("Inspected native decoder is certified only for square 64x64 heatmaps")
    if np.any(points < 0) or np.any(points > heatmap_resolution):
        raise CoordinateError("Heatmap edge coordinates outside grid")
    stage_index = points * extent / heatmap_resolution - .5
    clamped = np.clip(stage_index, 0, extent - 1)
    raw = np.asarray(trace["actual_ul_int"]) + clamped
    low, high = np.floor(raw), np.ceil(raw)
    size = np.asarray(trace["raw_png_size"])
    valid = np.all((low >= 0) & (high < size), axis=1)
    return {"raw_pixel_center_indices": raw.tolist(),
            "unclamped_raw_pixel_center_indices": (np.asarray(trace["actual_ul_int"]) + stage_index).tolist(),
            "crop_pixel_center_indices": (points * trace["crop_resolution"] / heatmap_resolution - .5).tolist(),
            "resize_stage_border_clamped": np.any(stage_index != clamped, axis=1).tolist(),
            "raw_interpolation_footprint_fully_in_image": valid.tolist(),
            "status": ["mapped_grid_only" if flag else "uncertain_padding" for flag in valid]}


def decode_heatmaps(heatmaps, trace):
    """Actual installed decoder output plus float crop-faithful coordinates."""
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    if heatmaps.ndim != 4 or heatmaps.shape[0] != 1 or heatmaps.shape[2:] != (64, 64) or not np.isfinite(heatmaps).all():
        raise CoordinateError("Expected finite heatmaps [1,N,64,64]")
    center = np.asarray(trace["center"], np.float64)
    q, native, scores = utils.get_preds_fromhm(heatmaps, center, float(trace["scale"]))
    argmax = np.argmax(heatmaps.reshape(1, heatmaps.shape[1], -1), axis=-1)[0]
    argmax_xy = np.column_stack((argmax % 64, argmax // 64))
    mapped = heatmap_to_raw(q[0], trace)
    h = 200 * trace["scale"]
    continuous_library = center - h / 2 + q[0].astype(np.float64) * h / 64
    raw = np.asarray(mapped["raw_pixel_center_indices"])
    return {"heatmap_decoded_edge_coordinates": q[0].tolist(),
            "heatmap_argmax_pixel_center_indices": argmax_xy.tolist(),
            "heatmap_quarter_sign_offsets": (q[0] - argmax_xy - .5).tolist(),
            "scores": scores[0].tolist(),
            "native_preds_orig_integer": native[0].tolist(),
            "ideal_unquantized_library_inverse": continuous_library.tolist(),
            "native_int32_truncation_vs_library_continuous": (native[0] - continuous_library).tolist(),
            "native_vs_actual_crop_grid": (native[0] - raw).tolist(),
            "continuous_library_vs_actual_crop_grid": (continuous_library - raw).tolist(), **mapped}
