"""Fixed paired-scan diagnostic. Geometry stays unchanged except global similarity.

Run as `python -m tools.model_bridge.scan_accuracy --help`. This is a declared
public-sample diagnostic, not the full FaceScape benchmark protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import platform

import cv2
import numpy as np
from scipy.spatial import cKDTree
import scipy

from .artifact import ModelArtifact, host_path, sha


def read_scan(path):
    """Read the pinned VCGLIB binary triangle PLY, rejecting unexpected layouts."""
    with Path(path).open("rb") as stream:
        header = []
        for _ in range(64):
            line = stream.readline(1024)
            header.append(line.decode("ascii").rstrip())
            if header[-1] == "end_header":
                break
        else:
            raise ValueError("Invalid PLY header")
        semantic = [s for s in header if not s.startswith("comment ")]
        if len(semantic) != 9 or semantic[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Unsupported PLY layout")
        n = int(semantic[2].removeprefix("element vertex "))
        m = int(semantic[6].removeprefix("element face "))
        if semantic[3:6] != ["property float x", "property float y", "property float z"] or semantic[7:] != [
                "property list uchar int vertex_indices", "end_header"]:
            raise ValueError("Unsupported PLY fields")
        vertices = np.fromfile(stream, dtype="<f4", count=n * 3).reshape(n, 3).astype(np.float64)
        records = np.fromfile(stream, dtype=np.dtype([("n", "u1"), ("ids", "<i4", (3,))]), count=m)
        if len(records) != m or np.any(records["n"] != 3) or stream.read(1):
            raise ValueError("Truncated/non-triangle/trailing PLY")
        faces = records["ids"].astype(np.int64)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= n:
        raise ValueError("Invalid scan geometry")
    return vertices, faces


def camera_rays(pixels, k, rt, distortion):
    normalized = cv2.undistortPoints(np.asarray(pixels, np.float64)[:, None, :], k, distortion)[:, 0]
    rotation, translation = rt[:, :3], rt[:, 3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(rotation) <= 0:
        raise ValueError("Camera rotation is not proper orthonormal")
    directions = np.c_[normalized, np.ones(len(normalized))] @ rotation
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return -rotation.T @ translation, directions


def first_ray_hits(origin, directions, triangles, chunk=100000):
    """Exact two-sided triangle intersections; choose the first positive surface."""
    origin, directions = np.asarray(origin, float), np.asarray(directions, float)
    points, hit_faces = [], []
    for direction in directions:
        best, best_id = np.inf, -1
        for start in range(0, len(triangles), chunk):
            tri = triangles[start:start + chunk]
            e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            p = np.cross(direction, e2)
            det = np.einsum("ij,ij->i", e1, p)
            valid = np.abs(det) > 1e-12 * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
            inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
            offset = origin - tri[:, 0]
            u = np.einsum("ij,ij->i", offset, p) * inv
            q = np.cross(offset, e1)
            v = q @ direction * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
            valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (t > 0)
            t[~valid] = np.inf
            i = int(np.argmin(t))
            if t[i] < best:
                best, best_id = float(t[i]), start + i
        if best_id < 0:
            raise ValueError("Anchor ray missed the scan; do not substitute a nearest vertex")
        points.append(origin + best * direction)
        hit_faces.append(best_id)
    return np.asarray(points), np.asarray(hit_faces)


def similarity(source, target):
    """Least-squares positive uniform similarity, with reflections prohibited."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    a, b = source.mean(0), target.mean(0)
    x, y = source - a, target - b
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("Insufficient alignment anchors")
    u, s, vt = np.linalg.svd(x.T @ y)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(signs) @ u.T
    scale = float(s @ signs / np.sum(x * x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid similarity scale")
    translation = b - scale * rotation @ a
    return scale, rotation, translation


def closest_on_triangles(point, triangles):
    """Exact closest point: interior plane projection or one of three segments."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    dot = lambda x, y: np.einsum("ij,ij->i", x, y)
    d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    d20, d21 = dot(ap, ab), dot(ap, ac)
    denominator = d00 * d11 - d01 * d01
    usable = denominator > np.finfo(float).eps * np.maximum(d00 * d11, np.finfo(float).tiny)
    v = np.divide(d11 * d20 - d01 * d21, denominator, out=np.zeros(len(a)), where=usable)
    w = np.divide(d00 * d21 - d01 * d20, denominator, out=np.zeros(len(a)), where=usable)
    interior = usable & (v >= 0) & (w >= 0) & (v + w <= 1)
    candidates = [a + v[:, None] * ab + w[:, None] * ac]
    for start, end in [(a, b), (b, c), (c, a)]:
        edge = end - start
        length = dot(edge, edge)
        t = np.divide(dot(point - start, edge), length, out=np.zeros(len(a)), where=length > 0).clip(0, 1)
        candidates.append(start + t[:, None] * edge)
    candidates = np.stack(candidates, axis=1)
    d2 = np.sum((candidates - point) ** 2, axis=2)
    d2[~interior, 0] = np.inf
    ids = np.argmin(d2, axis=1)
    return candidates[np.arange(len(a)), ids]


class TriangleSurface:
    """Centroid acceleration with a conservative bound, not nearest-vertex error.

    For any point q on triangle i, ||p-c_i|| <= ||p-q|| + r_i, where r_i is
    max vertex-to-centroid distance. Thus every potentially nearer triangle is
    inside a query ball of an existing upper bound plus max(r_i). The per-triangle
    lower bound can further discard candidates without omitting a closer surface.
    """
    def __init__(self, triangles):
        self.triangles = np.asarray(triangles, np.float64)
        self.centers = self.triangles.mean(1)
        self.radii = np.linalg.norm(self.triangles - self.centers[:, None], axis=2).max(1)
        self.tree = cKDTree(self.centers)

    def closest(self, points):
        result, face_ids, counts = [], [], []
        max_radius = float(self.radii.max())
        for point in points:
            _, initial_id = self.tree.query(point)
            initial = closest_on_triangles(point, self.triangles[initial_id:initial_id + 1])[0]
            upper = float(np.linalg.norm(initial - point))
            slack = 1e-12 * max(1., np.linalg.norm(point), upper, max_radius)
            ids = np.asarray(self.tree.query_ball_point(point, upper + max_radius + slack), np.int64)
            lower = np.linalg.norm(self.centers[ids] - point, axis=1) - self.radii[ids]
            ids = ids[lower <= upper + slack]
            closest = closest_on_triangles(point, self.triangles[ids])
            best = int(np.argmin(np.sum((closest - point) ** 2, axis=1)))
            result.append(closest[best])
            face_ids.append(int(ids[best]))
            counts.append(len(ids))
        return np.asarray(result), np.asarray(face_ids), np.asarray(counts)


def stats(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def project(vertices, k, rt):
    cv = vertices @ rt[:, :3].T + rt[:, 3]
    if np.any(cv[:, 2] <= 0):
        raise ValueError("Geometry behind calibrated camera")
    homogeneous = cv @ k.T
    return homogeneous[:, :2] / homogeneous[:, 2:3], cv[:, 2]


def review_board(image_path, k, rt, distortion, vertices, faces, vertex_errors, anchors, out, saturation):
    """Calibrated undistorted visual aid; metrics never use this painter rendering."""
    original = cv2.imread(str(image_path))
    if original is None:
        raise ValueError("Missing source photo")
    factor = 512 / original.shape[1]
    size = (512, round(original.shape[0] * factor))
    scaled_k = k.copy()
    scaled_k[:2] *= factor
    # Same resize-then-undistort convention as FaceScape's public projection demo.
    photo = cv2.undistort(cv2.resize(original, size), scaled_k, distortion)
    pixels, depth = project(vertices, scaled_k, rt)
    panel = np.full_like(photo, 230)
    overlay = photo.copy()
    values = np.nan_to_num(vertex_errors, nan=0).clip(0, saturation) / saturation
    colors = cv2.applyColorMap((values * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[:, 0]
    region_faces = np.isfinite(vertex_errors[faces]).all(1)
    # Sorting is only for this visual aid; no image or depth approximation enters distances.
    for index in np.argsort(depth[faces].mean(1))[::-1]:
        polygon = np.rint(pixels[faces[index]]).astype(np.int32)
        color = tuple(int(v) for v in colors[faces[index]].mean(0)) if region_faces[index] else (170, 170, 170)
        cv2.fillConvexPoly(panel, polygon, color)
        if region_faces[index]:
            cv2.polylines(overlay, [polygon], True, (200, 160, 40), 1)
    anchor_pixels, _ = project(anchors, scaled_k, rt)
    for p in np.rint(anchor_pixels).astype(int):
        cv2.circle(overlay, tuple(p), 4, (0, 255, 0), -1)
    result = np.concatenate([photo, overlay, panel], axis=1)
    footer = np.full((48, result.shape[1], 3), 245, np.uint8)
    for i, title in enumerate(["Calibrated undistorted photo", "Fixed alignment mesh / green anchors",
                               "Triangle surface error; grey outside mask"]):
        cv2.putText(footer, title, (i * 512 + 8, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1)
    cv2.putText(footer, f"Blue: 0, red: >= {saturation:g} x outer-eye span. Display range is not a pass threshold.",
                (8, 39), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1)
    result = np.concatenate([result, footer])
    if not cv2.imwrite(str(out), result):
        raise RuntimeError("Failed to write review board")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True, help="Pinned trusted FLAME_masks.pkl only")
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("facescape_scan_protocol.json"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("format") != "facescape_public_scan_diagnostic_v1":
        raise ValueError("Unsupported protocol")
    for field, path in [("scan", args.scan), ("camera", args.cameras), ("mask", args.mask), ("embedding", args.embedding)]:
        if sha(path) != protocol[field + "_sha256"]:
            raise ValueError("Pinned diagnostic asset changed: " + field)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new output directory; preserve previous evidence")
    first = ModelArtifact(args.manifest)
    first.verify_sources()
    artifacts = [ModelArtifact(args.manifest, i) for i in range(len(first.manifest["images"]))]
    photos = {host_path(a.image["input"]).stem: a for a in artifacts}
    anchor_artifact = photos[protocol["anchor_photo"]]
    cameras = json.loads(args.cameras.read_text())
    with np.load(args.embedding, allow_pickle=False) as embedding:
        landmark_ids = embedding["landmark_indices"].tolist()
    # Hash was checked before deserializing this installed, trusted model asset.
    with args.mask.open("rb") as stream:
        masks = pickle.load(stream, encoding="latin1")
    region = np.asarray(masks[protocol["source_region"]], np.int64)
    anchor_ids = protocol["anchor_mediapipe_ids"]
    indices = [landmark_ids.index(i) for i in anchor_ids]
    name = protocol["anchor_photo"]
    def camera(n):
        if cameras[n + "_valid"] is not True:
            raise ValueError("Invalid calibrated camera")
        return (np.asarray(cameras[n + "_K"], float), np.asarray(cameras[n + "_Rt"], float),
                np.asarray(cameras[n + "_distortion"], float))
    k, rt, distortion = camera(name)
    print("Reading paired scan and intersecting fixed anchor rays", flush=True)
    scan_vertices, scan_faces = read_scan(args.scan)
    triangles = scan_vertices[scan_faces]
    origin, rays = camera_rays(anchor_artifact.arrays["detected_landmarks"][anchor_ids, :2], k, rt, distortion)
    target_anchors, target_faces = first_ray_hits(origin, rays, triangles)
    span_ids = [anchor_ids.index(i) for i in protocol["normalization_span_ids"]]
    span = float(np.linalg.norm(target_anchors[span_ids[0]] - target_anchors[span_ids[1]]))
    if span <= 0:
        raise ValueError("Invalid fixed normalization span")
    print("Building exact triangle-surface query index", flush=True)
    surface = TriangleSurface(triangles)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"format": "facescape_paired_scan_accuracy_v1", "protocol": protocol,
              "protocol_sha256": sha(args.protocol), "manifest_sha256": sha(args.manifest),
              "source_revision": first.manifest["git_revision"], "raw_parameters_modified": False,
              "normalization_span_scan_units": span, "scan_vertices": len(scan_vertices),
              "scan_triangles": len(scan_faces), "target_anchors": target_anchors.tolist(),
              "target_anchor_triangle_ids": target_faces.tolist(), "images": [],
              "anchor_artifact_sha256": anchor_artifact.image["sha256"],
              "anchor_photo_sha256": anchor_artifact.image["input_sha256"],
              "evaluator_sha256": sha(Path(__file__)),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "opencv": cv2.__version__},
              "accuracy_status": "single_expression_paired_scan_diagnostic_not_general_certification"}
    for name, artifact in photos.items():
        image_path = host_path(artifact.image["input"])
        if sha(image_path) != artifact.image["input_sha256"]:
            raise ValueError("Input photo changed")
        image_shape = artifact.image["input_image_shape"]
        if image_shape[:2] != [cameras[name + "_height"], cameras[name + "_width"]]:
            raise ValueError("Image and camera resolutions differ")
        source, faces = artifact.mesh()
        source_anchors = artifact.arrays["geometry__landmarks_mp"][0, indices]
        scale, rotation, translation = similarity(source_anchors, target_anchors)
        placed = scale * source @ rotation.T + translation
        aligned_anchors = scale * source_anchors @ rotation.T + translation
        print("Measuring unchanged face surface for photo " + name, flush=True)
        nearest, triangle_ids, counts = surface.closest(placed[region])
        distances = np.linalg.norm(nearest - placed[region], axis=1)
        errors = np.full(len(source), np.nan)
        errors[region] = distances / span
        npz = args.out / (name + "_surface.npz")
        np.savez_compressed(npz, placed_vertices=placed, faces=faces, region_vertex_ids=region,
                            nearest_scan_surface=nearest, nearest_scan_triangle_ids=triangle_ids,
                            distances_scan_units=distances, normalized_distances=distances / span,
                            candidate_counts=counts)
        board = args.out / (name + "_calibrated_review.png")
        review_board(image_path, *camera(name), placed, faces, errors, target_anchors, board,
                     protocol["display_saturation_normalized"])
        held_out_boards = []
        for other_name, other_artifact in photos.items():
            if other_name == name:
                continue
            other_path = host_path(other_artifact.image["input"])
            if sha(other_path) != other_artifact.image["input_sha256"]:
                raise ValueError("Held-out photo changed")
            other_board = args.out / (name + "_in_camera_" + other_name + ".png")
            review_board(other_path, *camera(other_name), placed, faces, errors, target_anchors,
                         other_board, protocol["display_saturation_normalized"])
            held_out_boards.append({"camera_photo": other_name, "file": other_board.name,
                                    "sha256": sha(other_board), "alignment_refit": False})
        regions = {}
        for key in ["nose", "lips", "forehead", "eye_region"]:
            selected = np.intersect1d(region, masks[key])
            if len(selected):
                regions[key] = {"vertices": len(selected), "normalized_error": stats(errors[selected])}
        report["images"].append({"photo": name, "input_sha256": artifact.image["input_sha256"],
            "artifact_sha256": artifact.image["sha256"], "similarity": {"scale": scale,
                "rotation": rotation.tolist(), "translation": translation.tolist()},
            "anchor_residual_normalized": stats(np.linalg.norm(aligned_anchors - target_anchors, axis=1) / span),
            "face_vertex_count": len(region), "surface_error_scan_units": stats(distances),
            "surface_error_normalized": stats(distances / span), "source_regions": regions,
            "surface_npz": npz.name, "surface_npz_sha256": sha(npz),
            "review_board": board.name, "review_board_sha256": sha(board),
            "held_out_camera_boards": held_out_boards})
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.out / "report.json"), "accuracy_status": report["accuracy_status"]}))


if __name__ == "__main__":
    main()
