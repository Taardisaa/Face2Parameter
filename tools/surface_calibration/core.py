"""CPU projection/ray and triangle correspondence scaffolds, no detectors/game IO.

All observations remain candidates. Geometry visibility is explicitly separated
from shader visibility and anatomical correspondence validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class ContractError(ValueError):
    pass


def finite_array(value, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ContractError(f"Expected shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ContractError("Non-finite numerical input")
    return result


def homogeneous_point(value):
    if abs(value[3]) < 1e-12:
        raise ContractError("Projection/unprojection has zero homogeneous W")
    return value[:3] / value[3]


@dataclass
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    max_distance: float


@dataclass
class Camera:
    width: int
    height: int
    view: np.ndarray
    projection: np.ndarray
    rect: np.ndarray
    ndc_depth: tuple[float, float] = (-1.0, 1.0)
    view_id: str = "unspecified"
    pixel_contract_validated: bool = False
    pose_pairing_validated: bool = False
    evidence: str = "unvalidated"

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int or self.width <= 0 or self.height <= 0:
            raise ContractError("Positive image dimensions required")
        self.view = finite_array(self.view, (4, 4))
        self.projection = finite_array(self.projection, (4, 4))
        self.rect = finite_array(self.rect, (4,))
        if self.rect[2] <= 0 or self.rect[3] <= 0:
            raise ContractError("Positive viewport rectangle dimensions required")
        depth = finite_array(self.ndc_depth, (2,))
        if depth[0] >= depth[1]:
            raise ContractError("CPU near/far NDC interval must be increasing")
        self.ndc_depth = tuple(depth.tolist())
        try:
            self.inverse_vp = np.linalg.inv(self.projection @ self.view)
        except np.linalg.LinAlgError as exc:
            raise ContractError("Singular camera matrix") from exc

    @classmethod
    def from_capture(cls, data: dict, *, certification: dict | None = None, pixel_certificate=None, diagnostic=False):
        c = data.get("capture_camera")
        if not isinstance(c, dict):
            raise ContractError("capture_camera required; Camera.main metadata is insufficient")
        if c.get("matrix_layout") != "row_major_16; column_vectors":
            raise ContractError("Unknown matrix layout; require row_major_16; column_vectors")
        if c.get("viewport_origin") != "bottom_left":
            raise ContractError("Unknown viewport origin")
        convention = c.get("image_coordinate_convention", "")
        if "top_left" not in convention:
            raise ContractError("A declared top-left image convention is required")
        cert = certification or {}
        pixels_valid = False
        pixel_evidence = None
        if pixel_certificate is not None:
            if __package__:
                from .pixel_certificate import PixelCertificate
            else:
                from pixel_certificate import PixelCertificate
            if not isinstance(pixel_certificate, PixelCertificate):
                raise ContractError("Pixel certificate must come from independent report verification")
            pixel_certificate.validate_capture(data)
            pixels_valid = True
            pixel_evidence = f"{pixel_certificate.report_sha256}:{pixel_certificate.case_id}"
        pair = data.get("paired_geometry") or {}
        signature = pair.get("pose_signature")
        paired = (
            data.get("paired_pose_unchanged") is True and isinstance(signature, str) and bool(signature.strip())
            and all(type(value) is int for value in [data.get("frame_count_before_render"), data.get("frame_count"), pair.get("frame_count")])
            and data.get("pose_signature_before_render") == signature
            and data.get("pose_signature_after_render") == signature
            and data.get("frame_count_before_render") == data.get("frame_count") == pair.get("frame_count")
        )
        if data.get("paired_pose_unchanged") is False:
            paired = False
        if not diagnostic and (not pixels_valid or not paired):
            raise ContractError("Unvalidated PNG pixel convention or pose pairing; use diagnostic mode explicitly")
        for name in ("world_to_camera", "projection"):
            if finite_array(c.get(name)).shape != (16,):
                raise ContractError(f"{name} must contain 16 row-major values")
        camera = cls(
            width=data["width"], height=data["height"],
            view=np.asarray(c["world_to_camera"]).reshape(4, 4),
            projection=np.asarray(c["projection"]).reshape(4, 4),
            rect=c["pixel_rect"], ndc_depth=c["cpu_ndc_depth_range"],
            view_id=str(data.get("view_id", data.get("path", "unspecified"))),
            pixel_contract_validated=pixels_valid, pose_pairing_validated=paired,
            evidence=pixel_evidence or cert.get("evidence_id", "unvalidated"),
        )
        if "camera_to_world" in c:
            inverse = finite_array(c["camera_to_world"]).reshape(4, 4)
            if not np.allclose(camera.view @ inverse, np.eye(4), atol=2e-5, rtol=2e-5):
                raise ContractError("camera_to_world is inconsistent with world_to_camera")
        return camera

    def _pixel_ndc(self, xy):
        u, v = finite_array(xy, (2,))
        # Canonical image pixel centers are at integer u/v, with top-left origin.
        x, y, w, h = self.rect
        nx = 2 * ((u + 0.5 - x) / w) - 1
        ny = 2 * ((self.height - v - 0.5 - y) / h) - 1
        if not (-1 <= nx <= 1 and -1 <= ny <= 1):
            raise ContractError("Pixel lies outside the active viewport")
        return nx, ny

    def ray(self, xy):
        nx, ny = self._pixel_ndc(xy)
        near = homogeneous_point(self.inverse_vp @ [nx, ny, self.ndc_depth[0], 1.0])
        far = homogeneous_point(self.inverse_vp @ [nx, ny, self.ndc_depth[1], 1.0])
        delta = far - near
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            raise ContractError("Degenerate near/far ray")
        return Ray(near, delta / length, length)

    def project(self, world):
        clip = self.projection @ self.view @ np.r_[finite_array(world, (3,)), 1.0]
        if clip[3] <= 1e-12:
            return {"status": "behind_camera", "xy": None}
        ndc = homogeneous_point(clip)
        x, y, w, h = self.rect
        xy = [x + (ndc[0] + 1) * w / 2 - 0.5,
              self.height - (y + (ndc[1] + 1) * h / 2) - 0.5]
        status = "in_view"
        if ndc[2] < self.ndc_depth[0] or ndc[2] > self.ndc_depth[1]:
            status = "outside_clip"
        elif not (-1 <= ndc[0] <= 1 and -1 <= ndc[1] <= 1):
            status = "outside_viewport"
        return {"status": status, "xy": xy, "ndc": ndc.tolist()}


@dataclass
class SurfaceMesh:
    renderer_path: str
    vertices: np.ndarray
    triangles: np.ndarray
    source_hash: str
    visible: bool = True
    submesh_ids: np.ndarray | None = None
    cull_mode: str = "unknown"
    world_policy_certified: bool = False
    visibility_uncertainties: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.vertices = finite_array(self.vertices)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ContractError("Vertices must be Nx3")
        raw_triangles = np.asarray(self.triangles)
        if raw_triangles.dtype.kind not in "iu" or raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3:
            raise ContractError("Triangles must be an integer Mx3 array")
        self.triangles = raw_triangles.astype(np.int64)
        if self.triangles.size and (self.triangles.min() < 0 or self.triangles.max() >= len(self.vertices)):
            raise ContractError("Triangle vertex index out of range")
        if self.cull_mode not in ("unknown", "off", "back", "front"):
            raise ContractError("Unknown cull mode")
        if self.submesh_ids is None:
            self.submesh_ids = np.full(len(self.triangles), -1, dtype=np.int64)
        if np.asarray(self.submesh_ids).shape != (len(self.triangles),):
            raise ContractError("Submesh labels must match triangles")

    def intersections(self, ray: Ray):
        if not self.visible or not len(self.triangles):
            return []
        points = self.vertices[self.triangles]
        edge1, edge2 = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        p = np.cross(ray.direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, p)
        valid = np.abs(determinant) > 1e-12
        if self.cull_mode == "back":
            valid &= determinant > 0
        elif self.cull_mode == "front":
            valid &= determinant < 0
        inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
        displacement = ray.origin - points[:, 0]
        u = np.einsum("ij,ij->i", displacement, p) * inverse
        q = np.cross(displacement, edge1)
        v = (q @ ray.direction) * inverse
        distance = np.einsum("ij,ij->i", edge2, q) * inverse
        eps = 1e-9
        valid &= (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (distance >= -eps) & (distance <= ray.max_distance + eps)
        hits = []
        for index in np.flatnonzero(valid):
            barycentric = np.array([1 - u[index] - v[index], u[index], v[index]])
            normal = np.cross(edge1[index], edge2[index])
            normal /= np.linalg.norm(normal)
            hits.append({
                "renderer_path": self.renderer_path, "source_geometry_sha256": self.source_hash,
                "triangle_id": int(index), "submesh_id": int(self.submesh_ids[index]),
                "vertex_ids": self.triangles[index].tolist(), "barycentric": barycentric.tolist(),
                "world": (barycentric @ points[index]).tolist(), "geometric_normal": normal.tolist(),
                "distance_from_near": float(distance[index]),
                "world_policy_certified": self.world_policy_certified,
                "visibility_uncertainties": self.visibility_uncertainties + (["culling_unknown"] if self.cull_mode == "unknown" else []),
            })
        return hits


def meshes_from_geometry(data: dict, *, candidate="scale_free_trs", certification=None, diagnostic=False):
    cert = certification or {}
    certified_hashes = cert.get("mesh_source_hashes", {})
    cert_candidate = cert.get("world_candidate")
    meshes = []
    for item in data.get("meshes", []):
        baked = item.get("baked", {})
        worlds = baked.get("world_candidates", {})
        if candidate not in worlds:
            raise ContractError(f"Missing world candidate {candidate}")
        chosen = worlds[candidate]
        path = item["renderer_path"]
        source_hash = item["source_geometry_sha256"]
        certified = chosen.get("validated") is True or (
            cert.get("world_policy_validated") is True and cert_candidate == candidate
            and certified_hashes.get(path) == source_hash
        )
        if not diagnostic and not certified:
            raise ContractError(f"World policy for {path} is not certified")
        triangles = np.asarray(baked["triangles"])
        if triangles.size % 3:
            raise ContractError("Flattened triangle count is not divisible by 3")
        triangles = triangles.reshape(-1, 3)
        if triangles.size == 0:
            triangles = np.empty((0, 3), dtype=np.int64)
        labels = []
        flattened = []
        for submesh in baked.get("submeshes", []):
            if submesh.get("topology") == "Triangles":
                indices = submesh["indices"]
                if len(indices) % 3:
                    raise ContractError("Invalid submesh triangle indices")
                flattened.extend(indices)
                labels.extend([submesh["submesh_index"]] * (len(indices) // 3))
        if labels and (flattened != triangles.reshape(-1).tolist()):
            raise ContractError("Baked submesh and flattened triangle order differ")
        uncertainties = ["shader_alpha_and_depth_unverified", "camera_layer_mask_unverified"]
        meshes.append(SurfaceMesh(
            renderer_path=path, vertices=chosen["vertices"], triangles=triangles, source_hash=source_hash,
            visible=item.get("enabled") is True and item.get("active_in_hierarchy") is True,
            submesh_ids=np.asarray(labels, dtype=np.int64) if labels else None,
            cull_mode=item.get("cull_mode", "unknown"), world_policy_certified=certified,
            visibility_uncertainties=uncertainties,
        ))
    if not meshes or len({mesh.renderer_path for mesh in meshes}) != len(meshes):
        raise ContractError("Nonempty scene with unique renderer paths required")
    return meshes


def all_intersections(meshes, ray):
    hits = [hit for mesh in meshes for hit in mesh.intersections(ray)]
    return sorted(hits, key=lambda hit: (hit["distance_from_near"], hit["renderer_path"], hit["triangle_id"]))


def observe(camera: Camera, meshes: list[SurfaceMesh], point: dict):
    name, xy = str(point["id"]), finite_array(point["xy"], (2,))
    result: dict[str, Any] = {
        "id": name, "view_id": camera.view_id, "xy": xy.tolist(),
        "pixel_contract_validated": camera.pixel_contract_validated,
        "pose_pairing_validated": camera.pose_pairing_validated,
        "anatomical_correspondence_validated": False,
    }
    kind = point.get("kind", "surface")
    if kind == "silhouette":
        camera._pixel_ndc(xy)
        return {**result, "kind": "silhouette", "status": "view_observation_only"}
    if kind != "surface":
        raise ContractError("Point kind must be surface or silhouette")
    ray = camera.ray(xy)
    hits = all_intersections(meshes, ray)
    allowed = point.get("allowed_renderer_paths")
    if allowed is not None and (not isinstance(allowed, list) or not allowed or not all(isinstance(path, str) for path in allowed)):
        raise ContractError("allowed_renderer_paths must be a nonempty list of paths")
    targets = [hit for hit in hits if allowed is None or hit["renderer_path"] in allowed]
    result.update(kind="surface", ray_origin=ray.origin.tolist(), ray_direction=ray.direction.tolist())
    if not targets:
        return {**result, "status": "no_target_intersection", "occlusion": "unknown"}
    hit = targets[0]
    blocker = hits[0] if hits and hits[0]["distance_from_near"] < hit["distance_from_near"] - 1e-7 else None
    tied = [h for h in hits if abs(h["distance_from_near"] - hit["distance_from_near"]) <= 1e-7]
    result.update(
        status="occluded" if blocker else "surface_candidate",
        surface=hit, occlusion="geometrically_occluded" if blocker else "no_geometric_blocker",
        blocker=blocker, visibility_kind="geometry_only",
        equal_distance_hit_count=len(tied), intersection_ambiguous=len(tied) > 1,
    )
    return result


def follow(surface: dict, meshes: list[SurfaceMesh]):
    mesh = next((m for m in meshes if m.renderer_path == surface["renderer_path"]), None)
    if mesh is None or mesh.source_hash != surface["source_geometry_sha256"]:
        raise ContractError("Renderer/source geometry identity changed; recalibration required")
    triangle = surface["triangle_id"]
    if isinstance(triangle, bool) or not isinstance(triangle, int) or not 0 <= triangle < len(mesh.triangles):
        raise ContractError("Triangle ID invalid for target mesh")
    vertex_ids = mesh.triangles[triangle]
    if vertex_ids.tolist() != surface["vertex_ids"]:
        raise ContractError("Triangle topology/order changed; recalibration required")
    barycentric = finite_array(surface["barycentric"], (3,))
    if not np.isclose(barycentric.sum(), 1, atol=1e-8) or barycentric.min() < -1e-8:
        raise ContractError("Invalid barycentric weights")
    points = mesh.vertices[vertex_ids]
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    length = np.linalg.norm(normal)
    if length < 1e-12:
        raise ContractError("Followed triangle became degenerate")
    return {**surface, "world": (barycentric @ points).tolist(), "geometric_normal": (normal / length).tolist(), "material_point_only": True, "renderer_visible": mesh.visible, "world_policy_certified": mesh.world_policy_certified, "visibility_uncertainties": mesh.visibility_uncertainties}


def validate_reprojection(surface: dict, camera: Camera, meshes: list[SurfaceMesh], *, observed_xy=None, max_error_px=2.0):
    if not np.isfinite(max_error_px) or max_error_px < 0:
        raise ContractError("max_error_px must be finite and nonnegative")
    current = follow(surface, meshes)
    projection = camera.project(current["world"])
    result = {
        "view_id": camera.view_id, "projection": projection,
        "pixel_contract_validated": camera.pixel_contract_validated,
        "pose_pairing_validated": camera.pose_pairing_validated,
        "anatomical_correspondence_validated": False,
        "world_policy_certified": current["world_policy_certified"],
        "independent_observation_present": observed_xy is not None,
        "visibility_kind": "geometry_only", "error_px": None,
    }
    if not current["renderer_visible"]:
        return {**result, "status": "target_renderer_not_visible"}
    if projection["status"] != "in_view":
        return {**result, "status": projection["status"]}
    ray = camera.ray(projection["xy"])
    hits = all_intersections(meshes, ray)
    target_distance = float(np.dot(np.asarray(current["world"]) - ray.origin, ray.direction))
    if not hits:
        return {**result, "status": "no_surface_intersection"}
    if hits[0]["distance_from_near"] < target_distance - 1e-6:
        return {**result, "status": "occluded", "blocker": hits[0]}
    if observed_xy is None:
        return {**result, "status": "projected_without_independent_observation"}
    error = float(np.linalg.norm(np.asarray(projection["xy"]) - finite_array(observed_xy, (2,))))
    return {**result, "status": "within_pixel_tolerance" if error <= max_error_px else "outside_pixel_tolerance", "error_px": error, "max_error_px": max_error_px}
