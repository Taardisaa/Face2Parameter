"""Explicit metric/asset/scope contracts for offline complete-head targets."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from src.hs2_mesh_deform import HeadRig
from src.hs2_deform_torch import TorchHeadRig
from surface import content_hash, topology_hash

CHIN = "cf_J_ChinTip_s"
SCOPE = {"surface_group": "o_head", "expression_blendshapes": "excluded", "external_ancestors": "excluded", "body_pose": "cached_rest"}
AB_IDENTITY = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], float)


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    metadata: dict


def validate_arrays(vertices, faces):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Expected finite nonempty Nx3 vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Expected nonempty integer triangle topology")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Triangle index outside vertex array")
    return vertices, faces.astype(np.int64)


def cache_fingerprint(rig):
    root, head = Path(rig.root_dir), Path(rig.data_dir)
    paths = [head / name for name in ("o_head_mesh.npz", "skeleton.json", "anmShapeHead.json")]
    paths += [root / name for name in ("enums.json", "customhead.json", "update_eqns.json")]
    rows = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    return {"provenance": "extracted cached HeadRig assets + shape equations, no live runtime assertion",
            "files": rows, "sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()}


def ab_array(trig, modifier=None):
    modifier = modifier or {}
    if set(modifier) - {CHIN}:
        raise ValueError("ABMX bones other than the narrow chin probe are unverified and disabled")
    result = np.tile(AB_IDENTITY, (len(trig.ab_names), 1))
    if modifier:
        if CHIN not in trig.ab_names or CHIN not in trig.name2bone:
            raise ValueError("Chin bone missing from this rig")
        row = modifier[CHIN]
        if set(row) != {"scale", "length", "position", "rotation"}:
            raise ValueError("ABMX modifier requires explicit scale/length/position/rotation")
        values = np.r_[row["scale"], row["length"], row["position"], row["rotation"]]
        if values.shape != (10,) or not np.isfinite(values).all():
            raise ValueError("ABMX modifier must contain ten finite actual values")
        result[trig.ab_names.index(CHIN)] = values
    return result


def ab_dict(chin):
    return {CHIN: {"scale": chin[:3].tolist(), "length": float(chin[3]),
                   "position": chin[4:7].tolist(), "rotation": chin[7:10].tolist()}}


def rig_mesh(head_id, native59, *, profile="vanilla", abmx=None, device="cpu"):
    if not isinstance(head_id, int) or isinstance(head_id, bool) or head_id not in (0, 1, 2, 3):
        raise ValueError("This cached comparison supports explicit integer head IDs 0, 1, 2, 3")
    native = np.asarray(native59, float)
    if native.shape != (59,) or not np.isfinite(native).all():
        raise ValueError("Rig target needs all 59 finite actual native values")
    rig = HeadRig(head_id, sampling_profile=profile)
    trig = TorchHeadRig(rig, device=device, dtype=torch.float64)
    modifiers = ab_array(trig, abmx)
    with torch.no_grad():
        vertices = trig(torch.as_tensor(native[None], dtype=trig.dtype, device=trig.device),
                        torch.as_tensor(modifiers[None], dtype=trig.dtype, device=trig.device))[0].cpu().numpy()
    vertices, faces = validate_arrays(vertices, rig.faces)
    metadata = {"head_id": int(head_id), "units": "hs2_cache_units", "coordinate_frame": "hs2_cached_head_fk",
                "scope": dict(SCOPE), "asset": cache_fingerprint(rig),
                "content_sha256": content_hash(vertices, faces), "topology_sha256": topology_hash(faces),
                "native59": native.tolist(), "abmx": abmx or {}, "sampling_profile": profile,
                "pose_removal": "none; cached FK frame, no fitted rotation/translation/scale"}
    return Mesh(vertices, faces, metadata), rig, trig


def proper_rigid(value):
    matrix = np.asarray(value, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError("Declared pose transform needs finite affine 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8):
        raise ValueError("Declared pose transform may contain only proper rotation/translation, no scale/reflection/shear")
    return matrix


def load_target(config, *, device="cpu"):
    if config.get("kind") == "rig_target":
        mesh, _, _ = rig_mesh(config["head_id"], config["native59"], profile=config.get("sampling_profile", "vanilla"),
                              abmx=config.get("abmx"), device=device)
        return mesh
    if config.get("kind") != "full_mesh" or config.get("schema_version") != 1:
        raise ValueError("Target must be explicit rig_target or full_mesh schema 1; bone centers/landmarks are not surfaces")
    vertices, faces = validate_arrays(config["vertices"], config["faces"])
    metadata = dict(config["metadata"])
    if metadata.get("content_sha256") != content_hash(vertices, faces) or metadata.get("topology_sha256") != topology_hash(faces):
        raise ValueError("Full mesh content/topology hashes missing or inconsistent")
    if not metadata.get("asset", {}).get("provenance") or not metadata.get("asset", {}).get("sha256"):
        raise ValueError("Unknown target asset provenance")
    asset_hash = metadata["asset"]["sha256"]
    if not isinstance(asset_hash, str) or len(asset_hash) != 64 or any(ch not in "0123456789abcdef" for ch in asset_hash):
        raise ValueError("Asset evidence SHA-256 must be a complete lowercase hex digest")
    if metadata.get("units") != "hs2_cache_units" or metadata.get("coordinate_frame") != "hs2_cached_head_fk":
        raise ValueError("Unknown/incompatible units or coordinate frame; no automatic metric scale calibration")
    if metadata.get("scope") != SCOPE:
        raise ValueError("Target expression/ancestor/body-pose/surface scope differs from cached rig scope")
    if "to_comparison_rigid" in config:
        if not config.get("pose_provenance"):
            raise ValueError("Declared rigid pose removal requires provenance")
        matrix = proper_rigid(config["to_comparison_rigid"])
        vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        metadata["pose_removal"] = {"matrix": matrix.tolist(), "provenance": config["pose_provenance"], "fitted": False}
    else:
        metadata["pose_removal"] = "none; input declares comparison frame"
    metadata["extra_groups_not_evaluated"] = [group.get("name") for group in config.get("extra_groups", [])]
    metadata["evaluated_content_sha256"] = content_hash(vertices, faces)
    return Mesh(vertices, faces, metadata)


def full_mesh_config(mesh):
    return {"schema_version": 1, "kind": "full_mesh", "vertices": mesh.vertices.tolist(),
            "faces": mesh.faces.tolist(), "metadata": mesh.metadata, "extra_groups": []}
