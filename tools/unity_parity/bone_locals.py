"""Compare actual captured bone-local TRS with offline native/ABMX FK inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.hs2_mesh_deform import _fk_world, _trs  # noqa: E402


def proper_rotation(linear):
    u, _, vt = np.linalg.svd(linear)
    correction = np.eye(3)
    correction[-1, -1] = 1 if np.linalg.det(u @ vt) > 0 else -1
    return u @ correction @ vt


def compare_bone_locals(snapshot, rig, abmx=None):
    live_by_name = {}
    for transform in snapshot["transforms"]:
        live_by_name.setdefault(transform["name"], []).append(transform)
    live_by_id = {transform["id"]: transform for transform in snapshot["transforms"]}
    world = _fk_world(rig, snapshot["character"]["shape_value_face"], abmx)
    rows, missing, ambiguous = [], [], []
    for pid in rig._topo:
        bone = rig.bones[pid]
        matches = live_by_name.get(bone["name"], [])
        if not matches:
            missing.append(bone["name"])
            continue
        if len(matches) != 1:
            ambiguous.append(
                {"name": bone["name"], "paths": [m.get("path") for m in matches]}
            )
            continue
        live = matches[0]
        parent = bone["parent"]
        offline_local = (
            np.linalg.inv(world[parent]) @ world[pid] if parent in world else world[pid]
        )
        live_local = _trs(
            live["local_position"], live["local_rotation_xyzw"], live["local_scale"]
        )
        off_rotation = proper_rotation(offline_local[:3, :3])
        live_rotation = proper_rotation(live_local[:3, :3])
        cosine = np.clip((np.trace(off_rotation.T @ live_rotation) - 1) / 2, -1, 1)
        off_scale = np.linalg.norm(offline_local[:3, :3], axis=0)
        live_scale = np.linalg.norm(live_local[:3, :3], axis=0)
        live_parent = live_by_id.get(live.get("parent_id"))
        cached_parent = rig.bones.get(parent)
        rows.append(
            {
                "name": bone["name"],
                "path": live.get("path"),
                "transform_id": live["id"],
                "is_head_skin_bone": bone["name"] in rig.skin_bone_names,
                "cached_parent": cached_parent["name"] if cached_parent else None,
                "live_parent": live_parent["name"] if live_parent else None,
                "position_error": float(
                    np.linalg.norm(offline_local[:3, 3] - live_local[:3, 3])
                ),
                "rotation_angle_deg": float(np.degrees(np.arccos(cosine))),
                "scale_error": float(np.linalg.norm(off_scale - live_scale)),
                "matrix_max_abs": float(np.max(np.abs(offline_local - live_local))),
                "offline_position": offline_local[:3, 3].tolist(),
                "live_position": live_local[:3, 3].tolist(),
                "offline_scale": off_scale.tolist(),
                "live_scale": live_scale.tolist(),
                "offline_local": offline_local.flatten().tolist(),
                "live_local": live_local.flatten().tolist(),
            }
        )
    return {
        "scope": "local TRS comparison; actual locals include any applied modifiers and animation",
        "matched_count": len(rows),
        "missing_names": missing,
        "ambiguous_names": ambiguous,
        "max_position_error": max(
            (row["position_error"] for row in rows), default=None
        ),
        "max_rotation_angle_deg": max(
            (row["rotation_angle_deg"] for row in rows), default=None
        ),
        "max_scale_error": max((row["scale_error"] for row in rows), default=None),
        "max_head_skin_position_error": max(
            (row["position_error"] for row in rows if row["is_head_skin_bone"]),
            default=None,
        ),
        "max_head_skin_rotation_angle_deg": max(
            (row["rotation_angle_deg"] for row in rows if row["is_head_skin_bone"]),
            default=None,
        ),
        "max_head_skin_scale_error": max(
            (row["scale_error"] for row in rows if row["is_head_skin_bone"]),
            default=None,
        ),
        "bones": rows,
    }
