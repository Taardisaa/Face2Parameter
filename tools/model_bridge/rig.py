"""Carry the original FLAME LBS state, without fitting or truncating weights."""
from __future__ import annotations

import base64
import importlib.util

import numpy as np
import torch

from .artifact import host_path


def replay_source_pose(artifact, pose):
    """Reference geometry from the original decoder, not the exported rig arrays.

    Identity/expression/eyelids remain the recorded model output. Only the
    explicit five-joint source pose is replaced for final rig acceptance.
    """
    artifact.verify_sources()
    mica = artifact.manifest["format"] == "mica_flame_raw_export_v1"
    suffix = "models/lbs.py" if mica else "src/FLAME/lbs.py"
    source = next(row for row in artifact.manifest["sources"] if row["path"].endswith(suffix))
    spec = importlib.util.spec_from_file_location("source_pose_reference_lbs", host_path(source["path"],
        wsl_distribution=artifact.manifest.get("runtime", {}).get("wsl_distribution")))
    lbs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbs)
    state = {k: torch.as_tensor(v) for k, v in artifact.state.items()}
    params = {k: torch.as_tensor(v) for k, v in artifact.parameters.items()}
    pose = np.asarray(pose, dtype=np.float32)
    if pose.shape != (15,) or not np.isfinite(pose).all():
        raise ValueError("Explicit finite five-joint source pose required")
    expression = torch.zeros((1, 100), dtype=params["shape_params"].dtype) if mica else params["expression_params"]
    with torch.no_grad():
        vertices, joints = lbs.lbs(torch.cat([params["shape_params"], expression], 1),
            torch.as_tensor(pose[None]), state["v_template"][None], state["shapedirs"],
            state["posedirs"], state["J_regressor"], state["parents"], state["lbs_weights"],
            dtype=state["v_template"].dtype)
        if not mica:
            vertices = vertices + state["r_eyelid"] * params["eyelid_params"][:, 1:2, None]
            vertices = vertices + state["l_eyelid"] * params["eyelid_params"][:, 0:1, None]
    return vertices[0].numpy(), joints[0].numpy()


def encode(values):
    values = np.ascontiguousarray(values, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite rig values")
    return {"encoding": "float32_le_base64", "shape": list(values.shape),
            "data": base64.b64encode(values.tobytes()).decode("ascii")}


def export_rig(artifact):
    artifact.verify_sources()
    mica = artifact.manifest["format"] == "mica_flame_raw_export_v1"
    suffix = "models/lbs.py" if mica else "src/FLAME/lbs.py"
    source = next(row for row in artifact.manifest["sources"] if row["path"].endswith(suffix))
    spec = importlib.util.spec_from_file_location("source_rig_lbs", host_path(source["path"],
        wsl_distribution=artifact.manifest.get("runtime", {}).get("wsl_distribution")))
    lbs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lbs)
    state = {k: torch.as_tensor(v) for k, v in artifact.state.items()}
    params = {k: torch.as_tensor(v) for k, v in artifact.parameters.items()}
    if not np.array_equal(artifact.state["parents"], [-1, 0, 1, 1, 1]):
        raise ValueError("Unimplemented source skeleton; expected audited five-joint FLAME")
    if mica:
        expression = torch.zeros((1, 100), dtype=params["shape_params"].dtype)
        pose = torch.cat([state["eye_pose"][:, :3], state["neck_pose"], state["eye_pose"][:, 3:], state["eye_pose"]], 1)
        post_skin = torch.zeros_like(state["v_template"])
    else:
        expression = params["expression_params"]
        pose = torch.cat([torch.zeros_like(params["pose_params"]), state["neck_pose"],
                          params["jaw_params"], state["eye_pose"]], 1)
        post_skin = state["r_eyelid"][0] * params["eyelid_params"][0, 1] + state["l_eyelid"][0] * params["eyelid_params"][0, 0]
    beta = torch.cat([params["shape_params"], expression], 1)
    with torch.no_grad():
        shaped = state["v_template"][None] + lbs.blend_shapes(beta, state["shapedirs"])
        joints = lbs.vertices2joints(state["J_regressor"], shaped)
    return {"format": "flame_lbs_rig_v1", "lbs_source_sha256": source["sha256"],
        "parents": artifact.state["parents"].tolist(), "original_pose": pose[0].tolist(),
        "v_shaped": encode(shaped[0].numpy()), "joints": encode(joints[0].numpy()),
        "posedirs": encode(artifact.state["posedirs"]), "weights": encode(artifact.state["lbs_weights"]),
        "post_skin_offsets": encode(post_skin.numpy()),
        "policy": "Original five-joint hierarchy, all weights, non-root rotation pose correctives, then post-skin eyelids; source shape/expression fixed"}
