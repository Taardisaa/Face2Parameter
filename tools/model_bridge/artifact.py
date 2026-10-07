"""Read unchanged SMIRK/MICA outputs; reuse their original FLAME LBS code."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def host_path(path, *, wsl_distribution=None):
    """Resolve existing source provenance across this machine's Windows/WSL boundary."""
    path = str(path)
    if os.name == "nt" and path.startswith("/mnt/") and len(path) > 7 and path[6] == "/":
        return Path(path[5].upper() + ":/" + path[7:])
    if os.name == "nt" and path.startswith("/") and wsl_distribution:
        if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in wsl_distribution):
            raise ValueError("Invalid WSL distribution in source provenance")
        return Path("//wsl.localhost/" + wsl_distribution + path)
    if os.name != "nt" and len(path) > 2 and path[1] == ":":
        return Path("/mnt/" + path[0].lower() + "/" + path[3:].replace("\\", "/"))
    return Path(path)


class ModelArtifact:
    def __init__(self, manifest_path, image_index=0):
        self.path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("format") not in {"smirk_flame_raw_export_v1", "mica_flame_raw_export_v1"}:
            raise ValueError("Unsupported source-model artifact")
        if self.manifest.get("raw_parameters_modified") is not False:
            raise ValueError("Raw output preservation is required")
        if type(image_index) is not int or image_index < 0:
            raise ValueError("Explicit nonnegative image index required")
        self.image = self.manifest["images"][image_index]
        self.state = self._read(self.manifest["flame_state"])
        self.arrays = self._read(self.image)
        self.parameters = {k.removeprefix("param__"): v for k, v in self.arrays.items()
                           if k.startswith("param__")}
        self.faces = self.arrays["faces"]
        if not np.array_equal(self.faces, self.state["faces_tensor"]):
            raise ValueError("Export topology differs from its source model")
        expected = set(self.image["parameter_fields"])
        if set(self.parameters) != expected:
            raise ValueError("Model output fields differ from export contract")
        for key, value in self.parameters.items():
            definition = self.image["parameter_fields"][key]
            if list(value.shape) != definition["shape"] or str(value.dtype) != definition["dtype"]:
                raise ValueError(f"Model output shape/dtype changed: {key}")
        if self.manifest["format"] == "mica_flame_raw_export_v1":
            for key, definition in self.image["raw_output_fields"].items():
                value = self.arrays["output__" + key]
                if list(value.shape) != definition["shape"] or str(value.dtype) != definition["dtype"]:
                    raise ValueError("MICA raw output shape/dtype changed: " + key)
            if not np.array_equal(self.parameters["shape_params"], self.arrays["output__pred_shape_code"]) or not np.array_equal(
                    self.arrays["geometry__vertices"], self.arrays["output__pred_canonical_shape_vertices"]):
                raise ValueError("MICA raw outputs and bridge aliases differ")
            if not np.array_equal(self.arrays["head_local__vertices"], self.arrays["geometry__vertices"]):
                raise ValueError("Canonical MICA geometry was changed for head-local export")

    def _read(self, row):
        path = (self.path.parent / row["file"]).resolve()
        if path.parent != self.path.parent or sha(path) != row["sha256"]:
            raise ValueError("Artifact path/digest mismatch")
        with np.load(path, allow_pickle=False) as data:
            result = {key: data[key].copy() for key in data.files}
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Nonfinite exported arrays")
        return result

    def verify_sources(self):
        distribution = self.manifest.get("runtime", {}).get("wsl_distribution")
        for row in self.manifest["sources"]:
            if sha(host_path(row["path"], wsl_distribution=distribution)) != row["sha256"]:
                raise ValueError(f"Source model changed: {row['path']}")

    def replay(self, *, head_local=False, neutral_identity=False, device="cpu"):
        """Evaluate saved parameters with upstream LBS, using saved exact model buffers.

        This is the source decoder, not a learned or fitted conversion to HS2.
        """
        import torch
        self.verify_sources()
        mica = self.manifest["format"] == "mica_flame_raw_export_v1"
        path_end = "models/lbs.py" if mica else "src/FLAME/lbs.py"
        row = next(r for r in self.manifest["sources"] if r["path"].endswith(path_end))
        spec = importlib.util.spec_from_file_location("model_bridge_source_flame_lbs", host_path(row["path"]))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = {k: torch.as_tensor(v, device=device) for k, v in self.state.items()}
        params = {k: torch.as_tensor(v, device=device) for k, v in self.parameters.items()}
        batch = len(params["shape_params"])
        if mica:
            # Literal defaults from official models/flame.py: pose uses eye_pose
            # as its six-value zero/default field; expression has 100 components.
            eye = state["eye_pose"].expand(batch, -1)
            full_pose = torch.cat([eye[:, :3], state["neck_pose"].expand(batch, -1), eye[:, 3:], eye], dim=1)
            expression = torch.zeros((batch, 100), dtype=state["v_template"].dtype, device=device)
            beta = torch.cat([params["shape_params"], expression], dim=1)
            vertices, _ = module.lbs(beta, full_pose, state["v_template"].unsqueeze(0).expand(batch, -1, -1),
                state["shapedirs"], state["posedirs"], state["J_regressor"], state["parents"],
                state["lbs_weights"], dtype=state["v_template"].dtype)
            return vertices.detach().cpu().numpy()
        if neutral_identity:
            params = {k: v.clone() for k, v in params.items()}
            for key in ["pose_params", "expression_params", "jaw_params", "eyelid_params"]:
                params[key].zero_()
        pose = torch.zeros_like(params["pose_params"]) if head_local else params["pose_params"]
        full_pose = torch.cat([pose, state["neck_pose"].expand(batch, -1), params["jaw_params"],
                               state["eye_pose"].expand(batch, -1)], dim=1)
        beta = torch.cat([params["shape_params"], params["expression_params"]], dim=1)
        vertices, _ = module.lbs(beta, full_pose, state["v_template"].unsqueeze(0).expand(batch, -1, -1),
                                 state["shapedirs"], state["posedirs"], state["J_regressor"],
                                 state["parents"], state["lbs_weights"], dtype=state["v_template"].dtype)
        vertices = vertices + state["r_eyelid"].expand(batch, -1, -1) * params["eyelid_params"][:, 1:2, None]
        vertices = vertices + state["l_eyelid"].expand(batch, -1, -1) * params["eyelid_params"][:, 0:1, None]
        return vertices.detach().cpu().numpy()

    def mesh(self, *, head_local=False):
        prefix = "head_local" if head_local else "geometry"
        return self.arrays[f"{prefix}__vertices"][0].copy(), self.faces.copy()
