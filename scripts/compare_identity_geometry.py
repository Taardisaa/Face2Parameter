"""Compare original MICA identity meshes and diagnostic neutral SMIRK copies.

This measures view consistency and displays shape, not accuracy against neutral
ground truth. No input changes, inference, fit, alignment, averaging or HS2 calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.model_bridge.artifact import ModelArtifact, host_path, sha
from smirk_audit_geometry import image, panel, write_image


def statistics(values):
    return {"mean": float(values.mean()), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smirk-manifest", type=Path, required=True)
    parser.add_argument("--mica-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--same-identity", action="store_true", required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new output directory")
    smirk, mica = ModelArtifact(args.smirk_manifest), ModelArtifact(args.mica_manifest)
    smirk.verify_sources()
    mica.verify_sources()
    # Compare actual buffers before treating these meshes as sharing coordinates.
    shared = {key: np.array_equal(smirk.state[key], mica.state[key]) for key in [
        "faces_tensor", "v_template", "posedirs", "J_regressor", "parents", "lbs_weights", "eye_pose", "neck_pose"]}
    shared["identity_shapedirs"] = np.array_equal(smirk.state["shapedirs"][:, :, :300], mica.state["shapedirs"][:, :, :300])
    if not all(shared.values()):
        raise ValueError("Models do not share verified geometry conventions: " + json.dumps(shared))
    if np.any(mica.state["eye_pose"]) or np.any(mica.state["neck_pose"]):
        raise ValueError("Canonical MICA defaults are not neutral")
    source_dir = host_path(smirk.manifest["model_dir"])
    os.chdir(source_dir)
    sys.path.insert(0, str(source_dir))
    from src.renderer.renderer import Renderer
    renderer = Renderer().cuda().eval()
    region = np.asarray(renderer.final_mask, np.int64)
    smirk_by_input = {r["input_sha256"]: i for i, r in enumerate(smirk.manifest["images"])}
    neutral = {"mica": [], "smirk": []}
    out.mkdir(parents=True, exist_ok=True)
    report = {"format": "source_identity_comparison_v1", "neutral_accuracy": "unverified_no_paired_neutral_truth",
        "same_identity_asserted": True, "alignment_refit": False, "averaging": False,
        "shared_buffers_exact": shared, "raw_parameters_modified": False,
        "smirk_manifest_sha256": sha(args.smirk_manifest), "mica_manifest_sha256": sha(args.mica_manifest),
        "evaluator_sha256": sha(Path(__file__)), "face_region_vertex_ids": region.tolist(), "images": [],
        "coordinates": "Shared unscaled FLAME coordinates; no normalization or mm relabeling"}
    for i, row in enumerate(mica.manifest["images"]):
        m = ModelArtifact(args.mica_manifest, i)
        s = ModelArtifact(args.smirk_manifest, smirk_by_input[row["input_sha256"]])
        mv = m.mesh()[0]
        replay_error = float(np.max(np.abs(m.replay(device="cuda")[0] - mv)))
        if replay_error > 1e-6:
            raise ValueError("MICA source replay differs from actual exported output")
        sv = s.replay(neutral_identity=True, device="cuda")[0]
        neutral["mica"].append(mv)
        neutral["smirk"].append(sv)
        columns = [panel(s.arrays["crop_rgb"], "Actual source crop")]
        fixed_cam = torch.tensor([[8., 0., 0.]], device="cuda")
        with torch.no_grad():
            for title, vertices in [("SMIRK neutral copy", sv), ("MICA original", mv)]:
                tensor = torch.as_tensor(vertices[None], device="cuda")
                for yaw in [-45, 0, 45]:
                    angle = np.deg2rad(yaw)
                    rotation = torch.tensor([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                        [-np.sin(angle), 0., np.cos(angle)]], device="cuda", dtype=tensor.dtype)
                    rgb = image(renderer(tensor @ rotation.T, fixed_cam)["rendered_img"])
                    columns.append(panel(rgb, f"{title} {yaw:+d}"))
        board = out / f"face_{i:04d}_identity.png"
        write_image(board, np.concatenate(columns, axis=1))
        npz = out / f"face_{i:04d}_neutral.npz"
        np.savez_compressed(npz, mica_original=mv, smirk_diagnostic_neutral=sv, faces=m.faces)
        report["images"].append({"input_sha256": row["input_sha256"], "mica_source_replay_max": replay_error,
            "model_difference": statistics(np.linalg.norm(mv[region] - sv[region], axis=1)),
            "board": board.name, "board_sha256": sha(board), "meshes": npz.name, "meshes_sha256": sha(npz)})
    if len(report["images"]) < 2:
        raise ValueError("At least two paired inputs required")
    report["view_consistency"] = {name: [statistics(np.linalg.norm(v[region] - meshes[0][region], axis=1))
        for v in meshes[1:]] for name, meshes in neutral.items()}
    report["limitations"] = ["Consistency cannot certify accuracy", "No neutral scan associated with these inputs",
        "SMIRK neutralization cannot prove expression/identity disentanglement", "No learned or analytical HS2 parameter mapping is produced"]
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "report.json"), "neutral_accuracy": report["neutral_accuracy"],
                      "view_consistency": report["view_consistency"]}))


if __name__ == "__main__":
    main()
