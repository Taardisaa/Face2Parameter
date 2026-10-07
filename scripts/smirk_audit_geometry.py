"""Inspect saved source geometry against its input, independently of HS2.

Uses SMIRK's own camera, landmarks and renderer. No image generator, shape fit,
camera refit or inference changes. Detector reprojection is a consistency check,
not independent 3D truth. Run in the existing WSL SMIRK environment.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.model_bridge.artifact import ModelArtifact, host_path, sha


def image(tensor):
    return (tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)


def panel(rgb, title):
    canvas = np.full((254, 224, 3), 245, dtype=np.uint8)
    canvas[:224] = rgb
    cv2.putText(canvas, title, (5, 244), cv2.FONT_HERSHEY_SIMPLEX, .4, (25, 25, 25), 1)
    return canvas


def write_image(path, rgb):
    if path.exists():
        raise FileExistsError(path)
    success, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError("PNG encoding failed")
    path.write_bytes(encoded.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--same-identity", action="store_true",
                        help="Explicit caller assertion; compare neutral meshes across these photos")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new audit output directory")
    first = ModelArtifact(manifest)
    first.verify_sources()
    model_dir = host_path(first.manifest["model_dir"])
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))
    from src.FLAME.FLAME import FLAME
    from src.renderer.renderer import Renderer
    from src.renderer.util import batch_orth_proj

    # Read the literal training landmark ordering without importing its dataset
    # and unrelated augmentation dependencies. Do not invent a nearest-point map.
    tree = ast.parse((model_dir / "datasets/base_dataset.py").read_text())
    training_indices = next(ast.literal_eval(node.value) for node in tree.body
                            if isinstance(node, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == "mediapipe_indices" for t in node.targets))
    with np.load(model_dir / "assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz",
                 allow_pickle=False) as embedding:
        indices = embedding["landmark_indices"].astype(np.int64)
    if not np.array_equal(indices, training_indices):
        raise ValueError("Asset and training landmark order disagree")
    flame = FLAME().to(args.device).eval()
    flame.load_state_dict({k: torch.as_tensor(v, device=args.device) for k, v in first.state.items()}, strict=True)
    renderer = Renderer().to(args.device).eval()
    out.mkdir(parents=True, exist_ok=True)
    report = {"format": "smirk_source_geometry_audit_v1", "manifest_sha256": sha(manifest),
              "model_revision": first.manifest["git_revision"], "raw_parameters_modified": False,
              "camera_refit": False, "neural_image_generator": False,
              "accuracy_status": "unverified_3d", "images": [],
              "diagnostic_sources": [{"path": str(model_dir / p), "sha256": sha(model_dir / p)}
                  for p in ["src/renderer/renderer.py", "src/renderer/util.py", "datasets/base_dataset.py",
                            "assets/head_template.obj", "assets/FLAME_masks/FLAME_masks.pkl"]]}
    neutral_meshes = []
    for index in range(len(first.manifest["images"])):
        artifact = ModelArtifact(manifest, index)
        arrays = artifact.arrays
        params = {k: torch.as_tensor(v.copy(), device=args.device) for k, v in artifact.parameters.items()}
        with torch.no_grad():
            posed = torch.as_tensor(arrays["geometry__vertices"].copy(), device=args.device)
            lm = torch.as_tensor(arrays["geometry__landmarks_mp"].copy(), device=args.device)
            rendering = renderer(posed, params["cam"])["rendered_img"]
            ndc = batch_orth_proj(lm, params["cam"])
            ndc[:, :, 1:] *= -1
            predicted = (ndc[0, :, :2].cpu().numpy() + 1) * 224 / 2
            # Neutral identity diagnostic: retain shape coefficients unchanged,
            # explicitly zero expression/jaw/eyelids and global rotation in a copy.
            neutral = {k: v.clone() for k, v in params.items()}
            for key in ["expression_params", "jaw_params", "eyelid_params", "pose_params"]:
                neutral[key].zero_()
            neutral_output = flame(neutral)["vertices"]
            neutral_meshes.append(neutral_output[0].cpu().numpy().copy())
            views = []
            fixed_cam = torch.tensor([[8.0, 0., 0.]], device=args.device)
            for degrees in [-45, 0, 45]:
                angle = np.deg2rad(degrees)
                rotation = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                                          [-np.sin(angle), 0, np.cos(angle)]], dtype=neutral_output.dtype,
                                        device=args.device)
                views.append(image(renderer(neutral_output @ rotation.T, fixed_cam)["rendered_img"]))
        detected = arrays["detected_landmarks"][indices, :2]
        homogeneous = np.c_[detected, np.ones(len(detected))]
        observed = (homogeneous @ arrays["crop_transform"].T)[:, :2]
        if predicted.shape != observed.shape:
            raise ValueError("Landmark correspondence shape mismatch")
        error = np.linalg.norm(predicted - observed, axis=1)
        crop = arrays["crop_rgb"]
        overlay = crop.copy()
        for a, b in zip(observed, predicted):
            aa, bb = tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))
            cv2.line(overlay, aa, bb, (255, 210, 60), 1)
            cv2.circle(overlay, aa, 1, (40, 220, 80), -1)
            cv2.circle(overlay, bb, 1, (220, 40, 180), -1)
        shaded = image(rendering)
        mask = np.any(shaded != 0, axis=-1)
        composite = crop.copy()
        composite[mask] = (.55 * shaded[mask] + .45 * crop[mask]).astype(np.uint8)
        panels = [panel(crop, "Input crop"), panel(shaded, "Raw posed geometry"),
                  panel(composite, "Geometry overlay"), panel(overlay, "Green:det / magenta:mesh")]
        panels += [panel(v, f"Neutral identity yaw {yaw:+d}") for v, yaw in zip(views, [-45, 0, 45])]
        file = out / f"face_{index:04d}_audit.png"
        write_image(file, np.concatenate(panels, axis=1))
        report["images"].append({"input_sha256": artifact.image["input_sha256"], "artifact_sha256": artifact.image["sha256"],
            "audit_image": str(file), "audit_image_sha256": sha(file),
            "landmark_count": len(indices), "landmark_ids": indices.tolist(),
            "landmark_error_crop_px": {"mean": float(error.mean()), "median": float(np.median(error)),
                                        "p95": float(np.quantile(error, .95)), "max": float(error.max())},
            "per_landmark_error_px": error.tolist(),
            "interpretation": "Agreement with the same detector/crop and model landmark supervision; not 3D truth or a likeness certificate"})
    if args.same_identity:
        if len(neutral_meshes) < 2:
            raise ValueError("At least two explicitly paired photos required")
        face_indices = np.asarray(renderer.final_mask, dtype=np.int64)
        distances = [np.linalg.norm(neutral_meshes[i][face_indices] - neutral_meshes[0][face_indices], axis=1)
                     for i in range(1, len(neutral_meshes))]
        report["same_identity_consistency"] = {"caller_asserted_pairing": True,
            "neutralization": "Only shape retained; expression/jaw/eyelids/global pose zero in diagnostic copies",
            "alignment_refit": False, "coordinates": "Unscaled FLAME coordinates; no assumed mm conversion",
            "face_vertex_ids": face_indices.tolist(),
            "against_first_image": [{"mean": float(d.mean()), "p95": float(np.quantile(d, .95)),
                                     "max": float(d.max())} for d in distances],
            "interpretation": "View consistency only. Agreement cannot prove correctness; disagreement does not identify which estimate is correct."}
    report["missing_evidence"] = ["Independent observed landmarks/contours", "Held-out calibrated views",
                                  "Registered paired scan accuracy with explicit face region and unit/alignment convention"]
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(out / "report.json"), "images": len(report["images"]),
                      "accuracy_status": report["accuracy_status"]}))


if __name__ == "__main__":
    main()
