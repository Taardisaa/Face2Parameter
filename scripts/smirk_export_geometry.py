"""Export existing SMIRK parameters and FLAME geometry, without image synthesis.

Run in the existing WSL SMIRK environment. The encoder output is saved unchanged;
an additional head-local mesh removes only the predicted global head rotation.
No expression neutralization, geometry fitting or HS2 parameter prediction occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
from skimage.transform import warp

from smirk_neutralize import crop_face, imread_bgr, list_images


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arrays(values):
    result = {}
    for key, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Unexpected non-tensor model output: {key}")
        result[key] = value.detach().cpu().numpy().copy()
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite model output: {key}")
    return result


def save_npz(path, values):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smirk-dir", type=Path, required=True)
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    model_dir = args.smirk_dir.resolve()
    input_path, output_dir = args.input.resolve(), args.out.resolve()
    checkpoint = (args.checkpoint or model_dir / "pretrained_models/SMIRK_em1.pt").resolve()
    paths = [Path(p).resolve() for p in list_images(str(input_path))]
    if not paths:
        raise ValueError("No input images")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Use a new output directory; existing artifacts are preserved")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))
    from src.smirk_encoder import SmirkEncoder
    from src.FLAME.FLAME import FLAME
    from utils.mediapipe_utils import run_mediapipe

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
    encoder = SmirkEncoder().to(args.device).eval()
    weights = torch.load(checkpoint, map_location="cpu")
    encoder.load_state_dict({k.removeprefix("smirk_encoder."): v for k, v in weights.items()
                             if k.startswith("smirk_encoder.")}, strict=True)
    flame = FLAME().to(args.device).eval()
    state = arrays(flame.state_dict())
    save_npz(output_dir / "flame_state.npz", state)
    files = ["src/smirk_encoder.py", "src/FLAME/FLAME.py", "src/FLAME/lbs.py",
             "utils/mediapipe_utils.py", "assets/FLAME2020/generic_model.pkl",
             "assets/landmark_embedding.npy", "assets/l_eyelid.npy", "assets/r_eyelid.npy",
             "assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz",
             "assets/face_landmarker.task"]
    revision = subprocess.check_output(["git", "-C", str(model_dir), "rev-parse", "HEAD"],
                                       text=True).strip()
    manifest = {
        "format": "smirk_flame_raw_export_v1", "model_dir": str(model_dir),
        "git_revision": revision, "device": args.device,
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "sources": [{"path": str(model_dir / p), "sha256": sha(model_dir / p)} for p in files],
        "exporter": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "flame_state": {"file": "flame_state.npz", "sha256": sha(output_dir / "flame_state.npz")},
        "raw_parameters_modified": False, "extra_learned_conversion_model": False,
        "head_local_policy": "Copy encoder outputs; set only pose_params to zero; keep shape, expression, jaw and eyelids unchanged",
        "coordinates": "Raw FLAME coordinates and topology; no scale, centering, axis conversion or HS2 mapping",
        "images": [], "failures": [],
    }
    for index, path in enumerate(paths):
        try:
            image = imread_bgr(str(path))
            landmarks = run_mediapipe(image)
            if landmarks is None:
                raise ValueError("No face detected")
            transform = crop_face(image, landmarks[:, :2], scale=1.4, image_size=224)
            crop = warp(image, transform.inverse, output_shape=(224, 224),
                        preserve_range=True).astype(np.uint8)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float().div(255).to(args.device)
            with torch.no_grad():
                parameters = encoder(tensor)
                raw_parameters = arrays(parameters)
                geometry = arrays(flame(parameters))
                head_local = {key: value.clone() for key, value in parameters.items()}
                head_local["pose_params"] = torch.zeros_like(parameters["pose_params"])
                local_geometry = arrays(flame(head_local))
                unchanged = arrays(parameters)
                if any(not np.array_equal(raw_parameters[k], unchanged[k]) for k in raw_parameters):
                    raise RuntimeError("FLAME changed the raw encoder outputs")
            payload = {f"param__{k}": v for k, v in raw_parameters.items()}
            payload.update({f"geometry__{k}": v for k, v in geometry.items()})
            payload.update({f"head_local__{k}": v for k, v in local_geometry.items()})
            payload.update({"faces": state["faces_tensor"], "crop_rgb": rgb,
                            "crop_transform": transform.params, "detected_landmarks": landmarks})
            name = f"face_{index:04d}_{sha(path)[:12]}.npz"
            save_npz(output_dir / name, payload)
            manifest["images"].append({
                "input": str(path), "input_sha256": sha(path), "file": name,
                "sha256": sha(output_dir / name),
                "input_image_shape": list(image.shape),
                "parameter_fields": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                                     for k, v in raw_parameters.items()},
                "vertex_count": int(geometry["vertices"].shape[1]),
                "face_count": int(state["faces_tensor"].shape[0]),
            })
            print(f"Exported parameters and geometry: {name}", flush=True)
        except Exception as exc:
            manifest["failures"].append({"input": str(path), "error": str(exc)})
            print(f"Failed: {path.name}: {exc}", file=sys.stderr, flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
