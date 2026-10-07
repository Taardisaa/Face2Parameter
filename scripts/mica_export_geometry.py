"""Export official MICA identity code and canonical geometry unchanged.

Run in the existing WSL SMIRK environment. Calls official Arcface and Generator,
using official antelopev2 detection and ArcFace alignment. No HS2 fitting.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from smirk_export_geometry import arrays, save_npz, sha
from smirk_neutralize import list_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mica-dir", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--flame-model", type=Path, required=True)
    parser.add_argument("--landmark-embedding", type=Path, required=True)
    parser.add_argument("--in", dest="input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    model_dir, asset_dir, out = args.mica_dir.resolve(), args.assets.resolve(), args.out.resolve()
    model_path, lmk_path = args.flame_model.resolve(), args.landmark_embedding.resolve()
    paths = [Path(p).resolve() for p in list_images(str(args.input.resolve()))]
    if not paths or out.exists() and any(out.iterdir()):
        raise ValueError("Inputs and a new output directory are required")
    checkpoint, bundle = asset_dir / "mica.tar", asset_dir / "antelopev2.zip"
    if sha(checkpoint) != "4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b":
        raise ValueError("Official MICA checkpoint differs from audited download")
    if sha(bundle) != "7353a5fdca5a90e11d2792e0236032b2fe42adc1ea23eaef5cf8c8b57e7e9393":
        raise ValueError("Official antelopev2 bundle differs from audited download")
    detector_dir = asset_dir / "models/antelopev2"
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_path = detector_dir / "scrfd_10g_bnkps.onnx"
    with zipfile.ZipFile(bundle) as archive:
        data = archive.read("scrfd_10g_bnkps.onnx")
    if detector_path.exists():
        if detector_path.read_bytes() != data:
            raise ValueError("Existing detector differs from pinned bundle")
    else:
        with detector_path.open("xb") as stream:
            stream.write(data)
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))
    from configs.config import get_cfg_defaults
    from models.arcface import Arcface
    from models.generator import Generator
    from datasets.creation.util import get_arcface_input, get_center
    from insightface.app import FaceAnalysis
    from insightface.app.common import Face
    from insightface.utils import face_align

    cfg = get_cfg_defaults()
    cfg.model.flame_model_path = str(model_path)
    cfg.model.flame_lmk_embedding_path = str(lmk_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose CPU explicitly")
    # These are the exact model objects called by official MICA.encode/decode.
    # Avoid unrelated training-mask initialization, not the inference computation.
    arcface = Arcface().to(args.device).eval()
    generator = Generator(512, 300, cfg.model.n_shape, cfg.model.mapping_layers,
                          cfg.model, args.device).to(args.device).eval()
    weights = torch.load(checkpoint, map_location="cpu")
    arcface.load_state_dict(weights["arcface"], strict=True)
    generator.load_state_dict(weights["flameModel"], strict=True)
    del weights
    state = arrays(generator.generator.state_dict())
    # The detector computation and thresholds are official defaults. CPU runtime
    # is explicit; no replacement detector/recognition weights or landmark guesses.
    detector = FaceAnalysis(name="antelopev2", root=str(asset_dir), allowed_modules=["detection"],
                            providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(224, 224))
    out.mkdir(parents=True, exist_ok=True)
    save_npz(out / "flame_state.npz", state)
    source_paths = [model_dir / p for p in ["demo.py", "configs/config.py", "micalib/models/mica.py",
        "utils/landmark_detector.py", "datasets/creation/util.py", "models/arcface.py",
        "models/generator.py", "models/flame.py", "models/lbs.py"]]
    source_paths += [model_path, lmk_path, checkpoint, detector_path,
                     Path(inspect.getfile(FaceAnalysis)), Path(inspect.getfile(face_align)),
                     Path(inspect.getfile(type(detector.det_model)))]
    manifest = {"format": "mica_flame_raw_export_v1", "model_dir": str(model_dir),
        "git_revision": subprocess.check_output(["git", "-C", str(model_dir), "rev-parse", "HEAD"], text=True).strip(),
        "sources": [{"path": str(p), "sha256": sha(p)} for p in source_paths],
        "checkpoint": {"path": str(checkpoint), "sha256": sha(checkpoint)},
        "detector_bundle": {"url": "https://drive.google.com/uc?id=16PWKI_RjjbE4_kqpElG-YFqe8FpXjads",
                            "sha256": sha(bundle)},
        "flame_state": {"file": "flame_state.npz", "sha256": sha(out / "flame_state.npz")},
        "exporter": {"path": str(Path(__file__).resolve()), "sha256": sha(Path(__file__))},
        "raw_parameters_modified": False, "extra_learned_conversion_model": False,
        "parameter_aliases": {"shape_params": "pred_shape_code"},
        "decoder_policy": "Official Generator(shape) calls FLAME with its own default expression/pose; no SMIRK decoder or vertex fitting",
        "head_local_policy": "Official output is already canonical; unchanged geometry is reused",
        "coordinates": "Raw canonical FLAME coordinates; official demo multiplies by 1000 only at mesh export; not applied here",
        "inference_path": "Official Arcface -> F.normalize -> official Generator; strict complete checkpoint loads",
        "detector": {"name": "antelopev2", "provider": "CPUExecutionProvider", "size": [224, 224],
                     "selection": "official get_center", "recognizer": "MICA checkpoint Arcface, not InsightFace recognition ONNX"},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                    "wsl_distribution": os.environ.get("WSL_DISTRO_NAME"),
                    "numpy": np.__version__, "opencv": cv2.__version__}, "images": []}
    for i, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Unreadable photo: " + str(path))
        boxes, keypoints = detector.det_model.detect(image, max_num=0, metric="default")
        if not len(boxes) or keypoints is None:
            raise ValueError("No official detector face/landmarks: " + str(path))
        selected = get_center(boxes, image)
        face = Face(bbox=boxes[selected, :4], kps=keypoints[selected], det_score=boxes[selected, 4])
        blob, crop = get_arcface_input(face, image)
        with torch.no_grad():
            faceid = F.normalize(arcface(torch.as_tensor(blob[None], device=args.device)))
            vertices, shape = generator(faceid)
            raw = arrays({"pred_shape_code": shape, "pred_canonical_shape_vertices": vertices, "faceid": faceid})
            landmarks = generator.generator.compute_landmarks(vertices).detach().cpu().numpy().copy()
        payload = {"output__" + key: value for key, value in raw.items()}
        payload.update({"param__shape_params": raw["pred_shape_code"].copy(),
                        "geometry__vertices": raw["pred_canonical_shape_vertices"].copy(),
                        "geometry__landmarks_fan_3d": landmarks, "head_local__vertices": raw["pred_canonical_shape_vertices"].copy(),
                        "faces": state["faces_tensor"], "crop_bgr": crop, "arcface_blob": blob,
                        "detected_bbox": face.bbox, "detected_landmarks_5": face.kps})
        name = f"face_{i:04d}_{sha(path)[:12]}.npz"
        save_npz(out / name, payload)
        manifest["images"].append({"input": str(path), "input_sha256": sha(path), "input_image_shape": list(image.shape),
            "file": name, "sha256": sha(out / name), "parameter_fields": {"shape_params": {
                "shape": list(raw["pred_shape_code"].shape), "dtype": str(raw["pred_shape_code"].dtype)}},
            "raw_output_fields": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in raw.items()},
            "vertex_count": len(raw["pred_canonical_shape_vertices"][0]), "face_count": len(state["faces_tensor"])})
        print("Exported original MICA outputs: " + path.name, flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out / "manifest.json"), "photos": len(manifest["images"]),
                      "neutral_identity_accuracy": "unverified"}))


if __name__ == "__main__":
    main()
