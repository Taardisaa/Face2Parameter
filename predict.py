"""Predict the 205-dim face-parameter vector from an image (no card write-back).

This is the lightweight "give a picture -> get params" path. It needs only the
trained head + the DINOv2 backbone -- no HS2ABMX.exe, no template card, no mtcnn.
The output vector is in the SAME normalized space as labels.json.

For best results give a roughly cropped, front-facing face (training faces are
aligned 224x224). With --detector it will mtcnn-align first, if mtcnn_ort is installed.

Usage:
    .venv/Scripts/python.exe predict.py --config dinov2_vits14 --image face.png
    .venv/Scripts/python.exe predict.py --image face.png --head exp/.../head_epoch_30_step_XXXX.pth
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from config import get_config, resolve_head
from src.img_utils import load_face_rgb
from src.models.face2param import Face2Param


def predict(cfg, head_path, image_path, use_detector=True):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = Face2Param.from_checkpoint(cfg, head_path, map_location="cpu").to(device).eval()
    img = load_face_rgb(image_path, cfg.img_size, use_detector)
    chw = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    x = torch.from_numpy(chw).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model(x).squeeze(0).cpu().numpy().astype(np.float32)
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--image", required=True)
    ap.add_argument("--head", default=None,
                    help="head weights (.pth); defaults to latest trained exp checkpoint, "
                         "else the bundled release/ head")
    ap.add_argument("--out", default=None, help="defaults to outputs/<image>.json (+.npy)")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip mtcnn face alignment; aspect-preserving center-crop instead")
    args = ap.parse_args()

    cfg = get_config(args.config)
    head_path = resolve_head(cfg, args.head)
    vec = predict(cfg, head_path, args.image, use_detector=not args.no_detector)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_json = args.out or os.path.join("outputs", stem + "_out.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    np.save(os.path.splitext(out_json)[0] + ".npy", vec)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(vec.tolist(), f)

    print(f"head : {os.path.basename(head_path)}")
    print(f"image: {args.image}")
    print(f"vector: dim={vec.shape[0]} | min={vec.min():.3f} max={vec.max():.3f} "
          f"mean={vec.mean():.3f}")
    print(f"first 8: {np.array2string(vec[:8], precision=3, floatmode='fixed')}")
    print(f"saved -> {out_json}  (+ .npy)")


if __name__ == "__main__":
    main()
