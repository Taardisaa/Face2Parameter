"""Stage 1 (new): cache frozen-backbone features to disk.

Replaces ``extract_latentvec.py`` (which ran the VAE encoder). Runs the configured
backbone over a directory of aligned face images and writes one ``features/<name>.npy``
per image. The head trainer then trains on these cached vectors, keeping Stage 2 fast
and data-light.

Usage:
    .venv/Scripts/python.exe extract_features.py --config dinov2_vits14
    .venv/Scripts/python.exe extract_features.py --config smoke
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from config import get_config
from src.models.backbone import build_backbone


def list_images(images_dir: str) -> list:
    exts = ("*.png", "*.jpg", "*.jpeg")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
    return sorted(paths)


def read_image(path: str, img_size: int) -> np.ndarray:
    # imdecode(fromfile) tolerates non-ASCII (CJK) paths on Windows.
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # CHW


def run(cfg, images_dir: str | None = None) -> int:
    images_dir = images_dir or os.path.join(cfg.data_dir, "images")
    save_dir = cfg.features_dir
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(cfg).to(device).eval()

    paths = list_images(images_dir)
    if not paths:
        print(f"[extract_features] no images found under {images_dir}")
        return 0
    print(f"[extract_features] {len(paths)} images | backbone={cfg.backbone} "
          f"| feature_dim={cfg.feature_dim} | device={device}")

    n_done = 0
    batch_paths, batch_imgs = [], []

    def flush():
        nonlocal n_done
        if not batch_paths:
            return
        x = torch.from_numpy(np.stack(batch_imgs)).float().div(255).to(device)
        with torch.no_grad():
            feats = backbone(x).cpu().numpy()
        for p, f in zip(batch_paths, feats):
            name = os.path.splitext(os.path.basename(p))[0]
            np.save(os.path.join(save_dir, name + ".npy"), f.astype(np.float32))
        n_done += len(batch_paths)
        print(f"[extract_features] {n_done}/{len(paths)}")
        batch_paths.clear()
        batch_imgs.clear()

    for p in paths:
        try:
            batch_imgs.append(read_image(p, cfg.img_size))
            batch_paths.append(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[extract_features] skip {p}: {exc}")
            continue
        if len(batch_paths) >= cfg.batch_size:
            flush()
    flush()

    print(f"[extract_features] done -> {save_dir} ({n_done} vectors)")
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--images-dir", default=None,
                    help="override; defaults to <data_dir>/images")
    args = ap.parse_args()
    run(get_config(args.config), images_dir=args.images_dir)


if __name__ == "__main__":
    main()
