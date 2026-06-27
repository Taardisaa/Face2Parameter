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
from torch.utils.data import DataLoader, Dataset

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


# variant -> image subdir (the feature output subdir comes from cfg, so it can be
# namespaced per backbone, e.g. features_arcface/)
VARIANT_IMAGES = {
    "game": "images",
    "aug": "aug_images",
}


class _ImageReadDataset(Dataset):
    """Parallel image decode (the bottleneck): workers decode/resize, main runs the backbone."""

    def __init__(self, paths, img_size):
        self.paths = paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            return p, read_image(p, self.img_size)  # CHW uint8 RGB
        except Exception:  # noqa: BLE001 - mark unreadable; skipped in main loop
            return "", np.zeros((3, self.img_size, self.img_size), dtype=np.uint8)


def _collate(batch):
    paths = [b[0] for b in batch]
    imgs = torch.from_numpy(np.stack([b[1] for b in batch]))
    return paths, imgs


def run(cfg, images_dir: str, save_dir: str, backbone=None, device=None) -> int:
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if backbone is None:
        backbone = build_backbone(cfg).to(device).eval()

    paths = list_images(images_dir)
    if not paths:
        print(f"[extract_features] no images found under {images_dir}")
        return 0
    print(f"[extract_features] {len(paths)} images | backbone={cfg.backbone} "
          f"| feature_dim={cfg.feature_dim} | device={device}")

    loader = DataLoader(_ImageReadDataset(paths, cfg.img_size),
                        batch_size=cfg.batch_size, num_workers=cfg.num_workers or 8,
                        collate_fn=_collate, persistent_workers=False)
    n_done = 0
    for batch_paths, imgs in loader:
        x = imgs.float().div(255).to(device)
        with torch.no_grad():
            feats = backbone(x).cpu().numpy()
        for p, f in zip(batch_paths, feats):
            if not p:
                continue
            name = os.path.splitext(os.path.basename(p))[0]
            np.save(os.path.join(save_dir, name + ".npy"), f.astype(np.float32))
        n_done += len(batch_paths)
        if n_done % (cfg.batch_size * 20) < cfg.batch_size:
            print(f"[extract_features] {n_done}/{len(paths)}")
    print(f"[extract_features] decoded {n_done} images")

    print(f"[extract_features] done -> {save_dir} ({n_done} vectors)")
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--variant", choices=["game", "aug", "both"], default="game",
                    help="game: images/->features/ ; aug: aug_images/->aug_features/")
    args = ap.parse_args()
    cfg = get_config(args.config)

    variants = ["game", "aug"] if args.variant == "both" else [args.variant]
    # build the (heavy) backbone once and reuse across variants
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    backbone = build_backbone(cfg).to(device).eval()
    feat_subs = {"game": cfg.features_subdir, "aug": cfg.aug_features_subdir}
    for v in variants:
        img_sub, feat_sub = VARIANT_IMAGES[v], feat_subs[v]
        print(f"=== variant '{v}': {img_sub}/ -> {feat_sub}/ ===")
        run(cfg, os.path.join(cfg.data_dir, img_sub),
            os.path.join(cfg.data_dir, feat_sub), backbone=backbone, device=device)


if __name__ == "__main__":
    main()
