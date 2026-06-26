"""Create a tiny synthetic dataset so the skeleton runs with no real data.

Produces, under the smoke config's ``data_dir`` (``data_smoke/``):
  - ``features/<name>.npy``  : random feature vectors (dim = cfg.feature_dim)
  - ``labels.json``          : random 205-dim target vectors
  - ``train_features.txt`` / ``val_features.txt`` : the split index files
  - ``images/<name>.png``    : random RGB images (so extract_features.py has input)

This validates the full wiring (dataset -> loader -> head -> loss -> checkpoint and
backbone -> feature cache) without the dataset, downloads, or the card serializer.

Usage:
    .venv/Scripts/python.exe tools/make_synthetic_data.py --n 64 --val 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from config import get_config


def make(n: int = 64, val: int = 8, img_size: int = 64, seed: int = 0) -> None:
    cfg = get_config("smoke")
    rng = np.random.default_rng(seed)

    root = cfg.data_dir
    feat_dir = os.path.join(root, "features")
    img_dir = os.path.join(root, "images")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    names = [f"sample_{i:05d}" for i in range(n)]
    labels = {}
    for name in names:
        np.save(os.path.join(feat_dir, name + ".npy"),
                rng.standard_normal(cfg.feature_dim).astype(np.float32))
        labels[name] = rng.standard_normal(cfg.out_dim).astype(np.float32).tolist()
        img = rng.integers(0, 256, size=(img_size, img_size, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, name + ".png"), img)

    with open(os.path.join(root, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f)

    val = min(val, n - 1)
    train_names, val_names = names[:-val], names[-val:]
    with open(os.path.join(root, "train_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_names) + "\n")
    with open(os.path.join(root, "val_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_names) + "\n")

    print(f"[make_synthetic_data] {root}: {n} samples "
          f"(train={len(train_names)}, val={len(val_names)}), "
          f"feature_dim={cfg.feature_dim}, out_dim={cfg.out_dim}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--val", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=64)
    args = ap.parse_args()
    make(n=args.n, val=args.val, img_size=args.img_size)


if __name__ == "__main__":
    main()
