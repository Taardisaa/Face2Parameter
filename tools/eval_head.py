"""Evaluate a trained head on a split (default: the held-out test set).

Reports MSE / L2 distance / cosine similarity in the normalized label space, on both
feature domains (game `features/` and realistic `aug_features/`) so we can see the
domain gap that matters for real-photo inference.

Usage:
    .venv/Scripts/python.exe tools/eval_head.py --config dinov2_vits14 --split test
    .venv/Scripts/python.exe tools/eval_head.py --config dinov2_vits14 --head <weights.pth>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.models.MLP.MLP import MLP


def latest_head(cfg) -> str:
    paths = glob.glob(os.path.join(cfg.weights_dir, "head_*.pth"))
    if not paths:
        raise FileNotFoundError(f"no head weights in {cfg.weights_dir}")
    return max(paths, key=lambda p: os.path.getmtime(p))


def load_split(cfg, split: str):
    with open(os.path.join(cfg.data_dir, f"{split}_features.txt"), encoding="utf-8") as f:
        names = [l.strip() for l in f if l.strip()]
    with open(os.path.join(cfg.data_dir, "labels.json"), encoding="utf-8") as f:
        all_labels = json.load(f)
    labels = np.stack([np.asarray(all_labels[n], dtype=np.float32) for n in names])
    return names, labels


@torch.no_grad()
def evaluate(cfg, head_path, split, feat_subdir, names, labels, device):
    model = MLP(cfg.feature_dim, cfg.out_dim, cfg.hidden_dim, cfg.num_layers).to(device).eval()
    ckpt = torch.load(head_path, map_location=device)
    model.load_state_dict(ckpt["weights"] if "weights" in ckpt else ckpt)

    feats, keep = [], []
    for i, n in enumerate(names):
        p = os.path.join(cfg.data_dir, feat_subdir, n + ".npy")
        if os.path.exists(p):
            feats.append(np.load(p))
            keep.append(i)
    if not feats:
        return None
    x = torch.from_numpy(np.stack(feats)).float().to(device)
    y = torch.from_numpy(labels[keep]).to(device)
    pred = model(x)

    mse = torch.nn.functional.mse_loss(pred, y).item()
    l2 = torch.nn.functional.pairwise_distance(pred, y).mean().item()
    cos = torch.nn.functional.cosine_similarity(pred, y, dim=1).mean().item()
    mae = (pred - y).abs().mean().item()
    return dict(n=len(keep), mse=mse, l2=l2, cos=cos, mae=mae)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--split", default="test")
    ap.add_argument("--head", default=None)
    args = ap.parse_args()

    cfg = get_config(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    head_path = args.head or latest_head(cfg)
    names, labels = load_split(cfg, args.split)
    print(f"head={os.path.basename(head_path)} | split={args.split} | n={len(names)}")

    for domain, sub in [("game", cfg.features_subdir), ("aug/realistic", cfg.aug_features_subdir)]:
        r = evaluate(cfg, head_path, args.split, sub, names, labels, device)
        if r is None:
            print(f"  {domain:14s}: (no features found in {sub}/)")
        else:
            print(f"  {domain:14s} (n={r['n']}): "
                  f"MSE {r['mse']:.5f} | L2 {r['l2']:.4f} | MAE {r['mae']:.5f} | cos {r['cos']:.4f}")


if __name__ == "__main__":
    main()
