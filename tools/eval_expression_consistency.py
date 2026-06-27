"""Measure expression-invariance: how stable are predicted params across expressions?

The real test of the ArcFace approach (see docs/expression-invariance.md). For a set of
identities, each photographed with several expressions, a good model predicts (nearly)
the SAME 205-dim params for every expression of the same person. We quantify the residual
wobble as the mean per-parameter std across each identity's variants (lower = more invariant).

Input layout: a directory with one subfolder per identity, each holding 2+ images:
    <dir>/alice/neutral.jpg
    <dir>/alice/smile.jpg
    <dir>/bob/...

Usage:
    .venv/Scripts/python.exe tools/eval_expression_consistency.py --dir <expr_dir> --config arcface
    # compare two backbones side by side:
    .venv/Scripts/python.exe tools/eval_expression_consistency.py --dir <expr_dir> --compare arcface dinov2_vits14
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from predict import latest_head, predict

_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def _identity_groups(root: str) -> dict:
    groups = {}
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        imgs = []
        for e in _EXTS:
            imgs += glob.glob(os.path.join(d, e))
        if len(imgs) >= 2:
            groups[sub] = sorted(imgs)
    return groups


def evaluate(cfg_name: str, groups: dict, use_detector: bool = True) -> dict:
    cfg = get_config(cfg_name)
    head = latest_head(cfg)
    per_identity_std = []
    for ident, imgs in groups.items():
        vecs = np.stack([predict(cfg, head, p, use_detector=use_detector) for p in imgs])
        # std per parameter across this identity's expressions, averaged over params
        per_identity_std.append(float(vecs.std(axis=0).mean()))
    arr = np.array(per_identity_std)
    return {"mean_std": float(arr.mean()), "max_std": float(arr.max()), "per_identity": per_identity_std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir of per-identity subfolders (2+ imgs each)")
    ap.add_argument("--config", default="arcface")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="config names to compare (overrides --config)")
    ap.add_argument("--no-detector", action="store_true")
    args = ap.parse_args()

    groups = _identity_groups(args.dir)
    if not groups:
        raise SystemExit(f"no identity subfolders with >=2 images under {args.dir}")
    print(f"identities: {len(groups)} | variants: {{ {', '.join(f'{k}:{len(v)}' for k,v in groups.items())} }}")

    configs = args.compare or [args.config]
    print(f"\n{'config':22s} {'mean_std':>10s} {'max_std':>10s}   (lower = more expression-invariant)")
    for name in configs:
        r = evaluate(name, groups, use_detector=not args.no_detector)
        print(f"{name:22s} {r['mean_std']:>10.4f} {r['max_std']:>10.4f}")


if __name__ == "__main__":
    main()
