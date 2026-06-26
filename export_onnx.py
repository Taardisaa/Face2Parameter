"""Export the combined model (backbone + head) to ONNX.

Refactors ``onnx_export.py``. Full-model export embeds the DINOv2 backbone, which is
much heavier than the old VAE encoder and may be slow or hit unsupported ops; a
``--head-only`` fallback exports just the MLP head (input = feature vector).

Usage:
    .venv/Scripts/python.exe export_onnx.py --config dinov2_vits14 \
        --head exp/dinov2_vits14_head/weights/head_epoch_30_step_XXXX.pth \
        --out outputs/face2param.onnx
"""

from __future__ import annotations

import argparse
import os

import torch

from config import get_config
from src.models.face2param import Face2Param
from src.models.MLP.MLP import MLP


def export(cfg, head_path: str, out_path: str, head_only: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if head_only:
        model = MLP(cfg.feature_dim, cfg.out_dim, cfg.hidden_dim, cfg.num_layers).eval()
        ckpt = torch.load(head_path, map_location="cpu")
        model.load_state_dict(ckpt["weights"] if "weights" in ckpt else ckpt)
        dummy = torch.zeros(1, cfg.feature_dim)
        in_names, out_names = ["feature"], ["vector"]
    else:
        model = Face2Param.from_checkpoint(cfg, head_path, map_location="cpu").eval()
        dummy = torch.zeros(1, 3, cfg.img_size, cfg.img_size)
        in_names, out_names = ["image"], ["vector"]

    torch.onnx.export(model, (dummy,), out_path,
                      input_names=in_names, output_names=out_names,
                      dynamic_axes={in_names[0]: {0: "batch"}, out_names[0]: {0: "batch"}})
    print(f"[export_onnx] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", default="outputs/face2param.onnx")
    ap.add_argument("--head-only", action="store_true",
                    help="export only the MLP head (input = feature vector)")
    args = ap.parse_args()
    export(get_config(args.config), args.head, args.out, head_only=args.head_only)


if __name__ == "__main__":
    main()
