"""Combined inference model: frozen backbone + MLP regression head.

Training happens in two stages (cache features -> train head), so the head is
trained separately on cached vectors. This module stitches the backbone and the
trained head together for end-to-end inference / ONNX export, mirroring the old
``Model`` in ``onnx_export.py`` (VAE encoder -> MLP) but with the DINOv2 backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.backbone import build_backbone
from src.models.MLP.MLP import MLP


class Face2Param(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.head = MLP(cfg.feature_dim, cfg.out_dim, cfg.hidden_dim, cfg.num_layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(img)
        return self.head(feat)

    def load_head(self, weights_path: str, map_location: str = "cpu") -> None:
        """Load head-only weights saved by ``train_head.py`` (``{'weights': ...}``)."""
        ckpt = torch.load(weights_path, map_location=map_location)
        state = ckpt["weights"] if isinstance(ckpt, dict) and "weights" in ckpt else ckpt
        missing, unexpected = self.head.load_state_dict(state, strict=False)
        print(f"[Face2Param] head loaded (missing={list(missing)}, "
              f"unexpected={list(unexpected)})")

    @classmethod
    def from_checkpoint(cls, cfg, head_weights_path: str,
                        map_location: str = "cpu") -> "Face2Param":
        model = cls(cfg)
        model.load_head(head_weights_path, map_location=map_location)
        model.eval()
        return model
