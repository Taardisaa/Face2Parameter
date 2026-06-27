"""Pluggable image feature extractor (backbone) for Face2Parameter.

The backbone replaces the old from-scratch VAE encoder. It maps a batch of
``[0, 1]`` RGB images ``[B, 3, H, W]`` to a feature vector ``[B, feature_dim]``.

Implementations
---------------
- ``DINOv2Backbone`` : frozen, pretrained DINOv2 ViT (loaded via ``torch.hub``).
- ``DummyBackbone``  : pure-torch pool + linear; no downloads, for the offline skeleton.

Use ``build_backbone(cfg)`` to construct the one named by ``cfg.backbone`` and
assert its real output dim matches ``cfg.feature_dim``.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv2's hub code warns once per layer that the optional xFormers speedup is
# absent; it silently falls back to standard attention (identical results). Mute it.
warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

# ImageNet statistics; backbones receive [0,1] RGB and normalize internally so the
# rest of the pipeline never has to know about the backbone's expected stats.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_DINOV2_HUB_NAME = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
}


class Backbone(nn.Module):
    """Interface: ``forward(img) -> [B, feature_dim]`` and a ``feature_dim`` property."""

    @property
    def feature_dim(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class _Normalize(nn.Module):
    def __init__(self, mean=_IMAGENET_MEAN, std=_IMAGENET_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class DINOv2Backbone(Backbone):
    """Frozen DINOv2 ViT feature extractor.

    feature_mode:
        "cls"          -> CLS token            (embed_dim)
        "cls+patchmean"-> [CLS, mean(patches)] (2 * embed_dim)
    """

    def __init__(self, name: str = "dinov2_vits14", feature_mode: str = "cls",
                 img_size: int = 224):
        super().__init__()
        if name not in _DINOV2_HUB_NAME:
            raise ValueError(f"Unknown DINOv2 variant '{name}'. "
                             f"Choose from {list(_DINOV2_HUB_NAME)}.")
        if img_size % 14 != 0:
            raise ValueError(f"DINOv2 needs img_size divisible by 14, got {img_size}.")
        self.feature_mode = feature_mode
        self.img_size = img_size
        self.normalize = _Normalize()

        # Lazy/heavy: downloads weights on first call. Kept out of module import.
        self.model = torch.hub.load("facebookresearch/dinov2", _DINOV2_HUB_NAME[name])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        embed_dim = int(self.model.embed_dim)
        self._feature_dim = embed_dim * (2 if feature_mode == "cls+patchmean" else 1)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @torch.no_grad()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-1] != self.img_size or img.shape[-2] != self.img_size:
            img = F.interpolate(img, size=(self.img_size, self.img_size),
                                mode="bilinear", align_corners=False)
        x = self.normalize(img)
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]
        if self.feature_mode == "cls+patchmean":
            patch_mean = out["x_norm_patchtokens"].mean(dim=1)
            return torch.cat([cls, patch_mean], dim=1)
        return cls


class DummyBackbone(Backbone):
    """Offline stand-in: adaptive-pool the image then project to feature_dim.

    Trainable-free in spirit (it just needs to *run*); used so the skeleton works
    with no network access and no real backbone weights.
    """

    def __init__(self, feature_dim: int = 64, pool: int = 8):
        super().__init__()
        self._feature_dim = feature_dim
        self.pool = nn.AdaptiveAvgPool2d(pool)
        self.proj = nn.Linear(3 * pool * pool, feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = self.pool(img).flatten(1)
        return self.proj(x)


def build_backbone(cfg) -> Backbone:
    """Construct the backbone named by ``cfg.backbone`` and verify its feature dim."""
    if cfg.backbone == "dummy":
        backbone = DummyBackbone(feature_dim=cfg.feature_dim)
    elif cfg.backbone in _DINOV2_HUB_NAME:
        backbone = DINOv2Backbone(name=cfg.backbone, feature_mode=cfg.feature_mode,
                                  img_size=cfg.img_size)
    else:
        raise ValueError(f"Unknown backbone '{cfg.backbone}'.")

    if backbone.feature_dim != cfg.feature_dim:
        raise ValueError(
            f"Config feature_dim={cfg.feature_dim} does not match backbone "
            f"'{cfg.backbone}' (feature_mode={getattr(cfg, 'feature_mode', None)}) "
            f"output dim {backbone.feature_dim}. Update Config.feature_dim."
        )
    return backbone
