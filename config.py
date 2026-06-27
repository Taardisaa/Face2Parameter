"""Central configuration for Face2Parameter.

Replaces the per-script hardcoded ``initConfig`` methods. A single ``Config``
dataclass drives the backbone, head, data paths and training hyperparameters;
named presets are returned by ``get_config(name)``.

Presets
-------
- ``"smoke"``         : DummyBackbone + tiny dims + synthetic ``data_smoke/``. Runs
                        end-to-end with no downloads and no real dataset.
- ``"dinov2_vits14"`` : frozen DINOv2 ViT-S/14 (384-dim) + real ``data/``. The
                        intended training config once the dataset is ready.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


@dataclass
class Config:
    # --- identity ---
    exp_name: str = "dinov2_vits14_head"

    # --- backbone (feature extractor) ---
    # "dummy" | "dinov2_vits14" | "dinov2_vitb14"
    backbone: str = "dinov2_vits14"
    feature_dim: int = 384          # MUST match the backbone output; asserted at build time
    feature_mode: str = "cls"       # "cls" | "cls+patchmean"
    img_size: int = 224             # DINOv2 needs a multiple of 14; 224 -> 16x16 patches

    # --- head (MLP regressor) ---
    out_dim: int = 205              # FaceData 205-dim simplified/without-right vector
    hidden_dim: int = 1024
    num_layers: int = 4

    # --- data ---
    data_dir: str = "data/"         # holds features/, labels.json, {train,val,test}_features.txt
    batch_size: int = 32
    num_workers: int = 8
    # P(use aug_features/ instead of features/) per sample during training. Mixes the
    # realistic (aug_images) and in-game (images) domains. 0 disables aug sampling.
    aug_prob: float = 0.5

    # --- optimization ---
    lr: float = 5e-5
    weight_decay: float = 0.0
    num_epoch: int = 100
    seed: int = 123

    # --- bookkeeping ---
    device: str = "cuda"
    exp_root: str = "exp"
    ckpt_save_interval: int = 1
    val_interval: int = 1

    # split index of the loss: [:base_dim] = shapeValueFace, [base_dim:] = bone params
    base_dim: int = 54

    # Bundled HS2 character card used by infer.py when --template is not given. The
    # model only predicts the 205 face-shape params; the template supplies everything
    # else (body/hair/clothes), and we overwrite just the face fields.
    default_template: str = "assets/default_template.png"

    @property
    def exp_dir(self) -> str:
        return os.path.join(self.exp_root, self.exp_name)

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.exp_dir, "ckpts")

    @property
    def weights_dir(self) -> str:
        return os.path.join(self.exp_dir, "weights")

    @property
    def tb_log_dir(self) -> str:
        return os.path.join(self.exp_dir, "tb_logs")

    @property
    def features_dir(self) -> str:
        return os.path.join(self.data_dir, "features")


# --- backbone -> feature_dim registry (asserted against the real model at build time) ---
BACKBONE_FEATURE_DIM = {
    "dummy": None,            # configurable; set by the preset / caller
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
}


_PRESETS = {
    "dinov2_vits14": Config(
        exp_name="dinov2_vits14_head",
        backbone="dinov2_vits14",
        feature_dim=384,
        data_dir="data/",
        num_epoch=30,
    ),
    "dinov2_vitb14": Config(
        exp_name="dinov2_vitb14_head",
        backbone="dinov2_vitb14",
        feature_dim=768,
        data_dir="data/",
    ),
    # Offline skeleton: no downloads, no real data, tiny + fast.
    "smoke": Config(
        exp_name="smoke",
        backbone="dummy",
        feature_dim=64,
        feature_mode="cls",
        hidden_dim=128,
        num_layers=3,
        data_dir="data_smoke/",
        batch_size=8,
        num_workers=0,
        num_epoch=2,
        aug_prob=0.0,
        device="cuda",
    ),
}


def get_config(name: str = "dinov2_vits14", **overrides) -> Config:
    """Return a named preset, optionally overriding individual fields.

    Example: ``get_config("smoke", num_epoch=1, device="cpu")``.
    """
    if name not in _PRESETS:
        raise KeyError(f"Unknown config preset '{name}'. Available: {sorted(_PRESETS)}")
    cfg = _PRESETS[name]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def list_presets() -> list:
    return sorted(_PRESETS)
