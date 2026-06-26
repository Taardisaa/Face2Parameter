# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Face2Parameter reconstructs the **facial parameters of game characters** (Illusion games: AiSyoujyo / Koikatsu Sunshine) from a face image. It is a two-stage pipeline:

1. **Stage 1 — frozen backbone** ([extract_features.py](extract_features.py) → `Backbone`): a pretrained **DINOv2 ViT-S/14** maps a 224×224 RGB face to a feature vector (384-dim CLS by default). Features are precomputed and cached to `features/<name>.npy`.
2. **Stage 2 — MLP head** ([train_head.py](train_head.py) → `MLP`): regresses the **205-dim facial parameter vector** from the cached feature.

Inference ([infer.py](infer.py)) ties them together via `Face2Param` (backbone + head) and writes the predicted vector back into a real game character card using `FaceData`.

> **History:** the project originally used a from-scratch **VAE** as the feature extractor. That path was retired to [legacy/](legacy/) (`vae_trainer.py`, `extract_latentvec.py`, `f2p_trainer.py`, `extractor.py`, `onnx_export.py`, `VAE/`) and replaced by the frozen DINOv2 backbone. See [the plan](../../.claude/plans/fluttering-orbiting-crystal.md) for rationale.

## Environment

Use the project venv `.venv` (Python 3.13, torch 2.11+cu130, verified on RTX 4080). Deps are pinned in [requirements.txt](requirements.txt). On this machine the venv was created with `--system-site-packages` to reuse the already-installed CUDA torch.

```bash
# verify the environment (imports + CUDA; add --with-dinov2 to probe the hub backbone)
.venv/Scripts/python.exe tools/check_env.py
```

**Windows console quirk:** torch's ONNX exporter prints emoji; on this GBK-codepage machine prefix ONNX commands with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` to avoid `UnicodeEncodeError`.

## Commands

All scripts are **config-driven** via [config.py](config.py) (`get_config(name)`), selected with `--config`. Presets: `smoke` (offline, DummyBackbone, tiny, synthetic `data_smoke/`), `dinov2_vits14`, `dinov2_vitb14`. Training auto-resumes from `exp/<exp_name>/ckpts/`.

```bash
# --- offline skeleton smoke (no dataset, no downloads) ---
.venv/Scripts/python.exe tools/make_synthetic_data.py        # build data_smoke/
.venv/Scripts/python.exe train_head.py     --config smoke    # train head on synthetic features
.venv/Scripts/python.exe extract_features.py --config smoke   # DummyBackbone over synthetic images

# --- real pipeline (once the dataset is ready) ---
.venv/Scripts/python.exe tools/gen_labels.py --cards-dir data/cards --out data/labels.json
.venv/Scripts/python.exe split_data.py                        # writes {train,val}.txt / *_features.txt
.venv/Scripts/python.exe extract_features.py --config dinov2_vits14   # cache DINOv2 features
.venv/Scripts/python.exe train_head.py      --config dinov2_vits14
.venv/Scripts/python.exe infer.py --config dinov2_vits14 --head <weights.pth> --image <img> --template <card.png>
PYTHONUTF8=1 .venv/Scripts/python.exe export_onnx.py --config dinov2_vits14 --head <weights.pth> --out outputs/m.onnx
```

Monitor training: `tensorboard --logdir exp`. There is no linter or test suite; the **skeleton acceptance** (the smoke block above producing a checkpoint under `exp/smoke/`) is the smoke test.

## Architecture notes / gotchas

- **`feature_dim` couples the two stages.** It must equal the backbone's output dim (DINOv2 ViT-S/14 = 384; `feature_mode="cls+patchmean"` doubles it). `build_backbone` in [src/models/backbone.py](src/models/backbone.py) **asserts** config vs. actual dim and raises on mismatch — so a wrong `feature_dim` fails loudly instead of silently.
- **The head is the generic `MLP`** in [src/models/MLP/MLP.py](src/models/MLP/MLP.py), reused unchanged (only `input_dim` differs). The head trains on cached features; for inference/export it is loaded onto `Face2Param` ([src/models/face2param.py](src/models/face2param.py)) which prepends the backbone.
- **DINOv2 is frozen and offline-gated.** `DINOv2Backbone` lazy-loads via `torch.hub` (one-time download) and normalizes `[0,1]` RGB internally with ImageNet stats. `DummyBackbone` (selected by `backbone="dummy"`) is a pure-torch pool+linear so the skeleton runs with no network and no real weights.
- **Stage-2 loss** is split MSE: `MSE(out[:54], lbl[:54]) + MSE(out[54:], lbl[54:])` — `[:54]` is the base `shapeValueFace`, `[54:]` is the bone-parameter block. The split index is `Config.base_dim`.
- **The data/card/label layer is the stable I/O contract** under [src/face_data_utils/](src/face_data_utils/) — kept verbatim from the VAE era:
  - `FaceData.to_vector(is_simplify=True, without_right=True, normalize=True, use_gaussian=False)` produces the **205-dim** label (54 base + masked params from 30 bones × 10). This is what [tools/gen_labels.py](tools/gen_labels.py) uses.
  - `FaceData.set_from_vector(..., denormalize=True, use_gaussian=False)` writes a predicted vector back into a card. **Train-time `normalize=True` must pair with infer-time `denormalize=True`** (and matching `use_gaussian`) or values drift — both are hardcoded consistently in `gen_labels.py`/`infer.py`.
  - Card write-back needs `HS2ABMX.exe` + stat JSONs (under `src/face_data_utils/`); face alignment needs `mtcnn_ort` (gitignored). **Head training needs none of these** — only cached features + `labels.json`.
- **Two parallel util trees still exist:** the canonical [src/face_data_utils/](src/face_data_utils/) (used by the new code) and the older near-duplicate [src/utils/](src/utils/) (whose `label_gen.py` emits an inconsistent 54-dim vector — superseded by `tools/gen_labels.py`). Edit the `face_data_utils` one.
- **Checkpoints:** `exp/<exp_name>/ckpts/ckpt_epoch_*_step_*.pth` (weights+optimizer, for resume) and `exp/<exp_name>/weights/head_epoch_*_step_*.pth` (head weights only, for inference/export). `_try_resume` picks the highest epoch.
- Images are read with `cv2.imdecode(np.fromfile(...))` (not `cv2.imread`) throughout to tolerate non-ASCII (CJK) paths on Windows.
- `data/`, `data_smoke/`, `exp/`, `outputs/`, `.venv/`, `taming/`, `src/mtcnn_ort/` are gitignored.
