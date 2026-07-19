# Beauty score (facial-attractiveness regressor)

`beauty_score(image) -> float` rates facial attractiveness on a 1–5 scale. It **reuses the
main pipeline unchanged** — frozen DINOv2 ViT-S/14 backbone → `MLP` head — with the head's
`out_dim` set to **1** instead of 205. Trained on **SCUT-FBP5500** (5500 real faces, 60-rater
mean beauty score in [1,5]).

## Result (frozen backbone + small MLP)

| metric | value | notes |
| --- | --- | --- |
| **Pearson r** | **0.867** | on the official 60/40 val split (n=2198) |
| **MAE** | **0.263** | on the 1–5 scale |

For reference the SCUT-FBP5500 paper's fine-tuned ResNeXt-50 reaches ~0.90 Pearson; we get
0.867 with a **frozen** backbone + a tiny head (no backbone fine-tuning), consistent with the
rest of this project.

## Domain caveat ⚠️

The model is trained on **real human faces**. Scores for **stylized / anime / rendered
game-character** faces are **out of distribution and not reliable** — feeding a game-character
card into it is an experiment, not a valid measurement. `beauty_score.py` prints this note on
every run.

## Reproduce

```bash
# 1. align SCUT images + build labels.json + official 60/40 split  (one-time, ~12 min)
.venv/Scripts/python.exe tools/prep_beauty_data.py
# 2. cache DINOv2 features                                          (CUDA)
.venv/Scripts/python.exe extract_features.py --config beauty_dinov2_vits14
# 3. train the 1-dim head (30 epochs)                               (CUDA)
.venv/Scripts/python.exe train_head.py --config beauty_dinov2_vits14
# 4. evaluate: quick (test only) OR full (train+test+per-subset, + optional OOD extras)
.venv/Scripts/python.exe beauty_score.py --eval
.venv/Scripts/python.exe tools/eval_beauty.py --extra tests/
```

Held-out test (n=2198): **Pearson 0.867 / MAE 0.263**; train (n=3294): 0.995 (some overfit).
Per-subset Pearson is balanced 0.85–0.89 (AF/AM/CF/CM).

Data lands under `face2beauty/data/` (`images/`, `features/`, `labels.json`,
`{train,val}_features.txt`); the trained head under
`exp/beauty_dinov2_vits14_head/weights/head_*.pth`. Both are gitignored.

## Use

```python
from beauty_score import beauty_score, BeautyScorer

beauty_score("face.jpg")            # one-shot -> float in [1,5]

scorer = BeautyScorer()             # load the model once, score many
scorer.score_many(["a.jpg", "b.jpg"])
```

CLI:

```bash
.venv/Scripts/python.exe beauty_score.py --image face.jpg      # one face
.venv/Scripts/python.exe beauty_score.py --image folder/       # rank a folder, high -> low
.venv/Scripts/python.exe beauty_score.py --image face.jpg --no-detector   # skip mtcnn align
```

## How it maps onto the existing pipeline

| piece | reuse |
| --- | --- |
| backbone | `src/models/backbone.py` `build_backbone` (frozen DINOv2, internal ImageNet norm) — unchanged |
| head | `src/models/MLP/MLP.py` `MLP` with `output_dim=1` — unchanged |
| stitch | `src/models/face2param.py` `Face2Param.from_checkpoint` — unchanged |
| preprocess/align | `src/img_utils.py` `load_face_rgb` (mtcnn align → 224 crop) — shared with training, so train crops == inference crops |
| features / dataset / trainer | `extract_features.py`, `FeatureDataset`, `train_head.py` — config-driven, no change except a 1-dim-safe `_loss` guard |
| inference | `predict.py` `_load_model` / `_embed_and_predict` — reused by `beauty_score.py` |

Net new code: one config preset (`beauty_dinov2_vits14`), a 1-dim `_loss` guard in
`train_head.py`, `tools/prep_beauty_data.py`, and `beauty_score.py`. Label handling: trained
on **raw** [1,5] scores (no normalize), output clamped to [1,5].
