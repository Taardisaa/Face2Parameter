# Face2Parameter: Modeling facial parameters of game characters from images

Predict the **facial parameters of game characters** (Tested on HoneySelect2) directly from a face image, using a **frozen pretrained backbone** followed by an MLP
regression head. The predicted vector can be written straight back into a real game character card.

```
                       Stage 1: frozen backbone            Stage 2: MLP head
                      ┌───────────────────────┐         ┌──────────────────┐
  face image  ─────►  │  DINOv2 ViT-S/14 (def) │  ────►  │   4-layer MLP     │  ────►  205-dim
  224×224 RGB         │  or ArcFace (512-d)    │ feature │  (regressor)      │         param vector
                      │  frozen, cached to disk│  vector └──────────────────┘            │
                      └───────────────────────┘                                          ▼
                                                                          write back into HS2 card
                                                                          (54 base + bone params)
```

> The current pipeline replaces the original from-scratch **VAE** feature extractor with a **frozen
> pretrained backbone** (DINOv2 by default; an expression-invariant ArcFace variant is also
> available). The regression head and the data/card/label layer are unchanged. The retired VAE code
> lives in [`legacy/`](legacy/).

## 0. Introduction

The model is a two-stage pipeline:

1. **Stage 1 — frozen backbone.** A pretrained backbone maps a 224×224 RGB face image to a feature
   vector. The backbone is frozen, so features are precomputed once and cached to disk.
   - **DINOv2 ViT-S/14** (default, 384-dim) — a strong general-purpose self-supervised backbone.
   - **ArcFace** (`w600k_r50`, 512-dim) — a face-recognition embedding that is **expression-invariant
     by construction**, which fixes the "smiling photo → too-wide face" failure. See
     [docs/expression-invariance.md](docs/expression-invariance.md).
2. **Stage 2 — MLP head.** A small MLP regresses the **205-dim facial parameter vector** (54-dim base
   `shapeValueFace` + masked bone parameters) from the cached feature.

Using a strong pretrained backbone instead of a from-scratch VAE gives better features with far less
training and no reconstruction objective to tune. The backbone is a pluggable interface
([src/models/backbone.py](src/models/backbone.py)), so swapping DINOv2 ↔ ArcFace needs no change to
the head, training loop, or card write-back.

## 1. Installation

Clone the project:

```bash
git clone https://github.com/Taardisaa/Face2Parameter.git
cd Face2Parameter
```

Create a virtual environment and install the pinned dependencies. `torch`/`torchvision` are CUDA
builds and come from the PyTorch index:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Verify the environment (imports + CUDA; add `--with-dinov2` to also probe the hub backbone):

```bash
python tools/check_env.py
```

> Tested on Python 3.13 + PyTorch 2.11 (CUDA 13) on an RTX 4080. Adjust the CUDA tag (`cu130`) to
> match your driver. The `mtcnn_ort` face detector, the ArcFace ONNX, and the `HS2ABMX.exe` card
> serializer are only needed for the alignment / inference / ArcFace paths, not for DINOv2 head
> training. `onnxruntime-directml` is pinned for this Windows box; on Linux/CPU swap it for plain
> `onnxruntime` (or `onnxruntime-gpu` with a supported CUDA).

## 2. Configuration

Everything is config-driven via [`config.py`](config.py). Pick a preset with `--config`:

- `smoke` — offline skeleton (a `DummyBackbone`, tiny dims, synthetic data); runs with no downloads
  and no dataset. Useful to validate the wiring.
- `dinov2_vits14` — the default training config (frozen DINOv2 ViT-S/14, 384-dim).
- `dinov2_vitb14` — larger ViT-B/14 variant (768-dim).
- `arcface` — expression-invariant ArcFace backbone (512-dim). Trains/validates on the realistic
  (`aug_images/`) domain. See [docs/expression-invariance.md](docs/expression-invariance.md).

## 3. Data layout

The real-data path expects, under `data/` (gitignored):

```
data/
  cards/         # HS2 character-card PNGs (source of the labels)
  images/        # aligned 224x224 in-game face crops
  aug_images/    # realistic (stable-diffusion) renders of the same faces (same basenames)
  labels.json    # name -> 205-dim vector, produced from cards/
```

`extract_features.py` then caches features into `features<tag>/` (from `images/`) and
`aug_features<tag>/` (from `aug_images/`). Training mixes the two domains per-sample (`aug_prob`) so
the head sees both the clean in-game look and the realistic look it will face at inference.

## 4. Train

```bash
# (once the dataset is ready)
python tools/gen_labels.py --cards-dir data/cards --out data/labels.json   # 205-dim labels from cards
python tools/make_splits.py --config dinov2_vits14 --val 1000 --test 200    # train/val/test index files
python extract_features.py --config dinov2_vits14 --variant both            # Stage 1: cache both domains
python train_head.py      --config dinov2_vits14                            # Stage 2: train the MLP head
```

Training auto-resumes from the latest checkpoint under `exp/<exp_name>/ckpts/`. Monitor with
`tensorboard --logdir exp`. In practice ~15–30 head epochs are sufficient. To train the
expression-invariant variant, swap `--config arcface` (its features land in `*_arcface/` so they
don't collide with the DINOv2 cache).

Evaluate a trained head on the held-out test split (MSE / L2 / cosine on both feature domains):

```bash
python tools/eval_head.py --config dinov2_vits14 --split test
```

**Smoke run (no dataset, no downloads):**

```bash
python tools/make_synthetic_data.py
python train_head.py      --config smoke
python extract_features.py --config smoke
```

## 5. Inference

Two entry points share the same backbone + head:

**Image → 205-dim vector** (lightweight; needs only the head + backbone — no card serializer, no
template):

```bash
python predict.py --config dinov2_vits14 --image test/my.png   # writes outputs/my_out.json (+.npy)
```

**Image → character card** (writes the predicted params into a real HS2 card; needs `HS2ABMX.exe`
+ stat JSONs, and `mtcnn_ort` for alignment):

```bash
python infer.py --config dinov2_vits14 \
    --head exp/dinov2_vits14_head/weights/head_epoch_30_step_XXXX.pth \
    --image test/my.png --out outputs/
```

`--head` defaults to the latest weights in the config's `exp` dir; `--template` defaults to the
bundled [`assets/default_template.png`](assets/default_template.png) (the model only predicts the
face-shape params — the template supplies body/hair/clothes). Pass `--no-detector` to skip mtcnn
alignment and use an aspect-preserving center-crop instead.

Export the combined model to ONNX (use `--head-only` to export just the regression head):

```bash
python export_onnx.py --config dinov2_vits14 --head <weights.pth> --out outputs/face2param.onnx
```

## Known Limitations

1. **Expression leakage on smiling inputs.** A strong smile can bleed into the predicted geometry
   (typically a slightly off mouth/jaw), because the DINOv2 features encode expression as well as
   identity. The expression-invariant ArcFace backbone largely fixes this — see
   [docs/expression-invariance.md](docs/expression-invariance.md).
2. **Not meant for 2D anime images.** The intended inputs are HS2 character screenshots and real
   human photos. A 2D illustration can still be mapped to a plausible-looking face vector, but the
   result tends to be uncanny: for 2D→3D, the goal isn't geometric face-fitting so much as matching
   the character's *style* (hairstyle, outfit, accessories), which this face-shape pipeline doesn't
   model. See [docs/stylization-and-anime-inputs.md](docs/stylization-and-anime-inputs.md).
3. **Face-shape params only.** The model predicts the 54-dim `shapeValueFace` + bone params and
   nothing else. Other details — head/body proportions, hairstyle, skin tone, eyelashes, eyebrow
   detail, etc. — come from the template card and need manual touch-up.

## About the HS_FACE dataset

The [HS_FACE](https://pan.baidu.com/s/1yPftN5rmtY5QDF7G2RjN4A?pwd=p8qd) dataset is a collection of
approximately 14w facial images of game characters. It consists of three parts: 1. Facial images of
game characters directly sampled from the game (→ `images/`); 2. Images generated with stable
diffusion using the images in (1) as a condition, close to real people (→ `aug_images/`); 3. Facial
parameters of the game characters in (1) (→ `cards/` → `labels.json`).

## License

MIT. This project builds on the original
[ChasonJiang/Face2Parameter](https://github.com/ChasonJiang/Face2Parameter).
</content>
</invoke>
