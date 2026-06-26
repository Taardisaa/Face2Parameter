# Face2Parameter: Modeling facial parameters of game characters from images

Predict the **facial parameters of game characters** (Illusion games: AiSyoujyo / Koikatsu
Sunshine) directly from a face image, using a **frozen pretrained DINOv2 backbone** followed by
an MLP regression head.

![architecture](assets/arch.jpg)

> The diagram above shows the original VAE-based design. The current pipeline replaces the
> from-scratch VAE feature extractor with a frozen **DINOv2 ViT-S/14** backbone (the regression
> head and the data/card/label layer are unchanged). The retired VAE code lives in [`legacy/`](legacy/).

## 0. Introduction

The model is a two-stage pipeline:

1. **Stage 1 — frozen backbone.** A pretrained DINOv2 ViT-S/14 maps a 224×224 RGB face image to a
   384-dim feature vector. The backbone is frozen, so features are precomputed once and cached.
2. **Stage 2 — MLP head.** A small MLP regresses the **205-dim facial parameter vector** from the
   cached feature. The vector is then written back into a real game character card.

Using a strong self-supervised backbone instead of a from-scratch VAE gives better features with
far less training and no reconstruction objective to tune.

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
> match your driver. The `mtcnn_ort` face detector and the `HS2ABMX.exe` card serializer are only
> needed for the alignment / inference paths, not for training.

## 2. Configuration

Everything is config-driven via [`config.py`](config.py). Pick a preset with `--config`:

- `smoke` — offline skeleton (a `DummyBackbone`, tiny dims, synthetic data); runs with no downloads
  and no dataset. Useful to validate the wiring.
- `dinov2_vits14` — the real training config (frozen DINOv2 ViT-S/14, 384-dim, `data/`).
- `dinov2_vitb14` — larger ViT-B/14 variant (768-dim).

## 3. Train

```bash
# (once the dataset is ready)
python tools/gen_labels.py --cards-dir data/cards --out data/labels.json   # 205-dim labels from cards
python split_data.py                                                       # train/val split index files
python extract_features.py --config dinov2_vits14                          # Stage 1: cache DINOv2 features
python train_head.py      --config dinov2_vits14                           # Stage 2: train the MLP head
```

Training auto-resumes from the latest checkpoint under `exp/<exp_name>/ckpts/`. Monitor with
`tensorboard --logdir exp`. In practice ~15–30 head epochs are sufficient.

**Smoke run (no dataset, no downloads):**

```bash
python tools/make_synthetic_data.py
python train_head.py      --config smoke
python extract_features.py --config smoke
```

## 4. Inference

Predict parameters for an image and write them into a character card:

```bash
python infer.py --config dinov2_vits14 \
    --head exp/dinov2_vits14_head/weights/head_epoch_30_step_XXXX.pth \
    --image test/my.png --template test/template.png --out outputs/
```

Export the combined model to ONNX (use `--head-only` to export just the regression head):

```bash
python export_onnx.py --config dinov2_vits14 --head <weights.pth> --out outputs/face2param.onnx
```

## About the HS_FACE dataset

The [HS_FACE](https://pan.baidu.com/s/1yPftN5rmtY5QDF7G2RjN4A?pwd=p8qd) dataset is a collection of
approximately 14w facial images of game characters. It consists of three parts: 1. Facial images of
game characters directly sampled from the game; 2. Images generated with stable diffusion using the
images in (1) as a condition, close to real people; 3. Facial parameters of the game characters in (1).

## License

MIT. This project builds on the original
[ChasonJiang/Face2Parameter](https://github.com/ChasonJiang/Face2Parameter).
