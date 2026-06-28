# Face2Parameter: Modeling facial parameters of game characters from images

**English** | [中文](README_CN.md)

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

`--head` defaults to the latest weights in the config's `exp` dir, and if none exist (a fresh
clone) it falls back to the bundled [`release/`](release/) head for that config — so `predict.py`
/ `infer.py` run out of the box with no training. `--template` defaults to the
bundled [`assets/default_template.png`](assets/default_template.png) (the model only predicts the
face-shape params — the template supplies body/hair/clothes). Pass `--no-detector` to skip mtcnn
alignment and use an aspect-preserving center-crop instead.

**Multiple photos of one person → a stabler vector.** Pass a *directory* to `--image` and the
per-image predictions are aggregated into one consolidated vector (defaults: aggregate in embedding
space with a per-dimension median, which rejects a bad frame). This averages out pose/lighting/
detector noise and — for DINOv2 — cancels expression leakage across varied shots. Aim for ~5–10
varied photos; diversity matters more than count. See
[docs/multi-image-aggregation.md](docs/multi-image-aggregation.md).

```bash
python predict.py --config arcface --image dir_of_photos/          # one averaged vector
python predict.py --config arcface --image dir_of_photos/ --aggregate trimmed --save-per-image
# optional ensemble: run both backbones and merge in param space
python predict.py --ensemble dinov2_vits14,arcface --image dir_of_photos/
```

Flags: `--aggregate {median,mean,trimmed}`, `--aggregate-space {embedding,param}`, `--ensemble
<configs>`, `--save-per-image`, and `--name <basename>` (names the output file so repeated runs don't
overwrite the same `<input>_out.png`). `infer.py` accepts the same directory + flags and writes the
merged result into a card (the most representative photo becomes the card thumbnail).

**Recommended: de-smile in parameter space (`--desmile`).** If every photo of a subject is smiling,
aggregation can't remove the smile (it's systematic) and it leaks into the geometry. The simplest fix is
*not* to edit the image at all, but to relax the smile **after prediction**: the smile lives in named
mouth/cheek bones of the 205-d vector, so we pull those toward their neutral population mean by a factor
`alpha`. Model-free, deterministic, runs in this venv in microseconds — no extra deps, GPU, or identity
gate. `alpha=0` is off (default), `alpha=1` is fully neutral; `0.5–0.8` partially relaxes. It does adjust
the subject's real mouth/cheek detail along with the smile, which is fine when exact identity isn't required.

```bash
python infer.py --config arcface --image inputs/person/ --desmile 0.7 --name person_desmile
```

**Older alternative: image-space neutralization (parked).** Before the parameter-space approach we tried
removing the smile in the *image* under a strict identity-preservation constraint. These backends still
exist but are no longer recommended (weaker results, much heavier). If every photo of a subject is smiling,
the optional `--neutralize liveportrait` step relaxes each input to a neutral expression *before* Stage 1,
then runs the normal pipeline. It uses **delta-zeroing** — taking the subject's own face keypoints and zeroing the
expression deviation (no driver image, so no cross-identity distortion); `--neutralize-alpha` keeps a
fraction of the expression (0 = full neutral). It's a **prototype**: LivePortrait runs in **this same
venv** (its torch is unpinned; a few extra deps + ~500 MB weights — one-time setup in
[docs/expression-invariance.md](docs/expression-invariance.md)). Just point `LIVEPORTRAIT_DIR` at the clone.

```bash
python tools/check_neutralizer.py                                   # verify the external install + flags
python tools/neutralize.py --image inputs/person/ --out outputs/_neutral_check   # inspect de-smiled crops first
python infer.py --config arcface --image inputs/person/ --neutralize liveportrait --name person_neutral
```

There's also a stronger backend, **`--neutralize kontext`**, which uses **FLUX.1 Kontext [dev]** (a
GGUF-quantized instruction image-editor) to edit the expression to neutral while preserving identity —
better at flattening a smile than LivePortrait. It runs in its own dedicated venv and needs an accepted
FLUX license + HF token (`hf_token.txt` at repo root, gitignored). Setup in
[docs/expression-invariance.md](docs/expression-invariance.md):

```bash
python infer.py --config arcface --image inputs/person/ --neutralize kontext --name person_kontext
```

Each edit passes an ArcFace identity gate (`--gate-threshold`); failures fall back to the original
crop. Default is `--neutralize off` (no external dependency).

Export the combined model to ONNX (use `--head-only` to export just the regression head):

```bash
python export_onnx.py --config dinov2_vits14 --head <weights.pth> --out outputs/face2param.onnx
```

## 6. Offline geometry: mesh, ratio diagnostics & editor

Once you have a card, you can analyse and **refine** it entirely offline — no game, no rendering engine.
The HS2 face-construction pipeline was reverse-engineered (managed C# + extractable assets, **no Ghidra**),
so a card's 205-dim vector can be turned into the actual **3D head mesh** it implies, anthropometric **facial
ratios** can be measured on that mesh, and parameters can be **nudged** to hit named ratio targets. Details:
[docs/hs2-renderer-and-mesh.md](docs/hs2-renderer-and-mesh.md) (the constructor) and
[docs/facial-harmony-metric.md](docs/facial-harmony-metric.md) (the metric + editor).

The key enabler: the rig's **named bones are the anthropometric landmarks** (eye corners, nose tip, mouth
corners, chin, cheek/face-width, brow), so ratios fall out of the bone positions with no vertex labeling and
the character side is always expression-neutral.

```bash
# card -> offline 3D head mesh (.obj) + a pure-numpy software render (no game)
python scripts/hs2_render_mesh.py --card tests/yua_desmile08_out.png
```

**One-time reference** — the harmony/fidelity scores compare against a population distribution of facial
ratios sampled from `labels.json` (cached to `data/hs2_head/ratio_reference.npz`):

```bash
python scripts/face_report.py --build-reference          # ~5000 cards, runs in a minute
```

**Harmony — is the face a coordinated/typical *combination*?** Per-ratio robust z + a Mahalanobis
percentile (low = typical). Surfaces "every feature fits but the whole reads off" as numbers. `--card-a/-b`
compares two cards in σ units.

```bash
python scripts/face_report.py --card tests/yua_desmile08_out.png
python scripts/face_report.py --card-a tests/yua_desmile08_out.png --card-b tests/HS2ChaF_20240901192905747.png
```

**Fidelity — does the card match the input photo?** Reduces both sides to the same ratios and reports
per-ratio drift in σ. Needs a 68-point detector (`pip install face-alignment`). `--render-backend` detects
on a render of the card with the *same* detector used on the photo, so landmark-definition bias cancels —
**trust those deltas** (without it, a rig-bone-vs-skin offset inflates differences).

```bash
python scripts/face_report.py --card tests/yua_desmile08_out.png --photo tests/yua_ariga_06.jpg --render-backend
```

**Editor — nudge named ratios → a new card.** Solves for the smallest parameter change that hits your
targets while holding the other ratios (damped Gauss-Newton over the normalized vector, clamped in range,
symmetric by construction). Reports the achieved targets, *coupled collateral drift*, and the harmony cost.

```bash
# presets: thinner face, eyes closer together  (-> new card + before/after render)
python scripts/face_edit.py --card tests/yua_desmile08_out.png --thinner --eyes-closer \
    --out tests/yua_thinner_out.png --render

# precise targets (absolute / percent / sigma); --dry-run reports without writing
python scripts/face_edit.py --card tests/yua_desmile08_out.png \
    --set "icd/facewidth=-1.5sigma" --set "face_height/width=+8%"
```

Presets: `--thinner --wider --eyes-closer --eyes-wider --bigger-eyes --smaller-eyes --longer-face
--rounder-face --slimmer-jaw --narrower-nose` (`--strength`, `--sliders-only` to edit only the in-game
sliders). Because ratios are face-width-normalized they encode *proportion*, not absolute size. Scripts
default their output to the gitignored `outputs/`; pass `--out tests/...` to keep results visible.

## Known Limitations

1. **Expression leakage on smiling inputs.** A strong smile can bleed into the predicted geometry
   (typically a slightly off mouth/jaw), because the DINOv2 features encode expression as well as
   identity. The expression-invariant ArcFace backbone largely fixes this; residual smile can be
   relaxed directly with `--desmile <alpha>` (parameter-space, no extra deps) — see
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
