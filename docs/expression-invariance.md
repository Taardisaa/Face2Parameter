# Expression-invariant facial parameters

## The problem

The pipeline predicts a game character's **neutral identity face shape** (the 205-dim
`shapeValueFace` + bone params). But the input is an arbitrary photo, which may carry a
**transient expression** — a smile widens the cheeks/mouth, a frown narrows the brow, etc.

Observed failure: feeding a *smiling* portrait yields a character with a **too-wide face**. The
model captured the momentary expression deformation instead of the person's true resting geometry.

### Root cause

Our backbone is **DINOv2**, a *general-purpose* vision model. Its feature vector encodes whatever
is in the image — pose, lighting, **and the current expression**. Nothing tells it to ignore
expression, so the downstream head reads "smiling-wide cheeks" as "this person has a wide face."

The target labels, by contrast, are **expression-free**: a HS2 card's `shapeValueFace` describes
the resting face; in-game smiles/blinks are separate runtime blendshapes that never appear in the
label. So expression in the input is pure noise we want the model to discard.

## Three directions (researched)

### 1. Expression-invariant identity backbone (ArcFace) — **chosen**

Replace/augment the backbone with **face-recognition embeddings (ArcFace)**. ArcFace is trained to
answer *"who is this person?"* across expression, pose, illumination, and occlusion — so its
embedding is **expression-invariant by construction**. **MICA** (ECCV 2022) demonstrates that these
ArcFace embeddings still carry enough geometry to regress a **neutral** FLAME face shape; FlowFace
and Pixel3DMM both rely on MICA's ArcFace-initialized identity to disambiguate identity from
expression.

- **Fit:** our `Backbone` is a pluggable interface ([src/models/backbone.py](../src/models/backbone.py)),
  so ArcFace is a drop-in alternative — no change to the head, training loop, or card write-back.
- **Pros:** invariance is "free" at the feature level; no inference-time neutralization; small change;
  literature-validated; reuses our cache→train flow.
- **Cons/risk:** ArcFace optimizes for *recognition*, so it may drop some fine geometric detail
  needed for the 205 params (esp. bone tweaks). Mitigation = **hybrid ArcFace ⊕ DINOv2** (below).
- **Implementation (Py3.13-safe, no heavy `insightface` build):** run the `w600k_r50` ArcFace ONNX
  (buffalo_l, 512-d) on **onnxruntime** (already a dep); align faces with **mtcnn_ort** (already
  installed) 5 landmarks → canonical ArcFace 112×112 template via skimage `SimilarityTransform`
  (same machinery as [FaceCrop.py](../src/face_data_utils/FaceCrop.py)).

### 2. Input neutralization (pre-process the photo to a neutral expression)

Convert *any* expression into a neutral-expression image first, then feed the existing pipeline.
Methods: an **expression-to-neutral GAN** (e.g. Deep Face Normalization, Pinscreen, SIGGRAPH Asia
2019), or fit a **3DMM** (DECA / EMOCA), zero the expression coefficients, and re-render a neutral
face.

- **Pros:** the *literal* "infer to neutral then extract" idea; reuses the current trained head; modular.
- **Cons:** adds a heavy generative dependency; the neutralized image carries GAN/render artifacts
  and must still land in our training distribution; an extra failure point and latency at inference.

#### Why a generic diffusion edit is unsafe

Prompting a general image generator for "the same person with a neutral expression" is tempting,
but the generator is free to alter exactly the geometry this project is trying to recover: lip
thickness, mouth width, jaw contour, cheek volume, and even apparent age. A visually plausible
result is not necessarily a geometrically faithful one. If neutralization is added, it should use
an explicitly controlled expression representation rather than unconstrained text-guided editing.

#### Option A — LivePortrait neutralizer (fastest plugin prototype)

> **Status: implemented (prototype), verified working.** Wired as an optional `--neutralize
> liveportrait` step in `predict.py`/`infer.py` via [src/neutralize.py](../src/neutralize.py), with an
> inspect-only [tools/neutralize.py](../tools/neutralize.py) and an install checker
> [tools/check_neutralizer.py](../tools/check_neutralizer.py). Default is **off**.
>
> Realities found while building it:
> - **LivePortrait runs in *this* venv** (py3.13 / torch2.11 / numpy2). Its torch is *unpinned*; the
>   "py3.9 / torch2.3" in its guide is just a recommendation, not a hard requirement. The plugin still
>   shells out to its `inference.py` (simple + isolated) but with **this** interpreter by default.
>   Setup that worked:
>   ```bash
>   git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git ../LivePortrait
>   # extra deps not already in our venv (numpy stays 2.x — do NOT install their requirements_base.txt,
>   # which pins numpy==1.26.4):
>   .venv/Scripts/python.exe -m pip install tyro imageio-ffmpeg pykalman huggingface_hub
>   # an ffmpeg binary (inference.py hard-checks `ffmpeg -version`); reuse imageio-ffmpeg's bundled one:
>   #   copy it to ../LivePortrait/ffmpeg/ffmpeg.exe  (the script adds ./ffmpeg to PATH)
>   # weights (~500 MB):
>   .venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download as s; \
>       s(repo_id='KwaiVGI/LivePortrait', local_dir='../LivePortrait/pretrained_weights', \
>         allow_patterns=['liveportrait/*','insightface/*'])"
>   export LIVEPORTRAIT_DIR=../LivePortrait
>   ```
> - **Windows console:** LivePortrait's rich progress bar prints a 🚀 emoji that crashes on a GBK
>   console; the plugin runs the subprocess with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
> - **Driver-image transfer was tried and abandoned.** There's no native "neutralize" flag
>   ([#500](https://github.com/KwaiVGI/LivePortrait/issues/500)), and the CLI recipe (drive the source
>   with a neutral face + `--animation_region exp --no-flag-relative-motion`) **fails**: absolute
>   *cross-identity* transfer imposes the driver's keypoint geometry on a differently-shaped face,
>   producing distortion (a neutral painting driver turned a slight smile into a pronounced frown).
>   Relative motion with a single image nets zero change. So driver-based neutralization is a dead end.
> - **Method used: delta-zeroing (driver-free)**, in [scripts/lp_neutralize.py](../scripts/lp_neutralize.py).
>   Take the subject's OWN LivePortrait keypoints and zero the expression deviation while keeping their
>   pose/scale/translation: `x_d = scale·(kp @ R_s + α·exp) + t`, `α=0` for full neutral, then
>   `stitching`/`warp_decode`. No foreign expression imported → no cross-identity warp. `--neutralize-alpha`
>   exposes `α` (try 0.0–0.3). The script loads models once and batches a whole folder.
> - **Identity gate:** each edit is accepted only if its ArcFace similarity to the original clears
>   `--gate-threshold` (reusing this repo's [ArcFaceONNX](../src/models/arcface.py)); else it falls back
>   to the original. Crops cache under `outputs/_neutralized/`. Delta-zeroing preserves identity far
>   better than the driver approach — verified id-sim **0.98 / 0.79** on real inputs (vs ~0.76 for the
>   driver), and visually a clean "same person, neutral".
> - **Licensing:** LivePortrait code is MIT, but its bundled InsightFace weights are
>   non-commercial-research only.
> - Still one subprocess per `Neutralizer` call (models load once per call, not per image). A fully
>   in-process path is a possible future optimization.

[LivePortrait](https://github.com/KlingAIResearch/LivePortrait) supports portrait animation,
expression retargeting, and precise expression editing. Its output remains photorealistic enough to
feed the current image backbones, so it is the shortest path to an inference-only plugin:

```
aligned face crop
    -> LivePortrait expression edit
    -> neutralized RGB crop
    -> existing DINOv2 / ArcFace backbone
    -> existing MLP head
```

The neutralizer must change **expression only**. It should retain the source identity, head pose,
scale, translation, crop, and appearance. In particular, the default relative-driving workflow is
not sufficient for this use case: it treats the source expression as the motion baseline, so a
smiling source may remain smiling when the driving sequence starts at neutral. The implementation
needs absolute expression replacement or direct expression-latent editing while preserving the
source pose.

This is the easiest route to test, but it is still a neural renderer and can subtly move facial
geometry. It therefore needs an identity/geometry quality gate before its output is accepted. Also
check deployment licensing: the LivePortrait code is MIT, while its license notes that bundled
InsightFace models are restricted to non-commercial research.

#### Option B — 3DMM/FLAME neutralization (more geometrically explicit)

A 3D face model such as [DECA](https://github.com/YadiraF/DECA) or
[SMIRK](https://github.com/georgeretsi/smirk) decomposes a face into approximately independent
factors:

```
identity shape beta + expression psi + jaw pose + pose + appearance/lighting
```

Neutralization then becomes an explicit operation:

```
psi      = neutral
jaw pose = neutral
```

The neutral face can be rendered and passed to the existing pipeline. This gives stronger control
than a prompt-based generator, although disentanglement is imperfect and a rendered face may fall
outside the training-image distribution.

For several photos of one person, fit or aggregate a **shared identity shape** instead of processing
each image independently:

```
shared beta across the subject
per-image psi_i / pose_i / lighting_i
set every psi_i to neutral
render several neutral views
```

Sharing the identity variable prevents independent edits from drifting toward slightly different
people. DECA and related FLAME models have non-commercial/research-oriented licensing, which must be
considered before making this a distributed dependency. SMIRK's repository is MIT, but it still
depends on FLAME assets and their terms.

#### Option C — regress directly from neutral identity geometry (cleanest architecture)

[MICA](https://github.com/Zielon/MICA) maps an ArcFace identity representation to a neutral FLAME
shape. Instead of rendering a neutral image, it could become a new feature extractor:

```
input photos -> MICA neutral shape -> MLP -> HS2 205-dim parameters
```

This avoids generative image artifacts entirely and makes expression removal part of the feature
representation. It is architecturally cleaner, but no longer a drop-in inference plugin: the MICA
features must be extracted for the training set and a matching MLP head must be trained. Its model
and FLAME dependencies also use research-oriented licenses.

#### Proposed neutralizer interface

Keep neutralization optional and in front of the existing `Face2Param` model:

```
input image(s)
    -> detect and align
    -> Neutralizer(off | liveportrait | 3dmm)
    -> quality gate
    -> current backbone + head
    -> multi-image aggregation
```

The plugin should consume and return aligned RGB crops, cache generated crops, and support a
`neutralize-only` mode so outputs can be inspected independently of parameter prediction.

The quality gate should reject a generated crop when:

- ArcFace similarity to the original identity drops too far (threshold calibrated on validation
  identities, not chosen arbitrarily);
- a face/expression estimator still reports a strong smile, open jaw, or other non-neutral motion;
- face detection or landmark alignment becomes unreliable; or
- estimated identity geometry changes materially between the original and neutralized image.

Rejected edits should fall back to the original crop or be excluded from multi-image aggregation;
they should never silently enter the prediction set.

#### Recommended MVP and A/B test

Start with a `LivePortraitNeutralizer`, but keep the experiment isolated from the production path.
For the same identity and image set, compare:

1. original images -> ArcFace head;
2. neutralized images -> DINOv2 head;
3. neutralized images -> ArcFace head; and
4. optional DINOv2/ArcFace param-space ensemble.

The second arm is important: ArcFace already suppresses much expression information but may lose
fine shape detail. Once the image itself is neutralized, DINOv2 may retain that detail without
receiving the original smile deformation.

Evaluate both **identity retention** and **expression consistency**. Useful measurements include
ArcFace similarity before/after neutralization, within-identity variance across expressions,
mouth/jaw parameter drift, and comparison against a real neutral photo of the subject when one is
available.

#### What neutralization cannot fix

Over-thick lips are not necessarily caused only by expression leakage. The MLP is trained with MSE
and can regress toward the mean HS2 face, particularly after aggregation reduces feature variance.
The training distribution may also contain expression/label correlations. If original and
neutralized inputs produce the same mouth bias, the remaining problem is in the head, labels, loss,
or training distribution—not the input expression. At that point the appropriate fixes are
expression-augmented training, a better loss, more balanced labels, or a calibrated mouth/jaw
residual model rather than a stronger neutralizer.

### 3. Expression-augmented training (learn invariance)

Keep DINOv2, but during training **synthesize expressions** on the face images (e.g. **LivePortrait**
reenactment) while keeping the **neutral** game-param labels. The head is thus forced to map
many-expressions → one-neutral-shape, learning to ignore expression.

- **Pros:** no new inference dependency; composes with (1).
- **Cons:** DINOv2 features still *entangle* expression, so the head must learn to undo it (harder,
  more data); needs an expression generator + a re-extract/retrain.

## Recommendation

Start with **(1) ArcFace identity backbone**. It attacks the problem at the feature level — the
cleanest, smallest, best-supported change for our architecture — and needs no inference-time
neutralization. If pure ArcFace loses too much shape fidelity (neutral-accuracy regresses too far),
fall back to **hybrid ArcFace ⊕ DINOv2** (concat 512+384 = 896-d): ArcFace supplies the
expression-invariant identity anchor, DINOv2 adds appearance detail. Expression-augmented training
(3) can later be layered on top of either for extra robustness.

For the observed **all-smiling multi-image** failure, however, aggregation and ArcFace alone cannot
guarantee a neutral mouth: the residual smile bias is systematic across every input. The next
targeted experiment should therefore be the optional LivePortrait neutralizer above. If it improves
mouth/jaw consistency without identity drift, keep it as an inference plugin. If it changes identity
geometry too often, move to shared-shape 3DMM neutralization or the MICA-to-HS2 route.

## How we measure success

Two axes — both matter:

1. **Neutral accuracy must not regress badly.** `tools/eval_head.py --split test` MAE vs the DINOv2
   baseline (~0.013). Some fidelity loss is acceptable if invariance improves a lot.
2. **Expression-consistency (the real metric).** For several identities each shown with multiple
   expressions (neutral + smile/etc.), predict params for every variant and measure the **variance
   across expressions of the same identity** — lower is better (0 = perfectly invariant). Compare
   ArcFace vs DINOv2. Expression variants come from a few same-person photos or LivePortrait-generated
   smiles. Plus a qualitative check: the smiling photo's predicted face should stop widening.

## References

- MICA — Towards Metrical Reconstruction of Human Faces (ECCV 2022): arXiv [2204.06607](https://arxiv.org/abs/2204.06607), code [github.com/Zielon/MICA](https://github.com/Zielon/MICA)
- DECA — Learning an Animatable Detailed 3D Face Model from In-The-Wild Images: arXiv [2012.04012](https://arxiv.org/abs/2012.04012)
- EMOCA / Pixel3DMM (MICA-initialized identity): [pixel3dmm](https://simongiebenhain.github.io/pixel3dmm/)
- SMIRK — 3D Facial Expressions through Analysis-by-Neural-Synthesis (CVPR 2024): arXiv [2404.04104](https://arxiv.org/abs/2404.04104), code [github.com/georgeretsi/smirk](https://github.com/georgeretsi/smirk)
- ArcFace / InsightFace (Additive Angular Margin, buffalo_l, 512-d): [insightface.ai/research/arcface](https://www.insightface.ai/research/arcface)
- Deep Face Normalization (expression/pose/lighting neutralization, SIGGRAPH Asia 2019): [hao-li.com](https://www.hao-li.com/publications/papers/siggraphAsia2019DFN.pdf)
- LivePortrait (portrait animation, stitching, and retargeting control): arXiv [2407.03168](https://arxiv.org/abs/2407.03168), code [github.com/KlingAIResearch/LivePortrait](https://github.com/KlingAIResearch/LivePortrait)
