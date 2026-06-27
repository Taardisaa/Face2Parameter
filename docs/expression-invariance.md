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
- ArcFace / InsightFace (Additive Angular Margin, buffalo_l, 512-d): [insightface.ai/research/arcface](https://www.insightface.ai/research/arcface)
- Deep Face Normalization (expression/pose/lighting neutralization, SIGGRAPH Asia 2019): [hao-li.com](https://www.hao-li.com/publications/papers/siggraphAsia2019DFN.pdf)
- LivePortrait (efficient portrait reenactment for expression augmentation): arXiv [2407.03168](https://arxiv.org/abs/2407.03168), code [github.com/KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)
