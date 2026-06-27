# Observed behavior on different input domains + the real→cartoon preprocessing idea

> Status: **observations + a parked idea.** Records (a) how the current pipeline actually
> behaves on real vs. anime inputs, and (b) a cheap real→cartoon stylization preprocessor we
> considered but did not build. Read alongside [expression-invariance.md](expression-invariance.md)
> (the smiling-mouth failure) and [self-supervised-rendering.md](self-supervised-rendering.md).

## What we actually observe at inference

- **Real human faces work well.** The DINOv2/ArcFace → head pipeline maps a real photo to
  plausible game params most of the time. This is the case the whole `aug_images` (SD-realistic)
  training domain was built to serve, and it pays off.
- **Occasional mouth distortion on smiling inputs.** A wide smile in the photo leaks into the
  predicted face (e.g. too-wide mouth/jaw). This is the expression-leakage failure that motivated
  the expression-invariant ArcFace backbone — see [expression-invariance.md](expression-invariance.md).
- **Anime / 2D-style character inputs fail completely.** Feeding a 2D illustration produces a
  result that is "realistic-ish" but maps to a **very uncanny 3D face**. The pipeline has no
  useful behavior here.

## Why anime inputs break — and what 2D→3D *actually* needs

The pipeline's entire premise is **"approximate the human face"**: backbone features encode face
geometry/identity, and the head regresses params that reproduce that face. That premise is sound
for photos but **largely void for 2D-stylized input**, for two reasons:

1. **The input isn't a human face.** Anime faces have non-human proportions (huge eyes, tiny nose,
   stylized jaw). A face-geometry encoder either reads them as a distorted human (→ uncanny params)
   or as out-of-distribution noise. ArcFace especially is a *real-face recognizer* — stylized faces
   are OOD for it.
2. **For 2D→3D, geometric face-matching is arguably the wrong objective.** The working hypothesis
   (Taardisaa): turning a 2D character into a good 3D character is **less about matching face shape
   to a human and more about matching the other attributes** — hair color/style, accessories,
   eye color, clothing, overall vibe. A 2D→3D "good result" is judged by *"does it read as the same
   character?"*, which those attributes carry, not by facial-landmark fidelity. Our 205-dim target
   is almost entirely **face-shape params** (54 `shapeValueFace` + bone tweaks) and encodes **none**
   of hair/clothes/accessories — so even a perfect predictor is solving the wrong problem for the
   anime case.

**Implication:** the anime→3D task is a *different task*, not a harder version of the current one.
It would need (a) different inputs/targets (attribute classification: hair/eye color, hairstyle,
accessory/outfit tags → the corresponding card fields) and likely (b) a different label space than
the face-shape vector this project predicts. Treating it as "feed an anime pic into the photo
pipeline" is expected to keep failing.

## The parked idea: real→cartoon stylization as a preprocessor

Original question: render a real photo into a cartoon/game style *first* (so facial detail is
exaggerated toward the training domain), then run the unchanged backbone+head — a pure
preprocessing plug-in, **no retraining**.

Candidate stylizers (all ONNX-able, would slot in before the backbone in `predict.py`/`infer.py`):

| Model | Notes |
|---|---|
| **DCT-Net** (ModelScope) | Portrait-specialized; `3d` style is closest to a game render. Best fit. |
| **AnimeGANv2 / v3** | Fast, light, but a generic *anime* look, not HS2's render style. |
| **JoJoGAN / DualStyleGAN / VToonify** | Higher quality, StyleGAN-based, heavier, need face alignment. |
| **SD img2img + anime LoRA / ControlNet** | We already run SD; controllable but slow to tune. |

### Why we didn't build it (the domain-direction problem)

The pipeline already bridges domains the **other** way: it takes game faces (`images/`) and makes
SD-realistic versions (`aug_images/`), training the head on **both**. So a real photo at inference
already lands in a domain the head has seen.

- **ArcFace config: expected to get worse.** Stylizing a real photo to cartoon is doubly OOD for a
  real-face recognizer, and `arcface` trains with `aug_prob=1.0` (realistic domain only).
- **DINOv2 config: only marginally worth a try, with a "third domain" risk.** The head has seen the
  clean in-game `images/` domain, so pushing input toward it *could* help — but a *generic* anime
  filter produces neither a real face nor an HS2 render. Those features may land in an unseen third
  domain and do worse. The only stylizer that truly matches would be HS2's own render style, which
  is essentially our SD pipeline in reverse (hard).

**Verdict:** low expected payoff for the effort. If ever revisited, try **only the DINOv2 head**
with **DCT-Net `3d`**, gated behind a `--stylize {none,dctnet,animegan}` flag in `predict.py`
(default `none`), as a quick A/B — not a default path.

## Bottom line

The current "approximate the human face" pipeline is the right design for **photo→3D** and works.
For **anime→3D** it's the wrong design, because that task is dominated by attribute matching
(hair/accessories/clothes), not face geometry, and our target vector encodes none of those.
Stylizing real photos as a preprocessor is cheap but expected to be low-payoff and is parked.
