# Future idea: self-supervised face params via analysis-by-synthesis

> Status: **parked / not implemented.** The current supervised pipeline (DINOv2 or ArcFace
> backbone → MLP head, trained on `labels.json`) works well enough. This doc records a more
> ambitious, label-free direction and — importantly — *why it's hard*, so we don't rediscover
> the bottlenecks later.

## The idea

Today the backbone→head is trained **supervised**: it needs ground-truth 205-dim params for
every training image (the `labels.json` we extracted from cards). The question: can we train it
**self-supervised**, with no param labels, in a GAN-like / feedback loop?

Proposed loop (this is the classic **analysis-by-synthesis** / "render-and-compare" paradigm,
the same trick modern 3D face reconstruction like DECA uses):

```
input face ──F──► params ──[render]──► character image ──[similarity]──► loss ──► improve F
```

`F` = our predictor. Supervision = "does the rendered game character look like the input face?"
No labels required — the signal comes from the reconstruction similarity.

## Why it's hard — the dominating bottleneck: the renderer is the game

Everything hinges on the `[render]` step, and HoneySelect2 is the worst possible renderer for a
training loop:

1. **Non-differentiable.** You cannot backprop the similarity loss through the game's Unity
   renderer — there is no `∂loss/∂params`. This kills clean gradient-based end-to-end training.
   The only escape is black-box optimization (RL / CMA-ES / evolutionary), which needs no
   gradients but runs into bottleneck #2.
2. **Throughput.** Black-box methods need thousands–millions of render samples. The game renders
   at *seconds per character* (launch / load card / pose / screenshot — not headless, not
   batchable). `HS2ABMX.exe` only *writes* cards, it doesn't render them. So black-box training
   is infeasible. (It could work as slow **per-image test-time refinement** — minutes to optimize
   one character against one photo — but not as a training signal.)
3. **Cross-domain similarity.** The render is anime/game-styled; the input is a real photo. Pixel
   similarity is meaningless across that gap, so the reward itself is noisy. You'd need an
   identity/geometry metric, and ArcFace is already shaky on stylized faces. (The dataset's
   SD-realistic `aug_images` exist precisely to bridge this gap.)

## The viable reframing: a learned differentiable surrogate renderer

Replace the game with a **learned, differentiable** renderer `G: params → face image`. Both
blockers vanish — `G` is differentiable *and* fast (one forward pass). And we already have the
training data for it: the **~70k (params, image) pairs** in `labels.json` + `images/`/`aug_images/`.

The self-supervised loop then becomes an **autoencoder**:

```
image ──F(encoder)──► params (205-d bottleneck) ──G(decoder/renderer)──► reconstructed image
                                                                          └── reconstruction loss
```

`F` trains self-supervised against a (frozen or jointly-trained) `G`, and can consume **unlabeled**
real faces. Note the irony: this reintroduces the **generative model we retired** (the from-scratch
VAE), except now it is **parameter-conditioned** (`G(params)→image`) rather than a free latent —
arguably the right instinct in the wrong place originally.

## Open challenges if we ever build it

- **`G` is the new project.** A param→realistic-face generator (conditional GAN or diffusion).
  70k pairs is enough data, but `G`'s fidelity caps `F`'s ceiling — a blurry/biased renderer
  teaches `F` wrong lessons.
- **Underdetermination.** Many param sets (especially the bone tweaks) render to near-identical
  faces, so pure reconstruction can drift into weird-but-valid param basins. Likely remedy:
  **semi-supervised** — keep some label loss to anchor params to game-plausible values, and add
  reconstruction for polish + to exploit unlabeled real photos.
- **The domain gap doesn't disappear, it moves into `G`.** Train `G` toward the `aug_images`
  (realistic) style so the reconstruction-vs-real-photo comparison is fair.

## Bottom line

The "loop through the actual game" version is blocked by non-differentiability + throughput —
fundamental, not a tuning issue. The **surrogate-renderer autoencoder** is the principled,
buildable version *because we already have the (param, image) data to learn the renderer*. It is a
meaningfully bigger build than the current supervised approach, to be revisited only if the
supervised model's accuracy or label availability becomes a real limitation.
