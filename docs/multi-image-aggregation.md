# Multi-image aggregation: stabler params from several photos of one person

> Status: **implemented.** `predict.py` / `infer.py` accept a *directory* of photos of the same
> person (varied angle / expression / lighting) and produce one consolidated, more stable param
> vector by aggregating across the images, with robust outlier rejection. Read alongside
> [expression-invariance.md](expression-invariance.md) (the failure this also helps with).

## Goal

A single photo bakes in that photo's expression, pose, lighting, and detector noise. Several
photos of the same identity let us **average out the noise and cancel systematic biases**
(especially expression leakage), the same way face-recognition systems build an *enrollment
template* from multiple shots instead of trusting one.

## Where to aggregate — two layers

1. **Param space** — run the full pipeline per image to a 205-d vector, then aggregate the
   vectors. Model-agnostic, interpretable (outlier rejection acts on real face-shape params),
   trivial to add.
2. **Embedding space** *(chosen)* — aggregate the **backbone features** across images into one
   "template" embedding, then run the head **once**. This is the canonical face-recognition
   multi-shot enrollment, and it is the *natural* layer for **ArcFace** (its L2-normalized
   embeddings are designed to be averaged → re-normalized into an identity template). Note the
   head is non-linear, so `aggregate(head(f_i)) ≠ head(aggregate(f_i))` — the two layers give
   different results.

**Decision:** aggregate in **embedding space** (with param space kept as a fallback switch). For
ArcFace, re-L2-normalize the aggregated embedding before the head so it matches the unit-norm
distribution the head trained on.

## How to aggregate / reject outliers

k-means is the wrong tool here: photo counts are small (a handful to a few dozen), k-means is
fragile at small N, needs a `k`, and *clusters* rather than *rejects* outliers.

- **Per-dimension median** *(chosen default)* — zero hyper-parameters, naturally robust. One
  wide-smile frame that pushes the mouth dims simply gets ignored by the median. Applied
  per-embedding-dimension across the image set.
- **Trimmed mean** *(optional)* — drop embeddings whose distance to the centroid exceeds
  `median + 3·MAD`, then average the rest. Explicit frame-level rejection when you'd rather
  discard a whole bad shot than median each dimension independently.
- **Plain mean** *(optional)* — baseline, no robustness.

## Bonus: a poor-man's expression invariance for DINOv2

DINOv2 features encode expression, so a single smiling photo leaks into the geometry (the
problem ArcFace was added to fix). Averaging several **different** expressions cancels that
drift — smile + neutral + pout → roughly neutral geometry. So multi-image aggregation directly
patches DINOv2's weak spot. ArcFace is already expression-invariant, so for it the multi-image
win is mostly pose / lighting / detector-noise reduction.

## How many photos?

- **Floor ≈ 5.** Median + outlier rejection needs a real majority of inliers; at 3 images one
  bad frame is 33% of the sample and the median is shaky.
- **Sweet spot ≈ 7–10.** Random noise falls like √N, so 1→4 halves it, 4→9 cuts it by another
  third; past ~10 the marginal gain is small.
- **Diminishing returns past ~10–15.**
- **Diversity > count.** Averaging only kills *random* noise via √N; the bigger prize is
  canceling *systematic* bias (expression/pose), which needs **spread**, not duplicates. Five
  varied shots beat fifteen near-identical selfies. Aim for a mix: a neutral front shot plus a
  few different angles and expressions.

## When aggregation hurts (observed)

Aggregating a real person's photos produced a result that looked **more uncanny and more generic**
than both a hand-tuned card and a single-image prediction — notably **over-thick lips**. The inputs
were a set of photos where the subject was **smiling in nearly all of them**. This is not a bug; two
mechanisms compound:

1. **Averaging cancels *random* error, not *systematic* bias.** Aggregation removes the part that
   *differs* between photos. If expressions vary (some smiling, some neutral), the smile is random
   across the set and the median cancels it → the mouth moves toward neutral. But if **every** photo
   smiles, the smile is *systematic* — every image "votes" for the same smile-stretched mouth, so the
   median doesn't remove it, it **locks it in**. A homogeneous (all-smiling) set is the worst case:
   multi-image cannot synthesize a neutral face out of uniformly non-neutral inputs. (This is the
   sharp edge of "diversity beats count" above.) ArcFace suppresses *some* expression but not all;
   the residual, made consistent across the set, becomes visible.

2. **Variance reduction pulls toward the model's mean face (regression to the mean).** The MLP head
   is an MSE regressor, which is biased toward the training-set mean ("the average HS2 face").
   Aggregation lowers variance, which pushes the result *closer to that mean point*. So a single-image
   prediction is noisier but keeps more of the person's idiosyncratic detail, while the aggregate is
   smoother, more generic, less recognizable. If the mean HS2 face has fuller lips, aggregating makes
   the lips fuller.

The **thick-lip symptom** is both mechanisms pointing at the mouth: a smile-stretched mouth read as a
fuller-lip shape param, plus fullness pulled toward the mean — and a uniformly-smiling set freezes it
in instead of averaging it out.

### Practical guidance (highest leverage first)
- **Add expression diversity, especially neutral / closed-mouth shots.** This is the only thing that
  lets the median cancel a smile. With an all-smiling set, multi-image won't fix the mouth.
- **A/B a single neutral photo vs the aggregate** (`--save-per-image`, then run the most neutral image
  alone). If the single neutral shot has a better mouth, the homogeneous-set effect is confirmed.
- **For ArcFace, try `--aggregate mean`** — the canonical face-recognition template is the mean of
  L2-normalized embeddings; `median` (our default) is more outlier-robust but offers no advantage when
  all inputs share the same expression.
- **Try `--ensemble dinov2_vits14,arcface`** and **`--aggregate-space param`** as quick comparisons.

**Bottom line:** aggregation is for erasing random jitter across *diverse* shots, not for distilling a
neutral identity from many similar (e.g. all-smiling) photos. Homogeneous inputs + the mean-pull
compound into a more generic, more uncanny result.

## Cross-backbone ensemble

ArcFace (512-d) and DINOv2 (384-d) have **incompatible feature spaces** — they cannot share an
embedding-space aggregation. But both heads emit the **same 205-dim normalized param space**
(trained on the same `labels.json`), so they can be ensembled **in param space**. Their inductive
biases / failure modes differ (DINOv2 expression leak; ArcFace fails on anime), so their errors are
partially decorrelated → a variance-reducing ensemble. Two-level: aggregate per backbone in
embedding space (above), then merge the per-backbone 205-d vectors with a (weighted) mean. Opt-in
via `--ensemble dinov2_vits14,arcface` (loads/runs both backbones; ArcFace first run pulls
buffalo_l ~288 MB).

## Usage

`--image` accepts a single file or a directory. All flags have sensible defaults (runs with none
passed). Aggregation logic lives in the reusable `src/aggregate.py`.

```
--aggregate {median,mean,trimmed}     # default: median
--aggregate-space {embedding,param}   # default: embedding
--ensemble <config1,config2>          # optional: run+merge multiple backbones in param space
--save-per-image                      # also dump each image's vector for inspection
```

When the detector is on, frames where **no face is found are skipped and reported** (a non-aligned
center-crop would inject a bad vector); per-dim median further protects the result. With
`--no-detector` no frames are skipped (every image is center-cropped and used).
