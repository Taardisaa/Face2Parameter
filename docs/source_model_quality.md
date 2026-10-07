# Source model quality: a separate requirement

2026-10-07. The user explicitly requires checking whether the photo model itself
recovers the person's geometry accurately. An exact decoder replay or unchanged
HS2 import cannot establish that. The current source model's 3D accuracy is
**unverified**; the bridge goal remains active.

## Source findings

The local SMIRK README and `configs/config_train.yaml` state that the main stage
freezes shape and pose while optimizing expression. Shape pretraining is supervised
by MICA predictions. This makes expression quality and identity geometry separate
questions. See the [official implementation](https://github.com/georgeretsi/smirk)
and [paper, section 4.1](https://arxiv.org/html/2404.04104v2#S4.SS1).

The paper's appendix B also reports scan-to-mesh evaluation on MultiFace, rather
than relying solely on reconstructed RGB. Published results concern that benchmark;
they do not certify our local checkpoint, input crop or an individual person.
See [appendix B](https://arxiv.org/html/2404.04104v2#A2).

Earlier `expression-invariance.md` rejected SMIRK **render-back** using generated
images. That conclusion does not isolate raw FLAME identity geometry. The old
`outputs/_smirk_test/candidate_01.jpg` is not the current `inputs/audited.jpg`
portrait; they must not be compared as a matched input/output pair.

## Implemented first-stage diagnostic

`scripts/smirk_audit_geometry.py` consumes existing raw export artifacts, verifies
their source hashes, loads their exact FLAME state, and uses the existing SMIRK
renderer/camera. It performs no new encoder inference, neural image generation,
HS2 operation, camera refit or shape adjustment. It writes a local review board:

1. Actual source crop, raw posed geometry and their overlay.
2. Detector landmarks and model landmarks using the asset's literal index mapping,
   checked against the training code's mapping (105 points in this installed code;
   do not replace it with a number from another paper/version).
3. Neutral identity views: keep shape unchanged; zero expression, jaw, eyelids and
   root rotation **only in diagnostic copies**, then view at fixed yaw angles.

Landmark projection calls upstream `batch_orth_proj`, applies the renderer's Y/Z
sign convention and the training code's `image_size` normalization. The source
camera is retained. No best camera is selected to improve residuals.

```bash
$HOME/envs/smirk/bin/python /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/smirk_audit_geometry.py \
  --manifest /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_bridge_20261007/smirk_raw_v1/manifest.json \
  --out /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/audited_v1 --device cuda
```

The command ran. Its report records errors in crop pixels and explicitly says
`unverified_3d`. These observations reuse a detector involved in preprocessing and
training supervision. They are useful for finding inconsistency, not independent
3D truth. No pass threshold was invented, and no makeup-based explanation was
accepted as evidence of accurate geometry.

## Actual paired-view check

The already acquired FaceScape public sample contains `sample_mview_data/4_anger/49.jpg`,
`50.jpg`, `params.json` and a separate `4_anger.ply` scan. Its own readme calls these
two photographs of a tuple. The two photos were each processed by the existing
checkpoint, with raw outputs preserved. Their neutral meshes were then compared
point-by-point in the same FLAME coordinates and on the source renderer's face
vertex set. No per-photo alignment, scaling, vertex fitting or averaging occurred.

```bash
$HOME/envs/smirk/bin/python /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/smirk_export_geometry.py \
  --smirk-dir /mnt/c/Users/13666/Workspace/smirk \
  --in /mnt/c/Users/13666/Workspace/HS2Mod/tools/external_face_assets/facescape_public_sample_v1/sample_mview_data/4_anger \
  --out /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/facescape_pair_raw_v1 --device cuda

$HOME/envs/smirk/bin/python /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/smirk_audit_geometry.py \
  --manifest /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/facescape_pair_raw_v1/manifest.json \
  --out /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/facescape_pair_audit_v1 \
  --device cuda --same-identity
```

Both commands ran. The identity estimates differ between the two views; the profile
also has larger detector reprojection residuals. Detailed numbers and local images
remain in the ignored report directories above. Differences flag view dependence,
but do not identify which estimate is correct or alone prove unacceptable accuracy.
Agreement would not prove correctness either. The neutral diagnostic removes facial
expression parameters; it does not certify that the encoder disentangled expression
and identity in its shape coefficients.

## Next independent evidence

Use the paired `4_anger` scan/photos for a source-model accuracy case. First recover
the dataset's camera/distortion, image-resolution, coordinate and unit conventions
from its implementation and verify this scan's association. Fix the face region
and any rigid/similarity evaluation alignment explicitly; do not deform either
surface or choose a new convention after seeing residuals. Inspect contour and
local nose/jaw/chin discrepancies alongside surface distances. The sample's neutral
TU model is **not** a replacement truth for these anger photos; previous provenance
has not established that they share identity or coordinates.

This paired scan evaluation is not implemented yet. A matched scan answers a
different question from raw replay, cross-view stability or ArcFace identity score.
For real reference portraits without scans, calibrated or withheld views provide
partial constraints; do not manufacture the unseen geometry or claim absolute
accuracy from a single photograph. Preserve uncertainty separately from bridge
errors. Model quality must inform whether to retain SMIRK, examine the existing MICA
route, or revisit the source model before investing in further game integration.

All photos, raw model data, scans and generated review boards remain ignored. The
FaceScape sample readme says not to distribute it; only source, commands and findings
are committed. No game screenshots or Computer Use were needed in this work.
