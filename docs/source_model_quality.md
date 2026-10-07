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

## Implemented independent paired-scan evidence

`tools.model_bridge.scan_accuracy` ran against the existing `4_anger.ply` and both
raw photo outputs. The fixed protocol is tracked in
`tools/model_bridge/facescape_scan_protocol.json`; it was written before measuring
surface residuals. No model coefficients, mesh vertices or topology are fitted.

Camera and association oracle: FaceScape revision
`6a43878cdb61834472eb6cac7009d91f14a972d4`, specifically its
[projection example](https://github.com/zhuhao-nju/facescape/blob/6a43878cdb61834472eb6cac7009d91f14a972d4/toolkit/demo_mview_projection.ipynb),
[camera code](https://github.com/zhuhao-nju/facescape/blob/6a43878cdb61834472eb6cac7009d91f14a972d4/toolkit/src/camera.py)
and [multiview documentation](https://github.com/zhuhao-nju/facescape/blob/6a43878cdb61834472eb6cac7009d91f14a972d4/doc/doc_mview_model.md).
The example explicitly uses the same `4_anger.ply` and photos 49/50, undistorts
resized photographs, and renders with their K and world-to-camera Rt. OpenCV
distortion ordering is `k1,k2,p1,p2,k3`. Both cameras are valid and image dimensions
match their calibration. Source hashes are retained locally at
`HS2Mod/tools/parameter_audit/source_model_quality_20261007/provenance.json`.

Seven fixed front-photo MediaPipe IDs `[33,133,362,263,4,61,291]` give alignment
anchors. Their undistorted rays intersect the actual scan triangles at the first
positive hit. The corresponding FLAME points use its existing static MediaPipe
embedding. Each reconstructed posed mesh receives one positive uniform similarity
from these anchors, with reflections prohibited. This removes overall pose, size
and position only. There is **no ICP, local warp, coefficient change, best view
selection or residual-dependent anchor change**. Detector annotation uncertainty
and anchor registration error remain; this is not independent landmark truth.

The metric uses the pinned FLAME `face` mask (1,787 vertices), measuring each source
vertex to the closest **scan triangle surface**, not the closest scan vertex.
Centroid indexing uses a conservative triangle-radius bound that contains every
potentially nearer triangle, followed by exact interior/edge/corner projection.
It is a one-direction distance, not a bidirectional Chamfer or coverage metric.
This is a public-sample diagnostic, **not** FaceScape's complete benchmark protocol;
the official evaluator's dataset-specific head transforms and pupil scales are not
available for this sample under a verified association.

Physical scale in this raw scan is not established. Report errors in native scan
units and as a fraction of the fixed scan span between outer eye corners 33/263;
do not relabel them millimetres. A fixed 0.1-span display saturation is only a color
range, not an accuracy threshold. Gray mesh regions are outside the face metric.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.scan_accuracy `
  --manifest outputs/model_quality_20261007/facescape_pair_raw_v1/manifest.json `
  --scan C:/Users/13666/Workspace/HS2Mod/tools/external_face_assets/facescape_public_sample_v1/sample_mview_data/4_anger.ply `
  --cameras C:/Users/13666/Workspace/HS2Mod/tools/external_face_assets/facescape_public_sample_v1/sample_mview_data/4_anger/params.json `
  --mask C:/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl `
  --embedding C:/Users/13666/Workspace/smirk/assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz `
  --out outputs/model_quality_20261007/facescape_scan_v2

.venv/Scripts/python.exe -m unittest tools.model_bridge.test_scan_accuracy -v
```

The command completed with surface arrays, provenance and calibrated review boards.
Each model is also projected through the other calibrated camera **without another
alignment**, exposing depth and silhouette discrepancies that its own photo can hide.
The earlier `facescape_scan_v1` is preserved; v2 adds board labels, evaluator/runtime
hashes and the held-out camera views. It uses the same anchors, mask and metric.
Seven geometry checks passed, including first-visible ray intersection, a nearest
surface whose centroid is far away, acceleration versus exhaustive triangle queries,
degenerate triangles and known similarity recovery. These checks validate the
measurement machinery, not the photo model's accuracy.

Observed case results (fractions of scan outer-eye span):

| Photo input | Mean face surface error | P95 face error | Mean anchor registration residual | Mean nose-region error |
|---|---:|---:|---:|---:|
| 49, front | 0.02059 | 0.05850 | 0.01884 | 0.01108 |
| 50, profile | 0.03494 | 0.09687 | 0.05273 | 0.02619 |

The profile prediction has larger surface and registration discrepancies. Calibrated
overlays and surface heatmaps show differences at the side of the face, around the
mouth and chin; they cannot be explained away as makeup. These errors include
registration, expression reconstruction, detector correspondence and source geometry;
they are not a clean estimate of identity-only error. The front reconstruction also
does not preserve the entire observed profile. One subject with an anger expression
does not establish a model-wide pass or fail, nor accurate neutral bone structure.
No arbitrary acceptance threshold was introduced.

Next: inspect the existing MICA identity route and its real checkpoint/crop/FLAME
conventions, then seek a genuinely matched neutral scan or additional associated
expressions to isolate identity from expression. The sample's separate neutral TU
model is **not** replacement truth for these photos; identity/coordinates have not
been established. For reference portraits without scans, calibrated or withheld
views provide partial constraints, not absolute unseen-geometry truth. Keep source
model quality separate from bridge errors when deciding which source to preserve.

## MICA source audit: local availability and exact input path

The existing code is SMIRK's **training teacher wrapper**, not an already configured
standalone photo-to-neutral-mesh pipeline. Local
`smirk/src/models/MICA/mica.py` strictly loads `assets/mica.tar`, runs ArcFace on a
112-pixel crop after RGB-to-BGR and `(image-.5)/.5`, normalizes its identity vector,
and returns the original mapping network's 300 `shape_params`. The trained MICA
mapping network is part of the source model; it is not a new HS2 conversion model.
At the initial audit, `smirk/assets/mica.tar` was absent. The separate official
standalone route has now been configured and executed, as recorded below; the
teacher wrapper was not modified or silently reused with another crop.

The teacher's `datasets/base_dataset.py` uses its original FAN68 eye-midpoint,
nose-index-32 and mouth-corner points for an ArcFace similarity crop. The saved
SMIRK export contains MediaPipe observations, **not those FAN68 observations**.
They must not silently substitute for the teacher's input correspondence. This
would change the model input convention and contaminate a model comparison.

The [official MICA demo](https://github.com/Zielon/MICA/blob/master/demo.py) instead
uses its RetinaFace detector, original five-point ArcFace alignment and explicit
checkpoint to output canonical FLAME vertices and identity code. Its mesh export
multiplies by 1,000; raw FLAME coordinates and exported millimetres must not be mixed.
Its [model](https://github.com/Zielon/MICA/blob/master/micalib/models/mica.py) decodes
the identity embedding through its own FLAME generator. A runnable comparison must
bind the real official checkpoint, detector, crop and FLAME assets, preserve its
raw outputs, and use neutral ground truth for an identity-only accuracy claim.
The anger scan remains valid posed-shape evidence; zeroing a prediction's expression
does not turn the anger scan into neutral truth. Obtaining those missing dependencies
and genuinely associated neutral evidence remains work, not an accuracy certificate.

## Executed official MICA baseline

Source revision: `af22e7a5810d474bc28a1433db533723d6bd2b07`, retained in ignored
`HS2Mod/tools/parameter_audit/source_mica_20261007`. Assets came from this revision's
README links. SHA256: MICA checkpoint
`4542a467d9e8f7521474a1d00eac89552bebef0b331b72bf7fbd6f065ff64d7b`;
antelopev2 bundle `7353a5fdca5a90e11d2792e0236032b2fe42adc1ea23eaef5cf8c8b57e7e9393`.
Weights, checkout, licensed inputs and generated results remain ignored.

`scripts/mica_export_geometry.py` calls the original Arcface and Generator used by
MICA encode/decode, with strict complete checkpoint loads. Only unrelated training
mask initialization is bypassed. It uses antelopev2 `scrfd_10g_bnkps.onnx`, original
FaceAnalysis, 224 detection size, center-face selection and `get_arcface_input`.
Only the detection module is loaded; its recognition ONNX never replaces trained
MICA Arcface. CPU detection and CUDA encoding/decoding are explicit. Original
outputs `pred_shape_code` (300 values), `pred_canonical_shape_vertices` and `faceid`
are preserved. Bridge `shape_params` is a checked alias. No camera/jaw/expression
prediction is fabricated, and canonical geometry is reused unchanged as head-local.
The actual decoder state and source files are hashed. Windows source verification
uses the manifest's explicit WSL distribution for native Linux package paths.
The v1 artifact is retained; v2 adds distribution provenance without changing geometry.

```powershell
wsl -d Ubuntu -- bash /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/mica_deps_wsl.sh
wsl -d Ubuntu -- bash /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/mica_fetch_wsl.sh /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/mica_assets_20261007

wsl -d Ubuntu -- /home/taardis/envs/smirk/bin/python /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/mica_export_geometry.py `
  --mica-dir /mnt/c/Users/13666/Workspace/HS2Mod/tools/parameter_audit/source_mica_20261007 `
  --assets /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/mica_assets_20261007 `
  --flame-model /mnt/c/Users/13666/Workspace/smirk/assets/FLAME2020/generic_model.pkl `
  --landmark-embedding /mnt/c/Users/13666/Workspace/smirk/assets/landmark_embedding.npy `
  --in /mnt/c/Users/13666/Workspace/HS2Mod/tools/external_face_assets/facescape_public_sample_v1/sample_mview_data/4_anger `
  --out /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/mica_pair_raw_v2 --device cuda

wsl -d Ubuntu -- /home/taardis/envs/smirk/bin/python /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/compare_identity_geometry.py `
  --smirk-manifest /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/facescape_pair_raw_v1/manifest.json `
  --mica-manifest /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/mica_pair_raw_v2/manifest.json `
  --out /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_quality_20261007/identity_comparison_v2 --same-identity

.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh `
  --manifest outputs/model_quality_20261007/mica_pair_raw_v2/manifest.json --image-index 0 `
  --head-local --out outputs/model_quality_20261007/mica_pair_raw_v2/source_head.json

.venv/Scripts/python.exe -m unittest tools.model_bridge.test_artifact tools.model_bridge.test_scan_accuracy -v
```

Commands completed. Actual identity basis, template, topology, pose basis, joint
regressor, parents, skin weights and fixed eye/neck values match SMIRK exactly.
Comparison uses shared raw coordinates without fitting. Boards show original MICA
and SMIRK copies with expression/jaw/eyelids/root rotation zeroed. Original MICA LBS
replays both outputs within `1e-6`; this proves decoding fidelity, not accuracy.

| Model | Mean cross-view distance | P95 | Max |
|---|---:|---:|---:|
| Original MICA | 0.001975 | 0.003915 | 0.004982 |
| SMIRK neutral diagnostic | 0.002134 | 0.004516 | 0.007080 |

These are **consistency distances in unscaled FLAME coordinates**, not errors against
neutral truth. MICA varies slightly less on this pair but still varies; stability
does not certify correct identity. Report/boards:
`outputs/model_quality_20261007/identity_comparison_v2`. Interchange now includes
model format, checkpoint and all source hashes alongside unchanged vertices and
original shape values. No MICA game import was performed in this milestone. Ten
focused geometry/preservation checks passed, with no new game samples.

A source discrepancy is confirmed: SMIRK's base dataset converts images to RGB,
then its MICA teacher flips RGB to BGR. Official MICA `get_arcface_input` uses
`swapRB=True` on OpenCV BGR input, producing RGB. Cropping and channel order differ.
This does **not** prove the released SMIRK checkpoint used that exact training path
or that this alone explains its observed errors. The new baseline follows official
inference; the teacher is not modified/retrained.

Neutral-ground-truth association remains absent. The sample readme says
`sample_tu_model/1_neutral.jpg` is a **texture map**, its PNG a displacement map;
they are not additional calibrated portrait photos. Do not treat them as matched
neutral inputs or certify their association with `4_anger`.

All photos, raw model data, scans and generated review boards remain ignored. The
FaceScape sample readme says not to distribute it; only source, commands and findings
are committed. No game screenshots or Computer Use were needed in this work.
