# HS2 offline renderer repair — 2026-10-12

The CPU **geometry and camera stage** now uses captured state and recovered
installed assets. A full game-equivalent RGB renderer is still unfinished.
Diagnostic images are not approved MediaPipe fitting inputs.

## Mechanisms repaired

- `CapturedHeadForward` keeps actual current local TRS, private ABMX baselines,
  historical length/direction, flags, modifier list order, rotation exclusions
  and explicit Apply count. It reuses the independently audited installed
  `BoneModifier.Apply` transition, not `position * length + offset`.
- Native59 uses the installed head keyframes, SliderUnlocker profile and 113
  source-derived assignments. Native updates refresh the corresponding ABMX
  caches via Reset → Face/Body Update → CollectBaseline. Body Update's two
  actual head-ancestor writers (`cf_N_height`, `cf_J_Head_s`) sample real
  `cf_custombody/cf_anmShapeBody` assets. Mouth masking uses installed ten IDs.
- Modifier creation appends an actual identity/cache collection; removal resets
  and rebuilds native head state. Effects resolve first matching modifier objects.
  Effect lists are checked against every requested surface bone and ancestor;
  a breast-only producer is excluded only when the dependency intersection is
  empty. Unknown overlapping producers remain failures, not disabled features.
- Missing expression deltas are recovered from the installed head's render
  prefab. Source positions, triangle order, indices, weights and bindposes must
  be float32-exact. Names/counts alone never choose a mesh. Normal/tangent frame
  deltas are retained too. Neutral `e00_defo=100` is preserved.
- All requested surfaces retain renderer paths, including three distinct
  `o_tang` entries. Skinning reconstructs world positions directly, avoiding
  repeating renderer scale on Unity's already-scaled BakeMesh output.
- `src/render/cpu.py` uses the actual offscreen `capture_camera`, homogeneous
  clipping, CPU OpenGL-depth projection, pixel centers, depth tests and
  perspective-correct barycentric/UV/attribute interpolation. It does not
  reframe each candidate or use `geometry.main_camera` by mistake.

Each context validates six installed assembly hashes. Mesh frames and their
bundle hash are recorded. Geometry and context capture times stay visible;
matching output does not assert they were captured in the same frame.

## CPU entry point

```powershell
.venv/Scripts/python.exe scripts/hs2_render_cpu.py `
  --geometry ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/baseline_front_geometry.json `
  --context ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/context.json `
  --capture ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/baseline_front_capture.json `
  --source-reference ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/source_zero.json `
  --out outputs/offline_renderer_repair_20261012/fresh_name
```

Optional `--patch` contains native bridge-style `{"bones":[...]}`. `--native`
contains all 59 values. `--apply-count` declares 1–32 calls, never a fitted
hidden variable. Output contains all source-order surfaces, pixels and triangle
visibility data. `geometric_visibility.png` uses categorical diagnostic colors;
it explicitly ignores shader alpha and is **not** the finished rendered face.
No executable, game HTTP, GPU or neural inference is invoked.

## Validation and remaining work

Five analytic tests cover perspective reprojection, clipping, near-depth order,
Reset history and CollectBaseline persistence; the reused ABMX suite has 28
tests. Existing native/source captures are checked without collecting samples:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tools/offline_renderer -p 'test_*.py' -v
.venv/Scripts/python.exe tools/offline_renderer/validate_saved_cases.py `
  --baseline-dir ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1 `
  --acceptance-dir ../HS2Mod/artifacts/chenger_native_inverse_20261011 `
  --out outputs/offline_renderer_repair_20261012/fresh_validation.json
```

Current baseline source-zero comparison passes the unchanged `1e-5` maximum
bbox-normalized gate for all ten surfaces. Native59 refresh is checked separately.
The older seven-bone/native acceptance has an eye pose different from the latest
capture: reference-eye errors are already present before the patch. They remain
reported and cannot certify cross-capture eye parity. No fitted correction is
applied. Full receipts preserve reference and candidate errors individually.

The RGB/material stage still has concrete source gaps: current GPU scene code
omits tear film and eyelid shadow, uses fitted ambient instead of actual SH,
disables translucency, lacks the real SSS postprocess, and does not implement
all runtime normal/tangent/detail and shader variants. Porting a few albedo
layers or feeding the categorical image to MediaPipe would not repair this.
Active breathing also requires its captured source time/FrameEffects; unknown
overlapping body effects remain explicit unsupported paths. These are pending
implementation/acceptance, not waived requirements.

The old card-only gray/textured scripts retain their legacy behavior and are
not installed-state compatibility entry points. The source-state CPU path can
already supply real parameter-controlled geometry for **3D reference fitting**;
the render → MediaPipe loop must wait for the remaining image pipeline repair.

## Normal-frame disagreement retained

An additional CPU audit found that source-mesh identity does not certify
blendshape-frame identity. The installed bundle has normal-only sparse entries
that an earlier native `GetBlendShapeFrameVertices` export does not retain.
For neutral `head.e00_defo`, the bundle has 35 entries and the API export has
five nonzero normal locations. No empirical pruning threshold was introduced.
Using actual API frames reduces the skinned-normal direction discrepancy from
about 0.186 to 0.00133, but neither input proves current normal compatibility.
The vertex-position gate remains passing; the normal/shader claim remains open.

```powershell
.venv/Scripts/python.exe tools/offline_renderer/audit_normal_frames.py `
  --baseline-dir ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1 `
  --native-frame-export ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_trace/four_heads/head_2_native_ABMX.json `
  --out outputs/offline_renderer_repair_20261012/fresh_normal_audit.json
```

## Same-frame loaded normal repair — resumed 2026-10-12

The new native read-only export at `HS2Mod/artifacts/chenger_reference_fit_20261012/
native_resume_v1/live_full_frames_materials.json` contains actual loaded API
normal/tangent frames and BakeMesh normals in the same native frame (13378).
Four-weight **linear** direction skinning now agrees on all ten surfaces at the
unchanged `1e-5` direction threshold; the largest error is `2.93e-7`. This resolves
the earlier `0.00133` residual as a cross-pose comparison, without inverse-transpose
substitution, a fitted correction, or changed lighting.

`CapturedHeadForward.evaluate` and `hs2_render_cpu.py` now carry world normals,
world tangents and original tangent handedness when loaded API frames are present.
Position-only contexts remain usable for geometry, but explicitly lack RGB
attributes. Asset sparse frames are not silently treated as equivalent to loaded
API frames. The installed-bundle/API disagreement is retained, not threshold-pruned.
Tangent deformation is implemented; a BakeMesh tangent witness is not yet in the
bridge's saved evidence, so tangent equality is not claimed.

```powershell
.venv/Scripts/python.exe tools/offline_renderer/validate_loaded_attributes.py `
  --geometry ../HS2Mod/artifacts/chenger_reference_fit_20261012/native_resume_v1/live_full_frames_materials.json `
  --out outputs/offline_renderer_repair_20261012/fresh_loaded_normal_receipt.json
```

The compact same-frame receipt is committed at
`tools/offline_renderer/loaded_normal_validation_20261012.json`.
The current actual materials use six AIT shaders and `Custom/Silhouette`, not
Hanmen. These observations do not complete the RGB renderer or certify another
character's material pipeline.
