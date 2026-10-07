# Unity geometry export parity

The tools in `tools/unity_parity/` read saved MakerGeometryService schema1 exports.
They do not contact the game, alter characters, or rebuild/install plugins.
Matrices are row-major arrays acting on column vectors; units are Unity asset/game
units, with no physical centimeter calibration.

MCP native range mode is a **write-validation policy**, not a switch that disables
installed SliderUnlocker patches. Selecting bounded native writes leaves the
installed18.2 runtime active, including its in-range hooks. The offline sampling
profiles model GetInfo's explicit contract; their in-range interpolation is the
same by design, not evidence that the game plugin has been unloaded.

## What is compared

1. Start with original shared-mesh vertices in unchanged vertex order.
2. Add the captured active blendshape vertex deltas before skinning.
3. Reconstruct world positions with
   `sum(weight * actual_bone_world * bindpose * morphed_bind_vertex)`.
4. Apply each exported candidate transform to the **raw** Unity BakeMesh vertices:
   `renderer_matrix` and `scale_free_trs`.
5. Report maximum vertex Euclidean error, RMS Euclidean error, maximum coordinate
   error, and errors divided by the reference mesh bounding-box diagonal.

Both candidates are evaluated independently; the tool does not choose the lower
error and automatically proclaim it correct. A candidate must meet the declared
normalized tolerance (default `1e-5`) and agree with its exported world vertices.
The snapshot must stay in one frame and preserve source/baked topology. If both
candidates match, numerical agreement is recorded but the scale convention remains
undistinguished. This is common for unit-scale renderers.

This certification applies to the captured snapshot, not every possible Unity
version, transform hierarchy or shader deformation. Unity2018.4 documents
renderer-relative BakeMesh vertices, so the scale question is resolved empirically
instead of inferred from wording. [Unity2018.4 BakeMesh documentation](https://docs.unity3d.com/2018.4/Documentation/ScriptReference/SkinnedMeshRenderer.BakeMesh.html)

## Expression and skin quality

Every nonzero blendshape weight requires all corresponding exported frame deltas.
Without those deltas, the mesh is explicitly **uncertifiable**; expression is never
silently discarded. Zero-weight shapes do not require their deltas.
The interpolation hypothesis is piecewise linear frame deltas, with an implicit
zero-weight base if absent and outer-segment extrapolation. Its numerical agreement
is verified at the exported current weights; negative/extrapolated weights or
unusual authored frames are not universally certified by a passing ordinary case.
[Unity2018.4 blendshape frame API](https://docs.unity3d.com/2018.4/Documentation/ScriptReference/Mesh.AddBlendShapeFrame.html)

The declared renderer/global skin quality selects a 1-, 2- or 4-influence hypothesis.
Four-influence reconstruction uses the original weights unchanged. Lower-quality
hypotheses retain the strongest weights and normalize their sum. The report also
compares all three hypotheses, so CPU BakeMesh behavior differing from the declared
render quality is visible. `--influences 4` explicitly overrides the hypothesis;
the report records the override. Nonfinite arrays, invalid active bone indices,
missing actual bone matrices and invalid weight sums prevent certification.

Reports keep duplicate mesh names as distinct list entries identified by renderer
path. An inactive body/silhouette `o_tang` must not overwrite the head's tongue entry.

## Commands

Always use the project interpreter:

```powershell
.venv/Scripts/python.exe tools/unity_parity/test_geometry.py -v
.venv/Scripts/python.exe tools/unity_parity/compare_export.py C:/captures/geometry.json --out outputs/unity_parity/report.json
.venv/Scripts/python.exe tools/unity_parity/compare_export.py C:/captures/geometry.json --out outputs/unity_parity/offline.json --offline --profile slider_unlocker_18_2 --renderer-uniform-scale
.venv/Scripts/python.exe tools/unity_parity/compare_series.py C:/captures/live_cases.json --out outputs/unity_parity/series --profile slider_unlocker_18_2 --renderer-uniform-scale
```

`compare_export.py` returns exit0 when every mesh's numerical reconstruction is
certified, or exit2 when any is not. Offline fitting remains a separate diagnostic.
The report includes the input snapshot SHA-256 and runtime metadata. Reports cannot
overwrite their source geometry snapshot or modifier input.

`compare_series.py` accepts the root runner's `live_cases.json` (`cases` containing
`name` and `geometry.path/sha256`), verifies each export hash, writes per-case reports
and `series.json`, and compares bone-local TRS against offline FK. For explicit ABMX
inputs supply `--abmx-cases <json>` mapping case names to bone-name modifier dicts
with `scale`, `length`, `position`, and `rotation`. This file can be built from the
manifest's reviewed `abmx_writes[].current.bones` responses; do not infer modifiers
by adjusting a fitted mesh.

## Offline HeadRig comparison

`--offline` requires an explicit sampling profile and the exact cached head ID.
It checks vertex count, triangle order, source coordinates, bone palette, weights,
and bindposes before asserting correspondence. Captured expression deltas are
applied to the cached source vertices, then native FK/LBS is evaluated. Optional
ABMX modifiers can be supplied with `--abmx-json` or the series case mapping.

Alignment permits proper rigid rotation and translation only: no reflection,
anisotropic scale, shear or free affine deformation. Any unit conversion is
explicit (`--unit-scale` plus `--unit-scale-reason`) and independently reported.
An optimal uniform scale is printed as a **diagnostic suggestion only**, never
automatically applied. Uncompensated errors remain in the report.

`--renderer-uniform-scale` additionally permits a scale independently established
from the exported renderer matrix, not from a vertex fit. It verifies equal column
lengths, orthogonal axes, positive determinant and consistency with recorded lossy
scale. Nonuniform transforms and shear/reflection are rejected. The report records
each axis, spread, orthogonality error, renderer ID/path, and non-unit ancestor
contributors. Physical ancestor scaling and declared unit conversion remain
separate fields, even though their product is applied before rigid alignment.

The bone-local report compares actual recorded local position/quaternion/scale with
offline native/ABMX locals. It identifies cache/live parent names, unmatched or
ambiguous bones, and which differences affect the head's skinning palette. Actual
locals include animation and plugin effects; a residual is not automatically blamed
on native sampling or a limited base. Eye-look bones can animate while head-skin
transforms remain correct.

## Verified exports on 2026-10-04

Root exported the existing head2 character plus controlled cases; this tool only
read those saved files. Reports are under `outputs/unity_parity_20261004/`.

The initial10-renderer export:

- Eight head-related renderers independently match `scale_free_trs`.
- Two disabled additional `o_tang` renderers match both candidates; their convention
  is undistinguished and their renderer paths remain distinct.
- Head reconstruction error is `3.09e-6` world units (`1.04e-6` bbox-normalized) for
  scale-free TRS, versus `0.1337` (`0.04479` normalized) for the full renderer matrix.
- Worst scale-free error over all10 entries is `4.70e-6` normalized.

An initially apparent offline4.2% head discrepancy was explained by independently
recorded ancestor scale: `cf_N_height=.9` and `cf_J_Head_s=1.0169462`, giving about
`.9152514`. Applying that recorded factor and rigid pose alignment makes the native
baseline mesh error `3.28e-6` units (`1.10e-6` normalized). No fitted scale was applied.

The baseline plus10 unlocked-native cases (indices `0,4,24,47,54`, each at `-.25`
and `1.25`) agree with offline sampling/FK to roughly `8.8e-7–1.2e-6` normalized
maximum error. Head-skin bone-local position error is at most `1.19e-8` units.
This is real sampling-profile parity for those controls on head2, not a guarantee
for every installed base, every slider or arbitrary extreme values.

Four captured `cf_J_ChinTip_s` ABMX probes also agree after explicitly supplying
their recorded modifiers: scale `[1.08,1.02,1.04]`, length `1.05`, position
`[.01,.02,0]`, and rotation `[2,0,1]`. Their normalized head errors are approximately
`9.1e-7–1.04e-6`. These tests use native baseline `.5`; they do not resolve the
general cached-baseline-length issue when native position changes after ABMX's
baseline capture, nor rotation-excluded bones or H-scene conditions.

Eleven analytical tests cover nonuniform bone scale, mixed weights and bindposes,
both BakeMesh conventions and unit-scale ambiguity, expression/missing-frame cases,
quality hypotheses, rigid alignment restrictions, independently recorded scale
validation, and local-control differences. Passing parity tests establish the
geometry foundation; they do not establish photograph likeness or base expressivity.
