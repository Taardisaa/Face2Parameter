# Source-bound cross-mesh intersection semantics

This offline diagnosis retains every new intersection. A triangle intersection,
shader-visible artifact and anatomical defect are three different claims.
The measurements below prove only the first. The first nine overlays retain
failed projection checks against both older pixel-marker certificates. A later
fresh matching marker certifies their coordinates only; the geometry and shader
claims do not change.

Inputs are the actual `HS2Mod/artifacts/infrastructure_live_20261005/abmx_lowered_v2`
captures and `outputs/abmx_stability_20261005/lowered_quality_v1` certificates.
Outputs are `outputs/crossmesh_semantics_20261005/actual_lowered_v2/`:

- `report.json`: file hashes, actual matrices, source and world triangles,
  barycentric endpoints, UV0/UV1, ordered submesh mapping, actual material and
  texture descriptors, baseline correspondence, cameras and geometric rays.
- `pair_table.json`: concise 28 pair records with three views per record.
- `asset_geometry_comparison.json`: direct read-only installed-bundle array
  comparison, with bundle SHA checked against the preserved asset evidence.
- `findings.json`: source-bound findings, test output and explicit missing proof.
- `capture_requirements.json`: concrete next acquisition options and requirements.
- Nine `*_diagnostic.png` overlays use only the recently acquired lowered PNGs.
  Lines are drawn two pixels wide and markers are enlarged; those drawn widths
  are not measured defect sizes. No heldout images or null-training annotations
  are opened by these tools.

## Actual new indexed pairs

Every index below is the exact zero-based baked/source triangle ID. Each selected
mesh has one recorded submesh, index 0, and one material slot, index 0. Engine
binding is recorded as one slot per submesh; that mapping is not independently
validated GPU binding. Source and baked triangle ordering are checked explicitly.

| Mesh A, triangle | Mesh B, triangle | Windows |
|---|---|---|
| o_eyebase_L, 13 | o_head, 5921 | early |
| o_eyebase_L, 60 | o_head, 5933 | late, far60 |
| o_eyebase_L, 61 | o_head, 5933 | late, far60 |
| o_eyebase_L, 62 | o_head, 5933 | late, far60 |
| o_eyebase_R, 234 | o_head, 7743 | early, late, far60 |
| o_eyebase_R, 238 | o_head, 7743 | early, late, far60 |
| o_eyeshadow, 158 | o_head, 3035 | early, late, far60 |
| o_eyeshadow, 429 | o_head, 3275 | early, late, far60 |
| o_head, 1558 | o_tooth, 827 | early, late, far60 |
| o_head, 8278 | o_tooth, 22 | early, late, far60 |
| o_head, 8278 | o_tooth, 24 | early, late, far60 |

There are 8/10/10 new pairs. The baseline already has 1652 crossing pairs;
neither those nor the new pairs are whitelisted. Reclassification of each listed
pair at its baseline pose returns no intersection under the original baseline
epsilon. Reclassification at the candidate pose reproduces `proper_crossing`
and the previous intersection length within `1e-12` game units.

World coordinates are recomputed from actual baked local vertices and the
snapshot-local LBS-certified matrix. No raw coordinates from different renderer
spaces are mixed. Tolerances remain the existing global-head-diagonal `1e-7`
term plus the exact two renderer LBS residual floors. They are not fitted to
make these pairs disappear. Plane-plane line endpoints retain the original
interval classifier. Endpoint barycentric reconstruction error is at most
`1.91e-15` game units. Tiny negative barycentric values down to approximately
`-1.34e-13` are retained, without clamping or anatomical interpretation.

## Baseline asset and UV correspondence

The installed bundle `E:/HoneySelect2_ArcticFox/abdata/chara/38/fo_head_38.unity3d`
and prefab `p_cf_head_02` contain these meshes. Direct UnityPy decoding matches
all eight live head meshes in ordered vertices, triangle indices, UV0, normals,
tangents, bone indices, weights and bindposes after the recorded float32/int64 conversion.
UV1 matches exactly for `o_head` and `o_namida`; it is absent in the other six
asset meshes and empty in their live exports. Missing UV1 is not replaced by UV0.
Installed mesh path IDs and renderer path IDs are preserved in the asset report.
No equality between asset rest pose and live posed world vertices is claimed.

The complete live `source` dictionaries and source hashes are unchanged between
baseline, early, late and far60. Exact triangle indices therefore provide stable
source correspondence in these files. For each endpoint the tool records its
two triangle barycentric weights, UV0, available UV1, and where those same weights
land on the baseline triangles. This transfers a material point, not an
anatomical landmark. Correspondence must be rebuilt if source topology changes.

For every actual texture descriptor, UV0 multiplied by its recorded scale and
offset is provided as a **candidate** texture coordinate. Per-texture pixel values
remain unknown because `contents_read_back=false`. These coordinates do not
validate the shader's UV stream selection, custom rect/rotation, parallax,
mip selection, color space, channel choice, alpha test or screen dither. In
particular the head shader can use UV1. Recorded material properties cover known
fields only, not every possible shader uniform or GPU material property block.

Actual shaders are `AIT/Eye Translucency`, `AIT/main eyeshadow lambert`,
`AIT/Skin True Face` and `AIT/Skin Translucency simple`. Their prefab definitions
are context, not proof of current fragment survival. The current Face2 port
`src/render/scene.py` explicitly defers eyeshadow and namida because their shader
formulas are incomplete. It cannot supply an original-game visibility certificate.

## Projected size and raw geometric depth

The actual paired 512×512 AA1 camera has orthographic size `1.12654757`, distance
`8.935848`, target `[0.09552551,14.7151775,0.04619336]`, near `0.08`, far `1000`,
and projection diagonal `0.8876678`. Both previous saved pixel-marker reports
reject `projection` mismatch. All nine views have correct actual same-frame pose
and geometry pairing; all nine still lack a pixel convention certificate in that
preserved first pass. The later coordinate-only update below leaves these
historical failures unchanged.

With the explicitly diagnostic CPU projection, segment lengths are approximately
`0.000824–0.294604` pixels. Subpixel length is an observation, not an exemption.
A nonzero crossing may influence neighboring pixels, shading, shadows or other
views; a zero visual change does not prove safety or invisibility.

Raw two-sided rays through all 84 intersection midpoints hit other geometry
closer than the crossing by more than that pair's existing epsilon. Eye crossings
first encounter lashes or namida; tooth crossings first encounter another head
triangle. Exact hit triangle IDs, distances, barycentric weights and depth margins
are recorded. Rays retain all eight active head meshes and do not apply unknown
shader culling or discard formulas. These observations can prioritize acquisition,
but cannot certify that a crossing is inside, alpha-clipped or actually occluded.
All visible/interior/anatomical classifications remain `unknown`.

## Minimal next native acquisition

The root agent owns game operations. `capture_requirements.json` supplies exact
MCP `camera`, `output` and `scene` options. Minimal additional evidence is:

1. A fresh 512×512 AA1 marker with the exact projection above and the actual
   loaded MVID/source/IL binding. Re-run the certificate on each of the nine old
   targets; projection equality is checked, not inferred from equal resolution.
2. One original-shader full capture of the **new actual lowered pose**, with
   `output.geometry_out`, `output.geometry_texture_dir` and
   `scene.freeze_pose=true`. Keep all eight head meshes, source topology,
   UV0/UV1, material/texture descriptors, transforms and blendshapes. Preserve
   actual texture PNGs, instance IDs, hashes and fresh 2×2 linear/sRGB readback
   calibration. A new pose requires fresh intersection analysis, not reuse of
   these world segments. A texture dump alone does not certify clipping.
3. Actual frame/pose/capture-state pairing, full saved native/MCP envelopes,
   and independent public snapshot, head/body base, expression and ABMX readbacks
   before/after. Preserve all renderer/light states and any restoration errors.
   Do not infer restoration from defaults or only a summary boolean.

If conditional RGB evidence is useful, retain the established eight ocular
blocker subsets plus a full repeat under `isolated_layer`. All existing geometry,
state, RGB noise and effect thresholds remain unchanged. Additional per-renderer
effects for eyebases, tooth and head can be measured separately. They answer only
whether that renderer changes final RGB under the particular other states; they
do not identify a specific intersecting triangle's surviving fragment.

The current bridge has no independently certified original-shader per-renderer
depth/fragment survival output. That is an explicit missing proof, not a request
to implement a shader feature now. A future depth proof would need actual shader
survival/depth/triangle identity at the same camera, pose and pixel sample.
Geometry depth, an override shader, texture alpha=1, or zero ablation RGB cannot
substitute for it. A magnified ROI changes the actual projection and requires a
fresh matching marker; enlarging this diagnostic PNG does not add evidence.

## Reproduction and checks

Use only `Face2Parameter/.venv/Scripts/python.exe`. `tools/crossmesh_semantics/`
contains `analyze.py`, `asset_compare.py`, `finish_report.py` and five meaningful
unit checks for a known crossing under translation, nonintersection, affine
barycentric UV transfer, ordered submesh mapping, and source/frame/state pairing
tampering. Tests pass. The report hashes source code and every input. No existing
production tool, quality gate, threshold, shader, DLL, goal or game state is edited.

## Fresh matching marker update

The root agent subsequently acquired
`HS2Mod/artifacts/infrastructure_live_20261005/lowered_projection_marker_v1/response.json`,
independently hash-verified as
`0bc2cf6393c4285738c9a0010c61228bb58fc674d5c3f5c7a938d96e78f126d0`.
Fresh PNG remeasurement validates top-left orientation and the half-pixel
convention. Its AA1 guarded offset interval is `[0.4726053,0.53339084]`; that uses
the unchanged preregistered 0.003-pixel guard rather than a fitted offset.

New output `outputs/crossmesh_semantics_20261005/actual_lowered_v2_freshmarker/`
retains a fresh marker measurement, nine individual `PixelCertificate` instances,
nine newly labeled overlays, and six scope rejection checks: projection, MVID,
AA, allowMSAA, near clip and color space tampering. All nine saved recent lowered
views pass exact scope and actual PNG hash binding; all six altered scopes reject.
The maximum projected segment length remains `0.294604` px, all 84 raw rays still
have foreground geometry, and shader visibility/interior/anatomy remain unknown.
The first-pass report and nine UNCERTIFIED overlays are preserved. Its source
snapshots are saved alongside it to retain the exact previous script bytes.

The fresh marker satisfies step 1 of the minimal acquisition list only. Current
pose-bound texture readback and original-shader fragment/depth proof remain missing.
