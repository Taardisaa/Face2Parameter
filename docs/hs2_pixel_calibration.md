# HS2 saved-PNG pixel calibration

`tools/pixel_calibration/analyze.py` is an offline observer of **actual saved PNG RGB samples**. It does not contact the game, change a character, initialize/download a model, or use exported `viewport_center` / `viewport_corners` as evidence. All Python commands use this project's virtual environment.

```powershell
& .venv/Scripts/python.exe tools/pixel_calibration/analyze.py `
  C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261004/pixel_calibration/manifest.json `
  --out outputs/pixel_calibration_20261004/report.json

& .venv/Scripts/python.exe -m unittest discover -s tools/pixel_calibration -v
```

The input can be one capture response JSON (optionally `--png alternate.png`) or a manifest:

```json
{"cases":[{"name":"ortho_aa1","metadata":"ortho_aa1.json","png":"ortho_aa1.png"}]}
```

The live capture manifest's `response_path` and `png_path` fields are also accepted. Relative paths resolve against the manifest/metadata directory. Response envelopes named `response`, `result`, or `data` are supported. Exit status is zero only when every requested case is `validated`; `failed` and `uncertain` produce a nonzero exit.

## Independent observations

The accepted renderer graph is `isolated_world_space_unlit_markers`, with at least six uniquely named markers whose RGB channels are exact 0 or 1, actual primitive world-space quad vertices, an opaque black background, and exported capture-camera row-major matrices using column vectors. The tool checks:

- Actual RGB/RGBA image dimensions, opaque alpha and black boundary.
- Exactly one connected color component for each marker; solid color interiors without holes; contiguous rows; no unknown colored pixels.
- Every unassigned pixel is exactly black, with at least half the image independently observed as background.
- Finite, nonsingular CPU view/projection matrices; valid non-square/subviewport pixel rectangles; positive homogeneous `w`; in-frustum vertices; four actual corners forming the claimed axis-aligned projected rectangle.
- Observed support bounding edges, actual integer-index centroids and, when supported by color evidence, coverage-weighted subpixel centroids.

Unknown colors, blurred/mixed interior pixels, multiple blobs, an invalid projection, or a contaminated background fail. PNG and metadata SHA-256 values provide provenance only; they do not establish correctness. The report also records renderer scope and color histograms, including on failed cases.

## Y direction and the half-pixel distinction

CPU `P * V * [world,1]` yields NDC; `pixel_rect` converts NDC to bottom-left **pixel-edge** coordinates. The analyzer evaluates both PNG y directions and both edge offsets `0` and `0.5` independently:

```text
PNG edge y = height - bottom_left_edge_y   # top-left PNG hypothesis
integer pixel index = PNG edge coordinate - 0.5
```

`image_y_direction.status` and `pixel_center_convention.status` are separate decisions. A confident observation of a bottom-left raster or zero offset is reported as that measured convention, with `matches_expected_top_left_half_pixel_convention=false`; it is not silently changed to the expected convention.

AA1 provides interval evidence stronger than rounding a projected center. If an observed row occupies indices `first..last` and its independently projected continuous edges are `L..R`, the common pixel sample offset must satisfy:

```text
L - first <= offset < L - first + 1
R - last - 1 <= offset < R - last
```

The tool intersects these inequalities over every marker and both image axes. It certifies the offset only if exactly one candidate (`0` or `0.5`) remains and the raw interval width is below 0.4 pixels. A fixed 0.003-pixel measurement guard covers exported floating-point camera arithmetic and rasterizer snapping; it was declared before examining live captures and is not a fitted offset. The report exports both raw and guarded intervals. Direct3D11 snaps viewport vertices to eight fractional bits, so the raw continuous-geometry interval can differ slightly from the raster interval. See the primary [Microsoft Direct3D11 functional specification, coordinate snapping](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm#3.4.1%20Coordinate%20Snapping). Contradictory intervals beyond the guard fail. Broad intervals remain `uncertain`, even if centroid RMSE happens to favor the expected answer.

## Multisampling evidence and limits

Declared render-texture `anti_aliasing=4` or `8` does **not** establish effective multisampling. After independently verifying uniform black background, the tool checks edge-channel levels against fixed `k/N` linear quantization and fixed sRGB quantization. It does not fit gamma, offset or a per-marker correction. A transfer encoding is accepted only when exactly one discrete model is compatible to one RGB code value, and identifiable markers must agree on a single capture-graph encoding. Pure full-coverage colors alone leave the encoding and effective multisampling undetermined.

For identified AA4/AA8 coverage, weighted integer-index centroids compare the two offsets using fixed acceptance bands: all selected residuals at most 0.20 pixels, alternative RMSE at least 0.32 pixels, and the RMSE separation at least 0.20 pixels. These are an explicit measurement precision guard, not a universal proof for arbitrary vendor sample patterns, resolve filters, or postprocessing. If evidence fails to distinguish a half pixel, the result is `uncertain`. The report includes actual levels, decoded coverage mass, per-marker residuals and all candidate RMSE values for review. Manifest cases with identical PNG hashes but different declared AA levels are listed separately.

This certificate applies only to the measured isolated unlit-marker capture graph and its camera. It does not certify character materials, postprocessing, surface landmarks, blendshape evaluation, skinning, or another graph. Downstream consumers must check both separate statuses, the measured convention, and renderer scope.

## Tests and live evidence

The synthetic parser/raster tests cover a correct half-pixel raster, an actual zero-offset raster, a reversed y raster, non-square/subviewport projection, malformed/singular projection, invalid RGB/background, duplicate blob, linear/sRGB multisampling, blurred edge levels, wrong dimensions and an intentionally ambiguous raster that must remain `uncertain`. These tests verify failure handling and measurement logic; they do not substitute for a Unity capture.

The first 2026-10-04 preflight captures are preserved separately in `outputs/pixel_calibration_20261004/preflight.json`. They fail the exact primary/secondary RGB marker contract because Unity's legacy `Color.yellow` exported and rendered `[255,235,4]`. Each PNG has only black plus six solid marker colors, with no fractional coverage colors. The square AA1/AA4 PNGs have identical SHA-256 values. Therefore those captures provide no effective multisampling certificate. The corrected captures below replace this color preflight; an effective MSAA certificate still requires independently observed coverage evidence.

### Corrected v2 live capture

The genuine stdio-captured `HS2Mod/artifacts/infrastructure_live_20261004/pixel_calibration_v2/manifest.json` was analyzed into `outputs/pixel_calibration_20261004/report_v2.json`. Its before/after manifest reports `state_preserved=true`. All five corrected PNGs passed pure-color, connected-component and black-background checks, and all five independently validated top-left PNG y direction. No PNG contains fractional coverage levels: each has exactly black plus six 0/255 colors.

| Case | Actual camera rendering path | allowMSAA | Y direction | Half-pixel offset |
|---|---|---|---|---|
| ortho_square_aa1, 512×512, roll 0 | Forward | false | validated | validated |
| ortho_square_aa4, 512×512, roll 0 | Forward | true | validated | uncertain |
| ortho_rect_roll_aa4, 641×479, roll −13 | Forward | true | validated | uncertain |
| perspective_rect_roll_aa4, 641×479, roll 17 | DeferredShading | true | validated | uncertain |
| perspective_small_aa8, 320×256, roll 0 | DeferredShading | true | validated | uncertain |

The AA1 case's independently measured support edges constrain the common center offset to `[0.4756060815, 0.5303994037]` pixels, which includes 0.5 and excludes 0.0. Therefore the certified conversion is `x_index = x_edge − 0.5`, `y_index = height − y_bottom_left_edge − 0.5`. The opposite-y hypothesis has 175.72-pixel RMSE. The ordinary centroid quantization RMSE is 0.15046 pixels; certification uses the stronger edge interval rather than treating this quantization error as a fitted correction.

The AA1 and AA4 square PNG SHA-256 values are identical. `allowMSAA=true` and render-texture AA metadata consequently provide no effective-MSAA certificate. The five-case report is intentionally `uncertain`, not an aggregate pass.

The v2-only certified scope is Unity `2018.4.11f1`, bridge `0.30.0`, loaded assembly MVID `f7becf90-9a8d-4997-8188-d9d39f0f9bcc`, Direct3D11, Linear color space, reversed-Z true, GPU UV starts at top true, **the isolated Unlit/Color marker graph**, CPU camera matrices with bottom-left viewport, Forward actual rendering, AA1/allowMSAA false, HDR true, quality AA 0, orthographic 512×512/full pixel rect/roll 0. Exported hashes and camera matrices are in the report. The v3 captures below add perspective-path evidence; no character-surface or shader certificate follows from this marker test.

### Final v3 live scope

`HS2Mod/artifacts/infrastructure_live_20261004/pixel_calibration_v3/manifest.json` contains eight genuine captures, including three additional AA1 variants. The final offline result is `outputs/pixel_calibration_20261004/report_v3.json`. The aggregate remains `uncertain` because the four AA4/AA8 cases still lack fractional coverage evidence. Consumers must examine individual case statuses.

All four AA1 cases independently certify top-left PNG direction and a 0.5 edge-coordinate offset:

| AA1 case | Size; camera orientation | Actual rendering path | Raw common offset interval | Guarded interval |
|---|---|---|---|---|
| ortho_square_aa1 | 512×512; yaw 0, pitch 0, roll 0 | Forward | [0.475605, 0.530398] | [0.472605, 0.533398] |
| ortho_rect_roll_aa1 | 641×479; yaw −30, pitch 0, roll −13 | Forward | [0.472708, 0.532726] | [0.469708, 0.535726] |
| perspective_rect_roll_aa1 | 641×479; yaw 30, pitch 0, roll 17 | DeferredShading | [0.466924, 0.520371] | [0.463924, 0.523371] |
| perspective_small_aa1 | 320×256; yaw 60, pitch 11, roll 0 | DeferredShading | [0.501599, 0.502027] | [0.498599, 0.505027] |

The last case's **raw** interval misses 0.5 by 0.001599 pixels, and certification explicitly depends on the predetermined 0.003-pixel guard. It must not be presented as a zero-tolerance exact result. Every guarded interval contains 0.5 and excludes 0.0 by a large margin; the smallest opposite-y RMSE is 87.88 pixels. No interval is adjusted to fit the expected answer.

For an ordinary perspective capture, the best available coordinate-transfer evidence is `perspective_rect_roll_aa1` (641×479, FOV 24, roll 17, yaw 30, pitch 0) or `perspective_small_aa1` (320×256, FOV 24, roll 0, yaw 60, pitch 11), using the exact exported camera parameters of that case. Bind to bridge MVID `f7becf90-9a8d-4997-8188-d9d39f0f9bcc`, bridge `0.30.0`, Unity `2018.4.11f1`, Direct3D11, Linear, ARGB32, AA1, allowMSAA false, HDR true, actual DeferredShading, quality AA 0, full pixel rect, bottom-left viewport, and matching PNG dimensions. Forward orthographic and DeferredShading perspective are separate measured scopes. Other dimensions, AA, runtime DLLs, rendering paths or serialization paths need new evidence; source/PNG hashes must be checked before consuming a report.

The game-source path is shared through `RenderHead -> cam.Render -> SaveRenderTexture`; the isolated marker branch changes culling/background and marker scene contents. A separate strict binding consumer must verify this transfer and the live ordinary-camera scope rather than promote the whole character renderer to a validated marker graph. Surface correspondence and character shader/material validation remain outside this certificate.
