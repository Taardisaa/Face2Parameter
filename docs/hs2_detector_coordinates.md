# Detector crop and heatmap coordinates

This tool validates **coordinate infrastructure**, using the locally installed `face_alignment` 1.5.0 utilities and OpenCV 4.13.0. It never constructs `FaceAlignment`, loads/downloads weights, runs FAN, operates HS2, or certifies anatomical landmarks. Synthetic heatmaps exercise the installed decoder; actual saved/reloaded coordinate-encoded PNGs exercise the installed crop and resize implementation.

```powershell
& .venv/Scripts/python.exe tools/detector_coordinates/verify.py `
  --out-dir outputs/detector_coordinates_20261004

& .venv/Scripts/python.exe -m unittest discover -s tools/detector_coordinates -v
```

The measured evidence is `outputs/detector_coordinates_20261004/report.json`, with raw/cropped PNGs, per-case trace JSON, file hashes, observed integer bounds, padding and actual resize calls.

## Installed implementation and source binding

The inspected `utils.py` SHA-256 is `1d93060352770d71a4ba63561a71b688b283bab8fe743e7bbd15e2bc5fa43824`; `api.py` SHA-256 is `74c8dd5134fc461b554a6ed3f59d8199c4e1f95ce18f7465f290a33bb9686d7d`. The benchmark records NumPy 2.4.3 and Torch 2.11.0+cu130. The actual installed source, runtime versions and RGB array hashes are saved in every crop trace. A consumer rejects changed utilities/API hashes, OpenCV/Torch/package versions, inconsistent stage dimensions or another interpolation method.

In the installed API, the detected bbox supplies its center; center y is shifted upward by 0.12 times bbox height, and scale is bbox width plus height divided by the **actual detector's** `reference_scale`. The benchmark uses 195. Production callers must preserve their actual center, scale and detector reference scale; a default reference scale is not evidence for another detector.

The actual crop implementation:

1. Calls Torch `transform([1,1], center,scale,resolution,invert=True)` and `transform([resolution,resolution], ...)`. It uses float32 matrix arithmetic, then integer conversion truncating toward zero.
2. Creates a uint8 staging image whose width/height are the actual integer `br − ul`, initialized to zero.
3. Copies the available raw-image rectangle into the appropriate staging-image offset, leaving real padding black.
4. Calls `cv2.resize(..., INTER_LINEAR)` to the requested square resolution.

`trace_crop` spies on those actual transform and resize calls, rather than merely reimplementing an ideal affine matrix. It restores both functions in `finally`. Instrumentation is process-local and **must run sequentially**, because temporarily replacing utility functions is not thread-safe.

## Centers, edges and the actual crop inverse

Raw PNG and crop-array coordinates use top-left **integer pixel-center indices**: `(0,0)` means the center of the first pixel. Edge coordinates put that center at `(0.5,0.5)`.

The installed 64×64 decoder uses one-based argmax indices, quarter-pixel sign offsets from neighboring heatmap values, then subtracts 0.5. Thus its decoded value `q` has the declared heatmap-edge convention:

```text
q = zero_based_argmax_index + 0.5 + quarter_sign_offset
crop_edge = q * crop_resolution / heatmap_resolution
crop_integer_center_index = crop_edge - 0.5

stage_integer_center_index = q * actual_stage_extent / heatmap_resolution - 0.5
raw_integer_center_index = actual_ul_int + stage_integer_center_index
```

OpenCV's stage-border replication clamps stage indices to `[0, actual_stage_extent − 1]`. The mapper records this clamp separately. Padding is not border replication: a stage pixel may be a zero pad pixel rather than raw-image content. Every mapped point therefore includes `raw_interpolation_footprint_fully_in_image` and status `mapped_grid_only` or `uncertain_padding`. It never substitutes a fitted offset or invents a valid raw observation in padding.

This identifies the geometric heatmap-to-crop **grid convention**. It does not establish the network's receptive-field anchor, learned heatmap target bias, semantic peak meaning, detector bbox localization, or FAN anatomical accuracy. The bbox is a recorded input to this experiment. Even an in-image mapped point remains `mapped_grid_only`, not an anatomical certificate.

The inspected native decoder hardcodes parts of its boundary logic to 64 and uses height in its row-index calculation. The tool consequently rejects non-square or non-64 heatmaps instead of silently extending its certificate to unsupported shapes.

## Why the default original coordinates lose precision

The installed native `preds_orig` does not invert the actual integer crop rectangle. It uses the ideal center/scale transform and then `astype(int32)`:

```text
ideal_library_float = center - 100*scale + q * (200*scale)/64
native_preds_orig = trunc_toward_zero(ideal_library_float)
```

This loses the fractional original coordinate and ignores differences caused by the actual `ul/br` truncation, actual resize extent and the edge-to-center `−0.5` translation. Negative values truncate toward zero with the opposite bias from positive values. The tool retains the native output, ideal unquantized library inverse, crop-faithful floating-point output, and both error terms. It does not overwrite or relabel native landmarks as already validated grid coordinates.

Stable integration API in `tools/detector_coordinates/mapping.py`:

- `trace_crop(uint8_rgb, center, scale, resolution=256)` returns the actual cropped array and trace.
- `heatmap_to_raw(decoded_q_Nx2, trace, heatmap_resolution=64)` maps the declared edge-coordinate grid and preserves float output and validity flags.
- `decode_heatmaps(heatmaps_1xNx64x64, trace)` executes the installed decoder and returns native integer coordinates alongside crop-faithful float coordinates, score, quarter-offset values and explicit error/validity fields.

An API wrapper can trace `api.crop` and capture `api.get_preds_fromhm` output without changing weights or the library's native return value. Preserve native coordinates separately, require the exact recorded trace, and distinguish grid validity from semantic acceptance.

## Independent PNG experiment and observed errors

Five cases cover multiple raw sizes, fractional bbox coordinates and padding on all four sides. The input coordinate code is `R=(4*coordinate) mod 256`, `G=floor(coordinate/64)`, `B=255`. Reading the actual cropped RGB gives an independent coordinate estimate `R/4 + 64*G`, with quarter-pixel code precision. The input is saved and reloaded before the actual library crop executes.

Every destination center is evaluated, with two explicit exclusions: interpolation footprints that touch padding/outside image, and footprints crossing the discontinuous 64-pixel encoding seam. These exclusions are counted in the report. No points are selected according to their error, and no slope/offset/gamma correction is fitted. The fixed maximum measurement guard is 0.25 pixels; it covers uint8 interpolation rounding and code quantization.

| Synthetic case | Raw image size | Actual ul → br | Padding left/top/right/bottom | Ramp x/y RMSE | Native integer vs actual grid maximum difference |
|---|---|---|---|---|---|
| small_inside | 160×128 | (24,14) → (127,117) | 0/0/0/0 | 0.08055 / 0.08028 px | 0.41797 px |
| fractional_square | 320×320 | (49,43) → (260,254) | 0/0/0/0 | 0.08070 / 0.07780 px | 0.58984 px |
| large_rect_inside | 641×479 | (177,94) → (427,344) | 0/0/0/0 | 0.08021 / 0.07860 px | 0.67969 px |
| edge_left_top_padding | 256×192 | (−66,−72) → (126,120) | 66/72/0/0 | 0.12462 / 0.12460 px | 1.25000 px |
| edge_right_bottom_padding | 513×257 | (370,120) → (593,343) | 0/0/80/86 | 0.08051 / 0.08204 px | 1.08203 px |

Each axis independently reads 24,963–64,768 valid output pixels. The maximum observed ramp error is 0.18555 pixels. Adding an incorrect `+0.5` center translation gives 0.37592–0.62432-pixel RMSE, distinctly above the measured correct-grid errors. The small residual biases from uint8 interpolation are retained; notably the padded case produces opposite-sign x/y rounding bias near 0.124 pixels. They are not fitted away.

The native integer truncation alone differs from the ideal library float by as much as 0.99427 pixels in this benchmark. These are **observed benchmark values, not general upper bounds** on the native-vs-crop-grid discrepancy. Root's separate seven-view real cached-FAN run, `outputs/surface_calibration_aa1_20261004/fan68_crop_grid_candidates.json`, observed native-vs-float-grid maximum differences up to 2.1211 pixels. Those real landmarks still have unvalidated semantic correspondence; the observation documents coordinate-convention differences, not anatomical improvement.

The resulting `validated_crop_grid` status certifies the measured crop/resize grid and decoder convention under recorded source/runtime contracts. Padding, discontinuous encoding seams, FAN anchors, learned biases and anatomy remain explicitly outside that certificate.
