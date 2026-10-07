# Offline multi-view fixed surface consensus

`tools/surface_consensus` consumes actual paired capture cameras, geometry, an independent LBS report, a saved-marker pixel report, and image-only FAN crop-grid observations. It creates concrete renderer/triangle/barycentric material definitions and tests independent views before marking a candidate operationally accepted. It does not contact or change HS2.

The real AA1 run on 2026-10-05 accepts **zero anatomical or operational anchors** under the fixed 2-pixel policy. Three numeric material definitions are retained as rejected proposals, so later work has exact reproducible surfaces rather than only landmark names. Their arithmetic and source binding are valid; their detector semantic definitions are unvalidated. `outputs/surface_consensus_20261005/material_definitions.json` is the machine-readable definition file, and `report.json` contains all measurements and rejection reasons.

## Actual result

All seven 512×512 PNGs were consumed. Extreme yaw observations (−90°, −60°, +60°, +90°) were explicitly excluded from stable-point fitting because unverified hidden-side FAN predictions cannot provide acceptance evidence. Only −30°, 0°, +30° participate. The detector heatmap score is never treated as a visibility probability.

| Candidate | Eligible views | All-view maximum error | Independent LOO maximum error | LOO normal spread | Result |
|---|---:|---:|---:|---:|---|
| FAN30 purported nose apex | 2, 3, 4 | 1.3583 px | 2.1961 px | 14.83° | fails heldout 2 px gate |
| FAN48 purported mouth corner | 2, 3, 4 | 2.6571 px | 3.1382 px | 11.55° | fails training and heldout gates |
| FAN54 purported opposite mouth corner | 2, 3, 4 | 1.6713 px | 2.2696 px | 33.08° | fails heldout and 20° normal stability gates |
| FAN36, 39, 42, 45 purported eye corners | none | — | — | — | each candidate ray to `o_head` is geometrically blocked in all eligible views |

Eye blockers are `o_eyeshadow`, `o_namida`, and/or `o_eyelashes`. This is a geometry-only rejection, **not proof of rendered opacity**: shader alpha, culling, depth policy and the complete capture renderer graph are not certified. These blockers must not be silently deleted to make the eye candidates pass. Separate material evidence may later justify a revised visibility model.

The proposed fixed definitions on the AA1 head source SHA `b2c3831da3e025b23a8dc6231a8cd73ed3540536f81b3aea1aa4b9459e360ece` are:

| Candidate | Triangle | Ordered vertex IDs | Barycentric weights |
|---|---:|---|---|
| FAN30 | 6072 | 3478, 3474, 3483 | 0.1030044301, 0.8943525980, 0.0026429719 |
| FAN48 | 6442 | 3627, 3629, 3630 | 0.3185714491, 0.2306644083, 0.4507641426 |
| FAN54 | 2097 | 940, 1279, 1280 | 0.2294612945, 0.2493844387, 0.5211542668 |

Full precision, renderer path, submesh, implementation hashes, capture/geometry/image-observation source hashes, head ID and pose signature are stored in the JSON. These are rejected proposals, not a training label set. Jaw indices 0–16 are explicitly view-dependent apparent contour observations and never receive fixed material definitions. Eyebrow texture samples and the remaining FAN indices are outside this stable-point candidate scope.

## Literal mesh definitions independent of FAN semantics

`geometric_definitions.json` additionally freezes `head_front_support_vertex`: the unique directional support maximum over every vertex referenced by actual `o_head` submesh 0, toward the certified front capture camera. On this capture it is vertex **3468**, world `[0.10432, 14.3191023, 0.863654137]`. This is a literal head-surface measurement; it is not labeled an anatomical nose apex. It is stored as an incident source-bound triangle with one-hot barycentric weights. Later deformation must follow those fixed weights, rather than reselecting the maximum and thereby moving the material identity.

The actual head has one asset-provided submesh, so this asset supplies no named nose/lip submesh region. The same output inventories all actual indexed boundary components: 366 boundary edges, seven connected components, and zero nonmanifold edges. Their vertex lists, degree-two loop status and world bounds are recorded. No position welding, UV guess or bone proxy is used. Two 32-vertex closed loops occur around world Y 14.008–14.156; their anatomical labels remain null. Indexed boundaries can represent seams, and a visual resemblance to a lip is not semantic certification.

For independent position review, `visual_review/support_7view_overlay_sheet.png` overlays a small cross and point ID on copies of all seven original PNGs; `support_7view_crop128_strip.png` shows a 128-pixel crop around each projection. Original PNGs remain unchanged. The source and derived PNG hashes, projection coordinates, crop bounds and geometric visibility statuses are recorded in `visual_review_manifest.json`. These are diagnostic projections, not independently measured landmark truth. Cyan means no geometric blocker, while red means geometric visibility or clipping rejected the point; neither certifies shader/material visibility.

## Evidence and algorithm

1. Verify capture, geometry, LBS report and observation file SHA bindings. Recalculate actual independent bone-world LBS versus BakeMesh with `analyze_snapshot` at fixed normalized tolerance `1e-5`; compare numerical residuals with the original report. The `scale_free_trs` world conversion must independently pass. Two inactive tongue copies have indistinguishable conversion hypotheses, and are allowed because this selected conversion numerically passes; no universal convention is inferred.
2. Create a `PixelCertificate` separately for every raw PNG by remeasuring saved marker pixels and checking the scoped runtime/camera/readback contract. Use `Camera.from_capture` without diagnostic mode. Pose pairing comes from matching actual before/after pose signatures, view and geometry signatures, and identical frame counts; a caller-supplied pose attestation is not accepted.
3. Reproduce each actual FAN crop using the raw PNG and installed implementation; verify the recorded trace and floating heatmap-to-raw mapping. The heatmap feature-anchor convention and FAN semantics remain unvalidated. Ignore cached geometry ray hits in the FAN report; use only independent float image coordinates and crop traces.
4. Screen each candidate observation with the existing `observe` geometry routine, its raw viewport, finite near/far ray, target `o_head`, and blockers. Require at least three eligible observations. Exclude profile/hidden-side-risk views before any fitting.
5. Solve bounded-ray triangulation: `A = Σ(I − ddᵀ)`, `b = Σ(I − ddᵀ)o`, and `X = A⁻¹b`. Reject parallel/anti-parallel rays, insufficient ray angle, condition number above 100, and points outside any near/far segment. Minimum useful ray angle is 10°.
6. For every target triangle, minimize the same perpendicular-ray-distance quadratic over barycentric weights. Enumerate the interior stationary point and all three clamped edge minima, then screen proposed minima in increasing objective order for in-view geometry visibility and normal incidence ≤75°. The retained simplex point is an exact numeric material definition. No camera, translation, scale, mesh deformation or facial parameters are fitted.
7. Fit again for **every leave-one-view-out split**, excluding the heldout coordinates from triangulation and material-point selection. Reuse `follow` and `validate_reprojection` to compare the fitted material point with the heldout independent image observation. All training and heldout pixel errors must be ≤2 px; all normals must remain within 20° across LOO fits; heldout visibility and incidence must pass. The all-view fit's pixel residual is not independent validation.

The ray-distance minima are exact on each triangle. Screening only those minima can conservatively reject a case whose valid visibility-constrained solution lies elsewhere inside the triangle; the solver does not claim a global optimum after adding visibility constraints. Normal incidence uses absolute normal dot products because winding/cull behavior is not certified. The reported inverse ray information is a numerical conditioning measure, not a calibrated detector confidence interval.

The 2 px policy was fixed before evaluating residuals. A changed threshold is a new explicit policy and cannot convert this saved run into success. Nose FAN30 is the closest numerical proposal but still fails. More accurate independent observations, feature-anchor calibration, and reviewed material visibility are needed before anatomical correspondence can be claimed. A later morph or expression can use the fixed barycentric point via `follow`, but needs independent transport verification; this snapshot does not establish that the FAN semantic point follows it.

## Run and verify

Run from the Face2Parameter repository root, always with its venv:

```powershell
.venv/Scripts/python.exe -m tools.surface_consensus.run `
  --capture C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261004/paired_capture_aa1/response.json `
  --geometry C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261004/paired_capture_aa1/geometry.json `
  --lbs-report outputs/unity_parity_20261004/paired_capture_aa1.json `
  --pixel-report outputs/pixel_calibration_20261004/report_v3.json `
  --points outputs/surface_calibration_aa1_20261004/fan68_crop_grid_cuda_candidates.json `
  --out outputs/surface_consensus_20261005

.venv/Scripts/python.exe -m unittest tools.surface_consensus.test_consensus -v
```

The CLI writes a diagnostic report even when no candidate passes, and exits normally for a valid analysis. A broken evidence contract exits with code 2. Check `accepted_ids` and each candidate's `accepted` field rather than interpreting CLI exit zero as anatomical success. The fifteen exact synthetic tests cover parallel and anti-parallel rays, finite near/far segments, viewport bounds, convex simplex interior/edge solutions, independent multi-view success without anatomical certification, occluded observations, shifted landmarks, heldout exclusion/rejection, source/model identity changes, triangle-order changes, profile exclusion, moving jaw observations, literal geometric support without anatomical labeling, fixed material follow despite a changed extremum, and pre-reconstruction source rejection.

## Fixed-point transport through actual native and ABMX probes

`tools.surface_consensus.transport` follows the existing vertex 3468 / triangle 6044 / barycentric `[0,0,1]` definition. It independently reconstructs each actual saved mesh using bone-world LBS plus active expression deltas, compares that point and its triangle normal with the independently validated BakeMesh world conversion, and verifies source SHA, renderer identity and ordered topology. It never recomputes an extremum, substitutes a triangle or transfers the definition across head models. World differences from a group baseline are actual snapshot differences, not fitted translation/scale compensation or an isolated parameter derivative.

Two capture generations were analyzed:

| Dataset | Actual cases | PNGs | Numeric fixed-point transport | Strict pixel / pose / geometry projection |
|---|---:|---:|---:|---|
| Old `head2_cases` and head2 subset of `paired_base_cases` | 18 | 24 | 18 pass | 0 pixel certificates; 9 actually paired PNGs; no uncertified projection performed |
| New `infrastructure_live_20261005/surface_transport` | 9 | 63 | 9 pass | 63/63 pixel certificates, 63/63 actual pose/frame pairings, 63/63 geometric projections |

The old 24 PNGs lack eleven runtime pixel-scope fields (including module MVID, API, AA and readback settings). Their projection scale also differs from the reference AA1 capture. Fifteen PNGs from `head2_cases` were captured on different frames without real pose-signature pairing. Nine `paired_base_cases` PNGs have actual pairing, but still lack strict pixel evidence. The old-input run retains geometry-only numerical results and explicit image rejection reasons in `outputs/surface_consensus_20261005/transport/report.json`; it does not copy later metadata, create a diagnostic camera, or project rejected images. The maximum fixed-point LBS/BakeMesh difference is `2.57404e-6` Unity units and maximum triangle-normal difference is `0.000501304°`.

The new capture module MVID is `68ab80e6-e0db-4ddd-99ca-46ade7e76020`. Its independently reviewed shared render/readback path and actual saved marker measurements in `outputs/pixel_calibration_20261005/texture_module_report.json` support fresh strict certificates. All nine cases preserve the original head source hash. Maximum fixed-point LBS/BakeMesh difference is **`1.16999e-6` Unity units** and maximum triangle-normal difference is **`0.000418643°`**. `outputs/surface_consensus_20261005/transport_new_module/report.json` contains the aggregate and per-view certificates; separate JSON files and seven-view overlay/crop diagnostics are stored for every case.

| New case | Actual fixed-point world displacement from new baseline | Triangle-normal rotation from baseline |
|---|---:|---:|
| Native 0 = −0.25 | 0.00257745 units | 0.300921° |
| Native 0 = 1.25 | 0.000541250 units | 0.236525° |
| Native 47 = −0.25 | 0.00360114 units | 0.00000191° |
| Native 47 = 1.25 | 0.00775153 units | 0.00000553° |
| ChinTip ABMX scale | 0.0107490 units | 0.00000579° |
| ChinTip ABMX length | 0.00772858 units | 0.00000768° |
| ChinTip ABMX position | 0.00481862 units | 0.000000854° |
| ChinTip ABMX rotation | 0.00179824 units | 0.0000151° |

These world displacements include inherited pose/runtime changes. The support vertex's actual source weights are `cf_J_Nose_tip = 0.507994831` and `cf_J_Nose_t = 0.492005169`; ABMX probes target `cf_J_ChinTip_s`. Bone names describe recorded skinning input and do not define anatomy. No isolated effectiveness claim is made for these interventions, and the nose-area support point is not treated as an affected ChinTip region. This task does not invent another chin point.

All 63 projections are geometric and have **no independent image semantic observation**. They establish source-bound numeric transport and scoped camera projection, not anatomical landmark accuracy or material alpha/depth visibility. Root independently reviewed the original seven-view crop as lying near the visual nose region; that limited visual review does not set anatomical correspondence true.

Reproduce the new-data acceptance without changing the game:

```powershell
.venv/Scripts/python.exe -m tools.surface_consensus.transport `
  --anchor outputs/surface_consensus_20261005/geometric_definitions.json `
  --manifest C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261005/surface_transport/live_cases.json `
  --pixel-report outputs/pixel_calibration_20261005/texture_module_report.json `
  --reference-report outputs/surface_consensus_20261005/report.json `
  --out outputs/surface_consensus_20261005/transport_new_module
```

`--reference-report` supplies a labeled historical scope comparison only. Actual image acceptance consumes the independently measured report supplied by `--pixel-report` for every view, and requires that view's real geometry SHA/signature/frame pairing. `image_acceptance` remains false for independent anatomical/semantic acceptance; calibrated geometry is counted separately in `strict_numeric_geometry_projection_png_count`.
