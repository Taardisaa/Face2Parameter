# HS2 differentiable head geometry chain (2026-10-25)

Purpose: the hybrid strategy of [offline-renderer-research.md](offline-renderer-research.md)
(strategy 2 as the optimisation inner loop, the exact renderer as verifier).  Spec: the
geometry items of HS2Mod `docs/hs2_offline_renderer_deferred_plan.md` (R1: card + ABMX stateful
semantics + all 59 native inputs incl. ears, bones, skinning, explicit sources; R4: gradient
paths, discrete game branches kept, no claim of differentiability everywhere).

Code: `src/render/diffgeo/` (package) and `tools/diffgeo/` (derivation, acceptance, gradient
check, proxy demo, timing, tests).  Bulk outputs: `E:/HS2_offline_renderer_data/diffgeo_20261025/`.
CPU only; no game, RenderDoc or GPU executor was used.  Nothing in `src/render/hs2pipe/`,
`src/render/chin_fit/`, `tools/chin_fit/`, `tools/model_bridge`, `tools/native_head` was changed.

## 1 Chain and sources

| Step | Module | Source of the computation |
|---|---|---|
| card + patch -> bone local TRS | `forward.TorchHeadForward` | statement-for-statement mirror of `CapturedHeadForward.evaluate` (`src/hs2_offline_state.py`), data from the exact chain's own model object (`paired_forward_context.build`) |
| ShapeHeadInfoFemale (native59) | `native_shape` | port of `src/hs2_native_shape.py`: `GetInfo` (+ SliderUnlocker clamp), `ChangeValue`, the generated `ShapeHeadInfoFemale.Update` program, Mono rounding points |
| ABMX 4.4.6 Apply / Reset / CollectBaseline | `abmx` | port of `tools/abmx_replay/model.py` and the reset/collect of `hs2_offline_state.py` |
| Quaternion.Euler, SetLocalR, Quaternion * | `unity` | managed parts as in `unity_transform_math`; native EulerToQuaternion lifted (below); CRT `sinf`/`cosf` executed from UnityPlayer.dll (`hs2pipe.unity_crt`) |
| hierarchy, palette, CPU skinning | `unity` + `native_generated` | lifted from UnityPlayer.dll (below) |
| FBS blend shapes | `paired_skin.unity_blend` (exact port, reused) | probe frame weights; parameter-independent |
| EyeLook `cf_J_look_L/R` | hosted installed IL (`eye_look_acceptance.Harness`) | evaluated exactly at the iterate, held constant (section 4) |
| bridge frozen root (unity_ObjectToWorld) | `session.frozen_root` | PoseFreeze rule of `BridgeCandidateRenderer.frozen_pose_check` |
| bridge camera at a yaw | `camera` | `hs2pipe.bridge_camera` (MakerRenderStyleService.Capture), record non-jittered projection |
| chin objective PROXY | `chin_proxy` | labelled proxy (section 6) |

**Lifted native routines.** `tools/diffgeo/derive_native.py` executes the straight-line SSE
bodies of the installed UnityPlayer.dll (SHA `5034cdc2…`) on symbolic lanes (capstone
disassembly; 4 lane expressions per xmm register; memory resolved through register regions;
bit operations accepted only in the forms the code uses: sign masks, the (0,0,0,~0) lane select,
sign-of-scale products) and writes `src/render/diffgeo/native_generated.py` with one binary32
rounding per float instruction:

| Function | Routine (RVA) | SSA statements |
|---|---|---|
| `gpr_step` | CalculateGlobalPositionAndRotation 0xdda60, loop 0xddb10..0xddc62 | 88 |
| `inv_noscale` | CalculateInverseGlobalMatrixNoScale tail 0xbfdd73..0xbfde77 + packing 0xbff997..0xbff9fb | 57 |
| `local_compose` | CalculateSkinningMatrices prepared, entry loop 0xbda830..0xbdaa72 | 159 |
| `bind_mul` | same, bone loop 0xbdaa90..0xbdabd5 | 112 |
| `skin4` | SkinVerticesGenericCPU<4,1,1> 0x776730, vertex body 0x776790..0x77695e | 132 |
| `euler_case4` | EulerToQuaternion 0x7cc120, order 4 case 0x7cc4ce..0x7cc5ed | 44 |

`derive_native.py --check` regenerates and compares (unit test).  Each function is bit-equal to
the Unicorn-executed original on random inputs, including negative scales and zero angles
(`NativePrimitiveTests`).  Performance-only, value-preserving devices: renderers sharing a root
share the prepared hierarchy (a pose row is the same product whatever the entry order);
gradient-free pose rows, root chains, Euler arguments and ABMX transitions are memoised by the
exact bits of their inputs.

## 2 Arithmetic modes

`fp.Arith('exact')`: float64 tensors rounded to binary32 (`R`) exactly where the game rounds:
after each native SSE operation, at Mono stores / float arguments / returns (C# float
expressions are evaluated in double).  One +, -, *, /, sqrt of binary32 operands rounded through
binary64 equals the binary32 operation (53 >= 2*24+2).  `Arith('f64')`: the same graph without
rounding and with torch sin/cos; used for finite differences.  Branch predicates are always
decided on binary32 values, so both modes follow the same branches.

## 3 Forward equality (exact mode)

`tools/diffgeo/forward_acceptance.py --all` (one process per window; receipt
`tools/diffgeo/forward_acceptance_20261025.json`, full reports `forward_acceptance_v3/`).
Compared bit for bit: the local TRS of every transform at N-2/N-1/N (2727 / 2712), the ABMX
branch lists, the 40-byte skinned vertices (position, normal, tangent) and unity_ObjectToWorld of
every mesh renderer at N-2/N-1/N (15 / 14 renderers: the 8 head surfaces, hair (v6.1), body,
clothes); the frozen bridge root of every head renderer in all 16 bridge renders; with
`--capture-link` the torch bytes against the RenderDoc-captured uploads.

| window | patch | locals | streams (vb + O2W) | branches | hosted EyeLook on torch locals | frozen root | torch = captured upload |
|---|---|---|---|---|---|---|---|
| w_766d8184 | baseline (card) | 2727/2727 | 45/45 | equal | 45/45 | 144/144 | 41/41 |
| w_75e17db3 | cf_J_ChinTip_s | 2727/2727 | 45/45 | equal | 45/45 | 144/144 | 41/41 |
| w_8a42b979 | cf_J_ChinLow | 2727/2727 | 45/45 | equal | 45/45 | 144/144 | 41/41 |
| w_0b3351d6 | cf_J_Chin_rs | 2727/2727 | 45/45 | equal | 45/45 | 144/144 | 41/41 |
| front | baseline (card) | 2712/2712 | 42/42 | equal | 42/42 | 128/128 | - |

Extra patches (v6.1 baseline window and the chin-fit front window): the chin controls of
docs/hs2_chin_fit.md through `controls.patch_from_q` (q = 0, four random q in [-1, 1]^36, three
single-control extremes incl. the mirrored CheekLow pair), two native59 vectors (all 59 sliders,
one with sliders exactly at the key ends 0 / 1 and at 0.5), and the removal of a card modifier
(Reset + body/face update path).  Unit tests add the `captured` baseline mode with and without
the UpdateBaseline native refresh, a new modifier on a bone without card modifier, and the
length branch (chin_rs window, length 1.0189).

| window | patch | streams | locals | EyeLook | exact s | torch s |
|---|---|---|---|---|---|---|
| w_766d8184 | chin_q0_encoded | 45/45 | 2727/2727 | window state | 0.53 | 0.90 |
| w_766d8184 | chin_q_random_0 | 45/45 | 2727/2727 | window state | 0.90 | 0.98 |
| w_766d8184 | chin_q_random_1 | 45/45 | 2727/2727 | window state | 0.88 | 0.95 |
| w_766d8184 | chin_q_random_2 | 45/45 | 2727/2727 | window state | 0.88 | 0.98 |
| w_766d8184 | chin_q_random_3 | 45/45 | 2727/2727 | window state | 0.79 | 0.96 |
| w_766d8184 | chin_tip_scale_x_+1 | 45/45 | 2727/2727 | window state | 0.90 | 0.94 |
| w_766d8184 | chin_low_pos_z_-1 | 45/45 | 2727/2727 | window state | 0.96 | 0.95 |
| w_766d8184 | cheeklow_pair_rot_y_+1 | 45/45 | 2727/2727 | window state | 0.83 | 0.90 |
| w_766d8184 | native59_card_perturbed | 45/45 | 2727/2727 | hosted | 1.60 | 3.45 |
| w_766d8184 | native59_random_with_key_ends | 45/45 | 2727/2727 | hosted | 1.60 | 3.36 |
| w_766d8184 | remove_cf_J_ChinLow | 45/45 | 2727/2727 | hosted | 1.09 | 1.87 |
| front | chin_q0_encoded | 42/42 | 2712/2712 | window state | 0.47 | 0.80 |
| front | chin_q_random_0 | 42/42 | 2712/2712 | window state | 0.59 | 0.86 |
| front | chin_q_random_1 | 42/42 | 2712/2712 | window state | 0.70 | 0.88 |
| front | chin_q_random_2 | 42/42 | 2712/2712 | window state | 0.67 | 0.89 |
| front | chin_q_random_3 | 42/42 | 2712/2712 | window state | 0.68 | 0.87 |
| front | chin_tip_scale_x_+1 | 42/42 | 2712/2712 | window state | 0.71 | 0.81 |
| front | chin_low_pos_z_-1 | 42/42 | 2712/2712 | window state | 0.59 | 0.77 |
| front | cheeklow_pair_rot_y_+1 | 42/42 | 2712/2712 | window state | 0.71 | 0.79 |
| front | native59_card_perturbed | 42/42 | 2712/2712 | hosted | 1.26 | 2.74 |
| front | native59_random_with_key_ends | 42/42 | 2712/2712 | hosted | 1.36 | 2.80 |
| front | remove_cf_J_ChinLow | 42/42 | 2712/2712 | hosted | 1.12 | 1.72 |

## 4 Parts that are not differentiable, and how they are handled

* **EyeLook** (`EyeLookController.LateUpdate`, installed IL on the game's Mono JIT; outputs
  `cf_J_look_L/R`).  Source: `EyeLookCalc.EyeUpdateCalc` reads the world TRS of `cf_J_FaceUp_tz`
  and `cf_J_EyePos_rz_L/R` (head_skinning §9.2).  Rule (`session.eye_inputs_untouched`): Apply
  writes only its own bone, so when no patched / removed bone lies in that input closure (the
  three transforms and all their ancestors) and native59 equals the window's, the look rotations
  are the exact chain's (checked bitwise on the evaluated frames).  Otherwise the hosted
  controller runs over N-4..N on the torch forward's detached binary32 locals (this reproduced
  the exact chain bit for bit for every window and both native59 patches).  Either way the
  rotations are constants of the step: their derivative is not propagated.
  *Chin parameters:* the five chin bones are not in the closure -> EyeLook is independent of
  them (source rule, confirmed bitwise).  *native59:* face sliders move `cf_J_FaceUp_tz` /
  `cf_J_EyePos_rz` -> the look rotations change; their derivative is the explicit gap.  Only the
  eyeball surfaces `o_eyebase_L/R` are skinned to bones under `cf_J_look_*`; o_head, o_namida,
  o_eyelashes and o_eyeshadow are not.
* **FBS blend shapes** (blink, eyes, eyebrows, mouth): FaceBlendShape reads timers, Random and
  the expression state, never geometry (§9.3) -> the probe frame weights are exact for every
  candidate; ApplyBlendShapes runs through the exact port once per (renderer, frame).
* **Mouth width** (UpdateBlendShapeVoice): no geometry input (§9.4); the ABMX LateUpdate value of
  `cf_J_MouthMove` is part of the differentiable chain.
* **Neck look / animation / PoseFreeze**: no face-bone input (§9.5); the probe values and the
  bridge pre_cull locals of the Animator write set are constants.
* **Body shape writes** (`cf_N_height`, `cf_J_Head_s`): depend only on the card's body values;
  exact port, constants.

## 5 Gradients

Autograd runs through the same operations.  `R` is a cast (derivative taken as 1).  Where the
derivative is zero or piecewise, by game semantics (forward never altered):

* **ABMX IsEmpty / CanApply per channel**: a channel equal to its identity exactly (scale 1,
  length 1, position 0, rotation 0 -- all three components) is not applied by the game, so
  autograd gives 0 for it.  The chin controls are all live at q = 0 (card scales != 1, rotation x
  != 0, positions partly != 0); length is 1 and has zero derivative (not a control).  Opt-in
  `Arith(..., identity_gradient='one_sided')`: the forward keeps the taken branch's bits, the
  backward uses the applying branch's derivative when both branch values agree up to binary32
  rounding (then both one-sided neighbourhoods apply the channel); otherwise (e.g. length exactly
  1 with a position offset: the length branch renormalises the current position) it stays 0 and
  is labelled.  Unit test: a new modifier at identity -> 0 in branch mode, = central FD in
  one-sided mode.
* **ABMX length** `< 0.1` historical-direction fallback and Vector3 approximate-zero tests:
  piecewise.  **SetLocalR** fallback (sum <= 1e-30): constant branch.
* **GetInfo**: `rate == 0` / `1` take the first / last key (zero derivative at exactly those
  values); the key segment `FloorToInt((n-1) rate)` is chosen from the forward value, so the
  derivative is one-sided at a key position (the upper segment); `Clamp01(t)`.  The derivative
  inside a segment is the keyframe lerp's.  SliderUnlocker out-of-range extrapolation raises
  (also unported in the exact chain).
* **Sign-of-scale** products in the hierarchy walk (`S`): zero derivative.
* **CRT sinf / cosf**: forward executed from UnityPlayer.dll, backward cos / -sin.

| parameter group | max relative error (autograd vs central FD, float64) |
|---|---|
| cf_J_ChinTip_s.scale | 1.1e-07 |
| cf_J_ChinTip_s.position | 1.4e-07 |
| cf_J_ChinTip_s.rotation | 9.5e-08 |
| cf_J_ChinLow.scale | 7.9e-08 |
| cf_J_ChinLow.position | 3.2e-08 |
| cf_J_ChinLow.rotation | 4.2e-07 |
| cf_J_Chin_rs.scale | 9.6e-08 |
| cf_J_Chin_rs.position | 1.4e-08 |
| cf_J_Chin_rs.rotation | 6.4e-08 |
| cf_J_CheekLow_L.scale | 1.9e-07 |
| cf_J_CheekLow_L.position | 1.4e-06 |
| cf_J_CheekLow_L.rotation | 6.3e-08 |
| cf_J_CheekLow_R.scale | 3.5e-07 |
| cf_J_CheekLow_R.position | 2.3e-07 |
| cf_J_CheekLow_R.rotation | 2.5e-08 |
| native59 | 9.2e-06 |

All 104 gradients are nonzero.  native59 indices 54-58 sit at 0.5, a key position of their 25-key tables: the central difference straddles the segment boundary; the one-sided difference on the selected (upper) segment agrees to 1.5e-07.  Exact-mode vs float64-mode autograd: max relative difference 1.0e-05 (binary32 rounding of intermediates).  Functional: a random linear functional of o_head + o_namida world positions and normals (chin-fit front window, N-2 stream, frozen root of bridge render 0).

## 6 Chin objective PROXY (`chin_proxy.py`) -- not the chin objective

The authoritative score remains `src/render/chin_fit/objective.py` on exact renders.  The proxy
reuses the chin fit's frozen frame and references read-only (`domain.json`, `baseline_score.json`,
`encoding.json` of a chin fit directory) and replaces the image measurements:

* front / oblique: each MediaPipe chain id is bound once to the o_head surface point under its
  detected baseline position (camera ray on the view's exact baseline stream; 7 front points
  that lie just off the head silhouette bind to the nearest silhouette edge point, 0.9-6.2 px);
  the proxy landmark is that point's projection.  Fixed bindings do not slide along the contour
  as a detector does.
* right / left: outer silhouette of o_head + body (rasterised union of projected triangles,
  boundary mesh edges, detached selection), signed distance along each fixed art station normal
  to the nearest crossing (the `profile_offsets` rule), differentiable through the crossing edge.
* baseline anchoring: per landmark / station constants make the proxy equal the exact
  measurement at the encoded baseline (q = 0).
* per view: the view window's exact baseline slot-0 stream plus the torch change on the
  geometry session's window (root-relative streams of the four windows differ by <= 1.4e-6 at
  baseline), the view's frozen root of render 0 (bit-equal to its record), its record camera
  (equal to `camera.generated_camera` for that yaw).

API: `chin_proxy.loss_and_grad(proxy, patch)` -> (cost, {patch path: gradient}, per-view RMS);
`patch_from_q_torch(q, rows)` for the 36 chin controls.

Evaluated patches of the stopped chin fit `fit_v2` (exact costs from its `evaluations.jsonl`; RMS in art px; `tools/diffgeo/proxy_demo.py`):

| evaluation | exact cost | proxy cost | front exact / proxy | oblique | right | left |
|---|---|---|---|---|---|---|
| 0 | 0.008522 | 0.008522 | 10.01 / 10.01 | 15.48 / 15.48 | 8.24 / 8.24 | 8.07 / 8.07 |
| 79 | 0.007324 | 0.008035 | 9.41 / 9.51 | 14.21 / 15.32 | 8.21 / 8.19 | 7.09 / 7.10 |
| 78 | 0.007407 | 0.008037 | 9.41 / 9.51 | 14.36 / 15.33 | 8.21 / 8.19 | 7.07 / 7.10 |
| 57 | 0.007441 | 0.008060 | 9.36 / 9.52 | 14.42 / 15.34 | 8.23 / 8.21 | 7.10 / 7.14 |
| 50 | 0.007460 | 0.008049 | 9.38 / 9.53 | 14.44 / 15.32 | 8.22 / 8.20 | 7.10 / 7.14 |
| 73 | 0.007510 | 0.008037 | 9.43 / 9.52 | 14.52 / 15.32 | 8.21 / 8.20 | 7.07 / 7.10 |
| 28 | 0.008087 | 0.008581 | 9.71 / 9.99 | 14.94 / 15.59 | 8.26 / 8.28 | 8.04 / 8.03 |
| 21 | 0.008413 | 0.008236 | 9.69 / 9.81 | 15.68 / 15.38 | 8.10 / 8.07 | 7.64 / 7.58 |
| 34 | 0.008499 | 0.008513 | 9.73 / 10.01 | 15.58 / 15.47 | 8.24 / 8.24 | 8.07 / 8.07 |
| 19 | 0.008546 | 0.008546 | 10.03 / 10.00 | 15.51 / 15.52 | 8.25 / 8.25 | 8.07 / 8.06 |
| 24 | 0.009218 | 0.009162 | 10.29 / 10.43 | 16.00 / 15.84 | 8.48 / 8.49 | 8.90 / 8.90 |

Rank correlation of the costs 0.87 (Pearson 0.92).  Profiles follow the exact silhouettes closely; the front chain within ~0.1-0.3 px RMS; the oblique proxy under-reports the improvement (fixed bindings do not slide along the jaw contour).  Directional derivative at q = 0 along -grad: autograd -0.00546, central FD -0.00547 (h = 1e-3); larger steps change the crossing silhouette edges (piecewise).

## 7 Timing (CPU, warm, median of 5; `tools/diffgeo/timing.py`)

| step | median s |
|---|---|
| exact chain as `_streams` runs it per candidate (5 frames forward, hosted EyeLook, emulated skinning of all head surfaces x 3 frames) | 0.75 |
| exact chain, same scope as the torch step (forward N-2, N-1; emulated skin of o_head N-2) | 0.19 |
| torch exact mode, forward (N-2, N-1, o_head) | 0.19 |
| torch exact mode, forward + backward | 0.37 |
| torch float64 mode, forward + backward | 0.20 |
| chin proxy cost + gradient, four views | 0.95 |
| exact renderer, one fit evaluation (GPU executor, 4 views, no gradient; fit_v2 log) | 15.72 |

A finite-difference gradient over the 36 chin controls on the exact renderer needs 37-73 such evaluations; one torch forward + backward gives all of them (and the 59 native sliders) at once.

## 8 Gaps (explicit)

1. EyeLook derivative (hosted Mono; non-differentiable): zero for parameters that reach its
   inputs (native59 face sliders, bones in the FaceUp_tz / EyePos closure); affects the eyeball
   surfaces only.
2. SliderUnlocker out-of-range extrapolation (`SliderMath`) is not ported (as in the exact chain).
3. GPU vertex-shader projection arithmetic is not part of the bit-exact claim (projection in
   float64).
4. Candidates that move a head surface root (e.g. native sliders writing `cf_J_FaceBase` /
   `cf_J_FaceUp_*`) are computed here (frozen root and O2W included), but the exact renderer
   rejects them (`CandidateUnsupported`: template bounds / culling / matrices), so they cannot be
   verified on exact renders yet.
5. The proxy's stated limitations (section 6); the feature guard is not part of it.
6. Expression changes (FBS weights other than the probe's) are outside this chain.

## 9 Reproduce

```
.venv/Scripts/python.exe tools/diffgeo/derive_native.py --check
.venv/Scripts/python.exe -m unittest tools.diffgeo.test_diffgeo
.venv/Scripts/python.exe tools/diffgeo/forward_acceptance.py --all --out E:/HS2_offline_renderer_data/diffgeo_20261025/forward_acceptance_v3
.venv/Scripts/python.exe tools/diffgeo/gradient_check.py --window front --out E:/HS2_offline_renderer_data/diffgeo_20261025/gradient_check_v1
.venv/Scripts/python.exe tools/diffgeo/proxy_demo.py --n 10 --out E:/HS2_offline_renderer_data/diffgeo_20261025/proxy_demo_v1
.venv/Scripts/python.exe tools/diffgeo/timing.py --out E:/HS2_offline_renderer_data/diffgeo_20261025/timing_v1
```

torch 2.11.0+cu130 is importable from `.venv` (system site packages); nothing was installed.
