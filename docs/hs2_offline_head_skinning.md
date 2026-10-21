# HS2 offline renderer — head skinning data path (2026-10-20)

Scope: which bytes of the real captured frame must change for a candidate head,
and a source-derived generator of those bytes. Evidence is the existing
unaccepted retry capture
`HS2Mod/artifacts/chenger_reference_fit_20261012/native_present_session_v1/unaccepted_retry_frame17675.rdc`
(SHA `eb5743bc…`), its official zip.xml export and three official `drawinputs`
replays. No game, new capture, injection or card change was used. The capture
was taken with card `Chenger_DINOv2_v1_chin_edge_working_portrait_20261012.png`
(`269b6553…`, the unaccepted working candidate, not the DINOv2 v1 baseline).

Compact receipt: [`tools/offline_renderer/head_skinning_capture_20261020.json`](../tools/offline_renderer/head_skinning_capture_20261020.json).
Full evidence: ignored `outputs/offline_renderer_skinning_20261020/analysis_v1/evidence.json`.

## 1 Skinning is CPU skinning; there is no GPU skinning or GPU blendshape

The hypothesis "6 compute shaders / 187 Dispatches = GPU skinning" is false.
The six compute shaders carry debug names and are all PostProcessing v2:
`MultiScaleVORender_interleaved` (68 dispatches), `MultiScaleVODownsample1/2`,
`MultiScaleVOUpSample`, `MultiScaleVOUpSample_blendout` (51) and
`Lut3DBaker.KGenLut3D_AcesTonemap` (17). Their disassembly (`fxc /dumpbin`,
reflection stripped) declares only typed `texture2d/2darray/3d` SRVs and UAVs;
no buffer SRV/UAV exists in the whole capture (UAV dimensions 4/5/8 only), no
stream-out, and both `GSSetShader` calls bind null.

Unity 2018.4.11f1 therefore skins on the CPU. Every active SkinnedMeshRenderer
owns a `D3D11_USAGE_DYNAMIC` vertex buffer (debug name `Buffer-1-<bytes>`) bound
to IA slot 0 with stride 40:

| offset | semantic | format |
|---:|---|---|
| 0 | POSITION | R32G32B32_FLOAT |
| 12 | NORMAL | R32G32B32_FLOAT (unnormalized: linear skin incl. scale) |
| 24 | TANGENT | R32G32B32A32_FLOAT (w = source handedness ±1) |

Slots 1/2 are DEFAULT buffers named after the mesh: slot 1 holds COLOR/UV0..UV2
(o_head/o_namida stride 28, others stride 8); slot 2 (stride 32) holds the four
float bone weights and four int bone indices and is **not referenced by any
input layout** — the GPU never skins. The IB is `o_xxx(Clone)` (R16_UINT).
All eight drawn surfaces' IBs and UV streams equal the installed source arrays
bit-exactly, and slot 2 equals the source weights/indices.

**Blend shapes are applied on the CPU before skinning.** There is no blendshape
stream or buffer; the uploaded positions contain the active `*.e00_defo` deltas:
omitting them moves o_head by up to 4.47e-3, o_namida 8.04e-3 and o_eyelashes
9.73e-4, while including them reproduces the VB to the levels in section 5.

All skinned renderers are mapped together (`Map WRITE_DISCARD` at chunks
37435–37453) and unmapped at 37541–37555: one batched CPU skinning upload per
player-loop frame.

## 2 Coordinate space

Skinned vertices are in the renderer's **actual root bone frame without scale**:
`v_vb = inverse(TRS(root.position, root.rotation, 1)) · Σ w_k (bone_k.localToWorld · bindpose_k) · (v + Δblend)`.
Scale stays inside the skinning matrices. The per-draw cbuffer
`unity_ObjectToWorld` is exactly that scale-free root TRS (orthonormal to 4.5e-8).
For all eight head surfaces the root bone is `cf_J_FaceUp_ty`, which carries an
ABMX modifier; ABMX/native changes on the root or any ancestor change both the
VB contents and the per-draw matrices. The rotation must be Unity's
`Transform.rotation` (product of local rotations); with ABMX non-uniform scale
the normalized world columns differ (2.5e-8 here, larger for candidates), so
`CapturedHeadForward.evaluate` now also returns `bone_local`.

Cbuffer layout from disassembly: UnityPerDraw (`ConstantBuffer-179-176`, one
use `-179-160`) registers 0–3 = `unity_ObjectToWorld` columns, 4–7 =
`unity_WorldToObject` (used by `dp3 NORMAL, cb[4..6]`), 8 `unity_LODFade`,
9 `unity_WorldTransformParams`. The motion-vector VS
(`Hidden/Internal-MotionVectors`) reads a 240-byte per-draw CB
(`ConstantBuffer-186-240`): registers 2–5 `_NonJitteredVP`, 6–9 `_PreviousVP`,
10–13 `_PreviousM`, 14.x `_HasLastPositionData`, 14.z depth bias; TEXCOORD4 is
the previous-position VB in slot 1.

Renderer world bounds equal `localBounds` transformed by the **full** root bone
localToWorld (3e-7 in the live export), so root changes move culling bounds.
Every light-probe SH write (`ConstantBuffer-208-768`) read by a head draw is the
same write read by non-head draws, so it is per pass, not per renderer position.

## 3 Upload points for the candidate head (exact chunk list in the receipt)

Ten source surfaces: eight are drawn; the two body `o_tang`
(`p_cf_body_00/.../N_tang[1]`, `p_cf_body_silhouette/.../N_tang[2]`) are
`active_in_hierarchy=false` and have no GPU resources in this frame.

| surface | skinned VB | capture-start contents (CreateBuffer chunk, blob) | previous VB (MV) | in-frame skin Unmap |
|---|---|---|---|---|
| o_eyebase_L | 29875 | 1329 / 1648 | 29862 (1307) | 37551 |
| o_eyebase_R | 29870 | 1319 / 1643 | 29863 (1309) | 37552 |
| o_eyelashes | 29869 | 1317 / 1642 | — (not in MV pass) | 37541 |
| o_eyeshadow | 29873 | 1325 / 1646 | 29864 (1311) | 37553 |
| o_head | 29868 | 1315 / 1641 | 29861 (1305) | 37549 |
| o_namida | 29874 | 1327 / 1647 | — | 37547 |
| o_tang (head) | 29872 | 1323 / 1645 | 29860 (1303) | 37548 |
| o_tooth | 29871 | 1321 / 1644 | 29865 (1313) | 37554 |

RenderDoc 1.46 stores a buffer's capture-start bytes in its CreateBuffer
`InitialData` (`Serialise_InitialState` copies the staging buffer into
`record->GetDataPtr()`); there is no separate Initial Contents chunk for
buffers. The official replay confirms it: `drawinputs` at event 717 (chunk 3060)
and event 1615 (MV draw) return o_head VB bytes equal to blob 1641 and the
previous VB equal to blob 1636; event 35341 (after 37549) returns blob 5545.

Of 1003 head draws, 944 precede the target readback (chunk 37429, event 35086).
Those use the capture-start VB contents; the in-frame Unmap only feeds the later
window-camera render. Per-draw uploads consumed by head draws: 359 UnityPerDraw
writes and 102 motion-vector writes, all `Map WRITE_DISCARD`. Unity skips
redundant per-draw uploads, so one write is shared by up to five head surfaces
(same root); **no head write is consumed by a non-head draw**, so all are
in-place patchable.

### Two recorded root poses

The frame contains two root poses (exact matrices in the receipt):

- `pose0` — the bridge's frozen capture renders before the target. O2W of 337
  writes before 37549; the capture-start current VB was skinned in this pose.
- `pose1` — the ordinary idle pose. `_PreviousM` of the bridge renders, O2W of
  the window render after 37549, the previous-position VB contents and the
  in-frame skin. The capture-start previous VB is **bit-identical** to the
  in-frame Unmap for all six MV surfaces.

`unity_WorldToObject` is the inverse of the current O2W in 323 writes, identity
in 34 MV-pass writes, and in the first write after each pose switch (chunks 2385
and 37573) the inverse of the *other* pose: the engine updates O2W but keeps the
previous W2O for that draw. This rule is recorded behaviour, not recovered from
UnityPlayer source; the generator reproduces each write's recorded class.

## 4 Generator (`src/render/gpu_skin_inputs.py`)

- `CaptureIndex` / `simulate`: exact chunk/blob/ordinal index and buffer write
  provenance per draw; rejects deferred contexts, buffer UAVs, stream-out,
  D3D11.1 CB ranges and unsupported buffer writers instead of skipping them.
- `bind_surfaces`: binds each source renderer by exact IB/UV/weight content;
  duplicate meshes resolve only by recorded renderer activity.
- `object_matrix_uses`: finds object matrices in each head VS disassembly.
- `upload_points` + `label_poses`: role, pose and W2O class for every write.
- `generate_patches(index, writes, points, {pose: {surface: SurfaceState}}, out)`
  writes replacement files and a `mutate_capture`-compatible plan
  (`chunk_index`, `buffer_ordinal`, `expected_buffer_index`, `byte_offset`,
  `expected_range_sha256`, `replacement_sha256`, `resource_id`, `semantic_role`).
  It patches the 40-byte vertices, O2W (64 B at 0), W2O (64 B at 64) and
  `_PreviousM` (64 B at 160); other per-draw fields stay original. A missing
  pose/surface state is a listed gap; another pose's state is never substituted.
- `surface_state_from_forward(CapturedHeadForward.evaluate(...), i, geometry)`
  requires loaded API normal/tangent frames and `bone_local`.

CPU tests: `python -m unittest tools.offline_renderer.test_gpu_skin_inputs` (8
pass), including a synthetic official-schema capture through
`mutate_capture` and back.

## 5 Baseline correspondence (static, no new samples)

No same-frame source state exists for this capture: its paired geometry export
failed (`Marked original capture not saved`). The nearest state is the live
export `native_resume_v1/live_full_frames_materials.json` (same card state hash,
other process, frame 13378). VB bytes are root-relative, so the different
whole-head pose does not enter; matrices do.

- **No upload matches bit-exactly.** Per-draw matrices: export root differs from
  `pose0`/`pose1` by up to 0.114/0.056 (head animation/pose of another frame).
- o_tang, o_tooth, o_eyeshadow, o_eyelashes: all vertices within 3.6e-6
  (normals/tangents ≤3.6e-7). Remaining difference is float32 arithmetic: the
  generator works in float64 then rounds; Unity's float32 SIMD skinning order is
  not recovered. Even inside the capture, pose0 vs pose1 skins of a static mesh
  differ by ~1.5e-6 purely through the root frame.
- o_eyebase_L/R: pose0 positions within 4.9e-6 (normals 3.0e-5); pose1 differs by 1.9e-3 (normals 1.2e-2)
  through `cf_J_eye_rs_L/R`. Eye look pattern 1 (TARGET) recomputes eye bones
  from camera and head pose each LateUpdate.
- o_head: 548 lip/mouth vertices differ up to 3.9e-4 in both poses. Diagnostic
  per-bone attribution, not used for generation, gives `cf_J_Mouth_L/R` ±1.97e-4
  x-translation and `cf_J_Mouthup/MouthLow` ~0.5% x-linear change. That matches
  a different `ScaleX` of the mouth-width target written by
  `UpdateBlendShapeVoice` (`mouthAdjustWidth=true`) in that frame. o_namida's 4
  mouth-corner vertices (1.6e-5) share this mechanism.
- Forward-model integration: `CapturedHeadForward` (source3d_v1 context,
  loaded frames transferred by sharedMesh hash) feeds the generator end to end.
  That context is a different ABMX state (`a04f…` vs capture `46de…`), so chin
  bones differ by 1.2e-2. This proves only the code path, not a baseline.

## 6 Gaps (explicit; not waived)

1. **Same-frame source state.** Bit-level baseline acceptance needs bone,
   gaze-driver and mouth-width state paired with the RDC frame. That depends on
   the unresolved RenderDoc Map/retry lifecycle; repeating the failing Present
   path was not attempted.
2. **Unity float32 skinning arithmetic** (matrix palette build and blend order
   in UnityPlayer.dll) is not recovered; generated bytes are float64-rounded.
   Unity's W2O is not a numpy float32/float64 inverse bit-for-bit (0 of 325).
3. **Previous-frame semantics.** The MV pass needs the candidate in the
   previous ordinary pose (pose1 skin, `_PreviousM`). Whether MV output reaches
   the target through CTAA/temporal history belongs to the frame graph.
4. **Stale W2O rule** is reproduced by recorded class only.
5. **Draw-list invariance.** Candidate root/ancestor changes move world bounds
   (full root matrix · localBounds). Camera/shadow culling must be shown
   unchanged before replaying the recorded draw list.
6. Material per-draw `$Globals`, texture and shadow-map dependencies are outside
   this skinning path.

## 7 Same-frame paired capture and Unity-exact skinning (2026-10-20, done)

Session: `HS2Mod/artifacts/chenger_reference_fit_20261012/paired_capture_session_20261020/`.
The receipt is `paired_capture_receipt.json`; its compact copy is
`tools/offline_renderer/paired_skin_capture_20261020.json`.

The DINOv2 v1 baseline card (`74d2625f…`) was loaded read-only. Snapshot, raw
ABMX, expression and renderstyle were recorded and unchanged at the end; no
card was saved. Two Present-boundary captures were taken with the probe.
Each time the RenderDoc log shows the predicted "Uncapped Map/Unmap" failure,
then the automatic retry capturing the next frame. The in-frame probe marker
names the exported frame: 12741 and 26060 (`Time.frameCount`). Window 2 also
contains the full render-scene state for every frame: 8 cameras, 5 lights,
82 renderers, 63 materials, 217 textures and 18 effect components.

**Result: every head upload is bit-exact in both captures.**

| Upload | Source | Result |
|---|---|---|
| in-frame skinned VB (8 surfaces) | frame F export | 8/8 bit-exact |
| previous-position VB (6 MV surfaces) | frame F-1 export | 6/6 bit-exact |
| `unity_ObjectToWorld` (16 writes) | float32 TRS(root.position, root.rotation, 1) | 16/16 |
| `unity_WorldToObject` (14 writes) | `InvertMatrix4x4_General3D(O2W)` | 14/14 (2 MV-pass writes identity) |
| `_PreviousM` (6 writes) | O2W of frame F-1 | 6/6 |

The generator (`generate_patches` with Unity-exact states) reproduces all 50
captured upload ranges of capture 2 with zero changed bytes and no gaps.

**How exactness was obtained (source, not fitting).** The matched UnityPlayer
PDB (GUID `BFA91A8C…`, age 1) names the routines. `src/render/unity_skin_emu.py`
runs their original instructions under Unicorn:

- `CalculateGlobalPositionAndRotation`: the root's world position/rotation
  (130/130 sampled transforms bit-equal to `Transform.position/rotation`).
- `GetWorldToLocalMatrixNoScale` (`CalculateInverseGlobalMatrixNoScale`): the
  root inverse.
- `SkinnedMeshRendererManager::CalculateSkinningMatrices`, prepared-renderer
  path: pose = base·TRS(top)·…·TRS(bone) built from **local** TRS by SIMD,
  then times the bindpose.
- `SkinVerticesGenericCPU<4,1,1>`: column-wise matrix blend
  `((w0M0+w1M1)+w2M2)+w3M3`, then `((x·c0+y·c1)+z·c2)+c3`; tangent.w copied.
- `InvertMatrix4x4_General3D`, called from `GfxDeviceD3D11Base::BeforeDrawCall`,
  for `unity_WorldToObject`. Its lazy dirty flag explains the stale W2O seen
  in frame 17675.

The fallback path (`Transform.localToWorldMatrix` · bindpose) and all four
declared float32 orders reach only about 25–28% word equality. The difference
is that Unity rebuilds the bone matrices from local TRS through the prepared
hierarchy chain. `skin_from_locals` needs only local TRS (for example
`CapturedHeadForward.evaluate(...)['bone_local']`) and reproduced capture 2
bit-exactly from locals alone.

**Remaining gaps**
- This card's only active blendshapes are `*.e00_defo=100` (k=1.0), so the
  float32 blend order (shape-index sequential `v + d·(w/frameWeight)`) is
  verified only for that case. `ApplyBlendShapes`/`ApplyBlendShapesToVertices`
  must be emulated or disassembled before expression candidates.
- Bone locals from `CapturedHeadForward` have not yet been compared with the
  probe's same-frame float32 locals. Unity-exact candidate bytes require the
  ABMX/native replay to produce Unity's exact float32 local TRS.
- Typed material properties exist only for the 37 shaders in the manifest;
  other materials export keywords, textures, queue and tags. Raw texels were
  not exported: `HS2_RenderAudit`'s config points at an old output directory,
  so it was not invoked.
- Why the 2 MV-pass W2O writes are identity (signed zeros) is recorded, not
  traced. The current VB's capture-start contents are never consumed by a draw
  and were not matched.
- Bounds/culling invariance under candidates, and CTAA temporal history, as
  before.

CPU tests: `python -m unittest tools.offline_renderer.test_paired_skin
tools.offline_renderer.test_gpu_skin_inputs` (14 pass; the emulator tests run
the installed UnityPlayer.dll).

## 8 Source chain to upload bytes, blendshapes (2026-10-20, offline)

Receipt: `tools/offline_renderer/forward_chain_acceptance_20261020.json`.
Bulky outputs: `E:/HS2_offline_renderer_data/skinning_20261020/`.

**Result: bit-exact on both paired captures** (frames 26060 and 12741, plus
their previous frames). The chain was card → head prefab locals → native59 face
→ body → ABMX `CollectBaselineCo` → ABMX `Apply` (steady state) → bone local
TRS → `skin_from_locals` → upload bytes.

| Stage | Result |
|---|---|
| 98 skin-chain bones' local pos/rot/scale | 0 fields differ (F and F-1) |
| ABMX baselines rebuilt from prefab + native59 | 30/30 head modifiers |
| skinned VB (current / previous) | 8/8, 6/6 |
| `unity_ObjectToWorld` + `unity_WorldToObject` | 16/16 |
| `_PreviousM` | 6/6 |

Mechanisms recovered and fixed. Each is backed by an installed binary or
decompiled IL, with the paired frame as acceptance; none is fitted.

- **Transform rotation setter**: `SetLocalR` normalizes every quaternion it
  stores (per-lane horizontal sums, sqrtps, divps, 1e-30 fallback). The port is
  `src/render/unity_transform_math.normalize_quaternion`, equal to the emulated
  original. It is applied in ABMX Apply/Reset and in the native setters.
- **`Quaternion.Euler`**: native `EulerToQuaternion` with UnityPlayer's own
  `sinf`/`cosf`, run under emulation. The numpy sin/cos version is off by 1 ulp
  in 3-axis cases.
- **Mono float semantics**: C# float32 expressions are evaluated in double.
  Register locals stay double; values are rounded at memory stores, float32
  arguments and returns. This covers `Quaternion.operator*`, `GetInfo`'s `num`
  local, `LerpAngle` and `Vector3.SqrMagnitude`.
- **Native face shape**: `src/hs2_native_shape.py` ports `AnimationKeyInfo.GetInfo`,
  `ShapeInfoBase.ChangeValue`, and the `ShapeHeadInfoFemale.Update` program.
  The program is generated from the decompiled source into
  `src/render/shape_head_female_program.py` (113 assignments). This replaces the
  former world-FK decomposition, which flipped quaternion signs and lost bits.
- **Native body**: `ShapeBodyInfoFemale` writes reaching the head chains are
  only `cf_N_height` and `cf_J_Head_s`, both ported exactly.
- **ABMX baselines**: the card-load path is `BoneController.CollectBaselineCo`.
  `LateUpdateForce` applies the native update, then a fresh `CollectBaseline`
  runs with no Reset. `CapturedHeadForward.evaluate(baseline_mode='card_load')`
  reproduces it. `Apply` reaches a stationary state from the 2nd call on (results
  identical for 2, 3 and 8 calls).
  - The slider-change path (`UpdateBaseline`: Reset → `UpdateShapeFace`/`Body` →
    CollectBaseline) is still the `native=` route. It applies `Reset`'s
    length-history renormalization, which can move unwritten channels by about
    1e-9. That is real engine behaviour after a slider change, not reproduced by
    this frame.
- **Blendshapes**: `src/render/unity_blendshape.py` ports `ApplyBlendShapes`,
  `CalculateBlendShapeWeights` and `ApplyBlendShapeTmpl`, and runs the original
  under Unicorn (dynamic_array reserve/grow hooked).
  - Weights are clamped to [0, 1] because `PlayerSettings.legacyClampBlendShapeWeights`
    is True.
  - Multi-frame shapes interpolate `(j, 1-t)` and `(j+1, t)`; accumulation is
    `dst += F(w·d)` in frame order.
  - Port vs native: 108/108 synthetic cases (2298 frame weights), plus the
    captured `e00_defo=100` case.

Explained items:
- The motion-vector-pass `unity_WorldToObject` equals
  `InvertMatrix4x4_General3D(identity)`. `GfxDeviceD3D11Base::BeforeDrawCall`
  recomputes the inverse only when the shader needs it (flag +0x56e0), so the
  slot keeps a stale inverse.
- The current skinned VB's capture-start bytes equal frame F-2's skin (8/8):
  Unity alternates two buffers per renderer.

Files outside this module changed (all report-worthy):
- `tools/abmx_replay/model.py`: native Euler, Mono product, setter
  normalization, Mono `SqrMagnitude`.
- `src/hs2_offline_state.py`: exact `_face_update`/`_body_update`, Reset setter
  normalization, `baseline_mode='card_load'`, and the earlier `bone_local`
  output.

The frozen-provenance guards in `tools/abmx_stability`,
`tools/abmx_stable_lowering` and `tools/native_radial_target` pin `model.py` by
SHA. They now reject their older frozen artifacts, which were produced by the
previous inexact model. Re-freezing those is a decision for their owner.

Gaps:
- Per-frame drivers (animation pose, eye look, mouth width, FBS weights): see
  section 9.
- SliderUnlocker's out-of-range extrapolation (`SliderMath`) is not ported;
  it raises.
- Signed zeros in bridge JSON baselines are unrecoverable, because .NET
  Framework's "R" format drops -0.
- The sparse blendshape rows are rebuilt from API dense frames. This matters
  only for -0.0 source components.

## 9 Per-frame drivers (2026-10-20, offline)

Receipt: `tools/offline_renderer/frame_drivers_acceptance_20261020.json`.
Bulky outputs: `E:/HS2_offline_renderer_data/skinning_20261020/drivers/`.
Tests: `tools/offline_renderer/test_frame_drivers.py`.

### 9.1 How the drivers are executed: the game's own code

The drivers are managed code in the installed `IL.dll`. Their float results
depend on HS2's Mono JIT: double evaluation, rounding at stores, and register
allocation across basic blocks. They are therefore not transcribed.
`src/render/hs2_hosted_runtime.py` embeds the installed
`MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll` and loads the installed
`IL.dll` and `UnityEngine.CoreModule.dll`, so the game's IL is compiled by the
game's JIT.

- **Mono init**: identical to `InitializeMonoFromMain`:
  `mono_set_defaults(0, mono_parse_default_optimizations(NULL))` (UnityPlayer
  RVA 0x8e99d5).
- **Native calls** are UnityPlayer.dll's own code, loaded in-process:
  - The Quaternion, Vector3 and Random bindings are registered directly as the
    Mono internal calls.
  - Transform internal calls use a scene model. It runs UnityPlayer's
    `CalculateGlobalPositionAndRotation`, `TransformPoint`,
    `InverseTransformPosition` and `InverseTransformRotation` on a
    TransformHierarchy built from the TRS records, plus the validated `SetLocalR`
    normalization.
  - SSE operands are 16-byte aligned.
  - `GetScriptingRand()` returns the `Rand*` stored at RVA 0x1551860. The host
    points that slot at its own `RandState`.
- **Harmony patches in the installed plugin set**: none patches the driver
  arithmetic in the Maker.
  - Hooah prefixes `EyeLookController.LateUpdate` and
    `NeckLookControllerVer2.LateUpdate`. They act only for Studio
    `HijackLook`/`HijackNeck` gimmicks.
  - PovX prefixes `NeckLookControllerVer2.LateUpdate`, only with PoV enabled.
  - BetterHScenes postfixes `FaceBlendShape.OnLateUpdate`, only in an H scene.
  - No preloader patcher targets `IL.dll` or `CoreModule`.

The installed MonoScript execution orders (`globalgamemanagers.assets`) fix the
LateUpdate sequence:

| Order | Writer |
|---:|---|
| -16000 | UniRx `MainThreadDispatcher` → `FaceBlendShape.OnLateUpdate` |
| -45 | `Manager.Character` → `ChaControl.LateUpdateForce` |
| 0 | ABMX `BoneController`, `CameraControl_Ver2` |
| 9996–10001 | FinalIK |
| 10050 | `NeckLookControllerVer2` |
| 10800 | `EyeLookController` |
| 12400 | `CorrectLightAngle` |
| 13100 | `DynamicBone` |

`CharaUtils.Expression` (12000) exists only in Studio. In the Update phase,
`Manager.Character` (-45) runs `ChaControl.UpdateForce` →
`UpdateBlendShapeVoice`.

### 9.2 Eye look: ported and accepted bit-exact

The eye look runs on `p_cf_head_bone` (`oo_base.unity3d`).

- **Init**: `EyeLookCalc.Awake → Init` runs inside `CommonLib.LoadAsset`'s
  Instantiate, before parenting and before `InitShapeFace`. The reference state
  is therefore prefab-only, and the hosted Init gives:
  - `referenceLookDir = +Z` and `referenceUpDir = +Y`;
  - `origRotation = (0,-0,-0,1)`.
- **Inputs in the Maker**:
  - `ptnNo = 1` (TARGET, leapSpeed 10). This comes from
    `CustomControl.Initialize` and `CustomBase._eyelook = 0`.
  - `target = Camera.main.transform`.
- **Candidate dependence**: `EyeUpdateCalc` reads the world TRS of
  `rootNode = cf_J_FaceUp_tz` and of the eye parents `cf_J_EyePos_rz_L/R`. Both
  move with face sliders and ABMX, so the eye look depends on the candidate and
  must be recomputed.
- **Outputs**: `cf_J_look_L/R`.
- **Hidden state**: `angleH`, `angleV`, `dirUp` per eye.

**Acceptance** (`tools/offline_renderer/eye_look_acceptance.py`). For each probe
frame F, the installed `EyeLookController.LateUpdate` runs with:
- frame F's exported local TRS for all 907 character transforms;
- Time.deltaTime of F;
- Camera.main's position;
- the hidden state from the hosted run of F-1.

The hosted world TRS of the eye ancestors and `chaF_001` equal the exported
world TRS bit for bit on every frame. Probe v4 does not export the window-start
hidden state. It is seeded by decomposing the first frame's look rotation, and
every frame is rerun with two deliberately different seeds:
- the Init state;
- the reconstruction with dirUp tilted by 1e-3 rad.

A frame is **accepted** when all seeds give the same bits and they equal the
export:

| Window | Accepted (bit-exact, seed-invariant) | Seed-dependent |
|---|---|---|
| 04d1e76f (marker 26060) | 26057–26062, both eyes (12/12) | 26056 (exact with the reconstructed seed) |
| 625aa0f2 (marker 12741) | 12741–12750 both eyes; L 12739–12740 (22) | 12736–12740 R and 12736–12738 L |

- **No seed-invariant frame differs.** The one inexact seed-dependent frame
  (12738 R: 16 and 32768 ulp in x/z, about 1e-9 absolute) is exact with the tilted seed.
  This proves the residual is the unexported dirUp state, not the logic.
- **Seeds stop mattering** once dt·leapSpeed ≥ 1 (Lerp clamps) and Vector3.Slerp's
  dirUp converges.
- **Camera source**: the window-2 scene export has a single enabled,
  MainCamera-tagged camera. Its `cameraToWorldMatrix` translation is
  (1.9058167e-6, 14, 21.8000011), constant over the window.
  - Window 1 has no scene export. The same position reproduces all its
    seed-invariant frames bit-exactly, so the camera did not move between the
    windows.
  - This translation is not proven to be `Transform.position` bit for bit.
    `_WorldSpaceCameraPos` is not either: it is the translation of
    `InvertMatrix4x4_Full(worldToCamera)` (`Camera::ExtractCameraRenderingParams`).
    v5 must export it.
- **Candidate path**: run the hosted driver over the probe window from its start.
  Hold an output as exact only when it is seed-invariant under the same test,
  or seed from v5-exported state.

### 9.3 FBS (blink, eyes, eyebrows, mouth)

FBS uses `p_cf_head_02`'s FaceBlendShape (`fo_head_38.unity3d`).

- **Targets**:
  - EyebrowCtrl: `o_head`;
  - EyesCtrl: `o_head`, `o_eyelashes`, `o_namida`;
  - MouthCtrl: `o_head`, `o_tooth`, `o_tang`, `o_namida`.
- **Eye-look link**: `FaceBlendShape.EyeLookController` is serialized null and
  never assigned, so the eye-look open correction is off.
- **Write order**: each controller writes every Close/Open index of its PtnSet.
  The mouth (last) rewrites index 0 of `o_head` and `o_namida` as 100. This is
  why `head.e00_defo` stays 100 while `e01_close` blinks.
- **Candidate dependence**: none. FBS reads timers, Random and the expression
  state, never geometry.

**Composition check** (`fbs_composition_check.py`). The hosted `OnLateUpdate`
ran in the two steady blink states the source defines:
- open: mode 0, openRate 1;
- closed: mode 1 past blinkTime, openRate 0.

Every FBS-written weight of all 24 probe frames equals exactly one of the two
hosted compositions. Window 2's sequence is open, open, closed×3, open×3.

This verifies the controllers, mappings and overwrite order. It is **not** a
blink-timing acceptance. Which frame blinks depends on
`FBSBlinkControl.blinkTime/blinkMode/calcSpeed/count` and on
`UnityEngine.Random.state` (xorshift128, consumed by
`SetForceClose`/`SetFrequency` and the mouth's `TimeProgressCtrlRandom`). Probe
v4 exports none of these.

### 9.4 Mouth width

`UpdateBlendShapeVoice` (Update) sets
`trfMouthAdjustWidth = cf_J_MouthMove` (CmpBoneHead.targetEtc) to
`localScale.x = mouthCtrl.GetAdjustWidthScale()`.
- With no playing voice, voiceValue = 0. Then `AdjustWidthScale` keeps
  `adjustWidthScale = 1`, because `0.2 < openRate` is false. The hosted run
  returns exactly 1.0.
- It still consumes Random when its timer wraps.
- ABMX `LateUpdate` writes the final `cf_J_MouthMove` scale (section 8). The
  driver input is constant (1.0) and does not depend on the candidate.

### 9.5 Neck look and animation pose

- **Active neck mode is ptn 3 (ANIMATION)**. `CustomBase._necklook = 1` gives
  `ChangeLookNeckPtn(3)`.
  - In this branch `NeckUpdateCalc` sets
    `neckBone.localRotation = Slerp(fixAngleBackup, animatorLocal, curve(timer))`
    on `cf_J_Neck` and `cf_J_Head`. It reads no position or target.
  - The exported neck/head rotations are raw clip-like values, for example
    (-0.0163, 0.00046, 0.00166, 0.99987). FORWARD (≈identity) and TARGET
    (computed) cannot produce them.
  - Native `Quaternion.Slerp(x, b, 1)` returns b bit-exactly for all 48 exported
    neck/head rotations, both for x = 0 (default `fixAngleBackup` when lookType
    never changed) and for x = identity.
  - The neck look is not executed hosted: `AnimationCurve.Evaluate` needs a
    native curve object. `curve(1)` is a v5 export item.
  - TARGET mode would also depend only on body transforms
    (`transformAim`, `boneCalcAngle`).
- **Animation (Mecanim + FinalIK)** writes Hips/Spine/Neck/Head from clip, time
  and body effectors. No face bone is an input, so a head-shape candidate cannot
  change it.
  - Holding the probe's frame-F values is legitimate for a candidate rendered at
    frame F. This holds while the candidate does not modify `cf_J_Neck`,
    `cf_J_Head` or their ancestors.
  - The Mecanim evaluation itself (clip sampling, state machine, FinalIK
    solve) is not ported. It is needed only to produce poses at times the probe
    did not export, such as the renderstyle freeze at normalizedTime 0.

### 9.6 What the bridge's renderstyle capture freezes (source)

`/maker/renderstyle/capture` runs synchronously in BepInEx `ThreadingHelper`
Update (via `KoikatuAPI.SynchronizedInvoke`) and calls `Camera.Render`.

- `PoseFreeze.Apply` sets the animator speed to 0, then
  `Play(fullPathHash, 0, 0)` and `Animator.Update(0)`. This writes the
  normalizedTime-0 sample of the current state to the **transforms** at once.
- `ChangeEyesBlinkFlag(false)` sets `fixedFlags` and the open-rate fields. They
  act only at the next `FaceBlendShape.OnLateUpdate`, which does not run inside
  the call.
- No LateUpdate runs inside the call, so eye look, neck look, ABMX and FBS keep
  their previous-frame results.
- **The rendered vertices are not re-skinned.**
  - `SkinnedMeshRenderer::AddAsRenderNode` (RVA 0x77abe0) skins only when one of
    these holds: the dirty byte (renderer+0x2c4, set only by
    `RendererBecameVisible`) is set, no skin buffer exists, or
    `forceMatrixRecalculationPerRender` (renderer+0x28e) is set.
  - Otherwise it draws the buffer produced by the player-loop callback that
    `SkinnedMeshRenderer::InitializeClass` registers. That callback, 0x784330,
    tail-calls `SkinnedMeshRendererManager::UpdateAllSkinnedMeshes` in
    PostLateUpdate.
  - The renderstyle capture does not set `forceMatrixRecalculationPerRender`.
- **The capture image therefore shows the previous frame's natural state**:
  unfrozen animation time, eye look, blink and mouth.
- **Its BakeMesh outputs show the frozen bone pose**: `face_surfaces` and
  `paired_geometry` with actor transforms. The previous frame's weights apply to
  them as well.
- **No driver is source-proven frozen in the rendered pixels.** The candidate
  path may hold a frozen value only for BakeMesh/transform outputs, and there
  only for the animated bones (normalizedTime 0).
- Open item: whether the renderer's root-bone matrix (`TransformInfo`) is
  refreshed inside `Camera.Render`. `SkinnedMeshRenderer::UpdateTransformInfo`
  is called from the manager's per-frame update; its trigger inside
  `Camera.Render` is not traced.

### 9.7 v5 probe requirements (exact list)

Capture each item at the stated hook, with float32 bits and the Unity frameCount.

**Eye look**: prefix and postfix on `EyeLookCalc.EyeUpdateCalc`.
- EyeUpdateCalc arguments (target, ptnNo).
- Per `eyeObjs[i]`: `angleH`, `angleV`, `dirUp`, `referenceLookDir`,
  `referenceUpDir`, `origRotation`.
- EyeLookCalc: `nowPtnNo`, `initEnd`, `targetPos`, `fixAngle[2]`,
  `angleHRate[2]`, `angleVRate`, and static `isEnabled`.
- `Time.deltaTime`.
- `EyeLookController.ptnNo`.
- Identity and `position` of `target`.
- Count of `HooahComponents.HijackLook._updatePositions`.

**Camera**: Camera.main identity and `transform.position`, plus local TRS of its
ancestors (CharaCustom/CustomControl/CharaCamera) and of `chaF_001`'s ancestors.

**Neck look**: prefix and postfix on `NeckLookControllerVer2.LateUpdate`.
- `ptnNo`, `rate`, target identity/position.
- NeckLookCalcVer2: `isEnabled`, `skipCalc`, `calcLerp`, private `lookType`,
  `changeTypeTimer`, `nowPtnNo`, `backupPos`, `nowAngle`.
- `changeTypeLerpCurve.Evaluate(InverseLerp(0, changeTypeLeapTime, changeTypeTimer))`.
- Per `aBones[i]`: `fixAngle`, `fixAngleBackup`, `backupLocalRotaionByTarget`,
  `angleH`, `angleV`.
- The neck/head `localRotation` at entry, which is the animator output.
- `neckTypeStates[ptn].isLimitBreakBackup`.

**FBS**: prefix and postfix on `FaceBlendShape.OnLateUpdate`.
- FBSBlinkControl: `fixedFlags`, `BlinkFrequency`, `blinkMode`, `BaseSpeed`,
  `calcSpeed`, `blinkTime`, `count`, `openRate`; and whether BlinkCtrlEx is
  present.
- `voiceValue`.
- Each controller: `openRate`, `OpenMin`, `OpenMax`, `FixedRate`,
  `correctOpenMax`, `dictBackFace`, `dictNowFace`, and `blendTimeCtrl`
  (count, rate, progressTime).
- Mouth: `useAjustWidthScale`, `tpcRand` (count, rate, progressTime, minTime,
  maxTime), `sclNow/sclStart/sclEnd`, `adjustWidthScale`.
- `UnityEngine.Random.state`, `Time.time`, `Time.deltaTime`.

**Mouth width**: whether `asVoice` is non-null and playing, plus
`fileStatus.mouthAdjustWidth`. Export `cf_J_MouthMove.localScale` after
`UpdateForce` and after the ABMX LateUpdate.

**Animation**: snapshot at the `FaceBlendShape.OnLateUpdate` prefix. It is the
first LateUpdate, after the animation phase.
- Animated-bone local TRS.
- For each Animator:
  - controller identity, `speed`, `updateMode`, `cullingMode`, `applyRootMotion`;
  - avatar (generic or human);
  - layer-0 current and next state: hash, normalizedTime, length,
    speedMultiplier;
  - `IsInTransition`.
- FinalIK `FullBodyBipedIK.enabled`, solver weights, and the Maker `MotionIK`
  data.

**Order certificate**: a per-frame sequence counter at the hooks above and at
`ChaControl.LateUpdateForce` and ABMX `BoneController.LateUpdate`. It confirms
the MonoScript order in effect.
