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
