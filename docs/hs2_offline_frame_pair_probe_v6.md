# HS2 offline renderer — frame-pair probe v6 (2026-10-25)

Probe v6 adds the fields that the generators still listed as missing after the v5.0.1 session:

- [orchestrator §8.6 and §9.5](hs2_offline_frame_orchestrator.md);
- [head skinning §9.7](hs2_offline_head_skinning.md).

It also adds a clean quit request. The window plan, markers, verification and fail-closed rules of
[v5](hs2_offline_frame_pair_probe_v5.md) are unchanged. Nothing was launched or deployed to build
or test it.

## 1 Build identity

| item | value |
|---|---|
| build | `outputs/offline_renderer_probe_v6_20261025/build_v6/HS2_FramePairProbe.dll` (deterministic: an independent rebuild gave the same bytes) |
| SHA-256 | `42da95fdbb8c51862dc31e7452f15ee834c327ce9a45ddededf5b8844c2c7868` |
| plugin | `codex.hs2.framepairprobe` **6.0.0**. The log line, receipts and every export carry `HS2_FramePairProbe 6.0.0` (v5.0.1 still said 5.0.0). |
| new references | `BepInEx/core/0Harmony.dll` (HarmonyX 2.9.0), `UnityEngine.AnimationModule`, `UnityEngine.AudioModule` |
| manifest | unchanged: `shader_manifest_v5.tsv` `e403c5ad…` |
| session script | `tools/offline_renderer/frame_pair_probe/paired_capture_session.ps1` with fresh dirs `HS2Mod/…/paired_capture_session_v6_20261025` and `E:/HS2_offline_renderer_data/paired_session_v6_20261025` |
| superseded (moved, not deleted) | v5.0.1 script → `frame_pair_probe/deprecated/paired_capture_session_v5_1.ps1`; intermediate v6 builds → `outputs/offline_renderer_probe_v6_20261025/deprecated/` |

New file: `DriverHooks.cs`.

Changed files:

- `ProbeCore.cs`
- `NativeUnity.cs`
- `RenderRecorder.cs`
- `SceneState.cs`
- `FramePairProbe.cs`
- `HS2_FramePairProbe.csproj`
- `verify_probe_v5_session.py`
- `paired_capture_session.ps1`
- `tests/ProbeCoreTests/Tests.cs`
- `test_frame_pair_probe_v5.py`

## 2 New fields and their sources

All native offsets come from the installed `UnityPlayer.dll` `5034cdc2…` (PDB `BFA91A8C…` age 1).
They are listed with provenance in `ProbeCore.cs` `UnityPlayerLayout`. They are read through
ReadProcessMemory and nothing is written.

| request | v6 field (where) | source |
|---|---|---|
| shadow-caster sort material hash | `shared_material_data.caster_pass_values_hash_0xd0`, also `state_block_values_hash_0xd4`, `flags_0xd8`, `material_instance_id_0xdc` (scene `materials[]`, every `pre_render` `material_sheets_changed[]`, dedup) | Material+0xb0 = SharedMaterialData*. `Material::UpdateHashes` 0x84e3b0 writes +0xd0 = `Pass::ComputePassValuesHash` and +0xd4. +0xdc is the instance id, validated equal to `GetInstanceID` for every material. |
| `_HasLastPositionData` | `native.smr_has_last_position_data_0x255` (every renderer row) | `SkinnedMeshRenderer::AddAsRenderNode` 0x77ae22 → RenderNode+0xe8 bit 9 |
| `_MotionVectorDepthBias` | `native.settings.build_settings_0xfa` | BuildSettings = ManagerContext[10]; branch in forward_values.py |
| GraphicsCaps +0xd0, +0x3fc, +0x3a1, +0x404 (+0xdd) | `native.settings.graphics_caps_0xd0_u32/_bit7`, `_0x3fc_u32/_f32`, `_0x3a1_u8`, `_0x404_u32/_f32`, `_0xdd_u8` | `RenderShadowMaps` 0x6a29db.., `ClearNoSkybox` 0x65f8e0, `PrepareShadowMaps` 0x69fecf. Raw u32 and f32 forms, because the meaning of +0x404 is not identified. |
| Camera +0x4ec, +0x558 and its RenderTexture+0xd0 | `camera_state.native.implicit_projection_flag_0x4ec`, `current_target_0x558`, `current_target_instance_id`, `current_target_antialiasing_0xd0` (pre_cull / pre_render, every camera) | `DoMotionVectorRenderLoop` 0x6c9b60. RenderTexture+0xd0 is validated equal to `RenderTexture.antiAliasing` for every RT. The instance id is read at `UnityEngine.Object.GetOffsetOfInstanceIDInCPlusPlusObject()`, validated for every renderer. |
| GfxDevice+0x21e8 | `camera_state.native.gfxdevice_single_pass_stereo_0x21e8`, `native.settings` | `SetupMotionMatrices` 0x6d011d; main-thread GfxDevice from TLS (`GetGfxDevice` 0x8cd260) |
| engine RT descriptors and sampling | `render_textures` in every `pre_render` and `post_render` (dedup) and `render_textures_compact` in the scene: every RenderTexture object (camera intermediates, Camera DepthTexture / MV texture, GrabPass Temp, CustomRender rtCamera, pooled TempBuffers). Each row has the full descriptor, filterMode, anisoLevel, mipMapBias, wrapModeU/V/W, IsCreated, and native TextureID / antiAliasing / temp-entry / desc key. | managed `RenderTexture` API + v5 native offsets |
| `_PreviousM`, transform update frame (previously gaps) | renderer `native.base_renderer_previous_local_to_world_0x48`, `base_renderer_local_to_world_0x08`, `base_renderer_transform_update_frame_0xbc`, `base_renderer_flags_0xf0`, and the raw `base_renderer_flatten_block_hex` (every renderer, scene and every pre_render) | `Renderer::AddAsRenderNode` 0x739400: BaseRenderer = Renderer+0x38. `BaseRenderer::FlattenBasicData` 0x650b50 copies +0x08..+0x88 → RenderNode+0x00..+0x80 and +0x88..+0xfc → +0x80..+0xf4 (node+0xb4 = +0xbc, node+0xe8 = +0xf0). The L2W is validated for ≥ 90 % of enabled MeshRenderers (see §4). |
| skin inputs of every skinned renderer | `Meshes=ALL` (session default): `static.json` lists every ChaControl SMR with `export` = exported / same_mesh_as / not_readable. Readable meshes are written once per mesh instance (vertices, normals, tangents, uv, uv2, triangles, bind poses, bone weights/indices, all blendshape frames). The scene lists `skinned_renderers_outside_any_chara`. Per-frame bones and weights are, as before, for all ChaControl SMRs. | managed `Mesh` API |
| skin double-buffer slot at bridge time | every SMR row: `smr_skin_buffer_0x258`, `smr_prev_skin_buffer_0x260`, `smr_dirty_0x28c`, `smr_force_matrix_recalc_0x256`, plus `skin_buffer_holds_frame` / `prev_skin_buffer_holds_frame` (scene and every pre_render, so in each of the 16 bridge renders). Driver hooks record the same at FBS, LateUpdateForce and ABMX. | `SkinnedMeshRenderer::AddAsRenderNode` 0x77abe0: `MeshRenderingData::Init(mesh, mesh, [+0x258], [+0x260])`, +0x28c cleared after skinning. Label rule in §3. |
| per-frame driver hooks | `frames/drivers_00k.json` (request `DriverHooks=1`), §3 | HarmonyX prefix/postfix |
| order certificate | `seq` in every driver record and every camera-callback record, from one global counter (`ProbeSequence`) | — |
| version string | 6.0.0 | — |
| clean quit | request `Mode=quit` → `quit_request_receipt.json`, then `ExitDialog.GameEnd(isCheck: false)` | decompiled `ExitDialog`: the dialog's Yes button calls exactly this. It sets `_isGameEndCheck=false`, so the `Application.wantsToQuit` handler returns true, then calls `Application.Quit()`. `Manager.Game.OnApplicationQuit` saves the game progress data, as on any normal exit. |

## 3 Driver hooks and skin slots

### Hook points

The drivers' float32 results depend on Mono JIT register allocation
([head skinning §9.1](hs2_offline_head_skinning.md)). A Harmony patch replaces the method with a
re-JITted copy, so the hooks are placed on **callers whose bodies contain no driver arithmetic**.

| hook | patched method | body |
|---|---|---|
| `eye` | `EyeLookController.LateUpdate` | `eyeLookScript.EyeUpdateCalc(target.position, ptnNo)` |
| `fbs` | `FaceBlendShape.'<Start>b__12_1'(int64)` | the UniRx `EveryLateUpdate` subscription calling `OnLateUpdate()` |
| `chara_update_force`, `chara_late_update_force` | `ChaControl.UpdateForce` / `LateUpdateForce` | calls only |
| `abmx` | `KKABMX.Core.BoneController.LateUpdate` | calls only |
| `neck` | `NeckLookControllerVer2.LateUpdate` | contains the target `Mathf.Lerp` loop; **jit_risk** recorded in every neck record |

Notes on placement:

- The `fbs` lambda is identified from IL: it is the single `<Start>` void(long) method whose code
  calls `OnLateUpdate`.
- The drivers themselves are never patched: EyeUpdateCalc, OnLateUpdate, NeckUpdateCalc and ABMX
  ApplyEffects.
- Prefixes run first and postfixes last, so the existing Hooah/PovX neck prefixes run inside the
  bracket.
- A missing target aborts the window before TriggerCapture. Hook errors are counted and never
  thrown into the game.

### Record contents per hook

| hook | contents |
|---|---|
| eye | call arguments (target position, ptnNo), the target, all fields of `EyeLookController` and `EyeLookCalc` (eyeObjs angleH/V, dirUp, reference vectors, origRotation, nowPtnNo, initEnd, targetPos, fixAngle, angle rates), static `EyeLookCalc.isEnabled`, deltaTime. Prefix only: Hooah `HijackLook._updatePositions` count (class `HijackLook` in `HS2_Hooah.dll`, global namespace). |
| neck | all fields of `NeckLookControllerVer2` (ptnNo, rate, target) and `NeckLookCalcVer2` (isEnabled, skipCalc, calcLerp, lookType, changeTypeTimer, nowPtnNo, backupPos, nowAngle, aBones fixAngle/fixAngleBackup/backupLocalRotaionByTarget/angleH/V, neckTypeStates isLimitBreakBackup), `changeTypeLerpCurve.Evaluate(InverseLerp(0, changeTypeLeapTime, changeTypeTimer))`, and the local TRS of each aBones referenceCalc/neckBone/controlBone at entry |
| fbs | all fields of `FaceBlendShape` (BlinkCtrl fixedFlags/BlinkFrequency/blinkMode/BaseSpeed/calcSpeed/blinkTime/count/openRate, BlinkCtrlEx, voiceValue, each controller's openRate/OpenMin/OpenMax/FixedRate/correctOpenMax/dictBackFace/dictNowFace/blendTimeCtrl, mouth useAjustWidthScale/tpcRand/sclNow/sclStart/sclEnd/adjustWidthScale), `UnityEngine.Random.state`, Time.time and deltaTime, skin slots |
| fbs prefix only (first LateUpdate, after the animation phase) | the animation snapshot: local TRS of every character transform; per Animator controller, avatar/isHuman, speed, updateMode, cullingMode, applyRootMotion, layer-0 current and next state (hash, normalizedTime, length, speed, speedMultiplier), transition info and IsInTransition; FinalIK `FullBodyBipedIK` enabled and its solver; Maker `CustomBase.customMotionIK`; Camera.main identity, position and ancestor local TRS; chaF_001 ancestor local TRS; mouth state |
| chara_update_force postfix | asVoice present/playing, `fileStatus.mouthAdjustWidth`, the local TRS of `cmpBoneHead.targetEtc.trfMouthAdjustWidth` and of `cf_J_MouthMove` |
| chara_late_update_force | skin slots |
| abmx postfix | mouth scales after ABMX, skin slots |

### Skin-slot label rule

The rule is from the session v5.0.1 mode-b pairing `5af92a08`:

- The frame's own skinning writes the buffer that is slot 0 at the end of that frame.
- A buffer therefore holds the skin of the last frame whose end-of-frame observation had it in
  slot 0. A buffer not yet seen as slot 0 is −1 (unknown).
- At the bridge renders of frame N this gives slot 0 = N−2 and slot 1 = N−1, as observed.

The raw pointers are exported with the labels, so the rule can be re-checked against the RDC.
The swap point itself is still a source trace (open). The hook-point observations bracket it.

## 4 Validation added (fail closed, before the trigger)

| check | fails the window when |
|---|---|
| native instance-id offset | it differs from `GetInstanceID` for any scene renderer |
| SharedMaterialData+0xdc | it differs from `Material.GetInstanceID` for any renderer material |
| RenderTexture+0xd0 | it differs from `RenderTexture.antiAliasing` for any RT |
| BaseRenderer+0x08 | fewer than 90 % of the enabled MeshRenderers are bit-equal to `MeshRenderer.localToWorldMatrix` |

All mismatches are listed. The 90 % bound is not a tolerance on values: RendererUpdateManager
refreshes the renderer copy at culling, so a renderer moved by a script after the last cull differs
legitimately, while a wrong layout matches none.

Driver hooks: a missing target, an FBS lambda that cannot be identified, or a failed patch aborts
the window.

## 5 Not obtainable (explicit gaps)

1. **Shadow-caster cull set.** `Camera::CustomRender` 0x661400 receives its CullResults by
   reference from `StandaloneCustomRender`'s stack and stores no pointer on the Camera
   (prologue 0x661400..0x66152b). The caster set (culling 0x6ac320) is therefore not readable from
   persistent state. It stays external:capture. The caster *order* is now computable from the exported
   SharedMaterialData+0xd0 and the existing node data.
2. **Meshes with `isReadable == false`** have no CPU vertex data. They are listed as `not_readable`.
3. **Built-in texenv storage** (v5 gap 3) is unchanged.
4. **Skin swap point** in source (see §3); the probe supplies observations, not the trace.
5. **Neck hook JIT risk:** the patched `NeckLookControllerVer2.LateUpdate` contains the target
   `Mathf.Lerp` loop. Whether the neck results stay bit-identical to an unhooked run must be
   confirmed by a window with `DriverHooks=0` against the hosted runtime.

## 6 Runbook (main agent, user approval to launch)

```powershell
$S = 'C:\Users\13666\Workspace\Face2Parameter\tools\offline_renderer\frame_pair_probe\paired_capture_session.ps1'
```

1. Run `& $S launch`. It checks the pins (probe `42da95fd…`, bootstrap, RenderDoc, bridge, readback,
   UnityPlayer, Doorstop config, manifest) and that the v6 session and bulk dirs are fresh. It
   creates the bulk dir, copies the manifest, deploys the probe and launches. At the menu, run
   `& $S restore-config`.
2. **BepInEx log check:** the log must show `FramePairProbe 6.0.0 ready`.
3. **Maker and card:** `/game/maker/start` (female), then a read-only `/maker/card/load` of the
   DINOv2 v1 baseline (`74d2625f…`). Record snapshot, ABMX, express and renderstyle before and
   after each window.
4. **Mode a:** `& $S request -RequestMode frame` (Frames 8, TriggerIndex 3, Meshes ALL,
   DriverHooks 1). Then convert the RDC and verify, as in v5.
5. **Mode b:** `& $S request -RequestMode renderstyle -BodyJson <body>`. The body is written with
   the Write tool, never through a shell escape. Take the target from the live `o_head` bounds
   centre; ortho 1.42995071 at 1086×1448. Then convert and verify.
6. **Optional, for the neck JIT check:** one more `frame` window with `-DriverHooks 0`.
7. **Verifier:** `verify_probe_v5_session.py` must print `accepted`. For v6 windows it adds:
   - `driver_hooks_installed` (6 hooks), `driver_hook_errors_zero`;
   - `drivers_export_of_captured_frame`, `drivers_single_frame`;
   - `driver_hooks_all_fired_for_character` (prefix and postfix of every hook).

   The observed LateUpdate order against the installed MonoScript order is reported under
   `observations` and is not part of acceptance.
8. **Quit:** `& $S request -RequestMode quit`. The receipt is
   `<session>\quit_<nonce>\quit_request_receipt.json`. Wait for the HoneySelect2 process to exit,
   then run `& $S cleanup`.

## 7 Tests (CPU)

| suite | result |
|---|---|
| `outputs/offline_renderer_probe_v6_20261025/core_tests/ProbeCoreTests.exe` | 53/53, including quit request parsing, the `DriverHooks` key, the skin-slot labels on the 8748/8749/8750 pattern, `GuardValue` and the sequence counter |
| `.venv/Scripts/python.exe -m unittest tools.offline_renderer.test_frame_pair_probe_v5` | 11/11: the v6 verifier accepts a complete hook set and rejects missing hooks or hook errors |

## 8 ABMX live records: same-session re-execution (research only, nothing run)

Both records were checked against the installed bridge 0.31.14 (`efa943cb…`, MVID `fd8a7dc8…`)
by decompiling it and its predecessors. Decompiled copies are in the session scratchpad.

| record | ran on | collector | outcome |
|---|---|---|---|
| `HS2Mod/artifacts/infrastructure_live_20261005/abmx_lowered_v2` | bridge 0.30.0 | `HS2Mod/tests/geometry_export/live_abmx_lowered.py` | re-executable |
| `HS2Mod/artifacts/infrastructure_live_20261006/native_radial_drivers_head3_v1` | bridge 0.30.1 | `live_native_radial_drivers_v2.py` | blocked as recorded |

- **Lowered is re-executable.** Every route and field it uses is accepted by 0.31.14: abmx GET,
  batch and trace; face base, shapes batch, express; snapshot; render. Faithfulness differs in two
  points:
  - the starting character was never recorded (likely
    `UserData/chara/female/HS2ChaF_20260627152012646.png` `acd722f6…`, to be confirmed by a live
    snapshot comparison);
  - the 2026-10-24 ABMX refreeze changes the CheekUp L/R offsets by 2.98e-8. The expected result is
    `E:/HS2_offline_renderer_data/abmx_refreeze_20261024/live_successors/abmx_lowered_v2_compiled_execution.json`.
- **Head3 is blocked as recorded.** `tools/native_radial_target/validate_drivers.py` refuses the
  driver contract `native_face_drivers_install_v1/driver_contract.json` (`cdd0f927…`). It is pinned
  to bridge MVID `e7aee083…`, so the run stops before any write. A new contract with the installed
  MVID is needed, made the way `live_physics_controlled_consumer_v3.py` lines 39–48 does it. The
  record's own downstream verdict was `passed: false` (tooth topology mismatch), and a re-run should
  reproduce that rejection.
- **Session constraints:**
  - run strictly after the last probe window and before the quit request; no overlap (bridge gate,
    probe queue check, ABMX trace patches);
  - neither script loads, saves or restores a card or `/maker/renderstyle`. Load the record's
    character first, then reload the paired-session card and re-check its state afterwards;
  - instance ids and ABMX caches change, so any later probe window is a new baseline.
- **Exact steps:** offline P1–P3, in-game steps 1–6 (all need user approval), command lines and
  acceptance comparisons. They are in the research report relayed to the coordinator; the commands
  run from `C:\Users\13666\Workspace\HS2Mod` with
  `Face2Parameter\.venv\Scripts\python.exe -B tests\geometry_export\live_abmx_lowered.py --out
  E:\HS2_offline_renderer_data\abmx_rerun_<date>\abmx_lowered_v2 --heads 2 --compiler-root
  C:\Users\13666\Workspace\Face2Parameter --contract
  C:\Users\13666\Workspace\Face2Parameter\outputs\abmx_replay_20261005\installed_contract_v2.json`,
  and the head3 analogue with `--head 3 --driver-contract <new contract>`.
