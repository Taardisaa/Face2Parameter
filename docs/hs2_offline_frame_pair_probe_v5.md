# HS2 offline renderer — frame-pair probe v5 for the second paired session (2026-10-20)

Probe v5 is in `tools/offline_renderer/frame_pair_probe/`. It extends the v4 probe of the first
paired session ([head skinning §7](hs2_offline_head_skinning.md)). It adds every field that the
generators listed as missing ([forward §4](hs2_offline_unity_forward.md),
[PPv2 §5](hs2_offline_ppv2.md), [plugins §4](hs2_offline_plugin_effects.md)). It also adds a
second capture mode for the frame in which the bridge's `/maker/renderstyle/capture` runs its 16
renders.

No game was launched to build or test it. Nothing was deployed into `E:/HoneySelect2_ArcticFox/`.

## 1 Build identity

| item | value |
|---|---|
| build | `outputs/offline_renderer_probe_v5_20261020/build_v5/HS2_FramePairProbe.dll` (`dotnet build -c Release`, deterministic: a second build gave the same bytes) |
| SHA-256 | `b720c453c83ff5776d207e164f632bebd85339424cf427ee2e67a0125b413172` |
| plugin | `codex.hs2.framepairprobe` 5.0.0; request file `HS2_FramePairProbe.request.ini` beside the DLL (unchanged) |
| shader manifest | `outputs/offline_renderer_probe_v5_20261020/shader_manifest_v5.tsv`, SHA-256 `e403c5ad…2036`: 95 shaders, 1161 P, 65 K, 531 G lines |
| session script | `tools/offline_renderer/frame_pair_probe/paired_capture_session.ps1` (pins the probe, manifest, bootstrap, RenderDoc, bridge, readback, UnityPlayer and Doorstop config) |
| native layouts | recovered from `UnityPlayer.dll` SHA-256 `5034cdc2…13923` (PDB GUID `BFA91A8C…` age 1). The probe refuses any other binary. |

Source files:

- `FramePairProbe.cs`: orchestration, markers, RenderDoc API, v4 frame/static/mesh exports.
- `SceneState.cs`: v5 scene export.
- `RenderRecorder.cs`: per-render records.
- `NativeUnity.cs`: native reads.
- `RawTextures.cs`: raw texels.
- `BridgeCapture.cs`: mode renderstyle.
- `PluginPaths.cs`: explicit plugin member paths.
- `ProbeCore.cs`: Unity-free core and the layout constants with their provenance.
- `build_probe_manifest.py`: builds the manifest.
- `verify_probe_v5_session.py`: success checks.

## 2 Frame plan and the two modes (from source)

**RenderDoc retry.** RenderDoc 1.46 starts a TriggerCapture capture at the next Present. This is in
`d3d11_device.cpp`, Present at 2803 and retry at 2450..2516. The first attempt fails with
"Uncapped Map()/Unmap()" because Unity keeps maps open across Present. The ordinary-failure path
then calls `AttemptCapture` and records the next frame. All three earlier captures behaved this way
(17674→17675, 12739→12740, 26058→26059 in RenderDoc numbering). Each time the in-frame marker named
**trigger frame + 2** in `Time.frameCount`.

Window index `t = TriggerIndex`. Every index is one end-of-frame point (WaitForEndOfFrame):

| index | both modes | mode `renderstyle` only |
|---|---|---|
| every k | flush the per-render records of the frame; v4 `frame_k`; v5 `scene_k`; v01 marker | |
| t | `TriggerCapture` | |
| t+1 (failed attempt) | queue the raw texel export (render thread, before Present t+1) = contents at the start of t+2 | block until the export is written, then POST the bridge capture and block until its action is queued |
| t+2 (captured) | the captured frame | the bridge's 16 renders run in this frame's Update, between HS2PAIRBRIDGEv05 begin/end markers |

**Bridge source chain.** These steps come from the installed `HS2_McpBridge.dll` (`efa943cb…`,
decompiled):

1. `HttpServer.HandleRequest` handles `/maker/renderstyle/capture` by calling `RunOnMainThread(func, 5)`.
2. `MainThreadDispatcher.Invoke` passes the work to `KoikatuAPI.SynchronizedInvoke`.
3. That calls BepInEx `ThreadingHelper.StartSyncInvoke`, which appends to `_invokeList` under `_invokeLock`.
4. `ThreadingHelper.Update` swaps the list out and runs it, so the capture runs in the Update of the
   first frame after the enqueue.

The probe makes that frame deterministic:

- It POSTs from its end-of-frame coroutine of frame t+1, after every Update of t+1.
- It blocks the main thread until the bridge closure is in `_invokeList`. A precise match is the
  dispatcher closure whose `action.Target.body` equals the posted body. The match kind is
  recorded as `queue_match`.

The bridge waits 5 s for the reply. That wait only changes the reply to 504 `operation_in_progress`.
The probe then polls `/operations/{id}`.

The installed bridge sends the capture through `HS2_RenderDocDispatch` if
`BepInEx/Plugins/HS2_McpBridge/HS2_RenderDocScope.request.json` exists. The probe and the script
refuse to run in that case.

**Plug-in events.** `GL.IssuePluginEvent` reaches `GfxDeviceClient::InsertCustomMarkerCallback`
(0x59a6d0). That function publishes the write position at once (`[buffer+0xc0]`), so the blocking
raw wait cannot deadlock the render thread.

**Callback order.** Taken from `Camera::CustomCull` 0x660f70 and `Camera::CustomRender` 0x661400:

- **pre_cull** fires after the component `OnPreCull`, so it sees the CTAA jitter and the PPv2 state.
- **pre_render** fires after culling and after DHH SSS `OnPreRender`. The nested profile and
  lighting-camera renders appear as child records.
- **post_render** fires after the component `OnPostRender`.

## 3 New fields and their sources

### Unity forward

| field | v5 source | where |
|---|---|---|
| light `localToWorldMatrix`, position, rotation (enabled and all) | `Light.transform` | `scene.lights[].transform`, every `pre_render.lights[]` |
| exact camera position | `Camera.transform.position` (+ rotation, L2W, W2L) | `scene.cameras[].transform`, every `camera_state.transform` |
| per-render camera state | matrices, pixelRect, clip planes, culling mask, depth mode, command buffers, full public property reflection | `renders_k.json` pre_cull / pre_render |
| `_PreviousVP`, implicit-projection flag | native `Camera+0x598`, `+0x5d8` (VP), `+0x4ec` (`UpdatePreviousViewProjectionMatrix` 0x66e570, `SetupMotionMatrices` 0x6d0249, `SetCameraShaderProps`) | `camera_state.native` |
| culling inputs per render | every renderer's enabled/active/layer/bounds/L2W/shadowCastingMode/receiveShadows/probe usages/renderingLayerMask/sorting/lightmap index/root-bone TRS, every light's culling and shadow fields | `pre_render.renderers[]`, `lights[]` |
| visibility bit | `Renderer.isVisible` and native `+0x128 & 0x1000`, `+0x160` (`Renderer_Get_Custom_PropIsVisible` 0x970780) | `pre_render.renderers[].native` |
| `ReflectionProbe.defaultTexture` and `defaultTextureHDRDecodeValues` at render time | static getters in every pre_render | `pre_render.reflection_probe_default`, `scene.statics` |
| `_Time` and all engine built-ins in OnPreRender | main-thread GfxDevice built-in params: 121 vectors at `+0x11e8`, 21 matrices at `+0x1978` (`ShaderScripting::GetGlobalVector` 0x711230, `GetGlobalMatrix` 0x711100; GfxDevice from TLS as in `GetGfxDevice` 0x8cd260). `_Time` is built-in vector 75. | `builtin_params` in pre_cull/pre_render/post_render and `scene.native` |
| global sheet, all globals | `g_SharedPassContext+0x28` ShaderPropertySheet with raw GPU-bound bytes | `global_sheet` per callback; `scene.native.global_sheet` |
| global keywords | `g_SharedPassContext+8` (256 bits) + keyword name map (`keywords::GetKeywordName` 0x82c990) | `global_keywords` per callback |
| QualitySettings at runtime, `SystemInfo.graphicsMemorySize`, `maxTextureSize` | statics reflection (as v4) | `scene.statics` |
| BuildSettings `+0x101`/`+0x102`, active tier, tier cascadedShadowMaps, caps video memory (float at `+0xd8`) and `+0xf8` | ManagerContext[10] / [6], GraphicsCaps (`PrecalcLightColor` 0x69f4a0, `GetGraphicsMemorySize` 0x2f8750) | `scene.native.settings` |
| renderer submeshes | `Mesh.GetIndexStart/Count/BaseVertex/Topology` | `scene.renderers[].submeshes` |
| material property blocks | native sheet behind `MaterialPropertyBlock.m_Ptr`, per slot and whole renderer | `scene.renderers[].property_blocks_per_slot[].native_sheet` |

### PPv2

| field | v5 source | where |
|---|---|---|
| `Behaviour.enabled` of each volume | `enabled` on every effect component and every camera/light component | `scene.effect_components[].enabled`, `plugin_member_paths.PostProcessManager` |
| `m_SortedVolumes` / `m_Volumes` order, `m_SortNeeded` | `PostProcessManager.instance` (registration order, priority, isGlobal, enabled) | `plugin_member_paths.PostProcessManager`; full statics in `plugin_static_fields` |
| every `Spline.curve` keys (time, value, in/out tangent, in/out weight, weightedMode) and wrap modes; `Spline.cachedData` | AnimationCurve case (`keys`, `preWrapMode`, `postWrapMode`, Keyframe private fields); depth limit 6 → 12 with a path cycle guard; output formats are v4's | inside profiles and the layer bundles in `effect_components` |
| temporary RT pool free list at frame start | native `RenderBufferManager::Textures`, see 3.1 | `render_buffer_pool` at every pre_cull/post_render and at the end of each frame |
| camera-target RT ("TempBuffer 100 3200x1800") | every `RenderTexture` object: descriptor, aniso/filter/wrap, `IsCreated()`, native TextureID, temp entry, desc key | `scene.render_texture_ids` + `scene.textures` |
| `MultiScaleVO.m_AmbientOnlyAO` settings | reached through the layer bundle (depth 6 < 12); its texture row has anisoLevel/filterMode/wrapMode | `scene.textures` |

### 3.1 Temporary RT pool layout

Recovered from GetTempBuffer 0x756540, ReleaseTempBuffer 0x75a610, GarbageCollect 0x754440 and
`List::push_front` 0x6294f0:

| item | layout |
|---|---|
| global | `RenderBufferManager*` at RVA 0x1547E70. `Textures` is at offset 0. |
| `Textures` | `+0` buckets, `+8` byte mask, `+0xc` key count, `+0x20` in-use `List` root, `+0x34` TempBuffer name counter, `+0x38` GC frame |
| bucket (0x40 bytes) | hash (−1 empty, −2 deleted); 11-dword `RenderTextureDesc` key at `+8`; free-list head at `+0x38` |
| TextureEntry | ListNode with `+0x10` last frame and `+0x14` RenderTexture instance id |
| list order | Release makes the released entry the head. Acquire takes the head and moves to `head->next`. The walker therefore reports **acquisition order (LIFO)**. |
| validation | Before the trigger, every entry is checked against its RenderTexture's `+0x138` back-pointer. |

### Plugins (§4 of the plugin doc)

`plugin_member_paths` holds every listed path with its value or its read error:

- **SSS:** all 25 fields including backing fields, plus `NoiseTexture` (texture row and raw texels).
- **SSS cameras:** `GameObject.Find` existence and the scene objects of both SSS cameras, the
  profile camera's `allowHDR`, and `SSS_convolution.initFOV` / `AllowMSAA` / `_BlurMaterial`.
- **CTAA:** `CTAA_PC.enabled` and all 19 listed fields. Its RTs are exported as texture rows with
  raw texels.
- **CustomRender:** `update` and `<rtCamera>k__BackingField`.
- **Engine values:** `Screen.width/height`, the five QualitySettings fields, and
  `DHH_Base.ConfigSSS.Value`.
- **DoF:** `DHH_Main.depthOfFieldLayer.focusDistance.value`.
- **Materials:** the renderers with `AIT/Liquid` / `AIT/main namida` materials. Their values are
  in `scene.materials` as native sheets plus typed reads.

`effect_components` still holds every field of every DHH/CTAA/PPv2/CustomRender component.
`plugin_static_fields` holds the statics of the DHH, BetterAA/CTAA and PPv2 assemblies.

In mode renderstyle the **bridge request body** (verbatim, plus its SHA-256), the HTTP replies and
the operation poll are in `capture_receipt.json.bridge`.

### Materials

`scene.materials[]` keeps the v4 fields. v5 adds:

- the pass list with `GetShaderPassEnabled`;
- `native_sheet`: the material's ShaderPropertySheet with every entry's name (from the native
  FastPropertyName table), type, array size, gamma flag and raw stored bytes. This is what is bound
  to the cbuffers. The managed getters return `LinearToGammaSpace(stored)` for gamma-flagged
  entries (`GetFloatValueInsRGBSpace` 0x83d3f0), so they are not exact.
- `typed_properties_managed`: managed reads for the manifest names of that shader that are already
  registered.

The manifest comes from `shader_library.py` reflection for **all** 95 library shaders. It includes
serialized properties, every non-built-in cbuffer variable (`uniform_map`) and every texture
binding (`resources`). It replaces the 37-shader `shader_manifest.tsv`.

The global G/K lists come from the library too:

- `global_uniforms_managed` cross-checks them.
- The complete values are in the native global sheet and the built-in params.

Per render, every material sheet is dumped in `material_sheets_changed` when its SHA-256 changed
since its last dump.

### Raw textures

`raw_textures.json` and `<RawOutDir>/texture_<instance id>/descriptor.json + subresource_*.bin`
come from `HS2_RenderReadback.dll` (`b1cb390a…`, the DLL that HS2_RenderAudit uses, loaded from its
installed path after a hash check). Its exports are called directly so that the output directory
comes from the request. `HS2_RenderAudit.Capture()` would instead read a sidecar config in the game
folder, and it covers only head renderers.

The selection is the union of:

- every texture referenced by the scene export (materials, effects, probes, RenderSettings, the
  character's composed `customTexCtrlFace/Body` textures);
- every texture whose native TextureID is bound in any material sheet, or in the global sheet at
  any recorded callback;
- every RenderTexture object.

Uncreated RTs are skipped with the reason "not created". The probe never creates one.

Before queueing, the probe checks that free space on the RawOutDir drive is at least the upper
bound (16 B/texel ×4/3 for mips) plus `RawMinFreeGB`.

Drives present on 2026-10-20:

| drive | free |
|---|---|
| C: | 6.3 GB |
| D: | 103 GB |
| E: | 1308 GB |
| Y: (NAS) | 1969 GB |

All bulk data goes to `E:\HS2_offline_renderer_data\paired_session_v5_20261020\` (user decision).

### v4 defects fixed

- `RenderTexture.colorBuffer` and `depthBuffer` were reflected. Their natives
  (`RenderTextureScripting::GetColorBuffer` 0x705390 and `GetDepthBuffer` 0x705400) call
  `RenderTexture::Create`.
- `Shader.IsKeywordEnabled` runs `keywords::Create` (0x712040), which registers unknown names.
- Managed `HasProperty` and `GetGlobal*` by string register unknown FastPropertyNames.

v5 avoids all three. Static fields are read only for types without a static constructor, or for
types whose statics are in use every frame. Otherwise the read would run the static constructor.

### 3.2 Pre-trigger validation (fail closed)

Every check must pass before TriggerCapture, or the window aborts with nothing captured:

- the native name table: `PropertyToID(_MainTex)`;
- the caps video memory against `SystemInfo.graphicsMemorySize`;
- the pool back-pointers;
- the renderer visibility bit against `isVisible`;
- material sheet floats and vectors (no gamma flag) and texture ids against `Material.Get*(int id)`;
- global sheet values against `Shader.GetGlobal*(int id)`;
- built-in vectors and matrices against `GetGlobalVector/Matrix(built-in id)`;
- the keyword bits against `IsKeywordEnabled` for every registered keyword.

## 4 Fields that cannot be obtained (explicit gaps)

1. **Per-camera visible-renderer list and per-light shadow-caster cull result.** Neither is exposed
   by the 2018.4 built-in pipeline API: there is no CullResults accessor, and SRP culling needs a
   ScriptableRenderContext. They live in the native CullResults inside the stack frame of
   `Camera::StandaloneCustomRender`, which persistent state cannot reach. `Renderer.isVisible` and its
   bit are frame-cumulative, not per camera.

   Recorded instead: every culling input per render, for the transcribed `CullScene` / `0x68b480`
   rules, and the visibility bit at each callback. The RDC draw list stays the acceptance oracle.
   Shadow caster culling (`PrepareShadowedLight` 0x67eb30 / `CalculateShadowCasterCull` 0x679140)
   must be transcribed by the forward sibling.
2. **Per-renderer previous L2W (`_PreviousM`), `_HasLastPositionData`, sorting fudge.** These are
   native renderer fields whose offsets were not recovered. `_PreviousM` was already reproduced from
   the frame F−1 export in session 1.
3. **Built-in texenv values** (`unity_SpecCube0`, `unity_Lightmap`, …). The 25 names are exported,
   but their storage in GfxDevice was not located. Textures bound to them are covered only when they
   are also referenced elsewhere.
4. **ShaderPropertySheet texture/buffer entries beyond their first dword** (TextureID /
   ComputeBufferID). The region is exported raw but not interpreted. `_ST` / `_TexelSize` / `_HDR`
   auxiliaries are separate vector entries and are exact.
5. **Shader `Shader Forge/FlatColorTransparent`** (used by a scene material in session 1) is not in
   the extracted library. That material gets its native sheet but no manifest reads.
6. **Static fields** of types with a static constructor that may not have run. They are listed as
   skipped so that the read does not run the constructor.
7. **Dynamic RT contents in mode renderstyle** are read at the end of frame t+1, i.e. at the start of
   the captured frame. RTs created inside the bridge frame (the bridge's own RT, the per-render SSS
   profile textures) have no pre-frame contents to export.
8. **Assumed capture offset.** Capture = trigger + 2 holds when the first attempt fails, as in all
   three observations. If an attempt ever succeeded on the first try, the RDC would hold frame t+1.
   For mode renderstyle that capture would then lack the bridge renders. `verify_probe_v5_session.py`
   rejects such a window instead of pairing it to another frame.

## 5 Runbook for the main agent (user approval required to launch)

All commands run from `C:\Users\13666\Workspace\Face2Parameter` in PowerShell 7. `$S` is the
session script:

```powershell
$S = 'tools\offline_renderer\frame_pair_probe\paired_capture_session.ps1'
$B = 'E:\HS2_offline_renderer_data\paired_session_v5_20261020'
```

### Step 0: before launch

These preconditions must hold. The script checks each of them and stops on a mismatch.

- HS2 is not running.
- `$B`, `HS2Mod\artifacts\chenger_reference_fit_20261012\paired_capture_session_v5_20261020` and
  `E:\HoneySelect2_ArcticFox\BepInEx\plugins\HS2_FramePairProbe` do not exist.
- `HS2_RenderDocScope.request.json` is absent.

### Step 1: launch

Run `pwsh $S launch`. It:

1. verifies all eight pins;
2. requires E: to have at least 100 GB free;
3. creates `$B` and copies the pinned manifest into it;
4. deploys the probe;
5. calls `early_scope_game.py launch`.

When the game reaches its menu, run `pwsh $S restore-config`.

### Step 2: prepare the state as in session 1

Do this through the bridge, exactly as in the first session:

1. `/game/maker/start` (female).
2. Load the DINOv2 v1 baseline card `74d2625f…` read-only.
3. Record these before and after each window: `GET /maker/snapshot`, the raw ABMX,
   `GET /maker/face/express` and `GET /maker/renderstyle`.

Keep SSS in the state of session 1 (`ConfigSSS.Value` true). Do not save any card.

### Step 3: mode `frame` (an ordinary Maker frame)

Run `pwsh $S request -RequestMode frame` (Frames 8, TriggerIndex 3).

The written `HS2_FramePairProbe.request.ini` (copy: `<session>\request_<nonce>.ini`) is:

```ini
Mode=frame
Nonce=<32 hex>
OutDir=E:\HS2_offline_renderer_data\paired_session_v5_20261020\window_<nonce>
RawOutDir=E:\HS2_offline_renderer_data\paired_session_v5_20261020\raw_<nonce>
Frames=8
TriggerIndex=3
PollSeconds=120
CapturePrefix=E:\HS2_offline_renderer_data\paired_session_v5_20261020\rdc_<nonce>
ExpectedRenderDocSha256=809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d
ExpectedUnityPlayerSha256=5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923
ReadbackDll=E:\HoneySelect2_ArcticFox\BepInEx\Plugins\HS2_McpBridge\HS2_RenderReadback.dll
ExpectedReadbackSha256=b1cb390aa8f927e749730ec0c1f1133a944d0d5bd1f6138c32480fdc50245cb3
RawTimeoutSeconds=900
RawMinFreeGB=50
ShaderManifest=E:\HS2_offline_renderer_data\paired_session_v5_20261020\shader_manifest_v5.tsv
Meshes=o_head,o_namida,o_tang,o_eyelashes,o_eyeshadow,o_eyebase_L,o_eyebase_R,o_tooth
```

Wait until `window_<nonce>\capture_receipt.json` exists. During the window:

- send no bridge calls;
- leave the camera untouched.

### Step 4: mode `renderstyle` (the bridge's 16-render frame)

1. Write a body file, e.g. `$B\bridge_body_input.json`. These are the native_present_session_v1
   values. Check `target` against the head pose of the loaded card from Step 2 (`cf_J_Head`); keep
   the values when it matches.

   ```json
   {"path":"E:\\HS2_offline_renderer_data\\paired_session_v5_20261020\\bridge_capture_1.png",
    "yaw":0.0,"target":[0.0930694,14.2913847,0.661002934],"ortho_size":1.42995071,
    "width":1086,"height":1448,"freeze_pose":true}
   ```

   Do not add `geometry_out_path`. The installed bridge `efa943cb…` does not accept it; the key exists
   only in the uncommitted source.
2. Run `pwsh $S request -RequestMode renderstyle -BodyJson $B\bridge_body_input.json`. The request
   holds the frame-mode lines with `Mode=renderstyle`, plus:

   ```ini
   BridgeUrl=http://127.0.0.1:43127/maker/renderstyle/capture
   BridgeBodyFile=E:\HS2_offline_renderer_data\paired_session_v5_20261020\bridge_body_<nonce>.json
   ExpectedBridgeSha256=efa943cb78099fcfccc11a10d4947693216e58e9525006b74ab2ee20d4ac8c21
   BridgeDispatchRequestFile=E:\HoneySelect2_ArcticFox\BepInEx\Plugins\HS2_McpBridge\HS2_RenderDocScope.request.json
   BridgeTimeoutSeconds=180
   ```
3. **Do not POST `/maker/renderstyle/capture` yourself, and send no bridge request of any kind until
   `capture_receipt.json` exists.** The probe posts at the end of index 4. Any other bridge call
   would occupy the request gate (409) or queue another ThreadingHelper action.
4. Expect a main-thread stall at index 4 while the raw texels are written (seconds to minutes), then
   the bridge capture's own stall in the next frame.

### Step 5: after each window

Re-read the state routes of Step 2 and confirm they are unchanged.

### Step 6: quit and clean up

Quit HS2 as in session 1, then run `pwsh $S cleanup`. The cleanup receipt records the probe removal,
the Doorstop hash, that the dispatch file is absent, and the file count and bytes in `$B`.

### Expected outputs per window (in `$B`)

- **`window_<nonce>\`:**
  - `request.ini`, `static.json` (v4 static data plus the native property/keyword tables),
    `mesh_*.json`;
  - `frames\frame_00k.json` (v4 layout), `frames\scene_00k.json` (v5 scene) and
    `frames\renders_00k.json` (per-render records), for k = 0..7;
  - `capture_receipt.json`, `raw_textures.json`, `probe_log.txt`.
- **`raw_<nonce>\texture_<id>\`:** `descriptor.json` and `subresource_*.bin`.
- **`rdc_<nonce>_frame<N>.rdc`:** about 290 MB. N is the RenderDoc frame number, which is
  `Time.frameCount − 1`.
- **Mode renderstyle:** also the bridge PNG named in the body.

### Success checks

1. In `probe_log.txt`, every `native ok:` validation line is present and there is no `FAILURE`.
   `aborted_before_trigger` is false and `captures_after − captures_before = 1`.
2. The RenderDoc log shows "Failed to capture, frame N: Uncapped Map()/Unmap()" followed by
   "Finished capture, Frame N+1".
3. Convert the RDC with the pinned RenderDoc 1.46 directory on PATH:

   ```powershell
   outputs\offline_renderer_repair_20261013\official_action_provenance_msvc_v1\renderdoc_replay_msvc.exe convert <rdc> rdc $B\parse_<nonce>\capture.zip.xml zip.xml
   ```

4. Run the verifier:

   ```powershell
   .venv\Scripts\python.exe tools\offline_renderer\frame_pair_probe\verify_probe_v5_session.py --xml $B\parse_<nonce>\capture.zip.xml --window $B\window_<nonce> --nonce <nonce> --out $B\verify_<nonce>
   ```

   It must print `accepted`. It checks:
   - one own v01 marker equal to `expected_capture_frame` (= trigger + 2);
   - exports and renders of that frame;
   - every raw job complete, or skipped with a reason.

   In mode renderstyle it also checks:
   - 16 bridge renders, ordinals 0..15, all in the captured frame;
   - the begin and end bridge markers in order before the v01 marker;
   - the end ordinal 15;
   - HTTP 200 directly or through `/operations`, the PNG written, and `temporal_render_count` 16.

   A rejection is kept as evidence. It is not re-paired to another frame.

## 6 Tests (CPU only)

```powershell
.venv/Scripts/python.exe -m unittest tools.offline_renderer.test_frame_pair_probe_v5 -v
```

| test | result |
|---|---|
| C# core (`tests/ProbeCoreTests`) | 42/42 |
| Python suite (manifest, verifier, core runner) | 8/8 |

The C# core test builds the fake native structures and covers:

- the pool walker against re-implementations of the disassembled ReleaseTempBuffer / GetTempBuffer
  list operations (LIFO order, count and ring checks);
- sheet, keyword-map and name-table parsing;
- request validation and the action order per index;
- float32 JSON round trip on 200,000 random bit patterns;
- the `Guard` error isolation and the bridge-body rules.

The Python suite covers the manifest builder (whole library, type conflicts) and the verifier
(accepting and rejecting synthetic captures).
