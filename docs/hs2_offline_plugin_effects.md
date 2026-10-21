# HS2 offline renderer — plugin effect generators (2026-10-20)

Scope: the plugin work inside each of the bridge's 16 capture renders (frame graph segments
0–15 of the retry capture, [frame graph](hs2_offline_frame_graph.md)), reimplemented as
generators under the [architecture contract](hs2_offline_renderer_architecture.md):

- DHH screen-space SSS
- HS2_BetterAA CTAA
- AIT/Liquid with Unity GrabPass
- the bridge capture loop

CPU only. No game launch, capture, deployment or GPU replay. This is generator evidence. It
is not pixel acceptance.

| file | content |
|---|---|
| `src/render/hs2pipe/dhh_sss.py` | `SSS.OnPreRender`/`OnPostRender`, `SSS_convolution.OnRenderImage` |
| `src/render/hs2pipe/ctaa.py` | `CTAA_PC` jitter, parameters, history ping-pong |
| `src/render/hs2pipe/liquid.py` | GrabPass semantics and placement in the forward loop, Liquid globals |
| `src/render/hs2pipe/capture_loop.py` | `MakerRenderStyleService.Capture` and the per-render callback order |
| `src/render/hs2pipe/plugin_common.py` | Mono float rule, RT records and D3D11 lowering, temp-RT pool, cbuffer packing, `PluginLowering` |
| `tools/offline_renderer/validate_plugin_effects.py` | byte-diff against the capture |
| `tools/offline_renderer/test_plugin_effects.py` | 19 tests |
| `tools/offline_renderer/plugin_effects_validation_20261020.json` | compact receipt |

Bulky output (decompiled code, disassembly, reports, plans) is in the ignored
`outputs/offline_renderer_plugins_20261020/`.

## 1 Sources (installed, read-only)

| component | file | SHA-256 | method |
|---|---|---|---|
| DHH Graphics Enhancer 1.15.30 | `BepInEx/Plugins/HS2_Plugins/DHH_HS2.dll` | `44137712ecaa9465340e2dc6abc123bed437fc94bfecea0f181c19253206ca3c` | ilspycmd 8.2 → `decompiled/DHH` |
| Better Anti-Aliasing 1.6 | `BepInEx/Plugins/HS2_Plugins/ManlyMarco/HS2_BetterAA.dll` | `f5ecfeea7a361f30733a1448dee42d5751690178a07796488e9e71a857f5a464` | ilspycmd → `decompiled/BetterAA` |
| Maker camera components | `HoneySelect2_Data/Managed/Assembly-CSharp.dll` | `d7e7e0d51403a0a3a71935a14487f063c6bf6f93f004b98b8f4b8b89047af770` | `CharaCustom.CustomRender`, `GameScreenShot`, `CameraControl_Ver2` |
| Unity 2018.4.11f1 engine | `UnityPlayer.dll` (public PDB, GUID BFA91A8C…) | via `unityplayer_disasm.py` | GrabPass, temp-RT pool, forward loop → `unity_disasm/` |
| DHH SSS shaders | embedded `sss` bundle | `ad7c8f1a…` | existing `sss_shader_programs_v2` export |
| CTAA shaders | embedded `ctaa` bundle | see `embedded/*.provenance.json` | existing `hs2_betteraa_ctaa` export |
| AIT/Liquid, AIT/main namida | `abdata/chara/oo_base.unity3d`, head bundle | existing exports | `body_shader_programs_v1`, `shader_programs_v2` |
| bridge | HS2Mod `plugins/HS2_McpBridge/MakerRenderStyleService.cs` | HEAD `b6d170a` (loop identical in working tree) | source |

`GraphicsSettings.dll` (`5b54c802…`) was hashed. It is not part of these generators.

## 2 Recovered mechanisms

### 2.1 Mono float arithmetic (all plugins)

The plugin code runs on Mono. Its `float` expressions are evaluated in double precision and
rounded to binary32 at each store or return. The capture establishes this; it was not assumed:

- Under this rule, the CTAA jitter matrices of all 16 iterations and `Mathf.Lerp` →
  `_AEMAXCTAA` match byte-for-byte.
- Pure binary32 evaluation mismatches 6 of the 16 matrices and `preEnhanceClamp`
  (`0x3d2b3679` vs captured `0x3d2b367a`).

`plugin_common.mono_f32/d/mathf_lerp/mathf_sqrt` implement the rule.

### 2.2 DHH SSS (`DHH.SSS`, `DHH.SSS_convolution`, `DHH.Util`)

**Where it runs.** `SSSManager.Initialize(Camera.main)` adds `SSS` to the Maker main camera.
`OnPreRender` therefore runs inside every bridge render, after CTAA's `OnPreCull`.

**Order of work in `OnPreRender`.** For each render it does the following:

1. Disable the global keyword `SCENE_VIEW`.
2. Compute `m_TextureSize = (float)pixelWidth / Downsampling` and truncate it to an int:
   1086×1448 / 2 → 543×724.
3. `CreateCameras` copies these fields to the lighting camera: allowMSAA, background colour,
   clear flags and culling mask.
4. Profile camera (`ProfilePerObject`):
   - Copy the main camera with `UpdateCameraModes`: far/near, stereoTargetEye, orthographic,
     aspect, `renderingPath = Forward`, orthographicSize, and fieldOfView (or the projection
     for a physical camera).
   - Set depth −254, allowMSAA false, culling mask = `SSS_Layer`, black solid clear.
   - Set `QualitySettings.pixelLightCount = 0` and `shadows = Disable` for this render, then
     restore them.
   - Enable the global keyword `SSS_PROFILES`.
   - `GetProfileRT`: ARGB32 Linear, depth 16, Point, Clamp, `autoGenerateMips = false`.
   - `RenderWithShader(Hidden/SSS_Profile, "RenderType")`.
   - Set the global textures `SSS_ProfileTex` and `SSS_ProfileTexR`.
5. Lighting camera:
   - `UpdateCameraModes`, plus `allowHDR = cam.allowHDR`, depth −846,
     culling mask = `SSS_Layer`, `depthTextureMode = DepthNormals`.
   - `GetRT` for `LightingTexture` and `SSSLightingTextureBlurred`: depth 24; ARGBFloat with
     Default read/write when HDR, otherwise Default format with Linear; Bilinear, Clamp.
     MSAA only when `cam.allowMSAA && QualitySettings.antiAliasing > 0`.
   - `RenderWithShader(Hidden/LightingPass, "RenderType")`.
6. Set the global textures `LightingTexBlurred/LightingTex`, plus the `R` aliases in mono.

**Blur material properties** (set only once `_BlurMaterial` exists):

| property | value |
|---|---|
| `DepthTest` | `Max(1e-5, DepthTest/20)` |
| `maxDistance` | `Max(0, int)`, uploaded as int32 |
| `NormalTest`, `ProfileColorTest`, `ProfileRadiusTest` | `Max(0.001, value)` |
| `EdgeOffset` | as set |
| `_SSS_NUM_SAMPLES` | `ShaderIterations + 1`, int32 |
| `sssColor` | as set |
| `DitherScale`, `DitherIntensity`, `NoiseTexture` | when `Dither` is on |

Keywords:

| keyword | enabled when |
|---|---|
| `RANDOMIZED_ROTATION` | `Dither` |
| `PROFILE_TEST` | `UseProfileTest && ProfilePerObject` |
| `DEBUG_DISTANCE` | `DEBUG_DISTANCE` |
| `OFFSET_EDGE_TEST` | `FixPixelLeaks` |
| `DITHER_EDGE_TEST` | `DitherEdgeTest` |

The executed variant adds the global keyword `SSS_PROFILES`.

**Convolution** (`[ImageEffectOpaque]` on the lighting camera):

- Radius: `BlurRadius = ScatteringRadius/Downsampling`, then scaled in place by
  `initFOV/fov` and by `sqrt(h²+w²)/2202.9072f`. At 543×724 this gives `0x3e525719`. The
  normal 1600×900 frame gives 0.416666657, which matches the audited field.
- Temporaries: `T1`, and `T2` when there is more than one iteration. Both are `rtFormat`,
  Linear, depth 0.
- Pass sequence: vertical `source→T1`; then (iterations−1) × [horizontal `T1→T2`, vertical
  `T2→T1`]; then horizontal `T1→blurred`. `Blit(source, destination)` follows.
- `iterations = 0` copies `source→blurred`.
- With 2 iterations this gives the 4 captured blur passes.

**Lighting camera view.** It is a child of the main camera at local identity, and its
projection comes from fov/aspect/clip. Its `unity_MatrixVP` is byte-equal to the main
camera's `_NonJitteredVP` in all 16 iterations. Visible effect: the SSS lighting/blur
buffers are unjittered and at half resolution, while the main camera that samples them is
jittered.

### 2.3 CTAA (`AI_stuff.BetterAA`, `CTAA_PC`)

**Setup.**

- BetterAA adds `CTAA_PC` to every camera with `PostProcessLayer.OnEnable` (Harmony postfix)
  and to `Camera.main`.
- Config `BepInEx/config/BetterAA.cfg` is copied in by `UpdateSettings`: TemporalStability 3,
  HdrResponse 0.001, AdaptiveEnhance 0.32, TemporalJitterScale 0.25, MicroShimmerReduction 3,
  StaticStabilityPower 0.5.
- `OnEnable` sets `depthTextureMode |= DepthNormals|Depth|MotionVectors`.

**`OnPreCull` → `jitterCam`.**

1. `ResetWorldToCameraMatrix`, then `ResetProjectionMatrix`.
2. `nonJitteredProjectionMatrix = projection`.
3. `m02 += (x_jit[i]·jitterScale·2 − 1)/pixelRect.width`, and the same for `m12` with
   `y_jit`/height.
4. `frameCounter = (i+1) % 16`.

The tables are the fixed 16-entry `x_jit/y_jit`. The `2x−1` bias is not centred. Visible
effect: every jittered render, and so the accumulated image, is shifted by
(x·0.5−1)/2 px ∈ [−0.72, −0.26] px horizontally, with the corresponding vertical shift,
relative to an unjittered render.

**`OnRenderImage`.**

1. `SetCTAA_Parameters` (`Lerp` as in 2.1).
2. The four persistent RTs (`new RenderTexture(w,h,0,source.format)`, Clamp,
   `afterPreEnhace` takes the source filter) are recreated when the size changes. 3200×1800 →
   1086×1448 happens in iteration 0.
3. `CTAA_Enhance_PC` pass 1 runs with `_AEXCTAA = 1/Screen.width` and
   `_AEYCTAA = 1/Screen.height`, which come from the window (3200×1800), not the source.
4. If `firstFrame`: blit `afterPreEnhace → rtAccum0`.
5. `CTAA_PC` (pass −1) runs with `_AdaptiveResolve = 3000`,
   `_ControlParams = (StaticStabilityPower, TemporalStability, HdrResponse, MicroShimmerReduction)`
   and `_Accum = rtAccum0` (swap) or `rtAccum1`. It writes the other accumulator.
6. Copy to `txaaOut`, blit `txaaOut → destination`, then `swap = !swap`.

When disabled or orthographic it sets `jitterScale = 0` and blits `source → destination`.

### 2.4 GrabPass and AIT/Liquid (UnityPlayer 2018.4.11f1)

**Shaders.**

- `AIT/Liquid`: pass 0 is an *unnamed* `GrabPass {}`, pass 1 is ForwardBase (queue
  Transparent−1, keywords `DIRECTIONAL LIGHTPROBE_SH _ALPHAPREMULTIPLY_ON`), pass 2 is
  ForwardAdd.
- `AIT/main namida` (queue Overlay+1) also starts with an unnamed GrabPass.

**`GrabPasses::Add`** (0x859740, called from `DoForwardShaderRenderLoop`):

- Builds a per-loop map by texture name. No name means `_GrabTexture`, flagged unnamed.
- New entries get an RT from 0x85b500. The descriptor (0x85ad60) is the active colour target's
  width/height/format, depth 0, named `"GrabPass Temp"`, Bilinear.
- The RT is published as a global texture with `_TexelSize/_ST/_HDR`.

**`ApplyGrabPass`** (0x859da0, from `Pass::ApplyPass`):

- An unnamed grab copies every time.
- A named grab copies only if it has not been grabbed yet.
- `ClearGrabPassFrameState` runs in `CleanupAfterRenderLoop`, so a named grab happens once per
  camera render loop.

**The copy.** `GrabPixels` → `GrabIntoRenderTexture` is a full-rect `CopySubresourceRegion`
(capture events 1860 and 1891).

**Placement.**

- With grab passes present, `StartRenderJobs` uses a single job.
- `BatchRenderer` defers a ForwardBase draw (`Add`) until a later flush. Neither
  `ApplyShaderPass` nor `ApplyGrabPass` flushes.
- So an object's grab executes **before the pending draw of the preceding object**. The
  capture's order is grab, Liquid, grab, eyelashes, namida, particles.
- Visible effect: the tears refract a colour buffer that does not yet contain the eyelashes.

### 2.5 Unity temporary RT pool (UnityPlayer)

- `ReleaseTempBuffer` (0x75a610) files a released RT under its **current**
  `RenderTextureDesc` and places it at the head of that key's list. `GetTemporary` therefore
  returns the most recently released match (LIFO).
- DHH sets `autoGenerateMips = false` after `GetProfileRT`. This changes the descriptor, so
  every render misses the pool and `SSS_ProfileTex` is a new texture in each of the 16
  iterations (captured 59344, 59689, …).
- `LightingTexture`, the blur temporaries, `GrabPass Temp` and the CTAA RTs keep one resource
  each. All of these are reproduced by the allocator simulation.

### 2.6 Bridge capture loop (`MakerRenderStyleService.Capture`, route `/maker/renderstyle/capture`)

**Setup.**

1. Validate the request.
2. Save camera, focus and transform state.
3. `PoseFreeze` (default on): every Animator gets `speed = 0`, `Play(hash, 0, 0)`,
   `Update(0)`, and blinking is turned off (`ChangeEyesBlinkFlag(false)`).
4. `new RenderTexture(w,h,24,ARGB32)` (sRGB view in Linear space).
5. Camera placement: `distance = size/Mathf.Tan(fov·Deg2Rad·0.5)`, position
   `target + Euler(0,yaw,0)·(0,0,d)`, then `LookAt`.
6. `aspect = (float)w/h`, culling mask = the OR of the character renderer layers, grey solid
   clear, DHH DoF `focusDistance = distance`.

**Loop.** `ResetTemporal` sets `firstFrame = swap = true` and `frameCounter = 0`. Then
`Camera.Render()` runs **16** times when an enabled `CTAA_PC` exists, because `jitterCam`
cycles through 16 entries; otherwise it runs once.

**Readback and restore.**

1. `ReadPixels`, then PNG.
2. Restore everything, setting `projectionMatrix`, `nonJitteredProjectionMatrix` and
   `worldToCameraMatrix` explicitly.
3. `ResetTemporal` again.

**Per-render callback order** (component order of `Camera.main`):

1. `OnPreCull`: PPv2, then CTAA jitter.
2. Culling.
3. `OnPreRender`: DHH SSS.
4. Forward passes.
5. The `OnRenderImage` chain:
   - `CharaCustom.CustomRender` (`update = true`) blits `source` into its 3200×1800 `rtCamera`
     (written each iteration, never read in the frame) and into `destination`.
   - `CTAA_PC` writes the capture RT.
   - `SSS_buffers_viewer` runs only when `toggleTexture ≠ None`.
6. `OnPostRender`: SSS re-enables `SCENE_VIEW`.

`/maker/render` (`RenderHead`) is a different route: it uses a fresh camera, renders once
and has no plugins.

## 3 Byte-diff against the retry capture

`validate_plugin_effects.py` checks against `frame_graph_v2` and the existing per-variable
cbuffer extraction (`captured_cbuffers_v1.json`, SHA `306bf22d…`). All checks are exact. Report:
`outputs/offline_renderer_plugins_20261020/validation_v3/report.json`.

| check | count | result |
|---|---|---|
| SeparableSSS `$Globals` (9 variables) and keyword variant | 64 blur draws | byte-equal |
| CTAA_Enhance / CTAA_PC `$Globals` | 32 draws | byte-equal (Enhance bytes are capture-start; the CB was never re-uploaded) |
| Jittered `unity_MatrixVP` regenerated from the same render's `_NonJitteredVP` and generated jitter | 16 iterations, index *k* = segment *k* | byte-equal |
| Plugin RT D3D11 descriptors and RTV formats (profile, lighting, blurred, T1/T2, grab, 4 CTAA RTs, capture RT + depth) | 11 | equal |
| Plugin pass order per segment (SSS block, transparent block, image-effect chain) | 16 | equal |
| CTAA ping-pong write/read and first-frame init (iteration 0 only) | 16 | equal |
| Resource identity across iterations (pool model) | 9 textures | equal pattern |
| GrabPass placement and grab→reader binding | 6 ops | equal |
| Plugin `Graphics.Blit` VS constants: GL.LoadOrtho GPU VP and identity `unity_ObjectToWorld` (via `unity_math`) | 161 blits | byte-equal |
| Unity sampler keys (filter/wrap) of plugin RTs | 8 slots | equal |

Inputs not taken from the capture come from `runtime_audit_hook_v3`, which is **another
frame and process**:

- DHH, CTAA and convolution fields
- fov/near/far
- component order

They are consistent with the capture: every derived byte matches. They are still not a
paired-state proof. Retained attempts:

| attempt | result |
|---|---|
| `validation_try1` | 3 failures: FIFO pool, a ForwardAdd pass with no lights flushed, missing `multiply` for −0 |
| `validation_try2` | passes |
| `validation_v1`, `validation_v2` | earlier generator hashes; same results |

## 4 Required scene state (for a paired capture)

Exact member paths. `SSS` = `Camera.main.GetComponent<DHH.SSS>()`.

| member path | type |
|---|---|
| `SSS.<Enabled>k__BackingField`, `<Dither>k__BackingField`, `DitherEdgeTest`, `FixPixelLeaks`, `ProfilePerObject`, `UseProfileTest`, `DEBUG_DISTANCE` | bool |
| `SSS.<Downsampling>k__BackingField`, `<ScatteringRadius>k__BackingField`, `<DitherIntensity>k__BackingField`, `<DitherScale>k__BackingField`, `DepthTest`, `NormalTest`, `ProfileColorTest`, `ProfileRadiusTest`, `EdgeOffset` | float |
| `SSS.<ScatteringIterations>k__BackingField`, `<ShaderIterations>k__BackingField`, `maxDistance` | int |
| `SSS.sssColor` | UnityEngine.Color |
| `SSS.SSS_Layer` | LayerMask (int m_Mask) |
| `SSS.toggleTexture` | DHH.SSS.ToggleTexture |
| `SSS.NoiseTexture` | Texture2D (pixels + filter/wrap/aniso/mips) |
| `SSS.LightingPassShader`, `SSS.ProfileShader` | Shader |
| `GameObject.Find("SSS profile Camera")`, `GameObject.Find("SSS Lighting Pass Camera")` | GameObject (existence) |
| `…("SSS profile Camera").GetComponent<Camera>().allowHDR` | bool (not set by DHH) |
| `…("SSS Lighting Pass Camera").GetComponent<DHH.SSS_convolution>().initFOV` | float |
| `….SSS_convolution.AllowMSAA` | bool |
| `….SSS_convolution._BlurMaterial` | Material (null or property sheet + keywords) |
| `DHH.DHH_Base.ConfigSSS.Value` | BepInEx ConfigEntry<bool> (true during capture; config file now false) |
| `Camera.main.GetComponent<CTAA_PC>().enabled`, `._enabled`, `.firstFrame`, `.swap`, `.PreEnhanceEnabled` | bool |
| `CTAA_PC.TemporalStability`, `.frameCounter` | int |
| `CTAA_PC.HdrResponse`, `.AdaptiveEnhance`, `.TemporalJitterScale`, `.MicroShimmerReduction`, `.StaticStabilityPower`, `.jitterScale`, `.preEnhanceStrength`, `.preEnhanceClamp`, `.AdaptiveResolve` | float |
| `CTAA_PC.rtAccum0/rtAccum1/txaaOut/afterPreEnhace` | RenderTexture (descriptor; contents only if no reset) |
| `Camera.main.fieldOfView`, `.nearClipPlane`, `.farClipPlane` | float |
| `Camera.main.allowHDR`, `.allowMSAA`, `.usePhysicalProperties`, `.stereoEnabled` | bool |
| `Camera.main.pixelRect` | Rect |
| `Camera.main.transform` (position, rotation) | Vector3 / Quaternion |
| `Camera.main.GetComponents<Behaviour>()` | ordered (type, enabled) |
| `Camera.main.GetComponent<CharaCustom.CustomRender>().update` | bool |
| `CustomRender.<rtCamera>k__BackingField` | RenderTexture |
| `UnityEngine.Screen.width`, `.height` | int |
| `QualitySettings.antiAliasing`, `.pixelLightCount` | int |
| `QualitySettings.shadows` | ShadowQuality |
| `QualitySettings.anisotropicFiltering` | AnisotropicFiltering (DHH sets ForceEnable) |
| `QualitySettings.activeColorSpace` | ColorSpace |
| `DHH_Main.depthOfFieldLayer.focusDistance.value` | float (handed to ppv2) |
| Liquid renderer material (`AIT/Liquid`) | `_Cutoff`, `_ChromaticAberration`, `_WeatheringAll`, `_WeatheringRange1..6` float; `_UVScalePattern`, `_WeatheringUV` Vector4; `_WeatheringMap`, `_WeatheringMask` Texture2D; `shaderKeywords` string[]; `renderQueue` int. Captured values equal the shader defaults. |
| namida material | the same, for `AIT/main namida` |
| Unity temp-RT pool at capture start: for every plugin `GetTemporary` of render 0 | the pooled RenderTexture with a matching desc (instance id, `anisoLevel`), if any |
| bridge request | path string, yaw float, target float[3], ortho_size float, width/height int, freeze_pose bool |

## 5 Interfaces required from sibling modules

`PluginLowering` emits RT resources/views (`texture/view`), GrabPass
`copy_subresource_region` and the packed material constants itself. It calls `forward` for
Unity-internal work, and raises if a method is missing:

| method | Unity work |
|---|---|
| `blit` | `Graphics.Blit`: LoadOrtho quad, BlitCopy, pass state |
| `render_camera` | DHH camera renders, with a callback at the opaque image-effect point |
| `engine_state` | keywords, globals, camera/quality fields |
| `draw_pass` | forward passes of sorted objects |
| `active_color_target` | the bound camera colour target |
| `engine_buffer` | image-effect source/destination, `rtCamera` |

The builder calls are `texture`, `view` and `copy_subresource_region`, with the signatures of
`src/render/frame_program.ProgramBuilder`. A test emits plugin RTs and a GrabPass copy into a real
`ProgramBuilder` and passes its validator. `ProgramRecorder` is an in-memory stand-in with the same
three methods.

## 6 Gaps (explicit)

1. **Sampler anisotropy of pooled temporaries.**
   - What is known:
     - DHH forces `anisotropicFiltering = ForceEnable`.
     - Persistent CTAA RTs (anisoLevel 1) always use the forced 9× sampler.
     - `LightingTexture` (anisoLevel 0) uses the 1× sampler.
     - T1/T2, `GrabPass Temp` and Unity's MV temp use the 9× sampler in iteration 0 and the
       1× sampler from iteration 1 on.
     - `GetTempBuffer` has a `TextureSettings::Reset` path (aniso 0, Bilinear, Clamp; 0x756915).
   - Not recovered: the exact condition that yields 9× in iteration 0.
   - D3D11 sampler creation belongs to the GfxDevice lowering. The keys (filter/wrap) are
     verified; the anisotropy is evidence only.
2. **Unity-internal passes of the DHH cameras are delegated.** These are:
   - the profile camera's HDR intermediate and resolve blit
   - the lighting camera's depth, depth-normals, shadow map and screen-space shadows
   - the opaque image-effect copy-back

   They are observed and listed in the sequence but lowered by `unity_forward`.
3. **GrabPass flush rule.** It is derived from the disassembly structure: no flush in
   `ApplyShaderPass/ApplyGrabPass`, flush in `Add/RenderBatch`. The exact flush call site in
   the forward job (0x6ca590 region) is not fully annotated. The capture order agrees.
4. **Camera transform for `yaw ≠ 0`.** `Quaternion.Euler`/`LookAt` are native, and
   `capture_position` raises for these. The view matrix is `unity_forward`'s.
5. **Branches that raise instead of being lowered:**
   - stereo/VR (DHH and `UpdateCameraModes`)
   - physical-camera projection copy (fields kept, not executed)
   - MSAA GrabPass resolve
   - mipmapped plugin RTs
   - first render after SSS enable (lazy `_BlurMaterial` with default properties)
6. **Inputs are not paired.** Component fields come from another process. The game owner's
   paired capture must supply the fields in section 4.
7. **No pixel acceptance.** No executor and no GPU replay were used. Pixel checks run later
   on the sibling executor.

## 7 Reproduce

```powershell
.venv/Scripts/python.exe -m unittest tools.offline_renderer.test_plugin_effects -v
$O="outputs"
.venv/Scripts/python.exe -m tools.offline_renderer.validate_plugin_effects `
  --frame-graph $O/offline_renderer_frame_graph_20261020/frame_graph_v2 `
  --captured-cbuffers $O/offline_renderer_unity_forward_20261020/captured_cbuffers_v1.json `
  --runtime-fields ../HS2Mod/artifacts/chenger_reference_fit_20261012/runtime_audit_hook_v3/runtime_fields_completion.json `
  --shader-manifest $O/offline_renderer_repair_20261012/sss_shader_programs_v2/manifest.json `
  --shader-manifest $O/offline_renderer_frame_graph_20261020/shader_exports/hs2_betteraa_ctaa/manifest.json `
  --shader-manifest $O/offline_renderer_repair_20261012/body_shader_programs_v1/manifest.json `
  --shader-manifest $O/offline_renderer_repair_20261012/shader_programs_v2/manifest.json `
  --out $O/offline_renderer_plugins_20261020/FRESH --receipt tools/offline_renderer/plugin_effects_validation_20261020.json
ilspycmd -p -o $O/offline_renderer_plugins_20261020/decompiled/DHH `
  -r E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed -r E:/HoneySelect2_ArcticFox/BepInEx/core `
  E:/HoneySelect2_ArcticFox/BepInEx/Plugins/HS2_Plugins/DHH_HS2.dll
```

`captured_cbuffers_v1.json` is produced by `compare_generated_cbuffers.py extract`. That
script belongs to the unity_forward sibling.
