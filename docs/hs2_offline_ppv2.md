# HS2 offline renderer — PostProcessing Stack v2 generator (2026-10-20)

Component of the reimplementation route ([architecture](hs2_offline_renderer_architecture.md),
[frame graph](hs2_offline_frame_graph.md)). It covers PPv2 work that runs for each
capture-camera render: MSVO ambient occlusion, CopyStd, depth of field, the Lut3DBaker LUT
bake, Uber, SMAA and FinalPass. The generator emits frame-program commands through
`src/render/frame_program.py`.

No game was launched. Two inputs were used:

- the existing retry RDC;
- the coordinator's paired capture (`paired_capture_session_20261020`, nonce 04d1e76f,
  marker frame 26060) together with its probe scene state.

Bulky outputs are in the ignored `outputs/offline_renderer_ppv2_20261020/`.

## 1 Sources (the oracle)

| what | identity |
|---|---|
| `HoneySelect2_Data/Managed/Unity.Postprocessing.Runtime.dll` | SHA-256 `f18eac42…a35`. Decompiled with ilspycmd 8.2 to `outputs/…/decompiled/ppv2` (ignored). |
| package version | **2.1.7**. Its identifier set matches tag `2.1.7` (commit `283c8187`) for every file. Branch 2.1.8 adds `m_OldResources` and dynamic resolution, which the installed DLL lacks. The upstream HLSL is in `outputs/…/upstream/` and was used only to read shader intent. |
| installed shaders | `sharedassets2.assets` (SHA-256 `2ec301cb…`): Uber, CopyStd, DepthOfField, SMAA, FinalPass and MultiScaleVO. The DXBC and Unity binding layouts come from the existing exports. |
| compute shaders | Exported from the same file by `tools/offline_renderer/extract_compute_shaders.py`. Each kernel gets DXBC, Unity cbuffer layouts, bind points and `dcl_thread_group`. |
| HS2 configuration | The CharaCustom scene `level3` holds two global volumes on layer 8: `PostProcessVolume2D` (profile "Custom Camera Profile") and `PostProcessVolume3D` (profile "Main Camera Profile Custom", under `Map3D`). `CameraEffector.ConfigEffectorVolume` sets their Bloom, AO, SSR and Vignette `active` flags from `UserData/config/system.xml`. The Main Camera `PostProcessLayer` uses volumeLayer 256, SMAA High and stopNaN. Extracted by `extract_ppv2_scene_state.py`, which decodes MonoBehaviours whose type trees were stripped; every object is consumed to its exact length. |
| DHH (`DHH_HS2.dll`) | SHA-256 `44137712…`. `DHH_Main.Init` adds the global `DHH_Post` volume (layer 8, priority 1000, weight = `GraphicFloat[45]`). `ApplyGraphicSetting` sets the effect overrides from `DHH_Data/GraphicSetting.dhs`, parsed with a transcription of `GetDataWithTag`. It also sets SMAA High, `allowHDR`, `anisotropicFiltering = ForceEnable`, and ACES HDR grading. |
| UnityPlayer.dll | 2018.4.11f1, PDB-matched. `AnimationCurveTpl<float>::Evaluate`, `FindIndexForSampling` and `CalculateCacheData` were transcribed from disassembly (`outputs/…/disasm/`). |

## 2 Recovered chain, in order per camera render

All of this happens in `PostProcessLayer.OnPreCull → BuildCommandBuffers`. Each step is
reproduced in `PostProcessLayerGen.on_pre_cull` and the `Executor`.

1. **Immediate work.**
   - `Shader.SetGlobalFloat(_RenderViewportScaleFactor, 1)`.
   - Volume blending: `ReplaceData` from the base defaults, then `OverrideSettings` per
     volume in stable priority order. The order is 2D, then 3D (both priority 0, weight 1),
     then DHH_Post (priority 1000, weight 0.8914894).
   - The ColorGrading curves texture is uploaded with `Texture2D.Apply`: `map_write` of the
     128×2 RGBAHalf staging texture, then `copy_subresource_region`.
2. **`BeforeImageEffectsOpaque`: `MultiScaleVO.RenderAfterOpaque`.**
   - In forward rendering with no resolved depth, a depth copy runs (MultiScaleVO pass 0,
     into an RFloat temporary).
   - Downsample1 dispatch, at `(W/16, H/16)`.
   - Downsample2 dispatch, at `(W/64, H/64)`.
   - Four `MultiScaleVORender_interleaved` dispatches at L3–L6 ×16 slices, with thread groups
     of 8×8×1.
   - Three `MultiScaleVOUpSample_blendout` dispatches, then one `MultiScaleVOUpSample` into
     the persistent R8 `m_AmbientOnlyAO`. Upsample groups are `(w+17)/16`.
   - The composite (pass 2: Zero / OneMinusSrcColor) draws into CameraTarget with the
     camera depth bound.
3. **`BeforeImageEffects`.**
   - `BuiltinBlit(CameraTarget → TargetPool0)` with CopyStd pass 1 (stopNaN). This draws
     Unity's built-in blit quad.
   - `RenderBuiltins`:
     - DoF: CoC (pass 0, R8 Linear), prefilter (pass 2, half resolution, ARGBHalf sRGB),
       bokeh (pass 3 + kernelSize), postfilter (pass 7), combine (pass 8).
     - ColorGrading in HDR mode with compute: `Lut3DBaker` `KGenLut3D_AcesTonemap` at 9×9×9
       into the 33³ "Color Grading Log Lut".
     - Uber with `COLOR_GRADING_HDR_3D`.
   - `RenderFinalPass`:
     - SMAA: a clear plus edge detection (pass 2 = High), blend weights (pass 5) and
       neighbourhood blending (pass 6).
     - `Dithering.Render`.
     - FinalPass into CameraTarget.

Not active in HS2 Maker: Bloom, Vignette, Grain, ChromaticAberration, LensDistortion,
AutoExposure, MotionBlur and SSR. SSR is enabled by DHH but is unsupported in forward
rendering. If any of these becomes active, the generator raises `PPv2Unimplemented`; it
does not skip them.

## 3 Mechanisms

### Numerics

PPv2's C# runs under Mono without float32 mode. Expressions are evaluated in double
precision and rounded to float32 at every store (`ppv2_unity.r4`/`dv`).

The capture settles this. MSVO `gInvThicknessTable` equals the double-intermediate result
bit for bit, while pure float32 evaluation is 1 ulp off in 11 of 12 entries.

`Mathf.*` functions are `(float)Math.*(double)`. Native Unity code (AnimationCurve, texel
sizes) uses SSE float32.

### Descriptors

| RenderTextureFormat | DXGI resource / view | rule |
|---|---|---|
| `DefaultHDR`, `ARGBHalf` | `R16G16B16A16_TYPELESS` / `R16G16B16A16_FLOAT` | `DefaultHDR` resolves to `ARGBHalf` |
| `RFloat` | `R32_TYPELESS` / `R32_FLOAT` | |
| `R8` | `R8_TYPELESS` / `R8_UNORM` | |
| `RHalf` with `enableRandomWrite` | `R32_TYPELESS` / `R32_FLOAT` | observed promotion |

Bind flags are SRV|RTV, plus UAV for random-write textures. 2D arrays use `ArraySize -1`
views, and 3D textures use `WSize -1` UAVs.

The SMAA LUTs and blue noise come from asset data, with Unity's RGB24→RGBA32 expansion.
All 19 PPv2 resource textures are byte-identical to the capture.

### Samplers

Samplers come from the texture object's filter, wrap and anisoLevel, combined with
`QualitySettings.anisotropicFiltering`:

- With ForceEnable, an anisoLevel of 1–8 is raised to 9.
- An anisoLevel of 0 stays off.
- Point filtering never uses anisotropy.

### Pass states

Pass states come from ShaderLab pass state:

- Blend uses Unity → D3D factor mapping.
- `ZTest Always` maps to D3D `ALWAYS`, with mirrored comparisons under reversed Z.
- `Cull Off` maps to `NONE`.

### Constant buffers

Unity keeps one shadow per `($Globals, layout size)`; examples are `ConstantBuffer-186-464`,
which is shared with `AIT/Skin True Face`, and `…-512`, which is shared by the DoF passes. A
draw writes only the variables of its VS and PS program layouts, then uploads the whole
shadow. Bytes outside those layouts keep earlier contents; this is residue, and no PPv2
program reads it.

Compute cbuffers are one shadow per ComputeShader. Vector parameters write `colCount`
floats; for example, `AdditionalParams` in MultiScaleVORender is a float2.

### Property resolution

Each variable is resolved in this order: the PropertySheet's MaterialPropertyBlock, then
the material's own values (`_MainTex`/`_MainTex_ST` set by `Blit`), then globals.

- Temporary RTs are global textures under their name ID.
- `<tex>_TexelSize` is derived from the bound texture.

### Blit mechanics

**`BlitFullscreenTriangle`:**

1. `SetGlobalTexture(_MainTex, src)`.
2. Set the render target. CameraTarget also binds the camera depth.
3. Clear only if requested; only the SMAA edge pass requests it.
4. `DrawMesh` of `RuntimeUtilities.fullscreenTriangle`: VB of (−1,−1,0), (−1,3,0), (3,−1,0);
   IB of u16 0,1,2; `DrawIndexed(3)`.
5. The viewport is the full target.

**`BuiltinBlit`** uses Unity's quad: POSITION+TEXCOORD0 at stride 20, with vertices
(0,0,0.1 | 0,0)…, indices 0,1,2,0,2,3, and `DrawIndexed(6)`.

### Property values (examples)

- MSVO `ZBufferParams` = (far/near − 1, 1, 0, 0).
- `tanHalfFov` = 1/`projectionMatrix[0,0]`.
- DoF `_Distance` = max(focus, f).
- `_LensCoeff` = f²/(A·(d−f)·0.024·H/1080·2).
- `_MaxCoC` = min(0.05, (k·4+6)/H).
- LUT: `ComputeColorBalance`, mixers/100, Lift·0.2, Gain·0.8, Gamma·0.8.
- Uber: `_Lut3D_Params` = (1/33, 32) and `_PostExposure` = Exp2(postExposure).
- `_UVTransform` = (1,−1,0,1).

### Time-dependent values

**Dithering.** `Dithering.m_NoiseTextureIndex` increments once per FinalPass. The index
selects `blueNoise64[idx]` for `_DitheringTex`. `_Dithering_Coords` =
(W/64, H/64, z, w), where z and w are two `System.Random(1234).NextDouble()` draws.
`ppv2_unity.SystemRandom` reimplements the Knuth subtractive generator.

**Spline caching.** `Spline.Cache` is keyed on `Time.renderedFrameCount`. It only decides
whether a spline is re-evaluated; it does not change values.

**Other values.**
- No grain, TAA jitter or history is active. CTAA jitter belongs to the CTAA sibling and
  only reaches PPv2 through the projection matrix.
- DoF has no CoC history without PPv2 TAA.

### Curves texture

- **Row 0:** hue-vs-hue, hue-vs-sat, sat-vs-sat and lum-vs-sat.
- **Row 1 (HDR mode):** not rewritten. It keeps the persistent `m_Pixels` from the last LDR
  render, which happened before DHH switched to HDR.

Native AnimationCurve evaluation reproduces all 1536 serialized `cachedData` samples
bit-exactly.

## 4 Validation

**Retry capture, all 16 bridge renders.** `ppv2_diff.py` runs **10,032 checks on 416 actions,
with 0 mismatches.** Diff: `outputs/…/diff_retry_v6/diff.json`. The checks cover:

- VS, PS and CS DXBC SHA;
- the bytes of every program-layout cbuffer variable;
- the full declared cbuffer range, with seeded residue;
- cbuffer sizes;
- resource and view descriptors of every SRV, RTV, DSV and UAV;
- sampler descriptors;
- blend, depth and raster states;
- viewports, sample mask and stencil reference;
- input layout, executed index values and vertex bytes;
- dispatch group counts;
- the clear colour;
- curves-upload bytes.

Inputs for the retry diff came from:

- `GraphicSetting.dhs`;
- the bridge request (focus distance = `ortho_size`/`tan(23°/2)`, giving 7.0284328);
- dithering prior calls (14193), solved from the captured (z, w);
- `_ZBufferParams`, taken from the capture as an external input.

**Paired capture (frame 26060, 3200×1800).** `ppv2_diff_paired.py` runs **627 checks on 26
actions, with 0 mismatches.** It uses only scene-state inputs:

- volumes and profiles as reflected live;
- layer config;
- Dithering state and `m_Pixels` from snapshot N−1, after a generated warm-up of frame N−1;
- camera, QualitySettings and RenderSettings;
- the `_ZBufferParams` global uniform.

Further results on the paired capture:

- The generated blend equals the layer's reflected bundle settings in all 113 parameters,
  bit for bit, in both frames. Splines are compared only on loop and zero value.
- The persistent state after each frame equals the next snapshot: the Dithering index and
  the full `System.Random` seed array, and all 256 `m_Pixels` entries.

Diff: `outputs/…/paired_04d1e76f/diff_v2/diff.json`.

**GPU acceptance.** `ppv2_execute_paired.py` executed the generated program with
`frame_executor.exe` (`496477cb…`) on the capture adapter, an RTX 4080 Laptop. Inputs were
initialised from the official replay read-back of the paired RDC:

- the camera target before the MSVO composite;
- the camera target before CopyStd;
- `_CameraDepthTexture`.

**All 16 checkpoints are bit-identical** to the official RenderDoc 1.46 replay: depth copy,
LinearDepth, the AO result, the composite, CopyStd, CoC, DoF prefilter, bokeh, postfilter and
combine, the 33³ LUT, Uber, SMAA edges, weights and blend, and the final camera target.
Receipt: `outputs/…/paired_04d1e76f/gpu_v1/acceptance.json`.

In this frame the DoF combine output equals its input; that equality is in the replay too.

## 5 Scene-state fields still needed from the game

The paired probe already supplies everything else. These are the remaining exact members:

1. `UnityEngine.Behaviour.enabled` (bool) of the three `PostProcessVolume` components. The
   probe reflects only `activeInHierarchy`.
2. `PostProcessManager.instance.m_SortedVolumes[256]`
   (`Dictionary<int, List<PostProcessVolume>>`) and `m_Volumes`: the registration order of
   the two priority-0 volumes. [2D, 3D] is currently selected by the DoF and LUT constants.
3. `Spline.curve.keys` (`Keyframe[]`: time, value, inSlope, outSlope, weightedMode,
   inWeight, outWeight) and `AnimationCurve.preWrapMode`/`postWrapMode` of every
   `SplineParameter` in the three instantiated profiles. Also `Spline.cachedData`
   (`float[128]`) of the layer bundle ColorGrading splines. The probe stops at depth limit;
   keys are currently taken from the profile assets.
4. The temporary RenderTexture pool at frame start (UnityPlayer `RenderBufferManager`
   free list). For each entry: `RenderTextureDescriptor`, `anisoLevel`, `filterMode`,
   `wrapMode` and `IsCreated`. Also the RenderTexture object behind CameraTarget
   ("TempBuffer 100 3200x1800"), which is not in the probe texture list.
5. `MultiScaleVO.m_AmbientOnlyAO` (RenderTexture) `anisoLevel`, `filterMode` and `wrapMode`.
   This is acceptance only; it is currently the RenderTexture default (Point/Clamp).

## 6 Gaps (explicit)

- **`rt_random_write_format_fallback`.** `RHalf` with random write becomes `R32`. This is
  observed; the UnityPlayer `GraphicsCaps::IsFormatSupported(LoadStore)` path is not
  transcribed.
- **`unity_sampler_mapping`.** The TextureSettings → D3D11 sampler rule is observed on every
  captured sampler, not transcribed.
- **`temporary_rt_reuse_aniso`.** A pool-reused RT samples without forced anisotropy, while
  a newly created RT gets anisotropy 9. This was observed in retry segment 0 versus 1–15
  and modelled as anisoLevel 0 on reuse.
- **`temporary_rt_pool_order`.** Which free entry Unity picks is not transcribed. PPv2
  temporaries are fully rewritten, so contents do not depend on it.
- **`raster_front_ccw`.** `FrontCounterClockwise` is false only for the first opaque-buffer
  draw. This is observed; CullMode is NONE.
- **`unset_uniform_policy`.** Layout variables set nowhere are written as 0. This is
  consistent with all captured bytes.
- **`blit_dynamic_ring`.** The built-in Blit quad is emitted as its own buffers. The bytes
  and indices are equal, but Unity's ring offsets are not reproduced.
- **Stale and default state.** The stale scissor rectangle (scissor disabled) is not emitted.
  The blend factor uses the D3D default.
- **Cbuffer residue.** Bytes outside the PPv2 program layouts come from other components'
  draws, so the full-buffer diff seeds them from the capture. The orchestrator should share
  one `UnityConstantBufferPool` across generators.
- **`texture_lerper`.** TextureLerper GPU blends (0 < t < 1 between two textures) are not
  implemented. Local (collider) volumes are not implemented either.
- **Unimplemented effects.** These raise `PPv2Unimplemented`:
  - Bloom, Vignette, Grain, ChromaticAberration, LensDistortion, AutoExposure, MotionBlur
    and SSR;
  - deferred fog, ScalableAO and ambient-only AO;
  - PPv2 TAA and FXAA;
  - LDR, 2D and External grading, and the Custom tonemapper;
  - custom bundles, stereo, MSAA and debug overlays.
- **Shader library adapter.** `hs2pipe/shader_library.py` does not exist yet.
  `ManifestShaderSource` is the adapter. It expects
  `program(shader, subshader, pass, stage, keywords)` and `kernel(compute, kernel)` from the
  sibling module.
- **Probe snapshot timing.** The probe snapshot of frame N is taken after its OnPreCull.
  This is shown by `Spline.frameCount == N` and the Dithering index having already stepped.
  Persistent PPv2 state is therefore taken from snapshot N−1.

## 7 Files and reproduce

- `src/render/hs2pipe/ppv2.py`: generator, executor of the recorded command buffers, shader
  adapter and assets.
- `src/render/hs2pipe/ppv2_settings.py`: effect defaults, ParameterOverride interpolation,
  volume blending, scene and DHH volumes.
- `src/render/hs2pipe/ppv2_unity.py`: Mono numerics, System.Random, RT, sampler and state
  mapping, constant-buffer pool, temporary-RT pool, native AnimationCurve.
- `src/render/hs2pipe/ppv2_probe.py`: generator inputs from the frame-pair probe scene state.
- `tools/offline_renderer/`:
  - `extract_ppv2_scene_state.py`, `extract_compute_shaders.py`, `ppv2_oracle.py`;
  - `ppv2_diff.py` (retry), `ppv2_diff_paired.py`, `ppv2_execute_paired.py`;
  - `test_ppv2.py` (20 tests; the integration tests rerun both byte diffs).

```powershell
.venv/Scripts/python.exe -m unittest tools.offline_renderer.test_ppv2 -v
.venv/Scripts/python.exe -m tools.offline_renderer.extract_ppv2_scene_state --out outputs/offline_renderer_ppv2_20261020/FRESH/characustom_ppv2.json
.venv/Scripts/python.exe -m tools.offline_renderer.extract_compute_shaders --asset E:/HoneySelect2_ArcticFox/HoneySelect2_Data/sharedassets2.assets --out outputs/offline_renderer_ppv2_20261020/FRESH
.venv/Scripts/python.exe -m tools.offline_renderer.ppv2_oracle --xml <zip.xml> --inspect <inspect stdout> --shader-manifest … --boundary-resource 59322 --out FRESH
.venv/Scripts/python.exe tools/offline_renderer/ppv2_diff_paired.py --oracle … --scene-state … --scene …/scene_005.json --prev-scene …/scene_004.json --prev-prev-scene …/scene_003.json --shader-manifest …/sharedassets2/manifest.json --compute-manifest …/compute_shaders_v1/manifest.json --out FRESH
.venv/Scripts/python.exe tools/offline_renderer/ppv2_execute_paired.py … --readback …/oracle_readback_v1/fast --executor outputs/offline_renderer_executor_20261020/build_v3/frame_executor.exe --out FRESH
```

The paired RDC's driver listing came from `renderdoc_replay_msvc.exe inspect`, with the pinned
RenderDoc 1.46 directory on PATH (`outputs/…/paired_04d1e76f/inspect/stdout2.bin`). The
replay read-back used `render_patched_capture.py render` with an identity plan.
