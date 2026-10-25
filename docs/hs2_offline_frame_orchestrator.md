# HS2 offline renderer — whole-frame orchestrator, ladder step 2 (2026-10-21)

Builds one frame program for the paired Maker frame 26059 (`rdc_04d1e76f…`, same-frame scene
export `window_04d1e76f…/frames/scene_005.json`). Every pass with a transcribed rule is
generated from scene state. Every other pass is spliced from the converted capture and labelled
`external:capture`. The program is then diffed per pass against the capture and executed. No game
was run and no git operation was performed.

Bulky outputs are under `E:/HS2_offline_renderer_data/orchestrator_20261020/`. Receipt:
`tools/offline_renderer/frame_orchestrator_paired_20261021.json`.

## 1 Files

| file | role |
|---|---|
| `src/render/hs2pipe/frame.py` | Camera/callback order (transcribed, see §2); `FrameEmitter` (generated draws: shaders, D3D11 states, cbuffers through the shared `UnityConstantBufferPool`; splicing; pool mirroring of spliced work); `capture_cbuffer_names`; `LibraryShaderSource` (PPv2 interface on `shader_library`) |
| `src/render/hs2pipe/unity_d3d11_state.py` | ShaderLab pass state → Gfx*State → D3D11 blend / depth-stencil / raster descriptors (RVAs in the module) |
| `src/render/hs2pipe/forward_values.py` | Built-in and material cbuffer values from scene state, including motion-vector values (`_NonJitteredVP`, `_PreviousVP`, `_PreviousM`, `_ForceNoMotion`) |
| `src/render/hs2pipe/unity_camera_passes.py` | Binds the sibling `unity_pass_sorters` to the export: depth texture, depth-normals and motion-vector plans; legacy shader rename table (78 pairs, `.rdata 0x14b6e90`) |
| `src/render/hs2pipe/external_capture.py` | Capture program index: object import, cbuffer contents per draw, upload history |
| `src/render/hs2pipe/unity_forward.py` | + transcribed reflection-probe default sample (`GenerateDefaultSample` 0x67d5d0, `SetReflections` 0x6c11e0, `GetTextureDecodeValues*` 0x7569d0/0x756ab0); CRT powf/sinf/cosf via `unity_crt.py` |
| `src/render/hs2pipe/unity_forward_plan.py` | + `render_objects()` (shared RO lists) |
| `tools/offline_renderer/frame_orchestrate_paired.py` | Generate, diff per pass, execute generated and oracle programs |
| `tools/offline_renderer/frame_structural_diff.py` | Earlier per-variable diff, now using `forward_values` |

## 2 Callback and pass order (UnityPlayer.dll, transcribed)

The order is:

1. `CustomCull`: OnPreCull at 0x6610a8, then culling.
2. `UpdateVelocity`.
3. `CustomRender` (0x661400):
   1. OnPreRender at 0x66157f.
   2. `UpdateDepthTextures` at 0x6616fa (depth texture first, then depth-normals).
   3. `SetupRender`.
   4. The render loop (0x6617d7). `DoRenderLoop` 0x6b9890 runs:
      1. forward opaque, via `PerformRendering` 0x6ccb30: shadow maps, SetupRender, ClearNoSkybox, CameraEvent 10/11;
      2. `RenderSkybox`;
      3. `DoMotionVectorRenderLoop`;
      4. halos;
      5. `RenderImageFilters(opaque)`: CameraEvent 12, then the [ImageEffectOpaque] chain, then CameraEvent 13;
      6. forward transparent (CameraEvent 16/17).
   5. **OnPostRender** at 0x6617fc. It runs before the OnRenderImage chain.
   6. `DoRenderPostLayers`.
   7. `RenderImageFilters(false)` at 0x6618ef: CameraEvent 18, the OnRenderImage chain, then CameraEvent 19.
   8. The GUI layer.
   9. CameraEvent 20.

The task brief listed OnPostRender after the OnRenderImage chain. The binary has it before.

## 3 Results on frame 26059 (delivered run `frame_v15`)

**Generated passes:** depth texture (14 draws), depth-normals (22), forward opaque (16), motion vectors
(12) and forward transparent (4). That is 68 draws.

**Structural diff (generated objects vs capture):**

| item | equal / 68 |
|---|---|
| VS DXBC | 68 |
| PS DXBC | 68 |
| blend desc | 68 |
| depth-stencil desc | 68 |
| raster desc | 68 |
| stencil ref | 68 |

**Cbuffer bindings:** 416 of 430 are byte-equal, as whole buffers including pool residue. The other
14 differ only in GammaToLinear material colours. Each of those is explained by the export's decimal
printing: a ≤ 8-ulp float neighbour of the printed value converts to the captured bytes exactly.

| pass | draws equal | cbuffers equal |
|---|---|---|
| depth texture | 14/14 | 78/78 |
| depth-normals | 22/22 | 100/100 |
| forward opaque | 9/16 | 155 + 12 colour-precision |
| motion vectors | 12/12 | 48/48 |
| forward transparent | 3/4 | 35 + 2 colour-precision |

**Variable counts:** 1010 cbuffer variables are generated. 409 are `external:capture`:

- clothes and body material properties (no typed export);
- `_WorldSpaceLightPos0` and `unity_LightShadowBias`;
- `_HasLastPositionData` and `_MotionVectorDepthBias`;
- particle `unity_ObjectToWorld`;
- `_Cutoff` / tree uniforms of the depth-normals replacement shader.

**Draw order:**

- Depth texture and depth-normals reproduce the capture order with the transcribed sorters.
- Motion vectors do not. The ties among the head parts and among the clothes fall to the
  RenderObject index, which comes from the scene node order, and that is not exported.

**Execution** (`frame_executor` build_v3, hash read-backs at the end of every capture pass). The
oracle is the verbatim capture program with identical read-backs; build_v3 was accepted bit-exact
against the RenderDoc replay.

- Delivered run: **38/58 checkpoints equal.** The camera colour target differs from the first forward
  pass on, because of the 14 colour values. The motion-vector target differs because of the order.
  Everything downstream of those two follows.
- Diagnostic runs, isolated and labelled, taking those inputs from the capture:
  - captured colours only: 58/58 before motion-vector generation was added (`frame_v8_diag_colours`);
  - captured colours plus captured motion-vector order: **58/58, all 430 cbuffers equal**
    (`frame_v14_diag`).

  So the generated passes are exact except for those two scene-state inputs.

## 4 Not finished (explicit)

- **PPv2 inside the frame**: implemented, opt-in (`--ppv2-scene-state`; `PPv2Run`, `LibraryShaderSource`, shared pool), but **not working yet**: `ppv2._any_kernel` reads `shaders.compute` (manifest-adapter internals), which `LibraryShaderSource` does not provide. The delivered run splices PPv2 from the capture.
- **Not generated; spliced from the capture with labels:**
  - BackCamera deferred reflections;
  - shadow map:
    - order needs sorter gap S1 (shadow-caster pass hash);
    - the light matrix is not exported;
    - the device depth-bias and scissor setters in `RenderShadowMaps` are not transcribed;
  - screen-space shadows;
  - skybox;
  - the motion-vector camera quad;
  - grab copies;
  - clears;
  - CTAA and CustomRender blits (`PluginLowering` forward interface not yet wired);
  - UI (81 draws).
- **Camera targets:** RT allocation, IA (mesh and skin buffers), SRVs, samplers and viewports are `external:capture` per draw.
- **Skinned vertex buffers** for head surfaces are not regenerated through `paired_skin` inside the program; they are external.
- **Bridge renderstyle capture (16 renders):** not built. `capture_loop.plan_capture` exists, but no paired bridge capture with scene state exists.

## 5 Gaps / field requests for the next paired session

1. Material colours and vectors as float32 bits; the decimal printing loses ulps.
2. Typed properties for all shaders, including AIT/Clothes True and body; and the material's
   shader source bundle. Clothes copies differ, and o_tang binds a UnityPerDraw of 160 bytes, a
   layout no installed library copy has.
3. Renderer scene-node / RendererScene index order (the RenderObject index tie-break; it decides the
   motion-vector order). Also Mesh+0x174 ids and the static-batch fields (lowest depth-key bits).
4. SkinnedMeshRenderer+0x255 (skinned last-position flag), BaseRenderer+0xbc (transform update
   frame) and BuildSettings+0xfa (`_MotionVectorDepthBias`).
5. Light `localToWorldMatrix`, for `_WorldSpaceLightPos0`, the shadow matrices and bias.
6. The shadow-caster pass hash (`Pass::ComputePassValuesHash` 0x855f70), or the shadow-caster order.
7. Particle system render matrix and simulation space.
8. ReflectionProbe components including disabled ones. Default-reflection inputs are now derived:
   Custom mode with null cubemap gives HDR decode (1,1,0,0).
9. Earlier list, still open:
   - exact camera `transform.position`;
   - culling-matrix and implicit-projection flags;
   - `_Time`;
   - LightManager order;
   - sorting-layer values;
   - GraphicsCaps+0x404;
   - the PPv2 fields in `hs2_offline_ppv2.md` §5.

## 6 Reproduce

```
P=../HS2Mod/artifacts/chenger_reference_fit_20261012/paired_capture_session_20261020
E=E:/HS2_offline_renderer_data/orchestrator_20261020
# renderdoc.dll of outputs/offline_renderer_repair_20261012/renderdoc_portable_1_46 on PATH
.venv/Scripts/python.exe -m tools.offline_renderer.frame_orchestrate_paired --xml $P/parse_04d1e76f/capture.zip.xml \
  --inspect outputs/offline_renderer_unity_forward_20261020/paired_04d1/inspect/stdout.bin \
  --window $P/window_04d1e76fdad14c11bbc6fe439b228e77 --window-index 5 \
  --library outputs/offline_renderer_shader_library_20261020/library_v1 \
  --capture-program $E/capture_program_04d1 [--reuse-oracle $E/frame_v7] \
  [--diagnostic-capture-colours] [--diagnostic-capture-mv-order] --out $E/FRESH
```

`E:` has no hard-link support, so each run copies the 1.3 GB of capture blobs. The runs `frame_v1`
and `frame_v2` are partial runs that failed; they were left in place because deletion was not
permitted.

## 7 Update 2026-10-24 (steps 1–4 of the resumed batch)

Bulky outputs: `E:/HS2_offline_renderer_data/orchestrator_20261024/`. Superseded runs from
this batch are in `deprecated/` there.

Receipt: `tools/offline_renderer/frame_orchestrator_20261024.json`.

### Step 1 — PPv2 in-frame (frame 26059)

**Changes:**

- `ppv2._any_kernel` asks the shader source for `any_kernel(compute)`.
  `LibraryShaderSource.any_kernel` returns the first kernel of the D3D11 variant from
  `shader_library`. This is valid because ComputeShader parameters are per shader.
- The frame's shared pool is now seeded at frame start with every capture constant buffer's
  pre-frame bytes (`FrameEmitter.seed_frame_start`). This is the cross-frame residue of Unity's
  (name, size) shadow copies, labelled `external:capture`.
- PPv2 uses the same pool. Spliced work mirrors into it.

**Results:**

- Diagnostic run (`step1_diag`, captured colours and captured motion-vector order):
  - **58/58 checkpoints** equal; all PPv2 intermediates are read back from the generated
    temporaries.
  - 430/430 forward-family cbuffers equal.
  - PPv2 actions structurally equal: 23/24 (shaders, cbuffer bytes in the declared ranges, SRV and
    sampler descriptors, states, viewport, targets).
  - The remaining action is CopyStd's `CommandBuffer.Blit` draw arguments: dynamic VB/IB ring
    placement, gap `blit_dynamic_ring`.
- Delivered run (`step1_delivered`): 38/58 checkpoints. Only the two known input gaps remain:
  material colour precision and RenderObject (scene-node) order. Three PPv2 cbuffers inherit the
  colour bytes through the shared `$Globals/464` key, the head material shared with SMAA, which
  confirms the pool sharing.

### Step 2 — forward interface and plugin blits (retry frame 17675, 16 renders)

**`frame.UnityForwardLowering`** implements `blit`, `render_camera`, `engine_state`,
`draw_pass`, `active_color_target` and `engine_buffer`. The blit is UnityPlayer's
`Graphics.Blit`:

| element | value |
|---|---|
| target | destination RT, plus its depth surface |
| viewport | full |
| projection | `GL.LoadOrtho` GPU projection |
| view, world | identity |
| `_MainTex_ST` | (1, 1, 0, 0) |
| geometry | 4-vertex quad |
| states | material pass, through `unity_d3d11_state` |
| samplers | `ppv2_unity.sampler_desc`, aniso 0 for re-pooled temporaries |
| cbuffers | shared pool |

**`tools/offline_renderer/frame_orchestrate_retry.py`** drives `PluginLowering` over the
`validate_plugin_effects` plan:

- SSS blurs and the convolution blit, GrabPass copies, CustomRender and CTAA blits, and every
  plugin RT come from the allocator.
- Spliced capture work that touches a plugin RT is rewritten to the generated texture through a
  per-render role map (`remap_command`).

**Results (`retry_v4`):**

- **177/177 generated blits are structurally equal.**
- 32 GrabPass copies are generated.
- Execution: **1138/1169 checkpoints equal**. Every colour output of every render is equal,
  including the final capture RT.
- The 31 mismatches are depth attachments of plugin RTs that Unity never writes in the frame:
  the SSSLightingTextureBlurred depth, and the SSS_ProfileTex depth from render 1 on. The capture
  converter clears resources created mid-capture; generated RTs start undefined.
- Still spliced:
  - the DHH cameras' object draws (profile / LightingPass replacement renders);
  - the lighting camera's depth, depth-normals, shadow and screen-space shadows;
  - the copy-back.

  The retry capture has no paired scene state for objects. The v5.0.1 mode-b capture now has one.

### Step 3 — head skinning in-frame (frame 26059)

**`hs2pipe.frame_skin`** locates the uploads consumed by the head draws with `gpu_skin_inputs`
(`simulate`, `bind_surfaces`, `upload_points`). It regenerates them from the probe export with
`paired_skin.unity_exact_vertices`, which runs UnityPlayer arithmetic through `unity_skin_emu`:

- 8 current VBs from frame N, emitted as in-frame `map_write`;
- 6 motion-vector previous VBs from frame N−1, emitted as frame-start buffer contents.

**14/14 are byte-identical to the capture.** Execution: 58/58 with the two diagnostics
(`step3_diag`). Per-draw matrices of the generated passes were already generated
(`ForwardValues`). The spliced shadow pass still carries captured matrices.

### Step 4 — shadow map (source part)

- **Device state:**
  - The caster draw loop (0x6a0b00) calls `SetGlobalDepthBias(1, 1)` (0x6a0cc7) and restores
    it at 0x6a1116.
  - `RenderShadowMaps` sets the directional scissor to the split viewport inset by 1 pixel
    (0x6a2b2c, `SetScissorRect` 0x54c7b0).
  - These live in `unity_d3d11_state.shadow_caster_device` and `shadow_scissor_rect`.
  - With them, raster state and scissor are **9/9 equal** on the frame-26059 shadow draws: bias −1,
    slope −1 after reversed Z, scissor (1, 1, 4095, 4095).
- **Order:** the caster key's material field `SharedMaterialData+0xd0` is computed by
  `Material::UpdateHashes` 0x84e3b0 → `Pass::ComputePassValuesHash` 0x855f70. It is XXH32
  (seed 0x8f37154b) over:
  - the `ShaderLab::Pass*` address;
  - the 32-byte keyword set;
  - the per-stage `PrepareOverridingValues` bytes.

  The input contains a heap address, so the order **cannot be computed offline**. Required
  field: `SharedMaterialData+0xd0` per material (or the caster draw order). Light matrices: an
  explicit gap for frame 26059; they are in the v5.0.1 export.
- The shadow pass itself is not yet emitted (still spliced).

## 8 Update 2026-10-24/25 — v5.0.1 mode a (`f8bf0704…`, frame 3605)

Run: `E:/HS2_offline_renderer_data/orchestrator_20261024/modea_f8bf/run_v26` (superseded runs in
`modea_f8bf/deprecated/`). Inputs: v5 scene/renders export (`hs2pipe.scene_v5`), native material
sheets (exact colours), light transform rotation, scene node handles. Numbers: §8.6.

### 8.1 Scene node order (proof, replaces the capture MV order)

- `Renderer+0x160` is the renderer's index in the RendererScene node array:
  `AddRendererInternal` (0x68ae10) appends a SceneNode and returns the previous count;
  `RemoveRenderer` (0x690f80) swap-removes and writes the moved node's new index back to its
  renderer (back pointer +0x38, +0x128 → +0x160, 0x691142).
- Static nodes are the indices below the static count, so list 0 then list 1 in node order is
  ascending index. `cull` keeps node order within a list; `render_node_queue_order`
  (BeginRenderQueueExtraction 0x68bc00) appends main-thread-prepared nodes (SkinnedMeshRenderer
  has no PrepareRenderNodes job callback) after job-prepared ones.
- Probe v5 field `native.scene_handle_0x160`; `SceneAdapter.node_queue`. `flags_0x128` is not
  needed for the order.

### 8.2 Shadows (generated except the caster set/order)

- `unity_shadow_pass.directional_shadow`: light axes = `GetLocalToWorldMatrixNoScale` from the
  world rotation (`SharedLightData::UpdateTransform` 0x6a6610); ShadowCullData, caster/receiver
  MinMaxAABB, CloseFit cascade (0x6a4400) with the focused light hull; shadow map size
  (0x69ad80). Caster `unity_MatrixVP`, `unity_WorldToShadow[0]` bit-equal to the capture.
- Unused cascades: `PrepareShadowMaps` 0x69fecf..0x69ffe1 zeroes the four header matrices, sets
  m[14] = 1 when ShadowMapJobHeader+0x11143 (GraphicsCaps+0xdd) and m[15] = 1.
- `unity_LightShadowBias` normal-bias ×3.65 branch (`RenderShadowMaps` 0x6a29db..0x6a2a60):
  light shadows == Soft (header+0x5b0 = SharedLightData+0xa4, 0x69fba8), GraphicsCaps+0xd0 bit 7,
  tier byte GraphicsSettings[tier]+0xb4, BuildSettings+0x101/+0x102, GraphicsCaps+0x3fc == 0,
  QualitySettings.shadows ≥ 2, shadowDistance > 0 (header fields filled by 0x69f180).
  `unity_forward.soft_normal_bias_branch`. Only the two GraphicsCaps bits are not exported: the
  branch outcome is taken from the capture (one boolean, external:capture).
- Depth-texture pass: the depth-pass render job (0x6c8840) writes `unity_LightShadowBias = 0` on
  the device (0x6c890e..0x6c8923) before its draws.
- Caster draws: `unity_camera_passes.shadow_caster_plan` (GetShadowCasterPassToUse per material,
  `SHADOWS_DEPTH`), `shadow_caster_device`, viewport = map size, scissor inset by 1. Membership
  and order are the explicitly listed external:capture input (`SharedMaterialData+0xd0`).
- Screen-space shadows (`CollectShadows` 0x6c8210 → `ComputeScreenSpaceShadowMap` 0x69b950):
  subshader by `ShadowmapFilter` tag — soft = the same caps/settings chain without the tier byte;
  force-inverse-projection = Camera+0x4ec == 0 (explicit projection: `SetProjectionMatrix`
  0x66bc84 writes 0, `ResetProjectionMatrix` 0x6690f0 writes 1) and projection row 3 ≠ (0,0,0,1);
  `LoadFullScreenOrthoMatrix(-1, 100)`; `SHADOWS_SINGLE_CASCADE`; `_ShadowMapTexture_TexelSize`.

### 8.3 Other generated draws

- Particles: every ParticleSystemRenderer render callback (0xb02e30, 0xb03530, 0xb03c10, 0xb042e0,
  0xb04780, 0xb04c30) calls `SetWorldMatrixAndType(identity, 0)` (GfxDevice vtable +0xd8 =
  0x8cf3b0); vertices are generated in world space.
- Motion-vector camera quad (DoMotionVectorRenderLoop 0x6c9b60): pass `r12+1`, r12 = 1 when
  Camera+0x558 (current target) is null or its antiAliasing (RenderTexture+0xd0) > 1;
  `_NonJitteredVP`/`_PreviousVP` on the internal material; `LoadFullScreenOrthoMatrix(0,
  GetProjectionFar)` (0x68fc70: SetOrtho → SetProjectionMatrix, SetViewMatrix(identity));
  `GetProjectionFar` (0x663750) transcribed; mono `RenderFullScreenQuad` sets no matrices, so the
  world matrix is the device state left by the previous generated draw (the skybox restores its
  own through `DeviceMVPMatricesState` 0x68a1d0).
- OnRenderImage chain of Main Camera (component order: CustomRender 6, CTAA_PC 7):
  `capture_loop` CustomRender ops + `ctaa.on_render_image` from the v5 CTAA fields, lowered
  through `PluginLowering` / `UnityForwardLowering` with `ChainDriver`. Graphics.Blit to the
  backbuffer (camera targetTexture null) does not invert the projection; a code-created blit
  material starts from the shader default sheet (`Material::BuildProperties` 0x849050 →
  `ShaderPropertySheet::CopyFrom` 0x8490bc), so BlitCopy `_Color` = (1,1,1,1).
- GrabPass copies: `hs2pipe.liquid` rule (whole active colour target; the grab runs before the
  pending ForwardBase draw of the preceding object).
- Clears: `GfxDeviceD3D11Base::Clear` (0x5445b0) translation; depth texture (0x6674bb),
  depth-normals replacement pass (0x668146, colour (0.5,0.5,1,1)), directional shadow map
  (0x6a2847), screen-space shadow map (0x69bbaa, flags 1, colour (1,0,1,0)), main camera
  (`Camera::ClearNoSkybox` 0x65fac0 → 0x65f8e0: Skybox with material → 6 + GraphicsCaps+0x3a1).

### 8.4 Results (run_v26, execution with a fresh oracle)

| pass | draws equal | cbuffers equal |
|---|---|---|
| depth_texture | 14/14 | 78/78 |
| depth_normals | 22/22 | 100/100 |
| shadow_caster | 9/9 | 48/48 |
| screenspace_shadows | 1/1 | 7/7 |
| forward_opaque | 16/16 | 167/167 |
| skybox | 1/1 | 4/4 |
| motion_vectors_camera | 1/1 | 6/6 |
| motion_vectors | 12/12 | 48/48 |
| forward_transparent | 4/4 | 37/37 |
| image_effect_chain (CustomRender + CTAA) | 6/6 | 22/22 |

86/86 draws, 517/517 cbuffers; grab copies 2/2; clears 6/6; head skin uploads 14/14; PPv2
actions 23/24 (CopyStd blit ring args); **execution 60/60 checkpoints equal**. Generated cbuffer
variables 1533, external 24 (`_HasLastPositionData` 12, `_MotionVectorDepthBias` 12).

### 8.5 Remaining external:capture

- Listed: shadow caster membership and order.
- One-boolean caps inputs: GraphicsCaps+0xd0 bit 7 / +0x3fc (normal-bias and PCF_SOFT branch),
  GraphicsCaps+0x3a1 (Skybox clear colour bit).
- `_HasLastPositionData` (SkinnedMeshRenderer+0x255), `_MotionVectorDepthBias` (BuildSettings+0xfa).
- Per generated draw: IA (mesh/skin/immediate-quad buffers), SRVs, samplers (engine textures),
  render-target allocation; chain/grab/engine texture identities and cross-frame contents.
- Graphics.Blit / PPv2 CopyStd dynamic VB ring placement.
- Not generated (spliced): BackCamera deferred (5 clears, depth copy, reflections), curves copy,
  PPv2-internal clear (event 1522), pre-UI stencil clear (event 1691), UI (81 draws), staging copy.
- Assumed and to be confirmed by the probe: GfxDevice+0x21e8 == 0 (no single-pass stereo);
  Camera+0x4ec inferred from the projection comparison; Camera+0x558 = the camera intermediate
  RT with antiAliasing from QualitySettings (creation not transcribed).

### 8.6 v6 field requests

SharedMaterialData+0xd0 per material (or the caster queue order) and the caster cull set;
SkinnedMeshRenderer+0x255; BuildSettings+0xfa; GraphicsCaps+0xd0, +0x3fc, +0x3a1, +0x404;
Camera+0x4ec, Camera+0x558 (+ its RenderTexture+0xd0); GfxDevice+0x21e8; per engine RT
(camera intermediates, motion-vector texture, GrabPass Temp) the RenderTexture descriptor and
filter/wrap/aniso for sampler generation.
