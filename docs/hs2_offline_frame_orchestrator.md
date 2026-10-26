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
  `GetProjectionFar` (0x663750) transcribed; mono `RenderFullScreenQuad` sets no matrices.
  Corrected 2026-10-25 (§13.2 item 10): `GfxDevice::SetViewMatrix` 0x8cf290 resets the world
  matrix (GfxDevice+0x2110) to identity, so the quad draws with an identity world matrix. The
  earlier "device state left by the previous generated draw" rule matched only because that
  draw (BG dust) was itself identity.
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

## 9 Update 2026-10-25 — bridge 16-render frame, v5.0.1 mode b (`5af92a08…`, frame 8750)

Bulky outputs: `E:/HS2_offline_renderer_data/orchestrator_20261024/modeb_5af9/`
(`inspect/`, `capture_program/` through event 27734, `frame_graph/` segmented at the capture
target 31690, `resource_acceptance/`, `resource_wiring/`, runs `run_v6` (oracle) and `run_v7`).

### 9.1 Structure

`frame_orchestrate_paired` is now job based: one job per camera render.

- **Mode a** is one job.
- **Mode b** has 16 jobs: the frame-graph segments that end with a write of the bridge
  capture target. Each job pairs with the Main Camera `pre_render` record of the same bridge
  ordinal.
- **Per-render scene:** `scene_v5.normalize(..., render_index)` overlays the camera
  properties of that render, which include the native `implicit_projection_flag_0x4ec`. It also
  takes keywords, sheets, lights (camera-attached) and renderer state (bounds, root bones) from
  that render.
- **Shared across jobs:**
  - one PostProcessLayer generator (`PPv2Run.add_job`), so temporal and pool state carry over
    between renders;
  - one `PluginLowering` / `ChainDriver`, so the CTAA history handles persist.
- CTAA state = the scene-export fields, then `ResetTemporal` before render 0. History textures
  are recreated in render 0 at 1086×1448; these allocations are generated.
- The segment-16 ordinary render, BackCamera and UI are spliced (outside the bridge target).

### 9.2 Bridge-specific rules

- **Renderer bounds and skinned matrices** come from the render record: TRS of the
  pose-frozen root bone (the `paired_skin.unity_object_matrices` rule). The end-of-frame export
  differs because the bridge's pose freeze moves the bones between the frame export and the
  renders.
- **DOF focus distance:** `MakerRenderStyleService.Capture` sets
  `DHH_Main.depthOfFieldLayer.focusDistance` = `capture_distance(ortho_size, fov)` (the
  `ppv2_parameter` setup op of `capture_loop.plan_capture`). Applied per bridge render to the
  DepthOfField settings object of the volume profile.
- **Depth-pass `_WorldSpaceLightPos0`:** device state of the previous render. The depth job
  0x6c8840 writes only `unity_LightShadowBias`; the forward setup (0x680be5) and the
  RenderShadowMaps prologue (0x6a1c4c) write the light position. Render 0 therefore sees frame
  N−1's light.
- **`_AEXCTAA` / `_AEYCTAA`** use `Screen.width/height` (3200×1800), not the RT size.
- **Skin streams** (session pairing, not traced):
  - the bridge reads capture-start slot 0 = frame N−2 skin and slot 1 = N−1;
  - the frame's own skinning after the bridge writes slot 0 = N.

  `bridge_skin_frame_rule` / `frame_skin.head_skin_uploads(frame_rule=…)`. The swap is not in
  `SkinOnGPU`, `PrepareVertexBufferForWriting`, `UpdateSkinnedMeshes` or `OncePerFrameUpdate`;
  the trace is open.

### 9.3 Resources

`resource_acceptance.py --wiring-out --raw-dir` writes scene-state bytes for every byte-equal
capture resource. The orchestrator's `--resource-wiring` replaces their capture-start contents.

- 168 resources come from `hs2pipe.resources` (assets and computed built-ins).
- 14 runtime textures come from the probe raw readback (frame 8749), byte-equal to the capture.
- Remaining capture-start content:
  - body/clothes/hair skinning streams (27; the probe exports only the head surfaces);
  - UI resources (font atlases, list textures, canvas geometry);
  - dynamic VB rings.

### 9.4 Results (run_v7 against the run_v6 oracle)

| pass (16 renders) | draws equal | cbuffers equal |
|---|---|---|
| depth_texture | 224/224 | 1248/1248 |
| depth_normals | 352/352 | 1600/1600 |
| shadow_caster | 144/144 | 768/768 |
| screenspace_shadows | 16/16 | 112/112 |
| forward_opaque | 256/256 | 2672/2672 |
| motion_vectors_camera | 16/16 | 96/96 |
| motion_vectors | 192/192 | 768/768 |
| forward_transparent | 64/64 | 592/592 |
| image_effect_chain (CustomRender + CTAA) | 97/97 | 356/356 |

- PPv2 368/384 actions; the 16 failures are the CopyStd ring arguments.
- Grab copies 32/32, clears 112/112, head skin uploads 22/22, resources wired 182.
- **Execution 959/959 checkpoints equal.**
- Final capture target 31690 after render 15: SHA `1480f828…` = oracle replay =
  `bridge_capture_2.png` RGBA rows in ReadPixels (bottom-up) order.

### 9.5 Remaining external:capture and v6 field requests (additions to §8.6)

- Everything in §8.5 (shadow caster set/order, caps booleans, MV sheet fields, IA/SRV/sampler
  bindings and RT allocations, ring placement).
- Body/clothes/hair skin streams: v6 request for the skin inputs (mesh, weights, bones) of
  every SkinnedMeshRenderer, not only the head surfaces.
- Skin double-buffer slot timing: source trace open. Probe request: a field naming the frame
  held by each skinning slot at bridge time.

## 10 Candidate rendering on the bridge frame (2026-10-25, offline)

API `src/render/hs2pipe/candidate.py` (`BridgeCandidateRenderer`, `SessionConfig`); CLI
`tools/offline_renderer/render_candidate.py`. Outputs: `E:/HS2_offline_renderer_data/candidate_20261025/`
(`run_v2`: unoptimised renders and spot-check images; `run_v5`: optimised, timing; `run_v6`:
native-slider and unsupported candidates).

### 10.1 Chain

1. **Head forward:** card + patch (bridge `{"bones": [...]}` ABMX format and/or `native59`) →
   `CapturedHeadForward.evaluate(baseline_mode='card_load')` for each probe frame N−4..N.
   Baseline: 98 skin-chain bones bit-equal to the probe in frames N−2..N.
2. **Eye look:** the hosted `EyeLookController.LateUpdate` (installed IL, Mono JIT) runs over
   N−4..N with the candidate locals; it rewrites `cf_J_look_L/R`.
3. **Skinning:** `skin_from_locals` with the probe blendshape weights produces 22 uploads:
   - slot 0 = N−2 (capture-start contents);
   - slot 1 = N−1;
   - post-bridge slot 0 = N.

   For the baseline card all 22 are bit-equal to the capture.
4. **Execution:** `frame_executor --serve` (`build_v5`) holds the run_v7 template program
   resident. Per request it restores every frame-written resource, substitutes the streams,
   runs to the capture target's last write (`stop_after`) and reads back 31690 in top-down
   rows.

### 10.2 Assumptions (stated)

- The candidate is held constant over N−5..N (a static card loaded with these values).
- The EyeLook hidden state at N−5 is reconstructed from the baseline export. On this window
  the hosted EyeLook reproduces every exported look rotation 8746–8752. From deliberately wrong
  seeds it converges by N−3 (8747), before the first consumed frame N−2.

### 10.3 Dependencies a candidate changes

Each candidate gets a computed check (`dependency_check`). The changed transforms are those
whose float32 locals differ from the baseline in any frame; their descendants are added.

- **Head surface root bone or transform moved** → `CandidateUnsupported`. ObjectToWorld,
  WorldToObject, `_PreviousM` and the renderer bounds are all functions of the root TRS:
  - bounds = local AABB × root matrix (`InitializePreparedInfo` 0xbdce90;
    `CalculateTransformInfo_Prepared` 0xbdaca0; local AABB written only by `SetLocalAABB`
    0x788010);
  - `updateWhenOffscreen` is asserted false for every head surface (`SetUpdateWhenOffscreen`
    0x788313).
- **Any other renderer's bones, root or transform moved** (body, clothes, hair, accessories)
  → `CandidateUnsupported`, because their skin and matrices come from the capture.
  Example: `cf_J_Head` scale raises for o_bra_a, o_top_camisole2_a, the hair, o_tooth, o_tang
  and o_namida.
- **Otherwise these inputs are provably unchanged:**
  - bounds, therefore culling and draw lists;
  - shadow caster cull set and cascade fit;
  - shadow caster order: its material hash (`Material::UpdateHashes` 0x84e3b0 →
    `Pass::ComputePassValuesHash` 0x855f70) has no mesh, bone or bounds input;
  - object matrices: asserted equal to the template per surface and frame.
- **Recomputed:** the MV previous skin stream (N−1) and every eye-look-dependent bone.

### 10.4 Results

- **Identity:** baseline card, empty patch, all 22 streams substituted → raw target SHA
  `1480f828…` = template replay; pixel-equal to `bridge_capture_2.png`.
- **Spot check (`run_v2/spot_check_overview.png`, `spot_check_crops.png`):**

  | candidate | pixels changed | bbox (x0,y0,x1,y1) |
  |---|---|---|
  | `cf_J_ChinTip_s` position y −0.02 | 66338 | 55,1040–896,1447 |
  | `cf_J_ChinLow` position z +0.02 | 13510 | 156,1051–896,1447 |
  | `cf_J_Chin_rs` scale x ×1.08 | 123920 | 45,725–1085,1447 |
  | native index 1 (upper-face depth) +0.1 | 108163 | 40,479–1074,1392 |

  The changes cover the chin and jaw outline, the chin's shadow and AO on the neck, and (for
  the wider jaw) the cheeks and hair edges. Nothing changes above the cheeks for the chin
  candidates.
- **Timing per candidate** (resident executor; an idle GPU, though the GPU is shared with the
  user's game):

  | step | time |
  |---|---|
  | forward (5 frames) | 0.1–0.5 s |
  | eye look | 0.02–0.1 s |
  | skin | 0.01–0.35 s |
  | executor round trip | 0.5–1.2 s (≈0.11–0.3 s inside the executor) |
  | **total without PNG** | **≈1–2 s** |
  | optional PNG encode | +1.6 s |

  First-time setup is about 40 s.
- **Optimisations, outputs byte-identical** (run_v3/v4/v5 target SHA = run_v2 for every
  candidate):
  - camera position cached;
  - card-load baseline memoized (pure function of `native59`);
  - typed copies;
  - skin reuse when a surface's bone closure has no changed transform (float32 bit
    comparison);
  - resident executor;
  - PNG optional.

### 10.5 Gaps

- Candidates moving a head surface's root, or any bone or transform of another renderer,
  are rejected: body/clothes/hair skin inputs aren't exported (the v6 request stands).
- The two stated assumptions in 10.2.
- The skin double-buffer swap timing is untraced (§9.2).
- Native acceptance of candidates happens later, in a game session.

## 11 v6 session (steps 1-3 done 2026-10-25; candidate work moved to the v6.1 session)

Outputs: `E:/HS2_offline_renderer_data/orchestrator_20261025/{modea_6a9a, modeb_2def}`.

### Done and verified

- **v6 conversions.** Driver inspect listings; capture programs (mode a through event 2226,
  mode b through 27492); mode-b frame graph segmented at the capture target 114653 (17
  segments: 16 bridge renders plus the ordinary frame; 0 mismatched driver actions).
- **Code for the v6 inputs.** v5 regression: `modea_f8bf/run_v30_v6code_regression` gives
  86/86 draws and 517/517 cbuffers, unchanged. Without v6 fields every path falls back to the
  earlier behaviour.
  - Shadow-caster order is generated:
    - `unity_camera_passes.shadow_caster_plan`: shadow queue in scene-node order, then
      `shadow_caster_parts`, then `shadow_caster_draws` with key 0x6a5880;
    - the material hash comes from `scene_v5` `shadow_caster_hash`
      (SharedMaterialData+0xd0, also per render);
    - node fields come from the BaseRenderer flatten block (block byte k = RenderNode+k, from
      `FlattenBasicData` 0x650b50);
    - it raises if two parts tie above bit 39, because RenderNode+0x100 is not exported.
  - GraphicsCaps+0xd0 bit 7, +0x3fc, +0x3a1 and +0x404 come from the v6
    `native.settings`. The captured bias is still cross-checked.
  - `_MotionVectorDepthBias` = −0.001 from BuildSettings+0xfa.
  - `_HasLastPositionData` stays external, because the v6 `smr_*` fields are invalid (probe
    §9.1).
  - `src/render/hs2pipe/bindings.py` (`BindingDeriver`) is written but **not wired or run**.
    For each slot it derives the SRV (material/global texture → capture resource via the
    acceptance table; RenderTexture rows by name and size), the sampler (`sampler_desc` from
    the texture/RT settings) and the IA buffers (mesh streams, skin streams). It checks each
    against the captured binding and counts the rest as external.

### Results of steps 1-3 (2026-10-25)

| Run | Checkpoints | Draws / cbuffers | Bindings derived / external |
|-----|-------------|------------------|-----------------------------|
| `modea_6a9a/run_v10_final` (oracle `run_v6_execute`, fresh) | 60/60 | 86/86, 517/517 | 524 / 363 |
| `modeb_2def/run_v5_execute` (oracle `run_v1`, fresh) | 959/959 | 1361/1361, 8212/8212 | 8351 / 5745 |

PPv2 actions stay 23/24 (mode a) and 368/384 (mode b). The failures are only the CopyStd
blit-ring `args` (as in v5).

**Step 1: caster order and ties.**
- The shadow-caster order is generated from SharedMaterialData+0xd0 plus the flatten-block
  node fields.
- Parts of *different* nodes that tie on (two-sided, hash, odd scale) are now labelled
  `tie` instead of raising: mode a has 1 group (2 entries), mode b has 32 entries. The
  caller adopts the captured order only inside a tie group, and parts of one node must keep
  their generated (subset) order.
  - Statuses: `equal` / `equal_tie_order_external`.
  - In both captures the generated order already equalled the capture inside the ties, so
    no `mesh_id_tiebreak` was needed for shadow casters.
- Depth texture and depth normals use the same `adopt_tie_order` helper.
- **Motion vectors (mode b):** the v6 capture draws o_body_cf before o_head in all 16 bridge
  renders; mode a and v5 drew head first.
  - Cause: `motion_vector_less` (0x6c6f20) compares `instancing_key` = Mesh id << 16 |
    subset before the exact distance. Head and body fall in the same distance bucket (0xC2).
  - The adapter's `mesh_id` is the managed sharedMesh instance id, a stand-in for
    RenderNode+0x100 = Mesh+0x174, which is not exported. It is now labelled
    (`mesh_id_source`).
  - Instanced entries of one (layering, queue, bucket) group carry `tie`. Their order inside
    the group comes from the capture (`mesh_id_tiebreak` motion_vectors, 16 renders).
  - Before this fix, `modeb_2def/run_v1` gave 914/959; the 45 failures were MV → CTAA →
    blit → target.
- **Open:** the same stand-in feeds the forward opaque `mesh_id` comparison
  (the `mesh_id` field compared by the `unity_forward_rules` opaque comparator). All forward lists were equal, but they are not
  proven independent of it. This needs the v7 field RenderNode+0x100 per renderer.

**Step 2: bindings.** `--acceptance-table <resource_acceptance/resources_table.json>` wires
`BindingDeriver` after each `emit_draw`. Each slot is counted as derived only when it equals
the binding the command carries. The report has `per_pass[*].bindings` and
`bindings_summary`.
- Engine textures (not in the pre-render sheets, which hold the previous camera's values)
  come from `engine_textures`:
  - `_CameraDepthTexture` / `_LastCameraDepthTexture`: Camera::RenderDepthTexture
    0x6675de / 0x66760e → depth target of this render's depth pass;
  - `_CameraDepthNormalsTexture`: 0x6c8736;
  - `_ShadowMapTexture` in the collect draw: ComputeScreenSpaceShadowMap 0x69bbdf → the
    cascaded map;
  - `_ShadowMapTexture` in forward draws: RenderLightShadowMaps 0x6cece3 stores the
    CollectShadows result, then EndLoopFlush 0x6cb58b → SetLightShadowProps 0x6cfc85 →
    "Screenspace ShadowMap".
- Remaining external, by category:
  - IA default vertex stream: the channel → stream rule is not transcribed;
  - skinned runtime VBs: body, clothes and hair skin outputs are not generated yet;
  - engine built-in geometry and the dynamic ring;
  - null/default material textures: the m_DefaultName → built-in table 0x14b8e40 /
    GetBuiltinTexEnvParamName 0x8a0e50 selection is not transcribed. Both the SRV and the
    sampler are listed as `builtin-default`;
  - unity_SpecCube0/1 and unity_ProbeVolumeSH (built-in defaults);
  - the comparison sampler of `_ShadowMapTexture` in the collect pass.

**Step 3: engine RT samplers.** `chain_engine_rec` attaches the probe-v6 `render_textures`
row (render record, unique name + size match), and `UnityForwardLowering.blit` generates the
sampler. The generated sampler must equal the captured one, else FrameGap.
- A pooled temporary (`temp_entry_0x138` ≠ 0) that exists at the render is handed out again
  by GetTempBuffer. That path runs TextureSettings::Reset (0x7074e0: Bilinear, aniso 1), then
  sets wrap U/V/W = Clamp (0x75691c..2d) and anisoLevel = 0 (0x756934).
- The pre-render row can still hold the previous user's settings: the first bridge render's
  TempBuffer 214 row had aniso 1, while the capture used Bilinear (Filter 20) at the blit.
- Results: 80/97 chain blits in mode b have no external sampler. The remaining
  `_CameraMotionVectorsTexture` SRV/sampler (16) and three `_MainTex` samplers in mode a
  have no unique row.

### Running when stopped

`resource_acceptance --wiring-out --raw-dir` for both v6 captures (background). Logs:
`orchestrator_20261025/mode*/resource_acceptance.log`. If the logs show no final counts line,
the run did not finish: re-run the same command into a fresh directory and move the partial one
to `deprecated/`.

### Next steps (exact)

1. Run `frame_orchestrate_paired`:
   - **v6 mode a:** window `6a9a1eb9…`, index 5, its capture program, `--resource-wiring`.
   - **v6 mode b:** add `--frame-graph modeb_2def/frame_graph` and `--bridge-request
     bridge_body_2def4f9f…json`.

   Check that the generated caster order equals the capture (`lists.shadow_caster` 'equal'),
   then execute against fresh oracles.
2. Wire `BindingDeriver` into `build()`. After each `emit_draw` call
   `derive(state, sps, material, fv.global_sheet-entries, renderer id)`, using the
   acceptance table of that capture and the skin-stream resource ids. Report the derived and
   external counts per pass; the remaining external list goes into the report.
3. Engine RT samplers for chain blits: build the engine records' `rt` from the v6
   `render_textures` rows (filter/wrap/aniso), and let `UnityForwardLowering.blit` generate
   samplers for those records.
4. Task 2 (waits on the skinning agent's `gpu_skin_inputs` coverage of body, clothes and
   hair):
   - generate those streams;
   - replace the `CandidateUnsupported` root/other-renderer rules in `candidate.py` with
     recomputation: bounds = local AABB × root (rule cited in §10.3) → culling, shadow caster
     set and fit, matrices;
   - raise only for the shadow-caster cull membership when the candidate can change it.
5. Task 3: candidate identity on the v6 bridge frame. Point `SessionConfig` at the v6 session
   (`state_before_renderstyle.json` in the v6 artifacts, template = the v6 mode-b run, target
   114653).

## 12 Native candidate acceptance (probe v6.1 session, 2026-10-25)

Data: `E:/HS2_offline_renderer_data/paired_session_v6_1_20261025/`; receipts in the HS2Mod
artifacts `paired_capture_session_v6_1_20261025/`. Outputs:
`E:/HS2_offline_renderer_data/candidate_native_20261025/w_<nonce8>/`. Tool:
`tools/offline_renderer/candidate_native_acceptance.py`.

**Inputs for each window.** The window's own capture is converted:
- driver inspect;
- capture program through Present−1;
- frame graph segmented at the window's 1086×1448 target (44998, 60885, 71318, 81031);
- resource acceptance and wiring (182 wired; same statuses as v6);
- `frame_orchestrate_paired` template run. All four give 959/959 checkpoints, 1361/1361
  draws, 8212/8212 cbuffers and **0 external cbuffer variables**.

The card is `state_before_baseline.json`, and the patch is that window's `patch_*.json`
(absolute ABMX values; empty for the baseline).

| window | patch | forward = export (5 frames) | skin uploads | frozen root TRS / bounds | target |
|---|---|---|---|---|---|
| `766d8184` baseline | none | all equal | 41/41 | 416/416, 400/400 | `fa848126…` = replay = `bridge_capture_baseline.png` |
| `75e17db3` | cf_J_ChinTip_s | all equal | 41/41 | 416/416, 400/400 | `e37e788a…` = replay = PNG |
| `8a42b979` | cf_J_ChinLow | all equal | 41/41 | 416/416, 400/400 | `e3f80fd5…` = replay = PNG |
| `0b3351d6` | cf_J_Chin_rs | all equal | 41/41 | 416/416, 400/400 | `6d19836e…` = replay = PNG |

- The patch is load-bearing. Without it, the card alone differs at the patched bone and in
  3 / 3 / 5 skin uploads.
- In the identity run all 41 streams are substituted from card + patch. They are
  `skin_from_locals` streams N−2 / N−1 (capture-start contents) and N (in-frame uploads)
  for all 30 meshes.
- The pixels equal `bridge_capture_<name>.png`: 0 differing pixels in every window.

**New source-derived rules.**
- **`_HasLastPositionData`:** SkinnedMeshRenderer+0x255 → RenderNode+0xe8 bit 9
  (AddAsRenderNode 0x77ae22..37). The MV loop sets 1.0/0.0 (0x6cc4e7..0x6cc508; 1.0 from
  0x6cc453). The input is the v6.1 `smr_has_last_position_data_0x28d`.
- **Bridge skin slots:** v6.1 `skin_buffer_holds_frame` / `prev_skin_buffer_holds_frame`
  cross-check the N−2/N−1 rule at every bridge render.
- **Frozen bridge pose** (`candidate.frozen_pose_check`, `frozen_pose.py`):
  - PoseFreeze runs `Animator.Play(hash, 0, 0)` + `Update(0)` inside frame N.
  - Transforms the Animator rewrites take the phase-0 sample from the bridge `pre_cull`
    export.
  - Every other transform keeps its N−1 end-of-frame local, i.e. the forward chain at N−1.
  - Root TRS comes from CalculateGlobalPositionAndRotation 0xdda60.
  - World AABB of local-AABB skinned renderers: CalculateGlobalMatrix 0xdd620 (emulated),
    then the inline path 0xbdbe7f..0xbdbf5c (centre = ((C1·cy + C2·cz) + C0·cx) + C3;
    extent = (|C0·ex| + |C1·ey|) + |C2·ez|).
- **Converter:** ID3D11DeviceContext::Flush (window 0b3351d6, event 25071) has no state or
  resource effect. It is now accepted by `capture_to_program` and `frame_graph`.

**Gaps (explicit).**
- **Hair bounds** (`Kamael_Valentine_Hair_03_HQ`, updateWhenOffscreen): this is the
  bone-AABB path of CalculateTransformInfo_Prepared 0xbdb41e..0xbdb9cc. It runs over the
  Mesh bone AABBs (`Mesh::GetCachedBonesBounds` 0x7a4770), which are not exported, and it is
  not transcribed.
  - Its bounds are the template's.
  - `dependency_check` raises when a candidate moves any of its bones, root or transform.
    None of the chin patches does.
- **Animator-rewritten set:** it is identified as "frozen local differs from the N−1
  export" (86 transforms in every window). The clip bindings are not parsed.
- **Shadow casters and motion vectors:** the captured order is still used inside the
  mesh-id tie groups (`mesh_id_tiebreak`, 16 renders each; RenderNode+0x100 not exported).
- **EyeLook:** the hidden state is seeded from the window's first frame, as in §10.2.

## 13 Multi-view (yaw ≠ 0) support (2026-10-25, offline)

Outputs: `E:/HS2_offline_renderer_data/multiview_20261025/`. Module:
`src/render/hs2pipe/bridge_camera.py`; check tool:
`tools/offline_renderer/bridge_camera_acceptance.py`.

**How yaw reaches the frame** (MakerRenderStyleService.Capture):
`position = target + Quaternion.Euler(0, yaw, 0) * (0, 0, distance); LookAt(target)`.
- **Euler:** managed Deg2Rad products, then native Internal_FromEulerRad (emulated).
- **`Quaternion * Vector3` and `Vector3 + Vector3`:** CoreModule IL (op_Multiply RVA
  0x12d3c). Mono evaluates on a double stack and rounds at each float32 stloc/stfld.
- **Position setter:** SetPosition 0xc01f60 → SetGlobalT 0x2030a0 →
  InverseTransformPosition 0x1fff50 (emulated).
- **LookAt:** Internal_LookAt 0x870400 (GetPosition / CalculateGlobalPosition 0x1dd810;
  LookRotationToQuaternion 0x7ce350 with an identity seed, plus the degenerate fallback
  SetFromToRotation 0x7da4c0 + MatrixToQuaternion 0x7ce390). Then SetRotation 0xc02230 →
  InverseTransformRotation 0x200180 + SetLocalR normalisation.
- **Matrices:** worldToCamera = Scale(1, 1, −1) · GetWorldToLocalMatrixNoScale (0x6647b0);
  cameraToWorld = InvertMatrix4x4_Full (0x662dc0); culling = jittered projection ·
  worldToCamera (0x662ea0).
- **Lights under the camera** (DHH_Light, Lights Custom/Key|Back) and the DHH SSS cameras
  (local identity) use CalculateGlobalPositionAndRotation over the new camera local.
- **Not yaw-dependent:**
  - `_PreviousVP` = Camera+0x598. It is updated from +0x5d8 only by
    UpdatePreviousFrameMatrices (0x66e4ce), once per frame, so it is constant over the 16
    renders (records agree).
  - CTAA jitter offsets the projection only.
  - DOF focus = distance.
  - The shadow cascade fit and culling follow from the generated camera and light matrices
    through the existing passes.
- `capture_loop.capture_position` now implements every yaw. No yaw raise path is left.
- `frame_orchestrate_paired.bridge_camera_pose` generates the camera and lights per bridge
  render and replaces the record values after a bit-exact check.

**Check (yaw 0, 4 windows × 16 renders):** camera local/world TRS, worldToCamera,
cameraToWorld, culling and three attached-light TRS are all bit-equal (`camera_v1`). The
template regression `regress_w_766d8184_template_v1` gives 959/959 with the generated camera.
Predicted poses for −90..180 are in `camera_v1/report.json`; they need yaw ≠ 0 windows for
acceptance.

**Hair bone-driven bounds (updateWhenOffscreen), now generated** (`frozen_pose.bone_aabbs`,
`prepared_local_matrix`, `prepared_compose`, `bone_world_aabb`):
- **Mesh bone AABBs** (GetCachedBonesBounds 0x7a4770 → 0x7a2890, no-blend-shape branch):
  every influence with weight > 0 contributes bindpose·v to a min/max box. They are
  converted to centre/extent at 0xbde941.
- **Bone matrices:** composed from the hierarchy top with the loop 0xbdb5d0..0xbdb895.
- **World box:** 0xbdb8d0..0xbdba06.
- **Result:** 64/64 bit-equal (4 windows × 16 renders). In `frozen_pose_check` the bounds
  are 416/416. **No extra export is needed:** the exported vertices, weights and bindposes
  suffice.
- **Still open:** the blend-shape branch 0x7a2ce3 (no current mesh needs it; it is reported
  if met).

### 13.1 Probe v7 multi-yaw acceptance (2026-10-25)

Outputs: `E:/HS2_offline_renderer_data/multiview_20261025/v7/w_<nonce8>/`. Each has
`inspect/` (with `exit_receipt.json`), `capture_program/`, `frame_graph/`, the resource
acceptance and wiring, `plan_v1/`, `template_v1/` and `identity_v1/`.

| yaw | window | target | template (fresh oracle) | identity: card, empty patch | raw target SHA |
|---|---|---|---|---|---|
| 0 | 2daddb57 | 15092 | 959/959 | = replay = PNG | `836bf157…` |
| +30 | c971d224 | 21735 | 959/959 | = replay = PNG | `d3e51716…` |
| +45 | 321808bd | 22898 | 959/959 | = replay = PNG | `e519af38…` |
| +60 | 2d90e5e1 | 24601 | 959/959 | = replay = PNG | `950fbb95…` |
| +90 | 34ac9218 | 30864 | 959/959 | = replay = PNG | `26ab5576…` |
| −90 | ed32db74 | 31817 | 959/959 | = replay = PNG | `85cfb80f…` |
| −45 | 5c276666 | 37000 | 959/959 | = replay = PNG | `375c4543…` |

- In every window:
  - all 41 skin uploads from the card are equal to the capture;
  - frozen root TRS 416/416 and bounds 416/416, hair bone bounds included;
  - 0 external cbuffer variables;
  - every draw list `equal` with **0 `mesh_id_tiebreak`** and no `equal_tie_order_external`;
  - MV order generated 16/16.
- Per bridge render, the camera (worldToCamera / cameraToWorld / culling) and the
  camera-attached lights are generated from yaw. They are checked bit-exact against the
  records (FrameGap otherwise) before use.

**v7 fields consumed:**
- **RenderNode+0x100** (`render_nodes`, post_render of the same render_id): these replace
  the sharedMesh stand-in and the in-tie capture orders. Depth texture, depth normals, shadow
  caster, motion vectors and forward opaque now sort on fully generated keys.
- **Animator write set:** `pose_clip_bindings_<nonce>.json` (Idle_00, 345 transforms).
  - The "differs from N−1" set is contained in it in every window; 260 written transforms
    keep their N−1 value.
  - One extra writer before the freeze: `Manager.Character.Update` → `ChaControl.UpdateForce`
    → `UpdateBlendShapeVoice` sets `cf_J_MouthMove` localScale.x = `FBSCtrlMouth.GetAdjustWidthScale()`.
    That value is 1f unless openRate > 0.2; ABMX re-applies its modifier only in LateUpdate.
    The rule is implemented and equals the frozen export.
- **CharaCamera / Lights Custom:** the `transform_chains` local TRS (with their −0.0
  components) replace the identity assumption.

### 13.2 External-input closure (2026-10-25, offline)

Outputs: `E:/HS2_offline_renderer_data/external_closure_20261025/` — per window `w_<nonce8>/`
(`template_v2`/`identity_v2`: items 1-6; `template_v3`/`identity_v3`: + items 7-8 with
`--library-extra` and `--material-sources`; `template_v4`/`identity_v4`: + item 9, four
windows, interrupted by the drive loss; `template_v5`/`identity_v5`: final code, + item 10;
`material_sources_v4.json`; `pre_v1`), `chin_a2/` (chin-fit windows), `regress/` (earlier
frames), `shader_library_card_v1/` (the card's bundles), `disasm/` (dumps cited below; the
item 10 dumps are in `disasm/mvquad/`). Superseded check-only probes are in
`w_<nonce8>/deprecated/`. Receipt key `external_closure_20261025`.

**Acceptance (final code, `template_v5` / `identity_v5`).** The E: USB SSD disconnected at
20:39 during the earlier v4 batch (exFAT volume bitmap repaired by chkdsk, 0 bad sectors). The
interrupted v4 runs are in `w_<nonce8>/deprecated/*_interrupted_20261025_2039`; every JSON left
in the output tree parses. Because item 10 changed shared emission code, the final batch reran
all seven windows with the final code instead of the four interrupted v4 cells. It ran one heavy
job at a time to keep the drive's sustained write load low. The table shows the
results; v2 and v3 identities also equal the v7 receipt SHAs.

| yaw | window | v2 | v3 | v5 (final code) template / identity | MV quad UnityPerDraw |
|---|---|---|---|---|---|
| 0 | 2daddb57 | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |
| +30 | c971d224 | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |
| +45 | 321808bd | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |
| +60 | 2d90e5e1 | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |
| +90 | 34ac9218 | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |
| −90 | ed32db74 | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |
| −45 | 5c276666 | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 959/959, = replay = PNG = v7 SHA | 16/16 |

Earlier frames with the final code (`regress/*_v2`): v5.0.1 mode a 60/60, mode b 959/959,
v6 mode a 60/60, v6.1 `766d8184` 959/959. v6 mode b stops on its pre-existing FrameGap
(`render 0 renderer -381402: skin slots hold (51349, -1), rule says (N-2, N-1)`, the invalid
v6 SMR fields), unchanged by this work.

Chin-fit hair-off windows (`E:/HS2_offline_renderer_data/chin_fit_20261025/windows_a2`,
outputs in `external_closure_20261025/chin_a2/`; `candidate_native_acceptance.py
--windows-config configs/windows_a2.json --window <view>`):

| view | window | template / identity (final code) | MV quad UnityPerDraw |
|---|---|---|---|
| front | w_246e80f9 | 959/959, = replay = PNG | 16/16 |
| oblique | w_5ed6d147 | 959/959, = replay = PNG | 16/16 |
| right | w_f28a13ef | 959/959, = replay = PNG | 16/16 |
| left | w_a6fa11a8 | 959/959, = replay = PNG | 16/16 |

**Closed (generated from scene state / installed assets; each checked against the capture):**

1. **Shadow-caster cull membership** (`hs2pipe.unity_shadow_cull`,
   `frame_orchestrate_paired.generated_caster_cull`). `CullDirectionalShadowCasters` 0x67af80:
   `PrepareShadowedLight` 0x67eb30 → `CalculateShadowCasterCull` 0x679140 (camera planes
   ShadowCullData+0x2ec facing the light, |n·L| > 0.01; silhouette edge planes 0x677cd0 over the
   edge tables .rdata 0x14ae7d8/0x14ae7e8 and NDC table 0x14ae790; at most 10 planes) →
   `CalculateCustomCullingParameters` 0x6403c0 (camera culling mask, scene mask, layer mode 0) →
   kernel `ProcessShadowCasterNodeVisibilityAndCullWithoutUmbra` 0x6ac320 (negated SoA planes,
   `rad < dist` rejects) → post-cull 0x6a9eb0. ShadowCullData+0x1c0 / +0x380 from
   `SetupShadowCullData` 0x6818ff..0x681a3a with the ScriptableCullingParameters cull-matrix
   selection 0x7b9201..0x7b93b4 (checked equal to the culling matrix the shadow path uses).
   SceneNodes from `AddToScene` 0x739540 / `AddRendererInternal` 0x68ae10 (scene mask =
   UnityScene ctor 0x820e56; bit 29 only `RemoveRenderer`). Result: generated set = captured set
   in 112/112 bridge renders (7 windows × 16); the captured draw list remains the check.
2. **Motion-vector inclusion** (`unity_camera_passes.motion_vector_included`).
   DoMotionVectorRenderLoop 0x6ca0a0..0x6ca12d with RenderNode+0xe8 = BaseRenderer+0xf0 and bit 9 =
   SkinnedMeshRenderer+0x28d (AddAsRenderNode 0x77ae22), `CompareApproximately` 0x7cb900 of
   node+0/+0x40, MotionVectors pass, ForceNoMotion. The renderer-update stamp
   (RendererUpdateManager+0x388) decides only a moved Object-mode RO without a MotionVectors pass;
   none occurs (16/16 renders generated in every window).
3. **Camera Transform position** is now separate from cameraToWorld column 3: they differ by one
   ulp at yaw +30. Transform::GetPosition feeds CullingParameters+0x194, the RO sort context
   (ctx+0x798), Camera::GetPosition (skybox) and the shadow cull; `_WorldSpaceCameraPos` keeps
   column 3 (SetCameraShaderProps r14+0x80). Bridge renders: the generated pose world position,
   checked against the record's `camera_state.transform.position`. Labels
   `camera_transform_position` / `camera_implicit_flags` now name these generated sources.
4. **EyeLook seed and target** (`candidate.eye_seed_state`): the game's own EyeObject state, the
   eye postfix of N−5 (checked equal to the prefix of N−4), and the EyeUpdateCalc target argument
   of each frame's prefix record (Camera.main.transform.position; the scene cameraToWorld
   translation was a stand-in). With these, every window's forward chain equals the export in all
   frames N−4..N (`pre_v1`: 0 differing transforms, 7/7 windows; two windows previously differed
   in N−4/N−3).
5. **Mouth width before the freeze** (`candidate.frozen_pose_check`): `FBSCtrlMouth.adjustWidthScale`
   of the N−1 fbs postfix record and `fileStatus.mouthAdjustWidth` of the N UpdateForce postfix
   (driver hooks), instead of the asserted 1f.
6. **Built-in TexEnvs** (`bindings.resolve_texenv`). `shaderprops::GetTexEnv` 0x85b850: built-in
   TexEnv name (GfxDevice+0x1eb8) → material sheet (texture of the requested dimension, else its
   +0x14 built-in default, 0x85aa80) → global sheet → `GetDefaultTexEnv` 0x755480. Built-in
   TexEnv k → texture from `ReinitBuiltinTextures` 0x759c30 (names from
   `GenerateBuiltinTextures` 0x754620); unity_SpecCube0/1 = the default reflection texture of
   `GenerateDefaultSample` 0x67d5d0; unity_ProbeVolumeSH = GetDefaultTexture(3D). Built-in
   TextureSettings: Texture ctor Reset 0x7074e0, BuildTexture aniso 0 (0x2d2389), cubemap
   builder Clamp (0x750622), re-applied by `Texture::SetAnisoLimit` 0x75b390 (FindObjectsOfType has
   no hide-flag filter). The exported `UnityWhite` / `UnityBlackCube` settings equal the rule.
7. **Viewports** of the camera passes (`frame_orchestrate_paired.camera_viewport`): SetViewport
   0x54d1e0 → FlipRectForSurface → SetViewportInternal; generated from the camera pixel rect.
   **Scissor rects** of draws with scissor disabled: `DisableScissor` 0x546e80 issues no
   RSSetScissorRects, so the rect is the device state of the preceding command
   (`FrameEmitter.device_scissors`).
8. **Shader copy of each material** (`hs2pipe.material_sources`, `--material-sources`) from the
   renderer's own bundle: acceptance prefab binding → component `m_Materials[slot]` → material
   `m_Shader` PPtr. Runtime replacements from IL: `ChaControl.ChangeSettingHairShader` (hair →
   `hair_shader_mat.unity3d` hair_dithering / hair_cutout by the card's `ChaFileHair.shaderType`),
   face/clothes MatData from the item's MainAB (`InitBaseCustomTextureFace`), and a unique-name
   lookup for the body material. The card's own bundles (fo_head_30, the body and hair zipmods) are
   a second library root (`--library-extra`), and the device active tier (GraphicsCaps+0xd4) is
   passed to program selection. This resolves o_top_camisole2_a (the `clothes_shader_bundle`
   gap); GraphicsSettings (`hs2pipe.graphics_settings`) proves the DepthNormals / MotionVectors /
   ScreenSpaceShadows library copies are the referenced objects (unity_builtin_extra 62/75/64).

9. **IA buffer slots and draw arguments of mesh draws** (`hs2pipe.ia_bindings`,
   `BindingDeriver.derive_ia`). `BindBuffersForDraw` 0x5438e0 sets slots [0, streamCount) and
   leaves higher slots as device state. `ExtractMeshBuffers` 0x7a3f40 takes the rendering data's
   streams (skinned: stream 0 = skinning output, `MeshRenderingData::Init` 0x7a5560); for motion
   vectors it truncates to `CalculateRequiredStreamCount` 0x793d50 and appends the
   previous-position stream. `AddDefaultStreamsToMeshBuffers` 0x793760 appends
   `GetDefaultVertexBuffer` 0x8cd0c0 when Normal/Tangent/Color, or a texcoord below the mesh's
   lowest one, is required but missing. Mesh streams come from the installed Mesh object of the
   acceptance row (16-byte aligned stream offsets in m_DataSize), compared byte-for-byte with the
   captured buffer. Draw arguments: SubMesh indexCount, firstByte / index size, baseVertex (a
   material slot past the submesh count draws the last submesh). Yaw 0 window: bindings derived
   13 230 / external 866 (before: 8 319 / 5 777); ±90: 12 991 / 625 and 12 990 / 626;
   draw_args external 1 264 -> 160. The per-draw
   `ia` label now separates the captured layout object from the derived slots.

10. **Camera motion-vector quad world matrix.** `LoadFullScreenOrthoMatrix` 0x68fc70 calls
    `GfxDevice::SetViewMatrix` (vtable +0xe8 at 0x68fcc4). `GfxDevice::SetViewMatrix` 0x8cf290
    runs `SetIdentity` 0x7d0a00 on the world matrix at GfxDevice+0x2110 (0x8cf2ed..0x8cf2f4),
    the same field `SetWorldMatrix` 0x8cf370 writes. The mono `RenderFullScreenQuad` path
    (eye 2) sets no matrix after it; only its stereo branch calls `SetWorldMatrix(identity)`
    (0x6ce481). So the quad's `unity_ObjectToWorld` / `unity_WorldToObject` are identity. The
    earlier rule, "device state left by the previous generated draw", held only while the last
    forward-opaque draw was the BG dust (identity). In the hair-off ±90 chin windows the last
    draw is `o_shoes_bones` (a bone-root matrix), and the templates gave 895/959 (16 UnityPerDraw
    cbuffers, then MV → CTAA → blits). With the fix they give 959/959. Test:
    `DllTables.test_camera_motion_quad_world_matrix`.

**Still external:capture (reason, the missing input):**

- *IA input layout object and topology* (every draw) and the vertex/index buffer slots of
  draws without an installed Mesh row (particles, the BG MeshRenderers, fullscreen quads), the
  stale slot above streamCount of the first hair draw after a non-program command, and the
  camisole motion-vector previous-position slot (its m_SourceMap-derived stream count differs).
  The rest is closed in item 9. The program's real required-channel mask (the VertexInputMasks
  passed to ExtractMeshBuffers) is not exported, so each derived slot is accepted only when it
  equals the captured binding.
- *Draw arguments* of non-mesh draws (160 per window: particles, Blit / CopyStd quads) and the
  *render-target allocation*: the program binds capture RT resources; the RenderTexture
  descriptors of the bridge temporaries are not in the pre-render `render_textures` rows
  (allocated inside the render).
- *Engine RT samplers*: `_ShadowMapTexture` (the Shadowmap temp: RenderShadowMaps
  0x6a19cc..0x6a1a2a sets its filter from ShadowMapJobHeader+0x3c, GraphicsCaps+0x401 and the
  tier byte; the comparison variant comes from D3D11Sampler flag bit 1, GetSampler 0x5520c0),
  `_CameraDepthTexture`, `_GrabTexture` and the CTAA `_CameraMotionVectorsTexture` SRV/sampler:
  their RenderTexture rows are not exported for this render.
- *Scissor rects* left by a command outside the program state (48 per window: the first draw
  of the frame and the first forward draw after the camera clear / SetActive).
- *Dynamic VB/IB ring placement* (PPv2 CopyStd `args`, 16 per window; Graphics.Blit quads):
  the ring offset advances by ~6789 vertices and 54 indices per bridge render, dominated by the
  BG-dust particle geometry, and wraps; it needs the ring write offsets at frame start and the
  particle vertex counts.
- *Shader copies not resolved*: the BG-dust particles (`Standard`, map bundle not identified:
  no acceptance row for particle renderers); o_tang binds a 160-byte UnityPerDraw while every
  installed copy of `AIT/Skin Translucency simple` (fo_head_30 = the material's PPtr target,
  fo_head_38, oo_base, the SAC_Innie zipmod) declares m_Size 176.
- *SceneNode LOD fields* (Renderer+0x1a0 LODGroup): taken as no LOD group in the shadow-caster
  cull, as in the camera cull.
- *Spliced work*: the ordinary frame-N render after the bridge (segment 16), UI, BackCamera and
  non-pass resource updates (unchanged).
- *EyeLook for a synthetic candidate*: seeded with the window's game state at N−5 (exact for the
  window's own card + patch); the candidate's own history before N−4 does not exist in the game.

**Probe v8 field requests:**

| field | where read | use |
|---|---|---|
| `[gRendererUpdateManager]+0x388` (u32; global .data 0x1547438), per Main Camera render at pre_render | DoRenderLoop copies it to RenderLoopContext+0x7bc (0x6b99c4..0x6b99d5); DoMotionVectorRenderLoop compares node+0xb4 with it (0x6ca0b7) | MV inclusion of moved Object-mode renderers without a MotionVectors pass |
| raw SceneNode (RendererScene+0x88 + 0x20 × Renderer+0x160, 32 bytes) and the RendererCullData list index per renderer, per render | AddToScene 0x739540 | LOD fields and list membership of the caster/camera cull (now assumed none / one list) |
| GraphicsCaps+0x3e0 (u32) and +0x401 (u8) | GetDefaultVertexBuffer 0x8cd0f9; RenderShadowMaps 0x6a19da | default-stream stride; Shadowmap filter |
| ParserBindChannels / VertexInputMasks of each drawn program (the `VertexInputMasks` passed to ExtractMeshBuffers), or the GpuProgram channel mask | ExtractMeshBuffers 0x7a3f40 (arg word at rsp+0x88) | IA stream count and default stream |
| D3D11 dynamic VBO/IB write offsets (GfxDeviceD3D11 dynamic buffer: current offset, size) at frame start and after each bridge render | DynamicVBO chunk allocation | CopyStd / Blit ring args |
| `render_textures` rows at post_render (not only pre_render) for the temporaries allocated in the render (Shadowmap, Screenspace ShadowMap, Camera DepthTexture, motion-vector RT, GrabPass Temp), with TextureSettings and the RenderTexture descriptor | RenderBufferManager::GetTempBuffer 0x7562a0 | engine RT samplers, RT allocation |
| per material: the Shader object's serialized identity (PersistentManager file path + local file id of `Material.shader`) | Object::GetInstanceID → PersistentManager::InstanceIDToSerializedObjectIdentifier | o_tang 160-byte layout, BG-dust `Standard` |
