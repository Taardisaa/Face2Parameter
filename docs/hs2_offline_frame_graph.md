# HS2 offline renderer — frame graph of the retry capture (2026-10-20)

Scope: the complete ordered frame graph of the existing capture
`HS2Mod/artifacts/chenger_reference_fit_20261012/native_present_session_v1/unaccepted_retry_frame17675.rdc`
(SHA `eb5743bc…`, frame 17675) and the binding of the head's real surfaces to every draw.
CPU interpretation of the official XML+ZIP export, plus one official `drawinputs` cross-check
on the existing RDC. No game, new capture, injection, deployment or card change.
This is evidence for the full-replay renderer. It is not candidate RGB acceptance.

Tool: `tools/offline_renderer/frame_graph.py` (tests: `test_frame_graph.py`).
Full JSON (ignored): `outputs/offline_renderer_frame_graph_20261020/frame_graph_v2/`
(`summary.json`, `passes.json`, `actions.json`, `resources.json`, `edges.json`,
`head_surfaces.json`). `frame_graph_v1/` is the same run before the three transfer labels
were added and is superseded.

## 1 Mechanisms (source-derived, not assumed)

- **Event IDs** come from the official driver listing (`retry_driver_inspect_v1/inspect/stdout.bin`).
  All 3260 driver actions map to the interpreted actions. Event ID, action ID and chunk index
  all match, with 0 mismatches and 0 extra actions.
- **Capture-start buffer bytes.** Pinned RenderDoc 1.46 `d3d11_initstate.cpp`
  `Serialise_InitialState` writes no chunk for `Resource_Buffer`. It copies the staged
  capture-start bytes into the buffer's record, so the `CreateBuffer` `InitialData` blob *is*
  the content at capture start. Textures use `Internal::Initial Contents` chunks, or the
  creation record when no chunk exists.
- **Map/Unmap.** Written bytes travel on the Unmap (`Serialise_Unmap`), and `WRITE_DISCARD`
  invalidates the whole buffer. Six Unmaps belong to Maps that were issued before the capture
  began; pinned `Serialise_Unmap` still serializes their data. They are applied and listed in
  `summary.notes`. 102 Maps are still open at the end of the capture, all issued after the
  readback (next-frame uploads). 96 of them target resources that the capture never created.
- **Used shader slots.** Unity strips RDEF, so used slots come from the mandatory SHDR/SHEX
  declaration tokens (`dcl_constantbuffer`, `dcl_resource*`, `dcl_sampler`, `dcl_uav*`).
  Vertex inputs come from ISGN. For all 101 matched programs the parser agrees with the
  installed export's disassembly. Bound but undeclared slots are kept as `bound_but_unused`.
  Constant-buffer reads are limited to the declared float4 count.
- **Shader identity.** Each shader is matched by the exact SHA-256 of its executed DXBC against
  installed exports. 101 of 136 shader objects match, and every matching variant is kept. Some
  vertex shaders are byte-identical across several installed shaders; all of those sources are
  listed. Slot names and cbuffer layouts come from the matched export. A program that matches
  nothing gets a cbuffer name only when the *same CB resource* is bound under that name by a
  matched program; that witness is recorded.
- **Producer edges.** Buffers are tracked per byte interval and textures per subresource. For a
  texture, a read lists the last full overwrite (a clear, a full copy or a DISCARD Unmap)
  followed by every later partial writer. Draws count as partial writers, because full pixel
  coverage would need execution. The latest writer is flagged. `edges.json` has 168,468 edges.
- **Pass labels** follow fixed rules on recorded debug names (shader names, target names). The
  rule for each label is stored with it. Nothing is labelled from rendered appearance.
- **Views** are identified by the actual `unity_MatrixVP` bytes of each draw, read from the
  reconstructed cbuffer contents.

## 2 Frame structure

There are no debug markers, so grouping uses the recorded output bindings. The frame has 934
passes and 17 segments. A segment ends when the capture target 59322 is written, or at Present.

**Segments 0–15 are sixteen complete renders of the capture camera** (events 24–35079, about
2188 events each). They come from the bridge's sixteen renders. Each one is the same pass
sequence; segment 0 is shown here:

| # | label | events | target(s) | shaders |
|---|---|---|---|---|
| 0 | post_curves_texture_copy | 24 | Internal Curves Texture ← staging 59343 | – |
| 1 | sss_profile | 49–61 | TempBuffer 543×724 (cleared 28/29) | Hidden/SSS_Profile |
| 2 | blit_copy | 86 | SSS_ProfileTex | Hidden/BlitCopy |
| 3 | depth_texture_prepass | 113–172 | Camera DepthTexture 543×724 | Hidden/LightingPass (replacement) |
| 4 | depth_normals | 188–341 | Camera DepthNormalsTexture 543×724 | Internal-DepthNormalsTexture |
| 5 | shadow_map | 373–432 | Shadowmap 4096² (14283) | Hidden/LightingPass |
| 6 | screenspace_shadows | 469 | Screenspace ShadowMap 543×724 | Internal-ScreenSpaceShadows |
| 7 | sss_lighting | 514–540 | LightingTexture 543×724 | Hidden/LightingPass |
| 8–11 | sss_separable_blur ×4 | 579–627 | TempBuffer 1922/1923 → SSSLightingTextureBlurred | Hidden/SeparableSSS |
| 12–13 | blit_copy | 648, 662 | ImageEffects Temp, LightingTexture | BlitCopy |
| 14 | depth_texture_prepass | 696–849 | Camera DepthTexture 1086×1448 (59404) | AIT ShadowCaster passes |
| 15 | depth_normals | 865–1019 | Camera DepthNormalsTexture 1086×1448 | Internal-DepthNormalsTexture |
| 16 | shadow_map | 1045–1131 | Shadowmap 14283 (same resource as pass 5) | AIT ShadowCaster passes |
| 17 | screenspace_shadows | 1164 | Screenspace ShadowMap 59416 | Internal-ScreenSpaceShadows |
| 18 | forward_scene (opaque) | 1191–1553 | TempBuffer 1913 color 59330 + depth 59334 | AIT forward base, BackGround, Standard |
| 19 | motion_vectors | 1580–1657 | TempBuffer 1927 (59420) on depth 59334 | Internal-MotionVectors |
| 20–31 | ambient_occlusion_msvo | 1671–1857 | MSVO pyramid; 10 compute dispatches; composite into 59330 | PostProcessing MultiScaleVO |
| 32/34 | grabpass_copy | 1860, 1891 | GrabPass Temp 59636 ← 59330 | – |
| 33 | forward_scene | 1890 | 59330 | AIT/Liquid |
| 35 | forward_scene (transparent) | 1910–1951 | 59330 | AIT/eyelashes, AIT/main namida, Particles/Additive |
| 36–41 | post CopyStd + DepthOfField ×5 | 1970–2035 | | PostProcessing |
| 42 | color_grading_lut_bake | 2046 | Color Grading Log Lut (3D) | Lut3DBaker (compute) |
| 43–47 | Uber, SMAA ×3, FinalPass | 2069–2125 | → 59330 | PostProcessing |
| 48 | blit_copy | 2151 | 11566, 3200×1800 (written each iteration, never read in this frame) | BlitCopy |
| 49–53 | blit, CTAA_Enhance, blit, CTAA_PC, blit | 2163–2245 | 59668 → 59672 → 59676 / 59680 → 59684 | HS2_BetterAA CTAA |
| 54 | blit_copy | 2259 | **59322** (the capture target) | BlitCopy |

What the capture contains, by unity_MatrixVP:

- **DHH SSS camera, 543×724.** It renders with the LightingPass replacement shader. Its VP is
  identical in all 16 iterations, so it has no jitter.
- **Capture camera, 1086×1448.** It has a **different jittered VP in each iteration**; row 3
  differs by about 1e-3. That is CTAA sub-pixel jitter.
- **Shadow light views.** The SSS camera's view is constant. The capture camera's view changes
  slightly per iteration.
- **Ortho blit view.**

Inferred from the rendered passes: the capture camera renders a depth texture, depth normals
and motion vectors. The `depthTextureMode` component value itself is not in the capture.

**Segment 16** covers events 35086–38147:

1. Readback copy 59322 → staging 59915 (event 35086, chunk 37429).
2. A staging upload 59916 → 59913.
3. The next normal Maker frame:
   - the batched CPU-skinning Map/Unmap (Maps at 35096–35105, Unmaps at 35198–35211);
   - deferred clears, reflections and shading;
   - the DHH SSS camera at 1600×900;
   - the main camera at 3200×1800 with the same chain as above;
   - UI (81 draws) and Present.

Segment 16 runs after event 35086, so it cannot affect the readback.

## 3 Head surface draws (segment 0; identical in every capture segment)

**Binding rule.** A draw binds to a source surface only when its executed index list equals
that source submesh's list exactly, with BaseVertexLocation applied. The IB bytes are the
capture-start record. Source: `source3d_v1/baseline_front_geometry.json`, 10 surfaces. No draw
is ambiguous.

Static UV/COLOR/UV2 streams are float32-exact against the source for every drawn surface.
TEXCOORD2/3 of o_head have no source field to compare against. Debug names agree, but they are
not used as evidence.

| surface | SSS cam: profile / depth / dnorm / shadow / lighting | capture cam: depth / dnorm / shadow / forward / MV | draws per segment |
|---|---|---|---|
| o_eyebase_L | 61 / 119 / 194 / 384 / 540 | 757 / 871 / 1068 / 1269 / 1600 | 10 |
| o_eyebase_R | 58 / 116 / 191 / 381 / 537 | 723 / 868 / 1065 / 1265 / 1605 | 10 |
| o_eyelashes | – / – / 249 / – / – | – / 927 / – / 1910 (transparent) / – | 3 |
| o_eyeshadow | 55 / 113 / 188 / – / 530 | – / 865 / – / 1390 / 1629 | 7 |
| o_head | – / 155 / 242 / 424 / – | 717 / 920 / 1120 / 1338 / 1615 | 8 |
| o_namida | – / – / 235 / – / – | – / 913 / – / 1931 (transparent) / – | 3 |
| o_tang (head) | 52 / 125 / 200 / 411 / 521 | 825 / 877 / 1111 / 1225 / 1595 | 10 |
| o_tooth | 49 / 122 / 197 / – / 514 | 817 / 874 / – / 1241 / 1610 | 8 |
| o_tang (body N_tang[1]) | none | none | 0 |
| o_tang (silhouette) | none | none | 0 |

Per-segment event lists for all 17 segments are in `head_surfaces.json`.

- o_head is **not** drawn in the DHH SSS_Profile or LightingPass lighting passes. Only the eyes,
  eyeshadow, tang and tooth are.
- The head has no ForwardAdd draw and no `Custom/Silhouette` draw.
- The body and silhouette o_tang index lists differ from the head o_tang. They have no draw in
  this frame, which agrees with `active_in_hierarchy=false` in the source export.
- Eyebrows are not a mesh. `c_t_eyebrow_17` is sampled directly by the o_head forward PS as
  `_Texture3`.

Exact programs (DXBC match):

- o_head forward: `AIT/Skin True Face` pass 0 `{DIRECTIONAL, SHADOWS_SCREEN, LIGHTPROBE_SH}`
  (VS 14302 / PS 14303).
- o_head ShadowCaster: pass 2 `SHADOWS_DEPTH`. The PS bytes also match the `SHADOWS_CUBE`
  variant; both are retained.
- Eyes: `AIT/Eye Translucency` with the same keywords.
- Eyelashes: `AIT/eyelashes {DIRECTIONAL, LIGHTPROBE_SH}`.
- Namida: `AIT/main namida` pass 1 `{…, _ALPHAPREMULTIPLY_ON}`.
- Motion vectors: `Hidden/Internal-MotionVectors` pass 0.
- Depth normals: `Hidden/Internal-DepthNormalsTexture`.
- SSS camera: `Hidden/LightingPass` and `Hidden/SSS_Profile` from the DHH resource.

## 4 How head vertex data reaches the GPU

**There is no GPU skinning in this frame.** The six compute shaders are PostProcessing MSVO
and the LUT baker. Unity skins on the CPU. The head-skinning document has the details.

Each drawn surface uses these streams:

- **Slot 0:** a DYNAMIC `Buffer-1-<n·40>` buffer with POSITION/NORMAL/TANGENT and stride 40.
- **Slot 1:** a DEFAULT buffer named after the mesh, holding COLOR/UV streams.
- **Index buffer:** `o_xxx(Clone)`.
- **Motion-vector draws only:** slot 1 is rebound to a *second* DYNAMIC `Buffer-1-<n·40>`
  buffer as `TEXCOORD4`, the previous-frame positions.

| surface | current VB (slot 0) | writer used by segments 0–15 | in-frame write (after readback) | previous VB (MV) |
|---|---|---|---|---|
| o_head | 29868 | CreateBuffer chunk 1315, blob 1641 | Unmap 35206 (Map 35097) | 29861 (blob 1636, never written) |
| o_eyelashes | 29869 | blob 1642 | Unmap 35198 | – |
| o_eyebase_R | 29870 | blob 1643 | Unmap 35209 | 29863 |
| o_tooth | 29871 | blob 1644 | Unmap 35211 | 29865 |
| o_tang | 29872 | blob 1645 | Unmap 35205 | 29860 |
| o_eyeshadow | 29873 | blob 1646 | Unmap 35210 | 29864 |
| o_namida | 29874 | blob 1647 | Unmap 35204 | – |
| o_eyebase_L | 29875 | blob 1648 | Unmap 35208 | 29862 |

**No head vertex or index buffer is written anywhere in segments 0–15.** All 16 capture
renders read the capture-start contents, which are the `CreateBuffer` record bytes. That record
is the correct substitution point for current geometry.

Official replay agrees with this CPU provenance. `frame_graph_drawinputs.py` compared every
bound VB and IB at 12 draws byte-for-byte against the official `drawinputs` replay: the eight
surfaces' segment-0 forward draws, the o_head MV draw 1615 (previous buffer), segment-15 draw
34184, and post-Unmap draws 36531 and 36826. All 12 runs passed and all buffers were
byte-identical. Receipt: `outputs/offline_renderer_frame_graph_20261020/drawinputs_crosscheck_v2/receipt.json`.
The `v1/` attempt is a retained failure caused by a relative path argument.

Per-draw transforms are written by an Unmap just before each draw. For example:

- o_head forward 1338: UnityPerDraw `unity_ObjectToWorld` ← Unmap 1336; `$Globals` ← 1333;
  UnityPerFrame ← 1185.
- MV 1615: `$Globals` `_NonJitteredVP` / `_PreviousVP` / … ← Unmap 1614; UnityPerDraw ← 1592.

Every CB producer for every draw is in `actions.json`.

## 5 Motion vectors and previous-frame dependence

- **The motion-vector pass is active** in every capture segment (pass 19). It writes 59420.
  Its only consumer is `Hidden/CTAA_PC` slot t0, once per segment. Inputs: current VB, previous
  VB (TEXCOORD4), `_PreviousVP`, `_NonJitteredVP` and the per-draw matrices.
- **The previous buffers are not equal to the current ones.** Capture-start maximum absolute
  differences:
  - o_head: 1.6e-6
  - eyes: 0.0124 (gaze)
  - o_tang/tooth/eyeshadow: ≤1.5e-6
  - body: 0.152 (breathing)

  A candidate must define previous positions for every MV-drawn surface. Changing only the
  current VB introduces motion vectors that the original frame does not have.
- **CTAA accumulates across the 16 iterations.** Iteration 0 initializes its history by
  blitting current into history (2201 → 59676). Iteration k then reads iteration k−1's output
  through a 59680/59676 ping-pong (`summary.temporal_dependencies.cross_segment_reads`). The
  final 59322 therefore depends on all 16 jittered renders.
- **No definite pre-capture history is read on the 59322 path.** Every GPU-writable texture read
  with no in-frame writer is a baked character texture with an Initial Contents chunk, such as
  the head `_MainTex` and `_DetailMainTex` RenderTextures 29730/29735. One exception: in
  iteration 0, the SSS-camera body shadow draw 373 has `_MainTex` = 11478 bound, which is the
  previous main-frame post temp. In later iterations the same slot holds 59648, the previous
  iteration's SMAA temp.
- Other capture-start bases on render targets are **potential** dependencies only. Fullscreen
  draws are modelled as partial writers.
- MSVO and LUT dispatch UAVs are modelled as read-write, because declarations do not say
  whether a UAV is write-only.

## 6 Unresolved gaps (explicit)

1. **Unmatched programs.** These have no installed export, so they have used slots but no slot
   names or layouts: `AIT/Clothes True`, `AIT/Skin True` (body), `Shader Forge/hair_07`
   (zipmod), `Skybox/MyCubemap`, and the six PostProcessing compute shaders.
   `extract_shader_programs.py` does not export ComputeShader assets.
2. **Coverage of draw writes is not decided.** Whether a fullscreen blit overwrites every pixel
   needs execution, so capture-start bases behind draw writes stay potential dependencies.
3. **Cameras are not bound to Unity objects.** Camera and pass identity rests on VP bytes,
   targets and debug names. Mapping each to a Unity `Camera` object (DHH SSS camera, bridge
   capture camera, Maker main camera) and reading the real `depthTextureMode` needs live state.
   That belongs to the main agent and the owner of the game process.
4. **Geometry sources differ in session and card.** The binding source geometry comes from a
   different session (source3d_v1, 2026-10-12 02:01) and a different card from this capture.
   Topology and static streams are asset data and bind exactly. Skinned positions are not
   compared here; that belongs to the head-skinning work.
5. **Draw-list invariance is open.** Unity's culling and draw order for a candidate with
   changed bounds cannot be established from one capture. The capture fixes the draw list.
6. **Previous-frame inputs are open.** A candidate needs previous-frame positions for the MV
   draws, the matching `_PreviousM` and per-draw matrices, and the same CTAA jitter sequence.
   The original state was not a frozen pair of identical frames.
7. **Scope of acceptance.** This frame graph and the drawinputs check do not accept candidate
   RGB or the full renderer.

## 7 Reproduce

```powershell
.venv/Scripts/python.exe -m unittest tools.offline_renderer.test_frame_graph -v
.venv/Scripts/python.exe -m tools.offline_renderer.frame_graph `
  --xml ../HS2Mod/artifacts/chenger_reference_fit_20261012/native_present_session_v1/retry_parser_v1/retry.zip.xml `
  --inspect ../HS2Mod/artifacts/chenger_reference_fit_20261012/native_present_session_v1/retry_driver_inspect_v1/inspect/stdout.bin `
  --shader-manifest outputs/offline_renderer_repair_20261012/shader_programs_v2/manifest.json `
  --shader-manifest outputs/offline_renderer_repair_20261012/body_shader_programs_v1/manifest.json `
  --shader-manifest outputs/offline_renderer_repair_20261012/sss_shader_programs_v2/manifest.json `
  --shader-manifest outputs/offline_renderer_frame_graph_20261020/shader_exports/unity_builtin_extra/manifest.json `
  --shader-manifest outputs/offline_renderer_frame_graph_20261020/shader_exports/sharedassets2/manifest.json `
  --shader-manifest outputs/offline_renderer_frame_graph_20261020/shader_exports/sharedassets3/manifest.json `
  --shader-manifest outputs/offline_renderer_frame_graph_20261020/shader_exports/resources_assets/manifest.json `
  --shader-manifest outputs/offline_renderer_frame_graph_20261020/shader_exports/hs2_betteraa_ctaa/manifest.json `
  --geometry ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/baseline_front_geometry.json `
  --segment-boundary-resource 59322 --out outputs/offline_renderer_frame_graph_20261020/FRESH
```

The added shader exports are read-only `extract_shader_programs.py` runs on installed files.
Each run was filtered with `--shader` to the names that occur in the frame:

- `HoneySelect2_Data/Resources/unity_builtin_extra`
- `sharedassets2.assets`
- `sharedassets3.assets`
- `resources.assets`
- the `HS2_BetterAA.ctaa` bundle embedded in `BepInEx/Plugins/HS2_Plugins/ManlyMarco/HS2_BetterAA.dll`,
  extracted with `extract_embedded_resource.ps1` and stored with a provenance JSON.

The GPU cross-check is `python -m tools.offline_renderer.frame_graph_drawinputs --rdc … --xml …
--inspect … --tool outputs/offline_renderer_repair_20261013/official_action_provenance_msvc_v1/renderdoc_replay_msvc.exe
--renderdoc-dir outputs/offline_renderer_repair_20261012/renderdoc_portable_1_46/extracted/RenderDoc_1.46_64
--event N … --out FRESH`. It uses one binary-logged process per event with no retries.
