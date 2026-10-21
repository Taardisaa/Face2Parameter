# HS2 offline renderer: shader library and Unity 2018.4 variant selection (2026-10-20)

This component gives the frame generators the installed shader programs. Given a shader asset,
a pass (or a LightMode or replacement route), the active keyword set and the platform (D3D11),
it returns:

- the exact installed DXBC for each stage;
- the DXBC's declaration reflection;
- the serialized Unity binding tables;
- the pass render state, with property-bound values resolved.

It supersedes the head-only `extract_shader_programs.py` exports. It also supersedes the provisional
`paired_capture_session_20261020/shader_manifest.tsv` (37 shaders).

Architecture contract: [hs2_offline_renderer_architecture.md](hs2_offline_renderer_architecture.md).
Frame evidence: [hs2_offline_frame_graph.md](hs2_offline_frame_graph.md).

| file | role |
|---|---|
| `tools/offline_renderer/extract_shader_library.py` | read-only extraction (`extract`) and the coverage/verification receipt (`coverage`) |
| `src/render/hs2pipe/shader_library.py` | library, variant selection, routing, render state, cbuffer layouts and packer |
| `tools/offline_renderer/test_shader_library.py` | 24 tests (synthetic + installed data + receipt) |
| `tools/offline_renderer/shader_library_coverage_20261020.json` | compact coverage and verification receipt |
| `outputs/offline_renderer_shader_library_20261020/` (ignored) | extracted programs, decompiled sources, PDB, bulky verification |

## 1 Extraction

`extract_shader_library.py extract --source ... --out FRESH` reads `Shader` and `ComputeShader`
objects with UnityPy. It reads AssetBundles, `.assets`/resource files, `ZIP::MEMBER` zipmod members
and extracted plugin-embedded bundles (their `.provenance.json` sidecar is embedded).

Nothing is converted or dropped.

- **Shaders.** Each Shader is kept with its properties, Fallback, Dependencies and every SubShader
  (tags, LOD). Each Pass keeps its type: normal, UsePass or GrabPass with its texture name. It also
  keeps the pass tags plus state tags, the raw `m_State` (each value constant or bound to a property)
  and the program mask.
- **Subprograms.** Every SerializedSubProgram of every stage (vs/ps/gs/hs/ds) is kept **in serialized
  order** with:
  - tier and GPU program type;
  - keyword indices, resolved through the pass name table;
  - requirements and the raw binding tables.

  The keyword names resolved through the name table are checked against the blob's own
  `m_Keywords` strings. A disagreement aborts the extraction; none occurred.
- **DXBC.** Program bytes are cut at the DXBC container's own size field and stored by content as
  `dxbc/<sha256>.dxbc`, with `asm/` (`D3DDisassemble`). Any Unity header prefix or trailing bytes are
  recorded.
- **Reflection.** `reflection.json` comes from the SM4/5 token stream (`frame_graph.reflect`
  helpers). It covers cbuffer float4 counts, SRV/sampler/UAV slots and ISGN/OSGN.
- **ComputeShader.** In 2018.4 there are no keyword variants. Each kernel of each
  (targetRenderer, targetLevel) variant is exported with its cbuffers, textures and buffers, and the
  thread group size.
- **Tiers.** Every one of the 7628 extracted subprograms has `m_ShaderHardwareTier` 3, which means
  tier-independent.

Installed sources found (all read-only):

| shader(s) | installed source | how it is reached at runtime |
|---|---|---|
| AIT/Skin True Face, Eye Translucency, eyelashes, main namida, main eyeshadow lambert, Skin Translucency simple | `abdata/chara/38/fo_head_38.unity3d` | head bundle |
| **AIT/Skin True** (body) | `abdata/chara/mm_base.unity3d` (material `cf_m_skin_body_00`) | `ChaControl.LoadBody` drawMatName |
| AIT/Liquid, Custom/Silhouette, AIT/mnpb | `abdata/chara/oo_base.unity3d` | body prefab slot 1 / silhouette |
| **Shader Forge/hair_07** | `abdata/chara/hair_shader_mat.unity3d` | `Manager.Character` loads materials `hair_dithering`/`hair_cutout`. `ChaControl.ChangeSettingHairShader` replaces the mod hair material's shader. The mod material keeps its own keywords. |
| **AIT/Clothes True** | `abdata/chara/00/fo_{top,bot,inner_t,inner_b,shoes}_00.unity3d` | Clothes bundles. The five copies **differ**: shader_feature stripping is per bundle, giving 400/320/160/160/80 subprograms. A generator must use the copy from the item's own bundle, via `source=`. |
| Internal-DepthNormalsTexture, ScreenSpaceShadows, MotionVectors, BlitCopy, Deferred*, UI/Default, Legacy Shaders/VertexLit … | `HoneySelect2_Data/Resources/unity_builtin_extra` | built-ins |
| PostProcessing shaders **and the 6 compute kernels** (MultiScaleVORender/Downsample1/2/Upsample, Lut3DBaker) | `HoneySelect2_Data/sharedassets2.assets` | PPv2 resources |
| Particles/Additive, BackGround 2 Color | `sharedassets3.assets` | Maker scene |
| Standard | `resources.assets` (filtered to `Standard`) | |
| Hidden/LightingPass, SSS_Profile, SeparableSSS | `DHH_HS2.dll` resource `DHH.Resources.sss` (SHA `ad7c8f1a…`) | DHH SSS |
| **Skybox/MyCubemap** | `E:/HoneySelect2_ArcticFox/DHH_Data/ModData/dhpost` | `DHH_Main` loads materials `hdri1..5` into `RenderSettings.skybox` |
| Hidden/CTAA_PC, CTAA_Enhance_PC | `HS2_BetterAA.dll` embedded bundle | BetterAA |

## 2 Coverage (exact DXBC SHA-256)

| capture | executed VS/PS/CS programs | matched | unmatched |
|---|---|---|---|
| retry frame 17675 (`retry.zip.xml`, 16 bridge renders + Maker frame) | 105 (44/55/6) | **105** | 0 |
| paired frame 26059 (ordinary Maker frame, 3200×1800, UI included) | 92 (39/47/6) | **92** | 0 |
| paired frame 12740 | 92 (the same program set as 26059) | **92** | 0 |

The union is 106 programs; the paired frames add `Internal-DeferredReflections` PS `e064cd77…`.

The earlier frame-graph gaps are all closed:

- AIT/Clothes True (clothes bundles);
- AIT/Skin True (mm_base);
- hair_07 (hair_shader_mat);
- Skybox/MyCubemap (DHH dhpost);
- the 6 compute shaders (sharedassets2).

Several hashes have many installed identities, such as the PPv2 Uber VS, or one VS shared by several
shaders. All of those identities are kept.

## 3 Variant selection (recovered from UnityPlayer.dll)

The source is the installed `UnityPlayer.dll`, CodeView `BFA91A8C44734BB4AABC5A975B304CFE` age 1. Its
public PDB came from `symbolserver.unity3d.com`. The disassembly used capstone; the PDB is kept under
ignored outputs.

1. **Load filter.** `SubProgram::CreateFromSerializedSubProgram`, RVA 0x8393e0, keeps a subprogram
   when all of these hold:
   - `m_ShaderHardwareTier >= 3`, or it equals `GraphicsCaps` active tier;
   - its requirement mask ⊆ caps;
   - the blob's GPU program type is supported (the DX11 types).

   `Pass::CreateFromSerializedPass`, RVA 0x838a20, appends the surviving subprograms in serialized
   order per stage.
2. **Requested set.** `ApplyMaterialPassAndKeywordsWithCache`, RVA 0x837340, ORs the material's
   256-bit keyword set into `ShaderPassContext.keywords`. Those context keywords are the global ones
   plus the keywords the renderer sets for the pass. `ShaderState::FindSubProgramsToUse`, RVA
   0x856570, passes the same set, unmasked, to each stage.

   2018.4 has only global keywords: one `ShaderKeywordSet`, no local keywords; blob
   `m_LocalKeywords` is empty.
3. **Match.** `Program::GetMatchingSubProgram`, RVA 0x85aff0, works in three steps:
   - It looks the set up in a cache keyed by XXH32 of the 32-byte set.
   - On a miss, it scores every variant with `keywords::ComputeKeywordMatch`, RVA 0x85a400:
     `popcount(req & var) − 16·popcount(var & ~req)` (Harley-Seal popcounts; `shl ecx,4`). It keeps
     the first strictly greater score, starting from INT_MIN and skipping variants marked unsupported.
   - It caches the result under the requested set.

   Consequences:
   - Requested keywords that are not in a variant cost nothing, which is how keywords absent from
     the shader end up ignored.
   - Each unrequested variant keyword costs 16.
   - Ties go to serialized order.
   - An exact variant always wins.
   - shader_feature and multi_compile do not differ at runtime. The difference is only in which
     variants the build contains, and that is visible in the serialized lists.

Routing rules in `shader_library.py`:

- **LightMode table.** It keeps the **first** pass of each LightMode
  (`SubShader::UpdateLightModeToPassIndexTable`, RVA 0x841800). A pass without a tag is `Always`
  (documented).
- **ShadowCaster.** `Shader::GetShadowCasterPassToUse`, RVA 0xbf96c0, takes the active subshader's
  own ShadowCaster pass, otherwise the shader-level pass resolved from the Fallback chain.
- **Replacement.** `Camera.RenderWithShader(shader, tag)`: the object's tag value selects the first
  replacement subshader that has an equal value. If none matches, the object is not drawn. Every
  pass of the matched subshader is drawn, in order. This rule is documented, and the capture
  verifies it. DHH calls it with `"RenderType"` (`DHH.Util.RenderToTarget`).
- **UsePass.** `"Shader/PASS"` resolves to the passes with that name. GrabPass is exposed as
  `Pass.type == 'grab'`.
- **Renderer keywords** (documented Unity behaviour, used by the verifier):
  - ForwardBase sets `DIRECTIONAL LIGHTPROBE_SH`, plus `SHADOWS_SCREEN` when the main light casts
    shadows, `receiveShadows` is on and the queue is ≤ 2500.
  - ShadowCaster and the depth texture set `SHADOWS_DEPTH`.
  - The camera depth texture and per-object motion vectors render opaque queues (≤ 2500) only.

## 4 Verification against captures

The receipt is `shader_library_coverage_20261020.json`; the bulky output is in `coverage_v14/`.
Two independent checks were run.

- **Routing + selection on the retry capture** (1392 object groups covering 1568 draws, 16 segments). For each
  (segment, camera, label, object), the rules above produce the expected sequence of draws:
  - the renderer's eligible material slots in queue order;
  - each slot multiplied by the passes its route selects;
  - each draw's selected VS/PS.

  The comparison is element-wise.
  - Groups: 1312 are equal, and 80 are equal with the hair material keywords unknown in that session.
    None differ, and no group has a draw-count difference.
  - D3D11 state (depth enable/write/func, cull, RT0 blend) through `to_d3d11`: all 1568 draw states
    are equal.
  - The check exercises these routes:
    - the DHH SSS_Profile and LightingPass replacement;
    - o_head's absence from SSS lighting (its RenderType TransparentCutout maps to the subshader with
      only ShadowCaster passes);
    - eyes, tooth, tang and eyeshadow taking their SSS depth pass from LightingPass's Fallback
      `Legacy Shaders/VertexLit`;
    - the clothes' DepthNormals `TreeTransparentCutout` subshader drawing both passes;
    - the body liquid slot;
    - the transparent-queue exclusions.
- **Pass-anchored keyword selection.** The pass is taken from the executed hash, and only the
  keyword choice is tested.
  - Retry capture: 1819 draws equal, plus 119 with unknown hair keywords.
  - Paired frame 26059: **86/86 equal**, using same-frame native material keywords and global
    keywords. Those globals are `SOFTPARTICLES_ON`. The draws include the hair (`anyi` keywords) and
    PPv2 Uber `COLOR_GRADING_HDR_3D` among 3456 variants.
  - Almost every selection is a closest match (`exact: false`), because material keywords such as
    `_EMISSION`/`_NODE_686_ON` are absent from the variants. The namida PS picks
    `_ALPHAPREMULTIPLY_ON`, which was not requested, because every variant has it. hair_07 picks its
    `*_OFF` keywords the same way.
- **Built-in cbuffers.** Layouts parsed from the 2018.4.11f1 editor
  `CGIncludes/UnityShaderVariables.cginc`, at
  `HS2Mod/zips/UnitySetup64-2018.4.11f1/Editor/Data/CGIncludes`, use the D3D11 standalone defines
  and HLSL packing. They agree with every installed program's serialized layout, with 0 mismatches:

  | cbuffer | programs | bytes |
  |---|---|---|
  | UnityPerDraw | 2098 | 176 |
  | UnityPerFrame | 2794 | 368 |
  | UnityLighting | 2027 | 768 |
  | UnityPerCamera | 2868 | 144 |
  | UnityShadows | 847 | 416 |
  | UnityProbeVolume | 1090 | 112 |
  | UnityFog | 636 | 32 |
  | UnityReflectionProbes | 355 | 128 |
  | UnityPerCameraRare | 88 | 352 |
  | UnityLightmaps | 4 | 32 |

## 5 API (`src/render/hs2pipe/shader_library.py`)

```python
lib = ShaderLibrary([root, ...])                 # extraction dirs (library.json + manifests + dxbc/)
sh  = lib.shader('AIT/Clothes True', source='fo_top_00')   # copies must agree unless source= is given
shader, sub = sh.resolve(tier=None, maximum_lod=None)      # active subshader or Fallback chain
p   = sub.first_pass_with_light_mode('ForwardBase')        # / passes_with_light_mode / passes_named
p   = sh.shadow_caster_pass()                              # own or fallback-chain ShadowCaster
sub = replacement_subshader(lib.shader('Hidden/LightingPass'), 'RenderType', sh)
sel = p.select('ps', keywords, tier=None)                  # Selection(subprogram, score, exact, ...)
sel.subprogram.dxbc()                                      # exact installed bytes
sel.subprogram.reflection(); .constant_buffers(); .uniform_map(); .resources()
state = p.render_state(material_floats)                    # property-bound values resolved
d3d   = to_d3d11(state, reversed_z=True)                   # blend/depth-stencil fields, cull as authored
blob  = pack_cbuffer(layout, values, base=None)            # exact bytes; coverage_mask() for padding
builtin = parse_builtin_cbuffers(cginc_path, D3D11_STANDALONE_DEFINES)
lib.compute_kernel('Lut3DBaker', 'KGenLut3D_AcesTonemap'); lib.match(sha); lib.hash_index()
```

`tier=None` admits only tier-independent subprograms. If a stage has tier-specific subprograms, it
raises, because the device tier would then be needed.

`pack_cbuffer` follows the `shader_bindings.constant_buffers` encoding:

- values are GPU encodings;
- matrices are flattened Unity column-major `Matrix4x4f`;
- int is int32 and bool is 0/1;
- array elements and matrix registers are float4-aligned;
- struct arrays (`UnityInstancing_*`) are flattened as `struct[e].member`.

## 6 Reproduce

```powershell
$E='E:/HoneySelect2_ArcticFox'; $O='outputs/offline_renderer_shader_library_20261020/library_v1'
.venv/Scripts/python.exe tools/offline_renderer/extract_shader_library.py extract --out $O/main `
  --source $E/abdata/chara/38/fo_head_38.unity3d --source $E/abdata/chara/oo_base.unity3d `
  --source $E/abdata/chara/mm_base.unity3d --source $E/abdata/chara/hair_shader_mat.unity3d `
  --source $E/abdata/chara/00/fo_top_00.unity3d --source $E/abdata/chara/00/fo_bot_00.unity3d `
  --source $E/abdata/chara/00/fo_inner_t_00.unity3d --source $E/abdata/chara/00/fo_inner_b_00.unity3d `
  --source $E/abdata/chara/00/fo_shoes_00.unity3d --source $E/HoneySelect2_Data/Resources/unity_builtin_extra `
  --source $E/HoneySelect2_Data/sharedassets2.assets --source $E/HoneySelect2_Data/sharedassets3.assets `
  --source outputs/offline_renderer_repair_20261012/dhh_sss_bundle.unity3d `
  --source outputs/offline_renderer_frame_graph_20261020/embedded/HS2_BetterAA.ctaa.unity3d
.venv/Scripts/python.exe tools/offline_renderer/extract_shader_library.py extract --out $O/resources_standard `
  --source $E/HoneySelect2_Data/resources.assets --shader Standard --compute __none__
.venv/Scripts/python.exe tools/offline_renderer/extract_shader_library.py extract --out $O/dhh_dhpost `
  --source $E/DHH_Data/ModData/dhpost
# coverage + verification (see the receipt's argv-equivalent fields; paired XMLs are the official converts)
.venv/Scripts/python.exe tools/offline_renderer/extract_shader_library.py coverage --library $O/main `
  --library $O/resources_standard --library $O/dhh_dhpost --retry-actions <frame_graph_v2>/actions.json `
  --retry-passes <..>/passes.json --retry-head-surfaces <..>/head_surfaces.json --retry-summary <..>/summary.json `
  --retry-geometry ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/baseline_front_geometry.json `
  --paired paired_frame26059 <P>/parse_04d1e76f/capture.zip.xml <P>/window_04d1e76f…/frames/scene_005.json `
  --paired paired_frame12740 <P>/parse_625aa0f2/capture.zip.xml - `
  --cginc ../HS2Mod/zips/UnitySetup64-2018.4.11f1/Editor/Data/CGIncludes/UnityShaderVariables.cginc `
  --out outputs/offline_renderer_shader_library_20261020/FRESH --receipt tools/offline_renderer/shader_library_coverage_20261020.json
.venv/Scripts/python.exe -m unittest tools.offline_renderer.test_shader_library -v
```

- The scene for the paired frame is `scene_005`: the marker frame 26060 is window index 5.
- Frame 12740 has no scene export, so it is checked for coverage only.
- Retained failed or superseded runs: `coverage_v1`–`v13`, `trial_v1`.
- A paired capture is interpreted without a driver listing. `FrameGraph` runs without
  `group_passes`, so there are no event IDs, and chunk indices are used instead.

## 7 Remaining gaps (explicit)

1. **Renderer keywords not exercised.** Point, spot and cookie lights, `VERTEXLIGHT_ON`, lightmaps
   and fog keywords occur in no captured draw. The keyword set for a ForwardAdd draw depends on which
   light it belongs to, and that per-draw light identity is not recovered: the verifier carries both
   hypotheses.
2. **Not disassembled.** The following rules are implemented from documentation and verified only by
   captures:
   - the active-subshader "supported" test and the LOD choice;
   - Fallback resolution of the shader-level ShadowCaster (`IntShader::ResolveFallback`);
   - replacement tag matching, including whether tag values are case-sensitive;
   - whether more than one ForwardBase pass of a subshader is drawn (no captured shader has two);
   - the queue limits used by the verifier.
3. **Requirement mask not evaluated.** No `GraphicsCaps` mask is known. Captured selections agree.
4. **Draws without keyword verification** (programs hash-matched only):
   - UI/Default: 81 draws, Canvas materials not exported;
   - Unity-internal blits, ScreenSpaceShadows, DeferredReflections, and DepthNormals or MV draws of
     unexported objects;
   - 4 Standard draws;
   - compute dispatches (no keywords in 2018.4).
5. **Global keyword timing.**
   - Retry capture: the globals are taken from DHH source code (`SSS_PROFILES`); there was no native
     read.
   - Paired capture: the globals come from a single native read inside the frame. DHH toggles
     keywords during `OnPreRender`, so that read is not per-draw state.
6. **Retry-session materials.** The clothes materials come from the installed bundle renderers,
   resolved by keyword agreement, because ChaControl replaces `_MainTex`. The hair keywords are
   unknown. The paired frame resolves both natively.
7. **`to_d3d11` stops at shader state.** It does not decide `FrontCounterClockwise` (render-target
   flip), depth-bias translation, stencil enable or the D3D11 view state; those belong to the
   renderer.
8. **Unsupported layouts.** Matrix params with RowCount ≠ 4 raise an error; none were observed. Only
   `Standard` was extracted from `resources.assets`, because no other shader in it is used by the
   captured frames.
