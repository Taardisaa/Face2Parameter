# HS2 offline renderer: exact GPU resources (2026-10-20)

This is the resource layer of the [reimplementation architecture](hs2_offline_renderer_architecture.md).
Given scene state and a request (a renderer's mesh, a material texture slot, or a texture
identity), `src/render/hs2pipe/resources.py` returns frame-program resource entries:

- the verbatim D3D11 descriptor;
- the exact `D3D11_SUBRESOURCE_DATA` bytes of every subresource;
- the SRV descriptor Unity creates;
- provenance: `asset` (installed file + `path_id`), `computed` (recovered code, with
  its location), or `external_runtime` (producer named; no bytes are invented).

Acceptance is byte-for-byte against the capture-start contents of the paired capture
`rdc_04d1e76f…_frame26059.rdc` (official zip.xml in `parse_04d1e76f/`, scene state
`window_04d1e76f…/frames/scene_005.json`). No game was launched; nothing was written to the
game directory.

## Files

| file | role |
|---|---|
| `src/render/hs2pipe/resources.py` | upload rules, source index, Sideloader-aware binding, plugin selections, API |
| `src/render/hs2pipe/sideloader_assets.py` | Sideloader mod load set and bundle redirection, reproduced from `HS2_Sideloader.dll` |
| `tools/offline_renderer/resource_acceptance.py` | capture oracle, mapping captured resources to scene requests, comparison, receipt |
| `tools/offline_renderer/test_resources.py` | 16 CPU tests (synthetic, installed data, receipt) |
| `tools/offline_renderer/resources_acceptance_20261020.json` | compact receipt |
| `E:/HS2_offline_renderer_data/resources_20261020/` (outside the repo) | source index, Sideloader scan, decompiled plugins, per-resource table, scratch |

Bulky outputs are on E: (user decision: C: is nearly full). Only hashes are compared; no
texture dump is written.

## Upload rules (recovered)

**Textures** (`Texture2D` and `Cubemap` assets):

- `m_TextureFormat` maps to a DXGI format (`UNITY_TEXTURE_FORMATS`). The `_SRGB` variant is
  used when `m_ColorSpace` is 1 and the project is linear (`QualitySettings.activeColorSpace`
  from the scene).
- `MipLevels` is `m_MipCount`. Cubemaps are 6-slice arrays with `MISC_TEXTURECUBE`.
- Image data is laid out image-major, then mip-major, each mip tightly packed (4x4 blocks
  for BC formats). That is the D3D11 subresource order.
- RGB24 is expanded to RGBA with alpha 255. ARGB32 is reordered to RGBA.
- `m_StreamData` is read from the bundle's `.resS` member or the sibling `.resS` file.
- Unity creates RGB24 textures without creation data and uploads them later. Their bytes
  equal the capture's `Internal::Initial Contents`.

**Meshes** (`m_MeshCompression` 0):

- `m_VertexData` holds its streams back to back, each starting on a 16-byte boundary.
  Unity creates one DEFAULT vertex buffer per stream.
- The index buffer is `m_IndexBuffer` verbatim.
- For skinned meshes, the POSITION/NORMAL/TANGENT stream is replaced by the skinning output
  at draw time. Its static bytes are returned with `bound_by_frame=False`; the frame never
  references them.

**Engine built-ins** (UnityPlayer.dll `5034cdc2…`, PDB `BFA91A8C…/1`):

- Built-in textures come from `builtintex::GenerateBuiltinTextures` (RVA 0x754620). The fill
  callbacks store one dword per texel: `EmptyNormalMap` stores 0xffff7f7f, `White` 0xffffffff,
  `Black` 0, `Empty2D` 0x80808080. Cubes and the 3D default are built the same way.
- The default vertex buffer, bound with stride 0, comes from `GfxDevice::CreateDefaultVertexBuffer`
  (RVA 0x8cb000).
- The skybox sphere comes from `SkyboxGenerator::GetCubemapSkyboxVB` (RVA 0x641740, helpers
  0x63d6e0 and 0x63dbb0, base table at RVA 0x1242710). It is 3 uniform 4-way subdivision
  passes followed by 2 adaptive passes, reproduced with single-precision SSE operations in
  the disassembled order. It is **byte-exact on the first attempt** (5040 vertices).

**PPv2 script objects:** `RuntimeUtilities.fullscreenTriangle` and `whiteTexture`.

## Which installed file is read (binding)

- **Source index** (`SourceIndex`). Type and name are indexed over:
  - the Maker-relevant `abdata` folders;
  - the player data and level files;
  - every zipmod bundle registered under one of those vanilla paths;
  - the item zipmods used here;
  - the Maker face-list thumbnail bundles.

  That is 860+ sources. The index also records the serialized-file (CAB) names, so external
  PPtrs can be resolved.
- **Prefab binding.** A scene renderer is bound to the prefab GameObject whose path below the
  instantiated root equals the end of the scene path. Its `m_Mesh` gives the mesh. Its
  material's `m_TexEnvs` PPtr gives each texture the prefab authored itself; textures that
  ChaControl assigns from list rows are resolved by identity instead.
  - Objects placed in a scene, such as the Maker background, are bound through the
    `level*` file's MeshFilter.
  - The probe does not export `MeshFilter.sharedMesh`, so MeshRenderers need `mesh_name`.
- **Sideloader** (`sideloader_assets.py`, from the decompiled `HS2_Sideloader.dll`):
  - Mods are deduplicated by GUID: highest version, then longest archive name, otherwise the
    newest file. They load in GUID order.
  - `BundleManager.TryGetObjectFromName` returns the first registered zipmod bundle of a
    path that `Contains` the asset; only if none does, the vanilla file is read.
  - Reproduced load set: 5346 of 5357 archives and the same 11 duplicate choices as the
    game's `LogOutput.log`.
- **UncensorSelector 3.11.5.** The card has no BodyGUID, and the config says `Random`. So the
  body is chosen from the random-eligible female bodies in Manifests order with
  `System.Random(birthDay + personality + (int)(voicePitch*100f)).Next(n)`. Mono's Knuth
  generator is reimplemented.
  - The seed is 102 and the index is 0, which selects `bp.sac_innie_v2`.
  - Its `o_body_cf` (11033 vertices) is byte-exact. The vanilla and patched `oo_base` bodies
    (9594 vertices) are not what the game draws.
- **XUnity.AutoTranslator** (decompiled `TextureTranslationCache`). A UI texture is replaced
  by `Translation/zh-CN/Texture/**/<name> [KEY-HASH].png`, where KEY is SHA-1(name)[:10].
  The PNG is decoded to RGBA and stored bottom-up. Both Maker sprite atlases are byte-exact
  this way; their raw assets differ from the capture.

## Results (paired frame 26059; receipt `resources_acceptance_20261020.json`)

The frame reads 253 resources (VB, IB, SRV). **152 are byte-exact**: 113 Texture2D, 2 cube,
1 3D and 36 buffers. **15 are runtime with exact descriptor/SRV**: the 14 composites. There are
**82 runtime** and **3 unresolved**.

There are **2 mismatches**, and both are Maker face-list thumbnails (`p_cf_head_02`
256² BC1, `preset_head_02`). A name-only identity request found a single installed
candidate, but that candidate is not the shown list row, which is UI state that is not
exported. All non-UI-list resources match. The per-class counts are in the receipt's `counts`.

| class | result |
|---|---|
| Asset textures (body, head, eyes, mouth, clothes, hair zipmod, PPv2 LUTs/blue noise, Maker UI sprites, BG, vanilla list thumbnails) | byte-exact: every subresource, descriptor and SRV |
| Asset meshes (streams 1..n and IBs of 15 skinned meshes incl. uncensor body and hair, Maker BG mesh) | byte-exact |
| Unity built-ins (7 textures), default VB, skybox VB | byte-exact (computed) |
| PPv2 fullscreen triangle, white texture | byte-exact (computed) |
| XUnity-translated UI atlases (2) | byte-exact (computed from PNG) |
| ChaControl composites (14 RTs: face, body, 5 clothes × main/detail) | descriptor and SRV equal; bytes are runtime (see gaps) |

## Runtime-generated resources

| resource | producer (decompiled) | class | status |
|---|---|---|---|
| face `_MainTex` / `_DetailMainTex` RT 2048² | `ChaControl.CreateFaceTexture` → `CustomTextureCreate.RebuildTextureAndSetMaterial` (clear, sRGBWrite, `Graphics.Blit` with mm_base `Create/skin color` / `Create/skin detail`, via KSOX `OverlayBlit`), then auto mips | a | descriptor exact; prelude not emitted |
| body RTs 4096² | `CreateBodyTexture` + KSOX underlay/overlay from the uncensor bundle | a | descriptor exact; prelude not emitted |
| clothes RTs (camisole, pantshot, bra, shorts, shoes) | `ChangeCustomClothes` (`Create/Clothes`, `Create/Clothes Detail`) + KCOX post-hook | a | descriptor exact; prelude not emitted |
| skinned vertices (current, previous) | Unity CPU skinning (Map) | a | owned by the skinning generator (already bit-exact there) |
| Unity dynamic VB/IB ring (Blit, deferred and screen-space quads) | GfxDevice immediate geometry | a | vertex bytes owned by the generators |
| ColorGrading curves texture | PPv2 in-frame upload | a | `hs2pipe.ppv2` |
| in-frame render targets (TempBuffers, depth, shadow, GrabPass, LUT) | camera/PPv2/CTAA/DHH | a | frame generators |
| `''` ARGBHalf 3200x1800 (resource 8745) | render target read before any write in this frame (temporal history) | b | needs a raw runtime export |
| UI Canvas geometry (4 VB/IB pairs) | UnityEngine.UI batching | b | UI layout state not exported |
| particles / dynamic batching VB/IB | Maker BG `ParticleSystem` | b | simulation state not exported |
| Font Texture ×2 (dynamic font atlas) | FreeType glyph cache since startup | b | needs a raw runtime export |
| `''` 28x28 RGBA8 (12287) | unidentified script Texture2D (UI) | b | needs a raw runtime export |
| mod face-list thumbnails `p_cf_head_02`, `preset_head_02` (5 resources) | Maker list row `ThumbAB`, which is UI scroll state | asset + UI state | 3 unresolved (several candidates), 2 mismatch (name-only identity picked the wrong row) |

## API

```python
from src.render.hs2pipe import resources as R, sideloader_assets as S
index = R.SourceIndex.load(INDEX)              # R.SourceIndex.build(R.default_index_sources(...) + extras, INDEX)
st = S.SideloaderState.load(SCAN)              # S.scan(GAME, SCAN)
body = R.uncensor_selector_body(card, st)      # card: sex, birth_day, personality, voice_pitch
res = R.SceneResources(scene, index, sideloader=st, xunity=R.XUnityTextureTranslations(),
                       plugin_meshes={n: {'source': body['source'], 'asset': body['asset']} for n in R.UNCENSOR_BODY_NAMES})
res.texture(scene_texture_id)                  # asset / builtin / PPv2 / runtime entry
res.material_texture(material_id, '_MainTex'); res.renderer_textures(renderer_id)
res.renderer_mesh(renderer_id[, mesh_name=])   # [vertex_stream_k..., index_buffer]
res.texture_identity(name, w, h, dxgi, mips, bundle_path=None, ui=False)
R.builtin_texture_entry(name); R.default_vertex_buffer_entry(); R.skybox_cubemap_vertex_buffer_entry()
rid, views = entry.add_to(builder)             # frame_program.ProgramBuilder; provenance stored on the resource
```

## Reproduce

```powershell
python E:/HS2_offline_renderer_data/resources_20261020/scratch/run_scan.py E:/.../sideloader_scan_v1.json
python E:/.../scratch/build_index5.py E:/.../source_index_v5.json
python tools/offline_renderer/resource_acceptance.py --xml <P>/parse_04d1e76f/capture.zip.xml --zip <P>/parse_04d1e76f/capture.zip `
  --scene <P>/window_04d1e76f…/frames/scene_005.json --index E:/.../source_index_v5.json --sideloader-scan E:/.../sideloader_scan_v1.json `
  --card-state <P>/state_after_04d1e76f….json --shader-library <lib>/main --shader-library <lib>/resources_standard `
  --shader-library <lib>/dhh_dhpost --mesh-renderer o_ai_mi_createBG00_00=134230 --out E:/.../acceptance_trial5 --receipt RECEIPT
python -m unittest tools.offline_renderer.test_resources -v
```

The index builder scripts in `scratch/` only call `SourceIndex.build` on
`default_index_sources()` plus three groups of zipmod members: Sideloader override
registrations, item mods, and `fo_head` `ThumbAB` bundles. Promoting them to a repository
CLI is a follow-up.

## Gaps (explicit)

1. **ChaControl composites are not computed.** The face, body and clothes composite bytes
   are produced at card load, before the capture. The producer chain is recovered:
   - materials, shaders and property IDs;
   - list rows;
   - KSOX `OverlayBlit` and the uncensor overlay/underlay;
   - clear, sRGBWrite and mip generation.

   A prelude still needs four things:
   - a Graphics.Blit lowering, the `unity_forward` `blit` interface that is still missing;
   - Unity's material Color gamma→linear conversion, to be recovered from UnityPlayer;
   - the global keyword state at load time;
   - the card's exact float32 values, read from the card file rather than the rounded bridge
     snapshot.

   Only the descriptors are verified.
2. **The scene export lacks `MeshFilter.sharedMesh`** (MeshRenderers need `mesh_name`), and
   the Maker list row and UI state needed for thumbnail identity.
3. **Mono culture collation is not reimplemented.** A GUID order is accepted only where the
   collation proxies agree, and anything else raises. For example, oo_base:
   `com.ArcticFox.Munc…` loads before `HS2_UncensorSelector.Base`.
4. **XUnity texture files with duplicate keys** are not resolved, because the `GetFiles`
   order decides between them. 36 keys are affected; the request raises.
5. **One classification label is wrong in the receipt.** The receipt labels the PPv2 curves
   texture (8379) `script_texture` because the scene path is taken before the name rule. Its
   real producer is the `hs2pipe.ppv2` in-frame upload.
6. **Malformed mod thumbnails are skipped.** These are `m_CompleteImageSize` larger than the
   mip chain, or `m_MipCount` 0. They are recorded and their upload is not modelled.
7. **Class-b runtime resources** need the HS2_RenderAudit raw readback: the temporal RT, the
   UI geometry, particles, font atlases and the unidentified 28x28 texture.
8. **The index builder CLI** lives in scratch scripts, not a repository tool. Index staleness
   is checked by size and mtime.
