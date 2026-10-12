# HS2 offline renderer repair — 2026-10-12

The CPU **geometry and camera stage** now uses captured state and recovered
installed assets. A full game-equivalent RGB renderer is still unfinished.
Diagnostic images are not approved MediaPipe fitting inputs.

## Mechanisms repaired

- `CapturedHeadForward` keeps actual current local TRS, private ABMX baselines,
  historical length/direction, flags, modifier list order, rotation exclusions
  and explicit Apply count. It reuses the independently audited installed
  `BoneModifier.Apply` transition, not `position * length + offset`.
- Native59 uses the installed head keyframes, SliderUnlocker profile and 113
  source-derived assignments. Native updates refresh the corresponding ABMX
  caches via Reset → Face/Body Update → CollectBaseline. Body Update's two
  actual head-ancestor writers (`cf_N_height`, `cf_J_Head_s`) sample real
  `cf_custombody/cf_anmShapeBody` assets. Mouth masking uses installed ten IDs.
- Modifier creation appends an actual identity/cache collection; removal resets
  and rebuilds native head state. Effects resolve first matching modifier objects.
  Effect lists are checked against every requested surface bone and ancestor;
  a breast-only producer is excluded only when the dependency intersection is
  empty. Unknown overlapping producers remain failures, not disabled features.
- Missing expression deltas are recovered from the installed head's render
  prefab. Source positions, triangle order, indices, weights and bindposes must
  be float32-exact. Names/counts alone never choose a mesh. Normal/tangent frame
  deltas are retained too. Neutral `e00_defo=100` is preserved.
- All requested surfaces retain renderer paths, including three distinct
  `o_tang` entries. Skinning reconstructs world positions directly, avoiding
  repeating renderer scale on Unity's already-scaled BakeMesh output.
- `src/render/cpu.py` uses the actual offscreen `capture_camera`, homogeneous
  clipping, CPU OpenGL-depth projection, pixel centers, depth tests and
  perspective-correct barycentric/UV/attribute interpolation. It does not
  reframe each candidate or use `geometry.main_camera` by mistake.

Each context validates six installed assembly hashes. Mesh frames and their
bundle hash are recorded. Geometry and context capture times stay visible;
matching output does not assert they were captured in the same frame.

## CPU entry point

```powershell
.venv/Scripts/python.exe scripts/hs2_render_cpu.py `
  --geometry ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/baseline_front_geometry.json `
  --context ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/context.json `
  --capture ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/baseline_front_capture.json `
  --source-reference ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1/source_zero.json `
  --out outputs/offline_renderer_repair_20261012/fresh_name
```

Optional `--patch` contains native bridge-style `{"bones":[...]}`. `--native`
contains all 59 values. `--apply-count` declares 1–32 calls, never a fitted
hidden variable. Output contains all source-order surfaces, pixels and triangle
visibility data. `geometric_visibility.png` uses categorical diagnostic colors;
it explicitly ignores shader alpha and is **not** the finished rendered face.
No executable, game HTTP, GPU or neural inference is invoked.

## Validation and remaining work

Five analytic tests cover perspective reprojection, clipping, near-depth order,
Reset history and CollectBaseline persistence; the reused ABMX suite has 28
tests. Existing native/source captures are checked without collecting samples:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tools/offline_renderer -p 'test_*.py' -v
.venv/Scripts/python.exe tools/offline_renderer/validate_saved_cases.py `
  --baseline-dir ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1 `
  --acceptance-dir ../HS2Mod/artifacts/chenger_native_inverse_20261011 `
  --out outputs/offline_renderer_repair_20261012/fresh_validation.json
```

Current baseline source-zero comparison passes the unchanged `1e-5` maximum
bbox-normalized gate for all ten surfaces. Native59 refresh is checked separately.
The older seven-bone/native acceptance has an eye pose different from the latest
capture: reference-eye errors are already present before the patch. They remain
reported and cannot certify cross-capture eye parity. No fitted correction is
applied. Full receipts preserve reference and candidate errors individually.

The RGB/material stage still has concrete source gaps: current GPU scene code
omits tear film and eyelid shadow, uses fitted ambient instead of actual SH,
disables translucency, lacks the real SSS postprocess, and does not implement
all runtime normal/tangent/detail and shader variants. Porting a few albedo
layers or feeding the categorical image to MediaPipe would not repair this.
Active breathing also requires its captured source time/FrameEffects; unknown
overlapping body effects remain explicit unsupported paths. These are pending
implementation/acceptance, not waived requirements.

The old card-only gray/textured scripts retain their legacy behavior and are
not installed-state compatibility entry points. The source-state CPU path can
already supply real parameter-controlled geometry for **3D reference fitting**;
the render → MediaPipe loop must wait for the remaining image pipeline repair.

## Normal-frame disagreement retained

An additional CPU audit found that source-mesh identity does not certify
blendshape-frame identity. The installed bundle has normal-only sparse entries
that an earlier native `GetBlendShapeFrameVertices` export does not retain.
For neutral `head.e00_defo`, the bundle has 35 entries and the API export has
five nonzero normal locations. No empirical pruning threshold was introduced.
Using actual API frames reduces the skinned-normal direction discrepancy from
about 0.186 to 0.00133, but neither input proves current normal compatibility.
The vertex-position gate remains passing; the normal/shader claim remains open.

```powershell
.venv/Scripts/python.exe tools/offline_renderer/audit_normal_frames.py `
  --baseline-dir ../HS2Mod/artifacts/chenger_reference_fit_20261012/source3d_v1 `
  --native-frame-export ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_trace/four_heads/head_2_native_ABMX.json `
  --out outputs/offline_renderer_repair_20261012/fresh_normal_audit.json
```

## Same-frame loaded normal repair — resumed 2026-10-12

The new native read-only export at `HS2Mod/artifacts/chenger_reference_fit_20261012/
native_resume_v1/live_full_frames_materials.json` contains actual loaded API
normal/tangent frames and BakeMesh normals in the same native frame (13378).
Four-weight **linear** direction skinning now agrees on all ten surfaces at the
unchanged `1e-5` direction threshold; the largest error is `2.93e-7`. This resolves
the earlier `0.00133` residual as a cross-pose comparison, without inverse-transpose
substitution, a fitted correction, or changed lighting.

`CapturedHeadForward.evaluate` and `hs2_render_cpu.py` now carry world normals,
world tangents and original tangent handedness when loaded API frames are present.
Position-only contexts remain usable for geometry, but explicitly lack RGB
attributes. Asset sparse frames are not silently treated as equivalent to loaded
API frames. The installed-bundle/API disagreement is retained, not threshold-pruned.
Tangent deformation is implemented; a BakeMesh tangent witness is not yet in the
bridge's saved evidence, so tangent equality is not claimed.

```powershell
.venv/Scripts/python.exe tools/offline_renderer/validate_loaded_attributes.py `
  --geometry ../HS2Mod/artifacts/chenger_reference_fit_20261012/native_resume_v1/live_full_frames_materials.json `
  --out outputs/offline_renderer_repair_20261012/fresh_loaded_normal_receipt.json
```

The compact same-frame receipt is committed at
`tools/offline_renderer/loaded_normal_validation_20261012.json`.
The current actual materials use six AIT shaders and `Custom/Silhouette`, not
Hanmen. These observations do not complete the RGB renderer or certify another
character's material pipeline.

## Installed shader execution stage — 2026-10-12

The remaining RGB stage now has an independent implementation path that executes
the **installed D3D11 bytecode** instead of replacing AIT with the legacy fitted
key/fill/ambient shader. `extract_shader_programs.py` preserves all compiled
branches, constant layouts, resource bindings and parsed pass state. It also
produces a typed native-read request including hidden `$Globals` outside the
Shader Properties list. Keywords must be selected exactly; unused material
keywords alone do not identify Unity's lighting/shadow variant.

`src/render/dxbc.py` is a float32/uint32 CPU instruction executor, including
masked branches, divergent loops, discard, Bayer integer operations and supplied
texture operations. `shader_bindings.py` packs compiled byte offsets and preserves
integer uniforms. Unknown instructions/resources/layouts fail; texture LOD and
lighting are not guessed. Numeric VM results do not imply hardware bit equality.

`d3d11_pass.cpp` independently runs original vertex/pixel DXBC through D3D11 WARP
(a CPU software driver), including supplied 2D/cube mip chains and real D3D sampler
filter/anisotropy, depth/cull/blend and MSAA primitives. The current single-pass
packet runner is a **component**, not the full multi-pass face renderer: complete
alpha state, depth sharing between passes, per-pass material overrides, 3D/compare
resources and the SSS/GrabPass/postprocess scheduling still need integration.

An installed `AIT/main eyeshadow lambert` program was actually executed with its
`DstColor Zero` blend on a declared analytic quad/texture fixture. WARP versus CPU
instruction output differs by at most `8.94e-8` (declared `1e-6` gate). This verifies
the executable pass, not a native face or MediaPipe input. The compact receipt is
`tools/offline_renderer/shader_execution_validation_20261012.json`; unsuccessful
fixture setup directories remain under ignored outputs.

Reproducible extraction and execution:

```powershell
.venv/Scripts/python.exe tools/offline_renderer/extract_shader_programs.py `
  --bundle E:/HoneySelect2_ArcticFox/abdata/chara/38/fo_head_38.unity3d `
  --out outputs/offline_renderer_repair_20261012/fresh_head_programs
g++ -std=c++17 -O2 -static-libgcc -static-libstdc++ `
  tools/offline_renderer/d3d11_pass.cpp `
  -o outputs/offline_renderer_repair_20261012/fresh_d3d11_pass.exe `
  -ld3d11 -ld3dcompiler -ldxguid
.venv/Scripts/python.exe tools/offline_renderer/validate_shader_execution.py `
  --programs outputs/offline_renderer_repair_20261012/fresh_head_programs `
  --backend outputs/offline_renderer_repair_20261012/fresh_d3d11_pass.exe `
  --out outputs/offline_renderer_repair_20261012/fresh_pass_validation
```

The installed DHH resource `DHH.Resources.sss` was recovered with read-only Cecil
resource extraction. Its actual `Hidden/LightingPass`, `Hidden/SSS_Profile`, and
`Hidden/SeparableSSS` programs are retained (including depth/normal/profile checks,
sample-count integer, blue-noise/dither and loop branches). The resource SHA is
`ad7c8f1a5140875b8314091d9583fcfaa095df6fe161c665546224bb7b926c8e`.
The original `SSS_convolution.OnRenderImage` schedules vertical then horizontal
blur passes for its actual iteration count, with radius scaled by initial/current
FOV and viewport diagonal / `2202.9072`. Merely applying a Gaussian RGB blur is not
that implementation and is not used here.

`Custom/Silhouette` comes from `chara/oo_base.unity3d` path ID
`6550598071349889345`, through the installed `ChaControl.LoadBody` simple-body
asset path; it is not an unknown AIT material or silently removed tongue entry.

### Independent read-only native audit helper

`tools/offline_renderer/native_audit/HS2_RenderAudit.csproj` builds a net472 DLL
with public API:

```csharp
Dictionary<string, object> HS2_RenderAudit.RenderAudit.Capture()
```

It reads config `HS2_RenderAudit.dll.json` beside the deployed DLL:

```json
{"shader_manifests":["ABSOLUTE/head/native_property_request.json",
                     "ABSOLUTE/sss/native_property_request.json",
                     "ABSOLUTE/silhouette/native_property_request.json"]}
```

The helper reads shared materials, requested typed uniforms (retaining unsuccessful
reads), native ambient/renderer-interpolated SH 3x9, global SH reads, property-block
presence, reflection-probe blend/texture/HDR descriptors, actual camera component
order/enabled states, and effect instance fields including current SSS hidden
state/convolution material. It invokes no setter, render/readback, material clone,
ABMX Apply or effect property getter. Built-in `GetGlobalVector` values and linear
color candidates are preserved without claiming they are final engine per-draw
cbuffer contents. Main-thread deployment/capture belongs to the main task alone.

Still required: effective property blocks, complete native texture mip/HDR/cube
data, exact per-draw/global color/probe/light/shadow state, pass scheduling and
shared depth, SSS lighting/profile/convolution integration, and final RGB native
acceptance. One ABMX Apply agreement is not persistent card/steady-state acceptance.

### Raw native texture transfer

The helper now has an optional native readback module, with config field
`raw_texture_output_directory` pointing at a fresh ignored directory.
On D3D11, Unity 2018's `Texture.GetNativeTexturePtr` returns `ID3D11Resource`
([Unity 2018 documentation](https://docs.unity3d.com/2018.4/Documentation/ScriptReference/Texture.GetNativeTexturePtr.html)).
Each actual material/global/effect/probe texture is AddRef'ed and queued once;
`GL.IssuePluginEvent` executes resource staging copies on Unity's render thread.
Neither texture formats nor source pixels are replaced. There is no Blit, PNG,
color-space conversion or reconstructed mip chain.

`HS2_RenderReadback.dll` maps every native mip/array face/volume slice and writes
tightly packed original-format bytes with a resource descriptor. HDR float/half,
BC blocks, cubemap arrays and Texture3D dimensions are retained. Unhandled formats
and MSAA depth/array resources remain failed jobs. Supported MSAA color resolve is
explicitly marked in the descriptor; sample-level data is not claimed retained.
Typeless/SRGB shader-view interpretation remains a separate native-state gap.

`Capture()` returns `raw_texture_export` with jobs/pending/complete/failed counts;
subsequent calls poll existing jobs instead of repeating exports. The standalone
`RenderAudit.PollRawTextures()` provides the same summary. Submitted field-capture
frame and render-thread timestamp stay distinct; completion does not certify
all resources were read in the earlier managed snapshot frame.

```powershell
dotnet build tools/offline_renderer/native_audit/HS2_RenderAudit.csproj -c Release
g++ -std=c++17 -O2 -shared -static `
  tools/offline_renderer/native_audit/d3d11_readback.cpp `
  -o tools/offline_renderer/native_audit/bin/Release/HS2_RenderReadback.dll `
  -ld3d11 -ldxguid
g++ -std=c++17 -O2 -static tools/offline_renderer/native_audit/test_readback.cpp `
  -o outputs/offline_renderer_repair_20261012/fresh_readback_test.exe `
  -ld3d11 -ldxguid
outputs/offline_renderer_repair_20261012/fresh_readback_test.exe `
  tools/offline_renderer/native_audit/bin/Release/HS2_RenderReadback.dll `
  outputs/offline_renderer_repair_20261012/fresh_readback_fixture
.venv/Scripts/python.exe tools/offline_renderer/validate_raw_resources.py `
  --root ABSOLUTE/native_raw_output --out outputs/offline_renderer_repair_20261012/fresh_resource_receipt.json
```

The native module was independently checked with WARP-created HDR16 cube textures
(six faces, four mips) and 3D textures (four mips). All **28 subresources** match
their original bytes on two consecutive exports; the original resources remain
unchanged. Receipt: `tools/offline_renderer/raw_resource_validation_20261012.json`.
Actual Unity plugin-event execution/material resources still require the main
task's one native capture; this is not yet full renderer or live-resource acceptance.

The first live render-thread export produced 43 complete resources (512,211,024
bytes), including native BC blocks, full runtime texture mip chains, a six-face
cube and HDR buffers. Its subsequent HTTP serialization failed on an opaque
`System.IntPtr` from an engine component wrapper field; the native resource files
and that failure remain preserved. Pointer/UIntPtr identities are now represented
explicitly by type and hex identity without dereferencing. Nonfinite scalar values
retain their IEEE bits/classification. The main task will repeat only the affected
read-only acceptance with this repaired helper; old process IDs are never rebound
to a new process merely by number or texture dimensions.

RenderDoc was not found locally. Source inspection confirms public read APIs for
D3D11 pipeline state, constant variables, buffer/texture bytes and replacements
([public replay API](https://github.com/baldurk/renderdoc/blob/v1.x/renderdoc/api/replay/renderdoc_replay.h)).
However, proxy-buffer creation/upload is an internal replay-driver facility,
not a ready public Python ReplayController method
([D3D11 implementation](https://github.com/baldurk/renderdoc/blob/v1.x/renderdoc/driver/d3d11/d3d11_replay.cpp)).
A capture could supply exact pass/cbuffer/SRV evidence, but fixed capture replay
is not an implemented general parameter renderer, and no game injection or
capture has been performed by this subtask.
