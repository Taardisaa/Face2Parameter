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

### Actual v3 resources and native input assembly

The main task's repaired helper completed **44 raw jobs, zero pending/failed**
(517,971,024 original-format bytes) in `runtime_audit_hook_v3`. The geometry
export in that folder is from the same process; prior-process instance IDs are
not rebound. All eight actual head renderer manifests are covered and their
property-block presence checks are false. The main task's public card/ABMX/
expression/renderstyle checks and restoration receipts remain the authority.

`src/render/native_inputs.py` implements Unity's exact SH polynomial coefficient
packing, captured Color versus Vector semantics, typed native-resource views,
and the installed DHH SSS vertical/horizontal pass schedule. Current captured
SSS fields generate four passes and the same float32 radius as the actual
convolution material. Unknown typeless view interpretation, hidden setter
semantics and missing raw data raise explicit errors. Material/global getters
returning zero are not used as a substitute for the engine's per-draw cbuffers.
The actual head interpolated probe SH is nonzero; these coefficients are retained.

CPU verification: `python -m unittest tools.offline_renderer.test_native_inputs
tools.offline_renderer.test_dxbc tools.offline_renderer.test_cpu` — 14 tests pass.
This is input assembly, not full native RGB acceptance. Complete draw/pass state,
shared depth, SSS/profile/GrabPass/postprocess execution and active shader variant
selection remain required.

### RenderDoc portable preparation and retained target-client failure

`tools/offline_renderer/prepare_renderdoc.py` downloads the pinned official
portable RenderDoc 1.46 archive and records its URL/hash. The portable package
contains Python 3.8 DLL/stdlib; this is not a binding compatible with the
workspace's Python 3.13. The external C++ prototype uses the version-matched
official headers and DLL rather than guessing a Python binding.

`prepare_renderdoc_headers.py --version 1.46 --out FRESH_IGNORED_DIRECTORY`
recursively obtains the official release's 21 replay headers, preserves licenses,
records per-file source URL/hash, and writes the explicit six-export import
library definition. It downloads data only, without loading any native DLL.

An independent WARP analytic installed-shader pass produced a real capture and
official XML+ZIP conversion. `export_renderdoc_xml.py` preserves every chunk
and original data blob (the fixture has 47 chunks and 17 blobs), including
uninterpreted branches. The fixture's capture is not HS2 evidence and does not
demonstrate parameter/resource replacement or the complete RGB renderer.

**The external target-control prototype is NOT accepted and must not be used
with HS2.** Enumeration succeeded and the official log records successful
handshakes, but the client then failed before emitting its connected message.
The two retained logs show `renderdoccmd.exe --crash` servers; the user's bug
reporter observation is consistent with this failed client path. CPU-only
disassembly locates an ABI defect at the first `GetTarget` call: the GNU caller
passes its output `rdcstr` in RCX and `this` in RDX. The official Windows DLL uses
the MS C++ ABI, which requires `this` before the aggregate return buffer
([LLVM Microsoft ABI implementation](https://github.com/Microsoft/clang/blob/master/lib/CodeGen/MicrosoftCXXABI.cpp)).
The source now explicitly supplies that ordering for `GetTarget`, `GetAPI` and
`ReceiveMessage`; a CPU syntax-only compilation passed. This source repair has
not received runtime acceptance, and no exact popup/crash-dump identity was
captured. The failure is not described as a pure network timeout. No new
reproduction was attempted after the stop instruction, and no RenderDoc process
was injected into HS2.

The experiment is isolated in `renderdoc_target.cpp` and
`renderdoc_control_fixture.cpp`; its failed acceptance metadata is
`tools/offline_renderer/renderdoc_target_failure_20261012.json`. Complete logs,
portable tools, source headers, binaries and captures are under the ignored
`outputs/offline_renderer_repair_20261012` tree. The read-only process inventory
after stopping found no live RenderDoc client, fixture or bugreporter. Other
applications and the game's UnityCrashHandler were left untouched. While the
user is gaming, work continues on CPU data/source only; graphically executing
this prototype requires a repaired ABI path and later explicit native-session
authorization. The full renderer goal remains unfinished.

### MSVC client source repair; graphical acceptance remains pending

After the user resumed GPU work, one necessary independent corrected-GNU-client
check reached an external 25-second timeout. The experiment's timeout handler
then incorrectly used `write_bytes` on a string and lost its buffered output.
That output is not reconstructed. The launch/enumeration/readiness files and
terminal exceptions are retained, and a read-only inventory found no remaining
related processes or newer TEMP/RenderDoc crash log. This timeout is not claimed
to be a proven repeat of the first crash. No new graphical attempt followed it.

CPU source inspection identified another concrete cross-toolchain defect:
`RENDERDOC_ProgressCallback` is `std::function<void(float)>`, not a plain function
pointer. The same official headers produce **64 bytes with MSVC and 32 bytes
with GNU**, while `rdcstr` and `TargetControlMessage` agree at 24 and 240 bytes.
Correcting only the member return ordering does not make the callback ABI
compatible. The target client now requires the installed MSVC x64 toolchain and
uses the official C++ interface directly; GNU compilation is rejected.

`build_target_client_msvc.py` compiles without executing a RenderDoc DLL, client,
or graphics program. The native MSVC assembly confirms `this` in RCX, output in
RDX and the MSVC callback object in R8. The independent CPU ABI mock passed for
24/512-byte nontrivial results and explicit argument identities. No repaired
MSVC client has yet received graphical/runtime acceptance; it remains unusable
for HS2 until that separate check is authorized and passes. Receipt:
`tools/offline_renderer/renderdoc_target_msvc_cpu_20261012.json`.

`logged_process.py` now directs output into binary files before process startup,
retains it on timeout/nonzero exit and records status/hashes. Its three CPU
regressions pass, including UTF-8 progress/error output surviving a timeout.
It never retries or stops a process discovered by name, only its own supplied
child. This repairs the evidence loss without inventing the lost earlier output.

### CPU interpretation of recorded draw inputs

`src/render/captured_pipeline.py` now interprets the original XML+ZIP's native
initial context state and ordered D3D11 resource/binding commands. It retains
input layouts, CB byte ranges, original textures/views/samplers, all shader
stages, depth/stencil/blend/raster states, viewport data and draw arguments.
Executed vertex/pixel DXBC bytes are identified by exact hash against installed
shader exports; identical programs with several keyword labels retain every
matching source identity instead of claiming one guessed active keyword set.

The pre-existing independent capture has 47 interpreted commands, 20 resources
and one draw. Nine original byte inputs (VS, PS, vertex/index data, four CBs and
texture pixels) match the declared original packet exactly. CPU-only command:

```powershell
python tools/offline_renderer/validate_capture_inputs.py `
  --manifest ABSOLUTE/exported_inputs/manifest.json `
  --packet ABSOLUTE/original_packet `
  --shader-manifest ABSOLUTE/shader_programs_v2/manifest.json `
  --out ABSOLUTE/fresh_receipt.json
```

Receipt: `tools/offline_renderer/captured_inputs_validation_20261012.json`.
This does not launch a graphical process. It is input interpretation, not
completed frame execution. Written initial-content payloads, dynamic updates,
deferred command lists, SRV/output subresource hazards and predicated query
results remain explicit rejected branches. Clear/copy/resolve commands are
retained for execution; the interpreter does not manufacture their pixels or
reuse stale creation bytes as updated CB contents. Shared targets and full
SSS/GrabPass/postprocess integration remain part of the unfinished renderer.

### Dynamic buffer lifetime in the CPU input interpreter

The pinned RenderDoc 1.46
[D3D11 serialization source](https://github.com/baldurk/renderdoc/blob/v1.46/renderdoc/driver/d3d11/d3d11_context_wrap.cpp)
defines mapped writes in `Serialise_Unmap` as MapType/MapFlags, the exact start/
end byte range and `MapWrittenData`. `Serialise_Map` only records bookkeeping;
actual transfer occurs at Unmap. The interpreter now applies those buffer writes
and buffer CopyResource operations in order and retains immutable byte versions
at every draw. CB reads use that draw's version, preventing later writes from
silently changing an earlier draw's parameters.

Discarded/uninitialized ranges remain unknown: no zero-filled payload is
manufactured. A draw input that depends on uncaptured bytes fails explicitly.
CPU regressions cover partial/no-overwrite preservation, earlier-draw retention,
overlapping latest writes and discarded undefined ranges. Three tests pass;
the pre-existing capture still passes all nine exact original-byte checks.
Texture map/update transfer, written initial payloads, GPU-written buffer
lifetimes and complete frame/SSS/GrabPass/postprocess execution remain required.
No graphical process was started for this implementation or validation.

### Texture transfers and shared-target dependencies

The CPU interpreter now retains every repeated `SubresourceContents` field in
the original capture, in actual mip/array order. Previously a multi-mip creation
would be rejected as duplicate XML fields. Original encoded bytes remain encoded;
there is no PNG conversion, BC decoding, color conversion or mip reconstruction.

`texture_storage.py` follows pinned `Serialise_CreateTextureData`, `MapIntercept`
and `UpdateDataSize` source: 1D/2D/3D and array layouts, block-format row sizes,
tightly packed Map/Unmap ranges, pitched boxed UpdateSubresource updates, idle
full-subresource contents and exact-descriptor CopyResource transfers. Row/slice
padding is excluded from texels. Discarded/uninitialized ranges stay unknown.
YUV layouts, unaligned encoded-block boxes and reinterpret-format copies remain
explicit unsupported branches.

`shared_targets.py` tracks resource/subresource identities and clear/copy/resolve/
potential draw producers. Subsequent SRV inputs refer to the appropriate producer,
including shared depth dependencies. Write-disabled depth and color masks retain
their earlier versions. Clear values and draw dependencies are executable command
data, not manufactured texture pixels: actual frame execution is still required.
Partial-volume render views and SRV/output overlap handling remain unresolved.

CPU verification: seven texture/resource checks plus three buffer lifetime
checks pass; the pre-existing capture still passes nine original-byte comparisons.
No game, RenderDoc or graphical fixture was started for these checks.

Source inspection also found a raw-exporter format enum defect: `R1_UNORM` is
DXGI **66**, while `R9G9B9E5_SHAREDEXP` is **67**. The source now assigns bit-packed
pitch to 66 and four bytes per texel to 67. The 44 actual v3 resource descriptors
contain neither format, so their recorded raw bytes are unaffected. These repaired
format branches have CPU layout checks but no new native readback acceptance;
the installed module was not replaced.

### Target-control acceptance guards

The pinned target server sends initial capture notifications asynchronously, one
per tick; `ReceiveMessage` can return Noop before they arrive. The client no longer
uses Noop as a completed-list barrier. An explicit trigger now requires a fresh
caller-owned filename prefix, no pre-existing matching files, and accepts only
a local D3D11 result whose actual new `.rdc` path/byte count and timestamp agree.
The eight CPU policy checks cover old timestamps, wrong API/local flag/paths,
incorrect file size and prefix reuse. No RenderDoc DLL is loaded in these checks.

`validate_target_control.py` inventories exact related PID, parent PID, executable,
command line and window title before and after its single explicit standalone
attempt. It verifies binaries against the MSVC build receipt and uses the binary
logger with bounded readiness/enumeration/client deadlines. It does not retry,
force a connection, launch HS2 or kill a process by name. Only its owned Popen
launcher/client may be stopped on deadline; surviving helper/fixture processes
remain explicitly reported for ownership review. The controlled fixture is 32x32.

Remaining runtime boundary: the repaired MSVC target interface and complete frame
rendering have not yet passed a graphical check. One independent controlled
fixture check is authorized by the main task after these CPU repairs are saved;
it must not be called HS2 or full RGB acceptance.

The one authorized MSVC standalone check subsequently **passed**: independent
32x32 WARP process PID52140, D3D11 frame73, fresh 2603-byte capture, exact file
size/path/timestamp match, client exit0 and empty stderr. Both before and after
related-process inventories are empty; the fixture exited normally. No retry,
game launch/injection or HS2 control occurred. The prior failures remain intact.
Receipt: `tools/offline_renderer/target_control_validation_20261012.json`.

Verified client: ignored `controlled_msvc_v1/renderdoc_target_msvc.exe`, SHA256
`96803069d054923dcf88086c99b9dad4df49087cd74d580a0a6344b0a7cc83f4`.
The calling process must put the pinned official portable directory on PATH
(the client uses its `renderdoc.dll`) and preserve direct binary stdout/stderr
logs. For a main-task-owned native process launched with its fresh capture path
template, the supported commands are:

```text
renderdoc_target_msvc.exe list
renderdoc_target_msvc.exe IDENT EXPECTED_PID trigger DEADLINE_SECONDS ABSOLUTE_FRESH_CAPTURE_PREFIX
```

The main task alone chooses/launches HS2, waits for the actual Maker/card/SSS
state, obtains its PID/target ident and triggers its one native frame. This
standalone check validates the repaired control interface, not the pending
complete scene/SSS/GrabPass/postprocess renderer or its native RGB agreement.

### Scoped original main-camera command capture

An independent strict-C-ABI native helper and net472 shim are now implemented and
CPU-checked against the pinned official application SDK. They only obtain an
already-loaded RenderDoc API, guard existing capture state and capture the
parent-selected actual native RT device. The installed bridge narrow-hook
contract surrounds its original frozen sixteen renders and synchronized saved
PNG; all SSS/postprocess and original restoration remain the bridge's computation.
Detailed deployment/ownership/API limits:
[scope_capture_contract.md](../tools/offline_renderer/native_audit/scope_capture_contract.md).

Twenty-two official-table CPU mock checks and managed opt-in rejection checks
pass. Native imports exclude RenderDoc/D3D11; official C table offsets are static
assertions. Runtime game scope acceptance remains pending. Actual fresh RDC file
metadata and SHA can be recorded, but driver/frame contents require subsequent
RDC parsing because the C SDK does not expose them. The API cannot distinguish an
unobserved external discard/restart; one capture writer remains required.

Next implementation uses official full driver replay as the baseline shader/
SSS/pass execution oracle and changes source-derived dynamic geometry bytes with
all shadow/motion/previous-cache dependencies accounted for. It does not reduce
the intended renderer to unchanged fixed-capture replay or a selected pass.

### Official full-driver execution and candidate byte transfer

The pinned v1.46 `CaptureFile`/XML codec source confirms full roundtrip support
uses **`zip.xml`**, not XML-only `xml` (which cannot be imported). The source is
retained with hashes in ignored `renderdoc_source/replay_codec_sources.json`.
Official source URLs:
[capture_file.cpp](https://github.com/baldurk/renderdoc/blob/v1.46/renderdoc/replay/capture_file.cpp),
[xml_codec.cpp](https://github.com/baldurk/renderdoc/blob/v1.46/renderdoc/serialise/codecs/xml_codec.cpp).

`renderdoc_replay_frame.cpp` now implements explicit full XML+ZIP/native conversion
and local D3D11 frame execution through the official driver. MSVC compilation
passed; no RenderDoc DLL/tool or GPU was run for this milestone. The public
interface uses no GNU C++ ABI. It verifies the pinned DLL commit, reads the actual
recorded driver, uses NoOptimisation, and requires an explicit existing event and
actual texture identity. It never selects a convenient substitute target.
It can save the original final 2D single-sample target with official PNG conversion
and official texture readback; transfer/layout/native PNG agreement remains to be
accepted with the actual scoped capture.

```text
renderdoc_replay_msvc.exe convert ORIGINAL.rdc rdc ORIGINAL.zip.xml zip.xml
renderdoc_replay_msvc.exe convert CANDIDATE.zip.xml zip.xml CANDIDATE.rdc rdc
renderdoc_replay_msvc.exe inspect ORIGINAL.rdc
renderdoc_replay_msvc.exe replay ORIGINAL.rdc EVENT_ID RESOURCE_ID OUTPUT.png OUTPUT.bin
```

The commands above are interfaces, not instructions to start a new game/capture.
Main alone authorizes/owns runtime execution and keeps logs/deadlines. Generated
MSVC tool SHA/build receipt: ignored `official_full_replay_msvc_v3/`.

`capture_mutation.py` transfers explicitly witnessed source bytes into one
specific original buffer occurrence, including data attached to CreateBuffer or
recorded dynamic updates. Shared blob references use copy-on-write; every original
blob and unknown command is retained. Index/resource IDs, byte lengths and all
other structured values are invariant. Hashes bind original capture, candidate
bytes and replaced original range; mismatched/overlapping patches fail. Pinned
codec zero-padded numeric ZIP members and thumbnail/log/D3D12 literal sections
are preserved; the existing structured exporter now also retains these valid
auxiliary sections instead of rejecting a legitimate full capture.

CPU acceptance on the pre-existing independent analytic capture preserved all
47 chunks and all 17 original ZIP members while changing exactly one diagnostic
first vertex word in a new isolated blob. Three regression cases with five
rejection subcases cover shared-blob isolation, auxiliary metadata preservation,
source hashes/selectors and overlap. Receipts preserve pending native roundtrip
and candidate/full-RGB status; no HS2 card was modified.

Remaining source integration is substantial: bind all ten actual surfaces to
every GPU resource lifetime/write; reproduce position/normal/tangent and previous
geometry, bone, shadow and motion inputs; determine culling/draw-list invariance
and every dependent cache; restore camera/material control through actual inputs;
accept official baseline replay and candidate RGB against native pipeline. The
full-driver baseline is the execution oracle, not the final parameter renderer.

The compiled official replay tool also has `drawinputs RDC EVENT NEW_DIRECTORY`.
For one explicitly selected actual draw, it exports every bound vertex/index
buffer's actual replayed bytes, resource identity, stride/offset, semantic layout
and actual draw arguments. This is the required source-correspondence input;
shader constant/SRV/pass data remain in the complete original capture. It does
not guess a surface by appearance or declare previous/shadow/motion dependencies
covered. Source/compile only, no runtime acceptance yet; newest ignored tool build
is `official_full_replay_msvc_v5/`. Its input receipt scope is specifically vertex/
index data, not all constant/texture inputs.

### Same-freeze geometry/RDC evidence

The optional geometry pair is implemented in the existing net472 scope helper.
It calls the exact loaded installed native exporter immediately before SDK Start
and after successful SDK End/PNG save, before original restoration. Both fresh
paths and the loaded narrow-hook bridge SHA are explicit opt-in fields. Native
serializer output, full actor transforms/blendframes/native-driver diagnostics
and file/frame/timestamp provenance are preserved. Failed before/after exports
reject pair completion, preserve evidence and retain owned-only abort behavior.
The three public hook signatures remain unchanged.

CPU protocol fixture variants validate the installed internal type/default
constructor IL, exact five-parameter method, optional legacy path, pinned SHA,
freshness, exact arguments, paired scalar metadata and before/after failures.
Native scope acceptance remains pending; no game or graphics were started. The
installed exporter lacks baked tangents, so actual GPU tangent correspondence is
still a separate required proof. See the scope contract for deployment hashes/
paths and the warning against deploying synthetic fixture bridge assemblies.

### Doorstop/RenderDoc early startup compatibility

The main-task-owned official injected HS2 attempt reached native D3D11 vanilla
menus, but not BepInEx/bridge initialization. Zero captures were accepted; owned
processes were stopped. Upstream Doorstop issue24 and pinned source explain the
competing GetProcAddress IAT hooks. An independent mscorlib-only net472 bootstrap
now loads the exact official RenderDoc library **inside Doorstop's already-entered
Mono invocation**, before the original installed BepInEx entry assembly loads.

All original entry calls verify the installed preloader SHA and invoke its actual
public static Main; its resolver/platform/runtime-fix/preloader chain is retained.
The correct original DOORSTOP_INVOKE_DLL_PATH is restored and timed. Disabled/no
request executes the original entry only; failed opt-in still runs the original
entry and explicitly rejects capture compatibility. Fresh receipts retain exact
load/order/errors; original files and feature settings are untouched.

CPU build/metadata and nineteen synthetic entry/sequence checks pass. No game,
RenderDoc DLL, installed preloader or graphics was executed by this child. Late
startup guards and full plugin/native capture acceptance remain pending. Main
alone controls config deployment/restore and the next actual startup.
Contract/provenance: [early_bootstrap/contract.md](../tools/offline_renderer/native_audit/early_bootstrap/contract.md).

The main task also performed one actual **parser/export-only** run of the repaired
MSVC replay tool against the existing independent32x32 WARP capture. Explicit
`convert ... rdc ... zip.xml` passed with exit0/empty stderr and empty related
process inventories before/after. Source RDC SHA18eafcb2..., tool SHA076aae9c....
No OpenCapture, GPU driver replay or HS2 was executed. Receipt:
`tools/offline_renderer/official_parser_validation_20261012.json`. This validates
the official C++ file parser/export interface; full replay, modified capture
roundtrip, actual HS2 capture and candidate RGB remain unaccepted.
