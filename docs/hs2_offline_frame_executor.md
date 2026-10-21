# HS2 offline renderer — frame-program executor (2026-10-20)

This is the executor layer of the [reimplementation architecture](hs2_offline_renderer_architecture.md).
Generators emit a **frame program**; a C++ D3D11 program executes it from exact bytes. A
mechanical converter turns a real capture into the same format, and that is how the executor
is accepted (ladder step 1).

**Result:** the whole retry frame (`unaccepted_retry_frame17675.rdc`, events 1–35086, all
16 capture renders) was converted and executed on the capture GPU. It equals the official
RenderDoc 1.46 replay **bit for bit at all 70 checkpoints**. The final target's top-down
bytes equal every RGBA pixel of the native `actual_main_camera.png`.

The executor does not use RenderDoc at runtime. No game was launched, nothing was injected and
no new capture was taken.

## Files

| file | role |
|---|---|
| `src/render/frame_program.py` | schema constants, `ProgramBuilder`, `validate`, content-addressed `BlobStore` (usage example at the top) |
| `tools/offline_renderer/frame_executor.cpp` | D3D11 executor (MSVC) |
| `tools/offline_renderer/build_frame_executor_msvc.py` | build into a fresh ignored directory, with a `build_receipt.json` |
| `tools/offline_renderer/capture_to_program.py` | official zip.xml capture → frame program up to an explicit event |
| `tools/offline_renderer/test_frame_program.py` | CPU tests of the schema, builder and validator |
| `tools/offline_renderer/test_frame_executor.py` | CPU converter tests; GPU fixture tests (opt-in); the `hs2-acceptance` runner |
| `tools/offline_renderer/frame_executor_acceptance_20261020.json` | compact acceptance receipt |

Bulky outputs are in the ignored `outputs/offline_renderer_executor_20261020/`:

- `build_v3/` is the accepted executor build.
- `acceptance_v1/` holds the first full run, with fresh oracle read-backs and raw files.
- `acceptance_v2/` is the final run.
- `trial_*` and `build_v1/v2` are earlier development runs, kept for the record.

## Contract changes (minimal; made in place in the architecture doc)

- **New `map_write` op.** D3D11 cannot `UpdateSubresource` a DYNAMIC or STAGING resource,
  and real captures upload through `Map`/`Unmap`. The op carries the map type and flags, the
  exact `[start,end)` range and its bytes.
- **Snapshot semantics made explicit.** Slot lists are sparse: an absent slot is unbound.
  Every action starts from `ClearState`.
- **`input_layouts[].signature_blob`.** This is the bytecode `CreateInputLayout` needs.
- **Readback outputs.** Optional buffer byte ranges are allowed. The output layout is
  specified.
- **Small clarifications.** An optional `provenance` (and `name`) field is allowed per entry.
  Ids are globally unique. `null` view desc is allowed. `texcube` means a Texture2D desc with
  the cube flag.

## Executor (`frame_executor.exe PROGRAM OUT [--warp] [--adapter-index N] [--debug-layer] [--hash-only]`)

Everything is created from the program's bytes and verbatim descriptors. JSON objects are read
strictly: an unknown field fails. Every blob is SHA-256 checked against its address before use.

- **Device.** By default it is the hardware adapter whose VendorId/DeviceId match the
  program's `provenance.capture_adapter`. RenderDoc also picks the best-matching adapter. If
  none matches, the run fails; nothing is substituted.
  - `--adapter-index` selects an adapter explicitly.
  - `--warp` creates a WARP device.
  - `execution.json` records the adapter, LUID and UMD driver version.
- **Initial data.** `initial` is passed as `D3D11_SUBRESOURCE_DATA` at creation. DEPTH_STENCIL
  resources cannot take creation data. For them the bytes go through a bind-less DEFAULT copy
  and `CopyResource`, using RenderDoc's depth→colour format mapping for its own initial-state
  textures. This is a bit copy.
- **Binding.** For each draw or dispatch:
  1. `ClearState`;
  2. bind outputs (RTVs/DSV, or CS UAVs with their initial counts), then OM and RS state;
  3. bind IA and per-stage shaders, CBs, SRVs and samplers;
  4. read the bound state back through `Get*`.

  If the runtime refused an SRV, UAV or OM binding (an input/output hazard inside the
  snapshot), the run fails. CB `{buffer, first_constant, num_constants}` uses
  `*SetConstantBuffers` for the legacy full range and `*SetConstantBuffers1` otherwise.
  CB `{blob}` gets an immutable buffer per blob.
- **`map_write`** reproduces pinned `Serialise_Unmap` replay exactly:
  - NO_OVERWRITE goes through an immutable temp buffer and `CopySubresourceRegion` at `start`.
  - Other types use a real `Map` with the recorded type (flags without DO_NOT_WAIT), then
    `MapIntercept::Init` and `CopyToD3D(start, end)`.
  - That includes its row/slice walk and the destination offset `start` in D3D memory.
- **`update_subresource` / `copy_*` / `resolve` / `generate_mips` / clears** are the
  corresponding D3D11 calls with the recorded arguments. A `copy_flags` value selects the
  D3D11.1 variant.
- **Readbacks** run after their command and follow `D3D11Replay::GetTextureData`:
  1. a STAGING copy of the descriptor (bind/misc flags cleared);
  2. `CopyResource`;
  3. `Map(READ)`;
  4. `MapIntercept::CopyFromD3D` tight packing.

  Buffers use a staging buffer. The run writes `<name>.raw.bin` and, for
  `unity_texture_rows_bottom_up` non-block 2D targets, `<name>.topdown.bin`.
  `execution.json` holds SHA-256 values, formats, extents and timings. `--hash-only` records
  the hashes without writing raw files.
- **Explicitly unsupported (fails):**
  - OM UAVs, stream-out, predication;
  - MSAA initial data and MSAA readback (no implicit resolve);
  - class linkage, rasterizer desc 1/2;
  - `MAP_READ_WRITE`, NO_OVERWRITE on textures;
  - unknown DXGI storage formats.
- `--debug-layer` needs the D3D11 SDK layers. They are not installed on this machine, so the
  executor's own `Get*` checks and HRESULT checks are the guards.

## Converter (`capture_to_program.py`)

Inputs:

- the official zip.xml and its ZIP;
- the official driver listing (event ids);
- optionally, the frame graph's `actions.json` (cross-check) and installed shader manifests
  (exact-DXBC source identities, recorded in `shaders[].source.installed_matches`).

It reuses `frame_graph.load_chunks`, `Blobs`, `ShaderLibrary` and the driver listing. It stops
after the requested event. Unknown constructs raise `Unconvertible`.

### Capture-start content (pinned `d3d11_initstate.cpp`, `resource_manager.h`)

| class | rule | HS2 count (full frame) |
|---|---|---|
| `capture_start_createbuffer_record` | buffer: `CreateBuffer` InitialData holds the capture-start bytes | 183 |
| `initial_contents_chunk` | listed texture with an `Internal::Initial Contents` chunk; its RowPitch (and DepthPitch for 3D) are kept verbatim | 32 |
| `renderdoc_clear_rtv_zero` | listed, first NeededInitials record `written=false`, MipLevels 1, RENDER_TARGET → `ClearRenderTargetView(0,0,0,0)`. The view is `GetTypedFormat`, TEXTURE2D/2DMS/3D, created as RenderDoc does (including on 4 MSVO arrays) | 69 |
| `renderdoc_clear_dsv_one_zero` | same, DEPTH_STENCIL → `ClearDepthStencilView(DEPTH|STENCIL, 1.0, 0)` via `GetDepthTypedFormat` | 27 |
| `creation_data` | creation `SubresourceContents`, tightly packed (`Serialise_CreateTextureData`) | 87 |
| `pre_capture_full_update` | no creation data, but the pre-capture `UpdateSubresource` with IsUpdate=false carries the full subresource. These are Unity's default 3D textures 69/84 | 2 |
| `undefined_at_creation_listed_copy_of_live` | listed, no chunk, no data: RenderDoc copies the live (undefined) resource | 17 |

- The first NeededInitials record of a resource wins, because later records see existing
  initial contents.
- Initial-content chunks of unlisted resources are freed by RenderDoc and are ignored here.
- Pre-capture `UpdateSubresource` calls run at load, before the copy-of-live snapshot. They
  are emitted as the first commands.
- RenderDoc's start-of-frame clears follow them, marked `provenance.initial_state`.

### Frame commands

- **Unmap.** Becomes `map_write` with the exact recorded range and bytes. An empty range does
  nothing on replay, so it is skipped and counted: 126 in the full frame. Maps issued before
  the capture still apply their Unmap bytes.
- **UpdateSubresource.** IsUpdate=true keeps the recorded box and pitches. IsUpdate=false is
  replayed without a box, with pitches `GetRowPitch(subWidth)` and
  `GetByteSize(subWidth, subHeight, 1)`. `CopyFlags` `~0` means plain `UpdateSubresource`.
- **Bindings** are tracked with the pinned `D3D11RenderState` rules:
  - `ChangeRefRead`: an SRV whose range is bound for write is forced to NULL.
  - `ChangeRefWrite`: binding an output unbinds overlapping reads and other writes.
  - `ValidOutputMerger`: invalid OM calls are ignored.
  - `ResourceRange`: includes its 4/12-bit fields and the depth/stencil read-only cases.

  Each resolution is logged. The HS2 frame has none.
- **Draws and dispatches** snapshot the full resolved state. A NULL blend factor becomes
  `{1,1,1,1}`, per the D3D11 spec. CS UAV initial counts are passed only on the first
  dispatch after the bind that set them.
- **Event queries.** `End` on an EVENT query has no GPU effect and is skipped and recorded.
  Any other query type, and any predication, is rejected.
- **Readback binding.** `(event, resource)` binds to the last command at or before the event.
  This matches RenderDoc `SetFrameEvent`, which replays through the event inclusive.

Only referenced objects are emitted. These are the commands, snapshots and readbacks, the
views' resources, and the resources reset by the start-of-frame clears. Unreferenced
resources cannot affect results; their count is reported.

`--link-blobs-from` hard-links identical blobs from an earlier program instead of rewriting
about 500 MB.

## Acceptance (ladder step 1)

Runner: `python -m tools.offline_renderer.test_frame_executor hs2-acceptance ...` (see Reproduce).

**Checkpoints.** There are 23 per capture segment, taken from the frame graph's passes. Each
is the last event of the pass, on the pass's own output resource:

| group | checkpoints |
|---|---|
| SSS camera | shadow map 14283; LightingTexture 59360; first blur temp 59388; SSSLightingTextureBlurred 59396 |
| capture camera | depth texture 59404; shadow map 14283; screen-space shadows 59416; forward opaque colour 59330 and depth 59334; motion vectors 59420 |
| MSVO and forward passes | MSVO occlusion 59338; colour after the AO composite; after the Liquid draw; after the transparent draws |
| PPv2 | DoF 59648; 3D colour-grading LUT 14970; Uber 59644; SMAA 59648; FinalPass 59330 |
| CTAA | CTAA_Enhance; CTAA_PC (59680, or 59676 in later iterations); the CTAA output blit 59684 |
| final | capture target 59322 |

**Programs:**

- *segment 0*: events 1–2259, 492 commands, 23 checkpoints.
- *whole frame to event 35086*: the readback copy of 59322, 6224 commands (2481 draws,
  176 dispatches, 3148 map writes). Checkpoints: segment 0 + segment 15 + the final 59322 at
  35086, 47 in all.

**Oracle.** `render_patched_capture.ReplaySession` with the identity plan, one job per
program, the same RDC on the same GPU (RTX 4080 Laptop, UMD 32.0.16.1714).

**Result:**

- **70/70 checkpoints bit-exact.**
- Repeated executor runs are byte-identical.
- The final target equals the earlier `original_target.bin` hash, and its top-down bytes
  equal the native PNG.
- The frame-graph cross-check matched every converted GPU action's event and kind: 6126
  actions, 0 mismatches.

Run 1 (`acceptance_v1`) used executor build_v2. It took fresh oracle read-backs and wrote
raw files.

The final run (`acceptance_v2`, receipt in the repo) used:

- the final converter and executor build_v3 with `--hash-only`;
- hard-linked blobs;
- the v1 oracle read-backs, reused only after checking the RDC hash, the identity plan, the
  replay exe/DLL hashes, the exact target set, and re-hashing every raw file.

The C: drive had about 0.5 GB free during this work. Its programs' executed content is
identical to v1's; the only change is two provenance labels, for resources 69 and 84.

**WARP** runs the segment-0 program without error. It is a different device and has no WARP
oracle, so its bytes are not an acceptance criterion: 3 of 23 read-backs equal the hardware
ones.

Execution cost, warm, full frame:

- device 0.15 s;
- creating 417 resources from about 500 MB of verified blobs, 0.7 s;
- the commands, about 0.15 s;
- 47 read-backs, 0.45 s.

The first run after the blobs are written is about 11 s slower, from cold reads of
newly written files.

Observations from read-back hashes (identical on both sides; the mechanism is not analysed here):

- the first SSS blur temp (59388 after event 579) equals the final SSSLightingTextureBlurred;
- the Liquid draw leaves 59330 unchanged in both checked segments;
- the DoF output 59648 equals the forward colour before it.

These are inputs for the SSS and PPv2 generator owners.

## Remaining gaps (explicit)

1. **Undefined capture content on the Uber path.** Unity writes each of the 16 "Internal
   Curves Texture" staging uploads (59343, 59688 … 59898) with `Map(WRITE)`. RenderDoc
   recorded only bytes `[1, 2048)`: its diff found byte 0 unchanged. The staging textures
   have no captured contents.

   Byte 0 is the low byte of the first half-float of curve row 0. It is therefore the
   creation-time (undefined) content in both the oracle and the executor. They agree because
   both use fresh allocations. The game's true value is not in the capture.
   `conversion_report.json` lists each case as `undefined_content_dependencies`.
2. **One capture, one GPU and driver.** Bit-exactness is established for this adapter/UMD
   only. WARP and other GPUs have no oracle here.
3. **Unsupported features.** These are rejected rather than approximated:
   - OM UAVs, stream-out, predication, non-event queries, deferred contexts;
   - MSAA initial data and readback;
   - Texture1D initial-content chunks;
   - `MAP_READ_WRITE`.

   None occurs in the HS2 frame.
4. **Debug layer.** The D3D11 debug layer is not installed, so invalid-but-silent API usage
   in a *generated* program is caught only by the executor's own checks. These are strict
   fields, blob hashes, binding read-back, HRESULTs and device-removed. `--debug-layer` exists
   for machines that have the layers.
5. **Unity camera binding** is a frame-graph gap. It stays open: the checkpoints are passes
   identified by recorded bytes, not Unity objects.

## Reproduce

```powershell
python tools/offline_renderer/build_frame_executor_msvc.py --vcvars "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat" --out outputs/offline_renderer_executor_20261020/FRESH_BUILD
python -m tools.offline_renderer.test_frame_program
$env:HS2_FRAME_EXECUTOR='ABS\FRESH_BUILD\frame_executor.exe'; python -m tools.offline_renderer.test_frame_executor
python tools/offline_renderer/capture_to_program.py --xml ...\retry_parser_v1\retry.zip.xml --inspect ...\retry_driver_inspect_v1\inspect\stdout.bin `
  --frame-graph-actions outputs/offline_renderer_frame_graph_20261020/frame_graph_v2/actions.json --through-event 35086 `
  --readback final:35086:59322:0:unity_texture_rows_bottom_up --out FRESH_PROGRAM
FRESH_BUILD\frame_executor.exe FRESH_PROGRAM FRESH_OUT            # add --warp for WARP
python -m tools.offline_renderer.test_frame_executor hs2-acceptance --executor EXE `
  --tool outputs/offline_renderer_pipeline_20261020/replay_msvc_v2/renderdoc_replay_msvc.exe `
  --renderdoc-dir outputs/offline_renderer_repair_20261012/renderdoc_portable_1_46/extracted/RenderDoc_1.46_64 `
  --rdc ...\unaccepted_retry_frame17675.rdc --index outputs/offline_renderer_pipeline_20261020/hs2_retry_frame17675_v1/index.json `
  --xml ...\retry.zip.xml --inspect ...\stdout.bin --frame-graph outputs/offline_renderer_frame_graph_20261020/frame_graph_v2 `
  --native-png ...\capture_v1\actual_main_camera.png [--shader-manifest M ...] --out FRESH_DIR --receipt RECEIPT.json
  # disk-light re-run: add --hash-only --oracle-reuse PREV --link-blobs-from PREV --previous-programs PREV
```
