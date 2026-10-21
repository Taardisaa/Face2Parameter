# HS2 offline replay pipeline: patch → capture → official replay → image

2026-10-20. This is the transport/execution layer of the offline renderer. A real
captured HS2 frame (RenderDoc 1.46 RDC) is executed by the **official D3D11
replay driver** after candidate bytes are substituted at their real upload
points. Every recorded shader, SSS, GrabPass, shared depth, postprocess and copy
runs unchanged. *Which* bytes to substitute (frame graph, head GPU skinning
payloads, previous/shadow/motion dependencies) is decided by the frame-graph and
skinning work. Nothing here certifies that a given patch set is a complete head
candidate; receipts keep `candidate_geometry_complete=false` and
`full_renderer_accepted=false`.

## Entry point

`tools/offline_renderer/render_patched_capture.py`, used as a library or CLI.

```powershell
$T = 'ABS\replay_msvc_v2\renderdoc_replay_msvc.exe'      # MSVC build with build_receipt.json
$R = 'ABS\renderdoc_portable_1_46\extracted\RenderDoc_1.46_64'
# once per capture: official offset index
python tools/offline_renderer/render_patched_capture.py --tool $T --renderdoc-dir $R index --rdc CAPTURE.rdc --out NEW_INDEX.json
# one job (fast route; --route official|both for the reference / equivalence routes)
python tools/offline_renderer/render_patched_capture.py --tool $T --renderdoc-dir $R render --job JOB.json --out NEW_DIR
```

For a fitting loop, keep one session and call `render` per candidate:

```python
import render_patched_capture as rpc
tools = rpc.Tools(T, R)
with rpc.ReplaySession(tools, rdc, index_json, fresh_session_dir) as s:
    for k, plan in enumerate(candidate_plans):            # plan=None renders the unmodified capture
        receipt = s.render(plan, targets, out_root / f'cand{k:04d}')
```

### Job spec (`hs2_patched_capture_job/1`)

```json
{"schema": "hs2_patched_capture_job/1",
 "capture_rdc": "ABS.rdc", "capture_rdc_sha256": "...", "index": "ABS/index.json",
 "plan": "ABS/plan.json or null", "official_zip_xml": "ABS.zip.xml (official/both only)",
 "keep_candidate_rdc": false,
 "targets": [
  {"name": "camera", "kind": "texture", "event_id": 35086, "resource_id": 59322,
   "orientation": "unity_texture_rows_bottom_up", "mip": 0, "slice": 0, "png": true},
  {"name": "sss_input", "kind": "texture", "event_id": E, "resource_id": RID, "orientation": "unity_texture_rows_bottom_up"},
  {"name": "skin_vb", "kind": "buffer", "event_id": E, "resource_id": RID, "byte_offset": 0, "byte_length": 0}]}
```

Targets are explicit `(event, resource)` pairs. There is no default target. Any
actual API event is accepted, including non-action events such as an `Unmap`,
and any texture or buffer can be read back. Intermediate targets such as SSS
buffers or GPU-skinning outputs can therefore be read for diagnostics. MSAA
read-back is rejected rather than resolved implicitly.

### Patch plan: the existing `capture_mutation` format

One plan file serves both routes, so no second format exists. Each patch has
`chunk_index, chunk_name, buffer_ordinal, expected_buffer_index,
expected_blob_sha256, byte_offset, expected_range_sha256, replacement_file`
(relative to the plan), `replacement_sha256, source_witness, semantic_role`, and
an optional `resource_id`. The `resource_id` must be one of the chunk's
non-Context resource fields. At plan level, `source_rdc_sha256` binds the fast
route and `source_xml_sha256`/`source_zip_sha256` bind the official route.

The fields in the original task wording map onto this format as follows:
`original_sha256` is `expected_range_sha256`, `new_bytes_path` is
`replacement_file`, and the buffer occurrence is `buffer_ordinal` plus
`expected_buffer_index`. `expected_buffer_index` is the official structured
buffer index, which is the same number as the zip.xml member name.

Rejections happen before any output is written. These cover a changed capture,
plan, index or replacement; a wrong chunk name or occurrence; a wrong blob or
original range hash; a range outside the recorded buffer; overlapping patches;
and a missing witness or semantic role. The C++ session re-verifies, against the
actual decoded section bytes, the size prefix, the blob hash, the range hash and
the replacement hash. A rejected job leaves only its `receipt.json`, and the
session continues.

## Mechanism

### Official offset index (`index` mode, once per capture)

Pinned v1.46 source (`renderdoc-e4bd23b6…`, local copy under
`outputs/offline_renderer_repair_20261013/renderdoc_source_pinned/`) gives the
layout:

- `serialise/rdcfile.cpp` defines the container: `FileHeader`,
  `BinaryThumbnail`, `CaptureMetaData`, then binary sections with
  `BinarySectionHeader` (40 bytes plus the name).
- `serialise/serialiser.cpp` reading `BeginChunk`/`EndChunk` defines the chunk
  header: a u32 id with flags, then optional callstack, thread, duration and
  timestamp fields, then a 32- or 64-bit length. Every chunk ends at a 64-byte
  boundary.
- `DoSerialise(SDObject*)`, used by `WriteStructuredFile` (the official
  zip.xml→rdc writer), gives the byte layout of every structured value. Integers,
  floats and resource IDs use their width. Booleans and characters take 1 byte.
  A Nullable value has a 1-byte presence flag. Arrays have a u64 count. Strings
  have a 32-bit length followed by the bytes. **Byte buffers are a u64 size,
  `AlignTo<64>` padding, then the raw bytes.** Opaque chunks are raw payload.

`index` walks the official `ICaptureFile::GetStructuredData()` with exactly
these rules. Every byte buffer it finds is then checked against the official
decompressed section (`GetSectionContents`): the size prefix, the zero padding
and the complete bytes. It also checks every chunk's id and length. HS2 capture:
40,492 chunks, **5,835 buffers verified**, 40,395 chunks walk to their exact
recorded length, and 97 have the conservative length slack the source documents.
Independently, the pinned LZ4 page decode of the file gives the same section
SHA-256 as the official decoder (`f7122a76…`). Run time is 6.5 s.

### Fast route: LZ4 page splice + official replay (`session` mode)

`serialise/lz4io.cpp` stores the frame section as 1 MiB pages, each written as
`[int32 compSize][LZ4_compress_fast_continue block]`, with the previous page as
the only history. LZ4 match offsets are at most `LZ4_DISTANCE_MAX = 65535`, so
decoding page *k* depends only on its own block and the last 64 KiB of decoded
page *k−1*.

The session holds the original compressed RDC (260 MB) plus a 64 KiB history
tail for each page (1,577 pages). It does no decompression of the whole
1.65 GB section per job. For each job it:

1. decodes only the pages containing the patch;
2. verifies the original bytes;
3. writes the replacement;
4. re-encodes the touched pages and each following page whose history changed,
   using the **pinned `renderdoc/3rdparty/lz4/lz4.c`** with
   `LZ4_loadDict`(history) + `LZ4_compress_fast_continue`, and decode-checks
   each re-encoded block against the same history the official decoder will
   have;
5. writes a complete standard RDC: the original header with the new
   `sectionCompressedLength`, the original compressed blocks copied unchanged,
   the re-encoded blocks, and all following sections;
6. runs official `OpenFile` → `OpenCapture(NoOptimisation)` → `SetFrameEvent` →
   `GetTextureData`/`GetBufferData` for each target.

Each candidate RDC is a normal capture file that stock qrenderdoc can open. By
default it is deleted after a successful job. It is retained on failure, or when
`keep_candidate_rdc` is set.

The in-memory alternative was not used. `SetMetadata` + `WriteSection` +
`OpenCapture` from memory is supported by the pinned API, but
`RDCFile::Open(bytebuf)`/`OpenBuffer` cannot read sections back in v1.46
(`m_MemorySections` is empty). The `WriteSection` route also copies the 1.65 GB
section twice inside the DLL, which exceeds the memory available on this
machine. A controller cannot replace recorded buffer contents in place either:
proxy upload is internal, and `ReplaceResource` substitutes resource identities
for shader editing. Creation-time data such as `CreateBuffer` InitialData is
consumed in `ReadLogInitialisation`, so each candidate needs its own official
`OpenCapture`.

### Official reference route

The reference route uses `capture_mutation.mutate_capture` on the official
zip.xml export (copy-on-write blob, all members retained), then official
zip.xml→rdc `Convert` (structured re-serialisation with zstd), then a replay-only
session on the converted file.

## Equivalence proof

The official identity roundtrip (rdc → zip.xml → rdc) reproduces the frame
section byte for byte: the original LZ4 section and the converted zstd section
have the same decompressed SHA-256 `f7122a76…`. The fast and official routes
must therefore produce the same section bytes for the same plan.

Results:

- **Fixture** (independent analytic 32x32 capture, not HS2): the fast and
  official routes give equal frame section hashes and equal read-backs (RT,
  copy and VB) for both the identity job and an isolated diagnostic VB patch.
- **HS2**: see the acceptance below.

## HS2 acceptance (existing capture only)

Script: `accept_patched_capture_hs2.py`. Compact evidence:
`tools/offline_renderer/patched_capture_pipeline_acceptance_20261020.json`.
Bulky outputs: `outputs/offline_renderer_pipeline_20261020/hs2_retry_frame17675_v1/`.
The capture is the 2026-10-12 unaccepted retry RDC (`eb5743bc…`). No game was
launched or injected and no new capture was taken. All 10 checks passed.

- **Identity.** Camera 59322 at event 35086 equals the earlier exact
  `original_target.bin` (`da5e176e…`). Its top-down output equals every RGBA
  pixel of the native `capture_v1/actual_main_camera.png`, and the written PNG
  pixels equal the top-down bytes.
- **Diagnostic (labelled, head-unrelated, isolated plan).**
  - What was patched: Unity "Internal Curves Texture" upload, Unmap chunk 35240
    into staging 59898, copied to 11214 at event 32898. Row-0 half floats
    1–511 were halved; these are blob bytes [1, 1023) of the recorded written
    range [1, 2048).
  - Prediction: texture 11214 read back at event 32898 equals the identity
    read-back with exactly mapped bytes [2, 1024) replaced. It matches exactly.
    The observed changed range is [3, 1024) because byte 2 happens to be
    unchanged after halving.
  - Consequence (observation, not a prediction): 1,571,443 of 1,572,528 camera
    pixels changed, maximum channel delta 105. Colour grading is global.
  - Size of the change: one LZ4 page changed and two pages were re-encoded.
- **Rejection.** A wrong `expected_range_sha256` was rejected
  (`Original selected byte range SHA256 mismatch`). Only `receipt.json` was
  written, and the session stayed usable.
- **Identity after patching.** In the same session, identity read-backs were
  byte-identical to the first identity job.
- **Repeat timings.** Three repeated diagnostic renders gave identical camera
  bytes.
- **Equivalence.** The official route was run with the same plans. For identity,
  both routes give frame section `f7122a76…`. For the diagnostic, both give
  `50e5f94b…`. Every read-back is equal.

## Timings (HS2 capture: 260 MB RDC, 1.65 GB frame section, 1086×1448 target)

| Stage | Official route (before) | Fast route (after) |
|---|---|---|
| once per capture | rdc→zip.xml export 8.4 s | `index` 6.5 s; session start 1.5 s (read 0.18 s, SHA-256 0.10 s, page decode and history cache 0.89 s) |
| apply patch | `mutate_capture` 25.5 s (XML parse/compare, re-zip of all members) | verify 0.5 ms + re-encode 4 ms |
| build candidate RDC | official zip.xml→rdc Convert 8.9–11.3 s | splice write 0.29–0.30 s (260 MB, hashed while written) |
| official OpenCapture + replay + read-back | 4.6 s (separate process, zstd candidate) | 1.35–1.54 s OpenCapture + 0.03–0.05 s SetFrameEvent + 0.04 s read-back |
| per patched candidate, total | **≈ 39 s** | **1.9–2.1 s tool**, 2.4–2.7 s Python wall including PNG encode (≈0.4 s; set `png: false` in loops) |
| identity job | 13 s (convert + replay) | 1.46 s |

The fast route's per-job time is now set mainly by the official `OpenCapture`
of the whole frame. That is the floor for any exact path through the public
v1.46 API (see the in-memory alternative above).

## Output orientation

The official read-back is in D3D subresource memory order: row 0 first, with no
flip, as in `D3D11Replay::GetTextureData` → staging `Map(READ)` →
`MapIntercept::CopyFromD3D`. RenderDoc's `SaveTexture` also writes memory row 0
first.

The native PNG comes from the bridge: `cam.targetTexture = rt` (ARGB32 RT) →
`Camera.Render` → `Texture2D.ReadPixels` → `EncodeToPNG`
(`MakerRenderStyleService.cs`). On Direct3D-like platforms Unity renders into a
RenderTexture with a flipped projection, so texture-space v=0, the visual
bottom, is stored in memory row 0. `EncodeToPNG` writes the visual top row first
(Unity manual, "Writing shaders for different graphics APIs: render texture
coordinates"). The Unity engine's native code is not available, so the
documented behaviour is cited. Exact equality on the real capture confirms it:
`replay[h−1−y] == native[y]` for every RGBA pixel.

`orientation: "unity_texture_rows_bottom_up"` therefore produces
`<name>.topdown.bin` by an exact row permutation. For 8-bit RGBA
Typeless/UNorm/sRGB formats it also produces `<name>.png` from those exact
bytes, without colour conversion. `"d3d_rows_top_down"` keeps memory order. Use
it for targets whose memory row 0 is the visual top, such as a swapchain back
buffer or the non-Unity fixture. The raw read-back `<name>.raw.bin` is always
kept. Block-compressed, planar and non-2D targets get raw bytes only, with an
explicit note.

## Receipts

`receipt.json` per job records:

- tool exe and DLL paths and SHA-256, plus the build receipt;
- capture RDC and frame-section SHA-256, and index SHA-256;
- plan path and SHA-256, with every resolved patch (exact section range, blob
  and range hashes, witness, role);
- candidate RDC SHA-256 and size, and the number of changed and re-encoded LZ4
  pages;
- per target: raw, top-down and PNG paths with SHA-256 and the format
  descriptor;
- C++ stage timings (verify, patch/re-encode, candidate write, official
  OpenFile/OpenCapture, SetFrameEvent, read-back, shutdown) and Python timings;
- the session PID and the scope flags.

Each session directory keeps the exact stdin directives, the stdout JSON lines,
stderr, and a close receipt. Deadlines stop only the session's own child
process.

## Tests

`tools/offline_renderer/test_render_patched_capture.py`:

- **CPU** (7 cases): plan binding and all rejection branches, target
  validation, exact row permutation, PNG eligibility.
- **GPU** (opt-in, run with `HS2_REPLAY_TOOL`, `HS2_RENDERDOC_DIR` and
  `HS2_REPLAY_TEST_ROOT` set): on the analytic fixture, identity; an isolated
  diagnostic patch (VB read-back equals exactly the declared bytes and the RT
  changes); a mismatched original-range hash rejected inside a live session; a
  repeat identity run after the patch; and fast/official equivalence for both
  jobs.

HS2 acceptance script: `tools/offline_renderer/accept_patched_capture_hs2.py`.

## Build

```powershell
python tools/offline_renderer/build_replay_frame_msvc.py --headers ABS\renderdoc_headers_146 `
  --vcvars "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" `
  --lz4 ABS\renderdoc_source_pinned\renderdoc-e4bd23b671d3d5a747ff5221dbe08a63eb6ca200\renderdoc\3rdparty\lz4 `
  --out FRESH_DIR
```

The build compiles the pinned `lz4.c` and links `bcrypt.lib`, which provides
SHA-256. The previous `convert`/`inspect`/`replay`/`drawinputs` modes are
unchanged; the new modes are `sections`, `index` and `session`. Never run a
GNU-built client.

## Gaps (explicit)

- **Zstd-compressed originals cannot be patched.** Patches require an LZ4 or
  uncompressed frame section; captures written by HS2/RenderDoc are LZ4.
  Uncompressed sections are parsed, but splicing them is not implemented in this
  build. Zstd files, such as official Convert outputs, are replay-only.
- **Patches cannot change lengths.** A patch replaces bytes of an existing
  recorded buffer at the same length. Patches that change structure, such as a
  different buffer size, draw count or new commands, are outside this layer.
- **Culling and command visibility stay as recorded.** Any CPU-side culling or
  command list that changes with the candidate is outside a byte patch and must
  be shown invariant by the frame-graph work.
- **Each candidate pays a full official `OpenCapture`.** On HS2 that is most of
  the per-job time; no public v1.46 API replays a second variant inside one
  controller.
- **MSAA and 3D-slice orientation are not handled.** MSAA read-back is rejected,
  and 3D-texture slice orientation is not reinterpreted.
- **Orientation is proven for one target.** The Unity orientation is
  documented behaviour, confirmed exactly for camera target 59322. Other
  intermediate RTs follow the same Unity convention, but each new target's
  semantic orientation is the caller's declaration.
- **Each candidate is a full-size file on disk.** Every job writes a complete
  RDC (260 MB on HS2); the disk write is part of the per-job cost.
