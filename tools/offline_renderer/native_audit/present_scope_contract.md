# Official Present-boundary capture protocol

The actual SDK-controlled mid-frame attempt in process54072 failed with four
uncaptured-Unmap warnings. This implementation uses official `TriggerCapture`
and the source-defined `WrappedID3D11Device::Present` start/end boundary. It does
not move a mid-frame Start to a guessed EndOfFrame, disable multithreading/SSS,
add warmup renders or force a mid-frame End. CaptureAllCmdLists is not the repair.

Only main owns startup/deployment, the real game and its single native acceptance.
This child performed source recovery, installed metadata audit, compilation and
CPU fixtures; no new RenderDoc DLL, game or graphical process was executed.

## Exact managed methods

All methods are on global `public static HS2_RenderDocScope`:

```text
int ArmPresentFromNativeTexture(IntPtr existingRealUnityTexture,string exactPngPath)
int PollPresentCapture()
int EnterOriginalPresentCapture(string exactPngPath)
void AbortPresentSafely()
void RecordOriginalCallResult(object actualOriginalCallbackDictionary)
void BeginFromNativeTexture(IntPtr actualOriginalRt,string exactPngPath) [existing hook]
void EndAfterSavedPng(string exactPngPath) [existing hook]
void AbortOwnedSafely() [existing finally hook]
string get_LastReceipt()
```

Arm0 means absent/disabled/unmatched ordinary capture, Arm4 means own trigger
reserved. Negative values reject the attempt. Poll4 means waiting (pending GPU
Present is not inferred from main frame count); Poll5 means actual capture ready;
Enter6 consumes that ready reservation **once**, before any original Capture
operations. Enter0 is ordinary path only. Poll2 means one fresh file saved;
`capture_accepted=false` remains until actual RDC annotations/readback proof and
native pipeline acceptance. Negative results never justify executing Capture.

Root must use its actual coroutine/main-thread dispatcher to wait across frames;
blocking Unity's main thread cannot wait for another Present. Run the complete
installed original Capture exactly once only after Enter6, including original
face-surface setup, freeze, sixteen Camera.Render calls, full SSS/postprocess,
ReadPixels, PNG write and all original cleanup. Root's narrow bridge/dispatcher
milestone is HS2Mod `edc12ea`; its original Capture implementation is retained.

Arm gets device identity from a read-only **existing real composed-face Unity
texture**. It does not create a diagnostic RT. The existing Begin hook retains
the actual original camera RT, verifies identical COM IUnknown device identity,
exports before-geometry with the original actor already frozen and queues the
target annotation. Existing End hook exports after-geometry while still frozen/
active, then marks original complete and queues the end annotation. It does
not call SDK End. Ordinary captures with a different PNG path do not bind/complete
or abort this reservation. Existing original finally preserves a completed
reservation for normal Present, and discards an incomplete owned one. Root must
call AbortPresentSafely on dispatch/timeout/original-return errors, including an
error occurring after the inner PNG/completion hook.

## Explicit sentinel

The existing scope JSON sidecar keeps its exact fields and adds:

```json
{
  "Enabled": true,
  "CaptureMode": "official_present",
  "ExpectedPid": 123,
  "ExactPngPath": "C:/absolute/fresh/actual_main_camera.png",
  "CapturePrefix": "C:/absolute/fresh/unique_scope",
  "ReceiptPath": "C:/absolute/fresh/sdk_scope_receipt.json",
  "ExpectedNativeSha256": "SEE_FINAL_BUILD_RECEIPT",
  "ExpectedRenderDocSha256": "809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d",
  "ExpectedUnityCoreSha256": "1dbc6ba5ecaf6c5f890ad12aa677f7897fd62356b6ace27c1c3fda73de16ae11",
  "GeometryBeforePath": "C:/absolute/fresh/geometry_before.json",
  "GeometryAfterPath": "C:/absolute/fresh/geometry_after.json",
  "ExpectedBridgeSha256": "EXACT_MAIN_DEPLOYED_NARROW_BRIDGE_HASH"
}
```

The sample values are a contract illustration, not a deployed sentinel. Parents
must exist; receipt/PNG/prefix and both geometry paths must be fresh and distinct.
The installed Unity Core SHA/API signature is checked before calling the already
loaded `GL.IssuePluginEvent(IntPtr,int)`. No Unity module is independently loaded.
Absent CaptureMode preserves the legacy protocol and all old hook signatures.

Official SDK version1.7 is explicitly obtained for the new mode. Header table
order has static_asserts; SetCommandAnnotation is slot28 and TriggerCapture15.
No C++ interface/std::function/foreign-runtime ABI is used. Native helper only
uses already loaded pinned RenderDoc; no injection/startup is performed here.

## Actual command-stream evidence

Three native UnityRenderingEvent callbacks use the documented x64 StdCall ABI.
The managed shim queues them through the installed GL method; it does not invoke
callback pointers. IDs are bound to the owned reservation, unknown old events
cannot touch a new reservation, and reordered/repeated/inactive events reject it.
Callbacks execute official SetCommandAnnotation on the render thread:

- `hs2_scope.begin`: string nonce `fresh_capture_prefix#owned_token`.
- `hs2_scope.target`: scalar APIObject, the actual retained original RT. Pinned
  D3D11 source converts this COM pointer to its real captured ResourceId.
- `hs2_scope.end`: the same nonce string after original PNG and paired geometry.

This does not alter object debug names, replace resources or change features.
The nonce/device/target pointers and actual callback thread IDs remain in native
receipts. Original resource/device AddRefs are released after final file/error
cleanup; they keep queued callback arguments valid after original RT destruction.

**Ordinary capture auto-retries matter:** after a failed ordinary frame,
RenderDoc can record a later frame. Callback booleans, file count/path, and an
original call while API-active do not alone prove that final file contains that
call. Root must convert the final RDC through the pinned official parser, then:

```text
renderdoc_replay_msvc.exe convert ACTUAL.rdc rdc ACTUAL.zip.xml zip.xml
python tools/offline_renderer/verify_present_scope_xml.py ACTUAL.zip.xml --nonce EXACT_NONCE --width ACTUAL_WIDTH --height ACTUAL_HEIGHT --out FRESH_PROOF.json
```

The verifier requires exactly one ordered begin/actual-target/end annotation in
the same nonzero immediate context, correct native union types, exact nonce,
actual camera RT creation/dimensions, recorded draws and a whole-subresource
copy/resolve path into real CPU-readable STAGING texture. Unsupported cropped,
shader-converted, unknown-context or subresource paths remain explicit failures.
RenderDoc omits READ Map/Unmap commands as CPU no-ops, so an invented read-map
chunk is not required; actual recorded GPU staging transfer is the source proof.
It returns copy/annotation provenance only, with pixel/full-RGB acceptancefalse.
Native screenshot/16-render/source-state/actual replay output are still required.

## Receipt and failure lifecycle

Managed root is `{schema_version:1,events:[...]}`. Final file event is
`stage=present_poll, code=2`; `native.operation=present_file_saved`, native owned
false, mode official_present, file/hash/size/time/count/nonce and callback evidence.
Same event `geometry.complete/before/after/bridge_sha256/export_method` carries
the paired source export provenance. Actual public-state restoration remains
root's separate receipt, not this helper's assertion.

Original SDK has no cancel-pending-trigger API. Abort while queued/inactive
returns-21 and retains reservation until active owned discard or root-owned
process shutdown. Every phase with SDKfalse/unchanged count now retains reservation:
pinned core's counter can befalse while the D3D11 driver retries/writes. It never
calls Discard on an assumed inactive device or falsely clears
a pending trigger. An external saved entry/capture is not discarded. Present
reservations reject legacy SDK End/Abort routes. Default hotkey/other capture
writers must remain absent by root's explicit exclusive-writer contract; the SDK
does not expose an atomic capture identity or pending-trigger cancellation.

## CPU validation and deployment files

Final build/receipt:
`outputs/offline_renderer_repair_20261012/present_scope_cpu_v3/build_receipt.json`.
Ship only its `hs2_renderdoc_scope_native.dll` and `HS2_RenderDocScope.dll`, with
the already pinned installed Newtonsoft.Json. Never ship fixture bridge or CPU
test assemblies. Source is compiled by installed MSVC14.50/x64, static runtime;
imports have no renderdoc.dll or d3d11.dll dependency.

- Native29 table/COM-identity/protocol checks, including pending cancellation,
  once-only enter, wrong actual device, queued annotation order, ownership release,
  premature ordinary file, fresh file/hash and no SDK Start/End.
- Legacy22 SDK checks and original optional/paired geometry fixture variants.
- Managed/schema45 checks without System.Data, including exact callback queuing
  and completed-original finally behavior; fake GL records IDs only.
- Three XML regressions reject missing/retry markers, wrong nonce, duplicates,
  wrong context/readback and retain pixel/RGB acceptancefalse.
- Installed Unity GL/texture/end-of-frame Cecil audit and banned JSON-call audit.

CPU fixtures are isolated diagnostics, not native images/captures. Native Present
attempt33736 failed and is preserved. The abort/result fix build does not resolve
its source lifecycle and must not be treated as another native-ready candidate.
See [present_retry_lifecycle_failure.md](present_retry_lifecycle_failure.md). Full renderer
still needs source-exact replay/candidate vertex/normal/tangent/bone/previous/
shadow/motion/culling/material/resource dependency integration and pixel checks.
