# Actual main-camera capture scope

This helper records the original installed `MakerRenderStyleService.Capture`
commands; it neither renders a replacement scene nor loads/injects RenderDoc.
The main task exclusively owns game deployment/restart/capture. Native runtime
acceptance remains pending; the current checks are CPU table mocks and compilation.

## Installed IL hook contract

Independent net472 assembly `HS2_RenderDocScope.dll`, global class
`HS2_RenderDocScope`, public static **void** methods:

```csharp
BeginFromNativeTexture(System.IntPtr actualRtResource, string capturePngPath)
EndAfterSavedPng(string capturePngPath)
AbortOwnedSafely()
// Read-only diagnostic string getter:
get_LastReceipt()
```

Parent's installed bridge audit: pose freeze Apply at040c; actual
`Camera.targetTexture` assignment0645; first ResetTemporal064c; sixteen original
Camera.Render calls through0669; synchronous ReadPixels06a7/Apply06ae; original
PNG FileStream.Dispose and inner finally complete06ef; result construction06f0;
outer restoring finally07f4, pose.Dispose08b6.

1. After targetTexture assignment, before ResetTemporal: get **that actual RT's**
   `GetNativeTexturePtr()` and call BeginFromNativeTexture(pointer, PNG pathname).
2. After the original synchronized readback and PNG save/inner finally, before
   result construction: call EndAfterSavedPng(the same PNG pathname).
3. At the existing outer restoring finally entry: call AbortOwnedSafely().
4. Preserve original exception ranges, all sixteen renders, SSS, CTAA, readback,
   resource lifetime and every restoration instruction. No other source rebuild.

All methods contain ordinary exceptions and expose them in the receipt. A failed
capture is not a completed capture, even when the native PNG operation succeeds.
Absent/disabled/nonmatching sentinel does not invoke or load the native helper.

## Explicit one-attempt sidecar

Copy the two helper DLLs beside the installed narrow-hook bridge. Its existing
Newtonsoft.Json.dll has the same SHA256 as the dependency used for compilation:
`e1e27af7b07eeedf5ce71a9255f0422816a6fc5849a483c6714e1b472044fa9d`.
The sidecar is `HS2_RenderDocScope.request.json` beside the managed helper:

```json
{
  "Enabled": true,
  "ExpectedPid": 12345,
  "ExactPngPath": "C:/absolute/real_capture.png",
  "CapturePrefix": "C:/absolute/fresh_rdc_prefix",
  "ReceiptPath": "C:/absolute/scope_receipt.json",
  "ExpectedNativeSha256": "ab77d4f2136c4a6b78ce979428729be9c16586d0737826aba4166d25c935b771",
  "ExpectedRenderDocSha256": "809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d"
}
```

12345 and the example paths must be replaced with the actual parent-owned process
and exact request paths. Every output parent must already exist. RDC prefix must
have no matching existing files; PNG may be the explicit capture output path.
The helper verifies the loaded official RenderDoc DLL's actual path/hash, expected
PID, its own native helper hash, selected PNG path and fresh saved RDC file.
It loads only its own native DLL by absolute path. The native DLL has no RenderDoc
or D3D11 import; it uses GetModuleHandle/GetProcAddress for already-loaded SDK.

`BeginFromNativeTexture` requires an actual live **D3D11 ID3D11Resource** returned
by Unity on the main thread. It obtains the resource's actual ID3D11Device and
holds an AddRef through the scope. Supplying an arbitrary address/other graphics
backend is outside the contract; the call does not validate arbitrary pointers.

## Ownership and evidence limits

Official pinned application SDK v1.6.0 C table layout is compile-checked. Its
Start/End functions receive primitive pointer arguments; no C++ library objects
cross the DLL boundary. SDK has **no atomic begin-if-idle** or capture identity
query: main must be the sole capture writer throughout, with no Trigger/hotkey/
external SDK capture. Observed external end/count replacement is rejected without
End/Discard. An unobserved discard/restart cannot be distinguished by this API;
the sole-writer requirement cannot be replaced with a claimed ownership guarantee.

End requires idle completion, exactly one newly saved SDK entry, actual fresh
nonempty local RDC file, matching size/timestamp and SHA256. Original filename
template is restored only if still the helper's installed template, including an
original empty template. Owned-only outer-finally Discard handles failed renders.
Receipts retain begin/end/abort events and native errors. `LastReceipt` exposes
receipt-write failures; returned game dictionaries must not ingest Newtonsoft
JTokens or native pointers. The C SDK GetCapture does **not** expose driver/API;
RDC parsing must verify D3D11 and actual frame/pass contents subsequently.

## Rebuild and validation

Prepare the pinned official portable v1.46 with prepare_renderdoc.py, then:

```text
python tools/offline_renderer/build_renderdoc_scope.py --sdk outputs/offline_renderer_repair_20261012/renderdoc_portable_1_46/extracted/RenderDoc_1.46_64 --vcvars "C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat" --out outputs/offline_renderer_repair_20261012/NEW_FRESH_SCOPE_BUILD
```

Current independently built outputs: ignored `scoped_capture_cpu_v7/`, hashes in
`scope_validation_20261012.json`. Twenty-two CPU official-table mock checks cover
no SDK, PID, busy guard before resource access, Start no-op, token ownership,
End/file/hash/prefix reuse, failed/successful Discard, external template/ended/
count-changed scopes and buffer bounds. Managed CPU checks verify missing,
disabled, nonmatching and wrong-PID requests without loading native modules.
MSVC import/export/assembly evidence is retained, alongside earlier compile/mock
failures in v1-v3. No graphics, RenderDoc DLL or game is executed by these checks.

Remaining: parent-controlled actual scoped frame capture; full replay/mutation
of every geometry/material/cache dependency; full standalone candidate RGB
agreement and detector-forward fitting. A baseline RDC replay alone is not the
completed arbitrary-parameter offline renderer.

## Optional same-freeze geometry pair

The managed helper accepts three optional sidecar strings:

```json
{
  "GeometryBeforePath": "C:/absolute/fresh_geometry_before.json",
  "GeometryAfterPath": "C:/absolute/fresh_geometry_after.json",
  "ExpectedBridgeSha256": "29a3b2aa8ead6ea16d61013930cd8de2af81dc7bca0e8d3645c491b4aadc5482"
}
```

These values supplement the enabled/selected-path request above. All three must
be supplied together; both JSON paths must be distinct, absolute, fresh and have
existing parents. They cannot alias the PNG, receipt or capture prefix. Use the
**actual deployed narrow-hook bridge hash**, not the pre-patch bridge hash.
Missing all three preserves the previous execution/optional receipt fields.
Absent/disabled/nonmatching sidecar never exports geometry or loads native DLLs.

Reflection resolves exactly one **already-loaded** HS2_McpBridge assembly, verifies
its file SHA and internal sealed BeforeFieldInit service type, the public default
constructor attributes and audited seven-byte object-only IL, and the exact
public instance Dictionary<string,object> ExportGeometry method. Five parameters
must be the installed ordered names/types. The helper invokes precisely:

```text
HS2Mod.McpBridge.MakerGeometryService.ExportGeometry(path,null,true,null,true)
```

The first export occurs immediately before SDK Start while the outer Capture
already holds its actor freeze/current temporary main camera. The second occurs
after SDK End succeeds and the PNG is saved, before the existing outer restoration.
The installed serializer writes the geometry files unchanged: default eight mesh
names include the actual working card's ten surfaces (three o_tang), full loaded
API blendframes, actor transforms, native drivers and private diagnostics.
No texture export is requested; existing resource capture handles GPU data.

Receipts bind both actual file paths/hashes/byte counts, native frame_count and
frame_count_end, native captured_utc and invocation/return UTC, native export
receipts, method signature, loaded assembly identity/MVID/SHA and constructor
attributes. Root metadata is read as a streaming JSON scalar pass, preserving
the native timestamp strings. Export/hash/metadata failure rejects a completed
paired result and retains files/errors. An SDK success preceding a failed AFTER
export remains evidence under `end_native_pending_geometry`, followed by a
failure event; it is not accepted as a complete pair. Active owned SDK failures
still reach owned-only outer-finally abort. Already saved/ended RDC is retained.

CPU good/wrong-signature/wrong-constructor fixtures validate exact arguments,
optional legacy behavior, freshness/alias/relative paths, loaded-file SHA,
before/after frame/file metadata and export/hash/late-freshness failure handling.
Fixture-only HS2_McpBridge.dll files live inside ignored **fixture_*** subfolders;
**never deploy those synthetic assemblies to the game**. Deploy only the two
real scope helper DLLs at the build root. Current build is scoped_geometry_cpu_v4.
All checks are CPU-only; actual same-freeze native capture is still pending.

Installed geometry has no **baked tangent** array. Source tangents and blendframe
delta_tangents are retained; GPU tangent identity remains unverified until actual
draw buffers are decoded. Neither same-freeze pairing nor full renderer completion
can be claimed from the CPU fixture.

## Stripped Unity JSON boundary repair

The first native scope attempt failed before PNG/geometry/RDC: generic reflection
serialization required unavailable System.Data4.0. The failure and exact state
restoration remain main-task evidence; no success or retry was inferred.

The helper now follows the installed bridge's actual JsonUtil dictionary/
sequence/primitive boundary. Installed WriteValue/ParseObject/Object IL and
bridge SHA are retained in the CPU receipt. Request parsing uses JsonTextReader
and duplicate-strict JObject.Load with explicit named boolean/uint32/string
fields. Request DTO output is an explicit field dictionary. Native receipt
objects use direct JToken.WriteTo; nested geometry/export metadata retains all
dictionary/list/primitive values. Unsupported types/fields/tokens and nonfinite
numeric values fail explicitly; no DataTable behavior is simulated or disabled.

Save and SafeSave use JsonTextWriter directly. A secondary failure envelope
preserves previous complete receipt, request and all error strings; if even that
writer/previous-JSON pass fails, an independent ASCII JSON escape routine retains
all raw diagnostics as strings in valid JSON. Primary receipt I/O failure can
produce a separate fresh `.failure.GUID.json` without throwing through native
restoration. The hook signatures and original sixteen renders are unchanged.

Production helper source/IL has no JsonConvert/JsonSerializer/DefaultContractResolver
call or System.Data assembly reference. A fresh .NET CPU process exercises all
request fields, numeric bounds, nested native JTokens, timestamp strings, Unicode,
wrong/unknown types, nonfinite values, primary file failures and double-writer/
invalid-history fallback: thirty-three checks pass with **no System.Data load**.
Old twenty-two SDK checks and paired geometry/freshness fixtures also pass.

Latest build: `scope_streaming_json_cpu_v3/HS2_RenderDocScope.dll` SHA256
`11f02550423db6c2dfd8de586e916b64b2c40b30d0580d9702bd234a7c5f887f`.
Rebuilt native source is unchanged, C API intact; its SHA is
`b93200cd15650c3e617d235350adc1d670bf35fab85a8fd1b57b13f87a159339`.
Existing Newtonsoft dependency hash remains e1e27af7... and matches installed.
The same managed filename/version is used; main must restart to replace the
already-loaded old assembly and update sidecar ExpectedNativeSha256 for the
chosen verified native binary. No game/GPU or deployment was done by this child.
Actual corrected native scope/RDC/GPU geometry/full RGB acceptance remains pending.
