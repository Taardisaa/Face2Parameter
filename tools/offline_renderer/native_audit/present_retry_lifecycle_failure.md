# Ordinary-retry SDK counter and CPU/GPU scope race

Main's single native Present attempt: process33736, normal working card/full SSS,
no repeated POST. Arm4→Poll5→Enter6 queued GL begin, but actual RT bind rejected
because the SDK reported inactive during RT creation/GetNativeTexturePtr sync.
The original PNG was saved; geometry/full scope was rejected. Main later verified
four public states/card SHA unchanged, then stopped only the owned process.

Evidence: `HS2Mod/artifacts/chenger_reference_fit_20261012/native_present_session_v1/`.
Preserved log `RenderDoc_app_2026.10.12_23.44.25.log` shows frame17674 failing six
uncaptured Unmaps, then automatic retry17675 succeeding. V3 helper incorrectly
treated Entered+SDKfalse+unchanged saved count as released and restored the default
prefix before that eventual259,623,782-byte RDC was saved. It is retained as
diagnostic data, never accepted as the annotated/same-freeze sixteen-render scope.

## Pinned source mechanism

RenderDoc v1.46 commit e4bd23b671d3d5a747ff5221dbe08a63eb6ca200:

- `core.cpp`1132 increments m_CapturesActive after forwarding void driver Start.
  EndFrameCapture1162 unconditionally decrements after driver End returns.
- `core.h`603 defines SDK IsFrameCapturing as m_CapturesActive>0.
- `d3d11_device.cpp`2450..2516 retains ActiveCapturing and calls AttemptCapture
  on ordinary failures<=5, then returnsfalse without another core Start/increment.
  App-controlled failures instead terminate and return background.

Thus first ordinary failure can produce driver-active retry with public counter0,
and successful retry End can decrement to-1. False SDK flag does not establish
driver inactivity, write completion or pending-trigger cancellation. The CPU
active snapshot also cannot prevent an intervening GPU Present from ending an
ordinary frame while the original CPU method is still preparing its RT.

Public Start cannot promote active ordinary capture: WrappedDevice Start2034
returns when non-background before m_AppControlledCapture=true2044. Core still
increments its counter. Calling Start(active) must not fabricate promotion or
repair the counter. Present itself also does not close all maps: this first
Present capture had six old Unmaps. Background high-traffic/dirty/deferred
directMap returns bypass m_OpenMaps, so that existing table alone is incomplete.

## Safety and original-result source fix

Abort now retains reservation/prefix in **every phase** when SDKfalse and saved
count unchanged. Only authoritative file completion, owned active discard or
root-owned shutdown resolves it; no fake inactive release. CPU checks reproduce
Entered→false→later file completion while retaining device/prefix.

New signature: `public static void HS2_RenderDocScope.RecordOriginalCallResult(object)`.
Root calls it immediately after the actual original callback returns, before
later scope checks. Stage original_call_returned/code0 stores the full callback
dictionary in original_result through direct streaming JSON/JToken, even after
capture/geometry failure. No reflection serializer/System.Data is used and no
fields are summarized away; unsupported result values fail with explicit evidence.

Build and checks: `outputs/offline_renderer_repair_20261013/present_reservation_cpu_v1/`.
This fixes safety/evidence only, not the recording lifecycle. V3 ordinary-ready
dispatch must not receive another native attempt unchanged.

## Required source implementation

Proceed in an independent pinned RenderDoc source/build, never modify installed
official DLL memory or relax Unmap checks. D3D11 extension must expose real driver
active/ordinary/app-controlled/attempt/failure and track every successful Map/
Unmap, including background direct returns. A render-thread atomic gate can
promote only the same successful ordinary attempt with no unsafe pre-capture
mapping, preserving all commands/serializers, then retain app-controlled capture
through the complete original method and final End. Core active accounting must
agree with driver retry/finish/discard lifecycle. New source/hash/build provenance
is required. Full rendering/SSS/sixteen calls/restoration remain intact. No new
graphics or installation occurred in this child milestone.
