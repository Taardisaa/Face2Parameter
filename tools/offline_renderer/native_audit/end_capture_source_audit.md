# Native SDK End failure: pinned D3D11 source

Source oracle: official RenderDoc `v1.46`, release commit
`e4bd23b671d3d5a747ff5221dbe08a63eb6ca200` and deployed official DLL SHA256
`809da38e3867d9fd09cc5c30dd5310500dee75e999166d7a14ad1ef6e9eca65d`.
Source URLs and byte hashes are in `end_capture_source_provenance.json`.
Extracted source remains ignored under
`outputs/offline_renderer_repair_20261012/end_capture_source_v1/` and the earlier
`renderdoc_source/` directory. No runtime probe or graphical process was started
for this investigation.

## Actual failed scope

Main's native attempt in process54072 retained the normal working card/full SSS,
the original sixteen Camera.Render calls, saved PNG and before-geometry export.
Begin returned1/ownedtrue; End returned-7/ownedfalse (`SDK End failed`). The
operation completed after the client HTTP deadline; main observed its actual
operation ID and did not issue another POST. No RDC/after-geometry pair was
accepted. Evidence is parent-owned:
`HS2Mod/artifacts/chenger_reference_fit_20261012/native_scope_early_boot_v2/capture_v1/`.
The previous stripped-Unity generic JSON failure is a separate repaired branch:
this attempt reached the official SDK End, not that serializer exception.

## Exact End call chain

1. Helper passes the held device obtained from the actual Unity RT resource and
   null window to official C SDK EndFrameCapture. Core `MatchFrameCapturer` in
   `core.cpp` explicitly supports offscreen null-window capture via device
   capturers. Core then forwards End to the chosen D3D11 driver and decrements its
   active-capture counter. Matching failure is separately logged.
2. `WrappedID3D11Device::EndFrameCapture` in `d3d11_device.cpp` acquires the D3D
   lock. An already inactive device returnstrue. The window/swapchain mismatch
   false branch only executes with a non-null window, so changing this helper's
   null window to a guessed HWND does not repair a missing command stream.
3. The driver asks its immediate context `HasSuccessfulCapture(reason)`. A true
   answer completes context recording and writes resources/initial contents/full
   frame chunks to RDC. Thumbnail, staging-map and file-writer paths occur here;
   they are not the documented context-invalid failure branch.
4. A false answer logs `Failed to capture, frame ...` and the selected reason.
   For an application-controlled capture it immediately gives up, clears data,
   exits active capture and returnsfalse. It does not run the ordinary UI
   next-frame retry loop. This explains an inactive scope/no RDC after failed End;
   calling End repeatedly would neither reconstruct omitted commands nor recover
   this discarded attempt.

## Both source-defined context failures

`CaptureFailReason` in `d3d11_context.h` contains success and two failure enums.
`d3d11_context_wrap.cpp` assigns the immediate failure flag in these branches:

- **Uncapped command list:** ExecuteCommandList during active capture sees
  `WrappedID3D11CommandList::IsCaptured()==false`. It logs the resource ID and
  marks the whole frame unsuccessful. A deferred context beginning capture with
  existing unrecorded commands is unsuccessful. Its first FinishCommandList
  creates an uncaptured list even though subsequent lists become recordable;
  executing that first list contaminates the current capture. Finishing sixteen
  full renders does not undo an earlier unsuccessful execute.
- **Uncapped Map()/Unmap():** Unmap during active capture cannot find its
  intercepted Map, or the Map began outside active capture and was neither READ
  nor WRITE_DISCARD. WRITE/WRITE_NO_OVERWRITE need shadow/diff state created while
  recording. A synchronous ReadPixels read map is permitted by this source
  condition; the mere existence of ReadPixels is not evidence of this failure.

Multiple offending operations can occur and later assignments overwrite the
selected enum. Preserve both warning types and final summary, rather than using
the last warning from another capture or only the helper's generic -7.

## Source-derived repair choices; select using real log evidence

For **uncaptured command lists**, official SDK option10 CaptureAllCmdLists records
deferred contexts continuously from creation. The context constructor makes them
ActiveCapturing/successful and FinishCapture retains that state when this option
is enabled. It must therefore be set and read back inside the already audited
early bootstrap, after SDK acquisition but before original BepInEx/Unity graphics
initialization. Setting it only when the failed scope begins does not recover
previously recorded commands. This records more original commands; it does not
disable Unity multithreading, SSS, CTAA, materials or plugins. Startup ordering
and new native scope acceptance would remain required.

For **uncaptured Map/Unmap**, CaptureAllCmdLists does not repair immediate-context
maps. The capture boundary must be moved/synchronized using the actual Unity
render-thread/resource-write lifecycle so required write interception precedes
those Maps and End follows execution. This needs the responsible installed
Unity/native queue path, not extra Camera.Render warmups, parameter sweeps,
force-single-thread rendering or arbitrary sleeps. Existing main-thread geometry
and PNG boundaries alone do not prove a clean GPU write boundary.

Main subsequently preserved the actual flushed log
`RenderDoc_app_2026.10.12_22.51.09.log` (4490 bytes). In process54072 its line28
starts capture at22:53:53, lines31–34 record four uncaptured-Unmap warnings, and
line35 rejects frame4294967295 at22:53:54 with `Uncapped Map()/Unmap()`. There is
no command-list warning in this attempt. The read-only classifier confirms the
Map/Unmap failure enum; its full input hash/line receipt is retained in
`end_capture_native_failure_20261012.json`. This selects the write-lifecycle
repair, **not CaptureAllCmdLists**. The log does not name the four resources or
give their original Map timestamps/types; those details remain unresolved.

Main also reports its subsequent normal-card-reload state check failed before
stopping the owned game. The dependent stop still ran and exact pre-stop reload
state was not persisted. Do not claim a verified final reload; accepted
after-capture public-state restoration and independent unchanged card backup are
separate evidence. No native graphics process remains for another probe.

## Read-only log classification

```text
python tools/offline_renderer/audit_renderdoc_capture_failure.py EXACT_NATIVE_LOG FRESH_RECEIPT.json
python -m unittest discover -s tools/offline_renderer -p test_capture_failure_audit.py -v
```

The classifier hashes the exact input, preserves individual attempts/line evidence
and warnings, and only establishes the selected reason from a final driver
summary in a started attempt. Partial/buffered logs remain incomplete; warnings
without an attempt start remain orphan evidence. A completed-driver log never
claims file/replay/full-RGB acceptance. Four CPU regression cases pass, including
two concurrent cause types with a later selected enum, separate attempts,
buffered/orphan evidence and unknown reasons.

Full offline RGB remains incomplete: a successful complete native RDC and paired
geometry are still needed, followed by original-driver replay acceptance and
all-surface/current/previous/shadow/motion/culling/resource dependency mutation.
