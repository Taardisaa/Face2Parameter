# Source-method head displacement viewer

This is the current implementation direction. The earlier empirical atlas is
retained as historical evidence, including its rejected predictions.

Run from the Face2Parameter checkout:

```powershell
.\.venv\Scripts\python.exe tools/parameter_atlas/forward_server.py
```

Open http://127.0.0.1:43128/ with Maker ready and the new HS2 bridge installed.
Capture a context, choose any native control or selected ABMX scalar, and calculate
an explicit candidate. The bridge evaluates copies using the installed game's
shape methods, SliderUnlocker hooks, ABMX methods and Unity BakeMesh. It requires
the game; this viewer does not implement an approximate deformation formula or
write the current character.

The optional displacement target solves one scalar within a user-declared bound.
It calls the complete forward evaluator for every candidate, returning endpoint
bracketing/finite-iteration limitations and an explicit residual. It does not
assume global monotonicity, infer slopes or promise a unique/optimal parameter.

The three projections show the actual returned vertex positions and displacement
vectors. Arrow magnification changes drawing only. Geometry units are game units,
not millimeters. Captured expression and pose stay fixed; reference is the original
coefficients under the same declared operation protocol.

Implementation, call order and scope are documented in
[HS2Mod native-forward contract](../../../HS2Mod/docs/hs2_native_forward.md).
Only loopback hosts and same-origin requests are accepted. Routes are allowlisted
to context/evaluation; `/api/guide` invokes evaluations only. New contexts replace
previous ones. Changed characters/heads require another capture.

Status: bridge 0.31.3 passed the predeclared small final acceptance set: ten
same-frame real-actor cases on head 2, all ten declared surfaces, including the
three historical endpoint inputs, ear motion and selected ABMX combinations.
Actor restoration passed. The actual browser/proxy guide and real stdio MCP
tools also passed. No game samples were collected to infer formulas during
implementation. Scope and retained failures are recorded in
[HS2Mod final acceptance](../../../HS2Mod/docs/hs2_native_forward_acceptance.md).

This does not recertify the older offline world-LBS formula. New candidates are
computed on copies, not individually tested on the real actor. Active advanced
accessory parenting, male rigs and future animation/physics frames remain explicit
gaps; unsupported installed hooks/versions reject capture/evaluation.

Browser acceptance can be reproduced with `live_forward_ui.cjs OUTPUT_DIRECTORY`.
Expose an installed Playwright through NODE_PATH and optionally set
FORWARD_CHROMIUM to an existing compatible Chromium executable. A third argument
replays preserved context/evaluation/guidance receipts without new game calls.
