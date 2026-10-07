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

Status: implementation and necessary mocked logic checks complete; new game
acceptance is pending. No game samples were collected during this implementation.
The page displays this limit explicitly; historical eye endpoint failures have
not been relabelled as fixed.
