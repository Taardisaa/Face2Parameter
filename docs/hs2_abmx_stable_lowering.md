# Explicit first-apply target lowering

The optional compiler in `tools/abmx_stable_lowering` converts a declared clean radial-length-plus-offset target into the installed ABMX baseline position branch. It preserves the desired **first clean Apply** local TRS within float32 tolerance and intentionally changes subsequent behavior. It does not repair or redefine vanilla combined ABMX semantics, fit coefficients, infer call counts, write private caches, or operate the game.

For each supported bone, let `p` be the cached native59 predictor's local position, `r` the numerical historical `_lenBaseline` from an earlier stopped clean identity trace, `L` the logical Length and `d` its logical PositionOffset. The target is:

```
t = ((p / sqrt((px*px + py*py) + pz*pz)) * r) * L + d
physical.Length = 1
physical.PositionOffset = t - p
physical.Scale = logical.Scale
physical.Rotation = logical.Rotation
```

Float32 normalization and the two separate multiplications follow the installed source order. The executed position branch produces `_posBaseline + PositionOffset`; `_posBaseline` must agree with the independently predicted `p`. Cancellation in the subtraction/addition can cause a rounding difference. The compiler checks this against both the existing raw-TRS thresholds and an explicit component budget of `min(1e-5, 8*float32_epsilon*max(1,abs(p),abs(t)))`. No rounding coefficient is fitted.

Supported names are `cf_J_Chin_rs`, `cf_J_ChinTip_s`, `cf_J_CheekUp_L`, and `cf_J_CheekUp_R`. The first two have a parent-child relation: lowering each local position does not flatten the hierarchy. Parent-first FK and LBS must still use the modified parent scale and rotation. The compiler does not claim mesh quality or likeness acceptance.

The selected logical branch requires nonzero native and historical positions, positive historical radius, nonidentity logical Length at least 0.1, clean flags, a non-H scene, no additional modifiers, and no rotation exclusion. Logical Length exactly 1 is refused because installed ABMX selects the baseline position branch instead of radial normalization. Missing fields, incomplete/dropped/error traces, external inter-call boundaries, changed source hashes and unsupported names are refused. Same-native source/candidate is supported by this new context; the old changed-native `EarlierHistoryProtocol` diagnostic remains intact.

## Root-callable interfaces

```python
from tools.abmx_stable_lowering.compiler import compile_history, verify_execution_guard

history = {
    "trace": stopped_identity_trace_path,
    "trace_sha256": trace_sha,
    "geometry": {"path": paired_identity_geometry_path, "sha256": geometry_sha},
}
declared = {
    "head_id": 2,
    "native59": complete_candidate_native59,
    "source_history_native59": complete_earlier_identity_native59,
    "logical_patches": four_complete_named_logical_patches,
    "sampling_profile": "slider_unlocker_18_2",
    "source_files": {frozen_logical_protocol_path: protocol_sha},
}
compiled = compile_history(history, declared, installed_contract_path)
# Freeze compiled to a new path and SHA before applying any physical patch.
guard = verify_execution_guard(
    compiled, fresh_identity_geometry, installed_contract_path,
    current_trace_metadata=fresh_identity_trace_start_metadata,
)
physical_rows = compiled["executed_patches"]
```

The six declaration keys are exact. Additional capture metadata in `history` is allowed. A frozen declaration/protocol file must independently contain the requested head, native59, source native59, profile, and original logical patches. The compiler never accepts candidate local TRS/after data to generate a target. The identity trace's observed states are used to certify the earlier clean boundary and historical private fields, then source and candidate native TRS are calculated through `HeadRig`/`TorchHeadRig`. Actual original after states are not target inputs.

Output contains complete `logical_patches` and `executed_patches`, `expected_candidate_native_baselines`, per-bone logical and executed first predictions, historical radius/direction, first-equivalence errors, a context digest, original compiler inputs, and hashes for the declaration, trace, geometry, installed ABMX/Unity assemblies and decompiled source, cached head data, shared native deformation data, and predictor/replay/compiler source. The original candidate trace may be bound as immutable provenance by the manifest CLI, but its states are not consumed.

`verify_compiled_artifact` reopens and validates those source inputs, regenerates native locals and all targets, and compares every generated field. A coherent alteration of the executed patch plus its target/diagnostic is refused; changing logical inputs without changing the frozen original declaration is also refused. A self-digest alone is not accepted as evidence.

The fresh guard checks complete native59/head, clean current flags, exact float32 persistent `_lenBaseline`/`_positionBaseline`, current baseline and actual local TRS against the native predictor, current ABMX MVID/source, bridge MVID from the actual new trace metadata, game assembly SHA, same source actor, and current bone-ID mapping. New modifier/bone instance IDs are permitted and recorded freshly; old IDs are not used as proof. A different historical numerical radius or direction requires regeneration from a **new stopped identity history** before execution. Different actors are refused by this interface.

The guard needs full fresh geometry plus trace-start metadata: this geometry schema has no bridge MVID field, so private-cache JSON alone cannot establish all identities. It validates one read-only boundary, not absence of future writers. No head/native rebuild, baseline recollect, restore, animation writer, or modifier change may occur between the guard and physical application or within the claimed stable sequence.

## Verification and scope

Run with the project interpreter:

```powershell
& .venv/Scripts/python.exe -m unittest tools.abmx_stable_lowering.test_lowering -v
& .venv/Scripts/python.exe -m tools.abmx_stable_lowering.prepare --manifest ../HS2Mod/artifacts/infrastructure_live_20261005/abmx_stability_v1/live_cases.json --manifest-sha256 0c1dda10567e0ef9e4b3c19a5bf48b27ccedf63d4123ae68e98b17846ebfb154 --contract outputs/abmx_replay_20261005/installed_contract_v2.json --out outputs/abmx_stable_lowering_20261005/original_combined_source_only/compiled_execution.json
```

The suite covers a closed-form target, float32 operation order, 512 simulated position-branch calls, preserved scale/rotation, an analytical parent-child FK example, all radial-input derivatives, a real cached native59 directional derivative, special-branch refusals, stale sources/head/native/history, source-declaration tampering, coherent target/patch tampering, and bridge/actor/history guard refusal using the immutable original head2 source identity evidence. These certify the compiler/verifier behavior, not actual stable execution.

An actual acceptance run must collect new complete traces and source-bound whole-head geometry for early/late/far windows, independently replay every observed call and compare the complete surface to a separately reconstructed logical first target. Existing full-head and temporal tolerances stay fixed. Expressions, recorded ancestor transforms, and non-head meshes remain explicit nuisance/scope issues. No actual runtime stability, ocular/anatomical correspondence, safe mesh quality, or character readiness is claimed by this compiler artifact; those output flags remain false.
