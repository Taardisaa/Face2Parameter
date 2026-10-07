# Independent held-out displacement guidance

Source-first follow-up (2026-10-06): installed `ShapeHeadInfoFemale` enums and all
113 Update assignments were reparsed and match this repository's cached tables.
The current recommendation still uses a game-measured finite secant instead of
the existing full forward deform model. All nine rejected native directions have
training windows crossing animation keyframe segments. Future guidance should
evaluate the source-derived keyframe/bone/skinning chain, then confirm selected
recommendations in the game. The retained review below describes the original
approximation and must not be read as a full implementation certificate. See
[installed source audit](../../HS2Mod/docs/hs2_native_source_audit.md).

`tools/parameter_atlas/review_guidance.py` checks newly collected game samples
against a SHA-bound predeclared plan. It never drives the game or adapts its
prediction to a held-out sample.

```powershell
.venv/Scripts/python.exe tools/parameter_atlas/review_guidance.py --manifest C:/Users/13666/Workspace/HS2Mod/artifacts/parameter_response_20261006/guidance_heldout_v1/manifest.json --out outputs/parameter_response_20261006/guidance_review_v1.json
.venv/Scripts/python.exe -m unittest discover -s tools/parameter_atlas -p test_review_guidance.py -q
```

The manifest contains a `plan_receipt`, fresh baseline geometry receipts and
native59 vectors, one case for each supported planned entry/sign, baseline
repeats, and the original/public-restored configuration. Training snapshots and
catalog JavaScript files remain local ignored data; all of their original hashes
are verified. Geometry uses the captured `cf_J_Head` coordinate frame and is
strictly scoped to `o_head`. This report does not repair the separately retained
eye endpoint failures or certify all captured renderers.

The reviewer reconstructs each directional gain from the original **unrounded
game geometry receipts**. For the configured requested target of 0.01% of the
training head bounding-box diagonal, its effective target is clipped to 40% of
the measured local training response. A target at or below three times measured
training repeat noise is recorded as unsupported. A skip cannot replace a
required passing native direction; all59 verification requires both signs of
every native control at every declared configuration. Selected ABMX directions
may explicitly remain noise guarded.

For a supported direction, the raw one-sided training vertex delta divided by
its signed training step predicts the newly requested signed step. The maximum
corresponding-vertex prediction error **and** error in achieved maximum
displacement must each be at most:

```text
0.05 * effective_target_units
+ 3 * training_head_diagonal * max(training_repeat_noise, fresh_repeat_noise)
```

The measured world LBS/BakeMesh gate, repeat drift gate and fresh-versus-training
baseline correspondence gate each retain a normalized limit of `1e-5`. Fresh
instance IDs may differ; source arrays, bindposes, bone-name palettes and full
blendshape source frames must agree. The catalog's complete public calibration
identity must match. Fresh baseline reuse uses exact vertex correspondence in
the recorded coordinate frame, with no fitted rotation, translation or scale.

Cases must form an exact bijection with the predeclared requests. Native
readback, other58 coefficients, isolated ABMX channel patches, captured
expression/blendshape weights and anchor local position/scale are checked.
Before/after public snapshot, expression and ABMX equality plus released physics
are required in addition to the restoration boolean. Failed geometry or
prediction witnesses remain failed report entries with their reason.

The resulting witness fields include `entry_id`, `sign`, `input_trusted`,
`verified_prediction`, `heldout_value`, `signed_step`, `target_units`,
`actual_max`, `max_vector_error`, `relative_target_error` and `error_budget`.
Catalog/plan/training/held-out/source-code receipts and per-baseline identity,
reuse and noise measurements bind the report to its actual evidence. A passing
point establishes that finite recommendation at that captured configuration;
it does **not** certify the entire continuous interval or an arbitrary target.

Nine analytical/evidence tests cover all59 two-sign coverage, missing/changed
requests, contradictory actual movement, changed native readback, mismatched
restoration, arbitrary noise skips, catalog JS tampering, altered recommended
steps, opposite movement with matching magnitude, baseline drift without fitting
and ABMX cross-axis changes. Live validation results are recorded in the
corresponding generated report rather than inferred from these fixtures.
