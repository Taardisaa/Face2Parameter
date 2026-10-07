# Live head parameter response

`tools/parameter_atlas/live_response.py` reads saved game snapshots. It never calls
the game or changes a character. This supplements the existing cached atlas:
the character's complete 59-control vector, selected head and fixed expression
are captured, rather than assuming every other control equals 0.5.

```powershell
.venv/Scripts/python.exe tools/parameter_atlas/live_response.py <manifest.json> --out <review.json>
.venv/Scripts/python.exe tools/parameter_atlas/live_response.py <manifest.json> --head-only --out <head_surface_review.json>
.venv/Scripts/python.exe -m unittest tools.parameter_atlas.test_live_response -q
```

The collector is owned by the companion HS2Mod repository:
`tests/geometry_export/live_parameter_response.py`. A manifest contains named
`baselines` with `{native59, geometry}`, and `cases` with `name`, `kind`,
`baseline_name`, `native59`, and `geometry`. Native cases also declare integer
`control`, `value` and `probe_role` (`local_minus`, `local_plus`, `native_min`,
`native_max`, `extended_min`, `extended_max`). `geometry` is a path and SHA-256 of
the **uncompressed** JSON bytes, even when stored as `.json.gz`. The independently
read predeclared plan has a separately checked SHA-256. Role requirements are
reported per baseline: one configuration can test endpoints while a second tests
only local responses. Incomplete pilots never establish complete coverage.

`--head-only` produces an explicit `analysis_mesh_scope: ["o_head"]` component
report. It checks every original raw renderer's source identity and expression
metadata, but evaluates LBS/BakeMesh agreement, displacement, response slopes,
repeat drift and prediction coverage only for the unique captured `o_head`.
Its component coverage does **not** certify eyeballs or other renderers:
`full_captured_mesh_set_certified` remains false. The default command still checks
the whole captured renderer set at the same threshold. A separate head component
report never replaces or upgrades an earlier failed full-renderer report.

## What the reviewer checks

Every source snapshot includes actual expression frame deltas. Probe snapshots
may omit those large arrays, but frame reuse requires identical renderer path,
renderer and source mesh instance IDs, source geometry, bone palette, weights,
bindposes, topology, and blendshape names/indexes/frame weights. Sharing the name
`o_tang` does not establish correspondence; all renderer paths remain separate.
An offscreen render can clone source meshes, so collect baseline source snapshots
**after** any setup render. Changing meshes requires a new source snapshot.

Snapshot-local LBS uses actual captured bone matrices and actual blendshape
weights. It is compared with Unity BakeMesh and both recorded world-conversion
candidates. Only candidates passing the numerical gate are used. Receipts,
readback of all 59 coefficients, fixed other native coefficients, public ABMX
modifiers and expression weights are checked independently.

Surface coordinates use a recorded ancestor transform, normally `cf_J_Head`.
There is no fitted per-probe registration. Anchor identity, position and scale
must remain unchanged. By default rotation is checked too; an explicit manifest
policy may permit removal of observed rigid head rotation when source control
destinations exclude that anchor. Each observed rotation change is reported.
Direct ABMX modification of the selected normalization anchor is rejected.

Repeated baseline samples detect residual motion. The gate is maximum vertex
displacement divided by the original head's bounding-box diagonal, default
`1e-5`. It is evaluated for each renderer **and** for the whole captured mesh
set. A stable `o_head` does not certify drifting eyeballs. A repeated baseline is
mandatory before complete coverage can pass; expression drift cannot be excused
by fitting the geometry. The report records incomplete or failed coverage rather
than silently widening the threshold.

## Measurements and tuning

Every accepted case reports maximum, RMS and percentile vertex displacement,
centroid movement and X/Y/Z spans, using captured anchor-local game units.
These are not certified millimetres. Normalization by the captured head diagonal
makes relative changes readable without claiming physical units.

For each control and renderer, paired minus/plus samples give a numerical surface
derivative. `max_surface_units_per_parameter_unit` answers how much the most
affected vertex moves per coefficient unit near **this** configuration.
`predicted_max_for_plus_0_01` gives its local estimate for increasing that control
by 0.01. `axis_span_slope_xyz` shows whether the mesh becomes wider, taller or
deeper in the recorded coordinate axes. These are local estimates; endpoint
measurements and another baseline expose nonlinearity/configuration dependence.
They do not identify calibrated anatomy or recommend changing a control merely
because its numeric value is large.

An optional cached native `o_head` predictor also checks the source cache against
captured vertices, topology, bone names, influences and bindposes. A recorded
baseline renderer/anchor uniform-scale ratio and one baseline rigid transform
are reused unchanged for all probes. No probe gets a fitted scale or alignment.
Both absolute and baseline-to-probe response residuals are reported, with a
separate prediction coverage gate. ABMX replay is not silently assumed by this
native predictor. Missing cache data leaves an explicit unavailable diagnostic
while preserving the independent game measurements.

ABMX cases have isolated bone/channel patches and captured public readback. They
receive the same geometry metrics, but a small selected-bone experiment cannot
certify every ABMX bone, channel, interaction or runtime history. Real-time
expression behavior is intentionally outside this measurement task; fixed actual
blendshape data is still included because omitting it corrupts skinning parity.
