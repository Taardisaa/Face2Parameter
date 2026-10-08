# Chenger deletion visualization and bottom-only candidate

The user rejected the head/neck cut after directly inspecting the original model
with deleted faces highlighted. The original source has 5,023 vertices and 9,976
triangles including eyeballs. The first `neck + boundary` mask cut removed 636
triangles. The installed `neck_v6` additionally removed 16 posterior triangles:
652 original triangles removed in total. `neck_v5b` had a different, rejected
28-triangle posterior cut; do not substitute its count for the installed version.

The initial cut was too broad. Preserving every face-mask vertex did not preserve
every incident triangle: 34 deleted triangles touched face-mask vertices, and
264 touched scalp-mask vertices. All 16 additional triangles touched scalp-mask
vertices. These intersection counts overlap; they are not disjoint area totals.
The visual yellow band lies at the lower posterior skull. Previous claims that
the cut only removed neck/base surface were therefore too strong.

## Explicit diagnostic view

`tools/native_head/deletion_view.py` compares literal full-source arrays with
the retained original face/vertex IDs, then keeps the entire original model.
It colors initially removed faces red, additional posterior faces yellow, and
retained faces gray. Canonical vertices, face order, original parameters and rig
remain unchanged; authored diagnostic UVs/colors are explicitly not identity
appearance. OBJ/MTL and face-ID metadata are exported for direct inspection.

The user explicitly requested runtime head replacement for this work, instead
of repeated game restarts. `/maker/face/model` is used for these review candidates.
This is a runtime source-model preview, distinct from the actual native base
asset pipeline. Existing native cards and original artifacts remain intact.

```powershell
.venv/Scripts/python.exe -m tools.native_head.deletion_view --source outputs/model_bridge_20261007/chenger_import_03111_v1/source.json --trim outputs/model_bridge_20261007/chenger_neck_native_v1/source_trimmed.json --neck-design outputs/native_head_20261007/neck_v6/neck_design.json --out outputs/native_head_20261007/deletion_view_v1 --show --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462
```

The actual runtime canonical and render-gather arrays were checked literally.
Native capture: HS2Mod `artifacts/chenger/deletion_highlight_v1_sheet.png`.

## Bottom-only candidate

After that inspection, the user requested removal of only a small bottom portion.
`tools/native_head/lower_trim.py` starts again from the full original source,
selects the lowest 5% of its vertical span as the bottom threshold, and removes
only triangles incident to vertices below it. This intentionally conservative
whole-triangle crop removes **28 triangles**, retains 9,948, and does not move
any remaining source vertices. None of the removed triangles touch face, scalp
or ear masks. The cut leaves the original upper neck and lower skull intact.

Rig vertex buffers and corner-UV mapping are literal indexed subsets; full-source
joint centres and original photo parameters stay unchanged. No contraction,
stretch, added connector, body edit or native zipmod replacement occurs in this
candidate. Its retained long neck still overlaps the game's body. **The seam is
not joined or visually finished.** The installed native base is not relabelled as
fixed on the strength of this candidate.

```powershell
.venv/Scripts/python.exe -m tools.native_head.lower_trim --source outputs/model_bridge_20261007/chenger_import_03111_v1/source.json --out outputs/native_head_20261007/bottom_only_v1 --fraction 0.05 --show --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462
```

Both candidates were loaded into the same running game process without restart.
Canonical Unity positions/triangles matched their exact requested arrays. The
bottom-only candidate remains visible and was saved separately as
`E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/程儿_底端裁切候选_01.png`.
Native final capture: HS2Mod `artifacts/chenger/bottom_only_v1_sheet.png`.
Fresh ignored output directories preserve all previous rejected attempts.

The earlier unfinished native builder/rim changes and overlay prototype drafts
remain separate working-tree changes; these diagnostic tools do not certify them.

## Local neck fit, continuing from the bottom-only candidate

`tools/native_head/neck_fit.py` now fits the retained lower neck locally. It
never returns to the aggressive mask crop. Every bottom-only parent triangle is
present, either unchanged or subdivided along the low boundary; no additional
original face is deleted. Face/chin/ear/eye vertices, upper cranial vertices and
their incident first ring are locked literally. Only the lower neck can move,
including its height; there is no overall or per-axis head scaling.

The body opening is recovered from exact bind-position/ordered-skin aliases and
the installed head-support skin weights. The captured Unity body and source
matrices convert its actual baked positions into the source frame. A constrained
biharmonic displacement distributes neck compression below the locks. A short
integrated annulus extends the actual body's adjacent one-ring surface upward.
Every native seam corner is inserted, with edge correspondence preserved between
the upper/lower rows; matching endpoint positions alone would leave polygon gaps.
This is explicit new mesh authoring, not a replacement for known HS2 skinning.

```powershell
.venv/Scripts/python.exe -m tools.native_head.neck_fit --source outputs/native_head_20261007/bottom_only_v1/source_bottom_only.json --native outputs/native_head_20261007/neck_local_v1/native_body.json --state outputs/native_head_20261007/neck_local_v1/prior_state.json --out outputs/native_head_20261007/neck_local_v1/reproduce --show
.venv/Scripts/python.exe -m unittest tools.native_head.test_neck_geometry
```

Current final candidate: `neck_local_v1/face_ring_locked/source_neck_fitted.json`.
289 original lower-neck vertices move. The original 28-triangle bottom crop is
unchanged. 86 seam/annulus vertices are added; the final mesh has 10,092 triangles.
Original-face provenance, literal protected positions and deterministic replay
are checked in `static_acceptance.json`; game import checks exact canonical arrays.
The same running Maker received the update without restart. The preview card is
`E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/程儿_局部脖子衔接候选_01.png`;
its save receipt confirms the source mesh is embedded.

The head still uses the Standard geometry-preview material, whereas the body
uses AIT/Skin True. The visible color boundary is unresolved. Current-body neutral
seam geometry is the scope: arbitrary poses/sliders/body configurations, normal
continuity, visual seamlessness and native zipmod integration are not certified.
FLAME parameters remain provenance; authored neck post-skin offsets and inserted
vertex buffers are explicitly distinguished from untouched model output. The
installed native zipmod is not updated by this preview tool.

Early `fitted`/`tangent_fitted` outputs remain preserved. The latter exposed an
edge-correspondence error when upper/lower azimuths differed; the final method
uses the same polygon edge/fraction and rejects any missing native corner.
