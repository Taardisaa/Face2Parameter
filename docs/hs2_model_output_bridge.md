# Existing model output bridge

The first implemented stage preserves the actual SMIRK outputs and the FLAME
geometry they generate. It does not train a converter, fit HS2 sliders, or
reconstruct a replacement photograph. HS2 parameter conversion and native head
integration remain separate stages.

The same artifact reader and unchanged-mesh interchange now also support official
MICA canonical identity outputs. Original `pred_shape_code`, canonical vertices and
identity embedding are retained; `shape_params` is a checked alias. MICA already
outputs canonical geometry, so head-local export does not invent pose/expression
parameters. Original MICA decoder replay uses its complete saved FLAME state and
own LBS, not SMIRK's decoder. The new interchange embeds model/checkpoint/source
hashes as provenance, which the existing card record retains with its raw JSON.
Actual source, commands and the limited paired-view quality findings are recorded
in [source model quality](source_model_quality.md). Export/replay alone does not
establish game acceptance or accurate neutral identity. The separate 0.31.6 live
source-rig acceptance below now includes MICA, within its declared scope.

## Export using the existing environment

```bash
$HOME/envs/smirk/bin/python /mnt/c/Users/13666/Workspace/Face2Parameter/scripts/smirk_export_geometry.py \
  --smirk-dir /mnt/c/Users/13666/Workspace/smirk \
  --in /mnt/c/Users/13666/Workspace/Face2Parameter/inputs/audited.jpg \
  --out /mnt/c/Users/13666/Workspace/Face2Parameter/outputs/model_bridge_20261007/smirk_raw_v1 \
  --device cuda
```

Use a new output directory. The artifact stores all encoder fields unchanged,
the original posed mesh and topology, model-defined landmarks, the crop and its
transform, model buffers and source hashes. An additional head-local mesh sets
only global `pose_params` to zero in a copy; expression, jaw, eyelids and shape
remain unchanged. This explicit coordinate preparation is not expression
neutralization. No generator or image renderer is used.

`tools.model_bridge.artifact.ModelArtifact` reads these artifacts and can replay
the saved parameters using the upstream `src/FLAME/lbs.py` and the exported exact
FLAME buffers. It resolves this machine's Windows/WSL paths and verifies source
hashes before replay. It does not substitute another face model.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh `
  --manifest outputs/model_bridge_20261007/smirk_raw_v1/manifest.json `
  --out outputs/model_bridge_20261007/smirk_raw_v1/source_head.json --head-local
```

This writes model vertices, triangles, raw parameters and provenance into the
`hs2_source_head_mesh_v1` interchange. Its coordinate placement, materials,
skeletal binding, expressions and persistence in HS2 are not implied by export.

## Implemented result, 2026-10-07

The existing checkpoint successfully processed the existing `inputs/audited.jpg`
and produced actual outputs under the command above. The original upstream LBS
replayed both exported geometry modes on Windows; the original posed mesh's
largest component discrepancy was `2.98e-8` in FLAME coordinates (CUDA export
versus CPU replay; fixed replay tolerance `1e-6`). Raw parameters were unchanged
before and after source decoding. The interchange contains the original 5,023
vertices and 9,976 triangles without fitting or remeshing.

This confirms parameter/geometry preservation at the source-model stage, not
photo likeness or compatibility with the HS2 deformation space. Outputs, model
buffers, weights and input photos stay ignored; reproducible source and commands
are tracked. Current cross-repository plan: `../HS2Mod/ROADMAP.md`.

## Native source mesh preview

Bridge 0.31.4 exposes GET/POST/DELETE `/maker/face/model`. Import the unchanged
head-local interchange with one positive uniform scale and a translation:

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.game_import `
  --mesh outputs/model_bridge_20261007/smirk_raw_v1/source_head.json `
  --receipt outputs/model_bridge_20261007/smirk_raw_v1/game_import_receipt.json
```

This command ran in female Maker on 2026-10-07. Unity's actual vertex and triangle
arrays equalled the exported arrays. Placement uses a native head bounds reference;
it never adjusts individual vertices or scales axes independently. The bridge
rejects a sheared, reflected or nonuniformly scaled parent at import time. The
preview follows the head rigidly and temporarily hides native head renderers.
DELETE restores their previous visibility. This is a fixed-expression geometry
preview with a plain Standard material; eye appearance, neck joining, native
facial controls and card persistence are not implemented in this version.

Raw export milestone: Face2Parameter `d11abe0465bc81d9b8b0ea1930a832d2ffc430c7`.
Runtime preview evidence remains ignored under the output directory above.

## Embedded card persistence (Bridge 0.31.5)

The registered HS2API character controller owns each character's mesh independently.
ExtensibleSaveFormat stores the full original interchange JSON, its SHA256, uniform
scale and translation in the character card. Reload reads that embedded content;
it does not read the interchange path, model checkpoint, FLAME buffers or photograph.
Normal card save/reload hooks are connected. The native HTTP loader explicitly reads
the current ChaFile, bypassing Maker's stale LastLoadedChaFile UI cache.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.card_roundtrip `
  --card C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_card_0315.png `
  --thumbnail C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_front.png `
  --receipt outputs/model_bridge_20261007/smirk_raw_v1/card_roundtrip_0315.json
```

This saves a **new** card, clears the preview, then replaces the current Maker
character by reloading that card. Existing card/receipt paths are refused. The
command passed in female Maker: source digest, actual vertex and triangle arrays,
uniform scale and translation survived. Native renderer visibility was restored
on clear. The thumbnail is reused from the earlier native capture; no new screenshot
or parameter sweep was needed. First import with the new persistence code failed
because generic JsonConvert loaded missing System.Data; the implementation now
uses the existing bridge JsonUtil serializer. Failure is retained separately.

This is persistable geometry preview, **not** a complete native base: fixed source
expression, plain Standard material, unjoined neck and no native facial controls.
The controller lifecycle supports character ownership, but Studio/multiple-character
scene persistence has not received live acceptance. Current reproducible integration
and source provenance are in `../HS2Mod/docs/hs2_model_parameter_bridge.md`.

The card command waits for the expected embedded record to become active after
load, because HS2API reload notifications may finish after the HTTP response,
especially during initial Maker loading. Vertex, triangle and placement equality
checks still apply to the final state; readiness waiting does not weaken them.
Final installed binary reload is recorded in `final_load_0315.json` in the same
ignored output directory.

## Original source rig (Bridge 0.31.6)

`--with-rig` emits `hs2_source_head_mesh_v2`. The unchanged decoded vertices,
triangles, parameters and provenance remain present. The additional bounded
`flame_lbs_rig_v1` data carries identity/expression-shaped vertices, original joint
centres, all five skin-weight columns, all 36 pose-corrective basis rows and
post-skin eyelid offsets. Buffers use explicit little-endian float32 base64;
parents must match the audited `[-1,0,1,1,1]` source hierarchy. No weight truncation
or normalization is performed. Both MICA and SMIRK use their own verified source
LBS to prepare these buffers, rather than a fitted deformation substitute.

The C# port follows original Rodrigues sampling (including `norm(vector+1e-8)`),
non-root rotation features, pose corrections, hierarchical joint transforms,
rest-joint subtraction with homogeneous w=0, weighted transforms and then SMIRK
eyelid offsets. Geometry is evaluated by this original CPU computation; the five
visible/debug transforms are not a substitute quaternion skinning implementation.
At import the port must reproduce the decoded source output within `1e-6` raw
units. Original-pose display and reset use the literal stored vertex array.

`POST /maker/face/model/pose` accepts only `{"pose":[15 axis-angle radians]}` or
`{"reset":true}`. Joint order is root, neck, jaw, left eye, right eye. Identity,
expression coefficients and eyelid coefficients remain fixed source outputs.
Active source pose is separate card state; raw model parameters are never rewritten.
The existing version-1 card record adds optional `active_pose`, preserving old
static cards. A rigged posed card reloads from its embedded payload without
accessing any source model file. This is source pose control, not HS2 expression
or native bone retargeting. Native sliders still do not deform this source mesh.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh `
  --manifest outputs/model_bridge_20261007/smirk_raw_v1/manifest.json `
  --head-local --with-rig --out outputs/model_bridge_20261007/smirk_raw_v1/source_head_rig.json

.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh `
  --manifest outputs/model_quality_20261007/mica_pair_raw_v2/manifest.json `
  --head-local --with-rig --out outputs/model_quality_20261007/mica_pair_raw_v2/source_head_rig.json

.venv/Scripts/python.exe -m tools.model_bridge.rig_acceptance `
  --smirk-manifest outputs/model_bridge_20261007/smirk_raw_v1/manifest.json `
  --smirk-mesh outputs/model_bridge_20261007/smirk_raw_v1/source_head_rig.json `
  --mica-manifest outputs/model_quality_20261007/mica_pair_raw_v2/manifest.json `
  --mica-mesh outputs/model_quality_20261007/mica_pair_raw_v2/source_head_rig.json `
  --backup-card C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/before_rig_0316.png `
  --thumbnail C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_front.png `
  --out outputs/model_bridge_20261007/rig_acceptance_0316_v2
```

The declared composite pose exercises all five joints. Expected geometry comes
directly from the original decoder with saved raw coefficients/model buffers,
not the exported rig or the C# implementation. Actual Unity vertices and posed
joints passed the fixed raw tolerance for both models; posed card save/clear/load
preserved geometry, source pose and placement. Reset recovered literal original
arrays. The supplied old static backup card was restored unchanged afterward.
No camera capture or geometry fitting was involved. Full numerical receipts and
NPZ geometry are ignored under the output path above.

The first acceptance helper run compared flattened interchange triangle indices
with the decoder's two-dimensional faces array and stopped before importing a
rig. Its failure/character-restoration receipt is retained under
`rig_acceptance_0316_v1`; the helper now compares the same declared array layout.
No game computation or tolerance was changed to make acceptance pass.

Installed 0.31.6 DLL SHA256:
`27b98fe39dc6a2ed940d61235fb7e30fea54b9da69e39fe9f416eb33c8d77ad7`.
Original LBS source SHA256: SMIRK
`e495f37fb0e5f1bb73ae958d2f75dfbeae0c9e757a3936686af7984c4ce4fbc7`;
MICA `8d7737eed5a22bae5b31c31cba1839d0c673de15684247357bc614256f1195c9`.

This resolves original source skin/pose computation in female Maker. Neck joining,
separate eyeball/mouth appearance, UV/material integration, native expressions,
runtime nonuniform body deformation and Studio/multi-character acceptance remain
open. It does not establish neutral photo-to-shape accuracy or a clean HS2 slider
mapping. Full offline renderer work remains deferred.

## Attachment asset findings after 0.31.6

The current imported full FLAME head is **not yet a replacement for every native
head surface**. Installed `ChaControl.LoadAsync` loads the real female render body
from `chara/oo_base.unity3d` using `ChaABDefine.BodyAsset(1)` (`p_cf_body_00`). Its
`o_body_cf` contains surface weighted to `cf_J_Head_s`, including vertices wholly
weighted to that bone. The existing preview hides native face/eye renderers but
does not remove that body surface. Thus old native posterior head geometry can
remain alongside the source full head. Original source-array preservation and
posed-card acceptance did not inspect or certify the composite head/body surface.
Do not call the present result a complete visible head replacement.

The actual `CmpBoneBody.targetEtc.trfHeadParent` pointer in `p_cf_anim` selects
`cf_J_Head_s`. This agrees with `LoadAsync` attaching `objHeadBone` under that
pointer. It is recovered from component data, not selected by bone-name similarity.
Positive head-bone weights alone are **not** an anatomical neck cutting mask.
The native render body's many indexed boundaries include UV/material seams, so
selecting the largest loop or deleting every head-influenced vertex is unjustified.
The next body replacement must establish the authored head/body seam and preserve
the remaining native body, then add an explicit connector without moving source
face vertices. No body deletion or vertex fitting has been implemented here.

Original FLAME topology has three connected surfaces: the head/neck and two
closed eyeballs. The head surface has two simple open loops, at the mouth interior
and neck. Source eyeballs already exist and must not silently be replaced by old
native eye geometry. Labels follow the source geometry/joint context; the stored
audit preserves the index sets and bounds instead of inventing native correspondence.

SMIRK's `Renderer.__init__` loads `assets/head_template.obj` UVs per face corner.
Its triangle order and vertex indices match the recorded FLAME faces exactly.
Its **template positions are different data and must not replace predicted vertices**.
The source UV layout needs 95 seam duplicates: Unity's one-UV-per-render-vertex
layout would gather 5,118 render vertices from the original 5,023 positions.
An exact corner gather can preserve every triangle's geometry while keeping all
UVs; choosing one UV per original vertex loses the seam. Pose evaluation must
remain on canonical source vertices, with positions/normals gathered afterward.
Actual textured rendering and this gather have not yet been integrated.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.attachment_audit `
  --manifest outputs/model_bridge_20261007/smirk_raw_v1/manifest.json `
  --source-obj C:/Users/13666/Workspace/smirk/assets/head_template.obj `
  --game-root E:/HoneySelect2_ArcticFox `
  --cha-control-source C:/Users/13666/Workspace/HS2Mod/tools/parameter_audit/source_model_bridge_20261007/ChaControl.cs `
  --cha-ab-source C:/Users/13666/Workspace/HS2Mod/tools/parameter_audit/source_model_bridge_20261007/ChaABDefine.cs `
  --out outputs/model_bridge_20261007/attachment_audit_v2
```

The command executed using source/model hashes and the exact render prefab;
collision/silhouette objects are excluded. Original index topology is retained,
with no tolerance welding, shape fitting, parameter changes or game sampling.
`report.json` contains component indices, boundary loops/degrees, bone weights,
actual head-parent pointer and source UV requirements. `native_body.npz` contains
the installed body mesh data; both stay ignored. Game assembly SHA256 remains
`d7e7e0d51403a0a3a71935a14487f063c6bf6f93f004b98b8f4b8b89047af770`;
body bundle SHA256 is
`9fc9f9cdea1067005fbbbb81a9b2a920c834f5721b1d267eea3f0c3c4a6e56c0`;
source UV OBJ SHA256 is
`dd5bfbce75adb99b1963f43bca7ec3557bd4a7321f8fc515f4100599e80d99f2`.

No new installed DLL or runtime geometry changes were made during this asset
audit. It changes the implementation order: native posterior head/body integration
must accompany the neck connector. A connector alone would leave old geometry
in the composite result. Source-model neutral accuracy remains a separate open item.

## Native neck cut and source collar construction

The previous description of retained body geometry as a **posterior head** was too
specific. Applying the actual `p_cf_anim` head-parent hierarchy places the vanilla
body's head-weighted surface at the upper neck/lower head interface. Bone influence
alone does not establish anatomical labels. This correction does not remove the
integration issue: the source FLAME includes its own neck, so merely retaining the
native upper-neck surface can leave an overlapping interface. The previous static
audit still establishes indexed topology and head support, not a visible skull test.

`tools.model_bridge.neck_patch` now constructs an explicit cut and separate collar.
This is **new mod attachment geometry**, not a claimed reconstruction of an existing
game neck-cut algorithm. The current source head vertices are never edited. Body
triangles outside the selected neck region retain their original indices and winding.
New cut endpoints store an original native edge and interpolation coefficient.
They must be evaluated from **actual posed native endpoints**: interpolating skin
weights and rest positions would introduce cross terms and would not preserve the
game's posed triangle surface. The body remains governed by native deformation.

A plane at the original neck-joint origin was tried as a construction definition.
Its source topology crosses shoulder-related surfaces, so it is not the selected
attachment design. `neck_cut_v1.json` retains that static proposal. The implemented
design places the plane below the lowest source neck-loop point, along the actual
native neck-up direction. Clearance is explicitly 5% of the native neck-to-head joint
distance. This is a declared collar design choice outside the preserved face, not
a fitted parameter multiplier or an inference about game slider behavior. The cut
selects plane-positive connected components seeded by pure head support; weights
alone are not the cutting mask.

One necessary read-only native snapshot was obtained using existing
`/maker/geometry?meshes=o_body_cf&include_actor_transforms=true&blendshape_frames=true`
and source-head state. No parameter writes, screenshots, sweeps, card changes or
DLL replacement were performed. The saved body has unit renderer scale; other
BakeMesh scale branches are explicitly refused. Current source pose is required.
Snapshot actor transforms locate the source head in the same body frame. Source
neck indices come from the explicit audited boundary, not a largest-loop heuristic.

The actual loaded body differs from the vanilla bundle: 11,033 versus 9,594 vertices,
with a different bone layout. The initial static-array preflight refused this mismatch
(`neck_cut_posed_v1_failure.json`). Installed Uncensor Selector **3.11.5** explains
the relevant branch: `ReloadCharacterBody` loads `BodyData.OOBase/Asset`, then
`UpdateMeshRenderer` replaces `sharedMesh` and `TransferBones` maps source bones by
name. The destination renderer name/path can remain `o_body_cf`, so it is not proof
that vanilla geometry is active. Defaults marked Random use a deterministic seed
from birthday, personality and voice pitch; default selection is not inferred from
mesh counts. Plugin SHA256:
`d3d994b275331df322c2a1110326bdfa3af5ec15a9894328c661fe7b5c8cb44c`.
Decompilation is ignored under `HS2Mod/tools/parameter_audit/source_model_bridge_20261007/UncensorSelector.cs`.
The exact selected zipmod/GUID remains unresolved; the proposal binds actual source
arrays and their geometry digest rather than pretending the vanilla bundle supplied them.

Actual source mesh data, original indices/weights/bindposes and current native
posed vertices are used together. This current body has no blendshapes; bodies
with blendshapes are refused pending explicit seam-equivalence handling. UV seam
copies are joined in the **connectivity map only** when their bind position and
ordered skin data match byte-for-byte. Their render vertices/UVs remain separate.
Edge evaluation uses identical endpoint order on seam copies. No spatial welding
tolerance or vertex averaging manufactures a closed ring.

The cut now has a closed neck boundary. A directed-boundary zipper connects it
to the source neck ring, adding only new triangles. Source and native ring indices
and collider design points remain explicit; no face coefficient or source vertex
is adjusted. Original untouched body triangles, opposing collar half-edges and
nonmanifold-edge absence passed final structural checks. These checks do **not**
certify runtime rendering, material continuity or absence of every possible
self-intersection/animation failure.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.neck_patch `
  --game-root E:/HoneySelect2_ArcticFox `
  --native-state outputs/model_bridge_20261007/neck_body_state_v1/native_body.json `
  --source-state outputs/model_bridge_20261007/neck_body_state_v1/source_head.json `
  --attachment-audit outputs/model_bridge_20261007/attachment_audit_v2/report.json `
  --out outputs/model_bridge_20261007/neck_cut_posed_v4.json
```

The command executed. Evidence remains ignored:
`neck_body_state_v1/`, `neck_cut_posed_v2.json` (pre-seam connectivity),
`neck_cut_posed_v4.json`, `neck_cut_posed_v4_validation.json`.
The v4 proposal stores renderer-local design vertices, source/native indices,
exact edge interpolation and original source-array hashes. It is not embedded in
the current source card and the installed DLL remains 0.31.6.

Next implement body/collar ownership and per-frame native endpoint evaluation in
the game bridge, preserving native material updates and actual body selection.
Use source canonical vertices for the other collar endpoint, release all temporary
surfaces on clear/reload, reject changed source topology, and then embed/reload the
attachment record. Native clothing/body visibility, UV/material integration, other
body scale/blendshape branches and selected asset provenance remain required work.
The face-parameter conversion and neutral source-model accuracy requirements remain open.

## Native body/collar runtime and card lifecycle (0.31.7)

`HS2Mod/plugins/HS2_McpBridge/SourceHeadAttachment.cs` now consumes the constructed
cut and collar. `attachment_runtime.py` packages exact native vertices/topology,
skin/UV hashes, bone order and source placement. An optional original-rig artifact
is accepted only when every decoded vertex and triangle equals the design source;
the new artifact hash is recorded explicitly. No template positions replace the
photo-model geometry. Source and native index spaces remain separate during
boundary checks even if numeric vertex indices happen to coincide.

The installed native body renderer remains the native skinning, material and
plugin target. Display-only cut geometry reads its current BakeMesh, preserving
all original vertices, normals, tangents, UVs and colors. New cut vertices are
interpolated **after** native deformation. The other collar ring uses the current
source vertices and exact source-to-body transform; original face vertices are
not edited. The collar is a declared new untextured surface, not a recovered game
deformation rule. Body shared material/property-block references follow the native
renderer. GameObject visibility follows ChaControl's original objBody, and added
renderers enter CmpBase's cached visibility list; release removes only those entries
and restores original native renderer visibility. Source-head visibility enters
the original face cache as well.

Native HTTP `POST /maker/face/model/attachment` takes `path` and `sha256`;
`DELETE` releases only the attachment. Existing source state includes attachment
status and same-frame native/collar geometry. There are no new stdio MCP tools.
Source card record and PluginData v2 embed attachment JSON/hash together with the
unchanged source artifact, placement and active source pose. Old v1 cards still
load. Pending/incompatible attachments cannot be saved. Uncensor's coroutine may
replace sharedMesh after HS2API OnReload; the character retries against the actual
body with a bounded pending window, and then only after mesh changes. Changed body
identity is never coerced back to vanilla.

The selected runtime BodyData is `bp.sac_innie_v2`, `[BP5] Innie 1`, with an empty
explicit GUID/default selection. `selected_body_asset.py` resolves the supplied
installed `[Female][HS2][BPV5]SAC_Innie.zipmod` manifest version6.1 and actual bundle
`abdata/chara/oo_base_bpsacinnie_v2.unity3d`, then checks all body geometry/skin/UV
hashes and bone order against the native descriptor. All match. Archive SHA256
`d3b2e3f0e5f0fb3404779ada171ed6f302ea9e8436d0150fa38fb8ef1bad4d0a`,
bundle SHA256 `4995d5a01cd375058c4e89c22db42f2ca1244c33eb25fd848dbcf035a6da824b`.
No extracted bundle is tracked or redistributed. Native normals are intentionally
outside the static identity hashes because original BustNormal changes them; the
renderer uses the actual current posed normals instead.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.attachment_runtime `
  --proposal outputs/model_bridge_20261007/neck_cut_posed_v4.json `
  --native-state outputs/model_bridge_20261007/neck_body_state_v1/native_body.json `
  --source-state outputs/model_bridge_20261007/neck_body_state_v1/source_head.json `
  --source-artifact outputs/model_bridge_20261007/smirk_raw_v1/source_head_rig.json `
  --out outputs/model_bridge_20261007/neck_attachment_rig_runtime_v1.json

.venv/Scripts/python.exe -m tools.model_bridge.attachment_acceptance `
  --attachment outputs/model_bridge_20261007/neck_attachment_rig_runtime_v1.json `
  --source outputs/model_bridge_20261007/smirk_raw_v1/source_head_rig.json `
  --backup-card C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/before_attachment_0317.png `
  --thumbnail C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_front.png `
  --out outputs/model_bridge_20261007/attachment_acceptance_0317_v2

.venv/Scripts/python.exe -m tools.model_bridge.selected_body_asset `
  --zipmod 'E:/HoneySelect2_ArcticFox/mods/PersonalMods - Exclusive HS2/Uncensor Selector/[Female][HS2][BPV5]SAC_Innie.zipmod' `
  --acceptance-receipt outputs/model_bridge_20261007/attachment_acceptance_0317_v1/receipt.json `
  --attachment outputs/model_bridge_20261007/neck_attachment_rig_runtime_v1.json `
  --out outputs/model_bridge_20261007/selected_body_asset_0317_v1.json
```

These commands executed. Output paths above are preserved and require fresh paths
to rerun. Installed DLL0.31.7 SHA256
`3933159f877a0ec1b06729c63880c6a9ae1c5b55e04cdf485ff252491b5b4da5`.
Final female Maker evidence `attachment_acceptance_0317_v2` passes literal source
face, incompatible skin rejection, one source neck/jaw pose, one native height
change, same-response endpoint geometry, posed attachment card reload/reset,
detach/clear and exact backup restoration. v1 is retained; v2 additionally asserts
that the declared source/body cases actually move their respective geometry. No
additional cases or relaxed thresholds were introduced. Detailed arrays and cards
remain ignored. This closes the runtime attachment gap, **not source accuracy**.

Supported body scope is readable single-submesh, no blendshapes, unit orthogonal
renderer frame and no cut across tangent handedness. Reference changes and basic
branch conditions are checked every frame; full identity hashes are checked on
bind, state read and save. An unknown plugin's in-place topology edits are not
continuously full-hash monitored. Arbitrary body variants, Studio, self-intersection
in extreme poses and shared-index watertight merging are not certified. Nonuniform
head-parent transforms are refused at import; ongoing protection against later
arbitrary ABMX head shear/nonuniform changes is still a gap. Source UV/material,
eye/mouth appearance, clean HS2 parameter mapping and neutral source identity
accuracy remain the next work. Existing paired-scan discrepancies cannot be
dismissed as a makeup-only effect.

## Source UV and explicit surface, Bridge 0.31.8

`surface.py` exports `hs2_source_head_mesh_v3`: the original canonical geometry,
parameters and rig remain unchanged. Source OBJ face corners must match the model's
literal topology and order. `(canonical vertex, authored UV index)` pairs create a
render-only seam gather; positions and normals are gathered after canonical rig
evaluation. No OBJ template position, tolerance welding or replacement shape is
used. The declared FLAME assets have 5,023 canonical and 5,118 render vertices.

The inspected SMIRK renderer uses its authored UV asset but constant gray shape
color. Official MICA defaults to white vertex color and its supplied head OBJ lacks
corner UVs. **Neither pipeline outputs inferred identity albedo.** MICA's v3 case
explicitly uses the topology-identical SMIRK UV OBJ as an external asset, not a
claimed MICA prediction. An optional externally supplied opaque PNG in that layout
is embedded with byte and decoded RGBA hashes. The native material is Unity
`Unlit/Texture`, sRGB, Clamp/Bilinear, without mipmaps. This is an explicit surface
display branch, not recovered appearance or game shader parity. Eye/mouth surface
appearance and textured neck blending remain unfinished.

Source files inspected: SMIRK `src/renderer/renderer.py` SHA256
`8d9c45d7c127715b9b0f24f35ed789b49afb5d8c43a81d288f458e0e0d06223e`;
official MICA renderer `550cb1f2b1ba4b5de82de613af1376642f8bba93b3d8ae7259d036447673d153`;
SMIRK `assets/head_template.obj`
`dd5bfbce75adb99b1963f43bca7ec3557bd4a7321f8fc515f4100599e80d99f2`.
The MICA OBJ's missing-UV rejection is retained as `surface_0318/mica_obj_uv_failure.json`.

Positive uniform world frames are now checked continuously. Nonuniform scale,
shear or reflection suppresses source display, releases the collar and restores
the native face/body fallback; returning to a supported frame restores the source.
No compensation is applied to canonical vertices. Native HTTP source-card save
refuses unsupported frames or unresolved attachments. This does **not** establish
that native Maker UI save aborts: HS2API catches controller save exceptions.

The first final run exposed stale render ownership after card reload: held arrays
were correct while the actual MeshFilter/material referenced copies. v1 failure is
preserved. `SourceRenderOwnership` now adopts only isolated equivalent copies with
literal position/topology/normal/tangent/UV/color equality and matching preview
material state, tracks their lifetime, and updates the actual bound asset. Unknown
replacements are refused. Actual source, cut-body and collar binding checks were
added. The responsible replacement caller remains unidentified; inspected direct
`MeshFilter.get_mesh` callsites did not establish the reload caller. No plugin is
blamed without source evidence. In-place edits to an already owned reference are
not continuously full-content monitored.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.surface `
  --mesh outputs/model_bridge_20261007/smirk_raw_v1/source_head_rig.json `
  --source-obj C:/Users/13666/Workspace/smirk/assets/head_template.obj `
  --texture outputs/model_bridge_20261007/surface_0318/diagnostic_uv_chart.png `
  --out outputs/model_bridge_20261007/surface_0318/smirk_surface.json

# Explicit external UV for MICA; no texture or albedo prediction invented.
.venv/Scripts/python.exe -m tools.model_bridge.surface `
  --mesh outputs/model_quality_20261007/mica_pair_raw_v2/source_head_rig.json `
  --source-obj C:/Users/13666/Workspace/smirk/assets/head_template.obj `
  --out outputs/model_bridge_20261007/surface_0318/mica_surface_explicit_uv.json

.venv/Scripts/python.exe -m tools.model_bridge.attachment_runtime `
  --proposal outputs/model_bridge_20261007/neck_cut_posed_v4.json `
  --native-state outputs/model_bridge_20261007/neck_body_state_v1/native_body.json `
  --source-state outputs/model_bridge_20261007/neck_body_state_v1/source_head.json `
  --source-artifact outputs/model_bridge_20261007/surface_0318/smirk_surface.json `
  --out outputs/model_bridge_20261007/surface_0318/smirk_attachment.json

.venv/Scripts/python.exe -m tools.model_bridge.surface_acceptance `
  --smirk outputs/model_bridge_20261007/surface_0318/smirk_surface.json `
  --mica outputs/model_bridge_20261007/surface_0318/mica_surface_explicit_uv.json `
  --attachment outputs/model_bridge_20261007/surface_0318/smirk_attachment.json `
  --backup-card C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/before_surface_0318.png `
  --thumbnail C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_front.png `
  --out outputs/model_bridge_20261007/surface_acceptance_0318_v2
```

Commands executed; use fresh output paths to repeat them. The supplied acceptance
texture was a generated 32×16 RGB diagnostic chart, not a face/model texture. PNG
SHA256 `6a8850314b9a351245bc43e2c4a71c3633e02140b9910d97cafa6894aab4dcad`;
SMIRK v3 artifact `090ed375cfdb7358a9e4fe902ed07da4072637d7bb6828c0f212db914ba6e60c`;
MICA v3 artifact `5b718ebae9d3b490c33b965e568dd5f9d6617ddccc2bb873c7a6786f2bf1f1c0`;
attachment `d00977624ea1d33b8c3284f99f10c6a7de1bdc1f6d5ed3690c7b1b0a5e0af1c0`.
Installed DLL0.31.8 SHA256
`aeefd027248d5e0788072aa2bc76d0bd91e5fbccf0f3710a19f19b2572738150`.

`surface_acceptance_0318_v2/receipt.json` passes the same predeclared cases after
the source fix: both models' literal canonical/render gather, authored UVs and
actual binding; one source pose each; embedded-card reload with external JSON/PNG
temporarily absent; exact reset; one nonuniform ABMX head-frame rejection and
recovery; source/body/collar binding; original backup restoration. v1 is retained,
thresholds unchanged. These are geometry/asset/lifecycle checks, not appearance
similarity or source-model accuracy measurements. `attachment_acceptance.py` now
requires the 0.31.8 binding fields; the historical 0.31.7 receipt covered its older
declared assertions. No screenshot campaign or new model inference was performed.

Next work remains independent matched neutral geometry evidence, explicit eye/mouth
appearance and correspondence, and clean native parameter mapping feasibility.
The paired anger-scan discrepancies and missing neutral truth are unchanged; see
[source model quality](source_model_quality.md). No full character fidelity claim.

Paired 0.31.8 implementation commits: HS2Mod
`21478b4a67186516eb933051bf8ff78fb402ccdc`; Face2Parameter
`6352aeedfbd915fdcac64c1b5037804b49fe031d`. Both pushed normally to their
configured origin branches. This note adds no geometry or accuracy claim.

## 0.31.9: original eyeball components and separate appearance slots

`components.py` derives the three original index-connected FLAME parts and
requires exact equality of the left/right eyeball components to the pinned
authored masks. Canonical vertices, topology, raw parameters, the complete original
five-joint rig and corner UVs remain literal. Small non-eye skin weights are kept;
there is no rigid-eye approximation. The source contains a head mouth boundary,
but no separate teeth/tongue asset. This exporter does not invent those parts.

Pinned mask SHA256:
`ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492`.
Component vertex counts are head3931, left546, right546; authored OBJ provenance
remains the 0.31.8 UV hash above. Original triangles interleave the parts. Six
maximal contiguous material runs, rather than three reordered face groups, keep
the complete face-corner sequence unchanged. `flame_component_uv_v1` explicitly
distinguishes this layout from the older single-material surface; older bridges
reject the new format. The total artifact still has the native 16 MiB bound.

Native `ChaControl.ChangeEyesKind/ChangeEyesWH` operates on native `rendEyes`
materials through `ChaShader.PupilTex/PupilLayout`, including original slider
Lerp semantics. It is not a correspondence to source eyeball UVs. The inspected
installed decompilation SHA256 is
`e7c303c4ca47f5dbfffd04f393ea11de207871d30d759ce9b5632400e37058a1`.
0.31.9 instead accepts optional explicit left/right opaque PNGs in the source
UV layout. Each overrides that component's appearance; absent overrides inherit
the existing head appearance/preview. All image byte/pixel checks, embedding,
similarity guards and native cut/collar support remain as declared previously.
No photo albedo, native eye-material retargeting or native facial animation is
claimed. Texture images in this run are diagnostic colors only.

Executed preparation from Face2Parameter (existing 0.31.8 inputs retained):

```powershell
# Reproduce the two explicit 32x16 diagnostic images. Use a fresh output directory.
@'
from pathlib import Path
import numpy as np
from PIL import Image
p=Path('outputs/model_bridge_20261007/components_0319')
p.mkdir(exist_ok=False)
y,x=np.indices((16,32))
rgba=np.stack([x*8,y*16,np.full_like(x,80),np.full_like(x,255)],-1).astype('uint8')
Image.fromarray(rgba).save(p/'left_eye_diagnostic.png')
rgba[:,:,:3]=rgba[:,:,[2,0,1]]
Image.fromarray(rgba).save(p/'right_eye_diagnostic.png')
'@ | .venv/Scripts/python.exe -

.venv/Scripts/python.exe -m tools.model_bridge.components `
  --mesh outputs/model_bridge_20261007/surface_0318/smirk_surface.json `
  --mask C:/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl `
  --left-eye-texture outputs/model_bridge_20261007/components_0319/left_eye_diagnostic.png `
  --right-eye-texture outputs/model_bridge_20261007/components_0319/right_eye_diagnostic.png `
  --out outputs/model_bridge_20261007/components_0319/smirk_components.json

.venv/Scripts/python.exe -m tools.model_bridge.components `
  --mesh outputs/model_bridge_20261007/surface_0318/mica_surface_explicit_uv.json `
  --mask C:/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl `
  --left-eye-texture outputs/model_bridge_20261007/components_0319/left_eye_diagnostic.png `
  --right-eye-texture outputs/model_bridge_20261007/components_0319/right_eye_diagnostic.png `
  --out outputs/model_bridge_20261007/components_0319/mica_components.json

.venv/Scripts/python.exe -m tools.model_bridge.attachment_runtime `
  --proposal outputs/model_bridge_20261007/neck_cut_posed_v4.json `
  --native-state outputs/model_bridge_20261007/neck_body_state_v1/native_body.json `
  --source-state outputs/model_bridge_20261007/neck_body_state_v1/source_head.json `
  --source-artifact outputs/model_bridge_20261007/components_0319/smirk_components.json `
  --out outputs/model_bridge_20261007/components_0319/smirk_attachment.json

.venv/Scripts/python.exe -m tools.model_bridge.surface_acceptance `
  --smirk outputs/model_bridge_20261007/components_0319/smirk_components.json `
  --mica outputs/model_bridge_20261007/components_0319/mica_components.json `
  --attachment outputs/model_bridge_20261007/components_0319/smirk_attachment.json `
  --backup-card C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/before_components_0319.png `
  --thumbnail C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_front.png `
  --out outputs/model_bridge_20261007/components_acceptance_0319_v1
```

The backup was saved through the existing native HTTP source-card writer before
restarting the verified game process with the new DLL. This run reused the
existing thumbnail, took no new screenshots, and ran no new model inference.
Before importing, static checks confirmed literal canonical/rig/raw-output/UV
arrays against the two existing inputs. Original pose geometry was replayed through
each model's original LBS source, not reconstructed from exported rig arrays.

Source hashes: left PNG
`d2d08065ff55683a82082e5776ccd4a63d2e0f7cef2a003e6628d4be559a2ec6`;
right PNG `5e29a056d709b76b1bab87cd67ced62b4bfa3e8b1c8bf727d72403075160291c`;
SMIRK component artifact
`509c89c7bfb8fdb3020ad1dfa96d9ee74d17771d4427917175d9a6508525d601`;
MICA `0f34c178ff40625c38d46470690c773e00d495ece42a76524c1bf90045e52834`;
attachment `5df3fbe2a916f7feb6d765578b4aa00d7aafceb42c783c3516206227159be0c4`.
Installed0.31.9 DLL
`947d5b0969cd0f10c2f77a77f343fe566c2cfdd0963b6f9bab239a0999abcf1c`.
Receipt `components_acceptance_0319_v1/receipt.json` SHA256
`287d5d3c495f73a1c2f66151df162963036f6c4894973386ffc9756161149f75`.

Final acceptance passed both models' canonical positions/topology, full render
face order/UV gather, literal component partitions, actual per-run material/PNG
and decoded-pixel hashes. One composite neck/jaw/left/right-eye pose per model
matches original decoder vertices and joints within the existing1e-6 raw bound
(vertex maxima SMIRK2.98e-8, MICA1.49e-8). External JSON and all explicit PNGs
were temporarily absent for embedded-card reload; posed geometry was preserved,
then reset to literal originals. The existing one nonuniform ABMX guard and native
body/collar actual binding checks passed; original backup restored. No sweeps,
threshold changes, numerical fitting or source-model accuracy claims.

A source-found old v1/v2 ownership issue also synchronizes an attachment's borrowed
canonical mesh after adopting an isolated exact render copy. That old non-surface
branch was not independently exercised by this v3 final run. Already owned objects'
arbitrary in-place edits remain outside the continuous ownership check scope.

Next integration gaps are genuine appearance/inner-mouth asset provenance and
retargeting boundaries, plus the remaining clean native parameter conversion
scope. Independent model accuracy remains governed by
[source_model_quality.md](source_model_quality.md): one associated neutral tracked
case does not certify arbitrary photos, and this geometry-preserving material
integration does not alter or supersede its findings.

Paired 0.31.9 implementation commits: HS2Mod
`e4820991c4572d095c5a90e666ca39b446c41e09`; Face2Parameter
`5d84250ba3f8fe6fd4b5ebe4cbee5d1ae3864eb3`. Paired notes do not change
the verified installed DLL or the original-decoder/accuracy evidence above.

## Selected photo provenance and native oral driver audit

Export now records `source.image_index` as well as the existing photo/NPZ and
manifest hashes. `ModelArtifact.from_game_source` validates that exact tuple and
the recorded model/checkpoint/source metadata. Legacy exports resolve a unique
matching photo+NPZ tuple; missing or ambiguous matches are refused rather than
silently selecting row zero. The final surface reference decoder now uses this
resolver. Its previous unconditional row-zero lookup would select the wrong
oracle for a second-image export. Both already executed0.31.9 game cases actually
selected row zero, confirmed from their source hashes; their evidence is unchanged.

The normal raw-model exporter can now include original rig, UV and authored
component slots in one invocation. This reduces intermediate artifact handling
while preserving source outputs; all branches enforce the existing native16 MiB
limit. Explicit optional eye textures still require the authored masks and UVs.

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh `
  --manifest outputs/model_quality_20261007/multiface_mica_raw_v1/manifest.json `
  --image-index 1 --head-local --with-rig `
  --source-obj C:/Users/13666/Workspace/smirk/assets/head_template.obj `
  --source-mask C:/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl `
  --out outputs/model_bridge_20261007/selected_image_source_v1/mica_second_components.json

.venv/Scripts/python.exe -m tools.model_bridge.export_game_mesh `
  --manifest outputs/model_quality_20261007/multiface_smirk_raw_v1/manifest.json `
  --image-index 1 --head-local --with-rig `
  --source-obj C:/Users/13666/Workspace/smirk/assets/head_template.obj `
  --source-mask C:/Users/13666/Workspace/smirk/assets/FLAME_masks/FLAME_masks.pkl `
  --out outputs/model_bridge_20261007/selected_image_source_v1/smirk_second_components.json
```

Executed on the existing raw neutral exports, without new inference or game calls.
Both select the second photo and its original identity/parameters, preserve literal
canonical vertices/topology, and replay that selected model's original decoder.
The recorded first-photo identity differs, so accidental row-zero selection is
detectable. Existing1e-6 replay bounds pass. Six artifact tests cover original
preservation, source-change rejection, second-photo selection, wrong tuple/index,
and ambiguous legacy records. Evidence `selected_image_source_v1/receipt.json`
SHA256 `ebb7ba8642341fc93732f4c92040bbde2a10ac81da5b1049c71702ee905da02c`.
This is provenance/preservation evidence, not additional model-accuracy evidence.

For mouth integration, `oral_audit.py` resolves the saved vanilla headID through
fresh installed lists and selects the exact render prefab, excluding other heads
and hit meshes. It reads the serialized native `FaceBlendShape` targets, per-target
Close/Open indices, original blendshape channels and skin bones, with code/asset
hashes. Sideloader GUID remapping and current runtime asset identity are explicitly
outside this static vanilla resolution, not silently assumed.

The selected head2 prefab's MonoScript points to **IL.dll**, not either
Assembly-CSharp assembly. Actual `FaceBlendShape` calls `MouthCtrl.CalcBlend`
in the UniRx late-update path. `FBSCtrlMouth` calls `FBSBase.CalculateBlendShape`:
original OpenMin/OpenMax/corrected/fixed rate, integer0..100 opening weights,
pattern transitions, then each target's own Close/Open indices. Teeth and tongue
are separately skinned meshes, both using `cf_J_MouthCavity`; their opening is
also driven by their distinct blendshape sets, not a standalone lower-jaw skin
bone. The controller also includes a tear target. Same pattern indices cannot
be copied across these meshes; same names do not establish a FLAME correspondence.

Original FLAME output contains the original head/eyeballs and mouth boundary, not
separate authored teeth/tongue geometry. Its jaw is a three-value axis-angle
rotation with full LBS and pose correctives. A native mouth rate is a different
control mechanism. No gain fit, index copy, rigid graft or automatic native inner
mouth retargeting has been implemented or declared compatible.

Executed decompilation and audit commands:

```powershell
# From HS2Mod; outputs are ignored local source material.
ilspycmd -t FaceBlendShape E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/IL.dll |
  Set-Content tools/parameter_audit/source_model_bridge_20261007/FaceBlendShape.cs -Encoding utf8
ilspycmd -t FBSBase E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/IL.dll |
  Set-Content tools/parameter_audit/source_model_bridge_20261007/FBSBase.cs -Encoding utf8
ilspycmd -t FBSTargetInfo E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/IL.dll |
  Set-Content tools/parameter_audit/source_model_bridge_20261007/FBSTargetInfo.cs -Encoding utf8
ilspycmd -t FBSCtrlMouth E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/IL.dll |
  Set-Content tools/parameter_audit/source_model_bridge_20261007/FBSCtrlMouth.cs -Encoding utf8
ilspycmd -t AIChara.CmpFace E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/Assembly-CSharp.dll |
  Set-Content tools/parameter_audit/source_model_bridge_20261007/CmpFace.cs -Encoding utf8

# From Face2Parameter, using the existing ChaControl decompilation as well.
.venv/Scripts/python.exe -m tools.model_bridge.oral_audit `
  --card C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/before_components_0319.png `
  --source-manifest outputs/model_bridge_20261007/smirk_raw_v1/manifest.json `
  --game-root E:/HoneySelect2_ArcticFox `
  --decompiled-dir C:/Users/13666/Workspace/HS2Mod/tools/parameter_audit/source_model_bridge_20261007 `
  --out outputs/model_bridge_20261007/oral_asset_audit_v2.json
```

Installed IL.dll SHA256
`2034ec89ec868feb44c161d71f06cac67b276853a71ba5dc0d0b8b1964c9f680`;
report `oral_asset_audit_v2.json`
`64789d376a8e1282e7ce2ccc3d67ba084cf9e48a2ddd03433a0fba8635d5ddc1`.
v1 is retained; v2 adds auditor hash and explicit vanilla-resolution scope.
No game mutation, screenshot, new numerical dataset or DLL change this milestone.

The next executable gap is also explicit: `attachment_runtime.package` permits
an alternate source only when every original vertex and triangle equals the old
attachment design's input. That protects rig/UV packaging, but does not attach
another photo's genuinely different identity to the body. A broader original
FLAME identity needs a new descriptor derived from verified shared topology and
actual placement/body state. Do not remove the old equality gate or relabel the
old single-face collar as a generic photo importer. Oral grafting additionally
needs authored source-space parts or a justified oral placement/driver relation.

Paired implementation commits for this source/provenance milestone: HS2Mod
`507676f84fb08cb3f76252090305eb2442335187`; Face2Parameter
`44c9930cba3388852f35a3b707a1cf8cc63467ef`. Installed DLL0.31.9 was unchanged.

## Different photo identities: new attachment descriptor

`tools.model_bridge.attachment_rebase` now provides a separate path for a genuinely
different original FLAME face. The old `attachment_runtime.package` equality gate
is unchanged. The new path first validates the old source/body cut design, then
validates both artifacts against their selected original photo/NPZ/decoder state.
It requires literal common template, complete face order, hierarchy, joint
regressor, full skin weights and pose correctives. Shape/expression basis sizes
need not be equal: SMIRK has350 coefficients and MICA400 in these saved states;
each artifact's original parameters, generated geometry and rig remain intact.
This is a shared topology/rig branch, not a claim that all model definitions match.

The new descriptor retains the original native cut edge indices/T and collar
connectivity, binding them to the new source hash and its **actual** positive
uniform placement. Source neck boundary half-edges must still oppose the collar.
It rejects changed topology/rig, posed or modified preparation geometry, unbound
display, a second active attachment and incompatible frames. It never welds,
fits or adjusts the new source face. At activation, the existing native runtime
independently checks the actual body vertices/indices/UV/skin/bindposes/bone order;
the saved body is not evidence for a different installed body.

The installed0.31.9 cut body and collar already follow actual native BakeMesh
and current source canonical endpoints. No DLL change was needed. This supports
the validated body branch and common FLAME index definition, not arbitrary body
meshes or collision-free collars for all identities/poses. Connector geometry is
new authored construction; it is not recovered native facial deformation.

Executed final acceptance, after implementation and12 preservation/rejection
logic checks:

```powershell
.venv/Scripts/python.exe -m tools.model_bridge.rebase_acceptance `
  --smirk outputs/model_bridge_20261007/selected_image_source_v1/smirk_second_components.json `
  --mica outputs/model_bridge_20261007/selected_image_source_v1/mica_second_components.json `
  --proposal outputs/model_bridge_20261007/neck_cut_posed_v4.json `
  --native-state outputs/model_bridge_20261007/neck_body_state_v1/native_body.json `
  --reference-state outputs/model_bridge_20261007/neck_body_state_v1/source_head.json `
  --thumbnail C:/Users/13666/Workspace/HS2Mod/artifacts/model_bridge_20261007/source_head_front.png `
  --out outputs/model_bridge_20261007/new_identity_attachment_acceptance_v2
```

Both cases select original image_index1, with different source vertices from the
old reference. Actual canonical/render mesh and UV/material bindings, unchanged
native body vertices, new collar endpoints/topology, one neck/jaw/eye composite
pose per model versus the original decoder, card reload with both source and
descriptor files absent, original reset, detach and backup restoration passed.
The native body/ABMX algorithm is unchanged; no new body sweep or guard campaign.
Final receipt SHA256
`bb4430e5aa4b1eb7422ee51179846b718cd0f7bb3c68de4118b7db2b97b0f66e`.
No new model inference, screenshots or empirical deformation fit.

v1 is retained: its new check erroneously compared Python's full decimal spelling
of a float32 placement with the native JSON spelling. Even the initial wire
state differed in spelling while its float32 values were identical; posed and
reloaded geometry were literally identical. The fix compares saved wire state
literally and descriptor values as literal float32, with no tolerance increase,
rounding or compensation. The geometry/decoder bounds remain unchanged. The
existing surface checker also no longer overwrites its evidence filename with a
component label; original/posed/reloaded/reset evidence stays separate.

This closes the different-photo descriptor gate, not source-model accuracy or
native slider correspondence. Original FLAME still supplies no teeth/tongue or
identity albedo; importing a face does not certify it resembles its photograph.
Independent findings remain in [source_model_quality.md](source_model_quality.md).

Paired implementation commits: Face2Parameter
`a7781795ab943d15b677cf3177b8df4a16ba569e`; HS2Mod
`97f480491d16f53b9c6bc594f65a8992f5da6e48`. Both pushed normally. Generated
cards, native states, source model arrays and acceptance evidence remain ignored.

### Visible state correction

The user-visible game still contained the old v1 SMIRK diagnostic head shell after
the preceding runs. Their backup restoration proves literal pre-run **source**
state restoration, not that an ordinary native character was left on screen.
The old head had no rig, texture or collar; existing native hair overlapped it.
Native capture and current state confirmed this distinction. The diagnostic was
saved to `user_visible_cleanup_v1/old_test_state.png`, then cleared; native
renderers were restored and a native screenshot confirms normal face/neck display.
Original character parameters were not changed. The underlying native character
is now active, with no imported source head. A usable import command must support
backing up this ordinary state too, not require a pre-existing diagnostic source.
The previous artifact/decoder preservation claims do not become appearance or
photograph-accuracy claims because their pre-run states were restored.
