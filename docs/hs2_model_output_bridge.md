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
