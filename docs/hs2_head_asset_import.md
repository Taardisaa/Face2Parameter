# Actual HS2 head-base asset route

2026-10-07. Current decision: build a registered head AssetBundle/zipmod using
the game's actual head-prefab contract. The old SourceHead overlay/display
replacement prototype does not satisfy this requirement and is not the basis
of this route.

The source-first design and exact loading/registering steps are recorded in
[HS2Mod's native head-base audit](../../HS2Mod/docs/hs2_native_head_base.md).
The game retains its body/common face skeleton, **destroys the old objHead**,
loads the selected prefab as the real objHead/CmpFace, copies the new prefab's
same-name rest TRS into the common skeleton and rebinds its skinned renderers.
It loads the new row's shape table/material/skin and expression references.
Retaining the public rig is native behavior; it does not imply overlaying a
second head on the old model.

Independent native-list/prefab audit:

```powershell
.venv/Scripts/python.exe -m tools.head_base_audit --game-root E:/HoneySelect2_ArcticFox --head-id 2 --zipmod 'E:/HoneySelect2_ArcticFox/mods/PersonalMods - Head Mod/[Chw] WesternType2.zipmod' --zipmod 'E:/HoneySelect2_ArcticFox/mods/PersonalMods - Head Mod/Head - Exclusive HS2/[XeyNiu][HS2] c008 Tifa {FF7R}.zipmod' --zipmod 'E:/HoneySelect2_ArcticFox/mods/PersonalMods - Head Mod/RoomGirl Backport/hooh_rg_head05.zipmod' --out outputs/head_base_audit_20261007/prefab_contract_v1.json
```

Use a fresh output; the auditor refuses overwrite. It scans the installed vanilla
list bundles rather than trusting cached IDs, then reads each package's own CSV,
manifest and explicitly selected render prefab, excluding collision-prefab roots.
The report stores real MonoScript identities and component typetrees, mesh sizes,
bone palette/bindpose counts, blendshape names, material/shader identities, prefab
rest transforms and list-selected asset hashes. It neither patches files nor
queries the running game.

Report SHA256 `b9f29402c2d55e8ea3e9116b1cd8df2b663b9c23cf22cf080c11fa0efa8ddef9`.
Licensed component data and package contents stay in ignored outputs; only the
auditor, commands and provenance/interpretation are tracked.

Findings:

- All four audited real prefabs have CmpFace and IL.dll FaceBlendShape roots and
  eight skinned render parts. The primary head vertex counts differ substantially
  (vanilla2:4439; Western:4438; Tifa:36057; RG05:4927). The new topology does not
  have to match a vanilla face.
- All four list-selected ShapeAnime TextAssets have the same exact byte hash,
  although RG05 names its copy differently. Existing bone shape tables can be
  reused by a genuinely different skinned model; this does not auto-transfer
  weights or expression vertex deltas.
- Sideloader category210/211 plus GUID, faceSkinInfo and actual HeadID remapping
  provide the standard independent head/skin/card-reference route.
- Source FLAME five-joint buffers are not native game face bindings. Native skin
  palette/bindposes, source UV/materials and correctly placed render parts must
  be authored into the asset. This adaptation is not yet implemented.
- The user allows facial expressions to be deferred. A neutral first asset must
  explicitly disable/defer incompatible expression patterns; copied index arrays
  or zero deltas do not establish expression support.
- Neck adaptation must act on the currently retained local neck-edge vertices
  with smooth falloff, per user instruction. Do not restore deleted neck/base
  faces or distort the entire head in the name of fitting the rim.

Recommended construction: clone a native prefab's serialized component shell
in an isolated bundle, replace its **actual o_head mesh asset**, author native
bindings and appropriate UV/parts, supply independent head/skin directory rows
and register the resulting zipmod. Installed SB3UGS has script APIs for mesh,
morph and bundle save; only static signatures were inspected this turn, no
replacement or output has been demonstrated. A fresh Unity build is a fallback
when serialized-template editing is insufficient, not an assumed prerequisite.

This checkpoint establishes asset/loading design, not a working import package,
pose adaptation, likeness or new real base. No game calls, screenshots, model
parameter samples or DLL installs were made during this research.
