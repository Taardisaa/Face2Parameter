# Chenger native head asset construction

This implements the asset route in [the head import contract](hs2_head_asset_import.md).
The output is a standard Sideloader zipmod with a real `CmpFace` prefab, not an
additional head renderer attached by McpBridge. Model/game assets stay ignored
and local; the scripts require the installed game and existing raw target export.

## Construction

`tools/native_head/build.py` clones the audited installed vanilla head2 bundle,
writes source-connected head and eyeball components into the actual `o_head`
and `o_eyebase_L/R` mesh assets, and preserves the prefab's serialized native
component references. It registers category210/211 and `faceSkinInfo` with GUID
`codex.chenger.mica.nativehead`. Private CAB/resource names prevent collision
with the vanilla bundle. The prefab's container entry is renamed as well.

UnityPy 1.25.0 writes explicit Unity2018 vertex channels, indices, bone weights,
bindposes, normals and tangents. Reopening the serialized bundle checks literal
vertex/triangle/skin arrays. This avoids requiring a newly installed editor or
using SB3UGS GUI. The previously audited SB3UGS route remains an alternative.

The existing recovered `hs2_mesh_deform` implementation supplies actual native
bone matrices at all59 face values0.5. Cached mesh positions/topology must equal
the freshly read installed template. New bindposes invert those native matrices.
Closest native triangles and barycentric interpolation author skin weights and
UV/color attributes; this **does not alter source face geometry or infer game
deformation logic**. Four skin influences are normalized after selection. This
is asset attribute authoring, not a claim of exact FLAME-to-HS2 expression mapping.
The source's photo parameters and parent export remain intact.

Expressions are deferred: the new topology has no copied/empty58-channel claim.
Native `FaceBlendShape` remains structurally present with disabled updates and
empty target arrays. Installed `FBSBase.Init` and `CalculateBlendShape` support
empty targets. Unsupported lashes, eye membranes, tears and oral geometry have
zero indices in the asset; a later native renderer-enable write cannot reveal
the old geometry. Eyes are source eyeballs using transferred native eye UVs;
appearance is provisional and is not inferred albedo.

## Existing rim adaptation

The optional `rim.py` step uses the saved actual native body's bind vertices and
`cf_J_Head_s` bindpose. Its boundary must have literal full support on that bone.
Every native contour corner survives; existing contour edges are subdivided to
the retained source-ring count. The **existing source-ring vertices** move onto
this contour. Harmonic displacement and smooth geodesic falloff restrict edits
to the specified band. No source faces are restored/added and no body vertices
are changed. Outside the band, original placed positions remain unchanged.

The band blends toward the public `cf_J_FaceRoot` binding, before FaceRoot_s's
face-only scale. Original native rim normals set its endpoint normals. This is
an authored local geometry adjustment authorized by the user, not a reconstruction
of native game behavior. Full neck/body variants and unrestricted shape/ABMX
combinations are not certified by this implementation.

## Reproduce

From this repository, with the existing audited/trimmed target artifacts:

```powershell
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/model_bridge_20261007/chenger_neck_native_v1/source_trimmed.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --native-neck outputs/model_bridge_20261007/chenger_neck_native_v1/native_after_trim.json --neck-descriptor outputs/model_bridge_20261007/chenger_neck_native_v1/upper_neck_attachment.json --out outputs/native_head_20261007/neck_v2
.venv/Scripts/python.exe -m tools.native_head.card --card E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/程儿_MICA原生颈_03113.png --out outputs/native_head_20261007/程儿_原生底模初始.png
```

Fresh output directories/cards are required. `card.py` uses the installed
Sideloader `ResolveInfo`/`CategoryProperty` contract: `ChaFileFace.headId` and
`ChaFileFace.skinId` with this GUID and original slot1. The normal loader chooses
actual runtime LocalSlots; no fixed resolved ID is guessed. It clears only the
old SourceHead record. Other plugin keys retain their original MessagePack wire
bytes through the existing splice writer. All59 native face values are0.5 in
this first asset's card.

Generated `receipt.json`, mesh NPZs and `neck_design.json` record build inputs,
local edits and asset serialization checks. Game loading and final visual
acceptance are separate from those build checks; they must not be inferred
from a successful offline bundle write.
