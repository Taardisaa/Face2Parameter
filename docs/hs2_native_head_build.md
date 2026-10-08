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
the old geometry. The initial source-eyeball UV transfer failed visual acceptance. The updated
`--native-eyes` option uses actual native eye hemispheres and their unchanged
UV/pupil domain, uniformly placed at the source eyeball centers. This authors
eye geometry; it does not claim the source full eyeball vertices are unchanged.
The corrected neck construction hard-locks the authored anatomical face/ears
and the anterior surface. All retained vertices preserve their original height.
Eye gaze pivot retargeting remains unverified.

`--plain-skin` authors uniform native skin inputs in a private skin bundle,
using the installed albedo median, open occlusion and neutral DXT5nm inputs.
The original cosmetic alpha/detail settings are cleared in the neutral card.
This corrects incompatible vanilla texture markings without adding invented
identity detail. It is a plain authoring skin, not recovered identity albedo;
full makeup/UV transfer remains future work.

## Posterior contraction and integrated under-jaw join

The optional `rim.py` step uses the saved actual native body's bind vertices and
`cf_J_Head_s` bindpose. Its boundary must have literal full support on that bone.
Every native contour corner survives; existing contour edges are subdivided to
the retained source-ring count. Pairing uses XZ azimuth to preserve contour order.
The previous entire-ring XYZ snap was wrong: the cut includes chin vertices and
therefore visibly pulled the lower face back. That output remains in `neck_v3`.

The corrected implementation loads the source's SHA-bound authored FLAME masks
and hard-locks face, ears, eye region, forehead, lips and nose. It additionally
locks the surface anterior to the body's front rim. Existing posterior vertices
contract only in width/depth, with a local harmonic/geodesic fade that cannot
propagate through locked vertices. **Every retained vertex keeps its height
(Unity Y), including the posterior vertices.** Build guards enforce these locks.

Since the existing cut crosses the chin, its front edge cannot be moved onto the
native neck without deforming the face. Four new shared-vertex transition rows
inside the actual `o_head` mesh connect the retained edge to the native body rim.
Hermite endpoint tangents use the adjacent surface planes. This is newly authored
under-jaw connection geometry, not restoration of deleted source neck faces or
an additional shell/renderer. Original retained triangles, protected positions
and protected surface normals remain intact. No body vertices are changed.

The posterior band blends toward the public `cf_J_FaceRoot` binding, before
FaceRoot_s's face-only scale. New transition vertices use that root directly;
original native rim normals set the inner endpoint normals. Tangents are
orthogonalized against the final authored normals. This is
an authored local geometry adjustment authorized by the user, not a reconstruction
of native game behavior. Full neck/body variants and unrestricted shape/ABMX
combinations are not certified by this implementation.

## Reproduce

From this repository, with the existing audited/trimmed target artifacts:

```powershell
.venv/Scripts/python.exe -m tools.native_head.build --source outputs/model_bridge_20261007/chenger_neck_native_v1/source_trimmed.json --audit outputs/head_base_audit_20261007/prefab_contract_v1.json --scale 9.851049 --translation 0.00339057157 0.458031476 0.603983462 --native-neck outputs/model_bridge_20261007/chenger_neck_native_v1/native_after_trim.json --neck-descriptor outputs/model_bridge_20261007/chenger_neck_native_v1/upper_neck_attachment.json --native-eyes --plain-skin --out outputs/native_head_20261007/neck_v4b
.venv/Scripts/python.exe -m tools.native_head.card --card E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/程儿_MICA原生颈_03113.png --out outputs/native_head_20261007/程儿_原生底模中性.png
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

Asset construction commits: Face2Parameter `b68857d`; HS2Mod `bbbf5b4`.
The follow-up card writer removes the old overlay key entirely instead of
writing an unknown/empty plugin version rejected by bridge card preflight.

## Corrected asset acceptance, 2026-10-07

Installed `neck_v4b/Chenger.MICA.NativeHead.zipmod` (manifest0.1.1), restarted
the game and normally loaded the same GUID-based native card. No bridge DLL
change or Computer Use was needed. One final native three-angle capture and
head/body geometry export confirm the actual `o_head` matches serialized output,
the original protected face/chin/ear positions are literal, all retained heights
are literal, and the native body's source geometry hash is unchanged. The new
inner contour coincides with body rim edges in the same frame; geometry evidence
is in ignored `neck_v4b/acceptance.json` and `head_body_final.json`.

The visible chin-pulling from the previous contraction is corrected. The head
and neck still have different material appearance; geometric connection does
not imply an invisible material seam. The plain skin/black eye appearance,
expressions, unrestricted sliders/ABMX and photo likeness remain incomplete.
The evidence explicitly records the visible material seam and does not certify
target identity. Native screenshot: HS2Mod
`artifacts/chenger/native_base_v4_sheet.png`.

Regression check: `.venv/Scripts/python.exe -m unittest tools.native_head.test_rim`.
It protects face/front and height locks, preserves retained topology, and checks
that the integrated strip shares the existing boundary with opposing winding.
Build guards additionally bind anatomical masks by SHA, check literal positions,
reject degenerate/nonmanifold additions, and independently reopen mesh assets.
