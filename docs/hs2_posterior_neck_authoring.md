# Posterior neck contour authoring — neutral geometry review

The user authorized replacing the lower posterior skull/neck profile with a
longer smooth curve, then asked to lower that transition to preserve more of the
original skull roundness. This milestone implements that posterior-only edit.
It does not install a native head, complete the front/chin join, or certify skin,
expression or HS2 slider behavior.

## Surface construction

Input is the existing audited integrated mesh in `interface_authoring_v3`, whose
native lower ring already matches the actual BP body interface. Keep that ring,
all anterior vertices, face/ear/eye regions and the upper skull literal. Do not
translate or stretch the whole head.

Use the original ear masks as anatomical markers. The transition threshold is
upper ear height minus one tenth of ear height. On this coarse mesh the actual
upper anchor is the first fixed posterior edge above that threshold, not an
invented vertex at the threshold. A harmonic coordinate along the connected
surface runs from the literal native ring to the fixed upper region.

A cubic Bezier joins the posterior midline endpoints. Its lower handle follows
the actual body-interface normal's upward tangent in the common Head_s frame.
Its upper handle follows the retained contour and matches its longitudinal
derivative. The curve guides a full posterior patch, with quintic lateral fade
and explicit facial/ear locks. Biharmonic **position** fairing lays out the
remaining surface; the old short connector's posterior interior rows are then
re-laid between their new endpoints. Front connector rows remain unchanged.

This is deliberate mesh authoring, not an inferred replacement for HS2 skinning.
The ear fraction and Bezier construction are declared design choices. There are
no person-specific vertex corrections or fitted game-response gains.

The input layout is explicitly checked: original FLAME crop recipes, matching
connector endpoint rows, six connector strips, and an audited 28-point native
interface. Other layouts fail rather than silently using an incorrect mapping.
This tool currently consumes an existing integrated baseline; it is not a
complete photo-to-installable-head command.

## Finite acceptance and limits

The same implementation, masks, body capture and placement were applied to
Chenger's selected img-002 MICA output and the existing AF1 `400016` identity.
Both passed literal interface/front/outside-patch preservation, unchanged
topology, nondegenerate triangles, and no new proper triangle intersections or
coplanar area overlaps. No intersections touch either newly edited posterior
region. Existing source-region intersections remain, recorded separately; this
does **not** certify the whole original head as intersection-free.

Early attempts are retained under `posterior_bezier_v1` through `v4`: a wrong
upper endpoint caused a kink, and directly transporting old collar rows left
crossed triangles. The reviewed final surface is
`outputs/native_head_20261007/posterior_bezier_review_v1/`.
Its actual vertices/triangles generate both section and shaded three-dimensional
views. Orange marks edited faces; textures and game shading are not simulated.
The body is only clipped for the diagnostic view, never modified in the asset.

Current native asset integration remains unfinished. The earlier `build.py`,
`author_neck_surface.py`, `neck_surface.py` and other head/attachment drafts are
not included in this milestone. No game state was changed.

## Reproduce from the existing audited local baseline

Run from the Face2Parameter root; use fresh output directories:

```powershell
.venv/Scripts/python.exe -m tools.native_head.posterior_neck --baseline outputs/native_head_20261007/interface_authoring_v3/chenger --body outputs/native_head_20261007/neck_local_v1/native_body.json --output outputs/native_head_20261007/posterior_bezier_review_v1/chenger
.venv/Scripts/python.exe -m tools.native_head.validate_posterior --baseline outputs/native_head_20261007/interface_authoring_v3/chenger --candidate outputs/native_head_20261007/posterior_bezier_review_v1/chenger
.venv/Scripts/python.exe -m tools.native_head.posterior_review --geometry outputs/native_head_20261007/posterior_bezier_review_v1/chenger/geometry.npz --body outputs/native_head_20261007/neck_local_v1/native_body.json --output outputs/native_head_20261007/posterior_bezier_review_v1/chenger
```

For the second identity, substitute `af1_400016` for `chenger` in the baseline
and output paths. Required licensed/model arrays remain ignored; input/output
digests and finite acceptance summaries are in
[the metadata receipt](hs2_posterior_neck_receipt.json).

The global neck goal remains active. Front/chin shaping, the completed native
asset build, native normals/skin integration and game acceptance are outstanding.
