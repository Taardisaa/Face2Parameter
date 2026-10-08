# Native base 02: chin to anterior neck

The user asked to select an original game base and inspect its chin transition
before editing the target's anterior geometry. Selected female head ID 2,
`p_cf_head_02`, resolved through the installed ChaList and read directly from its
actual bundle. Its paired body is the installed vanilla `p_cf_body_00/o_body_cf`.

The diagnostic uses literal bind positions expressed in the coincident
FaceRoot_s and Head_s attachment frames. No slider sweep, fitted alignment,
runtime model replacement or game mutation occurred. The native reference is
**asset bind geometry**, not a claim about the current game's exact animated
state or the result of setting every face slider to 0.5.

The current target is the accepted posterior-only Chenger geometry from
`posterior_bezier_review_v1/chenger`. Its front was not edited. Its BP body uses
the existing captured source bind geometry in the same Head_s frame. The two
panels use their respective real body assets; no vertex-index substitution is
made between vanilla and BP bodies.

## Visible mechanism

The native chin's exterior bottom is a shallow curve that lies slightly above
the actual front head/body interface. It runs rearward into that interface.
The body's topmost opening continues further upward inside the head, drawn
dashed in the section; it is not the external join to target.

The current target has a deeper bottom contour below the interface, so its
under-chin profile descends and rises back toward the neck. This identifies a
relative placement/profile issue worth discussing before another local mesh
edit. It does not prove that the MICA identity is wrong or that the chin should
be flattened to match the native face. No front correction has been performed.

The interface audit confirms the native 28-point lower row against the actual
body internal row and root support. Full provenance and results are in
`outputs/native_head_20261007/native_chin_review_v2/receipt.json`; a metadata-only
copy is tracked as [the receipt](hs2_native_chin_receipt.json). The plotted full
head also contains the genuine internal mouth section; the two zoom panels
focus below it on the external under-chin surface.

## Reproduce

```powershell
.venv/Scripts/python.exe -m tools.native_head.native_chin_review --current outputs/native_head_20261007/posterior_bezier_review_v1/chenger/geometry.npz --body-capture outputs/native_head_20261007/neck_local_v1/native_body.json --output outputs/native_head_20261007/native_chin_review_v2
```

Use a new output path when rerunning. `native_chin_sections.png` shows the native
whole-head section and two under-chin zooms. `native_chin_3d.png` shows the actual
native mesh from side and front oblique views, with its lower face highlighted.
Assets/mesh coordinates remain local and ignored. This is a read-only reference
milestone; front shaping and native asset integration remain unfinished.
