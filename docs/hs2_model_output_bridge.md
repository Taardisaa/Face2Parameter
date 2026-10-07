# Existing model output bridge

The first implemented stage preserves the actual SMIRK outputs and the FLAME
geometry they generate. It does not train a converter, fit HS2 sliders, or
reconstruct a replacement photograph. HS2 parameter conversion and native head
integration remain separate stages.

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
