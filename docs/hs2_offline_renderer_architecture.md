# HS2 offline renderer — reimplementation architecture (2026-10-20)

User decision 2026-10-20: the offline renderer **reimplements** the game's
render computation (Unity 2018.4.11f1 built-in forward pipeline + installed
plugins), driven by scene parameters, so camera, lights, render settings,
materials and geometry can all change. RenderDoc replay of real captures is the
**oracle** for pass-by-pass acceptance, not the renderer. Fixed-capture byte
patching ([pipeline](hs2_offline_replay_pipeline.md)) remains a diagnostic tool.

Evidence for the target frame structure: [frame graph](hs2_offline_frame_graph.md),
[head skinning](hs2_offline_head_skinning.md). The repo/workspace rules in
`AGENTS.md` apply: source-first, no fitted substitutes, explicit gaps.

## Layers

```
scene state (exported from game, JSON + raw resources)
      │
      ▼
generators (Python, src/render/hs2pipe/)       ← reimplemented Unity + plugin logic
  unity_forward  dhh_sss  ppv2  ctaa  liquid    ← each emits commands via builder
  shader_library (installed DXBC + variant selection)
  skinning (CapturedHeadForward / CPU skin, already source-derived)
      │  frame program
      ▼
executor (C++ D3D11, tools/offline_renderer/frame_executor.cpp)
      │  readbacks
      ▼
acceptance (per-pass diff vs official RenderDoc replay of a paired capture)
```

## Frame program (executor input) — contract

A directory with `program.json` and a `blobs/` folder (content-addressed by
SHA-256). All bytes are exact; nothing is decoded or converted implicitly.

- `resources[]`: `{id, kind: buffer|tex1d|tex2d|tex3d|texcube, desc{...D3D11 desc
  fields verbatim: ByteWidth/Width/Height/Depth/MipLevels/ArraySize/Format(DXGI int)/
  SampleDesc/Usage/BindFlags/CPUAccessFlags/MiscFlags/StructureByteStride},
  initial: [{subresource, blob, row_pitch, slice_pitch}] | null, name}`
- `views[]`: `{id, kind: srv|rtv|dsv|uav, resource, desc{verbatim D3D11 view desc}}`
- `shaders[]`: `{id, stage: vs|ps|gs|hs|ds|cs, blob, source{shader, subshader,
  pass, keywords[], asset, path_id}}` — DXBC bytes from installed assets.
- `input_layouts[]`, `blend_states[]`, `depth_states[]`, `raster_states[]`,
  `samplers[]`: verbatim D3D11 descriptors.
- `commands[]` ordered; ops: `clear_rtv`, `clear_dsv`, `clear_uav_float|uint`,
  `update_subresource` (blob + box), `copy_resource`, `copy_subresource_region`,
  `resolve`, `generate_mips`, `draw`, `draw_indexed`, `draw_instanced`,
  `draw_indexed_instanced`, `dispatch`. Each draw/dispatch carries its **complete
  binding snapshot** (shaders, IA buffers/offsets/strides/topology/layout, CBs per
  stage with blob or buffer+range, SRVs/samplers/UAVs per stage, RTVs/DSV,
  viewports/scissors, blend factor/sample mask, stencil ref, all states). No
  implicit inherited state.
- `readbacks[]`: `{name, after_command, resource, subresource, orientation}`.
- `provenance`: generator versions, scene-state hash, capture hash if converted.

`src/render/frame_program.py` is the Python builder/validator for this schema.

## Acceptance ladder

1. **Executor**: a frame program converted mechanically from a real capture
   reproduces the official replay bit-exactly at every chosen checkpoint (same GPU).
2. **Generators**: from exported scene state of a *paired* capture, generated
   commands/cbuffer bytes/resources equal the captured ones (structural + byte
   diff per pass); then executed outputs equal replay per pass and at the final
   target. Residuals are traced to the responsible source path, never fitted.
3. **Parameter changes**: a few final native captures with changed camera /
   lights / render settings, accepted the same way.

Scope is the whole rendered scene (body, hair, clothes, map, UI exclusion rules),
not only the head; the head surfaces are the first priority inside it.
