# Driving HS2's renderer / getting the face mesh as a function call

## Verified pipeline (reverse-engineered 2026-06-28 — all managed C# + data, no Ghidra)

How HS2 builds a female face from a card, traced via UnityPy + ILSpy decompile of `Assembly-CSharp.dll`:

1. **Head mesh** = `o_head` (3254 verts, skinned to 43 bones) in `abdata/chara/00/fo_head_00.unity3d`
   (`fo_head_NN` = female heads; `ao_head` = hats; `mo_head` = male). It carries **58 blendshapes that are
   ALL expressions** (`e*` eyes, `g*` brows, `k*` mouth/visemes) — **none are shape sliders**. So for a
   neutral static face, expression blendshapes = 0.
2. **`shapeValueFace` (54) is BONE-DRIVEN**, not blendshapes. `ChaControl.InitShapeFace` →
   `sibFace = new ShapeHeadInfoFemale()` (a `ShapeInfoBase`) → `sibFace.InitShapeInfo(... "list/customshape.unity3d", assetAnmShapeFace, ChaABDefine.ShapeHeadListAsset(sex) ...)` then
   `sibFace.ChangeValue(i, fileFace.shapeValueFace[i])` for each of the 54.
3. **Shape data = a keyframe table** (TextAsset). `ShapeInfoBase.ChangeValue(category, value)` interpolates
   per-category keyframes of **bone transforms** (scale / position / rotation — uses `Mathf.LerpAngle` for
   rotation) and writes them to the `cf_J_*` face bones. Body's table is `cf_anmShapeBody` (present in
   `list/customshape.unity3d`, alongside `cf_custombody`/`cf_customhead` TextAssets); the **face** table
   (`assetAnmShapeFace`) is passed in per-head (location TBD — find via the `InitShapeFace` caller / head list).
4. **ABMX** then applies *additional* `cf_J_` bone mods on top (math known: `scale*=mod`,
   `pos=base*len+offset`, `rot=base*Euler(mod)` — ABMX `BoneModifier.cs`).
5. **Final mesh** = `o_head` linear-blend-skinned by the resulting `cf_J_` bone poses. The `cf_J_*` skeleton
   rest poses live in `oo_base.unity3d` (female base body+skeleton; `o_body_cf` + 55 face bones).

**So the offline constructor = LBS(o_head, bones) where bone_pose = rest ∘ shapeValueFace-keyframe-interp ∘ ABMX; expressions=0.**
Everything is extractable (UnityPy) or decompilable (ILSpy) — no native RE. Tooling confirmed present:
`Assembly-CSharp.dll`, `dotnet` (+ `ilspycmd` 8.2 installed), Blender 4.5.3, SB3UGS, UnityPy.

## Goal

Ideally a pure function `params -> face mesh` (and optionally `-> rendered image`) so we can:
- compute facial-proportion ratios directly in **3D** (no camera/pose/projection — see
  [facial-harmony-metric.md](facial-harmony-metric.md)), and
- eventually close a render-in-the-loop optimization.

## Reality check

HS2 is a **Unity** game with no headless/CLI render API. All programmatic access goes through
**BepInEx** C# plugins running *inside* the game (this repo already ships the HS2/BepInEx/KKAPI/HS2API
assemblies under `src/face_data_utils/MsgToJson/`, and uses `HS2ABMX.exe` to (de)serialize ABMX bone
data — but that tool only touches *data*, it can't deform/render without the Unity runtime + asset
bundles). So "invoke the renderer" means one of three routes.

The face shape we already control = **`shapeValueFace` (54 sliders = head blendshape weights)** +
**ABMX bones** (the 30 in `META_DATA_LIST`, bone transforms). A mesh evaluator needs the base head mesh,
those blendshape deltas, and the face rig.

## Traced internals (from existing open-source mods — read these, don't RE the binaries)

The community already reverse-engineered the deform pipeline; we read their code.

- **Bone deform math — solved.** [ABMX `BoneModifier.cs`](https://github.com/ManlyMarco/ABMX/blob/master/Shared/Core/BoneModifier.cs)
  applies modifiers exactly how our `ABData` (scale[3], length, position[3], rotation[3]-euler) is shaped:
  ```
  localScale    = baselineScale * ScaleModifier                     # component-wise multiply
  localPosition = baselinePos   * LengthModifier + PositionModifier
  localRotation = baselineRot   * Quaternion.Euler(RotationModifier)
  ```
  So the 30-bone block of the 205-d vector is a fully known transform.
- **Deformed-mesh capture — one Unity call.** `SkinnedMeshRenderer.BakeMesh(mesh)` snapshots the fully
  deformed head (blendshapes from `shapeValueFace` **and** ABMX bones, both already applied by Unity) into
  a plain `Mesh`, vertices relative to the SMR transform. **In-game export therefore needs *zero* knowledge
  of the slider→blendshape mapping** — Unity does the deform; we bake + write OBJ. KKBP_Exporter and
  BlendshapeCreator already do exactly this.
- **Not yet traced (only needed for the fully-offline Route A):** the `shapeValueFace` (54) →
  head-blendshape-weight mapping (some sliders interpolate a min/max morph pair) and pulling the base mesh
  + blendshape deltas from the asset bundles. Both are encoded in KKBP (exporter + Blender importer) and the
  game's `ChaControl`/face-blendshape definitions — **tracing, not binary RE.**

## Route A — offline asset extraction + pure-Python mesh evaluator  (the "ideal" function; hardest)

Extract once with **AssetStudio / UABEA / uTinyRipper** from HS2's asset bundles: base head/face mesh,
its blendshapes, face bone hierarchy + bind poses. Then replicate the deform in Python:

```
mesh(params) = base_verts + Σ_i  weight_i · blendshape_delta_i      # shapeValueFace
            then linear-blend-skin by the ABMX bone transforms      # the 30 bones
```

- **Pros:** pure, offline, deterministic, batchable, no game running; gives 3D geometry directly.
- **Remaining work (mostly *tracing* KKBP, not RE from binaries):**
  - The bone transform math is already known (see Traced internals / ABMX `BoneModifier.cs`).
  - The one genuine unknown: `shapeValueFace` → blendshape-weight mapping (some sliders interpolate a
    min/max morph pair) — recover from KKBP / the game's face-blendshape definitions.
  - Replicate standard Unity blendshape (`base + Σ wᵢ·deltaᵢ`) + LBS skinning; extract base mesh +
    deltas with AssetStudio. The face is several meshes (head/eyes/brows); the head + its morphs suffice
    for proportions.

## Route B — in-game BepInEx plugin bridge  (pragmatic; game must run; least RE)

A small plugin (or adapt existing ones) loads a card, then exports the head mesh and/or a screenshot;
a Python side drops `card + request` into a watched folder and reads back `OBJ`/`PNG`. Effectively a
function call, but the game runs (can be minimized; not truly headless) with per-card load latency (~seconds).

The core is **trivial** thanks to `SkinnedMeshRenderer.BakeMesh()`: find the head SMR → `BakeMesh(m)` →
write OBJ. No blendshape-mapping knowledge needed (Unity already deformed it). A from-scratch plugin is a
few dozen lines; or reuse BlendshapeCreator's existing OBJ export and just add the card-load + file-bridge.

Reusable building blocks that already exist:
- **[BlendshapeCreator](https://github.com/ShalltyB/BlendshapeCreator)** (supports HS2) — select a mesh,
  **export `.OBJ`**; ships a Blender addon for import/export.
- **[BepisPlugins](https://github.com/IllusionMods/BepisPlugins) Screencap** — unified screenshot config/API.
- **[HS2-Sandbox](https://github.com/SuitIThub/HS2-Sandbox) CopyScript** — bridges Studio to an external
  service on the PC (the automation hook to turn manual export into a loop).
- KKAPI/HS2API charadata load API for applying a card in Maker/Studio.

## Route C — manual one-off export  (to prototype the metric NOW)

Use BlendshapeCreator to export a head `.OBJ` for one generated card + the reference, by hand. Enough to
build and validate the **3D-ratio harmony/fidelity metric** before investing in any automation.

## Recommendation (staged)

1. **Prototype with Route C**: export 2–3 head OBJs by hand → build the 3D facial-ratio metric → confirm
   it gives sensible fidelity/harmony numbers. Cheapest way to de-risk the whole idea.
2. **Automate with Route B**: wrap BlendshapeCreator-style OBJ export behind a file/HTTP bridge for a
   `params -> mesh` Python call. This is the realistic "function call" with the least reverse-engineering.
3. **Only if B's throughput/latency hurts**, invest in **Route A** (fully offline evaluator).

## Crude fallback (no mesh at all)

The ABMX bones are literal 3D transforms positioned at face features (chin, cheek, nose, mouth, eyes —
see `META_DATA_LIST`). Bone positions approximate some landmarks, but miss the `shapeValueFace`
blendshape contribution and aren't surface vertices — too crude for reliable proportions. Not recommended
beyond a sanity check.

## Tools

| Tool | Use |
|---|---|
| AssetStudio / UABEA / uTinyRipper | extract meshes + blendshapes from Unity asset bundles (Route A) |
| BlendshapeCreator (HS2) | in-game mesh → OBJ export (Routes B/C) |
| BepisPlugins Screencap | in-game screenshot/render API (Route B, if pixels wanted) |
| HS2-Sandbox CopyScript | Studio ↔ external-service automation bridge (Route B) |
| KKAPI / HS2API | load a card's params in Maker/Studio (Route B) |
