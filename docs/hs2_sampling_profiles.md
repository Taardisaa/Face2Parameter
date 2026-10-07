# Native animation sampling profiles

Native deformation now has an explicit runtime profile. This does not change ML
label normalization, trained models, card inference, or their default domain.

```python
from src.hs2_mesh_deform import HeadRig, build_mesh
from src.hs2_deform_torch import TorchHeadRig

rig = HeadRig(1, sampling_profile="slider_unlocker_18_2")
trig = TorchHeadRig(rig, device="cuda")
numpy_vertices, faces = build_mesh(rig, native59, ab_data=None)
torch_vertices = trig(native59_tensor)
```

`HeadRig` defaults to `vanilla`. `TorchHeadRig` inherits the input rig's profile
unless explicitly overridden. Select matching profiles when comparing them.
The same sampled source values/FK drive the head and all submeshes, including eyes
and lashes. Unknown profiles and nonfinite inputs/keyframes are rejected.

## Contracts

| Case | `vanilla` | `slider_unlocker_18_2` |
| --- | --- | --- |
| Within `0–1` | Existing interpolation unchanged | Same interpolation |
| Outside, position/scale | Endpoint clamp | Global first-to-last extrapolation |
| Outside, ordinary Euler rotation | Endpoint clamp | Global direction-based extrapolation |
| Outside, exempt rotation | Endpoint clamp | Original clamped rotation |

The unlocked implementation follows the installed SliderUnlocker 18.2 decompilation
at `C:/Users/13666/Workspace/HS2Mod/tools/SliderUnlocker.decompiled.txt`, specifically
the GetInfo prefix/postfix and SliderMath methods. The plugin clamps for the
original method, then replaces selected values only outside the domain.
Removing the clamp or extending the nearest segment would be incorrect.

Position/scale use `first + (last-first)*rate`, including tables with many keys.
For rotation, the sign of raw `second-first` selects the unwrap direction of
`last-first`, adjusting by one turn when signs disagree. Zero counts as positive.
Do not apply a shortest-arc rule or normalize authored angles before this step.
Below zero, use `first + delta*rate`; above one, `last + delta*(rate-1)`.

These case-sensitive predicates exempt rotation while still extending position/scale:

- Contains `cf_s_Mune`, `cf_s_Mouth`, `cf_s_LegLow`, or `cf_s_MayuTip`.
- Contains both `thigh` and `01`.
- Starts with `cf_a_bust` and ends with `_size`.

One-key vanilla tables return a constant pose. Unlocked one-key position/scale and
exempt/unused rotation are also constant. Ordinary requested rotation outside
the domain raises an explicit error: the installed plugin accesses the second
keyframe and cannot safely evaluate this case. The offline profile does not
invent a runtime result or gradient.

## Compatibility and limits

The scalar default is checked exactly against its frozen predecessor; default
Torch source arithmetic is preserved. A preexisting endpoint Euler representation
difference remains: NumPy returns the raw final angle, whereas Torch can return
an equivalent last-segment angle differing by 360 degrees. Direct quaternion
conversion makes them equivalent, but affine Update equations may combine angles,
so this remains a parity caveat. The new unlocked outside-range branch uses raw
endpoints, including exemption cases. Within-range behavior stays unchanged.

This change addresses native sampling, not full ABMX parity. Existing offline
ABMX multiplies local position by length, while installed ABMX can normalize the
current direction and use a cached baseline length, along with runtime conditions
and rotation exclusions. Those formulas are not universally equivalent. Use no
ABMX for native parity checks and assess its residual separately. Neutral offline
geometry also does not apply expression blendshapes.

## Validation

Always use the project interpreter:

```powershell
.venv/Scripts/python.exe tools/range_validation/test_sampling_profiles.py -v
.venv/Scripts/python.exe tools/range_validation/validate_profiles.py --out outputs/range_validation --device cpu
.venv/Scripts/python.exe -m src.hs2_deform_torch --card tests/HS2ChaF_20240901192905747.png
```

The 7 contract tests cover exact vanilla scalar compatibility, three-keyframe
global extrapolation, both external sides, rotation unwrap, all exception predicates,
batched NumPy/Torch agreement, autograd gradcheck, finite guards and one-key cases.

The report script never contacts the game. By default it reads every cached head,
uses native `.5` as baseline, and tests sliders `0,4,24,27,47,53,54` at
`-.25,.23,.77,1.25` with both profiles. It compares the head and all cached submeshes,
checks outside-range head gradients, and records cache SHA-256 hashes, individual
errors, assumptions and compressed NPZ mesh fixtures in `validation.json`.

- Repeat `--head-id` to select cached bases.
- Supply `--shape-file native59.json` to use the full live native vector as baseline.
- Select controls/rates with `--sliders 0,4,53 --rates=-.25,.23,.77,1.25`.
- Use `--device cuda` for GPU evaluation; the NumPy reference remains float64.

2026-10-04 offline results for heads `0,1,2`: 174 cases, 1,392 mesh comparisons,
maximum NumPy/Torch vertex error `4.44e-16`; all 48 gradient probes pass. All 24
unlocked outside-range probes have nonzero gradients, and vanilla outside gradients
stay zero. The legacy card self-check retains `2.22e-16` maximum error with all
10 gradient probes passing. This proves numerical agreement, not Unity parity
or likeness to a photograph.

## Unity export comparison gate

1. Cache the exact live head ID via the card-driven extractor; match mesh, animation,
   skeleton and plugin versions. Bundle filenames do not determine head IDs.
2. Record all native values, including ears, and pass them as `--shape-file`.
   Establish a neutral expression and no effective ABMX for native comparisons.
3. Export baseline and controlled single-slider head/eye/lash geometry from Unity,
   including root/renderer transforms, runtime native values and modifiers.
4. Match topology and vertex order; transform both results into the same coordinate
   system using recorded transforms. Report absolute and mesh-scale-normalized
   errors. Independent shape fitting could conceal wrong scale or pose.
5. Preserve/restore the original character state and inspect fixed-camera multiview
   captures. Record native, ABMX, expression and rendering residuals separately.

Only after these parity checks should residuals be used to assess base expressivity.
