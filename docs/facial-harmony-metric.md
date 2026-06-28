# Facial harmony & reconstruction-fidelity diagnostic

## Why this exists

Generated cards sometimes look "off / not pretty," and it's tempting to build a "beauty score" and
optimize toward it. We tested that assumption empirically and it does **not** hold:

- The generated card for `yua_ariga_06` is **on-manifold and typical**: Mahalanobis at the 48th
  percentile of 134k training labels; GMM log-likelihood at the 41st percentile. Only 2/205 dims are
  marginal outliers, and both are degenerate near-constant dims. **The parameter combination is not weird.**
- A hand-tuned "prettier" card of the same person is **less** typical: GMM log-likelihood at the **11th**
  percentile, and **farther** from the population mean (0.592 vs 0.488). Of the 40 params changed most,
  22 moved *away* from the mean.

**Conclusion: "prettier" is a distinct, off-center *ideal* direction — not typicality/averageness.**
This matches the literature: statistical typicality correlates with *likability/normality*
([PNAS, *From likely to likable*](https://www.pnas.org/doi/10.1073/pnas.1912343117)), but peak
attractiveness is a **separate dimension** from averageness
([*Typicality & Attractiveness reflect an ideal dimension*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9899519/);
[Nature 2025](https://www.nature.com/articles/s41598-025-86974-0)).

### What this means for design

A harmony-as-typicality metric measures **"coordinated / normal,"** which is the **right tool for a
DIAGNOSTIC** but the **wrong target for beautification** — optimizing toward it pulls toward the average
(blander), the opposite of what manual beautification does. So we build an *interpretable diagnostic*,
not an auto-beautifier. (Avoid black-box "FBP" beauty regressors trained on rated photos like
[SCUT-FBP5500](https://arxiv.org/abs/1801.06345): not interpretable, dataset-biased, and they conflate
makeup/lighting/pose with shape.)

## What to build: an interpretable facial-ratio report (two references)

Same machinery, two comparisons:

1. **Reconstruction fidelity** — `ratios(generated render)` vs `ratios(input photo)`. Answers the real
   question ("do the params match this person?") and localizes drift: e.g. *"eye spacing +0.4σ wider than
   the photo, midface too long, jaw too wide."* Best-defined (no external dataset needed), and because
   both faces go through the **same** detector, systematic detector bias cancels. **Highest-value output.**
2. **Harmony / typicality** — `ratios(face)` vs a **reference population** distribution (per-ratio z +
   Mahalanobis percentile). Flags genuinely unusual *combinations* (e.g. this nose-width + face-width pair).

Both emit an interpretable report ("midface: slightly long; eye spacing: wider than reference; overall
harmony: 30th pctile"), never a fake "8.2/10."

## Pipeline (lightweight, CPU, no large models)

1. Landmarks: MediaPipe Face Mesh (468 pts) or dlib-68, on the input photo and the render.
2. Procrustes-align; normalize scale by interocular distance (or face width).
3. Extract ~25–30 anthropometric ratios: face height/width, interocular/face width, nose width/face
   width, mouth width/face width, upper/mid/lower-face thirds, jaw/cheekbone width, chin/lower-face,
   eye width/inter-eye, nose length/midface, etc.
4. Score: per-ratio z vs photo (fidelity) and per-ratio z + Mahalanobis percentile vs population (harmony).

## Caveats / honest gaps

- **Don't use fixed ideals.** Neoclassical canons / golden ratio are *not* valid across population, sex,
  age ([PMC4369102](https://pmc.ncbi.nlm.nih.gov/articles/PMC4369102/),
  [PMC9452610](https://pmc.ncbi.nlm.nih.gov/articles/PMC9452610/)). Use a *statistical reference
  distribution*, not a target ratio.
- **Landmarks on the render**: HS2 output is realistic 3D, so detectors should work (unlike anime).
  Fidelity (render-vs-photo through one detector) is robust to render style; absolute harmony less so.
- **Expression skews ratios** (smile → mouth width, cheeks, eye openness, jaw). Restrict to
  expression-robust bone-structure ratios, or neutralize first. (See [expression-invariance.md](expression-invariance.md).)
- **Diagnostic, not auto-fix.** The report says *what* is off; turning that into parameter changes is
  still manual (nudge the flagged sliders) or needs a render-in-the-loop optimizer.

## Better than 2D landmarks: measure on the 3D mesh

The cleanest version skips rendering and 2D landmarks entirely: if we can obtain the **3D face mesh**
implied by a parameter vector, facial proportions can be measured **directly in 3D** from known vertex
indices — no camera projection, no pose, and the mesh is already in a neutral metric space. This needs a
`params -> face mesh` evaluator; see [hs2-renderer-and-mesh.md](hs2-renderer-and-mesh.md).

## Implemented (v1) — `src/face_metrics/` + `scripts/face_report.py`

Built on the offline mesh constructor. **The rig's named bones are the anthropometric landmarks** (eye
corners `cf_J_Eye01/03_s`, nose tip `cf_J_Nose_tip`, mouth `cf_J_Mouth_L/R`, chin `cf_J_ChinTip_s`,
cheek/face-width `cf_J_CheekUp`/`cf_J_EarBase_s`, brow `cf_J_Mayu`) — a landmark is a bone's world
position after deform stages 1-4 (`hs2_mesh_deform.bone_world`, no skinning). So **no vertex
hand-labeling**, and the character side is always expression-neutral.

- `landmarks.py` — 27 canonical points, mesh + dlib-68 backends; `ratios.py` — 25 scale-invariant ratios
  (subsets `HARMONY` / `FULL` / `STRUCTURAL`); `reference.py` — population stats over N=5000 sampled
  `labels.json` cards → `data/hs2_head/ratio_reference.npz`; `report.py` — harmony / compare / fidelity.
- **Harmony** `--card`: per-ratio robust z + Mahalanobis percentile. On `yua_desmile08_out` (the predicted
  card) → **82nd pctile** (somewhat unusual *combination*; flags morpho-height, eye level, mouth width),
  vs the manual `HS2ChaF` card at the **19th** pctile. Note the predicted card's raw params were *typical*
  (48th pctile in param space) yet its **geometry** is less typical — the "each feature fits but the whole
  looks off" signal, made measurable.
- **Compare** `--card-a/--card-b`: the predicted vs manual card differ most at `eye_vert_level` (−3.1σ) and
  brow (−2.6σ) — consistent with the per-vertex region diff (eye 0.060 ≫ mouth 0.020).
- **Reference build flags must match `gen_labels.py`**: `set_from_vector(v, is_simplify=True,
  without_right=True, denormalize=True, use_gaussian=False)`. Note `set_from_vector` writes `fd.base_data`
  (not `card_data.Custom`), so `hs2_mesh.fd_to_inputs` reads `base_data`.

### Ratio-knob editor — `src/face_metrics/edit.py` + `scripts/face_edit.py`
Turns the metric from a *report* into a *knob*: name ratio targets, solve for the smallest param change that
hits them. Forward model `p → ratios` is exe-free, so we finite-difference a Jacobian and take damped,
regularized Gauss-Newton steps over the normalized 205-vec (clamped to [0,1]; `without_right` keeps edits
symmetric). Holds non-targeted ratios (`hold_weight=0.05`) and min-norms the param change (`λ=1e-3`).

    face_edit.py --card C --thinner --eyes-closer --out O --render   # presets
    face_edit.py --card C --set "icd/facewidth=-1.5sigma" --set "face_height/width=+8%"

On `yua_desmile08_out`, `--thinner --eyes-closer` hit all three targets to <0.05σ and the render shows a
visibly slimmer face with closer eyes. **Honest reporting is the point:** it surfaces *coupled collateral*
(moving `icd/facewidth` drags the shared-ICD/face-width ratios ~1σ — fundamental, not hidable) and the
*plausibility cost* (harmony percentile 82→95 at 1.5σ strength — dial `--strength` down for subtler edits).
Presets map to ratio targets; since ratios are face-width-normalized, "thinner" = height/width UP (proportion,
not absolute size). `--sliders-only` restricts edits to the 54 `shapeValueFace` sliders (no ABMX).

### Phase-2 fidelity (working) — and why `--render-backend` matters
`fidelity(card, photo)` reduces both sides to the same ratio subvector and reports per-ratio signed Δσ/Δ%.
Photo landmarks: bulat **`face-alignment`** (68-pt, torch; `pip install face-alignment`). The old repo file
that shadowed the import name was renamed `face_alignment.py → mtcnn_align.py`. On this box torch-inductor
has no MSVC `cl`, so `_get_fa` sets `torch._dynamo.config.suppress_errors=True` to fall back to eager.

Two char backends, and the gap between them **is the diagnostic**:
- **mesh-bones** (default): char landmarks = rig bone positions. Fast, but bones ≠ the skin-surface points
  the detector marks on the photo → a per-ratio definitional offset.
- **`--render-backend`**: run the *same* 68-detector on a render of the card, so that offset cancels.

On `yua_desmile08_out` vs the `yua_ariga_06` photo (STRUCTURAL), switching bone→render collapsed the scary
artifacts (`noselength/facewidth` −5.1σ→−0.9σ; `noselength/midface` −5.0σ→0.0σ; `nosewidth/icd`
−2.9σ→+0.5σ) while the **real** gaps persisted: **`face_height/width` ≈ −2.3σ in both backends** (generated
face shorter/wider than the person), plus brow-height and eye vertical level. **Trust the render-backend
deltas; read the bone-vs-render difference as the landmark-definition bias.** (Caveat: the photo may carry
mild smile/pose; STRUCTURAL drops expression ratios but vertical ratios still feel pitch.)
