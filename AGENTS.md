# HS2 implementation work

This repository supports the HS2 head/parameter tooling in
`C:\Users\13666\Workspace\HS2Mod`. Follow that repository's
[source-first and Git synchronization rules](../HS2Mod/AGENTS.md).

- Analyze the current game's and existing plugins' code and actual assets first.
  Trace the implemented call chain; do not guess deformation behavior.
- Match the game's implementation, including interpolation, bone mappings,
  transforms, plugin hooks, cached baselines, update order, and special cases.
  Do not substitute empirical slopes, fitted gains, or approximate deformation
  models for available implementation logic. Clearly mark unresolved branches.
- Reduce numerical testing. Collect new game samples only after implementation
  is complete, as a small set of final acceptance cases. During implementation,
  use source inspection, static comparisons, existing evidence, and necessary
  logic checks; do not repeatedly sweep or sample to infer known logic.
- If final acceptance differs, fix the source-derived implementation before
  repeating affected cases. Retain failures and the original thresholds.
- Report mechanisms and visible effects; keep detailed numbers in evidence files.
  Prefer native screenshot/geometry capture and avoid Computer Use.
- Inspect Git status before edits, stage explicit task paths, preserve unrelated
  changes, and promptly commit/push completed milestones in both changed repos.
  Keep generated outputs, game assets, decompiled code, and model data ignored.
