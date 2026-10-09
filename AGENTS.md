# HS2 implementation work

This repository supports the HS2 head/parameter tooling in
`C:\Users\13666\Workspace\HS2Mod`. Follow that repository's
[source-first and Git synchronization rules](../HS2Mod/AGENTS.md).

- The user's prohibition on shortcuts and simplification applies to ALL work,
  not only the current goal or a particular subsystem. Do not omit required steps
  or substitute constants, placeholders, disabled features, approximations,
  reduced scope, or weakened acceptance to make the work easier. Diagnostic
  substitutions must remain identified, isolated, and recoverable; they must not
  become delivered assets or support claims of compatibility/completion. Preserve
  the full requested end state and implement unresolved paths without silently
  replacing it with a smaller outcome. Follow the full fidelity rule in HS2Mod.
- The installed game and plugins are the implementation oracle. Recover their
  computation through decompilation, IL inspection, shader disassembly, asset
  extraction, and call-chain analysis. Use runtime inspection only for concrete
  state or a specific source gap; do not brute-force screenshot comparisons or
  parameter sweeps to discover recoverable logic.
- The user dislikes repeated reminders to "take screenshots and compare/test."
  Do not center plans or progress reports on that, or make it a prerequisite for
  implementation. Directly implement and integrate already recovered logic;
  small final acceptance checks are secondary, not an exploratory campaign.
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
