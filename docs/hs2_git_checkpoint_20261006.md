# HS2 infrastructure source checkpoint — 2026-10-06

This checkpoint preserves the existing working-tree backlog: deformation sampling,
parameter surface atlases, geometry/pixel/quality diagnostics, stateful ABMX and
controlled runtime consumers, independent target readers, tests and documentation.
It is a source checkpoint, not a claim that the infrastructure or character goal is
complete. Historical experiments and proposals retain their individual scope.

The current user priority is explaining parameter changes as actual surface motion
and making that response useful for tuning. The native atlas is an offline cached
geometry measurement; full current-character guidance and broad runtime calibration
remain incomplete. Existing beauty-related optimizer changes and `proposal.md` are
preserved as found, rather than treated as the acceptance criterion for this work.

Checks performed for this sync:

- All 159 pending Python source files parsed successfully without executing them.
- The parameter-atlas analytical suite passed all 7 tests.
- The companion HS2Mod MCP contract suite passed all 82 tests.
- Staged whitespace checks accept the original Windows CRLF source bytes.

No game operations, training, fitting, asset downloads or dependency upgrades were
performed for this Git sync. Existing source bytes were not reformatted to make the
checkpoint; saved evidence can bind exact implementation bytes.

Generated `outputs/`, extracted `data/`, dataset samples and model assets retain
their ignore rules and are not uploaded. Many runtime validators intentionally
require these local assets and saved native evidence; a fresh checkout alone cannot
reproduce the live certificates. The companion repository is
`C:/Users/13666/Workspace/HS2Mod` (remote `Taardisaa/HS2MCP`); its
`docs/hs2_git_checkpoint_20261006.md` records the corresponding Face2Parameter commit.
