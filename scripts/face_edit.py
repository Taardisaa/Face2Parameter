"""Ratio-knob face editor CLI — nudge a card's params to hit named facial-ratio targets.

    # the user's ask: thinner face, eyes closer together
    PYTHONPATH=. .venv/Scripts/python.exe scripts/face_edit.py \
        --card outputs/yua_desmile08_out.png --thinner --eyes-closer \
        --out outputs/yua_thinner_out.png --render

    # explicit targets (absolute / percent / sigma)
    PYTHONPATH=. .venv/Scripts/python.exe scripts/face_edit.py --card <c> \
        --set "icd/facewidth=-1.5sigma" --set "face_height/width=+8%"

Targets and presets nudge the card's CURRENT ratios; the solver holds the other ratios and keeps the param
change small. Needs the ratio reference (scripts/face_report.py --build-reference). See the plan / docs.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.face_metrics import edit, reference

_PRESET_FLAGS = list(edit.PRESETS.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", dest="sets", action="append", default=[],
                    help='explicit target "ratio_name=VALUE[%%|sigma]" (repeatable)')
    for name in _PRESET_FLAGS:
        ap.add_argument(f"--{name}", action="store_true", help=f"preset: {edit.PRESETS[name]}")
    ap.add_argument("--strength", type=float, default=1.5, help="preset magnitude in sigma (default 1.5)")
    ap.add_argument("--sliders-only", action="store_true", help="move only shapeValueFace sliders, no ABMX")
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--hold-weight", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--max-iters", type=int, default=10)
    ap.add_argument("--tol-sigma", type=float, default=0.05)
    ap.add_argument("--max-step", type=float, default=0.5)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write a card")
    args = ap.parse_args()

    from src.hs2_mesh import _rig
    from src.face_data_utils.utils import FaceData
    rig = _rig()
    ref = reference.load_reference()
    names = list(ref["ratio_names"])
    subset = ref["subset"]
    mad_by = dict(zip(names, np.asarray(ref["mad"], float)))

    # current ratios (for relative/preset targets)
    fd = FaceData(args.card)
    p0 = edit.current_p(fd)
    r_before = edit.forward_ratios(rig, fd, p0, subset, names)
    r0_by = dict(zip(names, r_before))

    # assemble targets: presets first, then --set overrides
    active = [n for n in _PRESET_FLAGS if getattr(args, n.replace("-", "_"))]
    targets = edit.expand_presets(active, args.strength, r0_by, mad_by)
    for spec in args.sets:
        n, g = edit.parse_target(spec, names, r0_by, mad_by)
        targets[n] = g
    if not targets:
        ap.error("no targets — pass a preset (e.g. --thinner) or --set \"ratio=VALUE\"")

    free_idx = range(54) if args.sliders_only else None
    p1, info = edit.solve(rig, fd, p0, targets, ref, lam=args.lam, hold_weight=args.hold_weight,
                          eps=args.eps, max_iters=args.max_iters, tol_sigma=args.tol_sigma,
                          max_step=args.max_step, free_idx=free_idx)

    out = args.out or os.path.join("outputs", os.path.splitext(os.path.basename(args.card))[0] + "_edited.png")
    mode = "sliders-only" if args.sliders_only else "all-params"
    print(edit.format_edit_report(info, args.card, out, mode))

    if args.dry_run:
        print("\n[dry-run] no card written.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    out_fd = FaceData(args.card)                       # fresh load preserves all non-face card data
    out_fd.set_from_vector(p1.astype(np.float32), is_simplify=True, without_right=True,
                           denormalize=True, use_gaussian=False)
    out_fd.save(out)
    print(f"\nwrote card -> {out}")

    if args.render:
        from PIL import Image
        from scripts.hs2_render_mesh import render
        from src.hs2_mesh import card_to_mesh
        vb, fb = card_to_mesh(args.card)
        va, fa = card_to_mesh(out)
        before, after = render(vb, fb, yaw=0), render(va, fa, yaw=0)
        gap = np.full((before.shape[0], 8), 30, np.uint8)
        combo = np.concatenate([before, gap, after], axis=1)
        rpath = os.path.splitext(out)[0] + "_edit_render.png"
        Image.fromarray(combo).save(rpath)
        print(f"wrote render (before | after) -> {rpath}")


if __name__ == "__main__":
    main()
