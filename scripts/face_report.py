"""Facial-ratio report CLI — harmony diagnostic, card-vs-card compare, and (Phase 2) fidelity-vs-photo.

    # one-time: build the population reference from labels.json
    PYTHONPATH=. .venv/Scripts/python.exe scripts/face_report.py --build-reference

    # harmony of one card (how typical/coordinated is the feature combination)
    PYTHONPATH=. .venv/Scripts/python.exe scripts/face_report.py --card outputs/yua_desmile08_out.png

    # compare two cards (where do they differ, in sigma units)
    PYTHONPATH=. .venv/Scripts/python.exe scripts/face_report.py \
        --card-a outputs/yua_desmile08_out.png --card-b tests/HS2ChaF_20240901192905747.png

    # fidelity: does the card match the input photo (Phase 2)
    PYTHONPATH=. .venv/Scripts/python.exe scripts/face_report.py \
        --card outputs/yua_desmile08_out.png --photo tests/yua_ariga_06.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.face_metrics import report, reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", help="harmony report for this card (or fidelity with --photo)")
    ap.add_argument("--card-a", dest="card_a")
    ap.add_argument("--card-b", dest="card_b")
    ap.add_argument("--photo", help="input photo for a fidelity report (with --card)")
    ap.add_argument("--build-reference", action="store_true", help="(re)build the population reference")
    ap.add_argument("--labels", default="data/labels.json")
    ap.add_argument("--template", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--n", type=int, default=5000, help="reference sample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", default="STRUCTURAL", help="fidelity ratio subset: STRUCTURAL / FULL")
    ap.add_argument("--render-backend", action="store_true",
                    help="fidelity: detect on a render of the card instead of using rig bones")
    args = ap.parse_args()

    if args.build_reference:
        out, names, n, dropped = reference.build_reference(args.labels, args.template, args.n, args.seed)
        print(f"built reference: {n} samples, {len(names)} ratios -> {out}")
        if dropped:
            print(f"dropped (near-zero spread): {', '.join(dropped)}")
        print("ratios:", ", ".join(names))
        return

    if args.card_a and args.card_b:
        print(report.format_compare(report.compare(args.card_a, args.card_b), args.card_a, args.card_b))
        return

    if args.card and args.photo:
        res = report.fidelity(args.card, args.photo, subset=args.subset, render_backend=args.render_backend)
        print(report.format_fidelity(res, args.card, args.photo))
        return

    if args.card:
        print(report.format_harmony(report.harmony(args.card), args.card))
        return

    ap.error("need --card, --card-a/--card-b, --card+--photo, or --build-reference")


if __name__ == "__main__":
    main()
