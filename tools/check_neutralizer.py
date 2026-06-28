"""Verify the LivePortrait install used by the optional (delta-zeroing) neutralizer.

Checks that LIVEPORTRAIT_DIR resolves to a LivePortrait checkout, that our neutralization script
is present, and (optionally, with --image) that a 1-image neutralization actually runs.

Run inside the project venv:
    .venv/Scripts/python.exe tools/check_neutralizer.py
    .venv/Scripts/python.exe tools/check_neutralizer.py --image inputs/person/one.jpg
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neutralize import Neutralizer, NeutralizerError, _SCRIPT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="optional: a single image for a smoke neutralization")
    ap.add_argument("--alpha", type=float, default=0.0)
    args = ap.parse_args()

    n = Neutralizer(mode="liveportrait", alpha=args.alpha)
    print("=== config ===")
    print(f"  LIVEPORTRAIT_DIR    = {n.lp_dir}")
    print(f"  LIVEPORTRAIT_PYTHON = {n.lp_python}")
    print(f"  neutralize script   = {_SCRIPT}")
    print(f"  alpha               = {n.alpha}  (0 = full neutral)")

    print("=== reachability ===")
    try:
        n._require_install()
        print("  [ok]   LivePortrait checkout + neutralization script found")
    except NeutralizerError as exc:
        print(f"  [FAIL] {exc}")
        return 1

    if args.image:
        print("=== 1-image smoke neutralization ===")
        try:
            kept, report = n([args.image])
            it = report["items"][0]
            print(f"  result: kept={it['kept']} sim={it.get('sim')} neutralized={it.get('neutralized')}")
            if not it.get("neutralized"):
                print("  [FAIL] no output produced")
                return 1
            print("  [ok]   neutralization ran")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {exc}")
            return 1

    print("=== result ===")
    print("  PASS: neutralizer looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
