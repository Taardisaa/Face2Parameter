"""Inspect-only expression neutralization: de-smile image(s) and eyeball the result.

Runs the LivePortrait-backed Neutralizer over a file or directory and writes the neutralized
crops plus a JSON report (per image: kept/rejected + ArcFace identity similarity) so you can judge
the edits BEFORE trusting them in predict.py/infer.py. Requires an external LivePortrait install
(see docs/expression-invariance.md); set LIVEPORTRAIT_DIR / LIVEPORTRAIT_PYTHON / LIVEPORTRAIT_NEUTRAL.

Usage:
    .venv/Scripts/python.exe tools/neutralize.py --image inputs/person/ --out outputs/_neutral_check
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.img_utils import list_images
from src.neutralize import Neutralizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image file OR a directory")
    ap.add_argument("--out", default="outputs/_neutral_check", help="where to copy neutral crops + report")
    ap.add_argument("--alpha", type=float, default=0.0, help="0=full neutral, 1=unchanged")
    ap.add_argument("--gate-threshold", type=float, default=0.6, help="min ArcFace id sim to accept")
    args = ap.parse_args()

    image_paths = list_images(args.image)
    if not image_paths:
        raise SystemExit(f"no images found at {args.image}")

    neut = Neutralizer(mode="liveportrait", alpha=args.alpha, gate_threshold=args.gate_threshold)
    kept, report = neut(image_paths)

    os.makedirs(args.out, exist_ok=True)
    # copy the produced neutral crops (even rejected ones) next to the report for visual inspection
    for it in report["items"]:
        if it.get("neutralized") and os.path.exists(it["neutralized"]):
            stem = os.path.splitext(os.path.basename(it["src"]))[0]
            tag = "kept" if it["kept"] else "REJECT"
            dst = os.path.join(args.out, f"{stem}_{tag}{os.path.splitext(it['neutralized'])[1]}")
            shutil.copyfile(it["neutralized"], dst)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[neutralize] {report['n_neutralized']}/{len(image_paths)} accepted "
          f"(threshold {args.gate_threshold}); crops + report.json -> {args.out}")


if __name__ == "__main__":
    main()
