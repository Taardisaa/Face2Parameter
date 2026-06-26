"""Generate the 205-dim ``labels.json`` from character-card PNGs (real-data path).

Supersedes the inconsistent 54-dim ``src/utils/label_gen.py``. Goes through the kept
``FaceData`` contract so the label normalization matches inference exactly:

    train:  to_vector(is_simplify=True, without_right=True, normalize=True, use_gaussian=False)
    infer:  set_from_vector(..., denormalize=True, use_gaussian=False)

Requires ``HS2ABMX.exe`` + the stat JSONs under ``src/face_data_utils/``. Not needed for
the skeleton smoke test (synthetic labels are produced by tools/make_synthetic_data.py).

Usage:
    .venv/Scripts/python.exe tools/gen_labels.py --cards-dir data/cards --out data/labels.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

# allow running as `python tools/gen_labels.py` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm

from src.face_data_utils.utils import FaceData


def gen_labels(cards_dir: str, out_path: str) -> int:
    paths = glob.glob(os.path.join(cards_dir, "**", "*.png"), recursive=True)
    labels = {}
    for path in tqdm(paths, desc="cards"):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            fd = FaceData(path)
            vec = fd.to_vector(is_simplify=True, without_right=True,
                               normalize=True, use_gaussian=False)
            labels[name] = vec.tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"[gen_labels] skip {name}: {exc}")
            continue

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    dim = len(next(iter(labels.values()))) if labels else 0
    print(f"[gen_labels] wrote {len(labels)} labels (dim={dim}) -> {out_path}")
    return len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-dir", required=True, help="dir of character-card PNGs")
    ap.add_argument("--out", default="data/labels.json")
    args = ap.parse_args()
    gen_labels(args.cards_dir, args.out)


if __name__ == "__main__":
    main()
