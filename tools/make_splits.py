"""Build train / val / test split index files for FeatureDataset.

Writes ``{train,val,test}_features.txt`` (one sample basename per line) into the
config's ``data_dir``. Names are the intersection of:
  - label keys in ``labels.json`` (read line-by-line to avoid loading 700+ MB), and
  - image basenames in ``data_dir/images`` (== aug_images; both share names).

The test split is held out entirely (never trained); use it for final metrics and
qualitative ``infer.py`` runs.

Usage:
    .venv/Scripts/python.exe tools/make_splits.py --config dinov2_vits14 --val 1000 --test 200
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config

_KEY_RE = re.compile(r'^\s*"(.+?)"\s*:\s*\[')


def label_keys(labels_path: str) -> set:
    keys = set()
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _KEY_RE.match(line)
            if m:
                keys.add(m.group(1))
    return keys


def image_names(images_dir: str) -> set:
    names = set()
    for fn in os.listdir(images_dir):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in (".png", ".jpg", ".jpeg"):
            names.add(stem)
    return names


def write_list(path: str, names: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")


def make(cfg, val: int, test: int, seed: int) -> None:
    labels_path = os.path.join(cfg.data_dir, "labels.json")
    images_dir = os.path.join(cfg.data_dir, "images")

    keys = label_keys(labels_path)
    imgs = image_names(images_dir)
    names = sorted(keys & imgs)
    missing_label = len(imgs - keys)
    missing_image = len(keys - imgs)
    print(f"[make_splits] labels={len(keys)} images={len(imgs)} "
          f"usable(intersection)={len(names)} "
          f"(imgs w/o label={missing_label}, labels w/o img={missing_image})")

    if len(names) < val + test + 1:
        raise ValueError(f"Not enough samples ({len(names)}) for val+test={val + test}.")

    random.Random(seed).shuffle(names)
    test_names = sorted(names[:test])
    val_names = sorted(names[test:test + val])
    train_names = sorted(names[test + val:])

    write_list(os.path.join(cfg.data_dir, "test_features.txt"), test_names)
    write_list(os.path.join(cfg.data_dir, "val_features.txt"), val_names)
    write_list(os.path.join(cfg.data_dir, "train_features.txt"), train_names)
    print(f"[make_splits] wrote train={len(train_names)} val={len(val_names)} "
          f"test={len(test_names)} -> {cfg.data_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    make(get_config(args.config), val=args.val, test=args.test, seed=args.seed)


if __name__ == "__main__":
    main()
