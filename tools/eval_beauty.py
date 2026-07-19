"""Evaluate the trained beauty head on the prepared SCUT splits (real faces).

Reports Pearson r / MAE / RMSE on the train and held-out test splits, plus a per-subset
breakdown (SCUT prefixes AF/AM/CF/CM = Asian/Caucasian x Female/Male). Optionally also
scores arbitrary extra images (e.g. out-of-distribution anime / game-character shots) with
mtcnn alignment, reporting whether a face was actually detected.

Splits/images/labels come from the beauty config's ``data_dir`` (built by
tools/prep_beauty_data.py). The aligned crops are scored WITHOUT re-detecting (they are
already the aligned faces the training features were cached from).

Usage:
    .venv/Scripts/python.exe tools/eval_beauty.py
    .venv/Scripts/python.exe tools/eval_beauty.py --extra tests/          # + OOD images
    .venv/Scripts/python.exe tools/eval_beauty.py --config beauty_dinov2_vits14 --head <w.pth>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beauty_score import BeautyScorer
from src.img_utils import list_images, load_face_rgb

SUBSET_NAMES = {"AF": "Asian Female", "AM": "Asian Male",
                "CF": "Caucasian Female", "CM": "Caucasian Male"}


def _read_stems(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _metrics(scorer, root, labels, stems):
    """(n, pearson, mae, rmse) over already-aligned crops (no re-detection)."""
    preds, gts = [], []
    for st in stems:
        p = os.path.join(root, "images", st + ".png")
        if not os.path.exists(p) or st not in labels:
            continue
        preds.append(scorer.score(p, use_detector=False))
        gts.append(float(labels[st][0]))
    preds, gts = np.array(preds), np.array(gts)
    if len(preds) < 2:
        return len(preds), float("nan"), float("nan"), float("nan")
    r = float(np.corrcoef(preds, gts)[0, 1])
    mae = float(np.mean(np.abs(preds - gts)))
    rmse = float(np.sqrt(np.mean((preds - gts) ** 2)))
    return len(preds), r, mae, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="beauty_dinov2_vits14")
    ap.add_argument("--head", default=None)
    ap.add_argument("--extra", default=None,
                    help="image file or dir of OOD images to also score (detector on)")
    args = ap.parse_args()

    scorer = BeautyScorer(args.config, args.head)
    root = scorer.cfg.data_dir
    with open(os.path.join(root, "labels.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)

    print("=== REAL FACES ===")
    print(f"{'split':16s} {'n':>5s} {'Pearson':>8s} {'MAE':>6s} {'RMSE':>6s}")
    for name, split in [("train", "train_features.txt"),
                        ("test (held-out)", "val_features.txt")]:
        stems = _read_stems(os.path.join(root, split))
        n, r, mae, rmse = _metrics(scorer, root, labels, stems)
        print(f"{name:16s} {n:5d} {r:8.4f} {mae:6.3f} {rmse:6.3f}")

    # per-subset on the held-out test split, grouped by leading alpha prefix
    test = _read_stems(os.path.join(root, "val_features.txt"))
    groups = {}
    for st in test:
        m = re.match(r"([A-Za-z]+)", st)
        if m:
            groups.setdefault(m.group(1), []).append(st)
    if len(groups) > 1:
        print("\n=== test set by subset ===")
        for g in sorted(groups):
            n, r, mae, rmse = _metrics(scorer, root, labels, groups[g])
            print(f"{g} {SUBSET_NAMES.get(g, ''):18s} n={n:4d}  "
                  f"Pearson={r:.4f}  MAE={mae:.3f}")

    if args.extra:
        paths = list_images(args.extra)
        print(f"\n=== OOD extras ({len(paths)} image(s), detector ON) ===")
        print("NOTE: model is trained on real faces; these scores are out-of-distribution "
              "and unreliable.")
        det = scorer._get_detector(True)
        for p in paths:
            img, detected = load_face_rgb(p, scorer.cfg.img_size, True,
                                          return_detected=True, detector=det)
            sc = scorer.score(img)
            tag = "" if detected else "   <- center-crop fallback (no face found)"
            print(f"  {os.path.basename(p):42s} face={detected!s:5s} score={sc:.3f}{tag}")


if __name__ == "__main__":
    main()
