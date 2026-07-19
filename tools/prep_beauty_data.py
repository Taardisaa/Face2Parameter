"""Prepare SCUT-FBP5500 into the project's Stage-2 data contract for the beauty head.

Turns the extracted SCUT-FBP5500 dataset into what ``extract_features.py`` /
``FeatureDataset`` expect under the beauty config's ``data_dir`` (``face2beauty/data/``):

  - ``images/<stem>.png``     : each face **mtcnn-aligned with the same aligner inference
                                uses** (``src.img_utils.load_face_rgb``), so train crops
                                == inference crops. Faces mtcnn can't find are dropped.
  - ``labels.json``           : ``{stem: [score]}`` (1-element list to match the
                                ``{name: [vector]}`` schema; raw 60-rater mean in [1,5]).
  - ``train_features.txt`` / ``val_features.txt`` : SCUT's official 60/40 split, filtered
                                to the images that survived alignment.

Label source is SCUT's pre-aggregated ``All_labels.txt`` (== the 60-rater mean, verified);
the 20 MB ``All_Ratings.xlsx`` is never touched.

Usage:
    .venv/Scripts/python.exe tools/prep_beauty_data.py
    .venv/Scripts/python.exe tools/prep_beauty_data.py --no-detector   # skip mtcnn (center-crop)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.img_utils import load_face_rgb

SCUT_ROOT = os.path.join(
    "face2beauty", "SCUT-FBP5500-Database-Release", "data", "SCUT-FBP5500_v2")
LABELS_TXT = os.path.join(SCUT_ROOT, "train_test_files", "All_labels.txt")
SPLIT_DIR = os.path.join(SCUT_ROOT, "train_test_files",
                         "split_of_60%training and 40%testing")


def _read_scores(path: str) -> dict:
    """All_labels.txt: '<stem>.jpg <score>' per line -> {stem: float}."""
    scores = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname, val = line.rsplit(" ", 1)
            scores[os.path.splitext(fname)[0]] = float(val)
    return scores


def _read_split(path: str) -> list:
    """A split file ('<stem>.jpg <score>' per line) -> ordered list of stems."""
    stems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fname = line.split(" ", 1)[0]
            stems.append(os.path.splitext(fname)[0])
    return stems


def _imwrite_unicode(path: str, img_bgr: np.ndarray) -> None:
    """cv2.imwrite that tolerates non-ASCII paths (mirrors the imdecode read pattern)."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    buf.tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="beauty_dinov2_vits14")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip mtcnn alignment; aspect-preserving center-crop instead")
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N images (smoke test)")
    args = ap.parse_args()

    cfg = get_config(args.config)
    use_detector = not args.no_detector
    out_root = cfg.data_dir
    img_out = os.path.join(out_root, "images")
    os.makedirs(img_out, exist_ok=True)

    scores = _read_scores(LABELS_TXT)
    train_stems = _read_split(os.path.join(SPLIT_DIR, "train.txt"))
    val_stems = _read_split(os.path.join(SPLIT_DIR, "test.txt"))
    print(f"[prep] labels={len(scores)}  official split: train={len(train_stems)} "
          f"val={len(val_stems)}")

    img_dir = os.path.join(SCUT_ROOT, "Images")
    names = sorted(scores.keys())
    if args.limit:
        names = names[:args.limit]

    # One shared FaceCrop (loads the mtcnn ONNX models once) reused across all images.
    detector = None
    if use_detector:
        from src.face_data_utils.FaceCrop import FaceCrop
        detector = FaceCrop()

    kept, dropped, missing = [], [], []
    for i, stem in enumerate(names):
        src = os.path.join(img_dir, stem + ".jpg")
        if not os.path.exists(src):
            missing.append(stem)
            continue
        try:
            img, detected = load_face_rgb(src, cfg.img_size, use_detector,
                                          return_detected=True, detector=detector)
        except Exception as exc:  # noqa: BLE001 - unreadable image: drop and report
            print(f"[prep]   drop (unreadable) {stem}: {exc}")
            dropped.append(stem)
            continue
        if use_detector and not detected:
            dropped.append(stem)
            continue
        _imwrite_unicode(os.path.join(img_out, stem + ".png"),
                         cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        kept.append(stem)
        if (i + 1) % 500 == 0:
            print(f"[prep]   {i + 1}/{len(names)}  kept={len(kept)} dropped={len(dropped)}")

    kept_set = set(kept)
    labels = {s: [scores[s]] for s in kept}
    with open(os.path.join(out_root, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f)

    train_kept = [s for s in train_stems if s in kept_set]
    val_kept = [s for s in val_stems if s in kept_set]
    with open(os.path.join(out_root, "train_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_kept) + "\n")
    with open(os.path.join(out_root, "val_features.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_kept) + "\n")

    vals = np.array([scores[s] for s in kept], dtype=np.float32)
    print(f"\n[prep] done -> {out_root}")
    print(f"[prep]   kept={len(kept)}  dropped(no-face)={len(dropped)}  "
          f"missing-file={len(missing)}")
    print(f"[prep]   split written: train={len(train_kept)} val={len(val_kept)}")
    if len(vals):
        print(f"[prep]   score stats: min={vals.min():.4f} max={vals.max():.4f} "
              f"mean={vals.mean():.4f}")


if __name__ == "__main__":
    main()
