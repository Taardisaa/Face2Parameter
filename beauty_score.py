"""Facial-attractiveness scoring: give a face photo -> get a 1-5 beauty score.

Reuses the exact Stage-1/Stage-2 pipeline as predict.py (frozen DINOv2 backbone + MLP
head), only the head is the 1-dim beauty regressor trained on SCUT-FBP5500
(``--config beauty_dinov2_vits14``). Preprocessing/alignment is shared with training via
``load_face_rgb`` so inference crops match the training crops.

    from beauty_score import beauty_score, BeautyScorer
    beauty_score("face.jpg")                 # -> float in [1,5]
    s = BeautyScorer(); s.score_many(paths)  # reuse one loaded model for many images

CLI:
    .venv/Scripts/python.exe beauty_score.py --image face.jpg
    .venv/Scripts/python.exe beauty_score.py --image folder_of_faces/
    .venv/Scripts/python.exe beauty_score.py --eval      # Pearson r + MAE on the val split

Domain caveat: the model is trained on REAL human faces (SCUT-FBP5500). Scores for
stylized / anime / rendered game-character faces are OUT OF DISTRIBUTION and not reliable.
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from config import get_config, resolve_head
from predict import _embed_and_predict, _load_model
from src.img_utils import list_images, load_face_rgb

SCORE_MIN, SCORE_MAX = 1.0, 5.0
OOD_NOTE = ("[beauty_score] note: trained on real faces (SCUT-FBP5500); scores for "
            "anime/rendered/game-character faces are out-of-distribution and unreliable.")


class BeautyScorer:
    """Loads the backbone + beauty head once; scores one or many images."""

    def __init__(self, config: str = "beauty_dinov2_vits14", head: str | None = None,
                 use_detector: bool = True):
        self.cfg = get_config(config)
        self.head_path = resolve_head(self.cfg, head)
        self.model, self.device = _load_model(self.cfg, self.head_path)
        self.use_detector = use_detector
        self._detector = None  # lazily-built, reused FaceCrop (loads mtcnn once)

    def _get_detector(self, use_detector: bool):
        if not use_detector:
            return None
        if self._detector is None:
            from src.face_data_utils.FaceCrop import FaceCrop
            self._detector = FaceCrop()
        return self._detector

    def _to_face_rgb(self, image, use_detector: bool) -> np.ndarray:
        """Path or RGB uint8 ndarray -> (img_size, img_size, 3) RGB uint8."""
        if isinstance(image, np.ndarray):
            img = image
            if img.shape[:2] != (self.cfg.img_size, self.cfg.img_size):
                img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size),
                                 interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(img)
        return load_face_rgb(image, self.cfg.img_size, use_detector,
                             detector=self._get_detector(use_detector))

    def score(self, image, use_detector: bool | None = None) -> float:
        """One image (path or RGB ndarray) -> clamped [1,5] beauty score."""
        ud = self.use_detector if use_detector is None else use_detector
        img = self._to_face_rgb(image, ud)
        _, vec = _embed_and_predict(self.model, img, self.device)
        return float(np.clip(vec[0], SCORE_MIN, SCORE_MAX))

    def score_many(self, paths, use_detector: bool | None = None) -> list:
        return [self.score(p, use_detector=use_detector) for p in paths]


def beauty_score(image, config: str = "beauty_dinov2_vits14", head: str | None = None,
                 use_detector: bool = True) -> float:
    """Convenience one-shot scorer (loads the model per call)."""
    return BeautyScorer(config, head, use_detector).score(image)


def _read_stems(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def evaluate(scorer: BeautyScorer) -> dict:
    """Pearson r + MAE of the head on the prepared val split (aligned images, no re-detect)."""
    root = scorer.cfg.data_dir
    with open(os.path.join(root, "labels.json"), "r", encoding="utf-8") as f:
        labels = json.load(f)
    stems = _read_stems(os.path.join(root, "val_features.txt"))
    preds, gts = [], []
    for stem in stems:
        img_path = os.path.join(root, "images", stem + ".png")
        if not os.path.exists(img_path) or stem not in labels:
            continue
        # Images are already the aligned crops from prep -> score without re-detecting,
        # matching how extract_features.py cached the training features.
        preds.append(scorer.score(img_path, use_detector=False))
        gts.append(float(labels[stem][0]))
    preds, gts = np.array(preds), np.array(gts)
    r = float(np.corrcoef(preds, gts)[0, 1]) if len(preds) > 1 else float("nan")
    mae = float(np.mean(np.abs(preds - gts))) if len(preds) else float("nan")
    return {"n": len(preds), "pearson": r, "mae": mae}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="beauty_dinov2_vits14")
    ap.add_argument("--image", help="image file OR a directory of faces")
    ap.add_argument("--head", default=None,
                    help="head weights (.pth); defaults to latest trained exp checkpoint")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip mtcnn alignment; aspect-preserving center-crop instead")
    ap.add_argument("--eval", action="store_true",
                    help="report Pearson r + MAE on the prepared val split, then exit")
    args = ap.parse_args()

    scorer = BeautyScorer(args.config, args.head, use_detector=not args.no_detector)

    if args.eval:
        m = evaluate(scorer)
        print(f"[eval] n={m['n']}  Pearson r={m['pearson']:.4f}  MAE={m['mae']:.4f}")
        return

    if not args.image:
        raise SystemExit("pass --image <file|dir> (or --eval)")
    paths = list_images(args.image)
    if not paths:
        raise SystemExit(f"no images found at {args.image}")

    print(OOD_NOTE)
    if len(paths) == 1:
        print(f"{os.path.basename(paths[0])}\t{scorer.score(paths[0]):.3f}")
    else:
        scores = scorer.score_many(paths)
        for p, s in sorted(zip(paths, scores), key=lambda t: -t[1]):
            print(f"{os.path.basename(p)}\t{s:.3f}")
        print(f"[beauty_score] {len(scores)} image(s), "
              f"mean={np.mean(scores):.3f} min={np.min(scores):.3f} max={np.max(scores):.3f}")


if __name__ == "__main__":
    main()
