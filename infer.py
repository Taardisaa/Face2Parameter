"""Inference: image(s) -> 205 face params -> game character card.

Refactors ``extractor.py`` to the new backbone+head model. The 205-dim vector is
written back into a real character card via the kept ``FaceData`` contract.

``--image`` may be a single file OR a directory of photos of one person (the latter is
aggregated into one stable vector; see docs/multi-image-aggregation.md). ``--ensemble``
runs more than one backbone and merges them in param space. The face-shape params are
written into the template card; the consolidated set's most representative (medoid) photo
becomes the card thumbnail.

Requires the real-data path: a trained head, the backbone (one-time download), plus
``HS2ABMX.exe`` and the stat JSONs under ``src/face_data_utils/`` (used by ``FaceData``)
and ``mtcnn_ort`` (used by ``FaceCrop``). Not exercised by the skeleton smoke test.

Usage:
    .venv/Scripts/python.exe infer.py --config dinov2_vits14 --image test/my.png --out outputs/
    .venv/Scripts/python.exe infer.py --config arcface --image dir_of_photos/ --out outputs/
    .venv/Scripts/python.exe infer.py --ensemble dinov2_vits14,arcface --image dir_of_photos/
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from config import get_config, resolve_head
from predict import ensemble_predict, predict_set
from src.img_utils import list_images


def write_card(vector, face_img, template_path: str, out_dir: str, out_name: str) -> str:
    """Write a 205-dim vector + thumbnail into a copy of the template card."""
    from src.face_data_utils.utils import FaceData  # lazy: needs .exe + stat JSONs

    os.makedirs(out_dir, exist_ok=True)
    face_data = FaceData(template_path)
    face_data.set_from_vector(vector, is_simplify=True, without_right=True,
                              denormalize=True, use_gaussian=False)
    face_data.set_image(np.asarray(face_img).astype(np.uint8))
    save_path = os.path.join(out_dir, out_name + "_out.png")
    face_data.save(save_path)
    print(f"[infer] wrote card -> {save_path}")
    return save_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--head", default=None,
                    help="head weights (.pth); defaults to latest trained exp checkpoint, "
                         "else the bundled release/ head (ignored with --ensemble)")
    ap.add_argument("--image", required=True, help="image file OR a directory of photos of one person")
    ap.add_argument("--template", default=None,
                    help="HS2 card to write into; defaults to the bundled assets/default_template.png")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip mtcnn face alignment; aspect-preserving center-crop instead")
    ap.add_argument("--aggregate", choices=["median", "mean", "trimmed"], default="median",
                    help="how to aggregate multiple images (default: median)")
    ap.add_argument("--aggregate-space", choices=["embedding", "param"], default="embedding",
                    help="aggregate backbone features (default) or per-image param vectors")
    ap.add_argument("--ensemble", default=None,
                    help="comma-separated configs to run and merge in param space, "
                         "e.g. dinov2_vits14,arcface (overrides --config)")
    args = ap.parse_args()

    image_paths = list_images(args.image)
    if not image_paths:
        raise SystemExit(f"no images found at {args.image}")
    use_detector = not args.no_detector

    if args.ensemble:
        config_names = [c.strip() for c in args.ensemble.split(",") if c.strip()]
        vector, results = ensemble_predict(config_names, image_paths, use_detector,
                                           method=args.aggregate, space=args.aggregate_space)
        rep = results[0][1]
        face_img = rep["representative_img"]
        tmpl_cfg = get_config(config_names[0])
        print(f"[infer] ensemble({'+'.join(config_names)}), "
              f"aggregate={args.aggregate}/{args.aggregate_space}")
    else:
        cfg = get_config(args.config)
        head = resolve_head(cfg, args.head)
        rep = predict_set(cfg, head, image_paths, use_detector,
                          method=args.aggregate, space=args.aggregate_space)
        vector = rep["vector"]
        face_img = rep["representative_img"]
        tmpl_cfg = cfg
        print(f"[infer] config={args.config}, aggregate={args.aggregate}/{args.aggregate_space}")

    print(f"[infer] {rep['n_used']} image(s) used"
          + (f", {len(rep['skipped'])} skipped" if rep['skipped'] else "")
          + f"; thumbnail from {os.path.basename(rep['representative_name'])}")

    template = args.template or tmpl_cfg.default_template
    if not os.path.exists(template):
        raise FileNotFoundError(f"template card not found: {template} (pass --template <card.png>)")

    base = os.path.basename(os.path.normpath(args.image))
    out_name = base if os.path.isdir(args.image) else os.path.splitext(base)[0]
    write_card(vector, face_img, template, out_dir=args.out, out_name=out_name)


if __name__ == "__main__":
    main()
