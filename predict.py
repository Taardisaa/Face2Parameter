"""Predict the 205-dim face-parameter vector from an image (no card write-back).

This is the lightweight "give a picture -> get params" path. It needs only the
trained head + the DINOv2 backbone -- no HS2ABMX.exe, no template card, no mtcnn.
The output vector is in the SAME normalized space as labels.json.

For best results give a roughly cropped, front-facing face (training faces are
aligned 224x224). With --detector it will mtcnn-align first, if mtcnn_ort is installed.

``--image`` may be a single file OR a directory of photos of one person: in the latter
case each image is run through the model and the per-image results are aggregated into one
stable vector (see docs/multi-image-aggregation.md). ``--ensemble`` additionally runs more
than one backbone and merges their outputs in param space.

Usage:
    .venv/Scripts/python.exe predict.py --config dinov2_vits14 --image face.png
    .venv/Scripts/python.exe predict.py --config arcface --image dir_of_photos/
    .venv/Scripts/python.exe predict.py --ensemble dinov2_vits14,arcface --image dir_of_photos/
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from config import get_config, resolve_head
from src.aggregate import aggregate, ensemble, medoid_index
from src.img_utils import list_images, load_face_rgb
from src.models.face2param import Face2Param


def _load_model(cfg, head_path):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = Face2Param.from_checkpoint(cfg, head_path, map_location="cpu").to(device).eval()
    return model, device


def _embed_and_predict(model, img_rgb, device):
    """One image -> (backbone feature (D,), 205-dim param vector)."""
    chw = np.ascontiguousarray(np.transpose(img_rgb, (2, 0, 1)))
    x = torch.from_numpy(chw).float().div(255).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.backbone(x)
        vec = model.head(feat)
    return (feat.squeeze(0).cpu().numpy().astype(np.float32),
            vec.squeeze(0).cpu().numpy().astype(np.float32))


def predict(cfg, head_path, image_path, use_detector=True):
    """Single image -> 205-dim vector. Kept stable for tools/eval_expression_consistency."""
    model, device = _load_model(cfg, head_path)
    img = load_face_rgb(image_path, cfg.img_size, use_detector)
    _, vec = _embed_and_predict(model, img, device)
    return vec


def predict_set(cfg, head_path, image_paths, use_detector=True,
                method="median", space="embedding"):
    """Aggregate several images of one person into one stable 205-dim vector.

    space="embedding": aggregate backbone features, then run the head once (ArcFace
    embeddings are re-L2-normalized first). space="param": aggregate the per-image
    205-dim outputs. Returns a dict with the vector, per-image vectors, skipped frames,
    and a representative (medoid) image for the card thumbnail.
    """
    model, device = _load_model(cfg, head_path)
    print(f"[predict_set] {len(image_paths)} image(s) found; backbone={cfg.backbone}, "
          f"detector={'on' if use_detector else 'off'}")
    feats, vecs, names, imgs, skipped = [], [], [], [], []
    for p in image_paths:
        nm = os.path.basename(p)
        try:
            img, detected = load_face_rgb(p, cfg.img_size, use_detector, return_detected=True)
        except Exception as exc:  # noqa: BLE001 - unreadable file: skip and report
            print(f"[predict_set]   skip (unreadable) {nm}: {exc}")
            skipped.append(p)
            continue
        if use_detector and not detected:
            print(f"[predict_set]   skip (no face)    {nm}")
            skipped.append(p)
            continue
        feat, vec = _embed_and_predict(model, img, device)
        feats.append(feat)
        vecs.append(vec)
        names.append(p)
        imgs.append(img)
        print(f"[predict_set]   ok                {nm}"
              + (" (face detected)" if use_detector else " (center-crop)"))

    if not feats:
        raise RuntimeError("no usable images (all skipped); try --no-detector or check inputs")
    feats, vecs = np.stack(feats), np.stack(vecs)
    n_used = len(feats)
    print(f"[predict_set] using {n_used}/{len(image_paths)} image(s)"
          + (f"; skipped {len(skipped)}" if skipped else ""))
    if n_used < 5:
        print(f"[predict_set] note: only {n_used} usable image(s); ~5-10 varied shots give the "
              f"most stable result (see docs/multi-image-aggregation.md)")

    if space == "embedding":
        agg_feat, mask = aggregate(feats, method)
        if cfg.backbone == "arcface":  # head trained on unit-norm ArcFace embeddings
            agg_feat = agg_feat / (np.linalg.norm(agg_feat) + 1e-9)
        x = torch.from_numpy(agg_feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            vector = model.head(x).squeeze(0).cpu().numpy().astype(np.float32)
        rep_idx = medoid_index(feats, agg_feat)
    elif space == "param":
        vector, mask = aggregate(vecs, method)
        rep_idx = medoid_index(vecs, vector)
    else:
        raise ValueError(f"unknown aggregate space '{space}' (embedding|param)")

    per_image = {os.path.splitext(os.path.basename(n))[0]: v for n, v in zip(names, vecs)}
    return {
        "vector": vector,
        "per_image": per_image,
        "skipped": skipped,
        "n_used": n_used,
        "representative_img": imgs[rep_idx],
        "representative_name": names[rep_idx],
        "kept_mask": mask,
    }


def ensemble_predict(config_names, image_paths, use_detector=True,
                     method="median", space="embedding", weights=None):
    """Run several backbones and merge their 205-dim outputs in param space.

    Each config resolves its own head via config.resolve_head (a single --head can't
    serve multiple backbones). Returns (merged_vector, [(config_name, result_dict), ...]).
    """
    results = []
    for name in config_names:
        cfg = get_config(name)
        head = resolve_head(cfg)
        print(f"[ensemble] running '{name}' (backbone={cfg.backbone}) ...")
        results.append((name, predict_set(cfg, head, image_paths, use_detector, method, space)))
    merged = ensemble(np.stack([r["vector"] for _, r in results]), weights=weights)
    return merged, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--image", required=True, help="image file OR a directory of photos of one person")
    ap.add_argument("--head", default=None,
                    help="head weights (.pth); defaults to latest trained exp checkpoint, "
                         "else the bundled release/ head (ignored with --ensemble)")
    ap.add_argument("--out", default=None, help="defaults to outputs/<image>_out.json (+.npy)")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip mtcnn face alignment; aspect-preserving center-crop instead")
    ap.add_argument("--aggregate", choices=["median", "mean", "trimmed"], default="median",
                    help="how to aggregate multiple images (default: median)")
    ap.add_argument("--aggregate-space", choices=["embedding", "param"], default="embedding",
                    help="aggregate backbone features (default) or per-image param vectors")
    ap.add_argument("--ensemble", default=None,
                    help="comma-separated configs to run and merge in param space, "
                         "e.g. dinov2_vits14,arcface (overrides --config)")
    ap.add_argument("--save-per-image", action="store_true",
                    help="also write each image's pre-aggregation 205-dim vector")
    args = ap.parse_args()

    image_paths = list_images(args.image)
    if not image_paths:
        raise SystemExit(f"no images found at {args.image}")
    use_detector = not args.no_detector

    if args.ensemble:
        config_names = [c.strip() for c in args.ensemble.split(",") if c.strip()]
        vec, results = ensemble_predict(config_names, image_paths, use_detector,
                                        method=args.aggregate, space=args.aggregate_space)
        label = f"ensemble({'+'.join(config_names)})"
        per_image_groups = {name: r["per_image"] for name, r in results}
        n_used = {name: r["n_used"] for name, r in results}
    else:
        cfg = get_config(args.config)
        head_path = resolve_head(cfg, args.head)
        res = predict_set(cfg, head_path, image_paths, use_detector,
                          method=args.aggregate, space=args.aggregate_space)
        vec = res["vector"]
        label = f"{args.config} (head={os.path.basename(head_path)})"
        per_image_groups = {args.config: res["per_image"]} if args.save_per_image else {}
        n_used = {args.config: res["n_used"]}

    base = os.path.basename(os.path.normpath(args.image))
    stem = base if os.path.isdir(args.image) else os.path.splitext(base)[0]
    out_json = args.out or os.path.join("outputs", stem + "_out.json")
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    np.save(os.path.splitext(out_json)[0] + ".npy", vec)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(vec.tolist(), f)

    if args.save_per_image and per_image_groups:
        per_dir = os.path.join(os.path.dirname(out_json) or ".", stem + "_per_image")
        for grp, per_image in per_image_groups.items():
            sub = per_dir if len(per_image_groups) == 1 else os.path.join(per_dir, grp)
            os.makedirs(sub, exist_ok=True)
            for nm, v in per_image.items():
                np.save(os.path.join(sub, nm + ".npy"), v)
        print(f"per-image vectors -> {per_dir}/")

    print(f"model : {label}")
    print(f"input : {args.image}  ({'dir, ' if os.path.isdir(args.image) else ''}"
          f"images used: {n_used}, aggregate={args.aggregate}/{args.aggregate_space})")
    print(f"vector: dim={vec.shape[0]} | min={vec.min():.3f} max={vec.max():.3f} "
          f"mean={vec.mean():.3f}")
    print(f"first 8: {np.array2string(vec[:8], precision=3, floatmode='fixed')}")
    print(f"saved -> {out_json}  (+ .npy)")


if __name__ == "__main__":
    main()
