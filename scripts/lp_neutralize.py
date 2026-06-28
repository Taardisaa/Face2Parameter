"""Driver-free expression neutralization via LivePortrait internals (delta-zeroing).

Instead of transferring a driver face's expression (cross-identity -> distortion), this takes the
subject's OWN implicit keypoints and zeroes the expression deviation `exp` while keeping their pose,
scale and translation -- relaxing *their* face to neutral. This is the principled neutralization the
LivePortrait CLI doesn't expose (feature request #500).

    x_d_neutral = scale * (kp_canonical @ R_source + alpha * exp) + t      # alpha=0 => full neutral
    x_d_neutral = stitching(x_source, x_d_neutral)
    image       = warp_decode(feature_3d, x_source, x_d_neutral)

Runs in this project's venv. Loads models once and processes a file or a folder.

    python scripts/lp_neutralize.py --lp-dir ../LivePortrait --in inputs/p/ --out outputs/_neutral --alpha 0.0
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch


def _list_images(path: str) -> list:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    if os.path.isdir(path):
        out = []
        for e in exts:
            out += glob.glob(os.path.join(path, "**", e), recursive=True)
        return sorted(out)
    return [path]


def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)  # CJK-safe
    if img is None:
        raise RuntimeError(f"could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp-dir", required=True, help="path to the cloned LivePortrait repo")
    ap.add_argument("--in", dest="inp", required=True, help="image file or directory")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="expression retain factor: 0=full neutral, 1=unchanged (try 0.0-0.3)")
    args = ap.parse_args()

    # resolve all user paths to absolute BEFORE chdir into the LivePortrait repo
    lp_dir = os.path.abspath(args.lp_dir)
    inp_abs = os.path.abspath(args.inp)
    out_abs = os.path.abspath(args.out)
    sys.path.insert(0, lp_dir)
    os.chdir(lp_dir)  # InferenceConfig uses weight paths relative to the repo root

    from src.config.inference_config import InferenceConfig
    from src.config.crop_config import CropConfig
    from src.live_portrait_wrapper import LivePortraitWrapper
    from src.utils.cropper import Cropper
    from src.utils.camera import get_rotation_matrix
    from src.utils.crop import prepare_paste_back, paste_back

    inf_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
    cropper = Cropper(crop_cfg=crop_cfg)

    os.makedirs(out_abs, exist_ok=True)
    paths = _list_images(inp_abs)
    print(f"[lp_neutralize] {len(paths)} image(s), alpha={args.alpha}")

    for p in paths:
        try:
            img = _imread_rgb(p)
            crop_info = cropper.crop_source_image(img, cropper.crop_cfg)
            if crop_info is None:
                print(f"  skip (no face): {os.path.basename(p)}")
                continue
            I_s = wrapper.prepare_source(crop_info["img_crop_256x256"])
            x_s_info = wrapper.get_kp_info(I_s)
            x_c_s = x_s_info["kp"]
            R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"])
            f_s = wrapper.extract_feature_3d(I_s)
            x_s = wrapper.transform_keypoint(x_s_info)

            delta = x_s_info["exp"] * args.alpha                 # zero (or scale down) expression
            t_new = x_s_info["t"].clone()
            t_new[..., 2].fill_(0)
            x_d_new = x_s_info["scale"] * (x_c_s @ R_s + delta) + t_new
            x_d_new = wrapper.stitching(x_s, x_d_new)

            out = wrapper.warp_decode(f_s, x_s, x_d_new)
            I_p = wrapper.parse_output(out["out"])[0]            # 512x512 RGB crop

            mask = prepare_paste_back(inf_cfg.mask_crop, crop_info["M_c2o"],
                                     dsize=(img.shape[1], img.shape[0]))
            I_full = paste_back(I_p, crop_info["M_c2o"], img, mask)

            stem = os.path.splitext(os.path.basename(p))[0]
            dst = os.path.join(out_abs, stem + ".jpg")
            cv2.imwrite(dst, cv2.cvtColor(I_full, cv2.COLOR_RGB2BGR))
            print(f"  ok: {os.path.basename(p)} -> {dst}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {os.path.basename(p)}: {exc}")


if __name__ == "__main__":
    main()
