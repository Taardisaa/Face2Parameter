"""Inference: image -> 205 face params -> game character card.

Refactors ``extractor.py`` to the new backbone+head model. The 205-dim vector is
written back into a real character card via the kept ``FaceData`` contract.

Requires the real-data path: a trained head checkpoint, the DINOv2 backbone (one-time
download), plus ``HS2ABMX.exe`` and the stat JSONs under ``src/face_data_utils/`` (used
by ``FaceData``) and ``mtcnn_ort`` (used by ``FaceCrop``). Not exercised by the skeleton
smoke test.

Usage:
    .venv/Scripts/python.exe infer.py --config dinov2_vits14 \
        --head exp/dinov2_vits14_head/weights/head_epoch_30_step_XXXX.pth \
        --image test/my.png --template test/template.png --out outputs/
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import torch

from config import get_config
from src.models.face2param import Face2Param


def read_face(path: str, img_size: int, use_face_detector: bool, device) -> torch.Tensor:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if use_face_detector:
        from src.face_data_utils.FaceCrop import FaceCrop  # lazy: needs mtcnn_ort
        faces = FaceCrop().crop(img)
        if len(faces) == 0:
            raise RuntimeError("No face detected in the image")
        if len(faces) > 1:
            raise RuntimeError("More than one face detected in the image")
        img = faces[0]
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    chw = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    return torch.from_numpy(chw).float().div(255).unsqueeze(0).to(device), img


def infer(cfg, head_path: str, image_path: str, template_path: str,
          out_dir: str = "outputs", use_face_detector: bool = True) -> list:
    from src.face_data_utils.utils import FaceData  # lazy: needs .exe + stat JSONs

    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = Face2Param.from_checkpoint(cfg, head_path, map_location="cpu").to(device).eval()

    face_tensor, face_img = read_face(image_path, cfg.img_size, use_face_detector, device)
    with torch.no_grad():
        vector = model(face_tensor).squeeze(0).cpu().numpy()

    face_data = FaceData(template_path)
    face_data.set_from_vector(vector, is_simplify=True, without_right=True,
                              denormalize=True, use_gaussian=False)
    face_data.set_image(face_img.astype(np.uint8))

    name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(out_dir, name + ".png")
    face_data.save(save_path)
    print(f"[infer] wrote card -> {save_path}")
    return vector.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov2_vits14")
    ap.add_argument("--head", required=True, help="trained head weights (.pth)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--template", required=True, help="character card to write into")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-detector", action="store_true")
    args = ap.parse_args()
    infer(get_config(args.config), args.head, args.image, args.template,
          out_dir=args.out, use_face_detector=not args.no_detector)


if __name__ == "__main__":
    main()
