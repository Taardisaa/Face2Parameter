"""Driver-free expression neutralization via SMIRK (3DMM analysis-by-neural-synthesis).

Encodes a face into FLAME params, scales the expression + jaw toward zero (alpha=0 => neutral),
re-renders the neutral geometry, and uses SMIRK's neural generator to composite it with the
original appearance into a *photoreal* neutral face (not a CG mesh). Unlike LivePortrait's
keypoint warp, the 3DMM models cheek/jaw expression explicitly, so it can flatten a smile's
cheek volume that LivePortrait leaves behind.

Runs inside the WSL py3.10 env (torch2.0.1+cu118 + pytorch3d). Modeled on smirk/demo.py.

    python scripts/smirk_neutralize.py --smirk-dir /mnt/c/.../smirk --in <dir> --out <dir> --alpha 0.0
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import estimate_transform, warp

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def list_images(path):
    if os.path.isdir(path):
        out = []
        for e in IMAGE_EXTS:
            out += glob.glob(os.path.join(path, "**", e), recursive=True)
        return sorted(out)
    return [path]


def imread_bgr(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)  # CJK-safe
    if img is None:
        raise RuntimeError(f"could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[:, :, :3]


def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left, right = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
    top, bottom = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    size = int(old_size * scale)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                        [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    return estimate_transform("similarity", src_pts, DST_PTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smirk-dir", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.0, help="expression retain: 0=neutral, 1=unchanged")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    smirk_dir = os.path.abspath(args.smirk_dir)
    inp = os.path.abspath(args.inp)
    out = os.path.abspath(args.out)
    ckpt = args.checkpoint or os.path.join(smirk_dir, "pretrained_models", "SMIRK_em1.pt")
    sys.path.insert(0, smirk_dir)
    os.chdir(smirk_dir)  # FLAME.py + assets use repo-relative paths

    from src.smirk_encoder import SmirkEncoder
    from src.smirk_generator import SmirkGenerator
    from src.FLAME.FLAME import FLAME
    from src.renderer.renderer import Renderer
    import src.utils.masking as masking_utils
    from utils.mediapipe_utils import run_mediapipe
    from datasets.base_dataset import create_mask

    device = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location="cpu")
    enc = SmirkEncoder().to(device)
    enc.load_state_dict({k.replace("smirk_encoder.", ""): v for k, v in ck.items() if "smirk_encoder" in k})
    enc.eval()
    gen = SmirkGenerator(in_channels=6, out_channels=3, init_features=32, res_blocks=5).to(device)
    gen.load_state_dict({k.replace("smirk_generator.", ""): v for k, v in ck.items() if "smirk_generator" in k})
    gen.eval()
    flame = FLAME().to(device)
    renderer = Renderer().to(device)

    os.makedirs(out, exist_ok=True)
    paths = list_images(inp)
    print(f"[smirk_neutralize] {len(paths)} image(s), alpha={args.alpha}, device={device}")

    for p in paths:
        try:
            image = imread_bgr(p)
            kpt = run_mediapipe(image)
            if kpt is None:
                print(f"  skip (no face): {os.path.basename(p)}")
                continue
            kpt = kpt[..., :2]
            tform = crop_face(image, kpt, scale=1.4, image_size=224)
            cropped = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T[:, :2]

            ct = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            ct = torch.tensor(ct).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)

            with torch.no_grad():
                outputs = enc(ct)
                # neutralize: scale expression + jaw toward zero (keep shape/pose/cam/eyelids)
                outputs["expression_params"] = outputs["expression_params"] * args.alpha
                outputs["jaw_params"] = outputs["jaw_params"] * args.alpha
                flame_output = flame.forward(outputs)
                rend = renderer.forward(flame_output["vertices"], outputs["cam"],
                                        landmarks_fan=flame_output["landmarks_fan"],
                                        landmarks_mp=flame_output["landmarks_mp"])
                rendered_img = rend["rendered_img"]

                # --- neural generator: composite neutral render with original appearance ---
                mask_ratio_mul, mask_ratio, mask_dilation_radius = 5, 0.01, 10
                hull_mask = create_mask(cropped_kpt, (224, 224))
                face_probs = masking_utils.load_probabilities_per_FLAME_triangle()
                rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
                npoints, _ = masking_utils.mesh_based_mask_uniform_faces(
                    rend["transformed_vertices"], flame_faces=flame.faces_tensor,
                    face_probabilities=face_probs, mask_ratio=mask_ratio * mask_ratio_mul)
                pmask = torch.zeros_like(rendered_mask)
                rsing = torch.randint(0, 2, (npoints.size(0),)).to(npoints.device) * 2 - 1
                rscale = torch.rand((npoints.size(0),)).to(npoints.device) * (mask_ratio_mul - 1) + 1
                rbound = (npoints.size(1) * (1 / mask_ratio_mul) * (rscale ** rsing)).long()
                for bi in range(npoints.size(0)):
                    pmask[bi, :, npoints[bi, :rbound[bi], 1], npoints[bi, :rbound[bi], 0]] = 1
                hull_mask_t = torch.from_numpy(hull_mask).float().unsqueeze(0).to(device)
                extra = ct * pmask
                masked_img = masking_utils.masking(ct, hull_mask_t, extra, mask_dilation_radius,
                                                   rendered_mask=rendered_mask)
                recon = gen(torch.cat([rendered_img, masked_img], dim=1))

            outimg = (recon.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            stem = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(out, stem + ".jpg"), cv2.cvtColor(outimg, cv2.COLOR_RGB2BGR))
            print(f"  ok: {os.path.basename(p)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {os.path.basename(p)}: {exc}")


if __name__ == "__main__":
    main()
