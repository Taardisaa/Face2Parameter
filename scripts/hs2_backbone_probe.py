"""Which feature space actually separates HS2 game renders?

Follow-up to the reward ablation (docs/beauty-guided-generation.md). That measurement showed the
beauty scorer is noise on renders, and traced it to the feature space rather than the head: on
DINOv2 ViT-S/14 a 3-degree yaw moves a render almost as far as swapping in a *different
character*. Any reward built on such a space inherits that.

Before spending money labelling renders, settle WHICH space to label in — labels collected on a
backbone that cannot tell these characters apart are wasted.

Two metrics, both parameter-free:

  * noise/signal = mean(1-cos) between a character and its own 3-degree render, over
    mean(1-cos) between different characters. <1 is necessary; the smaller the better. A ratio
    near 1 means the representation cannot distinguish "same face, slightly turned" from
    "different face at all".
  * rank-1 retrieval = given the 3-degree render, is the nearest neighbour among the 0-degree
    renders the same character? This is the blunt version of the same question and needs no
    threshold. A backbone below ~90% here has no business carrying a reward.

Uses the image pairs already captured by hs2_baldness_ablation.py --control (game renders, hair
on both sides, 3-degree yaw the only difference), so it measures the backbone and nothing else.

    .venv/Scripts/python.exe scripts/hs2_backbone_probe.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "hs2_gt", "_baldness")


def pairs(suffix="__ctrl"):
    out = []
    for a in sorted(glob.glob(os.path.join(OUT_DIR, f"*__hair{suffix}.png"))):
        b = a.replace(f"__hair{suffix}.png", f"__bald{suffix}.png")
        if os.path.exists(b):
            out.append((a, b))
    return out


# ---------------------------------------------------------------- backbones
def feats_dinov2(paths, variant="dinov2_vits14"):
    import torch
    from beauty_score import BeautyScorer
    sc = BeautyScorer()                      # reuse its exact preprocessing
    dev = sc.device
    model = torch.hub.load("facebookresearch/dinov2", variant).to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for p in paths:
        img = sc._to_face_rgb(p, False)      # (H,W,3) float [0,1], same crop for every backbone
        x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(dev)
        with torch.no_grad():
            out.append(model((x - mean) / std).squeeze(0).cpu().numpy())
    return np.stack(out)


def feats_arcface(paths):
    """ArcFace on its own 5-point aligned 112x112 crop — the preprocessing it was trained with."""
    import cv2
    from src.models.arcface import ArcFaceONNX, align_112
    from src.face_data_utils.FaceCrop import FaceCrop              # importing it sets cv2.cv2

    rec = ArcFaceONNX()
    det = FaceCrop().detector
    out, missed = [], 0
    for p in paths:
        bgr = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        aligned = None
        try:
            lm = det.detect_faces_raw(rgb)[1]
            if lm.size == 10:
                aligned = align_112(rgb, lm.reshape(2, 5).T.astype(np.float32))
        except Exception:
            pass
        if aligned is None:
            missed += 1
            aligned = cv2.resize(rgb, (112, 112))                  # honest fallback, counted
        out.append(rec.embed(aligned)[0])
    if missed:
        print(f"    (mtcnn missed {missed}/{len(paths)}; those used a plain resize)")
    return np.stack(out)


# ---------------------------------------------------------------- metrics
def report(name, A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)

    noise = 1.0 - (A * B).sum(1)                       # same character, 3-degree apart
    off = A @ A.T
    iu = np.triu_indices(len(A), 1)
    signal = 1.0 - off[iu]                             # different characters

    sim = A @ B.T                                      # (i=yaw0, j=yaw3)
    rank1 = (sim.argmax(0) == np.arange(len(B))).mean()

    print(f"  {name:22s} dim={A.shape[1]:4d}  "
          f"noise={noise.mean():.4f}  signal={signal.mean():.4f}  "
          f"ratio={noise.mean() / max(signal.mean(), 1e-9):.3f}  rank-1={rank1 * 100:5.1f}%")
    return {"name": name, "ratio": float(noise.mean() / max(signal.mean(), 1e-9)),
            "rank1": float(rank1)}


def real_photo_control(n=40):
    """Same measurement on REAL photographs, so a bad score can be blamed on the right thing.

    Without this, "ArcFace scores 17% rank-1 on renders" is unreadable: it could equally mean the
    alignment code here is wrong. Rotating a real photo by the same 3 degrees is the positive
    control — if a face model cannot ace THAT, the probe is broken, not the domain.
    """
    import cv2
    imgs = (sorted(glob.glob("face2beauty/data/images/*.jpg"))
            or sorted(glob.glob("face2beauty/data/images/*.png")))[:n]
    if not imgs:
        return []

    def rot3(p):
        img = cv2.cvtColor(cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
        out = os.path.join(OUT_DIR, "_ctrl_real", os.path.basename(p))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, cv2.cvtColor(cv2.warpAffine(img, m, (w, h), borderValue=(127, 127, 127)),
                                      cv2.COLOR_RGB2BGR))
        return out

    rotated = [rot3(p) for p in imgs]
    print(f"\n  --- positive control: {len(imgs)} REAL photographs, same 3-degree perturbation ---")
    rows = [report("dinov2_vits14 (real)", feats_dinov2(imgs), feats_dinov2(rotated))]
    try:
        rows.append(report("arcface (real)", feats_arcface(imgs), feats_arcface(rotated)))
    except Exception as e:
        print(f"  arcface (real) FAILED: {str(e)[:90]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-arcface", action="store_true")
    ap.add_argument("--no-control", action="store_true", help="skip the real-photo control")
    args = ap.parse_args()

    ps = pairs()
    if not ps:
        raise SystemExit("no control pairs — run hs2_baldness_ablation.py --control first")
    a_paths = [a for a, _ in ps]
    b_paths = [b for _, b in ps]
    print(f"[probe] {len(ps)} game-render pairs (same character, 3-degree yaw)\n")
    print("  lower ratio = better; rank-1 is the blunt check\n")

    rows = []
    for variant in ("dinov2_vits14", "dinov2_vitb14"):
        try:
            rows.append(report(variant, feats_dinov2(a_paths, variant),
                               feats_dinov2(b_paths, variant)))
        except Exception as e:
            print(f"  {variant:22s} FAILED: {str(e)[:90]}")
    if not args.skip_arcface:
        try:
            rows.append(report("arcface_w600k_r50", feats_arcface(a_paths), feats_arcface(b_paths)))
        except Exception as e:
            print(f"  {'arcface_w600k_r50':22s} FAILED: {str(e)[:90]}")

    ctrl = [] if args.no_control else real_photo_control()

    if rows:
        best = min(rows, key=lambda r: r["ratio"])
        print(f"\n  best on renders: {best['name']} (ratio {best['ratio']:.3f}, "
              f"rank-1 {best['rank1'] * 100:.1f}%)")
    if ctrl and rows:
        # The interesting column is `signal`, not `ratio`. If a backbone separates real people
        # fine but not these characters, the deficit is in the CHARACTERS -- they share one base
        # mesh and differ only by sliders -- not in the backbone or in our renderer.
        print("\n  Compare the SIGNAL column across the two blocks. A backbone that separates real\n"
              "  people but not these characters is telling you the characters are near-identical,\n"
              "  which no amount of backbone-swapping or renderer fidelity will change.")


if __name__ == "__main__":
    main()
