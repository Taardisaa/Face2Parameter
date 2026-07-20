"""Side-by-side: the game's own render vs ours, at matching yaws.

The single most useful picture for judging the offline renderer, and previously assembled by
hand — which meant it could not be regenerated after a fix without redoing the fiddly bits.

Rows are yaws; each row is [game | ours]. The game captures come from
`scripts/hs2_capture_gt.py` (data/hs2_gt/<stem>/bald_yaw*.png), so this needs no running game.

Framing is NOT pixel-matched: the bridge frames on `SkinnedMeshRenderer.bounds`, which returns
conservative whole-character bounds for a skinned mesh, so the game shots are wider and include
shoulders. Read this for shape and shading, not for alignment — a per-pixel comparison has to
wait for the BakeMesh fix in MakerRenderService.

    .venv/Scripts/python.exe scripts/hs2_gt_vs_ours.py --card tests/HS2ChaF_20240901192905747.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fit(img: np.ndarray, size: int) -> np.ndarray:
    """Letterbox onto a white square so rows line up regardless of source aspect/size."""
    if img.shape[2] == 4:                       # composite RGBA over white
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = np.asarray(Image.fromarray(img).resize((max(1, int(w * s)), max(1, int(h * s))),
                                               Image.LANCZOS))
    out = np.full((size, size, 3), 255, np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--yaws", default="0,25,-25")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="outputs/gt_vs_ours.png")
    args = ap.parse_args()

    from src.render.scene import HeadScene, render

    stem = os.path.splitext(os.path.basename(args.card))[0]
    gt_dir = os.path.join(ROOT, "data", "hs2_gt", stem)
    scene = HeadScene(args.card)
    meshes = scene.deform()
    print(f"[scene] {os.path.basename(scene.head_dir)}  submeshes={scene.present}")

    rows, labels = [], []
    for y in [float(v) for v in args.yaws.split(",") if v.strip()]:
        gt_path = os.path.join(gt_dir, f"bald_yaw{int(y):+04d}.png")
        if not os.path.exists(gt_path):
            print(f"  yaw {y:+.0f}: no game capture at {gt_path} — skipped")
            continue
        gt = fit(np.asarray(Image.open(gt_path).convert("RGB")), args.size)
        ours = render(scene, meshes, yaw=y, res=args.size).detach().cpu().numpy()
        ours = fit((np.clip(ours, 0, 1) * 255).astype(np.uint8), args.size)
        gap = np.full((args.size, 10, 3), 255, np.uint8)
        rows.append(np.concatenate([gt, gap, ours], axis=1))
        labels.append(y)
        print(f"  yaw {y:+.0f}: game | ours")

    if not rows:
        raise SystemExit(f"no game captures under {gt_dir} — run scripts/hs2_capture_gt.py first")

    hgap = np.full((10, rows[0].shape[1], 3), 255, np.uint8)
    combo = np.concatenate([x for r in rows for x in (r, hgap)][:-1], axis=0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(combo).save(args.out)
    print(f"\nsaved (left=game, right=ours; rows are yaw {labels}) -> {args.out}")


if __name__ == "__main__":
    main()
