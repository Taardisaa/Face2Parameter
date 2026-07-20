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
    # A near-frontal pair says little: the silhouette is where geometry error shows, so sweep out
    # to profile. Laid out as game-row over ours-row so one angle is directly above its twin.
    ap.add_argument("--yaws", default="-90,-60,-30,0,30,60,90")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--out", default="outputs/gt_vs_ours.png")
    ap.add_argument("--capture", action="store_true",
                    help="re-shoot the game side first (needs the game running); uses the clean "
                         "protocol — hide_hair + freeze_pose + settle, with a discarded warm-up")
    args = ap.parse_args()

    from src.render.scene import HeadScene, render

    stem = os.path.splitext(os.path.basename(args.card))[0]
    gt_dir = os.path.join(ROOT, "data", "hs2_gt", stem)
    scene = HeadScene(args.card)
    meshes = scene.deform()
    print(f"[scene] {os.path.basename(scene.head_dir)}  submeshes={scene.present}")

    yaws = [float(v) for v in args.yaws.split(",") if v.strip()]

    if args.capture:
        from scripts.hs2_capture_gt import call, aligned_framing
        os.makedirs(gt_dir, exist_ok=True)
        call("/maker/card/load", "POST", {"path": os.path.abspath(args.card)}, timeout=180)
        # Discarded warm-up: the character keeps settling well past what one render can wait for.
        call(f"/maker/render?w=640&h=640&yaw=0&hide_hair=1&out={gt_dir}/_warm.png", timeout=90)
        # Without this the game's capture is 27% larger than ours with the eye line 0.055 off,
        # so any difference we then attribute to shading is partly just a different camera.
        frame = aligned_framing()
        for y in yaws:
            p = os.path.join(gt_dir, f"bald_yaw{int(y):+04d}_p{int(args.pitch):+03d}.png")
            call(f"/maker/render?w=640&h=640&yaw={y}&pitch={args.pitch}&hide_hair=1&{frame}&out={p}",
                 timeout=90)
        print(f"  captured {len(yaws)} game views (camera aligned to ours)")

    top, bot, labels = [], [], []
    for y in yaws:
        gt_path = os.path.join(gt_dir, f"bald_yaw{int(y):+04d}_p{int(args.pitch):+03d}.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(gt_dir, f"bald_yaw{int(y):+04d}.png")   # older captures
        if not os.path.exists(gt_path):
            print(f"  yaw {y:+.0f}: no game capture — skipped (use --capture)")
            continue
        top.append(fit(np.asarray(Image.open(gt_path).convert("RGB")), args.size))
        ours = render(scene, meshes, yaw=y, pitch=args.pitch, res=args.size).detach().cpu().numpy()
        bot.append(fit((np.clip(ours, 0, 1) * 255).astype(np.uint8), args.size))
        labels.append(y)

    if not top:
        raise SystemExit(f"no game captures under {gt_dir} — re-run with --capture")

    gap = np.full((args.size, 6, 3), 255, np.uint8)
    row_g = np.concatenate([x for v in top for x in (v, gap)][:-1], axis=1)
    row_o = np.concatenate([x for v in bot for x in (v, gap)][:-1], axis=1)
    hgap = np.full((10, row_g.shape[1], 3), 255, np.uint8)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Image.fromarray(np.concatenate([row_g, hgap, row_o], axis=0)).save(args.out)
    print(f"\nsaved (top=game, bottom=ours; yaws {labels}, pitch {args.pitch:+.0f}) -> {args.out}")


if __name__ == "__main__":
    main()
