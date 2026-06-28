"""Render an HS2 head mesh to an image WITHOUT the game — a tiny pure-numpy software rasterizer.

How it works (no HS2, no Blender): we already have the deformed mesh (verts + triangles) from the offline
constructor. Rendering a mesh is just: project 3D verts to 2D, then for each triangle fill its pixels with
a z-buffer (nearest wins) and shade by the face normal vs a light direction. That's all a "renderer" is.

    .venv/Scripts/python.exe scripts/hs2_render_mesh.py --card tests/HS2ChaF_20240901192905747.png
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.hs2_mesh import card_to_mesh, save_obj


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def render(verts, faces, yaw=0.0, W=380, H=480, light=(0.3, 0.45, 0.85)):
    V = verts @ _rot_y(yaw).T                       # view space
    # orthographic fit
    mn, mx = V.min(0), V.max(0)
    c = (mn + mx) / 2
    scale = 0.82 * min(W / (mx[0] - mn[0]), H / (mx[1] - mn[1]))
    u = (V[:, 0] - c[0]) * scale + W / 2
    v = H / 2 - (V[:, 1] - c[1]) * scale            # flip y
    z = V[:, 2]
    P = np.stack([u, v], 1)

    L = np.array(light, float); L /= np.linalg.norm(L)
    img = np.zeros((H, W), np.float32)
    zbuf = np.full((H, W), -1e18, np.float32)

    tri = faces
    p0, p1, p2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    z0, z1, z2 = z[tri[:, 0]], z[tri[:, 1]], z[tri[:, 2]]
    # face normals (view space) for shading
    e1 = V[tri[:, 1]] - V[tri[:, 0]]
    e2 = V[tri[:, 2]] - V[tri[:, 0]]
    n = np.cross(e1, e2)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    n[n[:, 2] < 0] *= -1                            # face toward camera
    shade = 0.25 + 0.75 * np.clip(n @ L, 0, 1)      # ambient + diffuse

    for k in range(len(tri)):
        a, b, cc = p0[k], p1[k], p2[k]
        minx = max(int(np.floor(min(a[0], b[0], cc[0]))), 0)
        maxx = min(int(np.ceil(max(a[0], b[0], cc[0]))), W - 1)
        miny = max(int(np.floor(min(a[1], b[1], cc[1]))), 0)
        maxy = min(int(np.ceil(max(a[1], b[1], cc[1]))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        xs, ys = np.meshgrid(np.arange(minx, maxx + 1), np.arange(miny, maxy + 1))
        d = (b[1] - cc[1]) * (a[0] - cc[0]) + (cc[0] - b[0]) * (a[1] - cc[1])
        if abs(d) < 1e-9:
            continue
        w0 = ((b[1] - cc[1]) * (xs - cc[0]) + (cc[0] - b[0]) * (ys - cc[1])) / d
        w1 = ((cc[1] - a[1]) * (xs - cc[0]) + (a[0] - cc[0]) * (ys - cc[1])) / d
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        zz = w0 * z0[k] + w1 * z1[k] + w2 * z2[k]
        yy, xx = ys[m], xs[m]
        zsel = zz[m]
        cur = zbuf[yy, xx]
        better = zsel > cur
        yy, xx, zsel = yy[better], xx[better], zsel[better]
        zbuf[yy, xx] = zsel
        img[yy, xx] = shade[k]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.card))[0]
    out_png = args.out or f"outputs/{stem}_head_render.png"
    out_obj = f"outputs/{stem}_head.obj"

    verts, faces = card_to_mesh(args.card)
    save_obj(verts, faces, out_obj)
    views = [render(verts, faces, yaw=y) for y in (0, -30, 30)]
    gap = np.full((views[0].shape[0], 6), 30, np.uint8)
    combo = np.concatenate([views[0], gap, views[1], gap, views[2]], axis=1)
    os.makedirs("outputs", exist_ok=True)
    Image.fromarray(combo).save(out_png)
    print(f"verts={len(verts)} faces={len(faces)}")
    print(f"saved mesh -> {out_obj}")
    print(f"saved render (front, -30, +30 yaw) -> {out_png}")


if __name__ == "__main__":
    main()
