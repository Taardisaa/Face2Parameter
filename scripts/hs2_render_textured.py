"""Differentiable, UV-textured RGB render of the offline HS2 head mesh (nvdiffrast).

Strategy-2 first brick (see docs/offline-renderer-research.md): take the deformed mesh from the
offline geometry pipeline + the extracted UVs, and render it with nvdiffrast so the whole
`geometry -> image` path is differentiable (gradients flow to verts/texture/lighting; param-space
gradients additionally need the numpy deform ported to torch — a later brick).

v1 skin is APPROXIMATE (user-approved): a flat skin tone (× an optional albedo texture via UV) with
smooth-normal Lambert + ambient. The game's real base skin diffuse (`_MainTex`) is external/runtime-
composed and is a later brick; here we mainly validate that the recipe-faithful *geometry* reads like
the game face.

    .venv/Scripts/python.exe scripts/hs2_render_textured.py --card tests/HS2ChaF_20240901192905747.png

Requires nvdiffrast (see requirements.txt). First run compiles its CUDA plugin (needs an MSVC env);
after that it runs from the cached plugin with the plain venv python.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.hs2_mesh import card_to_mesh, save_obj

HEAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hs2_head")
_GLCTX = None


def _glctx():
    global _GLCTX
    if _GLCTX is None:
        import nvdiffrast.torch as dr
        _GLCTX = dr.RasterizeCudaContext()
    return _GLCTX


def _rot_y(deg, device):
    a = np.radians(deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32, device=device)


def vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Smooth per-vertex normals (area-weighted face-normal accumulation). v:(V,3) f:(F,3) long."""
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)  # (F,3), area-weighted
    n = torch.zeros_like(v)
    n.index_add_(0, f[:, 0], fn)
    n.index_add_(0, f[:, 1], fn)
    n.index_add_(0, f[:, 2], fn)
    return n / (n.norm(dim=1, keepdim=True) + 1e-9)


def _to_clip(v_view: torch.Tensor, margin: float = 0.85) -> torch.Tensor:
    """View-space verts -> clip space for a square viewport, aspect-preserving orthographic fit.

    Returns (V,4) with w=1. NDC z is mapped so nearer (larger view-z) -> smaller NDC z (OpenGL near).
    """
    mn, mx = v_view.min(0).values, v_view.max(0).values
    c = (mn + mx) / 2
    ext = (mx - mn)
    s = 2.0 * margin / torch.max(ext[0], ext[1])           # uniform -> preserves aspect (square view)
    x = (v_view[:, 0] - c[0]) * s
    y = (v_view[:, 1] - c[1]) * s
    z = (v_view[:, 2] - c[2]) * s
    z = -z / (z.abs().max() + 1e-6)                         # nearer -> -1 (OpenGL near plane)
    return torch.stack([x, y, z, torch.ones_like(x)], dim=1)


def render_head(verts, faces, uv=None, tex=None, yaw=0.0, res=512,
                skin=(0.90, 0.75, 0.66), light=(0.35, 0.45, 0.82),
                ambient=0.38, bg=1.0, device="cuda"):
    """Render (verts,faces) to an (res,res,3) uint8 RGB image with nvdiffrast.

    Differentiable end-to-end (this returns a numpy image, but the internal torch graph w.r.t.
    verts/tex/light is intact if you keep the tensors). uv:(V,2), tex:(th,tw,3) in [0,1] optional.
    """
    import nvdiffrast.torch as dr
    v = torch.as_tensor(np.ascontiguousarray(verts), dtype=torch.float32, device=device)
    f = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.int32, device=device)
    fl = f.long()

    v_view = v @ _rot_y(yaw, device).T
    n = vertex_normals(v_view, fl)                          # normals in the same (rotated) view space
    clip = _to_clip(v_view)[None]                           # (1,V,4)

    rast, _ = dr.rasterize(_glctx(), clip, f, resolution=[res, res])
    mask = (rast[..., 3:4] > 0).float()                    # (1,H,W,1)

    # smooth-normal Lambert + ambient (two-sided so back-facing interpolated normals don't go black)
    n_img, _ = dr.interpolate(n[None], rast, f)            # (1,H,W,3)
    n_img = n_img / (n_img.norm(dim=-1, keepdim=True) + 1e-9)
    L = torch.tensor(light, dtype=torch.float32, device=device)
    L = L / L.norm()
    ndl = (n_img * L).sum(-1, keepdim=True).abs()          # two-sided
    shade = ambient + (1.0 - ambient) * ndl.clamp(0, 1)    # (1,H,W,1)

    if tex is not None and uv is not None:
        uvt = torch.as_tensor(np.ascontiguousarray(uv), dtype=torch.float32, device=device)
        uv_img, _ = dr.interpolate(uvt[None], rast, f)     # (1,H,W,2)
        text = torch.as_tensor(tex, dtype=torch.float32, device=device)[None]  # (1,th,tw,3)
        albedo = dr.texture(text, uv_img, filter_mode="linear")               # (1,H,W,3)
    else:
        albedo = torch.tensor(skin, dtype=torch.float32, device=device).view(1, 1, 1, 3)

    color = albedo * shade
    img = color * mask + bg * (1.0 - mask)                 # composite over background
    img = dr.antialias(img, rast, clip, f)                 # silhouette AA
    img = img[0].flip(0).clamp(0, 1)                       # flip: nvdiffrast row0 = NDC y=-1 (bottom)
    return (img.detach().cpu().numpy() * 255).astype(np.uint8)


def load_uv():
    npz = np.load(os.path.join(HEAD_DIR, "o_head_mesh.npz"))
    return npz["uv"] if "uv" in npz.files else None


def load_texture(path, flip_v=True):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    if flip_v:                                             # Unity UV origin is bottom-left
        arr = arr[::-1].copy()
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--out", default=None)
    ap.add_argument("--texture", default=None,
                    help="albedo texture PNG to map via UV (default: flat skin tone)")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--yaws", default="0,-25,25")
    ap.add_argument("--compare", action="store_true",
                    help="also save our front render beside the card's in-game portrait (face-cropped)")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.card))[0]
    out_png = args.out or f"outputs/{stem}_head_textured.png"
    os.makedirs("outputs", exist_ok=True)

    verts, faces = card_to_mesh(args.card)
    save_obj(verts, faces, f"outputs/{stem}_head.obj")
    uv = load_uv()
    tex = load_texture(args.texture) if args.texture else None
    if args.texture:
        print(f"[render] albedo texture: {args.texture} {tex.shape}")
    else:
        print("[render] flat skin tone (approx skin; base diffuse is a later brick)")

    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    views = [render_head(verts, faces, uv=uv, tex=tex, yaw=y, res=args.res) for y in yaws]
    gap = np.full((views[0].shape[0], 8, 3), 255, np.uint8)
    combo = np.concatenate([x for v in views for x in (v, gap)][:-1], axis=1)
    Image.fromarray(combo).save(out_png)
    print(f"verts={len(verts)} faces={len(faces)} uv={'yes' if uv is not None else 'no'}")
    print(f"saved render (yaws={yaws}) -> {out_png}")

    if args.compare:
        # The card PNG's visible image IS an in-game portrait; face-crop it (reuse load_face_rgb)
        # and place our front render beside it. Rough eyeball: our render lacks eyes/skin/hair, so
        # this compares FACE SHAPE / proportions, not appearance.
        from src.img_utils import load_face_rgb
        ours = render_head(verts, faces, uv=uv, tex=tex, yaw=0.0, res=args.res)
        portrait = load_face_rgb(args.card, args.res, use_detector=True)  # RGB uint8 aligned face
        gap = np.full((args.res, 12, 3), 255, np.uint8)
        cmp = np.concatenate([portrait, gap, ours], axis=1)
        cmp_png = f"outputs/{stem}_compare.png"
        Image.fromarray(cmp).save(cmp_png)
        print(f"saved compare (game portrait | our render) -> {cmp_png}")


if __name__ == "__main__":
    main()
