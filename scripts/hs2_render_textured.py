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


def load_card_manifest(card_path):
    """The card's extraction manifest (head dir, resolved texture assets, ids)."""
    stem = os.path.splitext(os.path.basename(card_path))[0]
    path = os.path.join(HEAD_DIR, "cards", f"{stem}.json")
    if not os.path.exists(path):
        raise SystemExit(f"no manifest for {stem} — run:\n"
                         f"    .venv/Scripts/python.exe scripts/hs2_extract_head.py --card {card_path}")
    return json.load(open(path, encoding="utf-8"))


def load_skin_material(head_dir, name="cf_m_skin_head_02"):
    """Real skin shading params extracted from the game material (<head_dir>/materials/)."""
    path = os.path.join(head_dir, "materials", f"{name}.json")
    d = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"colors": {}, "floats": {}}
    c, f = d.get("colors", {}), d.get("floats", {})
    pick = lambda k, dv: (c[k][:3] if k in c else dv)
    return {
        "base": pick("_Color", [0.80, 0.70, 0.63]),
        "sss": pick("_ColorTranslucency", [0.80, 0.55, 0.44]),
        "spec": pick("_SpecColor", [0.5, 0.5, 0.5]),
        "smoothness": float(f.get("_Smoothness", 0.65)),
        "rim": float(f.get("_Rim", 0.5)),
        "rim_exp": float(f.get("_RimExp", 1.0)),
    }


def render_head(verts, faces, uv=None, tex=None, yaw=0.0, res=512, mat=None,
                skin=None, light=(0.35, 0.45, 0.82),
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

    n_img, _ = dr.interpolate(n[None], rast, f)            # (1,H,W,3)
    n_img = n_img / (n_img.norm(dim=-1, keepdim=True) + 1e-9)
    n_img = torch.where(n_img[..., 2:3] < 0, -n_img, n_img)   # orient toward camera (view dir +z)

    # Skin shading driven by the GAME's real material params (<head_dir>/materials/).
    if mat is None:
        raise ValueError("mat= (load_skin_material result) is required")
    base = torch.tensor(skin if skin is not None else mat["base"], dtype=torch.float32, device=device)
    if tex is not None and uv is not None:                  # Unity: albedo = _MainTex * _Color
        uvt = torch.as_tensor(np.ascontiguousarray(uv), dtype=torch.float32, device=device)
        uv_img, _ = dr.interpolate(uvt[None], rast, f)     # (1,H,W,2)
        text = torch.as_tensor(tex, dtype=torch.float32, device=device)[None]
        albedo = dr.texture(text, uv_img, filter_mode="linear") * base
    else:
        albedo = base.view(1, 1, 1, 3)

    view = torch.tensor([0.0, 0.0, 1.0], device=device)
    # Soft character-maker-ish rig (2 balanced lights so it isn't blown out).
    rig = [((0.30, 0.30, 0.90), (1.00, 0.97, 0.93), 0.80),   # warm key, front-upper
           ((-0.55, -0.15, 0.60), (0.82, 0.88, 1.00), 0.28)]  # cool fill, lower-side
    sss_col = torch.tensor(mat["sss"], dtype=torch.float32, device=device)      # _ColorTranslucency
    spec_col = torch.tensor(mat["spec"], dtype=torch.float32, device=device)    # _SpecColor
    shininess = 2.0 ** (mat["smoothness"] * 6.0 + 2.0)     # _Smoothness -> Blinn exponent
    diffuse = torch.zeros_like(albedo)
    specular = torch.zeros_like(albedo)
    for d, c, inten in rig:
        Ld = torch.tensor(d, dtype=torch.float32, device=device)
        Ld = Ld / Ld.norm()
        col = torch.tensor(c, dtype=torch.float32, device=device) * inten
        wrap = ((n_img * Ld).sum(-1, keepdim=True) * 0.5 + 0.5).clamp(0, 1)     # half-Lambert
        diffuse = diffuse + col * wrap
        diffuse = diffuse + sss_col * (wrap * (1.0 - wrap)) * inten * 0.5       # subsurface (translucency)
        half = Ld + view
        half = half / half.norm()
        specular = specular + spec_col * inten * (
            (n_img * half).sum(-1, keepdim=True).clamp(0, 1) ** shininess) * 0.35
    ndv = (n_img * view).sum(-1, keepdim=True).clamp(0, 1)
    rim = ((1.0 - ndv) ** (1.0 + mat["rim_exp"])) * mat["rim"] * 0.12          # subtle rim
    ambient_col = torch.tensor([0.42, 0.44, 0.48], device=device) * 0.5
    color = albedo * (diffuse + ambient_col) + specular + rim
    color = color.clamp(0, 1)

    img = color * mask + bg * (1.0 - mask)                 # composite over background
    img = dr.antialias(img, rast, clip, f)                 # silhouette AA
    img = img[0].flip(0).clamp(0, 1)                       # flip: nvdiffrast row0 = NDC y=-1 (bottom)
    return (img.detach().cpu().numpy() * 255).astype(np.uint8)


def load_uv(head_dir):
    npz = np.load(os.path.join(head_dir, "o_head_mesh.npz"))
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
    ap.add_argument("--composite", action="store_true",
                    help="compose _MainTex from the card (skin tone + makeup) instead of the raw skin texture")
    ap.add_argument("--tex-res", type=int, default=2048, help="composition resolution")
    ap.add_argument("--yaws", default="0,-25,25")
    ap.add_argument("--compare", action="store_true",
                    help="also save our front render beside the card's in-game portrait (face-cropped)")
    args = ap.parse_args()

    stem = os.path.splitext(os.path.basename(args.card))[0]
    out_png = args.out or f"outputs/{stem}_head_textured.png"
    os.makedirs("outputs", exist_ok=True)

    man = load_card_manifest(args.card)
    head_dir = os.path.join(HEAD_DIR, man["head_dir"])
    mat = load_skin_material(head_dir, man["head_material"])

    verts, faces = card_to_mesh(args.card)
    save_obj(verts, faces, f"outputs/{stem}_head.obj")
    uv = load_uv(head_dir)

    # Albedo. --composite reproduces the game's runtime texture composition (skin tone recolour +
    # lipstick/eyeshadow/blush/mole); otherwise we use the raw card-selected skin texture, which
    # is the composition's *input* and so carries no makeup.
    print(f"[render] head={man['head_dir']} prefab={man['prefab']} mat={man['head_material']}")
    tex_path = args.texture
    if args.composite and tex_path is None:
        from src.render.composite import FaceCompositor, linear_to_srgb
        comp = FaceCompositor.from_card(args.card, res=args.tex_res,
                                        device="cuda" if torch.cuda.is_available() else "cpu")
        # the compositor returns LINEAR light; this renderer still shades display-referred, so
        # encode back to sRGB here (Phase 3b moves shading into linear and drops this).
        tex = linear_to_srgb(comp.color_pass()).detach().cpu().numpy()[::-1].copy()
        print(f"[render] albedo: composed _MainTex {tex.shape} (makeup applied)")
    else:
        if tex_path is None:
            slot = man["textures"].get("ft_skin_f.MainTex")
            if slot and slot.get("file"):
                tex_path = os.path.join(HEAD_DIR, slot["file"])
        tex = load_texture(tex_path) if tex_path else None
        print(f"[render] albedo: {tex_path or 'flat skin tone'}"
              + (f" {tex.shape}" if tex is not None else ""))

    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    views = [render_head(verts, faces, uv=uv, tex=tex, yaw=y, res=args.res, mat=mat) for y in yaws]
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
        ours = render_head(verts, faces, uv=uv, tex=tex, yaw=0.0, res=args.res, mat=mat)
        portrait = load_face_rgb(args.card, args.res, use_detector=True)  # RGB uint8 aligned face
        gap = np.full((args.res, 12, 3), 255, np.uint8)
        cmp = np.concatenate([portrait, gap, ours], axis=1)
        cmp_png = f"outputs/{stem}_compare.png"
        Image.fromarray(cmp).save(cmp_png)
        print(f"saved compare (game portrait | our render) -> {cmp_png}")


if __name__ == "__main__":
    main()
