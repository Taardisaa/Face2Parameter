"""Differentiable multi-submesh render of a complete HS2 head (nvdiffrast).

The face is not one mesh: skin, eyeballs, lashes, tear film, eyeshadow, teeth and tongue are
separate skinned submeshes with different shaders, render queues and blend modes. Rendering them
correctly means reproducing the game's DRAW ORDER, which was read off the shader assets
(docs/hs2-shader-recipe.md §4), not guessed:

    opaque  (AIT/Skin True Face   AlphaTest-40, ZWrite on, Cull Back)
            (AIT/Eye Translucency Geometry,     ZWrite on)
            (AIT/Skin Translucency simple — teeth / tongue)
      then  lashes  (Transparent+0, Blend One OneMinusSrcAlpha = PREMULTIPLIED, Cull Off)
      then  eyeshadow (AlphaTest+0, Blend SrcAlpha Zero)
      then  tear film (namida; GrabPass refraction approximated as plain alpha)

Everything runs on the shared bone_world of `TorchHeadRig`, so the whole image stays
differentiable w.r.t. the 205 card parameters as well as textures/lighting.

    .venv/Scripts/python.exe -m src.render.scene --card tests/HS2ChaF_20240901192905747.png
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from ..hs2_deform_torch import TorchHeadRig
from ..hs2_mesh_deform import HeadRig
from .composite import FaceCompositor, linear_to_srgb
from .shading import MATERIAL_KIND, shade

_GLCTX = None


def glctx():
    global _GLCTX
    if _GLCTX is None:
        import nvdiffrast.torch as dr
        _GLCTX = dr.RasterizeCudaContext()
    return _GLCTX


# draw groups, in the order the game renders them (see module docstring)
OPAQUE = ["o_head", "o_eyebase_L", "o_eyebase_R", "o_tooth", "o_tang"]
# Blend modes read off the shader assets' rtBlend0 (Unity BlendMode enum: 1=One, 2=DstColor,
# 10=OneMinusSrcAlpha). NB eyeshadow is srcBlend=2 = DstColor -> a MULTIPLY, not a replace.
BLENDED = [("o_eyelashes", "premultiplied")]

# Two overlays are deliberately NOT drawn, because our port of each is demonstrably wrong and a
# wrong layer is worse than a missing one. Both are visible as a dark polygonal patch over the
# cheeks that the game's own render plainly does not have.
#
#  o_namida   — the tear film (`AIT/main namida`). It uses a GrabPass refraction we never
#               decompiled, so it was standing in as flat 25% alpha over a large mesh.
#  o_eyeshadow— the eyelid's cast shadow (`c_m_eyekage`), NOT the makeup eyeshadow (makeup is
#               composited into _MainTex). Its mesh spans the whole 0..1 UV range, its texture
#               alpha averages 0.74 there and `_Color` is (0.2,0.2,0.2,1), so the shader's
#               `dst *= lerp(1,_Color,pow(a,_ShadowScale))` darkens both cheeks. Something still
#               gates it that the decompile has not revealed.
DEFERRED = [("o_namida", "alpha"), ("o_eyeshadow", "multiply")]

# `o_eyeshadow` (material c_m_eyekage) is the eyelid's cast shadow, NOT the makeup eyeshadow —
# makeup is composited into _MainTex (see composite.py). It is disabled because our port of it is
# demonstrably wrong and a wrong layer is worse than a missing one: the mesh spans the whole 0..1
# UV range, its texture `c_t_eyeshadow_01` has alpha mean 0.74 across that range, and `_Color` is
# (0.2,0.2,0.2,1) — so `dst *= lerp(1,_Color,pow(a,_ShadowScale))` darkens both cheeks, which the
# game's own render clearly does not do. Something still gates it (mesh extent, a runtime _Color
# override, or inverted alpha semantics) that the decompile has not told us yet.



def vertex_normals(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """Smooth per-vertex normals (area-weighted face-normal accumulation). v:(V,3) f:(F,3)."""
    fn = torch.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]], dim=1)
    n = torch.zeros_like(v)
    n.index_add_(0, f[:, 0], fn)
    n.index_add_(0, f[:, 1], fn)
    n.index_add_(0, f[:, 2], fn)
    return n / (n.norm(dim=1, keepdim=True) + 1e-9)


class HeadScene:
    """A card's full head: deformed submeshes + their materials, ready to rasterize."""

    def __init__(self, card_path: str, device="cuda", dtype=torch.float32, tex_res=2048):
        from ..hs2_mesh import card_head_id
        self.card = card_path
        self.device, self.dtype = torch.device(device), dtype
        self.rig = HeadRig(card_head_id(card_path))
        self.trig = TorchHeadRig(self.rig, device=device, dtype=dtype)
        self.comp = FaceCompositor.from_card(card_path, device=device, dtype=dtype, res=tex_res)
        self.man = self.comp.man
        self.head_dir = self.rig.data_dir

        with open(os.path.join(self.head_dir, "submeshes", "manifest.json"), encoding="utf-8") as f:
            self.manifest = {e["mesh"]: e for e in json.load(f)}
        self.mats = {}
        for e in self.manifest.values():
            if e["material"]:
                p = os.path.join(self.head_dir, "materials", f"{e['material']}.json")
                if os.path.exists(p):
                    self.mats[e["mesh"]] = json.load(open(p, encoding="utf-8"))
        # the head's RUNTIME material is the list's MatData, not the one baked into the prefab
        p = os.path.join(self.head_dir, "materials", f"{self.man['head_material']}.json")
        if os.path.exists(p):
            self.mats["o_head"] = json.load(open(p, encoding="utf-8"))

        self._albedo: dict[str, torch.Tensor] = {}
        self.present = [m for m in OPAQUE + [b[0] for b in BLENDED] if self._has(m)]

    def _has(self, name):
        if name == "o_head":
            return True
        return os.path.exists(os.path.join(self.head_dir, "submeshes", f"{name}.npz"))

    # ---------------------------------------------------------------- geometry
    def deform(self, shape_face=None, ab=None):
        """-> {mesh: (verts (V,3), faces, uv, normals)} for every present submesh, batch 1."""
        if shape_face is None:
            shape_face, ab = self.trig.from_card(self.card)
        world = self.trig.bone_world(shape_face, ab)
        out = {}
        for name in self.present:
            if name == "o_head":
                skin = world.index_select(1, self.trig.skin_bone) @ self.trig.bindpose
                M = skin[:, self.trig.bone_idx.reshape(-1)].view(
                    world.shape[0], *self.trig.bone_idx.shape, 4, 4)
                tv = torch.einsum("bvkij,vj->bvki", M, self.trig.verts_h)[..., :3]
                v = (self.trig.bone_w.unsqueeze(0).unsqueeze(-1) * tv).sum(dim=2)
                # Cached rather than re-read per call. (Modest: profiling showed the real cost of
                # an argument-less deform() was `from_card` shelling out to HS2ABMX.exe to
                # re-parse the card — 121 ms of the 122. Pass shape_face/ab in, as an optimisation
                # loop does, and a deform is 8 ms.)
                if not hasattr(self, "_head_attrs"):
                    d = np.load(os.path.join(self.head_dir, "o_head_mesh.npz"))
                    self._head_attrs = (d["uv1"], d["colors"])
                uv1, vcol = self._head_attrs
                faces, uv = self.rig.faces, self.rig.uv
            else:
                sub = self.trig.load_submesh(name)
                v = self.trig.skin(sub, world)
                faces, uv, uv1, vcol = sub["faces"], sub["uv"], None, None
            out[name] = {"verts": v[0], "faces": faces, "uv": uv, "uv1": uv1, "vcol": vcol}
        return out

    # ---------------------------------------------------------------- appearance
    def albedo(self, name: str) -> torch.Tensor:
        """The RGBA texture this submesh samples, assembled per its shader (see shading.py)."""
        if name not in self._albedo:
            from .shading import build_albedo
            self._albedo[name] = build_albedo(self, name)
        return self._albedo[name]


def _to_clip(v_view: torch.Tensor, margin=0.85, center=None, extent=None):
    """View-space -> clip space, aspect-preserving orthographic fit over a FIXED framing.

    `center`/`extent` are taken from the head so every submesh shares one camera (fitting each
    mesh separately would scale the eyeballs to fill the screen).
    """
    mn, mx = v_view.min(0).values, v_view.max(0).values
    c = (mn + mx) / 2 if center is None else center
    ext = (mx - mn) if extent is None else extent
    s = 2.0 * margin / torch.max(ext[0], ext[1])
    x, y = (v_view[:, 0] - c[0]) * s, (v_view[:, 1] - c[1]) * s
    z = (v_view[:, 2] - c[2]) * s
    z = -z / 4.0                                   # fixed depth scale: keeps layers comparable
    return torch.stack([x, y, z, torch.ones_like(x)], dim=1)


def _rot_y(deg, device, dtype):
    a = np.radians(deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=dtype, device=device)


def _rot_x(deg, device, dtype):
    a = np.radians(deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=dtype, device=device)


def render(scene: HeadScene, meshes=None, yaw=0.0, res=512, bg=1.0, light=None, pitch=0.0):
    """Rasterize the whole head, blending groups in the game's order. -> (res,res,3) float [0,1].

    `yaw`/`pitch` name the same convention the bridge's /maker/render uses, so a comparison
    against a game capture at the same angles lines up without hand-matching the camera.
    """
    import nvdiffrast.torch as dr
    meshes = meshes or scene.deform()
    dev, dt = scene.device, scene.dtype
    R = _rot_y(yaw, dev, dt)
    if pitch:
        R = _rot_x(pitch, dev, dt) @ R

    head = meshes["o_head"]["verts"] @ R.T
    mn, mx = head.min(0).values, head.max(0).values
    center, extent = (mn + mx) / 2, (mx - mn)

    color = torch.zeros(res, res, 3, dtype=dt, device=dev)
    alpha = torch.zeros(res, res, 1, dtype=dt, device=dev)
    depth = torch.full((res, res, 1), 1e9, dtype=dt, device=dev)

    def draw(name, blend):
        m = meshes[name]
        v = m["verts"] @ R.T
        f = torch.as_tensor(np.ascontiguousarray(m["faces"]), dtype=torch.int32, device=dev)
        clip = _to_clip(v, center=center, extent=extent)[None]
        rast, _ = dr.rasterize(glctx(), clip, f, resolution=[res, res])
        mask = (rast[..., 3:4] > 0).float()
        if mask.sum() == 0:
            return

        n = vertex_normals(v, f.long())
        n_img, _ = dr.interpolate(n[None], rast, f)
        n_img = n_img / (n_img.norm(dim=-1, keepdim=True) + 1e-9)
        n_img = torch.where(n_img[..., 2:3] < 0, -n_img, n_img)
        attrs = {}
        for key in ("uv", "uv1", "vcol"):
            a = m.get(key)
            if a is None:
                attrs[key] = None
                continue
            t = torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
            attrs[key], _ = dr.interpolate(t[None].contiguous(), rast, f)

        rgba = shade(scene, name, attrs, n_img, light)           # (1,H,W,4) premult-free
        rgb, a = rgba[0, ..., :3], rgba[0, ..., 3:4] * mask[0]
        z = (rast[0, ..., 2:3] * 0.5 + 0.5)                      # NDC z -> [0,1]

        nonlocal color, alpha, depth
        if blend == "opaque":
            keep = ((z < depth) & (a > 0.5)).float()             # AlphaTest + z-test
            color = color * (1 - keep) + rgb * keep
            alpha = torch.maximum(alpha, keep)
            depth = depth * (1 - keep) + z * keep
        else:
            front = (z <= depth + 1e-4).float()                   # behind opaque -> hidden
            if blend == "multiply":                               # Blend DstColor Zero
                # tints what is already there (white = no change); needs coverage, not alpha
                color = color * (1 - front * mask[0]) + color * rgb * (front * mask[0])
            elif blend == "premultiplied":                        # Blend One OneMinusSrcAlpha
                vis = front * a
                color = rgb * front * mask[0] + color * (1 - vis)
                alpha = torch.maximum(alpha, vis)
            else:                                                 # plain alpha
                vis = front * a
                color = color * (1 - vis) + rgb * vis
                alpha = torch.maximum(alpha, vis)

    for name in OPAQUE:
        if name in meshes:
            draw(name, "opaque")
    for name, blend in BLENDED:
        if name in meshes:
            draw(name, blend)

    img = color * alpha + bg * (1 - alpha)
    return img.flip(0).clamp(0, 1)     # nvdiffrast row 0 = NDC y=-1 (bottom)


def main():
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--yaws", default="0,-25,25")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated submesh subset (debug)")
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    scene = HeadScene(args.card)
    print(f"[scene] head={os.path.basename(scene.head_dir)} submeshes={scene.present}")
    meshes = scene.deform()
    for n, m in meshes.items():
        print(f"[scene]   {n:14s} {len(m['verts']):5d} verts  {len(m['faces']):5d} tris  "
              f"mat={scene.manifest.get(n, {}).get('material')}  "
              f"shader={scene.manifest.get(n, {}).get('shader')}")
    if args.only:
        keep = set(args.only.split(","))
        meshes = {k: v for k, v in meshes.items() if k in keep}

    stem = os.path.splitext(os.path.basename(args.card))[0]
    os.makedirs("outputs", exist_ok=True)
    views = []
    for y in [float(x) for x in args.yaws.split(",") if x.strip()]:
        img = render(scene, meshes, yaw=y, res=args.res)
        views.append((img.detach().cpu().numpy() * 255).astype(np.uint8))
    gap = np.full((args.res, 8, 3), 255, np.uint8)
    combo = np.concatenate([x for v in views for x in (v, gap)][:-1], axis=1)
    out = args.out or f"outputs/{stem}_scene.png"
    Image.fromarray(combo).save(out)
    print(f"saved -> {out}")

    if args.compare:
        from ..img_utils import load_face_rgb
        ours = (render(scene, meshes, yaw=0.0, res=args.res).detach().cpu().numpy() * 255).astype(np.uint8)
        portrait = load_face_rgb(args.card, args.res, use_detector=True)
        g = np.full((args.res, 12, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([portrait, g, ours], axis=1)).save(f"outputs/{stem}_compare.png")
        print(f"saved -> outputs/{stem}_compare.png")


if __name__ == "__main__":
    main()
