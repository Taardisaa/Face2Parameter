"""Differentiable torch port of HS2's runtime face-texture composition.

`AIT/Skin True Face`'s `_MainTex` is not a file on disk: at load time `ChaControl` blits the card's
choices (skin tone, lipstick, eyeshadow, blush, mole, tattoos) through the `create_skin_face`
material into a 2048^2 RenderTexture, and the same again through `create_skin detail_face` for
`_DetailMainTex`. Reproducing that is a prerequisite for the renderer showing makeup at all.

Ported line-by-line from the DECOMPILED shaders (`data/hs2_head/shaders/Create/*.shader`, recovered
with USCSandbox) — not from the C#, because the C# assignment order is NOT the blend order. See
docs/hs2-shader-recipe.md §1b. Two things worth knowing before editing:

  * the skin base is an **additive-HSV recolour**, not `albedo * _Color`:
        h = |h_tex| + |h_target|,  s = s_tex + s_target - 0.5,  v = (v_tex - 0.1) * v_target
    with a -0.05 hue offset baked into the HSV->RGB reconstruction;
  * blend order is **paint01 -> paint02 -> cheek -> lip -> eyeshadow -> mole**, so eyeshadow sits
    *above* lipstick.

Everything is plain tensor arithmetic, so the output is differentiable w.r.t. every card colour
(and the textures themselves) — that is what lets makeup participate in a gradient objective later.

    from src.render.composite import FaceCompositor
    comp = FaceCompositor.from_card("tests/HS2ChaF_20240901192905747.png")
    main, detail = comp()                      # (H,W,3), (H,W,4), both float [0,1]

Preview / self-check:
    .venv/Scripts/python.exe -m src.render.composite --card tests/HS2ChaF_20240901192905747.png
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "hs2_head")

# layer -> (texture slot, colour slot, layout slot, the card fields that drive them).
# Order is the SHADER's blend order (see module docstring), which is what matters.
COLOR_LAYERS = [
    #  tex slot     colour slot  layout slot     card texture key          card colour key
    ("_Texture5", "_Color5", "_Texture5UV", "st_paint.AddTex0", "paint0"),
    ("_Texture6", "_Color6", "_Texture6UV", "st_paint.AddTex1", "paint1"),
    ("_Texture10", "_Color10", "_Texture10UV", "st_cheek.AddTex", "cheekColor"),
    ("_Texture9", "_Color9", "_Texture9UV", "st_lip.AddTex", "lipColor"),
    ("_Texture11", "_Color11", "_Texture11UV", "st_eyeshadow.AddTex", "eyeshadowColor"),
    ("_Texture12", "_Color12", "_Texture12UV", "st_mole.AddTex", "moleColor"),
]
ROTATED = {"_Texture5", "_Texture6"}          # only the paint layers get a rotator


# ---------------------------------------------------------------- colour space
def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp(min=0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    c = c.clamp(0, 1)
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def rgb_to_hsv(c: torch.Tensor) -> torch.Tensor:
    """(...,3) linear-ish RGB -> (...,3) HSV, matching Unity's `RGBtoHSV` in UnityCG.cginc."""
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, _ = c.max(dim=-1)
    mn, _ = c.min(dim=-1)
    d = mx - mn
    eps = 1e-10
    # hue by sector, expressed branchlessly the way the compiled shader does
    h = torch.where(mx == r, (g - b) / (6.0 * d + eps),
                    torch.where(mx == g, (b - r) / (6.0 * d + eps) + 1.0 / 3.0,
                                (r - g) / (6.0 * d + eps) + 2.0 / 3.0))
    h = torch.where(d <= eps, torch.zeros_like(h), h)
    return torch.stack([h.abs(), d / (mx + eps), mx], dim=-1)


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unity's `HSVtoRGB`, with the -0.05 hue offset the game's shader bakes in.

    The compiled constants are (0.95, 0.6166667, 0.2833333) = (1, 2/3, 1/3) - 0.05.
    """
    off = torch.tensor([0.95, 0.6166667, 0.2833333], dtype=h.dtype, device=h.device)
    p = torch.frac(h.unsqueeze(-1) + off) * 6.0 - 3.0
    p = (p.abs() - 1.0).clamp(0, 1)
    return (1.0 + s.unsqueeze(-1) * (p - 1.0)) * v.unsqueeze(-1)


# ---------------------------------------------------------------- uv helpers
def _base_uv(h, w, device, dtype):
    """(H,W,2) UV grid. Unity's texture origin is bottom-left; arrays are top-down, so v is flipped."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    v, u = torch.meshgrid(1.0 - ys, xs, indexing="ij")
    return torch.stack([u, v], dim=-1)


def _sample(tex: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Bilinear tex2D with clamp-to-edge. tex:(h,w,C) uv:(H,W,2) -> (H,W,C)."""
    grid = (uv * 2.0 - 1.0).unsqueeze(0)                       # (1,H,W,2) in [-1,1]
    grid = torch.stack([grid[..., 0], -grid[..., 1]], dim=-1)  # v-flip: grid_sample y is top-down
    out = torch.nn.functional.grid_sample(
        tex.permute(2, 0, 1).unsqueeze(0), grid,
        mode="bilinear", padding_mode="border", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0)


def layer_uv(uv: torch.Tensor, layout, rotator: float | None = None) -> torch.Tensor:
    """The composition shader's placement transform.

        uv' = rot(uv + layout.zw, rotator*pi)  [paint only]
        uv' = uv' * layout.xy + (1 - layout.xy) * 0.5      # scale about the texture centre

    `layout` is the material's `_TextureNUV` vector, e.g. lip = (4,4,0,0.18): a quarter-size decal
    centred on the mouth.
    """
    sx, sy, ox, oy = [float(t) for t in layout]
    out = uv + torch.tensor([ox, oy], dtype=uv.dtype, device=uv.device)
    if rotator is not None and abs(rotator) > 0:
        a = rotator * np.pi
        ca, sa = float(np.cos(a)), float(np.sin(a))
        c = out - 0.5
        out = torch.stack([c[..., 0] * ca + c[..., 1] * sa,
                           -c[..., 0] * sa + c[..., 1] * ca], dim=-1) + 0.5
    scale = torch.tensor([sx, sy], dtype=uv.dtype, device=uv.device)
    return out * scale + (1.0 - scale) * 0.5


# ---------------------------------------------------------------- compositor
class FaceCompositor:
    """Reproduces `create_skin_face` (-> `_MainTex`) and `create_skin detail_face` (-> `_DetailMainTex`)."""

    def __init__(self, manifest: dict, data_dir=_DATA, res=2048, device="cuda",
                 dtype=torch.float32):
        self.man, self.data_dir, self.res = manifest, data_dir, res
        self.device, self.dtype = torch.device(device), dtype
        self.ids = manifest["ids"]
        self.mat = json.load(open(os.path.join(data_dir, "composite", "create_skin_face.json"),
                                  encoding="utf-8"))
        self.mat_detail = json.load(open(
            os.path.join(data_dir, "composite", "create_skin detail_face.json"), encoding="utf-8"))
        self._cache: dict[str, torch.Tensor | None] = {}
        self.uv = _base_uv(res, res, self.device, dtype)
        pool = os.path.join(data_dir, "textures", "pool.json")
        self.pool = json.load(open(pool, encoding="utf-8")) if os.path.exists(pool) else {}

    # ---- asset access
    def _load(self, path: str, srgb: bool, channels=4) -> torch.Tensor | None:
        """Read a PNG to a float tensor. sRGB-flagged textures are LINEARISED, matching how Unity
        samples them (the game renders in linear space — see `_save_textures_from`). Only the
        colour channels are converted; alpha/masks are always linear data."""
        if not os.path.exists(path):
            return None
        im = Image.open(path).convert("RGBA" if channels == 4 else "RGB")
        t = torch.as_tensor(np.asarray(im, np.float32) / 255.0, dtype=self.dtype, device=self.device)
        if srgb:
            t = torch.cat([srgb_to_linear(t[..., :3]), t[..., 3:]], dim=-1) if t.shape[-1] == 4 \
                else srgb_to_linear(t)
        return t

    def tex(self, key: str, channels=4) -> torch.Tensor | None:
        """A card-selected texture by manifest slot key (e.g. "st_lip.AddTex"), or None if unset."""
        if key in self._cache:
            return self._cache[key]
        slot = self.man["textures"].get(key)
        t = None
        if slot and slot.get("file"):
            t = self._load(os.path.join(self.data_dir, slot["file"]),
                           bool(slot.get("srgb", True)), channels)
        self._cache[key] = t
        return t

    def pool_tex(self, asset: str, channels=4) -> torch.Tensor | None:
        """A texture from the shared pool by asset name (masks etc.)."""
        key = f"@{asset}"
        if key not in self._cache:
            info = self.pool.get(asset, {})
            self._cache[key] = self._load(os.path.join(self.data_dir, "textures", f"{asset}.png"),
                                          bool(info.get("srgb", False)), channels)
        return self._cache[key]

    def color(self, rgba) -> torch.Tensor:
        """A material/card colour as the shader sees it: Unity converts SetColor values to linear.

        A tensor is passed through (graph intact) so callers can optimise a colour directly;
        a plain list is lifted to a constant.
        """
        c = rgba if torch.is_tensor(rgba) else torch.as_tensor(
            [float(x) for x in rgba], dtype=self.dtype, device=self.device)
        return torch.cat([srgb_to_linear(c[:3]), c[3:]])

    def _vec(self, mat, name, default=(1, 1, 1, 1)):
        return mat["colors"].get(name, list(default))

    def _color(self, name, card_key=None):
        """Card colour if the card supplies one, else the material's stored default."""
        if card_key:
            if card_key.startswith("paint"):
                i = int(card_key[-1])
                info = self.ids.get("paintInfo") or []
                if i < len(info):
                    return info[i]["color"]
            elif card_key in self.ids:
                return self.ids[card_key]
        return self._vec(self.mat, name)

    # ---- the two passes
    def skin_base(self) -> torch.Tensor:
        """The HSV-recoloured skin, before any makeup layer. (H,W,3)"""
        base = self.tex("ft_skin_f.MainTex", 3)
        if base is None:
            raise SystemExit("card manifest has no ft_skin_f.MainTex — re-run hs2_extract_head.py")
        base = _sample(base, self.uv)

        skin_c = self.color(self.ids["skinColor"])[:3].expand_as(base).clone()

        sun = self.tex("ft_sunburn.AddTex", 4)                 # id 0 = none on most cards
        if sun is not None:
            sc = self.color(self.ids["sunburnColor"])
            a = (_sample(sun, self.uv)[..., :1] * sc[3])
            skin_c = skin_c + a * (sc[:3] - skin_c)
        nail = self.pool_tex("nail_mask", 4)                   # absent for the face
        if nail is not None:
            c13 = self.color(self._vec(self.mat, "_Color13"))[:3]
            m = _sample(nail, self.uv)[..., :1]
            skin_c = skin_c + m * (c13 - skin_c)

        hm = rgb_to_hsv(base)
        ht = rgb_to_hsv(skin_c)
        return hsv_to_rgb(hm[..., 0] + ht[..., 0],                       # hues ADD
                          hm[..., 1] + ht[..., 1] - 0.5,                 # saturations ADD, -0.5
                          (hm[..., 2] - 0.1) * ht[..., 2]).clamp(0, 1)   # values MULTIPLY

    def color_pass(self) -> torch.Tensor:
        """`Create/skin color` -> the composed `_MainTex`. (H,W,3)"""
        dst = self.skin_base()
        paint_mask = self.pool_tex("paint_mask_face", 4)
        pm = _sample(paint_mask, self.uv)[..., :1] if paint_mask is not None else None

        for tex_slot, col_slot, uv_slot, tex_key, card_key in COLOR_LAYERS:
            t = self.tex(tex_key, 4)
            if t is None:
                continue
            rot = self.mat["floats"].get(f"{tex_slot}Rotator", 0.0) if tex_slot in ROTATED else None
            uv = layer_uv(self.uv, self._vec(self.mat, uv_slot, (1, 1, 0, 0)), rot)
            s = _sample(t, uv)
            col = self.color(self._color(col_slot, card_key))

            if tex_slot in ROTATED:                       # paint: .b is a colourise mask
                layer = s[..., :1] + s[..., 2:3] * (s[..., :1] * col[:3] - s[..., :1])
                a = s[..., 3:4] * col[3] * (pm if pm is not None else 1.0)
            else:                                        # cheek / lip / eyeshadow / mole
                layer = s[..., :1] * col[:3]
                a = s[..., 3:4] * col[3]
            dst = dst + a * (layer - dst)                # lerp(dst, layer, a)
        return dst.clamp(0, 1)

    def detail_pass(self) -> torch.Tensor:
        """`Create/skin detail` -> `_DetailMainTex`. Channels: (r=metallic, g=normal, b=gloss, a)."""
        m = self.mat_detail
        base = self.pool_tex("black2048", 4)
        dst = (_sample(base, self.uv) if base is not None
               else torch.zeros(self.res, self.res, 4, dtype=self.dtype, device=self.device))

        for i, (tex_slot, col_slot) in enumerate((("_Texture5", "_Color5"), ("_Texture6", "_Color6"))):
            t = self.tex(f"st_paint.GlossTex{i}", 4)
            if t is None:
                continue
            rot = m["floats"].get(f"{tex_slot}Rotator", 0.0)
            s = _sample(t, layer_uv(self.uv, self._vec(m, f"{tex_slot}UV", (1, 1, 0, 0)), rot))
            a = s[..., 3:4] * self._color(col_slot, f"paint{i}")[3]
            met = torch.full_like(a, m["floats"].get(f"_Metallic{5 + i}", 0.0))
            glo = torch.full_like(a, m["floats"].get(f"_Gloss{5 + i}", 0.0))
            bump = s[..., 1:2] * m["floats"].get(f"_BumpScale{5 + i}", 1.0)
            dst = torch.maximum(dst, torch.cat([torch.minimum(a, met), bump,
                                                torch.minimum(a, glo), a], dim=-1))
        pm = self.pool_tex("paint_mask_face", 4)
        if pm is not None:
            dst = dst * _sample(pm, self.uv)[..., :1]

        gloss = torch.zeros(self.res, self.res, 1, dtype=self.dtype, device=self.device)
        for tex_slot, uv_slot, tex_key, gloss_key in (
                ("_Texture10", "_Texture10UV", "st_cheek.AddTex", "cheekGloss"),
                ("_Texture9", "_Texture9UV", "st_lip.AddTex", "lipGloss"),
                ("_Texture11", "_Texture11UV", "st_eyeshadow.AddTex", "eyeshadowGloss")):
            t = self.tex(tex_key, 4)
            if t is None:
                continue
            s = _sample(t, layer_uv(self.uv, self._vec(m, uv_slot, (1, 1, 0, 0))))
            gloss = torch.maximum(gloss, s[..., 3:4] * float(self.ids.get(gloss_key, 0.0)))
        zero = torch.zeros_like(gloss)
        return torch.maximum(dst, torch.cat([zero, zero, gloss, zero], dim=-1)).clamp(0, 1)

    def __call__(self):
        return self.color_pass(), self.detail_pass()

    @classmethod
    def from_card(cls, card_path: str, **kw):
        stem = os.path.splitext(os.path.basename(card_path))[0]
        path = os.path.join(kw.get("data_dir", _DATA), "cards", f"{stem}.json")
        if not os.path.exists(path):
            raise SystemExit(f"no manifest for {stem} — run:\n"
                             f"    .venv/Scripts/python.exe scripts/hs2_extract_head.py --card {card_path}")
        return cls(json.load(open(path, encoding="utf-8")), **kw)


def _save(t: torch.Tensor, path: str, encode_srgb=True):
    """Preview PNG. The compositor works (and returns) in LINEAR light, so encode for display."""
    x = linear_to_srgb(t) if encode_srgb else t.clamp(0, 1)
    a = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(a).save(path)
    print(f"saved {path}  {tuple(t.shape)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--res", type=int, default=2048)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    comp = FaceCompositor.from_card(args.card, res=args.res, device=dev)
    stem = os.path.splitext(os.path.basename(args.card))[0]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[composite] {stem}  res={args.res}  device={dev}")
    print(f"[composite] skinColor={[round(c, 3) for c in comp.ids['skinColor']]}  "
          f"lipId={comp.ids['lipId']} eyeshadowId={comp.ids['eyeshadowId']} "
          f"cheekId={comp.ids['cheekId']} moleId={comp.ids['moleId']}")
    for slot, _, uv_slot, key, _ in COLOR_LAYERS:
        t = comp.tex(key)
        if t is not None:
            print(f"[composite]   layer {slot:11s} <- {key:22s} {tuple(t.shape)} "
                  f"uv={comp._vec(comp.mat, uv_slot)}")

    base = comp.skin_base()
    main, detail = comp()
    _save(base, f"{args.out_dir}/{stem}_skin_base.png")
    _save(main, f"{args.out_dir}/{stem}_maintex.png")
    _save(detail[..., :3], f"{args.out_dir}/{stem}_detailtex.png")

    d = (main - base).abs()
    print(f"[composite] makeup changed {(d.max(-1).values > 2/255).float().mean() * 100:.2f}% of texels"
          f"  (max delta {d.max():.3f})")
