"""Per-material albedo assembly + lighting for the HS2 head submeshes.

Each submesh has its own AIT shader; this builds the RGBA each one samples, following the
decompiled layer order (data/hs2_head/shaders/, docs/hs2-shader-recipe.md). Notable points that
came out of the decompile rather than intuition:

  * the EYEBROW is not part of the composed face texture — it is a `_Texture3` layer inside
    `AIT/Skin True Face`, placed with UV1 (not UV0), rotated about (0.3, 0.4), and scaled with
    offset factors (-0.1, -0.43). Using the composition convention puts brows in the wrong place.
  * the EYE stacks sclera -> iris -> pupil -> highlight in one material, with per-layer UV
    rectangles (`_texture2uv`, `_texture3uv`, `_Texture4UV`).

Lighting is still the calibratable rig from the earlier prototype (a key + fill + ambient);
Phase 4 fits it against real game renders, and the full Unity Standard BRDF + translucency lands
with it. Kept deliberately simple here so the geometry/appearance work can be judged on its own.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .composite import _sample, layer_uv, srgb_to_linear

# which assembly routine each submesh uses, keyed by its shader name
MATERIAL_KIND = {
    "AIT/Skin True Face": "skin",
    "AIT/Skin Translucency": "skin",
    "AIT/Skin Translucency simple": "simple",
    "AIT/Eye Translucency": "eye",
    "AIT/eyelashes": "cutout",
    "AIT/main eyeshadow lambert": "cutout",
    "AIT/main namida": "namida",
}


def gl_tex(t: torch.Tensor) -> torch.Tensor:
    """Unity UV origin is bottom-left, PNG/compositor arrays are top-down: flip for sampling.
    Pairs with the final `img.flip(0)` in scene.render() — the same combination the earlier,
    verified-correct `hs2_render_textured.render_head` used."""
    return t.flip(0).contiguous()


def _tex(scene, asset: str, srgb=None) -> torch.Tensor | None:
    """Load a texture from the shared pool by asset name, honouring its sRGB flag."""
    if not asset:
        return None
    info = scene.comp.pool.get(asset, {})
    if srgb is None:
        srgb = bool(info.get("srgb", False))
    return scene.comp._load(os.path.join(scene.comp.data_dir, "textures", f"{asset}.png"),
                            srgb, 4)


def _card_tex(scene, key: str) -> torch.Tensor | None:
    """A card-selected texture by manifest slot (e.g. "st_eye.AddTex")."""
    return scene.comp.tex(key, 4)


def _slot(scene, mesh: str, slot: str) -> str | None:
    """The material's texture asset for a slot (from the extracted material json)."""
    m = scene.mats.get(mesh)
    if not m:
        return None
    return (m.get("tex", {}).get(slot) or {}).get("asset")


def _vec(scene, mesh, name, default=(1, 1, 1, 1)):
    m = scene.mats.get(mesh) or {}
    return m.get("colors", {}).get(name, list(default))


def _flt(scene, mesh, name, default=0.0):
    m = scene.mats.get(mesh) or {}
    return float(m.get("floats", {}).get(name, default))


# ---------------------------------------------------------------- per-shader albedo
def _skin_albedo(scene) -> torch.Tensor:
    """The composed face texture. (H,W,4) linear.

    The eyebrow is NOT baked in here: `AIT/Skin True Face` gates it with the interpolated vertex
    colour, so it has to be applied per-pixel — see `brow_layer`.
    """
    base = scene.comp.color_pass()
    return torch.cat([base, torch.ones_like(base[..., :1])], dim=-1)


def _lerp(a, b, t):
    return a + (b - a) * float(t)


def brow_layout(scene):
    """(`_Texture3UV`, `_Texture3Rotator`) as the GAME computes them from the card.

    `ChaControl.ChangeEyebrowLayout` / `ChangeEyebrowTilt` map the card's 0..1 sliders through
    fixed Lerps; the material's stored values are just defaults and are NOT what ships on screen.
    Using the defaults draws the brow at scale (1,1) instead of (1.21,1.48) — i.e. far too thick.
    """
    ids = scene.comp.ids
    lay = ids.get("eyebrowLayout") or [0.5, 0.5, 0.5, 0.5]
    tilt = ids.get("eyebrowTilt", 0.5)
    return ([_lerp(-0.2, 0.2, lay[0]), _lerp(0.16, 0.0, lay[1]),
             _lerp(2.0, 0.5, lay[2]), _lerp(2.0, 0.5, lay[3])],
            _lerp(-0.15, 0.15, tilt))


def brow_uv(scene, uv1: torch.Tensor) -> torch.Tensor:
    """The eyebrow's placement transform — rotate about (0.3,0.4), scale with (-0.1,-0.43) offsets.

    Deliberately different from the composition layers' "scale about 0.5"; copying that convention
    puts the brows in the wrong place (decompiled from AIT/Skin True Face).
    """
    lay, rot_v = brow_layout(scene)
    dev, dt = uv1.device, uv1.dtype
    c = uv1 + torch.tensor(lay[:2], dtype=dt, device=dev)
    piv = torch.tensor([0.3, 0.4], dtype=dt, device=dev)
    rot = rot_v * np.pi
    if abs(rot) > 0:
        ca, sa = float(np.cos(rot)), float(np.sin(rot))
        d = c - piv
        c = torch.stack([d[..., 0] * ca + d[..., 1] * sa,
                         -d[..., 0] * sa + d[..., 1] * ca], -1) + piv
    sc = torch.tensor(lay[2:], dtype=dt, device=dev)
    return c * sc + (sc - 1.0) * torch.tensor([-0.1, -0.43], dtype=dt, device=dev)


def brow_layer(scene, albedo: torch.Tensor, uv1_img: torch.Tensor, vcol_img: torch.Tensor):
    """Blend the eyebrow over the skin albedo, per-pixel, exactly as the shader does:

        a   = brow.b * _Color3.a * (1 - vertexColor.b)
        rgb = lerp(albedo, brow.b * _Color3.rgb, a)

    Two traps the decompile exposed: the mask is the texture's **blue** channel (its alpha is 1.0
    everywhere, so using alpha paints the whole face), and the vertex-colour term is what confines
    the brow to the brow region at all.
    """
    brow = _card_tex(scene, "st_eyebrow.AddTex")
    if brow is None:
        return albedo
    import nvdiffrast.torch as dr
    uvb = brow_uv(scene, uv1_img)
    s = dr.texture(gl_tex(brow)[None], uvb.contiguous(), filter_mode="linear")
    mask = s[..., 2:3]                                    # BLUE channel, not alpha
    col = scene.comp.color(scene.comp.ids.get("eyebrowColor")
                           or _vec(scene, "o_head", "_Color3"))
    gate = (1.0 - vcol_img[..., 2:3]).clamp(0, 1)
    a = mask * col[3] * gate
    return albedo + a * (mask * col[:3] - albedo)


def _eye_albedo(scene, mesh: str) -> torch.Tensor:
    """AIT/Eye Translucency: sclera -> iris -> pupil -> highlight, each on its own UV rect."""
    res = 1024
    uv = scene.comp.uv if scene.comp.res == res else None
    if uv is None:
        from .composite import _base_uv
        uv = _base_uv(res, res, scene.device, scene.dtype)

    white = _tex(scene, _slot(scene, mesh, "_MainTex"), srgb=True)
    dst = (_sample(white, uv)[..., :3] if white is not None
           else torch.ones(res, res, 3, dtype=scene.dtype, device=scene.device))
    dst = dst * scene.comp.color(scene.comp.ids.get("whiteColor", _vec(scene, mesh, "_Color")))[:3]

    # (…, mask channel). These differ per layer and each texture only carries its shape in one of
    # them, so a single rule does not work:
    #   iris  — VERIFIED from the decompile (`tmp7 = tmp5.xxxz * _Color2` -> colour .r, mask .b)
    #   pupil — INFERRED: c_t_eyeblack_* is white in RGB with the shape in alpha (⏳ confirm vs GT)
    #   hl    — INFERRED: c_t_eyehigh_* has structure in R and a flat alpha (⏳ confirm vs GT)
    layers = [("st_eye.AddTex", "_Texture2", "_texture2uv", "_Color2", "pupilColor", 2),
              ("st_eyeblack.AddTex", "_Texture3", "_texture3uv", "_Color3", None, 3),
              ("st_eye_hl.AddTex", "_Texture4", "_Texture4UV", "_Color4", "hlColor", 0)]
    for card_key, slot, uv_slot, col_slot, card_col, mask_ch in layers:
        t = _card_tex(scene, card_key)
        if t is None:                                   # card picked none -> material default
            t = _tex(scene, _slot(scene, mesh, slot))
        if t is None:
            continue
        lay = _vec(scene, mesh, uv_slot, (1, 1, 0, 0))
        # the eye UV rects use (scale.xy, offset.zw) like the composition layers
        s = _sample(t, layer_uv(uv, lay))
        col = scene.comp.color(scene.comp.ids[card_col] if card_col and card_col in scene.comp.ids
                               else _vec(scene, mesh, col_slot))
        mask = s[..., mask_ch:mask_ch + 1]
        dst = dst + (mask * col[3]) * (s[..., :1] * col[:3] - dst)
    return torch.cat([dst, torch.ones_like(dst[..., :1])], dim=-1)


def _lash_albedo(scene, mesh: str) -> torch.Tensor:
    """`AIT/eyelashes`, decompiled:

        rgb = t.r * _Color.rgb
        a   = min(pow(t.r, (1 - _Color.a) * 2 + _CutoutScale), 1)

    Coverage comes from the RED channel through a power curve — the texture's alpha is 1.0
    everywhere, so using it floods the whole lash quad black. Output is effectively premultiplied,
    matching the shader's `Blend One OneMinusSrcAlpha`.
    """
    t = _card_tex(scene, "st_eyelash.AddTex")
    if t is None:
        t = _tex(scene, _slot(scene, mesh, "_MainTex"))
    if t is None:
        return torch.zeros(4, 4, 4, dtype=scene.dtype, device=scene.device)
    col = scene.comp.color(scene.comp.ids.get("eyelashesColor")
                           or _vec(scene, mesh, "_Color"))
    k = (1.0 - float(col[3])) * 2.0 + _flt(scene, mesh, "_CutoutScale", 1.0)
    r = t[..., :1]
    a = torch.clamp(r.clamp(min=1e-6) ** max(k, 1e-3), max=1.0)

    # The shader's alpha TEST, which this was missing entirely. Decompiled:
    #     tmp1.x = saturate(_Color.w + _Color.w);
    #     tmp1.x = tmp0.x * tmp1.x;          // alpha, scaled
    #     ... 4x4 Bayer dither on screen position ...
    #     tmp1.x = max(dither_passed, tmp1.x);
    #     if (tmp1.x - _Cutoff < 0) discard;
    # Without it we kept every faint fringe texel the game throws away, which is what made our
    # lashes carry 44% more total darkness than the game's and gave the eye a soft speckled rim.
    # The dither is a screen-door pattern for the cheap transparency; it only ever RESCUES a texel
    # (`max`), so ignoring it is conservative — it can cost a little coverage, never add any.
    cutoff = _flt(scene, mesh, "_Cutoff", 0.1)
    keep = (a * min(2.0 * float(col[3]), 1.0) >= cutoff).to(a.dtype)
    a = a * keep
    return torch.cat([r * col[:3] * keep, a], dim=-1)


def _eyeshadow_albedo(scene, mesh: str) -> torch.Tensor:
    """`AIT/main eyeshadow lambert`, decompiled:

        a   = min(pow(t.a, _ShadowScale), 1) * _Color.a
        rgb = lerp(1, _Color.rgb, a)

    and the pass is `Blend DstColor Zero` — a MULTIPLY, so white leaves the skin untouched. (The
    blend enum is 2 = DstColor; reading it as SrcAlpha is what made this layer flood the cheeks.)
    """
    t = _tex(scene, _slot(scene, mesh, "_MainTex"))
    if t is None:
        return torch.ones(4, 4, 4, dtype=scene.dtype, device=scene.device)
    col = scene.comp.color(scene.comp.ids.get("eyeshadowColor")
                           or _vec(scene, mesh, "_Color"))
    scale = _flt(scene, mesh, "_ShadowScale", 1.0)
    a = torch.clamp(t[..., 3:4].clamp(min=1e-6) ** max(scale, 1e-3), max=1.0) * col[3]
    rgb = 1.0 + a * (col[:3] - 1.0)
    return torch.cat([rgb, torch.ones_like(a)], dim=-1)


def _cutout_albedo(scene, mesh: str, card_key: str | None, card_col: str | None) -> torch.Tensor:
    """Lashes / eyeshadow: a single `_MainTex` tinted by `_Color`, alpha carried through."""
    t = _card_tex(scene, card_key) if card_key else None
    if t is None:
        t = _tex(scene, _slot(scene, mesh, "_MainTex"))
    if t is None:
        return torch.zeros(4, 4, 4, dtype=scene.dtype, device=scene.device)
    col = scene.comp.color(scene.comp.ids[card_col] if card_col and card_col in scene.comp.ids
                           else _vec(scene, mesh, "_Color"))
    rgb = t[..., :1] * col[:3] if t.shape[-1] >= 3 else t[..., :1] * col[:3]
    return torch.cat([rgb, t[..., 3:4] * col[3]], dim=-1)


def _simple_albedo(scene, mesh: str) -> torch.Tensor:
    t = _tex(scene, _slot(scene, mesh, "_MainTex"), srgb=True)
    if t is None:
        c = scene.comp.color(_vec(scene, mesh, "_Color"))
        return c.view(1, 1, 4).expand(4, 4, 4).clone()
    col = scene.comp.color(_vec(scene, mesh, "_Color"))
    return torch.cat([t[..., :3] * col[:3], torch.ones_like(t[..., 3:4])], dim=-1)


def build_albedo(scene, mesh: str) -> torch.Tensor:
    """(H,W,4) linear RGBA the submesh samples with its UV."""
    shader = (scene.manifest.get(mesh) or {}).get("shader")
    if mesh == "o_head":
        return _skin_albedo(scene)
    kind = MATERIAL_KIND.get(shader, "simple")
    if kind == "eye":
        return _eye_albedo(scene, mesh)
    if kind == "cutout":
        if mesh == "o_eyelashes":
            return _lash_albedo(scene, mesh)
        if mesh == "o_eyeshadow":
            return _eyeshadow_albedo(scene, mesh)
        return _cutout_albedo(scene, mesh, None, None)
    if kind == "namida":
        a = _simple_albedo(scene, mesh)
        return torch.cat([a[..., :3], a[..., 3:4] * 0.25], dim=-1)   # tear film: faint overlay
    return _simple_albedo(scene, mesh)


# OFF, and measured rather than assumed. Ablating `_Gloss` in the game (POST /maker/material,
# value 0) moves 18% of pixels by a mean of just [-1.05, -0.77, -0.61] out of 255 — the game's skin
# specular is nearly negligible. Ours, isolated the same way, moves 5.4% of pixels by +[15.7, 17.3,
# 17.1]: about 16x too strong, and it renders as a plastic-looking sheen across the forehead.
#
# The reason is in the decompile: perceptual roughness is `1 - tmp2.w`, and tmp2.w is built
# per-pixel from `_DetailMainTex` scaled by `_DetailMainScale` — skin is mostly rough with gloss
# only where the map says so. Feeding it the flat material `_Gloss` makes the whole face glossy.
#
# So this stays off until that map is wired: no specular is measurably closer to the game than a
# specular that is 16x too strong, and picking a scale factor to tame it would be a fitted constant.
SPECULAR = False


def standard_pbs(scene, mesh, albedo, n_img, rig, ambient, view=(0.0, 0.0, 1.0)):
    """`BRDF1_Unity_PBS` + the ASE translucency term, transcribed from the decompiled shader.

    Source: data/hs2_head/shaders/AIT/Skin True Face.shader, the
    `DIRECTIONAL && LIGHTPROBE_SH && SHADOWS_SCREEN` pixel variant — the one the maker actually
    runs (one directional with shadows, skybox SH ambient, no instancing, no vertex lights). Every
    constant below is read off that listing rather than recalled:

        oneMinusReflectivity = 0.96 * (1 - metallic)      `-metallic * 0.96 + 0.96`
        specColor            = lerp(0.04, albedo, metallic)
        roughness            = max(perceptualRoughness^2, 0.002)
        D (GGX)              = a2 / (pi * (NdotH^2*(a2-1) + 1)^2)   `0.3183099` = 1/pi
        V (Smith-Joint)      = 0.5 / (NdotL*(NdotV*(1-a)+a) + NdotV*(NdotL*(1-a)+a))
        specularTerm         = V * D * pi * NdotL
        diffuseTerm          = disneyDiffuse(NdotV, NdotL, LdotH, pr) * NdotL
        translucency         = lerp(Lcol, Lcol*atten, _TransShadow)
                               * (saturate(dot(V, -(N*_TransNormalDistortion + L)))^_TransScattering
                                  * _TransDirect + indirect * _TransAmbient)
        result               = albedo*transMask*translucency*_Translucency + standardPBS

    Deliberately NOT implemented yet, and each would only add detail rather than change the level:
    tangent-space normal mapping (we shade with vertex normals), the detail/weathering layers, the
    occlusion map, and image-based indirect specular — the maker's ambient is a flat SH here, so
    the grazing reflection term is approximated by the same ambient. Shadow attenuation is 1: the
    offline renderer casts no shadows.

    This replaces a half-Lambert stand-in. That stand-in was a guess, and it is what made our skin
    read pinker with a harder terminator than the game's.
    """
    dt, dev = albedo.dtype, albedo.device
    f = (scene.mats.get(mesh) or {}).get("floats", {})
    metallic = float(f.get("_Metallic", 0.0))
    # `_Gloss`, not `_Smoothness`: the extracted material JSON lists a `_Smoothness` that the
    # RUNTIME material does not have (the bridge's /maker/material returns 404 for it). Same trap
    # as the eyebrow layout and the runtime material swap — extracted data is not runtime truth.
    smooth = float(f.get("_Gloss", 0.5))
    pr = 1.0 - smooth                                   # perceptual roughness
    a = max(pr * pr, 0.002)
    a2 = a * a

    one_minus_refl = (1.0 - 0.04) * (1.0 - metallic)
    diff_color = albedo * one_minus_refl
    spec_color = 0.04 + (albedo - 0.04) * metallic

    V = torch.tensor(view, dtype=dt, device=dev)
    V = V / V.norm()
    n = n_img
    ndv = (n * V).sum(-1, keepdim=True).abs()

    out = diff_color * ambient                          # indirect diffuse
    trans_accum = torch.zeros_like(albedo)

    for d, c, inten in rig:
        L = torch.tensor(d, dtype=dt, device=dev)
        L = L / L.norm()
        lcol = torch.tensor(c, dtype=dt, device=dev) * inten
        H = L + V
        H = H / H.norm().clamp(min=1e-6)

        ndl = (n * L).sum(-1, keepdim=True).clamp(0, 1)
        ndh = (n * H).sum(-1, keepdim=True).clamp(0, 1)
        ldh = float((L * H).sum().clamp(0, 1))

        # Disney diffuse
        f90 = 0.5 + 2.0 * ldh * ldh * pr
        light_scatter = 1.0 + (f90 - 1.0) * (1.0 - ndl).clamp(min=0) ** 5
        view_scatter = 1.0 + (f90 - 1.0) * (1.0 - ndv).clamp(min=0) ** 5
        diffuse_term = light_scatter * view_scatter * ndl

        # GGX + Smith-Joint visibility
        lambda_v = ndl * (ndv * (1.0 - a) + a)
        lambda_l = ndv * (ndl * (1.0 - a) + a)
        vis = 0.5 / (lambda_v + lambda_l + 1e-5)
        dterm = ndh * ndh * (a2 - 1.0) + 1.0
        dist = a2 * 0.3183099 / (dterm * dterm + 1e-7)
        spec_term = (vis * dist * np.pi * ndl).clamp(min=0)

        fres = spec_color + (1.0 - spec_color) * (1.0 - ldh) ** 5
        out = out + diff_color * lcol * diffuse_term
        if SPECULAR:
            out = out + spec_term * lcol * fres

        # Translucency direction/scattering — computed, but see below for why it is not applied.
        ldir = n * float(f.get("_TransNormalDistortion", 0.5)) + L
        vdl = (V * -ldir).sum(-1, keepdim=True).clamp(min=1e-6)
        vdl = vdl ** float(f.get("_TransScattering", 1.0))
        trans_accum = trans_accum + lcol * (vdl * float(f.get("_TransDirect", 0.0))
                                            + ambient * float(f.get("_TransAmbient", 0.0)))

    # DELIBERATELY OFF. `_Translucency` is 30.0, so the term only makes sense against the shader's
    # per-pixel translucency MASK — `tmp4.yzw`, which the decompile builds through an RGB->HSV->RGB
    # round trip and then scales by `tmp5.www`, a factor whose source is not yet traced. Applying
    # the term without that mask blows the whole face to white (verified: it does exactly that).
    #
    # Guessing a mask value to make the picture look right is the failure mode this codebase keeps
    # paying for, so the term stays off until `tmp5.www` is read out of the listing. What is
    # implemented above is the part that IS verified: BRDF1_Unity_PBS.
    _ = trans_accum
    return out


# ---------------------------------------------------------------- lighting
# MEASURED from the running game via `/maker/lights` (2026-07-19), not hand-tuned:
# the maker rig has exactly ONE effective light — "Directional Light Key", intensity 1.00,
# colour (0.95, 0.91, 0.88), travelling along (0.34, -0.18, -0.92). The other two directionals
# ("DHH_Light", "Directional Light Back") are present but at intensity 0.
# `Ld` here points TOWARDS the light, i.e. -forward.
DEFAULT_RIG = [((-0.34, 0.18, 0.92), (0.95, 0.91, 0.88), 1.00)]

# Ambient is the one term the endpoint cannot report: RenderSettings.ambientMode is `Skybox`, so
# the real value is the skybox convolved to spherical harmonics (ambientIntensity 1.053, sky
# colour (0.666, 0.737, 0.953)). Fitted instead in closed form against the game's own render, by
# matching the foreground mean per channel — see scripts/hs2_capture_gt.py. The fit lands on the
# same blue-leaning ratio as the reported sky colour (normalised 0.79/0.93/1.00 vs 0.70/0.78/1.00),
# which is a decent independent check that it is standing in for the right thing.
AMBIENT = (1.029, 1.209, 1.303, 1.0)


def shade(scene, mesh: str, attrs: dict, n_img: torch.Tensor, light=None):
    """(1,H,W,4): albedo sampled at the interpolated attributes, lit.

    `attrs` carries uv (UV0), uv1 and vcol (vertex colour) because some layers are not pure UV0
    lookups — the eyebrow needs UV1 *and* the vertex-colour gate. Alpha passes through for the
    blend stage.
    """
    import nvdiffrast.torch as dr
    tex = scene.albedo(mesh)
    rgba = dr.texture(gl_tex(tex)[None], attrs["uv"].contiguous(), filter_mode="linear")
    albedo, alpha = rgba[..., :3], rgba[..., 3:4]
    if mesh == "o_head" and attrs.get("uv1") is not None and attrs.get("vcol") is not None:
        albedo = brow_layer(scene, albedo, attrs["uv1"], attrs["vcol"])

    # The eyeshadow's frag is a pure tint, so lighting it again would double-shade it.
    if mesh == "o_eyeshadow":
        return torch.cat([albedo.clamp(0, 1), alpha], dim=-1)

    dev, dt = scene.device, scene.dtype
    rig = light or DEFAULT_RIG
    amb = torch.tensor(AMBIENT[:3], dtype=dt, device=dev) * AMBIENT[3]

    if mesh == "o_head":
        return torch.cat([standard_pbs(scene, mesh, albedo, n_img, rig, amb).clamp(0, 1),
                          alpha], dim=-1)

    # `AIT/eyelashes` is plain Lambert, transcribed from the decompile (no half-Lambert wrap, no
    # specular at all):
    #     out.rgb = albedo * (atten * _LightColor0 * max(dot(N,L),0) + ambient)
    # An earlier version skipped lighting for this mesh entirely on a wrong reading of the shader.
    # It happens to be invisible on a card whose eyelashesColor is black (0 * anything = 0), which
    # is exactly why it survived — it only shows on coloured lashes.
    lambert = mesh == "o_eyelashes"
    diffuse = torch.zeros_like(albedo)
    for d, c, inten in rig:
        Ld = torch.tensor(d, dtype=dt, device=dev)
        Ld = Ld / Ld.norm()
        col = torch.tensor(c, dtype=dt, device=dev) * inten
        ndl = (n_img * Ld).sum(-1, keepdim=True)
        shade_term = ndl.clamp(min=0) if lambert else (ndl * 0.5 + 0.5).clamp(0, 1)
        diffuse = diffuse + col * shade_term
    return torch.cat([(albedo * (diffuse + amb)).clamp(0, 1), alpha], dim=-1)
