"""Capture ground truth from the running game, to settle what the decompile alone cannot.

Talks to the HS2_McpBridge plugin (localhost:43127) and pulls, for the character currently loaded
in the Character Maker:

  * `/maker/lights`  - the real light rig, ambient, and the project COLOUR SPACE. The colour space
    independently confirms (or refutes) the linear-space assumption the compositor is built on.
  * `/maker/facetex` - the composed 2048^2 `_MainTex`. This is the exact target
    `src/render/composite.py` reproduces, so comparing them settles the value term
    `V = (V_main - 0.1) * V_target` and the per-layer mask channels — the open ⏳ items in
    docs/hs2-shader-recipe.md — by measurement instead of by eye.
  * `/maker/render`   - offscreen renders of the head at fixed yaws, for photometric comparison.

Requires: the game running, with the Character Maker open.

    .venv/Scripts/python.exe scripts/hs2_capture_gt.py --probe          # just report lights/status
    .venv/Scripts/python.exe scripts/hs2_capture_gt.py --card tests/HS2ChaF_20240901192905747.png
    .venv/Scripts/python.exe scripts/hs2_capture_gt.py --compare-facetex <card>   # ours vs game
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.environ.get("HS2_MCP_BASE_URL", "http://127.0.0.1:43127")
GT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hs2_gt")


def call(path: str, method="GET", body=None, timeout=30):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise SystemExit(f"[{e.code}] {path}\n{detail}")
    except urllib.error.URLError as e:
        raise SystemExit(
            f"cannot reach the HS2 bridge at {url} ({e.reason}).\n"
            f"Start HoneySelect2.exe with HS2_McpBridge loaded and open the Character Maker.")


def aligned_framing(margin=0.85, res=640):
    """Framing params that make a game capture match `src/render/scene.py`'s.

    The bridge's own auto-fit under-measures the head — it clips the top of the skull, and a
    capture came out 26.7% larger than ours with the eye line 0.055 of frame height off.

    Cause, from the `bounds_variants` the bridge now reports: `BakeMesh` output already carries
    the bones' world scale, and the bridge multiplies it by `smr.transform.localToWorldMatrix`,
    applying the transform's 0.7705 scale a SECOND time. The evidence is arithmetic —
    `baked_local` (1.6146) x lossyScale (0.7705) = 1.244 = `baked_world`, and the size that
    actually matches our render is `baked_local`, not `baked_world`.

    So take the baked size as-is and only translate it into world position. Verified: interocular
    error 26.7% -> -2.2%, eye line -0.055 -> +0.0105. The remaining ~2% is a genuine geometry
    difference (our deformed mesh is 1.298x the game's baked one where 1.26x would match), not
    framing.

    Returns a query fragment to append to /maker/render. The bridge's default is still wrong;
    this fixes it caller-side so no plugin rebuild is needed to move on.
    """
    import os
    probe_png = os.path.join(os.environ.get("TEMP", "."), "_hs2_frame_probe.png")
    r = call(f"/maker/render?w={res}&h={res}&yaw=0&hide_hair=1&out={probe_png}")
    bv = r.get("bounds_variants")
    if not bv or "baked_local" not in bv:
        raise SystemExit("bridge is older than v0.14.0 — it does not report bounds_variants")
    size = bv["baked_local"]["size"]
    ctr = bv["baked_local"]["center"]
    pos = bv["smr_position"]
    ortho = max(size[0], size[1]) / 2.0 / margin
    tgt = [pos[i] + ctr[i] for i in range(3)]
    return f"ortho_size={ortho}&target={tgt[0]},{tgt[1]},{tgt[2]}"


def probe():
    """Report status + the light rig. Safe to run any time the game is up."""
    st = call("/status")
    print(f"[status] inside_maker={st.get('inside_maker')} character_ready={st.get('character_ready')} "
          f"mode={st.get('game_mode')}")
    if not st.get("inside_maker"):
        print("  -> open the Character Maker; every capture endpoint needs it")
        return st, None

    lights = call("/maker/lights")
    print(f"[lights] colour space = {lights['color_space']}   "
          f"(the compositor assumes Linear — see docs/hs2-shader-recipe.md §1b)")
    print(f"[lights] ambient mode={lights['ambient_mode']} intensity={lights['ambient_intensity']:.3f} "
          f"sky={[round(c, 3) for c in lights['ambient_sky']]}")
    on = [l for l in lights["lights"] if l["enabled"]]
    print(f"[lights] {len(on)} enabled of {len(lights['lights'])}:")
    for l in on[:12]:
        print(f"    {l['name'][:28]:28s} {l['type']:11s} I={l['intensity']:.2f} "
              f"colour={[round(c, 2) for c in l['color']]} dir={[round(c, 2) for c in l['forward']]}")
    cam = lights.get("camera")
    if cam:
        print(f"[camera] {cam['name']} ortho={cam['orthographic']} fov={cam['fov']:.1f} "
              f"pos={[round(c, 2) for c in cam['position']]}")
    return st, lights


def capture(stem: str, yaws, res: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    got = {}

    for idx, label in ((0, "maintex"), (1, "detailtex")):
        dst = os.path.join(out_dir, f"{label}.png")
        r = call(f"/maker/facetex?index={idx}&out={dst}")
        got[label] = r
        print(f"[facetex] {label}: {r['width']}x{r['height']} -> {r['path']}")
        if idx == 0:
            print(f"          draw material={r['draw_material']} shader={r['draw_shader']}")

    for yaw in yaws:
        dst = os.path.join(out_dir, f"bald_yaw{int(yaw):+04d}.png")
        # yaw is relative to the character's own facing, and the head bbox drives the framing,
        # so these line up with src/render/scene.py's renders without hand-matching the camera.
        r = call(f"/maker/render?w={res}&h={res}&yaw={yaw}&out={dst}")
        got[f"yaw{yaw}"] = r
        if yaw == yaws[0]:
            print(f"[render]  head bbox size={[round(v, 3) for v in r['head_size']]} "
                  f"ortho_size={r['ortho_size']:.3f} hid_hair={r['hid_hair']}")
        print(f"[render]  yaw={yaw:+.0f} -> {r['path']}")

    with open(os.path.join(out_dir, "capture.json"), "w", encoding="utf-8") as f:
        json.dump({"stem": stem, "captures": got, "lights": call("/maker/lights")}, f, indent=2)
    return got


def compare_facetex(card: str, gt_png: str, res=2048):
    """Measure our composed _MainTex against the game's — the decisive test for Phase 3a."""
    import numpy as np
    import torch
    from PIL import Image
    from src.render.composite import FaceCompositor, linear_to_srgb

    gt = np.asarray(Image.open(gt_png).convert("RGB"), np.float32) / 255.0
    comp = FaceCompositor.from_card(card, res=gt.shape[0],
                                    device="cuda" if torch.cuda.is_available() else "cpu")
    ours = linear_to_srgb(comp.color_pass()).detach().cpu().numpy()

    if ours.shape != gt.shape:
        raise SystemExit(f"shape mismatch: ours {ours.shape} vs game {gt.shape}")

    mask = gt.max(-1) > 0.02
    d = np.abs(ours - gt)[mask]
    print(f"[compare] composed _MainTex, {mask.sum()} texels")
    print(f"  ours mean {ours[mask].mean(0).round(4)}   game mean {gt[mask].mean(0).round(4)}")
    print(f"  L1  = {d.mean():.4f}   p95 = {np.percentile(d, 95):.4f}   max = {d.max():.4f}")
    print(f"  PSNR= {10 * np.log10(1.0 / max(((ours - gt)[mask] ** 2).mean(), 1e-12)):.2f} dB")

    stem = os.path.splitext(os.path.basename(card))[0]
    side = np.concatenate([gt, np.ones((gt.shape[0], 16, 3), np.float32), ours], axis=1)
    out = f"outputs/{stem}_facetex_compare.png"
    Image.fromarray((side * 255).astype(np.uint8)).save(out)
    print(f"  saved game | ours -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default=None, help="card to load in the maker before capturing")
    ap.add_argument("--probe", action="store_true", help="only report status + light rig")
    ap.add_argument("--yaws", default="0,-25,25")
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare-facetex", default=None, metavar="CARD",
                    help="compare our composed _MainTex against an already-captured game one")
    args = ap.parse_args()

    if args.compare_facetex:
        stem = os.path.splitext(os.path.basename(args.compare_facetex))[0]
        gt = os.path.join(args.out or os.path.join(GT_DIR, stem), "maintex.png")
        if not os.path.exists(gt):
            raise SystemExit(f"no captured game texture at {gt} — run a capture first")
        compare_facetex(args.compare_facetex, gt)
        return

    st, _ = probe()
    if args.probe:
        return
    if not st.get("inside_maker"):
        raise SystemExit("Character Maker is not open")

    if args.card:
        card = os.path.abspath(args.card)
        r = call("/maker/card/load", "POST", {"path": card}, timeout=120)
        print(f"[card] loaded {r.get('name')} headId={r.get('head_id')} skinId={r.get('skin_id')}")
        stem = os.path.splitext(os.path.basename(card))[0]
    else:
        stem = "current"

    out_dir = args.out or os.path.join(GT_DIR, stem)
    yaws = [float(y) for y in args.yaws.split(",") if y.strip()]
    capture(stem, yaws, args.res, out_dir)
    print(f"\n[done] -> {out_dir}")
    if args.card:
        print(f"next:  .venv/Scripts/python.exe scripts/hs2_capture_gt.py --compare-facetex {args.card}")


if __name__ == "__main__":
    main()
