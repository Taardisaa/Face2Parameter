"""Instruction-based expression neutralization via FLUX.1 Kontext [dev] (GGUF, low-VRAM).

Edits each face toward a neutral expression with a constrained "change expression only, keep
identity/pose/lighting/background" prompt, using a GGUF-quantized FLUX Kontext transformer +
sequential CPU offload so it fits a ~5-12 GB GPU. Output feeds the existing pipeline; an ArcFace
identity gate (in src/neutralize.py) rejects any edit that drifts identity.

Runs in the dedicated FLUX env (see docs/expression-invariance.md). The gated base components
(T5/CLIP/VAE) need an accepted license + HF token (HF_TOKEN env, set by the caller).

    python scripts/flux_neutralize.py --model <kontext.gguf> --in <dir> --out <dir>
"""

import argparse
import glob
import os

import torch
from PIL import Image

BASE_REPO = "black-forest-labs/FLUX.1-Kontext-dev"
# DEFAULT_PROMPT = (
#     "Change only the facial expression to a calm, neutral expression with the mouth gently "
#     "closed and relaxed cheeks (no smile). "
#     # "If the cheeks are raised or puffed up from smiling "
#     # "(apple cheeks / bulging cheekbone area), flatten them back to a relaxed, non-smiling state. "
#     "Keep the same person, identity, and other facial features, "
#     "hairstyle, makeup, head pose, camera angle, lighting, and background unchanged."
# )

# DEFAULT_PROMPT = """
# Modify the woman's facial expression to completely remove the smile. Change her mouth to a neutral, closed-lip position with lips gently pressed together and no teeth visible. Relax all smiling muscles: lower the raised apple cheeks (zygomatic area) back to their natural resting state so they appear less lifted and more relaxed, as in a non-smiling face. Keep the eyes looking straight ahead with a natural, non-squinting expression. Preserve every other detail exactly the same as the original photo: short dark brown bob haircut with center part, white sleeveless top with lace neckline, arms raised with hands behind head, wooden post, tiled floor background, lighting, skin texture, body pose, and photorealistic quality. Seamless photorealistic edit with high fidelity to the original reference.
# """

DEFAULT_PROMPT = """
neutral relaxed expression, mouth gently closed, no smile, flatten the raised apple cheeks;
smooth even youthful skin, remove nasolabial folds, smile lines and any cheek/fat-crease wrinkles,
soft flawless complexion, refined and beautiful;
keep the same hairstyle, head pose and lighting; replace the background with solid pure blue
"""

# Only used when --true-cfg > 1: the things real CFG actively steers AWAY from. This is the
# strongest lever for "remove the smile" — far more effective than cranking --guidance.
DEFAULT_NEGATIVE = (
    "smile, smiling, grin, teeth showing, open mouth, raised apple cheeks, "
    "squinting eyes, nasolabial folds, smile lines, laugh lines, cheek wrinkles"
)

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.webp")


def list_images(path):
    if os.path.isdir(path):
        out = []
        for e in IMAGE_EXTS:
            out += glob.glob(os.path.join(path, "**", e), recursive=True)
        return sorted(out)
    return [path]


def fit(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the long side <= max_side and both dims are multiples of 16 (FLUX-friendly)."""
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    w, h = max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)
    return img.resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to the FLUX.1-Kontext-dev GGUF transformer")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5,
                    help="FLUX distilled guidance (NOT real CFG). Useful range ~2.5-6; >7 = artifacts.")
    ap.add_argument("--true-cfg", dest="true_cfg", type=float, default=1.0,
                    help="real classifier-free guidance. >1 enables it (uses --negative-prompt); "
                         "try 2-4. Costs ~2x compute/step but is the strongest 'remove the smile' lever.")
    ap.add_argument("--negative-prompt", dest="negative_prompt", default=DEFAULT_NEGATIVE,
                    help="only used when --true-cfg > 1; what the edit is steered AWAY from.")
    ap.add_argument("--max-side", type=int, default=768,
                    help="cap output long side (also keeps mtcnn happy; it fails above ~1024)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[args.prompt]: {args.prompt}")
    if args.true_cfg > 1.0:
        print(f"[true_cfg={args.true_cfg}] negative: {args.negative_prompt}")

    from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

    print(f"[flux_neutralize] loading GGUF transformer: {os.path.basename(args.model)}")
    transformer = FluxTransformer2DModel.from_single_file(
        args.model,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=BASE_REPO,
        subfolder="transformer",
    )
    pipe = FluxKontextPipeline.from_pretrained(BASE_REPO, transformer=transformer,
                                               torch_dtype=torch.bfloat16)
    # NOTE: sequential offload breaks GGUF (accelerate's meta-device move drops quant_type ->
    # KeyError: None). model_cpu_offload is the GGUF-compatible low-VRAM path. enable VAE tiling/slicing
    # too. If this OOMs on a tight GPU, use a smaller GGUF (Q3_K_M / Q2_K).
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    os.makedirs(os.path.abspath(args.out), exist_ok=True)
    paths = list_images(os.path.abspath(args.inp))
    print(f"[flux_neutralize] {len(paths)} image(s) | steps={args.steps} guidance={args.guidance} "
          f"true_cfg={args.true_cfg}")

    # real CFG (true_cfg_scale > 1) needs a negative prompt; the distilled guidance_scale stays on too.
    cfg_kwargs = {}
    if args.true_cfg > 1.0:
        cfg_kwargs = {"true_cfg_scale": args.true_cfg, "negative_prompt": args.negative_prompt}

    for p in paths:
        try:
            img = fit(Image.open(p).convert("RGB"), args.max_side)
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            # pass height/width so Kontext preserves the input aspect (else it defaults to a square
            # bucket and horizontally stretches the face).
            out = pipe(image=img, prompt=args.prompt, height=img.height, width=img.width,
                       guidance_scale=args.guidance, num_inference_steps=args.steps,
                       generator=gen, **cfg_kwargs).images[0]
            stem = os.path.splitext(os.path.basename(p))[0]
            out.save(os.path.join(os.path.abspath(args.out), stem + ".jpg"), quality=95)
            print(f"  ok: {os.path.basename(p)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {os.path.basename(p)}: {exc}")


if __name__ == "__main__":
    main()
