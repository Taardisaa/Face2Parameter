# Pretrained MLP heads

Trained Stage-2 regression heads, ready to use with no training. Everything else the
pipeline needs is fetched automatically on first run (DINOv2 / ArcFace backbones, mtcnn
face detector) or already ships in the repo (the HS2 card serializer, the default template),
so these `.pth` files are the only artifacts you need to download.

| File | Config | Backbone | Notes |
|---|---|---|---|
| `head_arcface.pth` | `arcface` | ArcFace `w600k_r50` (512-d) | **Expression-invariant** — best for real photos, robust to smiling. First run downloads InsightFace `buffalo_l` (~288 MB). |
| `head_dinov2_vits14.pth` | `dinov2_vits14` | DINOv2 ViT-S/14 (384-d) | General-purpose. First run downloads the DINOv2 hub weights. |

Both were trained to epoch 30 on the HS_FACE dataset.

## Use

After `git clone` + `pip install -r requirements.txt` (see the top-level [README](../README.md)),
these heads are used **automatically** — with no `exp/` from training, `predict.py`/`infer.py` fall
back to the `release/` head for the chosen `--config` (resolved by `config.resolve_head`):

```bash
# image -> 205-dim param vector (no card serializer needed)
python predict.py --config arcface --image my.png

# image -> a full HS2 character card (uses assets/default_template.png)
python infer.py   --config arcface --image my.png --out outputs/
```

Swap `--config dinov2_vits14` to use the DINOv2 head. Pass an explicit
`--head release/head_arcface.pth` to force a specific file, or `--no-detector` to skip mtcnn
alignment (uses a center-crop instead).
