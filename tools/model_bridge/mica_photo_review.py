"""Display unchanged MICA outputs from a small, explicitly selected photo set.

This is a geometry review board, not an HS2 renderer or an accuracy metric.
Run in the existing WSL SMIRK environment (PyTorch3D, matplotlib, Pillow).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
import torch
from pytorch3d.renderer import (
    BlendParams, FoVOrthographicCameras, HardPhongShader, Materials,
    MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings,
    TexturesVertex, look_at_view_transform,
)
from pytorch3d.structures import Meshes

from .artifact import ModelArtifact, host_path, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    if len(manifest['images']) != 3 or len(selection) != 3:
        raise ValueError('This review requires exactly three selected photographs')
    args.out.mkdir(parents=True, exist_ok=False)
    models = [ModelArtifact(args.manifest, i) for i in range(3)]
    rows = []
    for model, selected in zip(models, selection):
        original = host_path(selected['original'])
        if sha(original) != selected['sha256'] or selected['sha256'] != model.image['input_sha256']:
            raise ValueError('Original photograph/copy/raw-output correspondence changed')
        vertices, faces = model.mesh(head_local=True)
        beta = model.parameters['shape_params'][0]
        # At zero expression/pose the exact upstream shape path is linear.
        decoded = model.state['v_template'] + np.einsum('vci,i->vc', model.state['shapedirs'][:, :, :300], beta)
        error = float(np.max(np.abs(decoded - vertices)))
        if error > 1e-7:
            raise ValueError('Saved raw vertices differ from their exact shape basis')
        rows.append(dict(photo=selected['name'], original=str(original),
            photo_sha256=selected['sha256'], raw_file=model.image['file'],
            raw_sha256=model.image['sha256'], linear_decode_max_error=error))
        # Readable raw coefficients, without renaming components as anatomy.
        np.savetxt(args.out / (original.stem + '_shape_code.csv'),
            np.column_stack([np.arange(300), beta]), delimiter=',',
            header='component_index,pred_shape_code', comments='', fmt=['%d', '%.9g'])
    template = models[0].state['v_template']
    faces = models[0].faces
    if any(not np.array_equal(m.faces, faces) or not np.array_equal(m.state['v_template'], template)
           or not np.array_equal(m.state['shapedirs'], models[0].state['shapedirs']) for m in models):
        raise ValueError('All panels must use the same original topology and shape basis')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Only the camera changes. Complete raw head/neck topology is retained.
    vs = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in
          [template] + [m.mesh(head_local=True)[0] for m in models]]
    fs = [torch.as_tensor(faces, dtype=torch.int64, device=device)] * 4
    meshes = Meshes(verts=vs, faces=fs, textures=TexturesVertex(
        verts_features=[torch.full_like(v, .75) for v in vs]))
    lights = PointLights(device=device, location=[[.3, .4, 1.]],
        ambient_color=((.55, .55, .55),), diffuse_color=((.65, .65, .65),),
        specular_color=((0., 0., 0.),))
    materials = Materials(device=device, specular_color=((0., 0., 0.),))
    images = []
    for yaw in (0, 35, 90):
        rotation, translation = look_at_view_transform(dist=1., elev=0., azim=yaw,
            at=((0., .02, -.035),), device=device)
        cameras = FoVOrthographicCameras(device=device, R=rotation, T=translation,
            min_x=-.145, max_x=.145, min_y=-.145, max_y=.145)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(
                image_size=512, blur_radius=0., faces_per_pixel=1)),
            shader=HardPhongShader(device=device, cameras=cameras, lights=lights,
                materials=materials, blend_params=BlendParams(background_color=(.94, .94, .92))))
        with torch.no_grad():
            images.append(renderer(meshes)[..., :3].cpu().numpy())
    font_path = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    titles = ['FLAME 默认模板'] + [r['photo'] for r in rows]
    fig, axes = plt.subplots(3, 4, figsize=(13, 10), facecolor='#f0f0eb')
    for row, view in enumerate(images):
        for col in range(4):
            axes[row, col].imshow(view[col]); axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values(): spine.set_visible(False)
            if row == 0: axes[row, col].set_title(titles[col], fontproperties=font)
        axes[row, 0].set_ylabel(['正面', '斜侧面 35°', '侧面 90°'][row], fontproperties=font)
    fig.suptitle('原始 MICA 几何：相同尺度、相机和光照，无贴图，无局部修改', fontproperties=font)
    fig.text(.5, .02, 'PyTorch3D 几何示意，非游戏截图；只裁画面，未删网格；不代表真实三维精度认证',
        ha='center', fontproperties=font, fontsize=10)
    fig.subplots_adjust(left=.04, right=.99, top=.94, bottom=.06, wspace=.02, hspace=.02)
    fig.savefig(args.out / 'raw_models.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10, 7), facecolor='#f0f0eb')
    for i, model in enumerate(models):
        with Image.open(rows[i]['original']) as photo:
            # Expanded detector bbox for inspecting the actual source face.
            x0, y0, x1, y1 = model.arrays['detected_bbox']
            pad = .25 * max(x1-x0, y1-y0)
            axes[0, i].imshow(photo.crop((max(0, x0-pad), max(0, y0-pad),
                min(photo.width, x1+pad), min(photo.height, y1+pad))))
        axes[1, i].imshow(model.arrays['crop_bgr'][:, :, ::-1], interpolation='nearest')
        axes[0, i].set_title(rows[i]['photo'])
        for row in range(2): axes[row, i].set_xticks([]); axes[row, i].set_yticks([])
    axes[0, 0].set_ylabel('原照片脸部区域', fontproperties=font)
    axes[1, 0].set_ylabel('实际 ArcFace 输入裁图', fontproperties=font)
    fig.suptitle('原照片与实际输入；未美化、未合并身份系数', fontproperties=font)
    fig.tight_layout(); fig.savefig(args.out / 'photo_inputs.png', dpi=150); plt.close(fig)
    report = dict(manifest=str(args.manifest.resolve()), manifest_sha256=sha(args.manifest),
        selection_sha256=sha(args.selection), cases=rows, raw_outputs_unchanged=True,
        decoder_basis_shared=True, views_degrees=[0, 35, 90],
        renderer='PyTorch3D HardPhongShader, neutral constant vertex color',
        camera_target=[0., .02, -.035], orthographic_bounds=[-.145, .145],
        screenshot=False, game_mutations=0, identity_accuracy_certified=False,
        interpretation='Visual review only; no camera/pose registration to photographs')
    (args.out / 'receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(out=str(args.out), cases=len(rows), raw_outputs_unchanged=True)))


if __name__ == '__main__':
    main()
