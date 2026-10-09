"""Illustrate normal layers and actual installed HS2 skin inputs.

Reads previously audited assets; does not render or mutate the game. Licensed
texture previews and receipts belong in ignored outputs, not the Git repository.
The conceptual figure is explicitly an illustration, not an HS2 shader result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
import UnityPy

from tools.native_head.audit_neck_shading import digest, provenance, texture


def decode_normal(image, strength=1.):
    """Installed directional shader's RG/AG decode, before TBN conversion."""
    rgba = np.asarray(image.convert('RGBA'), dtype=float)/255
    xy = np.stack((2*rgba[..., 0]*rgba[..., 3]-1, 2*rgba[..., 1]-1), -1)*strength
    z = np.sqrt(1-np.minimum(1, np.sum(xy*xy, -1)))
    return np.concatenate((xy, z[..., None]), -1)


def schematic(out, font):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), layout='constrained')
    x = np.linspace(-1, 1, 7)
    points = np.c_[x, .54-.38*x*x]
    edge = np.diff(points, axis=0)
    fn = np.c_[-edge[:, 1], edge[:, 0]]
    fn /= np.linalg.norm(fn, axis=1, keepdims=True)
    vn = np.vstack([fn[0], fn[:-1]+fn[1:], fn[-1]])
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    for i, ax in enumerate(axes):
        ax.plot(*points.T, color='#555555', lw=2)
        ax.scatter(*points.T, color='#555555', s=13)
        if i == 0:
            origins, directions = (points[:-1]+points[1:])/2, fn
        else:
            t = np.linspace(0, len(points)-1, 23)
            origins = np.c_[np.interp(t, np.arange(len(points)), points[:, 0]),
                            np.interp(t, np.arange(len(points)), points[:, 1])]
            directions = np.c_[np.interp(t, np.arange(len(points)), vn[:, 0]),
                               np.interp(t, np.arange(len(points)), vn[:, 1])]
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            if i == 2:
                angle = .18*np.sin(t*3.4)
                a, b = directions.T.copy()
                directions = np.c_[a*np.cos(angle)-b*np.sin(angle),
                                    a*np.sin(angle)+b*np.cos(angle)]
        ax.quiver(*origins.T, *directions.T, color='#1565a8', angles='xy',
                  scale_units='xy', scale=4.3, width=.007)
        ax.set(xlim=(-1.2, 1.2), ylim=(0, 1.18), aspect='equal')
        ax.axis('off')
        ax.set_title(['面法线：垂直于各个面', '顶点法线：插值后平滑受光',
                      '再加法线贴图：局部方向变化'][i], fontproperties=font, fontsize=13)
        ax.text(0, -.03, ['每段面内方向相同', '受光方向可以偏离面的垂线',
                         '网格不动，只改变受光方向'][i], ha='center',
                fontproperties=font, fontsize=12)
    fig.suptitle('同一个网格，可以有不同的受光方向', fontproperties=font, fontsize=18)
    fig.text(.5, .02, '蓝色箭头＝受光法线；灰线＝相同网格剖面。讲解示意，不是游戏画面。',
             ha='center', fontproperties=font, fontsize=12)
    fig.savefig(out/'normal_layers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def previews(out, rows, font):
    fig, axes = plt.subplots(len(rows), 4, figsize=(12.5, 3.35*len(rows)))
    fig.subplots_adjust(left=.14, right=.99, top=.92, bottom=.075, wspace=.035, hspace=.17)
    titles = ['基础法线：解码后的方向', 'AO 图 R：高光相关',
              'AO 图 G：环境反射遮蔽', 'AO 图 B：细节相关']
    for row, (label, normal, packed) in enumerate(rows):
        # Only display resolution changes. Receipt fingerprints original pixels.
        image = normal.resize((384, 384), Image.Resampling.NEAREST)
        n = decode_normal(image)
        axes[row, 0].imshow(np.clip((n+1)/2, 0, 1))
        p = np.asarray(packed.convert('RGBA').resize((384, 384), Image.Resampling.NEAREST))
        for col in range(1, 4):
            axes[row, col].imshow(p[..., col-1], cmap='gray', vmin=0, vmax=255)
        for col, ax in enumerate(axes[row]):
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('#888888'); spine.set_linewidth(.4)
            if row == 0:
                ax.set_title(titles[col], fontproperties=font, fontsize=12)
        if row == len(rows)-1:
            for col, annotation in enumerate(['近中性方向', '全白：区域差异丢失',
                                              '全白：区域差异丢失', '全黑']):
                axes[row, col].set_xlabel(annotation, fontproperties=font, fontsize=11)
        axes[row, 0].text(-.08, .5, label, transform=axes[row, 0].transAxes,
                           va='center', ha='right', fontproperties=font, fontsize=12)
    fig.suptitle('真实资产：原生头、当前身体、当前导入头', fontproperties=font, fontsize=17)
    fig.text(.5, .013, '法线颜色编码的是方向，不是肤色。通道图固定 0–255 灰度，不做对比度增强。\n'
             '这些是展开的贴图输入，不是游戏渲染，也不能直接当作最后的亮度。',
             ha='center', fontproperties=font, fontsize=12)
    fig.savefig(out/'actual_skin_inputs.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def explain(manifest, audit_path, out):
    if out.exists():
        raise FileExistsError('Keep previous evidence; use a fresh output directory')
    out.mkdir(parents=True)
    manifest_data = json.loads(manifest.read_text(encoding='utf-8-sig'))
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    source = []
    def original(name):
        row = next(t for t in manifest_data['textures'] if t['asset'] == name)
        path = Path(row['output'])
        if digest(path.read_bytes()) != row['output_sha256']:
            raise ValueError('Original extracted texture changed: '+name)
        source.append(dict(kind='original_native_head', asset=name,
                           bundle=row['source_bundle'], bundle_sha256=row['source_bundle_sha256'],
                           extracted=provenance(path)))
        return Image.open(path).convert('RGBA')
    head_n, head_o = original('cf_head_02_00_n'), original('cf_head_02_00_o')
    detail = next(Path(b['path']) for b in audit['installed_bundles']
                  if Path(b['path']).name == 'ft_detail_b_00.unity3d')
    expected = next(b['sha256'] for b in audit['installed_bundles'] if b['path'] == str(detail))
    if digest(detail.read_bytes()) != expected:
        raise ValueError('Audited body dependency changed')
    env = UnityPy.load(str(detail))
    body_n, body_o, body_n2 = [texture(env, name).image.convert('RGBA')
                             for name in ('cf_body_base_n', 'cf_body_00_o', 'cf_body_00_n')]
    source.append(dict(kind='captured_body_binding_dependency', bundle=provenance(detail),
                       assets=['cf_body_base_n', 'cf_body_00_o', 'cf_body_00_n']))
    package = Path(audit['package']['path'])
    if digest(package.read_bytes()) != audit['package']['sha256']:
        raise ValueError('Audited imported package changed')
    with zipfile.ZipFile(package) as z:
        skin = UnityPy.load(z.read('abdata/chara/codex/chenger/skin.unity3d'))
    current_n, current_o = [texture(skin, name).image.convert('RGBA')
                            for name in ('cf_head_02_00_n', 'cf_head_02_00_o')]
    if (not np.all(np.asarray(current_n) == [255, 128, 128, 128])
            or not np.all(np.asarray(current_o) == [255, 255, 0, 255])):
        raise ValueError('Current texture values changed; update the constant-input annotations')
    source.append(dict(kind='current_imported_head', package=provenance(package)))
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    schematic(out, font)
    previews(out, [('原生头\nhead2 / skin20', head_n, head_o),
                   ('当前身体\ndetail0', body_n, body_o),
                   ('当前导入头\n版本 0.1.6', current_n, current_o)], font)
    body_n2.resize((768, 768), Image.Resampling.NEAREST).save(out/'body_second_normal_raw.png')
    receipt = dict(format='hs2_skin_inputs_explanation_v1', game_mutated=False,
        source_manifest=provenance(manifest), source_audit=provenance(audit_path), sources=source,
        normal_decode='X=2*(R*A)-1, Y=2*G-1; scale XY; Z=sqrt(1-min(1,dot(XY,XY)))',
        actual_body_second_normal_strength=next(x['materials'][0]['float'] for x in audit['runtime_scalars']
            if x['mesh']=='o_body_cf' and x['property']=='_BumpScale2'),
        figures=[provenance(out/name) for name in ('normal_layers.png', 'actual_skin_inputs.png',
                                                 'body_second_normal_raw.png')],
        limitations=['Conceptual normal arrows are illustrative, not an extracted HS2 mesh',
            'Normal colors are decoded tangent-space direction visualization at strength 1, not the final shading normal',
            'Body second normal, strength and TBN still participate in actual shader shading; not combined in texture overview',
            'Packed-map roles refer to the recovered installed directional shader branch',
            'No image measures the contribution to final game color, and no continuity fix is claimed'])
    (out/'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out.resolve()), game_mutated=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'audit', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    explain(args.manifest, args.audit, args.out)
