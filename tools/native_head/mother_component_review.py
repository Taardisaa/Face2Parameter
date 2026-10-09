"""Complete component geometry review: scientific illustrations, no game calls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.font_manager import FontProperties
import numpy as np

from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json, source_file


NAMES = {'o_head': '头部皮肤及口内', 'o_eyebase_L': '左眼球', 'o_eyebase_R': '右眼球',
         'o_eyeshadow': '眼部阴影层', 'o_eyelashes': '睫毛', 'o_tooth': '牙齿',
         'o_tang': '舌头', 'o_namida': '泪液层'}


def rotated(points, yaw):
    angle = np.deg2rad(yaw)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    return points@rotation.T


def surface(ax, parts, yaw):
    triangles, colors = [], []
    for name, arrays in parts.items():
        tri = rotated(arrays['authored_closed_reference'], yaw)[arrays['faces']]
        normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), np.finfo(float).tiny)
        palette = [.76, .71, .65] if name=='o_head' else [.56, .69, .78]
        rgb = np.asarray(palette)[None]*(.3+.7*np.maximum(normal@np.array([-.3, .4, .86]), 0))[:, None]
        selected = normal[:, 2] > 0
        triangles.append(tri[selected]); colors.append(rgb[selected])
    tri, rgb = np.concatenate(triangles), np.concatenate(colors)
    order = np.argsort(tri[:, :, 2].mean(1))
    ax.add_collection(PolyCollection(tri[order][:, :, :2], facecolors=rgb[order],
                                    edgecolors='none', antialiased=False))


def framing(ax, limits):
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def review(bindings, out):
    if out.exists():
        raise FileExistsError('Preserve earlier reviews; use fresh output')
    receipt = json.loads((bindings/'receipt.json').read_text())
    parts = {}
    for row in receipt['renderer_outputs']:
        if sha(row['arrays']['path']) != row['arrays']['sha256']:
            raise ValueError('Authored component changed')
        parts[row['mesh']] = dict(np.load(row['arrays']['path'], allow_pickle=False))
    if set(parts) != set(NAMES):
        raise ValueError('Complete eight native components required')
    out.mkdir(parents=True)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    head = parts['o_head']['authored_closed_reference']
    cy = (head[:, 1].min()+head[:, 1].max())/2
    span = np.ptp(head[:, 1])*1.1
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.3))
    # Transparent/layered secondary surfaces are displayed separately below.
    # Do not pretend opaque scientific colors reproduce native shaders.
    selected = {name: arrays for name, arrays in parts.items()
                if name in ('o_head', 'o_eyebase_L', 'o_eyebase_R')}
    for ax, yaw, title in zip(axes, (0, -40, -90), ('正面', '斜侧', '侧面')):
        surface(ax, selected, yaw)
        x = rotated(head, yaw)[:, 0]
        cx = (x.min()+x.max())/2
        framing(ax, ((cx-span*.43, cx+span*.43), (cy-span*.5, cy+span*.5)))
        ax.set_title(title, fontproperties=font)
    fig.suptitle('完整头部皮肤 + 原生眼球 · 已适配默认参考', fontproperties=font, fontsize=15)
    fig.text(.5, .02, '无贴图科学示意 · 不是游戏效果 · 其余部件见下图 · 尚未安装',
             fontproperties=font, ha='center')
    fig.tight_layout(rect=(0, .05, 1, .94)); fig.savefig(out/'head_with_native_eyes.png', dpi=145); plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    component_policy = []
    for index, (name, arrays) in enumerate(parts.items()):
        vertices, faces = arrays['authored_closed_reference'], arrays['faces']
        edges = np.unique(np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1), axis=0)
        row, pair = divmod(index, 2)
        for view, yaw in enumerate((0, -90)):
            ax = axes[row, pair*2+view]
            v = rotated(vertices, yaw)
            ax.add_collection(LineCollection(v[edges][:, :, :2], colors='#466d87', linewidths=.35, alpha=.6))
            low, high = v[:, :2].min(0), v[:, :2].max(0)
            center = (low+high)/2
            extent = max(high-low)*.55
            framing(ax, ((center[0]-extent, center[0]+extent), (center[1]-extent, center[1]+extent)))
            ax.set_title(NAMES[name]+(' · 正面' if view==0 else ' · 侧面'), fontproperties=font, fontsize=11)
        component_policy.append(dict(mesh=name, vertices=len(vertices), faces=len(faces),
                                    all_edges_shown=True, component_autozoom=True))
    fig.suptitle('八个原生部件逐项展开 · 每项独立缩放 · 线框不隐藏背面', fontproperties=font, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(out/'all_eight_components.png', dpi=145); plt.close(fig)
    save_json(out/'receipt.json', dict(format='native_mother_component_review_v1',
        bindings=source_file(bindings/'receipt.json'), code=source_file(Path(__file__)),
        head_surface_components=list(selected), wireframe_components=component_policy,
        images=[source_file(out/'head_with_native_eyes.png'),source_file(out/'all_eight_components.png')],
        screenshot=False, game_mutated=False, deliverable=False,
        surface_display='Original face connectivity; painter-sorted front faces, artificial colors without native shaders',
        wireframe_display='Every component edge including back-facing/internal edges; individual autozoom',
        geometry_certified=False, animation_certified=False))
    print(str(out.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review(args.bindings.resolve(), args.out.resolve())
