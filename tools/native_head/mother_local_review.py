"""Full-head before/after and explicit local edit regions, at identical scale."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
import numpy as np

from tools.native_head.mother_candidate_review import shade
from tools.native_head.mother_template_inputs import save_json, source_file


def review(base, candidate, out):
    if out.exists():
        raise FileExistsError('Keep previous review')
    original = dict(np.load(base/'o_head_candidate.npz', allow_pickle=False))
    repaired = dict(np.load(candidate/'o_head_candidate.npz', allow_pickle=False))
    regions = json.loads((candidate/'local_patch_regions.json').read_text())
    active = np.zeros(len(repaired['verts']), bool)
    active[regions['moved_render_vertex_ids']] = True
    if not np.array_equal(original['verts'][~active], repaired['verts'][~active]):
        raise ValueError('Protected exterior changed')
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    faces = original['faces']
    patch_faces = active[faces].any(1)
    fig, axes = plt.subplots(3, 3, figsize=(13, 13), facecolor='#f5f4f0')
    labels = ('正面', '斜侧', '侧面')
    for col, angle in enumerate((0., 45., 90.)):
        rad = np.deg2rad(angle)
        R = np.array([[np.cos(rad), 0., -np.sin(rad)], [0., 1., 0.],
                      [np.sin(rad), 0., np.cos(rad)]])
        before, after = original['verts']@R.T, repaired['verts']@R.T
        all_v = np.vstack([before, after])
        low, high = all_v.min(0), all_v.max(0)
        span = max(high[1]-low[1], high[0]-low[0])*1.08
        cx, cy = (high[0]+low[0])/2, (high[1]+low[1])/2
        for row, vertices in enumerate((before, after, after)):
            ax = axes[row, col]
            ax.set_facecolor('#e8e7e2')
            if row < 2:
                shade(ax, vertices, faces, 0, 2)
            else:
                triangles = vertices[faces]
                n = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                order = np.argsort(triangles[:, :, 2].mean(1))
                order = order[n[order, 2] > 0]
                colors = np.tile(np.array([.72, .73, .74]), (len(faces), 1))
                colors[patch_faces] = [.94, .55, .22]
                ax.add_collection(PolyCollection(triangles[order][:, :, [0, 1]],
                    facecolors=colors[order], edgecolors='none', antialiased=False))
            ax.set_xlim(cx-span/2, cx+span/2)
            ax.set_ylim(cy-span/2, cy+span/2)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(labels[col], fontproperties=font, fontsize=14)
            if col == 0:
                ax.set_ylabel(('修正前', '局部修正后', '允许修改的区域')[row], fontproperties=font, fontsize=13)
    fig.suptitle('保留整体头形 · 只重整局部细节', fontproperties=font, fontsize=18)
    fig.text(.5, .018, '橙色：修改区域涉及的面；灰色区域所有顶点完全不动。实际网格示意，无贴图，未安装游戏。',
             ha='center', fontproperties=font, fontsize=11)
    fig.subplots_adjust(left=.055, right=.985, top=.94, bottom=.04, hspace=.04, wspace=.04)
    out.mkdir(parents=True)
    fig.savefig(out/'before_after_regions.png', dpi=130); plt.close(fig)
    save_json(out/'receipt.json', dict(base=source_file(base/'receipt.json'),
        candidate=source_file(candidate/'receipt.json'), regions=source_file(candidate/'local_patch_regions.json'),
        renderer=source_file(Path(__file__)), exterior_unchanged_exactly=True,
        common_scale=True, screenshot=False, game_mutated=False))
    print(str(out.resolve()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    review(args.base.resolve(), args.candidate.resolve(), args.out.resolve())
