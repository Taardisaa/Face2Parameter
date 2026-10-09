"""Show complete native/FLAME/candidate geometry; scientific plots, not screenshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.font_manager import FontProperties
import numpy as np

from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json, source_file


def section(vertices, faces):
    segments = {}
    def add(a, b):
        if np.array_equal(a, b):
            return
        points = np.asarray([a, b], dtype="<f8")
        key = tuple(sorted(p.tobytes() for p in points))
        segments[key] = points[:, [2, 1]]
    for tri in vertices[faces]:
        points = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if a[0] == b[0] == 0:
                add(a, b)
            elif a[0] == 0:
                points.append(a)
            elif b[0] == 0:
                points.append(b)
            elif a[0]*b[0] < 0:
                points.append(a+(b-a)*(-a[0]/(b[0]-a[0])))
        if points:
            unique = np.unique(points, axis=0)
            if len(unique) == 2:
                add(*unique)
    return list(segments.values())


def shade(ax, vertices, faces, horizontal, depth):
    tri = vertices[faces]
    normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), np.finfo(float).tiny)
    light = np.zeros(3)
    light[depth], light[1], light[horizontal] = .8, .5, -.35
    light /= np.linalg.norm(light)
    rgb = np.array([.75, .71, .65])[None]*(.3+.7*np.maximum(normal@light, 0))[:, None]
    order = np.argsort(tri[:, :, depth].mean(1))
    order = order[normal[order, depth] > 0]
    ax.add_collection(PolyCollection(tri[order][:, :, [horizontal, 1]],
        facecolors=rgb[order], edgecolors="none", antialiased=False))


def review(candidate_dir, out):
    if out.exists():
        raise FileExistsError("Keep earlier renderings; use a fresh directory")
    arrays = dict(np.load(candidate_dir/"o_head_candidate.npz", allow_pickle=False))
    receipt = json.loads((candidate_dir/"receipt.json").read_text(encoding="utf-8"))
    if sha(receipt["inputs"]["path"]) != receipt["inputs"]["sha256"]:
        raise ValueError("Candidate input receipt changed")
    out.mkdir(parents=True)
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    meshes = [(arrays["original_vertices"], arrays["faces"]),
              (arrays["reference_vertices"], arrays["reference_faces"]),
              (arrays["verts"], arrays["faces"])]
    original_title = "原生 02 号\n默认闭嘴 / 睁眼参考" if 'default_reference' in receipt else "原生 02 号\n原始 bind 网格"
    titles = [original_title, "FLAME 零身份参考\n仅全局摆放", "临时头壳候选\n完整部件与表情适配待做"]
    all_v = np.vstack([v for v, f in meshes])
    low, high = all_v.min(0), all_v.max(0)
    span = (high[1]-low[1])*1.07
    cy = (high[1]+low[1])/2
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.5), facecolor="#f5f4f0")
    for col, (v, f) in enumerate(meshes):
        for row, (horizontal, depth) in enumerate(((0, 2), (2, 0))):
            ax = axes[row, col]
            ax.set_facecolor("#e8e7e2")
            shade(ax, v, f, horizontal, depth)
            cx = (high[horizontal]+low[horizontal])/2
            ax.set_xlim(cx-span*.44, cx+span*.44)
            ax.set_ylim(cy-span*.5, cy+span*.5)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(titles[col], fontproperties=font, fontsize=13, pad=14)
            if col == 0:
                ax.set_ylabel("正面" if row == 0 else "侧面", fontproperties=font, fontsize=12)
    fig.text(.5, .02, "实际网格的正交示意 · 相同比例 · 无贴图 · 不是游戏截图 · 候选未安装",
             fontproperties=font, ha="center", fontsize=11)
    fig.subplots_adjust(left=.035, right=.98, top=.91, bottom=.06, wspace=.08, hspace=.03)
    fig.savefig(out/"complete_shells.png", dpi=145)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 10))
    names = ["原生 02", "FLAME 零身份", "临时头壳"]
    for (v, f), color, name in zip(meshes, ("#607d8b", "#be7628", "#218f70"), names):
        ax.add_collection(LineCollection(section(v, f), colors=color, linewidths=1.4, label=name))
    ax.set_xlim(low[2]-.05*span, high[2]+.05*span)
    ax.set_ylim(low[1]-.05*span, high[1]+.05*span)
    ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("后 / 前（原生模型单位）", fontproperties=font)
    ax.set_ylabel("高度（原生模型单位）", fontproperties=font)
    ax.set_title("完整中央剖面 · 包括内部口腔交线", fontproperties=font)
    ax.legend(prop=font)
    fig.tight_layout(); fig.savefig(out/"complete_sections.png", dpi=145); plt.close(fig)
    save_json(out/"receipt.json", dict(format="mother_shell_geometry_review_v1",
        candidate=source_file(candidate_dir/"receipt.json"),
        arrays=source_file(candidate_dir/"o_head_candidate.npz"),
        renderer=source_file(Path(__file__)), screenshot=False, game_mutated=False,
        geometry_transforms_added=False, display_culling="Painter-sorted front-facing triangles for visual inspection only",
        section_policy="Exact X=0 triangle intersections, including edges on the plane; coincident lines deduplicated",
        limitations=receipt["pending"], deliverable=False))
    print(str(out.resolve()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    review(args.candidate.resolve(), args.out.resolve())
