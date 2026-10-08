"""Static authored/source/body section views; no screenshot-based inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def sections(vertices, faces):
    segments = []
    for tri in vertices[faces]:
        points = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if a[0]*b[0] < 0:
                points.append(a+(b-a)*(-a[0]/(b[0]-a[0])))
        if len(points) == 2:
            segments.append(np.asarray(points)[:, [2, 1]])
    return segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--body', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    a = dict(np.load(args.geometry))
    receipt = json.loads(args.receipt.read_text())
    body = json.loads(args.body.read_text())
    mesh = next(m for m in body['meshes'] if m['mesh_name'] == 'o_body_cf')
    transform = next(t for t in body['transforms'] if t['name'] == 'cf_J_Head_s')
    if mesh['renderer_lossy_scale'] != [1., 1., 1.]:
        raise ValueError('Captured BakeMesh scale convention requires unit renderer')
    world = np.c_[mesh['baked']['vertices'], np.ones(mesh['vertex_count'])] @ np.asarray(mesh['renderer_local_to_world']).reshape(4, 4).T
    bv = (world @ np.asarray(transform['world_to_local']).reshape(4, 4).T)[:, :3]
    bf = np.asarray(mesh['source']['triangles']).reshape(-1, 3)
    original = a['original_vertices']*receipt['scale']+np.asarray(receipt['translation'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for ax in axes:
        ax.add_collection(LineCollection(sections(original, a['original_faces']), color='#bb9984', linewidth=1, label='Original complete FLAME'))
        label='Posterior edit; front/upper skull unchanged' if receipt.get('posterior_only') else 'Authored neck; retained face/skull unchanged'
        ax.add_collection(LineCollection(sections(a['vertices'], a['faces']), color='#3a954a', linewidth=1.7, label=label))
        ax.add_collection(LineCollection(sections(bv, bf), color='#3272aa', linewidth=1.4, label='Unmodified captured BP body'))
        ax.set_aspect('equal'); ax.grid(); ax.set_xlabel('Front/back'); ax.set_ylabel('Height')
    axes[0].set_xlim(-1.1, 1.5); axes[0].set_ylim(-1.3, 1.9)
    axes[1].set_xlim(-.7, 1.1); axes[1].set_ylim(-.8, .4)
    axes[0].legend(loc='upper left', fontsize=7)
    fig.tight_layout(); fig.savefig(args.output, dpi=135); plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
