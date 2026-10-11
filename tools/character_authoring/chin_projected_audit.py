"""Bind actual source vertex identity to a recorded portrait projection.

Projection diagnostic uses reported frozen head TRS and the existing main-camera
capture's source-derived perspective framing. It excludes CTAA jitter, shader
displacement, depth visibility and unreported transform shear certification.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def load(path):
    raw = path.read_bytes()
    return json.loads(raw.decode('utf-8-sig')), hashlib.sha256(raw).hexdigest()


def project(geometry: Path, capture: Path, width=768, height=1024):
    d, geometry_sha = load(geometry)
    response, capture_sha = load(capture)
    c = response['capture']
    if c['pipeline'] != 'actual_main_camera' or c['projection'] != 'perspective':
        raise ValueError('Require the actual main-camera perspective response')
    matches = [m for m in d['meshes'] if m['mesh_name'] == 'o_head']
    if len(matches) != 1:
        raise ValueError('Require one native head mesh')
    m = matches[0]
    heads = [t for t in d['transforms'] if t['name'] == 'cf_J_Head']
    if len(heads) != 1:
        raise ValueError('Require actual head matrix')
    head = heads[0]
    q = Rotation.from_quat(head['world_rotation_xyzw']).as_matrix()
    actual = np.asarray(head['local_to_world']).reshape(4, 4)
    represented = q @ np.diag(head['lossy_scale'])
    # Preserve the matrix/TRS discrepancy rather than certifying arbitrary shear.
    representation_error = float(np.abs(actual[:3, :3] - represented).max())
    v = np.c_[m['baked']['vertices'], np.ones(m['vertex_count'])]
    head_vertices = v @ np.asarray(m['baked']['world_candidates']['scale_free_trs']['matrix']).reshape(4, 4).T
    head_vertices = (head_vertices @ np.asarray(head['world_to_local']).reshape(4, 4).T)[:, :3]
    pose = c['head_pose']
    canonical = Rotation.from_quat(pose['rotation_xyzw']).as_matrix() @ np.diag(pose['scale'])
    world = head_vertices @ canonical.T + pose['position']
    fov = float(d['main_camera']['fov_degrees'])
    if not 0 < fov < 180:
        raise ValueError('Invalid actual exported FOV')
    tan = np.tan(np.deg2rad(fov / 2))
    distance = c['ortho_size'] / tan
    theta = np.deg2rad(c['yaw'])
    xaxis = np.array([np.cos(theta), 0, -np.sin(theta)])
    zaxis = np.array([np.sin(theta), 0, np.cos(theta)])
    rel = world - c['target']
    depth = distance - rel @ zaxis
    if np.any(depth <= 0):
        raise ValueError('A mesh vertex lies behind the diagnostic capture camera')
    focal = height / 2 / tan
    pixels = np.c_[width / 2 - (rel @ xaxis) * focal / depth,
                   height / 2 - rel[:, 1] * focal / depth]
    source = np.asarray(m['source']['vertices'])
    selected = np.flatnonzero((source[:, 1] < -.30) & (source[:, 2] > .50))
    if not len(selected):
        raise ValueError('Empty recorded front-bottom source selection')
    lowest = int(selected[pixels[selected, 1].argmax()])
    indices = m['source']['bone_indices']
    weights = m['source']['bone_weights']
    def row(i):
        return {'index': i, 'source_xyz': source[i].tolist(),
                'head_xyz': head_vertices[i].tolist(), 'projected_xy': pixels[i].tolist(),
                'weights': [{'bone': m['bone_names'][j], 'weight': w}
                            for j, w in zip(indices[i], weights[i]) if w]}
    return {'scope': __doc__, 'geometry': {'path': str(geometry), 'sha256': geometry_sha},
            'capture': {'path': str(capture), 'sha256': capture_sha},
            'image': {'path': c['path'], 'sha256': hashlib.sha256(Path(c['path']).read_bytes()).hexdigest()},
            'source_geometry_sha256': m['source_geometry_sha256'],
            'fov_from_export_degrees': fov, 'resolution': [width, height],
            'export_head_matrix_vs_TRS_max_component_error': representation_error,
            'camera_scope': 'FOV/resolution are recorded inputs, not fields reported by older capture responses. Canonical head TRS reconstruction; no CTAA jitter/shader or arbitrary-shear pixel certification.',
            'source_selection': 'Y<-.30,Z>.50 in actual shared mesh; diagnostic selection, not an anatomical/occlusion label',
            'projected_most_inferior_selected_vertex': row(lowest),
            'source_min_y_selected_vertex': row(int(selected[source[selected, 1].argmin()])),
            'all_projected_xy': pixels.tolist(), 'all_head_xyz': head_vertices.tolist(),
            'selected_indices': selected.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--figure', type=Path)
    a = p.parse_args()
    if not 64 <= a.width <= 4096 or not 64 <= a.height <= 4096:
        raise ValueError('Invalid recorded resolution')
    if a.out.exists() or (a.figure is not None and a.figure.exists()):
        raise FileExistsError('Fresh audit and figure paths required')
    r = project(a.geometry, a.capture, a.width, a.height)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(r, f, indent=2)
    if a.figure is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.imshow(Image.open(r['image']['path']))
        for key, color, text_offset in [('projected_most_inferior_selected_vertex', '#31d6db', (8, -18)),
                                       ('source_min_y_selected_vertex', '#ffbe45', (8, 14))]:
            row = r[key]
            x, y = row['projected_xy']
            ax.plot(x, y, '+', color=color, markersize=13, markeredgewidth=2)
            ax.annotate(str(row['index']), (x, y), xytext=text_offset,
                        textcoords='offset points', color=color, fontsize=11)
        low = r['projected_most_inferior_selected_vertex']['projected_xy']
        ax.set_xlim(low[0]-100, low[0]+100)
        ax.set_ylim(low[1]+25, low[1]-110)
        ax.set_title('Projected source IDs on actual main-camera image\nTRS diagnostic; excludes CTAA jitter/shader visibility')
        ax.set_xlabel('image X pixels'); ax.set_ylabel('image Y pixels')
        a.figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.figure, dpi=160)
        plt.close(fig)
    print(json.dumps({k: r[k] for k in ['projected_most_inferior_selected_vertex',
                      'source_min_y_selected_vertex', 'export_head_matrix_vs_TRS_max_component_error']}, indent=2))


if __name__ == '__main__':
    main()
