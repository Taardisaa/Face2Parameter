"""Source OBJ corner UVs and optional explicit PNG appearance, without changing shape.

SMIRK/MICA do not infer identity albedo. PNG input is therefore an explicit
external surface, never labelled a recovered model output or texture truth.
"""
from __future__ import annotations

import base64
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .artifact import sha


def corner_uv(obj, expected_faces, vertex_count):
    uv, faces, corners = [], [], []
    vertices = 0
    for line in obj.read_text(encoding='utf-8').splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'v':
            vertices += 1  # Template positions are deliberately never read.
        elif fields[0] == 'vt':
            uv.append([float(v) for v in fields[1:3]])
        elif fields[0] == 'f':
            if len(fields) != 4:
                raise ValueError('Triangular source OBJ required')
            parts = [v.split('/') for v in fields[1:]]
            if any(len(v) < 2 or not v[1] for v in parts):
                raise ValueError('Each source corner needs its authored UV index')
            faces.append([int(v[0])-1 for v in parts])
            corners.append([int(v[1])-1 for v in parts])
    faces, corners, uv = np.asarray(faces), np.asarray(corners), np.asarray(uv, dtype=np.float32)
    if vertices != vertex_count or not np.array_equal(faces, expected_faces) or uv.ndim != 2 or uv.shape[1] != 2 or (
            not np.isfinite(uv).all() or corners.min() < 0 or corners.max() >= len(uv)):
        raise ValueError('Source OBJ vertex count/topology/corner UV correspondence differs')
    pairs, inverse = np.unique(np.column_stack([faces.reshape(-1), corners.reshape(-1)]), axis=0, return_inverse=True)
    if len(pairs) > 65535 or not np.array_equal(pairs[inverse, 0], faces.reshape(-1)):
        raise ValueError('UV gather changed source face-corner order')
    return {'format': 'flame_corner_uv_v1', 'render_to_canonical': pairs[:, 0].tolist(),
            'render_uv': uv[pairs[:, 1]].tolist(), 'render_triangles': inverse.tolist(),
            'source_obj': {'path': str(obj.resolve()), 'sha256': sha(obj)},
            'policy': 'Authored source corner UVs; exact position/normal gather from canonical model vertices. No template positions or UV welding.'}


def embed_texture(path):
    data = path.read_bytes()
    if len(data) > 4 * 1024 * 1024 or data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Explicit PNG up to 4 MiB required')
    with Image.open(path) as image:
        if image.format != 'PNG' or not 1 <= image.width <= 2048 or not 1 <= image.height <= 2048:
            raise ValueError('Bounded PNG dimensions required')
        rgba = np.asarray(image.convert('RGBA'), dtype=np.uint8)
        if np.any(rgba[:, :, 3] != 255):
            raise ValueError('Opaque texture required; alpha/cutout material branch is not implemented')
    return {'format': 'embedded_srgb_opaque_png_v1', 'png_base64': base64.b64encode(data).decode('ascii'),
            'sha256': hashlib.sha256(data).hexdigest(), 'rgba8_top_down_sha256': hashlib.sha256(rgba.tobytes()).hexdigest(),
            'width': rgba.shape[1], 'height': rgba.shape[0], 'source_path': str(path.resolve()),
            'material': 'unlit_srgb_opaque', 'model_inferred_albedo': False}


def add_surface(payload, obj, texture=None):
    surface = corner_uv(obj, np.asarray(payload['triangles']).reshape(-1, 3), len(payload['vertices']))
    if texture is not None:
        surface['texture'] = embed_texture(texture)
    payload['format'] = 'hs2_source_head_mesh_v3'
    payload['surface'] = surface
    payload['game_support'] = 'Original canonical source geometry/optional rig; authored corner-UV gather and explicit external opaque PNG. Model identity albedo and native eye/mouth appearance are not inferred.'
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True, help='Existing exact v1/v2 head-local artifact; source parameters/rig retained')
    parser.add_argument('--source-obj', type=Path, required=True)
    parser.add_argument('--texture', type=Path, help='Explicit external opaque PNG, not a source model prediction')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.mesh.read_text(encoding='utf-8'))
    if payload.get('format') not in {'hs2_source_head_mesh_v1', 'hs2_source_head_mesh_v2'} or payload.get('geometry_mode') != 'head_local':
        raise ValueError('Exact original head-local source artifact required')
    add_surface(payload, args.source_obj, args.texture)
    payload['surface']['input_artifact'] = {'path': str(args.mesh.resolve()), 'sha256': sha(args.mesh)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(',', ':'))+'\n')
    print(json.dumps({'artifact': str(args.out.resolve()), 'sha256': sha(args.out),
        'canonical_vertices': len(payload['vertices']), 'render_vertices': len(payload['surface']['render_to_canonical']),
        'model_inferred_albedo': False}))


if __name__ == '__main__':
    main()
