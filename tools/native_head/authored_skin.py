"""Basic anatomical pigment on the unchanged imported head, not identity albedo.

The native atlas is not anatomically interchangeable with FLAME. This temporary
skin uses a continuous front cylindrical UV chart and FLAME's lip landmarks.
The back chart seam has uniform pigment; no head positions are fitted or moved.
"""
import hashlib

import numpy as np
from PIL import Image, ImageFilter, ImageDraw


def author(vertices, out, source, raw_vertices, size=2048):
    vertices = np.asarray(vertices, float)
    center = (vertices.min(0) + vertices.max(0))/2
    angle = np.arctan2(vertices[:, 0]-center[0], vertices[:, 2]-center[2])
    uv = np.column_stack([angle/(2*np.pi)+.5,
        (vertices[:, 1]-vertices[:, 1].min())/np.ptp(vertices[:, 1])])
    # FLAME's broad 'lips' mask includes surrounding mouth skin. The decoder's
    # exact 68-landmark embedding defines the vermilion outline more narrowly.
    from tools.model_bridge.artifact import host_path
    record = next(x for x in source['source']['decoder_sources'] if x['path'].endswith('/landmark_embedding.npy'))
    embedding_path = host_path(record['path'])
    if hashlib.sha256(embedding_path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Source landmark embedding changed')
    embedding = np.load(embedding_path, allow_pickle=True, encoding='latin1').item()
    raw_faces = np.asarray(source['triangles']).reshape(-1, 3)
    landmarks = np.sum(np.asarray(raw_vertices)[raw_faces[embedding['full_lmk_faces_idx'][0]]] *
        embedding['full_lmk_bary_coords'][0, :, :, None], axis=1)
    lips = landmarks[48:60]
    outline = np.column_stack([np.arctan2(lips[:, 0]-center[0], lips[:, 2]-center[2])/(2*np.pi)+.5,
        (lips[:, 1]-vertices[:, 1].min())/np.ptp(vertices[:, 1])])
    pixels = outline*(size-1); pixels[:, 1] = size-1-pixels[:, 1]
    raster = Image.new('L', (size, size)); ImageDraw.Draw(raster).polygon([tuple(p) for p in pixels], fill=255)
    mask = np.asarray(raster.filter(ImageFilter.GaussianBlur(2)), float)/255
    # Generic native-skin median and subdued authored lip tint. These are design
    # choices, not MICA outputs or a reconstruction of the person's complexion.
    skin = np.array([211, 180, 165.]); lip = np.array([188, 128, 124.])
    rgb = skin[None, None, :] + mask[:, :, None]*(lip-skin)
    rgba = np.concatenate([np.uint8(np.clip(rgb, 0, 255)), np.full((size, size, 1), 255, np.uint8)], axis=-1)
    image = Image.fromarray(rgba)
    image.save(out/'authored_albedo.png')
    return uv, image, dict(method='front_cylindrical_chart_with_source_FLAME_outer_lip_landmarks',
        positions_modified=False,
        landmark_embedding_sha256=record['sha256'], lip_outline_indices=list(range(48, 60)),
        identity_albedo=False, native_anatomical_atlas_reused=False,
        normal_map='flat DXT5nm', occlusion_map='neutral',
        limitation='Basic skin/lips only; no photo albedo, pores, eye makeup or eyebrow reconstruction')
