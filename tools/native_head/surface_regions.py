"""Topology-bound semantic regions and authored UV correspondences, not detection.

All region membership is computed on the decoder's fixed reference topology.
Identity shape, placement, and the final neck fairing cannot change these labels.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath

from tools.model_bridge.artifact import ModelArtifact, host_path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(source, uv_template):
    model = ModelArtifact.from_game_source(source['source'])
    mask_info = source['surface']['components']['source_mask']
    mask_path = host_path(mask_info['path'])
    if digest(mask_path.read_bytes()) != mask_info['sha256']:
        raise ValueError('Source region masks changed')
    masks = pickle.loads(mask_path.read_bytes(), encoding='latin1')
    lines = uv_template.read_text().splitlines()
    uv = np.array([[float(x) for x in line.split()[1:3]]
                   for line in lines if line.startswith('vt ')])
    records = [line.split()[1:] for line in lines if line.startswith('f ')]
    faces = np.array([[int(x.split('/')[0])-1 for x in row] for row in records])
    uv_faces = np.array([[int(x.split('/')[1])-1 for x in row] for row in records])
    if not np.array_equal(faces, model.faces):
        raise ValueError('UV template is not the exact decoder topology/order')
    embedding = next(x for x in source['source']['decoder_sources']
                     if x['path'].endswith('/landmark_embedding.npy'))
    path = host_path(embedding['path'])
    if digest(path.read_bytes()) != embedding['sha256']:
        raise ValueError('Decoder landmark embedding changed')
    data = np.load(path, allow_pickle=True, encoding='latin1').item()
    indices = np.asarray(data['full_lmk_faces_idx'])[0]
    bary = np.asarray(data['full_lmk_bary_coords'])[0]
    corner_uv = uv[uv_faces]
    landmarks = (corner_uv[indices]*bary[:, :, None]).sum(1)
    centers = corner_uv.mean(1)
    membership = {name: np.isin(faces, ids).mean(1) for name, ids in masks.items()}
    # 'lips' is a broad source mask including perioral skin. Keep it separately.
    outer = PolygonPath(np.r_[landmarks[48:60], landmarks[48:49]])
    inner = PolygonPath(np.r_[landmarks[60:68], landmarks[60:61]])
    mouth = inner.contains_points(centers)
    vermilion = outer.contains_points(centers) & ~mouth & (membership['lips'] > 0)
    membership['lip_vermilion'] = vermilion.astype(float)
    membership['mouth_inner'] = mouth.astype(float)
    membership['perioral_skin'] = (membership['lips'] > 0) & ~vermilion & ~mouth
    report = dict(format='flame_topology_surface_regions_v1',
        topology_sha256=digest(np.asarray(faces, '<i4').tobytes()),
        uv_template_sha256=digest(uv_template.read_bytes()),
        masks_sha256=mask_info['sha256'], embedding_sha256=embedding['sha256'],
        rule='Reference UV triangle-centroid partition of decoder outer/inner lip contours; broad masks retain vertex membership fractions',
        boundary_policy='Whole-face centroid labels; contour-crossing triangles remain boundaries, not exact subtriangle segmentation',
        identity_dependent=False,
        regions={k: dict(face_ids=np.flatnonzero(v > 0).tolist()) for k, v in membership.items()},
        lip_outer_landmarks=list(range(48, 60)), lip_inner_landmarks=list(range(60, 68)))
    if not vermilion.any() or not membership['perioral_skin'].any():
        raise ValueError('Lip and perioral partitions must both exist')
    return corner_uv, landmarks, membership, report


def inherited_faces(arrays, face_keep, regions):
    """Compose refinement -> crop -> decoder face indices; never infer by position."""
    refined = np.asarray(arrays['source_parent_faces'], int)
    original = np.asarray(arrays['crop_parent_faces'], int)[refined]
    parents = np.full(len(arrays['faces']), -1, int)
    parents[:len(original)] = original
    selected = parents[face_keep]
    labels = {}
    for name, values in regions.items():
        out = np.zeros(len(selected), float)
        valid = selected >= 0
        out[valid] = values[selected[valid]]
        labels[name] = np.flatnonzero(out > 0)
    labels['authored_neck_transition'] = np.flatnonzero(selected < 0)
    return selected, labels


def save(report, parents, labels, out):
    report = dict(report, integrated_parent_face_ids=parents.tolist(),
                  integrated_regions={k: v.tolist() for k, v in labels.items()},
                  lineage='decoder face -> crop_parent_faces -> source_parent_faces -> actual o_head face')
    (out/'surface_regions.json').write_text(json.dumps(report, indent=2)+'\n')
