"""Prepare accepted integrated placement geometry for the actual native prefab.

Position arrays are literal authoring outputs. Nearest-triangle projection is
used only for native surface attributes, never for reshaping the source face.
"""
import numpy as np

from tools.native_head.placement_review import normals


def prepare(arrays, design, original_labels, native_vertices, template, correspondence):
    if design['format'] != 'source_head_similarity_review_v1':
        raise ValueError('Accepted similarity review geometry required')
    v, f = arrays['vertices'], arrays['faces']
    width = len(arrays['source_ring'])
    native_count = len(arrays['native_original_ids'])
    native_start = len(v) - 5 * width - native_count
    if native_start != design['source_prefix_count']:
        raise ValueError('Integrated source/collar/bridge layout changed')
    native_ids = np.arange(native_start, native_start + native_count)
    if not np.isin(arrays['native_outer_ring_authored_ids'], native_ids).all():
        raise ValueError('Native interface is outside its actual collar')
    recipes = arrays['crop_recipes']; a, b = recipes[:, :2].astype(int).T
    if not np.array_equal(original_labels[a], original_labels[b]):
        raise ValueError('Crop recipe crosses source components')
    labels = np.zeros(len(v), int); labels[:len(recipes)] = original_labels[a]
    if not np.all(labels[f] == labels[f[:, :1]]):
        raise ValueError('Integrated mesh crosses eye/head components')
    n = normals(v, f)
    raw_n = normals(arrays['placed_original_vertices'], arrays['original_faces'])
    original_ids = arrays['crop_original_ids']
    retained = np.flatnonzero((original_ids >= 0) & (arrays['posterior_fade'][:len(original_ids)] == 0))
    n[retained] = raw_n[original_ids[retained]]
    # Interpolate directions from the full actual native surface, not an
    # isolated collar. This authors normals/UVs without altering its positions.
    collar_reference = arrays['before_vertices'][native_ids]
    triangle_ids, bary, projected = correspondence(collar_reference, native_vertices, template['faces'])
    if np.linalg.norm(projected - collar_reference, axis=1).max() > 1e-8:
        raise ValueError('Retained collar no longer lies on the audited native surface')
    corners = template['faces'][triangle_ids]
    attrs = {k: np.sum(template[k][corners] * bary[:, :, None], axis=1)
             for k in ('uv', 'uv1', 'colors')}
    policy = {'method': 'exact_retained_native_surface_barycentric_attributes',
              'positions_modified': False}
    native_n = np.sum(template['normals'][corners] * bary[:, :, None], axis=1)
    native_n /= np.linalg.norm(native_n, axis=1, keepdims=True)
    # The reviewed posterior fairing deliberately reshaped part of this band.
    # Those points keep final-surface normals; the unchanged seam uses the
    # original native field. UV provenance remains on its original collar.
    unchanged = arrays['posterior_fade'][native_ids] == 0
    n[native_ids[unchanged]] = native_n[unchanged]
    if not np.isfinite(n).all() or not np.allclose(np.linalg.norm(n, axis=1), 1, atol=2e-6):
        raise ValueError('Integrated render normals are not finite unit vectors')
    return labels, n, native_start, native_ids, attrs, policy
