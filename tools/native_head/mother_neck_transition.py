"""Re-author the complete native neck collar with clamped surface directions.

Adapts the displacement fairing in posterior_neck and the two-surface endpoint
contract in placement_review to intact native topology. This is mesh authoring,
not a replacement for HS2 skinning. No faces, UVs or expression rows are removed.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve

from tools.native_head.audit_neck_shading import seam_pairs, source_frame
from tools.native_head.mother_surface_warp import SurfaceWarp
from tools.native_head.mother_template_inputs import save_json, source_file
from tools.model_bridge.artifact import sha


def collar(original, authored, faces, seam, protected, lift, rings=7):
    """Clamped biharmonic displacement: native lower derivative, rigid exterior.

    Fix two native rows at the bottom, two authored rows at the top. Unlike a
    positional rim pin, this retains the outgoing native surface direction.
    Solve displacement, rather than absolute positions, to retain native collar
    detail. Positive graph weights avoid negative cotangents on narrow triangles.
    Ring width and lift are explicit authoring choices, never empirical game gains.
    """
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]],
        faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    a, b = edges.T
    length = np.linalg.norm(original[a]-original[b], axis=1)
    if np.any(length <= 0):
        raise ValueError('Collapsed logical native edge')
    graph = sparse.coo_matrix((np.ones(len(a)*2),
        (np.r_[a, b], np.r_[b, a])), shape=(len(original),)*2).tocsr()
    distance = dijkstra(graph, directed=False, indices=seam,
                        unweighted=True, min_only=True)
    lower = distance <= 1
    upper = (distance >= rings-1) | protected
    if np.any(lower & protected):
        raise ValueError('Native tangent row intersects a protected feature')
    weights = sparse.coo_matrix((np.r_[1/length, 1/length],
        (np.r_[a, b], np.r_[b, a])), shape=graph.shape).tocsr()
    degree = np.asarray(weights.sum(1)).ravel()
    L = sparse.diags(1/degree)@(sparse.diags(degree)-weights)
    energy = L.T@L
    free = ~(lower | upper)
    fixed = ~free
    displacement = authored-original+lift
    displacement[lower] = 0
    displacement[free] = spsolve(energy[free][:, free].tocsc(),
        -energy[free][:, fixed]@displacement[fixed])
    result = original+displacement
    result[lower] = original[lower]
    result[upper] = authored[upper]+lift
    if not np.isfinite(result).all():
        raise ValueError('Unconstrained native collar component')
    if not np.array_equal(result[seam], original[seam]):
        raise ValueError('Body interface moved')
    return result, dict(distance=distance, lower=lower, upper=upper, free=free)


def author(candidate, capture, out, rings=7, lift_fraction=.25):
    if out.exists():
        raise FileExistsError('Preserve earlier candidate; use a fresh output')
    receipt = json.loads((candidate/'receipt.json').read_text())
    if sha(receipt['candidate_arrays']['path']) != receipt['candidate_arrays']['sha256']:
        raise ValueError('Candidate changed')
    arrays = dict(np.load(candidate/'o_head_candidate.npz', allow_pickle=False))
    inputs = Path(receipt['inputs']['path']).parent
    regions = json.loads((inputs/'native_regions.json').read_text())
    actual = json.loads(capture.read_text(encoding='utf-8-sig'))
    h, b = [next(m for m in actual['meshes'] if m['mesh_name']==n)
            for n in ('o_head', 'o_body_cf')]
    ring, _, contract = seam_pairs(h, b)
    if set(ring) != set(regions['physical_neck_boundary']):
        raise ValueError('Runtime interface differs from prepared native topology')
    for key, exported in [('faces', 'triangles'), ('uv', 'uv')]:
        if not np.array_equal(arrays[key].astype(np.float32),
                              np.asarray(h['source'][exported]).reshape(arrays[key].shape).astype(np.float32)):
            raise ValueError('Actual head has a different '+key)
    canon = np.array([int(regions['graph_aliases'].get(str(i), i))
                      for i in range(len(arrays['verts']))])
    unique, inverse = np.unique(canon, return_inverse=True)
    original, before = arrays['original_vertices'][unique], arrays['verts'][unique]
    faces = inverse[arrays['faces']]
    names = arrays['bone_names'].tolist()
    # Native source-supported features are hard rigid locks. Jaw underside is
    # deliberately free; lip/ear/eye/nose geometry never participates in fairing.
    bones = [i for i, name in enumerate(names)
             if any(s in name for s in ('Ear', 'Eye', 'Mouth', 'Nose', 'Mayu'))]
    support = (arrays['bone_w']*np.isin(arrays['bone_idx'], bones)).sum(1)
    protected = support[unique] > 0
    protected[inverse[regions['inner_mouth_component_vertex_ids']]] = True
    ear_bones = [i for i, name in enumerate(names) if 'Ear' in name]
    ear_ids = np.flatnonzero((arrays['bone_w']*
        np.isin(arrays['bone_idx'], ear_bones)).sum(1) > .5)
    if not len(ear_ids):
        raise ValueError('Native ear support missing')
    lift = np.array([0., np.ptp(arrays['verts'][ear_ids, 1])*lift_fraction, 0.])
    result, domain = collar(original, before, faces, np.unique(inverse[ring]),
                            protected, lift, rings)
    final = result[inverse]
    # The closed-reference and FLAME reference move together; secondary parts
    # and eye joints must be adapted from the new placement, not left behind.
    revised = {**arrays, 'verts': final,
               'reference_vertices': arrays['reference_vertices']+lift}
    field = SurfaceWarp(arrays['original_vertices'], final, arrays['faces'])
    field.vertex_gradients({int(k):v for k,v in regions['graph_aliases'].items()})
    out.mkdir(parents=True)
    np.savez_compressed(out/'o_head_candidate.npz', **revised)
    save_json(out/'neck_domain.json', dict(
        lower_native_render_ids=np.flatnonzero(domain['lower'][inverse]).tolist(),
        free_render_ids=np.flatnonzero(domain['free'][inverse]).tolist(),
        rigid_exterior_render_ids=np.flatnonzero(domain['upper'][inverse]).tolist(),
        protected_feature_render_ids=np.flatnonzero(protected[inverse]).tolist(),
        all_render_copies_included=True, body_interface=contract))
    updated = copy.deepcopy(receipt)
    updated['placement']['translation'] = (np.array(receipt['placement']['translation'])+lift).tolist()
    updated.update(candidate_arrays=source_file(out/'o_head_candidate.npz'),
        code=source_file(Path(__file__)), neck_transition=dict(
            input_candidate=source_file(candidate/'receipt.json'),
            actual_head_body=source_file(capture), domain=source_file(out/'neck_domain.json'),
            body_geometry_sha256=b['source_geometry_sha256'], lift=lift.tolist(),
            lift_fraction_of_ear_height=lift_fraction, transition_rings=rings,
            method='Clamped biharmonic native-collar displacement; two endpoint rows on each side',
            previous_rules=['placement_review.rebuilt_baseline two-surface directions',
                            'posterior_neck.fair_posterior squared-Laplacian patch'],
            front_and_rear=True, native_lower_rows_literal=True,
            outside_collar_rigid_translation_only=True, topology_unchanged=True),
        installed=False, game_mutated=False, deliverable=False)
    save_json(out/'receipt.json', updated)
    review(arrays['verts'], final, arrays['faces'], b, out)
    print(json.dumps(dict(output=str(out.resolve()), lift=lift.tolist(),
                          front_and_rear=True, topology_unchanged=True)))


def review(before, after, faces, body, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from tools.native_head.neck_section_review import sections
    bp, _ = source_frame(body, 'cf_J_Head_s')
    bf = np.asarray(body['source']['triangles']).reshape(-1, 3)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax in axes:
        for v, f, color, label in [(before, faces, '#a28d7e', 'Before'),
            (after, faces, '#28964b', 'Authored native head'),
            (bp, bf, '#2d78bd', 'Unchanged BP body')]:
            ax.add_collection(LineCollection(sections(v, f), color=color, label=label))
        ax.set_aspect('equal'); ax.grid(); ax.set_xlabel('Front/back'); ax.set_ylabel('Height')
    axes[0].set_xlim(-1.3, 1.3); axes[0].set_ylim(-1.2, 1.9); axes[0].legend(fontsize=8)
    axes[1].set_xlim(-1.1, -.3); axes[1].set_ylim(-.4, .8); axes[1].set_title('Rear neck')
    axes[2].set_xlim(.2, 1.15); axes[2].set_ylim(-.8, .1); axes[2].set_title('Under jaw')
    fig.tight_layout(); fig.savefig(out/'complete_neck_sections.png', dpi=150); plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rings', type=int, default=7)
    p.add_argument('--lift-fraction', type=float, default=.25)
    a=p.parse_args()
    if a.rings<4 or not np.isfinite(a.lift_fraction) or a.lift_fraction<0:
        raise ValueError('At least four rings and finite nonnegative lift required')
    author(a.candidate.resolve(), a.capture.resolve(), a.out.resolve(), a.rings, a.lift_fraction)
