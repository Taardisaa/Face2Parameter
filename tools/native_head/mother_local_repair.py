"""Repair small folded patches without refitting the accepted head silhouette.

This is asset authoring, not a replacement for the game's deformation logic.
Source default-pose detail supplies the rest geometry inside documented patches;
every vertex outside those patches is an exact Dirichlet constraint. No FLAME
closest-point queries or whole-head displacement field are used here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import splu

from tools.model_bridge.artifact import sha
from tools.native_head.mother_arap import SpokesARAP
from tools.native_head.mother_template_inputs import save_json, source_file


def similarity(source, target):
    """Proper uniform similarity fitted only to each patch's fixed boundary."""
    a, b = source.mean(0), target.mean(0)
    x, y = source-a, target-b
    u, _, vt = np.linalg.svd(x.T@y)
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u@vt)
    rotation = (u*correction)@vt
    denominator = np.sum(x*x)
    if denominator == 0:
        raise ValueError('Collapsed patch boundary')
    scale = np.sum((x@rotation)*y)/denominator
    if scale <= 0:
        raise ValueError('Reflected/collapsed patch placement')
    return scale, rotation, b-a@rotation*scale


def local_solve(original, base, faces, active, fixed, *, fidelity=0., iterations=400,
                method='harmonic', detail_core=None):
    """Extend fixed boundary displacements through source detail in each patch.

    Boundary registration places donor detail; it never moves the accepted
    boundary itself. Global indexing is retained. Returned exterior is bitwise
    equal, including independent components not involved in any patch.
    """
    active = np.asarray(active, bool) & ~np.asarray(fixed, bool)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]],
        faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    adjacency = sparse.coo_matrix((np.ones(len(edges)*2),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(base), len(base))).tocsr()
    selected = np.flatnonzero(active)
    count, labels = connected_components(adjacency[selected][:, selected], directed=False)
    result = base.copy()
    notes = []
    for component in range(count):
        free_global = selected[labels == component]
        patch_method = method
        if method == 'hybrid':
            if detail_core is None:
                raise ValueError('Hybrid repair needs explicit source rig support')
            patch_method = 'detail_blend' if np.mean(detail_core[free_global]) > .5 else 'harmonic'
        affected_faces = np.flatnonzero(np.isin(faces, free_global).any(1))
        used = np.unique(faces[affected_faces])
        local_faces = np.searchsorted(used, faces[affected_faces])
        free = np.flatnonzero(np.isin(used, free_global))
        pins = np.setdiff1d(np.arange(len(used)), free)
        if len(pins) < 3:
            raise ValueError('Local patch has insufficient fixed boundary')
        scale, rotation, translation = similarity(original[used[pins]], base[used[pins]])
        rest = original[used]@rotation*scale+translation
        arap = SpokesARAP(rest, local_faces)
        A = arap.stiffness+sparse.eye(len(used))*fidelity
        factor = splu(A[free][:, free].tocsc())
        A_pin = A[free][:, pins]
        if patch_method not in ('harmonic', 'arap', 'detail_blend'):
            raise ValueError('Unknown local repair method')
        # Solve for displacement from the intact donor detail. Unlike another
        # closest-point fit, this retains the relation between its thin layers.
        rhs = arap.stiffness@rest+fidelity*base[used]
        x = rest.copy()
        x[pins] = base[used[pins]]
        if patch_method == 'detail_blend':
            graph = arap.stiffness.copy()
            graph.setdiag(0); graph.eliminate_zeros(); graph.data[:] = 1.
            distance = dijkstra(graph, directed=False, indices=pins, min_only=True,
                                unweighted=True)
            transition_span = min(3., float(distance[free].max()))
            t = np.clip(distance/transition_span, 0., 1.)
            blend = t*t*(3.-2.*t)
            # Common similarity in the core keeps opposed surfaces together;
            # the existing two-ring halo joins it to the fixed accepted skin.
            x[free] = base[used[free]]+(rest[free]-base[used[free]])*blend[free, None]
        else:
            x[free] = factor.solve(rhs[free]-A_pin@base[used[pins]])
        history = []
        for iteration in range(iterations if patch_method == 'arap' else 0):
            rhs = arap.rhs(x)+fidelity*base[used]
            next_free = factor.solve(rhs[free]-A_pin@base[used[pins]])
            change = float(np.linalg.norm(next_free-x[free], axis=1).max())
            x[free] = next_free
            history.append(change)
            if change < 1e-8:
                break
        result[free_global] = x[free]
        notes.append(dict(free_logical_vertex_ids=free_global.tolist(),
            fixed_logical_boundary_ids=used[pins].tolist(),
            affected_face_ids=affected_faces.tolist(),
            boundary_similarity=dict(scale=float(scale), rotation=rotation.tolist(),
                                     translation=translation.tolist()),
            method=patch_method, iterations=len(history), final_max_update=history[-1] if history else 0.))
    if not np.array_equal(result[~active], base[~active]):
        raise ValueError('Local solve moved protected exterior')
    return result, notes


def repair(candidate, quality, out, rings=2):
    if out.exists():
        raise FileExistsError('Preserve earlier candidates; use a new path')
    receipt = json.loads((candidate/'receipt.json').read_text())
    report = json.loads((quality/'receipt.json').read_text())
    if sha(candidate/'o_head_candidate.npz') != report['arrays']['sha256']:
        raise ValueError('Intersection evidence belongs to a different candidate')
    arrays = dict(np.load(candidate/'o_head_candidate.npz', allow_pickle=False))
    inputs = Path(receipt['inputs']['path']).parent
    regions = json.loads((inputs/'native_regions.json').read_text())
    canonical = np.array([int(regions['graph_aliases'].get(str(i), i))
                          for i in range(len(arrays['verts']))])
    unique, inverse = np.unique(canonical, return_inverse=True)
    base, original = arrays['verts'][unique], arrays['original_vertices'][unique]
    faces = inverse[arrays['faces']]
    pairs = np.asarray(report['added_crossings'], int)
    if not pairs.size:
        raise ValueError('No new crossing patches to repair')
    seeds = np.unique(faces[np.unique(pairs)])
    active = np.zeros(len(base), bool)
    active[seeds] = True
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]],
        faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    adjacency = sparse.coo_matrix((np.ones(len(edges)*2),
        (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
        shape=(len(base), len(base))).tocsr()
    for _ in range(rings):
        active |= np.asarray(adjacency@active) > 0
    fixed = np.zeros(len(base), bool)
    protected_ids = regions['neck_boundary_all_render_copies']+[
        i for loop in regions['physical_eye_boundaries'].values() for i in loop]
    fixed[inverse[protected_ids]] = True
    dominant = arrays['bone_names'][arrays['bone_idx'][
        np.arange(len(arrays['verts'])), np.argmax(arrays['bone_w'], axis=1)]]
    mouth_detail = np.asarray([str(name).startswith('cf_J_Mouth') for name in dominant])[unique]
    result, patches = local_solve(original, base, faces, active, fixed,
                                  method='hybrid', detail_core=mouth_detail)
    moved = active & ~fixed
    render_moved = moved[inverse]
    final = result[inverse]
    if not np.array_equal(final[~render_moved], arrays['verts'][~render_moved]):
        raise ValueError('Exterior moved after UV-alias expansion')
    for group in regions['identical_bind_position_and_skin_groups']:
        if not np.all(final[group] == final[group[0]]):
            raise ValueError('UV seam copies diverged')
    out.mkdir(parents=True)
    output = out/'o_head_candidate.npz'
    np.savez_compressed(output, **{**arrays, 'verts': final})
    reopened = dict(np.load(output, allow_pickle=False))
    for key in arrays:
        if key != 'verts' and not np.array_equal(arrays[key], reopened[key]):
            raise ValueError('Local repair changed source data: '+key)
    save_json(out/'local_patch_regions.json', dict(
        seed_logical_ids=seeds.tolist(), ring_count=rings,
        moved_render_vertex_ids=np.flatnonzero(render_moved).tolist(),
        protected_render_vertex_ids=np.flatnonzero(~render_moved).tolist(),
        neck_and_eye_boundary_ids=protected_ids, patches=patches,
        displacement=(final-arrays['verts']).tolist()))
    result_receipt = {**receipt, 'format': 'native_mother_local_repair_v1',
        'candidate_arrays': source_file(output), 'code': source_file(Path(__file__)),
        'local_repair': dict(base_candidate=source_file(candidate/'receipt.json'),
            crossing_evidence=source_file(quality/'receipt.json'),
            region_manifest=source_file(out/'local_patch_regions.json'),
            exterior_unchanged_exactly=True, eye_boundary_unchanged_exactly=True,
            neck_boundary_unchanged_exactly=True,
            whole_head_refit=False, closest_surface_queries=False,
            fidelity_weight=0., patch_halo_rings=rings,
            method='mouth-dominant patch: common donor detail core with smooth halo; other patches: harmonic boundary displacement; fixed exterior',
            geometry_quality_checked=False),
        'installed': False, 'deliverable': False, 'game_mutated': False}
    save_json(out/'receipt.json', result_receipt)
    print(json.dumps(dict(output=str(out.resolve()), outside_patches_unchanged=True,
                         installed=False, deliverable=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    repair(args.candidate.resolve(), args.quality.resolve(), args.out.resolve())
