"""Small source-ordered contact corrections inside an existing repair mask.

Source-guided separation is followed, when requested, by small harmonic patches
and simultaneous contact constraints on the actual authored planes. Extrinsic
source half-plane order is not an anatomical invariant. Exterior vertices and
original contour pins remain fixed. This is local asset authoring, not recovered
game deformation, a whole-head refit, or full expression certification.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar, minimize, LinearConstraint
from scipy.sparse.linalg import spsolve

from tools.geometry_quality.mesh_quality import triangle_intersection
from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json, source_file
from tools.native_head.mother_arap import SpokesARAP


def axes_for(triangles):
    edges = np.roll(triangles, -1, axis=1)-triangles
    normals = np.cross(edges[:, 0], edges[:, 1])
    axes = [*normals]
    axes.extend(np.cross(edges[0, :, None], edges[1, None]).reshape(-1, 3))
    for normal in normals:
        axes.extend(np.cross(normal, edges.reshape(-1, 3)))
    axes = np.asarray(axes)
    lengths = np.linalg.norm(axes, axis=1)
    return axes[lengths > 1e-12]/lengths[lengths > 1e-12, None]


def separate_pair(source, positions, triangles, movable, rotation, epsilon=1e-8, scale=1.):
    shared = np.intersect1d(triangles[0], triangles[1])
    moving = movable[triangles] & ~np.isin(triangles, shared)
    choices = []
    axis_candidates = np.vstack([axes_for(source[triangles]),
                                 axes_for(positions[triangles])@rotation.T])
    for axis in axis_candidates:
        projected_source = source[triangles]@axis
        for sign in (1., -1.):
            source_p = projected_source*sign
            if source_p[0].max() > source_p[1].min()+epsilon:
                continue  # Never invent an order opposite to the intact donor.
            direction = axis@rotation*sign
            p = positions[triangles]@direction
            source_plane = (source_p[0].max()+source_p[1].min())/2
            # Retain the donor's projected thickness. Merely pushing to a
            # zero-width contact plane can collapse a shared-edge triangle.
            clearance = np.maximum(0., np.stack([source_plane-source_p[0],
                                                 source_p[1]-source_plane]))*scale
            # Shared vertices define the common contact plane, not two copies.
            clearance[np.isin(triangles, shared)] = 0.
            low = (p[0]+clearance[0])[~moving[0]].max() if (~moving[0]).any() else p.min()
            high = (p[1]-clearance[1])[~moving[1]].min() if (~moving[1]).any() else p.max()
            if low > high+epsilon:
                continue
            high = max(high, low)
            def displacement(t):
                d = np.stack([np.minimum(0., t-clearance[0]-p[0]),
                              np.maximum(0., t+clearance[1]-p[1])])
                return np.where(moving, d, 0.)
            if high == low:
                t = low
            else:
                t = minimize_scalar(lambda t: float(np.sum(displacement(t)**2)),
                    bounds=(low, high), method='bounded', options={'xatol': 1e-12}).x
            d = displacement(t)
            cost = float(np.sum(d*d))
            if cost > 1e-20:
                update = d[:, :, None]*direction
                trial = positions[triangles]+update
                normal = np.cross(trial[:, 1]-trial[:, 0], trial[:, 2]-trial[:, 0])
                # This is a static repair of an already folded surface. Its old
                # facing direction must not prevent unfolding. Keep triangles
                # nondegenerate; the complete crossing inventory checks the
                # authored result rather than assuming a local angle guard
                # proves validity.
                if np.all(np.isfinite(normal)) and np.all(np.linalg.norm(normal, axis=1) > 0):
                    choices.append((cost, update))
    if not choices:
        return False
    _, update = min(choices, key=lambda row: row[0])
    # A vertex shared by the two triangles has zero update in both copies.
    for ids, delta in zip(triangles, update):
        positions[ids] += delta
    return True


def dirty_crossings(positions, faces, dirty, exempt):
    """Exact existing narrow phase on dirty triangles against the entire head."""
    triangles = positions[faces]
    low, high = triangles.min(1), triangles.max(1)
    found = set(); tested = set()
    for first in np.flatnonzero(dirty):
        candidates = np.flatnonzero(np.all(low <= high[first]+1e-8, axis=1) &
                                    np.all(high >= low[first]-1e-8, axis=1))
        for second in candidates:
            pair = tuple(sorted((int(first), int(second))))
            if first == second or pair in tested or pair in exempt:
                continue
            tested.add(pair)
            contact = triangle_intersection(*triangles[list(pair)], 1e-8)
            if contact and contact['kind'] in ('proper_crossing', 'coplanar_overlap'):
                found.add(pair)
    return found


def joint_contact_step(source, positions, faces, inventory, movable, rotations, guards=None):
    """Solve the remaining tiny contact cluster together, avoiding pair cycles.

    Only vertices belonging to the actual crossing triangles are variables.
    Triangle area cannot decrease; outside vertices and original contour pins
    are absent from the variable vector. Zero contact clearance is the exact
    nonpenetration condition, not a replacement skin thickness.
    """
    involved = np.unique(faces[np.unique(np.asarray(sorted(inventory)).ravel())])
    free = involved[movable[involved]]
    if not len(free):
        return positions.copy(), dict(success=False, reason='all contact vertices fixed')
    lookup = {int(vertex): i for i, vertex in enumerate(free)}
    rows, bounds = [], []
    current_plane_fallback=[]
    guards = guards or set()
    for pair in sorted(inventory | guards):
        tri_ids = faces[list(pair)]
        u, _, vt = np.linalg.svd(rotations[np.unique(tri_ids)].mean(0)); R = u@vt
        choices = []
        shared = np.intersect1d(*tri_ids)
        moving = movable[tri_ids] & ~np.isin(tri_ids, shared)
        axes = np.vstack([axes_for(source[tri_ids]), axes_for(positions[tri_ids])@R.T])
        for axis in axes:
            for sign in (1., -1.):
                sp = source[tri_ids]@axis*sign
                if pair not in guards and sp[0].max() > sp[1].min()+1e-8:
                    continue
                direction = axis@R*sign; p = positions[tri_ids]@direction
                if pair in guards:
                    if p[0].max() <= p[1].min()+1e-8:
                        # Preserve the established side of neighbouring faces.
                        # Prefer a separating direction with actual support,
                        # not the uninformative normal of a coplanar pair.
                        choices.append((-float(p[1].mean()-p[0].mean()), direction))
                    continue
                low = p[0][~moving[0]].max() if (~moving[0]).any() else p.min()
                high = p[1][~moving[1]].min() if (~moving[1]).any() else p.max()
                if low > high+1e-8:
                    continue
                high = max(low, high)
                def cost(t):
                    da = np.minimum(0., t-p[0])*moving[0]
                    db = np.maximum(0., t-p[1])*moving[1]
                    return float(da@da+db@db)
                t = low if high == low else minimize_scalar(cost, bounds=(low, high),
                    method='bounded', options={'xatol':1e-12}).x
                if cost(t) > 1e-20:
                    choices.append((cost(t), direction))
        if not choices:
            # Extrinsic source half-plane order is not invariant under a new
            # curved eyelid shape. If it leaves no feasible direction, use the
            # actual authored triangle planes. Nonpenetration and the complete
            # geometry check remain unchanged; this does not certify anatomy.
            for direction in axes_for(positions[tri_ids]):
                for sign in (1.,-1.):
                    direction=direction*sign;p=positions[tri_ids]@direction
                    low=p[0][~moving[0]].max() if (~moving[0]).any() else p.min()
                    high=p[1][~moving[1]].min() if (~moving[1]).any() else p.max()
                    if low>high+1e-8:
                        continue
                    high=max(low,high)
                    def fallback_cost(t):
                        a=np.minimum(0.,t-p[0])*moving[0];b=np.maximum(0.,t-p[1])*moving[1]
                        return float(a@a+b@b)
                    t=low if high==low else minimize_scalar(fallback_cost,bounds=(low,high),method='bounded').x
                    if fallback_cost(t)>1e-20:
                        choices.append((fallback_cost(t),direction.copy()))
            if choices:
                current_plane_fallback.append(list(pair))
            else:
                continue
        direction = min(choices, key=lambda item:item[0])[1]
        for a in tri_ids[0]:
            for b in tri_ids[1]:
                if a == b:
                    continue
                row = np.zeros(len(free)*3)
                if int(a) in lookup:
                    row[lookup[int(a)]*3:lookup[int(a)]*3+3] -= direction
                if int(b) in lookup:
                    row[lookup[int(b)]*3:lookup[int(b)]*3+3] += direction
                rhs = -float((positions[b]-positions[a])@direction)
                if np.any(row):
                    rows.append(row); bounds.append(rhs)
    if not rows:
        return positions.copy(), dict(success=False, reason='no feasible source-ordered contact direction')
    C, b = np.asarray(rows), np.asarray(bounds)
    affected = np.flatnonzero(np.isin(faces, free).any(1))
    original_tri = positions[faces[affected]]
    minimum_area = np.linalg.norm(np.cross(original_tri[:,1]-original_tri[:,0],
                                          original_tri[:,2]-original_tri[:,0]), axis=1)
    def build(delta):
        x = positions.copy(); x[free] += delta.reshape(-1,3)
        return x
    def area_guard(delta):
        tri = build(delta)[faces[affected]]
        return np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)-minimum_area
    fit = minimize(lambda d: .5*float(d@d), np.zeros(len(free)*3), jac=lambda d:d,
        constraints=[LinearConstraint(C,b,np.full(len(b),np.inf)),
                     {'type':'ineq','fun':area_guard}], method='SLSQP',
        options={'maxiter':200,'ftol':1e-13})
    valid = np.all(C@fit.x-b >= -1e-8) and np.all(area_guard(fit.x) >= -1e-10)
    return (build(fit.x) if valid else positions.copy()), dict(success=bool(fit.success),
        constraints_satisfied=bool(valid), message=str(fit.message),
        authored_plane_fallback_pairs=current_plane_fallback,
        variable_logical_vertex_ids=free.tolist(), affected_face_ids=affected.tolist(),
        max_move=float(np.linalg.norm(fit.x.reshape(-1,3),axis=1).max()))


def crossing_extent(positions, faces, inventory):
    extent = 0.
    for pair in inventory:
        result = triangle_intersection(*positions[faces[list(pair)]], 1e-8)
        if result:
            extent += result.get('intersection_length', np.sqrt(result.get('overlap_area', 0.)))
    return float(extent)


def smooth_contact_patch(positions, faces, inventory, movable):
    """Position-only harmonic smoothing of the remaining tiny fold, with pins."""
    vertices = np.unique(faces[np.unique(np.asarray(sorted(inventory)).ravel())])
    halo = np.unique(faces[np.isin(faces,vertices).any(1)])
    free = halo[movable[halo]]
    used_faces = np.flatnonzero(np.isin(faces,free).any(1))
    used = np.unique(faces[used_faces]); local_faces=np.searchsorted(used,faces[used_faces])
    local_free=np.flatnonzero(np.isin(used,free)); pins=np.setdiff1d(np.arange(len(used)),local_free)
    L=SpokesARAP(positions[used],local_faces).stiffness
    x=positions.copy()
    x[free]=spsolve(L[local_free][:,local_free].tocsc(),-L[local_free][:,pins]@positions[used[pins]])
    triangles=x[faces[used_faces]]
    areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
    if not np.all(np.isfinite(x)) or np.any(areas==0):
        return positions.copy(),dict(accepted=False,reason='collapsed/nonfinite local surface')
    return x,dict(variable_logical_vertex_ids=free.tolist(),affected_face_ids=used_faces.tolist(),
        max_move=float(np.linalg.norm(x[free]-positions[free],axis=1).max()),
        method='positive edge-weight harmonic positions; all surrounding vertices fixed')


def correct(candidate, quality, out, passes=12, restore_source_detail=False,
            restore_canthus_plane=False, joint=False, correct_cut_correspondence=False,
            smooth_residual=False):
    if out.exists():
        raise FileExistsError('Keep previous candidates')
    receipt = json.loads((candidate/'receipt.json').read_text())
    q = json.loads((quality/'receipt.json').read_text())
    if sha(candidate/'o_head_candidate.npz') != q['arrays']['sha256']:
        raise ValueError('Contact evidence is stale')
    arrays = dict(np.load(candidate/'o_head_candidate.npz', allow_pickle=False))
    regions = json.loads((Path(receipt['inputs']['path']).parent/'native_regions.json').read_text())
    canonical = np.array([int(regions['graph_aliases'].get(str(i), i)) for i in range(len(arrays['verts']))])
    unique, inverse = np.unique(canonical, return_inverse=True)
    source = arrays['original_vertices'][unique]
    base = arrays['verts'][unique]; x = base.copy(); faces = inverse[arrays['faces']]
    mask = json.loads((candidate/'local_patch_regions.json').read_text())
    movable = np.zeros(len(base), bool)
    movable[inverse[mask['moved_render_vertex_ids']]] = True
    vertex_rotation = np.tile(np.eye(3), (len(base), 1, 1))
    vertex_scale = np.ones(len(base))
    for patch in mask['patches']:
        ids = np.union1d(patch['free_logical_vertex_ids'], patch['fixed_logical_boundary_ids']).astype(int)
        vertex_rotation[ids] = patch['boundary_similarity']['rotation']
        vertex_scale[ids] = patch['boundary_similarity']['scale']
    pairs = np.asarray(q['added_crossings'], int)
    touched = np.unique(faces[np.unique(pairs)])
    reset_notes = []
    cut_notes = []
    # Repair the surviving small folds coherently rather than move opposed
    # skin layers independently. All edits remain inside the original mask.
    dominant = arrays['bone_names'][arrays['bone_idx'][
        np.arange(len(arrays['verts'])), np.argmax(arrays['bone_w'], axis=1)]][unique]
    if correct_cut_correspondence:
        for side, render_loop in regions['physical_eye_boundaries'].items():
            loop = inverse[render_loop]
            sign = np.sign(source[loop,0].mean())
            ids = touched[movable[touched] & (np.sign(source[touched,0]) == sign)]
            ids = ids[np.array([str(n).startswith('cf_J_Eye') for n in dominant[ids]],bool)]
            if not len(ids):
                continue
            cut = loop[np.argmin(np.abs(source[loop,0]))]
            neighbours = np.unique(faces[np.isin(faces,cut).any(1)])
            cuff = np.setdiff1d(neighbours,loop)
            if not len(cuff):
                raise ValueError('No distinct source inner eye cuff')
            visible = cuff[np.argmax(source[cuff,2])]
            from tools.native_head.mother_local_repair import similarity
            scale, _, _ = similarity(source[loop],base[loop])
            halo = np.unique(faces[np.isin(faces,ids).any(1)])
            affected = np.union1d(halo[movable[halo]], [cut])
            target_visible = base[cut].copy()
            x[affected] = target_visible+(source[affected]-source[visible])*scale
            movable[cut] = True
            vertex_rotation[affected] = np.eye(3); vertex_scale[affected] = scale
            cut_notes.append(dict(side=side, cut_logical_id=int(cut),
                visible_cuff_logical_id=int(visible), affected_logical_ids=affected.tolist(),
                source_cut=source[cut].tolist(), source_visible=source[visible].tolist(),
                visible_target=target_visible.tolist(), revised_cut=x[cut].tolist(), scale=float(scale),
                semantic_status='source-topology/depth candidate annotation, not certified anatomical landmarks'))
        new_mask = movable[inverse]
        mask['moved_render_vertex_ids'] = np.flatnonzero(new_mask).tolist()
        mask['protected_render_vertex_ids'] = np.flatnonzero(~new_mask).tolist()
        mask['eye_cut_correspondence_adjustments'] = cut_notes
    for patch in mask['patches'] if restore_source_detail else []:
        free = np.asarray(patch['free_logical_vertex_ids'])
        involved = np.intersect1d(free, touched)
        if not len(involved):
            continue
        labels = dominant[free].tolist()
        placement = patch['boundary_similarity']
        R = np.asarray(placement['rotation']); s = placement['scale']; t = np.asarray(placement['translation'])
        reason = 'surviving folded source detail'
        if np.mean([str(name).startswith('cf_J_Mouth') for name in labels]) > .5:
            # One placement for both contacting lips, not separate pushes.
            involved = free
            reason = 'common intact upper/lower lip placement'
        elif np.mean([str(name).startswith('cf_J_Eye') for name in labels]) > .5:
            # Physical cut contour anchors this inner lid fold's source frame.
            # Keep that contour fixed and restore the depth of its free cuff.
            side = next(side for side, loop in regions['physical_eye_boundaries'].items()
                        if np.sign(source[inverse[loop], 0].mean()) == np.sign(source[free, 0].mean()))
            loop = inverse[regions['physical_eye_boundaries'][side]]
            from tools.native_head.mother_local_repair import similarity
            s, R, t = similarity(source[loop], base[loop])
            reason = 'inner lid depth from intact donor, contour fixed'
        x[involved] = source[involved]@R*s+t
        vertex_rotation[involved] = R
        vertex_scale[involved] = s
        reset_notes.append(dict(logical_vertex_ids=involved.tolist(), reason=reason,
                                scale=float(s), rotation=R.tolist(), translation=t.tolist()))
    if restore_canthus_plane:
        for side, render_loop in regions['physical_eye_boundaries'].items():
            loop = inverse[render_loop]
            sign = np.sign(source[loop, 0].mean())
            ids = touched[movable[touched] & (np.sign(source[touched, 0]) == sign)]
            ids = ids[np.asarray([str(name).startswith('cf_J_Eye') for name in dominant[ids]], dtype=bool)]
            if not len(ids):
                continue
            anchor = loop[np.argmin(np.abs(source[loop, 0]))]
            span_source = np.ptp(source[loop, :2], axis=0)
            span_target = np.ptp(base[loop, :2], axis=0)
            if np.any(span_source == 0):
                raise ValueError('Collapsed source eye contour')
            scale_xy = span_target/span_source
            # Recover the cuff's native side of the inner cut in the facial
            # plane. Its current depth is retained, avoiding the failed forward
            # extrusion experiment. All ten contour points remain fixed.
            x[ids, :2] = base[anchor, :2]+(source[ids, :2]-source[anchor, :2])*scale_xy
            reset_notes.append(dict(reason='inner eye cuff lateral/height order; depth unchanged',
                logical_vertex_ids=ids.tolist(), side=side, fixed_anchor=int(anchor),
                scale_xy=scale_xy.tolist(), depth_unchanged_exactly=bool(np.array_equal(x[ids, 2], base[ids, 2]))))
    history = []
    inventory = {tuple(sorted(map(int, pair))) for pair in pairs}
    exempt = {tuple(sorted(pair)) for pair in q['source_crossings']}
    if restore_source_detail or restore_canthus_plane or correct_cut_correspondence:
        dirty = np.any(np.any(x[faces] != base[faces], axis=2), axis=1)
        inventory = {p for p in inventory if not dirty[list(p)].any()} | dirty_crossings(x, faces, dirty, exempt)
    best_positions = x.copy(); best_inventory = inventory.copy(); best_pass = -1
    for iteration in range(passes):
        previous_positions = x.copy()
        repaired = remaining = 0
        for pair in sorted(inventory):
            triangles = faces[list(pair)]
            contact = triangle_intersection(*x[triangles], 1e-8)
            if not contact or contact['kind'] not in ('proper_crossing', 'coplanar_overlap'):
                continue
            remaining += 1
            # Polar rotation of the pair's source-supported patch placements.
            u, _, vt = np.linalg.svd(vertex_rotation[np.unique(triangles)].mean(0))
            R = u@vt
            if np.linalg.det(R) <= 0:
                raise ValueError('Invalid local contact rotation')
            repaired += int(separate_pair(source, x, triangles, movable, R,
                                         scale=float(vertex_scale[np.unique(triangles)].mean())))
        dirty = np.any(np.any(x[faces] != previous_positions[faces], axis=2), axis=1)
        inventory = {p for p in inventory if not dirty[list(p)].any()} | dirty_crossings(x, faces, dirty, exempt)
        history.append(dict(pass_index=iteration, known_crossings=remaining,
                            pairs_moved=repaired, complete_new_crossing_inventory=len(inventory)))
        if len(inventory) < len(best_inventory):
            best_positions = x.copy(); best_inventory = inventory.copy(); best_pass = iteration
        if not inventory or repaired == 0:
            break
    x = best_positions; inventory = best_inventory
    smoothing_notes=[]
    if smooth_residual:
        for patch_pass in range(3):
            if not inventory:
                break
            proposed,note=smooth_contact_patch(x,faces,inventory,movable)
            dirty=np.any(np.any(proposed[faces]!=x[faces],axis=2),axis=1)
            next_inventory={p for p in inventory if not dirty[list(p)].any()}|dirty_crossings(proposed,faces,dirty,exempt)
            note['crossings_before']=len(inventory);note['crossings_after']=len(next_inventory)
            note['accepted']=len(next_inventory)<len(inventory) or (
                len(next_inventory)==len(inventory) and crossing_extent(proposed,faces,next_inventory)<crossing_extent(x,faces,inventory))
            smoothing_notes.append(note)
            if not note['accepted']:
                break
            x=proposed;inventory=next_inventory
    joint_notes = []
    guards = set()
    if joint:
        for iteration in range(8):
            if not inventory:
                break
            proposed, note = joint_contact_step(source, x, faces, inventory, movable, vertex_rotation, guards)
            dirty = np.any(np.any(proposed[faces] != x[faces], axis=2), axis=1)
            next_inventory = {p for p in inventory if not dirty[list(p)].any()} | dirty_crossings(proposed, faces, dirty, exempt)
            note['crossings_before'] = len(inventory); note['crossings_after'] = len(next_inventory)
            before_extent = crossing_extent(x, faces, inventory)
            after_extent = crossing_extent(proposed, faces, next_inventory)
            note['extent_before'] = before_extent; note['extent_after'] = after_extent
            joint_notes.append(note)
            improves = len(next_inventory) < len(inventory) or (
                len(next_inventory) == len(inventory) and after_extent < before_extent)
            if not improves:
                new_guards = next_inventory-inventory-guards
                if new_guards:
                    guards |= new_guards
                    continue
                break
            x = proposed; inventory = next_inventory
    if not np.array_equal(x[~movable], base[~movable]):
        raise ValueError('Contact repair escaped the local mask')
    out.mkdir(parents=True)
    final = x[inverse]
    output = out/'o_head_candidate.npz'
    np.savez_compressed(output, **{**arrays, 'verts': final})
    original_candidate = Path(mask.get('original_base_candidate',
        receipt['local_repair']['base_candidate']['path'])).parent
    original_base = np.load(original_candidate/'o_head_candidate.npz')['verts']
    mask['original_base_candidate'] = str(original_candidate/'receipt.json')
    mask['displacement'] = (final-original_base).tolist()
    save_json(out/'local_patch_regions.json', mask)
    save_json(out/'receipt.json', {**receipt, 'candidate_arrays': source_file(output),
        'local_repair': {**receipt['local_repair'],
            'eye_boundary_unchanged_exactly': False if cut_notes else receipt['local_repair']['eye_boundary_unchanged_exactly'],
            'region_manifest': source_file(out/'local_patch_regions.json')},
        'eye_constraint_residual': float(np.linalg.norm(x[inverse[[int(i) for loop in regions['physical_eye_boundaries'].values() for i in loop]]]-base[inverse[[int(i) for loop in regions['physical_eye_boundaries'].values() for i in loop]]],axis=1).max()) if cut_notes else receipt['eye_constraint_residual'],
        'contact_correction': dict(base=source_file(candidate/'receipt.json'),
            quality=source_file(quality/'receipt.json'), code=source_file(Path(__file__)),
            history=history, source_detail_restoration=reset_notes,
            selected_best_pass=best_pass,
            remaining_new_crossing_pairs=sorted(inventory),
            joint_local_contacts=joint_notes,
            cut_correspondence_adjustments=cut_notes,
            small_fold_smoothing=smoothing_notes,
            outside_mask_unchanged_exactly=True,
            max_contact_displacement=float(np.linalg.norm(x-base, axis=1).max()),
            full_quality_checked=False),
        'installed': False, 'deliverable': False, 'game_mutated': False})
    print(json.dumps(dict(output=str(out.resolve()), history=history,
                         installed=False, deliverable=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--restore-source-detail', action='store_true',
                        help='Isolated coherent-restoration experiment; not the default repair')
    parser.add_argument('--restore-canthus-plane', action='store_true',
                        help='Restore surviving inner eye cuff order without changing its depth')
    parser.add_argument('--joint', action='store_true', help='Resolve surviving tiny contact cluster together')
    parser.add_argument('--correct-cut-correspondence', action='store_true',
                        help='Explicit inner cut/cuff correspondence experiment; preserve actual neck')
    parser.add_argument('--smooth-residual',action='store_true',help='Smooth only surviving tiny contact patch with surrounding vertices fixed')
    args = parser.parse_args()
    correct(args.candidate.resolve(), args.quality.resolve(), args.out.resolve(),
            restore_source_detail=args.restore_source_detail,
            restore_canthus_plane=args.restore_canthus_plane, joint=args.joint,
            correct_cut_correspondence=args.correct_cut_correspondence,
            smooth_residual=args.smooth_residual)
