"""Provisional native-topology shell registration; not a deliverable head asset.

Official zero-identity FLAME is the geometry target. Original native topology,
UV splits and mouth interiors remain. The neck is a hard constraint; eyelid
targets come from the original source landmark embedding. Lip/ear semantic
constraints and expression/component adaptation are explicitly pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from tools.model_bridge.artifact import sha
from tools.model_bridge.attachment_audit import topology
from tools.native_head.mother_template_inputs import save_json, source_file
from tools.native_head.mother_uv_anchors import constraints as uv_constraints
from tools.native_head.mother_arap import SpokesARAP
from tools.native_head.mother_surface_targets import RegionalTargets
from tools.native_head.mother_oriented_surface import logical_normals


def path_between(loop, start, end):
    loop = list(map(int, loop))
    a, b = loop.index(start), loop.index(end)
    rotated = loop[a:]+loop[:a]
    k = rotated.index(end)
    return rotated[:k+1], [start]+rotated[:k-1:-1]


def parameters(points):
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if arc[-1] == 0:
        raise ValueError("Collapsed contour")
    return arc/arc[-1]


def sample_curve(points, fractions):
    t = parameters(points)
    return np.column_stack([np.interp(fractions, t, points[:, k]) for k in range(3)])


def eye_constraints(native, regions, reference, landmarks):
    endpoint_pairs, native_paths = [], {}
    # These are the audited native/FLAME coordinate sides, not screen labels.
    source_paths = {"L": ([36, 37, 38, 39], [39, 40, 41, 36]),
                    "R": ([45, 44, 43, 42], [42, 47, 46, 45])}
    for side, loop in regions["physical_eye_boundaries"].items():
        xyz = native[loop]
        outer = loop[int(np.argmax(np.abs(xyz[:, 0])))]
        inner = loop[int(np.argmin(np.abs(xyz[:, 0])))]
        paths = path_between(loop, outer, inner)
        upper = max(paths, key=lambda ids: float(native[ids, 1].mean()))
        lower = next(p for p in paths if p != upper)[::-1]
        native_paths[side] = (upper, lower)
        endpoint_pairs += [(source_paths[side][0][0], outer),
                           (source_paths[side][0][-1], inner)]
    # Both audited canonical frames use X lateral, Y height and Z forward.
    # Eye slant is shape, not a reason to rotate the entire reference skull.
    source_endpoints = landmarks[[i for i, j in endpoint_pairs]]
    native_endpoints = native[[j for i, j in endpoint_pairs]]
    s = np.ptp(native_endpoints[:, 0])/np.ptp(source_endpoints[:, 0])
    rotation = np.eye(3)
    translation = native_endpoints.mean(0)-source_endpoints.mean(0)*s
    target = reference@rotation.T*s+translation
    target_landmarks = landmarks@rotation.T*s+translation
    constraints, profiles = {}, {}
    for side, paths in native_paths.items():
        profiles[side] = {}
        for name, ids, source_ids in zip(("upper", "lower"), paths, source_paths[side]):
            fraction = parameters(native[ids])
            values = sample_curve(target_landmarks[source_ids], fraction)
            for i, value in zip(ids, values):
                if i in constraints and not np.allclose(constraints[i], value, atol=1e-12, rtol=0):
                    raise ValueError("Conflicting eye-corner constraints")
                constraints[i] = value
            profiles[side][name] = dict(native_vertex_ids=ids, source_landmark_ids=source_ids,
                native_arc_fractions=fraction.tolist(), targets=values.tolist())
    return target, constraints, dict(scale=s, rotation=rotation.tolist(), translation=translation.tolist(),
        alignment="Canonical axes preserved; uniform scale from outer-canthus lateral span, translation from four-canthus centers",
        eye_contours=profiles,
        limitation="Piecewise linear embedded landmark polylines; not full eyelid surface registration")


def normals(vertices, faces):
    tri = vertices[faces]
    return np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])


def candidate(inputs, out, native_profile, texture_manifest, source_masks, default_reference):
    if out.exists():
        raise FileExistsError("Keep previous candidates; use a fresh output path")
    receipt = json.loads((inputs/"receipt.json").read_text(encoding="utf-8"))
    for key in ("native_original_copy", "native_prefab_export", "regions", "flame_zero_identity"):
        row = receipt[key]
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Prepared input changed: "+key)
    row = next(m for m in receipt["all_renderers"] if m["mesh"] == "o_head")
    if sha(row["array_export"]["path"]) != row["array_export"]["sha256"]:
        raise ValueError("Native donor arrays changed")
    arrays = dict(np.load(row["array_export"]["path"], allow_pickle=False))
    original_bind = arrays['verts'].copy()
    default_receipt = json.loads((default_reference/'receipt.json').read_text())
    if sha(default_receipt['inputs']['path']) != default_receipt['inputs']['sha256']:
        raise ValueError('Native default reference inputs changed')
    closed = next(r for r in default_receipt['renderer_references'] if r['mesh']=='o_head')['reference']
    if sha(closed['path']) != closed['sha256']:
        raise ValueError('Native default reference geometry changed')
    default_arrays = dict(np.load(closed['path'], allow_pickle=False))
    for key in ('verts', 'faces', 'uv', 'bone_idx', 'bone_w'):
        if not np.array_equal(default_arrays[key], arrays[key]):
            raise ValueError('Native default reference uses a different donor')
    arrays['verts'] = default_arrays['closed_reference_vertices']
    regions = json.loads((inputs/"native_regions.json").read_text(encoding="utf-8"))
    flame = dict(np.load(inputs/"flame_zero_identity.npz", allow_pickle=False))
    fv, ff = flame["v_template"].astype(float), flame["faces_tensor"]
    landmarks = (fv[ff[flame["full_lmk_faces_idx"][0]]]*
                 flame["full_lmk_bary_coords"][0, :, :, None]).sum(1)
    native_v, native_f = arrays["verts"].astype(float), arrays["faces"]
    target, eyes, placement = eye_constraints(native_v, regions, fv, landmarks)
    aliases = {int(k): int(v) for k, v in regions["graph_aliases"].items()}
    canonical = np.asarray([aliases.get(i, i) for i in range(len(native_v))])
    unique, inverse = np.unique(canonical, return_inverse=True)
    v0, f = native_v[unique], inverse[native_f]
    for group in regions['identical_bind_position_and_skin_groups']:
        if not np.all(native_v[group] == native_v[group[0]]):
            raise ValueError('Default expression separates logical seam aliases')
    if not np.array_equal(native_v[regions['neck_boundary_all_render_copies']],
                          original_bind[regions['neck_boundary_all_render_copies']]):
        raise ValueError('Default expression moves the source body interface')
    lookup = {int(old): i for i, old in enumerate(unique)}
    pinned = {lookup[int(canonical[i])]: native_v[i] for i in regions["neck_boundary_all_render_copies"]}
    for i, xyz in eyes.items():
        pinned[lookup[int(canonical[i])]] = xyz
    pin_ids = np.asarray(sorted(pinned), int)
    pin_positions = np.asarray([pinned[i] for i in pin_ids])
    free = np.setdiff1d(np.arange(len(v0)), pin_ids)
    target_landmarks = landmarks@np.asarray(placement["rotation"]).T*placement["scale"]+placement["translation"]
    C, anchor_positions, anchor_receipt = uv_constraints(arrays, inverse, target_landmarks,
                                                       native_profile, texture_manifest)
    C_free, C_pin = C[:, free], C[:, pin_ids]
    # Only the head-shell index component is a target, never FLAME eyeballs.
    ft = topology(fv, ff)
    components = [c for c in ft["components"] if any(
        set(b["vertices"]).issubset(c["vertices"]) for b in ft["boundaries"])]
    if len(components) != 1:
        raise ValueError("Cannot identify source shell from actual mouth boundary")
    head_ids = components[0]["vertices"]
    target_faces = ff[np.isin(ff, head_ids).all(1)]
    surface = RegionalTargets(arrays, inverse, target, target_faces, source_masks, anchor_receipt)
    data_weight = np.ones(len(v0))
    oral = inverse[regions["inner_mouth_component_vertex_ids"]]
    data_weight[oral] = 0  # Do not snap oral interiors onto external face skin.
    data_weight[surface.preserve_native_lips] = 0
    arap = SpokesARAP(v0, f)
    source_normals = logical_normals(v0, f)
    x = v0.copy()
    history = []
    # Shape-preserving asset authoring. Orientation-guarded line search avoids
    # applying a large nasal/lip/eye displacement in a fold-producing jump.
    # Oral interiors have zero target attraction but retain full ARAP energy.
    for strength in (16., 8., 4., 2.):
        A = (sparse.diags(data_weight)+strength*arap.stiffness).tocsr()
        A_free = A[free][:, free].tocsc()
        A_pin = A[free][:, pin_ids]
        system = sparse.bmat([[A_free, C_free.T],
                              [C_free, sparse.csc_matrix((C.shape[0], C.shape[0]))]], format="csc")
        factor = splu(system)
        for iteration in range(24):
            correspondence_normals = np.einsum('nij,nj->ni', arap.rotations(x), source_normals)
            q = surface.closest(x, correspondence_normals)
            rhs = data_weight[:, None]*q+strength*arap.rhs(x)
            solved = factor.solve(np.vstack([rhs[free]-A_pin@pin_positions,
                                             anchor_positions-C_pin@pin_positions]))
            proposed = x.copy()
            proposed[pin_ids] = pin_positions
            proposed[free] = solved[:len(free)]
            old_n = normals(x, f)
            fraction = 1.
            while fraction >= 1/1024:
                step = x+(proposed-x)*fraction
                new_n = normals(step, f)
                if np.all(np.sum(old_n*new_n, axis=1) > 0):
                    break
                fraction /= 2
            if fraction < 1/1024:
                history.append(dict(stiffness=strength, iteration=iteration, stopped="local_orientation_guard"))
                break
            movement = float(np.max(np.linalg.norm(step-x, axis=1)))
            x = step
            history.append(dict(stiffness=strength, iteration=iteration, step_fraction=fraction,
                                neck_hard_constraint_exact=bool(np.array_equal(
                                    x[[lookup[int(canonical[i])] for i in regions["physical_neck_boundary"]]],
                                    v0[[lookup[int(canonical[i])] for i in regions["physical_neck_boundary"]]]))))
            if movement < 1e-8:
                break
    result = x[inverse]
    if not np.array_equal(result[regions["neck_boundary_all_render_copies"]],
                          native_v[regions["neck_boundary_all_render_copies"]]):
        raise ValueError("Neck hard constraint moved")
    for group in regions["identical_bind_position_and_skin_groups"]:
        if not np.all(result[group] == result[group[0]]):
            raise ValueError("UV seam copies diverged")
    out.mkdir(parents=True)
    candidate_path = out/"o_head_candidate.npz"
    np.savez_compressed(candidate_path, **{**arrays, "verts": result},
                        original_vertices=native_v, original_bind_vertices=original_bind,
                        reference_vertices=target, reference_faces=ff)
    reopened = np.load(candidate_path, allow_pickle=False)
    for key, value in arrays.items():
        if key != 'verts' and not np.array_equal(value, reopened[key]):
            raise ValueError('Source attribute changed during candidate serialization: '+key)
    save_json(out/"receipt.json", dict(format="native_mother_shell_candidate_v2",
        inputs=source_file(inputs/"receipt.json"), code=source_file(Path(__file__)), placement=placement,
        uv_anchor_code=source_file(Path(__file__).with_name("mother_uv_anchors.py")),
        arap_code=source_file(Path(__file__).with_name("mother_arap.py")),
        regional_target_code=source_file(Path(__file__).with_name("mother_surface_targets.py")),
        oriented_surface_code=source_file(Path(__file__).with_name('mother_oriented_surface.py')),
        regional_targets=surface.receipt,
        default_reference=source_file(default_reference/'receipt.json'),
        candidate_geometry_state='Fitted neutral closed-mouth/open-eye reference; not an authored bind mesh',
        bind_mesh_authored=False,
        candidate_arrays=source_file(candidate_path),
        authoring_energy="Positive inverse-edge-length spokes ARAP; original interior shape retained by edge rotations, no exterior attraction on oral interior",
        candidate_anatomical_constraints=anchor_receipt,
        candidate_anatomical_constraint_residual=float(np.linalg.norm(C@x-anchor_positions, axis=1).max()),
        iteration_history=history, cut_faces=[], welded_vertices=False,
        topology_and_uv_unchanged=True, neck_boundary_preserved_exactly=True,
        identity_offsets_on_uv_copies_identical=True,
        eye_constraint_residual=float(np.linalg.norm(x[pin_ids]-pin_positions, axis=1).max()),
        original_complete_bundle_preserved=receipt["native_original_copy"],
        candidate_generated=True, deliverable=False, installed=False, game_mutated=False,
        pending=["Reviewed inner/outer lip and ear-root correspondences",
                 "Full expression, bone pivot and attached component adaptation",
                 "Final geometric normals/tangents and actual BP body interface normals",
                 "Surface accuracy, self-intersection and complete candidate review"],
        limitations=["Initial shell fit, not complete semantic registration",
                     "Local orientation guard is not a global self-intersection certificate",
                     "Stored original normals/frames are source data, not final retargeted shading/expression",
                     "All original component/controller data remains in prepared donor; no package installed"]))
    print(json.dumps(dict(output=str(out.resolve()), candidate_generated=True,
                         neck_preserved=True, deliverable=False, game_mutated=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-profile", type=Path, default=Path("tools/native_head/native_atlas_profile.json"))
    parser.add_argument("--texture-manifest", type=Path,
                        default=Path("../HS2Mod/artifacts/chenger/native_skin_retarget_20261008/manifest.json"))
    parser.add_argument("--source-masks", type=Path,
                        default=Path("../smirk/assets/FLAME_masks/FLAME_masks.pkl"))
    parser.add_argument('--default-reference', type=Path,
                        default=Path('outputs/native_mother_template_20261008/default_reference_v1'))
    args = parser.parse_args()
    candidate(args.inputs.resolve(), args.out.resolve(), args.native_profile.resolve(),
              args.texture_manifest.resolve(), args.source_masks.resolve(), args.default_reference.resolve())
