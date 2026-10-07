"""Compute literal surface observables; never infer RGB or anatomical equivalence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contract import AppearancePixel, FormalSurface, Manifest, MaterialCandidate, Unobservable, digest, validate_sources


def arrays(mesh):
    vertices = np.asarray(mesh['baked']['vertices'], dtype=float)
    raw_faces = np.asarray(mesh['baked']['triangles'])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Finite actual BakeMesh Nx3 vertices required')
    if raw_faces.ndim != 1 or raw_faces.size == 0 or raw_faces.size % 3 or raw_faces.dtype.kind not in 'iu':
        raise ValueError('Nonempty integer actual triangle index stream required')
    faces = raw_faces.reshape(-1, 3)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid triangle vertex index')
    if not np.array_equal(raw_faces, np.asarray(mesh['source']['triangles'])):
        raise ValueError('Source and BakeMesh ordered topology differ')
    return vertices, faces


def region_faces(mesh, submesh_id):
    matches = [s for s in mesh['baked']['submeshes'] if s.get('submesh_index') == submesh_id]
    if len(matches) != 1:
        raise ValueError('Actual submesh identity missing/ambiguous')
    submesh = matches[0]
    if submesh.get('topology') != 'Triangles' or submesh.get('indices_apply_base_vertex') is not True:
        raise ValueError('Actual triangle topology with already-applied base vertex required')
    raw = np.asarray(submesh['indices'])
    if raw.ndim != 1 or not raw.size or raw.size % 3 or raw.dtype.kind not in 'iu':
        raise ValueError('Integer nonempty submesh triangle stream required')
    combined = np.concatenate([np.asarray(s['indices']) for s in mesh['baked']['submeshes']])
    if not np.array_equal(combined, np.asarray(mesh['baked']['triangles'])):
        raise ValueError('Actual full stream and ordered submesh streams differ')
    return raw.reshape(-1, 3)


def plane_segments(vertices, faces, normal, offset, tolerance):
    segments, tangencies, coplanar = [], [], []
    for triangle_id, ids in enumerate(faces):
        points = vertices[ids]
        distance = points @ normal - offset
        on_plane = np.abs(distance) <= tolerance
        if on_plane.all():
            coplanar.append(triangle_id)
            continue
        hits = [p for p, on in zip(points, on_plane) if on]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (distance[i] < -tolerance and distance[j] > tolerance) or (distance[j] < -tolerance and distance[i] > tolerance):
                fraction = distance[i] / (distance[i] - distance[j])
                hits.append(points[i] + fraction * (points[j] - points[i]))
        distinct = []
        for hit in hits:
            if not any(np.linalg.norm(hit-other) <= tolerance for other in distinct):
                distinct.append(hit)
        if len(distinct) == 2:
            segments.append({'region_triangle_id': triangle_id, 'endpoints': [point.tolist() for point in distinct]})
        elif len(distinct) == 1:
            tangencies.append({'region_triangle_id': triangle_id, 'point': distinct[0].tolist()})
        elif len(distinct) > 2:
            raise ValueError('Ambiguous plane intersection beyond two noncoplanar points')
    return {'segments': segments, 'tangent_points': tangencies, 'coplanar_region_triangle_ids': coplanar,
            'unique_ordered_curve_claimed': False, 'position_welding_applied': False,
            'status': 'ambiguous_coplanar_surface' if coplanar else ('measured_segments' if segments else 'empty_or_tangent_slice')}


def formal_measure(feature: FormalSurface, mesh, vertices):
    faces = region_faces(mesh, feature.region.submesh_id)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Submesh index outside actual vertices')
    used = np.unique(faces)
    result = {'referenced_vertex_count': len(used), 'region_triangle_count': len(faces)}
    if feature.operation == 'surface_area':
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2
        result.update(area_game_units_squared=float(areas.sum()), zero_area_region_triangle_ids=np.flatnonzero(areas == 0).tolist())
    elif feature.operation == 'plane_intersection_segments':
        result.update(plane_segments(vertices, faces, np.asarray(feature.axis), feature.plane_offset, feature.numerical_tolerance_game_units))
    else:
        support = vertices[used] @ np.asarray(feature.axis)
        maximum, minimum = float(support.max()), float(support.min())
        max_ids = used[np.abs(support-maximum) <= feature.numerical_tolerance_game_units].tolist()
        min_ids = used[np.abs(support-minimum) <= feature.numerical_tolerance_game_units].tolist()
        result.update(max_support_game_units=maximum, maximizing_source_vertex_ids=max_ids,
                      unique_maximum_under_declared_tolerance=len(max_ids) == 1,
                      extremum_recomputed=True, fixed_material_identity_created=False)
        if feature.operation == 'directional_span':
            result.update(min_support_game_units=minimum, minimizing_source_vertex_ids=min_ids,
                          span_game_units=maximum-minimum)
    return result


def execute(manifest: Manifest):
    _, mesh = validate_sources(manifest)
    vertices, faces = arrays(mesh)
    results = []
    for feature in manifest.features:
        record = {'id': feature.id, 'kind': feature.kind, 'claims': feature.claims.model_dump()}
        if isinstance(feature, FormalSurface):
            record.update(definition=feature.model_dump(), result=formal_measure(feature, mesh, vertices),
                          interpretation='Literal full asset submesh in raw renderer game units; no named facial anatomy or fixed material extremum.')
        elif isinstance(feature, MaterialCandidate):
            if feature.triangle_id >= len(faces) or tuple(faces[feature.triangle_id]) != feature.ordered_vertex_ids:
                raise ValueError('Material triangle/order does not match actual source topology')
            point = np.asarray(feature.barycentric) @ vertices[faces[feature.triangle_id]]
            record.update(result={'point_renderer_baked_raw': point.tolist(), 'status': 'source_bound_geometry_candidate_only'},
                          definition=feature.model_dump(), interpretation='Fixed material identity; image visibility and anatomical meaning remain unverified.')
        elif isinstance(feature, AppearancePixel):
            record.update(result={'status': 'appearance_candidate_only' if feature.xy is not None else 'unobservable',
                                  'xy': feature.xy, 'uncertainty_radius_px': feature.uncertainty_radius_px,
                                  'eligible_as_fixed_skin_point_evidence': False},
                          definition=feature.model_dump(), interpretation='Image appearance cue; no skin depth, fixed material identity or anatomy inferred.')
        elif isinstance(feature, Unobservable):
            record.update(result={'status': 'unobservable', 'value': None, 'reason': feature.reason}, definition=feature.model_dump())
        results.append(record)
    return {'schema_version': 1, 'geometry_scope': manifest.geometry.model_dump(), 'results': results,
            'producer_source_sha256': {name: digest(Path(__file__).with_name(name)) for name in ('contract.py', 'measure.py')},
            'manifest_contract_valid': True, 'byte_sources_verified': True,
            'camera_pixel_lbs_certified_by_this_tool': False, 'anatomical_correspondence_validated': False,
            'material_visibility_validated': False, 'cross_base_semantic_equivalence_validated': False,
            'fixed_image_gate_max_error_px': 2.0, 'full_infrastructure_goal_complete': False}


def compare_spans(first: Manifest, second: Manifest):
    """Same-source formal diagnostics only. Cross-base/frame correspondence is refused."""
    a, b = first.geometry, second.geometry
    for name in ('head_id', 'renderer_path', 'source_geometry_sha256', 'coordinate_space', 'units', 'frame_policy_id'):
        if getattr(a, name) != getattr(b, name):
            raise ValueError('Cross-base/source/frame comparison needs independent correspondence contract: '+name)
    left = {f.id: f for f in first.features if isinstance(f, FormalSurface) and f.operation == 'directional_span'}
    right = {f.id: f for f in second.features if isinstance(f, FormalSurface) and f.operation == 'directional_span'}
    if not left or set(left) != set(right) or any(left[k] != right[k] for k in left):
        raise ValueError('Identical nonempty span definitions required')
    ar, br = execute(first), execute(second)
    av = {r['id']: r['result']['span_game_units'] for r in ar['results'] if r['id'] in left}
    bv = {r['id']: r['result']['span_game_units'] for r in br['results'] if r['id'] in right}
    return {'status': 'same_source_raw_frame_diagnostic_only', 'differences_game_units': {k: bv[k]-av[k] for k in left},
            'first_scope': a.model_dump(), 'second_scope': b.model_dump(), 'frame_policy_independently_certified': False,
            'facial_width_length_depth_anatomically_certified': False, 'global_scale_or_pose_fit_applied': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('New output path required')
    manifest = Manifest.model_validate_json(args.manifest.read_text(encoding='utf-8'))
    report = execute(manifest)
    report['manifest_sha256'] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'out': str(args.out.resolve()), 'features': len(report['results']), 'semantic_certification': False}))


if __name__ == '__main__':
    main()
