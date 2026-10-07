"""Verify receipt-bound live head sweeps and measure surface response, offline only."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.parameter_atlas.core import summarize_displacement
from tools.unity_parity.geometry import (
    Uncertifiable,
    blendshape_delta,
    finite_array,
    matrix,
    quality_influences,
    recorded_uniform_renderer_scale,
    rigid_alignment,
    skin_world,
    transform_points,
    vertex_errors,
)

ROLES = ('local_minus', 'local_plus', 'native_min', 'native_max',
         'extended_min', 'extended_max')
IDENTITY = ('renderer_id', 'renderer_path', 'mesh_instance_id', 'mesh_name',
            'source_geometry_sha256', 'bone_names', 'bone_transform_ids',
            'root_bone_transform_id', 'vertex_count', 'triangle_count')


def load_receipt(receipt, directory):
    path = Path(receipt['path'])
    if not path.is_absolute():
        path = Path(directory) / path
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest() != receipt['sha256']:
        raise Uncertifiable(f'Geometry receipt SHA-256 mismatch: {path}')
    return json.loads(raw.decode('utf-8-sig'))


def mesh_map(snapshot):
    result = {mesh['renderer_path']: mesh for mesh in snapshot['meshes']}
    if len(result) != len(snapshot['meshes']) or not result:
        raise Uncertifiable('Renderer paths must be nonempty and unique')
    return result


def attach_frames(source, candidate):
    """Do not reuse a frame merely because two renderers share a mesh name."""
    if set(mesh_map(source)) != set(mesh_map(candidate)):
        raise Uncertifiable('Renderer coverage changed from source snapshot')
    for key in ('head_id', 'transform_id'):
        if key in source['character'] and source['character'][key] != candidate['character'].get(key):
            raise Uncertifiable(f'Character {key} changed')
    output = dict(candidate)
    output['meshes'] = []
    for path, current in mesh_map(candidate).items():
        original = mesh_map(source)[path]
        for key in IDENTITY:
            if key not in current or current[key] != original.get(key):
                raise Uncertifiable(f'{path}: changed/missing identity {key}')
        if current['source'] != original['source']:
            raise Uncertifiable(f'{path}: changed source vertices/topology/weights/bindposes')
        a, b = original.get('blendshapes', []), current.get('blendshapes', [])
        if len(a) != len(b):
            raise Uncertifiable(f'{path}: blendshape count changed')
        merged = dict(current)
        merged['blendshapes'] = []
        for before, after in zip(a, b):
            for key in ('name', 'shape_index'):
                if key not in after or before.get(key) != after[key]:
                    raise Uncertifiable(f'{path}: blendshape {key} changed')
            af, bf = before.get('frames', []), after.get('frames', [])
            if len(af) != len(bf) or any(x.get('weight') != y.get('weight') for x, y in zip(af, bf)):
                raise Uncertifiable(f'{path}: blendshape frame metadata changed')
            shape = dict(after)
            shape['frames'] = af
            # Check even zero-weight shapes: a later sample may activate any frame.
            for frame in af:
                finite_array(frame['delta_vertices'], (current['vertex_count'], 3))
            merged['blendshapes'].append(shape)
        output['meshes'].append(merged)
    return output


def require_native(snapshot, expected, *, tolerance=2e-6):
    for vector in (expected, snapshot['character']['shape_value_face']):
        if len(vector) != 59 or any(type(value) not in (int, float) for value in vector):
            raise Uncertifiable('Native59 coefficients must be finite numbers, not booleans')
    values = finite_array(expected, (59,))
    observed = finite_array(snapshot['character']['shape_value_face'], (59,))
    if np.max(np.abs(values - observed)) > tolerance:
        raise Uncertifiable('Captured native59 readback differs from requested values')
    return values


def anchor(snapshot, manifest):
    wanted = manifest.get('coordinate_transform_id')
    transforms = {item['id']: item for item in snapshot['transforms']}
    if len(transforms) != len(snapshot['transforms']):
        raise Uncertifiable('Duplicate transform identities')
    if wanted is not None:
        selected = [transforms[wanted]] if wanted in transforms else []
    else:
        name = manifest.get('coordinate_transform_name', 'cf_J_Head')
        selected = [item for item in transforms.values() if item['name'] == name]
    if len(selected) != 1:
        raise Uncertifiable('Coordinate anchor must identify exactly one captured transform')
    item = selected[0]
    return item, np.linalg.inv(matrix(item['local_to_world']))


def require_same_anchor(first, current, *, allow_rotation=False):
    for key in ('id', 'path', 'parent_id'):
        if first.get(key) != current.get(key):
            raise Uncertifiable(f'Coordinate anchor {key} changed')
    keys = ('local_position', 'local_scale') if allow_rotation else ('local_position', 'local_rotation_xyzw', 'local_scale')
    for key in keys:
        if key not in first or key not in current or not np.allclose(first[key], current[key], atol=2e-6, rtol=0):
            raise Uncertifiable(f'Coordinate anchor {key} changed; normalization would erase a real driver')


def expression_changes(first, current):
    changes = []
    if first['character'].get('expression') != current['character'].get('expression'):
        changes.append('character.expression')
    for path, mesh in mesh_map(first).items():
        other = mesh_map(current)[path]
        for a, b in zip(mesh.get('blendshapes', []), other.get('blendshapes', [])):
            if abs(float(a['current_weight']) - float(b['current_weight'])) > 1e-6:
                changes.append(f'{path}/blendshape/{a["name"]}')
    return changes


def public_abmx(snapshot):
    runtime = snapshot.get('abmx_runtime') or {}
    return {bone['name']: {key: bone[key] for key in ('scale', 'length', 'position', 'rotation')}
            for bone in runtime.get('bones', [])}


def require_abmx(snapshot, baseline, case):
    observed, original = public_abmx(snapshot), public_abmx(baseline)
    if case['kind'] != 'abmx':
        if observed != original:
            raise Uncertifiable('ABMX public modifiers changed during native/repeat probe')
        return
    patch = case['patch']
    expected = copy.deepcopy(original)
    expected[patch['name']] = {key: patch[key] for key in ('scale', 'length', 'position', 'rotation')}
    if set(observed) != set(expected) or any(not np.allclose(observed[name][key], expected[name][key], atol=2e-6, rtol=0)
                                           for name in expected for key in expected[name]):
        raise Uncertifiable('Captured ABMX modifiers do not match isolated requested patch')


def measure_snapshot(snapshot, manifest, *, tolerance=1e-5, head_only=False):
    if snapshot.get('schema_version') != 1 or snapshot.get('frame_count') != snapshot.get('frame_count_end'):
        raise Uncertifiable('Unsupported or internally unstable geometry snapshot')
    transforms = {item['id']: item for item in snapshot['transforms']}
    origin, inverse = anchor(snapshot, manifest)
    surfaces, parity = {}, []
    all_meshes = mesh_map(snapshot)
    selected = all_meshes
    if head_only:
        selected = {path: mesh for path, mesh in all_meshes.items() if mesh['mesh_name'] == 'o_head'}
        if len(selected) != 1:
            raise Uncertifiable('Head-only analysis requires exactly one captured o_head renderer')
    for path, mesh in selected.items():
        count = quality_influences(mesh, snapshot)
        if count is None:
            raise Uncertifiable(f'{path}: unknown skin quality')
        if mesh['source']['triangles'] != mesh['baked']['triangles']:
            raise Uncertifiable(f'{path}: baked/source topology differs')
        predicted, active = skin_world(mesh, transforms, influences=count)
        raw = finite_array(mesh['baked']['vertices'], predicted.shape)
        matching, candidates = [], {}
        for name, candidate in mesh['baked']['world_candidates'].items():
            world = transform_points(raw, candidate['matrix'])
            error = vertex_errors(predicted, world)
            consistency = vertex_errors(world, candidate['vertices'])
            candidates[name] = {'lbs_vs_bake': error, 'export_consistency': consistency}
            if error['max_normalized'] is not None and error['max_normalized'] <= tolerance and consistency['max_normalized'] is not None and consistency['max_normalized'] <= tolerance:
                matching.append((name, world))
        if not matching:
            raise Uncertifiable(f'{path}: no BakeMesh world candidate passes LBS parity')
        if len(matching) > 1 and any(vertex_errors(matching[0][1], other)['max_normalized'] > tolerance for _, other in matching[1:]):
            raise Uncertifiable(f'{path}: matching BakeMesh world candidates remain geometrically ambiguous')
        surfaces[path] = transform_points(matching[0][1], inverse)
        parity.append({'renderer_path': path, 'mesh_name': mesh['mesh_name'],
                       'vertex_count': len(raw), 'influences': count,
                       'active_blendshapes': active,
                       'matching_candidates': [name for name, _ in matching],
                       'convention_distinguished': len(matching) == 1,
                       'candidate_errors': candidates})
    return surfaces, parity, origin


def response_stats(first, current, *, head_diagonal):
    result = {}
    for path, vertices in first.items():
        row, _, _ = summarize_displacement(vertices, current[path])
        row['normalized_by_baseline_head_diagonal'] = {
            key: value / head_diagonal for key, value in row['displacement'].items()}
        row['axis_spans_xyz'] = row['bbox']['extent']
        row['axis_span_changes_xyz'] = row['bbox']['extent_change']
        result[path] = row
    return result


class OfflinePredictor:
    """One baseline rigid transform, reused unchanged for every native probe."""
    def __init__(self, snapshot, surfaces, manifest):
        from src.hs2_mesh_deform import HeadRig, build_mesh
        self.build = build_mesh
        self.rig = HeadRig(snapshot['character']['head_id'], sampling_profile=manifest.get('sampling_profile', 'slider_unlocker_18_2'))
        cache_paths = [Path(self.rig.data_dir) / name for name in ('o_head_mesh.npz', 'skeleton.json', 'anmShapeHead.json')]
        cache_paths += [Path(self.rig.root_dir) / name for name in ('enums.json', 'customhead.json', 'update_eqns.json')]
        self.cache_receipts = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in cache_paths]
        self.mesh = next(m for m in snapshot['meshes'] if m['mesh_name'] == 'o_head')
        source = self.mesh['source']
        if self.rig.verts.shape != np.asarray(source['vertices']).shape or not np.allclose(self.rig.verts, source['vertices'], atol=1e-6, rtol=0):
            raise Uncertifiable('Offline cached source vertices differ')
        if not np.array_equal(self.rig.faces.reshape(-1), source['triangles']) or self.rig.skin_bone_names != self.mesh['bone_names'] or not np.array_equal(self.rig.bone_idx, source['bone_indices']) or not np.allclose(self.rig.bone_w, source['bone_weights'], atol=1e-7, rtol=0) or not np.allclose(self.rig.bindpose, np.asarray(source['bindposes']).reshape(-1, 4, 4), atol=1e-6, rtol=0):
            raise Uncertifiable('Offline cached topology/palette/weights/bindposes differ')
        self.scale = float(manifest.get('offline_unit_scale', 1.0))
        if self.scale != 1 and not manifest.get('offline_unit_scale_reason'):
            raise Uncertifiable('Offline nonunit scale needs explicit provenance')
        recorded_scale, evidence = recorded_uniform_renderer_scale(self.mesh, {t['id']: t for t in snapshot['transforms']})
        _, inverse = anchor(snapshot, manifest)
        axes = np.linalg.norm(np.linalg.inv(inverse)[:3, :3], axis=0)
        anchor_scale = float(axes.mean())
        gram = (np.linalg.inv(inverse)[:3, :3] / axes).T @ (np.linalg.inv(inverse)[:3, :3] / axes)
        if not np.allclose(axes, anchor_scale, atol=1e-6, rtol=1e-5) or not np.allclose(gram, np.eye(3), atol=1e-5, rtol=0):
            raise Uncertifiable('Offline ancestor scale removal requires uniform unsheared anchor')
        self.scale *= recorded_scale / anchor_scale
        self.path = self.mesh['renderer_path']
        original = self.predict_raw(snapshot)
        self.baseline = surfaces[self.path]
        _, self.alignment = rigid_alignment(original, self.baseline, unit_scale=self.scale)
        self.alignment['recorded_baseline_renderer_scale'] = evidence
        self.alignment['recorded_baseline_anchor_scale'] = anchor_scale
        self.alignment['scale_policy'] = 'Recorded baseline renderer/anchor ratio, reused for all cases; never vertex fitted'
        self.rotation = np.asarray(self.alignment['rotation_row_vector'])
        self.translation = np.asarray(self.alignment['translation'])
        self.baseline_predicted = self.predict(snapshot)

    def predict_raw(self, snapshot):
        mesh = mesh_map(snapshot)[self.path]
        rig = copy.copy(self.rig)
        rig.verts = rig.verts + blendshape_delta(mesh)[0]
        return self.build(rig, snapshot['character']['shape_value_face'], None)[0]

    def predict(self, snapshot):
        return self.predict_raw(snapshot) * self.scale @ self.rotation + self.translation

    def compare(self, snapshot, surface):
        predicted = self.predict(snapshot)
        return {'absolute_residual': vertex_errors(surface[self.path], predicted),
                'response_residual': vertex_errors(surface[self.path] - self.baseline,
                                                  predicted - self.baseline_predicted)}


def analyze_manifest(manifest, directory, *, tolerance=1e-5, offline=True, head_only=False):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be finite and positive')
    report = {'schema_version': 1, 'scope': 'Captured head IDs/configurations only; no arbitrary-character certificate',
              'units': 'Captured anchor-local game units, not certified millimetres',
              'normalized_tolerance': tolerance, 'expression_policy': 'Fixed captured expression; changed weights invalidate isolated response',
              'coordinate_policy': 'Captured stable ancestor frame, no per-case fitted alignment or scale',
              'cases': [], 'baselines': {}, 'local_slopes': {},
              'offline_policy': 'o_head native diagnostic, one baseline rigid alignment, no per-case fit; baseline ABMX not modeled'}
    report['analysis_mesh_scope'] = ['o_head'] if head_only else ['all_captured_renderers']
    report['full_captured_mesh_set_certified'] = False
    if head_only:
        report['scope'] = 'Captured o_head component only; no eyeball, other-renderer or arbitrary-character certificate'
    dependency_paths = [Path(__file__), ROOT / 'tools/parameter_atlas/core.py',
                        ROOT / 'tools/unity_parity/geometry.py', ROOT / 'src/hs2_mesh_deform.py',
                        ROOT / 'src/hs2_sampling.py']
    report['reviewer_source_receipts'] = [{'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in dependency_paths]
    plan = None
    if manifest.get('predeclared_plan'):
        path = Path(manifest['predeclared_plan'])
        if not path.is_absolute():
            path = Path(directory) / path
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['predeclared_plan_sha256']:
            raise Uncertifiable('Predeclared plan receipt SHA-256 mismatch')
        plan = json.loads(raw.decode('utf-8-sig'))
        report['predeclared_plan_sha256'] = manifest['predeclared_plan_sha256']
    allow_rotation = manifest.get('allow_anchor_rotation_change') is True or manifest.get('coordinate_policy') == 'remove recorded rigid head pose; anchor local position and scale must stay fixed'
    if allow_rotation:
        enums_path = ROOT / 'data/hs2_head/enums.json'
        enums_raw = enums_path.read_bytes()
        destinations = json.loads(enums_raw)['dst']
        report['anchor_native_control_exclusion'] = {'path': str(enums_path), 'sha256': hashlib.sha256(enums_raw).hexdigest(),
                                                    'native_destination_names': destinations}
    cache = {}
    for name, declaration in manifest['baselines'].items():
        snapshot = load_receipt(declaration['geometry'], directory)
        snapshot = attach_frames(snapshot, snapshot)
        require_native(snapshot, declaration['native59'])
        if plan and (name not in plan['baselines'] or declaration['native59'] != plan['baselines'][name]):
            raise Uncertifiable('Baseline coefficients disagree with the predeclared plan')
        surfaces, parity, origin = measure_snapshot(snapshot, manifest, tolerance=tolerance, head_only=head_only)
        if allow_rotation and origin['name'] in destinations:
            raise Uncertifiable('Native control destination includes the normalization anchor')
        headpaths = [m['renderer_path'] for m in snapshot['meshes'] if m['mesh_name'] == 'o_head']
        if len(headpaths) != 1:
            raise Uncertifiable('Baseline needs exactly one o_head renderer')
        head_diagonal = float(np.linalg.norm(np.ptp(surfaces[headpaths[0]], axis=0)))
        if head_diagonal <= 0:
            raise Uncertifiable('Baseline head is degenerate')
        cache[name] = {'snapshot': snapshot, 'surfaces': surfaces, 'anchor': origin,
                       'diagonal': head_diagonal, 'local': {}, 'repeat': [], 'roles': {}, 'predictor': None}
        row = {'head_id': snapshot['character']['head_id'], 'native59': declaration['native59'],
               'head_diagonal': head_diagonal, 'renderer_count': len(surfaces), 'parity': parity}
        if offline:
            try:
                predictor = OfflinePredictor(snapshot, surfaces, manifest)
                cache[name]['predictor'] = predictor
                row['offline_baseline_alignment'] = predictor.alignment
                row['offline_cache_receipts'] = predictor.cache_receipts
            except (FileNotFoundError, ValueError, KeyError, StopIteration) as exc:
                row['offline_unavailable_reason'] = str(exc)
        report['baselines'][name] = row
    for case in manifest['cases']:
        row = {key: case.get(key) for key in ('name', 'kind', 'baseline_name', 'control', 'probe_role', 'value', 'bone', 'channel', 'axis')}
        row['trusted_isolated_response'] = False
        report['cases'].append(row)
        try:
            base = cache[case['baseline_name']]
            snapshot = attach_frames(base['snapshot'], load_receipt(case['geometry'], directory))
            requested = require_native(snapshot, case['native59'])
            require_abmx(snapshot, base['snapshot'], case)
            baseline = np.asarray(base['snapshot']['character']['shape_value_face'])
            control = case.get('control')
            if case['kind'] == 'native':
                if type(control) is not int or not 0 <= control < 59:
                    raise Uncertifiable('Native probe control must be an integer 0..58')
                if abs(requested[control] - float(case['value'])) > 2e-6:
                    raise Uncertifiable('Probe value disagrees with native59')
                role = case.get('probe_role')
                expected_value = {'local_minus': baseline[control] - float(manifest.get('local_step', plan.get('local_step', .05) if plan else .05)),
                                  'local_plus': baseline[control] + float(manifest.get('local_step', plan.get('local_step', .05) if plan else .05)),
                                  'native_min': 0., 'native_max': 1.,
                                  'extended_min': -.25, 'extended_max': 1.25}.get(role)
                if expected_value is None or abs(float(case['value']) - expected_value) > 2e-6:
                    raise Uncertifiable('Probe role/value disagrees with the declared sampling protocol')
                if plan:
                    declared = [row for row in plan['native_cases'] if row['name'] == case['name']]
                    if len(declared) != 1 or any(declared[0].get(key) != case.get(key) for key in ('baseline_name', 'control', 'probe_role', 'value', 'native59')):
                        raise Uncertifiable('Observed native case differs from predeclared plan')
                keep = np.arange(59) != control
                if np.max(np.abs(requested[keep] - baseline[keep])) > 2e-6:
                    raise Uncertifiable('Other native controls changed in isolated probe')
            elif case['kind'] in ('baseline', 'abmx'):
                if np.max(np.abs(requested - baseline)) > 2e-6:
                    raise Uncertifiable('Repeat/ABMX probe changed native59')
            else:
                raise Uncertifiable('Unknown case kind')
            surfaces, parity, origin = measure_snapshot(snapshot, manifest, tolerance=tolerance, head_only=head_only)
            require_same_anchor(base['anchor'], origin, allow_rotation=allow_rotation)
            if case['kind'] == 'abmx' and case.get('bone') == origin['name']:
                raise Uncertifiable('ABMX probe directly modifies the normalization anchor')
            changes = expression_changes(base['snapshot'], snapshot)
            row.update({'parity': parity, 'expression_changes': changes,
                        'anchor_rotation_change_max_abs': float(np.max(np.abs(np.asarray(origin['local_rotation_xyzw']) - np.asarray(base['anchor']['local_rotation_xyzw'])))),
                        'meshes': response_stats(base['surfaces'], surfaces, head_diagonal=base['diagonal'])})
            if case['kind'] == 'abmx':
                runtime = snapshot.get('abmx_runtime') or {}
                row['abmx_requested_patch'] = case['patch']
                row['abmx_rotation_excluded_bones'] = runtime.get('rotation_excluded_bones')
                row['requested_rotation_excluded'] = case['channel'] == 'rotation' and case['bone'] in (runtime.get('rotation_excluded_bones') or [])
                row['abmx_runtime_diagnostics'] = [bone for bone in runtime.get('bones', []) if bone['name'] == case['bone']]
                row['abmx_offline_replay_status'] = 'Not predicted by the cached native-only mapper; captured game surface response only'
            if changes:
                raise Uncertifiable('Expression settings or blendshape weights changed')
            if base['predictor'] is not None and case['kind'] != 'abmx':
                row['offline_native_diagnostic'] = base['predictor'].compare(snapshot, surfaces)
                row['offline_native_diagnostic']['response_error_normalized_by_baseline_head_diagonal'] = row['offline_native_diagnostic']['response_residual']['max_l2'] / base['diagonal']
            row['trusted_isolated_response'] = True
            if case['kind'] == 'native':
                role = case.get('probe_role')
                key = (control, role)
                if key in base['roles']:
                    raise Uncertifiable('Duplicate control/probe_role observation')
                base['roles'][key] = row
                if role in ('local_minus', 'local_plus'):
                    base['local'][key] = (float(case['value']), surfaces)
            elif case['kind'] == 'baseline':
                base['repeat'].append(row)
        except (Uncertifiable, ValueError, KeyError, IndexError, OSError) as exc:
            row['trusted_isolated_response'] = False
            row['untrusted_reason'] = str(exc)
    for name, base in cache.items():
        required = tuple(manifest.get('required_probe_roles', ROLES))
        if plan:
            roles = {row['probe_role'] for row in plan['native_cases'] if row['baseline_name'] == name}
            required = tuple(role for role in ROLES if role in roles) or ('local_minus', 'local_plus')
        slopes = []
        for control in range(59):
            minus, plus = base['local'].get((control, 'local_minus')), base['local'].get((control, 'local_plus'))
            if minus is None or plus is None or plus[0] <= minus[0]:
                continue
            denominator = plus[0] - minus[0]
            meshes = {}
            for path, baseline in base['surfaces'].items():
                derivative = (plus[1][path] - minus[1][path]) / denominator
                length = np.linalg.norm(derivative, axis=1)
                meshes[path] = {'max_surface_units_per_parameter_unit': float(length.max()),
                                'rms_surface_units_per_parameter_unit': float(np.sqrt(np.mean(length**2))),
                                'p95_surface_units_per_parameter_unit': float(np.percentile(length, 95)),
                                'mean_direction_xyz_per_parameter_unit': derivative.mean(axis=0).tolist(),
                                'axis_span_slope_xyz': ((np.ptp(plus[1][path], axis=0) - np.ptp(minus[1][path], axis=0)) / denominator).tolist(),
                                'predicted_max_for_plus_0_01': float(length.max() * .01)}
            slopes.append({'control': control, 'sample_interval': [minus[0], plus[0]], 'meshes': meshes})
        report['local_slopes'][name] = slopes
        missing = [{'control': control, 'probe_role': role} for control in range(59) for role in required if (control, role) not in base['roles']]
        repeats = base['repeat']
        driftmax = max((metric['normalized_by_baseline_head_diagonal']['max'] for repeat in repeats for metric in repeat['meshes'].values()), default=None)
        complete = not missing and len(slopes) == 59 and bool(repeats) and driftmax <= tolerance
        report['baselines'][name]['coverage'] = {'unique_native_controls': len({key[0] for key in base['roles']}),
                  'required_probe_roles': list(required), 'trusted_probe_count': len(base['roles']),
                  'local_slope_control_count': len(slopes), 'missing_probes': missing,
                  'baseline_repeat_count': len(repeats), 'max_repeat_drift_normalized': driftmax,
                  'repeat_drift_pass': driftmax is not None and driftmax <= tolerance,
                  'all_native_controls_verified_at_this_configuration': complete}
        per_renderer = {}
        for path in base['surfaces']:
            drift = max((repeat['meshes'][path]['normalized_by_baseline_head_diagonal']['max'] for repeat in repeats), default=None)
            per_renderer[path] = {'mesh_name': mesh_map(base['snapshot'])[path]['mesh_name'],
                                 'max_repeat_drift_normalized': drift,
                                 'repeat_drift_pass': drift is not None and drift <= tolerance,
                                 'all_native_controls_verified_at_this_configuration': not missing and len(slopes) == 59 and drift is not None and drift <= tolerance}
        report['baselines'][name]['coverage']['per_renderer'] = per_renderer
        prediction_rows = [row for (control, role), row in base['roles'].items() if role in required]
        prediction_failures = [{'control': row['control'], 'probe_role': row['probe_role'],
                                'max_response_error_normalized': row.get('offline_native_diagnostic', {}).get('response_error_normalized_by_baseline_head_diagonal')}
                               for row in prediction_rows if row.get('offline_native_diagnostic', {}).get('response_error_normalized_by_baseline_head_diagonal', float('inf')) > tolerance]
        report['baselines'][name]['coverage']['offline_native_o_head_prediction'] = {
            'probe_count': len(prediction_rows), 'failed_or_unavailable': prediction_failures,
            'all_required_predictions_within_tolerance': not missing and not prediction_failures and bool(prediction_rows),
            'scope': 'o_head native response from one cached baseline mapping; actual game-surface repeat gates remain separate'}
        for row in report['cases']:
            if row['baseline_name'] != name:
                continue
            row['interpretable_isolated_response_per_renderer'] = {path: row['trusted_isolated_response'] and status['repeat_drift_pass'] for path, status in per_renderer.items()}
            row['interpretable_isolated_response_analysis_scope'] = row['trusted_isolated_response'] and all(status['repeat_drift_pass'] for status in per_renderer.values())
            row['interpretable_isolated_response_all_renderers'] = not head_only and row['interpretable_isolated_response_analysis_scope']
    report['all_captured_native_configurations_verified'] = bool(report['baselines']) and manifest.get('complete', True) and all(row['coverage']['all_native_controls_verified_at_this_configuration'] for row in report['baselines'].values()) and all(row['trusted_isolated_response'] for row in report['cases'] if row['kind'] != 'abmx')
    report['full_captured_mesh_set_certified'] = not head_only and report['all_captured_native_configurations_verified']
    report['all_captured_native_configurations_verified_scope'] = report['analysis_mesh_scope']
    report['abmx_scope'] = {'trusted_sample_count': sum(row['kind'] == 'abmx' and row['trusted_isolated_response'] for row in report['cases']),
                          'interpretable_all_renderer_sample_count': sum(row['kind'] == 'abmx' and row['interpretable_isolated_response_all_renderers'] for row in report['cases']),
                          'interpretable_analysis_scope_sample_count': sum(row['kind'] == 'abmx' and row['interpretable_isolated_response_analysis_scope'] for row in report['cases']),
                          'interpretable_sample_count_per_renderer': {path: sum(row['kind'] == 'abmx' and row.get('interpretable_isolated_response_per_renderer', {}).get(path, False) for row in report['cases']) for path in next(iter(cache.values()))['surfaces']} if cache else {},
                          'all_bones_channels_verified': False}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--no-offline', action='store_true')
    parser.add_argument('--head-only', action='store_true', help='Certify only captured o_head surfaces; all raw source identity/expression checks remain')
    parser.add_argument('--tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    report = analyze_manifest(manifest, args.manifest.parent, tolerance=args.tolerance, offline=not args.no_offline, head_only=args.head_only)
    report['source_manifest_path'] = str(args.manifest.resolve())
    report['source_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(args.out), 'native_complete': report['all_captured_native_configurations_verified'],
                      'case_count': len(report['cases'])}))


if __name__ == '__main__':
    main()
