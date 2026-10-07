"""Independently verify predeclared, held-out head displacement recommendations."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_live_response import (
    abmx_base_value,
    calibration_identity,
    canonical_hash,
    identity_mismatches,
)
from live_response import (
    Uncertifiable,
    anchor,
    attach_frames,
    expression_changes,
    load_receipt,
    measure_snapshot,
    require_abmx,
    require_native,
    require_same_anchor,
)

THRESHOLDS = {'geometry_normalized': 1e-5, 'repeat_normalized': 1e-5,
              'relative_prediction_error': .05, 'noise_multiplier': 3,
              'baseline_reuse_normalized': 1e-5}


def finite(value, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value) or (positive and value <= 0):
        raise Uncertifiable('Expected finite numeric value')
    return float(value)


def same_number(a, b, message):
    if not np.isclose(finite(a), finite(b), rtol=1e-8, atol=1e-12):
        raise Uncertifiable(message)


def unique(rows, field):
    keys = [row.get(field) for row in rows]
    if any(not isinstance(key, str) or not key for key in keys) or len(keys) != len(set(keys)):
        raise Uncertifiable(f'Duplicate or invalid {field}')
    return dict(zip(keys, rows))


def load_payload(entry, directory):
    path = (Path(directory) / entry['file']).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise Uncertifiable('Catalog data escapes its directory')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise Uncertifiable('Catalog data JS SHA-256 mismatch')
    text = raw.decode('utf-8-sig').strip()
    if not text.startswith('window.HS2_RESPONSE(') or not text.endswith(');'):
        raise Uncertifiable('Unsupported catalog callback encoding')
    payload = json.loads(text[len('window.HS2_RESPONSE('):-2])
    for key in ('id', 'kind', 'baseline_name', 'control', 'bone', 'channel', 'axis'):
        if payload.get(key) != entry.get(key):
            raise Uncertifiable('Catalog entry/data metadata differs')
    if payload.get('mesh') != 'o_head':
        raise Uncertifiable('Recommendations require explicit head-surface scope')
    return payload


def isolation(case, source):
    expected = np.asarray(source['character']['shape_value_face'], dtype=np.float64)
    requested = np.asarray(case['native59'], dtype=np.float64)
    if requested.shape != (59,) or not np.isfinite(requested).all():
        raise Uncertifiable('Invalid declared native59')
    if case['kind'] == 'native':
        index = case.get('control')
        if type(index) is not int or not 0 <= index < 59:
            raise Uncertifiable('Invalid native control index')
        expected[index] = finite(case['value'])
    elif case['kind'] == 'abmx':
        bone = case['bone']
        patch = case['patch']
        original = next((item for item in source.get('abmx_runtime', {}).get('bones', [])
                         if item['name'] == bone),
                        {'scale': [1, 1, 1], 'position': [0, 0, 0],
                         'rotation': [0, 0, 0], 'length': 1})
        wanted = {key: json.loads(json.dumps(original[key]))
                  for key in ('scale', 'position', 'rotation', 'length')}
        channel, axis = case['channel'], case['axis']
        if channel == 'length':
            wanted[channel] = finite(case['value'])
        elif channel in ('scale', 'position', 'rotation') and type(axis) is int and 0 <= axis < 3:
            wanted[channel][axis] = finite(case['value'])
        else:
            raise Uncertifiable('Invalid ABMX channel or axis')
        if patch.get('name') != bone or any(patch.get(key) != value for key, value in wanted.items()):
            raise Uncertifiable('ABMX patch changes another channel')
    elif case['kind'] != 'baseline':
        raise Uncertifiable('Unknown probe kind')
    if np.max(np.abs(expected - requested)) > 2e-6:
        raise Uncertifiable('Probe changed another native coefficient')


def head_measure(source, case, directory, manifest, tolerance):
    isolation(case, source)
    snapshot = attach_frames(source, load_receipt(case['geometry'], directory))
    require_native(snapshot, case['native59'])
    require_abmx(snapshot, source, case)
    if expression_changes(source, snapshot):
        raise Uncertifiable('Expression changed in isolated probe')
    measured, _, origin = measure_snapshot(snapshot, manifest, tolerance=tolerance, head_only=True)
    original, _ = anchor(source, manifest)
    require_same_anchor(original, origin, allow_rotation=True)
    if case.get('bone') == origin['name']:
        raise Uncertifiable('ABMX probe changes normalization anchor')
    return next(iter(measured.values()))


def original_source_arrays_match(first, current):
    if len(first['meshes']) != len(current['meshes']):
        raise Uncertifiable('Fresh baseline mesh count changed')
    for a, b in zip(first['meshes'], current['meshes']):
        for key in ('mesh_name', 'source_geometry_sha256', 'source', 'bone_names'):
            if a.get(key) != b.get(key):
                raise Uncertifiable(f'Fresh baseline source/palette changed: {key}')
        if [(s['name'], s['frames']) for s in a.get('blendshapes', [])] != [
                (s['name'], s['frames']) for s in b.get('blendshapes', [])]:
            raise Uncertifiable('Fresh baseline expression source frames changed')


def prediction_metrics(training_delta, training_signed_step, heldout_signed_step,
                       actual_delta, *, target, diagonal, training_noise, fresh_noise):
    predicted = training_delta / training_signed_step * heldout_signed_step
    actual_max = float(np.linalg.norm(actual_delta, axis=1).max())
    vector_error = float(np.linalg.norm(actual_delta - predicted, axis=1).max())
    budget = .05 * target + 3 * diagonal * max(training_noise, fresh_noise)
    magnitude_error = abs(actual_max - target)
    return {'actual_max': actual_max, 'target_units': target,
            'max_vector_error': vector_error,
            'relative_target_error': magnitude_error / target,
            'error_budget': budget, 'target_magnitude_error': magnitude_error,
            'verified_prediction': vector_error <= budget and magnitude_error <= budget}


def analyze(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode('utf-8-sig'))
    directory = manifest_path.parent
    if manifest.get('complete') is not True or manifest.get('public_configuration_restored') is not True:
        raise Uncertifiable('Complete collection and public restoration are required')
    if manifest.get('restoration_errors'):
        raise Uncertifiable('Restoration errors must remain empty')
    for key in ('snapshot', 'expression', 'abmx'):
        if key not in manifest.get('before', {}) or manifest['before'][key] != manifest.get('after', {}).get(key):
            raise Uncertifiable(f'Before/after public configuration differs: {key}')
    if manifest.get('after', {}).get('physics', {}).get('active') is not False:
        raise Uncertifiable('Physics freeze release was not verified')
    plan = load_receipt(manifest['plan_receipt'], directory)
    if plan.get('schema_version') != 1 or plan.get('thresholds') != THRESHOLDS:
        raise Uncertifiable('Unsupported predeclared plan or changed numerical thresholds')
    catalog = load_receipt(plan['catalog_receipt'], directory)
    training = load_receipt(plan['training_manifest_receipt'], directory)
    training_path = Path(plan['training_manifest_receipt']['path'])
    training_directory = training_path.parent if training_path.is_absolute() else directory / training_path.parent
    catalog_path = Path(plan['catalog_receipt']['path'])
    catalog_directory = catalog_path.parent if catalog_path.is_absolute() else directory / catalog_path.parent
    if catalog.get('complete') is not True or training.get('complete') is not True:
        raise Uncertifiable('Incomplete training collection/catalog')
    if catalog['provenance']['source_manifest_sha256'] != plan['training_manifest_receipt']['sha256']:
        raise Uncertifiable('Catalog does not bind the training manifest')
    catalog_entries = unique(catalog['entries'], 'id')
    declarations = unique(plan['entries'], 'name')
    cases = unique(manifest['cases'], 'name')
    if set(cases) != set(declarations):
        raise Uncertifiable('Observed/predeclared cases are not a bijection')
    for name, entry in declarations.items():
        if any(cases[name].get(key) != value for key, value in entry.items()):
            raise Uncertifiable('Observed case differs from predeclared plan')
    keys = [(entry['entry_id'], entry['sign']) for entry in plan['entries'] + plan['skipped']]
    expected_keys = {(key, sign) for key in catalog_entries for sign in (-1, 1)}
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise Uncertifiable('Every catalog entry needs exactly two declared signs or noise skips')
    if set(manifest['baselines']) != set(catalog['baselines']) or plan['baselines'] != catalog['baselines']:
        raise Uncertifiable('Baseline coverage/catalog declarations differ')
    training_cases = {case['geometry']['sha256']: case for case in training['cases']}
    report = {'schema_version': 1, 'analysis_mesh_scope': ['o_head'],
              'scope': 'Independent held-out points only; no whole-interval or arbitrary-state certificate',
              'source_manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'plan_receipt': manifest['plan_receipt'],
              'training_manifest_sha256': plan['training_manifest_receipt']['sha256'],
              'catalog_receipt': plan['catalog_receipt'], 'thresholds': THRESHOLDS,
              'entries': [], 'skipped': [], 'baselines': {}, 'coverage': {},
              'public_configuration_restored': True}
    cache = {}
    for name, declaration in manifest['baselines'].items():
        old = attach_frames(*[load_receipt(training['baselines'][name]['geometry'], training_directory)]*2)
        fresh = attach_frames(*[load_receipt(declaration['geometry'], directory)]*2)
        require_native(old, training['baselines'][name]['native59'])
        require_native(fresh, declaration['native59'])
        identity = calibration_identity(fresh, manifest)
        expected_identity = catalog['baselines'][name]['identity']
        if identity_mismatches(expected_identity, identity):
            raise Uncertifiable('Fresh complete calibration identity differs')
        if catalog['baselines'][name]['identity_sha256'] != canonical_hash(expected_identity):
            raise Uncertifiable('Catalog baseline identity hash changed')
        original_source_arrays_match(old, fresh)
        old_surfaces, _, _ = measure_snapshot(old, training, head_only=True)
        fresh_surfaces, _, _ = measure_snapshot(fresh, manifest, head_only=True)
        old_points, points = next(iter(old_surfaces.values())), next(iter(fresh_surfaces.values()))
        diagonal = finite(catalog['baselines'][name]['head_diagonal'], positive=True)
        same_number(diagonal, np.linalg.norm(np.ptp(old_points, axis=0)), 'Training head diagonal differs')
        reuse = float(np.linalg.norm(points-old_points, axis=1).max()/diagonal)
        training_noise = 0.
        for case in training['cases']:
            if case['kind'] == 'baseline' and case['baseline_name'] == name:
                value = head_measure(old, case, training_directory, training, 1e-5)
                training_noise = max(training_noise, float(np.linalg.norm(value-old_points, axis=1).max()/diagonal))
        same_number(training_noise, catalog['baselines'][name]['repeat_drift_normalized'], 'Training repeat noise differs')
        repeats = [dict(row, kind='baseline') for row in manifest.get('repeats', []) if row['baseline_name'] == name]
        if not repeats:
            raise Uncertifiable('Fresh baseline repeats are required')
        fresh_noise = 0.
        for case in repeats:
            value = head_measure(fresh, case, directory, manifest, 1e-5)
            fresh_noise = max(fresh_noise, float(np.linalg.norm(value-points, axis=1).max()/diagonal))
        usable = max(training_noise, fresh_noise) <= 1e-5 and reuse <= 1e-5
        report['baselines'][name] = {'identity_sha256': canonical_hash(expected_identity),
            'training_repeat_noise': training_noise, 'fresh_repeat_noise': fresh_noise,
            'baseline_reuse_normalized': reuse, 'baseline_reuse_pass': reuse <= 1e-5,
            'repeat_drift_pass': max(training_noise, fresh_noise) <= 1e-5,
            'repeat_count': len(repeats), 'head_diagonal': diagonal}
        cache[name] = (old, fresh, old_points, points, diagonal, training_noise, fresh_noise, usable)
    for declaration in plan['entries'] + plan['skipped']:
        entry = catalog_entries[declaration['entry_id']]
        name, sign = entry['baseline_name'], declaration['sign']
        old, fresh, old_points, points, diagonal, training_noise, fresh_noise, usable = cache[name]
        payload = load_payload(entry, catalog_directory)
        if payload['identity'] != catalog['baselines'][name]['identity']:
            raise Uncertifiable('Payload identity differs from bound baseline')
        choices = []
        receipts = payload['verification']['case_receipts']
        if len(receipts) != len(payload['roles']) or len(receipts) != len(payload['levels']):
            raise Uncertifiable('Payload sample declarations differ')
        for role, level, receipt in zip(payload['roles'], payload['levels'], receipts):
            case = training_cases.get(receipt['sha256'])
            if case is None or case['geometry'] != receipt:
                raise Uncertifiable('Payload receipt absent from training manifest')
            if case['kind'] != entry['kind'] or case['baseline_name'] != name:
                raise Uncertifiable('Payload receipt changes baseline or kind')
            for key in ('control', 'bone', 'channel', 'axis'):
                if case.get(key) != entry.get(key):
                    raise Uncertifiable('Payload receipt changes controlled channel')
            baseline_level = (float(old['character']['shape_value_face'][entry['control']])
                              if entry['kind'] == 'native' else abmx_base_value(case, old))
            same_number(level, case['value'], 'Payload sample level differs from raw request')
            step = float(case['value']) - baseline_level
            if np.sign(step) == sign and (entry['kind'] == 'abmx' or role == ('local_plus' if sign == 1 else 'local_minus')):
                choices.append((case, step, baseline_level))
        if len(choices) != 1:
            raise Uncertifiable('Unique one-sided training probe required')
        training_case, signed_training_step, baseline_level = choices[0]
        training_points = head_measure(old, training_case, training_directory, training, 1e-5)
        training_delta = training_points - old_points
        gain = float(np.linalg.norm(training_delta/signed_training_step, axis=1).max())
        target = min(diagonal*.01/100, gain*abs(signed_training_step)*.4)
        noise_guarded = gain <= 0 or target <= 3*diagonal*training_noise
        if declaration in plan['skipped']:
            if not noise_guarded:
                raise Uncertifiable('Skipped recommendation is not below measured noise floor')
            report['skipped'].append({**declaration, 'training_gain': gain,
                                     'noise_guard_verified': True, 'verified_prediction': False})
            continue
        if noise_guarded:
            raise Uncertifiable('Below-noise recommendation cannot claim a finite step')
        same_number(declaration['requested_target_percent'], .01, 'Requested target changed')
        same_number(declaration['max_probe_fraction'], .4, 'Probe fraction changed')
        same_number(declaration['training_step'], abs(signed_training_step), 'Training step changed')
        same_number(declaration['target_units'], target, 'Target differs from unrounded raw gain')
        same_number(declaration['target_percent'], target/diagonal*100, 'Effective target percent differs')
        same_number(declaration['step'], target/gain, 'Recommendation step differs from raw gain')
        same_number(declaration['signed_step'], sign*target/gain, 'Signed recommendation differs')
        same_number(declaration['value'], baseline_level+sign*target/gain, 'Held-out value differs')
        row = {key: declaration[key] for key in ('entry_id', 'sign', 'kind', 'baseline_name', 'signed_step')}
        row.update({'input_trusted': False, 'verified_prediction': False,
                    'heldout_value': declaration['value'], 'target_units': target,
                    'training_gain': gain, 'training_case_receipt': training_case['geometry'],
                    'baseline_identity_sha256': canonical_hash(catalog['baselines'][name]['identity']),
                    'baseline_reuse_pass': report['baselines'][name]['baseline_reuse_pass']})
        try:
            actual = head_measure(fresh, cases[declaration['name']], directory, manifest, 1e-5)
            row['input_trusted'] = True
            row.update(prediction_metrics(training_delta, signed_training_step,
                declaration['signed_step'], actual-points, target=target,
                diagonal=diagonal, training_noise=training_noise, fresh_noise=fresh_noise))
            row['verified_prediction'] = row['verified_prediction'] and usable
            if not usable:
                row['unverified_reason'] = 'Fresh baseline reuse or repeat-drift gate failed'
        except (ValueError, KeyError, OSError, IndexError) as exc:
            row['unverified_reason'] = str(exc)
        report['entries'].append(row)
    for name in cache:
        native_ids = {entry['id'] for entry in catalog_entries.values()
                      if entry['kind'] == 'native' and entry['baseline_name'] == name}
        if len(native_ids) != 59 or {entry['control'] for entry in catalog_entries.values()
            if entry['kind'] == 'native' and entry['baseline_name'] == name} != set(range(59)):
            raise Uncertifiable('All59 native controls required in each baseline')
        verified = {(row['entry_id'], row['sign']) for row in report['entries']
                    if row['baseline_name'] == name and row['verified_prediction']}
        report['coverage'][name] = {'native_controls': 59,
            'verified_native_signs': sum((key, sign) in verified for key in native_ids for sign in (-1, 1)),
            'all59_both_signs_verified': all((key, sign) in verified for key in native_ids for sign in (-1, 1))}
    report['all59_native_guidance_verified'] = all(row['all59_both_signs_verified'] for row in report['coverage'].values())
    report['selected_abmx_guidance_verified_or_noise_guarded'] = all(
        row['verified_prediction'] for row in report['entries'] if row['kind'] == 'abmx') and all(
        row['noise_guard_verified'] for row in report['skipped'])
    report['all_planned_predictions_verified'] = all(row['verified_prediction'] for row in report['entries'])
    report['source_receipts'] = [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in (Path(__file__), Path(__file__).with_name('live_response.py'),
                     Path(__file__).with_name('explore_live_response.py'),
                     Path(__file__).parents[1]/'unity_parity/geometry.py')]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'all_planned_predictions_verified': report['all_planned_predictions_verified'],
                      'entries': len(report['entries']), 'noise_guarded': len(report['skipped'])}))


if __name__ == '__main__':
    main()
