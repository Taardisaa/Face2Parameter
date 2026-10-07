"""Create a raw old-artifact example; no new images or game operations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import Manifest, digest, json_digest
from .measure import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--renderer-path', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        parser.error('New example directory required')
    snapshot = json.loads(args.geometry.read_text(encoding='utf-8'))
    matches = [m for m in snapshot['meshes'] if m['renderer_path'] == args.renderer_path]
    if len(matches) != 1:
        parser.error('One exact renderer identity required')
    mesh = matches[0]
    vertices = np.asarray(mesh['baked']['vertices'], float)
    features = []
    for i, axis_name in enumerate('xyz'):
        axis = np.eye(3)[i].tolist()
        definition = {'kind': 'formal_surface', 'region': {'kind': 'full_asset_submesh', 'submesh_id': 0},
                      'axis': axis, 'numerical_tolerance_game_units': 1e-7,
                      'recomputation_rule': 'recompute_on_each_actual_surface'}
        features.append(definition | {'id': f'raw_{axis_name}_span', 'operation': 'directional_span'})
        features.append(definition | {'id': f'raw_{axis_name}_mid_bounds_slice', 'operation': 'plane_intersection_segments',
                        'plane_offset': float((vertices[:, i].min()+vertices[:, i].max())/2)})
    features.append({'id': 'full_submesh_area', 'kind': 'formal_surface', 'operation': 'surface_area',
                     'region': {'kind': 'full_asset_submesh', 'submesh_id': 0}, 'axis': None,
                     'numerical_tolerance_game_units': 1e-7, 'recomputation_rule': 'recompute_on_each_actual_surface'})
    features.append({'id': 'raw_z_support_set', 'kind': 'formal_surface', 'operation': 'directional_support_set',
                     'region': {'kind': 'full_asset_submesh', 'submesh_id': 0}, 'axis': [0., 0., 1.],
                     'numerical_tolerance_game_units': 1e-7, 'recomputation_rule': 'recompute_on_each_actual_surface'})
    for id in ['N', 'AL', 'AR', 'ML', 'MR', 'C']:
        features.append({'id': f'unobservable_named_{id}', 'kind': 'unobservable',
                         'intended_feature': f'Previously registered named skin hypothesis {id}',
                         'reason': 'Prior raw training observations remain ambiguous/null; formal geometry does not fill them.'})
    data = {'schema_version': 1, 'purpose': 'measurement_diagnostic_only',
            'geometry': {'path': str(args.geometry.resolve()), 'sha256': digest(args.geometry),
                'head_id': snapshot['character']['head_id'], 'renderer_path': args.renderer_path,
                'source_geometry_sha256': mesh['source_geometry_sha256'], 'frame_count': snapshot['frame_count'],
                'pose_signature': snapshot['pose_signature'], 'coordinate_space': 'renderer_baked_raw',
                'native_face_values_sha256': json_digest(snapshot['character']['shape_value_face']),
                'expression_configuration_sha256': json_digest(snapshot['character']['expression']),
                'abmx_parameter_values_status': 'unverified_not_bound_by_this_contract',
                'units': 'game_units_unscaled', 'frame_policy_id': 'literal_raw_renderer_axes_no_cross_base_equivalence_v1'},
            'features': features}
    manifest = Manifest.model_validate(data)
    report = execute(manifest)
    args.out_dir.mkdir(parents=True)
    manifest_path = args.out_dir/'manifest.json'
    manifest_path.write_text(json.dumps(manifest.model_dump(), indent=2, allow_nan=False), encoding='utf-8')
    report['manifest_sha256'] = digest(manifest_path)
    report['example_scope'] = 'Real previously exported old snapshot, literal raw mesh math only; no new anatomical acceptance.'
    (args.out_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (args.out_dir/'manifest.schema.json').write_text(json.dumps(Manifest.model_json_schema(), indent=2), encoding='utf-8')
    print(json.dumps({'features': len(features), 'out': str(args.out_dir.resolve()),
                      'raw_spans': {row['id']: row['result']['span_game_units'] for row in report['results'] if 'span_game_units' in row['result']},
                      'anatomical_acceptance': False}))


if __name__ == '__main__':
    main()
