"""Source-bound four-regime actual stability audit, only after successful restoration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.abmx_stability.actual import audit_case
from tools.abmx_stability.classify import source_contract
from tools.abmx_multibone.geometry import DEFAULT_BONES
from tools.abmx_multibone.run import restored_flags
from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_multibone_quality.run import DEFAULT_ACCEPTANCE

REGIMES = {'scale_rotation_only', 'length_only', 'position_only', 'combined'}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--prediction-count', type=int, default=512)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('New output directory required')
    manifest, contract = read(args.manifest), read(args.contract)
    flags = restored_flags(manifest)
    for a, b in (('before', 'after'), ('expression_before', 'expression_after'), ('modifiers_before', 'modifiers_after')):
        if manifest[a] != manifest[b]:
            raise ValueError('Actual restoration records disagree: '+a)
    if sha(manifest['protocol_path']) != manifest['protocol_sha256']:
        raise ValueError('Predeclared protocol SHA changed')
    declared = read(manifest['protocol_path'])
    if declared['sampling_profile'] != 'slider_unlocker_18_2' or declared['normalized_whole_head_gate'] != 1e-5:
        raise ValueError('Installed sampler and fixed whole-head gate required')
    if declared['common_apply_count'] is not None or declared['common_count_runtime_certified'] is not False:
        raise ValueError('Observed counts do not certify a controlled common count')
    if declared['heads'] != [2] or len(declared['names']) != 4 or set(declared['names']) != set(DEFAULT_BONES):
        raise ValueError('This audit requires the declared head2 four-bone group')
    if declared['windows'] != [{'name': 'early', 'settle_frames': 5}, {'name': 'late', 'settle_frames': 18}, {'name': 'far60', 'settle_frames': 60}]:
        raise ValueError('Native settle cap60 requires exactly early5/late18/far60 window declarations')
    groups = {row['name']: row for row in declared['regimes']}
    if set(groups) != REGIMES or len(declared['regimes']) != 4:
        raise ValueError('Four unique declared regimes required')
    if len(manifest['cases']) != 4 or {case['regime'] for case in manifest['cases']} != REGIMES:
        raise ValueError('Four unique actual regime cases required')
    evidence = source_contract(contract)
    paths = [Path(__file__), Path(__file__).with_name('actual.py'), Path(__file__).with_name('classify.py'),
             ROOT/'tools/abmx_replay/model.py', ROOT/'tools/abmx_replay/validate_trace.py',
             ROOT/'tools/abmx_multibone/geometry.py', ROOT/'tools/abmx_multibone/run.py',
             ROOT/'tools/abmx_multibone_quality/run.py', ROOT/'tools/base_comparison/surface.py',
             ROOT/'tools/base_comparison/search.py', ROOT/'tools/unity_parity/geometry.py']
    report = {'schema_version': 1, 'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
              'protocol_path': manifest['protocol_path'], 'protocol_sha256': manifest['protocol_sha256'],
              'contract_path': str(args.contract.resolve()), 'contract_sha256': sha(args.contract),
              'installed_source_contract': evidence, 'restored_flags': flags,
              'implementation_sha256': {str(path.resolve()): sha(path) for path in paths},
              'whole_head_normalized_tolerance': 1e-5, 'temporal_surface_thresholds': DEFAULT_ACCEPTANCE,
              'temporal_evaluation_samples': 4096, 'temporal_evaluation_seed': 7381,
              'regimes': [], 'common_N_runtime_certified': False,
              'global_infinite_time_stability_certified': False, 'character_ready': False,
              'far90_collected_or_verified': False, 'far_window_native_settle_cap': 60}
    args.out.mkdir(parents=True)
    for case in manifest['cases']:
        label = case['regime']
        if case['head_id'] != 2 or set(case['names']) != set(DEFAULT_BONES):
            raise ValueError('Case head/bone group differs')
        for key in ('baseline_expected_native59', 'candidate_expected_native59', 'source_history_expected_native59'):
            if not np.array_equal(np.asarray(case[key], dtype=np.float32), np.asarray(declared[key], dtype=np.float32)):
                raise ValueError('Case expected native59 differs from predeclaration')
        group_declared = {**declared, 'patches': groups[label]['patches']}
        try:
            row = audit_case(case, group_declared, contract, args.out, label=label, long_count=args.prediction_count)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            row = {'label': label, 'rejected': True, 'rejection': str(exc), 'actual_temporal_stable_in_observed_windows': False,
                   'full_head_passed_count': 0, 'temporal_pairs_passed_count': 0}
        save(args.out/(label+'.json'), row)
        compact = {key: row[key] for key in ('label', 'full_head_passed_count', 'temporal_pairs_passed_count', 'actual_temporal_stable_in_observed_windows')}
        compact.update(rejected=row.get('rejected', False), rejection=row.get('rejection'), report_path=str((args.out/(label+'.json')).resolve()))
        if 'windows' in row:
            compact['window_counts'] = {w['window']: w['observed_candidate_counts'] for w in row['windows']}
            compact['temporal_surface'] = [{'pair': [p['earlier'], p['later']], 'rms': p['surface']['symmetric']['rms'],
                                           'p95': p['surface']['symmetric']['p95'], 'within_existing_thresholds': p['gate']['within_existing_surface_tolerances']}
                                          for p in row['temporal']]
            compact['supported_selected_branch_bones'] = 4-len(row['unsupported_selected_branches'])
        report['regimes'].append(compact)
        save(args.out/'progress.json', report)
        print(json.dumps(compact), flush=True)
    report['full_head_passed_count'] = sum(row['full_head_passed_count'] for row in report['regimes'])
    report['temporal_pairs_passed_count'] = sum(row['temporal_pairs_passed_count'] for row in report['regimes'])
    report['source_files_unchanged'] = (sha(args.manifest) == report['manifest_sha256'] and sha(args.contract) == report['contract_sha256']
                                       and sha(manifest['protocol_path']) == report['protocol_sha256']
                                       and all(sha(path) == digest for path, digest in report['implementation_sha256'].items()))
    if not report['source_files_unchanged']:
        raise ValueError('Sources changed during offline audit')
    save(args.out/'summary.json', report)
    # An observed temporal failure is a useful completed audit, not a reason to erase evidence.
    return 0 if report['full_head_passed_count'] == 12 else 2


if __name__ == '__main__':
    raise SystemExit(main())
