"""Actual stable-lowering audit: per-call replay, temporal gate, source-only target."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from tools.abmx_multibone.run import restored_flags
from tools.abmx_replay.model import require
from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_stability.actual import audit_case
from tools.abmx_stability.lowered_target import compare_window_to_target, prepare_target
from tools.abmx_stable_lowering.compiler import (
    verify_compiled_artifact,
    verify_execution_guard,
)


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--artifact', required=True, type=Path)
    parser.add_argument('--contract', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), 'New output directory required')
    manifest, artifact, contract = read(args.manifest), read(args.artifact), read(args.contract)
    restored = restored_flags(manifest)
    for before, after in (('before', 'after'), ('expression_before', 'expression_after'), ('modifiers_before', 'modifiers_after')):
        require(manifest[before] == manifest[after], 'Actual restoration records differ: '+before)
    require(sha(manifest['protocol_path']) == manifest['protocol_sha256'], 'Predeclared execution protocol changed')
    declared = read(manifest['protocol_path'])
    require(len(manifest['cases']) == 1, 'Exactly one physical lowered case required')
    case = manifest['cases'][0]
    require(declared['sampling_profile'] == 'slider_unlocker_18_2' and declared['normalized_whole_head_gate'] == 1e-5,
            'Explicit profile and unchanged wholehead gate required')
    require(declared['common_apply_count'] is None and declared['common_count_runtime_certified'] is False,
            'Observed counts cannot certify common N')
    require(declared['windows'] == [{'name': 'early', 'settle_frames': 5}, {'name': 'late', 'settle_frames': 18},
                                  {'name': 'far60', 'settle_frames': 60}], 'Native-cap60 window declaration required')
    require(declared['patches'] == artifact['executed_patches'] and declared['logical_patches'] == artifact['logical_patches'],
            'Physical/logical controls differ from frozen compiler')
    inputs = artifact['compiler_inputs']
    require(case['head_id'] == inputs['declared']['head_id'] and case['regime'] == 'position_only', 'Wrong physical head/regime')
    require(np.array_equal(np.asarray(declared['native59'], dtype=np.float64), np.asarray(inputs['declared']['native59'], dtype=np.float64)),
            'Execution native differs from compiler declaration')
    require(case['source_history']['trace_sha256'] == inputs['history']['trace_sha256']
            and case['source_history']['geometry']['sha256'] == inputs['history']['geometry']['sha256'],
            'Logical target source and observed case history differ')
    if 'compiler_artifact_sha256' in manifest:
        require(manifest['compiler_artifact_sha256'] == sha(args.artifact), 'Manifest compiler SHA mismatch')
    if 'compiler_artifact' in manifest:
        require(Path(manifest['compiler_artifact']['path']).resolve() == args.artifact.resolve()
                and manifest['compiler_artifact']['sha256'] == sha(args.artifact), 'Manifest compiler descriptor differs')
    paths = {str(args.manifest.resolve()): sha(args.manifest), str(args.artifact.resolve()): sha(args.artifact),
             str(args.contract.resolve()): sha(args.contract), manifest['protocol_path']: manifest['protocol_sha256']}
    for path in (Path(__file__), Path(__file__).with_name('lowered_target.py'), Path(__file__).with_name('actual.py'),
                 Path(__file__).with_name('classify.py'), ROOT/'src/hs2_mesh_deform.py', ROOT/'src/hs2_sampling.py',
                 ROOT/'tools/abmx_multibone/geometry.py', ROOT/'tools/abmx_multibone/run.py',
                 ROOT/'tools/abmx_multibone_quality/run.py', ROOT/'tools/base_comparison/surface.py',
                 ROOT/'tools/base_comparison/search.py', ROOT/'tools/unity_parity/geometry.py',
                 ROOT/'tools/abmx_replay/model.py', ROOT/'tools/abmx_replay/validate_trace.py'):
        paths[str(path.resolve())] = sha(path)
    paths.update(artifact['provenance']['source_files'])
    paths[case['trace']] = case['trace_sha256']
    descriptors = [(case['source_history']['geometry'], 'source_history_expected_native59'),
                   (case['baseline']['paired_geometry'], 'baseline_expected_native59'),
                   (case['identity_capture']['paired_geometry'], 'candidate_expected_native59')]
    descriptors.extend((window['geometry'], 'candidate_expected_native59') for window in case['windows'])
    for descriptor, key in descriptors:
        require(sha(descriptor['path']) == descriptor['sha256'], 'Snapshot source SHA differs')
        native_snapshot = read(descriptor['path'])
        require(np.array_equal(np.asarray(native_snapshot['character']['shape_value_face'], dtype=np.float32),
                               np.asarray(declared[key], dtype=np.float32)), 'Actual snapshot native differs: '+key)
        paths[descriptor['path']] = descriptor['sha256']
    if 'execution_guard' in manifest:
        descriptor = manifest['execution_guard']
        require(sha(descriptor['path']) == descriptor['sha256'], 'Producer execution guard changed')
        paths[descriptor['path']] = descriptor['sha256']
    verify_compiled_artifact(artifact)
    # Build and freeze target before loading candidate trace/geometry.
    target = prepare_target(artifact, contract)
    args.out.mkdir(parents=True)
    np.savez(args.out/'source_only_first_target.npz', vertices=target['vertices'],
             triangles=np.asarray(target['source_mesh']['source']['triangles']))
    save(args.out/'source_only_first_target.json', target['evidence'])
    trace = read(case['trace'])
    require(sha(case['trace']) == case['trace_sha256'], 'Candidate trace changed')
    guard_descriptor = case['identity_capture']['paired_geometry']
    require(sha(guard_descriptor['path']) == guard_descriptor['sha256'], 'Identity guard geometry changed')
    guard = verify_execution_guard(artifact, read(guard_descriptor['path']), args.contract, current_trace_metadata=trace['metadata'])
    save(args.out/'execution_guard.json', guard)
    actual = audit_case(case, declared, contract, args.out, label='position_only', long_count=2)
    save(args.out/'actual_physical_replay_temporal.json', actual)
    comparisons = [compare_window_to_target(target, window) for window in case['windows']]
    save(args.out/'source_only_first_target_comparisons.json', comparisons)
    require(all(sha(path) == digest for path, digest in paths.items()), 'Frozen input/algorithm changed during audit')
    summary = {'schema_version': 1, 'source_bindings': paths, 'restored_flags': restored,
               'full_head_passed_count': actual['full_head_passed_count'],
               'temporal_pairs_passed_count': actual['temporal_pairs_passed_count'],
               'logical_first_target_passed_count': sum(row['passed'] for row in comparisons),
               'window_observed_counts': {window['window']: window['observed_candidate_counts'] for window in actual['windows']},
               'target_error_max_normalized': {row['window']: row['comparison']['rigid_errors']['max_normalized'] for row in comparisons},
               'target_vertices_sha256': sha(args.out/'source_only_first_target.npz'),
               'target_prediction_sha256': sha(args.out/'source_only_first_target.json'),
               'source_files_unchanged': True, 'actual_candidate_after_used_to_generate_target': False,
               'semantic_scope': 'Stable physical position branch approximates FIRST clean logical combined apply; later logical radial iteration deliberately replaced.',
               'common_N_runtime_certified': False, 'arbitrary_candidates_runtime_certified': False,
               'global_infinite_stability_certified': False, 'character_ready': False}
    save(args.out/'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('full_head_passed_count', 'temporal_pairs_passed_count',
                                                  'logical_first_target_passed_count', 'target_error_max_normalized')}))
    return 0 if summary['full_head_passed_count'] == summary['temporal_pairs_passed_count'] == summary['logical_first_target_passed_count'] == 3 else 2


if __name__ == '__main__':
    raise SystemExit(main())
