"""Refreeze outputs/native_radial_target_20261005/source_smoke_frozen_v1 from its recorded inputs.

The original freeze (2026-10-05) left no driver script; its recorded inputs are the
artifact's own `compiler_inputs` (history, declared, contract_path) and the source
manifest named by report.json `source_manifest_sha256`. This reruns the tool's own
freeze path: compile_native_target -> prepare_execution_guard (full regeneration) ->
GuardContext.verify_snapshot on the recorded source snapshot/trace metadata, then the
same 11-test NativeTarget suite, and writes the same four files. No game access.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import unittest
from pathlib import Path

from tools.abmx_replay.validate_trace import read, sha
from tools.native_radial_target.compiler import compile_native_target, prepare_execution_guard

ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261005/native_radial_source_v1/live_cases.json'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old', type=Path, required=True, help='deprecated/ copy of the previous freeze directory')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    old_report, old = read(args.old/'report.json'), read(args.old/'compiled.json')
    if sha(SOURCE_MANIFEST) != old_report['source_manifest_sha256']:
        raise ValueError('Recorded source manifest changed')
    if sha(args.old/'compiled.json') != old_report['artifact_sha256']:
        raise ValueError('Deprecated artifact is not the one the old report froze')
    for name in ('compiled.json', 'source_guard.json', 'report.json', 'tests.txt'):
        if (args.out/name).exists():
            raise ValueError('New immutable output path required: '+name)
    inputs = old['compiler_inputs']
    t0 = time.perf_counter()
    compiled = compile_native_target(inputs['history'], inputs['declared'], inputs['contract_path'])
    t1 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    artifact = args.out/'compiled.json'
    artifact.write_text(json.dumps(compiled, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    digest = sha(artifact)
    t2 = time.perf_counter()
    context = prepare_execution_guard(read(artifact), inputs['contract_path'], artifact_path=str(artifact), artifact_sha256=digest)
    t3 = time.perf_counter()
    history = inputs['history']
    guard = context.verify_snapshot(read(history['geometry']['path']), read(history['trace'])['metadata'])
    t4 = time.perf_counter()
    (args.out/'source_guard.json').write_text(json.dumps(guard, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromName('tools.native_radial_target.test_native_target')
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (args.out/'tests.txt').write_text(stream.getvalue(), encoding='utf-8')
    pkg = ROOT/'tools/native_radial_target'
    report = {'passed': guard['passed'] and result.wasSuccessful(), 'tests_run': result.testsRun,
              'source_calls': len(read(history['trace'])['events']), 'source_bones': len(compiled['all30_order']),
              'compile_seconds': t1-t0, 'prepare_seconds': t3-t2, 'guard_seconds': t4-t3,
              'physical_length_all_one': all(p['length'] == 1.0 for p in compiled['executed_patches']),
              'sources': {str(p.resolve()): sha(p) for p in (pkg/'adapter.py', pkg/'compiler.py', pkg/'test_native_target.py', pkg/'__init__.py')},
              'artifact_sha256': digest, 'source_manifest_sha256': sha(SOURCE_MANIFEST),
              'candidate_runtime_certified': False,
              'refrozen_from': {'path': str((args.old/'compiled.json').resolve()), 'sha256': old_report['artifact_sha256'],
                                'reason': 'tools/abmx_replay/model.py exact revision (Face2Parameter c810e75)'}}
    (args.out/'report.json').write_text(json.dumps(report, indent=1, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('passed', 'tests_run', 'artifact_sha256', 'physical_length_all_one')}))


if __name__ == '__main__':
    main()
