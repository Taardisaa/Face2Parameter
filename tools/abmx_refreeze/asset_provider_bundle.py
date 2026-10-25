"""Refreeze outputs/native_radial_target_20261006/asset_provider_final_v1 for a new head3 identity source.

The staged V1 asset contract binds one earlier clean identity history. The 2026-10-25
live head3 re-run recorded a new history, so its contract is staged again through the
tool's own `asset_provider.stage_from_identity` with the same explicit override rule as
`test_asset_provider` (the o_tooth explicit-zipmod row of source_asset_audit_v2), then
`prepare_asset_provider`, `verify_snapshot_assets` on the new source snapshot and the
source-only `independent_target` of the new compiled artifact. The records keep the
original bundle's schema (source_report/source_asset_guard/source_only_target/tests/
interface) and the current sources are copied next to them as `files_frozen`.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import unittest
from pathlib import Path

from tools.abmx_replay.validate_trace import read, sha
from tools.native_radial_target.asset_provider import prepare_asset_provider, stage_from_identity

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT/'tools/native_radial_target'
AUDIT = ROOT/'outputs/native_radial_target_20261006/source_asset_audit_v2/head3_source.json'


def overrides_from_audit():
    report = read(AUDIT)
    mesh = next(m for m in report['meshes'] if m['mesh_name'] == 'o_tooth')
    asset = next(v for v in mesh['asset_candidates'] if v['source'] == 'explicit_zipmod')
    o = asset['origin']
    return {mesh['renderer_path']: {'archive': o['path'], 'archive_sha256': o['sha256'], 'member': o['member'],
                                    'member_sha256': o['member_sha256'], 'prefab': o['prefab'],
                                    'renderer_path_id': asset['renderer_path_id'], 'mesh_path_id': asset['mesh_path_id']}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', type=Path, required=True, help='compiled artifact of the live run whose history is staged')
    ap.add_argument('--old-bundle', type=Path, required=True, help='deprecated copy of the previous bundle')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists() and sorted(x.name for x in args.out.iterdir()) != ['deprecated']:
        raise ValueError('New bundle directory required (only a deprecated/ subfolder may already exist)')
    artifact = read(args.artifact)
    history, abmx = artifact['compiler_inputs']['history'], artifact['compiler_inputs']['contract_path']
    overrides = overrides_from_audit()
    t0 = time.perf_counter()
    descriptor = stage_from_identity(history, abmx, overrides, args.out/'staged')
    t1 = time.perf_counter()
    provider = prepare_asset_provider(descriptor['path'], descriptor['sha256'])
    t2 = time.perf_counter()
    source = read(history['geometry']['path'])
    guard = provider.verify_snapshot_assets(source)
    t3 = time.perf_counter()
    target = provider.independent_target(artifact, read(abmx), source_gaze_policy='frozen_source_locals')
    t4 = time.perf_counter()
    checks = target['evidence']['source_mesh_checks']
    worst = max(c['source_error']['max_normalized'] for c in checks.values())
    save = lambda name, value: (args.out/name).write_text(json.dumps(value, indent=1, allow_nan=False, default=str)+'\n', encoding='utf-8')  # noqa: E731
    save('source_asset_guard.json', guard)
    save('source_only_target.json', target['evidence'])
    save('source_report.json', {'descriptor': descriptor, 'timings_seconds': {'stage': t1-t0, 'prepare': t2-t1, 'observer_asset_guard': t3-t2,
                                                                                'source_target': t4-t3},
                                'all8_source_passed': all(c['native_source_model_passed'] for c in checks.values()) and len(checks) == 8,
                                'worst_source_max_normalized': worst, 'only_historical_source_tested': True,
                                'new_actual_execution_certified': False,
                                'source_artifact': {'path': str(args.artifact.resolve()), 'sha256': sha(args.artifact)}})
    stream = io.StringIO()
    s0 = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('tools.native_radial_target.test_asset_provider'))
    sources = {}
    for name in ('asset_provider.py', 'validate_asset_runtime.py', 'test_asset_provider.py'):
        shutil.copyfile(PKG/name, args.out/name)
        sources[str((PKG/name).resolve())] = sha(PKG/name)
    save('tests.json', {'tests': result.testsRun, 'passed': result.wasSuccessful(), 'failures': len(result.failures),
                        'errors': len(result.errors), 'elapsed_seconds': time.perf_counter()-s0, 'output': stream.getvalue(),
                        'source_files': sources, 'only_historical_source_tested': True, 'new_live_execution_tested': False})
    old_interface = read(args.old_bundle/'interface.json')
    interface = {**old_interface, 'files_frozen': sources, 'original_files_unchanged': True,
                 'contract': descriptor, 'test_report': str((args.out/'tests.json').resolve()),
                 'source_report': str((args.out/'source_report.json').resolve()),
                 'refrozen_from': {'path': str((args.old_bundle/'staged/asset_contract.json').resolve()),
                                   'sha256': sha(args.old_bundle/'staged/asset_contract.json'),
                                   'reason': 'new head3 identity source history (2026-10-25 live re-run)'}}
    save('interface.json', interface)
    print(json.dumps({'descriptor': descriptor, 'guard_passed': guard['passed'], 'worst_source_max_normalized': worst,
                      'tests': result.testsRun, 'tests_passed': result.wasSuccessful()}, indent=1))


if __name__ == '__main__':
    main()
