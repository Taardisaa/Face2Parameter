"""Refreeze a strict native-radial asset contract (identity_source_assets_strict_v2).

Reruns the tool's own path: asset_provider_strict.wrap_existing_v1 on the unchanged
V1 descriptor recorded in the deprecated contract, prepare_strict_asset_provider,
verify_snapshot_assets on the recorded source snapshot and, with --record, the
strict test suite, writing freeze.json/interface.json/source_guard.json/tests.txt in
the original schema. No game access.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tools.abmx_replay.validate_trace import read, sha
from tools.native_radial_target import asset_provider_strict as strict

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-contract', type=Path, required=True)
    ap.add_argument('--out-contract', type=Path, required=True)
    ap.add_argument('--record', type=Path, help='directory receiving freeze.json/interface.json/source_guard.json/tests.txt')
    ap.add_argument('--v1', type=Path, help='wrap this re-staged V1 asset contract instead of the recorded descriptor')
    args = ap.parse_args()
    if args.out_contract.exists():
        raise ValueError('New immutable contract path required')
    old = read(args.old_contract)
    v1 = old['v1_descriptor']
    if args.v1 is not None:      # V1 re-staged through stage_from_identity for a new identity history
        v1 = {'path': str(args.v1.resolve()), 'sha256': sha(args.v1)}
    elif sha(v1['path']) != v1['sha256']:
        raise ValueError('Recorded V1 descriptor changed')
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=args.out_contract.parent.parent) as tmp:
        desc = strict.wrap_existing_v1({'path': v1['path'], 'sha256': v1['sha256']}, Path(tmp)/'strict')
        args.out_contract.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(desc['path'], args.out_contract)
    t1 = time.perf_counter()
    descriptor = {'path': str(args.out_contract.resolve()), 'sha256': sha(args.out_contract),
                  'revision': desc['revision'], 'mesh_count': desc['mesh_count']}
    provider = strict.prepare_strict_asset_provider(descriptor['path'], descriptor['sha256'])
    t2 = time.perf_counter()
    source = read(read(v1['path'])['source_history']['geometry']['path'])
    guard = provider.verify_snapshot_assets(source)
    t3 = time.perf_counter()
    print(json.dumps({'descriptor': descriptor, 'guard_passed': guard['passed']}))
    if not args.record:
        return
    rec = args.record
    for name in ('freeze.json', 'interface.json', 'source_guard.json', 'tests.txt'):
        if (rec/name).exists():
            raise ValueError('New immutable record required: '+name)
    (rec/'source_guard.json').write_text(json.dumps(guard, indent=1, allow_nan=False)+'\n', encoding='utf-8')
    seconds = {'wrap_existing_v1': t1-t0, 'prepare': t2-t1, 'verify_observer': t3-t2}
    scope = 'offline historical identity-source smoke; no new runtime certification'
    sources = {str((ROOT/'tools/native_radial_target'/n).resolve()): sha(ROOT/'tools/native_radial_target'/n)
               for n in ('asset_provider_strict.py', 'test_asset_provider_strict.py', 'validate_strict_asset_runtime.py')}
    interface = {'descriptor': descriptor, 'seconds': seconds, 'scope': scope, 'source_files': sources}
    (rec/'interface.json').write_text(json.dumps(interface, indent=1)+'\n', encoding='utf-8')
    stream = io.StringIO()
    s0 = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('tools.native_radial_target.test_asset_provider_strict'))
    s1 = time.perf_counter()
    (rec/'tests.txt').write_text(stream.getvalue(), encoding='utf-8')
    ruff = subprocess.run([sys.executable, '-m', 'ruff', 'check', *sources], capture_output=True, text=True)
    freeze = {**interface, 'tests': {'passed': result.testsRun-len(result.failures)-len(result.errors), 'ran': result.testsRun,
              'seconds': round(s1-s0, 1), 'output_path': str((rec/'tests.txt').resolve()), 'output_sha256': sha(rec/'tests.txt'),
              'Ruff': 'passed' if ruff.returncode == 0 else 'failed: '+(ruff.stdout+ruff.stderr)[-400:]},
              'refrozen_from': {'path': str(args.old_contract.resolve()), 'sha256': sha(args.old_contract),
                                'reason': 'tools/abmx_replay/model.py exact revision (Face2Parameter c810e75)'}}
    freeze.pop('source_files')
    freeze = {'descriptor': descriptor, 'seconds': seconds, 'scope': scope, 'source_files': sources, **{k: v for k, v in freeze.items() if k not in ('descriptor', 'seconds', 'scope')}}
    (rec/'freeze.json').write_text(json.dumps(freeze, indent=1)+'\n', encoding='utf-8')
    print(json.dumps(freeze['tests']))


if __name__ == '__main__':
    main()
