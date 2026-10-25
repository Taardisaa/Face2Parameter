"""Refreeze outputs/abmx_stability_20261005/original_source_only_target from the refrozen compiler artifact.

Same path as tools/abmx_stability/lowered.py's target freeze: lowered_target.prepare_target
on the frozen source-only compiler artifact, evidence JSON (exclusive create) and the
target vertices, followed by the 16-test abmx_stability suite recorded in source_freeze.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_stability.lowered import save
from tools.abmx_stability.lowered_target import prepare_target

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--old', type=Path, required=True, help='deprecated copy of the previous freeze directory')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    old = read(args.old/'source_freeze.json')
    target = prepare_target(read(args.artifact), read(args.contract))
    args.out.mkdir(parents=True)
    save(args.out/'evidence.json', target['evidence'])
    np.savez(args.out/'vertices.npz', vertices=target['vertices'])
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tools/abmx_stability', '-p', 'test_*.py', '-v']
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = run.stderr.strip().splitlines()
    ran = next((int(line.split()[1]) for line in tail if line.startswith('Ran ')), 0)
    failed = run.returncode != 0
    bindings = {str(p.resolve()): sha(p) for p in (args.artifact, ROOT/'tools/abmx_stability/lowered.py',
                ROOT/'tools/abmx_stability/lowered_target.py', ROOT/'tools/abmx_stability/test_lowered_target.py',
                args.out/'evidence.json', args.out/'vertices.npz')}
    value = {'bindings': bindings, 'original_source_only_fixture': True, 'runtime_candidate_certified': False,
             'unittest_command': old['unittest_command'], 'tests_passed': 0 if failed else ran, 'tests_run': ran,
             'tests_returncode': run.returncode, 'tests_skipped': sum('skipped' in line for line in run.stderr.splitlines()),
             'test_scope': old['test_scope'],
             'refrozen_from': {'path': str((args.old/'source_freeze.json').resolve()), 'sha256': sha(args.old/'source_freeze.json'),
                               'reason': 'tools/abmx_replay/model.py exact revision (Face2Parameter c810e75)'}}
    with (args.out/'source_freeze.json').open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    (args.out/'tests.txt').write_text(run.stderr, encoding='utf-8')
    print(json.dumps({k: value[k] for k in ('tests_passed', 'tests_run', 'tests_returncode')}))


if __name__ == '__main__':
    main()
