"""Installed-source branch classification of a stopped trace; no runtime-stability verdict."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.abmx_stability.classify import classify, source_contract
from tools.abmx_replay.validate_trace import read, sha, validate
from tools.abmx_multibone.geometry import DEFAULT_BONES
from tools.abmx_replay.model import IDENTITY


def active(value):
    return any(not np.array_equal(np.asarray(value[key], dtype=np.float32), np.asarray(IDENTITY[key], dtype=np.float32))
               for key in IDENTITY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=512)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('New output file required')
    trace, contract = read(args.trace), read(args.contract)
    binding = source_contract(contract)
    replay = validate(trace, contract)
    if not replay['passed'] or replay['inter_call_boundaries']:
        raise ValueError('Complete trace must independently replay without external boundaries')
    classifications = {}
    for name in DEFAULT_BONES:
        candidates = [event for event in trace['events'] if event['bone_name'] == name and active(event['resolved_modifier'])]
        if not candidates:
            raise ValueError('No actual active candidate Apply for '+name)
        classifications[name] = {'first_active_sequence': candidates[0]['sequence'],
                                 'observed_active_call_count': len(candidates),
                                 'classification': classify(candidates[0], long_count=args.count)}
    report = {'schema_version': 1, 'trace_path': str(args.trace.resolve()), 'trace_sha256': sha(args.trace),
              'contract_sha256': sha(args.contract), 'source_contract': binding,
              'trace_total_calls_independently_replayed': len(trace['events']), 'classifications': classifications,
              'actual_after_used_as_prediction_input': False, 'long_run_is_runtime_evidence': False,
              'actual_temporal_geometry_measured': False, 'actual_runtime_stability_proven': False,
              'controlled_common_N_runtime_certified': False,
              'implementation_sha256': {str(path.resolve()): sha(path) for path in (Path(__file__),
                  Path(__file__).with_name('classify.py'), ROOT/'tools/abmx_replay/model.py', ROOT/'tools/abmx_replay/validate_trace.py')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({name: row['classification']['regime'] for name, row in classifications.items()}))


if __name__ == '__main__':
    main()
