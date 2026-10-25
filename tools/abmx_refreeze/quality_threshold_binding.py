"""Refreeze outputs/stateful_multibone_fresh_quality_20261005/threshold_binding_v1.json.

The binding pins the implementation SHA of the established quality chain. Commit
1a02d3f changed tools/geometry_quality/mesh_quality.py: self_intersections gained the
keyword `exclude_shared_vertex=True` (default = previous behaviour) and its report a
new `exclude_shared_vertex` key. The established path (analyze_mesh) calls it without
the keyword. This tool (1) loads the pinned previous mesh_quality.py, identified by
its recorded SHA, (2) runs analyze_snapshot with the old and the current module on
recorded actual geometry and requires identical reports apart from that one added
key, (3) rewrites the binding with the same fields from the same sources (the
established config, Thresholds(), DEFAULT_ACCEPTANCE, the six implementation files)
plus `refrozen_from` and the equivalence evidence. The old binding is kept under
deprecated/. No thresholds or acceptance values change.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from tools.abmx_replay.validate_trace import read, sha
from tools.geometry_quality import mesh_quality as current
from tools.stateful_multibone_fresh_quality.run import DEFAULT_ACCEPTANCE, THRESHOLD_BINDING, Thresholds

ROOT = Path(__file__).resolve().parents[2]
MESH_QUALITY = ROOT/'tools/geometry_quality/mesh_quality.py'


IGNORED = ('exclude_shared_vertex',   # key added by 1a02d3f
           'elapsed_seconds')         # wall-clock timing of self_intersections, not a result


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in IGNORED}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-binding', type=Path, required=True, help='deprecated copy of the previous binding')
    ap.add_argument('--old-revision', required=True, help='git revision holding the pinned mesh_quality.py (read only)')
    ap.add_argument('--geometry', type=Path, action='append', required=True, help='recorded actual MakerGeometryService snapshot')
    ap.add_argument('--work', type=Path, required=True)
    args = ap.parse_args()
    if THRESHOLD_BINDING.exists():
        raise ValueError('Move the previous binding to deprecated/ first; new immutable path required')
    old = read(args.old_binding)
    pinned = old['implementation_sha256'][str(MESH_QUALITY.resolve())]
    text = subprocess.run(['git', 'show', f'{args.old_revision}:tools/geometry_quality/mesh_quality.py'], cwd=ROOT,
                          capture_output=True, check=True).stdout
    args.work.mkdir(parents=True, exist_ok=True)
    old_file = args.work/'mesh_quality_pinned.py'
    old_file.write_bytes(text)
    if sha(old_file) != pinned:
        raise ValueError('Revision does not hold the pinned mesh_quality.py bytes')
    spec = importlib.util.spec_from_file_location('mesh_quality_pinned', old_file)
    previous = importlib.util.module_from_spec(spec)
    sys.modules['mesh_quality_pinned'] = previous      # dataclasses resolve their module during exec
    spec.loader.exec_module(previous)
    evidence = []
    for g in args.geometry:
        snapshot = read(g)
        a = previous.analyze_snapshot(copy.deepcopy(snapshot), mesh_names={'o_head'})
        b = current.analyze_snapshot(copy.deepcopy(snapshot), mesh_names={'o_head'})
        equal = json.dumps(strip(a), sort_keys=True, default=str) == json.dumps(strip(b), sort_keys=True, default=str)
        evidence.append({'geometry': str(g.resolve()), 'geometry_sha256': sha(g), 'meshes': ['o_head'],
                         'reports_identical_except': list(IGNORED), 'equal': equal})
        if not equal:
            raise ValueError('mesh_quality change is not behaviour-neutral on recorded geometry: '+str(g))
    config = read(old['established_config_path'])
    if sha(old['established_config_path']) != old['established_config_sha256']:
        raise ValueError('Established configuration changed')
    if asdict(Thresholds()) != old['thresholds'] or DEFAULT_ACCEPTANCE != old['surface_thresholds']:
        raise ValueError('Threshold definitions changed; this is not a refreeze')
    binding = {'established_config_path': old['established_config_path'], 'established_config_sha256': sha(old['established_config_path']),
               'thresholds': asdict(Thresholds()), 'surface_thresholds': dict(DEFAULT_ACCEPTANCE),
               'evaluation_samples': config['search']['evaluation_samples'], 'evaluation_seed': config['evaluation_seed'],
               'implementation_sha256': {p: sha(p) for p in old['implementation_sha256']},
               'previous_threshold_contract_path': old['previous_threshold_contract_path'],
               'previous_threshold_contract_sha256': old['previous_threshold_contract_sha256'],
               'previous_results_reused': False, 'actual_fresh_samples_evaluated': False,
               'refrozen_from': {'path': str(args.old_binding.resolve()), 'sha256': sha(args.old_binding),
                                 'changed_implementation': {str(MESH_QUALITY.resolve()): {'old': pinned, 'new': sha(MESH_QUALITY)}},
                                 'change': '1a02d3f: self_intersections(..., *, exclude_shared_vertex=True) and report key exclude_shared_vertex; '
                                           'established analyze_mesh call uses the default',
                                 'equivalence_on_recorded_geometry': evidence}}
    if binding['evaluation_samples'] != old['evaluation_samples'] or binding['evaluation_seed'] != old['evaluation_seed']:
        raise ValueError('Evaluation sampling changed')
    with THRESHOLD_BINDING.open('x', encoding='utf-8') as handle:
        json.dump(binding, handle, indent=2)
    print(json.dumps({'binding': str(THRESHOLD_BINDING), 'sha256': sha(THRESHOLD_BINDING), 'evidence': evidence}, indent=1))


if __name__ == '__main__':
    main()
