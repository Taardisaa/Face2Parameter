"""Separate actual quality gates relative to earlier identity; stability is separate."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.abmx_multibone.run import restored_flags
from tools.abmx_multibone_quality.run import (
    DEFAULT_ACCEPTANCE,
    absolute_summary,
    certify,
    identity_bound,
    load_snapshot,
    nuisance,
    save,
    sha,
    state,
    surface_gate,
)
from tools.abmx_replay.model import IDENTITY, require
from tools.base_comparison.search import quality_gate
from tools.base_comparison.surface import evaluate_surface
from tools.geometry_quality.cross_mesh import compare_crossings, cross_intersections
from tools.geometry_quality.mesh_quality import (
    Thresholds,
    analyze_mesh,
    baseline_comparison,
)
from tools.stateful_multibone_fresh_quality.run import (
    THRESHOLD_BINDING,
    validate_threshold_binding,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--replay-dir', required=True, type=Path)
    parser.add_argument('--established-config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    require(not args.out.exists(), 'New quality output directory required')
    read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    manifest, replay = read(args.manifest), read(args.replay_dir/'summary.json')
    restored_flags(manifest)
    for before, after in (('before', 'after'), ('expression_before', 'expression_after'), ('modifiers_before', 'modifiers_after')):
        require(manifest[before] == manifest[after], 'Actual restoration differs')
    require(replay['source_bindings'][str(args.manifest.resolve())] == sha(args.manifest), 'Replay binds another manifest')
    require(replay['full_head_passed_count'] == replay['temporal_pairs_passed_count'] == replay['logical_first_target_passed_count'] == 3,
            'Actual replay/temporal/target proof incomplete')
    require(all(sha(path) == digest for path, digest in replay['source_bindings'].items()), 'Frozen actual replay sources changed')
    threshold_binding = validate_threshold_binding(args.established_config)
    config = read(args.established_config)
    require({**DEFAULT_ACCEPTANCE, **config.get('acceptance', {})} == DEFAULT_ACCEPTANCE
            and config['search']['evaluation_samples'] == 4096 and config['evaluation_seed'] == 7381,
            'Established quality/surface contract changed')
    require(len(manifest['cases']) == 1, 'One actual lowered case required')
    case = manifest['cases'][0]
    require(case['head_id'] == 2 and {w['window'] for w in case['windows']} == {'early', 'late', 'far60'}, 'Actual case scope differs')
    binding = {str(args.manifest.resolve()): sha(args.manifest), str((args.replay_dir/'summary.json').resolve()): sha(args.replay_dir/'summary.json'),
               str(args.established_config.resolve()): sha(args.established_config), str(THRESHOLD_BINDING): sha(THRESHOLD_BINDING)}
    for path in (Path(__file__), ROOT/'tools/abmx_multibone_quality/run.py', ROOT/'tools/geometry_quality/mesh_quality.py',
                 ROOT/'tools/geometry_quality/cross_mesh.py', ROOT/'tools/unity_parity/geometry.py', ROOT/'tools/base_comparison/search.py',
                 ROOT/'tools/base_comparison/surface.py', ROOT/'tools/stateful_multibone_fresh_quality/run.py'):
        binding[str(path.resolve())] = sha(path)
    descriptor = case['source_history']['geometry']
    base, base_mesh, base_path = load_snapshot(descriptor, 2)
    binding[descriptor['path']] = descriptor['sha256']
    baseline_state = state(base, base_mesh)
    require(len(baseline_state['selected_actual_abmx_parameters']) == 4
            and all(value == IDENTITY for value in baseline_state['selected_actual_abmx_parameters'].values()), 'Earlier source not four-bone identity')
    require(np.array_equal(np.asarray(baseline_state['native59'], dtype=np.float32),
                           np.asarray(case['source_history_expected_native59'], dtype=np.float32)), 'Earlier identity native differs')
    args.out.mkdir(parents=True)
    bv, bf, baseline_cross_meshes, baseline_certificate = certify(base, base_mesh, descriptor, base_path, args.out/'baseline_lbs.json')
    quality_baseline, baseline_arrays = analyze_mesh(base_mesh, {t['id']: t for t in base['transforms']}, Thresholds(), space='baked_raw')
    baseline_cross = cross_intersections(baseline_cross_meshes, Thresholds())
    save(args.out/'baseline_absolute.json', {'source_sha256': descriptor['sha256'], 'state': baseline_state,
         'world_certificate': baseline_certificate, 'quality_report': quality_baseline, 'absolute': absolute_summary(quality_baseline),
         'cross_mesh_report': baseline_cross, 'absolute_anatomical_or_aesthetic_quality_certified': False})
    actual = read(args.replay_dir/'actual_physical_replay_temporal.json')
    binding[str((args.replay_dir/'actual_physical_replay_temporal.json').resolve())] = sha(args.replay_dir/'actual_physical_replay_temporal.json')
    summaries = []
    for window in case['windows']:
        current, mesh, path = load_snapshot(window['geometry'], 2)
        binding[str(path)] = window['geometry']['sha256']
        identity_bound(mesh, base_mesh)
        replay_windows = [w for w in actual['windows'] if w['window'] == window['window']]
        require(len(replay_windows) == 1 and replay_windows[0]['full_head_replay']['geometry_sha256'] == window['geometry']['sha256']
                and replay_windows[0]['full_head_replay']['passed'], 'Quality input differs from actual replay')
        cv, cf, current_cross_meshes, certificate = certify(current, mesh, window['geometry'], path, args.out/(window['window']+'_lbs.json'))
        require(np.array_equal(cf, bf), 'Actual quality ordered topology changed')
        quality, arrays = analyze_mesh(mesh, {t['id']: t for t in current['transforms']}, Thresholds(), space='baked_raw')
        quality['baseline'] = baseline_comparison(quality, quality_baseline, arrays, baseline_arrays, Thresholds())
        quality['baseline']['match_policy'] = 'exact_renderer_path_source_topology_prechecked'
        gate = quality_gate({'meshes': [quality]}, {'meshes': [quality_baseline]})
        current_state = state(current, mesh)
        for value in current_state['selected_actual_abmx_parameters'].values():
            require(np.isfinite(value['scale']).all() and np.all(np.asarray(value['scale']) > 0), 'Nonpositive/nonfinite actual selected modifier scale')
        surface = evaluate_surface(cv, cf, bv, bf, count=4096, seed=7381)
        crossings = cross_intersections(current_cross_meshes, Thresholds())
        cross_comparison = compare_crossings(current_cross_meshes, baseline_cross_meshes, crossings, baseline_cross)
        report = {'window': window['window'], 'geometry_sha256': window['geometry']['sha256'], 'baseline_geometry_sha256': descriptor['sha256'],
                  'quality_report': quality, 'relative_quality_gate': gate, 'absolute': absolute_summary(quality),
                  'world_certificate': certificate, 'cross_mesh_report': crossings, 'cross_mesh_baseline_comparison': cross_comparison,
                  'full_surface_vs_earlier_identity': surface, 'baseline_surface_threshold_diagnostic': surface_gate(surface, DEFAULT_ACCEPTANCE),
                  'baseline_to_candidate_nuisance': nuisance(baseline_state, current_state),
                  'positive_actual_selected_modifier_scale': True, 'likeness_or_anatomical_acceptance_certified': False}
        save(args.out/(window['window']+'_quality.json'), report)
        compact = {'window': window['window'], 'relative_quality_gate': gate, 'absolute': absolute_summary(quality),
                   'cross_mesh_new_pairs': len(cross_comparison.get('new_crossing_pairs', [])),
                   'identity_surface_rms': surface['symmetric']['rms'], 'identity_surface_p95': surface['symmetric']['p95'],
                   'identity_surface_diagnostic_passed': report['baseline_surface_threshold_diagnostic']['within_existing_surface_tolerances']}
        summaries.append(compact)
        print(json.dumps(compact), flush=True)
    require(all(sha(path) == digest for path, digest in binding.items()), 'Quality sources changed during audit')
    summary = {'schema_version': 1, 'source_bindings': binding, 'threshold_binding': threshold_binding,
               'thresholds': asdict(Thresholds()), 'surface_thresholds': DEFAULT_ACCEPTANCE,
               'evaluation_samples': 4096, 'evaluation_seed': 7381, 'baseline_absolute': absolute_summary(quality_baseline),
               'baseline_cross_mesh_crossing_count': baseline_cross['crossing_count'], 'windows': summaries,
               'relative_quality_passed_count': sum(w['relative_quality_gate']['quality_valid'] for w in summaries),
               'source_files_unchanged': True, 'existing_baseline_flags_retained': True,
               'scope': 'Actual complete o_head relative distortion/self-intersection/bone-frame gate, certified active head cross-mesh diagnostics. Baseline absolute flags retained; no anatomical/aesthetic/likeness certification.',
               'stability_and_first_target_proof_modified': False, 'character_ready': False}
    save(args.out/'summary.json', summary)
    return 0 if summary['relative_quality_passed_count'] == 3 else 2


if __name__ == '__main__':
    raise SystemExit(main())
