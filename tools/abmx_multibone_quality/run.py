"""Audit actual recorded geometry with unchanged established quality gates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools/base_comparison'))
sys.path.insert(0, str(ROOT/'tools/geometry_quality'))

import numpy as np
from scipy.spatial.transform import Rotation
from tools.geometry_quality.mesh_quality import Thresholds, analyze_mesh, baseline_comparison
from tools.geometry_quality.cross_mesh import load_certified_head, cross_intersections, compare_crossings
from tools.unity_parity.geometry import analyze_snapshot as certify_lbs
from tools.base_comparison.search import quality_gate, DEFAULT_ACCEPTANCE
from tools.base_comparison.surface import evaluate_surface, topology_hash, content_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load_snapshot(descriptor, head):
    path = Path(descriptor['path'])
    require(path.is_absolute() and sha(path) == descriptor['sha256'], 'Actual geometry bytes/path changed')
    snapshot = json.loads(path.read_text(encoding='utf-8'))
    require(snapshot['schema_version'] == 1 and snapshot['snapshot_kind'] == 'maker_live_skinned_geometry', 'Wrong geometry kind')
    require(snapshot['character']['head_id'] == head, 'Wrong same-head baseline/candidate')
    require(snapshot['frame_count'] == snapshot['frame_count_end'] == descriptor['frame_count'], 'Geometry frame mismatch')
    require(snapshot['pose_signature'] == descriptor['pose_signature'], 'Geometry pose mismatch')
    meshes = [m for m in snapshot['meshes'] if m['mesh_name'] == 'o_head' and m['enabled'] and m['active_in_hierarchy']]
    require(len(meshes) == 1, 'Need exact one actual active head surface')
    return snapshot, meshes[0], path


def identity_bound(current, baseline):
    require(current['renderer_path'] == baseline['renderer_path'], 'Renderer identity differs')
    require(current['source_geometry_sha256'] == baseline['source_geometry_sha256'], 'Source asset differs')
    require(np.array_equal(current['source']['triangles'], baseline['source']['triangles']), 'Source ordered topology differs')
    require(np.array_equal(current['baked']['triangles'], baseline['baked']['triangles']), 'Actual baked topology differs')
    require(np.array_equal(current['source']['vertices'], baseline['source']['vertices']), 'Actual source vertices differ')
    require(np.array_equal(current['source']['bindposes'], baseline['source']['bindposes']), 'Actual bindposes differ')
    require(np.array_equal(current['source']['bone_indices'], baseline['source']['bone_indices']), 'Actual skin indices differ')
    require(np.array_equal(current['source']['bone_weights'], baseline['source']['bone_weights']), 'Actual skin weights differ')


def certify(snapshot, mesh, descriptor, path, output):
    certificate = certify_lbs(snapshot, normalized_tolerance=1e-5)
    certificate.update(snapshot_path=str(path), snapshot_sha256=descriptor['sha256'])
    save(output, certificate)
    head_meshes, skipped = load_certified_head(snapshot, descriptor['sha256'], certificate)
    selected = [m for m in head_meshes if m.name == 'o_head' and m.path == mesh['renderer_path']]
    require(len(selected) == 1, 'Actual head world conversion not independently certified')
    world = selected[0]
    q = np.asarray(mesh['renderer_rotation_xyzw'], float)
    require(q.shape == (4,) and np.isfinite(q).all() and abs(np.linalg.norm(q)-1) <= 1e-5, 'Invalid recorded renderer proper rotation')
    rotation = Rotation.from_quat(q).as_matrix()
    require(abs(np.linalg.det(rotation)-1) <= 1e-12, 'Recorded renderer removal must be proper rigid')
    position = np.asarray(mesh['renderer_position'], float)
    canonical = (world.vertices-position) @ rotation
    return canonical, world.faces, head_meshes, {
        'certificate_path': str(output.resolve()), 'certificate_sha256': sha(output),
        'selected_world_candidate': world.candidate, 'residual_floor_game_units': world.residual_floor,
        'recorded_renderer_rotation_xyzw': q.tolist(), 'recorded_renderer_position': position.tolist(),
        'proper_rigid_removal': 'actual recorded renderer translation/rotation only, never vertex-fit',
        'fitted_rotation': False, 'fitted_translation': False, 'fitted_scale': False, 'fitted_affine': False,
        'canonical_content_sha256': content_hash(canonical, world.faces), 'skipped_renderers': skipped}


def state(snapshot, mesh):
    selected_names = {'cf_J_Chin_rs', 'cf_J_ChinTip_s', 'cf_J_CheekUp_L', 'cf_J_CheekUp_R'}
    bones = {b['name']: {k: b[k] for k in ('scale', 'length', 'position', 'rotation')}
             for b in snapshot['abmx_runtime']['bones'] if b['name'] in selected_names}
    return {'head_id': snapshot['character']['head_id'], 'frame_count': snapshot['frame_count'],
        'pose_signature': snapshot['pose_signature'], 'visibility_signature': snapshot.get('visibility_sample_signature'),
        'native59': snapshot['character']['shape_value_face'], 'expression_configuration': snapshot['character']['expression'],
        'active_blendshapes': [{'name': b.get('name'), 'weight': b['current_weight']} for b in mesh.get('blendshapes', []) if b['current_weight'] != 0],
        'selected_actual_abmx_parameters': bones, 'trace_cursor': snapshot.get('abmx_trace_cursor'),
        'renderer_rotation_xyzw': mesh['renderer_rotation_xyzw'], 'renderer_position': mesh['renderer_position'],
        'renderer_lossy_scale': mesh['renderer_lossy_scale'], 'source_geometry_sha256': mesh['source_geometry_sha256'],
        'renderer_path': mesh['renderer_path'], 'renderer_enabled': mesh['enabled'], 'renderer_active': mesh['active_in_hierarchy']}


def nuisance(a, b):
    return {key+'_identical': a[key] == b[key] for key in ('head_id', 'native59', 'expression_configuration',
            'active_blendshapes', 'selected_actual_abmx_parameters', 'source_geometry_sha256', 'renderer_path',
            'renderer_enabled', 'renderer_active', 'renderer_rotation_xyzw', 'renderer_position', 'renderer_lossy_scale',
            'pose_signature', 'visibility_signature')} | {
            'frame_difference': b['frame_count']-a['frame_count'], 'early': a, 'late': b,
            'external_writers_identified': False, 'parameter_identity_implies_pose_identity': False}


def surface_gate(report, thresholds):
    outcomes = {key: report['symmetric'][key] <= thresholds[key] for key in ('rms', 'p95', 'max_sampled')}
    outcomes['normal_p95_degrees'] = report['oriented_normal_degrees']['p95'] <= thresholds['normal_p95_degrees']
    return {'within_existing_surface_tolerances': all(outcomes.values()), 'checks': outcomes,
            'thresholds': thresholds, 'interpretation': 'Deviation from reference only; not target likeness or anatomical acceptance'}


def absolute_summary(report):
    return {'degenerate_triangle_ids': report['degenerate_triangle_ids'],
        'crossing_count': report['self_intersections']['true_crossing_or_area_overlap_count'],
        'contact_count': report['self_intersections']['contact_count'],
        'nonmanifold_edges': report['topology']['nonmanifold_edge_count'],
        'nonmanifold_vertices': report['topology']['nonmanifold_vertex_count'],
        'aspect_warning_count': len(report['aspect_warning_triangle_ids']),
        'negative_bone_world_frames': report['bone_world_determinants']['negative_count'],
        'singular_bone_world_frames': report['bone_world_determinants']['near_singular_count'],
        'absolute_anatomical_quality_certified': False,
        'preexisting_flags_are_retained_not_automatically_acceptable': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--replay-summary', type=Path, required=True)
    parser.add_argument('--established-config', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out_dir.exists(), 'New output directory required')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    replay = json.loads(args.replay_summary.read_text(encoding='utf-8'))
    require(replay['manifest_sha256'] == sha(args.manifest), 'Replay summary binds different live manifest')
    require(all(manifest.get(k) is True for k in ('state_restored', 'expression_restored', 'bone_restored')), 'Restoration flags missing')
    require(manifest['before'] == manifest['after'] and manifest['expression_before'] == manifest['expression_after']
            and manifest['modifiers_before'] == manifest['modifiers_after'], 'Recorded restoration evidence differs')
    require({c['head_id'] for c in manifest['cases']} == {0,1,2,3} and len(manifest['cases']) == 4, 'Exactly four actual base cases required')
    config = json.loads(args.established_config.read_text(encoding='utf-8'))
    thresholds = {**DEFAULT_ACCEPTANCE, **config.get('acceptance', {})}
    count, seed = config['search']['evaluation_samples'], config['evaluation_seed']
    require(count == 4096 and seed == 7381, 'Established common evaluation contract changed')
    require(thresholds == DEFAULT_ACCEPTANCE, 'This audit must use existing common thresholds unchanged')
    args.out_dir.mkdir(parents=True)
    files = [Path(__file__), ROOT/'tools/geometry_quality/mesh_quality.py', ROOT/'tools/geometry_quality/cross_mesh.py',
             ROOT/'tools/unity_parity/geometry.py', ROOT/'tools/base_comparison/search.py', ROOT/'tools/base_comparison/surface.py']
    header = {'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
        'replay_summary_path': str(args.replay_summary.resolve()), 'replay_summary_sha256': sha(args.replay_summary),
        'established_config_path': str(args.established_config.resolve()), 'established_config_sha256': sha(args.established_config),
        'thresholds': asdict(Thresholds()), 'surface_thresholds': thresholds, 'evaluation_samples': count, 'evaluation_seed': seed,
        'implementation_sha256': {str(p):sha(p) for p in files}}
    save(args.out_dir/'audit_contract.json', header)
    summaries = []
    for case in manifest['cases']:
        head = case['head_id']; begin = time.perf_counter()
        descriptor = case['baseline']['paired_geometry']
        base, bm, bp = load_snapshot(descriptor, head)
        require(base['character']['shape_value_face'] == [.5]*59, 'Actual baseline native59 is not all.5')
        base_state = state(base, bm)
        for values in base_state['selected_actual_abmx_parameters'].values():
            require(values == {'scale':[1.,1.,1.], 'length':1., 'position':[0.,0.,0.], 'rotation':[0.,0.,0.]}, 'Actual four-bone baseline not identity')
        require(len(base_state['selected_actual_abmx_parameters']) == 4, 'Baseline group coverage incomplete')
        bv, bf, bcross_meshes, bcert = certify(base,bm,descriptor,bp,args.out_dir/f'head_{head}_baseline_lbs.json')
        transforms = {t['id']:t for t in base['transforms']}
        bquality, barrays = analyze_mesh(bm, transforms, Thresholds(), space='baked_raw')
        bcross = cross_intersections(bcross_meshes, Thresholds())
        save(args.out_dir/f'head_{head}_baseline_quality.json', {'source_path':str(bp),'source_sha256':descriptor['sha256'],
             'state':base_state,'world_certificate':bcert,'quality_report':bquality,'absolute':absolute_summary(bquality),
             'cross_mesh_report':bcross,'baseline_absolute_quality_validated':False})
        windows = {}
        for window in case['windows']:
            name = window['name']; require(window['window'] in ('early','late'), 'Unknown capture window')
            require(window['window'] not in windows, 'Duplicate actual window')
            snapshot, mesh, path = load_snapshot(window['geometry'],head); identity_bound(mesh,bm)
            rv = [r for r in replay['windows'] if r['name'] == name]
            require(len(rv) == 1, 'Numerical replay window missing/duplicate')
            replay_report = json.loads(Path(rv[0]['report_path']).read_text(encoding='utf-8'))
            require(replay_report['geometry_sha256'] == window['geometry']['sha256'], 'Replay geometry bytes mismatch')
            cv, cf, cross_meshes, cert = certify(snapshot,mesh,window['geometry'],path,args.out_dir/(name+'_lbs.json'))
            require(np.array_equal(cf,bf), 'Certified actual head topology mismatch')
            quality, qarrays = analyze_mesh(mesh,{t['id']:t for t in snapshot['transforms']},Thresholds(),space='baked_raw')
            quality['baseline'] = baseline_comparison(quality,bquality,qarrays,barrays,Thresholds())
            quality['baseline']['match_policy'] = 'exact_renderer_path_source_and_topology_prechecked'
            gate = quality_gate({'meshes':[quality]},{'meshes':[bquality]})
            surface = evaluate_surface(cv,cf,bv,bf,count=count,seed=seed)
            crossings = cross_intersections(cross_meshes,Thresholds())
            cross_comparison = compare_crossings(cross_meshes,bcross_meshes,crossings,bcross)
            current_state = state(snapshot,mesh)
            report = {'name':name,'source_path':str(path),'source_sha256':window['geometry']['sha256'],
                'baseline_path':str(bp),'baseline_sha256':descriptor['sha256'],'topology_sha256':topology_hash(cf),
                'numerical_replay_passed':rv[0]['passed'],'numerical_replay_report_path':rv[0]['report_path'],
                'numerical_replay_report_sha256':sha(rv[0]['report_path']), 'world_certificate':cert,
                'quality_report':quality,'quality_gate':gate,'absolute':absolute_summary(quality),
                'full_surface_vs_same_head_baseline':surface,'baseline_surface_threshold_diagnostic':surface_gate(surface,thresholds),
                'cross_mesh_report':crossings,'cross_mesh_baseline_comparison':cross_comparison,
                'baseline_to_patch_nuisance':nuisance(base_state,current_state),
                'quality_acceptance_is_not_numerical_replay_or_likeness':True,'likeness_validated':False,
                'full_face_anatomical_region_validated':False}
            save(args.out_dir/(name+'_quality.json'), report)
            windows[window['window']] = {'vertices':cv,'faces':cf,'raw':np.asarray(mesh['baked']['vertices'],float),
                                       'state':current_state,'report':report}
            print(json.dumps({'phase':'actual_quality','name':name,'quality_valid':gate['quality_valid'],'reasons':gate['reasons']}),flush=True)
        require(set(windows) == {'early','late'}, 'Incomplete early/late coverage')
        early,late = windows['early'],windows['late']
        temporal = evaluate_surface(late['vertices'],late['faces'],early['vertices'],early['faces'],count=count,seed=seed)
        raw_delta = late['raw']-early['raw']; canonical_delta=late['vertices']-early['vertices']
        temporal_report = {'head_id':head,'early_source_sha256':early['report']['source_sha256'],
            'late_source_sha256':late['report']['source_sha256'],'topology_sha256':topology_hash(bf),
            'full_surface_early_late':temporal,'existing_threshold_diagnostic':surface_gate(temporal,thresholds),
            'full_corresponding_vertices_raw':{'rms_l2':float(np.sqrt(np.mean(np.sum(raw_delta**2,axis=1)))),
                                            'max_l2':float(np.linalg.norm(raw_delta,axis=1).max())},
            'full_corresponding_vertices_recorded_rigid_frame':{'rms_l2':float(np.sqrt(np.mean(np.sum(canonical_delta**2,axis=1)))),
                                                              'max_l2':float(np.linalg.norm(canonical_delta,axis=1).max())},
            'nuisance':nuisance(early['state'],late['state']), 'fitted_scale':False,'fitted_affine':False,
            'shape_or_pose_writers_causally_identified':False,'elapsed_seconds':time.perf_counter()-begin,
            'likeness_or_complete_infrastructure_validated':False}
        save(args.out_dir/f'head_{head}_temporal.json',temporal_report)
        summaries.append({'head_id':head,'baseline_absolute':absolute_summary(bquality),
            'baseline_cross_mesh_crossing_count':bcross['crossing_count'],
            'windows':[{'name':w['report']['name'],'quality_gate':w['report']['quality_gate'],
                        'absolute':w['report']['absolute'],'full_surface_vs_baseline':w['report']['full_surface_vs_same_head_baseline'],
                        'new_cross_mesh_pairs':len(w['report']['cross_mesh_baseline_comparison'].get('new_crossing_pairs',[]))}
                        for w in (early,late)],'temporal_report':str((args.out_dir/f'head_{head}_temporal.json').resolve()),
            'temporal_surface':temporal,'temporal_threshold_diagnostic':surface_gate(temporal,thresholds)})
        print(json.dumps({'phase':'head_complete','head_id':head,'elapsed_seconds':time.perf_counter()-begin,
                          'temporal_rms':temporal['symmetric']['rms']}),flush=True)
    save(args.out_dir/'summary.json',header | {'heads':summaries,'actual_geometry_count':12,'patch_quality_pass_count':sum(
         w['quality_gate']['quality_valid'] for h in summaries for w in h['windows']), 'patch_window_count':8,
         'baseline_absolute_quality_validated':False,'likeness_validated':False,'full_infrastructure_goal_complete':False,
         'scope':'Complete actual o_head geometric surface and certified active head cross-mesh diagnostics; no calibrated facial region/anatomy or shader/aesthetic certification'})


if __name__ == '__main__':
    main()
