"""Fresh four-head same-native audit with unchanged thresholds and actual histories."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.abmx_multibone_quality import run as helpers
from tools.abmx_multibone_quality.run import (sha, save, require, load_snapshot, identity_bound, certify,
    Thresholds, analyze_mesh, baseline_comparison, cross_intersections, compare_crossings,
    quality_gate, DEFAULT_ACCEPTANCE, evaluate_surface, topology_hash, content_hash, surface_gate, absolute_summary)
import numpy as np
THRESHOLD_BINDING=ROOT/'outputs/stateful_multibone_fresh_quality_20261005/threshold_binding_v1.json'
SELECTED={'cf_J_Chin_rs','cf_J_ChinTip_s','cf_J_CheekUp_L','cf_J_CheekUp_R'}

def require_native(snapshot,expected,label):
    require(len(expected)==59 and all(type(x) in (float,int) and np.isfinite(x) for x in expected), 'Expected finite full native59 required')
    require(snapshot['character']['native_count']==59 and np.array_equal(np.asarray(snapshot['character']['shape_value_face'],dtype=np.float32),np.asarray(expected,dtype=np.float32)), 'Actual native59 differs: '+label)

def state(snapshot,mesh):
    result=helpers.state(snapshot,mesh)
    selected=[b for b in snapshot['abmx_runtime']['bones'] if b['name'] in SELECTED]
    require(len(selected)==4 and {b['name'] for b in selected}==SELECTED, 'Selected actual bone histories incomplete or duplicate')
    result['all_selected_actual_runtime_bone_records']=selected
    result['all_actual_abmx_runtime']=snapshot['abmx_runtime']
    root=snapshot['character']['head_root_transform_id']; transforms={t['id']:t for t in snapshot['transforms']}; scoped=set()
    for start in mesh['bone_transform_ids']:
        current=start; visited=set()
        while current is not None:
            require(current in transforms and current not in visited,'Missing/cyclic actual skin ancestor')
            visited.add(current); scoped.add(current)
            if current==root:break
            current=transforms[current]['parent_id']
    result['actual_palette_ancestor_records']=[transforms[k] for k in sorted(scoped)]
    result['selected_actual_local_transforms']=[t for t in snapshot['transforms'] if t['name'] in SELECTED]
    require(len(result['selected_actual_local_transforms'])==4, 'Selected actual local transform identities ambiguous')
    return result

def history_diff(a,b):
    result={}
    for key,identity in [('all_selected_actual_runtime_bone_records','name'),('actual_palette_ancestor_records','path')]:
        av={row[identity]:row for row in a[key]}; bv={row[identity]:row for row in b[key]}
        require(set(av)==set(bv),'Actual history identity scope differs')
        result[key]={'changed_records':[{identity:name,'early':av[name],'late':bv[name]} for name in sorted(av) if av[name]!=bv[name]],
                     'record_count':len(av),'exactly_identical':av==bv}
    result.update(external_writers_identified=False, parameter_identity_implies_pose_identity=False)
    return result

def nuisance(a,b):
    return helpers.nuisance(a,b) | {'all_selected_history_comparison':history_diff(a,b)}

def raw_spans(mesh):
    from tools.observable_features.measure import region_faces, arrays
    vertices,_=arrays(mesh)
    rows=[]
    for submesh in mesh['baked']['submeshes']:
        faces=region_faces(mesh,submesh['submesh_index']); used=np.unique(faces); points=vertices[used]
        rows.append({'submesh_index':submesh['submesh_index'],'triangle_count':len(faces),'referenced_vertex_count':len(used),
            'span_xyz_renderer_raw_game_units':np.ptp(points,axis=0).tolist()})
    return {'submeshes':rows,'coordinate_frame':'actual BakeMesh renderer raw','frame_policy_independently_certified':False,
        'anatomical_face_width_or_length_certified':False,'single_factor_physical_response_certified':False,
        'scale_parameters_are_physical_distances':False,'crossbase_equivalence_certified':False,
        'interpretation':'Whole asset submesh raw extents; no anatomy, real-world unit, or controlled-channel response claim.'}

def preserve_history(case,head,base,bm,out):
    source=case['source_history']; descriptor=source.get('geometry') or source['capture']['paired_geometry']
    neutral,nm,path=load_snapshot(descriptor,head)
    identity_bound(nm,bm); require_native(neutral,[.5]*59,'neutral history')
    ns=state(neutral,nm); bs=state(base,bm)
    for params in ns['selected_actual_abmx_parameters'].values():
        require(params=={'scale':[1.,1.,1.],'length':1.,'position':[0.,0.,0.],'rotation':[0.,0.,0.]}, 'Neutral history actual modifier not identity')
    _,_,_,certificate=certify(neutral,nm,descriptor,path,out/f'head_{head}_neutral_history_lbs.json')
    records=[]
    for label,value in [('neutral',source),('candidate',case)]:
        if 'trace' in value:
            trace=Path(value['trace']); require(trace.is_absolute() and sha(trace)==value['trace_sha256'],'Actual trace bytes/path differ')
            records.append({'role':label,'path':str(trace),'sha256':sha(trace)})
    save(out/f'head_{head}_source_history.json',{'head_id':head,'source_history_manifest_record':source,
        'neutral_geometry_path':str(path),'neutral_geometry_sha256':descriptor['sha256'],
        'neutral_actual_state':ns,'varied_baseline_actual_state':bs,'actual_history_difference':history_diff(ns,bs),
        'neutral_world_certificate':certificate,'trace_sources':records,
        'neutral_used_as_candidate_quality_baseline':False,'old_protocol_prediction_certified':False})

def validate_threshold_binding(config_path):
    binding=json.loads(THRESHOLD_BINDING.read_text(encoding='utf-8'))
    require(sha(config_path)==binding['established_config_sha256'],'Established configuration byte hash changed')
    require(asdict(Thresholds())==binding['thresholds'] and DEFAULT_ACCEPTANCE==binding['surface_thresholds'], 'Established threshold definitions changed')
    for path,digest in binding['implementation_sha256'].items():
        require(sha(path)==digest,'Established measurement implementation changed: '+path)
    return {'threshold_binding_path':str(THRESHOLD_BINDING),'threshold_binding_sha256':sha(THRESHOLD_BINDING)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--established-config', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out_dir.exists(), 'New output directory required')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    protocol_path=Path(manifest['protocol_path'])
    require(protocol_path.is_absolute() and sha(protocol_path)==manifest['protocol_sha256'],'Fresh capture protocol byte binding differs')
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
    require(protocol['heads']==[0,1,2,3] and set(protocol['names'])==SELECTED,'Fresh protocol source scope differs')
    require(all(manifest.get(k) is True for k in ('state_restored', 'expression_restored', 'bone_restored')), 'Restoration flags missing')
    require(manifest['before'] == manifest['after'] and manifest['expression_before'] == manifest['expression_after']
            and manifest['modifiers_before'] == manifest['modifiers_after'], 'Recorded restoration evidence differs')
    require({c['head_id'] for c in manifest['cases']} == {0,1,2,3} and len(manifest['cases']) == 4, 'Exactly four actual base cases required')
    threshold_binding=validate_threshold_binding(args.established_config)
    config = json.loads(args.established_config.read_text(encoding='utf-8'))
    thresholds = {**DEFAULT_ACCEPTANCE, **config.get('acceptance', {})}
    count, seed = config['search']['evaluation_samples'], config['evaluation_seed']
    require(count == 4096 and seed == 7381, 'Established common evaluation contract changed')
    require(thresholds == DEFAULT_ACCEPTANCE, 'This audit must use existing common thresholds unchanged')
    args.out_dir.mkdir(parents=True)
    files = [Path(__file__), Path(helpers.__file__), ROOT/'tools/geometry_quality/mesh_quality.py', ROOT/'tools/geometry_quality/cross_mesh.py',
             ROOT/'tools/unity_parity/geometry.py', ROOT/'tools/base_comparison/search.py', ROOT/'tools/base_comparison/surface.py']
    header = {'manifest_path': str(args.manifest.resolve()), 'manifest_sha256': sha(args.manifest),
        'established_config_path': str(args.established_config.resolve()), 'established_config_sha256': sha(args.established_config),
        'thresholds': asdict(Thresholds()), 'surface_thresholds': thresholds, 'evaluation_samples': count, 'evaluation_seed': seed,
        'implementation_sha256': {str(p):sha(p) for p in files}}
    header.update(threshold_binding)
    header.update(protocol_path=str(protocol_path),protocol_sha256=sha(protocol_path),native_validation='Exact float32-equivalent values, never geometric tolerance')
    save(args.out_dir/'audit_contract.json', header)
    summaries = []
    for case in manifest['cases']:
        head = case['head_id']; begin = time.perf_counter()
        require(set(case['names'])==SELECTED and len(case['names'])==4,'Actual four-bone declared group differs')
        for key in ('baseline_expected_native59','candidate_expected_native59','source_history_expected_native59'):
            require(case[key]==protocol[key],'Case expected native59 differs from SHA-bound predeclared protocol: '+key)
        expected_plan=np.asarray([[.48,.52,.47,.53][i%4] for i in range(59)],dtype=np.float32)
        require(np.array_equal(np.asarray(case['baseline_expected_native59'],dtype=np.float32),expected_plan),'Fresh variednative preregistered plan differs')
        descriptor = case['baseline']['paired_geometry']
        base, bm, bp = load_snapshot(descriptor, head)
        require_native(base, case['baseline_expected_native59'], 'baseline')
        require(case['baseline_expected_native59'] == case['candidate_expected_native59'], 'Candidate same-native baseline differs')
        preserve_history(case, head, base, bm, args.out_dir)
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
             'raw_full_submesh_spans':raw_spans(bm),
             'cross_mesh_report':bcross,'baseline_absolute_quality_validated':False})
        windows = {}
        for window in case['windows']:
            name = window['name']; require(window['head_id']==head, 'Window declared head differs')
            require(window['window'] in ('early','late'), 'Unknown capture window')
            require(window['window'] not in windows, 'Duplicate actual window')
            snapshot, mesh, path = load_snapshot(window['geometry'],head); identity_bound(mesh,bm)
            require_native(snapshot, case['candidate_expected_native59'], name)
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
            for patch in protocol['patches']:
                actual=current_state['selected_actual_abmx_parameters'][patch['name']]
                for key in ('scale','length','position','rotation'):
                    require(np.array_equal(np.asarray(actual[key],dtype=np.float32),np.asarray(patch[key],dtype=np.float32)),'Actual candidate ABMX parameter differs from protocol: '+patch['name']+' '+key)
            report = {'name':name,'source_path':str(path),'source_sha256':window['geometry']['sha256'],
                'baseline_path':str(bp),'baseline_sha256':descriptor['sha256'],'topology_sha256':topology_hash(cf),
                'candidate_old_protocol_predictability_certified':False, 'world_certificate':cert,
                'quality_report':quality,'quality_gate':gate,'absolute':absolute_summary(quality),
                'full_surface_vs_same_head_baseline':surface,'baseline_surface_threshold_diagnostic':surface_gate(surface,thresholds),
                'cross_mesh_report':crossings,'cross_mesh_baseline_comparison':cross_comparison,
                'baseline_to_patch_nuisance':nuisance(base_state,current_state),
                'raw_full_submesh_spans':raw_spans(mesh),
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
            'nuisance':nuisance(early['state'],late['state']),
            'selected_cache_and_actual_local_drift':history_diff(early['state'],late['state']),
            'fitted_scale':False,'fitted_affine':False,
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
    save(args.out_dir/'summary.json',header | {'heads':summaries,'actual_geometry_count':12,'neutral_history_geometry_count':4,'candidate_old_protocol_predictability_certified':False,'patch_quality_pass_count':sum(
         w['quality_gate']['quality_valid'] for h in summaries for w in h['windows']), 'patch_window_count':8,
         'baseline_absolute_quality_validated':False,'likeness_validated':False,'full_infrastructure_goal_complete':False,
         'scope':'Complete actual o_head geometric surface and certified active head cross-mesh diagnostics; no calibrated facial region/anatomy or shader/aesthetic certification'})


if __name__ == '__main__':
    main()
