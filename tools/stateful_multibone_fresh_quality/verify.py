"""Re-read fresh geometry and bind produced diagnostics; no copied quality verdicts."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from .run import sha,save,require,load_snapshot,identity_bound,state,history_diff,require_native,Thresholds,DEFAULT_ACCEPTANCE,validate_threshold_binding

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'Immutable new verification output required')
    summary=json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'))
    require(sha(summary['manifest_path'])==summary['manifest_sha256'],'Manifest changed')
    require(sha(summary['protocol_path'])==summary['protocol_sha256'],'Protocol changed')
    require(all(sha(path)==digest for path,digest in summary['implementation_sha256'].items()),'Measurement producer changed')
    validate_threshold_binding(summary['established_config_path'])
    require(summary['thresholds']==asdict(Thresholds()) and summary['surface_thresholds']==DEFAULT_ACCEPTANCE,'Measurement thresholds changed')
    require(summary['evaluation_samples']==4096 and summary['evaluation_seed']==7381,'Quadrature contract changed')
    require(summary['actual_geometry_count']==12 and summary['neutral_history_geometry_count']==4 and summary['patch_window_count']==8,'Actual geometry coverage incomplete')
    manifest=json.loads(Path(summary['manifest_path']).read_text(encoding='utf-8'));verified=[];baseline_records=[]
    for case in manifest['cases']:
        head=case['head_id'];base,bm,_=load_snapshot(case['baseline']['paired_geometry'],head)
        require_native(base,case['baseline_expected_native59'],'fresh baseline')
        for bone in base['abmx_runtime']['bones']:
            require(all(bone[key]==identity for key,identity in [('scale',[1.,1.,1.]),('length',1.),('position',[0.,0.,0.]),('rotation',[0.,0.,0.])]),'Actual baseline registered modifier not identity')
        baseline=json.loads((args.run_dir/f'head_{head}_baseline_quality.json').read_text(encoding='utf-8'))
        require(baseline['state']==state(base,bm),'Recorded baseline actual history differs')
        require(baseline['baseline_absolute_quality_validated'] is False,'Absolute baseline validity illegally assumed')
        history=json.loads((args.run_dir/f'head_{head}_source_history.json').read_text(encoding='utf-8'))
        neutral,nm,_=load_snapshot(case['source_history']['geometry'],head);identity_bound(nm,bm)
        require_native(neutral,case['source_history_expected_native59'],'neutral history')
        require(history['neutral_actual_state']==state(neutral,nm) and history['varied_baseline_actual_state']==state(base,bm),'Neutral/varied history evidence changed')
        require(history['actual_history_difference']==history_diff(state(neutral,nm),state(base,bm)),'Neutral/varied history comparison changed')
        require(history['neutral_used_as_candidate_quality_baseline'] is False,'Neutral confused with candidate baseline')
        for source in history['trace_sources']:require(sha(source['path'])==source['sha256'],'Raw observed trace changed')
        states={};meshes={}
        for window in case['windows']:
            name=window['name'];snapshot,mesh,_=load_snapshot(window['geometry'],head);identity_bound(mesh,bm)
            require_native(snapshot,case['candidate_expected_native59'],name)
            report=json.loads((args.run_dir/(name+'_quality.json')).read_text(encoding='utf-8'))
            require(report['source_sha256']==window['geometry']['sha256'] and report['baseline_sha256']==case['baseline']['paired_geometry']['sha256'],'Same-native source binding changed')
            require(report['baseline_to_patch_nuisance']['late']==state(snapshot,mesh),'Actual candidate history changed')
            require(report['quality_gate']['thresholds']==asdict(Thresholds()),'Relative quality gate changed')
            require(report['quality_report']['baseline']['source_hash_identical'] is True,'Actual source correspondence missing')
            certificate=report['world_certificate'];require(sha(certificate['certificate_path'])==certificate['certificate_sha256'],'Actual LBS evidence changed')
            require(all(certificate[key] is False for key in ('fitted_rotation','fitted_translation','fitted_scale','fitted_affine')),'Fitted alignment hides shape drift')
            states[window['window']]=state(snapshot,mesh);meshes[window['window']]=mesh
            verified.append({'name':name,'head_id':head,'source_sha256':window['geometry']['sha256'],'quality_valid':report['quality_gate']['quality_valid']})
        temporal=json.loads((args.run_dir/f'head_{head}_temporal.json').read_text(encoding='utf-8'))
        delta=np.asarray(meshes['late']['baked']['vertices'])-np.asarray(meshes['early']['baked']['vertices'])
        require(float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))==temporal['full_corresponding_vertices_raw']['rms_l2'],'Actual full raw temporal RMS changed')
        require(float(np.linalg.norm(delta,axis=1).max())==temporal['full_corresponding_vertices_raw']['max_l2'],'Actual full raw temporal maximum changed')
        require(temporal['selected_cache_and_actual_local_drift']==history_diff(states['early'],states['late']),'Selected cache/skin-ancestor difference dropped or changed')
        require(temporal['existing_threshold_diagnostic']['thresholds']==DEFAULT_ACCEPTANCE,'Temporal gate changed')
        baseline_records.append({'head_id':head,'all_actual_registered_baseline_modifiers_identity':True,'actual_native59_float32_exact':True,'absolute_baseline_quality_validated':False})
    artifact_hashes={str(path.resolve()):sha(path) for path in sorted(args.run_dir.glob('*.json'))}
    save(args.out,{'schema_version':1,'verified_window_count':len(verified),'actual_primary_geometry_count':12,'neutral_history_geometry_count':4,
        'windows':verified,'baselines':baseline_records,'artifact_sha256':artifact_hashes,'verification_source_sha256':sha(__file__),
        'manifest_sha256':summary['manifest_sha256'],'thresholds_unchanged':True,'actual_geometry_and_state_reread':True,
        'full_raw_temporal_vertex_difference_recomputed':True,'private_cache_and_ancestor_history_difference_recomputed':True,
        'verification_reran_triangle_intersections_or_full_surface_quadrature':False,
        'candidate_old_protocol_predictability_certified':False,'anatomical_or_likeness_acceptance':False,'full_infrastructure_goal_complete':False})
    print(json.dumps({'windows':len(verified),'actual_primary_geometry':12,'neutral_history_geometry':4,'out':str(args.out.resolve())}))

if __name__=='__main__':main()
