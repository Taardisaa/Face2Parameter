"""Verify actual audit artifact binding and unchanged quality contracts offline."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .run import sha, save, require, load_snapshot, identity_bound, Thresholds, DEFAULT_ACCEPTANCE


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cache-drift',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),'New verification report required')
    summary=json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads(Path(summary['manifest_path']).read_text(encoding='utf-8'))
    require(sha(summary['manifest_path'])==summary['manifest_sha256'],'Live manifest changed')
    require(sha(summary['replay_summary_path'])==summary['replay_summary_sha256'],'Replay evidence changed')
    require(sha(summary['established_config_path'])==summary['established_config_sha256'],'Established contract changed')
    require(summary['thresholds']==asdict(Thresholds()) and summary['surface_thresholds']==DEFAULT_ACCEPTANCE,'Thresholds changed')
    require(summary['evaluation_samples']==4096 and summary['evaluation_seed']==7381,'Sampling contract changed')
    require(all(sha(path)==value for path,value in summary['implementation_sha256'].items()),'Producer source changed')
    require(summary['actual_geometry_count']==12 and summary['patch_window_count']==8,'Incomplete actual coverage')
    verified=[]
    for case in manifest['cases']:
        base,bm,_=load_snapshot(case['baseline']['paired_geometry'],case['head_id'])
        require(base['character']['shape_value_face']==[.5]*59,'Not actual nativeall.5 baseline')
        all_identity=all(all(b[f]==identity for f,identity in [('scale',[1.,1.,1.]),('length',1.),('position',[0.,0.,0.]),('rotation',[0.,0.,0.])])
                         for b in base['abmx_runtime']['bones'])
        require(all_identity,'Recorded ABMX baseline includes nonidentity registered modifiers')
        snapshots={}
        for window in case['windows']:
            name=window['name'];snapshot,mesh,_=load_snapshot(window['geometry'],case['head_id']);identity_bound(mesh,bm)
            report=json.loads((args.run_dir/(name+'_quality.json')).read_text(encoding='utf-8'))
            require(report['source_sha256']==window['geometry']['sha256'],'Quality report bound wrong actual geometry')
            certificate=report['world_certificate'];require(sha(certificate['certificate_path'])==certificate['certificate_sha256'],'LBS certificate changed')
            require(certificate['fitted_scale'] is False and certificate['fitted_affine'] is False and certificate['fitted_rotation'] is False,'Hidden shape alignment applied')
            require(report['quality_gate']['thresholds']==asdict(Thresholds()),'Quality thresholds differ')
            require(report['quality_report']['baseline']['source_hash_identical'] is True,'Source correspondence absent')
            snapshots[window['window']]=mesh
            verified.append({'name':name,'quality_valid':report['quality_gate']['quality_valid'],'source_sha256':window['geometry']['sha256'],
                             'report_sha256':sha(args.run_dir/(name+'_quality.json'))})
        temporal=json.loads((args.run_dir/f"head_{case['head_id']}_temporal.json").read_text(encoding='utf-8'))
        delta=np.asarray(snapshots['late']['baked']['vertices'])-np.asarray(snapshots['early']['baked']['vertices'])
        actual_rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=1))))
        actual_max=float(np.linalg.norm(delta,axis=1).max())
        require(actual_rms==temporal['full_corresponding_vertices_raw']['rms_l2'] and actual_max==temporal['full_corresponding_vertices_raw']['max_l2'],'Raw actual vertex differences changed')
        require(temporal['existing_threshold_diagnostic']['thresholds']==DEFAULT_ACCEPTANCE,'Temporal threshold changed')
    cache=json.loads(args.cache_drift.read_text(encoding='utf-8'))
    require(cache['manifest_sha256']==summary['manifest_sha256'],'Cache report references other capture')
    require(cache['implementation_sha256']==sha(Path(__file__).with_name('cache_drift.py')),'Cache producer changed')
    artifacts={str(path.resolve()):sha(path) for path in sorted(args.run_dir.glob('*.json'))}
    artifacts[str(args.cache_drift.resolve())]=sha(args.cache_drift)
    save(args.out,{'schema_version':1,'verified_actual_window_count':len(verified),'actual_geometry_count':12,
        'verified_windows':verified,'source_geometry_binding_rechecked':True,'recorded_all_baseline_abmx_parameters_identity':True,
        'established_thresholds_unchanged':True,'actual_raw_temporal_vertex_differences_recomputed':True,
        'main_job_independently_reconstructed_each_snapshot_lbs':True,'verification_did_not_rerun_triangle_intersections_or_quadrature':True,
        'artifact_sha256':artifacts,'verification_source_sha256':sha(__file__),
        'frozen_training_annotation_sha256':sha(ROOT/'outputs/anatomical_candidates_20261005/pitched_training_v1/frozen_train_annotations.json'),
        'anatomical_correspondence_validated':False,'temporal_stability_accepted':False,'full_infrastructure_goal_complete':False})
    print(json.dumps({'windows':len(verified),'thresholds_unchanged':True,'actual_raw_vertex_difference_recomputed':True,'out':str(args.out.resolve())}))


ROOT=Path(__file__).resolve().parents[2]
if __name__=='__main__':main()
