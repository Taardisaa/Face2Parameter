"""Re-read full private fields and local transforms without causal writer claims."""
import argparse,json
from pathlib import Path
from .run import sha,save,require,load_snapshot,state,history_diff

def classify(a,b):
    all_comparison=history_diff(a,b); private=[]; changed_local=[]
    av={row['name']:row for row in a['all_selected_actual_runtime_bone_records']};bv={row['name']:row for row in b['all_selected_actual_runtime_bone_records']}
    for name in sorted(av):
        x=av[name]['runtime_baseline'];y=bv[name]['runtime_baseline'];require(set(x['fields'])==set(y['fields']),'Private field scope changed')
        private.append({'name':name,'changed_private_fields':{key:{'early':x['fields'][key],'late':y['fields'][key]} for key in x['fields'] if x['fields'][key]!=y['fields'][key]},
            'changed_runtime_metadata_keys':[key for key in x if key!='fields' and x[key]!=y[key]],
            'public_parameter_values_identical':all(av[name][key]==bv[name][key] for key in ('scale','length','position','rotation'))})
    at={row['path']:row for row in a['actual_palette_ancestor_records']};bt={row['path']:row for row in b['actual_palette_ancestor_records']}
    require(set(at)==set(bt),'Actual transform path scope changed')
    for path in sorted(at):
        changes={key:{'early':at[path][key],'late':bt[path][key]} for key in ('local_position','local_rotation_xyzw','local_scale') if at[path][key]!=bt[path][key]}
        if changes:changed_local.append({'name':at[path]['name'],'path':path,'fields':changes})
    return {'selected_private_field_differences':private,'actual_palette_local_differences':changed_local,
        'all_palette_ancestor_count':len(at),'full_record_changed_count':len(all_comparison['actual_palette_ancestor_records']['changed_records']),
        'full_record_reference_contains_world_and_frame_metadata':True,'external_writers_causally_identified':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    require(not args.out.exists(),'Immutable new investigation output required')
    summary=json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'));manifest=json.loads(Path(summary['manifest_path']).read_text(encoding='utf-8'))
    require(sha(summary['manifest_path'])==summary['manifest_sha256'],'Actual manifest changed');heads=[]
    for case in manifest['cases']:
        head=case['head_id'];ns,nm,_=load_snapshot(case['source_history']['geometry'],head);bs,bm,_=load_snapshot(case['baseline']['paired_geometry'],head)
        actual={}
        for window in case['windows']:
            s,m,_=load_snapshot(window['geometry'],head);actual[window['window']]=state(s,m)
        temporal=json.loads((args.run_dir/f'head_{head}_temporal.json').read_text(encoding='utf-8'))
        quality=json.loads((args.run_dir/f'head_{head}_baseline_quality.json').read_text(encoding='utf-8'))
        heads.append({'head_id':head,'neutral_to_varied_history':classify(state(ns,nm),state(bs,bm)),
            'early_to_late_history':classify(actual['early'],actual['late']), 'temporal_surface':temporal['full_surface_early_late'],
            'temporal_gate':temporal['existing_threshold_diagnostic'],'nuisance_equal_flags':{key:value for key,value in temporal['nuisance'].items() if key.endswith('_identical')},
            'baseline_absolute_anomalies':quality['absolute'],'neutral_used_as_candidate_baseline':False})
    save(args.out,{'schema_version':1,'manifest_sha256':summary['manifest_sha256'],'fresh_summary_path':str((args.run_dir/'summary.json').resolve()),
        'fresh_summary_sha256':sha(args.run_dir/'summary.json'),'source_sha256':sha(__file__),'heads':heads,
        'actual_states_independently_reread':True,'all_selected_private_fields_retained':True,'writer_causal_identification':False,
        'anatomy_or_likeness_certified':False,'single_factor_physical_response_certified':False})
    print(json.dumps([{'head_id':r['head_id'],'changed_temporal_local_names':[x['name'] for x in r['early_to_late_history']['actual_palette_local_differences']],
        'changed_private_field_count':sum(len(x['changed_private_fields']) for x in r['early_to_late_history']['selected_private_field_differences'])} for r in heads]))

if __name__=='__main__':main()
