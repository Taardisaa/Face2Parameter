"""Few existing native cases, no new game sampling or fitting gains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.hs2_offline_state import CapturedHeadForward
from tools.unity_parity.geometry import vertex_errors, analyze_snapshot

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir',type=Path,required=True)
    ap.add_argument('--acceptance-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():
        raise FileExistsError('New receipt path required')
    geometry=read(a.baseline_dir/'baseline_front_geometry.json')
    context=read(a.baseline_dir/'context.json')
    forward=CapturedHeadForward(geometry,context)
    source=read(a.baseline_dir/'source_zero.json')
    refs={r['renderer_path']:r for r in source['surfaces']}
    rows=[]
    for name,native in [('captured_modifiers',None),('native59_refresh',forward.identity['native59'])]:
        result=forward.evaluate(native=native)
        rows.append({'name':name,'surfaces':[{'renderer_path':r['renderer_path'],
            'mesh_name':r['mesh_name'],'error':vertex_errors(np.asarray(refs[r['renderer_path']]['reference']),r['head_vertices'])}
                                           for r in result['surfaces']]})
    snapshot_analysis=analyze_snapshot(forward.geometry,influences=4)
    cases=[]
    for stem in ('five_jaw_xyz_scale_length_rotation','native_angle_plus_seven_bone_update'):
        reference=read(a.acceptance_dir/f'{stem}_acceptance.json')
        request=read(a.acceptance_dir/f'{stem}_request.json')
        original=forward.evaluate()
        result=forward.evaluate(request.get('bones',[]),native=request.get('native_values'))
        target={r['renderer_path']:r for r in reference['surfaces']}
        cases.append({'name':stem,'same_native':reference['identity']['native59']==forward.identity['native59'],
            'same_ordered_private_modifiers':reference['identity']['abmx_ordered_modifiers']==forward.identity['abmx_ordered_modifiers'],
            'surfaces':[{'renderer_path':r['renderer_path'],'mesh_name':r['mesh_name'],
                'reference_pose_difference':vertex_errors(np.asarray(target[r['renderer_path']]['reference']),o['head_vertices']),
                'candidate_difference':vertex_errors(np.asarray(target[r['renderer_path']]['candidate']),r['head_vertices'])}
                for o,r in zip(original['surfaces'],result['surfaces'])],
            'scope':'Cross-capture comparison. Gaze is retained from baseline capture; source acceptance from another pose/time cannot certify eye surfaces.'})
    receipt={'schema_version':1,'normalized_gate':1e-5,'new_native_samples':0,'source_provenance':forward.source_provenance,
             'asset_provenance':forward.asset_provenance,'baseline_source_cases':rows,
             'independent_snapshot_lbs':snapshot_analysis,'existing_parameter_cases':cases,
             'appearance_certified':False,'gaps':['AIT material/shader/SSS/GrabPass image pipeline','active breathing time state','PushUp affecting requested geometry beyond the head dependency set']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'receipt':str(a.out),'baseline_source_passed':all(s['error']['max_normalized']<=1e-5 for r in rows for s in r['surfaces']),
                      'head_parameter_cases_passed':all(s['candidate_difference']['max_normalized']<=1e-5 for c in cases for s in c['surfaces'] if s['mesh_name']=='o_head')}))

if __name__=='__main__':
    main()
