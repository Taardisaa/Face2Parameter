"""Offline installed ABMX Reset/CollectBaseline diagnosis. No game or target fitting."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]; HS2=ROOT.parent/'HS2Mod'
sys.path.insert(0,str(ROOT))
from tools.abmx_replay.model import F, unity_zero, unity_sqrmag, cache_state, serialized
from tools.native_radial_target.compiler import _native
from src.hs2_mesh_deform import HeadRig


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def binding(path):
    path=Path(path).resolve(); return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def reset(state, *, bone_exists=True):
    state=cache_state(state)
    if not bone_exists or not state['_hasBaseline']:
        return {'branch':'no_op','local':None,'cache':serialized(state)}
    pos=state['_posBaseline'].copy()
    branch='restore_latest_baseline'
    if not unity_zero(state['_positionBaseline']):
        if state['_lenModNeedsPositionRestore'] or unity_zero(pos):
            pos=state['_positionBaseline'].copy(); state['_lenModNeedsPositionRestore']=False
            branch='restore_first_position_history'
        else:
            mag=F(np.sqrt(unity_sqrmag(pos)))
            pos=(pos/mag)*F(state['_lenBaseline'])
            branch='latest_position_normalized_to_first_radius'
    return {'branch':branch,'local':{'local_position':pos.tolist(),'local_scale':state['_sclBaseline'].tolist(),
                                    'local_rotation_xyzw':state['_rotBaseline'].tolist()},'cache':serialized(state)}


def collect(state,local):
    state=cache_state(state)
    for dst,src in [('_posBaseline','local_position'),('_sclBaseline','local_scale'),('_rotBaseline','local_rotation_xyzw')]:
        state[dst]=np.asarray(local[src],dtype=F).copy()
    changed=unity_zero(state['_positionBaseline'])
    if changed:
        pos=np.asarray(local['local_position'],dtype=F)
        state['_lenBaseline']=F(np.sqrt(unity_sqrmag(pos)))
        state['_positionBaseline']=pos.copy();state['_lenModNeedsPositionRestore']=False
    state['_hasBaseline']=True
    return {'cache':serialized(state),'length_history_recollected':changed}


def write_mask(rig,name):
    result={'local_position':[], 'local_rotation_xyzw':[], 'local_scale':[]}
    for eq in rig.eqns:
        if rig.enums['dst'][eq['dst']]!=name:continue
        target={'pos':'local_position','rot':'local_rotation_xyzw','scl':'local_scale'}[eq['target']]
        result[target].extend(list(range(4 if target=='local_rotation_xyzw' else 3)) if eq['axis'] is None else [eq['axis']])
    return {key:sorted(set(value)) for key,value in result.items()}


def native_partial(local,native,mask):
    result=deepcopy(local)
    for key,axes in mask.items():
        for axis in axes:result[key][axis]=float(F(native[key][axis]))
    return result


def execute(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    directories=[HS2/'artifacts/infrastructure_live_20261005'/n for n in
        ['native_radial_source_v1','native_radial_head2_v1','native_radial_head0_v1']]
    inputs=[]; datasets=[]
    for directory in directories:
        manifest=read(directory/'live_cases.json'); protocol=read(directory/'predeclared_protocol.json')
        trace=read(directory/'source_history_trace.json'); snapshot=read(directory/'source_history.json')
        for name in ['live_cases.json','predeclared_protocol.json','source_history_trace.json','source_history.json','settled_identity.json']:
            inputs.append(binding(directory/name))
        assert binding(directory/'predeclared_protocol.json')['sha256']==manifest['protocol_sha256']
        assert binding(directory/'source_history_trace.json')['sha256']==manifest['source_history']['trace_sha256']
        assert binding(directory/'source_history.json')['sha256']==manifest['source_history']['geometry']['sha256']
        assert trace['trace_complete'] and not trace['dropped_events'] and not trace['pending_calls'] and not trace['observer_errors']
        byname={name:[e for e in trace['events'] if e['bone_name']==name] for name in protocol['names']}
        assert len(byname)==30 and all(byname.values())
        rig=HeadRig(protocol['head_id'],sampling_profile=protocol['sampling_profile'])
        native=_native(rig,protocol['native59'])
        original_head=manifest['before']['face_base']['head_id']
        original_rig=HeadRig(original_head,sampling_profile=protocol['sampling_profile'])
        original_values=[row['value'] for row in manifest['before']['face_shapes']['items']]
        original_native=_native(original_rig,original_values)
        records=[]
        for name,events in byname.items():
            assert all(e['resolved_modifier']=={'scale':[1,1,1],'length':1,'position':[0,0,0],'rotation':[0,0,0]}
                       and not e['additional_modifiers'] for e in events)
            e=events[-1]; state=e['before']['cache']['fields']
            r=reset(state)
            after_original_native=native_partial(r['local'],original_native[name],write_mask(original_rig,name))
            recollected=collect(state,after_original_native)
            second=reset(recollected['cache'])
            expected=native[name]
            position=np.asarray(e['before']['local_position']); ideal=np.asarray(expected['local_position'])
            records.append({'name':name,'calls':len(events),'first_before':events[0]['before'],'last_before':e['before'],
                'last_after':e['after'],'native_source_only_expected':expected,'native_write_mask':write_mask(rig,name),
                'native_position_max_abs_error':float(np.max(abs(position-ideal))),
                'native_scale_max_abs_error':float(np.max(abs(np.asarray(e['before']['local_scale'])-expected['local_scale']))),
                'source_only_reset_prediction':r,
                'source_only_cleanup_lifecycle':{'original_native_source_expected':original_native[name],
                    'original_native_write_mask':write_mask(original_rig,name),
                    'reset_before_original_native':r,'after_original_native':after_original_native,
                    'collect_after_original_native':recollected,'remove_modifier_second_reset':second},
                'reset_position_delta_from_latest_baseline':(np.asarray(r['local']['local_position'])-state['_posBaseline']).tolist(),
                'collect_latest_after_reset':collect(state,r['local'])})
        datasets.append({'name':directory.name,'head_id':protocol['head_id'],'cases_actually_executed':len(manifest['cases']),
            'compiler_or_physical_candidate_executed':bool(manifest['cases']),
            'failure':manifest.get('failure'),'trace_calls':len(trace['events']),
            'public_restore_values':{k:manifest[k] for k in ['state_restored','expression_restored','bone_restored','restore_failures']},
            'public_actual_equality':{a:manifest[a]==manifest[b] for a,b in [('before','after'),('expression_before','expression_after'),('modifiers_before','modifiers_after')]},
            'original_existing_modifiers':manifest['modifiers_before'],'actor_transform_id':snapshot['character']['root_transform_id'] if 'root_transform_id' in snapshot['character'] else snapshot['capture_state']['actor_transform_id'],
            'source_bone_transform_ids':{name:byname[name][-1]['before']['bone_transform_id'] for name in byname},
            'records':records})
    transitions=[]
    for previous,following in zip(datasets,datasets[1:]):
        previous_by={r['name']:r for r in previous['records']}
        for target in following['records']:
            source=previous_by[target['name']]
            # Exact source-only arithmetic; following actual is comparison only, never a desired target.
            predicted=native_partial(source['source_only_cleanup_lifecycle']['remove_modifier_second_reset']['local'],target['native_source_only_expected'],target['native_write_mask'])
            observed={key:target['last_before'][key] for key in predicted}
            error={key:float(np.max(abs(np.asarray(predicted[key])-observed[key]))) for key in predicted}
            transitions.append({'from':previous['name'],'to':following['name'],'name':target['name'],
                'same_bone_transform_id':source['last_before']['bone_transform_id']==target['last_before']['bone_transform_id'],
                'predicted_cleanup_then_partial_native':predicted,'following_actual_for_comparison_only':observed,'error':error,
                'native_write_mask':target['native_write_mask'],'prediction_is_target_for_lowering':False,
                'actual_reset_call_observed_by_apply_trace':False,
                'scope':'source-only two-Reset cleanup/refresh model compared to recorded subsequent identity boundary; Reset/CollectBaseline calls themselves are not traced'})
    sources=[HS2/'tools'/p for p in ['ABMX_BoneController.cs','ABMX_BoneModifier.cs','ABMX_Core.cs','ChaControl.decompiled.txt']]
    sources += [HS2/'plugins/HS2_McpBridge'/p for p in ['MakerAbmxService.cs','MakerMorphService.cs','MakerFaceBaseService.cs']]
    sources += [HS2/'tests/geometry_export/live_native_radial.py',ROOT/'tools/abmx_replay/model.py',ROOT/'tools/native_radial_target/compiler.py',Path(__file__)]
    sources += [ROOT/'data/hs2_head'/p for p in ['enums.json','update_eqns.json']]
    for head in [0,2]:sources.extend(ROOT/f'data/hs2_head/head_{head}'/p for p in ['skeleton.json','anmShapeHead.json'])
    result={'scope':'read-only source/identity-history diagnosis; no physical candidate patch, no private rebase, no game calls',
        'inputs':inputs,'source_bindings':[binding(p) for p in sources],'datasets':datasets,'transitions':transitions,
        'no_fitted_factors_or_candidate_target_substitution':True,
        'public_restore_is_full_skeleton_restore_certificate':False,
        'clean_actor_sufficiency_certified':False,
        'source_findings':[
            'CollectBaseline refreshes latest TRS, but only initializes length/history when old position history is approximately zero.',
            'Reset unconditionally restores latest baseline TRS then normalizes position to first radius unless forced-history restore applies.',
            'RemoveModifier performs Reset before removing modifier and only schedules native face/body refresh.',
            'UpdateBaseline resets only native dictDst-member modifiers, then native face/body writes and recollects TRS; length history can survive.',
            'Identity Apply may return without restoring private baseline; public identity does not erase caches.',
            'Mouthup native rule writes only position.Y; Mouth_L/R write Y and rotation. Reset-corrupted X/Z survive later native59 refresh.',
            'ChangeHead destroys objHead, preserves objHeadBone and copies transforms from persistent bones into new head; it is not a clean actor operation.',
            'Eye_s native writes scale only; inherited position cannot be made a fresh-head-rest position by native59 alone.',
            'Current finally refreshes head/native before RemoveModifier restoration, allowing post-refresh Reset to change unowned position components.']}
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    mouth=[x for x in transitions if x['name'] in ['cf_J_Mouthup','cf_J_Mouth_L','cf_J_Mouth_R','cf_J_Eye_s_L','cf_J_Eye_s_R']]
    (out/'selected_predictions.json').write_text(json.dumps(mouth,indent=2),encoding='utf-8')
    print([(x['from'],x['to'],x['name'],x['predicted_cleanup_then_partial_native']['local_position'],x['error']['local_position']) for x in mouth])


if __name__=='__main__':execute(sys.argv[1])
