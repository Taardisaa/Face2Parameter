"""Optional transform-only anchor proof; does not replace strict execution audit.

The proof covers the installed Apply transition, not writes after Apply, snapshots,
unknown future writers or whole-mesh/character readiness.
"""
from __future__ import annotations
import copy
import numpy as np
from tools.abmx_replay.model import (IDENTITY,CACHE_FIELDS,modifier,cache_state,unity_zero,
    replay_apply,serialized,require)
from tools.abmx_replay.validate_trace import local_only,trs_errors,cache_errors

POLICY='anchored_transform_only_v1'


def exact_cache(a,b):
    left,right=cache_state(a),cache_state(b)
    return all(np.array_equal(np.asarray(left[k]),np.asarray(right[k])) for k in CACHE_FIELDS)


def prove_anchor(cache,physical):
    """Structural installed-source branch proof for every finite incoming TRS."""
    state=cache_state(cache);p=modifier(physical)
    require(state['_hasBaseline'],'Missing baseline')
    require(p['length']==np.float32(1),'Physical Length must be exactly1')
    require(np.any(p['scale']!=np.float32(1)) and np.all(p['scale']>0),'Active positive complete Scale channel required')
    require(np.any(p['rotation']!=np.float32(0)),'Active Rotation channel required')
    require(np.any(p['position']!=np.float32(0)),'Active Position channel required')
    require(not state['_lenModForceUpdate'] or unity_zero(state['_positionBaseline']),
            'Current-direction/length normalization branch would depend on incoming TRS')
    arbitrary={'local_position':[0.,0.,0.],'local_rotation_xyzw':[0.,0.,0.,0.],'local_scale':[0.,0.,0.]}
    result=replay_apply(before=arbitrary,cache=serialized(state),coordinate_modifiers=[physical],coordinate=0,
        additional_modifiers=[],bone_exists=True,rotation_excluded=False,is_during_h_scene=False)
    required=('scale_from_cached_baseline','rotation_baseline_times_euler_zxy','position_only_cached_baseline_plus_offset')
    require(all(b in result['branches'] for b in required) and not any(b.startswith('length_') for b in result['branches']),
            'Installed branch does not replace every incoming TRS channel')
    return {'policy':POLICY,'predicted_after_from_cache_not_incoming':result['after'],
        'branches':result['branches'],'all_finite_incoming_trs_independent_by_branch_structure':True,
        'hypothetical_before_is_not_observed_after':True,'future_or_post_call_writer_absence_certified':False}


def prove_observed_gap(previous,event,physical):
    require(previous['bone_name']==event['bone_name'] and
        previous['modifier_instance_identity']==event['modifier_instance_identity'] and
        previous['after']['bone_transform_id']==event['before']['bone_transform_id'],'Gap bone/modifier identity differs')
    require(previous['sequence']<event['sequence'] and previous['completed_frame']<=event['frame'],'Gap ordering differs')
    require(event['additional_modifiers']==[] and event['no_rotation_excluded'] is False and
        event['is_during_h_scene'] is False and event['coordinate_specific'] is False,'Unsupported observed branch')
    require(exact_cache(previous['after']['cache']['fields'],event['before']['cache']['fields']),
            'Observed gap changed one or more of the12 private fields')
    require(serialized(modifier(event['resolved_modifier']))==serialized(modifier(physical)),
            'Observed gap physical modifier differs')
    proof=prove_anchor(event['before']['cache']['fields'],physical)
    error=trs_errors(proof['predicted_after_from_cache_not_incoming'],local_only(event['after']))
    require(error['passed'],'Anchored Apply prediction differs from actual after')
    return {'sequence':event['sequence'],'previous_sequence':previous['sequence'],'bone_name':event['bone_name'],
        'all12_private_fields_exactly_unchanged':True,'local_change':trs_errors(local_only(previous['after']),local_only(event['before'])),
        'anchor_proof':proof,'predicted_vs_actual_after':error,'writer_cause_identified':False,
        'post_call_snapshot_state_certified':False}


def diagnostic_cursor(snapshot,trace,report,names):
    """Return all observed cursor errors without pretending a failed match passed."""
    require(report['passed'],'Only independently replay-passed traces supported')
    cursor=snapshot['abmx_trace_cursor'];sequence=cursor['last_completed_sequence']
    require(type(sequence) is int and 0<sequence<=len(trace['events']),'Invalid actual cursor')
    require(cursor['observed_calls']==cursor['completed_calls']==sequence and
        snapshot['frame_count']==snapshot['frame_count_end']==cursor['frame'],'Incomplete actual cursor/frame')
    require(cursor['session_id']==trace['metadata']['session_id'] and all(cursor[k]==0 for k in ('pending_calls','dropped_events','observer_error_count')),'Cursor session/loss mismatch')
    require(snapshot['character']['transform_id']==trace['metadata']['character_transform_id'],'Cursor actor differs')
    predicted={};rows=[];selected_ids=set()
    for name in names:
        candidates=[r for r in report['rows'] if r['bone_name']==name and r['sequence']<=sequence]
        require(len(candidates)>=2,'Sparse actual cursor coverage '+name)
        require(len({(r['bone_transform_id'],r['modifier_instance_identity']) for r in candidates})==1,'Cursor identity changed '+name)
        row=candidates[-1];age=cursor['frame']-row['frame'];require(0<=age<=1,'Cursor call stale/future '+name)
        require(row['bone_transform_id'] not in selected_ids,'Duplicate cursor bone');selected_ids.add(row['bone_transform_id'])
        transforms=[t for t in snapshot['transforms'] if t['id']==row['bone_transform_id'] and t['name']==name]
        wrappers=[b['runtime_baseline'] for b in snapshot['abmx_runtime']['bones'] if b['name']==name]
        require(len(transforms)==len(wrappers)==1,'Actual cursor transform/private cache missing '+name)
        wrapper=wrappers[0];require(wrapper['missing_fields']==[] and wrapper['frame_count']==cursor['frame'] and
            wrapper['bone_transform_id']==row['bone_transform_id'] and wrapper['assembly_mvid']==trace['metadata']['plugin_mvid'] and
            wrapper['modifier_type']=='KKABMX.Core.BoneModifier','Cursor private source mismatch '+name)
        prediction=copy.deepcopy(row.get('chain_prediction',row['prediction']))
        local=trs_errors(prediction['after'],local_only(transforms[0]));private=cache_errors(prediction['cache_after'],wrapper['fields'])
        predicted[name]=prediction['after'];rows.append({'bone_name':name,'selected_observed_sequence':row['sequence'],
            'selected_call_frame':row['frame'],'frames_since_selected_call':age,'observed_prefix_call_count':len(candidates),
            'snapshot_local_error':local,'snapshot_cache_error':private,'prediction':prediction,
            'passed':local['passed'] and private['passed']})
    return predicted,{'cursor':cursor,'selected_bones':rows,'passed':all(r['passed'] for r in rows),
        'later_calls_excluded':len(trace['events'])-sequence,'diagnostic_only_no_strict_override':True,
        'external_boundaries_at_or_before_cursor':[b for b in report['inter_call_boundaries'] if b['sequence']<=sequence]}
