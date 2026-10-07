"""NEW strict metadata revision layered over the immutable v1 lowering guard.

The serial-complete cursor relationship is an explicitly supported consumer
policy, not a universal guarantee about reentrant observer completion order.
"""
from __future__ import annotations
from pathlib import Path
from tools.abmx_replay.validate_trace import read, sha, validate
from tools.abmx_stable_lowering.compiler import verify_execution_guard, check_files, digest_json
from tools.abmx_multibone.geometry import DEFAULT_BONES

REVISION='strict_metadata_v1'
FROZEN_COMPILER_SHA256='0f721b16d3812e4e04a5e29bd57156e896e44876ec9b75adf7337061084cb8d8'
METADATA_CONTRACT_SHA256='30585454692ed9c6c4370f4d937619dc91968582a2f162240856b8940938b868'
DEFAULT_METADATA_CONTRACT=Path(__file__).resolve().parents[3]/'HS2Mod/artifacts/ocular_visibility_audit_20261005/stable_lowering_review_v1/new_strict_metadata_contract.json'


def require(value,message):
    if not value: raise ValueError(message)


def exact_int(value,label,nonnegative=True):
    require(type(value) is int and (not nonnegative or value>=0),'Exact integer required: '+label)


def bind_contract(path=DEFAULT_METADATA_CONTRACT):
    require(sha(path)==METADATA_CONTRACT_SHA256,'Strict metadata contract SHA changed')
    contract=read(path)
    check_files({row['path']:row['sha256'] for row in contract['producer_sources']})
    check_files({contract['consumer_source']['path']:contract['consumer_source']['sha256']})
    require(contract['proposed_new_strict_revision']['wrapper_modifier_type']['required_value']=='KKABMX.Core.BoneModifier',
            'Unsupported reflected modifier type')
    return contract


def verify_metadata(snapshot,current_trace_metadata,artifact,*,metadata_contract_path=DEFAULT_METADATA_CONTRACT,completed_trace=None,installed_contract=None):
    """Strict metadata-only check; complete numerical v1 guard is separate.

    At a live boundary completed_trace may be unavailable until observation ends.
    If present it independently certifies the entire stopped trace and exact
    recorded prefix; otherwise count equality is a readback invariant only.
    """
    contract=bind_contract(metadata_contract_path)
    cursor=snapshot.get('abmx_trace_cursor')
    require(isinstance(cursor,dict),'Missing cursor')
    for key in ('frame','observed_calls','completed_calls','last_completed_sequence','pending_calls','dropped_events','observer_error_count'):
        exact_int(cursor.get(key),'cursor.'+key)
    require(type(cursor.get('active')) is bool and cursor['active'] is True,'Guard requires actual active boolean true')
    require(type(cursor.get('session_id')) is str and bool(cursor['session_id']),'Nonempty actual cursor session required')
    require(cursor['pending_calls']==cursor['dropped_events']==cursor['observer_error_count']==0,'Incomplete/lost/error cursor')
    require(cursor['observed_calls']==cursor['completed_calls']==cursor['last_completed_sequence'],
            'Unsupported incomplete/nonserial cursor prefix')
    for key in ('frame_count','frame_count_end'):
        exact_int(snapshot.get(key),'snapshot.'+key)
    require(cursor['frame']==snapshot['frame_count']==snapshot['frame_count_end'],'Exact snapshot/cursor frame mismatch')
    metadata=current_trace_metadata
    require(isinstance(metadata,dict),'Missing fresh actual trace metadata')
    exact_int(metadata.get('started_frame'),'metadata.started_frame')
    exact_int(metadata.get('character_transform_id'),'metadata.character_transform_id',False)
    require(type(metadata.get('session_id')) is str and bool(metadata['session_id']),'Nonempty actual metadata session required')
    require(cursor['session_id']==metadata['session_id'] and cursor['frame']>=metadata['started_frame'],'Wrong/future trace session/frame')
    names=metadata.get('requested_names')
    require(type(names) is list and len(names)==4 and all(type(n) is str for n in names) and set(names)==set(DEFAULT_BONES),
            'Exact four-bone observer filter required')
    p=artifact['provenance']
    require(metadata['bridge_mvid']==p['history_observed_bridge_mvid'] and metadata['plugin_mvid']==p['expected_abmx_mvid']
        and metadata['apply_method_il_sha256']==p['expected_apply_il_sha256'] and metadata['pre_existing_patch_owners']==[],
        'Source bridge/ABMX/IL/patch-owner mismatch')
    exact_int(snapshot['character'].get('transform_id'),'snapshot.character.transform_id',False)
    require(metadata['character_transform_id']==snapshot['character']['transform_id']==p['history_observed_character_transform_id'],
            'Same source actor required')
    wrappers={}
    ids=set()
    for name in DEFAULT_BONES:
        rows=[r for r in snapshot['abmx_runtime']['bones'] if r['name']==name]
        require(len(rows)==1,'Missing/ambiguous wrapper '+name)
        wrapper=rows[0]['runtime_baseline']
        require(type(wrapper.get('modifier_type')) is str and wrapper['modifier_type']=='KKABMX.Core.BoneModifier',
                'Unsupported actual runtime modifier type: '+name)
        exact_int(wrapper.get('frame_count'),'wrapper.frame_count')
        exact_int(wrapper.get('bone_transform_id'),'wrapper.bone_transform_id',False)
        require(type(wrapper.get('missing_fields')) is list and wrapper['missing_fields']==[],'Incomplete private wrapper '+name)
        require(wrapper['assembly_mvid']==p['expected_abmx_mvid'] and wrapper['frame_count']==cursor['frame'],
                'Actual wrapper source/frame mismatch '+name)
        require(wrapper['bone_transform_id'] not in ids,'Required bones share an actual transform')
        ids.add(wrapper['bone_transform_id'])
        matches=[t for t in snapshot['transforms'] if t['id']==wrapper['bone_transform_id'] and t['name']==name]
        require(len(matches)==1,'Exact actual transform mapping missing/ambiguous '+name)
        wrappers[name]={k:wrapper[k] for k in ('modifier_type','assembly_mvid','frame_count','bone_transform_id','missing_fields')}
    prefix_verified=False
    trace_binding=None
    if completed_trace is not None:
        require(installed_contract is not None,'Installed contract required for actual trace verification')
        replay=validate(completed_trace,installed_contract)
        require(replay['passed'],'Completed trace fails independent call replay')
        m=completed_trace['metadata']
        for key in ('session_id','bridge_mvid','plugin_mvid','apply_method_il_sha256','character_transform_id','requested_names','started_frame'):
            require(m[key]==metadata[key],'Completed trace differs from fresh metadata: '+key)
        n=cursor['last_completed_sequence']
        require(n<=len(completed_trace['events']),'Cursor extends beyond actual trace')
        prefix=completed_trace['events'][:n]
        for i,event in enumerate(prefix,1):
            exact_int(event.get('sequence'),'event.sequence')
            exact_int(event.get('frame'),'event.frame')
            exact_int(event.get('completed_frame'),'event.completed_frame')
            require(event['sequence']==i and event['frame']<=event['completed_frame']<=cursor['frame'],
                    'Noncanonical or future completed prefix event')
        prefix_verified=True
        trace_binding={'actual_total_events':len(completed_trace['events']),'actual_selected_prefix_events':n,
                       'future_events_excluded':len(completed_trace['events'])-n,'complete_trace_replay_passed':True}
    return {'revision':REVISION,'passed':True,'cursor':cursor,'actual_wrappers':wrappers,
        'supported_policy':'serial complete ordered prefix; zero/zero/zero allowed; count never inferred from settle_frames',
        'completed_recorded_prefix_independently_verified':prefix_verified,'trace_binding':trace_binding,
        'metadata_contract_path':str(Path(metadata_contract_path).resolve()),'metadata_contract_sha256':METADATA_CONTRACT_SHA256,
        'producer_source_bindings':contract['producer_sources'],'loaded_producer_il_newly_certified':False,
        'utc_age_or_future_stability_certified':False}


def verify_execution_guard_v2(artifact,snapshot,contract_path,*,current_trace_metadata,
        metadata_contract_path=DEFAULT_METADATA_CONTRACT,completed_trace=None):
    """Original numerical/source guard FIRST, then this explicit new revision."""
    core=Path(__file__).with_name('compiler.py')
    require(sha(core)==FROZEN_COMPILER_SHA256,'Frozen v1 compiler changed')
    original=verify_execution_guard(artifact,snapshot,contract_path,current_trace_metadata=current_trace_metadata)
    strict=verify_metadata(snapshot,current_trace_metadata,artifact,metadata_contract_path=metadata_contract_path,
                           completed_trace=completed_trace,installed_contract=read(contract_path))
    return {'revision':REVISION,'passed':True,'original_v1_guard':original,'strict_metadata':strict,
        'revision_binding':{'v1_compiler_sha256':FROZEN_COMPILER_SHA256,'strict_guard_sha256':sha(__file__),
                            'metadata_contract_sha256':METADATA_CONTRACT_SHA256,'artifact_content_sha256':digest_json(artifact)},
        'compiled_artifact_mutated':False,'old_live_guard_retroactively_claimed_v2':False,
        'future_or_runtime_stability_certified':False}
