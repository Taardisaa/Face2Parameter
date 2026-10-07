"""ALL30 source-only physical compiler and prepared clean-identity guard."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig
from src.hs2_deform_torch import TorchHeadRig
from src.face_data_utils.utils import PARAMETER_FLAGS,ONLY_RIGHT_INDEX
from tools.abmx_replay.model import FLAG_FIELDS,IDENTITY,modifier,serialized,cache_state,replay_apply,unity_zero
from tools.abmx_replay.validate_trace import read,sha,trs_errors,local_only
from tools.abmx_multibone.geometry import select_cursor_predictions
from tools.abmx_stability.classify import source_contract
from tools.abmx_stable_lowering.compiler import check_files,digest_json
from .adapter import MODE,PROFILE,NAMES,patch_array,native_source_bindings,assert_coverage

_CAPABILITY=object()


def require(value,message):
    if not value:raise ValueError(message)


def _clean(fields):
    state=cache_state(fields)
    require(state['_hasBaseline'] and not any(state[k] for k in ('_changedScale','_changedRotation','_changedPosition','_forceApply')),
        'Identity boundary needs baseline and cleared channel/forceApply flags')
    # Installed HasLenBaseline is _positionBaseline != Vector3.zero. Under a
    # fixed approximately-zero history, Length=1 cannot enter normalization even
    # with sticky lenForceUpdate. NeedsPositionRestore is unused by Apply there.
    if not unity_zero(state['_positionBaseline']):
        require(not state['_lenModForceUpdate'] and not state['_lenModNeedsPositionRestore'],
                'Nonzero history with pending length/restore flag unsupported')
    return state


def _native(rig,values):
    native=np.asarray(values,dtype=np.float64)
    require(native.shape==(59,) and np.isfinite(native).all(),'Explicit full59 finite native input required')
    model=TorchHeadRig(rig,device='cpu',dtype=torch.float64)
    with torch.no_grad():arrays=[v[0].numpy() for v in model.local_transforms(torch.as_tensor(native[None],dtype=torch.float64),None)]
    result={}
    for name in NAMES:
        i=model.name2bone[name]
        result[name]={key:array[i].tolist() for key,array in zip(('local_position','local_rotation_xyzw','local_scale'),arrays)}
    return result


def _patch_map(value):
    if isinstance(value,list):
        require(len(value)==30 and len({r['name'] for r in value})==30,'Unique all30 source patch rows required')
        return {r['name']:{k:r[k] for k in IDENTITY} for r in value}
    return value


def _declaration_origin(declared):
    require(type(declared['source_files']) is dict and declared['source_files'],'Frozen logical protocol source required')
    check_files(declared['source_files'])
    for path in declared['source_files']:
        try:d=read(path)
        except (ValueError,UnicodeError):continue
        if not isinstance(d,dict):continue
        patches=d.get('logical_patches',d.get('patches'))
        try:equal=np.array_equal(patch_array(_patch_map(patches)),patch_array(declared['logical_patches']))
        except (ValueError,TypeError,KeyError):continue
        head=d.get('head_id',d.get('heads'))
        if (equal and d.get('semantic_mode')==MODE and (head==declared['head_id'] or head==[declared['head_id']]) and
            d.get('sampling_profile')==PROFILE and d.get('native59')==declared['native59'] and
            d.get('source_history_native59',d.get('source_history_expected_native59'))==declared['source_history_native59']):return
    raise ValueError('Native/mode/head/profile/logical30 not bound to frozen protocol')


def compile_native_target(history,declared,contract_path):
    """Root API: stopped ALL30 identity history + predeclared raw logical map.

    No target candidate observation is accepted. Expected native TRS is generated
    from cached source; no actual after is substituted as the desired target.
    """
    require(set(declared)=={'semantic_mode','head_id','native59','source_history_native59','sampling_profile','logical_patches','source_files'},'Explicit all30 semantic declaration required')
    require(declared['semantic_mode']==MODE and declared['sampling_profile']==PROFILE and type(declared['head_id']) is int,'Explicit mode/profile/head required')
    logical=patch_array(declared['logical_patches'])
    _declaration_origin(declared)
    contract=read(contract_path);source_contract(contract)
    require(not set(NAMES)&set(contract['no_rotation_bones']),'Excluded all30 bone unsupported')
    files={str(Path(contract_path).resolve()):sha(contract_path),history['trace']:history['trace_sha256'],
        history['geometry']['path']:history['geometry']['sha256'],**declared['source_files']}
    check_files(files)
    trace,snapshot=read(history['trace']),read(history['geometry']['path'])
    require(snapshot['character']['head_id']==declared['head_id'],'Source history head differs')
    require(np.array_equal(np.asarray(snapshot['character']['shape_value_face'],dtype=np.float32),np.asarray(declared['source_history_native59'],dtype=np.float32)),'Declared source native differs')
    _,binding,replay=select_cursor_predictions(snapshot,trace,contract,required_names=NAMES)
    require(not replay['inter_call_boundaries'],'Unidentified external writer boundary unsupported')
    rig=HeadRig(declared['head_id'],sampling_profile=PROFILE);assert_coverage(rig)
    from tools.abmx_replay.validate_geometry import cache_correspondence
    heads=[m for m in snapshot['meshes'] if m['mesh_name']=='o_head']
    require(len(heads)==1,'Unique source head renderer required for asset correspondence')
    source_mesh_correspondence=cache_correspondence(rig,heads[0])
    native_source=_native(rig,declared['source_history_native59']);native=_native(rig,declared['native59'])
    history_fields={};source_checks={}
    for name in NAMES:
        events=[e for e in trace['events'] if e['bone_name']==name]
        first=events[0];fields=first['before']['cache']['fields'];state=_clean(fields)
        identity=(first['before']['bone_transform_id'],first['modifier_instance_identity'])
        for e in events:
            require(not e['additional_modifiers'] and not e['no_rotation_excluded'] and not e['is_during_h_scene'] and e['coordinate_specific'] is False,'Unsupported additional/excluded/scene/coordinate branch')
            require((e['before']['bone_transform_id'],e['modifier_instance_identity'])==identity,'Source bone/modifier identity changed')
            require(serialized(modifier(e['resolved_modifier']))==serialized(modifier(IDENTITY)),'Source history must be entirely identity')
            for phase in ('before','after'):
                s=_clean(e[phase]['cache']['fields'])
                require(all(np.array_equal(np.asarray(s[k]),np.asarray(state[k])) for k in state),'Source clean baseline/history changed')
        baseline={key:fields[field] for key,field in (('local_position','_posBaseline'),('local_rotation_xyzw','_rotBaseline'),('local_scale','_sclBaseline'))}
        source_checks[name]={'native_vs_before':trs_errors(native_source[name],local_only(first['before'])),
                            'native_vs_baseline':trs_errors(native_source[name],baseline)}
        require(all(c['passed'] for c in source_checks[name].values()),'Source native/baseline mismatch: '+name)
        history_fields[name]=serialized(state)
    # Reject unsupported active foreign modifiers rather than ignoring writer state.
    for row in snapshot['abmx_runtime']['bones']:
        if row['name'] not in NAMES:
            require(serialized(modifier({k:row[k] for k in IDENTITY}))==serialized(modifier(IDENTITY)),
                'Nonidentity extra public modifier outside ALL30 unsupported: '+row['name'])
    files.update(native_source_bindings(rig))
    root=Path(__file__).resolve().parents[2]
    for path in (Path(__file__),root/'tools/abmx_replay/model.py',root/'tools/abmx_replay/validate_trace.py',
        root/'tools/abmx_multibone/geometry.py',root/'tools/abmx_stability/classify.py'):
        files[str(path.resolve())]=sha(path)
    files.update({contract['assembly_path']:contract['assembly_sha256'],contract['unity_core_path']:contract['unity_core_sha256']})
    files.update({r['path']:r['sha256'] for r in contract['sources']})
    rows={};physical=[];targets={}
    for name,m in zip(NAMES,logical):
        n={k:np.asarray(v,dtype=np.float32) for k,v in native[name].items()}
        require(np.isfinite(np.r_[*n.values()]).all() and np.all(n['local_scale']>0),'Invalid/nonpositive source native TRS')
        target=n['local_position']*m[3]+m[4:7]
        physical_modifier={'scale':m[:3].tolist(),'length':1.,'position':(target-n['local_position']).tolist(),'rotation':m[7:].tolist()}
        cache={**history_fields[name],'_posBaseline':n['local_position'].tolist(),'_rotBaseline':n['local_rotation_xyzw'].tolist(),'_sclBaseline':n['local_scale'].tolist()}
        prediction=replay_apply(before=serialized(n),cache=cache,coordinate_modifiers=[physical_modifier],coordinate=0,
            additional_modifiers=[],bone_exists=True,rotation_excluded=False,is_during_h_scene=False)
        # Position target is independent of all historical radius/direction fields.
        error=float(np.abs(np.asarray(prediction['after']['local_position'])-target).max())
        require(error<=1e-6,'Float32 physical position roundtrip exceeds existing local tolerance')
        logical_modifier={k:declared['logical_patches'][name][k] for k in IDENTITY}
        rows[name]={'native_baseline':serialized(n),'desired_native_radius_position':target.tolist(),'physical_prediction':prediction,
            'logical_modifier':logical_modifier,'executed_modifier':physical_modifier,'position_roundtrip_max_abs':error,
            'persistent_identity_fields':history_fields[name], 'historical_radius_used_for_target':False,
            'identity_flag_policy':'zero_history_inert_length_flags' if unity_zero(np.asarray(history_fields[name]['_positionBaseline'],dtype=np.float32)) else 'strict_nonzero_history_clean_flags'}
        targets[name]=prediction['after'];physical.append({'name':name,**physical_modifier})
    provenance={'source_files':files,'head_id':declared['head_id'],'sampling_profile':PROFILE,'native59':declared['native59'],
        'source_history_native59':declared['source_history_native59'],'source_cursor':binding['cursor'],
        'source_bridge_mvid':trace['metadata']['bridge_mvid'],'source_actor_id':snapshot['character']['transform_id'],
        'expected_abmx_mvid':contract['plugin_mvid'],'expected_apply_il_sha256':contract['apply_method_il_sha256'],
        'game_assembly_sha256':snapshot['game']['game_assembly_sha256'],'source_checks':source_checks,
        'source_head_mesh_correspondence':source_mesh_correspondence,
        'source_complete_identity_calls':len(trace['events']),'source_external_boundaries':[]}
    result={'schema_version':1,'semantic_mode':MODE,'semantic_definition':'t=p_native*logical_Length+logical_offset; NOT installed historical first Apply',
        'source_bindings':provenance,'all30_order':list(NAMES),'parameter_layout':['scale_x','scale_y','scale_z','length','position_x','position_y','position_z','rotation_x','rotation_y','rotation_z'],
        'mask_contract':{'raw_bone_values':300,'model_native_values':54,'runtime_native_values':59,'label_simplified_without_right':205,
            'source_parameter_flags':list(PARAMETER_FLAGS),'source_only_right_indices':list(ONLY_RIGHT_INDEX),
            'physical_offsets_may_use_masked_logical_channels':True,'physical_to_205_roundtrip_lossless':False},
        'logical_patches':[{'name':n,**declared['logical_patches'][n]} for n in NAMES],'executed_patches':physical,
        'native_baselines':native,'desired_target_locals':targets,'bone_diagnostics':rows,
        'compiler_inputs':{'history':copy.deepcopy(history),'declared':copy.deepcopy(declared),'contract_path':str(Path(contract_path).resolve())},
        'dtype_contract':{'native_predictor':'float64','physical_lowering_and_installed_replay':'float32'},
        'quality_policy':{'finite':True,'positive_scale_and_length':True,'self_cross_mesh_quality_passed':False},
        'fitted_coefficients':False,'fitted_counts':False,'candidate_actual_after_input':False,
        'all30_runtime_certified':False,'arbitrary_candidate_runtime_certified':False,'character_ready':False}
    # convert NumPy scalar flags/indices before JSON freeze
    result=serialized(result)
    result['artifact_binding_sha256']=digest_json(result)
    return result


class GuardContext:
    """Trusted in-process context, created only after full source regeneration."""
    def __init__(self,artifact,contract,*,_capability,artifact_path=None,artifact_sha256=None):
        require(_capability is _CAPABILITY,'Use prepare_execution_guard; JSON trust tokens unsupported')
        self._source_artifact_ref=artifact
        self._artifact=copy.deepcopy(artifact);self._contract=copy.deepcopy(contract)
        self._digest=digest_json(self._artifact)
        self._artifact_path=artifact_path;self._artifact_sha=artifact_sha256

    def __reduce__(self):raise TypeError('GuardContext is nonserializable; prepare from original sources in this process')

    def verify_snapshot(self,snapshot,current_trace_metadata):
        # Retain actual cryptographic source checks; no inference/history replay.
        a=self._artifact;p=a['source_bindings'];meta=current_trace_metadata
        require(digest_json(a)==self._digest,'Prepared context mutated')
        require(digest_json(self._source_artifact_ref)==self._digest,'Caller compiled dictionary changed after preparation')
        check_files(p['source_files'])
        if self._artifact_path:require(sha(self._artifact_path)==self._artifact_sha,'Frozen artifact file changed')
        cursor=snapshot['abmx_trace_cursor']
        for key in ('frame','observed_calls','completed_calls','last_completed_sequence','pending_calls','dropped_events','observer_error_count'):
            require(type(cursor.get(key)) is int and cursor[key]>=0,'Exact nonnegative cursor int required: '+key)
        require(cursor.get('active') is True and type(cursor.get('active')) is bool,'Actual active boolean required')
        require(cursor['observed_calls']==cursor['completed_calls']==cursor['last_completed_sequence'],'Supported complete serial cursor prefix required')
        require(cursor['pending_calls']==cursor['dropped_events']==cursor['observer_error_count']==0,'Incomplete observer prefix')
        require(type(meta.get('session_id')) is str and meta['session_id'] and cursor['session_id']==meta['session_id'],'Actual session mismatch')
        require(type(meta.get('started_frame')) is int and 0<=meta['started_frame']<=cursor['frame'],'Actual started frame mismatch')
        require(all(type(snapshot[k]) is int and snapshot[k]==cursor['frame'] for k in ('frame_count','frame_count_end')),'Frame binding mismatch')
        require(type(meta.get('requested_names')) is list and len(meta['requested_names'])==30 and set(meta['requested_names'])==set(NAMES),'All30 observer filter required')
        require(meta['bridge_mvid']==p['source_bridge_mvid'] and meta['plugin_mvid']==p['expected_abmx_mvid'] and
            meta['apply_method_il_sha256']==p['expected_apply_il_sha256'] and meta['pre_existing_patch_owners']==[],'Source DLL/IL/external patch mismatch')
        character=snapshot['character']
        require(type(character['transform_id']) is int and character['transform_id']==meta['character_transform_id']==p['source_actor_id'],'Same source actor required')
        require(character['head_id']==p['head_id'] and np.array_equal(np.asarray(character['shape_value_face'],dtype=np.float32),np.asarray(p['native59'],dtype=np.float32)),'Full59/base differs from source target')
        require(snapshot['game']['game_assembly_sha256']==p['game_assembly_sha256'],'Game source assembly changed')
        checks={};ids=set()
        for row in snapshot['abmx_runtime']['bones']:
            require(serialized(modifier({k:row[k] for k in IDENTITY}))==serialized(modifier(IDENTITY)),'Guard requires all actual public modifiers identity; extra active bone refused')
        for name in NAMES:
            rows=[r for r in snapshot['abmx_runtime']['bones'] if r['name']==name]
            require(len(rows)==1 and rows[0]['exists'] is True,'Missing/ambiguous actual30 bone '+name)
            w=rows[0]['runtime_baseline']
            require(w['modifier_type']=='KKABMX.Core.BoneModifier' and w['assembly_mvid']==p['expected_abmx_mvid'] and w['missing_fields']==[], 'Actual modifier/source fields mismatch')
            require(type(w['frame_count']) is int and w['frame_count']==cursor['frame'] and type(w['bone_transform_id']) is int,'Actual wrapper frame/ID invalid')
            require(w['bone_transform_id'] not in ids,'Shared actual bone ID')
            ids.add(w['bone_transform_id'])
            state=_clean(w['fields']);expected=a['bone_diagnostics'][name]['persistent_identity_fields']
            for key in ('_lenBaseline','_positionBaseline'):
                require(np.array_equal(np.asarray(state[key],dtype=np.float32),np.asarray(expected[key],dtype=np.float32)),'Identity history differs; regenerate context')
            for key in FLAG_FIELDS:
                require(state[key]==expected[key],'Identity boundary flag lifecycle differs; regenerate context')
            transforms=[t for t in snapshot['transforms'] if t['name']==name and t['id']==w['bone_transform_id']]
            require(len(transforms)==1,'Exact actual30 transform mapping invalid')
            baseline={key:serialized(state[field]) for key,field in (('local_position','_posBaseline'),('local_rotation_xyzw','_rotBaseline'),('local_scale','_sclBaseline'))}
            checks[name]={'native_vs_before':trs_errors(a['native_baselines'][name],local_only(transforms[0])),
                'native_vs_cache':trs_errors(a['native_baselines'][name],baseline),'current_bone_id':w['bone_transform_id']}
            require(checks[name]['native_vs_before']['passed'] and checks[name]['native_vs_cache']['passed'],'Native30 baseline/current TRS differs '+name)
        return {'passed':True,'semantic_mode':MODE,'all30_checks':checks,'actual_cursor':cursor,'prepared_context_binding':self._digest,
            'old_bone_ids_reused':False,'inference_or_trace_replay_during_guard':False,'counts_inferred_from_settle':False,
            'future_writer_absence_or_stability_certified':False}


def prepare_execution_guard(compiled,contract_path,*,artifact_path=None,artifact_sha256=None):
    """Expensive source regeneration BEFORE starting candidate observer."""
    inputs=compiled['compiler_inputs']
    require(str(Path(contract_path).resolve())==inputs['contract_path'],'Installed contract differs')
    expected=compile_native_target(inputs['history'],inputs['declared'],contract_path)
    require(compiled==expected,'Compiled native target differs from independently reopened source inputs')
    if artifact_path:
        require(type(artifact_sha256) is str and sha(artifact_path)==artifact_sha256,'Explicit frozen artifact file SHA required')
    return GuardContext(compiled,read(contract_path),_capability=_CAPABILITY,artifact_path=artifact_path,artifact_sha256=artifact_sha256)
