"""Source-bound radial target -> installed baseline-anchored position branch.

This is an alternative executed control definition, not a change to installed
ABMX or a claim that a combined length/position patch is naturally stable.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig
from src.hs2_deform_torch import TorchHeadRig
from tools.abmx_replay.model import (FLAG_FIELDS, IDENTITY, cache_state, modifier,
    replay_apply, serialized, unity_zero, unity_sqrmag)
from tools.abmx_replay.validate_trace import read, sha, trs_errors
from tools.abmx_multibone.geometry import DEFAULT_BONES, select_cursor_predictions
from tools.abmx_multibone.run import restored_flags
from tools.abmx_stability.classify import source_contract
from tools.stateful_multibone_fresh.adapter import clean, persistent_equal, local_target

PROFILE = 'slider_unlocker_18_2'
ROOT = Path(__file__).resolve().parents[2]


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def check_files(files):
    if not files or any(sha(path) != expected for path, expected in files.items()):
        raise ValueError('Stale source/history/native predictor provenance')


def verify_declaration_origin(declared):
    """Bind requested targets to values in a frozen declaration/protocol file."""
    check_files(declared['source_files'])
    for path in declared['source_files']:
        try: value=read(path)
        except (ValueError,UnicodeError): continue
        if not isinstance(value,dict): continue
        candidates=[]
        if 'logical_patches' in value: candidates.append(value['logical_patches'])
        for row in value.get('regimes',[]):
            if row.get('name')=='combined': candidates.append(row.get('patches'))
        head=value.get('head_id',value.get('heads'))
        head_match=head==declared['head_id'] or head==[declared['head_id']]
        source=value.get('source_history_native59',value.get('source_history_expected_native59'))
        if (head_match and value.get('native59')==declared['native59'] and source==declared['source_history_native59']
            and value.get('sampling_profile')==declared['sampling_profile'] and declared['logical_patches'] in candidates):
            return
    raise ValueError('Logical/native/head inputs not found in a frozen declaration source')


def radial_target_torch(position, radius, length, offset):
    """Differentiable selected-branch formula, no actual candidate after input.

    Caller retains floating dtype. Float32 uses installed normalization/multiply
    order; float64 is useful for analytical derivative checks.
    """
    if position.shape[-1:] != (3,) or offset.shape != position.shape:
        raise ValueError('Three-component position/offset required')
    if any(not torch.isfinite(v).all() for v in (position, radius, length, offset)):
        raise ValueError('Nonfinite radial inputs')
    square = (position[..., 0]*position[..., 0] + position[..., 1]*position[..., 1]) + position[..., 2]*position[..., 2]
    if (square < 9.9999994e-11).any() or (radius <= 0).any() or (length < .1).any() or (length == 1).any():
        raise ValueError('Unsupported zero/radius/special/identity-length branch')
    return ((position / torch.sqrt(square)[..., None]) * radius[..., None]) * length[..., None] + offset


def lower_bone(native, history, logical):
    """Return complete logical/executed modifier and first-target diagnostics.

    Cache TRS is generated from declared native; only persistent history is read.
    The input history must be an already source-bound clean identity boundary.
    """
    state = cache_state(history)
    clean(serialized(state), 'lowering')
    m = modifier(logical)
    p = np.asarray(native['local_position'], dtype=np.float32)
    q = np.asarray(native['local_rotation_xyzw'], dtype=np.float32)
    s = np.asarray(native['local_scale'], dtype=np.float32)
    if (p.shape != (3,) or q.shape != (4,) or s.shape != (3,) or
            not np.isfinite(np.r_[p,q,s]).all() or abs(float(np.linalg.norm(q))-1) > 1e-5):
        raise ValueError('Invalid native local TRS')
    if (unity_zero(p) or unity_zero(state['_positionBaseline']) or state['_lenBaseline'] <= 0 or
            m['length'] < np.float32(.1) or m['length'] == np.float32(1)):
        raise ValueError('Unsupported logical radial branch; no identity/small-length/historical-direction lowering')
    # Exact installed arithmetic: normalize, radius multiply, length multiply,
    # then offset. Do not collapse the two multiplies to R*L beforehand.
    target = ((p / np.float32(np.sqrt(unity_sqrmag(p)))) * np.float32(state['_lenBaseline'])) * m['length'] + m['position']
    execution = {**serialized(m), 'length': 1., 'position': (target-p).tolist()}
    baseline = {**serialized(state), '_posBaseline': p.tolist(), '_rotBaseline': q.tolist(), '_sclBaseline': s.tolist()}
    before = {'local_position':p.tolist(), 'local_rotation_xyzw':q.tolist(), 'local_scale':s.tolist()}
    kwargs = dict(coordinate=0, additional_modifiers=[], bone_exists=True, rotation_excluded=False, is_during_h_scene=False)
    original = replay_apply(before=before, cache=baseline, coordinate_modifiers=[serialized(m)], **kwargs)
    lowered = replay_apply(before=before, cache=baseline, coordinate_modifiers=[execution], **kwargs)
    error = trs_errors(original['after'], lowered['after'])
    rounding_budget = float(min(1e-5, 8*np.finfo(np.float32).eps*max(1.,float(np.abs(p).max()),float(np.abs(target).max()))))
    position_error = float(np.abs(np.asarray(original['after']['local_position'])-np.asarray(lowered['after']['local_position'])).max())
    if not error['passed'] or position_error > rounding_budget:
        raise ValueError('Float32 first-target equivalence exceeds fixed tolerance')
    return {'logical_modifier': serialized(m), 'executed_modifier':execution,
            'native_baseline':before, 'persistent_history':{k:serialized(state[k]) for k in ('_lenBaseline','_positionBaseline')},
            'desired_first_clean_target':target.tolist(), 'execution_offset':execution['position'],
            'logical_first_prediction':original, 'executed_first_prediction':lowered,
            'first_equivalence':error, 'position_component_max_error':position_error,
            'position_rounding_budget':rounding_budget,
            'target_source':'declared cached native local position + earlier identity radius + logical patch; no observed candidate after',
            'stable_branch_assumptions':['Fixed native TRS baseline and physical patch.', 'Clean identity flags before first execution.',
                'No native rebuild, restore, baseline recollect, animator, additional modifier or unobserved writer between calls.'],
            'actual_stability_certified':False}


def native_locals(head_id, values):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (59,) or not np.isfinite(values).all():
        raise ValueError('Complete finite declared native59 required')
    rig = HeadRig(head_id, sampling_profile=PROFILE)
    predictor = TorchHeadRig(rig, device='cpu', dtype=torch.float64)
    with torch.no_grad():
        arrays = [v[0].cpu().numpy() for v in predictor.local_transforms(torch.as_tensor(values[None], dtype=torch.float64))]
    result = {}
    for name in DEFAULT_BONES:
        if name not in predictor.name2bone:
            raise ValueError('Missing four-bone predictor')
        i = predictor.name2bone[name]
        result[name] = {key:array[i].tolist() for key,array in zip(('local_position','local_rotation_xyzw','local_scale'), arrays)}
    return result


class HistoryAnchor:
    """Immutable logical compiler context for one declared base/native/history.

    Same-native source/candidate is supported here; the existing fresh diagnostic
    and its changed-native requirement are not altered.
    """
    def __init__(self, provenance, history, candidate_native, native, logical_patches):
        self.provenance = copy.deepcopy(provenance)
        self.history = copy.deepcopy(history)
        self.native59 = list(candidate_native)
        self.native = copy.deepcopy(native)
        self.logical_patches = copy.deepcopy(logical_patches)
        self.binding = digest_json({'provenance':self.provenance, 'history':self.history,
                                    'native59':self.native59, 'native':self.native,
                                    'logical_patches':self.logical_patches})

    @classmethod
    def from_history(cls, history, declared, contract_path):
        """History-only interface for a new live experiment.

        history: {trace, trace_sha256, geometry:{path,sha256}}.
        declared: {head_id,native59,source_history_native59,logical_patches,
                   sampling_profile,source_files:{predeclared_path:sha256}}.
        Declaration must be frozen before this compiler consumes its inputs.
        No candidate trace, snapshot, local TRS or after data is accepted.
        """
        if set(declared)!={'head_id','native59','source_history_native59','logical_patches','sampling_profile','source_files'}:
            raise ValueError('Explicit history-only declaration fields required; candidate observations forbidden')
        if declared['sampling_profile']!=PROFILE or type(declared['head_id']) is not int:
            raise ValueError('Explicit installed sampling profile and integer head ID required')
        verify_declaration_origin(declared)
        patches=declared['logical_patches']
        if len(patches)!=4 or {p['name'] for p in patches}!=set(DEFAULT_BONES) or any(set(p)!=set(IDENTITY)|{'name'} for p in patches):
            raise ValueError('Exactly four supported complete unique logical patches required')
        contract=read(contract_path)
        source_contract(contract)
        files={str(Path(contract_path).resolve()):sha(contract_path),history['trace']:history['trace_sha256'],
               history['geometry']['path']:history['geometry']['sha256'],**declared['source_files']}
        check_files(files)
        trace,snapshot=read(history['trace']),read(history['geometry']['path'])
        _,binding,replay=select_cursor_predictions(snapshot,trace,contract)
        if replay['inter_call_boundaries'] or 'stopped_frame' not in trace['metadata']:
            raise ValueError('Complete stopped history without external boundary required')
        head_id=declared['head_id']
        source_native=np.asarray(declared['source_history_native59'],dtype=np.float64)
        candidate_native=np.asarray(declared['native59'],dtype=np.float64)
        if snapshot['character']['head_id']!=head_id or source_native.shape!=(59,) or not np.array_equal(source_native.astype(np.float32),np.asarray(snapshot['character']['shape_value_face'],dtype=np.float32)):
            raise ValueError('Head/source native59 differs from declaration')
        predicted_source=native_locals(head_id,source_native)
        predicted_candidate=native_locals(head_id,candidate_native)
        fields_by_name,identities,checks={},{},{}
        for name in DEFAULT_BONES:
            events=[e for e in trace['events'] if e['bone_name']==name]
            first=events[0]
            fields=first['before']['cache']['fields']
            cache_state(fields)
            identity=(first['before']['bone_transform_id'],first['modifier_instance_identity'])
            for e in events:
                if (e['before']['bone_transform_id'],e['modifier_instance_identity'])!=identity or e['additional_modifiers'] or e['is_during_h_scene'] or e['no_rotation_excluded']:
                    raise ValueError('Unsupported identity/history branch')
                if serialized(modifier(e['resolved_modifier']))!=serialized(modifier(IDENTITY)):
                    raise ValueError('Earlier history must be entirely identity')
                for phase in ('before','after'):
                    clean(e[phase]['cache']['fields'],name)
                    if not persistent_equal(e[phase]['cache']['fields'],fields):
                        raise ValueError('Persistent history changed')
            baseline={key:fields[field] for key,field in (('local_position','_posBaseline'),('local_rotation_xyzw','_rotBaseline'),('local_scale','_sclBaseline'))}
            checks[name]={'source_before':trs_errors(predicted_source[name],local_target(first['before'])),
                          'source_baseline':trs_errors(predicted_source[name],baseline)}
            if not all(r['passed'] for r in checks[name].values()):
                raise ValueError('Native predictor fails earlier clean source boundary')
            fields_by_name[name]=copy.deepcopy(fields)
            identities[name]={'observed_bone_transform_id':identity[0],'observed_modifier_instance_identity':identity[1]}
        cache_dir=ROOT/'data/hs2_head'/('head_'+str(head_id))
        for path in cache_dir.rglob('*'):
            if path.is_file(): files[str(path.resolve())]=sha(path)
        for path in (ROOT/'src/hs2_mesh_deform.py',ROOT/'src/hs2_deform_torch.py',ROOT/'src/hs2_sampling.py',
                     ROOT/'data/hs2_head/enums.json',ROOT/'data/hs2_head/customhead.json',ROOT/'data/hs2_head/update_eqns.json',
                     ROOT/'tools/stateful_multibone_fresh/adapter.py',ROOT/'tools/abmx_replay/model.py',
                     ROOT/'tools/abmx_replay/validate_trace.py',ROOT/'tools/abmx_multibone/geometry.py',
                     ROOT/'tools/abmx_stability/classify.py',Path(__file__)):
            files[str(path.resolve())]=sha(path)
        files.update({contract['assembly_path']:contract['assembly_sha256'],contract['unity_core_path']:contract['unity_core_sha256']})
        files.update({r['path']:r['sha256'] for r in contract['sources']})
        provenance={'source_files':files,'head_id':head_id,'sampling_profile':PROFILE,'candidate_native59':candidate_native.tolist(),
            'source_history_native59':source_native.tolist(),'source_history_cursor':binding['cursor'],
            'source_history_session_id':trace['metadata']['session_id'],'history_observed_character_transform_id':snapshot['character']['transform_id'],
            'history_observed_bone_identities':identities,'history_observed_bridge_mvid':trace['metadata']['bridge_mvid'],
            'expected_game_assembly_sha256':snapshot['game']['game_assembly_sha256'],
            'expected_abmx_mvid':contract['plugin_mvid'],'expected_apply_il_sha256':contract['apply_method_il_sha256'],
            'source_native_predictor_checks':checks,'source_history_clean_identity_certified':True,
            'runtime_instance_ids_are_guard_expectations':False,
            'live_guard_policy':'New identities permitted only with exact float32 persistent numeric history and independently checked native TRS/clean flags.',
            'candidate_observations_accepted':False}
        return cls(provenance,fields_by_name,candidate_native.tolist(),predicted_candidate,patches)

    def compile(self, *, expected_binding, expected_head_id, expected_native59):
        now = digest_json({'provenance':self.provenance,'history':self.history,'native59':self.native59,'native':self.native,'logical_patches':self.logical_patches})
        if expected_binding!=self.binding or now!=self.binding or expected_head_id!=self.provenance['head_id'] or not np.array_equal(np.asarray(expected_native59),np.asarray(self.native59)):
            raise ValueError('Stale head/native/history/compiler binding')
        check_files(self.provenance['source_files'])
        rows={}
        for patch in self.logical_patches:
            name=patch['name']
            rows[name]=lower_bone(self.native[name],self.history[name],{k:patch[k] for k in IDENTITY})
        logical=[{'name':name,**rows[name]['logical_modifier']} for name in DEFAULT_BONES]
        executed=[{'name':name,**rows[name]['executed_modifier']} for name in DEFAULT_BONES]
        return {'schema_version':1,'kind':'optional_first_clean_radial_target_to_position_only_lowering',
                'compiler_binding_sha256':self.binding,'provenance':self.provenance,'logical_patches':logical,'executed_patches':executed,
                'expected_candidate_native_baselines':self.native,'bone_diagnostics':rows,
                'expected_execution_initial_flags':{k:k=='_hasBaseline' for k in FLAG_FIELDS},
                'semantic_change':'Preserves desired FIRST clean logical apply TRS within float32 tolerance; deliberately replaces later radial iteration by fixed baseline position branch.',
                'coefficients_fitted':False,'call_counts_fitted':False,'actual_candidate_after_used_for_target':False,
                'actual_stability_certified':False,'whole_head_runtime_certified':False,'character_ready':False}


def compile_history(history, declared, contract_path):
    """Root-callable complete four-row artifact; inputs described in from_history."""
    anchor=HistoryAnchor.from_history(history,declared,contract_path)
    result=anchor.compile(expected_binding=anchor.binding,expected_head_id=declared['head_id'],expected_native59=declared['native59'])
    result['compiler_inputs']={'history':copy.deepcopy(history),'declared':copy.deepcopy(declared),'contract_path':str(Path(contract_path).resolve())}
    return result


def verify_compiled_artifact(artifact):
    """Recompute targets from immutable inputs, not self-declared numerical rows."""
    inputs=artifact['compiler_inputs']
    predicted=compile_history(inputs['history'],inputs['declared'],inputs['contract_path'])
    for key,value in predicted.items():
        if artifact.get(key)!=value:
            raise ValueError('Compiler artifact differs from source-recomputed '+key)
    return predicted


def verify_execution_guard(artifact, snapshot, contract_path, *, current_trace_metadata):
    """Read-only fresh identity snapshot guard; does not certify future stability.

    Recreated bone/modifier IDs are recorded freshly rather than compared to old
    IDs. Numerical persistent cache fields must match exactly after float32 cast.
    A differing radius/history requires a NEW frozen compiler artifact.
    """
    verify_compiled_artifact(artifact)
    if str(Path(contract_path).resolve())!=artifact['compiler_inputs']['contract_path']:
        raise ValueError('Guard installed contract differs from frozen compiler input')
    source_contract(read(contract_path))
    p=artifact['provenance']
    check_files(p['source_files'])
    metadata=current_trace_metadata
    cursor=snapshot['abmx_trace_cursor']
    if (metadata['bridge_mvid']!=p['history_observed_bridge_mvid'] or metadata['plugin_mvid']!=p['expected_abmx_mvid'] or
        metadata['apply_method_il_sha256']!=p['expected_apply_il_sha256'] or metadata['pre_existing_patch_owners'] or
        metadata['character_transform_id']!=snapshot['character']['transform_id'] or
        snapshot['character']['transform_id']!=p['history_observed_character_transform_id'] or
        snapshot['game']['game_assembly_sha256']!=p['expected_game_assembly_sha256'] or
        cursor['session_id']!=metadata['session_id'] or set(metadata['requested_names'])!=set(DEFAULT_BONES)):
        raise ValueError('Fresh bridge/ABMX/actor/trace source identity mismatch')
    if any(cursor.get(k)!=0 for k in ('pending_calls','dropped_events','observer_error_count')) or cursor['frame']!=snapshot['frame_count']:
        raise ValueError('Fresh cursor incomplete or wrong frame')
    if snapshot['frame_count']!=snapshot['frame_count_end'] or snapshot['character']['head_id']!=p['head_id'] or not np.array_equal(np.asarray(snapshot['character']['shape_value_face'],dtype=np.float32),np.asarray(p['candidate_native59'],dtype=np.float32)):
        raise ValueError('Fresh guard snapshot has wrong head/native59 or changing frame')
    result={}
    for name in DEFAULT_BONES:
        matches=[r for r in snapshot['abmx_runtime']['bones'] if r['name']==name]
        if len(matches)!=1:
            raise ValueError('Ambiguous/missing current guard bone')
        row=matches[0]
        if not row['exists'] or serialized(modifier({k:row[k] for k in IDENTITY}))!=serialized(modifier(IDENTITY)):
            raise ValueError('Execution requires fresh actual identity modifier')
        wrapper=row['runtime_baseline']
        if wrapper['missing_fields'] or wrapper['assembly_mvid']!=p['expected_abmx_mvid'] or wrapper['frame_count']!=snapshot['frame_count']:
            raise ValueError('Incomplete/stale current cache/source')
        fields=cache_state(wrapper['fields'])
        clean(serialized(fields),name)
        expected=artifact['bone_diagnostics'][name]['persistent_history']
        if not persistent_equal(serialized(fields),expected):
            raise ValueError('Historical numerical radius/position differs; regenerate before execution')
        transforms=[r for r in snapshot['transforms'] if r['id']==wrapper['bone_transform_id'] and r['name']==name]
        if len(transforms)!=1:
            raise ValueError('Current snapshot bone ID/path mismatch')
        actual_baseline={key:serialized(fields[field]) for key,field in (('local_position','_posBaseline'),('local_rotation_xyzw','_rotBaseline'),('local_scale','_sclBaseline'))}
        predicted=artifact['expected_candidate_native_baselines'][name]
        checks={'actual_before':trs_errors(predicted,local_target(transforms[0])), 'baseline':trs_errors(predicted,actual_baseline)}
        if not all(r['passed'] for r in checks.values()):
            raise ValueError('Fresh native/cache TRS differs from declared predictor')
        result[name]={'current_bone_transform_id':wrapper['bone_transform_id'],'checks':checks,'persistent_history_float32_exact':True}
    return {'passed':True,'snapshot_frame':snapshot['frame_count'],'current_character_transform_id':snapshot['character']['transform_id'],
            'bones':result,'old_instance_ids_reused_as_proof':False,'future_or_runtime_stability_certified':False}
