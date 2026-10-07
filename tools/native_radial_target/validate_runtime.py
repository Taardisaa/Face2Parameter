"""Read-only ALL30 native-radius target, installed-call and surface audit.

Desired targets are generated from source-only native FK and logical declaration.
Candidate after states are acceptance objects, never target-generation inputs.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tools/geometry_quality'))
sys.path.insert(0,str(ROOT/'tools/base_comparison'))
from src.hs2_mesh_deform import HeadRig,_fk_world
from tools.abmx_replay.model import IDENTITY,modifier,serialized,replay_apply,require
from tools.abmx_replay.validate_trace import read,sha,trs_errors,local_only,validate
from tools.abmx_replay.validate_geometry import cache_correspondence,verify_pairs
from tools.abmx_multibone.geometry import select_cursor_predictions,parent_first_world,skin_vertices,verify_nonoverridden_ancestors
from tools.abmx_multibone.run import verify_capture_pair_signatures
from tools.abmx_multibone_quality.run import load_snapshot,identity_bound,evaluate_surface
from tools.unity_parity.geometry import analyze_snapshot,blendshape_delta,recorded_uniform_renderer_scale,rigid_alignment,vertex_errors
from tools.geometry_quality.cross_mesh import load_certified_head
from tools.native_radial_target.adapter import MODE,PROFILE,NAMES,native_numpy,patch_array
from tools.native_radial_target.compiler import prepare_execution_guard
from tools.abmx_stable_lowering.compiler import check_files
from tools.native_radial_target.observed_boundary import diagnostic_cursor

TOLERANCE=1e-5
TEMPORAL_RMS=.002
TEMPORAL_P95=.005


def save(path,data):
    with Path(path).open('x',encoding='utf-8') as handle:
        json.dump(serialized(data),handle,ensure_ascii=False,indent=2,allow_nan=False)


def vector(patch):
    p=modifier({k:patch[k] for k in IDENTITY})
    return np.r_[p['scale'],p['length'],p['position'],p['rotation']].astype(np.float32)


def certify(snapshot,descriptor):
    cert=analyze_snapshot(snapshot,normalized_tolerance=TOLERANCE)
    cert.update(snapshot_sha256=descriptor['sha256'],snapshot_path=descriptor['path'])
    meshes,skipped=load_certified_head(snapshot,descriptor['sha256'],cert)
    require(len({(m.name,m.path) for m in meshes})==len(meshes),'Duplicate visible renderer identity')
    return meshes,cert,skipped


def cached_mesh(rig,mesh):
    """Exact cached asset binding; duplicate names never overwrite renderer paths."""
    if mesh['mesh_name']=='o_head':
        cache_correspondence(rig,mesh)
        return rig
    path=Path(rig.data_dir)/'submeshes'/(mesh['mesh_name']+'.npz')
    require(path.is_file(),'Missing exact cached visible submesh '+mesh['renderer_path'])
    with np.load(path,allow_pickle=False) as d:
        sub=copy.copy(rig)
        for field,key in (('verts','verts'),('faces','faces'),('bone_idx','bone_idx'),('bone_w','bone_w'),('bindpose','bindpose')):
            setattr(sub,field,np.array(d[key]))
        sub.skin_bone_names=[rig.bones[str(pid)]['name'] for pid in d['skin_bone_pids']]
    cache_correspondence(sub,mesh)
    return sub


def native_world(rig,values,overrides):
    native=_fk_world(rig,values,None)
    local={pid:np.linalg.solve(native[rig.bones[pid]['parent']],native[pid])
           if rig.bones[pid]['parent'] in native else native[pid] for pid in rig._topo}
    return parent_first_world(rig,local,overrides)


def independent_target(artifact,contract,*,source_gaze_policy='reject'):
    """Independent NumPy native baseline; no actual candidate input accepted."""
    inputs=artifact['compiler_inputs'];history=inputs['history'];d=inputs['declared']
    check_files(artifact['source_bindings']['source_files'])
    require(d['semantic_mode']==MODE and artifact['candidate_actual_after_input'] is False,'Semantic mode/target leakage')
    snapshot,head,_=load_snapshot(history['geometry'],d['head_id'])
    trace=read(history['trace'])
    _,binding,replay=select_cursor_predictions(snapshot,trace,contract,required_names=NAMES)
    require(not replay['inter_call_boundaries'],'Source identity contains unidentified external writer')
    rig=HeadRig(d['head_id'],sampling_profile=PROFILE)
    native=native_numpy(rig,d['native59'])
    logical=patch_array(d['logical_patches']);targets={};checks={}
    for name,row in zip(NAMES,logical):
        n={k:np.asarray(v,dtype=np.float32) for k,v in native[name].items()}
        event=next(e for e in trace['events'] if e['bone_name']==name)
        check=trs_errors(serialized(n),local_only(event['before']))
        require(check['passed'],'Independent NumPy native differs from clean actual source '+name)
        # Target position is the new declared semantics, not historical radius.
        position=n['local_position']*row[3]+row[4:7]
        physical={'scale':row[:3].tolist(),'length':1.,'position':(position-n['local_position']).tolist(),'rotation':row[7:].tolist()}
        fields=copy.deepcopy(event['before']['cache']['fields'])
        for key,field in (('local_position','_posBaseline'),('local_rotation_xyzw','_rotBaseline'),('local_scale','_sclBaseline')):
            fields[field]=n[key].tolist()
        prediction=replay_apply(before=serialized(n),cache=fields,coordinate_modifiers=[physical],coordinate=0,
            additional_modifiers=[],bone_exists=True,rotation_excluded=False,is_during_h_scene=False)
        targets[name]=prediction['after']
        compiler_check=trs_errors(targets[name],artifact['desired_target_locals'][name])
        require(compiler_check['passed'],'Source-only independent target differs from compiler '+name)
        checks[name]={'native_vs_clean_source':check,'compiler_target_comparison':compiler_check,
            'independent_target':targets[name],'desired_position_float32':position.tolist(),
            'source_first_sequence':event['sequence'],'historical_radius_used':False}
    actual,certificate,skipped=certify(snapshot,history['geometry'])
    by_path={m['renderer_path']:m for m in snapshot['meshes']}
    require(source_gaze_policy in ('reject','frozen_source_locals'),'Explicit source gaze policy unsupported')
    gaze={}
    if source_gaze_policy=='frozen_source_locals':
        for name in ('cf_J_look_L','cf_J_look_R'):
            matches=[t for t in snapshot['transforms'] if t['name']==name]
            require(len(matches)==1 and name in rig.name2pid,'Source-only gaze nuisance coverage missing')
            gaze[name]=local_only(matches[0])
    native_fk=native_world(rig,d['native59'],gaze)
    target_fk=native_world(rig,d['native59'],{**gaze,**targets})
    targets_by_path={};sources_by_path={};source_errors={};factors={}
    h=next(m for m in actual if m.name=='o_head')
    delta,_=blendshape_delta(head)
    head_native=skin_vertices(rig,native_fk,delta)
    factor,_=recorded_uniform_renderer_scale(head,{t['id']:t for t in snapshot['transforms']})
    _,pose=rigid_alignment(head_native,h.vertices,unit_scale=factor)
    require(pose['rigid_errors']['max_normalized']<=TOLERANCE,'Source-only full native head does not match actual')
    # Freeze ONE pose from the earlier clean head. It is reused for every mesh
    # and candidate; no candidate vertex fit or per-mesh shape concealment.
    rotation=np.asarray(pose['rotation_row_vector']);translation=np.asarray(pose['translation'])
    for actual_mesh in actual:
        mesh=by_path[actual_mesh.path];sub=cached_mesh(rig,mesh)
        delta,active=blendshape_delta(mesh)
        s,scale_evidence=recorded_uniform_renderer_scale(mesh,{t['id']:t for t in snapshot['transforms']})
        source_prediction=skin_vertices(sub,native_fk,delta)*s@rotation+translation
        error=vertex_errors(actual_mesh.vertices,source_prediction)
        targets_by_path[actual_mesh.path]=skin_vertices(sub,target_fk,delta)*s@rotation+translation
        sources_by_path[actual_mesh.path]=mesh;factors[actual_mesh.path]=scale_evidence
        source_errors[actual_mesh.path]={'mesh_name':mesh['mesh_name'],'source_error':error,'active_source_blendshapes':active,
            'native_source_model_passed':error['max_normalized']<=TOLERANCE,
            'target_scope_supported':error['max_normalized']<=TOLERANCE}
    return {'rig':rig,'snapshot':snapshot,'targets':targets,'vertices':targets_by_path,'source_meshes':sources_by_path,'source_gaze_locals':gaze,
        'evidence':{'source_cursor':binding,'source_lbs_certificate':certificate,'skipped_renderers':skipped,
            'independent_native_target_checks':checks,'source_mesh_checks':source_errors,
            'source_pose_from_identity_head':pose,'source_recorded_uniform_scale':factors,
            'source_gaze_policy':source_gaze_policy,'source_only_gaze_nuisance':gaze,
            'source_gaze_is_not_native59_predicted_or_runtime_writer_identified':True,
            'candidate_actual_after_or_expression_used_for_target':False,
            'candidate_fitted_rotation_translation_scale_affine':False}}


def verify_physical_trace(trace,artifact,source_trace,contract):
    report=validate(trace,contract)
    require(report['passed'],'Actual ALL30 calls did not pass installed replay')
    require(set(trace['metadata']['requested_names'])==set(NAMES),'Incomplete all30 observation filter')
    expected={r['name']:r for r in artifact['executed_patches']}
    starts={};totals={}
    for name in NAMES:
        source=next(e for e in source_trace['events'] if e['bone_name']==name)
        events=[e for e in trace['events'] if e['bone_name']==name]
        require(events,'Unobserved required bone '+name)
        indices=[i for i,e in enumerate(events) if not np.array_equal(vector(e['resolved_modifier']),vector(IDENTITY))]
        require(indices and indices[0]>0,'Need observed identity before constant physical patch '+name)
        start=indices[0];starts[name]=events[start]['sequence'];totals[name]=len(events)-start
        identity=(source['before']['bone_transform_id'],source['modifier_instance_identity'])
        for i,event in enumerate(events):
            require((event['before']['bone_transform_id'],event['modifier_instance_identity'])==identity,'Source/candidate bone/modifier changed '+name)
            require(not event['additional_modifiers'] and not event['no_rotation_excluded'] and not event['is_during_h_scene'] and event['coordinate_specific'] is False,'Unsupported actual branch '+name)
            require(np.array_equal(vector(event['resolved_modifier']),vector(IDENTITY if i<start else expected[name])),'Actual physical patch changed '+name)
            for phase in ('before','after'):
                fields=event[phase]['cache']['fields'];original=source['before']['cache']['fields']
                for key in ('_lenBaseline','_positionBaseline'):
                    require(np.array_equal(np.asarray(fields[key],dtype=np.float32),np.asarray(original[key],dtype=np.float32)),'Actual persistent radius/history changed '+name)
    return report,starts,totals


def verify_local_restoration(manifest):
    """Recompute all30 terminal local restoration from actual paired geometry."""
    snapshots=[];descriptors=[]
    for key in ('original_pose','restored_pose'):
        descriptor=manifest[key]['paired_geometry']
        snapshot=read(descriptor['path'])
        require(sha(descriptor['path'])==descriptor['sha256'],'Local restoration geometry changed')
        require(snapshot['frame_count']==snapshot['frame_count_end']==descriptor['frame_count'],'Local restoration geometry frame unstable')
        verify_pairs({'capture':manifest[key],'geometry':descriptor},snapshot,descriptor['sha256'])
        verify_capture_pair_signatures({'capture':manifest[key]},snapshot)
        snapshots.append(snapshot);descriptors.append(descriptor)
    a,b=snapshots
    require(a['character']['head_id']==b['character']['head_id'] and
        a['character']['transform_id']==b['character']['transform_id'],'Original/restored head/actor differ')
    checks={}
    for name in NAMES:
        before=[t for t in a['transforms'] if t['name']==name]
        after=[t for t in b['transforms'] if t['name']==name]
        require(len(before)==len(after)==1,'Incomplete actual terminal local restoration coverage')
        checks[name]=trs_errors(local_only(before[0]),local_only(after[0]))
    return {'passed':all(c['passed'] for c in checks.values()),'all30_checks':checks,
        'original_geometry_sha256':descriptors[0]['sha256'],'restored_geometry_sha256':descriptors[1]['sha256'],
        'scope':'Actual ALL30 local TRS plus independent public restoration; not all bones/world/material/private-history restoration'}


def audit(manifest_path,contract_path,out,*,source_gaze_policy='reject'):
    out=Path(out);require(not out.exists(),'New report directory required');out.mkdir(parents=True)
    manifest=read(manifest_path);contract=read(contract_path)
    result={'schema_version':1,'semantic_mode':MODE,'manifest_path':str(Path(manifest_path).resolve()),
        'manifest_sha256':sha(manifest_path),'implementation_sha256':sha(__file__),'passed':False,
        'normalization_tolerance':TOLERANCE,'temporal_rms_gate':TEMPORAL_RMS,'temporal_p95_gate':TEMPORAL_P95,
        'anatomy_likeness_or_quality_certified':False,'arbitrary_candidate_or_infinite_time_stability_certified':False}
    try:
        require(all(manifest.get(k) is True for k in ('state_restored','expression_restored','bone_restored')),'Missing public restoration')
        require(manifest['before']==manifest['after'] and manifest['expression_before']==manifest['expression_after'] and
                manifest['modifiers_before']==manifest['modifiers_after'] and manifest.get('restore_failures')==[],'Actual restoration evidence differs')
        result['all30_local_restoration']=verify_local_restoration(manifest)
        require(result['all30_local_restoration']['passed'],'Actual ALL30 terminal local restoration failed')
        require(len(manifest['cases'])==1,'One immutable actual base per manifest required')
        descriptor=manifest['compiler_artifact'];require(sha(descriptor['path'])==descriptor['sha256'],'Compiler artifact changed')
        artifact=read(descriptor['path'])
        context=prepare_execution_guard(artifact,contract_path,artifact_path=descriptor['path'],artifact_sha256=descriptor['sha256'])
        case=manifest['cases'][0];identity=read(case['identity_capture']['paired_geometry']['path'])
        require(sha(case['identity_capture']['paired_geometry']['path'])==case['identity_capture']['paired_geometry']['sha256'],'Identity geometry changed')
        result['fresh_identity_guard_posthoc']=context.verify_snapshot(identity,case['trace_start'])
        require(sha(case['trace'])==case['trace_sha256'],'Actual candidate trace changed')
        trace=read(case['trace']);source_trace=read(case['source_history']['trace'])
        require(source_trace['metadata']['stopped_frame']<trace['metadata']['started_frame'],'Source/candidate lifecycle overlap')
        report,starts,totals=verify_physical_trace(trace,artifact,source_trace,contract)
        save(out/'actual_call_replay.json',report);result['actual_calls']={'total':len(trace['events']),'passed':report['passed_call_count'],'active_per_bone_totals':totals}
        result['unsupported_external_boundaries']=report['inter_call_boundaries']
        result['external_writers_identified']=False
        target=independent_target(artifact,contract,source_gaze_policy=source_gaze_policy);save(out/'independent_target.json',target['evidence'])
        result['source_only_target']=target['evidence'];windows={};rows=[]
        for window in case['windows']:
            snapshot,mesh,_=load_snapshot(window['geometry'],case['head_id'])
            verify_pairs(window,snapshot,window['geometry']['sha256']);verify_capture_pair_signatures(window,snapshot)
            require(snapshot['character']['expression']==target['snapshot']['character']['expression'],'Candidate expression differs from source target')
            gaze_checks={}
            for name,expected in target['source_gaze_locals'].items():
                candidates=[t for t in snapshot['transforms'] if t['name']==name]
                require(len(candidates)==1,'Incomplete actual gaze nuisance observation')
                gaze_checks[name]=trs_errors(expected,local_only(candidates[0]))
            require(np.array_equal(np.asarray(snapshot['character']['shape_value_face'],dtype=np.float32),np.asarray(artifact['source_bindings']['native59'],dtype=np.float32)),'Candidate native59 changed')
            try:
                predicted,binding,_=select_cursor_predictions(snapshot,trace,contract,required_names=NAMES)
                binding['passed']=True
            except ValueError as error:
                predicted,binding=diagnostic_cursor(snapshot,trace,report,NAMES)
                binding['strict_selector_rejection']=str(error)
            checks={n:trs_errors(target['targets'][n],predicted[n]) for n in NAMES}
            require(all(v['passed'] for v in checks.values()),'Desired ALL30 locals differ from replay snapshot')
            counts={n:sum(e['bone_name']==n and starts[n]<=e['sequence']<=binding['cursor']['last_completed_sequence'] for e in trace['events']) for n in NAMES}
            require(all(c>0 for c in counts.values()),'Missing actual active calls at capture')
            actual,certificate,skipped=certify(snapshot,window['geometry']);save(out/(window['window']+'_lbs.json'),certificate)
            require(set(m.path for m in actual)==set(target['vertices']),'Visible renderer set changed')
            ancestors=verify_nonoverridden_ancestors(snapshot,target['rig'],{**target['source_gaze_locals'],**target['targets']})
            comparisons={};canonical={};by_path={m['renderer_path']:m for m in snapshot['meshes']}
            source_head=next(m for m in target['source_meshes'].values() if m['mesh_name']=='o_head')
            current_head=next(m for m in snapshot['meshes'] if m['mesh_name']=='o_head' and m['enabled'] and m['active_in_hierarchy'])
            source_rotation=Rotation.from_quat(source_head['renderer_rotation_xyzw']).as_matrix()
            current_rotation=Rotation.from_quat(current_head['renderer_rotation_xyzw']).as_matrix()
            source_position=np.asarray(source_head['renderer_position']);current_position=np.asarray(current_head['renderer_position'])
            row_rotation=source_rotation@current_rotation.T
            row_translation=current_position-source_position@row_rotation
            for live in actual:
                current=by_path[live.path];source=target['source_meshes'][live.path];identity_bound(current,source)
                require([(b.get('name'),b['current_weight']) for b in current.get('blendshapes',[])]==[(b.get('name'),b['current_weight']) for b in source.get('blendshapes',[])],'Candidate expression blendshape weights changed')
                # Remove only the observed shared proper head pose. No candidate
                # vertex fitting; one matrix applies to every visible renderer.
                sm=np.asarray(source['renderer_local_to_world']).reshape(4,4)
                expected_matrix=sm.copy();expected_matrix[:3,:3]=row_rotation.T@sm[:3,:3]
                expected_matrix[:3,3]=sm[:3,3]@row_rotation+row_translation
                require(np.allclose(current['renderer_local_to_world'],expected_matrix.reshape(-1),atol=2e-6,rtol=0),
                        'Renderer relative pose/scale cannot be explained by shared recorded proper head pose')
                transported_target=target['vertices'][live.path]@row_rotation+row_translation
                error=vertex_errors(live.vertices,transported_target)
                source_supported=target['evidence']['source_mesh_checks'][live.path]['target_scope_supported']
                gaze_unchanged=all(c['passed'] for c in gaze_checks.values())
                if live.name in ('o_eyebase_L','o_eyebase_R') and not gaze_unchanged:source_supported=False
                comparisons[live.path]={'mesh_name':live.name,'errors':error,
                    'source_target_scope_supported':source_supported,
                    'passed':source_supported and error['max_normalized']<=TOLERANCE}
                rotation=Rotation.from_quat(current['renderer_rotation_xyzw']).as_matrix()
                canonical[live.path]={'vertices':(live.vertices-np.asarray(current['renderer_position']))@rotation,'faces':live.faces}
            rows.append({'window':window['window'],'geometry_sha256':window['geometry']['sha256'],'passed':binding['passed'] and all(c['passed'] for c in comparisons.values()),
                'all30_local_target_errors':checks,'exact_cursor_binding':binding,'observed_active_counts':counts,
                'full_mesh_comparisons':comparisons,'nonoverridden_ancestors':ancestors,'skipped_renderers':skipped,
                'frozen_source_gaze_vs_actual':gaze_checks,
                'recorded_shared_head_pose_removal':{'rotation_row_vector':row_rotation.tolist(),'translation':row_translation.tolist(),
                    'source_head_position':source_position.tolist(),'actual_head_position':current_position.tolist(),
                    'vertex_fitted':False,'scale_fitted_or_removed':False},
                'target_uses_candidate_after':False,'candidate_pose_or_scale_fitted':False,
                'unknown_boundary_unconditional_execution_certified':False})
            require(window['window'] not in windows,'Duplicate window');windows[window['window']]=canonical
        require(set(windows)=={'early','late','far60'},'Need all three actual windows')
        temporal=[]
        for a,b in (('early','late'),('late','far60'),('early','far60')):
            for path in windows[a]:
                left,right=windows[a][path],windows[b][path];require(np.array_equal(left['faces'],right['faces']),'Temporal topology changed')
                surface=evaluate_surface(right['vertices'],right['faces'],left['vertices'],left['faces'],count=4096,seed=7381)
                delta=right['vertices']-left['vertices']
                temporal.append({'earlier':a,'later':b,'renderer_path':path,'surface':surface,
                    'passed':surface['symmetric']['rms']<=TEMPORAL_RMS and surface['symmetric']['p95']<=TEMPORAL_P95,
                    'corresponding_vertex_rms':float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),
                    'corresponding_vertex_max':float(np.linalg.norm(delta,axis=1).max()),'fitted_scale_or_pose':False})
        result.update(windows=rows,temporal=temporal,head_id=case['head_id'],
            o_head_target_passed_windows=sum(all(c['passed'] for c in r['full_mesh_comparisons'].values() if c['mesh_name']=='o_head') for r in rows),
            full_visible_mesh_scope_supported=all(c['target_scope_supported'] for c in target['evidence']['source_mesh_checks'].values()),
            passed=not report['inter_call_boundaries'] and all(r['passed'] for r in rows) and all(r['passed'] for r in temporal))
        if report['inter_call_boundaries']:result['strict_execution_rejection']='Unidentified inter-call external writer boundaries retained; numerical target agreement is diagnostic only'
    except (ValueError,KeyError,TypeError,OSError,StopIteration) as error:
        result['rejection']=str(error)
    save(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--contract',required=True,type=Path);p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--source-gaze-policy',choices=('reject','frozen_source_locals'),default='reject');a=p.parse_args()
    report=audit(a.manifest,a.contract,a.out_dir,source_gaze_policy=a.source_gaze_policy)
    print(json.dumps({'passed':report['passed'],'rejection':report.get('rejection'),'report':str(a.out_dir/'summary.json')}))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
