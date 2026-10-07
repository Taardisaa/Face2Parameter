"""Native X-only writer source and static target binding versus actual Apply gaps."""
from pathlib import Path
import json
import sys
import UnityPy
from analyze import ROOT,HS2,read,binding


def execute(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    directory=HS2/'artifacts/infrastructure_live_20261005/native_radial_head2_v2'
    trace_path=directory/'candidate_trace.json';trace=read(trace_path)
    events=[e for e in trace['events'] if e['bone_name']=='cf_J_MouthMove']
    gaps=[]
    for a,b in zip(events,events[1:]):
        if a['after']['local_scale']==b['before']['local_scale']:continue
        changed=[i for i in range(3) if a['after']['local_scale'][i]!=b['before']['local_scale'][i]]
        gaps.append({'after_sequence':a['sequence'],'before_sequence':b['sequence'],
            'after_frame':a['completed_frame'],'before_frame':b['frame'],'after_scale':a['after']['local_scale'],
            'before_scale':b['before']['local_scale'],'changed_components':changed,
            'private_cache_unchanged':a['after']['cache']['fields']==b['before']['cache']['fields'],
            'after_position':a['after']['local_position'],'before_position':b['before']['local_position'],
            'after_rotation':a['after']['local_rotation_xyzw'],'before_rotation':b['before']['local_rotation_xyzw'],
            'bone_transform_id_same':a['after']['bone_transform_id']==b['before']['bone_transform_id']})
    bundle_path=Path('E:/HoneySelect2_ArcticFox/abdata/chara/oo_base.unity3d')
    env=UnityPy.load(str(bundle_path)); objects={o.path_id:o for o in env.objects}
    mono=[];errors=[]
    for obj in env.objects:
        if obj.type.name!='MonoBehaviour':continue
        try:
            tree=obj.read_typetree()
            script=objects.get(tree['m_Script']['m_PathID'])
            if script is None:continue
            script_tree=script.read_typetree()
            if script_tree.get('m_ClassName')!='CmpBoneHead':continue
            pointer=tree.get('targetEtc',{}).get('trfMouthAdjustWidth')
            record={'component_path_id':obj.path_id,'script':script_tree,'m_GameObject':tree['m_GameObject'],
                    'targetEtc':tree.get('targetEtc'),'width_target_pointer':pointer}
            if pointer and pointer['m_FileID']==0 and pointer['m_PathID'] in objects:
                target=objects[pointer['m_PathID']].read_typetree();record['width_target_transform']=target
                go=objects[target['m_GameObject']['m_PathID']].read_typetree();record['width_target_gameobject_name']=go['m_Name']
                chain=[]; cursor=target
                while True:
                    go=objects[cursor['m_GameObject']['m_PathID']].read_typetree();chain.append(go['m_Name'])
                    parent=cursor['m_Father']
                    if parent['m_PathID']==0 or parent['m_FileID']!=0:break
                    cursor=objects[parent['m_PathID']].read_typetree()
                record['target_asset_path']='/'.join(reversed(chain))
            mono.append(record)
        except (ValueError,KeyError,AttributeError) as exc:
            errors.append({'path_id':obj.path_id,'error':str(exc)})
    current=read(directory/'live_cases.json')
    result={'scope':'observed inter-Apply X-only boundary plus source/static-asset writer candidate; no writer instrumentation',
        'inputs':[binding(trace_path),binding(directory/'live_cases.json'),binding(bundle_path),binding(HS2/'tools/ChaControl.decompiled.txt')],
        'mouthmove_calls':len(events),'changed_scale_boundary_count':len(gaps),'gaps':gaps,
        'all_changes_x_only':all(g['changed_components']==[0] for g in gaps),
        'all_x_before_equals_one':all(g['before_scale'][0]==1 for g in gaps),
        'all_private_caches_equal':all(g['private_cache_unchanged'] for g in gaps),
        'static_CmpBoneHead_bindings':mono,'asset_decode_errors':errors,
        'source_writer':{'path':str(HS2/'tools/ChaControl.decompiled.txt'),'entry':'UpdateForce, line542',
            'method':'UpdateBlendShapeVoice, lines5996–6025',
            'condition':'loadEnd in UpdateForce; fileStatus.mouthAdjustWidth, objHeadBone != null, target transform != null',
            'target':'cmpBoneHead.targetEtc.trfMouthAdjustWidth',
            'value':'x=1 unless mouthCtrl exists, then GetAdjustWidthScale()',
            'operation':'SetLocalScaleX(x)','actual_runtime_target_binding_directly_read':False,
            'actual_runtime_mouth_adjust_width_directly_read':False,
            'actual_runtime_writer_identity_traced':False},
        'physical_candidate_cases':len(current['cases']),
        'do_not_merge_with_three_identity_only_datasets':True,
        'conditional_policy_only':'Anchoring all channels from cache can make next Apply independent of X-only entry scale; this does not remove the external writer or certify final-frame render ordering. Validate actual paired geometry and guard scope.',
        'source':binding(__file__)}
    (out/'mouth_writer.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'gaps':len(gaps),'x_only':result['all_changes_x_only'],'bindings':[(r.get('width_target_gameobject_name'),r.get('target_asset_path')) for r in mono],'decode_errors':len(errors)}))


if __name__=='__main__':execute(sys.argv[1])
