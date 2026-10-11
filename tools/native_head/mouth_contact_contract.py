"""Recover virtual closed-lip correspondences from native Close/Open frames."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy import sparse
from tools.geometry_quality.mesh_quality import self_intersections,triangle_intersection,plane_slice
from tools.native_head.mother_default_pose import dense_frame,controller_weights
from tools.native_head.mother_template_inputs import source_file,save_json
from tools.model_bridge.artifact import sha

def segment_points(a,b,epsilon=1e-8):
    na=np.cross(a[1]-a[0],a[2]-a[0]);nb=np.cross(b[1]-b[0],b[2]-b[0])
    na/=np.linalg.norm(na);nb/=np.linalg.norm(nb)
    direction=np.cross(na,nb);length=np.linalg.norm(direction)
    if length<=1e-10:return None
    direction/=length
    pa=plane_slice(a,(a-b[0])@nb,epsilon);pb=plane_slice(b,(b-a[0])@na,epsilon)
    if not len(pa) or not len(pb):return None
    origin=np.linalg.solve(np.vstack([na,nb,direction]),[na@a[0],nb@b[0],0.])
    sa=(pa-origin)@direction;sb=(pb-origin)@direction
    low=max(sa.min(),sb.min());high=min(sa.max(),sb.max())
    if high<low-epsilon:return None
    return origin+np.asarray([low,(low+high)/2,high])[:,None]*direction

def barycentric(point,tri):
    A=np.column_stack([tri[1]-tri[0],tri[2]-tri[0]])
    uv=np.linalg.lstsq(A,point-tri[0],rcond=None)[0]
    result=np.array([1-uv.sum(),*uv])
    if np.linalg.norm(result@tri-point)>1e-7 or result.min()<-1e-6:
        raise ValueError("Contact embedding outside source triangle")
    return result

def contact_constraints(closed,opened,faces,face_ids):
    sub=faces[face_ids];tri=closed[sub]
    report=self_intersections(closed,sub,tri,np.ones(len(sub),bool),1e-8,exclude_shared_vertex=True)
    records=[];rows=[];cols=[];values=[]
    for item in report['pairs']:
        local=item['triangles'];ids=face_ids[local];pair=faces[ids]
        p=closed[pair];normals=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        if normals[0]@normals[1]>=0:continue
        if triangle_intersection(*opened[pair],1e-8) is not None:continue
        points=segment_points(*p)
        if points is None:continue
        embeddings=[]
        for point in points:
            w=np.stack([barycentric(point,t) for t in p])
            row=len(embeddings)+len(records)*3
            for side,sign in ((0,1),(1,-1)):
                rows.extend([row]*3);cols.extend(pair[side]);values.extend(w[side]*sign)
            embeddings.append(w.tolist())
        records.append(dict(face_ids=ids.tolist(),source_classification=item,point_barycentric=embeddings,source_points=points.tolist()))
    C=sparse.coo_matrix((values,(rows,cols)),shape=(len(records)*3,len(closed))).tocsr()
    if records and np.max(np.abs(C@closed))>1e-7:raise ValueError('Virtual native contact does not coincide')
    return C,records,report

def build(inputs,mother,out,pattern=0):
    if out.exists():raise FileExistsError('Use fresh contact-contract output')
    r=json.loads((inputs/'receipt.json').read_text());m=json.loads((mother/'receipt.json').read_text())
    for row in (r['native_prefab_export'],r['regions'],m['default_reference']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Source contract changed')
    if m['inputs']['sha256']!=sha(inputs/'receipt.json') or m['candidate_arrays']['sha256']!=sha(mother/'o_head_candidate.npz'):
        raise ValueError('Native donor/mother geometry changed')
    audit_row=r['reference_audit']
    if sha(audit_row['path'])!=audit_row['sha256']:raise ValueError('Implementation source audit changed')
    audit=json.loads(Path(audit_row['path']).read_text(encoding='utf-8-sig'))
    for row in audit['assemblies']+audit['decompiled_sources']:
        if sha(row['path'])!=row['sha256']:raise ValueError('Installed implementation changed: '+row['path'])
    prefab=json.loads(Path(r['native_prefab_export']['path']).read_text())
    default=json.loads(Path(m['default_reference']['path']).read_text())
    regions=json.loads(Path(r['regions']['path']).read_text())
    head=next(x for x in prefab['renderers'] if x['mesh']=='o_head')
    ref=next(x for x in default['renderer_references'] if x['mesh']=='o_head')
    for row in (head['array_export'],head['blendshape_export']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Native mouth source changed')
    a=dict(np.load(head['array_export']['path'],allow_pickle=False));shapes=json.loads(Path(head['blendshape_export']['path']).read_text())
    ctrl=next(b['original_typetree'] for b in prefab['behaviours'] if b['class_name']=='FaceBlendShape')['MouthCtrl']
    obj=head['original_renderer_typetree']['m_GameObject']['m_PathID']
    def evaluate(rate):
        weights={int(k):v for k,v in ref['weights'].items()}
        weights.update(next(v for target,v in controller_weights(ctrl,rate,pattern) if target==obj))
        result=a['verts'].astype(float).copy()
        for channel,weight in weights.items():
            if channel!=-1 and weight:result+=dense_frame(shapes,channel,len(result))['vertex']*weight/100
        return result
    canonical=np.array([int(regions['graph_aliases'].get(str(i),i)) for i in range(len(a['verts']))])
    unique,inverse=np.unique(canonical,return_inverse=True);faces=inverse[a['faces']]
    closed,opened=evaluate(0)[unique],evaluate(1)[unique]
    lip=np.unique(inverse[m['regional_targets']['candidate_vermilion_vertex_ids']])
    face_ids=np.flatnonzero(np.isin(faces,lip).any(1))
    C,records,report=contact_constraints(closed,opened,faces,face_ids)
    names=a['bone_names'].tolist()
    up=names.index('cf_J_Mouthup');low=names.index('cf_J_MouthLow')
    upper=(a['bone_w']*(a['bone_idx']==up)).sum(1)[unique]
    lower=(a['bone_w']*(a['bone_idx']==low)).sum(1)[unique]
    all_records=records
    records=[]
    for index,row in enumerate(all_records):
        pair=faces[row['face_ids']]
        u,l=upper[pair].mean(1),lower[pair].mean(1)
        roles=['upper' if x>y else 'lower' if y>x else 'shared_or_unknown' for x,y in zip(u,l)]
        role='upper_lower' if set(roles)=={'upper','lower'} else 'same_'+roles[0] if roles[0]==roles[1] else 'unclassified'
        row.update(rig_layer_preferences=roles,rig_preference_pair=role,
            original_upper_lower_rig_weights=np.column_stack([u,l]).tolist())
        # Skin weights are mixed at corners and cannot certify anatomical
        # layers. Keep every actual source Close contact; no role-based filter.
        records.append(row)
    out.mkdir(parents=True);sparse.save_npz(out/'contact_matrix.npz',C)
    np.savez_compressed(out/'source_references.npz',closed=closed,opened=opened,unique_render_ids=unique,render_to_logical=inverse,faces=faces,lip_ids=lip)
    save_json(out/'contact_pairs.json',records)
    save_json(out/'all_source_contacts.json',all_records)
    save_json(out/'receipt.json',dict(format='native_mouth_close_open_contact_contract_v1',inputs=source_file(inputs/'receipt.json'),mother=source_file(mother/'receipt.json'),default_reference=m['default_reference'],original_arrays=head['array_export'],original_frames=head['blendshape_export'],pattern=pattern,matrix=source_file(out/'contact_matrix.npz'),pairs=source_file(out/'contact_pairs.json'),source_references=source_file(out/'source_references.npz'),original_mouth_pattern_table=ctrl['FBSTarget'],contact_pair_count=len(records),constraints=C.shape[0],source_formula='Original FBSBase endpoint weights, actual per-target indices, original sparse mesh frames',classification='Opposed nonadjacent native lip surfaces intersect/contact in original Close endpoint and separate in original Open endpoint; virtual points embedded on both original triangles',limitations=['Static source mesh frames before skinning/width adjustment','Coplanar contact pairs not converted to virtual segments','Contact support does not replace complete inner/outer lip anatomical validation'],game_mutated=False))
    receipt=json.loads((out/'receipt.json').read_text())
    receipt.update(code=source_file(Path(__file__)),source_audit=audit_row,
        all_source_contacts=source_file(out/'all_source_contacts.json'),
        rig_preferences_are_anatomical_certificates=False,
        classification='Every opposed source Close contact that separates in its original Open endpoint; includes intra-lip folds, never filtered by mixed bone preference')
    save_json(out/'receipt.json',receipt)
    print(json.dumps(dict(output=str(out.resolve()),contact_pairs=len(records),constraints=C.shape[0])))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs','mother','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pattern',type=int,default=0)
    a=p.parse_args();build(a.inputs,a.mother,a.out,a.pattern)
