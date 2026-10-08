"""Author a locally fitted neck, preserving the conservative cut and entire face.

This is new asset authoring, not an approximation of HS2's deformation rules.
The body opening is recovered from literal skin/position aliases. Its live
vertices are transformed through the captured source frame. No body is edited.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from tools.model_bridge.artifact import sha
from tools.model_bridge.game_import import request
from tools.model_bridge.rig import encode
from tools.native_head.lower_trim import decode
from tools.native_head.neck_geometry import original_neck_boundary, ordered_loops, radial_samples, corresponding_lower_row


def smooth_displacement(v, f, fixed, values):
    """Biharmonic displacement with literal Dirichlet constraints.

    Uniform adjacency avoids negative cotangent weights on the coarse neck.
    Squared Laplacian penalizes a sudden change at the protected boundary.
    """
    edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
    rows=np.r_[edges[:,0],edges[:,1]];cols=np.r_[edges[:,1],edges[:,0]]
    adjacency=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(v),len(v))).tocsr()
    lap=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    energy=lap.T@lap
    free=np.flatnonzero(~fixed);bound=np.flatnonzero(fixed)
    delta=values.copy()
    if len(free):delta[free]=spsolve(energy[free][:,free],-energy[free][:,bound]@delta[bound])
    if not np.isfinite(delta).all():raise ValueError('Unconstrained neck component')
    return v+delta


def fit(payload, native, placement, lock_height=.15):
    if payload.get('trim',{}).get('format')!='flame_bottom_only_trim_v1':
        raise ValueError('Start from the conservative bottom-only source, never the broad anatomical cut')
    scale=float(placement['local_scale'][0]);offset=np.asarray(placement['local_position'])
    if not np.array_equal(placement['local_scale'],[scale]*3):raise ValueError('Uniform placement required')
    raw=np.asarray(payload['vertices'],np.float32);v=raw.astype(float)*scale+offset
    f=np.asarray(payload['triangles'],int).reshape(-1,3)
    mesh=native['meshes'][0]
    if mesh['mesh_name']!='o_body_cf' or mesh['renderer_lossy_scale']!=[1.,1.,1.]:
        raise ValueError('Explicit unscaled native body required')
    body_ring,_=original_neck_boundary(mesh['source'],mesh['bone_names'])
    source_frames=[x for x in native['transforms'] if x['name']=='HS2_SourceModelHead']
    if len(source_frames)!=1:raise ValueError('One source transform required')
    # Installed Unity 2018 BakeMesh includes renderer scale. For this supported
    # unit body frame its captured local_to_world is the literal conversion.
    body_world=np.c_[mesh['baked']['vertices'],np.ones(mesh['vertex_count'])]@np.asarray(mesh['renderer_local_to_world']).reshape(4,4).T
    source_raw=body_world@np.asarray(source_frames[0]['world_to_local']).reshape(4,4).T
    body=source_raw[body_ring,:3]*scale+offset
    loops=ordered_loops(f)
    if len(loops)!=2:raise ValueError('Expected neck opening and unchanged mouth opening')
    ring=np.asarray(min(loops,key=lambda x:v[x,1].mean()))
    center=(body.min(0)+body.max(0))/2
    angle=lambda xyz:np.arctan2(xyz[:,0]-center[0],xyz[:,2]-center[2])
    theta=angle(v[ring]);steps=np.arctan2(np.sin(np.roll(theta,-1)-theta),np.cos(np.roll(theta,-1)-theta))
    sense=np.sign(steps.sum())
    if not np.all(steps*sense>0):raise ValueError('Source lower rim must be ordered about the neck')
    relative=lambda x:((x-theta[0])*sense)%(2*np.pi)
    outer_angles=relative(theta);outer_angles[0]=0
    # Extend the actual body's adjacent surface upward a short distance. This
    # creates a real tangent-aligned lower row, rather than forcing a large neck
    # compression to terminate abruptly at the seam itself.
    body_all=source_raw[:,:3]*scale+offset
    body_faces=np.asarray(mesh['source']['triangles']).reshape(-1,3)
    outgoing=[]
    for idx in body_ring:
        near=np.unique(body_faces[np.any(body_faces==idx,axis=1)])
        near=near[~np.isin(near,body_ring)]
        outgoing.append(body_all[near].mean(0)-body_all[idx])
    outgoing=np.asarray(outgoing)
    if np.any(outgoing[:,1]>=-1e-6):raise ValueError('Native one-ring does not point down the neck')
    upper_body=body-outgoing*(.12/np.abs(outgoing[:,1,None]))
    target,_=radial_samples(upper_body,upper_body,theta,center)
    masks_info=payload['surface']['components']['source_mask'];mask_path=Path(masks_info['path'])
    if sha(mask_path)!=masks_info['sha256']:raise ValueError('Source masks changed')
    masks=pickle.loads(mask_path.read_bytes(),encoding='latin1')
    protected_original=np.unique(np.concatenate([masks[k] for k in ['face','left_ear','right_ear','eye_region']]))
    original_ids=np.asarray(payload['trim']['original_vertex_ids'])
    head_ids=payload['surface']['components']['components'][0]['canonical_vertex_ids']
    head=np.zeros(len(v),bool);head[head_ids]=True
    protected=np.isin(original_ids,protected_original)|~head|(v[:,1]>=lock_height)
    # Protect incident triangles as well as named vertices. The previous mask
    # crop preserved chin vertices while destroying adjacent chin faces. Here
    # the entire first ring around the face/cranial locks stays literal.
    protected[np.unique(f[np.any(protected[f],axis=1)])]=True
    if protected[ring].any():raise ValueError('Neck rim touches protected head')
    values=np.zeros_like(v);values[ring]=target-v[ring]
    fixed=protected.copy();fixed[ring]=True
    fitted=smooth_displacement(v,f,fixed,values)
    fitted[protected]=v[protected];fitted[ring]=target
    if not np.array_equal(fitted[protected],v[protected]):raise ValueError('Protected head moved')
    # Preserve the original low cut. Insert every native corner into its matching
    # source boundary edge; otherwise a straight source edge shortcuts the body
    # polygon and leaves a gap even when its endpoints match.
    additions=[];splits={};native_to_head=[]
    for point,t in zip(upper_body,relative(angle(upper_body))):
        nearest=np.argmin(np.abs(np.arctan2(np.sin(theta-angle(point[None])[0]),np.cos(theta-angle(point[None])[0]))))
        if np.linalg.norm(fitted[ring[nearest]]-point)<1e-7:
            native_to_head.append(int(ring[nearest]));continue
        j=int(np.searchsorted(outer_angles,t,side='right')-1)%len(ring)
        a,b=int(ring[j]),int(ring[(j+1)%len(ring)])
        end=outer_angles[j+1] if j+1<len(ring) else 2*np.pi
        fraction=(t-outer_angles[j])/(end-outer_angles[j])
        idx=len(v)+len(additions);additions.append((a,b,float(fraction),point))
        splits.setdefault((a,b),[]).append((fraction,idx));native_to_head.append(idx)
    render_f=np.asarray(payload['surface']['render_triangles']).reshape(-1,3)
    uv=np.asarray(payload['surface']['render_uv'])[render_f]
    new_faces=[];new_uv=[];parents=[]
    for parent,(tri,tex) in enumerate(zip(f,uv)):
        selected=[(j,splits[(int(tri[j]),int(tri[(j+1)%3]))]) for j in range(3)
                  if (int(tri[j]),int(tri[(j+1)%3])) in splits]
        if not selected:new_faces.append(tri.tolist());new_uv.append(tex.tolist());parents.append(parent);continue
        if len(selected)!=1:raise ValueError('Multiple split boundary edges in one source triangle')
        j,items=selected[0];a,b,c=map(int,tri[[j,(j+1)%3,(j+2)%3]])
        entries=[(0.,a),*sorted(items),(1.,b)]
        for (ta,ia),(tb,ib) in zip(entries,entries[1:]):
            new_faces.append([ia,ib,c]);new_uv.append([(1-ta)*tex[j]+ta*tex[(j+1)%3],
                (1-tb)*tex[j]+tb*tex[(j+1)%3],tex[(j+2)%3]]);parents.append(parent)
    # Top boundary now follows the literal upward extension of every native
    # edge. Add a narrow integrated annulus down to every original body corner.
    top=np.vstack([fitted,*[x[3][None] for x in additions]])
    top_faces=np.asarray(new_faces,int)
    matching=[x for x in ordered_loops(top_faces) if int(ring[0]) in x]
    if len(matching)!=1:raise ValueError('Fitted neck boundary lost')
    top_ring=np.asarray(matching[0])
    bottom=corresponding_lower_row(top[top_ring],upper_body,body)
    bottom_corner_ids=[]
    for a,point in zip(top_ring,bottom):
        # Added vertices are explicitly authored from this current body's rim.
        # Their original FLAME buffers inherit the corresponding top corner.
        idx=len(v)+len(additions);additions.append((int(a),int(a),0.,point))
        bottom_corner_ids.append(idx)
    # Seam attributes/rig need recursive interpolation for the top corners
    # added earlier; retain the same parent chain rather than guessing weights.
    uv_lookup={}
    for tri,tex in zip(new_faces,new_uv):
        for a,t in zip(tri,tex):uv_lookup.setdefault(int(a),np.asarray(t))
    for j,(a,b) in enumerate(zip(top_ring,np.roll(top_ring,-1))):
        c=bottom_corner_ids[j];d=bottom_corner_ids[(j+1)%len(top_ring)]
        new_faces.extend([[int(b),int(a),c],[int(b),c,d]])
        new_uv.extend([[uv_lookup[int(b)],uv_lookup[int(a)],uv_lookup[int(a)]],
                       [uv_lookup[int(b)],uv_lookup[int(a)],uv_lookup[int(b)]]])
        parents.extend([-1,-1])
    authored=np.vstack([fitted,*[x[3][None] for x in additions]])
    native_to_head=[bottom_corner_ids[int(np.where(top_ring==idx)[0][0])] for idx in native_to_head]
    if np.linalg.norm(authored[native_to_head]-body,axis=1).max()>1e-7:
        raise ValueError('Native seam corners are not reproduced literally')
    faces=np.asarray(new_faces,int)
    xyz=authored[faces];areas=np.linalg.norm(np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0]),axis=1)
    if areas.min()<1e-10:raise ValueError('Authored neck has a degenerate triangle')
    if len(ordered_loops(faces))!=2:raise ValueError('Boundary splitting changed head topology')
    result=copy.deepcopy(payload);result['vertices']=((authored-offset)/scale).astype(np.float32).tolist()
    # Write locked raw vertices literally, without roundtrip numerical changes.
    for i in np.flatnonzero(protected):result['vertices'][int(i)]=payload['vertices'][int(i)]
    result['triangles']=faces.ravel().tolist()
    surface=result['surface'];surface['render_to_canonical']=faces.ravel().tolist()
    surface['render_triangles']=list(range(faces.size));surface['render_uv']=np.asarray(new_uv).reshape(-1,2).tolist()
    surface['components']['components'][0]['canonical_vertex_ids']+=list(range(len(v),len(authored)))
    labels=np.full(len(authored),-1,int)
    for c,part in enumerate(surface['components']['components']):labels[part['canonical_vertex_ids']]=c
    face_labels=labels[faces[:,0]];starts=np.r_[0,np.flatnonzero(np.diff(face_labels))+1];ends=np.r_[starts[1:],len(faces)]
    surface['components']['material_runs']=[dict(first_triangle=int(a),triangle_count=int(b-a),component=int(face_labels[a])) for a,b in zip(starts,ends)]
    # The source rig is preserved as provenance, but this geometry preview has
    # authored neck offsets. Neutral MICA only; posed FLAME/native retargeting is
    # outside this asset-authoring step.
    rig=result['rig']
    if np.any(payload['rig']['original_pose']):raise ValueError('Neutral source rig required')
    def extended(array):
        rows=list(array)
        for a,b,t,_ in additions:rows.append((1-t)*rows[a]+t*rows[b])
        return np.asarray(rows,dtype=array.dtype)
    shaped=decode(payload['rig']['v_shaped']);post=decode(payload['rig']['post_skin_offsets'])
    rig['v_shaped']=encode(extended(shaped));rig['weights']=encode(extended(decode(payload['rig']['weights'])))
    post=extended(post);post+=np.asarray(result['vertices'],np.float32)-extended(raw)
    rig['post_skin_offsets']=encode(post)
    dirs=decode(payload['rig']['posedirs']).reshape(36,len(v),3)
    rig['posedirs']=encode(extended(dirs.transpose(1,0,2)).transpose(1,0,2).reshape(36,-1))
    rig['policy']+='; explicit authored neck post-skin offsets and seam subdivisions; neutral display only, no posed seam certification'
    result['neck_fit']=dict(format='hs2_local_neck_fit_v1',protected_indices=np.flatnonzero(protected).tolist(),
        lock_height_head_frame=lock_height,protected_positions_literal=True,
        face_chin_ear_eye_positions_literal=True,additional_source_faces_deleted=0,
        source_faces_subdivided=len(splits),native_corner_head_indices=native_to_head,
        neck_ring_indices=bottom_corner_ids,integrated_tangent_annulus_height=.12,
        native_body_ring_indices=body_ring,native_body_vertices_modified=0,
        parent_face_indices=parents,added_edge_vertices=[dict(a=a,b=b,t=t) for a,b,t,_ in additions],
        changed_original_vertices=np.flatnonzero(np.any(fitted!=v,axis=1)).tolist(),
        method='Constrained biharmonic neck displacement, literal face/ear/cranial locks; actual body one-ring extension and exact seam corner insertion',
        scope='Neutral current-body geometry preview; native skin binding and seam normals require asset integration')
    result['shape_policy']='Original conservative crop retained; local neck authored to current native body opening, entire face and upper skull literal'
    report=dict(original_triangles=len(payload['triangles'])//3,result_triangles=len(faces),
        original_faces_deleted_total=len(payload['trim']['removed_original_face_ids']),
        additional_source_faces_deleted=0,face_positions_literal=True,head_above_lock_literal=True,
        native_body_unchanged=True,native_corners_present=True,native_seam_error_head_units=float(np.linalg.norm(authored[native_to_head]-body,axis=1).max()),
        changed_neck_vertices=len(result['neck_fit']['changed_original_vertices']),new_seam_vertices=len(additions),
        source_geometry_accuracy_certified=False,posed_seam_certified=False,native_base_asset_updated=False)
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','native','state','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--lock-height',type=float,default=.15);p.add_argument('--show',action='store_true')
    p.add_argument('--base',default='http://127.0.0.1:43127');args=p.parse_args()
    payload,native,state=[json.loads(x.read_text(encoding='utf-8')) for x in [args.source,args.native,args.state]]
    if sha(args.source)!=state['artifact_sha256']:raise ValueError('Captured source state does not match input')
    result,report=fit(payload,native,state,args.lock_height)
    args.out.mkdir(parents=True,exist_ok=False)
    result['neck_fit']['provenance']=[dict(path=str(x.resolve()),sha256=sha(x)) for x in [args.source,args.native,args.state]]
    path=args.out/'source_neck_fitted.json';path.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    obj=args.out/'neck_fitted.obj';obj.write_text('\n'.join(['# Locally authored neck; protected face/skull literal']+
        ['v '+' '.join(repr(x) for x in row) for row in result['vertices']]+
        ['f '+' '.join(str(x+1) for x in row) for row in np.asarray(result['triangles']).reshape(-1,3)])+'\n')
    if args.show:
        live=request(args.base,'POST',dict(path=str(path.resolve()),sha256=sha(path),scale=state['local_scale'][0],translation=state['local_position']))
        if not np.array_equal(np.asarray(live['canonical_vertices'],np.float32),np.asarray(result['vertices'],np.float32)) or not np.array_equal(live['canonical_triangles'],result['triangles']):
            raise ValueError('Live canonical geometry differs from authored mesh')
        report.update(runtime_loaded=True,game_restarted=False,runtime_arrays_exact=True)
    report['artifact']=dict(path=str(path.resolve()),sha256=sha(path))
    (args.out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
