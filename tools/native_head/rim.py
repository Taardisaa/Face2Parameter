"""Posterior contraction with hard face/height locks and an integrated join.

The cut crosses the chin. Never move face vertices onto the body rim. A newly
authored under-jaw strip connects them in the head asset, with no restored source
neck faces or additional runtime renderers.
"""
from __future__ import annotations

import heapq
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def native_contour(native, descriptor):
    mesh=native['meshes'][0];data=mesh['source']
    ids=np.asarray(descriptor['native_ring_cut_indices'],int)
    bone=mesh['bone_names'].index('cf_J_Head_s')
    indices=np.asarray(data['bone_indices']);weights=np.asarray(data['bone_weights'])
    if not np.array_equal((weights[ids]*(indices[ids]==bone)).sum(1),np.ones(len(ids))):
        raise ValueError('Native rim needs literal cf_J_Head_s support')
    bind=np.asarray(data['bindposes']).reshape(-1,4,4)[bone]
    xyz=np.asarray(data['vertices'])[ids]
    points=(np.c_[xyz,np.ones(len(xyz))]@bind.T)[:,:3]
    normals=np.asarray(data['normals'])[ids]@np.linalg.inv(bind[:3,:3])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    return points,normals


def paired_contour(source, points, normals):
    """Keep native contour corners; pair by azimuth rather than 3D distance."""
    count=len(source)
    if count<len(points):raise ValueError('Too few retained corners for native contour')
    subdivisions=np.ones(len(points),int)
    lengths=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    for _ in range(count-len(points)):subdivisions[np.argmax(lengths/subdivisions)]+=1
    target=[];ns=[]
    for i,n in enumerate(subdivisions):
        for t in np.arange(n)/n:
            target.append((1-t)*points[i]+t*points[(i+1)%len(points)])
            ns.append((1-t)*normals[i]+t*normals[(i+1)%len(points)])
    target=np.asarray(target);ns=np.asarray(ns)
    def area(v):return np.sum(v[:,0]*np.roll(v[:,2],-1)-v[:,2]*np.roll(v[:,0],-1))
    if area(target)*area(source)<0:target=target[::-1];ns=ns[::-1]
    center=(points.min(0)+points.max(0))/2
    angles=lambda v:np.arctan2(v[:,0]-center[0],v[:,2]-center[2])
    a=angles(source);b=angles(target)
    def cost(shift):
        d=a-np.roll(b,shift);return np.sum(np.arctan2(np.sin(d),np.cos(d))**2)
    shift=min(range(count),key=cost)
    target=np.roll(target,shift,axis=0);ns=np.roll(ns,shift,axis=0)
    ns/=np.linalg.norm(ns,axis=1,keepdims=True)
    return target,ns,center


def conform(vertices, faces, ring, native, descriptor, bandwidth, protected):
    vertices=np.asarray(vertices,float);ring=np.asarray(ring,int);faces=np.asarray(faces,int)
    protected=np.asarray(protected,bool)
    if protected.shape!=(len(vertices),) or bandwidth<=0:raise ValueError('Face locks and positive band required')
    points,normals=native_contour(native,descriptor)
    target,target_normals,center=paired_contour(vertices[ring],points,normals)
    neighbours=[{} for _ in vertices]
    for tri in faces:
        for a,b in zip(tri,np.roll(tri,-1)):
            length=float(np.linalg.norm(vertices[a]-vertices[b]))
            neighbours[a][b]=length;neighbours[b][a]=length
    front=float(points[:,2].max())
    locks=protected | (vertices[:,2]>=front)
    rear=np.clip((front-vertices[:,2])/(front-center[2]),0,1)
    rear=rear*rear*(3-2*rear);rear[locks]=0
    ring_delta=target-vertices[ring]
    ring_delta[:,1]=0  # Unity Y is height: no retained vertex may stretch vertically.
    ring_delta*=rear[ring,None]
    original_radial=vertices[ring][:,[0,2]]-center[[0,2]]
    desired_radial=original_radial+ring_delta[:,[0,2]]
    outward=np.linalg.norm(desired_radial,axis=1)>np.linalg.norm(original_radial,axis=1)
    ring_delta[outward]=0;ring_delta[locks[ring]]=0
    distance=np.full(len(vertices),np.inf);queue=[]
    movable=ring[np.linalg.norm(ring_delta,axis=1)>0]
    for a in movable:distance[a]=0;heapq.heappush(queue,(0,int(a)))
    while queue:
        d,a=heapq.heappop(queue)
        if d!=distance[a] or d>=bandwidth:continue
        for b,length in neighbours[a].items():
            if locks[b]:continue
            nd=d+length
            if nd<distance[b]:distance[b]=nd;heapq.heappush(queue,(nd,int(b)))
    t=np.clip(1-distance/bandwidth,0,1);factor=t*t*(3-2*t)*rear
    factor[locks]=0
    fixed=dict(zip(map(int,ring),ring_delta))
    free=np.flatnonzero((factor>0) & ~np.isin(np.arange(len(vertices)),ring))
    lookup=dict(zip(map(int,free),range(len(free))));rows=[];cols=[];values=[];rhs=np.zeros((len(free),3))
    for a in free:
        row=lookup[int(a)];rows.append(row);cols.append(row);values.append(len(neighbours[a]))
        for b in neighbours[a]:
            if b in lookup:rows.append(row);cols.append(lookup[b]);values.append(-1)
            elif b in fixed:rhs[row]+=fixed[b]
    result=vertices.copy()
    if len(free):
        lap=csr_matrix((values,(rows,cols)),shape=(len(free),len(free)))
        result[free]+=factor[free,None]*spsolve(lap,rhs)
    result[ring]+=ring_delta
    result[locks]=vertices[locks];result[:,1]=vertices[:,1]
    if not np.isfinite(result).all() or not np.array_equal(result[protected],vertices[protected]):
        raise ValueError('Protected face changed or invalid authored coordinates')
    affected=np.any(result!=vertices,axis=1)
    factor[~affected]=0
    design=dict(format='hs2_posterior_neck_contraction_v2',ring_indices=ring.tolist(),
        retained_ring_after_contraction=result[ring].tolist(),target_ring=target.tolist(),target_normals=target_normals.tolist(),
        protected_vertex_indices=np.flatnonzero(protected).tolist(),hard_lock_indices=np.flatnonzero(locks).tolist(),
        protected_positions_literal=True,retained_height_positions_literal=True,height_axis='Unity Y',
        affected_vertices=np.flatnonzero(affected).tolist(),bandwidth=bandwidth,
        native_body_vertices_modified=0,original_neck_faces_restored=0,
        retained_boundary_forced_to_body=False,integrated_underjaw_transition_required=True,
        policy='Hard face/ear/front locks; posterior contraction only; all retained heights fixed')
    return result,factor,design


def transition(vertices, faces, ring, target, start_normals, end_normals, steps=4):
    """Shared-vertex, tangent-continuous strip inside o_head, not a second shell."""
    ring=np.asarray(ring,int);a=vertices[ring];b=np.asarray(target,float)
    chord=b-a;n0=np.asarray(start_normals,float);n1=np.asarray(end_normals,float)
    start=chord-n0*np.sum(chord*n0,axis=1,keepdims=True)
    end=chord-n1*np.sum(chord*n1,axis=1,keepdims=True)
    rows=[ring];extra=[];extra_normals=[];count=len(vertices);width=len(ring)
    for k in range(1,steps+1):
        t=k/steps
        p=(2*t**3-3*t*t+1)*a+(t**3-2*t*t+t)*start+(-2*t**3+3*t*t)*b+(t**3-t*t)*end
        if k==steps:p=b.copy()
        extra.append(p);normal=(1-t)*n0+t*n1
        normal/=np.linalg.norm(normal,axis=1,keepdims=True);extra_normals.append(normal)
        rows.append(np.arange(count,count+width));count+=width
    added=[]
    for outer,inner in zip(rows,rows[1:]):
        for i in range(width):
            j=(i+1)%width
            added.extend([[outer[j],outer[i],inner[i]],[outer[j],inner[i],inner[j]]])
    all_vertices=np.concatenate([vertices,*extra]);all_faces=np.concatenate([faces,np.asarray(added,int)])
    return all_vertices,all_faces,np.concatenate(extra_normals),rows[-1]
