"""Exact mesh boundaries and explicit polygon correspondences for neck authoring."""
from collections import Counter

import numpy as np


def ordered_loops(faces, aliases=None, edge_filter=None):
    aliases=aliases or {};counts=Counter()
    for tri in np.asarray(faces).reshape(-1,3):
        for a,b in zip(tri,np.roll(tri,-1)):
            counts[(aliases.get(int(a),int(a)),aliases.get(int(b),int(b)))]+=1
    if any(n>1 for n in counts.values()):raise ValueError('Duplicate directed mesh edge')
    edges={(a,b) for a,b in counts if (b,a) not in counts}
    if edge_filter is not None:edges={e for e in edges if edge_filter(*e)}
    following=dict(edges)
    if len(following)!=len(edges) or len({b for a,b in edges})!=len(edges):
        raise ValueError('Mesh boundary branches')
    loops=[]
    while following:
        first=a=min(following);loop=[]
        while a in following:loop.append(a);a=following.pop(a)
        if a!=first:raise ValueError('Mesh boundary is an open chain')
        loops.append(loop)
    return loops


def original_neck_boundary(original,bone_names):
    """Cancel only byte-identical bind-position/ordered-skin UV duplicates."""
    vertices=np.asarray(original['vertices'],np.float32)
    indices=np.asarray(original['bone_indices'],np.int32)
    weights=np.asarray(original['bone_weights'],np.float32)
    unique={};aliases={}
    for i in range(len(vertices)):
        key=vertices[i].tobytes()+indices[i].tobytes()+weights[i].tobytes()
        unique.setdefault(key,i);aliases[i]=unique[key]
    head=bone_names.index('cf_J_Head_s')
    head_weight=(weights*(indices==head)).sum(1)
    loops=ordered_loops(original['triangles'],aliases,lambda a,b:head_weight[a]==1 and head_weight[b]==1)
    if len(loops)!=1:raise ValueError('Expected one native head-support opening')
    return loops[0],[[i,a] for i,a in aliases.items() if i!=a]


def radial_samples(polygon,values,angles,center):
    """Ray/edge intersection: outputs lie on actual polygon edges."""
    polygon=np.asarray(polygon,float);values=np.asarray(values,float)
    q=polygon[:,[0,2]]-np.asarray(center)[[0,2]];edge=np.roll(q,-1,axis=0)-q
    rays=np.column_stack([np.sin(angles),np.cos(angles)])
    cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
    den=cross(edge[None],rays[:,None])
    u=np.divide(-cross(q[None],rays[:,None]),den,out=np.full_like(den,np.inf),where=np.abs(den)>1e-14)
    radius=np.sum((q[None]+u[:,:,None]*edge)*rays[:,None],axis=2)
    radius=np.where((u>=-1e-9)&(u<=1+1e-9)&(radius>=0),radius,np.inf)
    chosen=np.argmin(radius,axis=1)
    if not np.isfinite(radius[np.arange(len(angles)),chosen]).all():raise ValueError('Nonradial neck polygon')
    u=np.clip(u[np.arange(len(angles)),chosen,None],0,1);next_ids=(chosen+1)%len(polygon)
    return (1-u)*polygon[chosen]+u*polygon[next_ids],(1-u)*values[chosen]+u*values[next_ids]


def corresponding_lower_row(top_points,upper,lower):
    """Retain edge identity/fraction when upper and lower azimuths differ."""
    edge=np.roll(upper,-1,axis=0)-upper
    t=np.clip(np.sum((top_points[:,None]-upper)*edge,axis=2)/np.sum(edge**2,axis=1),0,1)
    distance=np.linalg.norm(top_points[:,None]-(upper+t[:,:,None]*edge),axis=2)
    selected=np.argmin(distance,axis=1);fraction=t[np.arange(len(top_points)),selected,None]
    if distance[np.arange(len(top_points)),selected].max()>1e-7:
        raise ValueError('Upper neck row left the native extension polygon')
    return (1-fraction)*lower[selected]+fraction*lower[(selected+1)%len(lower)]
