"""Conform only the retained head's neck band to an actual native bind rim.

No faces added or restored. Keep every body corner by subdividing its existing
edge contour to the retained source ring count, then use a smooth geodesic falloff.
"""
from __future__ import annotations

import heapq
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def conform(vertices, faces, ring, native, descriptor, bandwidth):
    mesh = native['meshes'][0]; data=mesh['source']; ring=np.asarray(ring,int)
    body_ids=np.asarray(descriptor['native_ring_cut_indices'],int)
    bone_index=mesh['bone_names'].index('cf_J_Head_s')
    indices=np.asarray(data['bone_indices']);weights=np.asarray(data['bone_weights'])
    support=(weights[body_ids]*(indices[body_ids]==bone_index)).sum(1)
    if not np.array_equal(support,np.ones(len(support))):
        raise ValueError('Native rim must be authored wholly to cf_J_Head_s')
    bind=np.asarray(data['bindposes']).reshape(-1,4,4)[bone_index]
    original=np.asarray(data['vertices'])[body_ids]
    points=(np.c_[original,np.ones(len(original))]@bind.T)[:,:3]
    normals=np.asarray(data['normals'])[body_ids]@np.linalg.inv(bind[:3,:3])
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    if len(ring)<len(points):
        raise ValueError('Retained ring has too few corners to preserve native contour')
    subdivisions=np.ones(len(points),int)
    lengths=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    for _ in range(len(ring)-len(points)):
        subdivisions[np.argmax(lengths/subdivisions)]+=1
    target=[];target_normals=[]
    for i,count in enumerate(subdivisions):
        for t in np.arange(count)/count:
            target.append((1-t)*points[i]+t*points[(i+1)%len(points)])
            target_normals.append((1-t)*normals[i]+t*normals[(i+1)%len(points)])
    target=np.asarray(target);target_normals=np.asarray(target_normals)
    options=[]
    for reverse in [False,True]:
        candidate=target[::-1] if reverse else target
        for shift in range(len(candidate)):
            cost=np.sum((vertices[ring][:,[0,2]]-np.roll(candidate,shift,axis=0)[:,[0,2]])**2)
            options.append((float(cost),reverse,shift))
    _,reverse,shift=min(options)
    if reverse:target=target[::-1];target_normals=target_normals[::-1]
    target=np.roll(target,shift,axis=0);target_normals=np.roll(target_normals,shift,axis=0)
    neighbours=[{} for _ in vertices]
    for triangle in faces:
        for a,b in zip(triangle,np.roll(triangle,-1)):
            length=float(np.linalg.norm(vertices[a]-vertices[b]))
            neighbours[a][b]=length;neighbours[b][a]=length
    distance=np.full(len(vertices),np.inf);nearest=np.full(len(vertices),-1,int);queue=[]
    for i,vertex in enumerate(ring):distance[vertex]=0;nearest[vertex]=i;heapq.heappush(queue,(0,int(vertex)))
    while queue:
        d,a=heapq.heappop(queue)
        if d!=distance[a] or d>=bandwidth:continue
        for b,length in neighbours[a].items():
            nd=d+length
            if nd<distance[b]:distance[b]=nd;nearest[b]=nearest[a];heapq.heappush(queue,(nd,int(b)))
    t=np.clip(1-distance/bandwidth,0,1);factor=t*t*(3-2*t)
    offsets=target-vertices[ring];result=vertices.copy()
    selected=factor>0
    # Harmonic extension avoids Voronoi/nearest-anchor creases in the band.
    # Fixed boundary displacement; unchanged exterior; smoothstep fade at its
    # upper edge. This is authored asset fairing, not inferred HS2 behavior.
    fixed=dict(zip(map(int,ring),offsets));free=np.flatnonzero(selected & (distance>0))
    free_map=dict(zip(map(int,free),range(len(free))));rows=[];cols=[];values=[];rhs=np.zeros((len(free),3))
    for a in free:
        row=free_map[int(a)];rows.append(row);cols.append(row);values.append(len(neighbours[a]))
        for b in neighbours[a]:
            if b in free_map:rows.append(row);cols.append(free_map[b]);values.append(-1)
            elif b in fixed:rhs[row]+=fixed[b]
    if len(free):
        laplacian=csr_matrix((values,(rows,cols)),shape=(len(free),len(free)))
        displacement=spsolve(laplacian,rhs)
        result[free]+=factor[free,None]*displacement
    result[ring]=target
    target_normals/=np.linalg.norm(target_normals,axis=1,keepdims=True)
    if not np.isfinite(result).all():raise ValueError('Invalid neck-band geometry')
    return result,factor,dict(ring_indices=ring.tolist(),target_ring=target.tolist(),
        target_normals=target_normals.tolist(),bandwidth=bandwidth,affected_vertices=np.flatnonzero(selected).tolist(),
        triangles_added=0,native_body_vertices_modified=0,original_neck_faces_restored=0,
        policy='Existing retained neck band only; preserve native contour corners; harmonic displacement and smooth geodesic falloff')
