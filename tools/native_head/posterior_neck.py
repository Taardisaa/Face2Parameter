"""Author only the posterior contour of an existing integrated neck mesh.

This is explicit mesh authoring, not an approximation of HS2 deformation.
The body interface and front stay fixed. A cubic Bezier replaces the longer
rear profile, fading toward the sides and back into the unedited upper skull.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from tools.model_bridge.artifact import sha


def midline_segments(vertices, faces, values):
    points, coordinates = [], []
    for tri in faces:
        p, q = [], []
        for a, b in zip(tri, np.roll(tri, -1)):
            if vertices[a, 0]*vertices[b, 0] < 0:
                t = -vertices[a, 0]/(vertices[b, 0]-vertices[a, 0])
                p.append(vertices[a]+t*(vertices[b]-vertices[a]))
                q.append(values[a]+t*(values[b]-values[a]))
        if len(p) == 2:
            points.append(p); coordinates.append(q)
    return np.asarray(points), np.asarray(coordinates)


def rear_at(segments, coordinates, requested):
    result, tangents = [], []
    for q in np.atleast_1d(requested):
        span = coordinates[:, 1]-coordinates[:, 0]
        valid = (np.abs(span)>1e-12) & (q>=coordinates.min(1)-1e-10) & (q<=coordinates.max(1)+1e-10)
        ids = np.flatnonzero(valid)
        if not len(ids):
            raise ValueError('No posterior midline intersection at requested coordinate')
        t = np.clip((q-coordinates[ids, 0])/span[ids],0,1)
        points = segments[ids, 0]+t[:, None]*(segments[ids, 1]-segments[ids, 0])
        i = points[:, 2].argmin(); index = ids[i]
        result.append(points[i])
        tangent=(segments[index,1]-segments[index,0])/span[index]
        tangents.append(tangent/np.linalg.norm(tangent))
    return np.asarray(result), np.asarray(tangents)


def harmonic_coordinate(vertices, faces, seam, upper_height):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T
    w=1/np.linalg.norm(vertices[a]-vertices[b],axis=1)
    adjacency=coo_matrix((np.r_[w,w],(np.r_[a,b],np.r_[b,a])),shape=(len(vertices),)*2).tocsr()
    _,components=connected_components(adjacency)
    component=components[seam[0]]
    if not np.all(components[seam]==component):raise ValueError('Disconnected seam')
    active=components==component
    fixed=(vertices[:,1]>=upper_height)|~active
    fixed[seam]=True
    q=np.ones(len(vertices));q[seam]=0
    from scipy.sparse import diags
    laplacian=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    free=~fixed
    q[free]=spsolve(laplacian[free][:,free],-laplacian[free][:,fixed]@q[fixed])
    if not np.isfinite(q).all() or q.min() < -1e-9 or q.max() > 1+1e-9:
        raise ValueError('Invalid longitudinal mesh coordinate')
    return np.clip(q,0,1),active


def fair_posterior(vertices, faces, target, fade):
    """Biharmonic position fairing with a Bezier midline and literal outside.

    Moving an already folded short collar with a scalar profile offset can leave
    overlapping triangles away from the midline. Fair the full posterior patch,
    rather than transporting those folds into the new surface.
    """
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T
    weights=np.ones(len(a))
    adjacency=coo_matrix((np.r_[weights,weights],(np.r_[a,b],np.r_[b,a])),shape=(len(vertices),)*2).tocsr()
    from scipy.sparse import diags
    laplacian=diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    laplacian=diags(1/np.maximum(np.asarray(adjacency.sum(1)).ravel(),1))@laplacian
    allowed=fade>0
    crossed=edges[vertices[a,0]*vertices[b,0]<0]
    guides=np.zeros(len(vertices),bool);guides[np.unique(crossed)]=True
    guides&=allowed
    fixed=~allowed|guides
    positions=vertices.copy();positions[guides]=target[guides]
    energy=laplacian.T@laplacian
    free=~fixed
    positions[free]=spsolve(energy[free][:,free],-energy[free][:,fixed]@positions[fixed])
    if not np.isfinite(positions).all():raise ValueError('Posterior fairing is singular')
    return positions


def author(arrays, receipt, masks, body):
    v,f=arrays['vertices'],arrays['faces']
    original=arrays['original_vertices']*receipt['scale']+receipt['translation']
    ear_ids=np.r_[masks['left_ear'],masks['right_ear']]
    ears=original[ear_ids]
    # Anatomical construction: just below upper ears by a tenth of ear height.
    # This is an authoring choice shared across identities, not a person-specific
    # coordinate or a claim that FLAME has a semantic "perfect cut" parameter.
    upper_height=float(ears[:,1].max()-.1*np.ptp(ears[:,1]))
    side_front=float(ears[:,2].mean())
    seam=np.asarray(receipt['native_outer_ring_authored_ids'])
    q,active=harmonic_coordinate(v,f,seam,upper_height)
    mid,coordinates=midline_segments(v,f,q)
    old,_=rear_at(mid,coordinates,q)
    p0,_=rear_at(mid,coordinates,[0])
    # The first fully fixed posterior edge can be above the anatomical threshold
    # on this coarse topology. Use that actual edge, otherwise the displacement
    # would jump back to zero at q=1 and create a new upper kink.
    p3,upper_tangent=rear_at(mid,coordinates,[1])
    p0,p3=p0[0],p3[0]
    mesh=next(m for m in body['meshes'] if m['mesh_name']=='o_body_cf')
    root=mesh['bone_names'].index('cf_J_Head_s')
    matrix=np.asarray(mesh['source']['bindposes'][root]).reshape(4,4)
    bp=np.asarray(mesh['source']['vertices'])@matrix[:3,:3].T+matrix[:3,3]
    bn=np.asarray(mesh['source']['normals'])@np.linalg.inv(matrix[:3,:3])
    bn/=np.linalg.norm(bn,axis=1,keepdims=True)
    rear_seam=seam[np.argsort(v[seam,2])[:2]]
    matching=np.linalg.norm(v[rear_seam,None]-bp,axis=2).argmin(1)
    normal=bn[matching].mean(0)
    lower_tangent=np.array([0.,-normal[2],normal[1]])
    lower_tangent/=np.linalg.norm(lower_tangent)
    if lower_tangent[1]<=0:raise ValueError('Native posterior tangent must point upward')
    upper_tangent=upper_tangent[0];upper_tangent[0]=0
    upper_tangent/=np.linalg.norm(upper_tangent)
    distance=np.linalg.norm(p3-p0)
    p1=p0+lower_tangent*distance/3
    near_upper,_=rear_at(mid,coordinates,[1-1e-5])
    upper_speed=np.linalg.norm((p3-near_upper[0])/1e-5)
    p2=p3-upper_tangent*upper_speed/3
    controls=np.asarray([p0,p1,p2,p3])
    t=q[:,None]
    curve=(1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t*t*p2+t**3*p3
    depth=np.clip((side_front-v[:,2])/np.maximum(side_front-old[:,2],1e-9),0,1)
    fade=depth**3*(10-15*depth+6*depth**2)
    fade[~active]=0
    # Map semantic facial/ear locks through the exact original crop recipes.
    locked_ids=np.unique(np.r_[masks['face'],ear_ids,masks['eye_region'],masks['left_eyeball'],masks['right_eyeball']])
    locked_ids=np.unique(arrays['original_faces'][np.isin(arrays['original_faces'],locked_ids).any(1)])
    recipes=arrays['crop_recipes'];count=len(recipes)
    locked=np.zeros(len(v),bool)
    locked[:count]=np.isin(recipes[:,:2].astype(int),locked_ids).any(1)
    fade[locked]=0;fade[seam]=0;fade[q>=1]=0
    delta=curve-old;delta[:,0]=0
    edited=v+fade[:,None]*delta
    untouched=(fade==0)
    edited[untouched]=v[untouched]
    edited=fair_posterior(v,f,edited,fade)
    # Re-lay the old short connector's interior rows between their newly faired
    # endpoints. Its previous return fold must not survive as crossed internal
    # rows. Only posterior vertices participate; front connector rows stay literal.
    width=len(arrays['source_ring'])
    if len(arrays['native_upper'])!=width:
        raise ValueError('Expected matching connector endpoint rows')
    start=edited[arrays['source_ring']]
    end=edited[arrays['native_upper']]
    interior_start=len(v)-5*width
    remap=np.r_[arrays['source_ring'],np.arange(interior_start,len(v)),arrays['native_upper']]
    expected=[]
    for row in range(6):
        for j in range(width):
            a=row*width+j;b=row*width+(j+1)%width;c=(row+1)*width+j;d=(row+1)*width+(j+1)%width
            expected.extend([[b,a,c],[b,c,d]])
    if not np.array_equal(f[-12*width:],remap[np.asarray(expected)]):
        raise ValueError('Unimplemented connector layout: expected six audited rows')
    for row,t in enumerate(np.linspace(0,1,7)[1:-1]):
        ids=np.arange(interior_start+row*width,interior_start+(row+1)*width)
        keep=fade[ids]>0
        edited[ids[keep]]=((1-t)*start+t*end)[keep]
    # No global translation, height scaling, topology replacement or front work.
    if not np.array_equal(edited[locked],v[locked]) or not np.array_equal(edited[seam],v[seam]):
        raise ValueError('Posterior edit moved face/ear/interface locks')
    front=v[:,2]>=side_front
    if not np.array_equal(edited[front],v[front]):raise ValueError('Front changed')
    return edited,dict(controls=controls.tolist(),upper_height=upper_height,
        side_front=side_front,
        upper_anchor='Actual first fixed posterior edge above upper-ear height minus tenth ear span',
        lower_tangent=lower_tangent.tolist(),upper_tangent=upper_tangent.tolist(),
        edited_vertices=int(np.count_nonzero(np.linalg.norm(edited-v,axis=1)>1e-12)),
        front_vertices_unchanged=True,face_ears_eyes_unchanged=True,interface_unchanged=True,
        global_head_translation=[0,0,0],topology_unchanged=True),q,fade


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--masks',type=Path,default=Path(r'C:\Users\13666\Workspace\smirk\assets\FLAME_masks\FLAME_masks.pkl'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output')
    arrays=dict(np.load(args.baseline/'geometry.npz',allow_pickle=False))
    receipt=json.loads((args.baseline/'receipt.json').read_text())
    masks=pickle.loads(args.masks.read_bytes(),encoding='latin1')
    if sha(args.masks)!=receipt['masks_sha256'] or sha(args.body)!=receipt['body_capture_sha256']:
        raise ValueError('Masks/body differ from audited baseline')
    edited,design,q,fade=author(arrays,receipt,masks,json.loads(args.body.read_text()))
    args.output.mkdir(parents=True)
    arrays['before_vertices']=arrays['vertices'].copy();arrays['vertices']=edited
    arrays['posterior_coordinate']=q;arrays['posterior_fade']=fade
    np.savez(args.output/'geometry.npz',**arrays)
    receipt.update(posterior_design=design,baseline_geometry_sha256=sha(args.baseline/'geometry.npz'),
        geometry_sha256=sha(args.output/'geometry.npz'),posterior_only=True,
        retained_original_vertices_displaced='Only posterior transition area, explicitly authorized; face/ears/front/upper skull remain fixed',
        native_asset_installed=False,game_mutations=False)
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (args.output/'neck.obj').write_text('\n'.join(
        ['v '+' '.join(map(str,p)) for p in edited]+['f '+' '.join(str(int(i)+1) for i in t) for t in arrays['faces']])+'\n')
    print(json.dumps(design))


if __name__=='__main__':main()
