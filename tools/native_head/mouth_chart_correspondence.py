"""Donor-bound lip correspondence in the real, unfolded surface UV charts.

Identity-independent asset authoring, not a replacement for native expression
drivers. The native cavity attachment is the inner contour; pigment embeddings
are the existing outer contour. All original source coefficient columns survive.
Neither spatial nearest normals nor coincident closed-lip positions select layers.
"""
import argparse,json
from pathlib import Path
import numpy as np
from tools.native_head.neck_geometry import ordered_loops
from tools.native_head.mother_identity import barycentric
from tools.model_bridge.scan_accuracy import TriangleSurface
from tools.native_head.mother_template_inputs import source_file,save_json
from tools.model_bridge.artifact import sha
from tools.native_head.mouth_chart_solver import oriented_chart

def read_uv(path,faces):
    lines=path.read_text().splitlines()
    uv=np.array([[float(x) for x in l.split()[1:3]] for l in lines if l.startswith('vt ')])
    records=[l.split()[1:] for l in lines if l.startswith('f ')]
    vertices=np.array([[int(x.split('/')[0])-1 for x in row] for row in records])
    indices=np.array([[int(x.split('/')[1])-1 for x in row] for row in records])
    if not np.array_equal(vertices,faces):raise ValueError('UV template is not exact FLAME topology/order')
    return uv[indices]

def inner_contour(uv,faces,oral_faces,source_inner):
    loops=ordered_loops(faces[oral_faces])
    if len(loops)!=1:raise ValueError('Expected one actual cavity/skin attachment contour')
    loop=np.array(loops[0],int)
    # Native chart endpoints and upper/lower midpoints come from its literal
    # skin-side contour, not a made-up inner-lip UV rectangle.
    points=uv[loop]
    left=int(np.argmax(points[:,0]));right=int(np.argmin(points[:,0]))
    def arc(a,b):
        ids=np.arange(a,b+1) if b>=a else np.r_[np.arange(a,len(loop)),np.arange(b+1)]
        return ids
    a,b=arc(left,right),arc(right,left)
    upper=a if points[a,1].mean()>points[b,1].mean() else b
    lower=b if np.array_equal(upper,a) else a
    middle=(points[left,0]+points[right,0])/2
    top=int(upper[np.argmin(abs(points[upper,0]-middle))])
    bottom=int(lower[np.argmin(abs(points[lower,0]-middle))])
    if len({left,top,right,bottom})!=4:raise ValueError('Inner contour landmarks are ambiguous')
    anchors=[left,top,right,bottom]
    # Orient the loop left -> upper -> right -> lower, matching FLAME 60..67.
    if top not in arc(left,right):
        loop=loop[::-1];points=uv[loop]
        anchors=[len(loop)-1-i for i in anchors]
    dst=np.zeros_like(points);assigned=np.zeros(len(loop),bool)
    for k,(start,end) in enumerate(zip(anchors,np.roll(anchors,-1))):
        indices=arc(start,end)
        poly=points[indices];length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(poly,axis=0),axis=1))]
        if length[-1]<=0:raise ValueError('Collapsed native inner contour arc')
        source=source_inner[[(2*k)%8,(2*k+1)%8,(2*k+2)%8]]
        dx=poly[-1,0]-poly[0,0]
        if dx==0:raise ValueError('Inner half-contour has no horizontal extent')
        fraction=(poly[:,0]-poly[0,0])/dx
        q=source[0,0]+fraction*(source[-1,0]-source[0,0])
        order=np.argsort(source[:,0])
        values=np.column_stack([q,np.interp(q,source[order,0],source[order,1])])
        if np.any(assigned[indices]) and not np.allclose(dst[indices[assigned[indices]]],values[assigned[indices]],atol=1e-12,rtol=0):
            raise ValueError('Shared native inner contour endpoints disagree')
        dst[indices]=values;assigned[indices]=True
    if not assigned.all():raise ValueError('Native inner contour incomplete')
    return loop,dst,dict(native_corner_ids=loop[anchors].tolist(),policy='Four actual skin-chart extrema/midpoints; horizontal contour coordinate on each actual quadrant; decoder contour edges retained')

def strip_map(points,outer,outer_values,inner,inner_values):
    """Two unfolded lip strips between actual outer and cavity contour edges."""
    left=int(np.argmin(outer[:,0]));right=int(np.argmax(outer[:,0]))
    il=int(np.argmin(inner[:,0]));ir=int(np.argmax(inner[:,0]))
    def path(poly,values,start,end):
        ids=np.arange(start,end+1) if end>=start else np.r_[np.arange(start,len(poly)),np.arange(end+1)]
        return poly[ids],values[ids]
    oa=path(outer,outer_values,left,right);ob=path(outer,outer_values,right,left)
    ia=path(inner,inner_values,il,ir);ib=path(inner,inner_values,ir,il)
    if oa[0][:,1].mean()<ob[0][:,1].mean():oa,ob=ob,oa
    if ia[0][:,1].mean()<ib[0][:,1].mean():ia,ib=ib,ia
    def extend(row):
        p,v=row
        if p[0,0]>p[-1,0]:p,v=p[::-1],v[::-1]
        return np.vstack([outer[left],p,outer[right]]),np.vstack([outer_values[left],v,outer_values[right]])
    ia,ib=extend(ia),extend(ib)
    def section(row,x):
        p,v=row
        if p[0,0]>p[-1,0]:p,v=p[::-1],v[::-1]
        # Exact contour endpoints can share X. A vertical edge has distinct
        # layer meaning; select the incident nonvertical edge, never average.
        d=np.diff(p[:,0]);eligible=np.flatnonzero((d>0)&(x>=p[:-1,0]-1e-12)&(x<=p[1:,0]+1e-12))
        if not len(eligible):raise ValueError('Skin query outside actual lip-strip chart')
        i=int(eligible[0]);t=(x-p[i,0])/d[i]
        return (1-t)*p[i,1]+t*p[i+1,1],(1-t)*v[i]+t*v[i+1]
    result=[];fractions=[]
    for x,y in points:
        ya,va=section(ia,x);yb,vb=section(ib,x)
        upper=y>=(ya+yb)/2
        yi,vi=(ya,va) if upper else (yb,vb)
        yo,vo=section(oa if upper else ob,x)
        if yo==yi:raise ValueError('Lip-strip collapses at queried skin node')
        t=(y-yi)/(yo-yi)
        if t < -1e-8:raise ValueError('Skin query enters cavity strip; explicit layer partition required')
        result.append(vi+t*(vo-vi));fractions.append(t)
    return np.asarray(result),np.asarray(fractions)

def build(candidate,uv_template,out):
    if out.exists():raise FileExistsError('Preserve earlier correspondences')
    r=json.loads((candidate/'receipt.json').read_text())
    mother=Path(r['mother_reference']['path']).parent
    for row in (r['inputs'],r['mother_reference']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Donor/mother contract changed')
    mother_receipt=json.loads((mother/'receipt.json').read_text())
    if sha(mother/'o_head_candidate.npz')!=mother_receipt['candidate_arrays']['sha256']:
        raise ValueError('Mother surface/UV changed')
    m=np.load(mother/'o_head_candidate.npz',allow_pickle=False)
    inputs=Path(r['inputs']['path']).parent
    regions=json.loads((inputs/'native_regions.json').read_text())
    source=np.load(inputs/'flame_zero_identity.npz',allow_pickle=False)
    corners=read_uv(uv_template,source['faces_tensor'])
    lmk=(corners[source['full_lmk_faces_idx'][0]]*source['full_lmk_bary_coords'][0,:,:,None]).sum(1)
    canonical=np.array([int(regions['graph_aliases'].get(str(i),i)) for i in range(len(m['verts']))])
    unique,inverse=np.unique(canonical,return_inverse=True);faces=inverse[m['faces']]
    native_uv=m['uv'][unique].astype(float)
    mapping=np.load(candidate/'mouth_correspondence.npz',allow_pickle=False)
    embedding_faces=[row['face_id'] for row in r['candidate_anatomical_constraints']['annotations']
        if 48<=row['target_flame_landmark']<=59]
    lip=np.unique(np.r_[mapping['lip_ids'],faces[embedding_faces].reshape(-1)])
    loop,inner,inner_report=inner_contour(native_uv,faces,mapping['oral_face_ids'],lmk[60:68])
    src=[];dst=[]
    for row in r['candidate_anatomical_constraints']['annotations']:
        if 48<=row['target_flame_landmark']<=59:
            src.append(row['uv']);dst.append(lmk[row['target_flame_landmark']])
    if len(src)!=12:raise ValueError('Original outer-lip embeddings required')
    src=np.asarray(src);dst=np.asarray(dst)
    basis=np.load(candidate/'identity_basis.npz',allow_pickle=False)
    mapped=np.einsum('vi,vij->vj',basis['barycentric'][unique[lip]],corners[basis['source_faces'][unique[lip]]])
    within=(native_uv[lip,0]>=src[:,0].min())&(native_uv[lip,0]<=src[:,0].max())
    fractions=np.full(len(lip),np.nan)
    mapped[within],fractions[within]=strip_map(native_uv[lip][within],src,dst,native_uv[loop],inner)
    lookup=np.full(len(unique),-1);lookup[lip]=np.arange(len(lip))
    face_ids=np.flatnonzero(np.isin(faces,lip).all(1));local=lookup[faces[face_ids]]
    def signed_area(tri):
        a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    folds=face_ids[signed_area(native_uv[lip][local])*signed_area(mapped[local])>=0]
    rows=[];values=[]
    # Virtual donor pigment embeddings remain exact constraints on the actual
    # target triangles, rather than only on a synthetic point triangulation.
    for annotation in r['candidate_anatomical_constraints']['annotations']:
        if 48<=annotation['target_flame_landmark']<=59:
            nodes=lookup[faces[annotation['face_id']]]
            if np.any(nodes<0):raise ValueError('Outer-lip embedding leaves actual source-bound support')
            row=np.zeros(len(lip));row[nodes]=annotation['barycentric']
            rows.append(row);values.append(lmk[annotation['target_flame_landmark']])
    pinned=set()
    for index,value in zip(lookup[loop],inner):
        if index<0:raise ValueError('Physical mouth rim absent from lip support')
        row=np.zeros(len(lip));row[index]=1;rows.append(row);values.append(value);pinned.add(int(index))
    mapped,chart_solver=oriented_chart(mapped,local,np.asarray(rows),np.asarray(values),native_uv[lip])
    tris=np.concatenate([corners,np.zeros(corners.shape[:2]+(1,))],axis=2)
    query=np.column_stack([mapped,np.zeros(len(mapped))])
    closest,ids,_=TriangleSurface(tris).closest(query)
    error=np.linalg.norm(closest-query,axis=1)
    if error.max()>1e-8:raise ValueError('Mapped skin points enter source UV gap; explicit layer repair required')
    bary=barycentric(closest,tris[ids])
    out.mkdir(parents=True)
    np.savez_compressed(out/'correspondence.npz',lip_ids=lip,source_faces=ids,barycentric=bary,
        source_uv=mapped,native_uv=native_uv[lip],native_anchor_uv=src,source_anchor_uv=dst,
        inner_contour_ids=loop,source_inner_uv=inner,strip_fraction=fractions)
    save_json(out/'receipt.json',dict(format='native_lip_surface_chart_correspondence_v1',candidate=source_file(candidate/'receipt.json'),
        code=source_file(Path(__file__)),solver_code=source_file(Path(__file__).with_name('mouth_chart_solver.py')),
        mother=r['mother_reference'],inputs=r['inputs'],source_uv_template=source_file(uv_template),
        correspondence=source_file(out/'correspondence.npz'),inner_contour=inner_report,
        chart_solver=chart_solver,initial_fold_face_ids=folds.tolist(),
        identity_independent=True,source_coefficient_filtering=False,
        chart_policy='Two actual upper/lower surface strips with literal cavity/skin boundary; source decoder contour edges; all native lip triangle orientations checked',
        limitations=['Outer pigment embeddings remain existing anatomical candidates','Perioral points extend the surface chart beyond pigment; source UV coverage checked','No cavity or expression adaptation performed here'],deliverable=False))
    print(json.dumps(dict(output=str(out.resolve()),identity_independent=True,deliverable=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','uv-template','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();build(a.candidate,a.uv_template,a.out)
