"""Full-detail source lip transport and native oral extension.

In-progress asset-authoring rule, not an HS2 deformation driver. The complete
source lip identity field drives an intact native mother. Topology fixes its
oral interface. Every identity recomputes the same field and oral solve;
crossing/component acceptance is mandatory. No affine-only fallback exists.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.sparse.linalg import splu
from tools.native_head.mother_arap import SpokesARAP
from tools.native_head.mother_local_repair import similarity

def full_control_field(points, controls, values):
    """C2 polyharmonic extension, exact at every authored source lip vertex.

    This replaces the discontinuous derivative of a tetrahedral extension;
    it does not reduce controls to landmarks or an affine component. The affine
    tail is only the kernel's solvability constraint, not the delivered field.
    Zero smoothing, all coefficient columns and all original lip nodes remain.
    """
    controls=np.asarray(controls,float);values=np.asarray(values,float)
    points=np.asarray(points,float)
    if (controls.ndim!=2 or controls.shape[1]!=3 or points.ndim!=2 or points.shape[1]!=3
        or len(controls)<4 or len(values)!=len(controls) or not np.isfinite(controls).all()
        or not np.isfinite(points).all() or not np.isfinite(values).all()):
        raise ValueError('Finite 3D source lip controls and complete value columns required')
    if len(np.unique(controls,axis=0))!=len(controls):
        raise ValueError('Duplicate control positions require explicit source-ID validation; do not average lip layers')
    center=controls.mean(0);scale=np.linalg.norm(controls-center,axis=1).max()
    if scale<=0:raise ValueError("Collapsed source lip controls")
    x=(controls-center)/scale;q=(points-center)/scale
    poly=np.column_stack([np.ones(len(x)),x])
    if np.linalg.matrix_rank(poly)!=4:raise ValueError("Source lip controls are not volumetric")
    distance=np.linalg.norm(x[:,None]-x[None],axis=2)
    matrix=np.block([[distance**3,poly],[poly.T,np.zeros((4,4))]])
    flat=values.reshape(len(x),-1)
    fit=np.linalg.solve(matrix,np.vstack([flat,np.zeros((4,flat.shape[1]))]))
    kernel=np.linalg.norm(q[:,None]-x[None],axis=2)**3
    result=np.column_stack([kernel,np.ones(len(q)),q])@fit
    residual=np.max(np.abs(matrix[:len(x)]@fit-flat))
    if residual>1e-10*max(1.,float(np.abs(flat).max())):
        raise ValueError("Full source lip field interpolation lost controls")
    return result.reshape((len(points),)+values.shape[1:]),dict(
        source_control_count=len(x),coefficient_columns=flat.shape[1],
        interpolation_residual=float(residual),condition_number=float(np.linalg.cond(matrix)),
        kernel="r^3 polyharmonic with degree-1 solvability tail, zero smoothing",
        source_control_values_filtered=False)

def edges_and_graph(faces,count):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T
    graph=sparse.coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(count,count)).tocsr()
    return edges,graph

def make_correspondence(mother,faces,lip_ids,oral_face_ids,halo_rings=3):
    lip_ids=np.unique(np.asarray(lip_ids,int))
    oral_faces=faces[oral_face_ids]
    skin_faces=np.delete(faces,oral_face_ids,axis=0)
    oral=np.unique(oral_faces);skin=np.unique(skin_faces)
    boundary=np.intersect1d(oral,skin);interior=np.setdiff1d(oral,boundary)
    if len(boundary)<3:raise ValueError("Native oral component has no shared physical interface")
    if np.intersect1d(lip_ids,interior).size:raise ValueError("Exterior lip support enters oral interior")
    _,graph=edges_and_graph(faces,len(mother))
    distance=dijkstra(graph,directed=False,indices=lip_ids,min_only=True,unweighted=True)
    halo=np.flatnonzero((distance>0)&(distance<halo_rings)&~np.isin(np.arange(len(mother)),interior))
    return dict(lip_ids=lip_ids,oral_face_ids=np.asarray(oral_face_ids),
        oral_boundary_ids=boundary,oral_interior_ids=interior,halo_ids=halo,halo_distance=distance[halo],
        halo_rings=halo_rings)

def deform_oral(reference,target,faces,oral_face_ids,boundary,interior,max_iterations=2000,tolerance=1e-9):
    """ARAP of intact mother oral structure, with literal shared skin rim fixed."""
    if max_iterations<1 or tolerance<=0:raise ValueError('Invalid oral solver limits')
    selected=faces[oral_face_ids];used=np.unique(selected)
    local=np.searchsorted(used,selected);pins=np.searchsorted(used,boundary);free=np.searchsorted(used,interior)
    if not len(free):return target.copy(),dict(iterations=0,converged=True)
    scale,rotation,translation=similarity(reference[boundary],target[boundary])
    rest=reference[used]@rotation*scale+translation
    arap=SpokesARAP(rest,local)
    count,labels=connected_components(arap.stiffness[free][:,free],directed=False)
    for label in range(count):
        rows=free[labels==label]
        if not np.any(arap.stiffness[rows][:,pins].data):raise ValueError("Unanchored oral component")
    factor=splu(arap.stiffness[free][:,free].tocsc())
    cross=arap.stiffness[free][:,pins]
    x=rest.copy();x[pins]=target[boundary]
    for iteration in range(max_iterations):
        nxt=factor.solve(arap.rhs(x)[free]-cross@x[pins])
        update=float(np.linalg.norm(nxt-x[free],axis=1).max())
        x[free]=nxt
        if update<=tolerance:break
    result=target.copy();result[interior]=x[free]
    if not np.array_equal(result[boundary],target[boundary]):raise ValueError("Oral solve moved skin interface")
    return result,dict(iterations=iteration+1,converged=update<=tolerance,max_update=update,
        tolerance=tolerance,uniform_initial_placement=dict(scale=float(scale),rotation=rotation.tolist(),translation=translation.tolist()))

def apply_full_controls(reference,faces,target,controls,control_basis,beta,mapping):
    """Continuous source identity field, with all lip controls retained exactly."""
    basis,interpolation=full_control_field(reference,controls,control_basis)
    result,report=apply_lip_basis(reference,faces,target,basis[mapping['lip_ids']],beta,mapping)
    report['source_field']=interpolation
    report['policy']='C2 full source lip identity field; '+report['policy']
    return result,report

def apply_lip_basis(reference,faces,target,lip_basis,beta,mapping):
    """Consume complete, explicitly corresponding source shape columns."""
    ids=mapping['lip_ids']
    if lip_basis.shape!=(len(ids),3,len(beta)) or not np.isfinite(lip_basis).all():
        raise ValueError('Complete source identity columns at all corresponding lip nodes required')
    desired=reference[ids]+lip_basis@beta
    result=target.copy();result[ids]=desired
    halo=mapping["halo_ids"]
    if len(halo):
        edges,_=edges_and_graph(faces,len(reference));a,b=edges.T
        lengths=np.linalg.norm(reference[a]-reference[b],axis=1)
        if np.any(lengths==0):raise ValueError("Zero physical edge")
        w=sparse.coo_matrix((np.r_[1/lengths,1/lengths],(np.r_[a,b],np.r_[b,a])),shape=(len(reference),)*2).tocsr()
        L=sparse.diags(np.asarray(w.sum(1)).ravel())-w
        correction=splu(L[halo][:,halo].tocsc()).solve(-L[halo][:,ids]@(desired-target[ids]))
        result[halo]=target[halo]+correction
    result,oral=deform_oral(reference,result,faces,mapping["oral_face_ids"],mapping["oral_boundary_ids"],mapping["oral_interior_ids"])
    if not np.array_equal(result[ids],desired):raise ValueError("Full source lip controls changed")
    return result,dict(policy="Hard skin lip targets, harmonic halo, source native oral ARAP at literal shared rim",
        lip_detail_filtered=False,source_coefficients_modified=False,
        full_detail_lip_constraints_exact=True,oral=oral,
        changed_logical_ids=np.flatnonzero(np.any(result!=target,axis=1)).tolist(),
        semantic_partition="Provisional pigment support plus actual physical oral rim; requires anatomical review",
        acceptance_required=["No added crossings","Native component/frame integration","Explicit out-of-domain repair; no affine fallback"])
