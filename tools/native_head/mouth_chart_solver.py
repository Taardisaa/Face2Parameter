"""Orientation-preserving donor surface-chart authoring with exact anchors.

This solves a correspondence once, independently of identity coefficients. It
does not smooth, filter or refit any FLAME geometry or coefficient field.
"""
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

def areas(points,faces):
    t=points[faces];a,b=t[:,1]-t[:,0],t[:,2]-t[:,0]
    return a[:,0]*b[:,1]-a[:,1]*b[:,0]

def oriented_chart(initial,faces,anchor_matrix,anchor_values,native_uv=None):
    center=initial.mean(0);scale=np.linalg.norm(initial-center,axis=1).max()
    if scale<=0:raise ValueError('Collapsed source surface chart')
    original=(initial-center)/scale
    A=np.asarray(anchor_matrix,float);target=(anchor_values-A.sum(1)[:,None]*center)/scale
    if np.linalg.matrix_rank(A)!=len(A):raise ValueError('Dependent surface-chart anchors; explicit correspondence repair required')
    if native_uv is not None:
        return regular_chart(initial,faces,A,anchor_values,native_uv,center,scale)
    n=len(initial);rows=np.arange(len(faces))[:,None]
    margin=np.finfo(float).eps*32
    # Project exact anatomical anchors first; only their support and inverted
    # triangles need unknowns. All other chart nodes stay literal, including
    # the outer transition. One incident ring includes affected neighbour faces.
    projected=original+A.T@np.linalg.solve(A@A.T,target-A@original)
    bad=faces[areas(projected,faces)<=margin].reshape(-1)
    active=np.union1d(bad,np.flatnonzero(np.any(A!=0,axis=0)))
    active=np.unique(faces[np.isin(faces,active).any(1)])
    selected=np.flatnonzero(np.isin(faces,active).any(1))
    faces=faces[selected];rows=np.arange(len(faces))[:,None]
    lookup=np.full(n,-1);lookup[active]=np.arange(len(active))
    def unpack(x):
        p=projected.copy();p[active]=x.reshape(len(active),2);return p
    def value(x):return areas(unpack(x),faces)-margin
    def jac(x):
        t=unpack(x)[faces];out=np.zeros((len(faces),len(active)*2))
        for k in range(3):
            a,b=t[:,(k+1)%3],t[:,(k+2)%3]
            participating=lookup[faces[:,k]]>=0
            out[rows[:,0][participating],2*lookup[faces[participating,k]]]=a[participating,1]-b[participating,1]
            out[rows[:,0][participating],2*lookup[faces[participating,k]]+1]=b[participating,0]-a[participating,0]
        return out
    E=np.kron(A[:,active],np.eye(2))
    inactive=np.setdiff1d(np.arange(n),active)
    rhs=(target-A[:,inactive]@projected[inactive]).reshape(-1)
    start=projected[active].reshape(-1)
    objective_reference=original[active].reshape(-1)
    result=minimize(lambda x:.5*np.sum((x-objective_reference)**2),start,
        jac=lambda x:x-objective_reference,method='SLSQP',
        constraints=[dict(type='eq',fun=lambda x:E@x-rhs,jac=lambda x:E),
                     dict(type='ineq',fun=value,jac=jac)],
        options=dict(ftol=1e-12,maxiter=1000))
    mapped=unpack(result.x)*scale+center
    residual=float(np.max(abs(A@mapped-anchor_values),initial=0))
    positive=bool(np.all(areas(mapped,faces)>0))
    if not result.success or not positive or residual>1e-9:
        raise ValueError('Surface-chart constraints not satisfied: '+str(dict(reason=result.message,positive=positive,anchor_residual=residual)))
    return mapped,dict(solver='SLSQP minimum UV correspondence displacement with exact source anatomical embeddings and strictly positive triangle orientation',
        iterations=int(result.nit),anchor_residual=residual,all_triangles_positive=positive,
        maximum_uv_correction=float(np.linalg.norm(mapped-initial,axis=1).max()),
        source_geometry_modified=False,identity_coefficients_used=False)

def regular_chart(initial,faces,A,values,native_uv,center,scale):
    """Exact anchors with a surface metric/log-area energy, avoiding flat cells.

    Trial inverted states use a C2 continuation of log at numeric precision;
    the returned map must be above that threshold, on the exact log branch.
    This continuation is solver machinery, never delivered field behavior.
    """
    rest=(native_uv-native_uv.mean(0))/scale;rest[:,0]*=-1
    t=rest[faces];R=np.stack([t[:,1]-t[:,0],t[:,2]-t[:,0]],axis=2)
    area=np.linalg.det(R)
    if np.any(area<=0):raise ValueError('Native lip chart orientation/topology changed')
    inverse=np.linalg.inv(R);w=area/area.sum()
    n=len(rest);E=np.kron(A,np.eye(2));rhs=((values-A.sum(1)[:,None]*center)/scale).reshape(-1)
    # Affine is a positive initial chart only; exact nonlinear anatomical
    # constraints and nondegenerate surface metric define the returned result.
    target=(values-A.sum(1)[:,None]*center)/scale
    scaffold=np.column_stack([np.ones(n),rest])
    fit=np.linalg.lstsq(A@scaffold,target,rcond=None)[0]
    start=(scaffold@fit).reshape(-1)
    threshold=np.sqrt(np.finfo(float).eps)
    def energy(x):
        p=x.reshape(n,2);tri=p[faces]
        edge=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)
        J=edge@inverse;d=np.linalg.det(J)
        q=(d-threshold)/threshold;safe=np.maximum(d,threshold)
        log=np.where(d>=threshold,np.log(safe),np.log(threshold)+q-.5*q*q)
        derivative=np.where(d>=threshold,1/safe,(1-q)/threshold)
        value=float(np.sum(w*(np.sum(J*J,axis=(1,2))-2*log)))
        cofactor=np.stack([J[:,1,1],-J[:,1,0],-J[:,0,1],J[:,0,0]],axis=1).reshape(-1,2,2)
        grad_edge=(2*J-2*derivative[:,None,None]*cofactor)@inverse.transpose(0,2,1)*w[:,None,None]
        grad=np.zeros_like(p)
        np.add.at(grad,faces[:,0],-grad_edge[:,:,0]-grad_edge[:,:,1])
        np.add.at(grad,faces[:,1],grad_edge[:,:,0]);np.add.at(grad,faces[:,2],grad_edge[:,:,1])
        return value,grad.reshape(-1)
    with threadpool_limits(limits=1):
        result=minimize(energy,start,jac=True,method='SLSQP',
            constraints=[dict(type='eq',fun=lambda x:E@x-rhs,jac=lambda x:E)],options=dict(ftol=1e-10,maxiter=500))
    mapped=result.x.reshape(n,2)*scale+center
    determinant=areas(mapped,faces)/(scale*scale*area)
    residual=float(np.max(abs(A@mapped-values),initial=0))
    if not result.success or np.any(determinant<=threshold) or residual>1e-9:
        raise ValueError('Regular surface chart not accepted: '+str(dict(reason=result.message,minimum_determinant=float(determinant.min()),anchor_residual=residual)))
    return mapped,dict(solver='Native surface metric plus exact log-area energy, exact anatomical embeddings; no collapsed-chart acceptance',
        iterations=int(result.nit),anchor_residual=residual,all_triangles_positive=True,
        minimum_relative_area=float(determinant.min()),numeric_trial_continuation_used_in_result=False,
        maximum_uv_correction=float(np.linalg.norm(mapped-initial,axis=1).max()),source_geometry_modified=False,identity_coefficients_used=False)
