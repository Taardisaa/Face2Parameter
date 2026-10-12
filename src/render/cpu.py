"""CPU triangle rasterization using recorded Unity cameras.

Geometry/visibility output, not an implementation of AIT/SSS shading.  CPU NDC
is deliberately used instead of Direct3D's reversed-Z render-texture matrix.
No auto-framing, per-candidate camera refit, CUDA or graphics context is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Raster:
    depth: np.ndarray
    triangle: np.ndarray
    barycentric: np.ndarray

    @property
    def mask(self):
        return self.triangle >= 0


def clip_vertices(world, camera):
    p = np.asarray(camera['projection'], float).reshape(4, 4)
    v = np.asarray(camera['world_to_camera'], float).reshape(4, 4)
    xyz = np.asarray(world, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite N x 3 world vertices required')
    return np.c_[xyz, np.ones(len(xyz))] @ (p @ v).T


def project(world, camera, width, height):
    clip = clip_vertices(world, camera)
    with np.errstate(divide='ignore', invalid='ignore'):
        ndc = clip[:, :3] / clip[:, 3:4]
    return np.c_[(ndc[:, 0]+1)*width/2, (1-ndc[:, 1])*height/2], ndc[:, 2]


def _clip_polygon(positions):
    # Carry original triangle barycentrics through homogeneous frustum clipping.
    poly = [(p, b) for p, b in zip(positions, np.eye(3))]
    for plane in ((1,0,0,1),(-1,0,0,1),(0,1,0,1),(0,-1,0,1),
                  (0,0,1,1),(0,0,-1,1)):
        if not poly:
            break
        result = []
        previous = poly[-1]
        dp = float(previous[0] @ plane)
        for current in poly:
            dc = float(current[0] @ plane)
            if (dp >= 0) != (dc >= 0):
                t = dp/(dp-dc)
                result.append((previous[0]+t*(current[0]-previous[0]),
                               previous[1]+t*(current[1]-previous[1])))
            if dc >= 0:
                result.append(current)
            previous, dp = current, dc
        poly = result
    return poly


def rasterize(world, triangles, camera, width, height, *, cull='off'):
    """Nearest surfaces plus perspective-correct original triangle coordinates.

    Pixel centers are x+.5/y+.5. Original IDs survive clipping. NDC depth is
    affine in screen barycentrics; attributes use reciprocal clip-W instead.
    Transparent/cutout material visibility is a separate shader pass.
    """
    if cull not in ('off', 'back', 'front') or width <= 0 or height <= 0:
        raise ValueError('Positive viewport and explicit cull mode required')
    clip = clip_vertices(world, camera)
    faces = np.asarray(triangles)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.array_equal(faces, faces.astype(int)):
        raise ValueError('Integer N x 3 triangles required')
    faces = faces.astype(int)
    if (faces < 0).any() or (faces >= len(clip)).any():
        raise ValueError('Triangle vertex index outside source mesh')
    out = Raster(np.full((height,width),np.inf),
                 np.full((height,width),-1,dtype=np.int32),
                 np.zeros((height,width,3)))
    for face_id, ids in enumerate(faces):
        polygon = _clip_polygon(clip[ids])
        for index in range(1,len(polygon)-1):
            rows = [polygon[0],polygon[index],polygon[index+1]]
            cv, original = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
            if (cv[:,3] <= 0).any():
                continue
            ndc = cv[:,:3]/cv[:,3:4]
            p = np.c_[(ndc[:,0]+1)*width/2,(1-ndc[:,1])*height/2]
            a,b,c = p
            denominator = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denominator)<1e-14:
                continue
            # Unity front-facing source winding becomes clockwise in top-down PNG.
            if (cull=='back' and denominator<0) or (cull=='front' and denominator>0):
                continue
            lo = np.maximum(np.ceil(p.min(0)-.5).astype(int),0)
            hi = np.minimum(np.floor(p.max(0)-.5).astype(int),[width-1,height-1])
            if (hi < lo).any():
                continue
            xx,yy = np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
            x,y=xx+.5,yy+.5
            w0=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denominator
            w1=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denominator
            screen=np.stack([w0,w1,1-w0-w1],-1)
            depth=screen @ ndc[:,2]
            inside=(screen>=-1e-12).all(-1)
            better=inside & (depth < out.depth[yy,xx])
            if not better.any():
                continue
            reciprocal=screen[better]/cv[:,3]
            original_bary=(reciprocal @ original)/reciprocal.sum(-1,keepdims=True)
            iy,ix=yy[better],xx[better]
            out.depth[iy,ix]=depth[better]
            out.triangle[iy,ix]=face_id
            out.barycentric[iy,ix]=original_bary
    return out


def interpolate(raster, triangles, attributes):
    """Interpolate source UVs/normals/colors; background stays zero."""
    attributes=np.asarray(attributes)
    if attributes.ndim==1:
        attributes=attributes[:,None]
    output=np.zeros((*raster.depth.shape,attributes.shape[1]))
    mask=raster.mask
    ids=np.asarray(triangles)[raster.triangle[mask]]
    output[mask]=np.einsum('ni,nij->nj',raster.barycentric[mask],attributes[ids])
    return output
