"""Explicit level sampling of GPU-encoded 2D resources.

No inferred LOD, regenerated mip chain, anisotropy approximation or automatic
sRGB guessing. D3D11 backend handles the complete native sampler paths.
"""
import numpy as np
from .dxbc import UnsupportedShader

def _index(values,size,mode):
    if mode=='Repeat':
        return values % size
    if mode=='Clamp':
        return np.clip(values,0,size-1)
    if mode=='Mirror':
        values=values%(2*size)
        return np.where(values<size,values,2*size-1-values)
    raise UnsupportedShader('Unsupported address mode '+mode)

def level_2d(pixels,uv,*,filter_mode,wrap_u,wrap_v):
    """Top-row-first source bytes; UV V=0 samples top row in raw D3D resources."""
    pixels=np.asarray(pixels,np.float32)
    if pixels.ndim!=3 or pixels.shape[2]!=4:
        raise ValueError('Explicit H x W x 4 GPU sample-encoded pixels required')
    uv=np.asarray(uv,np.float32)
    height,width=pixels.shape[:2]
    if filter_mode=='Point':
        ix=_index(np.floor(uv[:,0]*width).astype(int),width,wrap_u)
        iy=_index(np.floor(uv[:,1]*height).astype(int),height,wrap_v)
        return pixels[iy,ix]
    if filter_mode!='Bilinear':
        raise UnsupportedShader('Explicit-level sampler supports point/bilinear only')
    xy=uv[:,:2]*[width,height]-.5
    low=np.floor(xy).astype(int)
    f=(xy-low).astype(np.float32)
    x0=_index(low[:,0],width,wrap_u);x1=_index(low[:,0]+1,width,wrap_u)
    y0=_index(low[:,1],height,wrap_v);y1=_index(low[:,1]+1,height,wrap_v)
    a=pixels[y0,x0]*(1-f[:,0:1])+pixels[y0,x1]*f[:,0:1]
    b=pixels[y1,x0]*(1-f[:,0:1])+pixels[y1,x1]*f[:,0:1]
    return a*(1-f[:,1:2])+b*f[:,1:2]
