"""Deform captured normal/tangent frames with the actual bone matrices.

API frame exports and installed serialized sparse frames are distinct inputs.
RGB callers must supply loaded API frames; recovering bundle positions alone
does not certify normal-only entries after Unity's mesh loading path.
"""
from __future__ import annotations
import copy
import numpy as np
from tools.unity_parity.geometry import blendshape_delta, matrix

def frame_delta(mesh, attribute):
    field={'normals':'delta_normals','tangents':'delta_tangents'}[attribute]
    proxy={'source':{'vertices':np.asarray(mesh['source'][attribute])[:,:3]},
           'blendshapes':copy.deepcopy(mesh.get('blendshapes',[]))}
    for shape in proxy['blendshapes']:
        if shape['current_weight']==0:
            continue
        for frame in shape['frames']:
            if field not in frame:
                raise ValueError(f'Loaded API {field} missing for '+shape['name'])
            frame['delta_vertices']=frame[field]
    return blendshape_delta(proxy)[0]

def deform_attributes(mesh, transforms, *, loaded_api_frames):
    if not loaded_api_frames:
        raise ValueError('RGB normal/tangent deformation requires loaded API frames')
    source=mesh['source']
    indices=np.asarray(source['bone_indices'],int)
    weights=np.asarray(source['bone_weights'],float)
    ids=mesh['bone_transform_ids']
    safe=np.where(weights>0,indices,0)
    matrices=np.stack([matrix(transforms[i]['local_to_world']) @ matrix(bp)
                      for i,bp in zip(ids,source['bindposes'])])[:,:3,:3]
    result={}
    for attribute in ('normals','tangents'):
        base=np.asarray(source[attribute],float)
        if base.ndim!=2 or base.shape[0]!=len(indices) or base.shape[1] not in (3,4):
            raise ValueError('Complete source attribute required: '+attribute)
        direction=base[:,:3]+frame_delta(mesh,attribute)
        deformed=np.einsum('vkij,vj,vk->vi',matrices[safe],direction,weights)
        result['world_'+attribute]=deformed
        if attribute=='tangents':
            if base.shape[1]!=4:
                raise ValueError('Source tangent handedness is required')
            result['tangent_handedness']=base[:,3].copy()
    result['attribute_scope']='Loaded frame attributes with four-weight linear skinning; normalize in shader vertex stage. Tangent BakeMesh witness not yet exported.'
    return result
