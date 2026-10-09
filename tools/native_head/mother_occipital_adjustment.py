"""Local upward/forward occipital authoring after the intact-neck stage.

An explicit user-directed asset edit, not HS2 parameter inference. Anatomical
native support and the recorded collar interface are hard locks. All mesh fields
except positions, and all original vertex/face identities, remain unchanged.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tools.model_bridge.artifact import sha
from tools.native_head.mother_neck_transition import review
from tools.native_head.mother_surface_warp import SurfaceWarp
from tools.native_head.mother_template_inputs import save_json, source_file


def smooth(t):
    t=np.clip(t,0,1)
    return t**3*(10-15*t+6*t*t)


def author(candidate, out, up_fraction=.18, forward_fraction=.15):
    if out.exists():
        raise FileExistsError('Use a fresh output')
    receipt=json.loads((candidate/'receipt.json').read_text())
    if sha(receipt['candidate_arrays']['path'])!=receipt['candidate_arrays']['sha256']:
        raise ValueError('Candidate changed')
    neck=receipt['neck_transition']
    for row in (neck['domain'],neck['actual_head_body']):
        if sha(row['path'])!=row['sha256']:
            raise ValueError('Neck provenance changed')
    domain=json.loads(Path(neck['domain']['path']).read_text())
    arrays=dict(np.load(candidate/'o_head_candidate.npz',allow_pickle=False))
    v=arrays['verts']
    names=arrays['bone_names'].tolist()
    ear_bones=[i for i,n in enumerate(names) if 'Ear' in n]
    support=(arrays['bone_w']*np.isin(arrays['bone_idx'],ear_bones)).sum(1)
    ears=v[support>.5]
    span=np.ptp(ears[:,1])
    low=float(v[domain['lower_native_render_ids'],1].max())
    full_low=float(ears[:,1].mean())
    full_high=float(ears[:,1].max()+.3*span)
    top=float(v[:,1].max())
    front=float(ears[:,2].min())
    rear=float(v[:,2].min())
    if not low<full_low<full_high<top or not rear<front:
        raise ValueError('Invalid ear/cranial authoring bounds')
    weight=smooth((front-v[:,2])/(front-rear))
    weight*=smooth((v[:,1]-low)/(full_low-low))
    weight*=1-smooth((v[:,1]-full_high)/(top-full_high))
    locked=np.unique(np.r_[domain['lower_native_render_ids'],
                            domain['protected_feature_render_ids']]).astype(int)
    weight[locked]=0
    delta=np.array([0.,up_fraction*span,forward_fraction*span])
    final=v+weight[:,None]*delta
    final[weight==0]=v[weight==0]
    regions=json.loads((Path(receipt['inputs']['path']).parent/'native_regions.json').read_text())
    for group in regions['identical_bind_position_and_skin_groups']:
        if not np.all(final[group]==final[group[0]]):
            raise ValueError('Physical UV copies separated')
    field=SurfaceWarp(arrays['original_vertices'],final,arrays['faces'])
    field.vertex_gradients({int(k):v for k,v in regions['graph_aliases'].items()})
    out.mkdir(parents=True)
    np.savez_compressed(out/'o_head_candidate.npz',**{**arrays,'verts':final})
    np.savez_compressed(out/'occipital_field.npz',weight=weight,delta=delta,
                        before_vertices=v,protected_render_ids=locked)
    updated=copy.deepcopy(receipt)
    updated.update(candidate_arrays=source_file(out/'o_head_candidate.npz'),
        code=source_file(Path(__file__)),occipital_adjustment=dict(
            input_candidate=source_file(candidate/'receipt.json'),
            field=source_file(out/'occipital_field.npz'),ear_height=span,
            full_displacement=delta.tolist(),up_fraction=up_fraction,
            forward_fraction=forward_fraction,
            bounds=dict(lower=low,full_low=full_low,full_high=full_high,top=top,
                        front=front,rear=rear),
            method='C2 spatial taper; upward/forward rear scalp displacement only',
            face_ears_and_lower_collar_unchanged=True,global_translation=False,
            topology_unchanged=True),installed=False,game_mutated=False,deliverable=False)
    save_json(out/'receipt.json',updated)
    actual=json.loads(Path(neck['actual_head_body']['path']).read_text(encoding='utf-8-sig'))
    body=next(m for m in actual['meshes'] if m['mesh_name']=='o_body_cf')
    review(v,final,arrays['faces'],body,out)
    print(json.dumps(dict(output=str(out.resolve()),posterior_only=True,
                          face_ears_and_interface_unchanged=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--up-fraction',type=float,default=.18)
    p.add_argument('--forward-fraction',type=float,default=.15)
    a=p.parse_args()
    if not np.isfinite([a.up_fraction,a.forward_fraction]).all():
        raise ValueError('Finite displacement required')
    author(a.candidate.resolve(),a.out.resolve(),a.up_fraction,a.forward_fraction)
