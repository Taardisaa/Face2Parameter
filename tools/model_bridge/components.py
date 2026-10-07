"""Partition exact source FLAME surface by authored eyeball components.

Use contiguous material runs to preserve the complete original face-corner order.
Optional eye PNGs use the supplied source UV atlas, not native HS2 eye UVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

from .artifact import sha
from .attachment_audit import topology
from .surface import embed_texture


def add_components(payload, mask, left_texture=None, right_texture=None):
    if payload.get('format') != 'hs2_source_head_mesh_v3' or payload['surface']['format'] != 'flame_corner_uv_v1':
        raise ValueError('Exact source corner-UV artifact required')
    # Only the already installed, audited mask is deserialized.
    digest = sha(mask)
    if digest != 'ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492':
        raise ValueError('Audited source FLAME mask differs')
    with mask.open('rb') as stream:
        masks = pickle.load(stream, encoding='latin1')
    vertices = np.asarray(payload['vertices'], np.float32)
    faces = np.asarray(payload['triangles'], np.int64).reshape(-1,3)
    parts = topology(vertices, faces)
    sets = [set(c['vertices']) for c in parts['components']]
    eyes = [set(int(i) for i in masks[k]) for k in ('left_eyeball','right_eyeball')]
    if len(sets) != 3 or any(eye not in sets for eye in eyes) or eyes[0] & eyes[1]:
        raise ValueError('Authored eye masks do not equal distinct connected components')
    head = next(s for s in sets if s not in eyes)
    labels = np.full(len(vertices), -1, np.int32)
    for i, ids in enumerate([head,*eyes]):
        labels[sorted(ids)] = i
    if np.any(labels<0) or np.any(labels[faces] != labels[faces][:,:1]):
        raise ValueError('Unclassified vertex or cross-component source triangle')
    face_labels = labels[faces[:,0]]
    starts = np.r_[0,np.flatnonzero(face_labels[1:]!=face_labels[:-1])+1]
    ends = np.r_[starts[1:],len(faces)]
    runs = [{'first_triangle':int(a),'triangle_count':int(b-a),'component':int(face_labels[a])} for a,b in zip(starts,ends)]
    if len(runs)>64 or runs[0]['component']!=0:
        raise ValueError('Unsupported source material run count/order')
    metadata = {'format':'flame_connected_components_v1','source_mask':{'path':str(mask.resolve()),'sha256':digest},
        'components':[{'name':name,'canonical_vertex_ids':sorted(ids)} for name,ids in zip(('head','left_eyeball','right_eyeball'),[head,*eyes])],
        'material_runs':runs,'textures':{},'geometry_policy':'Original index connectivity; exact masks, no coordinate fitting/welding/rebinding',
        'mouth':'Original head mouth boundary retained; no inferred teeth or tongue',
        'rig_policy':'Original full FLAME weights/correctives remain; no pure-eye rigid approximation'}
    for name,path in [('left_eyeball',left_texture),('right_eyeball',right_texture)]:
        if path is not None:
            metadata['textures'][name] = embed_texture(path)
    payload['surface']['format'] = 'flame_component_uv_v1'
    payload['surface']['components'] = metadata
    payload['game_support'] = 'Exact canonical source geometry/rig/UV, component material runs and explicit external head/eye PNGs. No inferred albedo, native eye UV retargeting, teeth or tongue.'
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--mask',type=Path,required=True)
    parser.add_argument('--left-eye-texture',type=Path)
    parser.add_argument('--right-eye-texture',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    payload = json.loads(args.mesh.read_text())
    add_components(payload,args.mask,args.left_eye_texture,args.right_eye_texture)
    payload['surface']['component_input_artifact'] = {'path':str(args.mesh.resolve()),'sha256':sha(args.mesh)}
    text = json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n'
    if len(text.encode('utf-8'))>16*1024*1024:
        raise ValueError('Combined source/rig/PNG artifact exceeds the native 16 MiB limit')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as stream:
        stream.write(text)
    print(json.dumps({'artifact':str(args.out.resolve()),'sha256':sha(args.out),
                      'components':[c['name'] for c in payload['surface']['components']['components']],
                      'material_runs':len(payload['surface']['components']['material_runs']),'source_vertices_modified':False}))


if __name__=='__main__':
    main()
