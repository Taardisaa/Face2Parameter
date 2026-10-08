"""Show exact deleted face IDs on the complete, unchanged source model.

This is an explicitly requested diagnostic runtime replacement, not the native
base deliverable. It changes appearance/UVs only, never source geometry or rig.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.model_bridge.artifact import sha
from tools.model_bridge.game_import import request
from tools.model_bridge.surface import embed_texture


def make(args):
    args.out.mkdir(parents=True,exist_ok=False)
    full=json.loads(args.source.read_text(encoding='utf-8'))
    trimmed=json.loads(args.trim.read_text(encoding='utf-8'))
    design=json.loads(args.neck_design.read_text(encoding='utf-8'))
    vertices=np.asarray(full['vertices'],np.float32)
    triangles=np.asarray(full['triangles'],int).reshape(-1,3)
    retained=np.asarray(trimmed['trim']['original_face_ids'],int)
    mapping=np.asarray(trimmed['trim']['original_vertex_ids'],int)
    if not np.array_equal(vertices[mapping],np.asarray(trimmed['vertices'],np.float32)):
        raise ValueError('Trimmed vertices are not literal subsets of full source')
    if not np.array_equal(triangles[retained],mapping[np.asarray(trimmed['triangles']).reshape(-1,3)]):
        raise ValueError('Original face IDs do not reproduce trimmed topology')
    initial=np.setdiff1d(np.arange(len(triangles)),retained)
    extra=retained[design['posterior_cut']['posterior_overlap_triangle_indices_removed']]
    if np.intersect1d(initial,extra).size:raise ValueError('Deletion stages overlap')
    labels=np.zeros(len(triangles),int);labels[initial]=1;labels[extra]=2
    palette=np.asarray([[205,205,205],[245,55,55],[255,195,25]],float)
    xyz=vertices[triangles];normal=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-20)
    light=np.asarray([.3,.6,.7]);light/=np.linalg.norm(light)
    # Baked diagnostic shading keeps the complete volume legible with unlit
    # categorical face colors. This is not inferred identity appearance.
    intensity=.55+.45*np.abs(normal@light)
    size=int(np.ceil(np.sqrt(len(triangles))))
    pixels=np.full((size,size,4),255,np.uint8)
    ids=np.arange(len(triangles));x=ids%size;y=ids//size
    pixels[y,x,:3]=np.round(palette[labels]*intensity[:,None]).astype(np.uint8)
    texture=args.out/'deletion_colors.png';Image.fromarray(pixels).save(texture)
    uv=np.c_[(x+.5)/size,1-(y+.5)/size]
    mtl=args.out/'original_untrimmed.mtl'
    mtl.write_text(''.join(f'newmtl {name}\nKd {r/255:.8f} {g/255:.8f} {b/255:.8f}\nKa .2 .2 .2\nd 1\n\n'
        for name,(r,g,b) in zip(['retained_gray','initial_deleted_red','additional_deleted_yellow'],palette)),encoding='utf-8')
    obj=args.out/'original_untrimmed.obj'
    lines=['# Complete original source geometry; deleted faces are shown, not removed.',
        'mtllib original_untrimmed.mtl']
    lines.extend('v '+' '.join(repr(float(v)) for v in row) for row in vertices)
    lines.extend('vt '+' '.join(repr(float(v)) for v in row) for row in uv)
    names=['retained_gray','initial_deleted_red','additional_deleted_yellow'];last=-1
    for i,(face,label) in enumerate(zip(triangles,labels)):
        if label!=last:lines.append('usemtl '+names[label]);last=label
        lines.append('f '+' '.join(f'{v+1}/{i+1}' for v in face))
    obj.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    diagnostic=copy.deepcopy(full)
    diagnostic['surface']=dict(format='flame_corner_uv_v1',
        render_to_canonical=triangles.reshape(-1).tolist(),render_uv=np.repeat(uv,3,axis=0).tolist(),
        render_triangles=np.arange(triangles.size).tolist(),source_obj=dict(path=str(obj.resolve()),sha256=sha(obj)),
        texture=embed_texture(texture),policy='Diagnostic per-face colors; complete original canonical geometry and rig unchanged')
    path=args.out/'original_untrimmed_highlighted.json'
    path.write_text(json.dumps(diagnostic,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    report=dict(format='source_face_deletion_visualization_v1',full_source_sha256=sha(args.source),
        trimmed_source_sha256=sha(args.trim),neck_design_sha256=sha(args.neck_design),
        original_vertex_count=len(vertices),original_triangle_count=len(triangles),
        initial_deleted_count=len(initial),additional_deleted_count=len(extra),
        total_deleted_count=len(initial)+len(extra),initial_deleted_face_ids=initial.tolist(),
        additional_deleted_face_ids=extra.tolist(),canonical_geometry_modified=False,
        original_source_parameters_preserved=True,legend={'gray':'retained original faces','red':'initial neck/base trim','yellow':'additional posterior trim'},
        obj=str(obj.resolve()),diagnostic_source=str(path.resolve()))
    if args.show:
        if args.scale is None or args.translation is None:raise ValueError('Use the existing literal placement for runtime diagnosis')
        state=request(args.base,'POST',dict(path=str(path.resolve()),sha256=sha(path),
            scale=args.scale,translation=args.translation))
        assert np.array_equal(np.asarray(state['canonical_vertices'],np.float32),vertices)
        assert np.array_equal(np.asarray(state['canonical_triangles'],int).reshape(-1,3),triangles)
        assert np.array_equal(np.asarray(state['render_vertices'],np.float32),vertices[triangles.reshape(-1)])
        assert np.array_equal(np.asarray(state['render_triangles']),np.arange(triangles.size))
        report.update(runtime_loaded=True,runtime_canonical_arrays_literal=True,active=state['active'],
            scale=args.scale,translation=args.translation)
    (args.out/'deletion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['original_triangle_count','initial_deleted_count','additional_deleted_count','total_deleted_count','canonical_geometry_modified','obj']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--trim',type=Path,required=True)
    parser.add_argument('--neck-design',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--show',action='store_true');parser.add_argument('--base',default='http://127.0.0.1:43127')
    parser.add_argument('--scale',type=float);parser.add_argument('--translation',nargs=3,type=float)
    make(parser.parse_args())


if __name__=='__main__':main()
