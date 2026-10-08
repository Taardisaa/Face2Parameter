"""Conservative bottom-only source trim, keeping literal head/upper-neck geometry."""
from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path
import pickle

import numpy as np

from tools.model_bridge.artifact import sha
from tools.model_bridge.game_import import request
from tools.model_bridge.rig import encode


def decode(row):
    if row['encoding']!='float32_le_base64':raise ValueError('Original float32 source rig required')
    return np.frombuffer(base64.b64decode(row['data'],validate=True),dtype='<f4').reshape(row['shape'])


def trim_bottom(source,fraction):
    result=copy.deepcopy(source);v=np.asarray(source['vertices'],np.float32)
    f=np.asarray(source['triangles'],int).reshape(-1,3)
    if not 0<fraction<=.1:raise ValueError('Conservative bottom fraction must be in (0, 0.1]')
    low=float(v[:,1].min());height=float(np.ptp(v[:,1]));cut=low+height*fraction
    remove=np.any(v[f,1]<cut,axis=1);keep=~remove
    mask_info=source['surface']['components']['source_mask'];mask_path=Path(mask_info['path'])
    if sha(mask_path)!=mask_info['sha256']:raise ValueError('Source anatomical mask changed')
    masks=pickle.loads(mask_path.read_bytes(),encoding='latin1')
    protected=np.unique(np.concatenate([masks[k] for k in ['face','scalp','left_ear','right_ear']]))
    if np.isin(f[remove],protected).any():raise ValueError('Bottom cut would touch face/scalp/ear triangles')
    ids=np.unique(f[keep]);inverse=np.full(len(v),-1,int);inverse[ids]=np.arange(len(ids))
    result['vertices']=v[ids].tolist();result['triangles']=inverse[f[keep]].reshape(-1).tolist()
    rig=result['rig']
    for name in ['v_shaped','weights','post_skin_offsets']:rig[name]=encode(decode(source['rig'][name])[ids])
    rig['posedirs']=encode(decode(source['rig']['posedirs']).reshape(36,len(v),3)[:,ids].reshape(36,-1))
    rig['policy']+='; conservative bottom-only literal subset; original full-source joints unchanged'
    old=source['surface'];surface=result['surface'];render_faces=np.asarray(old['render_triangles']).reshape(-1,3)[keep]
    render_ids=np.unique(render_faces);ri=np.full(len(old['render_to_canonical']),-1,int);ri[render_ids]=np.arange(len(render_ids))
    surface['render_to_canonical']=inverse[np.asarray(old['render_to_canonical'])[render_ids]].tolist()
    surface['render_uv']=np.asarray(old['render_uv'])[render_ids].tolist()
    surface['render_triangles']=ri[render_faces].reshape(-1).tolist()
    components=surface['components'];labels=np.full(len(ids),-1,int)
    for i,c in enumerate(components['components']):
        c['canonical_vertex_ids']=inverse[np.intersect1d(c['canonical_vertex_ids'],ids)].tolist()
        labels[c['canonical_vertex_ids']]=i
    face_labels=labels[inverse[f[keep,0]]];starts=np.r_[0,np.flatnonzero(np.diff(face_labels))+1];ends=np.r_[starts[1:],len(face_labels)]
    components['material_runs']=[dict(first_triangle=int(a),triangle_count=int(b-a),component=int(face_labels[a])) for a,b in zip(starts,ends)]
    result['trim']=dict(format='flame_bottom_only_trim_v1',original_vertex_ids=ids.tolist(),
        original_face_ids=np.flatnonzero(keep).tolist(),removed_original_face_ids=np.flatnonzero(remove).tolist(),
        bottom_fraction=fraction,source_cut_height=cut,face_scalp_ear_triangles_removed=0,
        retained_positions_literal=True,original_full_mesh_joints_retained=True,
        no_contraction_or_connector=True)
    result['shape_policy']='Only lowermost source base cropped; face/chin/scalp/ears and retained neck unchanged'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--fraction',type=float,default=.05);parser.add_argument('--show',action='store_true')
    parser.add_argument('--base',default='http://127.0.0.1:43127');parser.add_argument('--scale',type=float)
    parser.add_argument('--translation',type=float,nargs=3);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);source=json.loads(args.source.read_text(encoding='utf-8'))
    result=trim_bottom(source,args.fraction);result['trim']['parent_artifact']=dict(path=str(args.source.resolve()),sha256=sha(args.source))
    path=args.out/'source_bottom_only.json';path.write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n',encoding='utf-8')
    v=np.asarray(result['vertices'],np.float32);f=np.asarray(result['triangles'],int).reshape(-1,3)
    obj=args.out/'bottom_only.obj';obj.write_text('\n'.join(['# Bottom-only trim; literal original source vertices']+
        ['v '+' '.join(repr(float(x)) for x in row) for row in v]+
        ['f '+' '.join(str(x+1) for x in row) for row in f])+'\n',encoding='utf-8')
    report=dict(source=str(args.source.resolve()),source_sha256=sha(args.source),
        original_triangles=len(source['triangles'])//3,deleted_triangles=len(result['trim']['removed_original_face_ids']),
        retained_triangles=len(f),retained_vertices=len(v),trim=result['trim'],obj=str(obj.resolve()),
        seam_joined=False,native_base_asset_updated=False)
    if args.show:
        if args.scale is None or args.translation is None:raise ValueError('Use existing source placement for live update')
        state=request(args.base,'POST',dict(path=str(path.resolve()),sha256=sha(path),scale=args.scale,translation=args.translation))
        assert np.array_equal(np.asarray(state['canonical_vertices'],np.float32),v)
        assert np.array_equal(np.asarray(state['canonical_triangles'],int).reshape(-1,3),f)
        report.update(runtime_loaded=True,runtime_canonical_arrays_literal=True,game_restarted=False,
            cut_height_head_bind=args.scale*result['trim']['source_cut_height']+args.translation[1])
    (args.out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['original_triangles','deleted_triangles','retained_triangles','seam_joined','obj']}))


if __name__=='__main__':main()
