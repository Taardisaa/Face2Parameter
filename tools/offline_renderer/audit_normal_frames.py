"""Preserve asset/API frame and skinned-normal disagreements without fitting."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.hs2_offline_state import CapturedHeadForward
from tools.unity_parity.geometry import blendshape_delta

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def delta_normals(mesh,frames):
    proxy=copy.deepcopy(mesh)
    proxy['source']['vertices']=proxy['source']['normals']
    proxy['blendshapes']=copy.deepcopy(frames)
    for active,shape in zip(mesh['blendshapes'],proxy['blendshapes']):
        if active['name']!=shape['name']:
            raise ValueError('Shape order mismatch')
        shape['current_weight']=active['current_weight']
        for frame in shape['frames']:
            frame['delta_vertices']=frame['delta_normals']
    return blendshape_delta(proxy)[0]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir',type=Path,required=True)
    ap.add_argument('--native-frame-export',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh audit output required')
    f=CapturedHeadForward(read(a.baseline_dir/'baseline_front_geometry.json'),read(a.baseline_dir/'context.json'))
    o=f.evaluate()
    mesh=next(m for m in f.meshes if m['mesh_name']=='o_head')
    donors=[m for m in read(a.native_frame_export)['meshes'] if m['source_geometry_sha256']==mesh['source_geometry_sha256']]
    if len(donors)!=1:
        raise ValueError('Unique full source-hash matched native frame export required')
    donor=donors[0]
    frame_rows=[]
    for asset,api in zip(mesh['blendshapes'],donor['blendshapes']):
        if asset['name']!=api['name']:
            raise ValueError('Asset/API shape inventory mismatch')
        for af,nf in zip(asset['frames'],api['frames']):
            row={'shape':asset['name'],'weight':af['weight']}
            for field in ('delta_vertices','delta_normals','delta_tangents'):
                difference=np.linalg.norm(np.array(af[field])-np.array(nf[field]),axis=1)
                row[field+'_max']=float(difference.max())
                row[field+'_changed_vertices']=int(np.count_nonzero(difference>1e-7))
            frame_rows.append(row)
    matrices=np.asarray([o['bone_world'][bone] @ np.asarray(bp).reshape(4,4)
                         for bone,bp in zip(mesh['bone_transform_ids'],mesh['source']['bindposes'])])[:,:3,:3]
    indices=np.asarray(mesh['source']['bone_indices'])
    weights=np.asarray(mesh['source']['bone_weights'])
    expected=np.asarray(mesh['baked']['normals']) @ Rotation.from_quat(mesh['renderer_rotation_xyzw']).as_matrix().T
    unit=lambda x:x/np.linalg.norm(x,axis=1,keepdims=True)
    checks=[]
    for name,frames in [('installed_asset_frames',mesh['blendshapes']),('earlier_native_api_frames',donor['blendshapes'])]:
        normals=np.asarray(mesh['source']['normals'])+delta_normals(mesh,frames)
        predicted=np.einsum('vkij,vj,vk->vi',matrices[indices],normals,weights)
        difference=np.linalg.norm(predicted-expected,axis=1)
        directions=np.linalg.norm(unit(predicted)-unit(expected),axis=1)
        checks.append({'input':name,'max_vector_difference':float(difference.max()),
                       'max_unit_direction_difference':float(directions.max()),
                       'worst_vertex':int(np.argmax(directions)),
                       'normal_compatibility_certified':False})
    output={'schema_version':1,'scope':'Disagreement audit. No cutoff, normal reconstruction, light adjustment or empirical correction applied.',
        'source_geometry_sha256':mesh['source_geometry_sha256'], 'asset_provenance':f.asset_provenance,
        'native_frame_export':str(a.native_frame_export.resolve()),
        'native_frame_export_sha256':hashlib.sha256(a.native_frame_export.read_bytes()).hexdigest(),
        'frame_comparisons':frame_rows,'normal_comparisons':checks,
        'current_frame_normals_certified':False,
        'remaining_source_gap':'Installed sparse frame arrays versus Unity loaded-frame API; prior export alone does not identify the complete current load/normal-skinning path.'}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(checks))

if __name__=='__main__':
    main()
