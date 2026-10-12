"""Same-frame loaded normal validation using saved BakeMesh evidence only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.render.skin_attributes import deform_attributes

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh validation output required')
    raw=args.geometry.read_bytes()
    geometry=json.loads(raw.decode('utf-8-sig'))
    if geometry['frame_count']!=geometry['frame_count_end']:
        raise ValueError('Snapshot crossed native frames')
    transforms={t['id']:t for t in geometry['transforms']}
    rows=[]
    for mesh in geometry['meshes']:
        actual=deform_attributes(mesh,transforms,loaded_api_frames=geometry['include_blendshape_frames'])
        expected=np.asarray(mesh['baked']['normals']) @ Rotation.from_quat(mesh['renderer_rotation_xyzw']).as_matrix().T
        unit=lambda n:n/np.linalg.norm(n,axis=1,keepdims=True)
        difference=np.linalg.norm(unit(actual['world_normals'])-unit(expected),axis=1)
        rows.append({'renderer_path':mesh['renderer_path'],'mesh_name':mesh['mesh_name'],
             'source_geometry_sha256':mesh['source_geometry_sha256'],
             'normal_direction_max':float(difference.max()),
             'normal_direction_rms':float(np.sqrt(np.mean(difference**2))),
             'normal_gate_passed':bool(difference.max()<=1e-5),
             'tangent_gate_passed':None})
    receipt={'schema_version':1,'geometry':str(args.geometry.resolve()),
         'geometry_sha256':hashlib.sha256(raw).hexdigest(),
         'captured_utc':geometry['captured_utc'],'frame_count':geometry['frame_count'],
         'tolerance':1e-5,'surfaces':rows,
         'all_normal_gates_passed':all(r['normal_gate_passed'] for r in rows),
         'tangent_bakemesh_witness_available':False,
         'rgb_renderer_complete':False,
         'scope':'Exact captured native bone TRS and loaded API frames, no game/GPU calls, no fitted normal correction. Does not certify a changed ABMX transition or full shading.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'surfaces':len(rows),'all_normal_gates_passed':receipt['all_normal_gates_passed']}))
    return 0 if receipt['all_normal_gates_passed'] else 2

if __name__=='__main__':
    raise SystemExit(main())
