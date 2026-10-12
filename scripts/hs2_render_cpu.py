"""CPU source-state geometry and fixed-camera visibility export.

This command never starts/contact HS2, allocates CUDA, or pretends diagnostic
colors reproduce game materials. Asset frames are recovered from installation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.hs2_offline_state import CapturedHeadForward
from src.render.cpu import rasterize, project
from tools.unity_parity.geometry import vertex_errors

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--context',type=Path,required=True)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--patch',type=Path)
    ap.add_argument('--native',type=Path)
    ap.add_argument('--apply-count',type=int,default=1)
    ap.add_argument('--source-reference',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resolution-scale',type=float,default=1.)
    a=ap.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh output directory required')
    geometry,context,capture=read(a.geometry),read(a.context),read(a.capture)
    forward=CapturedHeadForward(geometry,context)
    patches=read(a.patch)['bones'] if a.patch else []
    native=read(a.native) if a.native else None
    result=forward.evaluate(patches,native=native,apply_count=a.apply_count)
    width=round(capture['width']*a.resolution_scale)
    height=round(capture['height']*a.resolution_scale)
    camera=capture['capture_camera']
    if not 0<a.resolution_scale<=4:
        raise ValueError('Explicit resolution scale in (0,4] required')
    reference=read(a.source_reference) if a.source_reference else None
    refs={r['renderer_path']:r for r in reference['surfaces']} if reference else {}
    merged_depth=np.full((height,width),np.inf)
    mesh_ids=np.full((height,width),-1,np.int32)
    triangles=np.full((height,width),-1,np.int32)
    bary=np.zeros((height,width,3))
    receipt={'schema_version':1,'device':'CPU; no game or GPU context',
      'appearance_compatible':False,'scope':'source-state deformation, fixed actual camera, geometric visibility; material transparency/alpha/shading not certified',
      'inputs':{str(p.resolve()):sha(p) for p in (a.geometry,a.context,a.capture,a.patch,a.native,a.source_reference) if p},
      'asset_provenance':forward.asset_provenance,'effect_dependency_audit':result['effect_decisions'],
      'source_provenance':forward.source_provenance,
      'apply_count':a.apply_count,'geometry_capture_frame':geometry['frame_count'],
      'source_context_frame':forward.identity['frame_count'],'same_frame_context':geometry['frame_count']==forward.identity['frame_count'],
      'surfaces':[],'normalized_tolerance':1e-5,'width':width,'height':height}
    a.out.mkdir(parents=True)
    for index,(surface,mesh) in enumerate(zip(result['surfaces'],forward.meshes)):
        pixels,depth=project(surface['world_vertices'],camera,width,height)
        np.savez_compressed(a.out/f'surface_{index:02d}.npz',
            world_vertices=surface['world_vertices'],head_vertices=surface['head_vertices'],
            triangles=surface['triangles'],pixels=pixels,ndc_depth=depth)
        row={'index':index,'renderer_path':surface['renderer_path'],'mesh_name':surface['mesh_name'],
             'source_geometry_sha256':mesh['source_geometry_sha256'],'vertex_count':len(pixels),
             'active_blendshapes':surface['active_blendshapes'],'rendered':mesh['enabled'] and mesh['active_in_hierarchy']}
        if refs:
            key='candidate' if patches or native is not None else 'reference'
            error=vertex_errors(np.asarray(refs[surface['renderer_path']][key]),surface['head_vertices'])
            row['source_reference_error']=error
            row['source_reference_gate_passed']=error['max_normalized'] is not None and error['max_normalized']<=1e-5
        receipt['surfaces'].append(row)
        if not row['rendered']:
            continue
        r=rasterize(surface['world_vertices'],surface['triangles'],camera,width,height,cull='off')
        front=r.depth<merged_depth
        merged_depth[front]=r.depth[front]; mesh_ids[front]=index
        triangles[front]=r.triangle[front]; bary[front]=r.barycentric[front]
    colors=np.array([[73,176,223],[79,157,199],[58,75,100],[208,156,88],[188,188,188],
                     [66,146,119],[213,91,100],[230,215,165],[160,80,90],[82,66,96]],np.uint8)
    image=np.full((height,width,3),127,np.uint8)
    mask=mesh_ids>=0
    image[mask]=colors[mesh_ids[mask]%len(colors)]
    Image.fromarray(image).save(a.out/'geometric_visibility.png')
    np.savez_compressed(a.out/'visibility.npz',mesh_index=mesh_ids,triangle=triangles,
                        barycentric=bary,ndc_depth=merged_depth)
    receipt['all_source_reference_gates_passed']=all(r.get('source_reference_gate_passed',False) for r in receipt['surfaces']) if refs else None
    (a.out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(a.out.resolve()),'surfaces':len(result['surfaces']),
                      'source_reference_passed':receipt['all_source_reference_gates_passed'],
                      'appearance_compatible':False}))
    return 0 if not refs or receipt['all_source_reference_gates_passed'] else 2

if __name__=='__main__':
    raise SystemExit(main())
