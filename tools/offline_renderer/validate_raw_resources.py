"""Validate and hash exported raw native resources, retaining incomplete jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(path):
    path=Path(path)
    descriptor=path/'descriptor.json'
    data=json.loads(descriptor.read_text())
    if data['schema_version']!=1 or data['dimension'] not in (3,4):
        raise ValueError('Unsupported native texture descriptor')
    if data['dimension']==4 and data['array_size']!=1:
        raise ValueError('Texture3D cannot have array slices')
    expected=data['mip_levels']*data['array_size']
    rows=data['subresources']
    if len(rows)!=expected or {r['index'] for r in rows}!=set(range(expected)):
        raise ValueError('Missing or duplicate subresource')
    resources=[]
    for r in rows:
        if r['index']!=r['mip']+r['array_slice']*data['mip_levels']:
            raise ValueError('Subresource index/order mismatch')
        dimensions=[max(1,data[name]>>r['mip']) for name in ('width','height','depth')]
        if dimensions!=[r[name] for name in ('width','height','depth')]:
            raise ValueError('Mip dimensions mismatch')
        target=(path/r['path']).resolve()
        if target.parent!=path.resolve():
            raise ValueError('Subresource outside declared output directory')
        if r['bytes']!=r['slice_pitch']*r['depth'] or target.stat().st_size!=r['bytes']:
            raise ValueError('Raw byte count differs from native tight pitches')
        resources.append({**r,'sha256':sha(target)})
    return {'path':str(path.resolve()),'descriptor_sha256':sha(descriptor),
            'native_descriptor':data,'hashed_subresources':resources,
            'total_bytes':sum(r['bytes'] for r in rows),
            'shader_view_format_certified':False,
            'scope':'Raw complete mip/face/volume byte inventory. Typeless/SRGB shader-view interpretation and render frame association require separate state.'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh validation output required')
    records=[];failures=[]
    for path in sorted(args.root.iterdir()):
        if not path.is_dir():
            continue
        try:
            records.append(validate(path))
        except Exception as e:
            failures.append({'path':str(path.resolve()),'error':type(e).__name__+': '+str(e)})
    receipt={'schema_version':1,'root':str(args.root.resolve()),'resources':records,
             'failures':failures,'resource_count':len(records),
             'total_bytes':sum(r['total_bytes'] for r in records),
             'complete':bool(records) and not failures,'rgb_renderer_complete':False}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({key:receipt[key] for key in ('resource_count','total_bytes','complete')}))
    return 0 if receipt['complete'] else 2

if __name__=='__main__':
    raise SystemExit(main())
