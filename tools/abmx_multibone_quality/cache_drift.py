"""Bind actual early/late local and cache drift; does not identify writers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .run import sha, save, require, load_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),'New output required')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    results=[]
    for case in manifest['cases']:
        windows={w['window']:w for w in case['windows']}
        early,em,_=load_snapshot(windows['early']['geometry'],case['head_id'])
        late,lm,_=load_snapshot(windows['late']['geometry'],case['head_id'])
        a={t['id']:t for t in early['transforms']};b={t['id']:t for t in late['transforms']}
        names=set(case['names']); palette=set(em['bone_transform_ids']); scoped=set()
        root=early['character']['head_root_transform_id']
        for start in palette:
            key=start;visited=set()
            while key is not None:
                require(key in a and key not in visited,'Missing/cyclic skin ancestor')
                visited.add(key);scoped.add(key)
                if key==root:break
                key=a[key]['parent_id']
        differences=[]
        for key in sorted(scoped):
            require(key in b and a[key]['name']==b[key]['name'] and a[key]['path']==b[key]['path'],'Actual ancestor identity differs')
            row={'name':a[key]['name'],'path':a[key]['path'],'transform_id':key,'selected_group':a[key]['name'] in names}
            changed=False
            for field in ('local_position','local_rotation_xyzw','local_scale'):
                delta=np.asarray(b[key][field])-np.asarray(a[key][field])
                row[field+'_max_abs_difference']=float(np.abs(delta).max())
                if np.any(delta!=0):changed=True
            if changed:differences.append(row)
        ab={v['name']:v for v in early['abmx_runtime']['bones'] if v['name'] in names}
        bb={v['name']:v for v in late['abmx_runtime']['bones'] if v['name'] in names}
        require(set(ab)==set(bb)==names,'Actual requested cache group incomplete')
        caches=[]
        for name in sorted(names):
            av=ab[name]['runtime_baseline']['fields'];bv=bb[name]['runtime_baseline']['fields']
            require(set(av)==set(bv),'Cache field scope changed')
            changed=[field for field in av if av[field]!=bv[field]]
            caches.append({'bone_name':name,'changed_fields':changed,
                'early_fields':av,'late_fields':bv,
                'parameter_values_identical':all(ab[name][f]==bb[name][f] for f in ('scale','length','position','rotation'))})
        results.append({'head_id':case['head_id'],'early_geometry_sha256':windows['early']['geometry']['sha256'],
            'late_geometry_sha256':windows['late']['geometry']['sha256'],
            'scoped_skin_ancestor_count':len(scoped),'changed_actual_local_count':len(differences),
            'changed_actual_locals':differences,'actual_selected_cache_differences':caches,
            'early_cursor':early.get('abmx_trace_cursor'),'late_cursor':late.get('abmx_trace_cursor'),
            'external_writers_identified':False,'all_anatomical_correspondence_validated':False})
    save(args.out,{'schema_version':1,'manifest_path':str(args.manifest.resolve()),'manifest_sha256':sha(args.manifest),
        'implementation_sha256':sha(__file__),'heads':results,
        'scope':'Exact actual skin-ancestor local and selected private cache differences; no scale/pose fit or causal writer attribution'})
    print(json.dumps([{'head_id':r['head_id'],'actual_changed_locals':r['changed_actual_local_count'],
                      'changed_cache_fields':{c['bone_name']:c['changed_fields'] for c in r['actual_selected_cache_differences']}}
                     for r in results]))


if __name__=='__main__':main()
