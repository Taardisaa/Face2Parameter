"""Read-only source asset correspondence audit; never refreshes shared caches.

Select each renderer by ID/path and head subtree. A supplied zipmod is inspected
as an explicitly named alternative asset source, not silently chosen as a cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from tools.abmx_replay.model import IDENTITY, modifier, serialized
from tools.abmx_replay.validate_trace import read, sha
from tools.native_radial_target.compiler import require


def array_report(actual, expected, dtype, tolerance=0.):
    a, b = np.asarray(actual, dtype=dtype), np.asarray(expected, dtype=dtype)
    def fingerprint(v):
        return hashlib.sha256(str(v.shape).encode('ascii')+v.tobytes(order='C')).hexdigest()
    result = {'actual_shape': list(a.shape), 'expected_shape': list(b.shape),
              'actual_canonical_sha256': fingerprint(a), 'expected_canonical_sha256': fingerprint(b),
              'canonical_dtype': str(a.dtype), 'tolerance': tolerance, 'shape_matched': a.shape == b.shape}
    if a.shape != b.shape:
        result.update(passed=False, max_abs=None, first_differing_index=None)
        return result
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'Nonfinite asset source arrays')
    diff = np.abs(a.astype(np.float64)-b.astype(np.float64))
    changed = np.argwhere(diff > tolerance)
    result.update(passed=not len(changed), exact_canonical_equal=np.array_equal(a,b),
                  max_abs=float(diff.max(initial=0)), first_differing_index=None if not len(changed) else changed[0].tolist())
    return result


def compare(mesh, data, bone_names):
    actual = mesh['source']
    specs = [('vertices','verts',np.float32,1e-6), ('triangles','faces',np.int64,0),
             ('bone_indices','bone_idx',np.int64,0), ('bone_weights','bone_w',np.float32,1e-7),
             ('bindposes','bindpose',np.float32,1e-6), ('normals','normals',np.float32,1e-6),
             ('tangents','tangents',np.float32,1e-6), ('uv','uv',np.float32,1e-6), ('uv2','uv1',np.float32,1e-6)]
    arrays = {}
    for src, cached, dtype, tol in specs:
        a = np.asarray(actual[src],dtype=dtype)
        if src=='triangles':a=a.reshape(-1,3)
        if src=='bindposes':a=a.reshape(-1,4,4)
        arrays[src] = array_report(a, data[cached], dtype, tol)
    palette = mesh['bone_names'] == bone_names
    required = ('vertices','triangles','bone_indices','bone_weights','bindposes')
    return {'deformation_correspondence_passed':palette and all(arrays[k]['passed'] for k in required),
            'all_exported_authored_arrays_passed':palette and all(v['passed'] for v in arrays.values()),
            'bone_palette_matched':palette,'actual_bone_names':mesh['bone_names'],'expected_bone_names':bone_names,
            'array_checks':arrays}


def select_head(snapshot):
    transforms={t['id']:t for t in snapshot['transforms']}
    require(len(transforms)==len(snapshot['transforms']),'Duplicate transform IDs')
    root=snapshot['character']['head_root_transform_id']
    require(root in transforms,'Source head root missing')
    selected=[];skipped=[];seen=set()
    for m in snapshot['meshes']:
        key=(m['renderer_id'],m['renderer_path'])
        require(key not in seen,'Duplicate renderer identity')
        seen.add(key)
        current=m['renderer_transform_id'];visited=set();in_head=False
        while current in transforms and current not in visited:
            if current==root:in_head=True;break
            visited.add(current);current=transforms[current]['parent_id']
        reason=None
        if not m['enabled'] or not m['active_in_hierarchy']:reason='disabled_or_inactive'
        elif not in_head:reason='outside_head_subtree'
        if reason:skipped.append({'mesh_name':m['mesh_name'],'renderer_path':m['renderer_path'],'reason':reason})
        else:selected.append(m)
    return selected,skipped


def asset_renderers(env,prefab):
    objects=list(env.objects);trans,go2t=_read_transforms(objects);rows=[]
    for obj in objects:
        if obj.type.name!='SkinnedMeshRenderer':continue
        renderer=obj.read()
        try:
            mesh=renderer.m_Mesh.read();transform=go2t[renderer.m_GameObject.path_id]
        except (KeyError,AttributeError):continue
        chain=_chain(trans,transform)
        if prefab not in chain:continue
        rows.append({'mesh_name':mesh.m_Name,'prefab':prefab,'renderer_path_id':obj.path_id,
                     'mesh_path_id':mesh.object_reader.path_id,'transform_path_id':transform,
                     'chain':chain,'data':_read_smr_mesh(mesh),
                     'bone_names':[trans[b.path_id]['name'] for b in renderer.m_Bones]})
    return rows


def audit(source_path,cache_dir,list_cache_path,archive=None,member=None):
    snapshot=read(source_path)
    require(snapshot['snapshot_kind']=='maker_live_skinned_geometry' and snapshot['frame_count']==snapshot['frame_count_end'],
            'Stable actual source snapshot required')
    for row in snapshot['abmx_runtime']['bones']:
        require(serialized(modifier({k:row[k] for k in IDENTITY}))==serialized(modifier(IDENTITY)),
                'Audit accepts clean identity source only; candidate geometry refused')
    head=snapshot['character']['head_id'];cache_dir=Path(cache_dir)
    require(cache_dir.name=='head_'+str(head),'Same-head cache explicitly required')
    skeleton_path=cache_dir/'skeleton.json';skeleton=read(skeleton_path)
    cached_list=read(list_cache_path);row=cached_list['by_cat']['210'][str(head)]
    bundle=Path(cached_list['ab_dir'])/row['MainAB'];prefab=row['MainData']
    alternatives=[('vanilla',asset_renderers(UnityPy.load(str(bundle)),prefab),{'path':str(bundle),'sha256':sha(bundle),'prefab':prefab})]
    if archive:
        require(member is not None and member=='abdata/'+row['MainAB'],'Archive member must match resolved head list bundle')
        with zipfile.ZipFile(archive) as z:
            blob=z.read(member);manifest=z.read('manifest.xml')
        alternatives.append(('explicit_zipmod',asset_renderers(UnityPy.load(blob),prefab),
            {'path':str(Path(archive).resolve()),'sha256':sha(archive),'member':member,
             'member_sha256':hashlib.sha256(blob).hexdigest(),'manifest_sha256':hashlib.sha256(manifest).hexdigest(),
             'manifest_utf8':manifest.decode('utf-8-sig'),'prefab':prefab}))
    meshes,skipped=select_head(snapshot);results=[];cache_hashes={str(skeleton_path.resolve()):sha(skeleton_path)}
    for mesh in meshes:
        name=mesh['mesh_name'];npz=cache_dir/('o_head_mesh.npz' if name=='o_head' else 'submeshes/'+name+'.npz')
        with np.load(npz,allow_pickle=False) as z:
            data={k:np.array(z[k]) for k in z.files}
        names=skeleton['skin_bone_names'] if name=='o_head' else [skeleton['bones'][str(pid)]['name'] for pid in data['skin_bone_pids']]
        result={'mesh_name':name,'renderer_id':mesh['renderer_id'],'renderer_path':mesh['renderer_path'],
                'actual_source_geometry_sha256':mesh['source_geometry_sha256'],'active_in_hierarchy':mesh['active_in_hierarchy'],
                'cache_npz':str(npz.resolve()),'cache_npz_sha256':sha(npz),'cache_comparison':compare(mesh,data,names),'asset_candidates':[]}
        cache_hashes[str(npz.resolve())]=sha(npz)
        for kind,rows,descriptor in alternatives:
            found=[r for r in rows if r['mesh_name']==name]
            if not found:
                result['asset_candidates'].append({'source':kind,'origin':descriptor,
                    'comparison':{'deformation_correspondence_passed':False,'all_exported_authored_arrays_passed':False},
                    'unsupported_reason':'No matching renderer inside exactly resolved prefab'})
                continue
            require(len(found)==1,'Ambiguous multiple renderers with same mesh name inside chosen prefab: '+name)
            r=found[0]
            result['asset_candidates'].append({'source':kind,'origin':descriptor,
                'renderer_path_id':r['renderer_path_id'],'mesh_path_id':r['mesh_path_id'],'chain':r['chain'],
                'comparison':compare(mesh,r['data'],r['bone_names'])})
        results.append(result)
    require(all(sha(p)==digest for p,digest in cache_hashes.items()),'Cache changed during read-only audit')
    return {'schema_version':1,'scope':'SOURCE identity asset arrays only; no candidate input or runtime target generation',
            'implementation_sha256':sha(__file__),'source_snapshot':{'path':str(Path(source_path).resolve()),'sha256':sha(source_path)},
            'head_id':head,'frame':snapshot['frame_count'],'list_cache':{'path':str(Path(list_cache_path).resolve()),'sha256':sha(list_cache_path)},
            'resolved_list_row':row,'cache_files_unchanged':cache_hashes,'meshes':results,'skipped':skipped,
            'same_head_cache_all_visible_deformation_correspondence':all(r['cache_comparison']['deformation_correspondence_passed'] for r in results),
            'failed_cached_meshes':[r['mesh_name'] for r in results if not r['cache_comparison']['deformation_correspondence_passed']],
            'cache_refreshed':False,'validator_or_tolerance_modified':False,'source_zipmod_priority_instrumented':False,
            'mesh_content_match_is_not_direct_runtime_loader_call_instrumentation':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-snapshot',required=True);p.add_argument('--cache-dir',required=True)
    p.add_argument('--list-cache',required=True);p.add_argument('--archive');p.add_argument('--member')
    p.add_argument('--out',required=True);args=p.parse_args()
    report=audit(args.source_snapshot,args.cache_dir,args.list_cache,args.archive,args.member)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'head_id':report['head_id'],'failed_cached_meshes':report['failed_cached_meshes'],'out':str(out)}))


if __name__=='__main__':main()
