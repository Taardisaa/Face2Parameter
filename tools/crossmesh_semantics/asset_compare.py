"""Direct installed head-prefab mesh comparison; never substitutes rest mesh for live pose."""
from pathlib import Path
import json
import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler
from analyze import ROOT, read, bound, require


def compare(output):
    evidence_path=Path('C:/Users/13666/Workspace/HS2Mod/tools/material_visibility_audit/asset_evidence.json')
    evidence=read(evidence_path)
    live_path=Path('C:/Users/13666/Workspace/HS2Mod/artifacts/infrastructure_live_20261005/abmx_lowered_v2/live_cases.json')
    source_path=Path(read(live_path)['cases'][0]['source_history']['geometry']['path'])
    source=read(source_path)
    rows=[]; bindings=[bound(evidence_path),bound(live_path),bound(source_path)]
    for bundle in evidence['bundles']:
        renderers=[r for r in bundle['renderers'] if r['prefab']=='p_cf_head_02']
        if not renderers: continue
        binding=bound(bundle['path']); require(binding['sha256']==bundle['sha256'],'Installed bundle changed from asset evidence')
        bindings.append(binding)
        env=UnityPy.load(bundle['path']); objects={o.path_id:o for o in env.objects}
        for renderer in renderers:
            path=renderer['mesh_pointer']['path_id']; obj=objects[path]; data=obj.read()
            handler=MeshHandler(data); handler.process()
            matching=[m for m in source['meshes'] if m['mesh_name']==renderer['mesh_name'] and '/n_head_cf2_c1[0]/' in m['renderer_path']]
            require(len(matching)==1,'Ambiguous live/asset renderer correspondence')
            live=matching[0]
            arrays={'vertices':handler.m_Vertices,'normals':handler.m_Normals,'tangents':handler.m_Tangents,
                    'uv':handler.m_UV0,'uv2':handler.m_UV1,'triangles':handler.m_IndexBuffer,
                    'bone_indices':handler.m_BoneIndices,'bone_weights':handler.m_BoneWeights,
                    'bindposes':[[getattr(matrix,f'e{i}{j}') for i in range(4) for j in range(4)] for matrix in data.m_BindPose]}
            comparisons={}
            for key,value in arrays.items():
                a=np.asarray(value); b=np.asarray(live['source'][key])
                if value is None and b.size==0:
                    comparisons[key]={'asset_channel_absent':True,'live_channel_empty':True,
                                      'absent_channel_compatible':True,'float32_or_int64_exact_equal':False,
                                      'uv0_substitution_claimed':False}; continue
                if key=='triangles': a=a.reshape(-1); b=b.reshape(-1)
                if a.shape!=b.shape:
                    comparisons[key]={'shape_asset':list(a.shape),'shape_live':list(b.shape),'equal':False}; continue
                integer=key in ('triangles','bone_indices')
                a=a.astype(np.int64 if integer else np.float32); b=b.astype(a.dtype)
                comparisons[key]={'shape':list(a.shape),'float32_or_int64_exact_equal':bool(np.array_equal(a,b)),
                                  'max_abs_difference':float(np.max(abs(a.astype(float)-b.astype(float))))}
            cached=ROOT/'data/hs2_head/head_2/submeshes'/f'{renderer["mesh_name"]}.npz'
            cache_comparison={}
            if cached.exists():
                bindings.append(bound(cached))
                with np.load(cached,allow_pickle=False) as z:
                    for key,cached_key in [('vertices','verts'),('triangles','faces'),('uv','uv'),('uv2','uv1')]:
                        a=np.asarray(live['source'][key]).astype(np.int64 if key=='triangles' else np.float32)
                        b=z[cached_key]
                        if key=='triangles': b=b.reshape(-1)
                        cache_comparison[key]={'shape_live':list(a.shape),'shape_cache':list(b.shape),
                                               'exact_equal':bool(np.array_equal(a,b))}
            rows.append({'renderer_path':live['renderer_path'],'mesh_name':live['mesh_name'],
                         'live_source_geometry_sha256':live['source_geometry_sha256'],
                         'prefab':renderer['prefab'],'asset_mesh_path_id':path,'asset_renderer_path_id':renderer['renderer_path_id'],
                         'direct_installed_asset_arrays':comparisons,'cached_arrays_exact_equal':cache_comparison,
                         'asset_material_slots':renderer['material_slots'],
                         'same_live_pose_as_asset_rest_pose_claimed':False,'shader_visibility_certified':False})
    result={'scope':'ordered source arrays only, float32 readback comparison; no live-vs-rest pose equivalence',
            'inputs':bindings,'renderers':rows}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print([(r['mesh_name'],{k:v.get('float32_or_int64_exact_equal',False) for k,v in r['direct_installed_asset_arrays'].items()}) for r in rows])


if __name__=='__main__':
    import sys
    compare(sys.argv[1])
