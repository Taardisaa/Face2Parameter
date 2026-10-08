"""Build a real head zipmod from an exact source artifact and installed head2.

Assets remain local. No runtime overlay, renderer hiding or source plugin required.
Native rest matrices come from the already recovered HS2 shape implementation.
Attribute transfer authors UV/skin only; it does not fit source face positions.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_transforms, _read_smr_mesh
from src.hs2_mesh_deform import HeadRig, _fk_world, build_mesh
from tools.model_bridge.scan_accuracy import TriangleSurface
from tools.native_head.mesh import write_mesh,basis
from tools.native_head.rim import conform

GUID = 'codex.chenger.mica.nativehead'
MAIN_AB = 'chara/codex/chenger/head.unity3d'
PREFAB = 'p_cf_head_chenger_mica'
SLOT = 1


def sha(data):
    return hashlib.sha256(data).hexdigest()


def correspondence(points, vertices, faces):
    closest, ids, _ = TriangleSurface(vertices[faces]).closest(points)
    triangles = vertices[faces[ids]]; a = triangles[:, 0]
    ab, ac, ap = triangles[:, 1]-a, triangles[:, 2]-a, closest-a
    dot = lambda x,y: np.sum(x*y, axis=1)
    d00,d01,d11 = dot(ab,ab),dot(ab,ac),dot(ac,ac)
    denominator = d00*d11-d01*d01
    if (denominator<=1e-20).any():
        raise ValueError('Attribute projection selected a degenerate triangle')
    v = (d11*dot(ap,ab)-d01*dot(ap,ac))/denominator
    w = (d00*dot(ap,ac)-d01*dot(ap,ab))/denominator
    bary = np.maximum(np.column_stack([1-v-w,v,w]),0)
    bary /= bary.sum(1,keepdims=True)
    return ids,bary,closest


def transfer(points, vertices, faces, arrays, bone_count):
    ids,bary,closest = correspondence(points, vertices, faces)
    corners = faces[ids]
    skin = np.zeros((len(points), bone_count))
    for i in range(3):
        for j in range(4):
            np.add.at(skin, (np.arange(len(points)),arrays['bone_idx'][corners[:,i],j]),
                      bary[:,i]*arrays['bone_w'][corners[:,i],j])
    chosen = np.argsort(-skin,axis=1,kind='stable')[:,:4]
    weights = np.take_along_axis(skin,chosen,axis=1)
    weights /= weights.sum(1,keepdims=True)
    attrs = {k: np.sum(np.asarray(arrays[k])[corners]*bary[:,:,None],axis=1) for k in ['uv','uv1','colors']}
    return chosen,weights,attrs,dict(method='closest_triangle_barycentric_attribute_transfer',
        source_positions_modified=False, max_attribute_projection_distance=float(np.linalg.norm(points-closest,axis=1).max()))


def csv_bytes(category, row, name):
    stream = io.StringIO(newline=''); writer = csv.writer(stream,lineterminator='\n')
    writer.writerow([category]);writer.writerow([0]);writer.writerow([name]);writer.writerow(row.keys());writer.writerow(row.values())
    return stream.getvalue().encode('utf-8-sig')


def build(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source_bytes = args.source.read_bytes(); source = json.loads(source_bytes)
    if source.get('trim',{}).get('format')!='flame_authored_neck_trim_v1':
        raise ValueError('Existing retained, explicitly trimmed target mesh required')
    if source['geometry_mode']!='head_local' or not np.isfinite([args.scale,*args.translation]).all() or args.scale<=0:
        raise ValueError('Head-local source and positive uniform placement required')
    report = json.loads(args.audit.read_text(encoding='utf-8'))
    native = next(x for x in report['native'] if x['id']==2)
    bundle_path = Path(native['bundle']); bundle_bytes = bundle_path.read_bytes()
    if sha(bundle_bytes)!=native['prefab_audit']['bundle_sha256']:
        raise ValueError('Installed native template changed since audited source')
    env = UnityPy.load(bundle_bytes); objects=list(env.objects); transforms,go = _read_transforms(objects)
    members = {pid for pid in transforms if 'p_cf_head_02' in _chain(transforms,pid)}
    root = next(o for o in objects if o.type.name=='GameObject' and o.read().m_Name=='p_cf_head_02')
    renderers = {}
    for obj in objects:
        if obj.type.name=='SkinnedMeshRenderer' and go[obj.read().m_GameObject.path_id] in members:
            smr=obj.read();renderers[smr.m_GameObject.read().m_Name]=(obj,smr,_read_smr_mesh(smr.m_Mesh.read()))
    rig=HeadRig(2); native_arrays=renderers['o_head'][2]
    if not np.array_equal(rig.verts.astype('<f4'),native_arrays['verts']) or not np.array_equal(rig.faces,native_arrays['faces']):
        raise ValueError('Recovered rig cache does not match actual installed asset')
    neutral_world=_fk_world(rig,np.full(59,.5),None)
    world_by_name={rig.bones[pid]['name']:matrix for pid,matrix in neutral_world.items()}
    native_neutral,_=build_mesh(rig,np.full(59,.5),None)
    canonical=np.asarray(source['vertices'],float)*args.scale+np.asarray(args.translation)
    original_placed=canonical.copy(); neck=None; neck_factors=np.zeros(len(canonical))
    if args.native_neck is not None:
        if args.neck_descriptor is None:raise ValueError('Native neck requires its exact boundary descriptor')
        canonical,neck_factors,neck=conform(canonical,np.asarray(source['triangles'],int).reshape(-1,3),
            source['trim']['neck_ring_indices'],json.loads(args.native_neck.read_text(encoding='utf-8')),
            json.loads(args.neck_descriptor.read_text(encoding='utf-8')),args.neck_bandwidth)
        neck['native_snapshot_sha256']=sha(args.native_neck.read_bytes())
        neck['boundary_descriptor_sha256']=sha(args.neck_descriptor.read_bytes())
        (args.out/'neck_design.json').write_text(json.dumps(neck,indent=2)+'\n',encoding='utf-8')
        np.savez(args.out/'source_placement.npz',original=original_placed,authored=canonical,falloff=neck_factors,
                 triangles=np.asarray(source['triangles']).reshape(-1,3))
    components=source['surface']['components']['components']
    labels=np.full(len(canonical),-1,int)
    for i,component in enumerate(components):labels[component['canonical_vertex_ids']]=i
    faces=np.asarray(source['triangles'],int).reshape(-1,3)
    if (labels<0).any() or not np.all(labels[faces]==labels[faces[:,0,None]]):
        raise ValueError('Source connected components are inconsistent')
    # Each source eye becomes a real eye renderer, not part of o_head's skin.
    names=['o_head','o_eyebase_L','o_eyebase_R']; mesh_reports=[]; expected_meshes={}
    # Sign determines actual asset side; authored model left/right conventions differ.
    eye_components=sorted([1,2],key=lambda i:canonical[labels==i,0].mean())
    component_indices=[0,*eye_components]
    for name,component_index in zip(names,component_indices):
        obj,smr,template=renderers[name]
        keep=labels[faces[:,0]]==component_index
        source_ids=np.unique(faces[keep]); inverse=np.full(len(canonical),-1,int);inverse[source_ids]=np.arange(len(source_ids))
        target_faces=inverse[faces[keep]];target_vertices=canonical[source_ids]
        bones=[b.read().m_GameObject.read().m_Name for b in smr.m_Bones]
        matrices=np.array([world_by_name[n] for n in bones]);bindposes=np.linalg.inv(matrices)
        if name=='o_head':ref_vertices=native_neutral
        else:
            # The original half-sphere UV/pupil design is translated to the source
            # eye before attribute projection; full source eye positions remain exact.
            homogeneous=np.c_[template['verts'],np.ones(len(template['verts']))]
            mats=matrices@template['bindpose']
            ref_vertices=sum(template['bone_w'][:,j,None]*(homogeneous[:,None,:]@mats[template['bone_idx'][:,j]].transpose(0,2,1))[:,0,:3] for j in range(4))
            ref_vertices += target_vertices.mean(0)-ref_vertices.mean(0)
        indices,weights,attrs,policy=transfer(target_vertices,ref_vertices,template['faces'],template,len(bones))
        override_normals=None
        if name=='o_head' and neck is not None:
            # Native FaceRoot (before FaceRoot_s slider scaling) lives in the
            # common rig. Its unchanged parent frame matches body Head_s support.
            root_id=next(pid for pid,t in transforms.items() if pid in members and t['name']=='cf_J_FaceRoot')
            tree=obj.read_typetree();tree['m_Bones'].append(dict(m_FileID=0,m_PathID=root_id));obj.save_typetree(tree)
            bindposes=np.concatenate([bindposes,np.linalg.inv(world_by_name['cf_J_FaceRoot'])[None]],axis=0)
            dense=np.zeros((len(source_ids),len(bindposes)))
            for column in range(4):np.add.at(dense,(np.arange(len(source_ids)),indices[:,column]),weights[:,column])
            fade=neck_factors[source_ids];dense*=1-fade[:,None];dense[:,-1]=fade
            indices=np.argsort(-dense,axis=1,kind='stable')[:,:4];weights=np.take_along_axis(dense,indices,axis=1)
            weights/=weights.sum(1,keepdims=True)
            override_normals,_=basis(target_vertices,target_faces,attrs['uv'])
            remap=dict(zip(map(int,source_ids),range(len(source_ids))))
            for original_id,normal in zip(neck['ring_indices'],neck['target_normals']):
                override_normals[remap[original_id]]=normal
        mesh_reader=smr.m_Mesh.read().object_reader
        result=write_mesh(mesh_reader,target_vertices,target_faces,attrs['uv'],indices,weights,bindposes,
                          uv1=attrs['uv1'],colors=attrs['colors'],normals=override_normals)
        # Do not discard the just-appended serialized root when clearing weights.
        tree=obj.read_typetree();tree['m_BlendShapeWeights']=[]
        if name=='o_head' and neck is not None:tree['m_Bones'].append(dict(m_FileID=0,m_PathID=root_id))
        obj.save_typetree(tree)
        # UnityPy readers retain original bytes until the bundle is serialized.
        # Readback therefore happens on the reopened serialized bundle below.
        expected_meshes[name]=(target_vertices.astype('<f4'),target_faces,source_ids,indices,weights.astype('<f4'))
        mesh_reports.append(dict(name=name,**result,attribute_policy=policy,face_geometry_fit=False))
    for name,(obj,smr,_) in renderers.items():
        if name in names:continue
        # Incompatible eye membranes/lashes and oral parts stay as referenced,
        # explicitly deferred assets; no runtime hiding helper is required.
        mesh_reader=smr.m_Mesh.read().object_reader
        mesh_tree=mesh_reader.read_typetree()
        mesh_tree['m_IndexBuffer']=[]
        for submesh in mesh_tree['m_SubMeshes']:submesh['indexCount']=0
        mesh_tree['m_Shapes']=dict(vertices=[],shapes=[],channels=[],fullWeights=[])
        mesh_reader.save_typetree(mesh_tree)
        tree=obj.read_typetree();tree['m_BlendShapeWeights']=[];obj.save_typetree(tree)
    for obj in objects:
        if obj.type.name=='MonoBehaviour':
            value=obj.read()
            if value.m_GameObject.path_id==root.path_id and value.m_Script.read().m_ClassName=='FaceBlendShape':
                tree=obj.read_typetree();tree['m_Enabled']=0
                for name in ['EyebrowCtrl','EyesCtrl','MouthCtrl']:tree[name]['FBSTarget']=[]
                tree['MouthCtrl']['useAjustWidthScale']=0;obj.save_typetree(tree)
    tree=root.read_typetree();tree['m_Name']=PREFAB;root.save_typetree(tree)
    # Avoid CAB collisions with the installed vanilla bundle, retaining every
    # path ID and real serialized component reference within the private bundle.
    old_cab=next(k for k,v in env.file.files.items() if hasattr(v,'objects'))
    new_cab='CAB-'+sha((GUID+sha(source_bytes)).encode())[:32]
    old_res=old_cab+'.resS';new_res=new_cab+'.resS'
    for obj in objects:
        if obj.type.name in ['Texture2D','Mesh']:
            tree=obj.read_typetree();stream=tree.get('m_StreamData')
            if stream and old_res in stream.get('path',''):
                stream['path']=stream['path'].replace(old_res,new_res);obj.save_typetree(tree)
        elif obj.type.name=='AssetBundle':
            tree=obj.read_typetree();tree['m_Name']=MAIN_AB
            tree['m_AssetBundleName']=MAIN_AB
            tree['m_Container']=[(key.replace('p_cf_head_02.prefab',PREFAB+'.prefab'),value)
                                 for key,value in tree['m_Container']]
            obj.save_typetree(tree)
    env.file.files={k.replace(old_cab,new_cab):v for k,v in env.file.files.items()}
    for k,v in env.file.files.items():
        if hasattr(v,'name'):v.name=k
    output_bundle=env.file.save(packer='lz4')
    # Reload the serialized bundle and locate new prefab/actual o_head asset.
    restored=UnityPy.load(output_bundle)
    if not any(o.type.name=='GameObject' and o.read().m_Name==PREFAB for o in restored.objects):
        raise ValueError('Independent bundle lost the registered prefab')
    rtrans,rgo=_read_transforms(list(restored.objects));checked=set()
    for obj in restored.objects:
        if obj.type.name!='SkinnedMeshRenderer':continue
        smr=obj.read();name=smr.m_GameObject.read().m_Name
        if name not in expected_meshes or PREFAB not in _chain(rtrans,rgo[smr.m_GameObject.path_id]):continue
        readback=_read_smr_mesh(smr.m_Mesh.read());vertices,faces,ids,indices,weights=expected_meshes[name]
        if not all([np.array_equal(readback['verts'],vertices),np.array_equal(readback['faces'],faces),
                    np.array_equal(readback['bone_idx'],indices),np.array_equal(readback['bone_w'],weights)]):
            raise ValueError(f'{name} serialized geometry/skin roundtrip failed')
        np.savez(args.out/(name+'.npz'),**readback,source_canonical_ids=ids);checked.add(name)
    if checked!=set(names):raise ValueError('Serialized prefab is missing its actual source render parts')
    head={k:v for k,v in native['row'].items() if not k.startswith('_')}
    head.update(ID=str(SLOT),Name='程儿 MICA 原生底模（中性）',MainManifest='abdata',MainAB=MAIN_AB,MainData=PREFAB,Preset='')
    skin={k:v for k,v in native['compatible_skin_rows'][0].items() if not k.startswith('_')}
    skin.update(ID=str(SLOT),HeadID=str(SLOT),Name='程儿 原生皮肤')
    manifest=ET.Element('manifest',{'schema-ver':'1'})
    for k,v in dict(guid=GUID,name='Chenger MICA native head',version='0.1.0',author='Codex',
                    description='Actual neutral head asset. Expressions deferred. Locally generated; requires installed HS2 assets.').items():
        ET.SubElement(manifest,k).text=v
    ET.SubElement(manifest,'faceSkinInfo',{'skinID':str(SLOT),'headID':str(SLOT),'headGUID':GUID})
    package=args.out/'Chenger.MICA.NativeHead.zipmod'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.xml',ET.tostring(manifest,encoding='utf-8',xml_declaration=True))
        archive.writestr('abdata/'+MAIN_AB,output_bundle)
        archive.writestr('abdata/list/characustom/codex_chenger_fo_head_00.csv',csv_bytes(210,head,'codex_chenger_fo_head_00'))
        archive.writestr('abdata/list/characustom/codex_chenger_ft_skin_f_00.csv',csv_bytes(211,skin,'codex_chenger_ft_skin_f_00'))
    receipt=dict(format='hs2_native_head_asset_build_v1',package=str(package.resolve()),package_sha256=sha(package.read_bytes()),
        guid=GUID,slot=SLOT,main_ab=MAIN_AB,prefab=PREFAB,source_sha256=sha(source_bytes),template_sha256=sha(bundle_bytes),
        source_parameters_preserved=True,outside_neck_band_positions_modified=False,neck_band_authored=neck is not None,
        placement=dict(scale=args.scale,translation=args.translation),
        native_reference_face_values=[.5]*59,mesh_assets=mesh_reports,source_overlay_required=False,
        expression_support=False,neck_fit_completed=neck is not None,appearance='Native UV/skin attribute transfer; no inferred albedo',
        deferred_parts=['o_eyelashes','o_eyeshadow','o_namida','o_tooth','o_tang'],game_loading_verified=False)
    (args.out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(package=str(package.resolve()),guid=GUID,actual_head_mesh_authored=True,game_loading_verified=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--scale',type=float,required=True);parser.add_argument('--translation',type=float,nargs=3,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--native-neck',type=Path);parser.add_argument('--neck-descriptor',type=Path)
    parser.add_argument('--neck-bandwidth',type=float,default=.55)
    build(parser.parse_args())


if __name__=='__main__':main()
