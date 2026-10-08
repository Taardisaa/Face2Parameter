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
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import UnityPy
from PIL import Image

from scripts.hs2_extract_head import _chain, _read_transforms, _read_smr_mesh
from src.hs2_mesh_deform import HeadRig, _fk_world, build_mesh
from tools.model_bridge.scan_accuracy import TriangleSurface
from tools.native_head.mesh import write_mesh,basis

GUID = 'codex.chenger.mica.nativehead'
MAIN_AB = 'chara/codex/chenger/head.unity3d'
PREFAB = 'p_cf_head_chenger_mica'
SLOT = 1


def neutral_skin(native, out):
    """Native shader inputs with neutral authoring textures, no albedo claim.

    UV transfer cannot make anatomical markings on the vanilla texture faithful
    to another topology. Use a plain skin first instead of inventing lip/eye spots.
    """
    row=native['compatible_skin_rows'][0];path=Path(native['bundle']).parents[2]/row['MainAB']
    if not path.exists():
        # bundle is abdata/chara/38/...; its third parent is installed abdata.
        raise FileNotFoundError(path)
    env=UnityPy.load(str(path));old_cab=next(k for k,v in env.file.files.items() if hasattr(v,'objects'))
    new_cab='CAB-'+sha((GUID+'.skin.v1').encode())[:32]
    for obj in list(env.objects):
        if obj.type.name=='Texture2D':
            value=obj.read();name=value.m_Name
            if name==row['MainTex']:
                pixels=np.asarray(value.image)
                color=tuple(map(int,np.median(pixels.reshape(-1,4),axis=0)))
                value.set_image(Image.new('RGBA',(16,16),color),target_format=4)
                value.save()
            elif name==row['OcclusionMapTex']:
                value.set_image(Image.new('RGBA',(16,16),(255,255,0,255)),target_format=4);value.save()
            elif name==row['NormalMapTex']:
                # Installed texture uses DXT5nm: X in alpha, Y in green.
                value.set_image(Image.new('RGBA',(16,16),(255,128,128,128)),target_format=4);value.save()
            else:
                tree=obj.read_typetree();stream=tree.get('m_StreamData')
                if stream and old_cab in stream.get('path',''):
                    stream['path']=stream['path'].replace(old_cab,new_cab);obj.save_typetree(tree)
        elif obj.type.name=='AssetBundle':
            tree=obj.read_typetree();tree['m_Name']='chara/codex/chenger/skin.unity3d'
            tree['m_AssetBundleName']='chara/codex/chenger/skin.unity3d';obj.save_typetree(tree)
    env.file.files={k.replace(old_cab,new_cab):v for k,v in env.file.files.items()}
    for k,v in env.file.files.items():
        if hasattr(v,'name'):v.name=k
    data=env.file.save(packer='lz4')
    return data,dict(source_bundle=str(path),sha256=sha(path.read_bytes()),
        policy='Plain authored native skin inputs; inherited skin color; no inferred albedo or native anatomical texture mapping claim')


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
    if args.authored_neck is None and source.get('trim',{}).get('format')!='flame_authored_neck_trim_v1':
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
    authored = None
    if args.authored_neck is not None:
        if args.native_neck is not None:
            raise ValueError('Authored integrated geometry cannot use legacy neck deformation')
        authored = dict(np.load(args.authored_neck, allow_pickle=False))
        receipt_path = args.authored_neck.with_name('receipt.json')
        design = json.loads(receipt_path.read_text(encoding='utf-8'))
        validation = json.loads(args.authored_neck.with_name('validation.json').read_text())
        if not validation['pass_static_geometry'] or validation['geometry_sha256'] != design['geometry_sha256']:
            raise ValueError('Integrated placement has no passing final geometry acceptance')
        if (design['geometry_sha256'] != sha(args.authored_neck.read_bytes())
                or design['native_bundle_sha256'] != sha(bundle_bytes)
                or design['raw_sha256'] != source['source']['artifact_sha256']
                or design['source_image_sha256'] != source['source']['image_sha256']
                or design['manifest_sha256'] != source['source']['manifest_sha256']
                or design['scale'] != args.scale or design['translation'] != args.translation
                or not np.array_equal(authored['original_vertices'], source['vertices'])
                or not np.array_equal(authored['original_faces'], np.asarray(source['triangles']).reshape(-1,3))):
            raise ValueError('Integrated geometry does not match exact source/template/placement provenance')
        canonical = authored['vertices']
        neck = dict(method=design['rules'],geometry_sha256=design['geometry_sha256'],
                    authoring_receipt_sha256=sha(receipt_path.read_bytes()),
                    body_capture_sha256=design['body_capture_sha256'],
                    body_source_geometry_sha256=design['body_source_geometry_sha256'],
                    body_mesh_edited=False, source_shape_preserved_up_to_similarity=True,
                    similarity_matrix=design['similarity_matrix'],
                    native_interface_error=design['native_interface_error'])
    if args.native_neck is not None:
        from tools.native_head.rim import conform,transition
        if args.neck_descriptor is None:raise ValueError('Native neck requires its exact boundary descriptor')
        mask_info=source['surface']['components']['source_mask']
        mask_path=Path(mask_info['path']);mask_bytes=mask_path.read_bytes()
        if sha(mask_bytes)!=mask_info['sha256']:raise ValueError('Authored anatomical masks changed')
        masks=pickle.loads(mask_bytes,encoding='latin1')
        protected_ids=np.unique(np.concatenate([masks[k] for k in
            ['face','left_ear','right_ear','eye_region','forehead','lips','nose']]))
        protected=np.isin(source['trim']['original_vertex_ids'],protected_ids)
        canonical,neck_factors,neck=conform(canonical,np.asarray(source['triangles'],int).reshape(-1,3),
            source['trim']['neck_ring_indices'],json.loads(args.native_neck.read_text(encoding='utf-8')),
            json.loads(args.neck_descriptor.read_text(encoding='utf-8')),args.neck_bandwidth,protected)
        neck['authored_mask_sha256']=sha(mask_bytes)
        neck['native_snapshot_sha256']=sha(args.native_neck.read_bytes())
        neck['boundary_descriptor_sha256']=sha(args.neck_descriptor.read_bytes())
        np.savez(args.out/'source_placement.npz',original=original_placed,authored=canonical,falloff=neck_factors,
                 triangles=np.asarray(source['triangles']).reshape(-1,3))
    components=source['surface']['components']['components']
    original_labels=np.full(len(original_placed),-1,int)
    for i,component in enumerate(components):original_labels[component['canonical_vertex_ids']]=i
    if authored is None:
        labels=original_labels
        faces=np.asarray(source['triangles'],int).reshape(-1,3)
    else:
        faces=authored['faces']
        from tools.native_head.asset_placement import prepare
        labels,authored_normals,native_start,native_ids,collar_attrs,collar_policy = prepare(
            authored,design,original_labels,native_neutral,native_arrays,correspondence)
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
            if args.native_eyes:
                # Keep the game's own eye UV/pupil domain. Its front hemisphere
                # is centered on the source eyeball and uniformly sized to it.
                source_center=(target_vertices.min(0)+target_vertices.max(0))/2
                native_center=(ref_vertices.min(0)+ref_vertices.max(0))/2
                native_center[2]=ref_vertices[:,2].min()
                size=(target_vertices[:,0].max()-target_vertices[:,0].min())/(ref_vertices[:,0].max()-ref_vertices[:,0].min())
                target_vertices=(ref_vertices-native_center)*size+source_center
                if authored is not None:
                    matrix=np.asarray(design['similarity_matrix'])[:3,:3]/design['relative_uniform_scale']
                    target_vertices=(target_vertices-source_center)@matrix.T+source_center
                target_faces=template['faces'];source_ids=np.full(len(target_vertices),-1,int)
        attribute_points=original_placed[source_ids] if name=='o_head' and authored is None else target_vertices
        indices,weights,attrs,policy=transfer(attribute_points,ref_vertices,template['faces'],template,len(bones))
        if name!='o_head' and args.native_eyes:
            indices=template['bone_idx'];weights=template['bone_w']
            attrs={k:template[k] for k in ['uv','uv1','colors']}
            policy=dict(method='native_eye_hemisphere_uniform_placement',source_eyeball_center_preserved=True,
                        original_native_uv_and_skin_weights_preserved=True)
        override_normals=None
        if authored is not None:
            if not (name!='o_head' and args.native_eyes):override_normals=authored_normals[source_ids]
            if name=='o_head':
                # The game's body row uses Head_s and the head rim uses the
                # coincident FaceRoot_s. Keep the whole new connector on that
                # audited head-side support; facial retargeting remains deferred.
                seam_and_bridge = source_ids >= native_start
                root_index=bones.index('cf_J_FaceRoot_s')
                indices[seam_and_bridge]=root_index;weights[seam_and_bridge]=[1,0,0,0]
                local=inverse[native_ids]
                for key in attrs:
                    attrs[key][local]=collar_attrs[key]
                neck.update(native_interface_root='cf_J_FaceRoot_s',
                            native_interface_normals_preserved=True,
                            transition_inside_actual_o_head=True)
        if name=='o_head' and neck is not None and authored is None:
            retained_count=len(source_ids);retained_faces=target_faces.copy()
            original_normals,_=basis(original_placed[source_ids],target_faces,attrs['uv'])
            retained_normals,_=basis(target_vertices,target_faces,attrs['uv'])
            locked=np.isin(source_ids,neck['hard_lock_indices'])
            retained_normals[locked]=original_normals[locked]
            ring=inverse[np.asarray(neck['ring_indices'])]
            target_vertices,target_faces,extra_normals,inner=transition(
                target_vertices,target_faces,ring,neck['target_ring'],retained_normals[ring],neck['target_normals'])
            if not np.array_equal(target_faces[:len(retained_faces)],retained_faces):
                raise ValueError('Transition changed retained source topology')
            if not np.array_equal(target_vertices[:retained_count][locked],original_placed[source_ids][locked]):
                raise ValueError('Transition moved locked face/chin vertices')
            added=target_faces[len(retained_faces):]
            triangle_xyz=target_vertices[added]
            if (np.linalg.norm(np.cross(triangle_xyz[:,1]-triangle_xyz[:,0],triangle_xyz[:,2]-triangle_xyz[:,0]),axis=1)<1e-12).any():
                raise ValueError('Transition has degenerate faces')
            edges=np.sort(np.concatenate([target_faces[:,[0,1]],target_faces[:,[1,2]],target_faces[:,[2,0]]]),axis=1)
            _,edge_counts=np.unique(edges,axis=0,return_counts=True)
            if (edge_counts>2).any():raise ValueError('Transition created nonmanifold edges')
            _,_,extra_attrs,_=transfer(target_vertices[retained_count:],ref_vertices,template['faces'],template,len(bones))
            attrs={k:np.concatenate([attrs[k],extra_attrs[k]]) for k in attrs}
            override_normals=np.concatenate([retained_normals,extra_normals])
            # Native FaceRoot (before FaceRoot_s slider scaling) lives in the
            # common rig. Its unchanged parent frame matches body Head_s support.
            root_id=next(pid for pid,t in transforms.items() if pid in members and t['name']=='cf_J_FaceRoot')
            tree=obj.read_typetree();tree['m_Bones'].append(dict(m_FileID=0,m_PathID=root_id));obj.save_typetree(tree)
            bindposes=np.concatenate([bindposes,np.linalg.inv(world_by_name['cf_J_FaceRoot'])[None]],axis=0)
            dense=np.zeros((len(target_vertices),len(bindposes)))
            for column in range(4):np.add.at(dense,(np.arange(retained_count),indices[:,column]),weights[:,column])
            fade=neck_factors[source_ids];dense[:retained_count]*=1-fade[:,None];dense[:retained_count,-1]=fade
            dense[retained_count:,-1]=1.
            indices=np.argsort(-dense,axis=1,kind='stable')[:,:4];weights=np.take_along_axis(dense,indices,axis=1)
            weights/=weights.sum(1,keepdims=True)
            neck.update(transition_vertices_added=len(target_vertices)-retained_count,
                transition_triangles_added=len(added),inner_boundary_head_indices=inner.tolist(),
                protected_original_normals_preserved=True,retained_triangles_preserved=True,
                transition_inside_actual_o_head=True)
            source_ids=np.r_[source_ids,np.full(len(target_vertices)-retained_count,-1,int)]
        mesh_reader=smr.m_Mesh.read().object_reader
        result=write_mesh(mesh_reader,target_vertices,target_faces,attrs['uv'],indices,weights,bindposes,
                          uv1=attrs['uv1'],colors=attrs['colors'],normals=override_normals)
        # Do not discard the just-appended serialized root when clearing weights.
        tree=obj.read_typetree();tree['m_BlendShapeWeights']=[]
        if name=='o_head' and neck is not None and authored is None:tree['m_Bones'].append(dict(m_FileID=0,m_PathID=root_id))
        obj.save_typetree(tree)
        # UnityPy readers retain original bytes until the bundle is serialized.
        # Readback therefore happens on the reopened serialized bundle below.
        original_ids=source_ids.copy()
        if authored is not None:
            original_ids[:]=-1
            retained=(source_ids>=0)&(source_ids<len(authored['crop_original_ids']))
            original_ids[retained]=authored['crop_original_ids'][source_ids[retained]]
        expected_meshes[name]=(target_vertices.astype('<f4'),target_faces,original_ids,indices,weights.astype('<f4'))
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
    new_cab='CAB-'+sha((GUID+sha(source_bytes)+sha(Path(__file__).read_bytes())+
        json.dumps(neck,sort_keys=True)).encode())[:32]
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
    skin_bundle=None;skin_policy=None
    if args.plain_skin:
        skin_bundle,skin_policy=neutral_skin(native,args.out)
        skin['MainAB']='chara/codex/chenger/skin.unity3d'
    manifest=ET.Element('manifest',{'schema-ver':'1'})
    for k,v in dict(guid=GUID,name='Chenger MICA native head',version='0.1.1',author='Codex',
                    description='Actual neutral head asset. Expressions deferred. Locally generated; requires installed HS2 assets.').items():
        ET.SubElement(manifest,k).text=v
    ET.SubElement(manifest,'faceSkinInfo',{'skinID':str(SLOT),'headID':str(SLOT),'headGUID':GUID})
    package=args.out/'Chenger.MICA.NativeHead.zipmod'
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('manifest.xml',ET.tostring(manifest,encoding='utf-8',xml_declaration=True))
        archive.writestr('abdata/'+MAIN_AB,output_bundle)
        if skin_bundle is not None:archive.writestr('abdata/chara/codex/chenger/skin.unity3d',skin_bundle)
        archive.writestr('abdata/list/characustom/codex_chenger_fo_head_00.csv',csv_bytes(210,head,'codex_chenger_fo_head_00'))
        archive.writestr('abdata/list/characustom/codex_chenger_ft_skin_f_00.csv',csv_bytes(211,skin,'codex_chenger_ft_skin_f_00'))
    receipt=dict(format='hs2_native_head_asset_build_v1',package=str(package.resolve()),package_sha256=sha(package.read_bytes()),
        guid=GUID,slot=SLOT,main_ab=MAIN_AB,prefab=PREFAB,source_sha256=sha(source_bytes),template_sha256=sha(bundle_bytes),
        source_parameters_preserved=True,outside_neck_band_positions_modified=False,neck_band_authored=neck is not None,
        placement=dict(scale=args.scale,translation=args.translation,
                       similarity=None if authored is None else design['similarity_matrix']),
        native_reference_face_values=[.5]*59,mesh_assets=mesh_reports,source_overlay_required=False,
        expression_support=False,neck_geometry_authored=neck is not None,neck_visual_acceptance=False,appearance='Native UV/skin attribute transfer; no inferred albedo',
        deferred_parts=['o_eyelashes','o_eyeshadow','o_namida','o_tooth','o_tang'],game_loading_verified=False)
    receipt['skin_policy']=skin_policy;receipt['native_eye_geometry_authored']=args.native_eyes
    if neck is not None:(args.out/'neck_design.json').write_text(json.dumps(neck,indent=2)+'\n',encoding='utf-8')
    (args.out/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(package=str(package.resolve()),guid=GUID,actual_head_mesh_authored=True,game_loading_verified=False)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--scale',type=float,required=True);parser.add_argument('--translation',type=float,nargs=3,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--native-neck',type=Path);parser.add_argument('--neck-descriptor',type=Path)
    parser.add_argument('--authored-neck',type=Path,help='Exact integrated author_neck_surface geometry.npz and adjacent receipt')
    parser.add_argument('--neck-bandwidth',type=float,default=.55)
    parser.add_argument('--plain-skin',action='store_true');parser.add_argument('--native-eyes',action='store_true')
    build(parser.parse_args())


if __name__=='__main__':main()
