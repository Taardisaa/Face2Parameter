"""Recover omitted expression deltas from the installed mesh asset on CPU.

Selection binds source positions, topology, all skin weights and bindposes;
mesh name or vertex count alone is insufficient (render/collision duplicates).
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler


def _arrays(mesh):
    h=MeshHandler(mesh); h.process()
    return {'vertices':np.asarray(h.m_Vertices,np.float32).reshape(-1,3),
        'triangles':np.asarray(h.m_IndexBuffer,np.int64),
        'bone_indices':np.asarray(h.m_BoneIndices,np.int64).reshape(-1,4),
        'bone_weights':np.asarray(h.m_BoneWeights,np.float32).reshape(-1,4),
        'bindposes':np.array([[[getattr(b,f'e{i}{j}') for j in range(4)] for i in range(4)]
                             for b in mesh.m_BindPose],np.float32)}


def recover_frames(geometry,bundle,prefab):
    bundle=Path(bundle)
    env=UnityPy.load(str(bundle))
    from scripts.hs2_extract_head import _read_transforms, _chain
    transforms,game_objects=_read_transforms(env.objects)
    mesh_ids=set()
    for reader in env.objects:
        if reader.type.name=='SkinnedMeshRenderer':
            renderer=reader.read()
            if prefab in _chain(transforms,game_objects[renderer.m_GameObject.path_id]):
                mesh_ids.add(renderer.m_Mesh.path_id)
    rows=[]
    requested={m['mesh_name'] for m in geometry['meshes']}
    for reader in env.objects:
        if reader.type.name!='Mesh' or reader.path_id not in mesh_ids:
            continue
        mesh=reader.read()
        if mesh.m_Name in requested:
            rows.append((reader.path_id,mesh))
    result=copy.deepcopy(geometry); provenance=[]
    decoded={}
    for target in result['meshes']:
        missing=any(shape['current_weight']!=0 and any('delta_vertices' not in f for f in shape['frames'])
                    for shape in target.get('blendshapes',[]))
        if not missing:
            continue
        matches=[]
        for path_id,mesh in rows:
            if mesh.m_Name!=target['mesh_name'] or mesh.m_VertexData.m_VertexCount!=target['vertex_count']:
                continue
            if path_id not in decoded:
                decoded[path_id]=_arrays(mesh)
            arrays=decoded[path_id]
            if all(np.array_equal(np.asarray(target['source'][key],dtype=array.dtype).reshape(array.shape),array)
                   for key,array in arrays.items()):
                matches.append((path_id,mesh))
        if len(matches)!=1:
            raise ValueError(f'Exactly one installed source asset match required for {target["renderer_path"]}; got {len(matches)}')
        path_id,mesh=matches[0]
        channels={ch.name:ch for ch in mesh.m_Shapes.channels}
        for shape in target['blendshapes']:
            channel=channels.get(shape['name'])
            if channel is None or channel.frameCount!=shape['frame_count']:
                raise ValueError('Expression channel identity/frame count mismatch')
            for index,frame in enumerate(shape['frames']):
                key=channel.frameIndex+index
                source=mesh.m_Shapes.shapes[key]
                if np.float32(frame['weight'])!=np.float32(mesh.m_Shapes.fullWeights[key]):
                    raise ValueError('Expression frame weight mismatch')
                delta={name:np.zeros((target['vertex_count'],3),np.float32)
                       for name in ('delta_vertices','delta_normals','delta_tangents')}
                for item in mesh.m_Shapes.vertices[source.firstVertex:source.firstVertex+source.vertexCount]:
                    for name,field in (('delta_vertices','vertex'),('delta_normals','normal'),('delta_tangents','tangent')):
                        vector=getattr(item,field)
                        delta[name][item.index]=[vector.x,vector.y,vector.z]
                frame.update({name:value.tolist() for name,value in delta.items()})
        provenance.append({'renderer_path':target['renderer_path'],'asset_path_id':path_id,
                           'selection':'float32-exact source positions, triangles, bone indices, weights, bindposes'})
    return result,{'bundle_path':str(bundle.resolve()),'prefab':prefab,'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),
                   'recovered_meshes':provenance}
