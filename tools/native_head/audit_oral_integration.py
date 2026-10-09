"""Compare native oral assets with the exact installed private head prefab.

This is a source/serialized-asset audit, not a graft or expression retargeter.
Keep extracted licensed data and complete target tables in ignored outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from tools.model_bridge.oral_audit import inspect_native
from tools.native_head.audit_neck_shading import digest, provenance


def inspect_private(data, prefab):
    env = UnityPy.load(data)
    objects = list(env.objects)
    transforms, gameobjects = _read_transforms(objects)
    meshes, controllers = {}, []
    for obj in objects:
        if obj.type.name not in ('SkinnedMeshRenderer', 'MonoBehaviour'):
            continue
        value = obj.read()
        go = value.m_GameObject.path_id
        if go not in gameobjects or prefab not in _chain(transforms, gameobjects[go]):
            continue
        if obj.type.name == 'MonoBehaviour':
            script = value.m_Script.read()
            if script.m_ClassName == 'FaceBlendShape':
                tree = obj.read_typetree()
                controllers.append(dict(enabled=bool(tree['m_Enabled']),
                    assembly=script.m_AssemblyName, path_id=obj.path_id,
                    controls={k: tree[k] for k in ('EyebrowCtrl', 'EyesCtrl', 'MouthCtrl')}))
            continue
        mesh = value.m_Mesh.read()
        if mesh.m_Name in meshes:
            raise ValueError('Ambiguous private-prefab renderer: '+mesh.m_Name)
        meshes[mesh.m_Name] = dict(mesh_path_id=mesh.object_reader.path_id,
            renderer_path_id=obj.path_id, index_bytes=len(mesh.m_IndexBuffer),
            triangles=sum(s.indexCount for s in mesh.m_SubMeshes)//3,
            vertex_count=mesh.m_VertexData.m_VertexCount,
            channels=[c.name for c in mesh.m_Shapes.channels],
            bones=[b.read().m_GameObject.read().m_Name for b in value.m_Bones],
            materials=[dict(file_id=p.file_id, path_id=p.path_id) for p in value.m_Materials])
    if len(controllers) != 1 or not {'o_head', 'o_tooth', 'o_tang'}.issubset(meshes):
        raise ValueError('Expected actual head/teeth/tongue and one controller in selected prefab')
    return dict(prefab=prefab, controller=controllers[0], meshes=meshes)


def audit(reference, package, build_source, out):
    if out.exists():
        raise FileExistsError('Keep previous evidence; use a fresh directory')
    out.mkdir(parents=True)
    previous = json.loads(reference.read_text(encoding='utf-8-sig'))
    for item in previous['assemblies']+previous['decompiled_sources']:
        if digest(Path(item['path']).read_bytes()) != item['sha256']:
            raise ValueError('Source oracle changed: '+item['path'])
    native_path = Path(previous['native']['bundle'])
    if digest(native_path.read_bytes()) != previous['native']['bundle_sha256']:
        raise ValueError('Native donor changed')
    native = inspect_native(native_path, previous['native']['prefab'])
    with zipfile.ZipFile(package) as archive:
        private = inspect_private(archive.read('abdata/chara/codex/chenger/head.unity3d'),
                                  'p_cf_head_chenger_mica')
    env = UnityPy.load(str(native_path))
    by_id = {o.path_id: o for o in env.objects}
    closed_references = []
    for target in native['targets']:
        reader = by_id[target['mesh_path_id']]
        arrays = _read_smr_mesh(reader.read())
        # Bone-index ordering, bindposes, normals, tangents, UVs and complete
        # original blendshape payload remain available without inventing a rig.
        np.savez(out/(target['mesh']+'_donor.npz'), **arrays,
                 bone_names=np.asarray([b['name'] for b in target['skin_bones']]))
        tree = reader.read_typetree()
        (out/(target['mesh']+'_blendshapes.json')).write_text(
            json.dumps(tree['m_Shapes'])+'\n', encoding='utf-8')
        channel_id = target['patterns'][0]['Close']
        if channel_id < 0:
            closed_references.append(dict(mesh=target['mesh'], channel=-1))
            continue
        shapes = tree['m_Shapes']
        channel = shapes['channels'][channel_id]
        frames = []
        for frame_id in range(channel['frameIndex'], channel['frameIndex']+channel['frameCount']):
            frame = shapes['shapes'][frame_id]
            rows = shapes['vertices'][frame['firstVertex']:frame['firstVertex']+frame['vertexCount']]
            frames.append(dict(frame_index=frame_id, weight=shapes['fullWeights'][frame_id],
                nonzero_position_deltas=sum(any(v['vertex'][k] != 0 for k in 'xyz') for v in rows),
                has_normals=frame['hasNormals'], has_tangents=frame['hasTangents']))
        closed_references.append(dict(mesh=target['mesh'], channel=channel_id,
                                      name=channel['name'], frames=frames))
    result = dict(format='native_oral_integration_audit_v1', game_mutated=False,
        reference=provenance(reference), installed_package=provenance(package),
        current_build_source=provenance(build_source), native=native, installed=private,
        installed_oral_renderable=all(private['meshes'][k]['triangles']>0 for k in ('o_tooth','o_tang')),
        installed_mouth_controller_active=bool(private['controller']['enabled']
            and private['controller']['controls']['MouthCtrl']['FBSTarget']),
        original_oral_payloads_exported=True, graft_implemented=False,
        deformation_retarget_implemented=False,
        native_default_pattern_closed_references=closed_references,
        requirements=['Authored mouth/lip/inner-surface correspondence between native A and target B',
            'Native teeth/tongue geometry, full blendshape frames, materials and dependency references',
            'Mouth-cavity surface and lip attachment; do not classify anatomy by one UV rectangle',
            'Mapped head mouth-expression frames, preserving target neutral identity',
            'Use evaluated native closed-reference state; default head frame has nonzero deltas',
            'Per-target Close/Open indices and all referenced renderer channels must be valid',
            'Original controller rates, transitions, voice opening, tongue-state and width hooks',
            'Coordinate imported-head neck atlas and newly added mouth-interior texture domains'],
        limitations=['Static installed package; not proof of current runtime renderer binding',
            'Exports preserve native donor data, but do not establish A-to-B fitting or expression compatibility'])
    (out/'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out.resolve()),
        installed_oral_renderable=result['installed_oral_renderable'],
        installed_mouth_controller_active=result['installed_mouth_controller_active'],
        graft_implemented=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'package', 'build-source', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    audit(args.reference, args.package, args.build_source, args.out)
