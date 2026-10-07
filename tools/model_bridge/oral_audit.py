"""Trace native mouth assets and source FLAME support without fitting a graft.

The selected card resolves through current installed lists, then the exact render
prefab's serialized FaceBlendShape targets, channels and original skin bones.
No runtime samples, index/name retargeting or jaw-angle-to-mouth-rate fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from src.hs2_assets import ChaList, face_ids
from .artifact import ModelArtifact, sha
from .attachment_audit import topology


def inspect_native(bundle, prefab):
    env = UnityPy.load(str(bundle)); objects = list(env.objects)
    transforms, gameobjects = _read_transforms(objects)
    renderers = {}
    for obj in objects:
        if obj.type.name != 'SkinnedMeshRenderer':
            continue
        renderer = obj.read(); go = renderer.m_GameObject.path_id
        if go in gameobjects and prefab in _chain(transforms, gameobjects[go]):
            if go in renderers:
                raise ValueError('Multiple renderers on a native mouth target')
            renderers[go] = (obj, renderer)
    controllers = []
    for obj in objects:
        if obj.type.name != 'MonoBehaviour':
            continue
        data = obj.read(); go = data.m_GameObject.path_id
        if go not in gameobjects or prefab not in _chain(transforms, gameobjects[go]):
            continue
        script = data.m_Script.read()
        if script.m_ClassName == 'FaceBlendShape':
            controllers.append((obj, obj.read_typetree(), script))
    if len(controllers) != 1:
        raise ValueError('Exactly one actual render-prefab FaceBlendShape required')
    obj, controller, script = controllers[0]
    if script.m_AssemblyName != 'IL.dll':
        raise ValueError('Installed native mouth controller assembly differs from inspected IL.dll')
    mouth = controller['MouthCtrl']; targets = []
    for target in mouth['FBSTarget']:
        ref = target['ObjTarget']
        if ref['m_FileID'] != 0 or ref['m_PathID'] not in renderers:
            raise ValueError('Mouth target is not a renderer in the exact selected prefab')
        renderer_obj, renderer = renderers[ref['m_PathID']]
        mesh = renderer.m_Mesh.read(); arrays = _read_smr_mesh(mesh)
        channels = mesh.m_Shapes.channels
        patterns = target['PtnSet']
        if any(type(row[k]) is not int or not -1 <= row[k] < len(channels)
               for row in patterns for k in ('Close', 'Open')):
            raise ValueError('Native mouth pattern references an unavailable channel')
        bones = []
        for i, bone in enumerate(renderer.m_Bones):
            if bone.file_id != 0 or bone.path_id not in transforms:
                raise ValueError('External/unknown native skin bone')
            bones.append({'index': i, 'path_id': bone.path_id,
                          'name': transforms[bone.path_id]['name'],
                          'chain': _chain(transforms, bone.path_id)})
        used = np.unique(arrays['bone_idx'][arrays['bone_w'] > 0]).tolist()
        targets.append({'mesh': mesh.m_Name, 'mesh_path_id': mesh.object_reader.path_id,
            'renderer_path_id': renderer_obj.path_id,
            'chain': _chain(transforms, gameobjects[ref['m_PathID']]),
            'vertex_count': len(arrays['verts']), 'triangle_count': len(arrays['faces']),
            'skin_bones': bones, 'positive_weight_bones': used,
            'patterns': patterns, 'channels': [{'index': i, 'name': c.name,
                'frame_index': c.frameIndex, 'frame_count': c.frameCount} for i, c in enumerate(channels)],
            'materials': [{'file_id': m.file_id, 'path_id': m.path_id} for m in renderer.m_Materials],
            'skinning_policy': 'Original full weights/bindposes; mouth motion also uses original blendshape channels'})
    names = [t['mesh'] for t in targets]
    if not {'o_head', 'o_tang', 'o_tooth'}.issubset(names) or any(
            name not in {'o_head', 'o_tang', 'o_tooth', 'o_namida'} for name in names) or len(set(names)) != len(names):
        raise ValueError('Selected native oral target set is outside the inspected support')
    return {'bundle': str(bundle.resolve()), 'bundle_sha256': sha(bundle), 'prefab': prefab,
            'controller_path_id': obj.path_id, 'controller_assembly': script.m_AssemblyName,
            'mouth_control': {k: v for k, v in mouth.items() if k != 'FBSTarget'}, 'targets': targets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--image-index', type=int, default=0)
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--decompiled-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Keep previous evidence; use a fresh output path')
    sources = {name: args.decompiled_dir / (name + '.cs') for name in
               ('ChaControl', 'CmpFace', 'FaceBlendShape', 'FBSBase', 'FBSTargetInfo', 'FBSCtrlMouth')}
    # These guards establish the responsible implementation, not a new numerical
    # approximation. Full decompiled sources and assembly digests stay in evidence.
    guards = {'ChaControl': ['AssignedWeightsAndSetBounds(base.objHead', 'base.mouthCtrl = base.fbsCtrl.MouthCtrl'],
        'FaceBlendShape': ['MouthCtrl.CalcBlend(voiceValue)', 'Observable.EveryLateUpdate()'],
        'FBSBase': ['int num3 = (int)Mathf.Clamp(num * 100f, 0f, 100f)', 'SetBlendShapeWeight(item.Key, item.Value)'],
        'FBSTargetInfo': ['ObjTarget.GetComponent<SkinnedMeshRenderer>()'],
        'CmpFace': ['GetObjectFromName("o_tang")'], 'FBSCtrlMouth': ['CalculateBlendShape()']}
    for name, path in sources.items():
        text = path.read_text(encoding='utf-8-sig')
        if any(marker not in text for marker in guards[name]):
            raise ValueError('Inspected installed call chain differs: ' + name)
    ids = face_ids(str(args.card)); cl = ChaList(str(args.game_root / 'abdata'), use_cache=False)
    row = cl.resolve('fo_head', ids['headId']); bundle = Path(cl.bundle_path(row))
    native = inspect_native(bundle, row['MainData'])
    artifact = ModelArtifact(args.source_manifest, args.image_index); artifact.verify_sources()
    vertices, faces = artifact.mesh(head_local=True)
    source_topology = topology(vertices, faces)
    source = {'manifest': str(artifact.path), 'manifest_sha256': sha(artifact.path),
        'image_index': artifact.image_index, 'input_sha256': artifact.image['input_sha256'],
        'raw_artifact_sha256': artifact.image['sha256'], 'model_format': artifact.manifest['format'],
        'parameter_fields': artifact.image['parameter_fields'],
        'index_connected_parts': [{'vertex_count': len(c['vertices']), 'canonical_vertex_ids': c['vertices']}
                                  for c in source_topology['components']],
        'boundaries': source_topology['boundaries'], 'inner_mouth_asset': None,
        'jaw_policy': 'Original five-joint FLAME axis-angle pose/LBS and pose correctives, not native mouth-rate patterns'}
    report = {'format': 'source_native_oral_audit_v1', 'card': str(args.card.resolve()), 'card_sha256': sha(args.card),
        'auditor_sha256': sha(Path(__file__)),
        'selection_scope': 'Saved vanilla card IDs resolved against current installed lists; Sideloader GUID remapping and current runtime asset identity are not certified by this static audit',
        'head_id': ids['headId'], 'selected_installed_list_row': row,
        'assemblies': [{'path': str((args.game_root/'HoneySelect2_Data/Managed'/name).resolve()),
                        'sha256': sha(args.game_root/'HoneySelect2_Data/Managed'/name)}
                       for name in ('Assembly-CSharp.dll', 'IL.dll')],
        'decompiled_sources': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in sources.values()],
        'native': native, 'source': source,
        'native_mouth_rate_policy': 'Original FBS open min/max/fixed rate, integer0..100 blend weights, pattern transition and per-target Close/Open indices',
        'native_mouth_skin_is_not_source_jaw': True,
        'native_inner_mouth_preservation_correspondence': None,
        'automatic_graft_implemented': False,
        'next_required_correspondence': 'Authored inner-mouth geometry in the source coordinate/rig convention, or an explicit validated native-to-source oral placement/driver relation. No names/index copy, fitted jaw gains or silent face deformation.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(args.out.resolve()), 'head_id': ids['headId'],
                      'native_targets': [t['mesh'] for t in native['targets']],
                      'native_to_source_correspondence': False, 'game_calls': 0}))


if __name__ == '__main__':
    main()
