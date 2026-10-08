"""Read native/custom head prefab contracts directly from installed assets.

No game calls or asset writes. Licensed typetrees/CSV stay in the ignored report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import UnityPy

from scripts.hs2_extract_head import _chain, _read_transforms
from src.hs2_assets import ChaList


def sha(data):
    return hashlib.sha256(data).hexdigest()


def inspect(data, prefab, row):
    env = UnityPy.load(data)
    objects = list(env.objects)
    transforms, gameobjects = _read_transforms(objects)
    root_ids = [i for i, t in transforms.items() if t['name'] == prefab]
    if len(root_ids) != 1:
        raise ValueError(f'Expected one exact prefab {prefab}, got {root_ids}')
    members = {i for i in transforms if prefab in _chain(transforms, i)}
    root_go = next(o.read() for o in objects if o.type.name == 'GameObject'
                   and gameobjects[o.path_id] == root_ids[0])
    root_components = []; controllers = []; renderers = []
    for component in root_go.m_Component:
        ref = component.component
        obj = ref.read(); entry = {'type': obj.object_reader.type.name, 'path_id': ref.path_id}
        if entry['type'] == 'MonoBehaviour':
            script = obj.m_Script.read()
            entry.update(class_name=script.m_ClassName, assembly=script.m_AssemblyName,
                         typetree=obj.object_reader.read_typetree())
        root_components.append(entry)
    for obj in objects:
        if obj.type.name not in ('SkinnedMeshRenderer', 'MonoBehaviour'):
            continue
        value = obj.read(); go = value.m_GameObject.path_id
        if gameobjects.get(go) not in members:
            continue
        if obj.type.name == 'MonoBehaviour':
            script = value.m_Script.read()
            if script.m_ClassName == 'FaceBlendShape':
                controllers.append({'path_id': obj.path_id, 'class_name': script.m_ClassName,
                    'assembly': script.m_AssemblyName, 'typetree': obj.read_typetree()})
            continue
        mesh = value.m_Mesh.read()
        materials = []
        for ref in value.m_Materials:
            material = ref.read()
            materials.append({'name': material.m_Name, 'shader': material.m_Shader.read().m_ParsedForm.m_Name})
        bones = [ref.read().m_GameObject.read().m_Name for ref in value.m_Bones]
        renderers.append({'path_id': obj.path_id, 'path': _chain(transforms, gameobjects[go]),
            'mesh': mesh.m_Name, 'mesh_path_id': value.m_Mesh.path_id,
            'vertices': mesh.m_VertexData.m_VertexCount, 'submeshes': len(mesh.m_SubMeshes),
            'bone_names': bones, 'bindpose_count': len(mesh.m_BindPose),
            'blendshape_channels': [channel.name for channel in mesh.m_Shapes.channels],
            'materials': materials})
    named_assets = {}
    for obj in objects:
        if obj.type.name not in ('TextAsset', 'Material', 'MonoBehaviour'):
            continue
        value = obj.read(); name = getattr(value, 'm_Name', '')
        if name in (row.get('ShapeAnime'), row.get('MatData'), row.get('Preset')):
            named_assets[name] = {'type': obj.type.name, 'path_id': obj.path_id}
            if obj.type.name == 'TextAsset':
                raw = value.m_Script.encode('utf-8', 'surrogateescape') if isinstance(value.m_Script, str) else bytes(value.m_Script)
                named_assets[name].update(bytes=len(raw), sha256=sha(raw))
    return {'prefab': prefab, 'bundle_sha256': sha(data), 'root_transform': transforms[root_ids[0]],
        'root_components': root_components, 'bone_transforms': [dict(path_id=i, **transforms[i]) for i in sorted(members)],
        'renderers': renderers, 'expression_controllers': controllers, 'list_named_assets': named_assets,
        'list_named_assets_missing_from_this_bundle': [row[k] for k in ('ShapeAnime','MatData','Preset') if row.get(k) not in named_assets]}


def inspect_zip(path):
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read('manifest.xml'); xml = ET.fromstring(manifest)
        lists = []
        for name in archive.namelist():
            if not name.lower().endswith('.csv') or not any(k in name.lower() for k in ('fo_head', 'ft_skin_f')):
                continue
            raw = archive.read(name); lines = raw.decode('utf-8-sig').splitlines()
            entries = list(csv.DictReader(lines[3:]))
            lists.append({'member': name, 'sha256': sha(raw), 'category': int(lines[0].split(',')[0]), 'rows': entries})
        heads = []
        for table in lists:
            if table['category'] != 210:
                continue
            for row in table['rows']:
                member = 'abdata/'+row['MainAB']
                heads.append({'row': row, 'member': member, 'prefab_audit': inspect(archive.read(member), row['MainData'], row)})
        return {'path': str(path.resolve()), 'sha256': sha(path.read_bytes()),
            'guid': xml.findtext('guid'), 'version': xml.findtext('version'),
            'manifest_sha256': sha(manifest), 'manifest': manifest.decode('utf-8-sig'),
            'lists': lists, 'heads': heads}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--zipmod', type=Path, action='append', default=[])
    parser.add_argument('--head-id', type=int, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use fresh evidence destination')
    cl = ChaList(str(args.game_root/'abdata'), use_cache=False)
    native = []
    for head_id in args.head_id:
        row = cl.resolve('fo_head', head_id); path = Path(cl.bundle_path(row))
        native.append({'id': head_id, 'row': row, 'bundle': str(path),
            'prefab_audit': inspect(path.read_bytes(), row['MainData'], row),
            'compatible_skin_rows': [r for r in cl.by_cat[211].values() if int(r['HeadID']) == head_id]})
    report = {'format': 'hs2_native_head_base_audit_v1', 'auditor_sha256': sha(Path(__file__).read_bytes()),
        'native': native, 'zipmods': [inspect_zip(p) for p in args.zipmod], 'game_calls': 0,
        'actual_runtime_loader_priority_claimed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report': str(args.out.resolve()), 'sha256': sha(args.out.read_bytes()),
        'heads': [{'guid': p.get('guid'), 'prefab': h['prefab_audit']['prefab'],
            'renderers': len(h['prefab_audit']['renderers'])} for p in report['zipmods'] for h in p['heads']], 'game_calls': 0}))


if __name__ == '__main__':
    main()
