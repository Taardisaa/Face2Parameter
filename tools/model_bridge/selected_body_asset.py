"""Resolve the recorded Uncensor BodyData against an explicit installed zipmod.

Read-only manifest/bundle extraction and exact native array comparison. Does not
infer a body selection from vertex counts or redistribute the licensed asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from .artifact import sha
from .neck_patch import digest_array


def resolve(archive, selected, descriptor):
    if not selected.get('available'):
        raise ValueError('Recorded runtime BodyData required')
    with ZipFile(archive) as zipmod:
        manifest_bytes = zipmod.read('manifest.xml')
        root = ElementTree.fromstring(manifest_bytes)
        entries = [entry for entry in root.findall('KK_UncensorSelector/body')
                   if entry.findtext('guid') == selected['BodyGUID']]
        if len(entries) != 1 or entries[0].findtext('oo_base/file') != selected['OOBase']:
            raise ValueError('Zipmod manifest does not supply the recorded selected body')
        bundle_entry = 'abdata/' + selected['OOBase']
        data = zipmod.read(bundle_entry)
    env = UnityPy.load(data)
    objects = list(env.objects)
    transforms, gameobjects = _read_transforms(objects)
    matches = []
    for obj in objects:
        if obj.type.name != 'SkinnedMeshRenderer':
            continue
        renderer = obj.read()
        chain = _chain(transforms, gameobjects[renderer.m_GameObject.path_id])
        if selected['Asset'] not in chain:
            continue
        mesh = renderer.m_Mesh.read()
        if mesh.m_Name == 'o_body_cf':
            matches.append((obj, renderer, mesh, chain))
    if len(matches) != 1:
        raise ValueError('Exactly one source-selected female render body required')
    obj, renderer, mesh, chain = matches[0]
    arrays = _read_smr_mesh(mesh)
    hashes = {field+'_sha256': digest_array(arrays[key], dtype) for key, field, dtype in [
        ('verts', 'vertices', '<f4'), ('faces', 'triangles', '<i4'), ('bone_idx', 'bone_indices', '<i4'),
        ('bone_w', 'bone_weights', '<f4'), ('bindpose', 'bindposes', '<f4'),
        ('uv', 'uv', '<f4'), ('uv1', 'uv2', '<f4')]}
    names = [bone.read().m_GameObject.read().m_Name for bone in renderer.m_Bones]
    expected = descriptor['native']
    checks = {field: value == expected[field] for field, value in hashes.items()}
    checks['bone_names'] = names == expected['bone_names']
    checks['vertex_count'] = len(arrays['verts']) == expected['vertex_count']
    return {'archive': {'path': str(archive.resolve()), 'sha256': sha(archive),
                        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                        'guid': root.findtext('guid'), 'version': root.findtext('version')},
            'selected_body': selected, 'bundle_entry': bundle_entry,
            'bundle_sha256': hashlib.sha256(data).hexdigest(),
            'mesh_path_id': mesh.object_reader.path_id, 'renderer_path_id': obj.path_id,
            'transform_chain': chain, 'native_identity_hashes': hashes, 'exact_checks': checks,
            'native_body_asset_verified': all(checks.values()),
            'scope': 'Actual selected body manifest and render mesh/skin/UV only; mutable BustNormal normals and material appearance not certified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zipmod', type=Path, required=True)
    parser.add_argument('--acceptance-receipt', type=Path, required=True)
    parser.add_argument('--attachment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.acceptance_receipt.read_text(encoding='utf-8'))
    descriptor = json.loads(args.attachment.read_text(encoding='utf-8'))
    result = resolve(args.zipmod, receipt['initial']['selected_body'], descriptor)
    result['inputs'] = [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
                       (args.acceptance_receipt, args.attachment)]
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    if not result['native_body_asset_verified']:
        raise RuntimeError('Selected archive differs from native identity arrays; report retained')
    print(json.dumps({'report': str(args.out.resolve()), 'native_body_asset_verified': True}))


if __name__ == '__main__':
    main()
