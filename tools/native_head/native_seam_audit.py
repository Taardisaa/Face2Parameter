"""Read authored HS2 neck interfaces, without game sampling or mesh edits.

Run from the repository root:
  .venv/Scripts/python.exe -m tools.native_head.native_seam_audit --output outputs/native_seam_audit_20261007
Licensed mesh arrays stay in the ignored output folder; the receipt records hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from src.hs2_assets import ChaList
from tools.model_bridge.attachment_audit import body_asset
from tools.native_head.neck_geometry import ordered_loops, original_neck_boundary


def extract(data, prefab):
    objects = list(UnityPy.load(data).objects)
    transforms, gameobjects = _read_transforms(objects)
    found = []
    for obj in objects:
        if obj.type.name != 'SkinnedMeshRenderer':
            continue
        renderer = obj.read()
        chain = _chain(transforms, gameobjects[renderer.m_GameObject.path_id])
        if prefab in chain and renderer.m_Mesh.read().m_Name == 'o_head':
            found.append(renderer)
    if len(found) != 1:
        raise ValueError(f'Expected one o_head below {prefab}, got {len(found)}')
    renderer = found[0]
    # The two bone-local frames coincide only for this audited attachment contract.
    # Refuse an independently authored root offset instead of silently comparing frames.
    checked = []
    for pid, t in transforms.items():
        if t['name'] in ('cf_J_FaceRoot', 'cf_J_FaceRoot_s') and prefab in _chain(transforms, pid):
            if (not np.allclose(t['pos'], 0, atol=1e-10, rtol=0)
                    or not np.allclose(t['rot'], [0, 0, 0, 1], atol=1e-10, rtol=0)
                    or not np.allclose(t['scale'], 1, atol=1e-10, rtol=0)):
                raise ValueError(f'Unimplemented nonidentity attachment transform: {t}')
            checked.append(t['name'])
    if set(checked) != {'cf_J_FaceRoot', 'cf_J_FaceRoot_s'} or len(checked) != 2:
        raise ValueError('Ambiguous attachment hierarchy')
    return _read_smr_mesh(renderer.m_Mesh.read()), [
        x.read().m_GameObject.read().m_Name for x in renderer.m_Bones]


def aliases(arrays):
    unique, result = {}, {}
    for i in range(len(arrays['verts'])):
        key = (arrays['verts'][i].tobytes() + arrays['bone_idx'][i].tobytes()
               + arrays['bone_w'][i].tobytes())
        unique.setdefault(key, i)
        result[i] = unique[key]
    return result


def nearest(points, target):
    distance = np.linalg.norm(points[:, None] - target[None], axis=2)
    ids = distance.argmin(1)
    return ids, distance[np.arange(len(points)), ids]


def audit_head(arrays, names, body, body_names, body_rim, body_in_head):
    root = names.index('cf_J_FaceRoot_s')
    weights = (arrays['bone_w'] * (arrays['bone_idx'] == root)).sum(1)
    loops = ordered_loops(arrays['faces'], aliases(arrays))
    rings = [r for r in loops if np.all(weights[r] == 1)]
    result = []
    for ring in rings:
        # Literal inverse bindposes put both surfaces in their attachment-bone frames.
        m = arrays['bindpose'][root].astype(float)
        points = arrays['verts'][ring] @ m[:3, :3].T + m[:3, 3]
        ids, distance = nearest(points, body_in_head)
        _, opening_distance = nearest(points, body_in_head[body_rim])
        normal = arrays['normals'][ring] @ np.linalg.inv(m[:3, :3])
        bm = body['bindpose'][body_names.index('cf_J_Head_s')].astype(float)
        body_normal = body['normals'][ids] @ np.linalg.inv(bm[:3, :3])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        body_normal /= np.linalg.norm(body_normal, axis=1, keepdims=True)
        body_head = body_names.index('cf_J_Head_s')
        body_weights = (body['bone_w'][ids] * (body['bone_idx'][ids] == body_head)).sum(1)
        _, to_rows = nearest(body_in_head, body_in_head[np.r_[body_rim, ids]])
        overlap_strip = body['faces'][(to_rows[body['faces']] < 1e-5).all(1)]
        strip_weights = (body['bone_w'][np.unique(overlap_strip)] *
                        (body['bone_idx'][np.unique(overlap_strip)] == body_head)).sum(1)
        adjacent = arrays['faces'][np.isin(arrays['faces'], ring).any(1)]
        adjacent_ids = np.unique(adjacent)
        adjacent_points = arrays['verts'][adjacent_ids] @ m[:3, :3].T + m[:3, 3]
        _, adjacent_distance = nearest(adjacent_points, body_in_head)
        result.append({
            'head_boundary_ids': ring, 'count': len(ring), 'head_binding': 'cf_J_FaceRoot_s:1',
            'body_matching_vertex_ids': ids.tolist(),
            'body_matches_in_opening_ring': int(np.isin(ids, body_rim).sum()),
            'max_distance_to_body_vertices': float(distance.max()),
            'min_distance_to_body_opening': float(opening_distance.min()),
            'max_distance_to_body_opening': float(opening_distance.max()),
            'max_normal_vector_difference': float(np.linalg.norm(normal-body_normal, axis=1).max()),
            'matched_body_vertices_all_Head_s_weight_one': bool(np.all(body_weights == 1)),
            'body_strip_between_opening_and_head_rim_triangle_count': len(overlap_strip),
            'body_strip_all_Head_s_weight_one': bool(np.all(strip_weights == 1)),
            'adjacent_head_triangle_count': len(adjacent),
            'adjacent_head_vertices_matching_body_within_1e-5': int((adjacent_distance < 1e-5).sum()),
            'adjacent_head_vertex_count': len(adjacent_ids),
            'comparison_tolerance_is_float_serialization_only': True,
        })
    return {'pure_root_boundary_rings': result, 'all_boundary_sizes': [len(x) for x in loops]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', type=Path, default=Path(r'E:\HoneySelect2_ArcticFox'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--zipmod-audit', type=Path, default=Path('outputs/head_base_audit_20261007/prefab_contract_v1.json'))
    args = p.parse_args()
    body, meta = body_asset(args.game / 'abdata/chara/oo_base.unity3d')
    transforms, _ = _read_transforms(list(UnityPy.load(str(args.game / 'abdata/chara/oo_base.unity3d')).objects))
    root = [t for t in transforms.values() if t['name'] == 'p_cf_head_bone']
    if len(root) != 1 or root[0]['pos'] != [0, 0, 0] or root[0]['rot'] != [0, 0, 0, 1] or root[0]['scale'] != [1, 1, 1]:
        raise ValueError('Unimplemented native head-bone attachment root')
    names = meta['skin_bones']
    body_rim, _ = original_neck_boundary(dict(vertices=body['verts'], triangles=body['faces'],
        bone_indices=body['bone_idx'], bone_weights=body['bone_w']), names)
    m = body['bindpose'][names.index('cf_J_Head_s')].astype(float)
    points = body['verts'] @ m[:3, :3].T + m[:3, 3]
    report = {'body': meta, 'body_opening_ring': body_rim, 'heads': [],
              'common_head_bone_root': root[0],
              'scope': 'Authored bind geometry. No runtime sweeps, mutations, or shader continuity claim.'}
    catalog = ChaList(ab_dir=str(args.game / 'abdata'))
    native = [catalog.resolve('fo_head', i) for i in range(4)]
    sources = []
    for row in native:
        path = args.game / 'abdata' / row['MainAB']
        sources.append((str(path), row['MainData'], path.read_bytes()))
    if args.zipmod_audit.exists():
        previous = json.loads(args.zipmod_audit.read_text(encoding='utf-8'))
        for z in previous['zipmods']:
            with zipfile.ZipFile(z['path']) as archive:
                for h in z['heads']:
                    sources.append((z['path']+'::'+h['member'], h['row']['MainData'], archive.read(h['member'])))
    args.output.mkdir(parents=True, exist_ok=True)
    for source, prefab, data in sources:
        row = {'source': source, 'prefab': prefab, 'bundle_sha256': hashlib.sha256(data).hexdigest()}
        try:
            head, head_names = extract(data, prefab)
            row.update(audit_head(head, head_names, body, names, body_rim, points))
        except ValueError as exc:
            row['unresolved'] = str(exc)
        report['heads'].append(row)
        print(prefab, row.get('unresolved') or [(r['count'], r['max_distance_to_body_vertices'])
                                               for r in row['pure_root_boundary_rings']])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
