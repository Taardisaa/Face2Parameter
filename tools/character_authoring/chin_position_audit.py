"""Inspect installed length-one chin translations using captured skinning data.

This is a source-derived differential for position ONLY. It does not fit game
samples, predict changing scale/rotation/length, or supply anatomical landmarks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(path: Path):
    raw = path.read_bytes()
    d = json.loads(raw.decode('utf-8-sig'))
    meshes = [m for m in d['meshes'] if m['mesh_name'] == 'o_head']
    if len(meshes) != 1:
        raise ValueError('Require one captured native o_head')
    m = meshes[0]
    t = {r['id']: r for r in d['transforms']}
    heads = [r for r in t.values() if r['name'] == 'cf_J_Head']
    if len(heads) != 1:
        raise ValueError('Require actual unique head transform')
    head_inv = np.asarray(heads[0]['world_to_local']).reshape(4, 4)
    v = np.asarray(m['source']['vertices'], float).copy()
    for shape in m['blendshapes']:
        if not shape['current_weight']:
            continue
        frames = shape.get('frames', [])
        if len(frames) != 1 or frames[0]['weight'] != 100 or not 0 <= shape['current_weight'] <= 100:
            raise ValueError('Unsupported active blendshape frames/weight; no extrapolation fallback')
        v += np.asarray(frames[0]['delta_vertices']) * (shape['current_weight'] / 100)
    indices = np.asarray(m['source']['bone_indices'])
    weights = np.asarray(m['source']['bone_weights'])
    matrices = np.asarray([
        np.asarray(t[i]['local_to_world']).reshape(4, 4) @ np.asarray(b).reshape(4, 4)
        for i, b in zip(m['bone_transform_ids'], m['source']['bindposes'])])
    world = np.einsum('vki,vk->vi', np.einsum(
        'vkij,vj->vki', matrices[indices], np.c_[v, np.ones(len(v))]), weights)[:, :3]
    result = {'input_sha256': hashlib.sha256(raw).hexdigest(),
              'source_geometry_sha256': m['source_geometry_sha256'],
              'game': d['game'], 'skin_quality': m['skin_quality'],
              'scope': 'Position-only differential from actual parent matrices, hierarchy, source weights and installed ABMX length=1 baseline+offset branch. All descendants included. No screenshot fitting, anatomical calibration or general renderer certification.',
              'bake_world_comparison': {}, 'bones': {}}
    for name, candidate in m['baked']['world_candidates'].items():
        baked_world = (np.c_[m['baked']['vertices'], np.ones(len(v))] @
                       np.asarray(candidate['matrix']).reshape(4, 4).T)[:, :3]
        error = np.linalg.norm(baked_world - world, axis=1)
        result['bake_world_comparison'][name] = {
            'max_distance_game_units': float(error.max()),
            'mean_distance_game_units': float(error.mean()),
            'scope': 'Observed complete captured mesh discrepancy, not an invented acceptance threshold.'}
    modifiers = {b['name']: b for b in d['abmx_runtime']['bones']}
    source = np.asarray(m['source']['vertices'])
    selected = np.flatnonzero((source[:, 1] < -.30) & (source[:, 2] > .50))
    if not len(selected):
        raise ValueError('Empty explicit source bottom selection')
    for name in ('cf_J_ChinTip_s', 'cf_J_ChinLow'):
        modifier = modifiers[name]
        if modifier['length'] != 1:
            raise ValueError('Length-dependent branch is excluded: '+name)
        cache = modifier['runtime_baseline']
        if cache['missing_fields'] or not cache['fields']['_hasBaseline']:
            raise ValueError('Captured installed cache unavailable: '+name)
        node = t[cache['bone_transform_id']]
        actual_pos = np.asarray(node['local_position'])
        expected_pos = np.asarray(cache['fields']['_posBaseline']) + modifier['position']
        # Do not invent a pass tolerance; preserve observed float32 arithmetic difference.
        baseline_error = actual_pos - expected_pos
        parent = np.asarray(t[node['parent_id']]['local_to_world']).reshape(4, 4)
        inherited = []
        for bone_id in m['bone_transform_ids']:
            cursor = bone_id
            while cursor is not None and cursor != node['id']:
                cursor = t[cursor]['parent_id']
            inherited.append(cursor == node['id'])
        influence = np.asarray(inherited)[indices]
        total_weight = (weights * influence).sum(1)
        # Translation affects each descendant bone equally in world space:
        # sum descendant weights * parent's world linear map * offset_delta.
        linear_head = head_inv[:3, :3] @ parent[:3, :3]
        jacobian = total_weight[:, None, None] * linear_head[None, :, :]
        result['bones'][name] = {
            'installed_modifier': modifier,
            'actual_local_position': actual_pos.tolist(),
            'baseline_plus_offset_component_error': baseline_error.tolist(),
            'inherited_bones': [n for n, yes in zip(m['bone_names'], inherited) if yes],
            'parent_linear_in_head_space': linear_head.tolist(),
            'selected_source_vertices': [
                {'index': int(i), 'source_xyz': source[i].tolist(),
                 'inherited_weight': float(total_weight[i]),
                 'head_xyz_change_per_local_xyz_offset': jacobian[i].tolist()}
                for i in selected]}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.geometry)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'bake_world_comparison': result['bake_world_comparison'],
                      'bone_baseline_errors': {k: v['baseline_plus_offset_component_error']
                                               for k, v in result['bones'].items()}}, indent=2))


if __name__ == '__main__':
    main()
