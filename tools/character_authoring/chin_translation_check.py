"""Compare one source-derived chin position change against actual raw BakeMesh.

No sampling, regression or fitted gains. Full residuals retained; this tool does
not manufacture a pass threshold or certify portrait appearance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def read(path):
    raw = path.read_bytes()
    d = json.loads(raw.decode('utf-8-sig'))
    matches = [m for m in d['meshes'] if m['mesh_name'] == 'o_head']
    if len(matches) != 1:
        raise ValueError('Require one actual o_head')
    m = matches[0]
    t = {r['id']: r for r in d['transforms']}
    heads = [r for r in t.values() if r['name'] == 'cf_J_Head']
    if len(heads) != 1:
        raise ValueError('Require unique actual head')
    hi = np.asarray(heads[0]['world_to_local']).reshape(4, 4)
    v = np.c_[m['baked']['vertices'], np.ones(m['vertex_count'])]
    head_vertices = (v @ np.asarray(m['baked']['world_candidates']['scale_free_trs']['matrix']).reshape(4, 4).T @ hi.T)[:, :3]
    return d, m, t, hi, head_vertices, hashlib.sha256(raw).hexdigest()


def compare(before, after):
    a, m, t, hi, va, sha_a = read(before)
    b, mb, tb, hb, vb, sha_b = read(after)
    if m['source_geometry_sha256'] != mb['source_geometry_sha256'] or va.shape != vb.shape:
        raise ValueError('Source geometry changed; cannot compare this position-only path')
    A = {r['name']: r for r in a['abmx_runtime']['bones']}
    B = {r['name']: r for r in b['abmx_runtime']['bones']}
    allowed = ('cf_J_ChinTip_s', 'cf_J_ChinLow')
    pred = np.zeros_like(va)
    deltas = {}
    indices = np.asarray(m['source']['bone_indices'])
    weights = np.asarray(m['source']['bone_weights'])
    for name in allowed:
        if any(A[name][k] != B[name][k] for k in ('scale', 'length', 'rotation')) or A[name]['length'] != 1:
            raise ValueError('Only length-one position changes are supported: '+name)
        cache = A[name]['runtime_baseline']
        if cache['missing_fields'] or not cache['fields']['_hasBaseline']:
            raise ValueError('Installed baseline is unavailable: '+name)
        delta = np.asarray(B[name]['position']) - A[name]['position']
        deltas[name] = delta.tolist()
        node = t[cache['bone_transform_id']]
        inherited = []
        for i in m['bone_transform_ids']:
            cur = i
            while cur is not None and cur != node['id']:
                cur = t[cur]['parent_id']
            inherited.append(cur == node['id'])
        total_weight = (weights * np.asarray(inherited)[indices]).sum(1)
        parent = np.asarray(t[node['parent_id']]['local_to_world']).reshape(4, 4)
        pred += total_weight[:, None] * (hi[:3, :3] @ parent[:3, :3] @ delta)
    obs = vb - va
    residual = np.linalg.norm(obs - pred, axis=1)
    return {'scope': __doc__, 'before_sha256': sha_a, 'after_sha256': sha_b,
            'source_geometry_sha256': m['source_geometry_sha256'], 'position_deltas': deltas,
            'max_residual_game_units': float(residual.max()),
            'mean_residual_game_units': float(residual.mean()),
            'largest_residual_vertex': int(residual.argmax()),
            'all_observed_head_delta': obs.tolist(), 'all_predicted_head_delta': pred.tolist(),
            'all_residual_game_units': residual.tolist(),
            'coverage': 'Captured pair only; other driver/ancestor differences are not suppressed and remain in residuals. No general animation/private-state compatibility claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    r = compare(a.before, a.after)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(r, f, indent=2)
    print(json.dumps({k: v for k, v in r.items() if not k.startswith('all_')}, indent=2))


if __name__ == '__main__':
    main()
