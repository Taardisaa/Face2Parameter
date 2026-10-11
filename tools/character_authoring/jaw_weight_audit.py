"""Inspect actual exported source weights; no deformation fitting or game sampling."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = a.geometry.read_bytes()
    d = json.loads(raw.decode('utf-8-sig'))
    matches = [m for m in d['meshes'] if m['mesh_name'] == 'o_head']
    if len(matches) != 1:
        raise ValueError('Expected exactly one actual o_head source mesh')
    m = matches[0]
    v = np.asarray(m['source']['vertices'])
    idx = np.asarray(m['source']['bone_indices'])
    weights = np.asarray(m['source']['bone_weights'])
    names = m['bone_names']
    # Source-space regions are explicit inspection selections, not anatomy labels
    # or an inferred deformation model. Preserve selected indices and all weights.
    bottom = (v[:, 1] < -.30) & (v[:, 2] > .50)
    groups = {'central_bottom': bottom & (np.abs(v[:, 0]) < .08),
              'lateral_bottom': bottom & (np.abs(v[:, 0]) >= .15) & (np.abs(v[:, 0]) < .35)}
    result = {'scope': 'Static source topology and direct LBS weights only. Parent transforms, blends and camera projection still matter; this does not predict screenshot width.',
              'input_sha256': hashlib.sha256(raw).hexdigest(),
              'source_geometry_sha256': m['source_geometry_sha256'],
              'game': d['game'], 'regions': {}}
    for label, mask in groups.items():
        ids = np.flatnonzero(mask)
        if not len(ids):
            raise ValueError('Empty source selection: '+label)
        averages = {n: float(np.where(idx[ids] == i, weights[ids], 0).sum(1).mean())
                    for i, n in enumerate(names)}
        result['regions'][label] = {
            'count': len(ids), 'selection': label,
            'mean_direct_weights': {n: w for n, w in averages.items() if w > 1e-7},
            'vertices': [{'index': int(i), 'source_xyz': v[i].tolist(),
                          'weights': [{'bone': names[int(j)], 'weight': float(w)}
                                      for j, w in zip(idx[i], weights[i]) if w > 1e-7]}
                         for i in ids]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: {'count': x['count'], 'mean_direct_weights': x['mean_direct_weights']}
                      for k, x in result['regions'].items()}, indent=2))


if __name__ == '__main__':
    main()
