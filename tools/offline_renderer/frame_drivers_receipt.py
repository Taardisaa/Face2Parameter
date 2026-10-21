"""Compact receipt of the per-frame driver acceptance (eye look, FBS) for the repo.

Reads the bulky outputs of eye_look_acceptance.py and fbs_composition_check.py
and writes tools/offline_renderer/frame_drivers_acceptance_20261020.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GAME = Path('E:/HoneySelect2_ArcticFox')
INPUTS = {
    'UnityPlayer.dll': GAME / 'UnityPlayer.dll',
    'mono-2.0-bdwgc.dll': GAME / 'MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll',
    'IL.dll': GAME / 'HoneySelect2_Data/Managed/IL.dll',
    'UnityEngine.CoreModule.dll': GAME / 'HoneySelect2_Data/Managed/UnityEngine.CoreModule.dll',
    'oo_base.unity3d': GAME / 'abdata/chara/oo_base.unity3d',
    'fo_head_38.unity3d': GAME / 'abdata/chara/38/fo_head_38.unity3d',
    'globalgamemanagers.assets': GAME / 'HoneySelect2_Data/globalgamemanagers.assets',
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eye', type=Path, required=True)
    ap.add_argument('--fbs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    eye = json.loads(args.eye.read_text(encoding='utf-8'))
    fbs = json.loads(args.fbs.read_text(encoding='utf-8'))
    windows = {}
    for w, rows in eye['windows'].items():
        frames = [r for r in rows if r.get('role') != 'state_seed']
        per = []
        for r in frames:
            per.append({'frame': r['frame'], 'delta_time': r['delta_time'], 'camera_source': r['camera_source'],
                        'world_chain_bit_equal': not r['world_mismatch'],
                        **{n: {'bit_equal': r[n]['bit_equal'], 'seed_invariant': r[n]['seed_invariant']}
                           for n in ('cf_J_look_L', 'cf_J_look_R')}})
        accepted = [(r['frame'], n) for r in frames for n in ('cf_J_look_L', 'cf_J_look_R')
                    if r[n]['bit_equal'] and r[n]['seed_invariant']]
        seed_dep = [(r['frame'], n, r[n]['bit_equal']) for r in frames for n in ('cf_J_look_L', 'cf_J_look_R')
                    if not r[n]['seed_invariant']]
        windows[w] = {'seed_frame': rows[0]['frame'], 'frames': per,
                      'accepted_seed_invariant_bit_exact': len(accepted),
                      'seed_dependent': [{'frame': f, 'bone': n, 'bit_equal_with_reconstructed_seed': e}
                                         for f, n, e in seed_dep],
                      'non_bit_equal_seed_invariant': [(r['frame'], n) for r in frames for n in ('cf_J_look_L', 'cf_J_look_R')
                                                       if r[n]['seed_invariant'] and not r[n]['bit_equal']]}
    receipt = {
        'schema': 'hs2_frame_drivers_acceptance_v1',
        'date': '2026-10-20',
        'inputs_sha256': {k: sha(v) for k, v in INPUTS.items()},
        'execution': 'installed managed IL on the installed Mono JIT (mono_set_defaults(0, '
                     'mono_parse_default_optimizations(NULL)) as UnityPlayer InitializeMonoFromMain RVA 0x8e99d5); '
                     'native calls are UnityPlayer.dll routines (direct icall registration or TransformAccess routines)',
        'eye_look': {'seeds': eye['seeds'], 'init_state': eye['init_state'], 'camera_main_path': eye['camera_main_path'],
                     'camera_position_used': eye['camera_window2'], 'windows': windows},
        'fbs_composition': {'expression_state': fbs['expression_state'],
                            'mouth_adjust_width_scale': fbs['mouth_adjust_width_scale'],
                            'windows': {w: rows for w, rows in fbs['windows'].items()},
                            'every_frame_matches_exactly_one_steady_state': all(
                                len(r['matches']) == 1 for rows in fbs['windows'].values() for r in rows)},
        'bulky_outputs': {'eye_look': str(args.eye), 'fbs': str(args.fbs)},
    }
    args.out.write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    for w, d in windows.items():
        print(w, 'accepted', d['accepted_seed_invariant_bit_exact'], 'seed-dependent', len(d['seed_dependent']),
              'seed-invariant mismatches', d['non_bit_equal_seed_invariant'])


if __name__ == '__main__':
    main()
