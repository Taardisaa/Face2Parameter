"""Chin/jaw controls (handoff parameter set) and the persistent ABMX encoding.

Parameter set (``chenger_reference_fit_20261012/guarded_predictor_fit.py``, handoff 4.2 / README
"DFO-LS uses the three central bones and the mirrored CheekLow, XYZ scale, position and
rotation, 36 variables"): groups cf_J_ChinTip_s, cf_J_ChinLow, cf_J_Chin_rs and the mirrored
pair cf_J_CheekLow_L/R; channels scale (x(1 + 0.10 q)), position (+0.025 q central, +0.012 q
pair) and rotation (+3 deg q), axes x/y/z, q in [-1, 1].  The second bone of the pair mirrors
position x and rotation y/z.  These are the five bones whose skinned region is the chin, the
jaw body and the lower cheek; none is an ancestor of the mouth, eye or nose bones and none is a
head-surface root (checked by ``BridgeCandidateRenderer.dependency_check`` per candidate).

Persistent encoding (``persistent_mapping.py``, installed HS2ABMX 4.4.6 BoneModifier.Apply):
a non-unit LengthModifier re-normalises the *current* localPosition every Apply, so a nonzero
PositionModifier with length != 1 drifts frame after frame.  Each of the five bones is
re-expressed with length exactly 1 and PositionModifier = (card-load local position after the
original modifier) - _posBaseline, so the game takes _posBaseline + offset every Apply.  The
baselines come from the forward model's card-load ``CollectBaseline`` (pure function of native59,
which stays fixed).
"""
from __future__ import annotations

import copy

import numpy as np

GROUPS = (('cf_J_ChinTip_s',), ('cf_J_ChinLow',), ('cf_J_Chin_rs',), ('cf_J_CheekLow_L', 'cf_J_CheekLow_R'))
NAMES = tuple(n for g in GROUPS for n in g)
CHANNELS = ('scale', 'position', 'rotation')
F = np.float32


def controls():
    out = []
    for group in GROUPS:
        for channel in CHANNELS:
            for axis in range(3):
                rng = {'scale': 0.10, 'position': 0.012 if len(group) == 2 else 0.025, 'rotation': 3.0}[channel]
                out.append({'names': list(group), 'channel': channel, 'axis': axis, 'range': rng})
    return out


CONTROLS = controls()


def encode_persistent(model):
    """Length-1 rows reproducing the card's five chin bones (card-load baseline)."""
    native = model.identity['native59']
    res = model.evaluate([], native=None, apply_count=2, baseline_mode='card_load')
    key = tuple(float(np.float32(v)) for v in native)
    cache = model._card_load_cache[key][1]
    rows, details = [], []
    for name in NAMES:
        mods = [m for m in cache if m['name'] == name]
        if len(mods) != 1 or name in model.additional:
            raise ValueError('unique modifier without an additional producer required: ' + name)
        m = mods[0]
        if len(m['coordinate_modifiers']) != 1 or m['coordinate_modifiers'][0] is None:
            raise ValueError('single explicit coordinate modifier required: ' + name)
        if not m['cache']['_hasBaseline']:
            raise ValueError('collected baseline required: ' + name)
        orig = m['coordinate_modifiers'][0]
        local = res['bone_local'][m['id']]['local_position']
        base = m['cache']['_posBaseline']
        row = {'name': name, 'scale': [float(x) for x in orig['scale']], 'rotation': [float(x) for x in orig['rotation']],
               'length': 1.0,
               'position': [float(x) for x in (np.asarray(local, F) - np.asarray(base, F))]}
        rows.append(row)
        details.append({'name': name, 'original': copy.deepcopy(orig), 'card_load_local_position': list(local),
                        'pos_baseline': list(base), 'encoded_position': row['position']})
    return rows, details


def patch_from_q(q, base_rows):
    q = np.asarray(q, float)
    if q.shape != (len(CONTROLS),) or not np.isfinite(q).all():
        raise ValueError('complete finite control vector required')
    rows = {r['name']: copy.deepcopy(r) for r in base_rows}
    for value, c in zip(q, CONTROLS):
        for j, name in enumerate(c['names']):
            ch, ax = c['channel'], c['axis']
            sign = -1 if j and ((ch == 'position' and ax == 0) or (ch == 'rotation' and ax in (1, 2))) else 1
            if ch == 'scale':
                rows[name][ch][ax] = float(rows[name][ch][ax] * (1 + value * c['range']))
            else:
                rows[name][ch][ax] = float(rows[name][ch][ax] + sign * value * c['range'])
    out = [rows[n] for n in NAMES]
    if any(r['length'] != 1.0 for r in out):
        raise RuntimeError('persistent length-1 branch lost')
    return {'bones': out}
