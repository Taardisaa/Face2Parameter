"""Gradient check of the torch head geometry chain: autograd vs float64 central differences.

Both sides use the same torch forward in the float64 arithmetic mode (no binary32 rounding, torch
sin/cos instead of the CRT): the autograd gradient of a fixed random linear functional of the
skinned head geometry is compared with central finite differences of that functional.

Parameters (the chin controls of docs/hs2_chin_fit.md plus all native sliders):

* the persistent chin rows (``controls.encode_persistent``) of cf_J_ChinTip_s, cf_J_ChinLow,
  cf_J_Chin_rs, cf_J_CheekLow_L / R: scale xyz, position xyz, rotation xyz (45 values);
* ``native59`` (59 values).

Functional: sum(c1 * world positions) + sum(c2 * skinned normals) over o_head and o_namida at
frame N-2, world = the frozen bridge root of render 0 (N-1 locals) applied to the slot-0 stream.
Native-slider perturbations change the EyeLook inputs; the hosted EyeLook is re-evaluated in
every forward (finite differences include it), while autograd holds it constant -- o_head and
o_namida have no EyeLook bone, so the functional does not depend on it.

Central differences use h = 1e-6 (positions), 1e-6 relative (scale), 1e-4 deg (rotation),
1e-6 (native rates); a native rate whose +-h straddles a key segment boundary
(FloorToInt((n-1) rate)) would be reported as such.  Also reported: the exact-mode autograd
gradient against the float64-mode one (rounding-level differences only).

    .venv/Scripts/python.exe tools/diffgeo/gradient_check.py --window front --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p_ in (str(ROOT), str(ROOT / 'tools' / 'offline_renderer')):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import numpy as np  # noqa: E402
import torch  # noqa: E402

F = np.float32
SURFACES = ('o_head', 'o_namida')


def leaf(v):
    return torch.tensor(np.asarray(v, F).astype(np.float64), dtype=torch.float64, requires_grad=True)


class Functional:
    def __init__(self, session, rows, seed=7):
        self.s = session
        self.rows = rows
        self.rids = [r for r in session.head_renderers if session.renderers[r]['name'] in SURFACES]
        rng = np.random.default_rng(seed)
        self.c = {rid: (torch.tensor(rng.normal(size=(session.win.meshes[rid]['n'], 3))),
                        torch.tensor(rng.normal(size=(session.win.meshes[rid]['n'], 3))))
                  for rid in self.rids}

    def params(self):
        P = {}
        for r in self.rows:
            for ch in ('scale', 'position', 'rotation'):
                P[(r['name'], ch)] = leaf(r[ch])
        P[('native59', None)] = leaf(self.s.card_native)
        return P

    def patch(self, P):
        bones = []
        for r in self.rows:
            bones.append({'name': r['name'], 'scale': P[(r['name'], 'scale')], 'position': P[(r['name'], 'position')],
                          'rotation': P[(r['name'], 'rotation')], 'length': r['length']})
        return {'bones': bones, 'native59': P[('native59', None)]}

    def __call__(self, session, P):
        s = session
        tr = s.torch_streams(self.patch(P), frames=(s.N - 2, s.N - 1), rids=self.rids)
        total = 0.0
        L1 = tr['locals'][s.N - 1]
        for rid in self.rids:
            vb = tr['skin'][s.N - 2][rid]['vb']
            _, _, o2w = s.frozen_root(L1, rid, 0)
            world = vb[:, 0:3] @ o2w[:3, :3].T + o2w[:3, 3]
            c1, c2 = self.c[rid]
            total = total + (c1 * world).sum() + (c2 * vb[:, 3:6]).sum()
        return total


def steps(key):
    return {'scale': 1e-6, 'position': 1e-6, 'rotation': 1e-4, None: 1e-6}[key[1]]


def run(window, out):
    from tools.diffgeo.forward_acceptance import window_config
    from src.render.chin_fit import controls as C
    from src.render.diffgeo.session import DiffGeoSession
    from src.render.diffgeo.fp import Arith
    cfg, _ = window_config(window)
    t0 = time.time()
    s_ex = DiffGeoSession(cfg, renderers='head')
    s64 = s_ex.with_arith(Arith('f64'))
    rows, _ = C.encode_persistent(s_ex.models[s_ex.N])
    fn = Functional(s_ex, rows)
    rep = {'window': window, 'init_s': time.time() - t0, 'surfaces': list(SURFACES), 'parameters': {}}
    # autograd, float64 mode
    P = fn.params()
    t1 = time.time()
    f0 = fn(s64, P)
    t2 = time.time()
    f0.backward()
    t3 = time.time()
    rep['f64_forward_s'], rep['f64_backward_s'] = t2 - t1, t3 - t2
    g64 = {k: v.grad.detach().clone() for k, v in P.items()}
    # autograd, exact mode
    Pe = fn.params()
    t4 = time.time()
    fe = fn(s_ex, Pe)
    t5 = time.time()
    fe.backward()
    rep['exact_forward_s'], rep['exact_backward_s'] = t5 - t4, time.time() - t5
    gex = {k: v.grad.detach().clone() for k, v in Pe.items()}
    counts = {}
    for row in s_ex.models[s_ex.N].rig.customhead:
        counts.setdefault(int(row['category']), set()).add(len(s_ex.models[s_ex.N].rig.anm[row['bone']]))
    rep['native_key_counts'] = {k: sorted(v) for k, v in counts.items()}
    # central differences, float64 mode
    worst = 0.0
    tfd = time.time()
    for key, val in P.items():
        rows_out = []
        base = val.detach().clone()
        for i in range(base.numel()):
            h = steps(key) * (abs(float(base[i])) if key[1] == 'scale' else 1.0)
            note = None
            if key[0] == 'native59':
                # GetInfo selects the key segment by FloorToInt((n-1) * rate) for every bone of the
                # category (key counts from the installed cf_anmShapeHead tables)
                for nk in sorted(counts.get(i, ())):
                    if int(np.floor((nk - 1) * (float(base[i]) - h))) != int(np.floor((nk - 1) * (float(base[i]) + h))):
                        note = 'straddles a key segment boundary (%d-key table)' % nk
            vals = []
            for sgn in (+1, -1):
                Q = {k: v.detach().clone() for k, v in P.items()}
                Q[key] = base.clone()
                Q[key][i] = base[i] + sgn * h
                with torch.no_grad():
                    vals.append(float(fn(s64, Q)))
            fd = (vals[0] - vals[1]) / (2 * h)
            one_sided = None
            if note is not None:
                # at a segment boundary the forward selects the segment FloorToInt gives at the
                # value itself (the upper one for an exact key position): one-sided difference there
                with torch.no_grad():
                    f_0 = float(fn(s64, {k: v.detach().clone() for k, v in P.items()}))
                one_sided = (vals[0] - f_0) / h
            ad = float(g64[key].view(-1)[i])
            exg = float(gex[key].view(-1)[i])
            scale_ = max(abs(fd), abs(ad), 1e-8)
            err = abs(ad - fd) / scale_
            worst = max(worst, err) if note is None else worst
            rows_out.append({'index': i, 'value': float(base[i]), 'h': h, 'fd': fd, 'autograd_f64': ad,
                             'autograd_exact': exg, 'rel_err_f64': err,
                             'rel_diff_exact_vs_f64': abs(exg - ad) / scale_, 'note': note,
                             'one_sided_fd_upper_segment': one_sided,
                             'rel_err_one_sided': (None if one_sided is None else
                                                   abs(ad - one_sided) / max(abs(one_sided), abs(ad), 1e-8))})
        rep['parameters']['%s.%s' % key if key[1] else key[0]] = rows_out
        print('%-28s max rel err %.2e' % ('%s.%s' % key if key[1] else key[0], max(r['rel_err_f64'] for r in rows_out)),
              flush=True)
    rep['fd_s'] = time.time() - tfd
    rep['max_rel_err_f64'] = worst
    rep['max_rel_err_one_sided_at_key_boundaries'] = max(
        [r['rel_err_one_sided'] for v in rep['parameters'].values() for r in v if r['rel_err_one_sided'] is not None],
        default=None)
    rep['key_boundary_parameters'] = [(k, r['index'], r['note']) for k, v in rep['parameters'].items() for r in v
                                      if r['note']]
    rep['max_rel_diff_exact_vs_f64'] = max(r['rel_diff_exact_vs_f64'] for v in rep['parameters'].values() for r in v)
    rep['zero_gradient_parameters'] = [(k, r['index']) for k, v in rep['parameters'].items() for r in v
                                       if r['autograd_f64'] == 0.0]
    out.mkdir(parents=True, exist_ok=True)
    (out / ('gradient_check_%s.json' % window)).write_text(json.dumps(rep, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('max_rel_err_f64', 'max_rel_err_one_sided_at_key_boundaries',
                                           'key_boundary_parameters', 'max_rel_diff_exact_vs_f64', 'f64_forward_s',
                                           'f64_backward_s', 'exact_forward_s', 'exact_backward_s', 'fd_s')}))
    print('zero-gradient parameters:', rep['zero_gradient_parameters'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', default='front')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    run(a.window, a.out)


if __name__ == '__main__':
    main()
