"""Demo: the chin PROXY (``src/render/diffgeo/chin_proxy.py``) against the exact chin objective.

Reads evaluated patches of a chin fit (``evaluations.jsonl``: patch, exact cost and per-view RMS
measured by the chin agent on exact renders) and evaluates the proxy on the same patches -- no
rendering, no fit.  Also reports the proxy gradient w.r.t. the 36 chin controls at q = 0 and a
finite-difference check of the proxy along that gradient.

    .venv/Scripts/python.exe tools/diffgeo/proxy_demo.py --fit-dir <chin fit dir> --out <dir>
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

FIT_V2 = Path('E:/HS2_offline_renderer_data/chin_fit_20261025/deprecated/fit_v2_stopped_by_user_dfols_cancelled')
WINDOWS = Path('E:/HS2_offline_renderer_data/chin_fit_20261025/configs/windows_a2.json')


def pick(evals, k):
    ok = [e for e in evals if e.get('classification') == 'rendered_and_scored' and e.get('view_rms_px')]
    ok.sort(key=lambda e: e['cost'])
    idx = sorted(set(int(round(x)) for x in np.linspace(0, len(ok) - 1, k)))
    chosen = [ok[i] for i in idx]
    q0 = [e for e in ok if e.get('tag') == 'q0_encoded_baseline']
    if q0 and q0[0] not in chosen:
        chosen.insert(0, q0[0])
    return chosen


def main(argv=None):
    from tools.diffgeo.forward_acceptance import window_config
    from src.render.diffgeo.session import DiffGeoSession
    from src.render.diffgeo.chin_proxy import ChinProxy, patch_from_q_torch, loss_and_grad
    from src.render.chin_fit import controls as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit-dir', type=Path, default=FIT_V2)
    ap.add_argument('--windows', type=Path, default=WINDOWS)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    cfg, _ = window_config('front')
    t0 = time.time()
    s = DiffGeoSession(cfg, renderers='head')
    proxy = ChinProxy(s, a.fit_dir, a.windows)
    rep = {'label': 'PROXY of the chin objective (not the authoritative score)', 'fit_dir': str(a.fit_dir),
           'setup_s': time.time() - t0,
           'views': {v: {k: (x if not isinstance(x, np.ndarray) else None) for k, x in p.notes.items()}
                     for v, p in proxy.pv.items()},
           'bindings': {v: p.bindings for v, p in proxy.pv.items() if p.bindings}}
    evals = [json.loads(l) for l in (a.fit_dir / 'evaluations.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    rows = []
    for e in pick(evals, a.n):
        t1 = time.time()
        with torch.no_grad():
            total, vr, _ = proxy.loss(e['patch'])
        rows.append({'evaluation': e['evaluation'], 'tag': e.get('tag'), 'exact_cost': e['cost'],
                     'proxy_cost': float(total), 'exact_view_rms_px': e['view_rms_px'],
                     'proxy_view_rms_px': {v: vr[v]['rms_px'] for v in vr}, 'proxy_s': time.time() - t1})
        print(json.dumps({k: rows[-1][k] for k in ('evaluation', 'exact_cost', 'proxy_cost')}),
              {v: (round(rows[-1]['exact_view_rms_px'][v], 2), round(rows[-1]['proxy_view_rms_px'][v], 2)) for v in vr},
              flush=True)
    ex = np.array([r['exact_cost'] for r in rows])
    pr = np.array([r['proxy_cost'] for r in rows])
    rk = lambda x: np.argsort(np.argsort(x))
    rep['patches'] = rows
    rep['spearman_cost'] = float(np.corrcoef(rk(ex), rk(pr))[0, 1]) if len(rows) > 2 else None
    rep['pearson_cost'] = float(np.corrcoef(ex, pr)[0, 1]) if len(rows) > 2 else None
    base_rows = json.loads((a.fit_dir / 'encoding.json').read_text(encoding='utf-8'))['rows']
    q = torch.zeros(len(C.CONTROLS), dtype=torch.float64, requires_grad=True)
    t2 = time.time()
    patch = patch_from_q_torch(q, base_rows)
    total, rep_v, _ = proxy.loss(patch)
    (gq,) = torch.autograd.grad(total, [q])
    rep['grad_q0'] = {'cost': float(total), 'grad': gq.tolist(), 'seconds_loss_and_grad': time.time() - t2,
                      'controls': [{'names': c['names'], 'channel': c['channel'], 'axis': c['axis']} for c in C.CONTROLS]}
    d = -gq / gq.norm()
    fd = []
    for h in (1e-2, 3e-3, 1e-3):
        with torch.no_grad():
            fp = float(proxy.loss(patch_from_q_torch(q.detach() + h * d, base_rows))[0])
            fm = float(proxy.loss(patch_from_q_torch(q.detach() - h * d, base_rows))[0])
        fd.append({'h': h, 'central_fd': (fp - fm) / (2 * h), 'autograd': float(gq @ d)})
    rep['directional_check'] = fd
    # per-patch gradient API (bone tensors) on the q = 0 patch
    leafed = {'bones': [dict(b, **{k: torch.as_tensor(np.asarray(b[k], float), dtype=torch.float64).requires_grad_(True)
                                   for k in ('scale', 'position', 'rotation')}) for b in base_rows]}
    t3 = time.time()
    c0, grads, _ = loss_and_grad(proxy, leafed)
    rep['loss_and_grad_api'] = {'cost': c0, 'seconds': time.time() - t3,
                                'grad_norms': {k: float(v.norm()) for k, v in grads.items() if v is not None}}
    (a.out / 'proxy_demo.json').write_text(json.dumps(rep, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('spearman_cost', 'pearson_cost', 'directional_check')}, default=str))
    print('loss+grad %.2fs' % rep['grad_q0']['seconds_loss_and_grad'])


if __name__ == '__main__':
    main()
