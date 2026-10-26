"""Per-step timing: torch head geometry chain vs the exact CPU chain (chin-fit front window, CPU).

Each figure is the median of ``--repeat`` warm evaluations of a chin-control patch (random q):

* exact chain, as ``BridgeCandidateRenderer._streams`` runs it for a candidate (CapturedHeadForward
  over N-4..N, hosted EyeLook, emulated skinning of every head surface at N-2 / N-1 / N with the
  unchanged-closure reuse);
* exact chain restricted to what the torch step evaluates (CapturedHeadForward at N-2 and N-1,
  emulated ``skin_from_locals`` of o_head at N-2);
* torch, exact arithmetic mode: forward (N-2, N-1, o_head) and forward + backward of a scalar;
* torch, float64 mode: forward + backward;
* chin PROXY: cost and gradient over the four views (``loss_and_grad`` path).

The exact renderer's wall time per fit evaluation (GPU frame executor, four views) is quoted from
the chin fit's ``evaluations.jsonl`` ``seconds`` field for context.

    .venv/Scripts/python.exe tools/diffgeo/timing.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import statistics
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


def med(fn, n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {'median_s': statistics.median(ts), 'min_s': min(ts), 'n': n}


def main(argv=None):
    from tools.diffgeo.forward_acceptance import window_config
    from src.render.diffgeo.session import DiffGeoSession
    from src.render.diffgeo.fp import Arith
    from src.render.diffgeo.chin_proxy import ChinProxy, patch_from_q_torch
    from src.render.chin_fit import controls as C
    from src.render.unity_skin_emu import skin_from_locals
    from src.render.hs2pipe.candidate import _frame_locals
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeat', type=int, default=5)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    cfg, _ = window_config('front')
    s = DiffGeoSession(cfg, renderers='head')
    rows, _ = C.encode_persistent(s.models[s.N])
    rng = np.random.default_rng(9)
    qs = [rng.uniform(-1, 1, len(C.CONTROLS)) for _ in range(a.repeat + 1)]
    it = iter(qs)
    head = next(r for r in s.head_renderers if s.renderers[r]['name'] == 'o_head')
    rep = {'window': 'front', 'cpu_only': True, 'threads': torch.get_num_threads()}

    rep['exact_candidate_streams'] = med(lambda: s.exact_streams(C.patch_from_q(next(it), rows)), a.repeat)
    it = iter(qs)

    def exact_min():
        p = C.patch_from_q(next(it), rows)['bones']
        L = {}
        for fc in (s.N - 2, s.N - 1):
            L[fc] = s.models[fc].evaluate(patches=p, native=None, apply_count=2, baseline_mode='card_load')['bone_local']
        parent, loc = _frame_locals(L[s.N - 2], s.tf, s.by_id)
        r = s.renderers[head]
        m = dict(s.win.meshes[head])
        bl = s.ps.unity_blend(s.win.meshes[head], s.win.by_frame[s.N - 2]['W'][head])
        m['v'], m['nrm'], m['t'] = bl[:, 0:3], bl[:, 3:6], bl[:, 6:10]
        skin_from_locals(s.emu, parent, loc, r['bones'], r['root_bone'], m)
    rep['exact_same_scope'] = med(exact_min, a.repeat)

    def torch_step(sess, backward):
        q = torch.tensor(next(it), dtype=torch.float64, requires_grad=backward)
        tr = sess.torch_streams(patch_from_q_torch(q, rows), frames=(s.N - 2, s.N - 1), rids=[head])
        vb = tr['skin'][s.N - 2][head]['vb']
        if backward:
            vb[:, 0:3].sum().backward()
    for name, sess, bw in (('torch_exact_forward', s, False), ('torch_exact_forward_backward', s, True),
                           ('torch_f64_forward_backward', s.with_arith(Arith('f64')), True)):
        it = iter(qs)
        torch_step(sess, bw)            # warm the constant-row memo of this mode
        it = iter(qs)
        rep[name] = med(lambda: torch_step(sess, bw), a.repeat)
    proxy = ChinProxy(s, FIT_V2, WINDOWS)
    it = iter(qs)

    def proxy_step():
        q = torch.tensor(next(it), dtype=torch.float64, requires_grad=True)
        total, _, _ = proxy.loss(patch_from_q_torch(q, rows))
        torch.autograd.grad(total, [q])
    rep['proxy_loss_and_grad_4_views'] = med(proxy_step, a.repeat)
    ev = [json.loads(l) for l in (FIT_V2 / 'evaluations.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    secs = [e['seconds'] for e in ev if e.get('seconds') and e.get('rendered')]
    rep['exact_renderer_fit_evaluation_s'] = {'median_s': statistics.median(secs), 'n': len(secs),
                                              'source': str(FIT_V2 / 'evaluations.jsonl'),
                                              'note': 'GPU frame executor, four views, MediaPipe and guard; one '
                                                      'objective value, no gradient'}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'timing.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
