"""Forward equality of the torch head geometry chain against the exact CPU chain (offline).

For one bridge window (one process per window: the hosted EyeLook needs Mono, initialised once
per process):

1. ``DiffGeoSession`` loads the window exactly as ``BridgeCandidateRenderer`` and evaluates the
   exact chain for the window's own patch (card + patch -> CapturedHeadForward -> hosted EyeLook
   -> skin_from_locals) for every mesh renderer at N-2 / N-1 / N;
2. the torch chain (exact arithmetic mode) evaluates the same patch: bone locals of every
   transform and frame, ABMX branch lists, the skinned 40-byte vertices and unity_ObjectToWorld
   of every (renderer, frame) are compared bit for bit;
3. the hosted EyeLook driven by the torch locals reproduces the exact look rotations;
4. the bridge renders' frozen root TRS (torch, N-1 locals + PoseFreeze write set) equals the
   render records' root_bone position / rotation (all 16 renders, every head renderer);
5. extra patches (``--extra``): the chin controls of docs/hs2_chin_fit.md (persistent encoding,
   ``controls.patch_from_q``) at q = 0, random q in [-1, 1]^36 and single-control extremes,
   two native59 vectors (all 59 sliders incl. ears; one hitting the 0 / 1 key ends), and a
   modifier removal; exact vs torch for every stream;
6. ``--capture-link``: afterwards the full ``BridgeCandidateRenderer`` (capture index) is built
   and the torch bytes of the window patch are compared with the captured upload bytes.

    .venv/Scripts/python.exe tools/diffgeo/forward_acceptance.py --window w_766d8184 --extra --out <dir>
    .venv/Scripts/python.exe tools/diffgeo/forward_acceptance.py --all --out <dir>     (one subprocess each)
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p_ in (str(ROOT), str(ROOT / 'tools' / 'offline_renderer')):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import numpy as np  # noqa: E402

OUT_DEFAULT = Path('E:/HS2_offline_renderer_data/diffgeo_20261025')
WORK = OUT_DEFAULT / 'work'
CHIN_WINDOWS = Path('E:/HS2_offline_renderer_data/chin_fit_20261025/configs/windows_a2.json')
V61 = ('w_766d8184', 'w_75e17db3', 'w_8a42b979', 'w_0b3351d6')
EXTRA_WINDOWS = ('w_766d8184', 'front')
F = np.float32


def window_config(key):
    import candidate_native_acceptance as cna
    if key not in cna.WINDOWS:
        cna.register_windows_config(CHIN_WINDOWS)
    return cna.config(key, None, WORK)


def extra_patches(session, seed=20261025):
    from src.render.chin_fit import controls as C
    rows, _ = C.encode_persistent(session.models[session.N])
    rng = np.random.default_rng(seed)
    out = [('chin_q0_encoded', C.patch_from_q(np.zeros(len(C.CONTROLS)), rows))]
    for k in range(4):
        out.append(('chin_q_random_%d' % k, C.patch_from_q(rng.uniform(-1, 1, len(C.CONTROLS)), rows)))
    for name, ch, axis, grp, val in (('chin_tip_scale_x_+1', 'scale', 0, 0, 1.0), ('chin_low_pos_z_-1', 'position', 2, 1, -1.0),
                                     ('cheeklow_pair_rot_y_+1', 'rotation', 1, 3, 1.0)):
        q = np.zeros(len(C.CONTROLS))
        i = next(j for j, c in enumerate(C.CONTROLS) if c['channel'] == ch and c['axis'] == axis
                 and c['names'] == list(C.GROUPS[grp]))
        q[i] = val
        out.append((name, C.patch_from_q(q, rows)))
    card = np.asarray(session.card_native, float)
    n1 = np.clip(card + rng.normal(0, 0.08, 59), 0.0, 1.0)
    n2 = rng.uniform(0, 1, 59)
    n2[[0, 13, 27, 41, 58]] = [0.0, 1.0, 0.0, 1.0, 0.5]
    out.append(('native59_card_perturbed', {'bones': [], 'native59': [float(F(v)) for v in n1]}))
    out.append(('native59_random_with_key_ends', {'bones': [], 'native59': [float(F(v)) for v in n2]}))
    out.append(('remove_cf_J_ChinLow', {'bones': [{'name': 'cf_J_ChinLow', 'remove': True}]}))
    return out


def compare(session, exact, torch_res):
    from src.render.diffgeo.forward import local_f32_bytes
    from src.render.diffgeo.session import stream_bytes, matrix_bytes_o2w, _row_bytes
    bad_streams, n = [], 0
    for (rid, fc), v in exact['skin_cache'].items():
        n += 1
        t = torch_res['skin'][fc][rid]
        if stream_bytes(t['vb']) != bytes(v[0]):
            a = np.frombuffer(bytes(v[0]), np.uint32)
            b = np.asarray(t['vb'].detach().numpy(), F).view(np.uint32).ravel()
            bad_streams.append({'renderer': session.renderers[rid]['name'], 'frame': fc, 'what': 'vb',
                                'words_differ': int((a != b).sum())})
        if matrix_bytes_o2w(t['o2w']) != bytes(v[1]):
            bad_streams.append({'renderer': session.renderers[rid]['name'], 'frame': fc, 'what': 'o2w'})
    bad_locals, nl = [], 0
    for fc in session.skin_frames:
        for tid, row in exact['locals'][fc].items():
            nl += 1
            if local_f32_bytes(torch_res['locals'][fc][tid]) != _row_bytes(row):
                bad_locals.append((fc, session.by_id[tid]['name'] if tid in session.by_id else tid))
    return {'streams': n, 'streams_bit_exact': n - len({(b['renderer'], b['frame']) for b in bad_streams}),
            'stream_failures': bad_streams[:20], 'locals': nl, 'locals_bit_exact': nl - len(bad_locals),
            'local_failures': bad_locals[:20]}


def frozen_roots(session, torch_res):
    fz = session.frozen_written()
    L1 = torch_res['locals'][session.N - 1]
    rows = eq = 0
    bad = []
    for k, rec in sorted(fz['records'].items()):
        for r in rec['data']['renderers']:
            if r['id'] not in session.head_renderers or not r.get('root_bone') or not r['enabled']:
                continue
            pos, rot, _ = session.frozen_root(L1, r['id'], k)
            rows += 1
            ok = (np.asarray(pos.detach().numpy(), F).tobytes() == np.asarray(r['root_bone']['position'], F).tobytes()
                  and np.asarray(rot.detach().numpy(), F).tobytes() == np.asarray(r['root_bone']['rotation'], F).tobytes())
            eq += ok
            if not ok:
                bad.append((k, r['id']))
    return {'rows': rows, 'bit_exact': eq, 'failures': bad[:10]}


def run_window(key, out, extra, capture_link):
    import torch
    from src.render.diffgeo.session import DiffGeoSession, stream_bytes
    cfg, patch = window_config(key)
    rep = {'window': key, 'nonce': cfg.nonce, 'patch': patch}
    t0 = time.time()
    s = DiffGeoSession(cfg)
    rep['init_s'] = time.time() - t0
    rep['renderers'] = [s.renderers[r]['name'] for r in s.mesh_renderers]
    rep['head_renderers'] = [s.renderers[r]['name'] for r in s.head_renderers]
    rep['frames'] = list(s.skin_frames)
    t1 = time.time()
    tr = s.torch_streams(patch)
    rep['torch_window_patch_s'] = time.time() - t1
    rep['eye_source'] = tr['eye_source']
    rep['window_patch'] = compare(s, s.baseline, tr)
    rep['window_patch']['branches_equal'] = all(
        [b['branches'] for b in tr['branches'][fc]] == [b['branches'] for b in s.models[fc].evaluate(
            patches=patch.get('bones') or (), native=None, apply_count=s.cfg.apply_count,
            baseline_mode='card_load')['branches']] for fc in s.skin_frames)
    th = s.torch_streams(patch, force_hosted_eye=True)
    rep['hosted_eye_on_torch_locals'] = dict(compare(s, s.baseline, th), source=th['eye_source'])
    try:
        rep['frozen_root'] = frozen_roots(s, tr)
    except Exception as e:          # noqa: BLE001 -- recorded, not hidden
        rep['frozen_root'] = {'error': repr(e)}
    if extra:
        rep['extra'] = []
        for name, p in extra_patches(s):
            t2 = time.time()
            ex = s.exact_streams(p)
            t3 = time.time()
            tt = s.torch_streams(p)
            t4 = time.time()
            c = compare(s, ex, tt)
            c.update({'name': name, 'eye_source': tt['eye_source'], 'exact_s': t3 - t2, 'torch_s': t4 - t3,
                      'patch': p})
            rep['extra'].append(c)
            print(json.dumps({k: c[k] for k in ('name', 'streams', 'streams_bit_exact', 'locals', 'locals_bit_exact')}),
                  flush=True)
    window_bytes = {(rid, fc): stream_bytes(tr['skin'][fc][rid]['vb']) for fc in tr['skin'] for rid in tr['skin'][fc]}
    harness = s.eye
    del s, tr, th
    gc.collect()
    if capture_link:
        import eye_look_acceptance as ela
        from src.render.hs2pipe.candidate import BridgeCandidateRenderer
        # Mono initialises once per process: the renderer receives this process's EyeLook
        # harness (built for the same window's static hierarchy) instead of a second runtime
        ela.Harness = lambda static, _h=harness: _h
        r = BridgeCandidateRenderer(cfg, start_executor=False)
        ok = n = 0
        bad = []
        for p in r.points:
            n += 1
            got = window_bytes.get((p['renderer'], p['frame']))
            if got is not None and got == p['captured']:
                ok += 1
            else:
                bad.append((p['chunk'], r.renderers[p['renderer']]['name'], p['frame']))
        rep['capture_link'] = {'captured_uploads': n, 'torch_equal_captured': ok, 'failures': bad[:10],
                               'exact_renderer_equal_captured': 'checked in BridgeCandidateRenderer.__init__ (raises otherwise)'}
    (out / ('%s.json' % key)).write_text(json.dumps(rep, indent=1, default=str), encoding='utf-8')
    print(json.dumps({'window': key, 'window_patch': {k: rep['window_patch'][k] for k in
                                                      ('streams', 'streams_bit_exact', 'locals', 'locals_bit_exact',
                                                       'branches_equal')},
                      'hosted_eye': rep['hosted_eye_on_torch_locals']['streams_bit_exact'],
                      'frozen_root': rep['frozen_root'], 'capture_link': rep.get('capture_link')}, default=str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window')
    ap.add_argument('--all', action='store_true', help='the four v6.1 windows and the chin-fit front window')
    ap.add_argument('--extra', action='store_true')
    ap.add_argument('--capture-link', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.all:
        for key in V61 + ('front',):
            cmd = [sys.executable, __file__, '--window', key, '--out', str(a.out)]
            if key in EXTRA_WINDOWS:
                cmd.append('--extra')
            if key in V61:
                cmd.append('--capture-link')
            print('==', ' '.join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        return
    run_window(a.window, a.out, a.extra, a.capture_link)


if __name__ == '__main__':
    main()
