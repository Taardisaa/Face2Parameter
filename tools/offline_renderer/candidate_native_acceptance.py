"""Native candidate acceptance on the probe v6.1 bridge windows (offline).

For each window (the game rendered card + that window's ABMX patch, or the baseline card):

1. head forward chain from the card + the window's patch (``CapturedHeadForward`` per probe
   frame, hosted EyeLook) -- the per-frame bone locals are compared with the window's own
   probe export (comparison only; nothing of the export geometry is used downstream);
2. skin streams N-2 / N-1 / N from those locals (``skin_from_locals``) -- every upload is
   compared with the captured bytes;
3. frozen-pose bridge matrices and bounds from the bridge ``pre_cull`` transforms with the
   forward locals (``BridgeCandidateRenderer.frozen_pose_check``) -- compared with the render
   records the template program was built from;
4. the window's own bridge frame program (``frame_orchestrate_paired`` template run of that
   capture) executed with *every* regenerated head stream substituted; the 1086x1448 target
   must equal that window's replay (oracle) and its ``bridge_capture_*.png``.

    python tools/offline_renderer/candidate_native_acceptance.py --window w_766d8184 [--pre-only]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

SESSION = Path('E:/HS2_offline_renderer_data/paired_session_v6_1_20261025')
ARTIFACTS = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_reference_fit_20261012/'
                 'paired_capture_session_v6_1_20261025')
OUT = Path('E:/HS2_offline_renderer_data/candidate_native_20261025')
WINDOWS = {
    'w_766d8184': {'nonce': '766d81849ec147a881b4214169a54185', 'name': 'baseline', 'patch': None, 'target': 44998},
    'w_75e17db3': {'nonce': '75e17db31ce44808a11abbeb7afef6a3', 'name': 'chin_tip_down_002',
                   'patch': 'patch_chin_tip_down_002.json', 'target': 60885},
    'w_8a42b979': {'nonce': '8a42b979a9454cdc8576bcb958bef814', 'name': 'chin_low_forward_002',
                   'patch': 'patch_chin_low_forward_002.json', 'target': 71318},
    'w_0b3351d6': {'nonce': '0b3351d62a2d439f8af95db8de79d712', 'name': 'chin_rs_wider_x108',
                   'patch': 'patch_chin_rs_wider_x108.json', 'target': 81031},
}
CARD_STATE = 'state_before_baseline.json'
# probe v7 multi-yaw baseline windows (empty patch); per-window card state before the window
V7_SESSION = Path('E:/HS2_offline_renderer_data/paired_session_v7_20261025')
V7_ARTIFACTS = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_reference_fit_20261012/'
                    'paired_capture_session_v7_20261025')
V7_OUT = Path('E:/HS2_offline_renderer_data/multiview_20261025/v7')
for _yaw, _tag, _nonce, _target in (('0', '0', '2daddb57cebe4294bdf792ffaf382127', 15092),
                                    ('+30', 'p30', 'c971d22436794d678e0b15fcc31f83db', 21735),
                                    ('+45', 'p45', '321808bd7d6a4e4d836b2cdbb3ff8481', 22898),
                                    ('+60', 'p60', '2d90e5e134114107921aba4827cfd9fd', 24601),
                                    ('+90', 'p90', '34ac9218fd964d9cb95fe20a80c7ddd2', 30864),
                                    ('-90', 'm90', 'ed32db74bb094d46bda6f1b096848f9e', 31817),
                                    ('-45', 'm45', '5c27666617b9483bb3ffb910a967ed48', 37000)):
    WINDOWS['v7_w_' + _nonce[:8]] = {
        'nonce': _nonce, 'name': 'yaw %s' % _yaw, 'patch': None, 'target': _target, 'session': V7_SESSION,
        'artifacts': V7_ARTIFACTS, 'out': V7_OUT / ('w_' + _nonce[:8]), 'card_state': 'state_before_yaw_%s.json' % _tag,
        'png': V7_SESSION / ('bridge_capture_%s_yaw%s.png' % (_nonce, _tag.replace('p', '') if _tag != '0' else '0'))}


def register_windows_config(path):
    """Register the views of a windows JSON (bulk session, artifacts, card_state, views) as windows."""
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    for view, v in c['views'].items():
        WINDOWS[view] = {'nonce': v['nonce'], 'name': view, 'patch': None, 'target': int(v['target']),
                         'session': Path(c['bulk']), 'artifacts': Path(c['artifacts']), 'out': Path(v['dir']),
                         'card_state': c['card_state'], 'png': Path(v['png'])}


def config(key, template_run, work=None):
    from src.render.hs2pipe.candidate import SessionConfig
    w = WINDOWS[key]
    patch = {'bones': []}
    session, artifacts = w.get('session', SESSION), w.get('artifacts', ARTIFACTS)
    out = w.get('out', OUT / key)
    if w['patch']:
        patch = json.loads((artifacts / w['patch']).read_text(encoding='utf-8'))
    cfg = SessionConfig(session=session, nonce=w['nonce'], artifacts=artifacts,
                        loaded_state=w.get('card_state', CARD_STATE),
                        template_run=template_run, capture_program=out / 'capture_program',
                        work=work or (out / 'candidate_work'), target_resource=w['target'], window_patch=patch)
    return cfg, patch


def forward_vs_export(r, streams):
    """Per probe frame: forward (card + patch) bone locals vs the window's probe export."""
    F = np.float32
    rows = []
    for fc, L in sorted(streams['locals'].items()):
        T = r.win.by_frame[fc]['T']
        diff = []
        for i, t in enumerate(r.tf):
            g = L.get(t['id'])
            if g is None:
                continue
            gb = np.concatenate([np.asarray(g['local_position'], F), np.asarray(g['local_rotation_xyzw'], F),
                                 np.asarray(g['local_scale'], F)]).tobytes()
            if gb != T[i, 0:10].tobytes():
                diff.append(t['name'])
        rows.append({'frame': fc, 'transforms': len(r.tf), 'differ': diff})
    return rows


def png_pixels(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGBA'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--window', required=True,
                    help='one of %s, or a view of --windows-config' % ', '.join(sorted(WINDOWS)))
    ap.add_argument('--windows-config', type=Path,
                    help='windows JSON with bulk / artifacts / card_state / views {view: nonce, dir, target, png} '
                         '(the tools/chin_fit layout); its views are registered as windows')
    ap.add_argument('--template-run', type=Path, help='frame_orchestrate_paired run of this window (template)')
    ap.add_argument('--pre-only', action='store_true', help='forward / skin / frozen-pose checks, no execution')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--work', type=Path, help='scratch directory (default: the window directory candidate_work)')
    a = ap.parse_args(argv)
    if a.windows_config:
        register_windows_config(a.windows_config)
    if a.window not in WINDOWS:
        ap.error('unknown window %r' % a.window)
    from src.render.hs2pipe.candidate import BridgeCandidateRenderer
    if a.out.exists():
        raise FileExistsError('Fresh output directory required: %s' % a.out)
    a.out.mkdir(parents=True)
    w = WINDOWS[a.window]
    cfg, patch = config(a.window, None if a.pre_only else a.template_run, a.work)
    t0 = time.time()
    report = {'window': a.window, 'nonce': w['nonce'], 'name': w['name'], 'patch': patch,
              'card_state': w.get('card_state', CARD_STATE)}
    try:
        r = BridgeCandidateRenderer(cfg, start_executor=not a.pre_only)
        report['skin_uploads'] = {'points': len(r.points), 'equal_captured': len(r.points)}
    except RuntimeError as e:
        report['init_error'] = str(e)
        (a.out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
        raise
    report['init_s'] = time.time() - t0
    report['eye_seed_source'] = getattr(r, 'eye_seed_source', None)
    report['eye_target_source'] = sorted(set(getattr(r, 'camera_source', {}).values()))
    report['forward_vs_export'] = forward_vs_export(r, r.baseline)
    report['frozen_pose'] = r.frozen_pose_check(r.baseline)
    if patch.get('bones'):
        # the patch is load-bearing: the card alone gives other locals and skin bytes
        card_only = r._streams(patches=(), native=None)
        ch = set()
        for v in r._changed_by_frame(card_only['locals']).values():
            ch |= v
        report['card_only_vs_window'] = {
            'transforms_differing': sorted(r.tf[r.index_of[t]]['name'] for t in ch if t in r.index_of),
            'skin_uploads_differing': sum(1 for p in r.points
                                          if card_only['bytes'][p['chunk']] != p['captured'])}
    if not a.pre_only:
        report['template_target_sha256'] = r.template_target_sha
        try:
            res = r.render(patch, a.out / 'identity', patch_all=True)
        finally:
            r.close()
        ref = png_pixels(w['png'] if 'png' in w else SESSION / ('bridge_capture_%s.png' % w['name']))
        diff = np.any(res.pixels != ref, axis=2)
        ys, xs = np.nonzero(diff)
        report['identity'] = {
            'target_sha256_raw': res.target_sha256, 'equal_template_replay': res.target_sha256 == r.template_target_sha,
            'equal_bridge_capture_png': bool(not diff.any()), 'pixels_differing_from_png': int(diff.sum()),
            'diff_bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if diff.any() else None,
            'patched': res.patched, 'changed_bones_vs_window': res.changed_bones, 'timings': res.timings}
    (a.out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: report.get(k) for k in ('window', 'name', 'skin_uploads', 'card_only_vs_window',
                                                  'identity', 'eye_seed_source', 'eye_target_source')}, default=str))
    print(json.dumps({'frozen': {k: v for k, v in report['frozen_pose'].items() if k != 'animator_rewritten'},
                      'forward_differ': [(x['frame'], len(x['differ'])) for x in report['forward_vs_export']]},
                     default=str))


if __name__ == '__main__':
    main()
