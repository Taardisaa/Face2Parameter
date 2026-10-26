"""One fitting window in its own process (BridgeCandidateRenderer + resident executor).

The EyeLook harness hosts the game's Mono runtime, which initialises once per process, so every
window of a fit runs in its own worker.  Started by ``run_fit.py``; talks over a local
``multiprocessing.connection`` (stdout/stderr are free for the renderer's own logging).

    python tools/chin_fit/view_worker.py --port P --authkey-hex K --view V --windows <json> --work <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'tools' / 'offline_renderer'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402


def make_renderer(cfg, view, work, start_executor=True):
    from src.render.hs2pipe.candidate import BridgeCandidateRenderer, SessionConfig
    w = cfg['views'][view]
    d = Path(w['dir'])
    sc = SessionConfig(session=Path(cfg['bulk']), nonce=w['nonce'], artifacts=Path(cfg['artifacts']),
                       loaded_state=cfg['card_state'], template_run=Path(w.get('template') or d / 'template_v1'),
                       capture_program=d / 'capture_program', work=Path(work),
                       target_resource=int(w['target']), window_patch=cfg.get('window_patch') or {'bones': []})
    return BridgeCandidateRenderer(sc, start_executor=start_executor)


class Worker:
    def __init__(self, cfg, view, work):
        t0 = time.time()
        self.view = view
        self.r = make_renderer(cfg, view, work)
        self.guard = None
        self.init_s = time.time() - t0

    def handle(self, req):
        from src.render.chin_fit import controls as ctl
        from src.render.chin_fit import protection as pr
        cmd = req['cmd']
        r = self.r
        if cmd == 'identity':
            res = r.render(dict(req['patch'], name='identity'), Path(req['out']), patch_all=True, write_png=True)
            return {'target_sha256_raw': res.target_sha256, 'template_target_sha256': r.template_target_sha,
                    'png': str(res.png), 'changed_bones': res.changed_bones}
        if cmd == 'encode':
            rows, details = ctl.encode_persistent(r.models[r.N])
            return {'rows': rows, 'details': details}
        if cmd == 'masks':
            masks, roles = pr.build_masks(r, req['lip_box'])
            return {'masks': {k: v.tolist() for k, v in masks.items()}, 'roles': roles}
        if cmd == 'anchors':
            # mesh-bound anchors (step-4 bindings) on the baseline slot-0 head stream, interpolated
            # in object space, then world (frozen root TRS) and the bridge camera of render 0
            cam, chunks, names = pr.bridge_camera(r), pr.slot0_chunks(r), pr.renderer_names(r)
            head = next(rid for rid, n in names.items() if n == 'o_head')
            v = pr.stream_positions(r.baseline['bytes'][chunks[head]])
            out = {}
            for bnd in req['bindings']:
                pt = np.asarray(bnd['barycentric'], float) @ v[bnd['vertex_ids']]
                out[int(bnd['id'])] = cam.pixels(cam.world(head, pt[None]))[0].tolist()
            return out
        if cmd == 'bind_pixel':
            return pr.raycast_binding(r, req['px'])
        if cmd == 'set_guard':
            self.guard = pr.ViewGuard(r, {k: np.asarray(v, bool) for k, v in req['masks'].items()},
                                      area_ratio=req['area_ratio'])
            return {'ok': True}
        if cmd == 'guard':
            st = r._streams(patches=req['bones'], native=None)
            g = self.guard.evaluate(st)
            g['changed_transforms'] = len(r.changed_transforms(st))
            return g
        if cmd == 'render':
            res = r.render({'name': req.get('name', 'eval'), 'bones': req['bones']}, Path(req['out']),
                           write_png=req.get('write_png', False))
            # pixels go back to the fitting process, which owns the detector (MediaPipe's
            # landmarker cannot be created in a process that hosts the game's Mono runtime)
            return {'target_sha256_raw': res.target_sha256, 'png': str(res.png) if res.png else None,
                    'changed_bones': res.changed_bones, 'timings': res.timings,
                    'rgb': np.ascontiguousarray(res.pixels[..., :3])}
        if cmd == 'quit':
            return {'bye': True}
        raise ValueError('unknown command ' + cmd)

    def close(self):
        self.r.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--authkey-hex', required=True)
    ap.add_argument('--view', required=True)
    ap.add_argument('--windows', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    a = ap.parse_args(argv)
    conn = Client(('127.0.0.1', a.port), authkey=bytes.fromhex(a.authkey_hex))
    cfg = json.loads(a.windows.read_text(encoding='utf-8'))
    try:
        w = Worker(cfg, a.view, a.work)
        conn.send({'ready': True, 'view': a.view, 'init_s': w.init_s})
    except Exception:
        conn.send({'error': traceback.format_exc(), 'view': a.view})
        raise
    try:
        while True:
            req = conn.recv()
            try:
                resp = {'ok': True, 'result': w.handle(req)}
            except Exception:
                resp = {'ok': False, 'error': traceback.format_exc()}
            conn.send(resp)
            if req.get('cmd') == 'quit':
                break
    finally:
        w.close()
        conn.close()


if __name__ == '__main__':
    main()
