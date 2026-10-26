"""DFO-LS chin/jaw fit through the offline candidate renderer (no game).

    python tools/chin_fit/run_fit.py --windows <windows.json> --out <fresh dir> [--maxfun 400]

``windows.json``: {"bulk": <session bulk dir>, "artifacts": <session receipt dir>,
"card_state": <state file name in artifacts>, "views": {"front": {"nonce", "dir", "target", "png"}, ...}}
(``dir`` = convert_window.py output with ``template_v1`` and ``capture_program``).

Every window runs in its own ``view_worker.py`` process (renderer + resident executor +
detector).  Per evaluation: control vector q -> persistent ABMX patch (``controls``) -> every
worker computes the forward chain and skin streams and its part of the feature guard
(``protection``) -> if the combined guard is feasible, every worker renders its bridge target
(at most two executors run at a time) and returns its pixels; the fitting process runs the fixed
MediaPipe contract -> residual vs the art (``objective.score``).  Infeasible proposals are not rendered; they get the baseline
residual plus 20 x the guard violations (as in guarded_predictor_fit.py); a detector-domain
rejection gets the baseline residual plus a unit flag.  Neither can become the best candidate.
Every evaluation is appended to ``evaluations.jsonl`` (q, patch, class, cost, per-view rms, raw
D3D target SHA-256 per view, guard metrics, timings).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Listener
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DFOLS_DEPS = Path('C:/Users/13666/Workspace/HS2Mod/tools/chenger_fit_deps')
for p in (ROOT, DFOLS_DEPS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from src.render.chin_fit import controls as ctl  # noqa: E402
from src.render.chin_fit import objective as obj  # noqa: E402
from src.render.chin_fit import protection as prot  # noqa: E402

PENALTY = 20.0
MAX_GPU = 2


def png_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'))


class Workers:
    def __init__(self, windows_path, cfg, out):
        self.key = secrets.token_bytes(16)
        self.listener = Listener(('127.0.0.1', 0), backlog=16, authkey=self.key)
        port = self.listener.address[1]
        self.procs, self.conns = {}, {}
        views = [v for v in obj.VIEWS if v in cfg['views']]
        (out / 'workers').mkdir(parents=True, exist_ok=True)
        for v in views:
            log = open(out / 'workers' / ('%s.log' % v), 'ab')
            self.procs[v] = subprocess.Popen(
                [sys.executable, str(ROOT / 'tools/chin_fit/view_worker.py'), '--port', str(port),
                 '--authkey-hex', self.key.hex(), '--view', v, '--windows', str(windows_path),
                 '--work', str(out / ('work_' + v))], stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        pending = [self.listener.accept() for _ in views]      # all handshakes first, then the hellos
        for c in pending:
            hello = c.recv()
            if 'error' in hello:
                raise RuntimeError('worker %s failed: %s' % (hello['view'], hello['error']))
            self.conns[hello['view']] = c
            print('worker', hello['view'], 'ready', round(hello['init_s'], 1), 's', flush=True)
        self.views = views

    def call(self, view, req):
        self.conns[view].send(req)
        resp = self.conns[view].recv()
        if not resp['ok']:
            raise RuntimeError('worker %s %s: %s' % (view, req['cmd'], resp['error']))
        return resp['result']

    def map(self, reqs, max_workers):
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {v: ex.submit(self.call, v, r) for v, r in reqs.items()}
            return {v: f.result() for v, f in futs.items()}

    def close(self):
        for v, c in self.conns.items():
            try:
                c.send({'cmd': 'quit'})
                c.recv()
            except Exception:
                pass
        for p in self.procs.values():
            try:
                p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                p.kill()
        self.listener.close()


def lip_box(points):
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C
    ids = sorted({i for e in C.FACE_LANDMARKS_LIPS for i in (e.start, e.end)})
    lips = np.asarray(points)[ids, :2]
    return [float(v) for v in (*lips.min(axis=0), *lips.max(axis=0))]


class Fit:
    def __init__(self, windows_path, cfg, out):
        self.cfg, self.out = cfg, out
        self.w = Workers(windows_path, cfg, out)
        self.views = self.w.views
        self.det = obj.FixedDetector()
        self.refs = {v: obj.ReferenceView.load(v) for v in self.views}
        self.best, self.n = None, 0
        self.log = (out / 'evaluations.jsonl').open('x', encoding='utf-8')

    def write(self, name, value):
        (self.out / (name + '.json')).write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str),
                                                  encoding='utf-8')

    # ---------------------------------------------------------------- setup
    def setup(self):
        ident = self.w.map({v: {'cmd': 'identity', 'patch': {'bones': []}, 'out': str(self.out / 'baseline' / v)}
                            for v in self.views}, MAX_GPU)
        base_rgb = {}
        for v in self.views:
            img = png_rgb(ident[v]['png'])
            live = png_rgb(self.cfg['views'][v]['png'])
            diff = np.any(img != live, axis=2)
            ident[v].update(equal_template_replay=ident[v]['target_sha256_raw'] == ident[v]['template_target_sha256'],
                            equal_live_png=bool(not diff.any()), pixels_differing=int(diff.sum()))
            base_rgb[v] = img
        gaps = self.cfg.get('known_identity_gaps') or {}
        for v, x in ident.items():
            if not x['equal_template_replay']:
                raise RuntimeError('offline baseline differs from its own template replay: ' + v)
            if not x['equal_live_png']:
                if v not in gaps:
                    raise RuntimeError('baseline identity failed: %s %r' % (v, x))
                d = np.abs(base_rgb[v].astype(int) - png_rgb(self.cfg['views'][v]['png']).astype(int)).max(axis=2)
                x['known_renderer_gap'] = gaps[v]
                x['max_abs_channel_diff'] = int(d.max())
        self.write('identity', ident)
        anchors = self.w.map({v: {'cmd': 'anchors', 'bindings': obj.load_bindings(v)} for v in self.views}, 4)
        doms, rect_search = {}, {}
        for v in self.views:
            rect = None
            if obj.MEASURE[v] == 'landmarks':
                rect, tried = obj.select_rect(self.det, base_rgb[v], v)
                rect_search[v] = {'chosen': rect, 'tried_count': len(tried),
                                  'accepted_count': sum(1 for t in tried if t['faces'] == 1)}
            doms[v] = obj.ChinObjective.build_domain(v, {int(k): x for k, x in anchors[v].items()}, self.refs[v],
                                                     rect, hashlib.sha256(Path(ident[v]['png']).read_bytes()).hexdigest())
        self.objective = obj.ChinObjective(doms, self.refs, self.det)
        front_pts = self.det.detect(base_rgb['front'], doms['front'].rect, 'front')
        self.lips = lip_box(front_pts)
        lip_bind = self.w.call('front', {'cmd': 'bind_pixel', 'px': front_pts[17, :2].tolist()})
        lip_bind['id'] = 17
        self.lip_binding = lip_bind
        for v in self.views:
            if obj.MEASURE[v] == 'silhouette':
                lp = self.w.call(v, {'cmd': 'anchors', 'bindings': [lip_bind]})
                self.objective.set_lip_translation(v, list(lp.values())[0], base_rgb[v])
        self.write('domain', {'views': {v: d.to_json() for v, d in doms.items()}, 'rect_search': rect_search,
                              'front_lip_box_render_px': self.lips, 'lower_lip_binding': self.lip_binding,
                              'profile_stations': {v: {'stations_art_px': st.tolist(), 'normals': nv.tolist()}
                                                   for v, (st, nv) in self.objective.stations.items()},
                              'silhouette': {'level': obj.SILHOUETTE_LEVEL, 'bg_strip': obj.BG_STRIP,
                                             'station_step_art_px': obj.STATION_STEP},
                              'references': {v: {'art': self.refs[v].art_path, 'sha256': self.refs[v].art_sha256,
                                                 'chain_ids': self.refs[v].chain_ids,
                                                 'anchor_ids': self.refs[v].anchor_ids,
                                                 'eye_unit_px': self.refs[v].eye_unit_px,
                                                 'step4_orientation': self.refs[v].step4_orientation}
                                             for v in self.views},
                              'detector': {'model': str(obj.MODEL), 'sha256': obj.MODEL_SHA256, 'options': obj.OPTIONS}})
        m = self.w.call('front', {'cmd': 'masks', 'lip_box': self.lips})
        self.write('guard_roles', m['roles'])
        self.w.map({v: {'cmd': 'set_guard', 'masks': m['masks'], 'area_ratio': v == 'front'} for v in self.views}, 4)
        enc = self.w.map({v: {'cmd': 'encode'} for v in self.views}, 4)
        rows0 = enc['front']['rows']
        for v, e in enc.items():
            if e['rows'] != rows0:
                raise RuntimeError('card-load chin baselines differ between windows: ' + v)
        self.base_rows = rows0
        self.write('encoding', {'rows': rows0, 'details': enc['front']['details'], 'controls': ctl.CONTROLS,
                                'native59_fixed': True, 'length_exactly_one': True})
        live_res, live_rep = self.objective.score({v: png_rgb(self.cfg['views'][v]['png']) for v in self.views})
        self.baseline_residual, rep = self.objective.score(base_rgb)
        self.write('baseline_live_vs_offline', {
            'live_cost': float(live_res @ live_res), 'offline_cost': float(self.baseline_residual @ self.baseline_residual),
            'max_abs_residual_difference_eye_units': float(np.abs(live_res - self.baseline_residual).max()),
            'views': {v: {'live_rms_px': live_rep[v]['rms_px'], 'offline_rms_px': rep[v]['rms_px']} for v in self.views}})
        self.slices, k0 = {}, 0
        for v in self.objective.views:
            n = (2 * len(self.refs[v].chain_ids) if obj.MEASURE[v] == 'landmarks'
                 else len(self.objective.stations[v][0]))
            self.slices[v] = (k0, k0 + n)
            k0 += n
        self.baseline_cost = float(self.baseline_residual @ self.baseline_residual)
        self.write('baseline_score', {'cost': self.baseline_cost, 'views': rep})
        print('baseline cost', self.baseline_cost, {v: round(x['rms_px'], 2) for v, x in rep.items()}, flush=True)

    # ---------------------------------------------------------------- evaluation
    def evaluate(self, q, tag=None):
        q = np.clip(np.asarray(q, float), -1, 1)
        t0 = time.time()
        k = self.n
        self.n += 1
        patch = ctl.patch_from_q(q, self.base_rows)
        rows = self.w.map({v: {'cmd': 'guard', 'bones': patch['bones']} for v in self.views}, 4)
        guard = prot.combine(rows)
        info = {'evaluation': k, 'tag': tag, 'q': q.tolist(), 'patch': patch, 'guard': guard}
        nb = len(self.baseline_residual)
        if not guard['feasible']:
            res = np.r_[self.baseline_residual, PENALTY * np.asarray(guard['violation']), 0.0]
            info.update(classification='source_guard_rejection', rendered=False)
        else:
            outs = self.w.map({v: {'cmd': 'render', 'bones': patch['bones'], 'name': 'eval_%05d' % k,
                                   'out': str(self.out / 'evals' / ('%05d' % k) / v),
                                   'rect': self.objective.domains[v].rect} for v in self.views}, MAX_GPU)
            info['target_sha256_raw'] = {v: o['target_sha256_raw'] for v, o in outs.items()}
            info['changed_bones'] = outs['front']['changed_bones']
            info['rendered'] = True
            try:
                meas, rep = self.objective.score({v: o['rgb'] for v, o in outs.items()})
            except obj.PredictionUnavailable as e:
                res = np.r_[self.baseline_residual, np.zeros(4), 1.0]
                info.update(classification='detector_domain_rejection', failed_view=e.view, faces=e.faces)
            else:
                res = np.r_[meas, np.zeros(5)]
                cost = float(meas @ meas)
                info.update(classification='rendered_and_scored', cost=cost,
                            view_rms_px={v: x['rms_px'] for v, x in rep.items()},
                            view_cost={v: float(np.sum(np.square(meas[a:b]))) for v, (a, b) in self.slices.items()})
                if self.best is None or cost < self.best['cost']:
                    self.best = dict(info, report=rep)
                    self.write('best', self.best)
        if len(res) != nb + 5:
            raise RuntimeError('residual length changed')
        info['residual_cost_incl_penalty'] = float(res @ res)
        info['seconds'] = round(time.time() - t0, 2)
        self.log.write(json.dumps(info, default=str) + '\n')
        self.log.flush()
        print(k, info['classification'], round(info.get('cost', float('nan')), 6),
              {v: round(x, 2) for v, x in info.get('view_rms_px', {}).items()}, info['seconds'], 's', flush=True)
        return res

    def finalize(self):
        """Re-render the best evaluated patch with PNGs and re-score it (must equal its record)."""
        if self.best is None:
            raise RuntimeError('no rendered and scored evaluation')
        patch = self.best['patch']
        outs = self.w.map({v: {'cmd': 'render', 'bones': patch['bones'], 'name': 'best', 'write_png': True,
                               'out': str(self.out / 'best_render' / v), 'rect': self.objective.domains[v].rect}
                           for v in self.views}, MAX_GPU)
        shas = {v: o['target_sha256_raw'] for v, o in outs.items()}
        meas, rep = self.objective.score({v: o['rgb'] for v, o in outs.items()})
        same = shas == self.best['target_sha256_raw']
        (self.out / 'best_render' / 'score.json').write_text(json.dumps(
            {'cost': float(meas @ meas), 'views': rep, 'target_sha256_raw': shas,
             'png': {v: o['png'] for v, o in outs.items()},
             'equal_to_recorded_evaluation': same, 'evaluation': self.best['evaluation']}, indent=1), encoding='utf-8')
        (self.out / 'best_patch.json').write_text(json.dumps(
            dict(patch, name='chin_fit_best_eval_%05d' % self.best['evaluation']), indent=1), encoding='utf-8')
        if not same:
            raise RuntimeError('best re-render differs from its recorded evaluation')
        print('best', self.best['evaluation'], self.best['cost'], flush=True)

    def close(self):
        self.log.close()
        self.w.close()
        self.det.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--windows', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--maxfun', type=int, default=400)
    ap.add_argument('--rhobeg', type=float, default=0.15)
    ap.add_argument('--rhoend', type=float, default=0.002)
    ap.add_argument('--setup-only', action='store_true', help='identity, domain, guard, encoding and q=0 only')
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError('fresh output directory required: %s' % a.out)
    a.out.mkdir(parents=True)
    cfg = json.loads(a.windows.read_text(encoding='utf-8'))
    (a.out / 'windows.json').write_text(json.dumps(cfg, indent=1), encoding='utf-8')
    import dfols
    fit = Fit(a.windows.resolve(), cfg, a.out)
    try:
        fit.setup()
        x0 = np.zeros(len(ctl.CONTROLS))
        fit.write('policy', {'controls': ctl.CONTROLS, 'bounds': [-1, 1], 'rhobeg': a.rhobeg, 'rhoend': a.rhoend,
                             'maxfun': a.maxfun, 'penalty': PENALTY, 'dfols': dfols.__version__,
                             'objective': 'renderer bridge target -> fixed MediaPipe -> art chain residual (objective.py)',
                             'baseline_cost': fit.baseline_cost, 'chin_accepted': False})
        fit.evaluate(x0, tag='q0_encoded_baseline')
        if a.setup_only:
            return
        t0 = time.time()
        sol = dfols.solve(fit.evaluate, x0, bounds=(-np.ones_like(x0), np.ones_like(x0)), rhobeg=a.rhobeg,
                          rhoend=a.rhoend, maxfun=a.maxfun, objfun_has_noise=False,
                          user_params={'restarts.use_restarts': False})
        fit.write('solver', {'flag': int(sol.flag), 'message': sol.msg, 'nf': int(sol.nf), 'obj': float(sol.obj),
                             'x': sol.x.tolist(), 'seconds': time.time() - t0,
                             'best_evaluation': fit.best['evaluation'] if fit.best else None})
        print('solver', sol.flag, sol.msg, sol.nf, sol.obj, flush=True)
        fit.finalize()
    finally:
        fit.close()


if __name__ == '__main__':
    main()
