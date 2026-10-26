"""Live acceptance of a chin fit: the saved and reloaded card, captured in the game, vs offline.

    python tools/chin_fit/live_acceptance.py --fit <fit dir> --windows-b <windows json of the
        acceptance session> --base-state <card state of the fit session> --out <fresh dir>

Session B loads the new card (base card + best patch, saved as a new file and reloaded), removes
the front hair and the blink exactly as the fit windows did, and captures the bridge renderstyle
at the fitting yaws through the probe.  Each window is converted (``convert_window.py``) with the
session-B card state.  This tool then checks, per window:

1. ABMX of the reloaded card = base card + best patch (the five chin bones equal the patch values,
   every other bone unchanged), from the bridge state files;
2. offline render of the reloaded card (its own forward chain, every stream substituted) =
   the window's template replay and = the live bridge PNG (pixel-equal);
3. re-applying the best patch offline gives the same target (the card holds exactly the patch);
4. the score of the live captures under the fit's frozen domain (``domain.json``) vs the score
   recorded for the best evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'tools' / 'chin_fit'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402

from src.render.chin_fit import objective as obj  # noqa: E402
from run_fit import Workers, png_rgb  # noqa: E402


def abmx_check(base_state, new_state, patch):
    base = {b['name']: b for b in base_state['abmx']['bones']}
    new = {b['name']: b for b in new_state['abmx']['bones']}
    want = {b['name']: b for b in patch['bones']}
    rows, ok = {}, set(base) == set(new)
    for name in sorted(set(base) | set(new)):
        nb = new.get(name)
        exp = want.get(name) or base.get(name)
        same = nb is not None and all(np.asarray(nb[k], np.float32).tobytes() == np.asarray(exp[k], np.float32).tobytes()
                                      for k in ('scale', 'position', 'rotation', 'length'))
        if name in want or not same:
            rows[name] = {'expected': {k: exp[k] for k in ('scale', 'position', 'rotation', 'length')},
                          'reloaded': None if nb is None else {k: nb[k] for k in ('scale', 'position', 'rotation', 'length')},
                          'equal_float32': same, 'patched': name in want}
        ok = ok and same
    return {'all_equal': ok, 'bones': rows, 'inventory_equal': set(base) == set(new)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit', required=True, type=Path)
    ap.add_argument('--windows-b', required=True, type=Path)
    ap.add_argument('--base-state', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    cfg = json.loads(a.windows_b.read_text(encoding='utf-8'))
    patch = json.loads((a.fit / 'best_patch.json').read_text(encoding='utf-8'))
    best_score = json.loads((a.fit / 'best_render' / 'score.json').read_text(encoding='utf-8'))
    dom = json.loads((a.fit / 'domain.json').read_text(encoding='utf-8'))
    report = {'fit': str(a.fit), 'windows_b': cfg, 'patch': patch}
    new_state = json.loads((Path(cfg['artifacts']) / cfg['card_state']).read_text(encoding='utf-8'))
    report['abmx'] = abmx_check(json.loads(a.base_state.read_text(encoding='utf-8')), new_state, patch)
    w = Workers(a.windows_b.resolve(), cfg, a.out)
    try:
        ident = w.map({v: {'cmd': 'identity', 'patch': {'bones': []}, 'out': str(a.out / 'card' / v)}
                       for v in w.views}, 2)
        repatch = w.map({v: {'cmd': 'render', 'bones': patch['bones'], 'name': 'best_patch_on_card', 'write_png': True,
                             'out': str(a.out / 'card_plus_patch' / v), 'rect': None} for v in w.views}, 2)
    finally:
        w.close()
    views = {}
    live, offline = {}, {}
    gaps = cfg.get('known_identity_gaps') or {}
    for v in ident:
        live[v] = png_rgb(cfg['views'][v]['png'])
        offline[v] = png_rgb(ident[v]['png'])
        d = np.abs(live[v].astype(int) - offline[v].astype(int)).max(axis=2)
        views[v] = {'offline_target_sha256_raw': ident[v]['target_sha256_raw'],
                    'template_target_sha256': ident[v]['template_target_sha256'],
                    'offline_equals_template_replay': ident[v]['target_sha256_raw'] == ident[v]['template_target_sha256'],
                    'offline_equals_live_png': bool(not d.any()), 'pixels_differing': int((d > 0).sum()),
                    'max_abs_channel_diff': int(d.max()),
                    'patch_reapplied_same_target': repatch[v]['target_sha256_raw'] == ident[v]['target_sha256_raw'],
                    'known_renderer_gap': gaps.get(v)}
    doms = {v: obj.ViewDomain.from_json(d) for v, d in dom['views'].items()}
    o = obj.ChinObjective(doms)
    r_live, rep_live = o.score(live)
    r_off, rep_off = o.score(offline)
    best_imgs = {v: png_rgb(best_score['png'][v]) for v in doms}
    for v in views:
        dd = np.abs(best_imgs[v].astype(int) - live[v].astype(int)).max(axis=2)
        views[v]['fit_session_best_render_vs_live_pixels_differing'] = int((dd > 0).sum())
        views[v]['fit_session_best_render_vs_live_max_abs'] = int(dd.max())
    report['views'] = views
    report['score'] = {'live_cost': float(r_live @ r_live), 'offline_session_b_cost': float(r_off @ r_off),
                       'fit_best_cost': best_score['cost'],
                       'live_view_rms_px': {v: x['rms_px'] for v, x in rep_live.items()},
                       'fit_best_view_rms_px': {v: x['rms_px'] for v, x in best_score['views'].items()},
                       'live_report': rep_live}
    report['accepted_mechanics'] = bool(report['abmx']['all_equal'] and all(
        x['offline_equals_template_replay'] and x['patch_reapplied_same_target'] and
        (x['offline_equals_live_png'] or x['known_renderer_gap']) for x in views.values()))
    report['chin_accepted'] = False
    (a.out / 'acceptance.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps({'abmx_equal': report['abmx']['all_equal'], 'views': {v: {k: x[k] for k in (
        'offline_equals_template_replay', 'offline_equals_live_png', 'pixels_differing', 'patch_reapplied_same_target')}
        for v, x in views.items()}, 'score': {k: report['score'][k] for k in ('live_cost', 'fit_best_cost',
                                                                              'live_view_rms_px')}}, indent=1))


if __name__ == '__main__':
    main()
