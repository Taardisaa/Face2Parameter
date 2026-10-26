"""Review images of a chin fit: approved art vs baseline vs fitted render, per yaw.

    python tools/chin_fit/report_images.py --fit <fit dir> [--candidate-dir <dir with <view>/bridge_target.png>]
        [--candidate-score <score json>] --out <fresh dir>

Per view writes ``<view>_compare.png``: the art with the reference (red: MediaPipe chain on the
front / three-quarter views, silhouette stations on the profiles), the baseline measurement mapped
through the frozen frame (blue) and the fitted one (green); then the baseline and the fitted
bridge renders (same 1086x1448 canvas, unscaled relative to each other).  Also
``overview.png`` (all views stacked).  Images are for review only; no pixel is edited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIEWS = ('front', 'oblique', 'right', 'left')


def draw_chain(d, pts, colour, r=7, w=3, line=True):
    pts = [tuple(map(float, p)) for p in pts]
    if line and len(pts) > 1:
        d.line(pts, fill=colour, width=2)
    for x, y in pts:
        d.ellipse([x - r, y - r, x + r, y + r], outline=colour, width=w)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fit', required=True, type=Path)
    ap.add_argument('--candidate-dir', type=Path, help='default <fit>/best_render')
    ap.add_argument('--candidate-score', type=Path, help='default <fit>/best_render/score.json')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    dom = json.loads((a.fit / 'domain.json').read_text(encoding='utf-8'))
    base = json.loads((a.fit / 'baseline_score.json').read_text(encoding='utf-8'))['views']
    cdir = a.candidate_dir or a.fit / 'best_render'
    cand = json.loads((a.candidate_score or cdir / 'score.json').read_text(encoding='utf-8'))['views']
    rows = []
    for v in VIEWS:
        if v not in dom['views']:
            continue
        art = Image.open(dom['references'][v]['art']).convert('RGB')
        d = ImageDraw.Draw(art)
        prof = base[v].get('measure') == 'silhouette'
        draw_chain(d, base[v]['reference_xy_art_px'], (255, 40, 40), r=3 if prof else 7, line=not prof)
        draw_chain(d, base[v]['candidate_xy_art_px'], (60, 120, 255), r=3 if prof else 5, line=not prof)
        draw_chain(d, cand[v]['candidate_xy_art_px'], (40, 220, 60), r=3 if prof else 5, line=not prof)
        d.text((12, 12), '%s  art (red ref) / baseline (blue) %.1f px / fitted (green) %.1f px rms'
               % (v, base[v]['rms_px'], cand[v]['rms_px']), fill=(255, 255, 255))
        b = Image.open(a.fit / 'baseline' / v / 'bridge_target.png').convert('RGB')
        c = Image.open(cdir / v / 'bridge_target.png').convert('RGB')
        for im, lab, sc in ((b, 'baseline render yaw %s' % dom['views'][v]['yaw'], base[v]),
                            (c, 'fitted render yaw %s' % dom['views'][v]['yaw'], cand[v])):
            dd = ImageDraw.Draw(im)
            if 'candidate_xy_render_px' in sc:          # landmark views; profiles are silhouettes
                draw_chain(dd, sc['candidate_xy_render_px'], (255, 220, 0), r=4, w=2)
            dd.text((12, 12), lab, fill=(255, 255, 255))
        W = art.width + b.width + c.width
        H = max(art.height, b.height, c.height)
        panel = Image.new('RGB', (W, H), (0, 0, 0))
        panel.paste(art, (0, 0))
        panel.paste(b, (art.width, 0))
        panel.paste(c, (art.width + b.width, 0))
        panel.save(a.out / ('%s_compare.png' % v))
        rows.append(panel.resize((W // 2, H // 2)))
    over = Image.new('RGB', (max(r.width for r in rows), sum(r.height for r in rows)))
    y = 0
    for r in rows:
        over.paste(r, (0, y))
        y += r.height
    over.save(a.out / 'overview.png')
    print(a.out)


if __name__ == '__main__':
    main()
