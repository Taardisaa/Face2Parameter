"""Render candidates on the paired bridge frame (v5.0.1 mode b) offline.

    python tools/offline_renderer/render_candidate.py --out-root E:/HS2_offline_renderer_data/candidate_20261025/run_vN \
        --identity [--candidate cand.json ...]

A candidate file is ``{"name": ..., "bones": [{"name", "position", "rotation", "scale", "length"}, ...]}``
(bridge ABMX format) and/or ``"native59": [59 floats]``.  ``--identity`` renders the
baseline card with an empty patch (every regenerated stream substituted) and compares the
target with the template replay and ``bridge_capture_2.png``.  See
``src/render/hs2pipe/candidate.py`` for the chain, the assumption and the dependency rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.render.hs2pipe.candidate import BridgeCandidateRenderer, CandidateUnsupported, SessionConfig  # noqa: E402


def png_pixels(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGBA'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-root', required=True, type=Path)
    ap.add_argument('--identity', action='store_true')
    ap.add_argument('--candidate', action='append', default=[], type=Path)
    ap.add_argument('--repeat', type=int, default=1, help='render each candidate N times (timing)')
    ap.add_argument('--no-png', action='store_true', help='skip PNG encoding (pixels are compared in memory)')
    ap.add_argument('--reference-png', type=Path,
                    default=Path('E:/HS2_offline_renderer_data/paired_session_v5_1_20261021/bridge_capture_2.png'))
    a = ap.parse_args(argv)
    if a.out_root.exists():
        raise FileExistsError('Fresh output root required')
    a.out_root.mkdir(parents=True)
    t0 = time.time()
    r = BridgeCandidateRenderer(SessionConfig(work=a.out_root / 'work'))
    report = {'init_s': time.time() - t0, 'executor_ready': r.ready, 'template_target_sha256': r.template_target_sha,
              'target_command': r.target_command, 'skin_points': len(r.points), 'renders': []}
    ref = png_pixels(a.reference_png)
    jobs = []
    if a.identity:
        jobs.append(('identity', {'name': 'identity', 'bones': []}, True))
    for c in a.candidate:
        cand = json.loads(c.read_text(encoding='utf-8'))
        jobs.append((cand.get('name') or c.stem, cand, False))
    try:
        for name, cand, patch_all in jobs:
            for k in range(a.repeat):
                d = a.out_root / ('%s_%02d' % (name, k))
                row = {'name': name, 'repeat': k}
                try:
                    res = r.render(cand, d, patch_all=patch_all, write_png=not a.no_png)
                except CandidateUnsupported as e:
                    row['unsupported'] = str(e)
                    report['renders'].append(row)
                    continue
                pix = res.pixels
                diff = np.any(pix != ref, axis=2)
                ys, xs = np.nonzero(diff)
                row.update({'png': str(res.png) if res.png else None, 'target_sha256_raw': res.target_sha256,
                            'equal_template_replay': res.target_sha256 == r.template_target_sha,
                            'equal_reference_png_pixels': bool(not diff.any()),
                            'pixels_differing_from_reference': int(diff.sum()),
                            'diff_bbox_xyxy': ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                                               if diff.any() else None),
                            'max_abs_channel_diff': int(np.abs(pix.astype(int) - ref.astype(int)).max()),
                            'changed_bones': res.changed_bones, 'patched': res.patched, 'timings': res.timings,
                            'dependency': {k: v for k, v in res.dependency.items() if k != 'citations'}})
                report['renders'].append(row)
                print(json.dumps({k: row[k] for k in ('name', 'repeat', 'equal_template_replay',
                                                      'equal_reference_png_pixels', 'pixels_differing_from_reference',
                                                      'diff_bbox_xyxy')}), json.dumps(res.timings, default=str))
    finally:
        r.close()
    report['citations'] = __import__('src.render.hs2pipe.candidate', fromlist=['x']).SOURCE_CITATIONS
    (a.out_root / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
