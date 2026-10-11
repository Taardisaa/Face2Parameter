"""Refine a reviewed portrait jaw annotation against its actual pixel transition.

This observes a 2D light/shadow boundary, not game deformation or hidden 3D.
Always inspect the result on the original portrait. Preserve prior annotations.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh annotation required')
    raw = a.prior.read_bytes()
    d = json.loads(raw.decode('utf-8-sig'))
    ref = d['reference']
    if hashlib.sha256(a.image.read_bytes()).hexdigest() != ref['image_sha256']:
        raise ValueError('Reference image hash mismatch')
    im = np.asarray(Image.open(a.image).convert('RGB'), float)
    lum = gaussian_filter(im @ np.array([.2126, .7152, .0722]), 2)
    dy = np.gradient(lum, axis=0)
    old = np.array(ref['left_xy'] + list(reversed(ref['right_xy'][:-1])))
    if np.any(np.diff(old[:, 0]) <= 0):
        raise ValueError('Reviewed contour must increase in X')
    xs = np.unique(np.r_[np.arange(old[0, 0], old[-1, 0]+1, 5),
                         old[:, 0], ref['chin_xy'][0]])
    ys = np.interp(xs, old[:, 0], old[:, 1])
    points = []
    for x, y in zip(xs, ys):
        lo, hi = int(y)-15, int(y)+16
        if lo < 0 or hi > len(dy) or not 0 <= x < dy.shape[1]:
            raise ValueError('Reviewed contour search band leaves image')
        found = lo + int(np.argmin(dy[lo:hi, int(x)]))
        if found in (lo, hi-1):
            raise ValueError('Boundary reaches search-band edge; inspect annotation')
        points.append([int(x), found])
    points = np.array(points)
    bottom = int(points[:, 1].max())
    center = int(np.median(points[points[:, 1] == bottom, 0]))
    ref['left_xy'] = points[points[:, 0] <= center].tolist()
    ref['right_xy'] = points[points[:, 0] >= center][::-1].tolist()
    ref['chin_xy'] = [center, bottom]
    d['scope'] = ('Reference annotation correction only: inspect actual diffuse-light/'
                  'shadow transition, retain previous 6px boundary uncertainty. '
                  'Original reference and old rejected results preserved; no character changes.')
    d['reference_annotation_provenance'] = {
        'prior_trace_sha256': hashlib.sha256(raw).hexdigest(),
        'method': ('RGB Rec709 luminance; Gaussian sigma2 pixels; most negative vertical '
                   'derivative inside prior manually reviewed contour +/-15px; inspect '
                   'against original image. Light/shadow transition is a 2D observation, '
                   'not proof of hidden 3D geometry.'),
        'search_radius_pixels': 15, 'sigma_pixels': 2, 'sampling_pixels': 5}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(json.dumps({'chin_xy': ref['chin_xy'], 'point_count': len(points)}))


if __name__ == '__main__':
    main()
