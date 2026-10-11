"""Compare hash-bound human jaw traces; retain scale and full visible contours.

Reads existing FAN eye/mouth points for alignment only. No image synthesis,
automatic parameter fitting, 3D reconstruction or acceptance threshold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from portrait_compare import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    observations = json.loads(args.measurements.read_text(encoding='utf-8'))
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    result = {'scope': trace['scope'], 'trace': trace, 'measurements_sha256':
              hashlib.sha256(args.measurements.read_bytes()).hexdigest()}
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)
    overlay = axes[2]
    # The old three sections missed the broad/flat terminal chin seen in v14.
    # Keep those sections and explicitly inspect the shoulder and tip as well.
    fractions = [.1, .25, .5, .75, .9, .95]
    for i, name in enumerate(('reference', 'candidate')):
        row, label = observations[name], trace[name]
        path = Path(row['path'])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != row['sha256'] or sha != label['image_sha256']:
            raise ValueError(f'Input hash mismatch: {name}')
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(path, np.uint8), 1),
                           cv2.COLOR_BGR2RGB)
        points = np.asarray(row['landmarks_pixels'], float)
        _, _, _, eye_mid, scale, rot = measure(points)
        mouth = points[[62, 66]].mean(0)
        normalize = lambda xy: (np.asarray(xy, float) - mouth) @ rot.T / scale
        left, right = normalize(label['left_xy']), normalize(label['right_xy'])
        chin, lip = normalize(label['chin_xy']), normalize(label['lower_lip_bottom_xy'])
        for poly in (left, right):
            if np.any(np.diff(poly[:, 1]) <= 0):
                raise ValueError('Trace must descend monotonically after de-roll')
        levels = np.array(fractions) * chin[1]
        lx = np.interp(levels, left[:, 1], left[:, 0])
        rx = np.interp(levels, right[:, 1], right[:, 0])
        lo, hi = fractions.index(.25), fractions.index(.75)
        vleft = np.array([lx[hi] - lx[lo], levels[hi] - levels[lo]])
        vright = np.array([rx[hi] - rx[lo], levels[hi] - levels[lo]])
        angle = float(np.degrees(np.arccos(np.clip(np.dot(vleft, vright) /
                         np.linalg.norm(vleft) / np.linalg.norm(vright), -1, 1))))
        result[name] = {'image_sha256': sha,
                        'widths_per_eye_distance': dict(zip(map(str, fractions), (rx-lx).tolist())),
                        'section_left_x_per_eye_distance': dict(zip(map(str, fractions), lx.tolist())),
                        'section_right_x_per_eye_distance': dict(zip(map(str, fractions), rx.tolist())),
                        'lower_lip_to_chin_per_eye_distance': float(chin[1]-lip[1]),
                        'mouth_to_chin_per_eye_distance': float(chin[1]),
                        'mid_contour_secant_included_angle_degrees': angle,
                        'angle_scope': 'Angle between 25%-75% contour secants; NOT anatomical mandibular angle.'}
        color = '#36bfc8' if i == 0 else '#f16553'
        ax = axes[i]
        ax.imshow(rgb)
        for key in ('left_xy', 'right_xy'):
            p = np.asarray(label[key]); ax.plot(p[:, 0], p[:, 1], '.-', color=color, lw=1.2, ms=3)
        ax.scatter(*label['lower_lip_bottom_xy'], marker='+', color='#ffee66')
        allp = np.vstack([label['left_xy'], label['right_xy']])
        ax.set_xlim(allp[:, 0].min()-20, allp[:, 0].max()+20)
        ax.set_ylim(allp[:, 1].max()+20, allp[:, 1].min()-25)
        ax.set_aspect('equal'); ax.set_title(name+' visible jaw trace'); ax.axis('off')
        for p in (left, right): overlay.plot(p[:, 0], p[:, 1], '.-', color=color, lw=2, ms=3)
        for j, y in enumerate(levels):
            overlay.plot([lx[j], rx[j]], [y, y], color=color, alpha=.55, lw=1)
        overlay.plot([], [], color=color, label=name)
    ref, cand = result['reference'], result['candidate']
    result['relative_difference_percent'] = {
        'width_at_'+f: (cand['widths_per_eye_distance'][f] / w - 1)*100
        for f, w in ref['widths_per_eye_distance'].items()}
    for key in ('lower_lip_to_chin_per_eye_distance', 'mouth_to_chin_per_eye_distance'):
        result['relative_difference_percent'][key] = (cand[key]/ref[key]-1)*100
    overlay.set_aspect('equal'); overlay.invert_yaxis(); overlay.grid(alpha=.2)
    overlay.set_xlabel('horizontal / eye-centroid distance')
    overlay.set_ylabel('below mouth / eye-centroid distance')
    overlay.set_title('Mouth aligned, uniform eye-distance scale'); overlay.legend()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'jaw_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig.savefig(args.out/'jaw_contour_comparison.png', dpi=160)
    print(json.dumps({'differences':result['relative_difference_percent'],
                      'secant_angle':{s:result[s]['mid_contour_secant_included_angle_degrees']
                                      for s in ('reference','candidate')}}, indent=2))


if __name__ == '__main__':
    main()
