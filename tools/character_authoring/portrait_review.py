"""Review cached FAN68 observations with explicit, hash-bound human chin labels.

Preserves original detector observations. This is image annotation, not a fitted
game response or a replacement anatomical ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from portrait_compare import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measurements', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.measurements.read_text(encoding='utf-8'))
    labels = json.loads(args.annotations.read_text(encoding='utf-8'))
    frames, result = [], {'original_measurements_sha256': hashlib.sha256(
        args.measurements.read_bytes()).hexdigest(), 'annotations': labels}
    for name in ('reference', 'candidate'):
        row, annotation = original[name], labels[name]
        path = Path(row['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row['sha256'] or digest != annotation['image_sha256']:
            raise ValueError(f'Image hash mismatch: {name}')
        rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2RGB)
        points = np.asarray(row['landmarks_pixels'], dtype=float).copy()
        chin = np.asarray(annotation['chin_pixels'], dtype=float)
        if chin.shape != (2,) or not np.isfinite(chin).all() or not (
                0 <= chin[0] < rgb.shape[1] and 0 <= chin[1] < rgb.shape[0]):
            raise ValueError(f'Invalid chin label: {name}')
        old_chin = points[8].copy()
        points[8] = chin
        aligned, levels, metrics, mid, scale, rotation = measure(points)
        matrix = np.empty((2, 3))
        matrix[:, :2] = rotation * (300 / scale)
        matrix[:, 2] = [350, 210] - matrix[:, :2] @ mid
        frame = cv2.warpAffine(rgb, matrix, (700, 900), borderValue=(210, 210, 210))
        old_dot = (old_chin - mid) @ rotation.T / scale * 300 + [350, 210]
        frames.append((name, frame, aligned * 300 + [350, 210], old_dot, levels))
        result[name] = {'path': str(path), 'sha256': digest,
                        'original_detector_chin_pixels': old_chin.tolist(),
                        'reviewed_chin_pixels': chin.tolist(),
                        'metrics_per_eye_centroid_distance': metrics}
    reference = result['reference']['metrics_per_eye_centroid_distance']
    candidate = result['candidate']['metrics_per_eye_centroid_distance']
    result['relative_difference_percent'] = {
        key: (candidate[key] / reference[key] - 1) * 100 for key in reference}
    result['scope'] = ('Only landmark8 is manually labelled on the visible chin contour. '
                       'All eye/nose/mouth points remain original FAN68 estimates. '
                       'No yaw/pitch correction or three-dimensional ground truth. '
                       'Annotation uncertainty is a judgement bound, not detector calibration.')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'reviewed_measurements.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (name, frame, points, old_dot, levels) in zip(axes, frames):
        ax.imshow(frame)
        ax.scatter(points[:, 0], points[:, 1], s=5, color='#27dcdd')
        ax.scatter(*old_dot, s=35, facecolors='none', edgecolors='#ff5555',
                   label='original FAN chin')
        for key in ('eye_line', 'nose_base', 'mouth_line', 'chin'):
            y = 210 + levels[key] * 300
            ax.axhline(y, color='#ffc247', linewidth=.8)
            ax.text(4, y - 3, key, color='#ffc247', fontsize=7)
        ax.set_title(name + ' (reviewed visible chin)')
        ax.set_xlim(0, 700); ax.set_ylim(900, 0); ax.axis('off')
    fig.savefig(args.out / 'reviewed_alignment.png', dpi=160)
    plt.close(fig)
    print(json.dumps(result['relative_difference_percent'], indent=2))


if __name__ == '__main__':
    main()
