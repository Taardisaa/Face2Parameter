"""Measure 2D portrait proportions with one installed FAN68 detector.

These are image-space observations, not inferred HS2 deformation or 3D truth.
Uniform scale/roll alignment uses the two eye-contour centroids, not gaze pupils.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import face_alignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def measure(points):
    left = points[36:42].mean(0)
    right = points[42:48].mean(0)
    midpoint = (left + right) / 2
    vector = right - left
    distance = float(np.linalg.norm(vector))
    angle = np.arctan2(vector[1], vector[0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    aligned = (points - midpoint) @ rotation.T / distance
    levels = {'eye_line': 0.0, 'nose_base': float(aligned[33, 1]),
              'mouth_line': float(aligned[[62, 66], 1].mean()),
              'chin': float(aligned[8, 1]),
              'brow_inner': float(aligned[[21, 22], 1].mean())}
    metrics = {
        'eye_to_chin': levels['chin'],
        'eye_to_nose_base': levels['nose_base'],
        'nose_base_to_mouth': levels['mouth_line'] - levels['nose_base'],
        'mouth_to_chin': levels['chin'] - levels['mouth_line'],
        'inner_brow_to_chin': levels['chin'] - levels['brow_inner'],
        'mouth_width': float(aligned[54, 0] - aligned[48, 0]),
        'nose_width': float(aligned[35, 0] - aligned[31, 0]),
        'contour_width_2_14': float(aligned[14, 0] - aligned[2, 0]),
    }
    return aligned, levels, metrics, midpoint, distance, rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                             flip_input=False, device='cpu', compile=False)
    results, frames = {}, []
    for label, path in [('reference', args.reference), ('candidate', args.candidate)]:
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'Unreadable image: {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detector.get_landmarks_from_image(rgb)
        if detections is None or len(detections) != 1:
            raise ValueError(f'Expected one detected face: {path}')
        points = np.asarray(detections[0], dtype=float)[:, :2]
        aligned, levels, metrics, mid, scale, rotation = measure(points)
        matrix = np.empty((2, 3), float)
        matrix[:, :2] = rotation * (300 / scale)
        matrix[:, 2] = [350, 210] - matrix[:, :2] @ mid
        normalized = cv2.warpAffine(rgb, matrix, (700, 900), borderValue=(210, 210, 210))
        frames.append((label, normalized, aligned * 300 + [350, 210], levels))
        results[label] = {'path': str(path.resolve()),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'landmarks_pixels': points.tolist(),
                          'eye_contour_centroid_distance_pixels': scale,
                          'metrics_per_eye_centroid_distance': metrics}
    ref = results['reference']['metrics_per_eye_centroid_distance']
    candidate = results['candidate']['metrics_per_eye_centroid_distance']
    results['relative_difference_percent'] = {
        key: (candidate[key] / ref[key] - 1) * 100 for key in ref}
    results['scope'] = ('FAN68 2D estimates, uniformly scaled/de-rolled by eye-contour centroids. '
                        'No pitch/yaw correction, 3D-depth inference or HS2 response fitting. '
                        'Review overlay before using detector estimates.')
    weights = Path.home() / '.cache/torch/hub/checkpoints/2DFAN4-11f355bf06.pth.tar'
    if weights.exists():
        results['detector_weights_sha256'] = hashlib.sha256(weights.read_bytes()).hexdigest()
    (args.out / 'measurements.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (label, rgb, points, levels) in zip(axes, frames):
        ax.imshow(rgb)
        ax.scatter(points[:, 0], points[:, 1], s=5, color='#27dcdd')
        for name, y in levels.items():
            if name == 'brow_inner':
                continue
            ax.axhline(210 + y * 300, color='#ffc247', linewidth=.8)
            ax.text(4, 207 + y * 300, name, color='#ffc247', fontsize=7)
        ax.set_title(label + ' (eye-centroid distance = 300 px)')
        ax.set_xlim(0, 700); ax.set_ylim(900, 0); ax.axis('off')
    fig.savefig(args.out / 'aligned_landmarks.png', dpi=160)
    plt.close(fig)
    print(json.dumps(results['relative_difference_percent'], indent=2))


if __name__ == '__main__':
    main()
