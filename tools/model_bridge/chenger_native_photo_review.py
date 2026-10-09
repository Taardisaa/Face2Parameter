"""Two-dimensional photo/native screenshot board; no head or game changes.

Eye anchors set display scale, roll and translation only. Screenshot anchors
are explicitly supplied visual estimates. This is not camera calibration,
landmark error measurement, geometry fitting, or proof of three-dimensional
identity. Photographic pose, expression, perspective and makeup remain.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from tools.native_head.mother_template_inputs import save_json, source_file


def display_image(path, eyes):
    eyes = np.asarray(eyes, dtype=float).reshape(2, 2)
    vector = eyes[1]-eyes[0]
    length = np.linalg.norm(vector)
    if not np.isfinite(eyes).all() or length <= 0:
        raise ValueError('Invalid visual eye anchors')
    # Image coordinates have Y down. Similarity only: no independent X/Y scale.
    c, s = vector/length
    matrix = np.array([[c, s], [-s, c]])*(130/length)
    offset = np.array([220., 180.])-matrix@eyes.mean(0)
    inverse = np.linalg.inv(matrix)
    affine = np.column_stack((inverse, -inverse@offset))
    image = Image.open(path).convert('RGB')
    view = image.transform((440, 520), Image.Transform.AFFINE,
        affine.ravel(), resample=Image.Resampling.BICUBIC, fillcolor=(135, 135, 140))
    return view, dict(eye_anchors_pixels=eyes.tolist(), linear=matrix.tolist(),
                     translation=offset.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('photo', 'mica-output', 'game-image', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--game-eye-centers', nargs=4, type=float, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    raw = np.load(a.mica_output, allow_pickle=False)
    photo, photo_map = display_image(a.photo, raw['detected_landmarks_5'][:2])
    game, game_map = display_image(a.game_image, a.game_eye_centers)
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), facecolor='#f0f0eb')
    for ax, view, title in zip(axes, [photo, game], ['程儿原照片 img-002', '当前游戏底模 v6']):
        ax.imshow(view); ax.axhline(180, color='#dddddd', lw=.6, alpha=.6)
        ax.set_title(title, fontproperties=font); ax.axis('off')
    fig.suptitle('仅按眼部参考点统一画面尺度、倾斜与位置', fontproperties=font)
    fig.text(.5, .025, '未拉伸脸部；照片与游戏的朝向、表情、镜头和妆容仍不同；不是三维误差测量',
             ha='center', fontproperties=font, fontsize=9)
    fig.subplots_adjust(left=.02, right=.98, bottom=.08, top=.89, wspace=.03)
    fig.savefig(a.out/'photo_native_board.png', dpi=150); plt.close(fig)
    save_json(a.out/'receipt.json', dict(photo=source_file(a.photo),
        mica_output=source_file(a.mica_output), game_image=source_file(a.game_image),
        code=source_file(Path(__file__)), photo_display_transform=photo_map,
        native_display_transform=game_map, screenshot_anchor_source='Manual visual pixel estimates',
        photo_anchor_source='Saved original MICA detector five points',
        camera_calibrated=False, geometry_changed=False, game_mutated=False,
        identity_accuracy_certified=False, local_image_warp=False))
    print(a.out/'photo_native_board.png')


if __name__ == '__main__':
    main()
