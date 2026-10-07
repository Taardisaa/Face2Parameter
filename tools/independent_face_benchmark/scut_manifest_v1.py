"""Freeze the predeclared SCUT AF1 2D annotation target, without model/game code."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from PIL import __version__ as pillow_version

SAMPLE_ID = 'AF1'
SCHEMA = 'independent_person_2d_scut86_v1'
F2 = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = F2 / 'face2beauty/SCUT-FBP5500-Database-Release'
CANONICAL = {'sort_keys': True, 'ensure_ascii': False, 'separators': (',', ':'), 'allow_nan': False}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def receipt(path: Path) -> dict:
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw), 'sha256': sha256(raw)}


def positive_dimension(value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError('Image dimension must be a positive integer')
    return value


def parse_pts(raw: bytes, *, width: int, height: int) -> np.ndarray:
    """Exact installed upstream i172f layout made explicit little-endian (692 bytes)."""
    width, height = positive_dimension(width), positive_dimension(height)
    if len(raw) != struct.calcsize('<i172f'):
        raise ValueError('Expected exactly 692 bytes; no truncation or trailing data accepted')
    count, *values = struct.unpack('<i172f', raw)
    if count != 86:
        raise ValueError('SCUT declared count must be 86')
    xy = np.asarray(values, dtype=np.float64).reshape(86, 2)
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite annotation')
    if ((xy < 0).any() or (xy[:, 0] >= width).any() or (xy[:, 1] >= height).any()):
        raise ValueError('Annotation outside raw image support [0,width) x [0,height)')
    return xy


def decode_photo(raw: bytes) -> tuple[Image.Image, dict]:
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        width, height = image.size
        positive_dimension(width)
        positive_dimension(height)
        if image.format != 'JPEG' or image.mode != 'RGB':
            raise ValueError('AF1 contract requires its actual RGB JPEG source')
        if image.getexif():
            raise ValueError('Unexpected EXIF: declare orientation/color handling in a new revision')
        return image.copy(), {'width': width, 'height': height, 'source_format': image.format,
                              'source_mode': image.mode, 'decode_mode': 'RGB',
                              'icc_profile_present': bool(image.info.get('icc_profile')),
                              'exif_present': False, 'exif_transpose_applied': False,
                              'color_space': 'JPEG decoded RGB; physical color calibration unknown'}


def make_manifest(source_root: Path = DEFAULT_SOURCE) -> tuple[dict, Image.Image, bytes, bytes]:
    """Only predeclared AF1; no sample selection, beauty labels, candidate, or scoring input."""
    source_root = source_root.resolve()
    data = source_root / 'data/SCUT-FBP5500_v2'
    photo, pts = data / 'Images/AF1.jpg', data / 'facial landmark/AF1.pts'
    readme, parser = source_root / 'README.md', source_root / 'pts2txt.py'
    diagram = source_root / 'SCUT FPB 5500-Facial Landmarks 86.png'
    photo_raw, pts_raw = photo.read_bytes(), pts.read_bytes()
    image, properties = decode_photo(photo_raw)
    xy = parse_pts(pts_raw, width=image.width, height=image.height)
    sources = {name: receipt(path) for name, path in (
        ('photo', photo), ('pts', pts), ('upstream_readme', readme),
        ('upstream_pts_parser', parser), ('upstream_landmark_diagram', diagram))}
    # The target payload cannot include HS2 or beauty state: there is no such argument.
    target = {'sample_id': SAMPLE_ID, 'photo_sha256': sha256(photo_raw), 'pts_sha256': sha256(pts_raw),
              'width': image.width, 'height': image.height, 'points_xy': xy.tolist(),
              'point_ids': [f'scut86_raw_{i:02d}' for i in range(86)],
              'units': 'original_photo_pixels', 'roi': 'complete_original_image_no_crop',
              'coordinate_transform': 'identity; no y flip, resize, roll removal, or .5 offset'}
    manifest = {'schema_version': 1, 'contract': SCHEMA, 'target': target,
                'target_sha256': sha256(json.dumps(target, **CANONICAL).encode('utf-8')),
                'sources': sources, 'image': properties,
                'parser': {'layout': '<i172f', 'endian': 'explicit_little_endian', 'bytes': 692,
                           'source_declared_point_count': 86, 'float_storage': 'IEEE754_float32',
                           'manifest_storage': 'exact_float32_value_promoted_to_float64_JSON',
                           'no_pickle': True, 'source_order_preserved': True},
                'point_definition': {
                    'ids': 'local zero-based serialized order IDs; not source anatomical names',
                    'evidence': 'upstream README describes GUI-located 86 significant facial points; diagram bound separately',
                    'anatomical_names_certified': False, 'fan68_mapping': None, 'bone_mapping': None,
                    'index_origin_in_image': 'not specified by source; values retained without correction',
                    'display_convention': 'upstream pts2txt display uses cv2.circle at int(x),int(y) on original image',
                    'coordinate_axes': 'stored x/y, displayed x right/y down by upstream parser; no physical frame',
                    'half_pixel_convention_certified': False,
                    'all_points_material_surface_points': False,
                    'silhouette_and_texture_semantics': 'not individually certified; no 3D correspondence inferred'},
                'validation': {'finite': True, 'count': 86, 'in_raw_image_bounds': True,
                               'bounds_definition': '[0,width) x [0,height)',
                               'xy_min': xy.min(0).tolist(), 'xy_max': xy.max(0).tolist()},
                'uncertainty': {'camera_intrinsics': None, 'camera_extrinsics': None, 'depth': None,
                                'metric_scale': None, 'lens_distortion': None, 'capture_lighting': None,
                                'expression': 'not measured or neutralized', 'yaw_pitch_roll': 'not measured',
                                'annotation_accuracy_pixels': None, 'color_calibration': None,
                                'roi': 'no calibrated face segmentation; point bbox is diagnostic only',
                                'identity': 'dataset sample AF1; real person identity not established'},
                'independence': {'sample_predeclared': 'AF1 selected before HS2 search by task declaration',
                                 'candidate_conditioned_target': False, 'hs2_candidate_input': False,
                                 'beauty_labels_read': False, 'beauty_labels_used_for_target': False,
                                 'beauty_labels_used_for_score': False, 'score_implemented': False,
                                 'requires_frozen_artifact_before_search': True},
                'certification': {'scope': 'source-bound real-person photograph and original 2D annotations',
                                  'local_upstream_source_metadata_present': True,
                                  'independent_3d_scan': False, 'metric_anatomy': False,
                                  'hs2_surface_correspondence': False, 'person_likeness_pass': False},
                'license_metadata': {'source': 'bound local upstream README',
                                     'declared_use': 'non-commercial research; academic research',
                                     'online_terms_verified': False, 'authorization_certified': False},
                'generator': receipt(Path(__file__)),
                'libraries': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': pillow_version},
                'artifacts': {'photo': 'AF1.jpg', 'pts': 'AF1.pts', 'overlay': 'AF1_overlay.png',
                              'overlay_scope': 'derived diagnostic; original photo/annotation bytes unmodified'}}
    return manifest, image, photo_raw, pts_raw


def overlay(image: Image.Image, xy: np.ndarray) -> Image.Image:
    """Scientific annotation display, no inferred lines/anatomy or photo replacement."""
    canvas = Image.new('RGB', (image.width + 240, image.height), '#ffffff')
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (x, y) in enumerate(xy):
        # Marker location is exact serialized x/y. Label displacement is cosmetic only.
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill='#00ffff', outline='#003344')
        draw.text((x + 3, y - 6), str(index), fill='#ffff00', stroke_width=1, stroke_fill='#000000')
    draw.multiline_text((image.width + 12, 14),
                        'SCUT AF1 | raw86\n\nSource 350 x 350\nCyan: original x/y\nIDs: serialized order 0..85\n\nNo crop / resize / y flip\nNo FAN68 / bone mapping\nNo beauty labels used\n\nDiagnostic overlay\nNot 3D / metric anatomy\nPixel origin & .5 unknown',
                        fill='#202020', spacing=6)
    return canvas


def freeze(out: Path, source_root: Path = DEFAULT_SOURCE) -> dict:
    if out.exists():
        raise ValueError('Refuse overwriting frozen output: use a new directory')
    manifest, image, photo_raw, pts_raw = make_manifest(source_root)
    # All parsing/source preflight is completed before any output mutation.
    out.mkdir(parents=True)
    (out / 'AF1.jpg').write_bytes(photo_raw)
    (out / 'AF1.pts').write_bytes(pts_raw)
    overlay(image, np.asarray(manifest['target']['points_xy'])).save(out / 'AF1_overlay.png')
    manifest['artifacts']['overlay_sha256'] = sha256((out / 'AF1_overlay.png').read_bytes())
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    descriptor = {'schema_version': 1, 'contract': SCHEMA, 'target_sha256': manifest['target_sha256'],
                  'outputs': {name: receipt(out / name) for name in ('manifest.json', 'AF1.jpg', 'AF1.pts', 'AF1_overlay.png')},
                  'frozen_before_hs2_search': True, 'no_candidate_accepted_or_scored': True}
    (out / 'freeze.json').write_text(json.dumps(descriptor, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    verify_artifact(out, sha256((out / 'freeze.json').read_bytes()))
    return descriptor


def verify_artifact(out: Path, expected_freeze_sha256: str) -> dict:
    """Consumer pins a trusted freeze hash, verifies bytes, and reparses original target."""
    raw = (out / 'freeze.json').read_bytes()
    if sha256(raw) != expected_freeze_sha256:
        raise ValueError('Freeze SHA mismatch')
    descriptor = json.loads(raw)
    if descriptor.get('contract') != SCHEMA or set(descriptor.get('outputs', {})) != {'manifest.json', 'AF1.jpg', 'AF1.pts', 'AF1_overlay.png'}:
        raise ValueError('Unknown freeze contract or artifact set')
    for name, entry in descriptor['outputs'].items():
        actual = (out / name).read_bytes()
        if len(actual) != entry['bytes'] or sha256(actual) != entry['sha256']:
            raise ValueError(f'Artifact bytes changed: {name}')
    manifest = json.loads((out / 'manifest.json').read_bytes())
    target = manifest['target']
    if manifest['contract'] != SCHEMA or target['sample_id'] != SAMPLE_ID:
        raise ValueError('Unexpected sample or manifest contract')
    digest = sha256(json.dumps(target, **CANONICAL).encode('utf-8'))
    if digest != manifest['target_sha256'] or digest != descriptor['target_sha256']:
        raise ValueError('Target binding changed')
    image, properties = decode_photo((out / 'AF1.jpg').read_bytes())
    xy = parse_pts((out / 'AF1.pts').read_bytes(), width=image.width, height=image.height)
    if (target['width'], target['height']) != image.size or manifest['image'] != properties:
        raise ValueError('Image properties changed')
    if target['points_xy'] != xy.tolist() or target['point_ids'] != [f'scut86_raw_{i:02d}' for i in range(86)]:
        raise ValueError('Target differs from raw source annotation/order')
    for name in ('photo', 'pts'):
        entry = descriptor['outputs'][manifest['artifacts'][name]]
        if entry['sha256'] != target[f'{name}_sha256'] or entry['sha256'] != manifest['sources'][name]['sha256']:
            raise ValueError('Raw source binding changed')
    if manifest['independence'] != {
        'sample_predeclared': 'AF1 selected before HS2 search by task declaration',
        'candidate_conditioned_target': False, 'hs2_candidate_input': False,
        'beauty_labels_read': False, 'beauty_labels_used_for_target': False,
        'beauty_labels_used_for_score': False, 'score_implemented': False,
        'requires_frozen_artifact_before_search': True}:
        raise ValueError('Independence contract changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--verify-freeze-sha256')
    args = parser.parse_args()
    if args.verify_freeze_sha256:
        manifest = verify_artifact(args.out, args.verify_freeze_sha256)
        print(json.dumps({'verified': True, 'target_sha256': manifest['target_sha256']}))
    else:
        result = freeze(args.out, args.source_root)
        print(json.dumps({'out': str(args.out.resolve()), 'target_sha256': result['target_sha256'],
                          'freeze_sha256': sha256((args.out / 'freeze.json').read_bytes())}))


if __name__ == '__main__':
    main()
