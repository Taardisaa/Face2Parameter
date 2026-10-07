"""Project separately named visual hypotheses to fixed triangle/barycentric data.

No detector result is promoted. Geometry agreement is not semantic acceptance.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tools.surface_calibration.core import Camera, ContractError, follow, meshes_from_geometry, observe, validate_reprojection
from tools.surface_calibration.pixel_certificate import certify_pixel_contract, digest, read_json
from tools.unity_parity.geometry import analyze_snapshot


def require(value, message):
    if not value:
        raise ContractError(message)


def load_scene(capture_path, geometry_path, lbs_path, pixel_path):
    raw = read_json(capture_path)
    capture = raw.get('bridge_result', raw)
    geometry, saved = read_json(geometry_path), read_json(lbs_path)
    geometry_sha = digest(geometry_path)
    require(saved.get('snapshot_sha256') == geometry_sha, 'Independent LBS snapshot SHA mismatch')
    require(geometry['frame_count'] == geometry['frame_count_end'], 'Non-atomic snapshot')
    fresh = analyze_snapshot(geometry, normalized_tolerance=1e-5)
    previous = {row['renderer_path']: row for row in saved['meshes']}
    require(len(previous) == len(saved['meshes']), 'Duplicate renderer paths in LBS report')
    computed = {row['renderer_path']: row for row in fresh['meshes']}
    hashes, included, excluded = {}, [], []
    for mesh in geometry['meshes']:
        path = mesh['renderer_path']
        if not (mesh.get('enabled') is True and mesh.get('active_in_hierarchy') is True):
            excluded.append({'renderer_path': path, 'enabled': mesh.get('enabled'), 'active_in_hierarchy': mesh.get('active_in_hierarchy')})
            continue
        row, old = computed[path], previous.get(path, {})
        require(row['certified'] and 'scale_free_trs' in row.get('matching_candidates', []), 'Visible mesh lacks independent scale-free LBS parity')
        require(row['source_geometry_sha256'] == old.get('source_geometry_sha256'), 'LBS source identity mismatch')
        for candidate, errors in row['candidate_errors'].items():
            for key in ('max_normalized', 'rms_normalized', 'max_l2'):
                require(np.isclose(errors[key], old['candidate_errors'][candidate][key], rtol=1e-8, atol=1e-12), 'Independent LBS residual changed')
        hashes[path] = row['source_geometry_sha256']
        included.append(mesh)
    visible_geometry = {**geometry, 'meshes': included}
    meshes = meshes_from_geometry(visible_geometry, candidate='scale_free_trs', certification={
        'world_policy_validated': True, 'world_candidate': 'scale_free_trs', 'mesh_source_hashes': hashes})
    targets = [mesh['renderer_path'] for mesh in included if mesh['mesh_name'] == 'o_head']
    require(len(targets) == 1, 'Exactly one visible source-bound o_head required')
    views, cameras, certs = capture.get('views') or [capture], [], []
    for view in views:
        pair = view.get('paired_geometry', {})
        require(pair.get('sha256') == geometry_sha and pair.get('pose_signature') == geometry['pose_signature'], 'View snapshot SHA/pose mismatch')
        require(pair.get('frame_count') == geometry['frame_count'], 'View snapshot frame mismatch')
        cert = certify_pixel_contract(pixel_path, view)
        camera = Camera.from_capture(view, pixel_certificate=cert)
        with Image.open(view['path']) as image:
            require(image.size == (camera.width, camera.height), 'Actual PNG dimensions mismatch')
        cameras.append(camera)
        certs.append(cert.evidence())
    return views, cameras, meshes, targets, {
        'sources': {name: {'path': str(Path(path).resolve()), 'sha256': digest(path)} for name, path in
                    [('capture', capture_path), ('geometry', geometry_path), ('lbs_report', lbs_path), ('pixel_report', pixel_path)]},
        'character_head_id': geometry['character']['head_id'], 'geometry_pose_signature': geometry['pose_signature'],
        'pixel_certificates': certs, 'excluded_inactive_renderers': excluded,
        'independent_lbs_recomputed': True, 'lbs_normalized_tolerance': 1e-5}


def validate_seed(seed, evidence, views):
    require(seed.get('schema_version') == 1 and seed.get('semantic_validated') is False, 'Unvalidated seed schema required')
    for name in ('capture', 'geometry'):
        require(seed.get(name+'_sha256') == evidence['sources'][name]['sha256'], 'Seed belongs to different '+name)
    require(seed.get('coordinate_convention') == 'top_left_pixel_centers_integer', 'Unknown seed pixels')
    index = seed.get('view_index')
    require(type(index) is int and 0 <= index < len(views), 'Invalid seed view')
    require(digest(views[index]['path']) == seed.get('raw_png_sha256'), 'Seed PNG changed')
    require(isinstance(seed.get('seed_author'), str) and bool(seed['seed_author']), 'Seed provenance missing')
    require(isinstance(seed.get('points'), list) and seed['points'], 'No seed points')
    ids = [point.get('id') for point in seed['points']]
    require(len(set(ids)) == len(ids) and all(isinstance(i, str) and i for i in ids), 'Duplicate/invalid point IDs')
    for point in seed['points']:
        xy = np.asarray(point.get('xy'), dtype=float)
        require(xy.shape == (2,) and np.isfinite(xy).all(), 'Invalid seed xy')
        require(isinstance(point.get('definition'), str) and bool(point['definition']), 'Explicit semantic definition required')
    return index


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def render(views, rows, output):
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 12)
    colors = [(0, 255, 255), (255, 210, 0), (255, 110, 225), (60, 250, 70), (255, 160, 60), (190, 150, 255)]
    overlay_tiles, raw_tiles, records = [], [], []
    for index, view in enumerate(views):
        original = Image.open(view['path']).convert('RGB')
        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        draw.text((7, 7), f'v{index} yaw {view.get("yaw", 0):+.0f} | hypotheses; NO semantic pass', fill='white', font=font, stroke_width=1, stroke_fill='black')
        xy_points = []
        for number, row in enumerate(rows):
            if 'views' not in row:
                continue
            current = row['views'][index]
            xy = current['projection']['xy']
            if xy is None:
                continue
            x, y = xy
            xy_points.append(xy)
            color = colors[number % len(colors)] if current['status'] == 'projected_without_independent_observation' else (255, 65, 65)
            draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=1)
            draw.text((x+5, y-13), row['id'], font=font, fill=color, stroke_width=1, stroke_fill='black')
        path = output/f'view_{index}_overlay.png'
        overlay.save(path)
        # Crops never resample. Original images remain immutable.
        coords = np.asarray(xy_points) if xy_points else np.array([[original.width/2, original.height/2]])
        left = max(0, math.floor(coords[:, 0].min())-38)
        top = max(0, math.floor(coords[:, 1].min())-38)
        right = min(original.width, math.ceil(coords[:, 0].max())+39)
        bottom = min(original.height, math.ceil(coords[:, 1].max())+39)
        crop = output/f'view_{index}_crop.png'
        overlay.crop((left, top, right, bottom)).save(crop)
        raw_crop = output/f'view_{index}_raw_crop.png'
        original.crop((left, top, right, bottom)).save(raw_crop)
        records.append({'view_index': index, 'yaw': view.get('yaw'), 'source_png': view['path'], 'source_png_sha256': digest(view['path']),
                        'overlay': str(path.resolve()), 'overlay_sha256': digest(path), 'crop': str(crop.resolve()),
                        'raw_crop': str(raw_crop.resolve()), 'crop_bounds_ltrb_exclusive': [left, top, right, bottom]})
        overlay_tiles.append(overlay)
        raw_tiles.append(original)
    width, height = max(image.width for image in raw_tiles), max(image.height for image in raw_tiles)
    for kind, images in [('overlay', overlay_tiles), ('original', raw_tiles)]:
        sheet = Image.new('RGB', (width*4, height*math.ceil(len(images)/4)), (32, 32, 32))
        for i, image in enumerate(images):
            sheet.paste(image, ((i % 4)*width, (i//4)*height))
        sheet.save(output/f'{kind}_seven_view_sheet.png')
    return records


def run(args):
    require(not args.out.exists(), 'New output directory required')
    views, cameras, meshes, targets, evidence = load_scene(args.capture, args.geometry, args.lbs_report, args.pixel_report)
    seed = read_json(args.seeds)
    index = validate_seed(seed, evidence, views)
    rows = []
    for point in seed['points']:
        hit = observe(cameras[index], meshes, {'id': point['id'], 'xy': point['xy'], 'kind': 'surface', 'allowed_renderer_paths': targets})
        row = {**point, 'seed_view_index': index, 'seed_author': seed['seed_author'], 'seed_hit': hit,
               'semantic_accepted': False, 'status': 'awaiting_independent_semantic_review',
               'same_seed_image_reprojection_is_independent_validation': False}
        if hit['status'] != 'surface_candidate' or hit.get('intersection_ambiguous'):
            row['status'] = 'seed_geometric_hit_rejected_or_ambiguous'
        if 'surface' in hit:
            material = follow(hit['surface'], meshes)
            row.update(material=material, views=[validate_reprojection(material, camera, meshes) for camera in cameras])
        rows.append(row)
    # FAN failures are immutable comparison evidence, never input to these seeds.
    prior = read_json(args.prior_consensus) if args.prior_consensus else None
    if prior:
        require(all(prior['sources'][name]['sha256'] == evidence['sources'][name]['sha256'] for name in ('capture', 'geometry')), 'Prior consensus has different source')
    report = {'schema_version': 1, 'kind': 'source_bound_visual_anatomical_hypotheses', **evidence,
              'seeds_path': str(args.seeds.resolve()), 'seeds_sha256': digest(args.seeds), 'points': rows,
              'semantic_accepted_ids': [], 'anatomical_correspondence_validated': False,
              'material_visibility_validated': False, 'camera_or_scale_fitted': False,
              'prior_fan_status_unchanged': [{'id': row['id'], 'status': row['status'], 'accepted': row['accepted'],
                                             'max_heldout_error_px': row.get('max_heldout_error_px')} for row in prior['results']] if prior else None,
              'semantic_gate': {'required_independent_reviewers': 1, 'reviewer_must_differ_from_seed_author': True,
                               'required_independent_eligible_view_annotations': 3, 'at_least_one_oblique_view': True,
                               'max_fixed_material_reprojection_error_px': 2., 'review_on_original_png_before_overlay': True,
                               'unknown_visibility_or_texture_anchor': 'unsupported_or_uncertain',
                               'nose_or_chin_apex_extra_requirement': 'independent profile/oblique review distinguishes anterior eminence from lowest silhouette',
                               'current_gate_passed': False},
              'limits': ['A front seed ray is a material-point construction, not proof of semantic or detector accuracy.',
                         'Actual renderer geometry is checked; shader alpha/depth and semantic visibility are still unknown.',
                         'No directional support extreme is relabeled as nose/chin; no failed FAN point is promoted.',
                         'Names remain hypotheses until independent original-image annotation and semantic review pass.',
                         'Coordinates and topology are head/source-specific. Other assets require new annotation or validated transport.'],
              'implementation_hashes': {str(path.resolve()): digest(path) for path in [Path(__file__), ROOT/'tools/surface_calibration/core.py', ROOT/'tools/unity_parity/geometry.py']}}
    args.out.mkdir(parents=True)
    overlays = args.out/'visual_review'
    overlays.mkdir()
    report['visual_review'] = render(views, rows, overlays)
    template = {'schema_version': 1, 'candidate_report_path': str((args.out/'report.json').resolve()),
                'reviewer_identity': None, 'reviewer_is_seed_author': None,
                'review_original_png_before_candidate_overlay': True,
                'instructions': 'Record original-image semantic pixels without seeing projected candidates first. Use null for occluded/ambiguous features; do not copy projections.',
                'points': [{'id': row['id'], 'definition': row['definition'], 'semantic_decision': 'pending',
                            'skin_vs_texture_anchor': 'unknown', 'notes': None,
                            'views': [{'view_index': i, 'raw_png_sha256': digest(view['path']), 'raw_png': view['path'],
                                       'visibility': 'unreviewed', 'independent_xy': None, 'annotation_method': None} for i, view in enumerate(views)]} for row in rows]}
    save(args.out/'report.json', report)
    template['candidate_report_sha256'] = digest(args.out/'report.json')
    save(args.out/'independent_review_template.json', template)
    print(json.dumps({'out': str(args.out.resolve()), 'points': [(row['id'], row['seed_hit']['status'], row.get('material', {}).get('triangle_id')) for row in rows],
                      'semantic_accepted': report['semantic_accepted_ids']}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('capture', 'geometry', 'lbs-report', 'pixel-report', 'seeds', 'out'):
        parser.add_argument('--'+field, type=Path, required=True)
    parser.add_argument('--prior-consensus', type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except (ContractError, KeyError, OSError, ValueError) as exc:
        parser.exit(2, 'Candidate contract rejected: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
