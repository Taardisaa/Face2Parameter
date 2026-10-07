"""Check independent raw-image annotations against frozen material hypotheses.

This accepts qualified visual-review evidence, not clinical ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.anatomical_candidates.run import load_scene, require, save
from tools.surface_calibration.core import ContractError, validate_reprojection
from tools.surface_calibration.pixel_certificate import digest, read_json
from tools.surface_consensus.consensus import Policy, triangulate


def review_identity(report, review, report_path):
    require(review.get('schema_version') == 1, 'Unsupported review schema')
    require(review.get('candidate_report_sha256') == digest(report_path), 'Candidate report changed')
    require(Path(review.get('candidate_report_path', '')).resolve() == report_path.resolve(), 'Wrong candidate report')
    reviewer = review.get('reviewer_identity')
    require(isinstance(reviewer, str) and bool(reviewer.strip()), 'Independent reviewer identity required')
    require(review.get('reviewer_is_seed_author') is False, 'Seed author cannot independently validate own hypothesis')
    require(all(reviewer != point['seed_author'] for point in report['points']), 'Reviewer is the seed author')
    require(review.get('review_original_png_before_candidate_overlay') is True, 'Independent original-image-first annotation required')
    rows = review.get('points', [])
    require(len(rows) == len(report['points']) and len({row['id'] for row in rows}) == len(rows), 'Review IDs incomplete/duplicated')
    require({row['id'] for row in rows} == {row['id'] for row in report['points']}, 'Wrong review point IDs')
    return reviewer


def evaluate_point(candidate, annotation, views, cameras, meshes):
    require(annotation.get('definition') == candidate['definition'], 'Semantic definition changed during review')
    decisions = ('pending', 'accepted', 'rejected', 'uncertain')
    require(annotation.get('semantic_decision') in decisions, 'Unknown semantic decision')
    require(annotation.get('skin_vs_texture_anchor') in ('unknown', 'skin', 'texture', 'ambiguous'), 'Unknown anchor class')
    observed, eligible, rejected = [], [], []
    records = annotation.get('views')
    require(isinstance(records, list) and len(records) == len(views), 'One review record per original view required')
    require({row.get('view_index') for row in records} == set(range(len(views))), 'Missing/duplicate review views')
    for row in records:
        index = row['view_index']
        view = views[index]
        require(row.get('raw_png_sha256') == digest(view['path']), 'Review PNG changed')
        require(Path(row.get('raw_png', '')).resolve() == Path(view['path']).resolve(), 'Wrong review raw PNG')
        visibility = row.get('visibility')
        require(visibility in ('unreviewed', 'visible', 'occluded', 'ambiguous'), 'Unknown semantic visibility')
        xy = row.get('independent_xy')
        if xy is None:
            require(visibility != 'visible', 'Visible review needs an independent coordinate')
            continue
        require(visibility == 'visible' and row.get('annotation_method') == 'original_png_before_overlay', 'Coordinates must come from independent visible raw-image annotation')
        value = np.asarray(xy, dtype=float)
        require(value.shape == (2,) and np.isfinite(value).all(), 'Invalid independent coordinate')
        require(0 <= value[0] < cameras[index].width and 0 <= value[1] < cameras[index].height, 'Independent pixel outside original image')
        if 'material' not in candidate:
            rejected.append({'view_index': index, 'reason': 'No geometric material hypothesis'})
            continue
        result = validate_reprojection(candidate['material'], cameras[index], meshes, observed_xy=value, max_error_px=2.)
        observed.append({'view_index': index, 'yaw': view.get('yaw'), 'independent_xy': value.tolist(), 'result': result})
        if result['status'] in ('within_pixel_tolerance', 'outside_pixel_tolerance'):
            eligible.append(index)
        else:
            rejected.append({'view_index': index, 'reason': result['status']})
    semantic = annotation['semantic_decision'] == 'accepted' and annotation['skin_vs_texture_anchor'] == 'skin'
    enough = len(eligible) >= 3 and any(abs(float(views[i].get('yaw', 0))) >= 30 for i in eligible)
    if candidate['hypothesis'] in ('nose_skin_apex', 'midline_chin_skin_eminence'):
        enough = enough and any(abs(float(views[i].get('yaw', 0))) >= 60 for i in eligible)
    geometric = enough and not rejected and all(row['result']['status'] == 'within_pixel_tolerance' for row in observed)
    conditioning = None
    if enough:
        # Numerical observability check only; triangulated world is never substituted.
        rays = [cameras[row['view_index']].ray(row['independent_xy']) for row in observed if row['view_index'] in eligible]
        try:
            conditioning, _, _ = triangulate(rays, Policy())
        except ContractError as exc:
            geometric = False
            rejected.append({'reason': 'Independent-ray observability failed: '+str(exc)})
    accepted = semantic and geometric and candidate['status'] == 'awaiting_independent_semantic_review'
    return {'id': candidate['id'], 'semantic_decision': annotation['semantic_decision'],
            'semantic_visual_review_passed': semantic, 'fixed_material_geometry_test_passed': geometric,
            'accepted_for_named_material_candidate': accepted,
            'status': 'review_supported_material_hypothesis' if accepted else 'unaccepted_material_hypothesis',
            'eligible_view_indices': eligible, 'independent_observations': observed,
            'rejections': rejected, 'ray_observability': conditioning,
            'scope': 'Reviewer declarations and fixed-point multiview agreement; not clinical accuracy or full shader proof'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-report', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.out.exists(), 'New review report path required')
        report, review = read_json(args.candidate_report), read_json(args.review)
        reviewer = review_identity(report, review, args.candidate_report)
        for source in report['sources'].values():
            require(digest(source['path']) == source['sha256'], 'Candidate source file changed')
        source = report['sources']
        views, cameras, meshes, _, _ = load_scene(*(source[name]['path'] for name in ('capture', 'geometry', 'lbs_report', 'pixel_report')))
        annotations = {row['id']: row for row in review['points']}
        results = [evaluate_point(candidate, annotations[candidate['id']], views, cameras, meshes) for candidate in report['points']]
        output = {'schema_version': 1, 'candidate_report_sha256': digest(args.candidate_report),
                  'review_sha256': digest(args.review), 'reviewer_identity': reviewer,
                  'reviewer_independence_evidence': 'Declared identity/original-image-first method; software cannot authenticate human independence',
                  'anatomical_correspondence_validated': False, 'material_visibility_validated': False,
                  'camera_or_scale_fitted': False, 'fixed_candidate_was_refit_to_review': False,
                  'accepted_for_named_material_candidate_ids': [row['id'] for row in results if row['accepted_for_named_material_candidate']],
                  'policy': {'max_error_px': 2., 'min_independent_eligible_views': 3, 'min_oblique_yaw': 30., 'apex_extra_yaw': 60.},
                  'results': results, 'implementation_sha256': digest(__file__)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save(args.out, output)
        print(json.dumps({'accepted_for_named_material_candidate_ids': output['accepted_for_named_material_candidate_ids'], 'out': str(args.out.resolve())}))
    except (ContractError, KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, 'Independent review rejected: '+str(exc)+'\n')


if __name__ == '__main__':
    main()
