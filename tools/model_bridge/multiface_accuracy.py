"""Evaluate fixed neutral predictions against associated Multiface tracked geometry.

Uses existing exact triangle distances and calibrated rays. Tracked reconstruction
error, neutral instruction, detector anchors and single-subject limits remain explicit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import platform

import numpy as np
import scipy
import cv2

from .artifact import ModelArtifact, host_path, sha
from .scan_accuracy import camera_rays, first_ray_hits, similarity, TriangleSurface, stats, review_board


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--geometry-manifest', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads(args.case.read_text()); exported = json.loads(args.geometry_manifest.read_text())
    if case['format'] != 'multiface_associated_neutral_v1' or exported['format'] != 'neutral_pair_original_geometry_v1' or exported['raw_parameters_modified'] is not False:
        raise ValueError('Unsupported associated geometry case')
    protocol = case['protocol']
    for row in [*case['source_files'], *case['images'], case['geometry']]:
        if sha(row['path']) != row['sha256']:
            raise ValueError('Associated source asset changed: ' + row['path'])
    # Verify original raw artifacts/source buffers again, not just a derived manifest.
    for model in ('smirk', 'mica'):
        path = host_path(exported[model+'_manifest'])
        if sha(path) != exported[model+'_manifest_sha256']:
            raise ValueError('Raw model manifest changed')
        a = ModelArtifact(path); a.verify_sources()
    if sha(args.mask) != 'ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492':
        raise ValueError('Trusted model face mask changed')
    with args.mask.open('rb') as stream:
        masks = pickle.load(stream, encoding='latin1')
    region = np.asarray(masks[protocol['source_region']], np.int64)
    if args.out.exists():
        raise FileExistsError('Preserve previous diagnostic; use a new output directory')
    data = {}
    for row in exported['images']:
        path = args.geometry_manifest.parent/row['file']
        if path.parent.resolve() != args.geometry_manifest.parent.resolve() or sha(path) != row['sha256']:
            raise ValueError('Derived geometry path/hash differs')
        if row['input_sha256'] in data:
            raise ValueError('Duplicate geometry association')
        with np.load(path,allow_pickle=False) as values:
            data[row['input_sha256']] = {key:values[key].copy() for key in values.files}
    if set(data) != {r['sha256'] for r in case['images']}:
        raise ValueError('Model predictions are not exactly the declared photos')
    with np.load(case['geometry']['path'],allow_pickle=False) as values:
        target = values['vertices']; target_faces = values['faces']
    triangles = target[target_faces]
    photos = {r['camera']:r for r in case['images']}
    anchor_photo = photos[protocol['anchor_camera']]
    anchor_data = data[anchor_photo['sha256']]
    def camera(name):
        row = case['cameras'][name]
        return tuple(np.asarray(row[key],float) for key in ('K','Rt','distortion'))
    ids = protocol['anchor_mediapipe_ids']
    origin, rays = camera_rays(anchor_data['detected_landmarks'][ids,:2], *camera(protocol['anchor_camera']))
    anchors, anchor_faces = first_ray_hits(origin,rays,triangles)
    span = float(np.linalg.norm(anchors[ids.index(protocol['normalization_span_ids'][0])] - anchors[ids.index(protocol['normalization_span_ids'][1])]))
    if not np.isfinite(span) or span <= 0:
        raise ValueError('Invalid outer-eye normalization span')
    embedded_ids = exported['landmark_ids']; selected = [embedded_ids.index(i) for i in ids]
    surface = TriangleSurface(triangles)
    args.out.mkdir(parents=True)
    report = {'format':'multiface_neutral_tracked_accuracy_v1','protocol':protocol,'case_sha256':sha(args.case),
              'geometry_manifest_sha256':sha(args.geometry_manifest),'mask_sha256':sha(args.mask),
              'evaluator_sha256':sha(Path(__file__)),'distance_evaluator_sha256':sha(Path(__file__).with_name('scan_accuracy.py')),
              'raw_parameters_modified':False,'target_anchors':anchors.tolist(),'target_anchor_triangle_ids':anchor_faces.tolist(),
              'normalization_span_tracked_units':span,'accuracy_status':'single_neutral_tracked_case_not_general_certification',
              'geometry_kind':case['geometry_kind'],'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                                                            'scipy':scipy.__version__,'opencv':cv2.__version__},'predictions':[]}
    modes = [('smirk_raw_posed','smirk_posed_landmarks'),('smirk_identity_diagnostic','smirk_identity_landmarks'),('mica_original','mica_landmarks')]
    for camera_id, photo in photos.items():
        values = data[photo['sha256']]; faces = values['faces']
        for mode, landmark_key in modes:
            vertices = values[mode]; source_anchors = values[landmark_key][selected]
            scale, rotation, translation = similarity(source_anchors,anchors)
            placed = scale*vertices@rotation.T+translation
            closest, triangle_ids, counts = surface.closest(placed[region])
            distances = np.linalg.norm(closest-placed[region],axis=1)
            errors = np.full(len(vertices),np.nan); errors[region] = distances/span
            name = camera_id+'_'+mode
            output = args.out/(name+'.npz')
            np.savez_compressed(output,placed_vertices=placed,faces=faces,region_vertex_ids=region,
                                nearest_tracked_surface=closest,nearest_tracked_triangle_ids=triangle_ids,
                                normalized_distances=distances/span,distances_tracked_units=distances,candidate_counts=counts)
            boards = []
            for view, other in photos.items():
                board = args.out/(name+'_camera_'+view+'.png')
                review_board(other['path'],*camera(view),placed,faces,errors,anchors,board,protocol['display_saturation_normalized'])
                boards.append({'camera':view,'path':board.name,'sha256':sha(board),'alignment_refit':False})
            regions = {}
            for part in ('nose','lips','forehead','eye_region'):
                selected_part = np.intersect1d(region,masks[part])
                if len(selected_part):
                    regions[part] = {'vertices':len(selected_part),'normalized_error':stats(errors[selected_part])}
            aligned_anchors = scale*source_anchors@rotation.T+translation
            report['predictions'].append({'input_camera':camera_id,'input_sha256':photo['sha256'],'model_geometry':mode,
                'surface_error_tracked_units':stats(distances),'surface_error_normalized':stats(distances/span),
                'anchor_residual_normalized':stats(np.linalg.norm(aligned_anchors-anchors,axis=1)/span),
                'similarity':{'scale':scale,'rotation':rotation.tolist(),'translation':translation.tolist()},
                'source_regions':regions,'geometry_npz':output.name,'geometry_npz_sha256':sha(output),'boards':boards})
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(args.out/'report.json'),'predictions':len(report['predictions']),'accuracy_status':report['accuracy_status']}))


if __name__ == '__main__':
    main()
