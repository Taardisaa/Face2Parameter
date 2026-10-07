"""Small final acceptance of the implemented native neck attachment lifecycle.

One source pose, one native body change, card roundtrip and backup restoration.
Same-response native bake and source transform provide the endpoint oracle; no
screenshots, inference sweeps, fitted coefficients or alignment thresholds.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time
from urllib.error import HTTPError

import numpy as np

from .artifact import sha
from .game_import import request


GEOMETRY_TOLERANCE = 1e-5  # Float32 multiply/add and source-to-body TRS, in game units.


def require(value, message):
    if not value:
        raise RuntimeError(message)


def check(state, descriptor, expected_faces, out, name):
    attachment = state.get('attachment', {})
    require(attachment.get('status') == 'active', 'Attachment is not active: ' + str(attachment.get('error')))
    count = descriptor['native']['vertex_count']
    body = np.asarray(attachment['body_vertices'], dtype=np.float32)
    native = np.asarray(attachment['native_baked_vertices'], dtype=np.float32)
    source = np.asarray(state['render_vertices'], dtype=np.float32)
    collar = np.asarray(attachment['collar_vertices'], dtype=np.float32)
    edges = np.asarray(descriptor['edge_interpolation'])
    t = edges[:, 2:3].astype(np.float32)
    endpoints = native[edges[:, 0].astype(int)] * (1-t) + native[edges[:, 1].astype(int)] * t
    matrix = np.asarray(attachment['source_to_body_matrix']).reshape(4, 4)
    source_ring = source[descriptor['source_ring_indices']]
    world_ring = source_ring @ matrix[:3, :3].T + matrix[:3, 3]
    n = len(source_ring)
    source_error = float(np.max(np.abs(collar[:n] - world_ring)))
    edge_error = float(np.max(np.abs(body[count:] - endpoints)))
    checks = {
        'untouched_native_vertices_literal': bool(np.array_equal(body[:count], native)),
        'body_cut_topology_literal': bool(np.array_equal(attachment['body_triangles'], descriptor['triangles'])),
        'source_topology_literal': bool(np.array_equal(state['render_triangles'], expected_faces)),
        'collar_topology_literal': bool(np.array_equal(attachment['collar_triangles'], descriptor['collar_triangles'])),
        'native_collar_endpoints_literal': bool(np.array_equal(collar[n:], body[descriptor['native_ring_cut_indices']])),
        'native_skin_target_retained': attachment['native_shared_mesh_unchanged'],
        'native_renderer_hidden': not attachment['native_renderer_enabled'],
        'visibility_cache_registered': attachment['visibility_cache_registered'],
        'material_references_preserved': attachment['materials_shared_with_native'],
        **{key: attachment[key] for key in ['original_posed_normals_preserved', 'original_posed_tangents_preserved',
            'original_uv_preserved', 'original_uv2_preserved', 'original_colors_preserved']},
        'replacement_follows_native_active': attachment['replacement_active_in_hierarchy'] == attachment['native_body_active_in_hierarchy'],
        'replacement_matches_baseline_enabled': attachment['replacement_renderer_enabled'] == attachment['native_original_enabled'],
    }
    report = {'checks': checks, 'edge_max_error': edge_error, 'source_endpoint_max_error': source_error,
              'tolerance': GEOMETRY_TOLERANCE, 'frame_count': attachment['frame_count'],
              'selected_body': attachment['selected_body'], 'scope': attachment['scope']}
    np.savez_compressed(out / (name + '.npz'), body=body, native=native, source=source,
                        collar=collar, source_to_body=matrix)
    (out / (name + '.json')).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    require(all(checks.values()) and edge_error <= GEOMETRY_TOLERANCE and source_error <= GEOMETRY_TOLERANCE,
            'Native attachment geometry/lifecycle contract differs; evidence retained')
    return report


def load(base, card, digest, attachment_digest=None):
    request(base, 'POST', {'path': str(card.resolve())}, route='/maker/card/load')
    deadline = time.monotonic() + 30
    while True:
        state = request(base, 'GET')
        if state.get('active') and state.get('artifact_sha256') == digest and (
                attachment_digest is None or state.get('attachment', {}).get('status') == 'active' and
                state['attachment'].get('artifact_sha256') == attachment_digest):
            return state
        require(time.monotonic() < deadline, 'Card source/attachment did not finish restoring: '+str(state.get('attachment')))
        time.sleep(.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:43127')
    parser.add_argument('--attachment', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--backup-card', type=Path, required=True)
    parser.add_argument('--thumbnail', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    descriptor = json.loads(args.attachment.read_text(encoding='utf-8'))
    source = json.loads(args.source.read_text(encoding='utf-8'))
    expected_vertices = np.asarray(source['vertices'], dtype=np.float32)
    expected_faces = np.asarray(source['triangles'], dtype=np.int32)
    backup = request(args.base, 'GET')
    require(backup.get('active'), 'Original source head required to verify backup restoration')
    report = {'plugin': request(args.base, 'GET', route='/health'),
              'attachment': {'path': str(args.attachment.resolve()), 'sha256': sha(args.attachment)},
              'source': {'path': str(args.source.resolve()), 'sha256': sha(args.source)},
              'backup_card_sha256': sha(args.backup_card), 'source_accuracy_certified': False}
    require(descriptor['source_artifact_sha256'] == sha(args.source), 'Source artifact differs from attachment')
    try:
        placement = descriptor['source_placement']
        imported = request(args.base, 'POST', {'path': str(args.source.resolve()), 'sha256': sha(args.source), **placement})
        require(np.array_equal(np.asarray(imported['render_vertices'], dtype=np.float32), expected_vertices), 'Original source vertices changed')
        bad = copy.deepcopy(descriptor); bad['native']['bone_weights_sha256'] = '0' * 64
        bad_path = args.out / 'deliberately_incompatible_skin.json'
        bad_path.write_text(json.dumps(bad)+'\n', encoding='utf-8')
        try:
            request(args.base, 'POST', {'path': str(bad_path.resolve()), 'sha256': sha(bad_path)}, route='/maker/face/model/attachment')
            raise RuntimeError('Incompatible native skin unexpectedly accepted')
        except HTTPError as exc:
            rejection = json.load(exc)
            require(exc.code == 409 and rejection.get('error') == 'body_changed', 'Wrong incompatibility rejection: '+str(rejection))
            report['incompatible_skin_rejected'] = rejection
        state = request(args.base, 'POST', {'path': str(args.attachment.resolve()), 'sha256': sha(args.attachment)},
                        route='/maker/face/model/attachment')
        require(np.array_equal(np.asarray(state['render_vertices'], dtype=np.float32), expected_vertices), 'Attachment changed source face')
        report['initial'] = check(state, descriptor, expected_faces, args.out, 'initial')
        original_pose = list(state['source_pose_axis_angle'])
        pose = list(original_pose); pose[4] += .035; pose[6] += .07
        state = request(args.base, 'POST', {'pose': pose}, route='/maker/face/model/pose')
        report['source_pose'] = check(state, descriptor, expected_faces, args.out, 'source_pose')
        posed = np.asarray(state['render_vertices'], dtype=np.float32)
        require(np.any(posed != expected_vertices), 'Declared source pose did not actually move source geometry')
        report['source_pose_geometry_changed'] = True
        before_native = np.asarray(state['attachment']['native_baked_vertices'], dtype=np.float32)
        body_shapes = request(args.base, 'GET', route='/maker/body/shapes')
        height = next(item['value'] for item in body_shapes['shapes'] if item['index'] == 0)
        target = .55 if height != .55 else .5
        report['native_body_change'] = {'index': 0, 'original': height, 'target': target}
        request(args.base, 'POST', {'index': 0, 'value': target}, route='/maker/body/shapes')
        time.sleep(.3)  # One native update propagation, not an inference sweep.
        state = request(args.base, 'GET')
        report['native_body_pose'] = check(state, descriptor, expected_faces, args.out, 'native_body_pose')
        changed_shapes = request(args.base, 'GET', route='/maker/body/shapes')
        require(np.float32(next(item['value'] for item in changed_shapes['shapes'] if item['index'] == 0)) == np.float32(target),
                'Declared native body parameter did not persist')
        require(np.any(before_native != np.asarray(state['attachment']['native_baked_vertices'], dtype=np.float32)),
                'Declared native body case did not actually change native geometry')
        report['native_body_geometry_changed'] = True
        require(np.array_equal(np.asarray(state['render_vertices'], dtype=np.float32), posed), 'Native body parameter changed source-local face')
        card = args.out / 'source_with_attachment.png'
        report['saved'] = request(args.base, 'POST', {'path': str(card.resolve()), 'thumbnail': str(args.thumbnail.resolve())},
                                 route='/maker/face/model/save')
        cleared = request(args.base, 'DELETE')
        report['cleared'] = cleared
        require(all(cleared['attachment_restoration'].values()) and cleared['original_renderer_visibility_restored'], 'Clear did not restore original native state')
        restored = load(args.base, card, sha(args.source), sha(args.attachment))
        report['reloaded'] = check(restored, descriptor, expected_faces, args.out, 'reloaded')
        require(np.array_equal(np.asarray(restored['render_vertices'], dtype=np.float32), posed) and
                restored['source_pose_axis_angle'] == pose and restored['local_position'] == state['local_position'] and
                restored['local_scale'] == state['local_scale'], 'Posed attachment card did not preserve source geometry/placement')
        request(args.base, 'POST', {'reset': True}, route='/maker/face/model/pose')
        reset = request(args.base, 'GET')
        require(np.array_equal(np.asarray(reset['render_vertices'], dtype=np.float32), expected_vertices), 'Reset changed source original arrays')
        report['reset'] = check(reset, descriptor, expected_faces, args.out, 'reset')
        detached = request(args.base, 'DELETE', route='/maker/face/model/attachment')
        require(detached['attachment']['status'] == 'absent' and all(detached['attachment_restoration'].values()), 'Detach did not restore body')
        report['detach_restoration'] = detached['attachment_restoration']
        report['passed'] = True
    except Exception as exc:
        report.update(passed=False, failure=str(exc))
        raise
    finally:
        try:
            restored_backup = load(args.base, args.backup_card, backup['artifact_sha256'])
            exact = np.array_equal(np.asarray(restored_backup['render_vertices'], dtype=np.float32),
                                   np.asarray(backup['render_vertices'], dtype=np.float32)) and (
                restored_backup['render_triangles'] == backup['render_triangles'] and
                restored_backup['local_position'] == backup['local_position'] and restored_backup['local_scale'] == backup['local_scale'])
            report['backup_restored'] = bool(exact and restored_backup.get('attachment', {}).get('status') == 'absent')
            require(report['backup_restored'], 'Backup character geometry/placement/attachment state changed')
        except Exception as exc:
            report['backup_restore_error'] = str(exc); report['passed'] = False
            raise
        finally:
            (args.out / 'receipt.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'receipt': str((args.out/'receipt.json').resolve()), 'passed': report['passed'],
                      'backup_restored': report['backup_restored']}))


if __name__ == '__main__':
    main()
