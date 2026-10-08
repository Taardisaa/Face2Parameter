"""Final different-photo identity attachment acceptance, not an accuracy score.

Two already-exported second-photo identities; one source composite pose each;
card reload without source/descriptor files, reset/detach and original restoration.
Source logic and new descriptor implementation are completed before this run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifact import ModelArtifact, host_path, sha
from .attachment_acceptance import check, load, require
from .attachment_rebase import original_artifact, rebase
from .game_import import request
from .rig import replay_source_pose
from .surface_acceptance import verify


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def accept(base, path, proposal, native, reference, reference_path, out, thumbnail):
    artifact, model = original_artifact(path)
    require(model.image_index == 1, 'Declared second-photo case required')
    before = request(base, 'GET')
    vertices = np.asarray(artifact['vertices'], np.float32)
    lo, hi = vertices.min(0), vertices.max(0)
    scale = float(before['reference_local_size'][1]/(hi[1]-lo[1]))
    placement = {'scale': scale, 'translation': (np.asarray(before['reference_local_center'])-scale*((lo.astype(float)+hi)/2)).tolist()}
    digest = sha(path)
    state = request(base, 'POST', {'path': str(path.resolve()), 'sha256': digest, **placement})
    dump(out/'actual_source_state.json', state)
    descriptor = rebase(proposal, native, reference, state, reference_path, path)
    descriptor_path = out/'new_identity_attachment.json'
    descriptor['packager_sha256'] = sha(Path(__file__).with_name('attachment_rebase.py'))
    descriptor['actual_source_state_sha256'] = sha(out/'actual_source_state.json')
    dump(descriptor_path, descriptor)
    require(descriptor_path.stat().st_size <= 4*1024*1024, 'Native descriptor size bound exceeded')
    descriptor_digest = sha(descriptor_path)
    report = {'image_index': model.image_index, 'source_sha256': digest,
        'source_photo_sha256': model.image['input_sha256'], 'original_raw_artifact_sha256': model.image['sha256'],
        'descriptor_sha256': descriptor_digest, 'source_accuracy_certified': False,
        'original': verify(state, artifact, out, 'original', original=True)}
    state = request(base, 'POST', {'path': str(descriptor_path.resolve()), 'sha256': descriptor_digest},
        route='/maker/face/model/attachment')
    report['initial_attachment'] = check(state, descriptor, artifact['triangles'], out, 'initial_attachment')
    report['attached_source'] = verify(state, artifact, out, 'attached_source', original=True)
    pose = list(state['source_pose_axis_angle'])
    pose[4] += .035; pose[6] += .07; pose[10] += .05; pose[13] -= .06
    expected, expected_joints = replay_source_pose(model, pose)
    state = request(base, 'POST', {'pose': pose}, route='/maker/face/model/pose')
    # Native pose/TRS are float32. JSON's decimal representation can differ from
    # Python's exact float32-to-double spelling; compare literal native values,
    # with no extra tolerance or rounding and retain the actual saved wire state.
    require(np.array_equal(np.asarray(state['source_pose_axis_angle'], np.float32), np.asarray(pose, np.float32)),
        'Requested native float32 pose changed')
    saved_pose = state['source_pose_axis_angle']
    saved_position = state['local_position']; saved_scale = state['local_scale']
    report['posed_source'] = verify(state, artifact, out, 'posed_source')
    report['posed_attachment'] = check(state, descriptor, artifact['triangles'], out, 'posed_attachment')
    posed = np.asarray(state['canonical_vertices'], np.float32)
    vertex_error = float(np.max(np.abs(posed-expected)))
    joint_error = float(np.max(np.abs(np.asarray(state['source_posed_joints'])-expected_joints)))
    np.savez_compressed(out/'original_decoder_pose.npz', expected=expected, actual=posed,
        expected_joints=expected_joints, pose=np.asarray(pose, np.float32))
    report['original_decoder_pose'] = {'vertex_error': vertex_error, 'joint_error': joint_error,
        'tolerance': 1e-6, 'pose': pose}
    require(np.any(posed != vertices) and vertex_error <= 1e-6 and joint_error <= 1e-6,
        'Selected original decoder pose differs; evidence retained')
    card = out/'new_identity_card.png'
    report['saved'] = request(base, 'POST', {'path': str(card.resolve()), 'thumbnail': str(thumbnail.resolve())},
        route='/maker/face/model/save')
    renamed = []
    try:
        for item in [path, descriptor_path]:
            hidden = item.with_name(item.name+'.temporarily_absent_for_card_reload')
            require(not hidden.exists(), 'Temporary card-reload path already exists')
            item.rename(hidden); renamed.append((item, hidden))
        cleared = request(base, 'DELETE')
        require(all(cleared['attachment_restoration'].values()) and cleared['original_renderer_visibility_restored'],
            'Clear failed to restore native body/visibility')
        state = load(base, card, digest, descriptor_digest)
        report['reloaded_source'] = verify(state, artifact, out, 'reloaded_source')
        report['reloaded_attachment'] = check(state, descriptor, artifact['triangles'], out, 'reloaded_attachment')
        report['embedded_state_checks'] = {
            'canonical_positions_literal': bool(np.array_equal(posed, np.asarray(state['canonical_vertices'], np.float32))),
            'saved_pose_wire_literal': state['source_pose_axis_angle'] == saved_pose,
            'saved_position_wire_literal': state['local_position'] == saved_position,
            'saved_scale_wire_literal': state['local_scale'] == saved_scale,
            'descriptor_position_float32_literal': bool(np.array_equal(np.asarray(state['local_position'], np.float32),
                np.asarray(descriptor['source_placement']['translation'], np.float32))),
            'descriptor_scale_float32_literal': bool(np.array_equal(np.asarray(state['local_scale'], np.float32),
                np.asarray([descriptor['source_placement']['scale']]*3, np.float32))),
        }
        dump(out/'embedded_state_checks.json', report['embedded_state_checks'])
        require(all(report['embedded_state_checks'].values()), 'Embedded source state changed')
        report['external_source_and_descriptor_unneeded'] = all(not item.exists() for item, _ in renamed)
    finally:
        for item, hidden in reversed(renamed):
            hidden.rename(item)
    state = request(base, 'POST', {'reset': True}, route='/maker/face/model/pose')
    report['reset_source'] = verify(state, artifact, out, 'reset_source', original=True)
    report['reset_attachment'] = check(state, descriptor, artifact['triangles'], out, 'reset_attachment')
    state = request(base, 'DELETE', route='/maker/face/model/attachment')
    require(state['attachment']['status'] == 'absent' and all(state['attachment_restoration'].values()), 'Detach failed')
    report['detach_restoration'] = state['attachment_restoration']
    report['passed'] = True
    dump(out/'receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:43127')
    parser.add_argument('--smirk', type=Path, required=True)
    parser.add_argument('--mica', type=Path, required=True)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--native-state', type=Path, required=True)
    parser.add_argument('--reference-state', type=Path, required=True)
    parser.add_argument('--thumbnail', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference_state.read_text(encoding='utf-8'))
    reference_path = host_path(reference['artifact_path'])
    native = json.loads(args.native_state.read_text(encoding='utf-8'))
    proposal = json.loads(args.proposal.read_text(encoding='utf-8'))
    # Verify both original artifacts/definitions before any game mutation.
    for path in [args.smirk, args.mica]:
        artifact, _ = original_artifact(path)
        synthetic = {'active': True, 'artifact_sha256': sha(path),
            'canonical_vertices': artifact['vertices'], 'canonical_triangles': artifact['triangles'],
            'local_scale': reference['local_scale'], 'local_position': reference['local_position'],
            'attachment': {'status': 'absent'}, **{key: True for key in (
                'source_vertices_unchanged', 'source_triangle_indices_unchanged', 'source_world_similarity_frame',
                'source_render_mesh_bound', 'source_render_material_bound', 'source_display_enabled')}}
        rebase(proposal, native, reference, synthetic, reference_path, path)
    backup = request(args.base, 'GET')
    require(backup.get('active') and backup.get('attachment', {}).get('status') == 'absent', 'Original unjoined backup required')
    card = args.out/'before_new_identity.png'
    request(args.base, 'POST', {'path': str(card.resolve()), 'thumbnail': str(args.thumbnail.resolve())}, route='/maker/face/model/save')
    report = {'plugin': request(args.base, 'GET', route='/health'), 'source_accuracy_certified': False,
        'protocol': __doc__, 'preflight': 'Synthetic source state only for static logic; actual imported state required for each final descriptor',
        'inputs': [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
            (args.smirk, args.mica, args.proposal, args.native_state, args.reference_state, reference_path)],
        'implementation': [{'path': str(Path(__file__).with_name(name)), 'sha256': sha(Path(__file__).with_name(name))} for name in
            ('attachment_rebase.py', 'rebase_acceptance.py', 'surface_acceptance.py')]}
    try:
        for name, path in [('smirk', args.smirk), ('mica', args.mica)]:
            out = args.out/name; out.mkdir()
            report[name] = accept(args.base, path, proposal, native, reference, reference_path, out, args.thumbnail)
        report['passed'] = True
    except Exception as exc:
        report.update(passed=False, failure=str(exc)); raise
    finally:
        try:
            state = load(args.base, card, backup['artifact_sha256'])
            require(np.array_equal(np.asarray(state['render_vertices'], np.float32), np.asarray(backup['render_vertices'], np.float32)) and
                state['render_triangles'] == backup['render_triangles'] and state['local_scale'] == backup['local_scale'] and
                state['local_position'] == backup['local_position'] and state['attachment']['status'] == 'absent', 'Original backup changed')
            report['backup_restored'] = True
        except Exception as exc:
            report.update(passed=False, backup_restore_error=str(exc)); raise
        finally:
            dump(args.out/'receipt.json', report)
    print(json.dumps({'receipt': str((args.out/'receipt.json').resolve()), 'passed': report['passed'], 'backup_restored': report['backup_restored']}))


if __name__ == '__main__':
    main()
