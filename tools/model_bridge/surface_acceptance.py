"""Final source corner-UV, explicit texture and similarity-guard acceptance.

Two declared source artifacts, one pose per artifact, one ABMX nonuniform guard
case, embedded-card reload without external artifact/texture and backup restore.
No screenshot sampling, appearance scores or model-accuracy certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError

import numpy as np

from .artifact import sha
from .attachment_acceptance import check as check_attachment, require
from .game_import import request


def verify(state, artifact, out, name, original=False):
    surface = state['surface']; layout = artifact['surface']
    canonical = np.asarray(state['canonical_vertices'], dtype=np.float32)
    canonical_faces = np.asarray(state['canonical_triangles'], dtype=np.int32)
    render = np.asarray(state['render_vertices'], dtype=np.float32)
    mapping = np.asarray(layout['render_to_canonical'], dtype=np.int32)
    render_faces = np.asarray(state['render_triangles'], dtype=np.int32)
    checks = {
        'canonical_topology_literal': bool(np.array_equal(canonical_faces, artifact['triangles'])),
        'render_mapping_literal': bool(np.array_equal(surface['render_to_canonical'], mapping)),
        'render_positions_exact_gather': bool(np.array_equal(render, canonical[mapping])),
        'render_face_order_literal': bool(np.array_equal(render_faces, layout['render_triangles'])),
        'render_face_corners_preserved': bool(np.array_equal(mapping[render_faces], canonical_faces)),
        'uv_literal': bool(np.array_equal(np.asarray(surface['uv'], dtype=np.float32), np.asarray(layout['render_uv'], dtype=np.float32))),
        'canonical_normals_gathered': surface['normal_gather_exact'],
        'render_mesh_bound': state['source_render_mesh_bound'],
        'render_material_bound': state['source_render_material_bound'],
        'world_similarity_frame': state['source_world_similarity_frame'],
        'source_display_enabled': state['source_display_enabled'],
        'no_inferred_albedo_claim': not surface['model_inferred_albedo'],
    }
    if original:
        checks['original_canonical_positions_literal'] = bool(np.array_equal(canonical, np.asarray(artifact['vertices'], dtype=np.float32)))
    if 'texture' in layout:
        checks['embedded_png_hash_equal'] = surface['texture_png_sha256'] == layout['texture']['sha256']
        checks['decoded_pixel_hash_equal'] = surface['decoded_rgba8_top_down_sha256'] == layout['texture']['rgba8_top_down_sha256']
        checks['material_mode'] = surface['material_mode'] == 'unlit_srgb_opaque' and state['shader'] == 'Unlit/Texture'
    else:
        checks['no_texture_invented'] = not surface['explicit_texture']
    np.savez_compressed(out/(name+'.npz'), canonical=canonical, render=render, mapping=mapping,
                        canonical_faces=canonical_faces, render_faces=render_faces,
                        uv=np.asarray(surface['uv'], dtype=np.float32))
    report = {'checks': checks, 'shader': state['shader'], 'canonical_vertices': len(canonical),
              'render_vertices': len(render), 'source_obj': surface['source_obj'],
              'texture_png_sha256': surface['texture_png_sha256'], 'source_accuracy_certified': False}
    (out/(name+'.json')).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    require(all(checks.values()), 'Source surface contract differs; geometry evidence retained')
    return report


def load(base, card, digest, attachment=None):
    request(base, 'POST', {'path': str(card.resolve())}, route='/maker/card/load')
    deadline = time.monotonic()+30
    while True:
        state = request(base, 'GET')
        if state.get('active') and state.get('artifact_sha256') == digest and (
                attachment is None or state.get('attachment', {}).get('status') == 'active'):
            return state
        require(time.monotonic() < deadline, 'Embedded source surface/attachment did not finish restoring')
        time.sleep(.25)


def guard(base, state, descriptor, artifact, out, thumbnail):
    bone = request(base, 'GET', route='/maker/abmx?names=cf_J_Head_s')['bones'][0]
    scale = list(bone['scale']); scale[0] *= 1.1
    report = {'original_modifier': bone, 'declared_nonuniform_scale': scale}
    before = np.asarray(state['canonical_vertices'], dtype=np.float32)
    try:
        request(base, 'POST', {'bones': [{'name': 'cf_J_Head_s', 'scale': scale}]}, route='/maker/abmx/batch')
        time.sleep(.3)
        blocked = request(base, 'GET')
        report['guard_checks'] = {
            'distorting_frame_detected': not blocked['source_world_similarity_frame'],
            'source_hidden': not blocked['source_display_enabled'],
            'native_face_fallback_restored': blocked['native_fallback_renderers_restored'],
            'attachment_released': blocked['attachment']['status'] == 'incompatible',
            'canonical_vertices_not_compensated': bool(np.array_equal(before, np.asarray(blocked['canonical_vertices'], dtype=np.float32))),
        }
        require(all(report['guard_checks'].values()), 'Nonuniform source-head guard failed')
        refused = out/'must_not_save_in_distorting_frame.png'
        try:
            request(base, 'POST', {'path': str(refused.resolve()), 'thumbnail': str(thumbnail.resolve())}, route='/maker/face/model/save')
            raise RuntimeError('Distorting frame unexpectedly saved by native HTTP writer')
        except HTTPError as exc:
            rejection = json.load(exc)
            require(exc.code == 409 and rejection.get('error') == 'non_similarity_parent' and not refused.exists(),
                    'Invalid-frame save rejection differs: '+str(rejection))
            report['save_rejection'] = rejection
    finally:
        patch = {'name': 'cf_J_Head_s', 'scale': bone['scale']} if bone['exists'] else {'name': 'cf_J_Head_s', 'remove': True}
        request(base, 'POST', {'bones': [patch]}, route='/maker/abmx/batch')
        time.sleep(.3)
        restored = request(base, 'GET')
        require(restored['source_world_similarity_frame'] and restored['source_display_enabled'] and
                restored['attachment']['status'] == 'active', 'Original head frame/attachment did not recover')
        report['recovered'] = verify(restored, artifact, out, 'guard_recovered')
        report['recovered_attachment'] = check_attachment(restored, descriptor, artifact['triangles'], out, 'guard_attachment')
        (out/'guard.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def accept(base, mesh_path, descriptor_path, out, thumbnail):
    artifact = json.loads(mesh_path.read_text(encoding='utf-8'))
    descriptor = None if descriptor_path is None else json.loads(descriptor_path.read_text(encoding='utf-8'))
    reference = request(base, 'GET')
    if descriptor is not None:
        placement = descriptor['source_placement']
    else:
        vertices = np.asarray(artifact['vertices'], dtype=np.float32)
        lo, hi = vertices.min(0), vertices.max(0)
        scale = float(reference['reference_local_size'][1]/(hi[1]-lo[1]))
        placement = {'scale': scale, 'translation': (np.asarray(reference['reference_local_center'])-scale*((lo.astype(float)+hi)/2)).tolist()}
    digest = sha(mesh_path)
    report = {'source_sha256': digest, 'source_path': str(mesh_path.resolve())}
    state = request(base, 'POST', {'path': str(mesh_path.resolve()), 'sha256': digest, **placement})
    report['original'] = verify(state, artifact, out, 'original', original=True)
    if descriptor is not None:
        state = request(base, 'POST', {'path': str(descriptor_path.resolve()), 'sha256': sha(descriptor_path)}, route='/maker/face/model/attachment')
        report['original_attachment'] = check_attachment(state, descriptor, artifact['triangles'], out, 'original_attachment')
    pose = list(state['source_pose_axis_angle']); pose[4] += .035; pose[6] += .07
    state = request(base, 'POST', {'pose': pose}, route='/maker/face/model/pose')
    report['posed'] = verify(state, artifact, out, 'posed')
    posed = np.asarray(state['canonical_vertices'], dtype=np.float32)
    require(np.any(posed != np.asarray(artifact['vertices'], dtype=np.float32)), 'Declared source pose did not move geometry')
    if descriptor is not None:
        report['guard'] = guard(base, state, descriptor, artifact, out, thumbnail)
    card = out/'source_surface_card.png'
    report['saved'] = request(base, 'POST', {'path': str(card.resolve()), 'thumbnail': str(thumbnail.resolve())}, route='/maker/face/model/save')
    # Both are task-generated artifacts. Rename in their own directory, then restore
    # in finally. Card restoration uses embedded JSON/PNG, never these external paths.
    paths = [mesh_path]
    if artifact['surface'].get('texture'):
        paths.append(Path(artifact['surface']['texture']['source_path']))
    renamed = []
    try:
        for path in paths:
            hidden = path.with_name(path.name+'.temporarily_absent_for_card_reload')
            require(not hidden.exists(), 'Temporary reload path already exists')
            path.rename(hidden); renamed.append((path, hidden))
        report['cleared'] = request(base, 'DELETE')
        restored = load(base, card, digest, descriptor)
        report['reloaded'] = verify(restored, artifact, out, 'reloaded')
        require(np.array_equal(posed, np.asarray(restored['canonical_vertices'], dtype=np.float32)) and
                restored['source_pose_axis_angle'] == pose, 'Embedded posed surface geometry changed')
        if descriptor is not None:
            report['reloaded_attachment'] = check_attachment(restored, descriptor, artifact['triangles'], out, 'reloaded_attachment')
        report['external_files_unneeded'] = all(not path.exists() for path, _ in renamed)
    finally:
        for path, hidden in reversed(renamed):
            hidden.rename(path)
    state = request(base, 'POST', {'reset': True}, route='/maker/face/model/pose')
    report['reset'] = verify(state, artifact, out, 'reset', original=True)
    report['passed'] = True
    (out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='http://127.0.0.1:43127')
    parser.add_argument('--smirk', type=Path, required=True)
    parser.add_argument('--mica', type=Path, required=True)
    parser.add_argument('--attachment', type=Path, required=True)
    parser.add_argument('--backup-card', type=Path, required=True)
    parser.add_argument('--thumbnail', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    backup = request(args.base, 'GET')
    report = {'plugin': request(args.base, 'GET', route='/health'), 'source_accuracy_certified': False}
    try:
        for name, mesh, attachment in [('smirk', args.smirk, args.attachment), ('mica', args.mica, None)]:
            out = args.out/name; out.mkdir()
            report[name] = accept(args.base, mesh, attachment, out, args.thumbnail)
        report['passed'] = True
    except Exception as exc:
        report.update(passed=False, failure=str(exc))
        raise
    finally:
        try:
            restored = load(args.base, args.backup_card, backup['artifact_sha256'])
            report['backup_restored'] = (np.array_equal(np.asarray(restored['render_vertices'], dtype=np.float32),
                np.asarray(backup['render_vertices'], dtype=np.float32)) and restored['render_triangles'] == backup['render_triangles'] and
                restored['local_scale'] == backup['local_scale'] and restored['local_position'] == backup['local_position'] and
                restored['attachment']['status'] == 'absent')
            require(report['backup_restored'], 'Original backup source state changed')
        except Exception as exc:
            report.update(passed=False, backup_restore_error=str(exc)); raise
        finally:
            (args.out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'receipt': str((args.out/'receipt.json').resolve()), 'passed': report['passed'], 'backup_restored': report['backup_restored']}))


if __name__ == '__main__':
    main()
