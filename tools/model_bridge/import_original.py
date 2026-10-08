"""Export a selected original model output, attach it and save a new HS2 card.

Default: restore the pre-run character after saving; --keep-in-game explicitly
leaves the chosen source model visible. Any failure attempts card restoration.
This does not infer photos, fit sliders, predict albedo or retarget native mouths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import numpy as np

from .artifact import ModelArtifact, sha
from .attachment_acceptance import check, require
from .attachment_profile import read_profile
from .attachment_rebase import rebase, validate_source_pair
from .export_game_mesh import export
from .game_import import request
from .surface_acceptance import verify


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def same_source(before, after):
    """Source card-owned state, deliberately separate from native parameters."""
    if after.get('active') != before.get('active'):
        return False
    if not before.get('active'):
        return True
    for key in ('artifact_sha256', 'local_scale', 'local_position', 'source_pose_axis_angle'):
        if after.get(key) != before.get(key):
            return False
    for key in ('render_vertices', 'render_triangles', 'canonical_vertices', 'canonical_triangles'):
        if (key in before) != (key in after) or key in before and not np.array_equal(
                np.asarray(before[key], np.float32 if 'vertices' in key else np.int32),
                np.asarray(after[key], np.float32 if 'vertices' in key else np.int32)):
            return False
    a, b = before.get('attachment', {}), after.get('attachment', {})
    return a.get('status') == b.get('status') and a.get('artifact_sha256') == b.get('artifact_sha256')


def restore(base, card, before, snapshot, *, transport=request):
    transport(base, 'POST', {'path': str(card.resolve())}, route='/maker/card/load')
    deadline = time.monotonic()+30
    while True:
        state = transport(base, 'GET')
        if same_source(before, state):
            current_snapshot = transport(base, 'GET', route='/maker/snapshot')
            if current_snapshot == snapshot:
                return {'pre_run_source_state_restored': True, 'native_parameter_snapshot_literal': True,
                    'ordinary_native_character': not before['active'],
                    'scope': 'Card-owned source state and exposed native snapshot; not all unknown plugin/transient animation state'}
        require(time.monotonic() < deadline, 'Pre-run source/native snapshot did not restore')
        time.sleep(.25)


def thumbnail(base, path, *, transport=request):
    # This produces the card's visible thumbnail, not a deformation comparison.
    params = {'out': str(path.resolve()), 'w': 256, 'h': 256, 'yaw': 0,
        'hide_hair': 'false', 'hide_accessories': 'false'}
    result = transport(base, 'GET', route='/maker/render?'+urlencode(params))
    require(Path(result['path']).resolve() == path.resolve() and path.is_file(), 'Native card thumbnail was not written')
    return result


def require_exclusive_display(state):
    rows = state.get('native_head_display_renderers', [])
    require(state.get('source_display_enabled') is True and
        state.get('native_head_display_exclusive') is True and rows and
        all(row.get('enabled') is False for row in rows),
        'Source head display overlaps native renderers or is disabled')


def apply_and_save(base, artifact, path, descriptor_inputs, descriptor_paths, out, *, keep_in_game=False, transport=request):
    """Recoverable native transaction. Export/provenance checks run beforehand."""
    health = transport(base, 'GET', route='/health')
    version = tuple(int(x) for x in health['version'].split('.'))
    require(version >= (0, 31, 11), 'Install bridge0.31.11+ for exclusive source display and native backup')
    before = transport(base, 'GET')
    snapshot = transport(base, 'GET', route='/maker/snapshot')
    write(out/'before_native_snapshot.json', snapshot)
    write(out/'before_source_state.json', before)
    thumb = out/'before.png'; thumbnail(base, thumb, transport=transport)
    backup = out/'before_character.png'
    saved = transport(base, 'POST', {'path': str(backup.resolve()), 'thumbnail': str(thumb.resolve())}, route='/maker/card/save')
    require(saved['source_head_embedded'] == before['active'] and backup.is_file(), 'Backup source/native classification differs')
    report = {'plugin': health, 'original_source': artifact['source'], 'source_sha256': sha(path),
        'backup_card': str(backup.resolve()), 'backup_card_sha256': sha(backup),
        'keep_in_game_requested': keep_in_game, 'source_accuracy_certified': False,
        'native_parameter_mapping': False, 'new_model_inference': False, 'source_parameters_modified': False,
        'support': 'Original source geometry/rig/UV/components and native cut/collar; fixed source identity/expression/eyelids; no teeth/tongue, inferred albedo or native expression retarget'}
    mutated = False
    try:
        vertices = np.asarray(artifact['vertices'], np.float32); lo, hi = vertices.min(0), vertices.max(0)
        scale = float(before['reference_local_size'][1]/(hi[1]-lo[1]))
        translation = (np.asarray(before['reference_local_center'])-scale*((lo.astype(float)+hi)/2)).tolist()
        mutated = True
        state = transport(base, 'POST', {'path': str(path.resolve()), 'sha256': sha(path), 'scale': scale, 'translation': translation})
        write(out/'actual_imported_state.json', state)
        require_exclusive_display(state)
        report['source_original'] = verify(state, artifact, out, 'source_original', original=True)
        descriptor = rebase(descriptor_inputs['proposal'], descriptor_inputs['native_state'], descriptor_inputs['reference_state'],
            state, descriptor_paths['reference_artifact'], path)
        descriptor['packager_sha256'] = sha(Path(__file__).with_name('attachment_rebase.py'))
        descriptor['actual_source_state_sha256'] = sha(out/'actual_imported_state.json')
        descriptor_path = out/'attachment.json'; write(descriptor_path, descriptor)
        require(descriptor_path.stat().st_size <= 4*1024*1024, 'Native attachment bound exceeded')
        state = transport(base, 'POST', {'path': str(descriptor_path.resolve()), 'sha256': sha(descriptor_path)}, route='/maker/face/model/attachment')
        require_exclusive_display(state)
        report['attached_original_source'] = verify(state, artifact, out, 'attached_original_source', original=True)
        report['actual_attachment'] = check(state, descriptor, artifact['triangles'], out, 'actual_attachment')
        thumbnail(base, out/'source.png', transport=transport)
        require_exclusive_display(transport(base, 'GET'))
        report['native_head_display_exclusive'] = True
        card = out/'source_character.png'
        report['saved'] = transport(base, 'POST', {'path': str(card.resolve()), 'thumbnail': str((out/'source.png').resolve())}, route='/maker/card/save')
        require(report['saved']['source_head_embedded'] is True and card.is_file(), 'Source card missing')
        report['source_card_sha256'] = sha(card)
        report['passed'] = True
        return report
    except Exception as exc:
        report.update(passed=False, failure=str(exc)); raise
    finally:
        try:
            if mutated and (not keep_in_game or not report.get('passed')):
                report['restoration'] = restore(base, backup, before, snapshot, transport=transport)
        except Exception as exc:
            report.update(passed=False, restoration_failure=str(exc)); raise
        finally:
            write(out/'receipt.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Original SMIRK/MICA raw export manifest')
    parser.add_argument('--image-index', type=int, required=True, help='Explicit selected original photo row')
    parser.add_argument('--source-obj', type=Path, required=True)
    parser.add_argument('--source-mask', type=Path, required=True)
    parser.add_argument('--attachment-profile', type=Path, required=True)
    parser.add_argument('--texture', type=Path)
    parser.add_argument('--left-eye-texture', type=Path)
    parser.add_argument('--right-eye-texture', type=Path)
    parser.add_argument('--base', default='http://127.0.0.1:43127')
    parser.add_argument('--out', type=Path, required=True, help='Fresh output directory; never overwrite cards')
    parser.add_argument('--keep-in-game', action='store_true', help='Explicitly leave this chosen original source visible after saving')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = ModelArtifact(args.manifest, args.image_index); model.verify_sources()
    path = args.out/'source.json'
    export(model, path, head_local=True, with_rig=True, source_obj=args.source_obj, source_mask=args.source_mask,
        texture=args.texture, left_eye_texture=args.left_eye_texture, right_eye_texture=args.right_eye_texture)
    inputs, paths = read_profile(args.attachment_profile)
    _, artifact, _, _ = validate_source_pair(inputs['proposal'], inputs['native_state'], inputs['reference_state'], paths['reference_artifact'], path)
    write(args.out/'preflight.json', {'source_sha256': sha(path), 'profile_sha256': sha(args.attachment_profile),
        'selected_image_index': model.image_index, 'original_definition_verified': True,
        'live_body_compatibility': 'Required at native attachment activation; not certified by static profile',
        'implementation_sha256': sha(Path(__file__))})
    report = apply_and_save(args.base, artifact, path, inputs, paths, args.out, keep_in_game=args.keep_in_game)
    print(json.dumps({'card': str((args.out/'source_character.png').resolve()), 'receipt': str((args.out/'receipt.json').resolve()),
        'source_preserved': report['passed'], 'left_in_game': args.keep_in_game,
        'ordinary_native_character_restored': report.get('restoration', {}).get('ordinary_native_character', False)}))


if __name__ == '__main__':
    main()
