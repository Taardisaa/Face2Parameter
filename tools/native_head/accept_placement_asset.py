"""Verify one loaded neutral placement asset against authored geometry and body.

This is final acceptance, not a parameter sweep or an empirical deformation fit.
The game export must include o_head and o_body_cf in the same frame.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.model_bridge.artifact import sha


def matrix(values):
    return np.asarray(values, float).reshape(4, 4)


def skin_world(mesh, transforms):
    source = mesh['source']; vertices = np.c_[source['vertices'], np.ones(mesh['vertex_count'])]
    ids = np.asarray(source['bone_indices']); weights = np.asarray(source['bone_weights'])
    bones = np.array([matrix(transforms[i]['local_to_world']) for i in mesh['bone_transform_ids']])
    mats = bones @ np.asarray(source['bindposes']).reshape(-1, 4, 4)
    return sum(weights[:, j, None] * np.einsum('nij,nj->ni', mats[ids[:, j]], vertices)
               for j in range(4))[:, :3]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'built', 'game', 'state', 'source', 'installed', 'sheet', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve earlier acceptance records')
    design = json.loads((args.candidate / 'receipt.json').read_text())
    build = json.loads((args.built / 'receipt.json').read_text())
    arrays = np.load(args.candidate / 'geometry.npz', allow_pickle=False)
    asset = np.load(args.built / 'o_head.npz', allow_pickle=False)
    game = json.loads(args.game.read_text(encoding='utf-8-sig'))
    state = json.loads(args.state.read_text(encoding='utf-8-sig'))
    source = json.loads(args.source.read_text())
    if state['active'] or game['frame_count'] != game['frame_count_end']:
        raise ValueError('Require a native head and a coherent head/body game frame')
    if sha(args.installed) != build['package_sha256'] or sha(args.source) != build['source_sha256']:
        raise ValueError('Installed package/source differs from the actual build')
    if sha(args.candidate / 'geometry.npz') != design['geometry_sha256']:
        raise ValueError('Candidate geometry changed')
    meshes = {x['mesh_name']: x for x in game['meshes']}
    head, body = meshes['o_head'], meshes['o_body_cf']
    if not head['enabled'] or not head['active_in_hierarchy']:
        raise ValueError('Actual native head is not visible')
    # ChaControl loads the selected prefab under the ct_head slot. Its runtime
    # root name is not a reliable serialized prefab identifier; literal mesh
    # and skin readback below binds the actual loaded object to the private asset.
    if '/ct_head[' not in head['renderer_path']:
        raise ValueError('Actual o_head is outside the native head slot')
    for key, saved in (('vertices', 'verts'), ('triangles', 'faces'), ('normals', 'normals'),
                       ('bone_indices', 'bone_idx'), ('bone_weights', 'bone_w'), ('bindposes', 'bindpose')):
        expected = asset[saved]
        if not np.array_equal(np.asarray(head['source'][key], dtype=expected.dtype).reshape(expected.shape), expected):
            raise ValueError('Game source differs from the serialized asset: ' + key)
    if body['source_geometry_sha256'] != design['body_source_geometry_sha256']:
        raise ValueError('Actual body no longer matches the authored interface')
    labels = np.full(len(arrays['original_vertices']), -1, int)
    for i, part in enumerate(source['surface']['components']['components']):
        labels[part['canonical_vertex_ids']] = i
    integrated_labels = np.zeros(len(arrays['vertices']), int)
    integrated_labels[:len(arrays['crop_recipes'])] = labels[arrays['crop_recipes'][:, 0].astype(int)]
    head_faces = arrays['faces'][integrated_labels[arrays['faces'][:, 0]] == 0]
    retained = np.unique(head_faces)
    if not np.array_equal(asset['verts'], arrays['vertices'][retained].astype(np.float32)):
        raise ValueError('Serialized head changed the reviewed placement')
    inverse = np.full(len(arrays['vertices']), -1, int); inverse[retained] = np.arange(len(retained))
    if not np.array_equal(asset['faces'], inverse[head_faces]):
        raise ValueError('Serialized head changed the integrated topology')
    transforms = {x['id']: x for x in game['transforms']}
    worlds = [skin_world(m, transforms) for m in (head, body)]
    baked = [np.asarray(m['baked']['world_candidates']['scale_free_trs']['vertices']) for m in (head, body)]
    bake_errors = [float(np.abs(w - b).max()) for w, b in zip(worlds, baked)]
    # Installed Unity bridge's scale-free TRS conversion is checked against the
    # actual LBS matrices in this frame, not assumed valid for arbitrary exports.
    if max(bake_errors) > 1e-5:
        raise ValueError('Raw BakeMesh conversion disagrees with actual skinning')
    head_root = body['bone_transform_ids'][body['bone_names'].index('cf_J_Head_s')]
    local = (np.c_[baked[0], np.ones(len(baked[0]))] @ matrix(transforms[head_root]['world_to_local']).T)[:, :3]
    placement_error = float(np.abs(local - asset['verts']).max())
    seam = inverse[arrays['native_outer_ring_authored_ids']]
    matched = np.asarray(design['body_matched_vertex_ids'])
    seam_error = float(np.linalg.norm(baked[0][seam] - baked[1][matched], axis=1).max())
    if max(placement_error, seam_error) > 1e-5:
        raise ValueError('Neutral loaded placement/interface differs from the candidate')
    result = dict(format='native_head_placement_game_acceptance_v1',
                  candidate_geometry_sha256=design['geometry_sha256'], package_sha256=sha(args.installed),
                  game_geometry_sha256=sha(args.game), source_state_sha256=sha(args.state),
                  screenshot=str(args.sheet.resolve()), screenshot_sha256=sha(args.sheet),
                  serialized_prefab=build['prefab'], actual_renderer_path=head['renderer_path'], frame_count=game['frame_count'],
                  actual_native_asset_matches=True, reviewed_head_positions_preserved=True,
                  native_body_source_unchanged=True, source_overlay_active=False,
                  raw_bake_lbs_max_errors=bake_errors, neutral_placement_max_error=placement_error,
                  actual_same_frame_seam_max_distance=seam_error, pass_neutral_geometry=True,
                  native_eye_geometry_authored=build['native_eye_geometry_authored'],
                  scope='This placement, actual BP body and neutral card only; materials, eyes/gaze, expressions and arbitrary sliders remain unverified')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('actual_native_asset_matches', 'native_body_source_unchanged', 'pass_neutral_geometry')}))


if __name__ == '__main__':
    main()
