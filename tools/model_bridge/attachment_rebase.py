"""Build a new neck descriptor for a different unchanged FLAME photo identity.

The existing exact-input package path remains untouched. This path first validates
that design, then proves shared original topology/template/rig definitions and
binds a new descriptor to the actual current artifact and uniform placement.
Native cut indices and collar connectivity are retained; no face fitting/welding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .artifact import ModelArtifact, host_path, sha
from .attachment_runtime import boundary, package
from .neck_patch import digest_array


SHARED_STATE = ('v_template', 'faces_tensor', 'parents', 'J_regressor', 'lbs_weights', 'posedirs')


def original_artifact(path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('format') not in {'hs2_source_head_mesh_v1', 'hs2_source_head_mesh_v2', 'hs2_source_head_mesh_v3'} or (
            payload.get('geometry_mode') != 'head_local'):
        raise ValueError('Original head-local model artifact required')
    model = ModelArtifact.from_game_source(payload['source']); model.verify_sources()
    vertices, faces = model.mesh(head_local=True)
    if not np.array_equal(np.asarray(payload['vertices'], np.float32), vertices) or not np.array_equal(
            payload['triangles'], faces.reshape(-1)) or set(payload['source_parameters']) != set(model.parameters) or any(
            not np.array_equal(payload['source_parameters'][key], value) for key, value in model.parameters.items()):
        raise ValueError('Interchange must preserve the complete original mesh and raw parameters')
    if 'rig' in payload:
        from .rig import export_rig
        expected = export_rig(model)
        if any(payload['rig'].get(key) != expected[key] for key in expected if key != 'policy'):
            raise ValueError('Interchange rig differs from the selected original model definition')
    return payload, model


def canonical(state):
    return (np.asarray(state.get('canonical_vertices', state.get('render_vertices')), np.float32),
            np.asarray(state.get('canonical_triangles', state.get('render_triangles'))))


def rebase(proposal, native, reference, current, reference_path, source_path):
    # Preserve the exact original design's validation and its immutable native
    # signatures, including complete skin weights/bindposes and authored UVs.
    result = package(proposal, native, reference)
    if sha(reference_path) != reference['artifact_sha256']:
        raise ValueError('Original attachment reference artifact changed')
    old, old_model = original_artifact(reference_path)
    new, model = original_artifact(source_path)
    old_positions, old_faces = canonical(reference)
    if not np.array_equal(old_positions, np.asarray(old['vertices'], np.float32)) or not np.array_equal(old_faces, old['triangles']):
        raise ValueError('Reference artifact does not equal the original design state')
    proof = {}
    for key in SHARED_STATE:
        a, b = old_model.state[key], model.state[key]
        if a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError('Original source definition differs: ' + key)
        proof[key] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': digest_array(a, a.dtype.str)}
    if new['triangles'] != old['triangles']:
        raise ValueError('Original face-corner topology/order differs')
    if not current.get('active') or current.get('artifact_sha256') != sha(source_path) or any(
            current.get(key) is not True for key in ('source_vertices_unchanged', 'source_triangle_indices_unchanged',
                'source_world_similarity_frame', 'source_render_mesh_bound', 'source_render_material_bound', 'source_display_enabled')):
        raise ValueError('Current exact original source display and compatible frame required')
    if current.get('attachment', {}).get('status') != 'absent':
        raise ValueError('Detach the current attachment before preparing another descriptor')
    positions, faces = canonical(current)
    if not np.array_equal(positions, np.asarray(new['vertices'], np.float32)) or not np.array_equal(faces, new['triangles']):
        raise ValueError('Actual current canonical mesh differs from the selected original artifact')
    scale = np.asarray(current['local_scale'], np.float32)
    position = np.asarray(current['local_position'], np.float32)
    if scale.shape != (3,) or position.shape != (3,) or not np.isfinite(scale).all() or not np.isfinite(position).all() or (
            scale[0] <= 0 or not np.all(scale == scale[0])):
        raise ValueError('Actual positive uniform source placement required')
    ring = result['source_ring_indices']; n = len(ring)
    collar_edges = boundary(result['collar_triangles'])
    join = {(ring[b], ring[a]) for a, b in collar_edges if a < n and b < n}
    wanted = set(ring)
    if len(join) != n or join != {(a, b) for a, b in boundary(new['triangles']) if a in wanted and b in wanted}:
        raise ValueError('Original source neck boundary correspondence differs')
    reference_design = result['design']
    result['source_artifact_sha256'] = sha(source_path)
    result['source_placement'] = {'scale': float(scale[0]), 'translation': position.tolist()}
    result['design'] = {'mode': 'shared_flame_template_new_identity_v1',
        'policy': 'New descriptor binds the unchanged selected source output; original native cut and index-boundary collar connectivity retained',
        'reference_cut_design': reference_design,
        'reference_source_artifact': {'path': str(reference_path.resolve()), 'sha256': sha(reference_path)},
        'current_source_artifact': {'path': str(source_path.resolve()), 'sha256': sha(source_path),
            'image_index': model.image_index, 'image_sha256': model.image['input_sha256'],
            'raw_artifact_sha256': model.image['sha256'], 'manifest_sha256': sha(model.path)},
        'shared_source_definition': proof, 'source_ring_indices_sha256': digest_array(ring, '<i4'),
        'canonical_positions_modified': False, 'raw_parameters_modified': False,
        'source_accuracy_certified': False, 'collar_self_intersection_certified': False,
        'placement_policy': 'Actual source state positive uniform local placement; current world similarity guard and native body signatures checked by runtime',
        'normals_hash_exclusion': reference_design['normals_hash_exclusion']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--native-state', type=Path, required=True, help='Original validated design body state')
    parser.add_argument('--reference-state', type=Path, required=True, help='Original validated design source state')
    parser.add_argument('--reference-artifact', type=Path, help='Defaults to original design state artifact_path')
    parser.add_argument('--source-artifact', type=Path, required=True)
    parser.add_argument('--source-state', type=Path, required=True, help='Actual current imported source, at original pose with no attachment')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_state.read_text(encoding='utf-8'))
    reference_path = args.reference_artifact or host_path(reference['artifact_path'])
    result = rebase(json.loads(args.proposal.read_text(encoding='utf-8')),
        json.loads(args.native_state.read_text(encoding='utf-8')), reference,
        json.loads(args.source_state.read_text(encoding='utf-8')), reference_path, args.source_artifact)
    result['provenance'] = [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
        (args.proposal, args.native_state, args.reference_state, reference_path, args.source_artifact, args.source_state)]
    result['packager_sha256'] = sha(Path(__file__))
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n'
    if len(encoded.encode('utf-8')) > 4 * 1024 * 1024:
        raise ValueError('Attachment JSON exceeds the existing native 4 MiB bound')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    print(json.dumps({'attachment': str(args.out.resolve()), 'sha256': sha(args.out),
                      'new_identity_source_preserved': True, 'hs2_parameter_mapping': False}))


if __name__ == '__main__':
    main()
