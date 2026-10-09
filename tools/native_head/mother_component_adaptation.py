"""Adapt all eight native components and every original expression frame.

The resulting NPZ/JSON files are authoring intermediates, not an installed or
certified head. Source topology, UV, weights, channel names and controller tables
remain intact. Bone pivots/bindposes, actual body interface shading and full
surface acceptance are separate unfinished requirements.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from tools.model_bridge.artifact import sha
from tools.model_bridge.attachment_audit import topology
from tools.native_head.mother_default_pose import dense_frame
from tools.native_head.mother_surface_warp import SurfaceWarp, direction_maps
from tools.native_head.mother_template_inputs import save_json, source_file


def verified_json(row):
    if sha(row['path']) != row['sha256']:
        raise ValueError('Input changed: '+row['path'])
    return json.loads(Path(row['path']).read_text(encoding='utf-8'))


def eye_maps(inputs, candidate, placement):
    flame = np.load(inputs/'flame_zero_identity.npz', allow_pickle=False)
    v, f = flame['v_template'], flame['faces_tensor']
    components = [c for c in topology(v, f)['components']
                  if not any(set(b['vertices']).issubset(c['vertices'])
                    for b in topology(v, f)['boundaries'])]
    if len(components) != 2:
        raise ValueError('Expected two closed source eye components')
    joint = flame['J_regressor']@v
    result = {}
    for side in ('L', 'R'):
        native = np.load(inputs/('o_eyebase_'+side+'.npz'), allow_pickle=False)
        names = native['bone_names'].tolist()
        i = names.index('cf_J_eye_rs_'+side)
        pivot = np.linalg.inv(native['bindpose'][i])[:3, 3].astype(float)
        sign = np.sign(pivot[0])
        comp = next(c for c in components if np.sign(v[c['vertices'], 0].mean()) == sign)
        # Match the actual eye mesh's skin support to the decoder eye joint;
        # names of left/right masks are not used to guess coordinate sides.
        weights = flame['lbs_weights'][comp['vertices']].mean(0)
        j = int(weights.argmax())
        if j not in (3, 4):
            raise ValueError('Source eyeball is not eye-joint supported')
        rotation = np.asarray(placement['rotation'])
        target_pivot = joint[j]@rotation.T*placement['scale']+placement['translation']
        eye = candidate['reference_vertices'][comp['vertices']]
        scale = np.ptp(eye[:, 0])/np.ptp(native['verts'][:, 0])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid authored eye size')
        result[side] = dict(native_pivot=pivot, target_pivot=target_pivot, scale=scale,
            native_bone='cf_J_eye_rs_'+side, source_joint=j,
            source_component_vertex_ids=comp['vertices'])
    return result


def map_eye(points, maps):
    authored = np.empty_like(points, dtype=float)
    jac = np.empty((len(points), 3, 3))
    if np.any(points[:, 0] == 0):
        raise ValueError('Eye component crosses the side plane')
    for row in maps.values():
        selected = np.sign(points[:, 0]) == np.sign(row['native_pivot'][0])
        authored[selected] = (points[selected]-row['native_pivot'])*row['scale']+row['target_pivot']
        jac[selected] = np.eye(3)*row['scale']
    return authored, jac


def adapt_frames(arrays, shapes, default_weights, authored_reference, jacobian):
    count = len(arrays['verts'])
    frames = [dense_frame(shapes, i, count) for i in range(len(shapes['channels']))]
    source_reference = {key: arrays[field].astype(float).copy() for key, field in
                        [('vertex', 'verts'), ('normal', 'normals')]}
    source_reference['tangent'] = arrays['tangents'][:, :3].astype(float).copy()
    for i, weight in default_weights.items():
        i = int(i)
        if i != -1 and weight:
            for key in source_reference:
                source_reference[key] += frames[i][key]*weight/100
    nmap, tmap = direction_maps(jacobian, source_reference['normal'], source_reference['tangent'])
    maps = dict(vertex=jacobian, normal=nmap, tangent=tmap)
    transformed = [{key: np.einsum('nij,nj->ni', maps[key], frame[key])
                    for key in maps} for frame in frames]
    target_ref = dict(vertex=authored_reference,
        normal=np.einsum('nij,nj->ni', nmap, source_reference['normal']),
        tangent=np.einsum('nij,nj->ni', tmap, source_reference['tangent']))
    bind = {key: value.copy() for key, value in target_ref.items()}
    for i, weight in default_weights.items():
        i = int(i)
        if i != -1 and weight:
            for key in bind:
                bind[key] -= transformed[i][key]*weight/100
    updated = copy.deepcopy(shapes)
    for i, channel in enumerate(shapes['channels']):
        shape = shapes['shapes'][channel['frameIndex']]
        begin, end = shape['firstVertex'], shape['firstVertex']+shape['vertexCount']
        for row in updated['vertices'][begin:end]:
            index = row['index']
            for key in maps:
                row[key] = dict(zip('xyz', map(float, transformed[i][key][index])))
    output = {**arrays, 'verts': bind['vertex'], 'normals': bind['normal'],
              'tangents': np.column_stack([bind['tangent'], arrays['tangents'][:, 3]])}
    rebuilt = {key: value.copy() for key, value in bind.items()}
    for i, weight in default_weights.items():
        i = int(i)
        if i != -1 and weight:
            for key in rebuilt:
                rebuilt[key] += transformed[i][key]*weight/100
    error = {key: float(np.max(np.linalg.norm(rebuilt[key]-target_ref[key], axis=1)))
             for key in rebuilt}
    if max(error.values()) > 1e-12:
        raise ValueError('Native controller reference double-baked or lost during adaptation')
    return output, updated, target_ref, error


def adapt(inputs, candidate_dir, out):
    if out.exists():
        raise FileExistsError('Preserve previous adaptation; use a fresh output')
    inputs_receipt = json.loads((inputs/'receipt.json').read_text())
    candidate_receipt = json.loads((candidate_dir/'receipt.json').read_text())
    if candidate_receipt['inputs']['sha256'] != sha(inputs/'receipt.json'):
        raise ValueError('Candidate belongs to different prepared inputs')
    default_receipt = verified_json(candidate_receipt['default_reference'])
    prefab = verified_json(inputs_receipt['native_prefab_export'])
    regions = verified_json(inputs_receipt['regions'])
    head = dict(np.load(candidate_dir/'o_head_candidate.npz', allow_pickle=False))
    field = SurfaceWarp(head['original_vertices'], head['verts'], head['faces'])
    oral_faces = head['faces'][regions['inner_mouth_component_face_ids']]
    oral = SurfaceWarp(head['original_vertices'], head['verts'], oral_faces)
    aliases = {int(k): int(v) for k, v in regions['graph_aliases'].items()}
    head_jac = field.vertex_gradients(aliases)
    eyes = eye_maps(inputs, head, candidate_receipt['placement'])
    out.mkdir(parents=True)
    reports = []
    for renderer in prefab['renderers']:
        ref_row = next(r for r in default_receipt['renderer_references'] if r['mesh']==renderer['mesh'])
        for source in (renderer['array_export'], ref_row['reference']):
            if sha(source['path']) != source['sha256']:
                raise ValueError('Native component input changed')
        arrays = dict(np.load(renderer['array_export']['path'], allow_pickle=False))
        reference = dict(np.load(ref_row['reference']['path'], allow_pickle=False))
        transform = prefab['transforms'][str(renderer['transform'])]
        head_transform = next(r['transform'] for r in prefab['renderers'] if r['mesh']=='o_head')
        t_head = prefab['transforms'][str(head_transform)]
        if any(transform[k] != t_head[k] for k in ('parent', 'pos', 'rot', 'scale')):
            raise ValueError('Unsupported differing renderer coordinate frame; must transform explicitly')
        name, points = renderer['mesh'], reference['closed_reference_vertices']
        correspondence = None
        if name == 'o_head':
            authored, jac = head['verts'], head_jac
            method = 'Exact authored native vertex positions; common incident-triangle gradient for physical UV copies'
        elif name.startswith('o_eyebase_') or name=='o_eyeshadow':
            authored, jac = map_eye(points, eyes)
            method = 'Shared eye similarity from native rotation pivot and source FLAME eye joint/diameter; native eye surfaces retained'
        elif name in ('o_tooth', 'o_tang'):
            authored, jac, correspondence = oral.map(points)
            method = 'Only original MouthCavity index component surface-gradient transport'
        else:
            authored, jac, correspondence = field.map(points)
            method = 'Original head surface-gradient transport with normal offsets; secondary skin attachment'
        shapes = verified_json(renderer['blendshape_export'])
        adapted, shape_output, target_ref, errors = adapt_frames(arrays, shapes, ref_row['weights'], authored, jac)
        for key in arrays:
            if key not in ('verts', 'normals', 'tangents') and not np.array_equal(arrays[key], adapted[key]):
                raise ValueError('Native component structure changed: '+key)
        if shape_output['channels'] != shapes['channels'] or shape_output['fullWeights'] != shapes['fullWeights']:
            raise ValueError('Native expression channel table changed')
        mesh_path, shape_path = out/(name+'.npz'), out/(name+'_blendshapes.json')
        np.savez_compressed(mesh_path, **adapted, authored_closed_reference=target_ref['vertex'],
            authored_reference_normals=target_ref['normal'], authored_reference_tangent_xyz=target_ref['tangent'],
            authoring_jacobian=jac)
        save_json(shape_path, shape_output)
        reports.append(dict(mesh=name, method=method, source=renderer['array_export'],
            original_frames=renderer['blendshape_export'], arrays=source_file(mesh_path), frames=source_file(shape_path),
            channel_count=len(shapes['channels']), all_sparse_rows_retained=len(shapes['vertices'])==len(shape_output['vertices']),
            source_reference_weights=ref_row['weights'], reference_reconstruction_error=errors,
            minimum_local_determinant=float(np.linalg.det(jac).min()), correspondence=correspondence))
    save_json(out/'receipt.json', dict(format='native_mother_component_adaptation_v1',
        inputs=source_file(inputs/'receipt.json'), candidate=source_file(candidate_dir/'receipt.json'),
        candidate_arrays=source_file(candidate_dir/'o_head_candidate.npz'),
        code=source_file(Path(__file__)), surface_warp_code=source_file(Path(__file__).with_name('mother_surface_warp.py')),
        renderer_outputs=reports, eye_authoring={side:{key: value.tolist() if isinstance(value,np.ndarray) else value
            for key,value in row.items()} for side,row in eyes.items()},
        controller_modified=False, prefab_modified=False, installed=False, game_mutated=False, deliverable=False,
        pending=['Bone pivot/bindpose adaptation', 'Actual BP body interface positions and authored shading',
                 'Anatomical correspondence review and full surface intersection/accuracy gates',
                 'Original bundle patch/package and native integration acceptance'],
        limitations=['Local surface transport is an asset-authoring extension, not a globally bijective volume mapping',
                     'All expression rows adapted but complete animation/slider compatibility is not certified',
                     'Bind frame normals/tangents are authored attributes; GPU normalization follows native shaders']))
    print(json.dumps(dict(output=str(out.resolve()), all_components_adapted=True,
                          all_original_channels_retained=True, installed=False, deliverable=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    adapt(args.inputs.resolve(), args.candidate.resolve(), args.out.resolve())
