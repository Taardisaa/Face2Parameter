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
    rotation = np.asarray(placement['rotation'])
    # Identity candidates carry the actual decoded, placed FLAME reference.
    # Recover canonical coordinates before using its unchanged joint regressor.
    reference = ((candidate['reference_vertices']-placement['translation'])/
                 placement['scale'])@rotation
    joint = flame['J_regressor']@reference
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


def adapt_frames(arrays, shapes, default_weights, authored_reference, jacobian,close_action=None):
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
    target_ref = dict(vertex=np.asarray(authored_reference,float).copy(),
        normal=np.einsum('nij,nj->ni', nmap, source_reference['normal']),
        tangent=np.einsum('nij,nj->ni', tmap, source_reference['tangent']))
    identity_bind={key:value.copy() for key,value in target_ref.items()}
    for i,weight in default_weights.items():
        if int(i)!=-1 and weight:
            for key in identity_bind:identity_bind[key]-=transformed[int(i)][key]*weight/100
    if close_action is not None:
        channel=int(close_action['channel'])
        if float(default_weights.get(str(channel),default_weights.get(channel,0)))!=100:
            raise ValueError('Close correction must use actual default 100-percent native endpoint')
        for key in target_ref:
            transformed[channel][key]+=close_action[key]
            target_ref[key]+=close_action[key]
    bind = {key: value.copy() for key, value in target_ref.items()}
    for i, weight in default_weights.items():
        i = int(i)
        if i != -1 and weight:
            for key in bind:
                bind[key] -= transformed[i][key]*weight/100
    if close_action is not None and max(np.max(abs(bind[k]-identity_bind[k])) for k in bind)>1e-12:
        raise ValueError('Expression correction was baked into identity bind')
    updated = copy.deepcopy(shapes);updated['vertices']=[]
    if len({c['frameIndex'] for c in shapes['channels']})!=len(shapes['channels']):
        raise ValueError('Shared native frame storage requires explicit authoring')
    for i, channel in enumerate(shapes['channels']):
        shape = shapes['shapes'][channel['frameIndex']]
        begin, end = shape['firstVertex'], shape['firstVertex']+shape['vertexCount']
        source_rows=shapes['vertices'][begin:end]
        original_ids={row['index'] for row in source_rows}
        extra=np.flatnonzero(np.any(np.column_stack([transformed[i][key] for key in maps])!=0,axis=1))
        indices=[row['index'] for row in source_rows]+sorted(set(extra)-original_ids)
        authored_shape=updated['shapes'][channel['frameIndex']]
        authored_shape['firstVertex']=len(updated['vertices']);authored_shape['vertexCount']=len(indices)
        for key,flag in [('normal','hasNormals'),('tangent','hasTangents')]:
            authored_shape[flag]=bool(shape[flag] or np.any(transformed[i][key]!=0))
        for index in indices:
            row={'index':int(index)}
            for key in maps:
                row[key] = dict(zip('xyz', map(float, transformed[i][key][index])))
            updated['vertices'].append(row)
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
    if candidate_receipt.get('identity_transfer',{}).get('nonlinear_per_identity_mouth_recomputation'):
        acceptance=candidate_receipt.get('mouth_acceptance',{})
        if (acceptance.get('status')!='accepted_static_head' or
            acceptance.get('candidate_arrays_sha256')!=sha(candidate_dir/'o_head_candidate.npz')):
            raise ValueError('Generic full-detail mouth requires accepted geometry before component/frame authoring')
        quality=verified_json(acceptance['quality'])
        if quality['added_crossings'] or quality['arrays']['sha256']!=acceptance['candidate_arrays_sha256']:
            raise ValueError('Generic mouth geometry acceptance does not match candidate')
    close_receipt=candidate_receipt.get('closed_pose_authoring')
    identity_head=head['verts']
    close_field=close_oral=None
    if close_receipt is not None:
        adjustment=np.load(close_receipt['adjustment']['path'],allow_pickle=False)
        if sha(close_receipt['adjustment']['path'])!=close_receipt['adjustment']['sha256']:
            raise ValueError('Closed-pose action changed')
        identity_head=adjustment['full_identity_reference_vertices']
        if not np.allclose(identity_head+adjustment['vertex_delta'],head['verts'],atol=1e-12,rtol=0):
            raise ValueError('Close endpoint does not reconstruct from full identity')
        close_field=SurfaceWarp(identity_head,head['verts'],head['faces'])
    field = SurfaceWarp(head['original_vertices'], identity_head, head['faces'])
    oral_faces = head['faces'][regions['inner_mouth_component_face_ids']]
    oral = SurfaceWarp(head['original_vertices'], identity_head, oral_faces)
    if close_field is not None:close_oral=SurfaceWarp(identity_head,head['verts'],oral_faces)
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
            authored, jac = identity_head, head_jac
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
        close_action=None
        if close_field is not None:
            if name=='o_head':closed,close_jac=head['verts'],close_field.vertex_gradients(aliases)
            elif name in ('o_tooth','o_tang'):closed,close_jac,_=close_oral.map(authored)
            elif name.startswith('o_eyebase_') or name=='o_eyeshadow':closed,close_jac=authored,np.broadcast_to(np.eye(3),(len(authored),3,3))
            else:closed,close_jac,_=close_field.map(authored)
            _,_,identity_ref,_=adapt_frames(arrays,shapes,ref_row['weights'],authored,jac)
            nmap,tmap=direction_maps(close_jac,identity_ref['normal'],identity_ref['tangent'])
            delta=dict(vertex=closed-authored,
                normal=np.einsum('nij,nj->ni',nmap,identity_ref['normal'])-identity_ref['normal'],
                tangent=np.einsum('nij,nj->ni',tmap,identity_ref['tangent'])-identity_ref['tangent'])
            ctrl=next(b['original_typetree'] for b in prefab['behaviours'] if b['class_name']=='FaceBlendShape')['MouthCtrl']
            obj=renderer['original_renderer_typetree']['m_GameObject']['m_PathID']
            target=next((t for t in ctrl['FBSTarget'] if t['ObjTarget']['m_PathID']==obj),None)
            if target is not None:
                close_action=dict(channel=target['PtnSet'][close_receipt['pattern']]['Close'],**delta)
            elif max(np.max(abs(v),initial=0) for v in delta.values())>1e-12:
                raise ValueError('Secondary component needs a mouth action but has no original MouthCtrl target: '+name)
        adapted, shape_output, target_ref, errors = adapt_frames(arrays, shapes, ref_row['weights'], authored, jac,close_action)
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
            channel_count=len(shapes['channels']), all_original_sparse_rows_retained=True,
            extra_authored_sparse_rows=len(shape_output['vertices'])-len(shapes['vertices']),
            close_action_channel=close_action['channel'] if close_action else None,
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
