"""Patch a complete native bundle in place, retaining all native asset data.

This is an isolated development bundle, not a registered or deliverable zipmod.
No calls to legacy write_mesh (which clears expressions), material replacement,
game mutation, or animation/controller suppression are performed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

from scripts.hs2_extract_head import _read_smr_mesh
from tools.model_bridge.artifact import sha
from tools.native_head.mesh import aabb
from tools.native_head.mother_component_adaptation import verified_json
from tools.native_head.mother_default_pose import dense_frame
from tools.native_head.mother_template_inputs import save_json, source_file


def patch_vertex_channels(mesh, tree, arrays):
    """Change only original position/normal/tangent bytes in existing streams."""
    handler = MeshHandler(mesh)
    data = tree['m_VertexData']
    if tree['m_StreamData']['size'] or tree['m_MeshCompression']:
        raise ValueError('External/compressed mesh requires its own exact writer')
    count = data['m_VertexCount']
    streams = handler.get_streams(mesh.m_VertexData.m_Channels, count)
    raw = bytearray(data['m_DataSize'])
    touched = np.zeros(len(raw), dtype=bool)
    for channel_index, field, width in ((0, 'verts', 3), (1, 'normals', 3), (2, 'tangents', 4)):
        channel = data['m_Channels'][channel_index]
        if channel['format'] != 0 or channel['dimension'] != width:
            raise ValueError('Original channel is not the audited float32 layout')
        values = np.asarray(arrays[field], dtype='<f4')
        if values.shape != (count, width) or not np.isfinite(values).all():
            raise ValueError('Invalid authored '+field)
        stream = streams[channel['stream']]
        start = stream.offset+channel['offset']
        end = start+(count-1)*stream.stride+width*4
        if end > len(raw):
            raise ValueError('Channel exceeds original vertex buffer')
        view = np.ndarray((count, width), dtype='<f4', buffer=raw,
                          offset=start, strides=(stream.stride, 4))
        view[:] = values
        for i in range(count):
            touched[start+i*stream.stride:start+i*stream.stride+width*4] = True
    before = np.frombuffer(data['m_DataSize'], dtype=np.uint8)
    after = np.frombuffer(raw, dtype=np.uint8)
    if not np.array_equal(before[~touched], after[~touched]):
        raise ValueError('Unrelated UV/color/skin/padding bytes changed')
    data['m_DataSize'] = bytes(raw)
    return hashlib.sha256(after[~touched].tobytes()).hexdigest()


def frame_bounds(arrays, frames):
    """Bounds include bind geometry and every actual serialized expression frame.

    This does not certify all simultaneous expression/bone combinations. Native
    AssignedWeightsAndSetBounds supplies its own character bounds at loading.
    """
    points = [arrays['verts'], arrays['authored_closed_reference']]
    points.extend(arrays['verts']+dense_frame(frames, i, len(arrays['verts']))['vertex']
                  for i in range(len(frames['channels'])))
    return aabb(np.vstack(points))


def float32_shapes(shapes):
    result = copy.deepcopy(shapes)
    for row in result['vertices']:
        for key in ('vertex', 'normal', 'tangent'):
            row[key] = {axis: float(np.float32(value)) for axis, value in row[key].items()}
    return result


def float32_values(value):
    if isinstance(value, float):
        return float(np.float32(value))
    if isinstance(value, list):
        return [float32_values(v) for v in value]
    if isinstance(value, dict):
        return {k: float32_values(v) for k, v in value.items()}
    return value


def build(inputs, bindings, out):
    if out.exists():
        raise FileExistsError('Preserve previous bundle candidate; use fresh output')
    source = json.loads((inputs/'receipt.json').read_text())
    original_prefab = verified_json(source['native_prefab_export'])
    binding = json.loads((bindings/'receipt.json').read_text())
    if binding['inputs']['sha256'] != sha(inputs/'receipt.json'):
        raise ValueError('Bindings belong to a different complete donor')
    authored_prefab = verified_json(binding['authored_prefab'])
    donor = source['native_original_copy']
    if sha(donor['path']) != donor['sha256']:
        raise ValueError('Original donor bundle changed')
    env = UnityPy.load(donor['path'])
    objects = {o.path_id: o for o in env.objects}
    original_raw = {pid: o.get_raw_data() for pid, o in objects.items()}
    trees = {}
    def edit(pid):
        if pid not in trees:
            trees[pid] = copy.deepcopy(objects[pid].read_typetree())
        return trees[pid]
    reports = []
    for row in original_prefab['renderers']:
        authored = next(r for r in binding['renderer_outputs'] if r['mesh']==row['mesh'])
        if sha(authored['arrays']['path']) != authored['arrays']['sha256']:
            raise ValueError('Authored arrays changed')
        arrays = dict(np.load(authored['arrays']['path'], allow_pickle=False))
        original = dict(np.load(row['array_export']['path'], allow_pickle=False))
        frames = verified_json(authored['frames'])
        original_frames = verified_json(row['blendshape_export'])
        for key in ('faces', 'uv', 'uv1', 'colors', 'bone_idx', 'bone_w', 'bone_names'):
            if not np.array_equal(arrays[key], original[key]):
                raise ValueError('Native structure changed: '+key)
        if frames['channels'] != original_frames['channels'] or frames['shapes'] != original_frames['shapes']:
            raise ValueError('Native frame/channel layout changed')
        if [r['index'] for r in frames['vertices']] != [r['index'] for r in original_frames['vertices']]:
            raise ValueError('Native sparse expression indices changed')
        tree = edit(row['mesh_path_id'])
        preserved_digest = patch_vertex_channels(objects[row['mesh_path_id']].read(), tree, arrays)
        tree['m_Shapes'] = frames
        tree['m_BindPose'] = [{f'e{r}{c}': float(matrix[r, c]) for r in range(4) for c in range(4)}
                             for matrix in arrays['bindpose']]
        bounds = frame_bounds(arrays, frames)
        tree['m_LocalAABB'] = bounds
        # All selected native meshes have one triangle submesh. Do not rebuild
        # indices/counts or collapse a future multi-submesh donor silently.
        if len(tree['m_SubMeshes']) != 1:
            raise ValueError('Need exact per-submesh bound handling for new donor')
        tree['m_SubMeshes'][0]['localAABB'] = bounds
        reports.append(dict(mesh=row['mesh'], mesh_path_id=row['mesh_path_id'],
                            unrelated_vertex_bytes_sha256=preserved_digest,
                            bounds_policy='Bind plus default reference and every original individual frame',
                            arrays=authored['arrays'], frames=authored['frames']))
    for changed in binding['edited_transforms']:
        pid = int(changed['transform'])
        tree = edit(pid)
        value = authored_prefab['transforms'][str(pid)]
        tree['m_LocalPosition'] = dict(zip('xyz', value['pos']))
        tree['m_LocalScale'] = dict(zip('xyz', value['scale']))
    root = objects[int(original_prefab['root_transform'])].read()
    prefab_name = 'p_cf_flame_mother_candidate'
    edit(root.m_GameObject.path_id)['m_Name'] = prefab_name
    old_cab = next(k for k, v in env.file.files.items() if hasattr(v, 'objects'))
    new_cab = 'CAB-'+hashlib.sha256((donor['sha256']+sha(bindings/'receipt.json')+
                                    sha(Path(__file__))).encode()).hexdigest()[:32]
    bundle_name = 'chara/codex/flame_mother_candidate/head.unity3d'
    for pid, obj in objects.items():
        if obj.type.name in ('Texture2D', 'Mesh'):
            original_tree = obj.read_typetree()
            stream = original_tree.get('m_StreamData')
            if stream and old_cab in stream.get('path', ''):
                edit(pid)['m_StreamData']['path'] = stream['path'].replace(old_cab, new_cab)
        elif obj.type.name == 'AssetBundle':
            tree = edit(pid)
            tree['m_Name'] = tree['m_AssetBundleName'] = bundle_name
            tree['m_Container'] = [(key.replace('/'+source['prefab'].lower()+'.prefab',
                '/'+prefab_name.lower()+'.prefab'), value) for key, value in tree['m_Container']]
    for pid, tree in trees.items():
        objects[pid].save_typetree(tree)
    # Rename archive AND resource filenames, keeping their payload byte exact.
    resource_hashes = {k: hashlib.sha256(v.bytes).hexdigest()
                       for k, v in env.file.files.items() if not hasattr(v, 'objects')}
    env.file.files = {k.replace(old_cab, new_cab): v for k, v in env.file.files.items()}
    for key, value in env.file.files.items():
        if hasattr(value, 'name'):
            value.name = key
    encoded = env.file.save(packer='lz4')
    restored = UnityPy.load(encoded)
    check = {o.path_id: o for o in restored.objects}
    if set(check) != set(objects):
        raise ValueError('Original serialized objects were lost or added')
    for pid in objects:
        if pid not in trees and check[pid].get_raw_data() != original_raw[pid]:
            raise ValueError('Untouched serialized object changed: '+str(pid))
        if pid in trees:
            expected = copy.deepcopy(trees[pid])
            # Float32 authoring values round on serialization, then verify all
            # non-authoring fields separately below rather than demanding double precision.
            actual = check[pid].read_typetree()
            excluded = {'m_VertexData','m_BindPose','m_Shapes','m_LocalAABB','m_SubMeshes'} if objects[pid].type.name=='Mesh' else set()
            if objects[pid].type.name=='Transform':
                excluded = {'m_LocalPosition','m_LocalScale'}
            if {k:v for k,v in expected.items() if k not in excluded} != {k:v for k,v in actual.items() if k not in excluded}:
                raise ValueError('Unexpected serialized fields changed: '+str(pid))
            for key in excluded:
                if actual[key] != float32_values(expected[key]):
                    raise ValueError('Authored serialized field differs: '+str(pid)+'/'+key)
    for report, row in zip(reports, original_prefab['renderers']):
        mesh = check[row['mesh_path_id']].read()
        decoded = _read_smr_mesh(mesh)
        arrays = dict(np.load(report['arrays']['path'], allow_pickle=False))
        for key in decoded:
            expected = arrays[key].astype(decoded[key].dtype)
            if not np.array_equal(decoded[key], expected):
                raise ValueError('Serialized component mismatch: '+row['mesh']+'/'+key)
        frames = verified_json(report['frames'])
        if check[row['mesh_path_id']].read_typetree()['m_Shapes'] != float32_shapes(frames):
            raise ValueError('Original expression frame roundtrip mismatch')
        # Renderer and controllers were untouched raw serialized objects above.
        report['serialized_arrays_and_all_frames_match'] = True
    for old_name, digest in resource_hashes.items():
        if hashlib.sha256(restored.file.files[old_name.replace(old_cab,new_cab)].bytes).hexdigest() != digest:
            raise ValueError('Original texture/resource payload changed')
    out.mkdir(parents=True)
    output = out/'head.unity3d'
    output.write_bytes(encoded)
    save_json(out/'receipt.json', dict(format='native_mother_bundle_candidate_v1',
        inputs=source_file(inputs/'receipt.json'), bindings=source_file(bindings/'receipt.json'),
        code=source_file(Path(__file__)), bundle=source_file(output), prefab=prefab_name,
        native_donor=donor, archive_identity=new_cab, original_resource_payload_hashes=resource_hashes,
        original_object_count=len(objects), changed_object_ids=list(trees), renderer_outputs=reports,
        original_controllers_materials_textures_and_skin_preserved=True,
        registered_zipmod=False, installed=False, game_mutated=False, deliverable=False,
        pending=['Anatomical correspondence and complete geometry/intersection review',
                 'Actual BP interface shading contract', 'Native UV distribution metrics retained; updated engine metric handling required',
                 'Independent list/skin registration and native final acceptance',
                 'Full expression/gaze/slider range compatibility is not certified']))
    print(json.dumps(dict(output=str(output.resolve()), all_native_parts_and_frames_roundtrip=True,
                          untouched_objects_and_texture_payloads_exact=True, installed=False, deliverable=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build(args.inputs.resolve(), args.bindings.resolve(), args.out.resolve())
