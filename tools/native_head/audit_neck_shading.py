"""Read-only neck material audit from installed assets and an existing capture.

No game calls, screenshot search, parameter sweep or fitted color correction.
Extracted textures and shader assembly must stay in an ignored output directory.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import UnityPy
from UnityPy.export.ShaderConverter import ShaderProgram
from UnityPy.helpers import CompressionHelper
from UnityPy.streams import EndianBinaryReader

from tools.native_head.neck_geometry import ordered_loops


def digest(data):
    return hashlib.sha256(data).hexdigest()


def provenance(path):
    return dict(path=str(path.resolve()), sha256=digest(path.read_bytes()))


def material(env, name):
    return next(o for o in env.objects if o.type.name == 'Material' and o.read().m_Name == name)


def texture(env, name):
    return next(o.read() for o in env.objects if o.type.name == 'Texture2D' and o.read().m_Name == name)


def disassemble(code):
    """Use the Windows SDK disassembler on the exact installed DXBC."""
    offset = code.index(b'DXBC')
    dxbc = code[offset:]
    dll = ctypes.WinDLL('d3dcompiler_47.dll')
    function = dll.D3DDisassemble
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint,
                        ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_long
    data = ctypes.create_string_buffer(dxbc)
    blob = ctypes.c_void_p()
    hr = function(data, len(dxbc), 0, None, ctypes.byref(blob))
    if hr < 0:
        raise RuntimeError(f'D3DDisassemble HRESULT {hr}')
    vtable = ctypes.cast(blob, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    try:
        pointer = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vtable[3])(blob)
        size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vtable[4])(blob)
        return ctypes.string_at(pointer, size).decode('utf-8').rstrip('\0'), digest(dxbc)
    finally:
        ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtable[2])(blob)


def shader_evidence(draw, out, prefix):
    shader = draw.read().m_Shader.read()
    tree = shader.object_reader.read_typetree()
    platform = list(shader.platforms).index(4)  # Installed D3D11 platform.
    entry = lambda field: field[platform][0] if isinstance(field[platform], list) else field[platform]
    offset, length, size = map(entry, (shader.offsets, shader.compressedLengths, shader.decompressedLengths))
    compressed = bytes(shader.compressedBlob)[offset:offset+length]
    program = ShaderProgram(EndianBinaryReader(CompressionHelper.decompress_lz4(compressed, size), endian='<'),
                            shader.object_reader.version)
    passes = tree['m_ParsedForm']['m_SubShaders'][0]['m_Passes']
    # Inspect the first directional forward branch, not every lighting variant.
    # The live variant is not selected or certified by this static audit.
    for p in passes:
        names = {value: key for key, value in p['m_NameIndices']}
        for sub in p['progFragment']['m_SubPrograms']:
            code = program.m_SubPrograms[sub['m_BlobIndex']]
            if code.m_Keywords != ['DIRECTIONAL']:
                continue
            assembly, bytecode_hash = disassemble(bytes(code.m_ProgramCode))
            path = out/(prefix+'_directional.asm')
            path.write_text(assembly, encoding='utf-8')
            return dict(name=shader.m_ParsedForm.m_Name, path_id=shader.object_reader.path_id,
                blob_index=sub['m_BlobIndex'], keywords=code.m_Keywords,
                dxbc_sha256=bytecode_hash, assembly=provenance(path),
                texture_bindings=[dict(name=names[t['m_NameIndex']], texture_register=t['m_Index'],
                                      sampler_register=t['m_SamplerIndex']) for t in sub['m_TextureParams']],
                constant_buffers=[dict(name=names[c['m_NameIndex']], vectors=[
                    dict(name=names[v['m_NameIndex']], byte_offset=v['m_Index'], dimension=v['m_Dim'])
                    for v in c['m_VectorParams']]) for c in sub['m_ConstantBuffers']],
                live_variant_verified=False)
    raise ValueError('No audited directional pixel branch in '+shader.m_ParsedForm.m_Name)


def point_texels(image, uv):
    """Nearest level-0 source texels; never claim filtered GPU sample equality."""
    pixels = np.asarray(image.convert('RGBA'))
    uv = np.asarray(uv)
    # UnityPy images have their first row at V=1. Explicit clamp is used here
    # because all audited interface UVs are inside the chart, not at wrap edges.
    x = np.clip(np.floor(uv[:, 0]*pixels.shape[1]).astype(int), 0, pixels.shape[1]-1)
    y = np.clip(pixels.shape[0]-1-np.floor(uv[:, 1]*pixels.shape[0]).astype(int), 0, pixels.shape[0]-1)
    return pixels[y, x]


def source_frame(mesh, bone):
    source = mesh['source']
    index = mesh['bone_names'].index(bone)
    # Unity bindpose maps renderer-local bind vertices into the bone's bind
    # frame. Its inverse would map in the opposite direction.
    matrix = np.asarray(source['bindposes'][index]).reshape(4, 4)
    points = np.asarray(source['vertices']) @ matrix[:3, :3].T + matrix[:3, 3]
    normals = np.asarray(source['normals']) @ np.linalg.inv(matrix[:3, :3])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def seam_pairs(head, body):
    hp, hn = source_frame(head, 'cf_J_FaceRoot_s')
    bp, bn = source_frame(body, 'cf_J_Head_s')
    source = head['source']
    ids = np.asarray(source['bone_indices'])
    weights = np.asarray(source['bone_weights'])
    root = head['bone_names'].index('cf_J_FaceRoot_s')
    aliases, unique = {}, {}
    for i, p in enumerate(hp):
        key = (tuple(np.round(p, 7)), tuple(ids[i]), tuple(weights[i]))
        unique.setdefault(key, i)
        aliases[i] = unique[key]
    root_weight = (weights*(ids == root)).sum(1)
    loops = ordered_loops(source['triangles'], aliases,
                          lambda a, b: root_weight[a] == 1 and root_weight[b] == 1)
    if len(loops) != 1:
        raise ValueError('Expected one pure root-weight head/body interface')
    ring = np.asarray(loops[0])
    tree = cKDTree(bp)
    distances, nearest = tree.query(hp[ring])
    if distances.max() > 1e-5:
        raise ValueError('Explicit source bone frames do not agree at the interface')
    candidates = tree.query_ball_point(hp[ring], 1e-5)
    # Compare every coincident body duplicate, not an arbitrary first UV copy.
    normal_gaps = [float(np.min(np.linalg.norm(bn[c]-hn[i], axis=1))) for i, c in zip(ring, candidates)]
    return ring, nearest, dict(head_vertex_ids=ring.tolist(), body_nearest_vertex_ids=nearest.tolist(),
        body_coincident_vertex_ids=[list(c) for c in candidates], source_frame_bones=['cf_J_FaceRoot_s', 'cf_J_Head_s'],
        position_distance=distances.tolist(), best_normal_vector_distance=normal_gaps,
        scope='Existing captured bind mesh in declared support-bone frames; not normal-map or rendered color equality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True, help='Native skin export manifest')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--game-root', type=Path, default=Path('E:/HoneySelect2_ArcticFox'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh directory to preserve earlier evidence')
    args.out.mkdir(parents=True)
    snapshot = json.loads(args.capture.read_text(encoding='utf-8-sig'))
    head = next(m for m in snapshot['meshes'] if m['mesh_name'] == 'o_head')
    body = next(m for m in snapshot['meshes'] if m['mesh_name'] == 'o_body_cf')
    ring, counterparts, seam = seam_pairs(head, body)
    head_uv = np.asarray(head['source']['uv'])[ring]
    body_uv = np.asarray(body['source']['uv'])[counterparts]
    manifest = json.loads(args.inputs.read_text(encoding='utf-8-sig'))
    inputs = {t['asset']: t for t in manifest['textures']}

    def exported(name):
        record = inputs[name]
        path = Path(record['output'])
        if digest(path.read_bytes()) != record['output_sha256']:
            raise ValueError('Changed native input '+name)
        return Image.open(path), record

    head_albedo, head_input = exported('cf_head_02_00_t')
    body_albedo, body_input = exported('cf_body_00_t')
    native_head_mask, mask_input = exported('cf_head_00_mask')
    head_color = point_texels(head_albedo, head_uv)
    body_color = point_texels(body_albedo, body_uv)
    body_bundle = args.game_root/'abdata/chara/mm_base.unity3d'
    body_env = UnityPy.load(str(body_bundle))
    body_draw = material(body_env, 'cf_m_skin_body_00')
    body_mask = dict(body_draw.read().m_SavedProperties.m_TexEnvs)['_NailMask'].m_Texture.read()
    body_mask.image.save(args.out/'body_region_mask.png')
    with zipfile.ZipFile(args.package) as archive:
        head_bytes = archive.read('abdata/chara/codex/chenger/head.unity3d')
        skin_bytes = archive.read('abdata/chara/codex/chenger/skin.unity3d')
    head_env = UnityPy.load(head_bytes)
    head_draw = material(head_env, 'cf_m_skin_head_02')
    current_mask = dict(head_draw.read().m_SavedProperties.m_TexEnvs)['_NailMask'].m_Texture.read()
    skin_env = UnityPy.load(skin_bytes)
    current_head_ao = texture(skin_env, 'cf_head_02_00_o')
    current_head_albedo = texture(skin_env, 'cf_head_02_00_t')
    if not np.array_equal(np.asarray(current_head_albedo.image), np.asarray(head_albedo)):
        raise ValueError('Package albedo does not match audited original head2 input')
    # AO comes from the current body detailId=0 list row, not its skinId row.
    detail_bundle = args.game_root/'abdata/chara/00/ft_detail_b_00.unity3d'
    detail_env = UnityPy.load(str(detail_bundle))
    body_ao = texture(detail_env, 'cf_body_00_o')
    rows = dict(head_uv=head_uv.tolist(), body_uv=body_uv.tolist(),
        raw_head_albedo=head_color.tolist(), raw_body_albedo=body_color.tolist(),
        raw_albedo_difference=(head_color[:, :3].astype(int)-body_color[:, :3]).tolist(),
        native_head_mask=point_texels(native_head_mask, head_uv).tolist(),
        native_body_mask=point_texels(body_mask.image, body_uv).tolist(),
        imported_head_mask=point_texels(current_mask.image, head_uv).tolist(),
        native_body_ao=point_texels(body_ao.image, body_uv).tolist(),
        imported_head_ao=point_texels(current_head_ao.image, head_uv).tolist(),
        sampling_scope='Source RGBA8 nearest level-0 texels, not GPU filtering/mipmap or calibrated rendered colors')
    bindings = {}
    for mesh in (head, body):
        draw = next(m for m in mesh['material_state']['materials'] if m['name'].startswith('cf_m_skin_'))
        bindings[mesh['mesh_name']] = dict(shader=draw['shader_name'], textures=[
            {k: t[k] for k in ('property', 'is_null', 'name', 'scale', 'offset') if k in t}
            for t in draw['textures']], source_geometry_sha256=mesh['source_geometry_sha256'])
    baselines = {}
    for name, draw in (('imported_head', head_draw), ('native_body', body_draw)):
        props = draw.read_typetree()['m_SavedProperties']
        wanted = ('_BumpScale', '_BumpScale2', '_DetailNormalMapScale', '_Riality',
                  '_Gloss', '_ExGloss', '_OcclusionStrength', '_Translucency',
                  '_TransDirect', '_TransShadow', '_TransScattering')
        baselines[name] = dict(floats={k: v for k, v in props['m_Floats'] if k in wanted},
            vectors={k: v for k, v in props['m_Colors'] if k in ('_Color', '_FresnelSetting')},
            scope='Serialized baseline only; game Change methods may replace these values')
    evidence = dict(format='native_neck_shading_audit_v1', game_mutated=False,
        snapshot=provenance(args.capture), package=provenance(args.package),
        package_head_bundle_sha256=digest(head_bytes), inputs_manifest=provenance(args.inputs),
        package_skin_bundle_sha256=digest(skin_bytes),
        package_albedo_pixels_match_native=True,
        source_texture_pixels_sha256=dict(body_mask=digest(np.asarray(body_mask.image).tobytes()),
            body_ao=digest(np.asarray(body_ao.image).tobytes()),
            imported_head_mask=digest(np.asarray(current_mask.image).tobytes())),
        native_input_provenance=[head_input, body_input, mask_input],
        body_base_bundle=provenance(body_bundle), body_detail_bundle=provenance(detail_bundle),
        game=snapshot['game'], interface=seam, source_texels=rows, captured_bindings=bindings,
        serialized_baselines=baselines,
        shaders=dict(head=shader_evidence(head_draw, args.out, 'head'),
                     body=shader_evidence(body_draw, args.out, 'body')),
        limitations=['No final image-error attribution or material fix',
                     'No runtime lighting-variant certification',
                     'No use of uncalibrated readback as source RGBA equality',
                     'Body normal-map/tangent-space contribution remains a separate source path'])
    path = args.out/'audit.json'
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(audit=str(path), game_mutated=False, source_shaders_disassembled=True)))


if __name__ == '__main__':
    main()
