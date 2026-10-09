"""Source/asset follow-up for the installed neck shading gap; no game writes.

Use existing native geometry and material bindings. Keep extracted assemblies,
textures and full per-vertex records in an ignored output directory.
"""
import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _read_smr_mesh, _read_transforms, _chain

from tools.native_head.audit_neck_shading import (
    digest, material, point_texels, provenance, seam_pairs, shader_evidence,
    source_frame, texture,
)


def audit(capture, package, scalars, out):
    if out.exists():
        raise FileExistsError('Preserve earlier evidence; use a fresh directory')
    out.mkdir(parents=True)
    snapshot = json.loads(capture.read_text(encoding='utf-8-sig'))
    head, body = [next(m for m in snapshot['meshes'] if m['mesh_name'] == name)
                  for name in ('o_head', 'o_body_cf')]
    ring, nearest, seam = seam_pairs(head, body)
    body_draw = next(m for m in body['material_state']['materials']
                     if m['name'] == 'cf_m_skin_body_00')
    bindings = {t['property']: t for t in body_draw['textures']}
    for prop, name in (('_OcclusionMap', 'cf_body_00_o'), ('_BumpMap2', 'cf_body_00_n')):
        if bindings[prop]['is_null'] or bindings[prop]['name'] != name:
            raise ValueError('Unresolved body detail source: '+prop)
    root = Path('E:/HoneySelect2_ArcticFox/abdata/chara')
    base_path, detail_path = root/'mm_base.unity3d', root/'00/ft_detail_b_00.unity3d'
    base, detail = UnityPy.load(str(base_path)), UnityPy.load(str(detail_path))
    body_ao, body_normal2 = texture(detail, 'cf_body_00_o'), texture(detail, 'cf_body_00_n')
    native_body_material = material(base, 'cf_m_skin_body_00')
    normal_pointer = dict(native_body_material.read().m_SavedProperties.m_TexEnvs)['_BumpMap'].m_Texture
    external_path = native_body_material.assets_file.externals[normal_pointer.m_FileID-1].path
    if external_path.rsplit('/',1)[-1].lower() not in [k.lower() for k in detail.file.files]:
        raise ValueError('Body base-normal dependency does not point to installed detail bundle')
    body_normal = next(o.read() for o in detail.objects if o.path_id == normal_pointer.m_PathID)
    if bindings['_BumpMap']['is_null'] or bindings['_BumpMap']['name'] != body_normal.m_Name:
        raise ValueError('Captured body base normal does not match installed drawing asset')
    with zipfile.ZipFile(package) as archive:
        head_env = UnityPy.load(archive.read('abdata/chara/codex/chenger/head.unity3d'))
        skin_env = UnityPy.load(archive.read('abdata/chara/codex/chenger/skin.unity3d'))
    transforms, objects = _read_transforms(list(head_env.objects))
    smr = next(o.read() for o in head_env.objects if o.type.name == 'SkinnedMeshRenderer'
               and o.read().m_GameObject.read().m_Name == 'o_head'
               and 'p_cf_head_chenger_mica' in _chain(transforms,objects[o.read().m_GameObject.path_id]))
    packed_mesh = _read_smr_mesh(smr.m_Mesh.read())
    for packed, captured in (('verts','vertices'),('normals','normals'),('uv','uv'),
                             ('faces','triangles'),('bone_idx','bone_indices'),('bone_w','bone_weights')):
        source = np.asarray(head['source'][captured],packed_mesh[packed].dtype)
        if packed == 'faces':
            source = source.reshape(-1,3)
        if not np.array_equal(packed_mesh[packed],source):
            raise ValueError('Package/captured head source mismatch: '+packed)
    head_material = material(head_env, 'cf_m_skin_head_02')
    head_ao, head_normal = [texture(skin_env, name) for name in ('cf_head_02_00_o', 'cf_head_02_00_n')]
    fields = {}
    for name, asset, uv in (
        ('head_ao', head_ao, np.asarray(head['source']['uv'])[ring]),
        ('head_normal', head_normal, np.asarray(head['source']['uv'])[ring]),
        ('body_ao', body_ao, np.asarray(body['source']['uv'])[nearest]),
        ('body_base_normal', body_normal, np.asarray(body['source']['uv'])[nearest]),
        ('body_detail_normal', body_normal2, np.asarray(body['source']['uv'])[nearest]),
    ):
        fields[name] = dict(asset=asset.m_Name, width=asset.m_Width, height=asset.m_Height,
            texture_format=int(asset.m_TextureFormat), serialized_color_space=int(asset.m_ColorSpace),
            pixels_sha256=digest(np.asarray(asset.image).tobytes()),
            interface_source_rgba=point_texels(asset.image, uv).tolist())
    _, head_normals = source_frame(head, 'cf_J_FaceRoot_s')
    _, body_normals = source_frame(body, 'cf_J_Head_s')
    # Preserve all coincident UV copies: neither an arbitrary nearest UV nor a
    # raw tangent-space normal can be treated as a portable surface direction.
    aliases = []
    for h, candidates in zip(ring, seam['body_coincident_vertex_ids']):
        uv = np.asarray(body['source']['uv'])[candidates]
        aliases.append(dict(head_vertex=int(h), body_vertices=candidates, body_uv=uv.tolist(),
            body_ao_rgba=point_texels(body_ao.image, uv).tolist(),
            body_normal_rgba=point_texels(body_normal.image, uv).tolist(),
            body_normal2_rgba=point_texels(body_normal2.image, uv).tolist(),
            head_bind_normal=head_normals[h].tolist(), body_bind_normals=body_normals[candidates].tolist(),
            body_source_tangents=np.asarray(body['source']['tangents'])[candidates].tolist(),
            head_source_tangent=head['source']['tangents'][h]))
    face_create, body_create = [material(base, name).read() for name in ('create_skin_face', 'create_skin_body')]
    result = dict(format='native_neck_lighting_followup_v1', game_mutated=False,
        capture=provenance(capture), package=provenance(package),
        runtime_scalar_receipt=provenance(scalars), runtime_scalars=json.loads(scalars.read_text()),
        installed_bundles=[provenance(base_path), provenance(detail_path)], game=snapshot['game'],
        body_source_geometry_sha256=body['source_geometry_sha256'], interface=seam,
        source_fields=fields, interface_uv_aliases=aliases,
        skin_color_composition=dict(face_material=face_create.m_Name,body_material=body_create.m_Name,
            same_installed_shader_pointer=face_create.m_Shader.path_id == body_create.m_Shader.path_id,
            shader_name=face_create.m_Shader.read().m_ParsedForm.m_Name,
            scope='Shared shader asset; not proof that every layer input is equal'),
        shaders=dict(head=shader_evidence(head_material,out,'head'),
                     body=shader_evidence(native_body_material,out,'body')),
        limitations=['Static bind frames and level-0 raw asset texels; no GPU filtered color equality',
                     'Lighting variant not independently selected',
                     'Scalar readback is later than geometry capture and contains no overrides',
                     'Normal maps must be decoded, composed with actual strengths and re-expressed in destination TBN',
                     'No AO or normal continuity fix implemented by this read-only audit'])
    (out/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=str(out/'audit.json'),game_mutated=False,source_paths_resolved=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('capture', 'package', 'scalars', 'out'):
        parser.add_argument('--'+flag,type=Path,required=True)
    args = parser.parse_args()
    audit(args.capture,args.package,args.scalars,args.out)
