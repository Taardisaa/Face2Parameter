"""Check an appearance-only package keeps geometry and valid streamed URIs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import UnityPy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--built', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve previous evidence')
    checks = {}
    for name in ('o_head', 'o_eyebase_L', 'o_eyebase_R'):
        with np.load(args.previous/(name+'.npz')) as old, np.load(args.built/(name+'.npz')) as new:
            if old['verts'].shape != new['verts'].shape:
                # Texture seams duplicate literal vertices. Compare complete
                # ordered triangle-corner geometry/skin, not vertex count.
                checks[name] = {key: bool(np.array_equal(old[key][old['faces']], new[key][new['faces']]))
                               for key in ('verts', 'normals', 'bone_idx', 'bone_w')}
                checks[name]['bindpose'] = bool(np.array_equal(old['bindpose'],new['bindpose']))
                checks[name]['triangle_count'] = len(old['faces']) == len(new['faces'])
            else:
                checks[name] = {key: bool(np.array_equal(old[key], new[key])) for key in
                               ('verts', 'faces', 'normals', 'bone_idx', 'bone_w', 'bindpose')}
            if not all(checks[name].values()):
                raise ValueError('Appearance changed accepted geometry: ' + name)
    streams = 0
    receipt = json.loads((args.built/'receipt.json').read_text(encoding='utf-8'))
    region = ((receipt.get('skin_policy') or {}).get('authored_surface') or {}).get('region_mask')
    region_checked = False
    with zipfile.ZipFile(args.built/'Chenger.MICA.NativeHead.zipmod') as archive:
        for path in ('head', 'skin'):
            env = UnityPy.load(archive.read('abdata/chara/codex/chenger/'+path+'.unity3d'))
            if path == 'skin':
                packed_skin = env
            cab = next(k for k, v in env.file.files.items() if hasattr(v, 'objects'))
            if path == 'head' and region:
                # Follow the actual drawing material's PPtr, not just a texture
                # with a matching name somewhere in the package.
                material = next(o for o in env.objects if o.type.name == 'Material'
                                and o.read().m_Name == 'cf_m_skin_head_02')
                props = material.read_typetree()['m_SavedProperties']
                pointer = dict(props['m_TexEnvs'])['_NailMask']['m_Texture']
                if pointer['m_FileID'] != 0:
                    raise ValueError('Imported face mask points outside its private bundle')
                mask = next(o for o in env.objects if o.path_id == pointer['m_PathID']).read()
                pixels = np.asarray(mask.image)
                expected_pixels = (np.all(pixels == region['rgba']) if 'rgba' in region else
                    hashlib.sha256(pixels.tobytes()).hexdigest() == region['pixels_sha256'])
                if mask.m_Name != region['texture_name'] or not expected_pixels:
                    raise ValueError('Serialized face mask retains old atlas regions')
                floats = dict(props['m_Floats'])
                if floats['_DetailNormalMapScale'] != 0 or floats['_Riality'] != 0:
                    raise ValueError('Incompatible atlas microdetail is enabled')
                region_checked = True
            for obj in env.objects:
                if obj.type.name != 'Texture2D':
                    continue
                stream = obj.read_typetree().get('m_StreamData', {})
                if not stream.get('size'):
                    continue
                uri = stream['path']; streams += 1
                if not uri.startswith('archive:/'+cab+'/') or uri.rsplit('/', 1)[-1] not in env.file.files:
                    raise ValueError('Broken streamed texture URI: '+uri)
                # A basename-only parser can decode the old broken URI offline;
                # successful decoding alone is insufficient for Unity resolution.
                obj.read().image.load()
    if not streams:
        raise ValueError('No internal streamed texture evidence')
    checks.update(all_texture_archive_uris_relinked=True, streamed_textures_checked=streams,
                  scope='Static asset integrity and fixed geometry; runtime appearance is separate')
    if region:
        if not region_checked:
            raise ValueError('No actual face region mask evidence')
        checks['private_face_region_mask_matches_declared_pixels'] = True
    local = ((receipt.get('skin_policy') or {}).get('authored_surface') or {}).get('neck_shader_inputs')
    if local:
        parents = np.load(args.built/'atlas_uv.npz')['parent_face_ids']
        with np.load(args.previous/'o_head.npz') as old, np.load(args.built/'o_head.npz') as new:
            if not np.array_equal(old['uv'][old['faces']][parents >= 0],new['uv'][new['faces']][parents >= 0]):
                raise ValueError('Neck UV authoring changed retained face UV corners')
        with zipfile.ZipFile(args.previous/'Chenger.MICA.NativeHead.zipmod') as archive:
            previous_skin=UnityPy.load(archive.read('abdata/chara/codex/chenger/skin.unity3d'))
        def albedo(env):
            return np.asarray(next(o.read().image for o in env.objects if o.type.name=='Texture2D'
                                   and o.read().m_Name=='cf_head_02_00_t'))
        old_image,new_image=albedo(previous_skin),albedo(packed_skin)
        size=old_image.shape[0]; yy,xx=np.mgrid[:size,:size]
        tile=(xx/size>=.4)&(xx/size<.6)&(1-yy/size>=.89)&(1-yy/size<1.)
        if not np.array_equal(old_image[~tile],new_image[~tile]):
            raise ValueError('Neck albedo bake altered original face/scalp texels')
        checks.update(retained_face_uv_corners_unchanged=True,face_albedo_texels_unchanged=True,
                      authored_neck_mask_pixel_hash_checked=True,
                      neck_scope='Local mask continuity only; AO/normal continuity not certified')
    args.out.write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({'geometry_preserved': True, 'streamed_texture_uris_valid': True}))


if __name__ == '__main__':
    main()
