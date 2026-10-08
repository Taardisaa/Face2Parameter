"""Check an appearance-only package keeps geometry and valid streamed URIs."""
import argparse
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
                if mask.m_Name != region['texture_name'] or not np.all(pixels == region['rgba']):
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
        checks['private_face_region_mask_uniform_skin'] = True
    args.out.write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({'geometry_preserved': True, 'streamed_texture_uris_valid': True}))


if __name__ == '__main__':
    main()
