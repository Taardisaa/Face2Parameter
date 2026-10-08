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
    with zipfile.ZipFile(args.built/'Chenger.MICA.NativeHead.zipmod') as archive:
        for path in ('head', 'skin'):
            env = UnityPy.load(archive.read('abdata/chara/codex/chenger/'+path+'.unity3d'))
            cab = next(k for k, v in env.file.files.items() if hasattr(v, 'objects'))
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
    args.out.write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({'geometry_preserved': True, 'streamed_texture_uris_valid': True}))


if __name__ == '__main__':
    main()
