"""Prepare a pinned official portable RenderDoc; never inject/launch a game."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if not all(c.isdecimal() or c=='.' for c in args.version):
        raise ValueError('Explicit numeric official version required')
    if args.out.exists():
        raise FileExistsError('Fresh portable tool directory required')
    args.out.mkdir(parents=True)
    url=f'https://renderdoc.org/stable/{args.version}/RenderDoc_{args.version}_64.zip'
    target=args.out/Path(url).name
    urllib.request.urlretrieve(url,target)
    extracted=args.out/'extracted'
    with zipfile.ZipFile(target) as archive:
        base=extracted.resolve()
        for name in archive.namelist():
            if not (base/name).resolve().is_relative_to(base):
                raise ValueError('Archive member escapes extraction root')
        archive.extractall(extracted)
    receipt={'url':url,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
             'bytes':target.stat().st_size,
             'official_release':f'https://github.com/baldurk/renderdoc/releases/tag/v{args.version}',
             'scope':'Portable tool extraction only; no game launch/injection, driver registration or global hook.'}
    (args.out/'download_provenance.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))

if __name__=='__main__':
    main()
