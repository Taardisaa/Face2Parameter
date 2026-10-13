"""Fetch pinned official replay headers for CPU compilation; never load RenderDoc.

Portable DLL and headers must have the same release. Licenses and source hashes
are retained in the caller's fresh ignored output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version',required=True)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if not re.fullmatch(r'\d+\.\d+',a.version):raise ValueError('Explicit official release number required')
    if a.out.exists():raise FileExistsError('Fresh header directory required')
    a.out.mkdir(parents=True);queue=['renderdoc_replay.h'];seen=set();sources=[]
    while queue:
        name=queue.pop(0)
        if name in seen:continue
        if not re.fullmatch(r'[A-Za-z0-9_]+\.h',name):raise ValueError('Unexpected external header path')
        seen.add(name)
        url=f'https://raw.githubusercontent.com/baldurk/renderdoc/v{a.version}/renderdoc/api/replay/{name}'
        data=urllib.request.urlopen(url,timeout=30).read();(a.out/name).write_bytes(data)
        sources.append({'path':name,'url':url,'sha256':hashlib.sha256(data).hexdigest()})
        queue.extend(re.findall(r'^#include "([^"/]+)"',data.decode(),re.M))
    exports=['RENDERDOC_AllocArrayMem','RENDERDOC_FreeArrayMem','RENDERDOC_CreateTargetControl',
        'RENDERDOC_EnumerateRemoteTargets','RENDERDOC_InitialiseReplay','RENDERDOC_ShutdownReplay']
    (a.out/'renderdoc.def').write_text('LIBRARY renderdoc.dll\nEXPORTS\n'+'\n'.join(exports)+'\n')
    (a.out/'provenance.json').write_text(json.dumps({'release':a.version,'headers':sources,
        'scope':'Download only; no DLL loading, graphics, injection, launch or capture'},indent=2))
    print(json.dumps({'headers':len(sources),'release':a.version,'out':str(a.out)}))
if __name__=='__main__':main()
