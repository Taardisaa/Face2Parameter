"""Export official RenderDoc XML+ZIP into complete structured shader inputs.

Does not inject, launch, mutate or replay HS2. Every chunk/blob is retained.
State interpretation is a separate stage; unknown initial-context/update/deferred
branches are not discarded or treated as compatible draw state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

def digest(data):
    return hashlib.sha256(data).hexdigest()

def structured(node):
    row={'kind':node.tag,'attributes':node.attrib.copy()}
    if node.text and node.text.strip():
        row['value']=node.text.strip()
    if len(node):
        row['children']=[structured(child) for child in node]
    return row

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    raw=args.xml.read_bytes();root=ET.fromstring(raw)
    archive_path=args.xml.with_suffix('')
    if root.tag!='rdc' or root.find('header/driver').text!='D3D11':
        raise ValueError('Actual D3D11 RenderDoc XML required')
    args.out.mkdir(parents=True)
    blobs=[]
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.isdecimal() or '/' in name or '\\' in name:
                raise ValueError('Unexpected external data-buffer member '+name)
            data=archive.read(name)
            target='blob_'+name+'.bin'
            (args.out/target).write_bytes(data)
            blobs.append({'buffer_index':int(name),'path':target,'bytes':len(data),'sha256':digest(data)})
    chunks=root.find('chunks')
    blob_indices={b['buffer_index'] for b in blobs}
    records=[]
    for chunk in chunks:
        for node in chunk.iter('buffer'):
            if int(node.text) not in blob_indices:
                raise ValueError('Referenced shader/resource/update buffer missing from ZIP')
        records.append(structured(chunk))
    names=[r['attributes']['name'] for r in records]
    receipt={'schema_version':1,'source_xml':str(args.xml.resolve()),
        'source_xml_sha256':digest(raw),'source_zip':str(archive_path.resolve()),
        'source_zip_sha256':digest(archive_path.read_bytes()),
        'header':structured(root.find('header')),'chunk_format_version':chunks.attrib['version'],
        'chunks':records,'blobs':blobs,'chunk_count':len(records),'blob_count':len(blobs),
        'scope':'All actual D3D11 resource creation, bytecode, data buffers and command chunks preserved. No selected-draw or dynamic scene compatibility claim.',
        'pending_state_interpretation':[n for n in sorted(set(names)) if any(word in n for word in
            ('Beginning of Capture','CommandList','Map','Unmap','UpdateSubresource','CopyResource','Dispatch'))],
        'game_captured_by_this_command':False,'rgb_renderer_complete':False}
    (args.out/'manifest.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({key:receipt[key] for key in ('chunk_count','blob_count','pending_state_interpretation')}))

if __name__=='__main__':
    main()
