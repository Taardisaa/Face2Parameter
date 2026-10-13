"""CPU byte transfer on a pre-existing independent capture; no driver execution."""
import argparse
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.render.capture_mutation import mutate_capture,digest

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml',required=True,type=Path);ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Fresh CPU receipt directory required')
    tree=ET.parse(a.xml).getroot()
    candidates=[c for c in tree.find('chunks') if c.attrib['name']=='ID3D11Device::CreateBuffer' and c.find(".//enum[@name='BindFlags']").text=='1']
    if len(candidates)!=1:raise ValueError('This diagnostic receipt requires the existing one-VB independent fixture')
    chunk=candidates[0];refs=list(chunk.iter('buffer'));node=chunk.find("buffer[@name='InitialData']")
    index=int(node.text)
    with zipfile.ZipFile(a.xml.with_suffix('')) as z:
        member=next(n for n in z.namelist() if n.isdecimal() and int(n)==index);old=z.read(member)
    value=struct.unpack_from('<f',old)[0];replacement=struct.pack('<f',value+0.125)
    a.out.mkdir(parents=True);(a.out/'replacement.bin').write_bytes(replacement)
    plan={'source_xml_sha256':digest(a.xml.read_bytes()),'source_zip_sha256':digest(a.xml.with_suffix('').read_bytes()),'patches':[{
      'chunk_index':int(chunk.attrib['chunkIndex']),'chunk_name':chunk.attrib['name'],'buffer_ordinal':refs.index(node),
      'expected_buffer_index':index,'expected_blob_sha256':digest(old),'byte_offset':0,'expected_range_sha256':digest(old[:4]),
      'replacement_file':'replacement.bin','replacement_sha256':digest(replacement),
      'semantic_role':'isolated diagnostic independent fixture first vertex float; not HS2 fitting',
      'source_witness':'pre-existing recorded immutable VB CreateBuffer ByteWidth/payload provenance'}]}
    path=a.out/'plan.json';path.write_text(json.dumps(plan,indent=2))
    out=a.out/'mutated.zip.xml';result=mutate_capture(a.xml,path,out)
    with zipfile.ZipFile(a.xml.with_suffix('')) as original,zipfile.ZipFile(out.with_suffix('')) as patched:
        checks=[]
        for member in original.namelist():
            checks.append({'member':member,'bytes':len(original.read(member)),'exact':original.read(member)==patched.read(member)})
        if not all(c['exact'] for c in checks):raise AssertionError('Original capture member changed')
        new=str(result['patches'][0]['new_buffer_index']).zfill(6)
        if patched.read(new)!=replacement+old[4:]:raise AssertionError('Source-derived exact byte transfer failed')
    result.update({'actual_original_members_checked':checks,'changed_vb_first_word_exact':True,
      'native_roundtrip_replay_accepted':False,'scope':'CPU mutation of pre-existing independent capture; no HS2 candidate/rendering acceptance'})
    (a.out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
