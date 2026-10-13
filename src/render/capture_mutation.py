"""Exact, copy-on-write recorded data replacement for official full-driver replay.

This is a byte-transfer primitive, not a completed HS2 geometry correspondence.
Every original command/unknown blob remains; explicit source witnesses bind each
patch to one buffer occurrence. Reusing a blob index does not authorize changing
every reference to it. Candidate dependency completeness stays pending.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

class MutationRejected(ValueError):
    pass

def digest(data):return hashlib.sha256(data).hexdigest()

# Actual v1.46 xml_codec.cpp literal/thumbnail archive entries, kept byte-exact.
AUXILIARY_MEMBERS={'thumb.jpg','thumb.png','thumb.raw','ext_thumb.jpg','ext_thumb.png','ext_thumb.raw','diagnostic.log','D3D12Core.dll','D3D12SDKLayers.dll'}

def mutate_capture(source_xml: Path, plan_path: Path, output_xml: Path):
    source_xml=Path(source_xml);output_xml=Path(output_xml);plan_path=Path(plan_path)
    output_zip=output_xml.with_suffix('');source_zip=source_xml.with_suffix('')
    if output_xml.suffix!='.xml' or output_xml==output_zip:raise MutationRejected('Official XML plus external ZIP pathname required')
    if output_xml.exists() or output_zip.exists():raise MutationRejected('Fresh XML and ZIP outputs required')
    if not output_xml.parent.is_dir():raise MutationRejected('Existing output parent required')
    plan=json.loads(plan_path.read_text(encoding='utf-8-sig'))
    raw=source_xml.read_bytes();zip_bytes=source_zip.read_bytes()
    if digest(raw)!=plan['source_xml_sha256'] or digest(zip_bytes)!=plan['source_zip_sha256']:raise MutationRejected('Original capture provenance changed')
    tree=ET.fromstring(raw);original=deepcopy(tree)
    if tree.tag!='rdc' or tree.findtext('header/driver')!='D3D11':raise MutationRejected('Actual D3D11 XML required')
    chunks={}
    for chunk in tree.find('chunks'):
        index=int(chunk.attrib['chunkIndex'])
        if index in chunks:raise MutationRejected('Duplicate original chunk index')
        chunks[index]=chunk
    with zipfile.ZipFile(source_zip) as archive:
        infos=archive.infolist();blobs={};members={}
        for info in infos:
            if info.filename in members or (not info.filename.isdecimal() and info.filename not in AUXILIARY_MEMBERS):raise MutationRejected('Unexpected/duplicate external data member')
            members[info.filename]=archive.read(info)
            if info.filename.isdecimal():
                index=int(info.filename)
                if index in blobs:raise MutationRejected('Duplicate numeric external data index')
                blobs[index]=members[info.filename]
    original_blob_count=len(blobs)
    for node in tree.iter('buffer'):
        key=int(node.text.strip())
        if key not in blobs or int(node.attrib['byteLength'])!=len(blobs[key]):raise MutationRejected('Original reference/byteLength disagrees with actual ZIP')
    patches=plan['patches']
    if not patches:raise MutationRejected('Explicit source-derived byte patches required')
    selected={};records=[];next_index=max(map(int,blobs),default=-1)+1
    for patch in patches:
        chunk=chunks.get(patch['chunk_index'])
        if chunk is None or chunk.attrib['name']!=patch['chunk_name']:raise MutationRejected('Original command selector mismatch')
        refs=list(chunk.iter('buffer'));ordinal=patch['buffer_ordinal']
        if not isinstance(ordinal,int) or not 0<=ordinal<len(refs):raise MutationRejected('Buffer occurrence selector out of bounds')
        node=refs[ordinal];old_key=int(node.text.strip())
        if old_key!=patch['expected_buffer_index'] or digest(blobs[old_key])!=patch['expected_blob_sha256']:raise MutationRejected('Original selected blob mismatch')
        if not patch.get('source_witness') or not patch.get('semantic_role'):raise MutationRejected('Source identity and semantic role required; never guessed')
        replacement=(plan_path.parent/patch['replacement_file']).read_bytes()
        if digest(replacement)!=patch['replacement_sha256']:raise MutationRejected('Candidate source bytes changed')
        offset=patch['byte_offset'];end=offset+len(replacement)
        if not isinstance(offset,int) or offset<0 or not replacement or end>len(blobs[old_key]):raise MutationRejected('Exact source byte patch range invalid')
        if digest(blobs[old_key][offset:end])!=patch['expected_range_sha256']:raise MutationRejected('Original selected byte range changed')
        key=(patch['chunk_index'],ordinal)
        if key not in selected:
            selected[key]={'node':node,'old_key':old_key,'old_text':node.text,'bytes':bytearray(blobs[old_key]),'ranges':[],'new_key':next_index};next_index+=1
        state=selected[key]
        if any(offset<last_end and end>last_start for last_start,last_end in state['ranges']):raise MutationRejected('Overlapping patches on one command occurrence')
        state['bytes'][offset:end]=replacement;state['ranges'].append((offset,end))
        records.append({k:patch[k] for k in ('chunk_index','chunk_name','buffer_ordinal','expected_buffer_index','byte_offset','source_witness','semantic_role','replacement_sha256')})
        records[-1].update({'replacement_bytes':len(replacement),'new_buffer_index':int(state['new_key'])})
    for state in selected.values():
        state['node'].text=str(state['new_key']);blobs[state['new_key']]=bytes(state['bytes'])
    # Only selected buffer indices changed in XML; commands, lengths and unknown
    # fields are semantically identical. Numeric buffer indices are SD references,
    # not raw command resource IDs; serializer reconstructs the original commands.
    restored=deepcopy(tree)
    restored_chunks={int(c.attrib['chunkIndex']):c for c in restored.find('chunks')}
    for (index,ordinal),state in selected.items():list(restored_chunks[index].iter('buffer'))[ordinal].text=state['old_text']
    if ET.tostring(restored)!=ET.tostring(original):raise MutationRejected('Unrequested structured command mutation')
    with zipfile.ZipFile(output_zip,'w') as archive:
        for info in infos:archive.writestr(info,members[info.filename])
        for state in selected.values():archive.writestr(f"{state['new_key']:06d}",blobs[state['new_key']])
    output_xml.write_bytes(ET.tostring(tree,encoding='utf-8',xml_declaration=True))
    result={'schema_version':1,'scope':'Exact recorded byte-transfer primitive; candidate dependency coverage not yet accepted',
      'source_xml_sha256':digest(raw),'source_zip_sha256':digest(zip_bytes),'plan_sha256':digest(plan_path.read_bytes()),
      'output_xml':str(output_xml.resolve()),'output_xml_sha256':digest(output_xml.read_bytes()),
      'output_zip':str(output_zip.resolve()),'output_zip_sha256':digest(output_zip.read_bytes()),
      'original_chunks_preserved':len(chunks),'original_blobs_preserved':original_blob_count,'original_archive_members_preserved':len(infos),'copy_on_write_occurrences':len(selected),
      'patches':records,'candidate_geometry_complete':False,'rgb_renderer_complete':False,
      'remaining_dependencies':['actual all10 surface-to-resource correspondence','every dynamic/previous geometry stream','shadow/motion/skinning/cache inputs','CPU culling/command visibility invariance','full camera/material control and native RGB agreement']}
    return result
