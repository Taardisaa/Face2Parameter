"""Create a new card selecting a separately registered native head by GUID.

Preserves the source card's other native/plugin sections. No runtime source-head
record is allowed; the complete head must be loaded by the native asset path.
"""
import argparse
import copy
import json
from pathlib import Path

import msgpack

from src.face_data_utils.chara_loader.AiSyoujyoCharaData import AiSyoujyoCharaData
from tools.native_head.mother_template_inputs import save_json, source_file


def create(source, registration, out, title):
    if out.exists():
        raise FileExistsError('Preserve existing card')
    record=json.loads(registration.read_text())
    card=AiSyoujyoCharaData.load(str(source),True)
    if 'com.hs2mod.mcpbridge.sourcehead' in card.KKEx.data:
        raise ValueError('Select a real native source card without a diagnostic overlay record')
    face=card.Custom['face']
    face['headId']=record['original_head_slot']
    face['skinId']=record['original_skin_slot']
    face['shapeValueFace']=[.5]*59
    card.Parameter['fullname']=title
    key='com.bepis.sideloader.universalautoresolver'
    version,data=copy.deepcopy(card.KKEx.data.get(key,[0,{}]))
    entries=[]
    for entry in data.get('info',[]):
        row=msgpack.unpackb(entry,raw=False,strict_map_key=False)
        if row['Property'] not in ('ChaFileFace.headId','ChaFileFace.skinId'):
            entries.append(entry)
    for category,property,slot in ((210,'headId',face['headId']),(211,'skinId',face['skinId'])):
        entries.append(msgpack.packb(dict(ModID=record['guid'],Slot=slot,LocalSlot=slot,
            Property='ChaFileFace.'+property,CategoryNo=category,Author='Codex',Website='',Name=title),use_bin_type=True))
    data['info']=entries
    card.KKEx[key]=[version,data]
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_bytes(bytes(card))
    restored=AiSyoujyoCharaData.load(str(out),True)
    if restored.KKEx[key][1]['info']!=entries or restored.Custom['face']['headId']!=face['headId']:
        raise ValueError('Native card GUID references did not roundtrip')
    save_json(out.with_suffix('.receipt.json'),dict(format='native_mother_card_v1',
        source=source_file(source),registration=source_file(registration),card=source_file(out),
        guid=record['guid'],native_face_values=[.5]*59,source_head_overlay=False,
        appearance_and_other_plugin_data_preserved=True,game_reload_verified=False))
    print(str(out.resolve()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--title',required=True)
    a=p.parse_args()
    create(a.source.resolve(),a.registration.resolve(),a.out.resolve(),a.title)
