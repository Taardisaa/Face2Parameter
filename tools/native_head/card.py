"""Author standard Sideloader GUID references in a native card, no overlay JSON."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import msgpack

from src.face_data_utils.chara_loader.AiSyoujyoCharaData import AiSyoujyoCharaData
from tools.native_head.build import GUID,SLOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--card',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Preserve previous cards; use a fresh path')
    card=AiSyoujyoCharaData.load(str(args.card),True)
    card.Custom['face']['headId']=SLOT;card.Custom['face']['skinId']=SLOT
    card.Custom['face']['shapeValueFace']=[.5]*59
    face=card.Custom['face'];face['detailPower']=0.
    # These pre-existing cosmetics belong to vanilla UVs. The neutral asset
    # explicitly starts without them; no claim of source makeup reconstruction.
    for name,value in face['makeup'].items():
        if name.lower().endswith('color') and isinstance(value,list) and len(value)==4:value[3]=0.
    face['eyebrowColor'][3]=0.;face['moleColor'][3]=0.
    card.Parameter['fullname']='程儿_MICA原生底模'
    key='com.bepis.sideloader.universalautoresolver'
    version,data=copy.deepcopy(card.KKEx.data.get(key,[0,{}]))
    info=[]
    for entry in data.get('info',[]):
        value=msgpack.unpackb(entry,raw=False,strict_map_key=False)
        if value['Property'] not in ['ChaFileFace.headId','ChaFileFace.skinId']:info.append(entry)
    for category,property in [(210,'headId'),(211,'skinId')]:
        value=dict(ModID=GUID,Slot=SLOT,LocalSlot=SLOT,Property='ChaFileFace.'+property,
                   CategoryNo=category,Author='Codex',Website='',Name='Chenger MICA native head')
        info.append(msgpack.packb(value,use_bin_type=True))
    data['info']=info;card.KKEx[key]=[version,data]
    # Remove the key altogether: the bridge validates even empty/unknown-version
    # records during card-load preflight. Preserve every other original wire value.
    old_key='com.hs2mod.mcpbridge.sourcehead'
    if old_key in card.KKEx.data:
        original=card.KKEx._raw;unpacker=msgpack.Unpacker(raw=False,strict_map_key=False)
        unpacker.feed(original);count=unpacker.read_map_header();entries=[]
        for _ in range(count):
            start=unpacker.tell();wire_key=unpacker.unpack();unpacker.unpack();end=unpacker.tell()
            if wire_key!=old_key:entries.append(original[start:end])
        card.KKEx._raw=msgpack.Packer().pack_map_header(len(entries))+b''.join(entries)
        card.KKEx.data.pop(old_key)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(bytes(card))
    reread=AiSyoujyoCharaData.load(str(args.out),True)
    if reread.Custom['face']['headId']!=SLOT or reread.KKEx[key][1]['info']!=info:
        raise ValueError('Native card reference roundtrip failed')
    print(str(args.out.resolve()))


if __name__=='__main__':main()
