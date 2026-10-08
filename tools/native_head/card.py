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
    # Clearing this one old plugin key prevents the old display prototype from
    # rebuilding itself after native head load. Other plugin wire bytes survive.
    if 'com.hs2mod.mcpbridge.sourcehead' in card.KKEx.data:
        card.KKEx['com.hs2mod.mcpbridge.sourcehead']=[0,{}]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_bytes(bytes(card))
    reread=AiSyoujyoCharaData.load(str(args.out),True)
    if reread.Custom['face']['headId']!=SLOT or reread.KKEx[key][1]['info']!=info:
        raise ValueError('Native card reference roundtrip failed')
    print(str(args.out.resolve()))


if __name__=='__main__':main()
