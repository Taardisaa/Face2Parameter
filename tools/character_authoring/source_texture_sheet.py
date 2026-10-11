"""Inspect native makeup source textures; this is not a game render or shader replica."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import UnityPy
from PIL import Image, ImageChops, ImageDraw

from src.hs2_assets import ChaList, CATEGORY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--category', choices=['st_eyebrow', 'st_lip', 'st_eyelash'], required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    lists = ChaList()
    rows = lists.by_cat[CATEGORY[args.category]]
    loaded = {}
    entries = []
    columns, cell_w, cell_h = 5, 240, 110
    sheet = Image.new('RGB', (columns*cell_w, ((len(rows)+columns-1)//columns)*cell_h), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (item_id, row) in enumerate(sorted(rows.items())):
        if row.get('MainAB') in ('0', '', None):
            x, y = (n % columns)*cell_w, (n // columns)*cell_h
            draw.text((x+6, y+4), f'{args.category} ID {item_id}: no texture', fill='black')
            entries.append(dict(id=item_id, source_row=row, status='source_has_no_bundle'))
            continue
        bundle = Path(lists.bundle_path(row))
        if bundle not in loaded:
            env = UnityPy.load(str(bundle))
            textures = {}
            for obj in env.objects:
                if obj.type.name == 'Texture2D':
                    texture = obj.read()
                    textures[texture.m_Name] = texture.image
            loaded[bundle] = textures
        raw = loaded[bundle][row['AddTex']].convert('RGBA')
        raw_path = args.out / f'{item_id}_raw.png'
        raw.save(raw_path)
        alpha = raw.getchannel('A')
        view = 'alpha' if alpha.getextrema()[0] != alpha.getextrema()[1] else 'RGB'
        im = alpha.convert('RGB') if view == 'alpha' else raw.convert('RGB')
        background = Image.new('RGB', im.size, im.getpixel((0, 0)))
        bbox = ImageChops.difference(im, background).getbbox()
        crop = im.crop(bbox) if bbox else im.copy()
        crop.thumbnail((cell_w-12, cell_h-28))
        x, y = (n % columns)*cell_w, (n // columns)*cell_h
        sheet.paste(crop, (x+6, y+24))
        draw.text((x+6, y+4), f'{args.category} ID {item_id} ({view})', fill='black')
        entries.append(dict(id=item_id, source_row=row, bundle=str(bundle),
                            bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
                            raw_path=str(raw_path), raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                            crop_bbox=bbox, diagnostic_channel=view))
    sheet.save(args.out / 'source_sheet.png')
    (args.out / 'manifest.json').write_text(json.dumps(dict(
        scope='Raw source RGBA preserved; cropped alpha/RGB inspection sheet; no shader, layout or game appearance claim',
        category=args.category, entries=entries), ensure_ascii=False, indent=2), encoding='utf-8')
    print(args.out / 'source_sheet.png')


if __name__ == '__main__':
    main()
