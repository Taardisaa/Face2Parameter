"""Static native-atlas evidence: chart correspondence, regions and donor pixels."""
import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import UnityPy

from tools.native_head.surface_regions import digest


def review(built):
    receipt = json.loads((built/'receipt.json').read_text(encoding='utf-8'))
    regions = json.loads((built/'surface_regions.json').read_text())
    native = UnityPy.load(receipt['skin_policy']['source_bundle'])
    with zipfile.ZipFile(built/'Chenger.MICA.NativeHead.zipmod') as archive:
        packed = UnityPy.load(archive.read('abdata/chara/codex/chenger/skin.unity3d'))
    def textures(env):
        return {o.read().m_Name: o.read() for o in env.objects if o.type.name == 'Texture2D'}
    original, copied = textures(native), textures(packed)
    checks = {}
    for name in ('cf_head_02_00_t',):
        a, b = np.asarray(original[name].image), np.asarray(copied[name].image)
        checks[name] = dict(original_pixels_unchanged=bool(np.array_equal(a, b)),
                           pixels_sha256=digest(a.tobytes()))
        if not checks[name]['original_pixels_unchanged']:
            raise ValueError('Native donor pixels were altered: '+name)
    atlas = np.asarray(original['cf_head_02_00_t'].image)
    with np.load(built/'atlas_uv.npz') as data:
        triangles = data['uv'][data['faces']]
    source = np.asarray(regions['atlas_correspondence']['source_uv'])
    donor = np.asarray(regions['atlas_correspondence']['donor_uv'])
    from scipy.spatial import Delaunay
    simplices = Delaunay(source).simplices
    def signed_area(p):
        a, b = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
        return a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    orientation = regions['atlas_correspondence']['chart_orientation']
    flipped = signed_area(source[simplices])*signed_area(donor[simplices])*orientation <= 0
    for name, expected in [('cf_head_02_00_n',[255,128,128,128]),('cf_head_02_00_o',[255,255,0,255])]:
        pixels = np.asarray(copied[name].image)
        checks[name] = dict(neutral_input=bool(np.all(pixels == expected)))
        if not checks[name]['neutral_input']:
            raise ValueError('Uncertified native normal/AO input enabled: '+name)
    checks['anchor_chart'] = dict(folded_or_degenerate_cells=int(flipped.sum()),
        interpretation='Authored atlas mapping cells; no anatomical accuracy claim')
    if flipped.any():
        checks['anchor_chart']['failed_cells'] = simplices[flipped].tolist()
    checks['regions'] = {k: len(v) for k, v in regions['integrated_regions'].items()}
    checks['topology_sha256'] = regions['topology_sha256']
    checks['geometry_scope'] = 'UV authoring only; verify_appearance_assets checks literal triangle-corner geometry'
    colors = dict(lip_vermilion='#ff2266',mouth_inner='#002aff',perioral_skin='#12e040',
                  left_eye_region='#04ddff',right_eye_region='#04ddff',
                  left_ear='#f5a000',right_ear='#f5a000',scalp='#a545ff')
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax in axes:
        ax.imshow(atlas, extent=(0, 1, 0, 1), origin='upper')
        for name, color in colors.items():
            ids = regions['integrated_regions'][name]
            ax.add_collection(PolyCollection(triangles[ids],facecolor='none',edgecolor=color,
                                            linewidth=.25,label=name))
        ax.scatter(donor[:, 0],donor[:, 1],s=4,c='white')
        ax.set_xlabel('native U'); ax.set_ylabel('native V')
    axes[0].set(xlim=(0,1),ylim=(0,1),title='Original donor pixels / mapped semantic faces')
    axes[0].legend(fontsize=7,loc='upper left')
    axes[1].set(xlim=(.38,.62),ylim=(.25,.4),title='Lip contour and perioral partition')
    fig.tight_layout(); fig.savefig(built/'atlas_regions.png',dpi=160); plt.close(fig)
    (built/'atlas_static_review.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(native_albedo_pixels_preserved=True,
        folded_or_degenerate_cells=int(flipped.sum()), regions=checks['regions'])))
    if flipped.any():
        raise ValueError('Authored UV anchors fold the source chart; see failed_cells')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--built',type=Path,required=True)
    review(parser.parse_args().built)
