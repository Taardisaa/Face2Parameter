"""Show the actual zero-shape FLAME template beside target and edited meshes.

Orthographic geometry inspection only; no inference, game mutation or renderer
development. Template comes from the exact decoder state of the selected photo.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.font_manager import FontProperties
import numpy as np

from tools.model_bridge.artifact import ModelArtifact, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    original=json.loads(args.source.read_text(encoding='utf-8'))
    candidate=json.loads(args.candidate.read_text(encoding='utf-8'))
    model=ModelArtifact.from_game_source(original['source'])
    template=model.state['v_template'];faces=model.faces
    vertices=np.asarray(original['vertices'])
    if not np.array_equal(faces,np.asarray(original['triangles']).reshape(-1,3)):
        raise ValueError('Original source topology is not the decoder topology')
    if not np.array_equal(model.mesh(head_local=True)[0],vertices):
        raise ValueError('Original source positions differ from raw selected output')
    meshes=[(template,faces),(vertices,faces),(np.asarray(candidate['vertices']),np.asarray(candidate['triangles']).reshape(-1,3))]
    font_path=Path('C:/Windows/Fonts/msyh.ttc')
    font=FontProperties(fname=str(font_path)) if font_path.exists() else None
    titles=['FLAME 默认头模\n形状、表情系数全为 0','程儿：照片推断的原始头模\n没有裁切或局部变形','当前修改版\n脖子被局部调整']
    # Identical source coordinates, scale and framing across every panel.
    all_v=np.vstack([v for v,f in meshes]);low=all_v.min(0);high=all_v.max(0)
    span=(high[1]-low[1])*1.1;cy=(high[1]+low[1])/2
    fig,axes=plt.subplots(2,3,figsize=(13.5,10.2),facecolor='#f1f1ee')
    for col,(v,f) in enumerate(meshes):
        tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-12)
        for row,(horizontal,depth) in enumerate([(0,2),(2,0)]):
            ax=axes[row,col];ax.set_facecolor('#e5e5e1')
            light=np.zeros(3);light[depth]=.8;light[1]=.5;light[horizontal]=-.35;light/=np.linalg.norm(light)
            shade=.35+.65*np.maximum(normal@light,0)
            rgb=np.array([.76,.70,.62])[None]*shade[:,None]
            visible=normal[:,depth]>-1e-9;order=np.argsort(tri[:,:,depth].mean(1));order=order[visible[order]]
            ax.add_collection(PolyCollection(tri[order][:,:,[horizontal,1]],facecolors=rgb[order],edgecolors='none',antialiased=False))
            cx=(high[horizontal]+low[horizontal])/2
            ax.set_xlim(cx-span*.43,cx+span*.43);ax.set_ylim(cy-span*.5,cy+span*.5)
            ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(titles[col],fontproperties=font,fontsize=13,pad=14)
            if col==0:ax.set_ylabel('正面' if row==0 else '侧面',fontproperties=font,fontsize=12)
    fig.text(.5,.02,'真实网格的正交示意 · 无贴图 · 相同比例与坐标 · 不是游戏截图',ha='center',fontproperties=font,fontsize=11)
    fig.subplots_adjust(left=.035,right=.98,top=.91,bottom=.06,wspace=.10,hspace=.03)
    path=args.out/'flame_template_and_chenger.png';fig.savefig(path,dpi=155);plt.close(fig)
    obj=args.out/'flame_zero_shape_template.obj'
    obj.write_text('\n'.join(['# Actual decoder v_template: zero shape/expression; complete uncropped topology']+
        ['v '+' '.join(repr(float(x)) for x in row) for row in template]+
        ['f '+' '.join(str(int(x)+1) for x in row) for row in faces])+'\n')
    report=dict(template='Actual v_template from original selected MICA decoder state',
        source_manifest=str(model.path),source_manifest_sha256=sha(model.path),
        original_source=dict(path=str(args.source.resolve()),sha256=sha(args.source)),
        candidate=dict(path=str(args.candidate.resolve()),sha256=sha(args.candidate)),
        template_vertices=len(template),template_faces=len(faces),template_parameters_zero=True,
        source_positions_original=True,shared_coordinates_scale=True,game_mutations=0,
        screenshot=False,inferred_new_model=False,image=str(path.resolve()),obj=str(obj.resolve()))
    (args.out/'receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':main()
