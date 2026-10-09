"""Compare captured native and authored occipital shapes in one head-bone frame.

Uses source vertices, actual single-frame blendshapes, bone matrices and
bindposes; never selects an unverified BakeMesh scale hypothesis. Read-only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.native_head.accept_placement_asset import skin_world, matrix
from tools.native_head.neck_section_review import sections
from tools.native_head.mother_template_inputs import save_json, source_file


def posed(path):
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    mesh=next(m for m in data['meshes'] if m['mesh_name']=='o_head')
    if data['character']['shape_value_face']!=[.5]*59:
        raise ValueError('Head driver states differ from nominal reference')
    vertices=np.asarray(mesh['source']['vertices']).copy()
    for channel in mesh['blendshapes']:
        if not channel['current_weight']:
            continue
        if channel['frame_count']!=1 or channel['frames'][0]['weight']!=100:
            raise ValueError('Unimplemented blendshape frame contract')
        vertices+=np.asarray(channel['frames'][0]['delta_vertices'])*channel['current_weight']/100
    transforms={t['id']:t for t in data['transforms']}
    world=skin_world({**mesh,'source':{**mesh['source'],'vertices':vertices}},transforms)
    root=next(t for t in data['transforms'] if t['name']=='cf_J_Head_s')
    inverse=matrix(root['world_to_local'])
    points=world@inverse[:3,:3].T+inverse[:3,3]
    return points,np.asarray(mesh['source']['triangles']).reshape(-1,3),data


def review(native,before,current,out):
    if out.exists():
        raise FileExistsError('Use a fresh output')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    rows=[(native,'Original native head 02','#3276b8'),
          (before,'Before local occipital edit (v4)','#aaaaaa'),
          (current,'Current (v5)','#dc702b')]
    meshes=[(posed(path),label,color) for path,label,color in rows]
    out.mkdir(parents=True)
    fig,axes=plt.subplots(1,3,figsize=(15,7))
    for ax,offset in zip(axes,[0.,0.,.35]):
        for (v,f,data),label,color in meshes:
            local=v.copy();local[:,0]-=offset
            ax.add_collection(LineCollection(sections(local,f),colors=color,
                linewidths=1.8,label=label))
        ax.set_aspect('equal');ax.grid(alpha=.3)
        ax.set_xlabel('Front/back (front positive)');ax.set_ylabel('Height')
    axes[0].set_xlim(-1.25,1.25);axes[0].set_ylim(-.8,1.9)
    axes[0].set_title('Complete head: common native neck frame');axes[0].legend(fontsize=8)
    for ax in axes[1:]:
        ax.set_xlim(-1.25,-.15);ax.set_ylim(-.15,1.9)
    axes[1].set_title('Posterior midline: height/depth distribution')
    axes[2].set_title('Side section: x = +0.35')
    fig.tight_layout();fig.savefig(out/'occipital_reference_sections.png',dpi=145);plt.close(fig)
    save_json(out/'receipt.json',dict(
        inputs=[source_file(p) for p in (native,before,current)],code=source_file(Path(__file__)),
        skinning_code=source_file(Path('tools/native_head/accept_placement_asset.py')),
        source_single_frame_blendshapes_and_actual_bones=True,
        common_frame='cf_J_Head_s',head_scale_or_ear_alignment_applied=False,
        game_mutated=False,scope='Shape distribution review; no identity or anatomical truth claim'))
    print(out/'occipital_reference_sections.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native','before','current','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    review(a.native.resolve(),a.before.resolve(),a.current.resolve(),a.out.resolve())
