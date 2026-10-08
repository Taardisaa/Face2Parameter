"""Inspect an installed native base's chin/neck surfaces without game mutation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from src.hs2_assets import ChaList
from tools.model_bridge.artifact import sha
from tools.model_bridge.attachment_audit import body_asset
from tools.native_head.native_seam_audit import extract,aliases,audit_head
from tools.native_head.neck_geometry import ordered_loops,original_neck_boundary
from tools.native_head.neck_section_review import sections
from tools.native_head.posterior_review import shaded


def in_bone(points,matrix):
    return points@matrix[:3,:3].T+matrix[:3,3]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game',type=Path,default=Path(r'E:\HoneySelect2_ArcticFox'))
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--body-capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a new output folder')
    row=ChaList(ab_dir=str(args.game/'abdata')).resolve('fo_head',2)
    head_path=args.game/'abdata'/row['MainAB']
    head,names=extract(head_path.read_bytes(),row['MainData'])
    body_path=args.game/'abdata/chara/oo_base.unity3d'
    body,meta=body_asset(body_path)
    hm=head['bindpose'][names.index('cf_J_FaceRoot_s')].astype(float)
    bm=body['bindpose'][meta['skin_bones'].index('cf_J_Head_s')].astype(float)
    hv=in_bone(head['verts'],hm);bv=in_bone(body['verts'],bm)
    rim,_=original_neck_boundary(dict(vertices=body['verts'],triangles=body['faces'],
        bone_indices=body['bone_idx'],bone_weights=body['bone_w']),meta['skin_bones'])
    audit=audit_head(head,names,body,meta['skin_bones'],rim,bv)
    if len(audit['pure_root_boundary_rings'])!=1:raise ValueError('Ambiguous native interface')
    interface=audit['pure_root_boundary_rings'][0]
    ring=interface['head_boundary_ids'];matched=interface['body_matching_vertex_ids']
    if interface['max_distance_to_body_vertices']>1e-5 or not interface['matched_body_vertices_all_Head_s_weight_one']:
        raise ValueError('Native source interface changed')
    rows=bv[np.r_[rim,matched]]
    distances=np.linalg.norm(bv[:,None]-rows,axis=2).min(1)
    overlap=body['faces'][(distances[body['faces']]<1e-5).all(1)]
    outside=body['faces'][~(distances[body['faces']]<1e-5).all(1)]
    current=dict(np.load(args.current,allow_pickle=False))
    capture=json.loads(args.body_capture.read_text())
    mesh=next(m for m in capture['meshes'] if m['mesh_name']=='o_body_cf')
    cm=np.asarray(mesh['source']['bindposes'][mesh['bone_names'].index('cf_J_Head_s')]).reshape(4,4)
    cv=in_bone(np.asarray(mesh['source']['vertices']),cm)
    cf=np.asarray(mesh['source']['triangles']).reshape(-1,3)
    args.output.mkdir(parents=True)
    fig,axes=plt.subplots(1,3,figsize=(15,7))
    for ax in axes[:2]:
        ax.add_collection(LineCollection(sections(hv,head['faces']),colors='#356fa0',linewidths=2,label='Native female head 02'))
        ax.add_collection(LineCollection(sections(bv,outside),colors='#689660',linewidths=2,label='Native body'))
        ax.add_collection(LineCollection(sections(bv,overlap),colors='#689660',linewidths=1.5,linestyles='dashed',label='Body overlap inside head'))
        seam=hv[ring];front=seam[seam[:,2].argmax()]
        ax.scatter([front[2]],[front[1]],s=35,color='#a14187',zorder=5,label='Actual front interface')
        ax.set_aspect('equal');ax.grid(alpha=.3);ax.set_xlabel('Front / back');ax.set_ylabel('Height')
    axes[0].set_xlim(-1.1,1.4);axes[0].set_ylim(-.8,1.9);axes[0].set_title('Installed native base 02: bind geometry')
    axes[0].legend(loc='upper left',fontsize=8)
    axes[1].set_xlim(.25,1.25);axes[1].set_ylim(-.65,-.22);axes[1].set_title('Native chin to front neck')
    axes[2].add_collection(LineCollection(sections(current['vertices'],current['faces']),colors='#d66c28',linewidths=2,label='Current target head'))
    axes[2].add_collection(LineCollection(sections(cv,cf),colors='#777777',linewidths=2,label='Current BP body: bind geometry'))
    axes[2].add_collection(LineCollection(sections(hv,head['faces']),colors='#356fa0',linewidths=1,linestyles='dotted',label='Native head reference'))
    axes[2].set_xlim(.25,1.25);axes[2].set_ylim(-.65,-.22);axes[2].set_aspect('equal');axes[2].grid(alpha=.3)
    axes[2].set_xlabel('Front / back');axes[2].set_ylabel('Height');axes[2].set_title('Current target: anterior region unedited');axes[2].legend(fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'native_chin_sections.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(11,7))
    for i,angle in enumerate([0,35]):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        view_faces=body['faces'][(bv[body['faces'],1].max(1)>-.8)&(bv[body['faces'],1].min(1)<.3)]
        shaded(ax,bv,view_faces,[.64,.68,.64])
        # Highlight the real lower head surface, not a drawn Bezier substitute.
        region=(hv[head['faces'],1].max(1)<.15)&(hv[head['faces'],2].mean(1)>.1)
        shaded(ax,hv,head['faces'],[.75,.77,.81],region)
        ax.set_xlim(-1,1);ax.set_ylim(-1,1.4);ax.set_zlim(-.8,1.9);ax.set_box_aspect([2,2.4,2.7])
        ax.set_proj_type('ortho');ax.view_init(elev=0,azim=angle);ax.set_axis_off()
        ax.set_title('Native base 02: '+('side' if i==0 else 'front oblique'))
    fig.text(.5,.04,'Actual installed meshes in their coincident attachment frames; no face slider or fitted alignment.',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(args.output/'native_chin_3d.png',dpi=150);plt.close(fig)
    report=dict(format='native_chin_asset_review_v1',head_id=2,prefab=row['MainData'],
        head_bundle=str(head_path),head_bundle_sha256=sha(head_path),body_bundle=str(body_path),body_bundle_sha256=sha(body_path),
        current_geometry_sha256=sha(args.current),current_body_capture_sha256=sha(args.body_capture),
        coordinate_policy='Literal asset bind positions transformed by FaceRoot_s / Head_s inverse bindposes; no shape slider evaluation or fitted placement',
        current_body_policy='BP captured source bind mesh, not animated BakeMesh',
        native_interface=interface,game_mutations=False,current_geometry_modified=False)
    (args.output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(head_id=2,prefab=row['MainData'],output=str(args.output),current_geometry_modified=False)))


if __name__=='__main__':main()
