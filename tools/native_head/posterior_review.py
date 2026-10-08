"""Render an honest before/after section and shaded view of actual mesh arrays."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from tools.native_head.neck_section_review import sections


def body_geometry(capture):
    mesh=next(m for m in capture['meshes'] if m['mesh_name']=='o_body_cf')
    transform=next(t for t in capture['transforms'] if t['name']=='cf_J_Head_s')
    if mesh['renderer_lossy_scale'] != [1.,1.,1.]:raise ValueError('Unit captured renderer required')
    world=np.c_[mesh['baked']['vertices'],np.ones(mesh['vertex_count'])]@np.asarray(mesh['renderer_local_to_world']).reshape(4,4).T
    v=(world@np.asarray(transform['world_to_local']).reshape(4,4).T)[:,:3]
    return v,np.asarray(mesh['source']['triangles']).reshape(-1,3)


def shaded(ax,v,f,colour,changed=None):
    tri=v[f][:,:,[0,2,1]]
    n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-20)
    light=np.array([.55,-.3,.8]);light/=np.linalg.norm(light)
    intensity=.65+.35*np.maximum(n@light,0)
    colours=np.tile(colour,(len(f),1))
    if changed is not None:colours[changed]=[.85,.56,.28]
    collection=Poly3DCollection(tri,facecolors=colours*intensity[:,None],edgecolors='none',linewidths=0)
    ax.add_collection3d(collection)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--body',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(args.geometry,allow_pickle=False));bv,bf=body_geometry(json.loads(args.body.read_text()))
    v,b,f=a['vertices'],a['before_vertices'],a['faces']
    fig,axes=plt.subplots(1,2,figsize=(11,7))
    for ax in axes:
        for points,faces,colour,label,width in [(b,f,'#949494','Before',1.2),(v,f,'#d66c28','Posterior edit',2),(bv,bf,'#3272aa','Unchanged body',1.4)]:
            ax.add_collection(LineCollection(sections(points,faces),colors=colour,linewidths=width,label=label))
        ax.set_aspect('equal');ax.grid(alpha=.3);ax.set_xlabel('Front / back');ax.set_ylabel('Height')
    axes[0].set_xlim(-1.05,1.4);axes[0].set_ylim(-.85,1.85);axes[0].legend(loc='upper left')
    axes[1].set_xlim(-1.02,-.32);axes[1].set_ylim(-.15,1.2);axes[1].set_title('Posterior detail: actual mesh sections')
    fig.tight_layout();fig.savefig(args.output/'posterior_sections.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(14,8))
    changed_vertices=np.linalg.norm(v-b,axis=1)>1e-12
    changed_faces=changed_vertices[f].any(1)
    body_faces=bf[(bv[bf,1].max(1)>-.9)&(bv[bf,1].min(1)<.3)]
    for i,(points,title,angle) in enumerate([(b,'Before: side',0),(v,'Posterior edit: side',0),(v,'Posterior edit: rear oblique',-45)]):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        shaded(ax,bv,body_faces,[.64,.65,.68])
        shaded(ax,points,f,[.79,.75,.69],changed_faces if i else None)
        ax.set_xlim(-1,1);ax.set_ylim(-1,1.4);ax.set_zlim(-.8,1.9)
        ax.set_box_aspect([2,2.4,2.7]);ax.set_proj_type('ortho');ax.view_init(elev=5,azim=angle)
        ax.set_axis_off();ax.set_title(title)
    fig.text(.5,.04,'Orange: edited posterior faces. Plain mesh diagnostic; no texture or game-render claim.',ha='center')
    fig.tight_layout();fig.savefig(args.output/'posterior_3d.png',dpi=150);plt.close(fig)
    print(args.output)


if __name__=='__main__':main()
