"""Uniform source-head size/pitch candidates, rebuilt against a fixed native rim.

Uses the exact original FLAME output and existing audited crop recipes. This is
explicit placement authoring, not a fit to the native reference face or HS2 logic.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from tools.model_bridge.artifact import ModelArtifact,sha
from tools.native_head.neck_section_review import sections
from tools.native_head.posterior_neck import author as author_posterior
from tools.native_head.posterior_review import shaded
from tools.native_head.native_seam_audit import extract,aliases
from tools.native_head.neck_geometry import ordered_loops
from src.hs2_mesh_deform import HeadRig,build_mesh


def normals(v,f):
    xyz=v[f];n=np.cross(xyz[:,1]-xyz[:,0],xyz[:,2]-xyz[:,0])
    out=np.zeros_like(v)
    for k in range(3):np.add.at(out,f[:,k],n)
    return out/np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-20)


def similarity(center,size,pitch):
    angle=np.deg2rad(pitch);c,s=np.cos(angle),np.sin(angle)
    # In the profile plot, +Z is right and +Y is up. Positive X-axis rotation
    # therefore pitches the face clockwise/down, not counterclockwise/up.
    rotation=np.array([[1,0,0],[0,c,-s],[0,s,c]])
    matrix=np.eye(4);matrix[:3,:3]=size*rotation
    matrix[:3,3]=center-size*rotation@center
    return matrix,rotation


def apply(v,m):return v@m[:3,:3].T+m[:3,3]


def native_endpoint_normals(base,r):
    """Interpolate the actual native authored normal field on its real upper rim.

    Recalculating normals from the tiny isolated last band loses the head's
    neighbouring surface directions, causing the original cubic columns to
    cross. This restores the native direction field used by that asset.
    """
    path=Path(r['native_bundle'])
    if sha(path)!=r['native_bundle_sha256']:raise ValueError('Native reference asset changed')
    arrays,names=extract(path.read_bytes(),'p_cf_head_02');rig=HeadRig(2)
    if (not np.array_equal(arrays['verts'],rig.verts.astype(np.float32))
        or not np.array_equal(arrays['faces'],rig.faces)
        or not np.array_equal(arrays['normals'],rig.normals)):
        raise ValueError('Recovered native rig differs from current installed asset')
    v,_=build_mesh(rig,np.full(59,.5),None)
    alias=aliases(arrays);f=np.array([[alias[int(i)] for i in tri] for tri in arrays['faces']])
    root=names.index('cf_J_FaceRoot_s');w=(arrays['bone_w']*(arrays['bone_idx']==root)).sum(1)
    outer=next(loop for loop in ordered_loops(f) if np.all(w[loop]==1))
    band=f[np.isin(f,outer).any(1)]
    upper=np.asarray(next(loop for loop in ordered_loops(band) if set(loop)!=set(outer)))
    points=base['vertices'][base['native_upper']];edge=v[np.roll(upper,-1)]-v[upper]
    t=np.clip(((points[:,None]-v[upper])*edge).sum(2)/(edge*edge).sum(1),0,1)
    error=np.linalg.norm(v[upper]+t[:,:,None]*edge-points[:,None],axis=2)
    which=error.argmin(1)
    if error[np.arange(len(points)),which].max()>1e-9:raise ValueError('Native endpoint left its original asset edge')
    t=t[np.arange(len(points)),which,None]
    n=(1-t)*arrays['normals'][upper[which]]+t*arrays['normals'][np.roll(upper,-1)[which]]
    return n/np.linalg.norm(n,axis=1,keepdims=True)


def rebuilt_baseline(base,r,placed,m,rotation,native_normals):
    a={k:v.copy() for k,v in base.items()}
    width=len(a['source_ring'])
    interior_start=len(a['vertices'])-5*width
    source_count=interior_start-len(a['native_original_ids'])
    if np.any(a['source_ring']>=source_count) or np.any(a['native_upper']<source_count):
        raise ValueError('Unexpected retained source/native collar layout')
    a['vertices'][:source_count]=apply(base['vertices'][:source_count],m)
    # Applying the similarity to already clipped source triangles is exactly the
    # same affine operation as transforming full raw output before clipping.
    recipe=a['crop_recipes'];i,j=recipe[:,:2].astype(int).T;t=recipe[:,2,None]
    expected=(1-t)*placed[i]+t*placed[j]
    if not np.allclose(a['vertices'][:len(recipe)],expected,atol=1e-12,rtol=0):
        raise ValueError('Source crop no longer agrees with unchanged raw topology')
    original_placed=base['original_vertices'].astype(float)*r['scale']+r['translation']
    original_normals=normals(original_placed,base['original_faces'])
    cut_normals=(1-t)*original_normals[i]+t*original_normals[j]
    cut=base['crop_vertices'];ring=base['crop_neck_ring']
    edge=cut[np.roll(ring,-1)]-cut[ring]
    start=base['vertices'][base['source_ring']]
    fractions=np.clip(((start[:,None]-cut[ring])*edge).sum(2)/(edge*edge).sum(1),0,1)
    distance=np.linalg.norm(cut[ring]+fractions[:,:,None]*edge-start[:,None],axis=2)
    which=distance.argmin(1)
    if distance[np.arange(width),which].max()>1e-9:raise ValueError('Refined source rim left its actual crop edges')
    fraction=fractions[np.arange(width),which,None]
    ns=((1-fraction)*cut_normals[ring[which]]+fraction*cut_normals[np.roll(ring,-1)[which]])@rotation.T
    ne=native_normals
    ns/=np.linalg.norm(ns,axis=1,keepdims=True);ne/=np.linalg.norm(ne,axis=1,keepdims=True)
    start=a['vertices'][a['source_ring']];end=a['vertices'][a['native_upper']]
    chord=end-start
    d0=chord-ns*np.sum(chord*ns,axis=1,keepdims=True)
    d1=chord-ne*np.sum(chord*ne,axis=1,keepdims=True)
    for row,t in enumerate(np.linspace(0,1,7)[1:-1]):
        points=(2*t**3-3*t*t+1)*start+(t**3-2*t*t+t)*d0+(-2*t**3+3*t*t)*end+(t**3-t*t)*d1
        a['vertices'][interior_start+row*width:interior_start+(row+1)*width]=points
    a['placed_original_vertices']=placed
    return a,source_count


def body_points(body):
    mesh=next(x for x in body['meshes'] if x['mesh_name']=='o_body_cf')
    root=mesh['bone_names'].index('cf_J_Head_s')
    m=np.asarray(mesh['source']['bindposes'][root]).reshape(4,4)
    return apply(np.asarray(mesh['source']['vertices']),m),np.asarray(mesh['source']['triangles']).reshape(-1,3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True,help='Existing pre-posterior native-band baseline')
    p.add_argument('--current',type=Path,required=True,help='Previously reviewed posterior result')
    p.add_argument('--body',type=Path,required=True)
    p.add_argument('--masks',type=Path,default=Path(r'C:\Users\13666\Workspace\smirk\assets\FLAME_masks\FLAME_masks.pkl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=float,default=.94)
    p.add_argument('--pitch',type=float,default=3.)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output')
    if not np.isfinite([args.size,args.pitch]).all() or args.size<=0:raise ValueError('Finite positive uniform scale required')
    r=json.loads((args.baseline/'receipt.json').read_text());base=dict(np.load(args.baseline/'geometry.npz',allow_pickle=False))
    if sha(args.body)!=r['body_capture_sha256'] or sha(args.masks)!=r['masks_sha256']:raise ValueError('Body/masks changed')
    artifact=ModelArtifact(r['source_manifest'],r['image_index']);raw,faces=artifact.mesh(head_local=True)
    if (sha(artifact.path)!=r['manifest_sha256'] or artifact.image['sha256']!=r['raw_sha256']
        or artifact.image['input_sha256']!=r['source_image_sha256']):raise ValueError('Original source provenance changed')
    if not np.array_equal(raw,base['original_vertices']) or not np.array_equal(faces,base['original_faces']):raise ValueError('Baseline does not match true original source mesh')
    original=raw.astype(float)*r['scale']+r['translation']
    masks=pickle.loads(args.masks.read_bytes(),encoding='latin1')
    ids=np.setdiff1d(np.unique(np.r_[masks['face'],masks['scalp'],masks['left_ear'],masks['right_ear']]),masks['neck'])
    center=(original[ids].min(0)+original[ids].max(0))/2
    body=json.loads(args.body.read_text());bv,bf=body_points(body)
    native_normals=native_endpoint_normals(base,r)
    args.output.mkdir(parents=True)
    candidates=[]
    for name,size,pitch in [('size_only',args.size,0),('size_and_pitch',args.size,args.pitch)]:
        output=args.output/name;output.mkdir()
        m,rotation=similarity(center,size,pitch);placed=apply(original,m)
        if not np.allclose(apply(placed,np.linalg.inv(m)),original,atol=1e-12,rtol=0):raise ValueError('Uniform placement changed source shape')
        arrays,source_count=rebuilt_baseline(base,r,placed,m,rotation,native_normals)
        np.savez(output/'pre_posterior.npz',**arrays)
        edited,posterior,q,fade=author_posterior(arrays,r,masks,body)
        arrays['before_vertices']=arrays['vertices'].copy();arrays['vertices']=edited
        arrays['posterior_coordinate']=q;arrays['posterior_fade']=fade
        seam=np.asarray(r['native_outer_ring_authored_ids'])
        if not np.array_equal(edited[seam],base['vertices'][seam]):raise ValueError('Native body interface moved')
        facial=np.unique(np.r_[masks['face'],masks['left_ear'],masks['right_ear'],masks['left_eyeball'],masks['right_eyeball']])
        old=arrays['crop_original_ids'];locked=(old>=0)&np.isin(old,facial)
        if not np.allclose(edited[:len(old)][locked],placed[old[locked]],atol=1e-12,rtol=0):raise ValueError('Face/ears/eyes acquired a local deformation')
        np.savez(output/'geometry.npz',**arrays)
        (output/'head.obj').write_text('\n'.join(['v '+' '.join(map(str,x)) for x in edited]+['f '+' '.join(str(int(i)+1) for i in tri) for tri in arrays['faces']])+'\n')
        receipt=dict(r,format='source_head_similarity_review_v1',geometry_sha256=sha(output/'geometry.npz'),
            source_baseline_geometry_sha256=sha(args.baseline/'geometry.npz'),previous_review_geometry_sha256=sha(args.current),
            similarity_matrix=m.tolist(),center=center.tolist(),center_policy='Source head semantic face/scalp/ears bounds excluding neck mask',
            relative_uniform_scale=size,clockwise_pitch_degrees=pitch,source_prefix_count=source_count,
            source_shape_preserved_up_to_similarity=True,face_ear_eye_positions_match_transformed_raw=True,
            native_interface_positions_unchanged=True,posterior_design=posterior,
            connector_policy='Cubic authored surface with exact retained-source normals and interpolated native asset normal directions; no normals recalculated from isolated native band',
            local_occipital_thinning=False,native_asset_installed=False,game_mutations=False,
            unfinished=['Geometry acceptance','Review final placement against reference photos','Native asset/skin/normal integration and game acceptance'])
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        candidates.append((name,edited,arrays['faces'],placed))
    current=dict(np.load(args.current,allow_pickle=False))
    if not np.array_equal(current['original_vertices'],base['original_vertices']):raise ValueError('Comparison head belongs to a different raw identity')
    fig,axes=plt.subplots(1,3,figsize=(16,8))
    views=[('Current reviewed head',current['vertices'],current['faces']),('Uniformly smaller',*candidates[0][1:3]),('Smaller + slight downward pitch',*candidates[1][1:3])]
    for ax,(title,v,f) in zip(axes,views):
        ax.add_collection(LineCollection(sections(current['vertices'],current['faces']),colors='#b0b0b0',linewidths=1.1,linestyles='dotted',label='Previous reviewed head'))
        ax.add_collection(LineCollection(sections(v,f),colors='#d66c28',linewidths=1.8,label='Candidate head'))
        ax.add_collection(LineCollection(sections(bv,bf),colors='#777777',linewidths=1.6,label='Fixed original body'))
        ax.set_xlim(-1.15,1.45);ax.set_ylim(-1.15,1.95);ax.set_aspect('equal');ax.grid(alpha=.25)
        ax.set_xlabel('Front / back');ax.set_ylabel('Height');ax.set_title(title,fontsize=11);ax.legend(loc='lower left',fontsize=8)
    fig.tight_layout();fig.savefig(args.output/'complete_placement_sections.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(15,8))
    body_view=bf[(bv[bf,1].max(1)>-.85)&(bv[bf,1].min(1)<.3)]
    for i,(title,v,f) in enumerate(views):
        ax=fig.add_subplot(1,3,i+1,projection='3d');shaded(ax,bv,body_view,[.65,.67,.69]);shaded(ax,v,f,[.78,.76,.72])
        ax.set_xlim(-1,1);ax.set_ylim(-1.1,1.45);ax.set_zlim(-.85,1.95);ax.set_box_aspect([2,2.55,2.8])
        ax.set_proj_type('ortho');ax.view_init(elev=0,azim=0);ax.set_axis_off();ax.set_title(title,fontsize=11)
    fig.tight_layout();fig.savefig(args.output/'placement_3d.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(output=str(args.output),candidate_names=[x[0] for x in candidates],game_mutations=False)))


if __name__=='__main__':main()
