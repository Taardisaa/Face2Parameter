"""Draw preserved image contours and real native vertices; no character mutation.

Manual observations, shadow transitions, and mesh envelopes stay separate.
Reference/generated profile angles are not calibrated 3D ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image
from chin_projected_audit import project

COLORS = {'reference': '#00aaff', 'game': '#ff423f', 'photo': '#00aaff'}


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refine_visible_profile(row):
    """Refine inspected profile seeds at their skin/background pixel transition.

    This local image observation is not a shader mask or game deformation model.
    Other/shadow/occluded curves are deliberately left untouched.
    """
    if row['angle'] not in ('left','right'):
        return
    im=np.asarray(Image.open(row['image_path']).convert('RGB'),float)
    skin=(im[:,:,0]-im[:,:,1]>12)&(im[:,:,0]-im[:,:,2]>18)&(im.mean(2)>80)
    changes=[]
    for curve in row['curves']:
        if curve['kind']!='visible_edge':
            continue
        old=np.asarray(curve['points_xy'],float)
        refined=[]
        for i,p in enumerate(old):
            tangent=old[min(i+1,len(old)-1)]-old[max(i-1,0)]
            normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
            offsets=np.linspace(-25,25,201)
            probes=p+offsets[:,None]*normal
            indices=np.rint(probes).astype(int)
            if np.any(indices<0) or np.any(indices[:,0]>=skin.shape[1]) or np.any(indices[:,1]>=skin.shape[0]):
                raise ValueError('Visible boundary search leaves original image')
            values=skin[indices[:,1],indices[:,0]]
            crossing=np.flatnonzero(values[:-1]!=values[1:])
            if not len(crossing):
                refined.append(p.tolist());changes.append({'point':i,'status':'no_local_color_transition; original manual seed retained'})
                continue
            deltas=(offsets[crossing]+offsets[crossing+1])/2
            delta=float(deltas[np.abs(deltas).argmin()])
            refined.append((p+delta*normal).tolist())
            changes.append({'point':i,'normal_offset_pixels':delta})
        curve['original_manual_points_xy']=curve['points_xy']
        curve['points_xy']=refined
        for key in ('D','P','C'):
            if key in row.get('landmarks_xy',{}):
                j=int(np.linalg.norm(old-row['landmarks_xy'][key],axis=1).argmin())
                row['landmarks_xy'][key]=refined[j]
    row['profile_pixel_refinement']={'method':'Nearest local warm-skin/color-background transition along original polyline normals, +/-25px at .25px sampling; R-G>12, R-B>18 and mean RGB>80. Raw manual points retained. Visually inspect: this is a local color observation, not a certified segmentation or exact 3D surface.','observations':changes}


def normalized(row, points):
    """One similarity transform from independent eye anchors; never jaw fitting."""
    a, b = np.asarray(row['alignment_anchors_xy'], float)
    v = b - a
    distance = np.linalg.norm(v)
    if distance <= 0:
        raise ValueError('Independent anchors required')
    angle = np.arctan2(v[1], v[0])
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    return (np.asarray(points)-a) @ rotation.T / distance


def draw(ax, row, mesh=None, fixed_ids=None):
    im = Image.open(row['image_path'])
    ax.imshow(im)
    color = COLORS[row['role']]
    if mesh is not None:
        xy, tri = mesh
        ax.add_collection(LineCollection(xy[tri][:, [0,1,2,0]],
                                        colors='#e1e1e1', linewidths=.22, alpha=.42))
        if fixed_ids is not None:
            for n, i in enumerate(fixed_ids, 1):
                x, y = xy[i]
                ax.plot(x, y, 'o', color='#ff3333', ms=4)
                ax.annotate('V'+str(n), (x, y), xytext=(4,-5),
                            textcoords='offset points', fontsize=8, color='#ff3333')
    for curve in row['curves']:
        p = np.asarray(curve['points_xy'])
        ax.plot(p[:,0], p[:,1], '.-' if curve['kind']=='visible_edge' else '.--',
                color=color, linewidth=1.8, markersize=2.4)
    for name, xy in row.get('landmarks_xy', {}).items():
        ax.plot(*xy, 'o', color='#ffc247', markersize=4)
        ax.annotate(name, xy, xytext=(4,-6), textcoords='offset points',
                    color='#ffc247', fontsize=8)
    x0,y0,x1,y1 = row['roi_xyxy']
    ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.axis('off')


def front_metrics(row):
    """Widths across the manually observed mouth-to-menton span, not a score."""
    p = normalized(row, row['curves'][0]['points_xy'])
    mouth = normalized(row, [row['landmarks_xy']['M']])[0]
    bottom_i = int(p[:,1].argmax())
    left, right = p[:bottom_i+1], p[bottom_i:][::-1]
    # Deroll may slightly disturb terminal point ordering. Intersections retain
    # the complete polyline and select both sides; no monotonic sorting or hull.
    fractions = [.1,.25,.5,.75,.9,.95,.97,.99]
    bottom_y = float(p[:,1].max())
    result = {}
    for f in fractions:
        y = mouth[1] + f*(bottom_y-mouth[1])
        hits=[]
        for a,b in zip(p[:-1],p[1:]):
            if (a[1]<=y<b[1]) or (b[1]<=y<a[1]):
                hits.append(float(a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1])))
        if len(hits)<2:
            raise ValueError('Incomplete manually observed section')
        result[str(f)] = {'left':min(hits),'right':max(hits),'width':max(hits)-min(hits)}
    return {'width_sections_per_eye_distance':result,
            'mouth_to_bottom_per_eye_distance':bottom_y-float(mouth[1]),
            'bottom_above_eye_axis':bottom_y,
            'scope':'Observed polyline widths; shadow boundaries and manual uncertainty retained. Extreme terminal widths are especially uncertainty-sensitive. No 3D or acceptance claim.'}


def source_differences(before, after):
    mb=next(m for m in before['meshes'] if m['mesh_name']=='o_head')
    ma=next(m for m in after['meshes'] if m['mesh_name']=='o_head')
    if mb['source_geometry_sha256'] != ma['source_geometry_sha256']:
        raise ValueError('Cannot compare different source meshes')
    def head_positions(d,m):
        t=next(t for t in d['transforms'] if t['name']=='cf_J_Head')
        w=np.c_[m['baked']['vertices'],np.ones(m['vertex_count'])] @ np.asarray(m['baked']['world_candidates']['scale_free_trs']['matrix']).reshape(4,4).T
        return (w @ np.asarray(t['world_to_local']).reshape(4,4).T)[:,:3]
    vb,va=head_positions(before,mb),head_positions(after,ma)
    indices=np.asarray(ma['source']['bone_indices']);weights=np.asarray(ma['source']['bone_weights'])
    ans={}
    for name in ['cf_J_CheekLow_L','cf_J_CheekLow_R','cf_J_ChinLow','cf_J_ChinTip_s']:
        bid=ma['bone_names'].index(name)
        w=np.sum(np.where(indices==bid,weights,0),axis=1)
        ids=np.flatnonzero(w>.05)
        delta=va[ids]-vb[ids]
        ans[name]={'diagnostic_weight_selection':'> .05 actual source weight; not an anatomical hard segmentation',
                   'vertex_count':len(ids),'mean_delta_head_xyz':delta.mean(0).tolist(),
                   'max_absolute_delta_head_xyz':np.abs(delta).max(0).tolist(),
                   'max_distance_head_units':float(np.linalg.norm(delta,axis=1).max())}
    return {'scope':'v17b native baseline versus current actual BakeMesh, transformed into each exact recorded head local matrix. Source topology identical. No inferred screenshot deformation gains.',
            'regions':ans}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--baseline-geometry',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--refine-visible-profiles',action='store_true')
    a=parser.parse_args()
    if a.out.exists():
        raise FileExistsError('Fresh review directory required')
    annotations=load(a.annotations)
    if annotations.get('status','').startswith('REJECTED'):
        raise ValueError('User rejected these annotations; they must not produce comparison metrics or authoring targets. Preserved historical outputs remain diagnostic failures.')
    rows=annotations['images']
    for r in rows:
        if sha(r['image_path'])!=r['image_sha256']:
            raise ValueError('Annotation/image identity mismatch: '+r['name'])
        im=Image.open(r['image_path'])
        for curve in r['curves']:
            p=np.asarray(curve['points_xy'])
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
                raise ValueError('Nonempty finite polyline required')
            if np.any(p<0) or np.any(p>np.asarray(im.size)-1):
                raise ValueError('Contour leaves original image')
        if a.refine_visible_profiles:
            refine_visible_profile(r)
    d=load(a.geometry);head=next(m for m in d['meshes'] if m['mesh_name']=='o_head')
    triangles=np.asarray(head['source']['triangles']).reshape(-1,3)
    raw_sha=hashlib.sha256(struct.pack('<i',head['vertex_count'])+
                           np.asarray(head['baked']['vertices'],dtype='<f4').tobytes()).hexdigest()
    game=[r for r in rows if r['role']=='game']
    projected={};matching={}
    for r in game:
        c=load(r['capture_path'])['capture']
        observed=next(m for m in c['face_surfaces'] if m['renderer']=='o_head')
        if observed['baked_vertices_sha256']!=raw_sha:
            raise ValueError('Main-camera pre-render head differs from paired mesh: '+r['name'])
        audit=project(a.geometry,Path(r['capture_path']))
        projected[r['angle']]=np.asarray(audit['all_projected_xy'])
        matching[r['angle']]={'pre_render_head_bake_matches_paired_geometry':True,
            'image_sha256':r['image_sha256'],'capture_sha256':sha(r['capture_path']),
            'TRS_representation_component_error':audit['export_head_matrix_vs_TRS_max_component_error'],
            'scope':'Exact raw pre-render local BakeMesh equality plus captured head TRS. Final shader fragments/post-render transforms and hidden contours are not certified.'}
    front=next(r for r in game if r['angle']=='front')
    xy=projected['front'];source=np.asarray(head['source']['vertices'])
    lower_front=np.flatnonzero((source[:,1]<.10)&(source[:,2]>.25))
    chosen=[]
    for p in annotations['mesh_seed_points_on_front_xy']:
        chosen.append(int(lower_front[np.linalg.norm(xy[lower_front]-p,axis=1).argmin()]))
    a.out.mkdir(parents=True)
    (a.out/'effective_annotations.json').write_text(json.dumps(annotations,indent=2),encoding='utf-8')
    report={'status':'Contour/source review only; user-rejected chin and cheeks remain unaccepted; no character parameter edits.',
        'annotations_sha256':sha(a.annotations),'geometry_sha256':sha(a.geometry),
        'source_geometry_sha256':head['source_geometry_sha256'],'head_raw_bake_sha256':raw_sha,
        'capture_bindings':matching,'fixed_source_vertex_ids':chosen,
        'fixed_vertex_scope':'Same V1..V5 actual source IDs projected in every view. Front-view seed picks are screen-nearest diagnostic markers, not certified anatomical landmarks or silhouette correspondences. Silhouette vertices can change with view.',
        'front_reference_observation':front_metrics(next(r for r in rows if r['name']=='reference_front')),
        'front_game_observation':front_metrics(front),
        'source_change_from_v17b':source_differences(load(a.baseline_geometry),d),
        'limitations':annotations['limitations']}
    for r in rows:
        fig,ax=plt.subplots(figsize=(6,7))
        draw(ax,r)
        ax.set_title(r['display_title'],fontsize=10)
        fig.savefig(a.out/(r['name']+'_marked.png'),dpi=200,bbox_inches='tight');plt.close(fig)
    angles=['front','left','right','quarter']
    fig,axes=plt.subplots(3,4,figsize=(16,14),constrained_layout=True)
    for j,angle in enumerate(angles):
        ref=next(r for r in rows if r['role']=='reference' and r['angle']==angle)
        actual=next(r for r in game if r['angle']==angle)
        draw(axes[0,j],ref);axes[0,j].set_title(ref['display_title'],fontsize=10)
        draw(axes[1,j],actual);axes[1,j].set_title('Game observed contour: '+angle,fontsize=10)
        draw(axes[2,j],actual,(projected[angle],triangles),chosen)
        axes[2,j].set_title('Same native head vertices: '+angle,fontsize=10)
    fig.suptitle('BLUE: reference | RED: game | GOLD: feature anchors | dashed: shadow boundary\nSame colors and meanings in all views. Generated profiles are design references, not calibrated 3D truth.',fontsize=13)
    fig.savefig(a.out/'multiview_contours_and_vertices.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,6),constrained_layout=True)
    for ax,r in zip(axes[:2],[r for r in rows if r['role']=='photo']):
        draw(ax,r);ax.set_title(r['display_title'],fontsize=10)
    draw(axes[2],front);axes[2].set_title('Current game lower cheek / chin')
    fig.savefig(a.out/'real_photos_cheek_review.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    for r in [next(r for r in rows if r['name']=='reference_front'),front]:
        p=normalized(r,r['curves'][0]['points_xy'])
        ax.plot(p[:,0],p[:,1],'.-',color=COLORS[r['role']],label=r['role'],lw=2,ms=3)
    ax.set_aspect('equal');ax.invert_yaxis();ax.legend();ax.grid(alpha=.25)
    ax.set_title('Front observed contours, fixed eye anchors\nUniform scale / roll only; no jaw or mouth alignment')
    ax.set_xlabel('X / eye distance');ax.set_ylabel('Y / eye distance')
    fig.savefig(a.out/'front_observed_overlay.png',dpi=180);plt.close(fig)
    (a.out/'review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'fixed_source_vertex_ids':chosen,'capture_bindings':matching,
          'front_reference':report['front_reference_observation'],
          'front_game':report['front_game_observation'],
          'source_change':report['source_change_from_v17b']},indent=2))


if __name__=='__main__':
    main()
