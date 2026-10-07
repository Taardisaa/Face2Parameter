"""Author-v10 metadata on the unchanged full FaceScape neutral TU sample.

An author-numbered point/UV mask is not an HS2 anatomical correspondence.
No pickle, network calls, source cropping or candidate-dependent transforms.
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import find_objects, label

from .facescape_target_v1 import (
    array_digest,
    binding,
    check,
    exact,
    freeze,
    load_raw_target,
    read,
    readonly,
    require,
    sha,
)

KIND = 'facescape_author_v10_source_semantics_v1'
ROLES = ['toolkit/predef/landmark_indices.npz','toolkit/predef/facial_mask_v10.png',
         'toolkit/src/camera.py','toolkit/src/renderer.py','toolkit/src/utility.py',
         'toolkit/demo_landmark.ipynb','toolkit/demo_mask.ipynb','toolkit/README.md','README.md','doc/doc_tu_model.md']


def upstream_sources(receipt_path):
    receipt=read(receipt_path);commit=receipt['commit']
    require(type(commit) is str and len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Frozen author commit required')
    sources={}
    for role in ROLES:
        url='https://raw.githubusercontent.com/zhuhao-nju/facescape/'+commit+'/'+role
        entries=[r for r in receipt['files'] if r['url']==url]
        require(len(entries)==1,'Missing/ambiguous pinned author source: '+role)
        row=entries[0]
        require(type(row['status']) is int and row['status']==200 and row['final_url']==url,'Author request failed/redirected to another source')
        descriptor={k:row[k] for k in ['path','sha256','bytes']};check(descriptor)
        sources[role]={'descriptor':descriptor,'url':url}
    commit_file=Path(receipt_path).parent/'commit.json'
    require(sha(commit_file)==receipt['commit_json_sha256'] and read(commit_file)['sha']==commit,'Commit API receipt differs')
    return sources,{'commit':commit,'receipt':binding(receipt_path),'commit_api_receipt':binding(commit_file)}


def notebook(path):
    doc=read(path)
    codes=[(i,cell) for i,cell in enumerate(doc['cells']) if cell['cell_type']=='code']
    require(codes,'Author notebook lacks code')
    return doc,codes


def demo_camera(path):
    _,codes=notebook(path);source=''.join(codes[0][1]['source']);tree=ast.parse(source)
    assignments={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ['K','Rt']:
                require(isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                        and node.value.func.attr=='array' and len(node.value.args)==1,'Unknown author camera expression')
                assignments[name]=ast.literal_eval(node.value.args[0])
    sizes=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='render_cvcam':
            for keyword in node.keywords:
                if keyword.arg=='rend_size':sizes.append(ast.literal_eval(keyword.value))
    require(assignments.keys()=={'K','Rt'} and sizes and all(s==sizes[0] for s in sizes),'Unambiguous explicit author camera required')
    k,rt=np.asarray(assignments['K'],float),np.asarray(assignments['Rt'],float)
    require(k.shape==(3,3) and rt.shape==(3,4) and np.isfinite(k).all() and np.isfinite(rt).all(),'Malformed author camera')
    require(len(sizes[0])==2 and all(type(v) is int and v>0 for v in sizes[0]),'Typed author render dimensions required')
    return {'K':k.tolist(),'Rt':rt.tolist(),'height':sizes[0][0],'width':sizes[0][1],
            'source_cell_index':codes[0][0],'fitted':False,'frame':'author CV image convention; source-derived projection, not HS2 frame'}


def validate_indices(indices,vertex_count,version='v10'):
    require(version=='v10','TU neutral sample requires author v10; in-range v16 is not a compatible fallback')
    indices=np.asarray(indices)
    require(indices.shape==(68,) and indices.dtype.kind in 'iu','Exact68 integer author-numbered indices required')
    require(type(vertex_count) is int and vertex_count>0 and indices.min()>=0 and indices.max()<vertex_count,'Vertex index outside unchanged source')
    require(len(np.unique(indices))==68,'Duplicate author landmark index')
    return readonly(indices,'<i8')


def attachments(target,indices):
    result=[]
    for number,index in enumerate(indices):
        hits=np.argwhere(target.faces==index)
        require(len(hits)>0,'Landmark vertex not used by surface topology')
        face,corner=map(int,hits[0]);uv_ids=sorted({int(target.face_uv_indices[f,c]) for f,c in hits})
        require(min(uv_ids)>=0,'Missing UV corner; no per-vertex UV assumption')
        bary=[0.,0.,0.];bary[corner]=1.
        result.append({'author_number_zero_based':number,'source_vertex_index':int(index),
                       'canonical_triangle_id':face,'barycentric':bary,'incident_triangle_count':len(hits),
                       'all_incident_uv_indices':uv_ids,'all_incident_uv':target.texcoords[uv_ids].tolist(),
                       'author_named_anatomical_definition':None})
    return result


def mask_array(path):
    with Image.open(path) as image:
        require(image.format=='PNG' and image.mode=='L','Authored grayscale PNG mask required, no conversion')
        array=np.array(image)
    require(array.shape==(4096,4096) and array.dtype==np.uint8,'Expected original4096 grayscale mask')
    return readonly(array,'u1')


def source_definition_checks(sources):
    lm=read(sources['toolkit/demo_landmark.ipynb']['descriptor']['path'])
    mask=read(sources['toolkit/demo_mask.ipynb']['descriptor']['path'])
    lm_text='\n'.join(''.join(c['source']) for c in lm['cells']);mask_text='\n'.join(''.join(c['source']) for c in mask['cells'])
    require("['v10']" in lm_text and 'verts[lm_ind]' in lm_text and 'openmesh' in lm_text
            and '../samples/sample_tu_model/1_neutral.obj' in lm_text,'Author exact sample/version/order applicability not established')
    require('facial_mask_v10.png' in mask_text and 'PIL.ImageOps.invert' in mask_text
            and 'mask>200' in mask_text and 'not strictly symmetric' in mask_text,'Author v10 mask operation/limit changed')
    return {'version_eligibility':'author demo explicitly references this neutral TU filename and v10; both v10/v16 being in-range alone is insufficient',
            'order_definition':'zero-based NumPy verts[lm_ind], author-numbered enumerate order0..67; independent reader retains OBJ vertices and corner indices',
            'named_landmark_definitions':'no individual anatomical-name table in inspected author sources; IDs/author visual demo only',
            'mask_definition':'author UV facial region, v10 not strictly symmetric; demo inverts PNG then rendered mask>200 becomes white',
            'direct_uv_to_array_row_formula':'not explicit in inspected author demo/renderer; trimesh/pyrender dependency conventions not reconstructed',
            'mask_white_face_interpretation':'white source mask retained after inversion/rendered threshold; direct CPU sampling is a diagnostic hypothesis, not shader-equivalent threshold',
            'head_surface_cropped':False,'hs2_semantic_bridge_certified':False}


def build_contract(raw_path,raw_sha,receipt_path):
    target=load_raw_target(raw_path,raw_sha)
    require(len(target.vertices)==26317 and len(target.faces)==52261,'This revision binds complete actual neutral TU topology only')
    sources,acquisition=upstream_sources(receipt_path)
    definitions=source_definition_checks(sources)
    npz_path=sources['toolkit/predef/landmark_indices.npz']['descriptor']['path']
    with np.load(npz_path,allow_pickle=False) as archive:
        require(set(archive.files)=={'v10','v16'},'Unexpected author index archive schema')
        indices=validate_indices(archive['v10'],len(target.vertices))
        v16=archive['v16'];require(v16.shape==(68,) and v16.dtype.kind in 'iu','Malformed alternate-version metadata')
    mask=mask_array(sources['toolkit/predef/facial_mask_v10.png']['descriptor']['path'])
    require(np.all(target.face_uv_indices>=0) and np.all(target.face_uv_indices<len(target.texcoords)),'Source UV corner mapping unavailable')
    require(np.isfinite(target.texcoords).all() and target.texcoords.min()>=0 and target.texcoords.max()<=1,'Out-of-unit authored UV; no silent wrap/clamp repair')
    unique,counts=np.unique(mask,return_counts=True)
    return {'schema_version':1,'kind':KIND,'loader':binding(__file__),
            'raw_reader':binding(Path(__file__).with_name('facescape_target_v1.py')),
            'raw_contract':binding(raw_path),'source_metadata':sources,'acquisition':acquisition,
            'sample_scope':'same original full26317/52261 neutral TU asset; no HS2 or anger tuple inputs',
            'landmarks':{'selected_version':'v10','count':68,'indices':indices.tolist(),
                         'indices_sha256':array_digest(indices,'<i8'),'all_unique_in_range':True,
                         'other_version_v16_also_in_range':bool(v16.min()>=0 and v16.max()<len(target.vertices)),
                         'vertex_attachments':attachments(target,indices)},
            'mask':{'shape':list(mask.shape),'mode':'L','values':unique.tolist(),'counts':counts.tolist(),
                    'array_sha256':array_digest(mask,'u1'),'uv_corner_mapping_hash':array_digest(target.face_uv_indices,'<i8'),
                    'uv_bounds':[target.texcoords.min(0).tolist(),target.texcoords.max(0).tolist()],
                    'direct_row_convention_certified':False,'direct_threshold_semantic_equivalence_certified':False},
            'definitions':definitions,'landmark_demo_camera':demo_camera(sources['toolkit/demo_landmark.ipynb']['descriptor']['path']),
            'mask_demo_camera':demo_camera(sources['toolkit/demo_mask.ipynb']['descriptor']['path']),
            'units':'unchanged unknown raw sample units','hs2_axes_or_scale_calibrated':False,
            'raw_target_modified_or_masked':False,'anatomical_acceptance_threshold':None}


def freeze_semantics(raw_path,raw_sha,receipt_path,out_contract):
    return freeze(out_contract,build_contract(raw_path,raw_sha,receipt_path))


@dataclass(frozen=True)
class Semantics:
    target:object
    landmark_indices:np.ndarray
    landmark_points:np.ndarray
    mask:np.ndarray
    contract:dict


def load_semantics(path,expected_sha):
    require(sha(path)==expected_sha,'Frozen semantics contract SHA differs')
    contract=read(path)
    require(type(contract.get('schema_version')) is int and contract['schema_version']==1 and contract.get('kind')==KIND,'Explicit author-v10 semantics contract required')
    check(contract['loader']);check(contract['raw_reader']);check(contract['raw_contract'])
    rebuilt=build_contract(contract['raw_contract']['path'],contract['raw_contract']['sha256'],contract['acquisition']['receipt']['path'])
    exact(contract,rebuilt)
    target=load_raw_target(contract['raw_contract']['path'],contract['raw_contract']['sha256'])
    indices=validate_indices(contract['landmarks']['indices'],len(target.vertices))
    mask=mask_array(contract['source_metadata']['toolkit/predef/facial_mask_v10.png']['descriptor']['path'])
    return Semantics(target,indices,readonly(target.vertices[indices],'<f8'),mask,contract)


def uv_mask_values(uv,mask,vertical_policy):
    require(vertical_policy in ['obj_v_bottom','image_v_top'],'Explicit declared UV-row hypothesis required; no auto-selected convention')
    uv=np.asarray(uv,dtype=float)
    require(uv.ndim==2 and uv.shape[1]==2 and np.isfinite(uv).all() and uv.min()>=0 and uv.max()<=1,'Finite unit UV required; no hidden wrapping')
    require(mask.ndim==2 and mask.dtype==np.uint8,'Grayscale raw mask required')
    h,w=mask.shape;x=uv[:,0]*w-.5;y=(1-uv[:,1] if vertical_policy=='obj_v_bottom' else uv[:,1])*h-.5
    # Declared diagnostic GL-like pixel-center sampling/clamp; not inferred shader state.
    x=np.clip(x,0,w-1);y=np.clip(y,0,h-1);x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);ax=x-x0;ay=y-y0
    return (1-ay)*((1-ax)*mask[y0,x0]+ax*mask[y0,x1])+ay*((1-ax)*mask[y1,x0]+ax*mask[y1,x1])


def mask_face_centroid_values(context,vertical_policy):
    require(isinstance(context,Semantics),'Source-verified semantics object required')
    uv=context.target.texcoords[context.target.face_uv_indices].mean(1)
    return uv_mask_values(uv,context.mask,vertical_policy)


def project_cv(vertices,camera):
    vertices=np.asarray(vertices,float);k=np.asarray(camera['K'],float);rt=np.asarray(camera['Rt'],float)
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all(),'Finite source points required')
    points=np.c_[vertices,np.ones(len(vertices))] @ rt.T
    require(np.all(points[:,2]>0),'Points behind/at author camera')
    homogeneous=points @ k.T
    return homogeneous[:,:2]/homogeneous[:,2,None],points[:,2]


def embedded_reference(path):
    _,cells=notebook(path);images=[]
    for output in cells[0][1].get('outputs',[]):
        encoded=output.get('data',{}).get('image/png')
        if encoded:images.append(base64.b64decode(''.join(encoded)))
    require(len(images)==1,'Unique first author TU-demo output required')
    return images[0]


def landmark_reference_check(image,xy):
    rgb=np.asarray(image.convert('RGB'));gray=np.all(rgb==100,axis=2)
    labels,count=label(gray,np.ones((3,3)));objects=find_objects(labels);components=[]
    for number,slices in enumerate(objects,1):
        y,x=slices;area=int(np.sum(labels[slices]==number))
        if area>=150 and x.stop-x.start==21 and y.stop-y.start==21:
            components.append({'center':[(x.start+x.stop-1)/2,(y.start+y.stop-1)/2],'area':area})
    rows=[];matched=set()
    for number,point in enumerate(xy):
        candidates=[(float(np.linalg.norm(np.asarray(c['center'])-point)),i,c) for i,c in enumerate(components)]
        error,index,component=min(candidates,key=lambda row:row[0])
        supported=error<=np.sqrt(.5**2+.5**2)+1e-12
        if supported:matched.add(index)
        rows.append({'author_number':number,'projected_xy':point.tolist(),'nearest_observed_isolated_gray_disk_center':component['center'],
                     'pixel_error':error,'rounding_halfpixel_bound_supported':supported,
                     'unmatched_is_not_anatomical_failure':'merged/overlapping author circles are deliberately not decoded'})
    return {'method':'independent exactRGB100 connected21x21 disks from bound author output; no camera fit or pixel alignment',
            'isolated_disk_component_count':len(components),'supported_isolated_point_count':sum(r['rounding_halfpixel_bound_supported'] for r in rows),
            'distinct_components_matched':len(matched),'points':rows,'full_author_68_semantic_certification':False}


def region_raster(context,camera,vertical_policy):
    """Double-sided CPU geometry/UV diagnostic; independent of original shader."""
    vertices=context.target.vertices;center=vertices.mean(0);angle=np.deg2rad(30)
    rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    vertices=(vertices-center)@rotation.T+center  # Explicit author mask demo rotation, no fit.
    xy,z=project_cv(vertices,camera);h,w=camera['height'],camera['width']
    depth=np.full((h,w),np.inf);gray=np.zeros((h,w),float);occupied=np.zeros((h,w),bool)
    for ids,uv_ids in zip(context.target.faces,context.target.face_uv_indices):
        p=xy[ids];x0=max(0,int(np.ceil(p[:,0].min())));x1=min(w-1,int(np.floor(p[:,0].max())))
        y0=max(0,int(np.ceil(p[:,1].min())));y1=min(h-1,int(np.floor(p[:,1].max())))
        if x1<x0 or y1<y0:continue
        denominator=(p[1,1]-p[2,1])*(p[0,0]-p[2,0])+(p[2,0]-p[1,0])*(p[0,1]-p[2,1])
        if abs(denominator)<1e-12:continue
        y,x=np.mgrid[y0:y1+1,x0:x1+1]
        a=((p[1,1]-p[2,1])*(x-p[2,0])+(p[2,0]-p[1,0])*(y-p[2,1]))/denominator
        b=((p[2,1]-p[0,1])*(x-p[2,0])+(p[0,0]-p[2,0])*(y-p[2,1]))/denominator
        bary=np.stack([a,b,1-a-b],-1);inside=np.all(bary>=-1e-12,axis=-1)
        inverse=bary/z[ids];den=inverse.sum(-1);d=1/den
        selection=inside & (d<depth[y0:y1+1,x0:x1+1])
        if not selection.any():continue
        weights=inverse[selection]/den[selection,None]
        uv=weights @ context.target.texcoords[uv_ids]
        # Numerical boundary roundoff only, not repair/wrapping of authored out-of-range UV.
        require(uv.min()>=-1e-10 and uv.max()<=1+1e-10,'Raster interpolated UV outside source')
        sampled=uv_mask_values(np.clip(uv,0,1),context.mask,vertical_policy)
        yy=y[selection];xx=x[selection];depth[yy,xx]=d[selection];gray[yy,xx]=sampled;occupied[yy,xx]=True
    color=np.full((h,w,3),255,np.uint8);color[occupied]=[130,130,130];color[occupied & (gray>200)]=[40,190,90]
    return Image.fromarray(color),occupied,gray


def run(raw_path,raw_sha,receipt_path,out):
    began=time.perf_counter();out=Path(out).resolve();require(not out.exists(),'New derived-output directory required')
    descriptor=freeze_semantics(raw_path,raw_sha,receipt_path,out/'semantics_contract.json')
    context=load_semantics(descriptor['path'],descriptor['sha256'])
    sources=context.contract['source_metadata'];lm_source=sources['toolkit/demo_landmark.ipynb']['descriptor'];mask_source=sources['toolkit/demo_mask.ipynb']['descriptor']
    author_lm=embedded_reference(lm_source['path']);lm_path=out/'author_landmark_reference.png';lm_path.write_bytes(author_lm)
    with Image.open(io.BytesIO(author_lm)) as im:reference=im.convert('RGB')
    camera=context.contract['landmark_demo_camera'];require(reference.size==(camera['width'],camera['height']),'Author PNG camera dimensions differ')
    xy,z=project_cv(context.landmark_points,camera)
    lm_check=landmark_reference_check(reference,xy)
    overlay=reference.copy();draw=ImageDraw.Draw(overlay)
    for number,(x,y) in enumerate(xy):
        draw.ellipse((x-4,y-4,x+4,y+4),outline=(255,0,255),width=1)
        draw.text((x+11,y-5),str(number),fill=(255,0,255),stroke_width=1,stroke_fill='white')
    overlay.save(out/'independent_landmark_overlay.png')
    author_mask=embedded_reference(mask_source['path']);mask_path=out/'author_mask_reference.png';mask_path.write_bytes(author_mask)
    mask_image=np.asarray(Image.open(io.BytesIO(author_mask)).convert('RGB'));mask_camera=context.contract['mask_demo_camera']
    require(mask_image.shape[:2]==(mask_camera['height'],mask_camera['width']*2),'Expected unmodified two-panel author mask image')
    left,right=mask_image[:,:mask_camera['width']],mask_image[:,mask_camera['width']:]
    original_occupied=np.any(left!=255,axis=2);author_kept=np.any(right!=255,axis=2)
    hypotheses=[]
    for policy in ['obj_v_bottom','image_v_top']:
        raster,occupied,gray=region_raster(context,mask_camera,policy);path=out/(policy+'_region_geometry.png');raster.save(path)
        kept=occupied & (gray>200);union=int(np.sum(kept|author_kept));intersection=int(np.sum(kept&author_kept))
        values=mask_face_centroid_values(context,policy)
        hypotheses.append({'vertical_policy':policy,'raster':binding(path),'cpu_occupied_pixels':int(occupied.sum()),'cpu_mask_white_retained_pixels':int(kept.sum()),
                           'author_image_original_occupied_pixels':int(original_occupied.sum()),'author_image_nonwhite_retained_pixels':int(author_kept.sum()),
                           'diagnostic_retained_mask_IoU':intersection/union if union else None,
                           'face_centroid_white_over200_count':int(np.sum(values>200)),'face_centroid_values_sha256':array_digest(values,'<f8'),
                           'axis_convention_certified':False,'not_original_render':'double-sided software zbuffer, no shader/light/AA or texture import reproduction; no image alignment',
                           'source_target_crop_or_region_selection_used':False})
    landmarks=[]
    for number,(index,point,uv,depth) in enumerate(zip(context.landmark_indices,context.landmark_points,xy,z)):
        landmarks.append({'author_number':number,'source_vertex_index':int(index),'source_point':point.tolist(),'projection_xy':uv.tolist(),'camera_depth':float(depth),
                          'in_image':bool(0<=uv[0]<camera['width'] and 0<=uv[1]<camera['height']),
                          'attachment':context.contract['landmarks']['vertex_attachments'][number]})
    report={'contract':descriptor,'raw_target_contract':context.contract['raw_contract'],'loader':binding(__file__),
            'author_reference_provenance':{'landmark':{'notebook':lm_source,'first_code_output_image':binding(lm_path)},'mask':{'notebook':mask_source,'first_code_output_image':binding(mask_path)}},
            'landmarks':landmarks,'author_number_definition_camera':camera,'independent_landmark_check':lm_check,
            'landmark_overlay':binding(out/'independent_landmark_overlay.png'),'mask_hypotheses':hypotheses,
            'mask_camera':mask_camera,'explicit_source_demo_yaw_degrees':30,'mask_demo_rotation_anchor':'raw vertex mean as author utility.py; no fit',
            'full_source_triangle_count_retained':len(context.target.faces),'topology_or_target_changed':False,
            'named_anatomical_table_available':False,'hs2_semantic_correspondence_certified':False,'physical_units_or_anatomical_axes_certified':False,
            'no_future_detector_candidates_or_HS2_results_used':True,'elapsed_seconds':time.perf_counter()-began}
    freeze(out/'verification.json',report);print(json.dumps({'contract':descriptor,'isolated_author_disks_supported':lm_check['supported_isolated_point_count'],'UV_hypothesis_IoU':[(r['vertical_policy'],r['diagnostic_retained_mask_IoU']) for r in hypotheses],'elapsed':report['elapsed_seconds']}))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw-contract',type=Path,required=True)
    parser.add_argument('--raw-sha256',required=True);parser.add_argument('--upstream-receipt',type=Path,required=True);parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args();run(args.raw_contract,args.raw_sha256,args.upstream_receipt,args.out_dir)


if __name__=='__main__':main()
