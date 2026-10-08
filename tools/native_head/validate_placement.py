"""Finite static acceptance of a source similarity and its rebuilt neck mesh."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from tools.geometry_quality.mesh_quality import self_intersections
from tools.model_bridge.artifact import sha
from tools.native_head.neck_geometry import ordered_loops
from tools.native_head.placement_review import apply


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--masks',type=Path,default=Path(r'C:\Users\13666\Workspace\smirk\assets\FLAME_masks\FLAME_masks.pkl'))
    args=p.parse_args();b=dict(np.load(args.baseline/'geometry.npz',allow_pickle=False));c=dict(np.load(args.candidate/'geometry.npz',allow_pickle=False))
    r=json.loads((args.candidate/'receipt.json').read_text())
    if r['source_baseline_geometry_sha256']!=sha(args.baseline/'geometry.npz') or r['geometry_sha256']!=sha(args.candidate/'geometry.npz') or r['masks_sha256']!=sha(args.masks):raise ValueError('Placement provenance mismatch')
    m=np.asarray(r['similarity_matrix']);size=r['relative_uniform_scale']
    if not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3)*size*size,rtol=0,atol=1e-12):raise ValueError('Placement contains anisotropic deformation')
    raw=b['original_vertices'].astype(float)*r['scale']+r['translation']
    expected=apply(raw,m)
    if not np.allclose(c['placed_original_vertices'],expected,atol=1e-12,rtol=0):raise ValueError('Original head is not the declared similarity')
    if not np.array_equal(c['faces'],b['faces']):raise ValueError('Unexpected integrated topology change')
    v,f=c['vertices'],c['faces'];seam=np.asarray(r['native_outer_ring_authored_ids'])
    if not np.array_equal(v[seam],b['vertices'][seam]):raise ValueError('Fixed native interface moved')
    masks=pickle.loads(args.masks.read_bytes(),encoding='latin1')
    protected=np.unique(np.r_[masks['face'],masks['left_ear'],masks['right_ear'],masks['left_eyeball'],masks['right_eyeball'],masks['eye_region']])
    protected=np.unique(b['original_faces'][np.isin(b['original_faces'],protected).any(1)])
    old=c['crop_original_ids'];locked=(old>=0)&np.isin(old,protected)
    error=float(np.abs(v[:len(old)][locked]-expected[old[locked]]).max())
    if error>1e-12:raise ValueError('Protected face acquired a local deformation')
    tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if areas.min()<=1e-12:raise ValueError('Degenerate connection triangle')
    reports=[]
    for points in (b['vertices'],v):reports.append(self_intersections(points,f,points[f],np.ones(len(f),bool),1e-8))
    true_pairs=lambda report:{tuple(sorted(x['triangles'])) for x in report['pairs'] if x['kind'] in ('proper_crossing','coplanar_overlap')}
    added=sorted(true_pairs(reports[1])-true_pairs(reports[0]))
    neck=[x for x in reports[1]['pairs'] if x['kind'] in ('proper_crossing','coplanar_overlap') and max(x['triangles'])>=r['source_face_count']]
    output=dict(format='head_placement_static_acceptance_v1',geometry_sha256=sha(args.candidate/'geometry.npz'),
        protected_face_similarity_error=error,native_interface_unchanged=True,uniform_similarity=True,
        topology_unchanged=True,boundary_sizes=[len(x) for x in ordered_loops(f)],minimum_double_area=float(areas.min()),
        added_true_crossings=added,neck_region_true_crossings=neck,existing_source_crossings=reports[1]['counts'],
        pass_static_geometry=not added and not neck,
        scope='Authored neutral similarity and rebuilt geometry only; native asset, render normals, materials, expressions and photo identity accuracy unverified')
    (args.candidate/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('pass_static_geometry','uniform_similarity','native_interface_unchanged')}))
    if not output['pass_static_geometry']:raise ValueError('New neck geometry intersects')


if __name__=='__main__':main()
