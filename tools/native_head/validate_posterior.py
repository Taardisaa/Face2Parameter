"""Finite final static acceptance of one authored posterior surface."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.geometry_quality.mesh_quality import self_intersections
from tools.model_bridge.artifact import sha
from tools.native_head.neck_geometry import ordered_loops


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    args=p.parse_args()
    c=dict(np.load(args.candidate/'geometry.npz',allow_pickle=False))
    b=dict(np.load(args.baseline/'geometry.npz',allow_pickle=False))
    receipt=json.loads((args.candidate/'receipt.json').read_text())
    if receipt['baseline_geometry_sha256']!=sha(args.baseline/'geometry.npz'):
        raise ValueError('Baseline provenance mismatch')
    v,f=c['vertices'],c['faces'];bv=b['vertices']
    if not np.array_equal(f,b['faces']) or not np.array_equal(c['before_vertices'],bv):
        raise ValueError('Unexpected topology/baseline changes')
    seam=np.asarray(receipt['native_outer_ring_authored_ids'])
    front=bv[:,2]>=receipt['posterior_design']['side_front']
    frozen=c['posterior_fade']==0
    if not np.array_equal(v[seam],bv[seam]) or not np.array_equal(v[front],bv[front]) or not np.array_equal(v[frozen],bv[frozen]):
        raise ValueError('Interface/front/outside edit area changed')
    tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if areas.min()<=1e-12:raise ValueError('Degenerate authored triangle')
    loops=ordered_loops(f)
    reports=[]
    for points in (bv,v):
        reports.append(self_intersections(points,f,points[f],np.ones(len(f),bool),1e-8))
    crossing=lambda r:{tuple(sorted(x['triangles'])) for x in r['pairs'] if x['kind'] in ('proper_crossing','coplanar_overlap')}
    added=crossing(reports[1])-crossing(reports[0])
    changed=np.linalg.norm(v-bv,axis=1)>1e-12
    affected=[x for x in reports[1]['pairs'] if x['kind'] in ('proper_crossing','coplanar_overlap') and changed[f[x['triangles']]].any()]
    output=dict(format='posterior_neck_static_acceptance_v1',geometry_sha256=sha(args.candidate/'geometry.npz'),
        baseline_sha256=sha(args.baseline/'geometry.npz'),topology_unchanged=True,
        interface_positions_unchanged=True,front_positions_unchanged=True,outside_patch_positions_unchanged=True,
        minimum_double_area=float(areas.min()),boundary_sizes=[len(x) for x in loops],
        added_true_crossings=sorted(added),edited_region_crossings=affected,
        baseline_existing_crossings=reports[0]['counts'],candidate_existing_crossings=reports[1]['counts'],
        pass_static_posterior=not added and not affected,
        scope='Posterior neutral mesh authoring only. Existing source-eye/mouth intersections retained; no whole-mesh-clean, material, rig or game acceptance claim.')
    (args.candidate/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('pass_static_posterior','front_positions_unchanged','interface_positions_unchanged')}))
    if not output['pass_static_posterior']:raise ValueError('Authored posterior surface has new intersections')


if __name__=='__main__':main()
