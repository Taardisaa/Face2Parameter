"""Archived whole-head experiment; excluded from the current local repair route.

The user explicitly required preserving the accepted overall head shape. The
kernel_v1 experiment failed geometry/Jacobian checks and was never installed.
Keep this source for provenance, not as a next step or production candidate.

Native triangles are unchanged. A single cubic RBF displacement field, with
affine polynomial and moment constraints, moves all physical points together.
Mesh-barycentric anchors constrain the actual triangles (not a nearby point).
This is new asset authoring, not HS2 runtime-driver replacement or a claim of
perfect FLAME correspondence. Source extrema remain candidate annotations.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

from tools.model_bridge.artifact import sha
from tools.native_head.mother_shell_candidate import eye_constraints
from tools.native_head.mother_surface_targets import MASK_SHA
from tools.native_head.mother_template_inputs import save_json, source_file
from tools.native_head.mother_uv_anchors import constraints as uv_constraints, embed


class CubicWarp:
    def __init__(self, vertices, constraints, targets):
        # Kernel centers are the actual material points defined by the anchors.
        points = np.asarray(constraints@vertices)
        self.origin = points.mean(0)
        self.scale = float(np.ptp(points, axis=0).max())
        if self.scale<=0:
            raise ValueError('Collapsed landmark field')
        self.centers = (points-self.origin)/self.scale
        normalized = (vertices-self.origin)/self.scale
        kernel = cdist(normalized,self.centers)**3
        polynomial = np.c_[np.ones(len(vertices)),normalized]
        moment = np.c_[np.ones(len(points)),self.centers]
        system = np.block([[constraints@kernel, constraints@polynomial],
                           [moment.T,np.zeros((4,4))]])
        right = np.vstack([targets-points,np.zeros((4,3))])
        if np.linalg.matrix_rank(system)!=len(system):
            raise ValueError('Conflicting or redundant material constraints')
        solved = np.linalg.solve(system,right)
        self.weights, self.affine = solved[:-4],solved[-4:]
        self.constraint_residual = float(np.max(np.linalg.norm(system@solved-right,axis=1)))
        if self.constraint_residual>1e-10:
            raise ValueError('Material constraints are not solved accurately')

    def map(self, vertices):
        normalized=(np.asarray(vertices,float)-self.origin)/self.scale
        offset=normalized[:,None]-self.centers[None]
        radius=np.linalg.norm(offset,axis=2)
        displacement=radius**3@self.weights+np.c_[np.ones(len(vertices)),normalized]@self.affine
        # Jacobian rows are output coordinates, columns input coordinates.
        derivative=3*radius[:,:,None]*offset/self.scale
        jacobian=np.eye(3)[None]+np.einsum('nki,kj->nji',derivative,self.weights)+self.affine[1:].T[None]/self.scale
        return np.asarray(vertices,float)+displacement,jacobian

    def arrays(self):
        return dict(kernel_centers=self.centers,kernel_origin=self.origin,kernel_scale=np.asarray(self.scale),
                    kernel_weights=self.weights,kernel_affine=self.affine)


def unique_constraints(matrix, targets):
    seen, selected = {}, []
    for i in range(matrix.shape[0]):
        row=matrix.getrow(i)
        key=(tuple(row.indices),tuple(row.data))
        if key in seen:
            if not np.array_equal(targets[i],targets[seen[key]]):
                raise ValueError('Duplicate source material point has conflicting targets')
        else:
            seen[key]=i;selected.append(i)
    return matrix[selected],targets[selected]


def candidate(inputs,default_reference,out,mask_path,native_profile,texture_manifest):
    if out.exists():
        raise FileExistsError('Preserve earlier candidates; use fresh output')
    source=json.loads((inputs/'receipt.json').read_text())
    default=json.loads((default_reference/'receipt.json').read_text())
    if default['inputs']['sha256']!=sha(inputs/'receipt.json'):
        raise ValueError('Default reference belongs to other inputs')
    native_row=next(r for r in source['all_renderers'] if r['mesh']=='o_head')
    for row in (native_row['array_export'],source['regions'],source['flame_zero_identity']):
        if sha(row['path'])!=row['sha256']:
            raise ValueError('Prepared source changed')
    arrays=dict(np.load(native_row['array_export']['path'],allow_pickle=False))
    native_bind=arrays['verts'].copy()
    ref=next(r for r in default['renderer_references'] if r['mesh']=='o_head')['reference']
    if sha(ref['path'])!=ref['sha256']:
        raise ValueError('Original default reference changed')
    native=np.load(ref['path'])['closed_reference_vertices'].astype(float)
    arrays['verts']=native
    regions=json.loads(Path(source['regions']['path']).read_text())
    flame=dict(np.load(source['flame_zero_identity']['path'],allow_pickle=False))
    fv,ff=flame['v_template'].astype(float),flame['faces_tensor']
    landmarks=(fv[ff[flame['full_lmk_faces_idx'][0]]]*flame['full_lmk_bary_coords'][0,:,:,None]).sum(1)
    target,eyes,placement=eye_constraints(native,regions,fv,landmarks)
    target_landmarks=landmarks*placement['scale']+placement['translation']
    aliases={int(k):int(v) for k,v in regions['graph_aliases'].items()}
    canonical=np.array([aliases.get(i,i) for i in range(len(native))])
    unique,inverse=np.unique(canonical,return_inverse=True)
    vertices=native[unique]
    pinned={int(inverse[i]):native[i] for i in regions['neck_boundary_all_render_copies']}
    pinned.update({int(inverse[i]):p for i,p in eyes.items()})
    ids=np.asarray(sorted(pinned))
    rows=[sparse.csr_matrix((np.ones(len(ids)),(np.arange(len(ids)),ids)),shape=(len(ids),len(unique)))]
    values=[np.asarray([pinned[i] for i in ids])]
    matrix,anchors,annotation=uv_constraints(arrays,inverse,target_landmarks,native_profile,texture_manifest)
    rows.append(matrix);values.append(anchors)
    profile=json.loads(native_profile.read_text())
    outline=[]
    for landmark in (0,8,16):
        ids0,bary,note=embed(profile['landmark_uv'][str(landmark)],arrays)
        rows.append(sparse.csr_matrix((bary,(np.zeros(3,int),inverse[ids0])),shape=(1,len(unique))))
        # Native texture U reverses mesh X, as in the original profile.
        target_id=16-landmark
        values.append(target_landmarks[target_id:target_id+1])
        outline.append(dict(**note,target_flame_landmark=target_id,semantically_reviewed=False))
    if sha(mask_path)!=MASK_SHA:
        raise ValueError('Original FLAME masks changed')
    masks=pickle.loads(mask_path.read_bytes(),encoding='latin1')
    ear_anchors=[]
    for side in ('L','R'):
        bones=[i for i,name in enumerate(arrays['bone_names']) if 'Ear' in name and name.endswith('_'+side)]
        support=(arrays['bone_w']*np.isin(arrays['bone_idx'],bones)).sum(1)
        ear=np.flatnonzero(support>.5)
        sign=np.sign(native[ear,0].mean())
        key=next(k for k in ('left_ear','right_ear') if np.sign(target[masks[k],0].mean())==sign)
        target_ear=np.asarray(masks[key],int)
        for axis in range(3):
            for name,func in (('min',np.argmin),('max',np.argmax)):
                a=int(ear[func(native[ear,axis])]);b=int(target_ear[func(target[target_ear,axis])])
                rows.append(sparse.csr_matrix(([1.],([0],[inverse[a]])),shape=(1,len(unique))))
                values.append(target[b:b+1])
                ear_anchors.append(dict(side=side,axis=axis,extremum=name,native_vertex=a,source_vertex=b,
                    policy='Candidate whole-ear extremum annotation from native bone support and original source ear mask'))
    C,positions=unique_constraints(sparse.vstack(rows,format='csr'),np.vstack(values))
    field=CubicWarp(vertices,C,positions)
    authored,jacobian=field.map(vertices)
    determinant=np.linalg.det(jacobian)
    # Retain diagnostic failure rather than treating sampled Jacobians as a
    # global diffeomorphism proof or silently accepting a bad field.
    for i in regions['neck_boundary_all_render_copies']:
        authored[inverse[i]]=native[i]
    result=authored[inverse]
    for group in regions['identical_bind_position_and_skin_groups']:
        if not np.all(result[group]==result[group[0]]):
            raise ValueError('Physical source copies separated')
    residual=float(np.linalg.norm(C@authored-positions,axis=1).max())
    if residual>1e-10:
        raise ValueError('Head material constraints do not match final triangles')
    out.mkdir(parents=True)
    path=out/'o_head_candidate.npz'
    np.savez_compressed(path,**{**arrays,'verts':result},original_vertices=native,
        original_bind_vertices=native_bind,reference_vertices=target,reference_faces=ff,
        authoring_jacobian=jacobian[inverse],**field.arrays())
    save_json(out/'receipt.json',dict(format='native_mother_kernel_candidate_v1',
        inputs=source_file(inputs/'receipt.json'),default_reference=source_file(default_reference/'receipt.json'),
        code=source_file(Path(__file__)),placement=placement,candidate_arrays=source_file(path),
        candidate_anatomical_constraints=annotation,outline_annotations=outline,ear_annotations=ear_anchors,
        source_masks=source_file(mask_path),constraint_residual=residual,
        minimum_material_jacobian_determinant=float(determinant.min()),
        material_jacobians_positive=bool((determinant>0).all()),
        algorithm='One cubic RBF displacement field with affine polynomial/moment constraints and actual mesh-barycentric landmarks',
        cut_faces=[],welded_vertices=False,neck_boundary_preserved_exactly=True,
        topology_and_uv_unchanged=True,installed=False,deliverable=False,game_mutated=False,
        pending=['Complete anatomical correspondence and target surface accuracy',
                 'Full surface intersection and component adaptation', 'Actual body shading and native integration'],
        limitations=['Ear extrema and original atlas outline are provisional anatomical annotations',
                     'Positive vertex Jacobians are not a global field/surface nonintersection certificate',
                     'Landmark matching is not exact target surface matching or an identity-basis transfer']))
    print(json.dumps(dict(output=str(out.resolve()),smooth_material_field=True,
        material_jacobians_positive=bool((determinant>0).all()),deliverable=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--default-reference',type=Path,default=Path('outputs/native_mother_template_20261008/default_reference_v1'))
    parser.add_argument('--source-masks',type=Path,default=Path('../smirk/assets/FLAME_masks/FLAME_masks.pkl'))
    parser.add_argument('--native-profile',type=Path,default=Path('tools/native_head/native_atlas_profile.json'))
    parser.add_argument('--texture-manifest',type=Path,default=Path('../HS2Mod/artifacts/chenger/native_skin_retarget_20261008/manifest.json'))
    args=parser.parse_args()
    candidate(args.inputs.resolve(),args.default_reference.resolve(),args.out.resolve(),args.source_masks.resolve(),
              args.native_profile.resolve(),args.texture_manifest.resolve())
