"""Fixed FLAME identity-basis transfer onto an authored native mother reference.

This is a new asset-authoring correspondence, not an HS2 slider approximation.
Once built, 300 coefficient application is linear and deterministic. The native
mother's surface offsets and oral structure remain; the actual body rim is pinned.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

from tools.model_bridge.artifact import ModelArtifact, sha
from tools.model_bridge.attachment_audit import topology
from tools.model_bridge.scan_accuracy import TriangleSurface
from tools.native_head.mother_oriented_surface import OrientedSurface, logical_normals
from tools.native_head.mother_template_inputs import save_json, source_file


def barycentric(points, triangles):
    a=triangles[:,1]-triangles[:,0]
    b=triangles[:,2]-triangles[:,0]
    p=points-triangles[:,0]
    dot=lambda x,y: np.einsum('ij,ij->i',x,y)
    aa,bb,ab=dot(a,a),dot(b,b),dot(a,b)
    determinant=aa*bb-ab*ab
    if np.any(determinant<=0):
        raise ValueError('Degenerate source correspondence triangle')
    u=(bb*dot(a,p)-ab*dot(b,p))/determinant
    v=(aa*dot(b,p)-ab*dot(a,p))/determinant
    return np.column_stack([1-u-v,u,v])


def neck_pin_basis(vertices, faces, basis, neck, transition_rings=6):
    """Zero rim displacement with a Dirichlet correction confined to neck rings.

    Ring count is an explicit new-asset transition design setting, not a guessed
    game gain. Outside this topological strip the FLAME displacement is untouched.
    Positive edge weights use actual mother edge lengths.
    """
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1)
    if np.any(length<=0):
        raise ValueError('Zero logical edge')
    a,b=edges.T
    adjacency=sparse.coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(vertices),)*2).tocsr()
    distance=dijkstra(adjacency,directed=False,indices=neck,min_only=True,unweighted=True)
    interior=np.flatnonzero((distance>0)&(distance<transition_rings))
    weights=sparse.coo_matrix((np.r_[1/length,1/length],(np.r_[a,b],np.r_[b,a])),shape=adjacency.shape).tocsr()
    L=sparse.diags(np.asarray(weights.sum(1)).ravel())-weights
    flat=basis.reshape(len(vertices),-1)
    correction=np.zeros_like(flat)
    correction[neck]=-flat[neck]
    if len(interior):
        correction[interior]=spsolve(L[interior][:,interior].tocsc(),-L[interior][:,neck]@correction[neck])
    result=(flat+correction).reshape(basis.shape)
    result[neck]=0
    return result,dict(rim=neck.tolist(),interior=interior.tolist(),transition_rings=transition_rings,
        policy='Positive inverse-edge-length harmonic displacement correction; fixed rim and fixed outside strip')


def coherent_mouth(vertices, faces, transferred, core, landmark_positions, landmark_basis):
    """Preserve native closed lip/oral layer ordering under one local affine field.

    FLAME has no native closed-mouth interior. Its 20 actual mouth landmarks
    determine a geometric affine component for each shape basis column. This
    deliberate local detail-preservation policy discards non-affine FLAME lip
    residuals, records them, and cannot be called an exact lip reconstruction.
    The surrounding two-edge transition blends into the original source field.
    """
    design=np.column_stack([landmark_positions,np.ones(len(landmark_positions))])
    affine=np.linalg.lstsq(design,landmark_basis.reshape(len(design),-1),rcond=None)[0]
    common=(np.column_stack([vertices,np.ones(len(vertices))])@affine).reshape(transferred.shape)
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    a,b=edges.T
    graph=sparse.coo_matrix((np.ones(len(a)*2),(np.r_[a,b],np.r_[b,a])),shape=(len(vertices),)*2).tocsr()
    distance=dijkstra(graph,directed=False,indices=core,min_only=True,unweighted=True)
    t=np.clip(distance/3,0,1)
    weight=1-t*t*(3-2*t)
    result=transferred*(1-weight[:,None,None])+common*weight[:,None,None]
    residual=landmark_basis.reshape(len(design),-1)-design@affine
    return result,dict(core=core.tolist(),transition=np.flatnonzero((weight>0)&(weight<1)).tolist(),
        policy='Shared affine mouth-landmark identity displacement in all native mouth-rig support and oral interior; cubic three-edge transition outside',
        nonaffine_source_basis_residual_norm=float(np.linalg.norm(residual)),
        exact_flame_lip_reconstruction=False)


def build(inputs, mother_dir, manifest, image_index, out, mouth_detail_preview=False):
    if out.exists():
        raise FileExistsError('Preserve identity candidates; use a fresh output')
    mother_receipt=json.loads((mother_dir/'receipt.json').read_text())
    if sha(inputs/'receipt.json')!=mother_receipt['inputs']['sha256']:
        raise ValueError('Different complete donor')
    mother=dict(np.load(mother_dir/'o_head_candidate.npz',allow_pickle=False))
    regions=json.loads((inputs/'native_regions.json').read_text())
    source=ModelArtifact(manifest,image_index)
    state=source.state
    if source.manifest['format']!='mica_flame_raw_export_v1':
        raise ValueError('This path uses official MICA neutral canonical output')
    if np.any(state['eye_pose']) or np.any(state['neck_pose']):
        raise ValueError('Nonzero decoder pose requires upstream posed basis; do not silently linearize')
    original=dict(np.load(inputs/'flame_zero_identity.npz',allow_pickle=False))
    for key in ('v_template','shapedirs','faces_tensor','J_regressor'):
        if not np.array_equal(original[key],state[key]):
            raise ValueError('Mother and identity FLAME state differ: '+key)
    beta=source.parameters['shape_params'][0]
    basis=state['shapedirs'][:,:,:300].astype(float)
    decoded=state['v_template'].astype(float)+np.einsum('vck,k->vc',basis,beta)
    # Official saved output is float32 LBS. The zero-pose linear identity path
    # above is its source equation; verify against the unchanged exported result.
    error=float(np.max(np.abs(decoded-source.arrays['output__pred_canonical_shape_vertices'][0])))
    if error>1e-7:
        raise ValueError('Official canonical output does not match zero-pose identity equation')
    place=mother_receipt['placement']
    rotation=np.asarray(place['rotation'])
    scale=place['scale']
    target=mother['reference_vertices']
    placed_basis=np.einsum('ab,vbk->vak',rotation,basis)*scale
    aliases={int(k):int(v) for k,v in regions['graph_aliases'].items()}
    canonical=np.array([aliases.get(i,i) for i in range(len(mother['verts']))])
    unique,inverse=np.unique(canonical,return_inverse=True)
    vertices=mother['verts'][unique]
    faces=inverse[mother['faces']]
    norms=logical_normals(vertices,faces)
    flame_faces=state['faces_tensor']
    components=topology(state['v_template'],flame_faces)
    shell=next(c['vertices'] for c in components['components'] if len(c['vertices'])==max(len(c['vertices']) for c in components['components']))
    selected=np.flatnonzero(np.isin(flame_faces,shell).all(1))
    region_by_node=np.full(len(unique),'general',dtype='U16')
    allowed={'general':selected}
    for label,row in mother_receipt['regional_targets']['regions'].items():
        region_by_node[inverse[row['native_vertex_ids']]]=label
        source_faces=np.asarray(row['source_faces'])
        lookup={tuple(f):i for i,f in enumerate(flame_faces)}
        allowed[label]=np.array([lookup[tuple(f)] for f in source_faces])
    # The separate native oral cavity has no FLAME interior counterpart.
    # Extend the lips' displacement into it; never project its geometry onto skin.
    lip_landmarks=np.arange(48,68)
    lmk_faces=state['full_lmk_faces_idx'][0][lip_landmarks]
    lip_vertices=np.unique(flame_faces[lmk_faces])
    mouth_source=np.flatnonzero(np.isin(flame_faces,lip_vertices).any(1))
    oral=np.unique(inverse[regions['inner_mouth_component_vertex_ids']])
    region_by_node[oral]='oral_extension'
    allowed['oral_extension']=mouth_source
    chosen=np.empty(len(unique),dtype=int)
    bary=np.empty((len(unique),3))
    reports={}
    for label,source_faces in allowed.items():
        ids=np.flatnonzero(region_by_node==label)
        if not len(ids):
            continue
        triangles=target[flame_faces[source_faces]]
        if label=='oral_extension':
            closest,face_ids,_=TriangleSurface(triangles).closest(vertices[ids])
        else:
            closest,face_ids,_=OrientedSurface(triangles).closest_oriented(vertices[ids],norms[ids])
        chosen[ids]=source_faces[face_ids]
        bary[ids]=barycentric(closest,triangles[face_ids])
        reports[label]=dict(logical_vertex_count=len(ids),max_projection_distance=float(np.linalg.norm(vertices[ids]-closest,axis=1).max()))
    transferred=np.einsum('vi,vick->vck',bary,placed_basis[flame_faces[chosen]])
    # A surface-only field assigns opposed native thin layers to different
    # triangles. Extend the *identity displacement* continuously inside the
    # source shell with P1 tetrahedral interpolation of its original samples.
    # This is a fixed interpolation scaffold, not a replacement volume mesh,
    # a refit of the mother, or a learned prediction. No identity-dependent
    # correspondences are chosen. Outside its convex hull keep the recorded
    # regional surface mapping; record both branches explicitly.
    scaffold=Delaunay(target[shell])
    cells=scaffold.find_simplex(vertices)
    inside=np.flatnonzero(cells>=0)
    tetra_weights=np.zeros((len(vertices),4))
    tetra_ids=np.full((len(vertices),4),-1,dtype=int)
    transforms=scaffold.transform[cells[inside]]
    first=np.einsum('nij,nj->ni',transforms[:,:3],vertices[inside]-transforms[:,3])
    tetra_weights[inside]=np.column_stack([first,1-first.sum(1)])
    tetra_ids[inside]=np.asarray(shell)[scaffold.simplices[cells[inside]]]
    transferred[inside]=np.einsum('vi,vick->vck',tetra_weights[inside],placed_basis[tetra_ids[inside]])
    # Physical eyelid openings use the already audited embedding/polyline mapping,
    # keeping corner identity displacements identical on the upper/lower paths.
    landmark_basis=(placed_basis[flame_faces[state['full_lmk_faces_idx'][0]]]*
        state['full_lmk_bary_coords'][0,:,:,None,None]).sum(1)
    landmark_positions=(target[flame_faces[state['full_lmk_faces_idx'][0]]]*
        state['full_lmk_bary_coords'][0,:,:,None]).sum(1)
    mouth_core=np.unique(inverse[np.union1d(regions['inner_mouth_component_vertex_ids'],
        mother_receipt['regional_targets']['native_mouth_rig_support_vertex_ids'])])
    mouth_report=dict(policy='Full source displacement extension; no lip detail filtering',
                      static_contact_acceptance_required=True)
    if mouth_detail_preview:
        transferred,mouth_report=coherent_mouth(vertices,faces,transferred,mouth_core,
            landmark_positions[48:68],landmark_basis[48:68])
    assigned={}
    for side,row in place['eye_contours'].items():
        for path in row.values():
            xyz=target[flame_faces[state['full_lmk_faces_idx'][0][path['source_landmark_ids']]]]
            xyz=(xyz*state['full_lmk_bary_coords'][0][path['source_landmark_ids'],:,None]).sum(1)
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
            arc/=arc[-1]
            weights=np.stack([np.interp(path['native_arc_fractions'],arc,np.eye(len(arc))[:,i]) for i in range(len(arc))],axis=1)
            curve=np.einsum('ij,jck->ick',weights,landmark_basis[path['source_landmark_ids']])
            for native_id,value in zip(path['native_vertex_ids'],curve):
                index=int(inverse[native_id])
                if index in assigned and not np.allclose(assigned[index],value,rtol=0,atol=1e-12):
                    raise ValueError('Eye-corner identity mapping disagrees')
                assigned[index]=value
                transferred[index]=value
    neck=np.unique(inverse[regions['neck_boundary_all_render_copies']])
    transferred,neck_report=neck_pin_basis(vertices,faces,transferred,neck)
    native_basis=transferred[inverse]
    delta=np.einsum('vck,k->vc',native_basis,beta)
    arrays={**mother,'verts':mother['verts']+delta,
            'reference_vertices':decoded@rotation.T*scale+place['translation']}
    if not np.array_equal(arrays['verts'][regions['neck_boundary_all_render_copies']],mother['verts'][regions['neck_boundary_all_render_copies']]):
        raise ValueError('Native body rim moved')
    for group in regions['identical_bind_position_and_skin_groups']:
        if not np.array_equal(delta[group],np.broadcast_to(delta[group[0]],(len(group),3))):
            raise ValueError('UV seam identity displacement differs')
    out.mkdir(parents=True)
    np.savez_compressed(out/'identity_basis.npz',native_identity_basis=native_basis,
        source_faces=chosen[inverse],barycentric=bary[inverse],shape_params=beta,
        source_identity_basis=placed_basis,tetrahedron_source_ids=tetra_ids[inverse],
        tetrahedron_weights=tetra_weights[inverse],interior_extension=cells[inverse]>=0)
    np.savez_compressed(out/'o_head_candidate.npz',**arrays)
    receipt=copy.deepcopy(mother_receipt)
    receipt.update(format='native_mother_identity_candidate_v1',code=source_file(Path(__file__)),
        mother_reference=source_file(mother_dir/'receipt.json'),candidate_arrays=source_file(out/'o_head_candidate.npz'),
        candidate_geometry_state='Identity-deformed native closed-reference surface; not bind geometry',
        identity_transfer=dict(basis=source_file(out/'identity_basis.npz'),
            source_manifest=source_file(manifest),image_index=image_index,
            raw_parameters=source_file(manifest.parent/source.image['file']),
            original_input=source.image['input'],source_decoder_revision=source.manifest['git_revision'],
            canonical_source_equation_error=error,coefficient_count=300,
            policy='Fixed P1 tetrahedral interior displacement extension plus regional surface barycentrics outside hull; official 300 identity fields; native offsets retained; exact embedded eyelid contours; pinned body rim',
            displacement_extension=dict(interior_logical_vertices=len(inside),
                outside_hull_logical_vertices=int(np.sum(cells<0)),
                source_scaffold='Delaunay tetrahedra of original FLAME head-shell sample positions; no source faces changed'),
            regions=reports,neck=neck_report,mouth_detail_preservation=mouth_report,
            extra_trained_model=False,hs2_sliders_used=False,
            mouth_detail_preview=mouth_detail_preview,
            flame_lip_detail_filtered=mouth_detail_preview,
            exact_continuous_surface_equivalence=False,
            limitations=['Native mean differs from FLAME; this transfer does not claim identical topology or exact continuous-surface equivalence',
                'Native oral interior receives lip-region displacement extension; no FLAME oral counterpart exists',
                'Native closed lip/interior layers share a mouth-landmark affine component; non-affine FLAME lip details are not transferred exactly',
                'Static geometry and full native integration must be reviewed separately']),
        installed=False,game_mutated=False,deliverable=False)
    save_json(out/'receipt.json',receipt)
    print(json.dumps(dict(output=str(out.resolve()),mica_image=source.image['input'],coefficients=300,native_neck_fixed=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--mother',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--image-index',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mouth-detail-preview',action='store_true',
        help='Explicit incomplete diagnostic: common mouth affine component preserves closed native detail; not exact FLAME lip transfer')
    a=p.parse_args()
    build(a.inputs.resolve(),a.mother.resolve(),a.manifest.resolve(),a.image_index,a.out.resolve(),a.mouth_detail_preview)
