"""Author a native closed-mouth endpoint from recovered virtual contact pairs.

Complete identity vertices remain unchanged in a separate input artifact. The
computed delta belongs to the Close expression, not the identity shape basis.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from tools.native_head.mouth_identity_transfer import edges_and_graph
from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import source_file,save_json

def close_adjustment(reference,identity,contact,free,anchors,tolerance=1e-9,faces=None):
    free=np.asarray(free,int)
    constraints=sparse.vstack([contact,anchors]).tocsr()
    rhs=np.vstack([-contact@(identity-reference),np.zeros((anchors.shape[0],3))])
    matrix=constraints[:,free].toarray()
    if faces is None:
        correction,residuals,rank,singular=np.linalg.lstsq(matrix,rhs,rcond=None)
        policy='Minimum Euclidean correction'
    else:
        # Smooth expression motion on the intact native surface, while keeping
        # all recovered contacts/outer embeddings exact. No identity column is
        # removed; this is authored Close-frame motion, never bind identity.
        edges,_=edges_and_graph(faces,len(reference));a,b=edges.T
        lengths=np.linalg.norm(reference[a]-reference[b],axis=1)
        if np.any(lengths<=0):raise ValueError('Degenerate physical native edge')
        w=sparse.coo_matrix((np.r_[1/lengths,1/lengths],(np.r_[a,b],np.r_[b,a])),shape=(len(reference),)*2).tocsr()
        L=sparse.diags(np.asarray(w.sum(1)).ravel())-w
        Q=(L[:,free].T@L[:,free]).tocsc()
        factor=splu(Q)
        u,s,vt=np.linalg.svd(matrix,full_matrices=False)
        rank=int(np.sum(s>max(matrix.shape)*np.finfo(float).eps*s[0]))
        reduced=vt[:rank];target=(u[:,:rank].T@rhs)/s[:rank,None]
        dual=factor.solve(reduced.T)
        correction=dual@np.linalg.solve(reduced@dual,target)
        policy='Minimum squared native surface-Laplacian Close motion with exact original virtual contacts and outer embeddings; fixed exterior'
    error=float(np.max(np.abs(matrix@correction-rhs))) if len(rhs) else 0.
    if error>tolerance:
        raise ValueError(f"Native close-contact and full outer-lip constraints conflict; residual {error}")
    delta=np.zeros_like(identity);delta[free]=correction
    if np.max(np.abs(anchors@delta),initial=0)>tolerance:
        raise ValueError("Closed-pose correction changed outer lip anchors")
    return delta,dict(rank=int(rank),constraint_count=constraints.shape[0],variable_vertices=len(free),
        maximum_constraint_residual=error,tolerance=tolerance,
        maximum_closed_pose_adjustment=float(np.linalg.norm(correction,axis=1).max(initial=0)),
        policy=policy+'; no penalty/gain tuning',
        source_identity_modified=False)

def author(candidate,contract,out):
    if out.exists():raise FileExistsError("Preserve prior closure candidates")
    r=json.loads((candidate/'receipt.json').read_text())
    c=json.loads((contract/'receipt.json').read_text())
    if r['inputs']['sha256']!=c['inputs']['sha256'] or r['mother_reference']['sha256']!=c['mother']['sha256']:
        raise ValueError('Closed-pose contract belongs to different donor/mother')
    for row in (c['matrix'],c['pairs'],c['source_references'],r['candidate_arrays'],r['mother_reference'],r['identity_transfer']['mouth_correspondence']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Native contact data changed')
    a=dict(np.load(candidate/'o_head_candidate.npz',allow_pickle=False))
    mother=Path(r['mother_reference']['path']).parent
    m=np.load(mother/'o_head_candidate.npz',allow_pickle=False)
    refs=np.load(c['source_references']['path'],allow_pickle=False)
    unique,inverse,faces=refs['unique_render_ids'],refs['render_to_logical'],refs['faces']
    if not np.array_equal(inverse[a['faces']],faces):raise ValueError('Native mouth topology changed')
    contact=sparse.load_npz(c['matrix']['path'])
    pairs=json.loads(Path(c['pairs']['path']).read_text())
    mapping=np.load(candidate/'mouth_correspondence.npz',allow_pickle=False)
    free=np.union1d(np.r_[mapping['lip_ids'],mapping['halo_ids'],mapping['oral_interior_ids']],
        faces[np.unique([face for pair in pairs for face in pair['face_ids']])])
    rows=[];cols=[];values=[];count=0
    for annotation in r['candidate_anatomical_constraints']['annotations']:
        if 48<=annotation['target_flame_landmark']<=59 and 'face_id' in annotation:
            rows.extend([count]*3);cols.extend(faces[annotation['face_id']]);values.extend(annotation['barycentric']);count+=1
    if count!=12:raise ValueError('Full original outer-lip embedding is required')
    anchors=sparse.coo_matrix((values,(rows,cols)),shape=(count,len(unique))).tocsr()
    delta,report=close_adjustment(m['verts'][unique],a['verts'][unique],contact,free,anchors,faces=faces)
    authored=a['verts']+delta[inverse]
    unaffected=~np.isin(inverse,free)
    if not np.array_equal(authored[unaffected],a['verts'][unaffected]):raise ValueError('Close action changed exterior')
    out.mkdir(parents=True)
    np.savez_compressed(out/'closed_pose_adjustment.npz',vertex_delta=delta[inverse],logical_vertex_delta=delta,
        full_identity_reference_vertices=a['verts'],free_logical_ids=free,outer_lip_constraint_matrix=anchors.toarray())
    np.savez_compressed(out/'o_head_candidate.npz',**{**a,'verts':authored})
    r.update(format='native_mother_full_identity_close_pose_candidate_v1',
        candidate_arrays=source_file(out/'o_head_candidate.npz'),closed_pose_authoring=dict(
            source_identity_candidate=source_file(candidate/'receipt.json'),full_identity_arrays=source_file(candidate/'o_head_candidate.npz'),
            original_contact_contract=source_file(contract/'receipt.json'),adjustment=source_file(out/'closed_pose_adjustment.npz'),
            solver=report,pattern=c['pattern'],identity_field_and_source_coefficients_unchanged=True,
            bind_and_original_closed_frame_integration_required=True,
            pending='Correction must be authored into the original Close frame, with normals/tangents; never bake it into identity bind vertices'),
        deliverable=False,installed=False,game_mutated=False)
    save_json(out/'receipt.json',r)
    print(json.dumps(dict(output=str(out.resolve()),identity_field_unchanged=True,static_acceptance_required=True)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','contract','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();author(a.candidate,a.contract,a.out)
