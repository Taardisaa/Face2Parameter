"""Merge a local full-detail mouth into an already authored native identity.

Only source-declared mouth support changes. Existing neck/skull/appearance
authoring remains exact. This entry supports a rigid local mouth placement;
other local edits require an explicit subsequent correspondence repair.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import source_file,save_json

def verify_digest(row):
    if sha(row['path'])!=row['sha256']:raise ValueError('Source changed: '+row['path'])
def verified(row):
    verify_digest(row)
    return json.loads(Path(row['path']).read_text())

def merge(accepted,identity_reference,mouth,out):
    if out.exists():raise FileExistsError('Preserve accepted and failed assets')
    ar=json.loads((accepted/'receipt.json').read_text());old=json.loads((identity_reference/'receipt.json').read_text());new=json.loads((mouth/'receipt.json').read_text())
    for r in (ar,old,new):
        verify_digest(r['candidate_arrays']);verified(r['inputs']);verified(r['mother_reference'])
    if len({r['inputs']['sha256'] for r in (ar,old,new)})!=1 or len({r['mother_reference']['sha256'] for r in (ar,old,new)})!=1:
        raise ValueError('Mouth and accepted identity require the same source donor/mother')
    accepted_arrays=dict(np.load(accepted/'o_head_candidate.npz',allow_pickle=False))
    baseline=np.load(identity_reference/'o_head_candidate.npz',allow_pickle=False)
    closed=np.load(mouth/'o_head_candidate.npz',allow_pickle=False)
    for field in ('faces','uv','bone_idx','bone_w','original_vertices'):
        if not np.array_equal(accepted_arrays[field],baseline[field]) or not np.array_equal(baseline[field],closed[field]):
            raise ValueError('Native source topology/attributes differ: '+field)
    for r in (old,new):
        verify_digest(r['identity_transfer']['basis'])
    old_basis=np.load(old['identity_transfer']['basis']['path'],allow_pickle=False)
    new_basis=np.load(new['identity_transfer']['basis']['path'],allow_pickle=False)
    if not np.array_equal(old_basis['shape_params'],new_basis['shape_params']):
        raise ValueError('This is local mouth completion, not replacement with another identity')
    action=verified(new['closed_pose_authoring']['source_identity_candidate'])
    mapping=np.load(action['identity_transfer']['mouth_correspondence']['path'],allow_pickle=False)
    adjustment=np.load(new['closed_pose_authoring']['adjustment']['path'],allow_pickle=False)
    if sha(new['closed_pose_authoring']['adjustment']['path'])!=new['closed_pose_authoring']['adjustment']['sha256']:
        raise ValueError('Closed action changed')
    regions=verified(json.loads(Path(ar['inputs']['path']).read_text())['regions'])
    canonical=np.array([int(regions['graph_aliases'].get(str(i),i)) for i in range(len(closed['verts']))])
    unique,inverse=np.unique(canonical,return_inverse=True)
    old_mouth=old['identity_transfer']['mouth_detail_preservation']
    logical=np.unique(np.r_[old_mouth.get('core',[]),old_mouth.get('transition',[]),mapping['lip_ids'],mapping['halo_ids'],mapping['oral_interior_ids'],adjustment['free_logical_ids']]).astype(int)
    selected=np.isin(inverse,logical)
    # No value-based clipping/filtering: support comes solely from the exact
    # source methods. Outside it, the two calculations must agree to roundoff.
    count=len(old_basis['shape_params'])
    gamma=count*np.finfo(float).eps/(1-count*np.finfo(float).eps)
    accumulated=np.abs(baseline['verts'])+np.einsum('vck,k->vc',np.abs(old_basis['native_identity_basis']),np.abs(old_basis['shape_params']))
    accumulated+=np.einsum('vck,k->vc',np.abs(new_basis['native_identity_basis']),np.abs(new_basis['shape_params']))
    magnitude=max(1.,float(accumulated.max()))
    bound=gamma*magnitude
    identity=adjustment['full_identity_reference_vertices']
    if np.max(abs((identity-baseline['verts'])[~selected]),initial=0)>bound or np.max(abs((closed['verts']-baseline['verts'])[~selected]),initial=0)>bound:
        raise ValueError('Nonlocal source identity change; preserve full data for explicit repair')
    ap,bp=ar['placement'],old['placement']
    linear=np.asarray(ap['rotation'])@np.asarray(bp['rotation']).T*(ap['scale']/bp['scale'])
    translation=np.asarray(ap['translation'])-linear@np.asarray(bp['translation'])
    neck=verified(ar['neck_transition']['domain'])
    protected_neck=np.unique(np.r_[neck['free_render_ids'],neck['lower_native_render_ids']]).astype(int)
    if np.max(abs((closed['verts']-baseline['verts'])[protected_neck]),initial=0)>bound:
        raise ValueError('Mouth change enters accepted neck; explicit local-domain repair required')
    selected[protected_neck]=False
    predicted=baseline['verts']@linear.T+translation
    if np.max(abs((predicted-accepted_arrays['verts'])[selected]),initial=0)>bound:
        raise ValueError('Accepted local mouth is not rigidly placed; preserve artistic edit and request correspondence repair')
    merged_identity=accepted_arrays['verts'].copy();merged_closed=accepted_arrays['verts'].copy()
    merged_identity[selected]+=((identity-baseline['verts'])@linear.T)[selected]
    merged_closed[selected]+=((closed['verts']-baseline['verts'])@linear.T)[selected]
    delta=merged_closed-merged_identity
    if not np.array_equal(merged_closed[~selected],accepted_arrays['verts'][~selected]):raise ValueError('Accepted exterior changed')
    out.mkdir(parents=True)
    np.savez_compressed(out/'o_head_candidate.npz',**{**accepted_arrays,'verts':merged_closed})
    np.savez_compressed(out/'merged_identity_reference.npz',**{**accepted_arrays,'verts':merged_identity})
    np.savez_compressed(out/'closed_pose_adjustment.npz',vertex_delta=delta,full_identity_reference_vertices=merged_identity,
        free_logical_ids=np.unique(inverse[np.flatnonzero(selected)]),source_declared_render_ids=np.flatnonzero(selected))
    r=copy.deepcopy(ar);r.pop('mouth_acceptance',None)
    r.update(format='native_authored_identity_local_mouth_merge_v1',candidate_arrays=source_file(out/'o_head_candidate.npz'),
        identity_transfer=copy.deepcopy(new['identity_transfer']),closed_pose_authoring=copy.deepcopy(new['closed_pose_authoring']),
        local_mouth_merge=dict(accepted=source_file(accepted/'receipt.json'),identity_reference=source_file(identity_reference/'receipt.json'),mouth=source_file(mouth/'receipt.json'),
            linear=linear.tolist(),translation=translation.tolist(),outside_source_support_byte_equal=True,accepted_neck_byte_equal=True,
            floating_point_identity_roundoff_bound=bound,source_detail_filtered=False),installed=False,game_mutated=False,deliverable=False)
    r['closed_pose_authoring']['full_identity_arrays']=source_file(out/'merged_identity_reference.npz')
    r['closed_pose_authoring']['adjustment']=source_file(out/'closed_pose_adjustment.npz')
    save_json(out/'receipt.json',r)
    print(json.dumps(dict(output=str(out.resolve()),accepted_neck_preserved=True,source_detail_filtered=False,deliverable=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('accepted','identity-reference','mouth','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();merge(a.accepted,a.identity_reference,a.mouth,a.out)
