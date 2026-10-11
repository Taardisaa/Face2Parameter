"""Apply a fixed native/FLAME surface correspondence without filtering identity."""
import argparse,json,copy
from pathlib import Path
import numpy as np
from tools.model_bridge.artifact import sha
from tools.native_head.mouth_identity_transfer import apply_lip_basis,make_correspondence
from tools.native_head.mother_template_inputs import source_file,save_json

def author(candidate,chart,out):
    if out.exists():raise FileExistsError('Preserve earlier identities')
    r=json.loads((candidate/'receipt.json').read_text())
    c=json.loads((chart/'receipt.json').read_text())
    if r['inputs']['sha256']!=c['inputs']['sha256'] or r['mother_reference']['sha256']!=c['mother']['sha256']:
        raise ValueError('Surface chart belongs to a different native donor/mother')
    for row in (c['mother'],c['inputs'],c['source_uv_template'],c['correspondence']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Anatomical correspondence source changed')
    for row in (r['candidate_arrays'],r['identity_transfer']['basis'],r['identity_transfer']['mouth_correspondence']):
        if sha(row['path'])!=row['sha256']:raise ValueError('Full source identity/correspondence changed')
    mother_receipt=json.loads(Path(c['mother']['path']).read_text())
    if sha(mother_receipt['candidate_arrays']['path'])!=mother_receipt['candidate_arrays']['sha256']:
        raise ValueError('Original native mother changed')
    m=dict(np.load(Path(c['mother']['path']).parent/'o_head_candidate.npz',allow_pickle=False))
    b=dict(np.load(candidate/'identity_basis.npz',allow_pickle=False))
    mapping=dict(np.load(candidate/'mouth_correspondence.npz',allow_pickle=False))
    uvmap=np.load(c['correspondence']['path'],allow_pickle=False)
    inputs=Path(c['inputs']['path']).parent
    reg=json.loads((inputs/'native_regions.json').read_text())
    canonical=np.array([int(reg['graph_aliases'].get(str(i),i)) for i in range(len(m['verts']))])
    unique,inverse=np.unique(canonical,return_inverse=True);faces=inverse[m['faces']]
    if not np.isin(mapping['lip_ids'],uvmap['lip_ids']).all():raise ValueError('Original full lip support was omitted')
    mapping=make_correspondence(m['verts'][unique],faces,uvmap['lip_ids'],mapping['oral_face_ids'])
    flame=np.load(inputs/'flame_zero_identity.npz',allow_pickle=False)
    # Shape columns retain the actual original decoder source values. The
    # chart selects the surface side once, before seeing any identity beta.
    lip_basis=np.einsum('vi,vick->vck',uvmap['barycentric'],b['source_identity_basis'][flame['faces_tensor'][uvmap['source_faces']]])
    beta=b['shape_params'];baseline=m['verts']+b['native_identity_basis']@beta
    authored,report=apply_lip_basis(m['verts'][unique],faces,baseline[unique],lip_basis,beta,mapping)
    old=dict(np.load(candidate/'o_head_candidate.npz',allow_pickle=False))
    arrays={**old,'verts':authored[inverse]}
    if not np.array_equal(arrays['verts'][reg['neck_boundary_all_render_copies']],m['verts'][reg['neck_boundary_all_render_copies']]):
        raise ValueError('Actual native neck rim moved')
    out.mkdir(parents=True)
    np.savez_compressed(out/'o_head_candidate.npz',**arrays)
    np.savez_compressed(out/'identity_basis.npz',**b)
    mapping['identity_authored_correction']=arrays['verts']-baseline
    np.savez_compressed(out/'mouth_correspondence.npz',**mapping)
    r=copy.deepcopy(r);r['candidate_arrays']=source_file(out/'o_head_candidate.npz')
    transfer=r['identity_transfer'];transfer['basis']=source_file(out/'identity_basis.npz')
    transfer['mouth_correspondence']=source_file(out/'mouth_correspondence.npz')
    transfer['mouth_detail_preservation']=dict(report,chart=source_file(chart/'receipt.json'),
        policy='Fixed actual inner/outer surface chart, full original 300-column barycentric identity sampling; '+report['policy'])
    r.update(format='native_mother_surface_chart_mouth_identity_v1',deliverable=False,installed=False,game_mutated=False)
    save_json(out/'receipt.json',r)
    print(json.dumps(dict(output=str(out.resolve()),oral_converged=report['oral']['converged'],source_coefficients_filtered=False,deliverable=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','chart','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();author(a.candidate,a.chart,a.out)
