"""Reproducible mouth contact repair with source-bound full-detail exterior pins.

This changes only explicitly unexposed native lip/oral vertices. It never calls
the affine mouth preview or replaces source coefficients. Front visibility is a
conservative geometric pin rule, not a certified anatomical segmentation.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json,source_file
from tools.native_head.mother_local_contacts import separate_pair,dirty_crossings,joint_contact_step
from tools.native_head.mother_local_repair import similarity

def front_exposed(vertices,faces,ids,epsilon=1e-8):
    """Parallel +Z ray against complete donor triangles, without raster sampling."""
    tri=vertices[faces]
    ab=tri[:,1,:2]-tri[:,0,:2];ac=tri[:,2,:2]-tri[:,0,:2]
    det=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
    usable=np.abs(det)>np.finfo(float).eps
    visible=[]
    for i in ids:
        p=vertices[i];ap=p[:2]-tri[:,0,:2]
        u=np.divide(ap[:,0]*ac[:,1]-ap[:,1]*ac[:,0],det,out=np.zeros(len(det)),where=usable)
        v=np.divide(ab[:,0]*ap[:,1]-ab[:,1]*ap[:,0],det,out=np.zeros(len(det)),where=usable)
        hit=usable&(u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)
        z=tri[:,0,2]+u*(tri[:,1,2]-tri[:,0,2])+v*(tri[:,2,2]-tri[:,0,2])
        if not np.any(hit&(z>p[2]+epsilon)):visible.append(int(i))
    return np.asarray(visible,int)

def repair(candidate,quality,out,max_rounds=32):
    if out.exists():raise FileExistsError("Preserve all mouth candidates and failures")
    r=json.loads((candidate/"receipt.json").read_text())
    q=json.loads((quality/"receipt.json").read_text())
    if sha(candidate/"o_head_candidate.npz")!=q["arrays"]["sha256"]:
        raise ValueError("Crossing report belongs to another candidate")
    mt=r["identity_transfer"]
    if mt["flame_lip_detail_filtered"] or not mt.get("nonlinear_per_identity_mouth_recomputation"):
        raise ValueError("Repair requires explicit full-detail mouth candidate")
    if sha(mt["mouth_correspondence"]["path"])!=mt["mouth_correspondence"]["sha256"]:
        raise ValueError("Full-detail source correspondence changed")
    inputs=Path(r["inputs"]["path"]).parent
    regions=json.loads((inputs/"native_regions.json").read_text())
    a=dict(np.load(candidate/"o_head_candidate.npz",allow_pickle=False))
    mother=Path(r["mother_reference"]["path"]).parent
    rest_arrays=np.load(mother/"o_head_candidate.npz",allow_pickle=False)
    canonical=np.array([int(regions["graph_aliases"].get(str(i),i)) for i in range(len(a["verts"]))])
    unique,inverse=np.unique(canonical,return_inverse=True)
    reference=rest_arrays["verts"][unique];faces=inverse[a["faces"]]
    target=a["verts"][unique];x=target.copy()
    mapping=dict(np.load(mt["mouth_correspondence"]["path"],allow_pickle=False))
    lip=np.asarray(mapping["lip_ids"]);oral=np.asarray(mapping["oral_interior_ids"])
    pins=front_exposed(reference,faces,lip)
    # The complete annotated outer contour remains fixed, including its embedded
    # barycentric support. No vertex outside the explicit mouth region is free.
    for annotation in r["candidate_anatomical_constraints"]["annotations"]:
        if 48<=annotation["target_flame_landmark"]<=59 and "face_id" in annotation:
            pins=np.union1d(pins,faces[annotation["face_id"]])
    movable=np.zeros(len(reference),bool);movable[np.union1d(lip,oral)]=True;movable[pins]=False
    dirty=np.isin(faces,np.flatnonzero(movable)).any(1)
    exempt={tuple(pair) for pair in q["source_crossings"]}
    inventory={tuple(pair) for pair in q["added_crossings"]}
    scale,R,t=similarity(reference[lip],target[lip])
    history=[];seen=set()
    for round_id in range(max_rounds):
        if not inventory:break
        signature=tuple(sorted(inventory))
        progress=False
        for pair in sorted(inventory):
            progress|=separate_pair(reference,x,faces[list(pair)],movable,R,scale=scale)
        remaining=dirty_crossings(x,faces,dirty,exempt)
        # Fully protected crossings are not omitted from the repair result.
        for pair in inventory:
            if not dirty[list(pair)].any():remaining.add(pair)
        history.append(dict(round=round_id+1,pairs_before=len(inventory),pairs_after=len(remaining),pair_projection_progress=bool(progress)))
        if remaining and (signature in seen or not progress):
            rotations=np.broadcast_to(R.T,(len(x),3,3)).copy()
            trial,note=joint_contact_step(reference,x,faces,remaining,movable,rotations)
            history[-1]["joint_step"]=note
            if note.get("constraints_satisfied"):x=trial
            remaining=dirty_crossings(x,faces,dirty,exempt)|{pair for pair in remaining if not dirty[list(pair)].any()}
        seen.add(signature);inventory=remaining
        if not progress and inventory and not history[-1].get("joint_step",{}).get("constraints_satisfied"):break
    if not np.array_equal(x[~movable],target[~movable]):raise ValueError("Repair moved protected exterior detail")
    final=x[inverse]
    out.mkdir(parents=True)
    np.savez_compressed(out/"o_head_candidate.npz",**{**a,"verts":final})
    np.savez_compressed(out/"mouth_repair_constraints.npz",movable_logical_ids=np.flatnonzero(movable),fixed_logical_ids=np.flatnonzero(~movable),front_exposed_lip_ids=pins,repair_delta=final-a["verts"])
    request=dict(format="native_full_detail_mouth_repair_request_v1",candidate=source_file(candidate/"receipt.json"),candidate_arrays=source_file(candidate/"o_head_candidate.npz"),quality=source_file(quality/"receipt.json"),correspondence=mt["mouth_correspondence"],constraints=source_file(out/"mouth_repair_constraints.npz"),unresolved_pairs=sorted(inventory),history=history,source_coefficients_modified=False,lip_detail_filtered=False,protected_exterior_exact=True,
        next_step="Recompute semantic lip-layer correspondence for unresolved protected contact pairs; retain original full source controls and rerun this contract. No affine fallback permitted.",
        scope="Static mouth repair only; complete component/frame/native acceptance remains mandatory")
    save_json(out/"repair_request.json",request)
    updated=copy.deepcopy(r)
    updated.update(format="native_mother_generic_mouth_repair_candidate_v1",candidate_arrays=source_file(out/"o_head_candidate.npz"),code=source_file(Path(__file__)),mouth_repair=source_file(out/"repair_request.json"),deliverable=False,installed=False,game_mutated=False)
    updated["identity_transfer"]["mouth_detail_preservation"]["post_repair_requires_full_geometry_acceptance"]=True
    updated["identity_transfer"]["identity_basis_alone_reconstructs_candidate"]=False
    save_json(out/"receipt.json",updated)
    print(json.dumps(dict(output=str(out.resolve()),remaining_pairs=len(inventory),protected_exterior_exact=True,deliverable=False)))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("candidate","quality","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();repair(a.candidate.resolve(),a.quality.resolve(),a.out.resolve())
