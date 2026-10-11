"""Stage generic full-detail mouth candidates; never publish failed geometry.

A fixed donor/mother contract is reused for all MICA identities. Failed stages
retain exact source fields, candidate arrays and repair inputs. No game mutation,
affine preview or automatic installation occurs in this authoring entry point.
"""
import argparse,copy,json,shutil
from pathlib import Path
from tools.native_head.mother_identity import build
from tools.native_head.mother_surface_quality import check
from tools.native_head.mouth_contact_repair import repair
from tools.native_head.mother_template_inputs import source_file,save_json
from tools.model_bridge.artifact import sha

def run(inputs,mother,manifest,image_index,source_quality,out):
    if out.exists():raise FileExistsError("Use a fresh per-identity pipeline directory")
    out.mkdir(parents=True)
    source=dict(inputs=source_file(inputs/"receipt.json"),mother=source_file(mother/"receipt.json"),
        manifest=source_file(manifest),image_index=image_index,source_quality=source_file(source_quality/"receipt.json"))
    candidate=out/"identity_full";quality=out/"identity_quality"
    try:
        build(inputs,mother,manifest,image_index,candidate,generic_mouth=True)
        check(candidate,quality,source_quality)
        original=json.loads((candidate/"receipt.json").read_text())
        oral=original["identity_transfer"]["mouth_detail_preservation"]["oral"]
        q=json.loads((quality/"receipt.json").read_text())
        if not oral["converged"]:
            raise ValueError("Native oral solve did not converge; preserve full source targets for continuation")
        if q["added_crossings"]:
            fixed=out/"contact_repair";repair(candidate,quality,fixed)
            fixed_quality=out/"contact_repair_quality";check(fixed,fixed_quality,source_quality)
            candidate,quality=fixed,fixed_quality
            q=json.loads((quality/"receipt.json").read_text())
        if q["added_crossings"]:
            raise ValueError("Protected full-detail mouth contract still has added crossings; semantic contact correspondence repair required")
        accepted=out/"accepted_static_head";accepted.mkdir()
        for name in ("o_head_candidate.npz","receipt.json"):
            shutil.copyfile(candidate/name,accepted/name)
        r=json.loads((accepted/"receipt.json").read_text())
        r["candidate_arrays"]=source_file(accepted/"o_head_candidate.npz")
        r["mouth_acceptance"]=dict(status="accepted_static_head",quality=source_file(quality/"receipt.json"),
            original_candidate_receipt=source_file(candidate/"receipt.json"),candidate_arrays_sha256=sha(accepted/"o_head_candidate.npz"),
            scope="Static head/closed-reference mouth only. Component/expression/native acceptance is still required.")
        save_json(accepted/"receipt.json",r)
        result=dict(format="generic_mouth_pipeline_v1",status="accepted_static_head",source=source,
            candidate=source_file(accepted/"receipt.json"),quality=source_file(quality/"receipt.json"),
            game_mutated=False,full_character_delivered=False,next_step="Original native eight-component/frame/bind adaptation and limited native acceptance")
    except (ValueError,RuntimeError) as error:
        result=dict(format="generic_mouth_pipeline_v1",status="requires_correspondence_repair",source=source,
            reason=str(error),candidate=source_file(candidate/"receipt.json") if (candidate/"receipt.json").exists() else None,
            quality=source_file(quality/"receipt.json") if (quality/"receipt.json").exists() else None,
            repair_request=source_file(out/"contact_repair/repair_request.json") if (out/"contact_repair/repair_request.json").exists() else None,
            source_control_filtering_allowed=False,affine_fallback_allowed=False,game_mutated=False,
            full_character_delivered=False,next_step="Revise source-bound upper/inner/lower lip contact correspondence, preserve full source controls and rerun into a fresh directory; do not publish current arrays.")
    save_json(out/"pipeline.json",result)
    print(json.dumps(dict(status=result["status"],receipt=str((out/"pipeline.json").resolve()),game_mutated=False)))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","mother","manifest","source-quality","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--image-index",type=int,required=True)
    a=p.parse_args();r=run(a.inputs.resolve(),a.mother.resolve(),a.manifest.resolve(),a.image_index,a.source_quality.resolve(),a.out.resolve())
    raise SystemExit(0 if r["status"]=="accepted_static_head" else 2)
