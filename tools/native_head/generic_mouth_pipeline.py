"""Stage generic full-detail mouth candidates; never publish failed geometry.

A fixed donor/mother contract is reused for all MICA identities. Failed stages
retain exact source fields, candidate arrays and repair inputs. No game mutation,
affine preview or automatic installation occurs in this authoring entry point.
"""
import argparse,copy,json,shutil
from pathlib import Path
from tools.native_head.mother_identity import build
from tools.native_head.mother_surface_quality import check
from tools.native_head.mouth_chart_correspondence import build as build_chart
from tools.native_head.mouth_chart_authoring import author as author_chart
from tools.native_head.mouth_contact_contract import build as build_contacts
from tools.native_head.mouth_close_pose import author as author_close
from tools.native_head.mother_template_inputs import source_file,save_json
from tools.model_bridge.artifact import sha

def run(inputs,mother,manifest,image_index,source_quality,out,lip_chart=None,close_contract=None,uv_template=None):
    if out.exists():raise FileExistsError("Use a fresh per-identity pipeline directory")
    out.mkdir(parents=True)
    source=dict(inputs=source_file(inputs/"receipt.json"),mother=source_file(mother/"receipt.json"),
        manifest=source_file(manifest),image_index=image_index,source_quality=source_file(source_quality/"receipt.json"))
    candidate=out/"identity_full";quality=out/"closed_quality"
    try:
        build(inputs,mother,manifest,image_index,candidate,generic_mouth=True)
        if lip_chart is None:
            if uv_template is None:
                profile=json.loads(Path(__file__).with_name('native_atlas_profile.json').read_text())
                uv_template=Path(profile['uv_template'])
            lip_chart=out/'donor_lip_chart';build_chart(candidate,uv_template,lip_chart)
        full=out/'surface_identity';author_chart(candidate,lip_chart,full);candidate=full
        original=json.loads((full/'receipt.json').read_text())
        oral=original['identity_transfer']['mouth_detail_preservation']['oral']
        if not oral["converged"]:
            raise ValueError("Native oral solve did not converge; preserve full source targets for continuation")
        if close_contract is None:
            close_contract=out/'donor_close_contacts';build_contacts(inputs,mother,close_contract)
        closed=out/'closed_pose';author_close(full,close_contract,closed);candidate=closed
        check(candidate,quality,source_quality)
        q=json.loads((quality/'receipt.json').read_text())
        if q["added_crossings"]:
            raise ValueError('Source-bound lip/Close correspondence still adds crossings; preserve full identity and action for local pipeline repair')
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
            lip_chart=source_file(lip_chart/'receipt.json') if lip_chart and (lip_chart/'receipt.json').exists() else None,
            close_contract=source_file(close_contract/'receipt.json') if close_contract and (close_contract/'receipt.json').exists() else None,
            source_control_filtering_allowed=False,affine_fallback_allowed=False,game_mutated=False,
            full_character_delivered=False,next_step="Revise source-bound upper/inner/lower lip contact correspondence, preserve full source controls and rerun into a fresh directory; do not publish current arrays.")
    save_json(out/"pipeline.json",result)
    print(json.dumps(dict(status=result["status"],receipt=str((out/"pipeline.json").resolve()),game_mutated=False)))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("inputs","mother","manifest","source-quality","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--image-index",type=int,required=True)
    for key in ('lip-chart','close-contract','uv-template'):p.add_argument('--'+key,type=Path)
    a=p.parse_args();r=run(a.inputs.resolve(),a.mother.resolve(),a.manifest.resolve(),a.image_index,a.source_quality.resolve(),a.out.resolve(),a.lip_chart,a.close_contract,a.uv_template)
    raise SystemExit(0 if r["status"]=="accepted_static_head" else 2)
