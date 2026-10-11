"""Read source-bound mouth topology and original controller frame support."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from tools.native_head.mother_template_inputs import source_file,save_json
from tools.native_head.mother_default_pose import dense_frame
from tools.model_bridge.artifact import sha
def audit(inputs,mother,out):
    if out.exists(): raise FileExistsError("Use fresh evidence directory")
    out.mkdir(parents=True)
    r=json.loads((inputs/"receipt.json").read_text())
    m=json.loads((mother/"receipt.json").read_text())
    if m['inputs']['sha256']!=sha(inputs/'receipt.json'):
        raise ValueError('Mother belongs to another donor')
    if m['candidate_arrays']['sha256']!=sha(mother/'o_head_candidate.npz'):
        raise ValueError('Mother geometry changed')
    for key in ('regions','native_prefab_export'):
        if sha(r[key]['path'])!=r[key]['sha256']:
            raise ValueError('Prepared source changed: '+key)
    regions=json.loads((inputs/"native_regions.json").read_text())
    a=dict(np.load(mother/"o_head_candidate.npz",allow_pickle=False))
    audit_row=r["reference_audit"]
    if sha(audit_row['path'])!=audit_row['sha256']:
        raise ValueError('Source audit changed')
    audit=json.loads(Path(audit_row["path"]).read_text(encoding="utf-8-sig"))
    for row in audit["assemblies"]+audit["decompiled_sources"]:
        if sha(row['path'])!=row['sha256']:
            raise ValueError('Installed implementation/source changed: '+row['path'])
    prefab=json.loads(Path(r["native_prefab_export"]["path"]).read_text())
    ctrl=next(b["original_typetree"] for b in prefab["behaviours"] if b["class_name"]=="FaceBlendShape")
    head=next(b for b in prefab["renderers"] if b["mesh"]=="o_head")
    obj=head["original_renderer_typetree"]["m_GameObject"]["m_PathID"]
    target=next(t for t in ctrl["MouthCtrl"]["FBSTarget"] if t["ObjTarget"]["m_PathID"]==obj)
    shapes=json.loads(Path(head["blendshape_export"]["path"]).read_text())
    if sha(head['blendshape_export']['path'])!=head['blendshape_export']['sha256']:
        raise ValueError('Original native expression frames changed')
    ptn=target["PtnSet"][0]
    closed=dense_frame(shapes,ptn["Close"],len(a["verts"]))["vertex"]
    opened=dense_frame(shapes,ptn["Open"],len(a["verts"]))["vertex"]
    opening=opened-closed
    lip=np.asarray(m["regional_targets"]["candidate_vermilion_vertex_ids"])
    selected=np.flatnonzero(np.isin(a["faces"],lip).any(1))
    tri=a["faces"][selected]
    fig,axs=plt.subplots(1,3,figsize=(20,7))
    axes=[(0,1,"Original native lip UV"),(0,1,"Closed native lip front"),(2,1,"Closed native lip side")]
    colors=opening[tri,1].mean(1)
    for ax,(x,y,title) in zip(axs,axes):
        v=a["uv"] if title.endswith("UV") else a["original_vertices"]
        polys=v[tri][:,:,[x,y]]
        pc=PolyCollection(polys,array=colors,cmap="coolwarm",edgecolors="k",linewidth=.3)
        ax.add_collection(pc);ax.autoscale_view();ax.set_aspect("equal");ax.set_title(title+"; color = original opening delta Y")
        for i in np.unique(tri):
            ax.text(v[i,x],v[i,y],str(i),fontsize=5)
    fig.colorbar(pc,ax=axs,shrink=.6)
    fig.savefig(out/"native_mouth_topology.png",dpi=170);plt.close(fig)
    np.savez_compressed(out/"native_mouth_support.npz",lip_ids=lip,face_ids=selected,original_opening_delta=opening)
    save_json(out/"receipt.json",dict(format="generic_mouth_source_contract_audit_v1",inputs=source_file(inputs/"receipt.json"),mother=source_file(mother/"receipt.json"),native_head_arrays=source_file(mother/"o_head_candidate.npz"),native_regions=source_file(inputs/"native_regions.json"),source_audit=source_file(audit_row["path"]),implementation_sources=audit["assemblies"]+audit["decompiled_sources"],original_expression_frames=head["blendshape_export"],mouth_controller=ctrl["MouthCtrl"],pattern0=ptn,inner_mouth_boundaries=regions["inner_mouth_component_boundaries"],vermilion_support=lip.tolist(),opening_support=source_file(out/"native_mouth_support.npz"),plot=source_file(out/"native_mouth_topology.png"),limitations=["Pigment envelope is source-bound candidate support, not a certified outer/inner lip split","Native mouth opening delta is recovered original data, not an empirical jaw gain","No candidate deformation or game mutation performed"]))
if __name__=="__main__":
    root=Path("outputs/native_mother_template_20261008")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,default=root/'inputs_v2')
    p.add_argument('--mother',type=Path,default=root/'local_contacts_v16')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.inputs,a.mother,a.out)
