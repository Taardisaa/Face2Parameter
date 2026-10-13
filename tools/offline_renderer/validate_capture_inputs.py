"""CPU-only byte acceptance using a pre-existing independent RenderDoc fixture.

Never launches/injects/replays a graphics program. This is input interpretation,
not native HS2 RGB or complete parameter-renderer acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.render.captured_pipeline import CapturedPipeline,UnresolvedPipeline

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--packet',required=True,type=Path)
    ap.add_argument('--shader-manifest',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError('Fresh receipt required')
    p=CapturedPipeline(a.manifest).interpret()
    if len(p.draws)!=1:raise UnresolvedPipeline('This receipt requires the recorded one-draw analytic fixture')
    draw=p.draws[0]; checks=[]
    def compare(label,actual,file):
        expected=(a.packet/file).read_bytes(); ok=actual==expected
        checks.append({'input':label,'source_file':file,'byte_count':len(actual),'sha256':digest(actual),'exact':ok})
        if not ok:raise AssertionError('Capture/declared original packet mismatch: '+label)
    for stage,file in [('VS','vertex.dxbc'),('PS','pixel.dxbc')]:
        compare(stage,p.resources[draw['state'][stage]['Object']]['data'],file)
    ia=draw['state']['IA'];compare('vertex data',p.resources[ia['VBs'][0]]['data'],'vertices.bin')
    compare('index data',p.resources[ia['IndexBuffer']]['data'],'indices.bin')
    for slot,row in p.constant_buffers(draw,'VS').items():compare('VS cb'+str(slot),row['data'],f'cb_0_{slot}.bin')
    for slot,row in p.constant_buffers(draw,'PS').items():compare('PS cb'+str(slot),row['data'],f'cb_1_{slot}.bin')
    srv=p.resources[draw['state']['PS']['SRVs'][0]]
    compare('original texture pixels',p.resources[srv['fields']['pResource']]['data'],'texture.bin')
    identities=p.shader_matches(draw,[a.shader_manifest])
    if any(not rows for rows in identities.values()):raise AssertionError('Executed shader source not identified')
    recorded_draw={k:v for k,v in draw.items() if k!='buffer_versions'}
    recorded_draw['buffer_versions']={ident:[{'offset':offset,'bytes':len(data),'sha256':digest(data)} for offset,data in segments] for ident,segments in draw['buffer_versions'].items()}
    receipt={'schema_version':1,'status':'pass','scope':'CPU original capture inputs; not HS2/full RGB acceptance',
        'capture_manifest':str(a.manifest.resolve()),'command_count':len(p.commands),'draw_count':len(p.draws),
        'resource_count':len(p.resources),'exact_byte_checks':checks,'shader_source_matches':identities,
        'draw_state':recorded_draw,'unimplemented_execution':['initial-content payloads','texture Map/Unmap and UpdateSubresource','deferred command lists','GPU command execution/shared targets','full scene/SSS/postprocess integration']}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'status':'pass','exact_byte_checks':len(checks),'commands':len(p.commands),'receipt':str(a.out)}))
if __name__=='__main__':main()
