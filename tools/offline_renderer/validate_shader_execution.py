"""Run an installed eyeshadow DXBC pass in WARP and compare CPU execution.

This analytic material fixture checks the actual shader/pass executor. It is
not a game face, fitted light rig, appearance acceptance or MediaPipe input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.render.dxbc import Program
from src.render.shader_bindings import constant_buffers, select_program
from src.render.texture_sampling import level_2d

def signature(assembly):
    before=assembly.split('// Output signature:')[0]
    return re.findall(r'^// ([A-Z_]+)\s+(\d+)\s+[xyzw]+\s+(\d+)\s+\S+\s+(?:float|uint|sint)',before,re.M)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--programs',type=Path,required=True)
    ap.add_argument('--backend',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    args.out.mkdir(parents=True)
    manifest=json.loads((args.programs/'manifest.json').read_text())
    shader=next(s for s in manifest['shaders'] if s['name']=='AIT/main eyeshadow lambert')
    vs=select_program(shader,0,'progVertex',['DIRECTIONAL'])
    ps=select_program(shader,0,'progFragment',['DIRECTIONAL'])
    uniform={'_Color':[.2,.3,.4,.8],'_MainTex_ST':[1,1,0,0],'_ShadowScale':.5,
         '_texcoord_ST':[1,1,0,0], 'unity_ObjectToWorld':np.eye(4).ravel(),
         'unity_WorldToObject':np.eye(4).ravel(),'unity_MatrixVP':np.eye(4).ravel()}
    vp=Program((args.programs/vs['assembly']).read_text())
    pp=Program((args.programs/ps['assembly']).read_text())
    constants=[constant_buffers(r,p,uniform) for r,p in ((vs,vp),(ps,pp))]
    vertices=signature((args.programs/vs['assembly']).read_text())
    stride=(max(int(i[2]) for i in vertices)+1)*16
    data=np.zeros((4,stride//16,4),np.float32)
    for name,index,register in vertices:
        register=int(register)
        if name=='POSITION':data[:,register]=[[-1,1,.5,1],[1,1,.5,1],[-1,-1,.5,1],[1,-1,.5,1]]
        elif name=='TEXCOORD':data[:,register]=[[0,0,0,0],[1,0,0,0],[0,1,0,0],[1,1,0,0]]
        elif name=='NORMAL':data[:,register]=[0,0,1,0]
        elif name=='TANGENT':data[:,register]=[1,0,0,1]
        elif name=='COLOR':data[:,register]=[1,1,1,1]
        else:raise ValueError('Unexpected actual input '+name)
    data.tofile(args.out/'vertices.bin')
    np.asarray([0,1,2,2,1,3],np.uint32).tofile(args.out/'indices.bin')
    shutil.copyfile(args.programs/vs['dxbc'],args.out/'vertex.dxbc')
    shutil.copyfile(args.programs/ps['dxbc'],args.out/'pixel.dxbc')
    image=np.ones((8,8,4),np.float32)
    y,x=np.mgrid[:8,:8];image[:,:,3]=(x+2*y)/21
    image.tofile(args.out/'texture.bin')
    width=32;height=32
    clear=np.array([.7,.6,.5,1],np.float32)
    packet=[f'{width} {height} 1 4 {stride} 6',
            '1 0 0 8 1 9 1 1', ' '.join(map(str,clear))]
    for stage,cbs in enumerate(constants):
        packet.append(str(len(cbs)))
        for slot,values in cbs.items():
            path=f'cb_{stage}_{slot}.bin';values.tofile(args.out/path)
            packet.append(f'{slot} {path}')
    # Exact raw float texture, one mip, clamp, linear texel sampling. The fixture
    # does not replace native textures or claim mip/anisotropy acceptance.
    packet.extend(['1','0 0 8 8 1 1 2 20 1 3 3 3','texture.bin 128'])
    (args.out/'packet.txt').write_text('\n'.join(packet)+'\n')
    run=subprocess.run([str(args.backend.resolve()),str(args.out.resolve())],capture_output=True,text=True)
    if run.returncode:
        raise RuntimeError(run.stderr)
    actual=np.fromfile(args.out/'rgba32f.bin',np.float32).reshape(height,width,4)
    yy,xx=np.mgrid[:height,:width]
    uv=np.zeros((width*height,4),np.float32)
    uv[:,0]=(xx.ravel()+.5)/width;uv[:,1]=(yy.ravel()+.5)/height
    sample=lambda opcode,t,s,coords,extra:level_2d(image,coords,filter_mode='Bilinear',wrap_u='Clamp',wrap_v='Clamp')
    cpu=pp.execute({'v1':uv},constants[1],sample=sample).outputs['o0'].reshape(height,width,4)
    expected=cpu.copy();expected[:,:,:3]*=clear[:3]
    maximum=float(np.max(np.abs(actual-expected)))
    receipt={'schema_version':1,'device':'D3D11 WARP (CPU software driver)',
         'bundle_sha256':manifest['bundle_sha256'],'vertex_dxbc_sha256':vs['dxbc_sha256'],
         'pixel_dxbc_sha256':ps['dxbc_sha256'],'backend_sha256':hashlib.sha256(args.backend.read_bytes()).hexdigest(),
         'shader':shader['name'],'keywords':['DIRECTIONAL'],'max_rgba_float_difference':maximum,
         'tolerance':1e-6,'pass_gate':maximum<=1e-6,'rgb_renderer_complete':False,
         'scope':'Synthetic analytic texture/quad/color fixture; installed original DXBC and DstColor Zero blend executed. No native face or appearance acceptance.',
         'stdout':run.stdout}
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
    return 0 if receipt['pass_gate'] else 2

if __name__=='__main__':
    raise SystemExit(main())
