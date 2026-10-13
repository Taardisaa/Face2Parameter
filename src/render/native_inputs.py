"""Consume captured material/probe/raw-resource facts without a fitted light rig.

These are inputs to shader execution, not certification of Unity's active draw
variant/cbuffer. Missing facts remain explicit, particularly typeless view state.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .dxbc import UnsupportedShader

def pack_sh(coefficients):
    """Unity's seven-vector packing of its already-rescaled SH L2 coefficients.

    Matches Unity-Technologies FPSSample/SphericalHarmonics.PackCoefficients and
    the installed shaders' unity_SH* polynomial; no intensity fitting/rescaling.
    """
    sh=np.asarray(coefficients,np.float32)
    if sh.shape!=(3,9) or not np.isfinite(sh).all():
        raise ValueError('Actual 3 x 9 SH coefficients required')
    out={}
    for c,label in enumerate('rgb'):
        out['unity_SHA'+label]=np.array([sh[c,3],sh[c,1],sh[c,2],sh[c,0]-sh[c,6]],np.float32)
        out['unity_SHB'+label]=np.array([sh[c,4],sh[c,5],sh[c,6]*3,sh[c,7]],np.float32)
    out['unity_SHC']=np.array([sh[0,8],sh[1,8],sh[2,8],1],np.float32)
    return out

def material_uniforms(material,shader,*,color_space):
    """Values backed by actual Material reads and ShaderLab property semantics.

    Color properties use the separately captured Unity Color.linear value in a
    Linear project. Vector properties are not automatically converted. Hidden
    uniforms lacking a ShaderLab property are listed as unresolved rather than
    accepting potentially zero Material.GetVector reads as engine globals.
    """
    definitions={p['name']:p for p in shader['properties']}
    values={};unresolved=[]
    for p in material['properties']:
        name=p['name'];definition=definitions.get(name)
        if p['type']=='texture':
            continue
        if not p['read_succeeded']:
            unresolved.append({'name':name,'reason':'native getter failed'})
            continue
        if definition is None:
            unresolved.append({'name':name,'reason':'compiled hidden/per-draw uniform; source setter or engine binding required','captured_material_value':p.get('value')})
            continue
        if definition['flags'] & 32 or 'Gamma' in definition['attributes']:
            unresolved.append({'name':name,'reason':'Gamma-marked non-color requires source conversion state'})
            continue
        if definition['type']=='color' and color_space=='Linear':
            if 'linear_candidate' not in p:
                raise UnsupportedShader('Native linear Color value missing: '+name)
            values[name]=np.asarray(p['linear_candidate'],np.float32)
        else:
            values[name]=p['value']
    return values,unresolved

def raw_resource(texture):
    job=texture.get('raw_export')
    if not job or job['status']!='complete':
        raise UnsupportedShader('Completed raw resource required')
    root=Path(job['path'])
    descriptor=json.loads((root/'descriptor.json').read_text())
    for row in descriptor['subresources']:
        path=root/row['path']
        if path.stat().st_size!=row['bytes']:
            raise ValueError('Native raw subresource byte count changed')
    return root,descriptor

def view_format(texture,descriptor):
    """Determine a typed input format only from recorded native format facts.

    Typed resource formats already define view encoding. Half/float RTs and
    explicit depth formats can resolve their typeless families. UNORM typeless
    gamma without recorded RenderTexture.sRGB remains unresolved.
    """
    fmt=descriptor['dxgi_format']
    if fmt not in (1,9,19,27,39,44,53,70,73,76,79,82,90,92,94,97):
        return fmt
    native=texture.get('format')
    if fmt==1 and native in ('RGBAFloat','ARGBFloat'):
        return 2
    if fmt==9 and native in ('RGBAHalf','ARGBHalf'):
        return 10
    if native=='Depth':
        if fmt==19:return 21
        if fmt==39:return 41
        if fmt==44:return 46
        if fmt==53:return 56
    if fmt==27 and native=='ARGB32' and 'srgb' in texture:
        return 29 if texture['srgb'] else 28
    raise UnsupportedShader(f'Typeless shader view unresolved: native {native}, DXGI {fmt}')

def sss_pass_schedule(sss_fields,convolution_fields,*,width,height,current_fov,stereo):
    """Installed SSS.OnPreRender + SSS_convolution.OnRenderImage scheduling.

    All operands are captured native fields. Samples/keywords/color/depth tests
    remain in the real installed shader; this returns the actual pass offsets.
    """
    count=int(sss_fields['<ScatteringIterations>k__BackingField'])
    if count<0 or width<=0 or height<=0 or current_fov<=0:
        raise ValueError('Invalid captured convolution operands')
    radius=np.float32(sss_fields['<ScatteringRadius>k__BackingField'])/np.float32(sss_fields['<Downsampling>k__BackingField'])
    if stereo:radius/=np.float32(2)
    radius*=np.float32(convolution_fields['initFOV'])/np.float32(current_fov)
    radius*=np.float32(np.sqrt(np.float32(height)*height+np.float32(width)*width))/np.float32(2202.9072)
    if count==0:return [{'operation':'copy_lighting_to_blurred','source_path':'SSS_convolution.OnRenderImage iterations <= 0'}]
    return [{'operation':'installed_blur_shader','axis':axis,'_TexelOffsetScale':
              np.array([0,radius,0,0] if axis=='vertical' else [radius,0,0,0],np.float32)}
            for _ in range(count) for axis in ('vertical','horizontal')]
