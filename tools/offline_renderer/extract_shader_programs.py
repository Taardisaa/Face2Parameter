"""Extract installed shader properties and exact D3D11 programs for offline use.

No live-game calls and no changed material/default substitution.  Parsed pass
state, resource bindings and compiled keywords are preserved per program.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import UnityPy
from UnityPy.export.ShaderConverter import ShaderProgram
from UnityPy.helpers import CompressionHelper
from UnityPy.streams import EndianBinaryReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.native_head.audit_neck_shading import disassemble

def digest(data):
    return hashlib.sha256(data).hexdigest()

def extract(bundle, out, only=None):
    bundle, out = Path(bundle), Path(out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    env = UnityPy.load(str(bundle))
    result = {'schema_version':1, 'bundle_path':str(bundle.resolve()),
              'bundle_sha256':digest(bundle.read_bytes()), 'shaders':[]}
    types = {0:'color',1:'vector',2:'float',3:'range',4:'texture'}
    for obj in env.objects:
        if obj.type.name != 'Shader':
            continue
        shader, tree = obj.read(), obj.read_typetree()
        parsed = tree['m_ParsedForm']
        if only is not None and parsed['m_Name'] not in only:
            continue
        props = [{'name':p['m_Name'], 'type':types[p['m_Type']],
                  'flags':p['m_Flags'], 'attributes':p['m_Attributes'],
                  'default':[p[f'm_DefValue[{i}]'] for i in range(4)],
                  'default_texture':p['m_DefTexture']} for p in parsed['m_PropInfo']['m_Props']]
        row = {'name':parsed['m_Name'], 'asset_path_id':obj.path_id,
               'properties':props, 'passes':[]}
        platform = list(shader.platforms).index(4)
        def entry(field):
            value = field[platform]
            return value[0] if isinstance(value,list) else value
        offset,length,size = map(entry,(shader.offsets,shader.compressedLengths,shader.decompressedLengths))
        code = CompressionHelper.decompress_lz4(bytes(shader.compressedBlob)[offset:offset+length],size)
        programs = ShaderProgram(EndianBinaryReader(code,endian='<'),obj.version)
        name = parsed['m_Name'].replace('/','__')
        for subshader_id, subshader in enumerate(parsed['m_SubShaders']):
            for pass_id, p in enumerate(subshader['m_Passes']):
                pr = {'subshader':subshader_id,'pass':pass_id,'type':p['m_Type'],
                      'state':p['m_State'],'name':p['m_Name'],
                      'texture_name':p['m_TextureName'],'tags':p['m_Tags'],'programs':[]}
                for stage in ('progVertex','progFragment'):
                    seen = set()
                    for compiled in p[stage]['m_SubPrograms']:
                        index=compiled['m_BlobIndex']
                        if index in seen:
                            continue
                        seen.add(index)
                        program=programs.m_SubPrograms[index]
                        data=bytes(program.m_ProgramCode)
                        if b'DXBC' not in data:
                            continue
                        # Preserve all compiled branches. Runtime selection is a
                        # later explicit operation, not a directional guess.
                        dxbc=data[data.index(b'DXBC'):]
                        assembly,sha=disassemble(data)
                        base=f'{name}__s{subshader_id}p{pass_id}__{stage}__{index}'
                        (out/(base+'.dxbc')).write_bytes(dxbc)
                        (out/(base+'.asm')).write_text(assembly,encoding='utf-8')
                        pr['programs'].append({'stage':stage,'blob_index':index,
                            'keywords':program.m_Keywords,'dxbc_sha256':sha,
                            'dxbc':base+'.dxbc','assembly':base+'.asm',
                            'bindings':compiled,'name_indices':p['m_NameIndices']})
                row['passes'].append(pr)
        result['shaders'].append(row)
    (out/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Small read request directly consumable by a native read-only exporter.
    requested=[]
    for r in result['shaders']:
        globals_by_name={}; other_by_name={}; textures=set()
        for p in r['passes']:
            for program in p['programs']:
                names={index:name for name,index in program['name_indices']}
                for texture in program['bindings']['m_TextureParams']:
                    textures.add(names[texture['m_NameIndex']])
                for buffer in program['bindings']['m_ConstantBuffers']:
                    target=globals_by_name if names[buffer['m_NameIndex']]=='$Globals' else other_by_name
                    for field in buffer['m_VectorParams']+buffer['m_MatrixParams']:
                        name=names[field['m_NameIndex']]
                        record={'name':name,'type':('matrix' if 'm_RowCount' in field else
                            'int' if field['m_Type']==1 else 'bool' if field['m_Type']==2 else
                            'float' if field['m_Dim']==1 else 'vector'),
                            'array_size':field['m_ArraySize'],
                            'dimensions':field.get('m_Dim',field.get('m_RowCount',0)*4),
                            'constant_buffer':names[buffer['m_NameIndex']]}
                        if name in target and target[name]!=record:
                            raise ValueError('Conflicting compiled uniform layouts: '+name)
                        target[name]=record
        requested.append({'name':r['name'],
            'properties':[{ 'name':p['name'],'type':p['type']} for p in r['properties']],
            'compiled_material_globals':list(globals_by_name.values()),
            'compiled_unity_globals':list(other_by_name.values()),
            'compiled_texture_bindings':sorted(textures)})
    request={'schema_version':2,'source_bundle_sha256':result['bundle_sha256'],
             'shaders':requested,
             'scope':'Compiled $Globals include uniforms outside Shader Properties. Do not skip solely because HasProperty=false; include a GetFloat/GetVector/GetInt read status. Unity globals require per-draw/render source state rather than a guessed light rig.'}
    (out/'native_property_request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    print(json.dumps({'manifest':str(out/'manifest.json'),'shader_count':len(result['shaders'])}))

if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shader',action='append',help='Explicit source shader identity filter, repeatable')
    args=ap.parse_args()
    extract(args.bundle,args.out,set(args.shader) if args.shader else None)
