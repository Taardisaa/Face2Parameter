"""Interpret actual RenderDoc D3D11 commands into resource-backed draw state.

CPU only. Original shaders, descriptors, blobs and ordered commands are retained.
Unknown commands fail rather than silently inventing per-draw state. This is an
input interpreter for the standalone renderer, not a fixed-frame replay renderer.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path

class UnresolvedPipeline(RuntimeError):
    pass

def decode(node):
    kind=node['kind']; attrs=node['attributes']
    if kind in ('uint','int','enum','ResourceId'):
        return int(node['value'])
    if kind in ('float','double'):
        return float(node['value'])
    if kind=='bool':
        if node['value'] not in ('true','false','1','0'):
            raise UnresolvedPipeline('Unknown serialized boolean')
        return node['value'] in ('true','1')
    if kind=='string':return node.get('value','')
    if kind=='null':return None
    if kind=='buffer':return {'buffer_index':int(node['value']),'byte_length':int(attrs['byteLength'])}
    if kind=='array':return [decode(c) for c in node.get('children',[])]
    if kind in ('chunk','struct'):
        result={}
        for c in node.get('children',[]):
            name=c['attributes'].get('name')
            if not name or name in result:raise UnresolvedPipeline('Missing/duplicate structured field')
            result[name]=decode(c)
        return result
    raise UnresolvedPipeline('Uninterpreted XML kind '+kind)

class CapturedPipeline:
    def __init__(self,manifest_path):
        path=Path(manifest_path);self.root=path.parent
        self.manifest=json.loads(path.read_text());self.resources={};self.state=None
        self.commands=[];self.draws=[];self.context=None
        self.needed_initials=[]
        self.blobs={b['buffer_index']:b for b in self.manifest['blobs']}

    def bytes(self,ref):
        b=self.blobs[ref['buffer_index']]; data=(self.root/b['path']).read_bytes()
        if len(data)!=ref['byte_length'] or hashlib.sha256(data).hexdigest()!=b['sha256']:
            raise UnresolvedPipeline('Original capture blob changed')
        return data

    def _resource(self,fields,out,kind,descriptor=None,data=None):
        ident=fields[out]
        if not ident or ident in self.resources:raise UnresolvedPipeline('Invalid/reused resource ID')
        self.resources[ident]={'kind':kind,'fields':fields,'descriptor':descriptor,'data':data}

    def _slots(self,stage,key,fields,array):
        start=fields['StartSlot']; values=fields[array]
        if start+len(values)>len(self.state[stage][key]):raise UnresolvedPipeline('Binding exceeds native slot array')
        self.state[stage][key][start:start+len(values)]=values

    def _check_view_hazards(self,reads,writes):
        """Subresource-overlap behavior must be implemented before aliasing is accepted."""
        def base(view):
            if not view:return None
            resource=self.resources.get(view)
            if not resource or 'pResource' not in resource['fields']:
                raise UnresolvedPipeline('Bound view has no captured base resource')
            return resource['fields']['pResource']
        written={base(v) for v in writes if v}
        if any(base(v) in written for v in reads if v):
            raise UnresolvedPipeline('Native automatic SRV/output subresource-overlap unbinding requires implementation')

    def interpret(self):
        for chunk in self.manifest['chunks']:
            name=chunk['attributes']['name']; f=decode(chunk); idx=int(chunk['attributes']['chunkIndex'])
            if name=='Internal::DeviceInitialisation':self.context=f['ImmediateContext']
            elif name=='Internal::Beginning of Capture':
                self.state=deepcopy(f['state'])
                if any(f['HiddenStreamOutCounters']) or f['numAnnotations']:
                    raise UnresolvedPipeline('Initial stream-out counters/annotation state require interpretation')
            elif name=='Internal::List of Initial Contents Resources':
                self.needed_initials=f['NeededInitials']
                if any(r['written'] for r in self.needed_initials):raise UnresolvedPipeline('Written initial-content resource payloads require interpretation')
            elif name in ('Internal::Driver Initialisation Parameters','Internal::Frame Metadata','Internal::End of Capture'):pass
            elif name.startswith('ID3D11Device::Create'):
                method=name.split('::')[1]
                if method.endswith('Shader') and 'pShaderBytecode' in f:
                    self._resource(f,'pShader',method,data=self.bytes(f['pShaderBytecode']))
                elif method=='CreateBuffer':
                    data=self.bytes(f['InitialData']) if 'InitialData' in f else None
                    if data is not None and len(data)!=f['pDesc']['ByteWidth']:raise UnresolvedPipeline('Buffer initial-data length mismatch')
                    self._resource(f,'pBuffer',method,f['pDesc'],data)
                elif method in ('CreateTexture1D','CreateTexture2D','CreateTexture3D'):
                    self._resource(f,'pTexture',method,f['Descriptor'],self.bytes(f['SubresourceContents']) if 'SubresourceContents' in f else None)
                elif method=='CreateInputLayout':self._resource(f,'pInputLayout',method,f['pInputElementDescs'],self.bytes(f['pShaderBytecodeWithInputSignature']))
                elif method in ('CreateShaderResourceView','CreateRenderTargetView','CreateDepthStencilView','CreateUnorderedAccessView'):
                    self._resource(f,'pView',method,f['pDesc'])
                elif method in ('CreateSamplerState','CreateRasterizerState','CreateDepthStencilState','CreateBlendState'):
                    self._resource(f,'pState',method,f['Descriptor'])
                else:raise UnresolvedPipeline('Uninterpreted resource creation '+method)
            elif name.startswith('ID3D11DeviceContext::'):
                if self.state is None:raise UnresolvedPipeline('Context command before captured initial state')
                if f['Context']!=self.context:raise UnresolvedPipeline('Deferred/foreign context must be implemented')
                method=name.split('::')[1]
                self._context(method,f,idx)
            else:raise UnresolvedPipeline('Uninterpreted capture command '+name)
            self.commands.append({'index':idx,'name':name,'fields':f})
        return self

    def _context(self,method,f,index):
        ia=self.state['IA']; om=self.state['OM']; rs=self.state['RS']
        if method=='IASetInputLayout':ia['Layout']=f['pInputLayout']
        elif method=='IASetVertexBuffers':
            self._slots('IA','VBs',f,'ppVertexBuffers');self._slots('IA','Strides',f,'pStrides');self._slots('IA','Offsets',f,'pOffsets')
        elif method=='IASetIndexBuffer':ia.update(IndexBuffer=f['pIndexBuffer'],IndexFormat=f['Format'],IndexOffset=f['Offset'])
        elif method=='IASetPrimitiveTopology':ia['Topo']=f['Topology']
        elif method[:2] in ('VS','HS','DS','GS','PS','CS') and method[2:]=='SetShader':
            self.state[method[:2]].update(Object=f['pShader'],Instances=f['ppClassInstances'],NumInstances=f['NumClassInstances'])
        elif method[:2] in ('VS','HS','DS','GS','PS','CS') and method[2:] in ('SetConstantBuffers','SetShaderResources','SetSamplers'):
            key,array={'SetConstantBuffers':('ConstantBuffers','ppConstantBuffers'),'SetShaderResources':('SRVs','ppShaderResourceViews'),'SetSamplers':('Samplers','ppSamplers')}[method[2:]]
            if key=='SRVs':self._check_view_hazards(f[array],om['RenderTargets']+[om['DepthView']]+om['UAVs']+self.state['CSUAVs'])
            self._slots(method[:2],key,f,array)
            if key=='ConstantBuffers':
                # D3D11 legacy binding resets D3D11.1 partial CB offsets/counts.
                n=len(f[array]);start=f['StartSlot'];self.state[method[:2]]['CBOffsets'][start:start+n]=[0]*n;self.state[method[:2]]['CBCounts'][start:start+n]=[4096]*n
        elif method=='OMSetRenderTargets':
            reads=[v for stage in ('VS','HS','DS','GS','PS','CS') for v in self.state[stage]['SRVs']]
            self._check_view_hazards(reads,f['ppRenderTargetViews']+[f['pDepthStencilView']])
            n=f['NumViews'];om['RenderTargets']=f['ppRenderTargetViews']+[0]*(len(om['RenderTargets'])-n);om['DepthView']=f['pDepthStencilView']
        elif method=='OMSetDepthStencilState':om.update(DepthStencilState=f['pDepthStencilState'],StencRef=f['StencilRef'])
        elif method=='OMSetBlendState':om.update(BlendState=f['pBlendState'],BlendFactor=f['BlendFactor'],SampleMask=f['SampleMask'])
        elif method=='RSSetState':rs['State']=f['pRasterizerState']
        elif method=='RSSetViewports':rs.update(NumViews=f['NumViewports'],Viewports=f['pViewports'])
        elif method=='RSSetScissorRects':rs.update(NumScissors=f['NumRects'],Scissors=f['pRects'])
        elif method in ('ClearRenderTargetView','ClearDepthStencilView','CopyResource','ResolveSubresource'):
            # Resource mutations are retained in ordered commands for execution.
            # They are not assigned fake pixel contents in this state interpreter.
            pass
        elif method in ('Draw','DrawIndexed','DrawInstanced','DrawIndexedInstanced','Dispatch'):
            if self.state['Predicate']:
                raise UnresolvedPipeline('Predicated draw execution requires captured query result')
            self.draws.append({'index':index,'method':method,'arguments':f,'state':deepcopy(self.state)})
        else:raise UnresolvedPipeline('Uninterpreted context command '+method)

    def shader_matches(self,draw,source_manifests):
        """Bind actual executed DXBC to all exact source identities, not keyword guesses."""
        matches={stage:[] for stage in ('VS','PS')}
        hashes={stage:hashlib.sha256(self.resources[draw['state'][stage]['Object']]['data']).hexdigest() for stage in matches}
        for path in map(Path,source_manifests):
            source=json.loads(path.read_text())
            for shader in source['shaders']:
                for p in shader['passes']:
                    for program in p['programs']:
                        binary=path.parent/program['dxbc']
                        digest=hashlib.sha256(binary.read_bytes()).hexdigest()
                        for stage in matches:
                            if digest==hashes[stage]:matches[stage].append({'bundle_sha256':source['bundle_sha256'],'shader':shader['name'],'subshader':p['subshader'],'pass':p['pass'],'program':program})
        return matches

    def constant_buffers(self,draw,stage):
        """Exact creation bytes for bound CBs whose lifetime has no pre-draw writes.

        Dynamic Map/Update is not silently replaced by creation data. D3D11.1
        range binding is represented in float4 units; no getter reconstruction.
        """
        state=draw['state'][stage]; result={}
        for slot,ident in enumerate(state['ConstantBuffers']):
            if not ident:continue
            resource=self.resources.get(ident)
            if not resource or resource['kind']!='CreateBuffer' or resource['data'] is None:
                raise UnresolvedPipeline('Bound CB has no captured byte payload')
            for c in self.commands:
                if c['index']>=draw['index']:break
                if c['name'] in ('ID3D11DeviceContext::CopyResource','ID3D11DeviceContext::ResolveSubresource') and c['fields']['pDstResource']==ident:
                    raise UnresolvedPipeline('Bound CB requires ordered resource mutation execution')
            start=state['CBOffsets'][slot]*16; count=state['CBCounts'][slot]*16
            if start>len(resource['data']):raise UnresolvedPipeline('CB range begins outside resource')
            result[slot]={'resource_id':ident,'byte_offset':start,'float4_count':state['CBCounts'][slot],
                          'data':resource['data'][start:start+count]}
        return result
