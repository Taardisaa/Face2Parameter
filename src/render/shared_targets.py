"""View subresource identity and ordered shared-target producer dependencies.

This retains actual clear/copy/draw producers; it does not render pixels or
replace GPU results with colors/CPU approximations. Frame execution is separate.
"""
class UnresolvedTarget(RuntimeError):pass

def view_subresources(resources,view):
    if not view:return None,()
    v=resources[view];ident=v['fields']['pResource'];r=resources[ident]
    if 'texture_layouts' not in r:raise UnresolvedTarget('Buffer views require explicit byte/element range interpretation')
    ls=r['texture_layouts'];desc=v['descriptor'];reading=v['kind']=='CreateShaderResourceView'
    mip_count=1+max(row['mip'] for row in ls);array_count=1+max(row['array'] for row in ls)
    mip=0;mips=mip_count if reading else 1;array=0;arrays=array_count
    if desc is not None:
        names=[name for name in desc if name.startswith('Texture')]
        if len(names)!=1:raise UnresolvedTarget('Actual texture view union member required')
        name=names[0];fields=desc[name]
        if reading:
            mip=fields.get('MostDetailedMip',0);mips=fields.get('MipLevels',1)
            if mips==0xffffffff:mips=mip_count-mip
        else:mip=fields.get('MipSlice',0);mips=1
        if 'FirstArraySlice' in fields:array=fields['FirstArraySlice'];arrays=fields['ArraySize']
        elif name=='TextureCubeArray':array=fields['First2DArrayFace'];arrays=fields['NumCubes']*6
        elif name=='TextureCube':array=0;arrays=6
        elif 'Array' not in name:array=0;arrays=1
        if name=='Texture3D' and not reading:
            depth=next(row['depth'] for row in ls if row['mip']==mip)
            if fields.get('FirstWSlice',0)!=0 or fields.get('WSize',depth) not in (depth,0xffffffff):
                raise UnresolvedTarget('Partial Texture3D render-view slice writes require volume-range tracking')
    if mip<0 or mips<1 or mip+mips>mip_count or array<0 or arrays<1 or array+arrays>array_count:raise UnresolvedTarget('View exceeds recorded mip/array storage')
    return ident,tuple(row['subresource'] for row in ls if mip<=row['mip']<mip+mips and array<=row['array']<array+arrays)

class SharedTargets:
    def __init__(self,resources):self.resources=resources;self.nodes=[];self.current={}

    def _node(self,operation,ident,sub,fields,dependencies=()):
        index=len(self.nodes);self.nodes.append({'node':index,'operation':operation,'resource':ident,'subresource':sub,
            'fields':fields,'dependencies':list(dependencies)})
        self.current[(ident,sub)]=index;return index

    def created(self,ident):
        r=self.resources[ident]
        if 'texture_layouts' not in r:return
        for row in r['texture_layouts']:
            known=bool(r['texture_segments'][row['subresource']])
            self._node('captured_initial_bytes' if known else 'undefined_initial_texture',ident,row['subresource'],{})

    def changed(self,ident,sub,operation,fields):
        prior=self.current[(ident,sub)]
        return self._node(operation,ident,sub,fields,[prior])

    def copy(self,dst,src,fields):
        r=self.resources[dst]
        for row in r.get('texture_layouts',[]):
            sub=row['subresource'];self._node('native_copy_resource',dst,sub,fields,[self.current[(src,sub)]])

    def resolve(self,fields):
        dst,sub=fields['pDstResource'],fields['DstSubresource'];src,source_sub=fields['pSrcResource'],fields['SrcSubresource']
        dr=self.resources[dst];sr=self.resources[src]
        if dr['descriptor'].get('SampleDesc',{}).get('Count',1)!=1 or sr['descriptor'].get('SampleDesc',{}).get('Count',1)<=1:
            raise UnresolvedTarget('Resolve requires actual multisample source and single-sample destination')
        a=dr['texture_layouts'][sub];b=sr['texture_layouts'][source_sub]
        if (a['width'],a['height'])!=(b['width'],b['height']):raise UnresolvedTarget('Resolve source/destination extent mismatch')
        self._node('native_resolve_subresource',dst,sub,fields,[self.current[(src,source_sub)]])
        dr['texture_segments'][sub]=[]

    def clear(self,view,operation,fields):
        ident,subs=view_subresources(self.resources,view)
        for sub in subs:
            self.changed(ident,sub,operation,fields)
            # No diagnostic clear pixels are manufactured. Actual clear is an
            # executable producer with the prior version (needed for aspect masks).
            self.resources[ident]['texture_segments'][sub]=[]

    def draw(self,draw):
        state=draw['state'];reads={};dependencies=[]
        for stage in ('VS','HS','DS','GS','PS','CS'):
            for slot,view in enumerate(state[stage]['SRVs']):
                if not view:continue
                ident,subs=view_subresources(self.resources,view)
                nodes=[self.current[(ident,sub)] for sub in subs]
                reads[f'{stage}:{slot}']={'view':view,'resource':ident,'subresources':list(subs),'versions':nodes}
                dependencies.extend(nodes)
        outputs=[];om=state['OM']
        depth_view=om['DepthView'];depth_write=False
        if depth_view:
            ident,subs=view_subresources(self.resources,depth_view)
            dependencies.extend(self.current[(ident,sub)] for sub in subs)
            ds=self.resources[om['DepthStencilState']]['descriptor'] if om['DepthStencilState'] else {'DepthEnable':True,'DepthWriteMask':1,'StencilEnable':False}
            flags=(self.resources[depth_view]['descriptor'] or {}).get('Flags',0)
            depth_write=ds['DepthEnable'] and ds['DepthWriteMask']!=0 and not flags&1
            if ds['StencilEnable'] and ds['StencilWriteMask'] and not flags&2:
                depth_write|=any(ds[face][operation]!=1 for face in ('FrontFace','BackFace') for operation in ('StencilFailOp','StencilDepthFailOp','StencilPassOp'))
        blend=self.resources[om['BlendState']]['descriptor'] if om['BlendState'] else None
        views=[]
        for slot,view in enumerate(om['RenderTargets']):
            if blend is not None and blend['RenderTarget'][slot if blend['IndependentBlendEnable'] else 0]['RenderTargetWriteMask']==0:continue
            views.append(view)
        if depth_write:views.append(depth_view)
        for view in views:
            if not view:continue
            ident,subs=view_subresources(self.resources,view)
            for sub in subs:
                node=self._node('potential_native_draw_target',ident,sub,{'draw_index':draw['index'],'view':view},
                    [self.current[(ident,sub)],*dependencies])
                self.resources[ident]['texture_segments'][sub]=[];outputs.append(node)
        draw['texture_inputs']=reads;draw['target_producers']=outputs
