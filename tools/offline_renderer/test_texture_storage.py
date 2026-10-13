"""CPU transfer checks with pitched rows/volumes, BC blocks and shared targets."""
import unittest
from src.render.texture_storage import layouts,update_box,UnresolvedTexture
from src.render.captured_pipeline import CapturedPipeline,decode,UnresolvedPipeline
from src.render.shared_targets import SharedTargets,view_subresources

def texture(fmt,w,h=1,d=1):
    kind='CreateTexture3D' if d>1 else 'CreateTexture2D'
    desc={'Width':w,'Height':h,'Depth':d,'ArraySize':1,'MipLevels':1,'Format':fmt,'SampleDesc':{'Count':1}}
    return layouts(kind,desc)[0]

class TextureStorageTests(unittest.TestCase):
    def read(self,segments,total):return CapturedPipeline.buffer_range(segments,0,total)

    def test_padded_rgba_box_preserves_surrounding_texels(self):
        row=texture(28,4,3);original=bytes(range(48))
        box={'left':1,'top':1,'front':0,'right':3,'bottom':3,'back':1}
        result=update_box([(0,original)],row,box,b'abcdefghXXXXijklmnop',12,0)
        expected=original[:20]+b'abcdefgh'+original[28:36]+b'ijklmnop'+original[44:]
        self.assertEqual(self.read(result,48),expected)

    def test_volume_pitch_padding_is_not_pixel_data(self):
        row=texture(54,2,2,2)
        result=update_box([],row,None,b'abcdXXefghYYijklXXmnop',6,12)
        self.assertEqual(self.read(result,16),b'abcdefghijklmnop')

    def test_bc_blocks_are_transferred_without_decoding(self):
        row=texture(71,8,8);original=bytes(range(32))
        box={'left':4,'top':0,'front':0,'right':8,'bottom':8,'back':1}
        result=update_box([(0,original)],row,box,b'abcdefghXXXXijklmnop',12,0)
        self.assertEqual(self.read(result,32),original[:8]+b'abcdefgh'+original[16:24]+b'ijklmnop')
        box['left']=3
        with self.assertRaises(UnresolvedTexture):update_box([(0,original)],row,box,b'abcdefghXXXXijklmnop',12,0)

    def test_r1_and_shared_exponent_have_distinct_pitch(self):
        self.assertEqual(texture(66,13)['row_pitch'],2)
        self.assertEqual(texture(67,13)['row_pitch'],52)

    def test_repeated_initial_mip_fields_not_dropped(self):
        def buf(i):return {'kind':'buffer','attributes':{'name':'SubresourceContents','byteLength':'4'},'value':str(i)}
        result=decode({'kind':'chunk','attributes':{},'children':[buf(0),buf(1)]})
        self.assertEqual([x['buffer_index'] for x in result['SubresourceContents']['__repeated_field__']],[0,1])

    def test_shared_depth_survives_color_only_draw(self):
        resources={}
        for ident in (1,2):
            desc={'Width':2,'Height':2,'ArraySize':1,'MipLevels':1,'Format':28,'SampleDesc':{'Count':1}}
            resources[ident]={'kind':'CreateTexture2D','descriptor':desc,'texture_layouts':layouts('CreateTexture2D',desc),'texture_segments':{0:[]}}
        for view,ident,kind in [(10,1,'CreateRenderTargetView'),(11,1,'CreateShaderResourceView'),(20,2,'CreateDepthStencilView')]:
            resources[view]={'kind':kind,'fields':{'pResource':ident},'descriptor':None}
        resources[30]={'descriptor':{'DepthEnable':True,'DepthWriteMask':0,'StencilEnable':False}}
        graph=SharedTargets(resources);graph.created(1);graph.created(2)
        graph.clear(20,'native_clear_depth_stencil',{'ClearFlags':1,'Depth':1})
        depth_version=graph.current[(2,0)]
        state={s:{'SRVs':[]} for s in ('VS','HS','DS','GS','PS','CS')}
        state['OM']={'RenderTargets':[10],'DepthView':20,'DepthStencilState':30,'BlendState':0}
        draw={'index':5,'state':state};graph.draw(draw)
        self.assertEqual(graph.current[(2,0)],depth_version)
        self.assertIn(depth_version,graph.nodes[draw['target_producers'][0]]['dependencies'])
        graph.copy(2,1,{})
        self.assertEqual(graph.nodes[graph.current[(2,0)]]['dependencies'],[graph.current[(1,0)]])

    def test_array_mip_views_select_actual_subresources(self):
        desc={'Width':4,'Height':4,'ArraySize':2,'MipLevels':3,'Format':28}
        resources={1:{'texture_layouts':layouts('CreateTexture2D',desc)},2:{'kind':'CreateShaderResourceView','fields':{'pResource':1},'descriptor':{'Texture2DArray':{'MostDetailedMip':1,'MipLevels':2,'FirstArraySlice':1,'ArraySize':1}}}}
        self.assertEqual(view_subresources(resources,2),(1,(4,5)))
if __name__=='__main__':unittest.main()
