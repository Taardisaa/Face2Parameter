import unittest
import xml.etree.ElementTree as ET
from verify_present_scope_xml import verify


def fixture():
    root = ET.fromstring('<rdc><header><driver>D3D11</driver></header><chunks/></rdc>')
    chunks = root.find('chunks')
    def chunk(name):
        return ET.SubElement(chunks, 'chunk', name=name, chunkIndex=str(len(chunks)), threadID='44')
    def value(node, name, text, tag='uint'):
        ET.SubElement(node, tag, name=name).text = str(text)
    for ident, usage, cpu in [(17, 0, 0), (18, 3, 0x20000)]:
        row=chunk('ID3D11Device::CreateTexture2D');value(row,'pTexture',ident,'ResourceId')
        desc=ET.SubElement(row,'struct',name='Descriptor')
        for key,num in [('Width',32),('Height',32),('Usage',usage),('CPUAccessFlags',cpu)]:value(desc,key,num)
    def annot(key, text, number):
        row=chunk('ID3D11DeviceContext::SetCommandAnnotation');value(row,'Context',5,'ResourceId')
        value(row,'key',key,'string');value(row,'valueType',number,'enum');value(row,'valueVectorWidth',1)
        union=ET.SubElement(row,'struct',name='value')
        if number==8:value(union,'string',text,'string')
        else:
            words=ET.SubElement(union,'array',name='vector.uint64')
            for word in [text,0,0,0]:ET.SubElement(words,'uint').text=str(word)
    annot('hs2_scope.begin','synthetic_nonce',8);annot('hs2_scope.target',17,9)
    chunk('ID3D11DeviceContext::DrawIndexed')
    row=chunk('ID3D11DeviceContext::CopyResource');value(row,'Context',5,'ResourceId');value(row,'pSrcResource',17,'ResourceId');value(row,'pDstResource',18,'ResourceId')
    annot('hs2_scope.end','synthetic_nonce',8)
    return root


class ScopeXmlTests(unittest.TestCase):
    def test_actual_schema_marker_and_copy_fixture(self):
        result=verify(fixture(),'synthetic_nonce',32,32)
        self.assertEqual(result['actual_target_resource'],17)
        self.assertEqual(result['readback_candidates'][0]['staging_resource'],18)
        self.assertFalse(result['native_pixel_equivalence_accepted'])

    def test_ordinary_retry_cannot_pass(self):
        root=fixture();chunks=root.find('chunks')
        for row in list(chunks):
            if row.attrib['name'].endswith('SetCommandAnnotation'):chunks.remove(row)
        with self.assertRaisesRegex(ValueError,'reject retry frame'):verify(root,'synthetic_nonce',32,32)

    def test_unrelated_context_copy_and_output_write_do_not_erase_root(self):
        root=fixture();chunks=root.find('chunks')
        unrelated=ET.fromstring('<chunk name="ID3D11DeviceContext::CopyResource" chunkIndex="99" threadID="88"><ResourceId name="Context">9</ResourceId><ResourceId name="pSrcResource">111</ResourceId><ResourceId name="pDstResource">112</ResourceId></chunk>')
        output=ET.fromstring('<chunk name="ID3D11DeviceContext::CopyResource" chunkIndex="100" threadID="44"><ResourceId name="Context">5</ResourceId><ResourceId name="pSrcResource">113</ResourceId><ResourceId name="pDstResource">17</ResourceId></chunk>')
        chunks.insert(5,unrelated);chunks.insert(6,output)
        self.assertTrue(verify(root,'synthetic_nonce',32,32)['marker_and_copy_provenance_passed'])

    def test_wrong_nonce_duplicate_context_and_readback_rejected(self):
        with self.assertRaisesRegex(ValueError,'nonce mismatch'):verify(fixture(),'other',32,32)
        root=fixture();root.find('chunks').append(root.find('chunks')[-1])
        with self.assertRaisesRegex(ValueError,'Duplicate'):verify(root,'synthetic_nonce',32,32)
        root=fixture();root.find('chunks')[-1].find('./ResourceId').text='6'
        with self.assertRaisesRegex(ValueError,'same actual'):verify(root,'synthetic_nonce',32,32)
        root=fixture();root.find('chunks')[5].find('./ResourceId[@name="pSrcResource"]').text='123'
        with self.assertRaisesRegex(ValueError,'staging transfer'):verify(root,'synthetic_nonce',32,32)


if __name__=='__main__':unittest.main()
