"""CPU transport policy checks; synthetic SD files are not HS2/RDC acceptance."""
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import unittest
import tempfile
from src.render.capture_mutation import mutate_capture,MutationRejected,digest

def source(tmp_path):
    xml=tmp_path/'source.zip.xml';zip_path=tmp_path/'source.zip'
    xml.write_text('<rdc><header><driver>D3D11</driver></header><chunks version="20"><chunk chunkIndex="0" name="ID3D11Device::CreateBuffer"><buffer name="InitialData" byteLength="8">0</buffer></chunk><chunk chunkIndex="1" name="ID3D11DeviceContext::Unmap"><buffer name="MapWrittenData" byteLength="8">0</buffer><string name="unknown">retained</string></chunk></chunks></rdc>')
    with zipfile.ZipFile(zip_path,'w') as z:
        z.writestr('000000',b'abcdefgh');z.writestr('000001',b'unknown data retained')
        z.writestr('thumb.png',b'original thumbnail');z.writestr('diagnostic.log',b'original diagnostics')
    data=tmp_path/'candidate.bin';data.write_bytes(b'XYZ')
    plan={'source_xml_sha256':digest(xml.read_bytes()),'source_zip_sha256':digest(zip_path.read_bytes()),'patches':[{
      'chunk_index':1,'chunk_name':'ID3D11DeviceContext::Unmap','buffer_ordinal':0,'expected_buffer_index':0,
      'expected_blob_sha256':digest(b'abcdefgh'),'byte_offset':2,'expected_range_sha256':digest(b'cde'),
      'replacement_file':'candidate.bin','replacement_sha256':digest(data.read_bytes()),
      'source_witness':'synthetic exact source transfer test','semantic_role':'current_positions'}]}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));return xml,path,plan

class MutationChecks(unittest.TestCase):
    def test_shared_blob_copy_on_write_preserves_other_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path=Path(directory);xml,path,plan=source(tmp_path);out=tmp_path/'candidate.zip.xml';receipt=mutate_capture(xml,path,out)
            with zipfile.ZipFile(out.with_suffix('')) as z:
                self.assertEqual(z.read('000000'),b'abcdefgh');self.assertEqual(z.read('000001'),b'unknown data retained')
                self.assertEqual(z.read('000002'),b'abXYZfgh')
                self.assertEqual(z.read('thumb.png'),b'original thumbnail');self.assertEqual(z.read('diagnostic.log'),b'original diagnostics')
            root=ET.parse(out).getroot();refs=list(root.iter('buffer'))
            self.assertEqual(refs[0].text,'0');self.assertEqual(refs[1].text,'2');self.assertEqual(root.find('.//string').text,'retained')
            self.assertEqual(receipt['copy_on_write_occurrences'],1);self.assertFalse(receipt['candidate_geometry_complete'])

    def test_wrong_provenance_and_selectors_fail_before_output(self):
        for field,value in [('expected_range_sha256','wrong'),('expected_buffer_index',99),('byte_offset',7),('source_witness',''),('chunk_name','other command')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                tmp_path=Path(directory);xml,path,plan=source(tmp_path);plan['patches'][0][field]=value;path.write_text(json.dumps(plan))
                out=tmp_path/'rejected.zip.xml'
                with self.assertRaises(MutationRejected):mutate_capture(xml,path,out)
                self.assertFalse(out.exists());self.assertFalse(out.with_suffix('').exists())

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path=Path(directory);xml,path,plan=source(tmp_path);plan['patches'].append(plan['patches'][0].copy());path.write_text(json.dumps(plan))
            with self.assertRaises(MutationRejected):mutate_capture(xml,path,tmp_path/'bad.zip.xml')

if __name__=='__main__':unittest.main()
