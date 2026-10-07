"""Synthetic parser/provenance negatives and density-independent normalization."""
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .facescape_target_v1 import (
    SHAPE_KIND,
    _common_coordinates,
    _validated_candidate_conversion,
    binding,
    freeze_raw_target,
    load_raw_target,
    parse_obj,
    proper_rigid,
    sha,
    surface_moments,
    transform_candidate,
)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.obj=self.root/'1_neutral.obj'
        self.readme=self.root/'readme'
        self.obj.write_text('v 0 0 0\nv 2 0 0\nv 0 2 0\nvt 0 0\nvt 1 0\nvt 0 1\nvt .5 .5\nvn 0 0 1\nf 1/4/1 2/2/1 3/3/1\n',encoding='utf-8')
        self.readme.write_text('sample FaceScape TU-model. Please do NOT distribute.',encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def acquire(self):
        archive=self.root/'sample.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for member,path in [('sample_tu_model/1_neutral.obj',self.obj),('sample_tu_model/readme',self.readme)]:
                info=tarfile.TarInfo(member);info.size=path.stat().st_size;tar.addfile(info,io.BytesIO(path.read_bytes()))
        receipt=self.root/'receipt.json';receipt.write_text(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size}),encoding='utf-8')
        extraction=self.root/'extraction.json'
        extraction.write_text(json.dumps({'TU_vertices':3,'TU_faces':1,'asset_receipts':[
            {**binding(self.obj),'member':'sample_tu_model/1_neutral.obj'},
            {**binding(self.readme),'member':'sample_tu_model/readme'}]}),encoding='utf-8')
        return receipt,extraction

    def frozen(self):
        receipt,extraction=self.acquire()
        return freeze_raw_target(self.obj,self.readme,receipt,extraction,self.root/'contract.json')

    def test_uv_indices_and_source_order_not_vertex_welded(self):
        target=parse_obj(self.obj)
        np.testing.assert_array_equal(target.faces,[[0,1,2]])
        np.testing.assert_array_equal(target.face_uv_indices,[[3,1,2]])
        np.testing.assert_array_equal(target.face_normal_indices,[[0,0,0]])
        self.assertEqual(target.face_obj_signed_indices.tolist(),[[[1,4,1],[2,2,1],[3,3,1]]])
        self.assertFalse(target.vertices.flags.writeable)
        self.assertFalse(target.faces.flags.writeable)

    def test_relative_indices_preserved_separately(self):
        self.obj.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n')
        target=parse_obj(self.obj)
        np.testing.assert_array_equal(target.faces,[[0,1,2]])
        np.testing.assert_array_equal(target.face_obj_signed_indices[:,:,0],[[-3,-2,-1]])
        np.testing.assert_array_equal(target.face_uv_indices,[[-1,-1,-1]])

    def test_nonfinite_out_of_range_zero_repeat_and_polygon_refuse(self):
        for text in ['v nan 0 0\nf 1 2 3', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4',
                     'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 2 3', 'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 2',
                     'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3 1',
                     'v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/10 2/2 3/3']:
            self.obj.write_text(text)
            with self.subTest(text=text),self.assertRaises(ValueError):parse_obj(self.obj)

    def test_actual_loader_contract_provenance_no_vertex_serialization(self):
        descriptor=self.frozen();target=load_raw_target(descriptor['path'],descriptor['sha256'])
        self.assertEqual(len(target.vertices),3)
        contract=json.loads(Path(descriptor['path']).read_text())
        self.assertNotIn('vertices',contract)
        self.assertEqual(contract['units'],'unknown_raw_sample_units')
        self.assertIsNone(contract['sample_identity']['subject_id'])

    def test_coherent_redigest_units_identity_and_type_reject(self):
        descriptor=self.frozen();original=json.loads(Path(descriptor['path']).read_text())
        for key,value in [('units','hs2_cache_units'),('schema_version',True),('sample_identity',{'subject_id':1})]:
            tamper=dict(original);tamper[key]=value
            path=self.root/'tamper.json';path.write_text(json.dumps(tamper))
            with self.subTest(key=key),self.assertRaises(ValueError):load_raw_target(path,sha(path))

    def test_source_and_archive_mutation_reject(self):
        descriptor=self.frozen();source=self.obj.read_text();self.obj.write_text(source+'# changed\n')
        with self.assertRaises(ValueError):load_raw_target(descriptor['path'],descriptor['sha256'])
        self.obj.write_text(source)
        archive=self.root/'sample.tar.gz';archive.write_bytes(archive.read_bytes()+b'x')
        with self.assertRaises(ValueError):load_raw_target(descriptor['path'],descriptor['sha256'])

    def test_license_and_receipt_counts_refuse(self):
        self.readme.write_text('TU-model without authorized sample restriction')
        receipt,extraction=self.acquire()
        with self.assertRaises(ValueError):freeze_raw_target(self.obj,self.readme,receipt,extraction,self.root/'bad.json')
        self.readme.write_text('TU-model. Please do NOT distribute.')
        receipt,extraction=self.acquire();data=json.loads(extraction.read_text());data['TU_vertices']=2;extraction.write_text(json.dumps(data))
        with self.assertRaises(ValueError):freeze_raw_target(self.obj,self.readme,receipt,extraction,self.root/'bad.json')

    def test_legacy_contract_refused(self):
        path=self.root/'legacy.json';path.write_text(json.dumps({'kind':'full_mesh','schema_version':1,'vertices':[[0,0,0]]}))
        with self.assertRaises(ValueError):load_raw_target(path,sha(path))
        with self.assertRaises(ValueError):transform_candidate(np.zeros((3,3)),[[0,1,2]],{'kind':SHAPE_KIND},{})

    def test_exact_surface_integrals_and_subdivision_invariance(self):
        vertices=np.array([[0.,0,0],[2.,0,0],[0.,2,0]])
        coarse=surface_moments(vertices,np.array([[0,1,2]]))
        vertices=np.vstack([vertices,vertices.mean(0)])
        fine=surface_moments(vertices,np.array([[0,1,3],[1,2,3],[2,0,3]]))
        np.testing.assert_allclose(coarse['centroid'],[2/3,2/3,0],atol=1e-15)
        self.assertAlmostEqual(coarse['rms_radius'],2/3)
        for key in ['area','centroid','rms_radius']:np.testing.assert_allclose(coarse[key],fine[key],atol=1e-14)

    def test_zero_area_kept_and_not_used_as_hidden_mask(self):
        vertices=np.array([[0.,0,0],[2.,0,0],[0.,2,0],[1.,0,0]])
        result=surface_moments(vertices,np.array([[0,1,2],[0,1,3]]))
        self.assertEqual(result['zero_area_triangle_count'],1)
        with self.assertRaises(ValueError):surface_moments(vertices,np.array([[0,1,3]]))

    def test_shared_scale_does_not_recenter_or_rescale_candidate(self):
        protocol={'kind':SHAPE_KIND,'hs2_common_reference':{'surface_moments':{'centroid':[1,2,3],'rms_radius':2.}}}
        vertices=np.array([[1.,2,3],[3.,4,5]])
        baseline=_common_coordinates(vertices,protocol)
        shifted=_common_coordinates(vertices+[2.,0,0],protocol)
        np.testing.assert_allclose(shifted-baseline,[[1,0,0],[1,0,0]])
        np.testing.assert_allclose(_common_coordinates((vertices-[1,2,3])*2+[1,2,3],protocol),baseline*2)

    def test_scale_reflection_rigid_and_unknown_candidate_units_reject(self):
        for diagonal in [[2,2,2,1],[-1,1,1,1]]:
            with self.assertRaises(ValueError):proper_rigid(np.diag(diagonal))
        metadata={'head_id':0,'units':'mm','coordinate_frame':'hs2_cached_head_fk'}
        with self.assertRaises(ValueError):_validated_candidate_conversion(np.zeros((3,3)),np.array([[0,1,2]]),{},metadata)


if __name__=='__main__':
    unittest.main()
