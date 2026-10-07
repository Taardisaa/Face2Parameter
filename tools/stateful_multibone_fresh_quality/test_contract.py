"""Synthetic contract checks only; not actual quality/anatomy certification."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .run import require_native,validate_threshold_binding,history_diff,raw_spans,ROOT

class ContractTests(unittest.TestCase):
    def test_unity_float32_equivalence_is_exact(self):
        expected=[.48,.52,.47,.53]*14+[.48,.52,.47]
        snap={'character':{'native_count':59,'shape_value_face':np.asarray(expected,dtype=np.float32).astype(float).tolist()}}
        require_native(snap,expected,'fixture')
        snap['character']['shape_value_face'][0]=float(np.nextafter(np.float32(.48),np.float32(1)))
        with self.assertRaises(ValueError):require_native(snap,expected,'fixture')
    def test_neutral_cannot_masquerade_as_varied_baseline(self):
        expected=[.48]*59
        with self.assertRaises(ValueError):require_native({'character':{'native_count':59,'shape_value_face':[.5]*59}},expected,'baseline')
    def test_nan_and_bool_expected_rejected(self):
        snap={'character':{'native_count':59,'shape_value_face':[.5]*59}}
        for invalid in (True,float('nan')):
            with self.assertRaises(ValueError):require_native(snap,[invalid]+[.5]*58,'fixture')
    def test_config_and_producer_byte_binding(self):
        validate_threshold_binding(ROOT/'outputs/stateful_base_comparison_20261005/comparison_config.json')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'changed.json'; path.write_text('{"acceptance":{"rms":0.1}}')
            with self.assertRaises(ValueError):validate_threshold_binding(path)
    def test_history_does_not_drop_private_or_unknown_fields(self):
        early={'all_selected_actual_runtime_bone_records':[{'name':'bone','scale':[1,1,1],'runtime_baseline':{'fields':{'length':1,'opaque_private':[2,3]}}}],
            'actual_palette_ancestor_records':[{'path':'a','local_scale':[1,1,1]}]}
        late=copy.deepcopy(early);late['all_selected_actual_runtime_bone_records'][0]['runtime_baseline']['fields']['opaque_private'][1]=4
        report=history_diff(early,late)
        self.assertFalse(report['all_selected_actual_runtime_bone_records']['exactly_identical'])
        self.assertEqual(report['all_selected_actual_runtime_bone_records']['changed_records'][0]['late'],late['all_selected_actual_runtime_bone_records'][0])
        self.assertTrue(report['actual_palette_ancestor_records']['exactly_identical'])
        self.assertFalse(report['external_writers_identified'])
    def test_history_identity_scope_drift_rejected(self):
        a={'all_selected_actual_runtime_bone_records':[{'name':'a'}],'actual_palette_ancestor_records':[]}
        b=copy.deepcopy(a);b['all_selected_actual_runtime_bone_records'][0]['name']='b'
        with self.assertRaises(ValueError):history_diff(a,b)
    def test_raw_span_has_no_distance_anatomy_or_response_claim(self):
        mesh={'baked':{'vertices':[[0,0,0],[2,0,0],[0,3,0]],'triangles':[0,1,2],
            'submeshes':[{'submesh_index':0,'topology':'Triangles','indices_apply_base_vertex':True,'indices':[0,1,2]}]},
            'source':{'triangles':[0,1,2]}}
        result=raw_spans(mesh)
        self.assertEqual(result['submeshes'][0]['span_xyz_renderer_raw_game_units'],[2,3,0])
        for key in ('anatomical_face_width_or_length_certified','single_factor_physical_response_certified','scale_parameters_are_physical_distances','crossbase_equivalence_certified','frame_policy_independently_certified'):
            self.assertFalse(result[key])

if __name__=='__main__':unittest.main()
