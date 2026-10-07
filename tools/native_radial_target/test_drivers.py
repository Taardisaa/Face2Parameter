"""Synthetic verifier tests, never a claim of actual driver runtime acceptance."""
from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from tools.abmx_replay.validate_trace import read, sha
from tools.native_radial_target.compiler import compile_native_target
from tools.native_radial_target.test_native_target import fixture
from tools.native_radial_target.validate_drivers import (
    ENUM,
    GAZE,
    POLICY,
    ROLES,
    inspect_snapshot,
    prepare_driver_guard,
)

ROOT = Path(__file__).resolve().parents[2]


def driver_fixture(directory):
    history, declared, contract, snapshot, meta, _ = fixture(directory)
    files = {}
    roles = {}
    for role in ROLES:
        p = directory / (role + '.synthetic.txt')
        p.write_text('synthetic verifier source role ' + role, encoding='utf-8')
        roles[role] = str(p.resolve()); files[str(p.resolve())] = sha(p)
    snapshot['game']['game_assembly_sha256'] = files[roles['game_pe']]
    eye_states = []
    for i, name in enumerate(GAZE):
        raw = {'name': name, 'id': 3000+i, 'path': '/synthetic/'+name,
               'local_position': [0.,0.,0.], 'local_scale': [1.,1.,1.], 'local_rotation_xyzw': [0.,0.,0.,1.]}
        snapshot['transforms'].append(raw)
        eye_states.append({k:v for k,v in raw.items() if k != 'id'} | {'transform_id': raw['id']})
    mouth = next(t for t in snapshot['transforms'] if t['name'] == 'cf_J_MouthMove')
    mouth = {k:v for k,v in mouth.items() if k != 'id'} | {'transform_id': mouth['id']}
    # Deliberately NO_LOOK index 1: integer 3 is not a verifier assumption.
    patterns = [{'index':i,'name':n,'value':ENUM[n]} for i,n in enumerate(('TARGET','NO_LOOK','AWAY','FORWARD','CONTROL'))]
    snapshot['native_face_drivers'] = {'schema_version':1,'frame':10,'actor_transform_id':123,
        'bridge_mvid':'synthetic-bridge','game_mvid':'synthetic-game-mvid',
        'configured':{'mouth_adjust_width':False,'eyes_look_pattern':1},'mouth_adjust_target':mouth,
        'gaze_bones':eye_states,'eye_controller':{'enabled':True,'active_and_enabled':True,'script_enabled':True,
            'global_enabled':True,'delta_time':.02,'pattern':1,'target':None,
            'fixed_angles_xyzw':[[0.,0.,0.,1.],[0.,0.,0.,1.]],'eye_objects':copy.deepcopy(eye_states),
            'patterns':patterns,'resolved_type':{'name':'NO_LOOK','value':0}}}
    snapshot['character']['expression'] = {'mouth_adjust_width':False,'eyes_look_pattern':1}
    p = Path(history['geometry']['path']); p.write_text(json.dumps(snapshot),encoding='utf-8')
    history['geometry']['sha256'] = sha(p)
    dc = {'schema_version':1,'policy':POLICY,'bridge_mvid':'synthetic-bridge','game_mvid':'synthetic-game-mvid',
          'eye_look_enum':ENUM,'source_roles':roles,'source_files':files}
    cp = directory/'driver_contract.json'; cp.write_text(json.dumps(dc),encoding='utf-8')
    artifact = compile_native_target(history, declared, contract)
    return artifact, contract, cp, snapshot, meta, dc


class DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = ROOT/'outputs/native_radial_target_20261006'; parent.mkdir(parents=True,exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(prefix='driver_verifier_',dir=parent)
        cls.fixture = driver_fixture(Path(cls.tmp.name))
        cls.artifact, cls.contract, cls.driver_path, cls.source, cls.meta, cls.dc = cls.fixture
        cls.context = prepare_driver_guard(cls.artifact, cls.contract, cls.driver_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def candidate(self):
        s = copy.deepcopy(self.source)
        for row in s['abmx_runtime']['bones']:
            p = next(r for r in self.artifact['executed_patches'] if r['name']==row['name'])
            row.update({k:v for k,v in p.items() if k!='name'})
        for name, target in self.artifact['desired_target_locals'].items():
            next(t for t in s['transforms'] if t['name']==name).update(copy.deepcopy(target))
        target = self.artifact['desired_target_locals']['cf_J_MouthMove']
        s['native_face_drivers']['mouth_adjust_target'].update(copy.deepcopy(target))
        return s

    def test_identity_and_candidate_dynamic_index(self):
        self.assertTrue(self.context.verify_identity(self.source,current_trace_metadata=self.meta)['passed'])
        r = self.context.verify_candidate(self.candidate(),current_trace_metadata=self.meta)
        self.assertEqual(r['actual_pattern_index'],1)
        self.assertFalse(r['candidate_used_to_generate_target'])
        self.assertEqual(len(r['all30_target_locals']),30)

    def test_mouth_driver_enabled_and_expression_mismatch_refused(self):
        for change in ('configured','expression'):
            s = self.candidate()
            block = s['native_face_drivers']['configured'] if change=='configured' else s['character']['expression']
            block['mouth_adjust_width'] = True
            with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_names_and_numeric_enum_not_interchangeable(self):
        for resolved in ({'name':'NONE','value':0},{'name':'NO_LOOK','value':3},{'name':'NO_LOOK','value':False}):
            s = self.candidate(); s['native_face_drivers']['eye_controller']['resolved_type'] = resolved
            with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_actual_controller_disagrees_with_status(self):
        s = self.candidate(); s['native_face_drivers']['eye_controller']['pattern'] = 3
        with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_eye_object_pointer_not_name_discovery(self):
        s = self.candidate(); s['native_face_drivers']['eye_controller']['eye_objects'][0]['transform_id'] += 99
        with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_duplicate_gaze_and_mouth_pointer_refused(self):
        for which in ('gaze','mouth'):
            s = self.candidate()
            if which=='gaze':s['native_face_drivers']['gaze_bones'][1]=copy.deepcopy(s['native_face_drivers']['gaze_bones'][0])
            else:s['native_face_drivers']['mouth_adjust_target']['path']='/different'
            with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_sameframe_and_cursor_prefix_required(self):
        for field in ('frame','completed_calls'):
            s = self.candidate()
            if field=='frame':s['native_face_drivers']['frame'] += 1
            else:s['abmx_trace_cursor'][field] += 1
            with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_source_fixed_angles_and_gaze_not_replaced_by_candidate(self):
        s = self.candidate(); eye=s['native_face_drivers']['eye_controller']
        q=[0.,.01,0.,.99995]
        eye['fixed_angles_xyzw'][0]=q
        eye['eye_objects'][0]['local_rotation_xyzw']=q
        s['native_face_drivers']['gaze_bones'][0]['local_rotation_xyzw']=q
        next(t for t in s['transforms'] if t['name']==GAZE[0])['local_rotation_xyzw']=q
        with self.assertRaises(ValueError): self.context.verify_candidate(s,self.meta)

    def test_actual_target_wrong_even_when_pointer_matches(self):
        s = self.candidate(); n='cf_J_MouthMove'
        next(t for t in s['transforms'] if t['name']==n)['local_scale'][0]=1.
        s['native_face_drivers']['mouth_adjust_target']['local_scale'][0]=1.
        with self.assertRaisesRegex(ValueError,'ALL30 locals'):self.context.verify_candidate(s,self.meta)

    def test_native_head_and_assembly_invalidations(self):
        for field in ('head','native','bridge'):
            s = self.candidate()
            if field=='head':s['character']['head_id']=0
            elif field=='native':s['character']['shape_value_face'][0]=.7
            else:s['native_face_drivers']['bridge_mvid']='different'
            with self.assertRaises(ValueError):self.context.verify_candidate(s,self.meta)

    def test_exact_bool_and_finite_fixed_angles(self):
        for bad in ('bool','nan'):
            s = self.candidate()
            if bad=='bool':s['native_face_drivers']['eye_controller']['enabled']=1
            else:s['native_face_drivers']['eye_controller']['fixed_angles_xyzw'][0][0]=float('nan')
            with self.assertRaises(ValueError):self.context.verify_candidate(s,self.meta)

    def test_view_status_true_not_sufficient(self):
        s=self.candidate();d=s['native_face_drivers']
        v={'native_face_drivers_before_render':copy.deepcopy(d),'native_face_drivers_after_render':copy.deepcopy(d),
           'paired_native_face_drivers_unchanged':True}
        self.assertTrue(self.context.verify_view(v,s)['passed'])
        v['native_face_drivers_after_render']['configured']['mouth_adjust_width']=True
        with self.assertRaises(ValueError):self.context.verify_view(v,s)

    def test_source_file_and_contract_changes_refused(self):
        p=Path(self.dc['source_roles']['enum_source']);old=p.read_bytes()
        try:
            p.write_bytes(old+b' changed')
            with self.assertRaises(ValueError):self.context.verify_candidate(self.candidate(),self.meta)
        finally:p.write_bytes(old)
        p=self.driver_path;old=p.read_bytes()
        try:
            p.write_bytes(old+b' ')
            with self.assertRaises(ValueError):self.context.verify_candidate(self.candidate(),self.meta)
        finally:p.write_bytes(old)

    def test_context_nonserializable_and_numeric_tamper(self):
        with self.assertRaises(TypeError):pickle.dumps(self.context)
        a=self.artifact;old=a['desired_target_locals']['cf_J_MouthMove']['local_scale'][0]
        try:
            a['desired_target_locals']['cf_J_MouthMove']['local_scale'][0]=1.
            with self.assertRaises(ValueError):self.context.verify_candidate(self.candidate(),self.meta)
        finally:a['desired_target_locals']['cf_J_MouthMove']['local_scale'][0]=old

    def test_legacy_actual_dataset_missing_drivers_rejected(self):
        path=ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261005/native_radial_head2_v2/source_history.json'
        self.assertTrue(path.is_file(),'Actual legacy source geometry expected')
        with self.assertRaisesRegex(ValueError,'legacy evidence refused'):
            inspect_snapshot(read(path),self.dc)


if __name__=='__main__':unittest.main()
