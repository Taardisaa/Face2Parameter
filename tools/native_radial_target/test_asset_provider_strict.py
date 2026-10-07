"""Strict revision tests using frozen actual identity SOURCE, never game calls."""
from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_stable_lowering.compiler import digest_json
from tools.native_radial_target import asset_provider_strict as strict

ROOT=Path(__file__).resolve().parents[2]


class StrictProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(prefix='strict_asset_tests_',dir=ROOT/'outputs/native_radial_target_20261006')
        cls.out=Path(cls.tmp.name)
        cls.v1=ROOT/'outputs/native_radial_target_20261006/asset_provider_final_v1/staged/asset_contract.json'
        cls.descriptor=strict.wrap_existing_v1({'path':str(cls.v1),'sha256':sha(cls.v1)},cls.out/'strict')
        cls.contract=read(cls.descriptor['path'])
        cls.provider=strict.prepare_strict_asset_provider(cls.descriptor['path'],cls.descriptor['sha256'])
        cls.original=read(cls.v1)
        cls.source=read(cls.original['source_history']['geometry']['path'])
        cls.old_files={str(p):sha(p) for p in (ROOT/'data/hs2_head/head_3').rglob('*') if p.is_file()}
        cls.v1_sha=sha(ROOT/'tools/native_radial_target/asset_provider.py')

    @classmethod
    def tearDownClass(cls):
        assert all(sha(p)==d for p,d in cls.old_files.items())
        assert sha(ROOT/'tools/native_radial_target/asset_provider.py')==cls.v1_sha
        cls.tmp.cleanup()

    def modified(self,name,change):
        c=copy.deepcopy(self.contract);change(c)
        c.pop('contract_binding_sha256');c['contract_binding_sha256']=digest_json(c)
        p=self.out/(name+'.json');p.write_text(json.dumps(c,allow_nan=False),encoding='utf-8')
        return p,sha(p)

    def test_actual_source_full8_and_complete_dependency_closure(self):
        self.assertTrue(self.provider.verify_snapshot_assets(self.source)['passed'])
        bound=self.contract['source_files']
        for rel in strict._HELPER_ROOTS:
            self.assertIn(str((ROOT/rel).resolve()),bound)
        self.assertGreater(len(bound),len(strict._HELPER_ROOTS))
        self.assertIn(str((ROOT/'src/hs2_mesh_deform.py').resolve()),bound)

    def test_three_previously_accepted_v1_aliases_rejected(self):
        base=ROOT.parent/'HS2Mod/artifacts/ocular_visibility_audit_20261006/asset_provider_review_v1'
        for name in ('float_actor_id','float_completed_count','integer_active_flag'):
            p=base/('final_tamper_'+name+'.json')
            with (self.subTest(name=name),
                  patch.object(strict.v1,'prepare_asset_provider',side_effect=AssertionError('expensive V1 preparation must not run')),
                  self.assertRaisesRegex(ValueError,'Exact')):
                strict.wrap_existing_v1({'path':str(p),'sha256':sha(p)},self.out/name)

    def test_recursive_source_binding_coherent_aliases(self):
        for name,change in (
            ('actor',lambda c:c['source_binding'].update(actor_id=float(c['source_binding']['actor_id']))),
            ('count',lambda c:c['source_binding']['source_cursor']['cursor'].update(completed_calls=float(c['source_binding']['source_cursor']['cursor']['completed_calls']))),
            ('active',lambda c:c['source_binding']['source_cursor']['cursor'].update(active=1)),
            ('head',lambda c:c['source_binding'].update(head_id=float(c['source_binding']['head_id']))),
            ('schema',lambda c:c.update(schema_version=2.0)),
            ('descriptor_count',lambda c:c['v1_descriptor'].update(mesh_count=8.0))):
            with self.subTest(name=name):
                p,d=self.modified(name,change)
                with self.assertRaisesRegex(ValueError,'Exact'):strict.prepare_strict_asset_provider(p,d)

    def test_coherent_missing_or_renamed_helper_rejected(self):
        for rel in ('tools/native_radial_target/compiler.py','tools/abmx_multibone/geometry.py','tools/abmx_replay/model.py','tools/abmx_stable_lowering/compiler.py'):
            key=str((ROOT/rel).resolve())
            p,d=self.modified('drop_'+Path(rel).stem,lambda c,k=key:c['source_files'].pop(k))
            with self.assertRaisesRegex(ValueError,'keys differ'):strict.prepare_strict_asset_provider(p,d)
        def rename(c):
            key=next(iter(c['source_files']));c['source_files'][key+'.renamed']=c['source_files'].pop(key)
        p,d=self.modified('renamed_helper',rename)
        with self.assertRaisesRegex(ValueError,'keys differ'):strict.prepare_strict_asset_provider(p,d)

    def test_actual_snapshot_type_aliases_and_adjacent_ids_refused(self):
        for field in ('actor','head','frame','completed','active','renderer','skin_bone'):
            s=copy.deepcopy(self.source)
            if field=='actor':s['character']['transform_id']=float(s['character']['transform_id'])
            elif field=='head':s['character']['head_id']=True
            elif field=='frame':s['frame_count']=float(s['frame_count'])
            elif field=='completed':s['abmx_trace_cursor']['completed_calls']=float(s['abmx_trace_cursor']['completed_calls'])
            elif field=='active':s['abmx_trace_cursor']['active']=1
            elif field=='renderer':s['meshes'][0]['renderer_id']=float(s['meshes'][0]['renderer_id'])
            else:s['meshes'][0]['bone_transform_ids'][0]=float(s['meshes'][0]['bone_transform_ids'][0])
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Exact'):self.provider.verify_snapshot_assets(s)

    def test_observer_check_no_unity_decode_and_nonserializable(self):
        with patch.object(strict.v1.UnityPy,'load',side_effect=AssertionError('no decode')):
            self.assertTrue(self.provider.verify_snapshot_assets(self.source)['passed'])
        with self.assertRaises(TypeError):pickle.dumps(self.provider)

    def test_source_only_target_uses_all8_unchanged_gate(self):
        artifact=read(ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261006/native_radial_drivers_head3_v1/compiled_execution.json')
        target=self.provider.independent_target(artifact,read(artifact['compiler_inputs']['contract_path']),source_gaze_policy='frozen_source_locals')
        checks=target['evidence']['source_mesh_checks']
        self.assertEqual(len(checks),8)
        self.assertTrue(all(c['native_source_model_passed'] for c in checks.values()))
        self.assertLess(max(c['source_error']['max_normalized'] for c in checks.values()),1e-5)
        self.assertFalse(target['evidence']['strict_asset_provider']['candidate_after_target_inputs'])

    def test_manifest_v1_fallback_or_typed_descriptor_rejected(self):
        for descriptor in ({'path':str(self.v1),'sha256':sha(self.v1)},
                           {**self.descriptor,'mesh_count':8.0}):
            p=self.out/'wrong_manifest.json';p.write_text(json.dumps({'asset_provider':descriptor}),encoding='utf-8')
            with self.assertRaises(ValueError):self.provider.runtime_audit(p,self.original['abmx_contract']['path'],self.out/'never_written')
        self.assertFalse((self.out/'never_written').exists())


if __name__=='__main__':unittest.main()
