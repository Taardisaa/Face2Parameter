"""Executable real-SOURCE and negative asset contract tests; no game operations."""
from __future__ import annotations

import copy
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.abmx_replay.validate_trace import read, sha
from tools.abmx_stable_lowering.compiler import digest_json
from tools.native_radial_target import validate_runtime
from tools.native_radial_target.asset_provider import (
    _array_digest,
    prepare_asset_provider,
    stage_from_identity,
)

ROOT=Path(__file__).resolve().parents[2]


class AssetProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out=ROOT/'outputs/native_radial_target_20261006';out.mkdir(exist_ok=True,parents=True)
        cls.tmp=tempfile.TemporaryDirectory(prefix='asset_provider_tests_',dir=out)
        cls.dir=Path(cls.tmp.name)
        cls.artifact=read(ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261006/native_radial_drivers_head3_v1/compiled_execution.json')
        cls.history=cls.artifact['compiler_inputs']['history'];cls.abmx=cls.artifact['compiler_inputs']['contract_path']
        cls.source=read(cls.history['geometry']['path'])
        report=read(ROOT/'outputs/native_radial_target_20261006/source_asset_audit_v2/head3_source.json')
        mesh=next(m for m in report['meshes'] if m['mesh_name']=='o_tooth');asset=next(v for v in mesh['asset_candidates'] if v['source']=='explicit_zipmod');o=asset['origin']
        cls.overrides={mesh['renderer_path']:{'archive':o['path'],'archive_sha256':o['sha256'],'member':o['member'],
            'member_sha256':o['member_sha256'],'prefab':o['prefab'],'renderer_path_id':asset['renderer_path_id'],'mesh_path_id':asset['mesh_path_id']}}
        cls.old_hashes={str(p):sha(p) for p in (ROOT/'data/hs2_head/head_3').rglob('*') if p.is_file()}
        cls.original_math_cached_mesh=staticmethod(validate_runtime.cached_mesh)
        cls.descriptor=stage_from_identity(cls.history,cls.abmx,cls.overrides,cls.dir/'staged')
        cls.contract=read(cls.descriptor['path'])
        cls.provider=prepare_asset_provider(cls.descriptor['path'],cls.descriptor['sha256'])

    @classmethod
    def tearDownClass(cls):
        assert all(sha(p)==v for p,v in cls.old_hashes.items())
        assert validate_runtime.cached_mesh is cls.original_math_cached_mesh
        cls.tmp.cleanup()

    def modified_contract(self,name,mutate):
        c=copy.deepcopy(self.contract);mutate(c)
        c.pop('contract_binding_sha256');c['contract_binding_sha256']=digest_json(c)
        p=self.dir/(name+'.json');p.write_text(json.dumps(c,allow_nan=False),encoding='utf-8')
        return p,sha(p)

    def test_real_source_all8_and150_authored_frames(self):
        r=self.provider.verify_snapshot_assets(self.source)
        self.assertTrue(r['passed']);self.assertEqual(len(r['meshes']),8)
        self.assertEqual(sum(len(b['frames']) for m in self.contract['meshes'] for b in m['source_inventory']['blendshape_frames']),150)
        kinds=[m['origin']['kind'] for m in self.contract['meshes']]
        self.assertEqual(kinds.count('explicit_zipmod'),1);self.assertEqual(kinds.count('existing_same_head_cache'),7)
        tooth=next(m for m in self.contract['meshes'] if m['mesh_name']=='o_tooth')
        self.assertEqual(tooth['all_staged_arrays']['verts']['shape'],[496,3])
        self.assertTrue(tooth['authored_correspondence']['runtime_uv2_absent'])
        self.assertFalse(tooth['authored_correspondence']['uv2_certified'])

    def test_source_only_full8_fk_lbs_unchanged_math(self):
        target=self.provider.independent_target(self.artifact,read(self.abmx),source_gaze_policy='frozen_source_locals')
        checks=target['evidence']['source_mesh_checks']
        self.assertEqual(len(checks),8);self.assertTrue(all(c['native_source_model_passed'] for c in checks.values()))
        self.assertLess(max(c['source_error']['max_normalized'] for c in checks.values()),1e-5)
        self.assertIs(validate_runtime.cached_mesh,self.original_math_cached_mesh)

    def test_wrong_contract_hash_head_profile_bridge_cursor_refused(self):
        with self.assertRaises(ValueError):prepare_asset_provider(self.descriptor['path'],'0'*64)
        for name,change in [('head',lambda c:c.update(head_id=2)),('profile',lambda c:c.update(sampling_profile='wrong')),
                            ('bridge',lambda c:c.update(source_bridge_mvid='wrong')),
                            ('cursor',lambda c:c['source_cursor'].update(cursor={}))]:
            p,d=self.modified_contract(name,change)
            with self.assertRaises(ValueError):prepare_asset_provider(p,d)

    def test_removed_mandatory_dependency_not_self_digest_token(self):
        p,d=self.modified_contract('missing_dependency',lambda c:c['source_files'].pop(str(ROOT/'tools/native_radial_target/asset_provider.py')))
        with self.assertRaisesRegex(ValueError,'Mandatory'):prepare_asset_provider(p,d)

    def test_wrong_member_hash_and_prefab_pathids_refused(self):
        def change(c,key,val):
            path=next(iter(c['explicit_overrides']));c['explicit_overrides'][path][key]=val
            next(m for m in c['meshes'] if m['mesh_name']=='o_tooth')['origin'][key]=val
        for key,val in [('member_sha256','0'*64),('prefab','p_cf_head_02'),('mesh_path_id',123)]:
            p,d=self.modified_contract(key,lambda c,k=key,v=val:change(c,k,v))
            with self.assertRaises(ValueError):prepare_asset_provider(p,d)

    def test_wrong_renderer_palette_topology_actual_snapshot(self):
        for field in ('renderer','palette','topology'):
            s=copy.deepcopy(self.source);m=next(m for m in s['meshes'] if m['mesh_name']=='o_tooth')
            if field=='renderer':m['renderer_id']+=1
            elif field=='palette':m['bone_names'][0]='cf_J_Head'
            else:m['source']['triangles'][0]+=1
            with self.assertRaises(ValueError):self.provider.verify_snapshot_assets(s)

    def test_active_or_authored_shape_change_refused(self):
        for field in ('weight','vertex','normal','name'):
            s=copy.deepcopy(self.source);m=next(m for m in s['meshes'] if m['mesh_name']=='o_tooth');b=m['blendshapes'][0]
            if field=='weight':b['current_weight']+=1
            elif field=='name':b['name']='wrong'
            elif field=='vertex':b['frames'][0]['delta_vertices'][0][0]=.001
            else:b['frames'][0]['delta_normals'][0][0]=.001
            with self.assertRaises(ValueError):self.provider.verify_snapshot_assets(s)

    def test_optional_uv2_absence_not_synthetic_zero_match(self):
        s=copy.deepcopy(self.source);m=next(m for m in s['meshes'] if m['mesh_name']=='o_tooth')
        m['source']['uv2']=[[0.,0.] for _ in m['source']['vertices']]
        with self.assertRaises(ValueError):self.provider.verify_snapshot_assets(s)

    def test_candidate_source_cannot_select_assets(self):
        candidate=ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261006/native_radial_drivers_head3_v1/early.json'
        h=copy.deepcopy(self.history);h['geometry']={'path':str(candidate),'sha256':sha(candidate)}
        with self.assertRaisesRegex(ValueError,'Candidate'):stage_from_identity(h,self.abmx,self.overrides,self.dir/'candidate_source_forbidden')
        self.assertFalse((self.dir/'candidate_source_forbidden').exists())

    def test_source_archive_hash_mismatch_refused(self):
        h=copy.deepcopy(self.history);h['geometry']['sha256']='0'*64
        with self.assertRaises(ValueError):stage_from_identity(h,self.abmx,self.overrides,self.dir/'wrong_source_hash')
        o=copy.deepcopy(self.overrides);next(iter(o.values()))['archive_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Archive SHA'):stage_from_identity(self.history,self.abmx,o,self.dir/'wrong_archive_hash')

    def test_no_fallback_and_no_unobserved_renderer(self):
        with self.assertRaises(ValueError):stage_from_identity(self.history,self.abmx,{},self.dir/'no_override')
        o={'/unobserved':next(iter(self.overrides.values()))}
        with self.assertRaises(ValueError):stage_from_identity(self.history,self.abmx,o,self.dir/'unobserved')

    def test_coherent_staged_array_tamper_cannot_be_redigested(self):
        e=next(m for m in self.contract['meshes'] if m['mesh_name']=='o_tooth');p=Path(e['stage_npz']);old=p.read_bytes()
        try:
            with np.load(p,allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
            data['faces'][0]=data['faces'][0,::-1];np.savez(p,**data)
            def mutate(c):
                row=next(m for m in c['meshes'] if m['mesh_name']=='o_tooth')
                row['stage_npz_sha256']=sha(p);c['stage_files'][str(p)]=sha(p)
                row['all_staged_arrays']['faces']=_array_digest(data['faces'],np.int64)
            cp,d=self.modified_contract('coherent_topology_tamper',mutate)
            with self.assertRaisesRegex(ValueError,'Staged arrays'):prepare_asset_provider(cp,d)
        finally:p.write_bytes(old)

    def test_signed_zero_canonical_but_nonzero_not_rounded(self):
        self.assertEqual(_array_digest([-0.]),_array_digest([0.]))
        self.assertNotEqual(_array_digest([1e-12]),_array_digest([0.]))
        with self.assertRaises(ValueError):_array_digest([1.5],np.int64)

    def test_contract_nonserializable_and_manifest_descriptor_bound(self):
        with self.assertRaises(TypeError):pickle.dumps(self.provider)
        p=self.dir/'wrong_manifest.json';p.write_text(json.dumps({'asset_provider':{'path':self.descriptor['path'],'sha256':'0'*64}}),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'descriptor'):self.provider.runtime_audit(p,self.abmx,self.dir/'wrong_runtime_report')

    def test_guard_does_not_decode_unity_assets_during_observer(self):
        with patch('tools.native_radial_target.asset_provider.UnityPy.load',side_effect=RuntimeError('must not decode during observer')):
            self.assertTrue(self.provider.verify_snapshot_assets(self.source)['passed'])


if __name__=='__main__':unittest.main()
