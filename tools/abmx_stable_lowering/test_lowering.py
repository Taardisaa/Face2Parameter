"""Analytical target, derivative, FK and stale-input rejection checks."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.hs2_mesh_deform import _trs
from tools.abmx_replay.model import IDENTITY,FLAG_FIELDS,replay_apply
from tools.abmx_replay.validate_trace import sha
from tools.abmx_multibone.geometry import DEFAULT_BONES
from tools.abmx_stable_lowering.compiler import HistoryAnchor,lower_bone,radial_target_torch,check_files


def history(radius=3.):
    return {**{k:k=='_hasBaseline' for k in FLAG_FIELDS},'_lenBaseline':radius,'_positionBaseline':[0.,3.,0.],
            '_posBaseline':[0.,2.,0.],'_rotBaseline':[0.,0.,0.,1.],'_sclBaseline':[1.,1.,1.]}


def native(p=(0.,2.,0.)):
    return {'local_position':list(p),'local_rotation_xyzw':[0.,0.,0.,1.],'local_scale':[1.,1.,1.]}


def logical(**kw): return {**copy.deepcopy(IDENTITY),'length':2.,'position':[1.,0.,0.],**kw}


class LoweringTests(unittest.TestCase):
    def test_closed_form(self):
        row=lower_bone(native(),history(),logical())
        np.testing.assert_array_equal(row['desired_first_clean_target'],[1.,6.,0.])
        np.testing.assert_array_equal(row['execution_offset'],[1.,4.,0.])
        self.assertEqual(row['executed_modifier']['length'],1)
        self.assertFalse(row['actual_stability_certified'])

    def test_repeated_stable_branch(self):
        row=lower_bone(native(),history(),logical(scale=[2.,1.,1.],rotation=[0.,0.,90.]))
        result=row['executed_first_prediction']
        for _ in range(511):
            result=replay_apply(before=result['after'],cache=result['cache_after'],coordinate_modifiers=[row['executed_modifier']],
                coordinate=0,additional_modifiers=[],bone_exists=True,rotation_excluded=False,is_during_h_scene=False)
            self.assertEqual(result['after'],row['executed_first_prediction']['after'])
            self.assertIn('position_only_cached_baseline_plus_offset',result['branches'])
            self.assertNotIn('length_use_current_direction',result['branches'])

    def test_parent_child_target_fk(self):
        a=lower_bone(native(),history(),logical(scale=[2.,2.,2.],rotation=[0.,0.,90.]))
        b=lower_bone(native((2.,0.,0.)),history(1.),logical(position=[0.,1.,0.]))
        mats=[]
        for row in (a,b):
            p=row['executed_first_prediction']['after']
            mats.append(_trs(p['local_position'],p['local_rotation_xyzw'],p['local_scale']))
        world=mats[0]@mats[1]
        np.testing.assert_allclose(world[:3,3],[-1.,10.,0.],atol=2e-6,rtol=0)
        np.testing.assert_allclose((world@np.array([1.,0.,0.,1.]))[:3],[-1.,12.,0.],atol=2e-6,rtol=0)

    def test_analytical_derivatives(self):
        p=torch.tensor([.3,-.4,.8],dtype=torch.float64,requires_grad=True)
        r=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
        length=torch.tensor(1.02,dtype=torch.float64,requires_grad=True)
        d=torch.tensor([.02,.01,-.03],dtype=torch.float64,requires_grad=True)
        jp,jr,jl,jd=torch.autograd.functional.jacobian(radial_target_torch,(p,r,length,d))
        pv=p.detach().numpy(); n=np.linalg.norm(pv)
        np.testing.assert_allclose(jp.detach(),.7*1.02*(np.eye(3)/n-np.outer(pv,pv)/n**3),atol=2e-15,rtol=0)
        np.testing.assert_allclose(jr.detach(),1.02*pv/n,atol=2e-15,rtol=0)
        np.testing.assert_allclose(jl.detach(),.7*pv/n,atol=2e-15,rtol=0)
        np.testing.assert_array_equal(jd.detach(),np.eye(3))
        for v in (jp,jr,jl,jd): self.assertTrue(torch.isfinite(v).all())

    def test_float32_order_and_offset_roundtrip(self):
        row=lower_bone(native((.013,-.087,.115)),history(.120832346),logical(length=.99,position=[.002,.003,0.]))
        t=radial_target_torch(torch.tensor([.013,-.087,.115],dtype=torch.float32),torch.tensor(.120832346),
            torch.tensor(.99),torch.tensor([.002,.003,0.]))
        np.testing.assert_array_equal(t.detach().numpy(),np.asarray(row['desired_first_clean_target'],dtype=np.float32))
        self.assertLessEqual(row['position_component_max_error'],row['position_rounding_budget'])

    def test_unsupported_special_branches(self):
        for h,n,m in ((history(),native((0,0,0)),logical()),(history(),native(),logical(length=.05)),
                      (history(),native(),logical(length=1.)),(history(0),native(),logical()),
                      ({**history(),'_positionBaseline':[0,0,0]},native(),logical()),
                      ({**history(),'_lenModForceUpdate':True},native(),logical())):
            with self.assertRaises(ValueError): lower_bone(n,h,m)

    def test_exact_scale_rotation_preserved(self):
        m=logical(scale=[.99,1.01,1.02],rotation=[.6,-.4,.3])
        row=lower_bone(native(),history(),m)
        self.assertEqual(row['executed_modifier']['scale'],row['logical_modifier']['scale'])
        self.assertEqual(row['executed_modifier']['rotation'],row['logical_modifier']['rotation'])
        self.assertEqual(row['logical_first_prediction']['after']['local_rotation_xyzw'],row['executed_first_prediction']['after']['local_rotation_xyzw'])
        self.assertEqual(row['logical_first_prediction']['after']['local_scale'],row['executed_first_prediction']['after']['local_scale'])

    def test_stale_file_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'declared.json';p.write_text('{}')
            files={str(p):sha(p)}
            check_files(files)
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError): check_files(files)

    def test_context_native_and_mutation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source.json';p.write_text('{}')
            anchor=HistoryAnchor({'source_files':{str(p):sha(p)},'head_id':2},
                {n:history() for n in DEFAULT_BONES},[.5]*59,{n:native() for n in DEFAULT_BONES},
                [{'name':n,**logical()} for n in DEFAULT_BONES])
            for kwargs in ({'expected_binding':'bad','expected_head_id':2,'expected_native59':[.5]*59},
                           {'expected_binding':anchor.binding,'expected_head_id':3,'expected_native59':[.5]*59},
                           {'expected_binding':anchor.binding,'expected_head_id':2,'expected_native59':[.51]*59}):
                with self.assertRaises(ValueError): anchor.compile(**kwargs)
            anchor.history[DEFAULT_BONES[0]]['_lenBaseline']=4.
            with self.assertRaises(ValueError): anchor.compile(expected_binding=anchor.binding,expected_head_id=2,expected_native59=[.5]*59)

    def test_torch_zero_refused_before_nan(self):
        with self.assertRaises(ValueError): radial_target_torch(torch.zeros(3),torch.tensor(1.),torch.tensor(1.1),torch.zeros(3))


class ActualSourceAnchorTests(unittest.TestCase):
    """Real earlier immutable identity evidence, no candidate target inputs."""
    @classmethod
    def setUpClass(cls):
        from tools.abmx_replay.validate_trace import read
        from tools.abmx_stable_lowering.compiler import compile_history,ROOT
        cls.contract=ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json'
        cls.manifest=ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261005/abmx_stability_v1/live_cases.json'
        expected='0c1dda10567e0ef9e4b3c19a5bf48b27ccedf63d4123ae68e98b17846ebfb154'
        if sha(cls.manifest)!=expected: raise ValueError('Authorized immutable actual manifest changed')
        manifest=read(cls.manifest);protocol=read(manifest['protocol_path'])
        case=next(c for c in manifest['cases'] if c['regime']=='combined')
        cls.history=case['source_history']
        cls.declared={'head_id':2,'native59':protocol['native59'],'source_history_native59':protocol['source_history_expected_native59'],
            'logical_patches':next(g for g in protocol['regimes'] if g['name']=='combined')['patches'],
            'sampling_profile':protocol['sampling_profile'],'source_files':{str(cls.manifest):expected,manifest['protocol_path']:manifest['protocol_sha256']}}
        cls.artifact=compile_history(cls.history,cls.declared,cls.contract)
        cls.snapshot=read(cls.history['geometry']['path'])
        cls.metadata=read(cls.history['trace'])['metadata']

    def test_actual_history_guard(self):
        from tools.abmx_stable_lowering.compiler import verify_execution_guard
        result=verify_execution_guard(self.artifact,self.snapshot,self.contract,current_trace_metadata=self.metadata)
        self.assertTrue(result['passed'])
        self.assertFalse(result['old_instance_ids_reused_as_proof'])
        for row in self.artifact['bone_diagnostics'].values():
            self.assertTrue(row['first_equivalence']['passed'])
        json.dumps(self.artifact,allow_nan=False)

    def test_coherent_numerical_artifact_tamper_refused(self):
        from tools.abmx_stable_lowering.compiler import verify_compiled_artifact
        value=copy.deepcopy(self.artifact);name=DEFAULT_BONES[0]
        # Coherently alter executed patch and its mirrored diagnostic/target.
        value['executed_patches'][0]['position'][0]+=.01
        row=value['bone_diagnostics'][name]
        row['executed_modifier']['position'][0]+=.01
        row['execution_offset'][0]+=.01
        row['desired_first_clean_target'][0]+=.01
        with self.assertRaises(ValueError): verify_compiled_artifact(value)

    def test_frozen_logical_declaration_tamper_refused(self):
        from tools.abmx_stable_lowering.compiler import verify_compiled_artifact
        value=copy.deepcopy(self.artifact)
        value['compiler_inputs']['declared']['logical_patches'][0]['length']=1.3
        with self.assertRaises(ValueError): verify_compiled_artifact(value)

    def test_bridge_actor_and_numeric_history_stale_refused(self):
        from tools.abmx_stable_lowering.compiler import verify_execution_guard
        for label in ('bridge','actor','history'):
            snap=copy.deepcopy(self.snapshot);meta=copy.deepcopy(self.metadata)
            if label=='bridge':meta['bridge_mvid']='wrong'
            if label=='actor':snap['character']['transform_id']+=1;meta['character_transform_id']+=1
            if label=='history':
                next(r for r in snap['abmx_runtime']['bones'] if r['name']==DEFAULT_BONES[0])['runtime_baseline']['fields']['_lenBaseline']+=.001
            with self.assertRaises(ValueError):verify_execution_guard(self.artifact,snap,self.contract,current_trace_metadata=meta)

    def test_native59_derivative_direction(self):
        from src.hs2_mesh_deform import HeadRig
        from src.hs2_deform_torch import TorchHeadRig
        rig=TorchHeadRig(HeadRig(2,sampling_profile='slider_unlocker_18_2'),device='cpu',dtype=torch.float64)
        x=torch.tensor([self.declared['native59']],dtype=torch.float64,requires_grad=True)
        i=rig.name2bone[DEFAULT_BONES[0]]
        radius=torch.tensor(self.artifact['bone_diagnostics'][DEFAULT_BONES[0]]['persistent_history']['_lenBaseline'],dtype=torch.float64)
        length=torch.tensor(1.01,dtype=torch.float64);off=torch.tensor([.002,-.003,.002],dtype=torch.float64)
        weight=torch.tensor([.4,-.3,.2],dtype=torch.float64)
        def f(v):return (radial_target_torch(rig.local_transforms(v)[0][0,i],radius,length,off)*weight).sum()
        grad=torch.autograd.grad(f(x),x)[0]
        direction=torch.tensor(np.random.default_rng(7361).normal(size=(1,59)),dtype=torch.float64)
        direction/=direction.norm();step=1e-5
        fd=(f(x.detach()+step*direction)-f(x.detach()-step*direction))/(2*step)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertAlmostEqual(float((grad*direction).sum()),float(fd.detach()),places=8)


if __name__=='__main__': unittest.main()
