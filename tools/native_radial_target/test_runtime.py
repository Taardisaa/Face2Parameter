"""Additional verifier checks; synthetic transitions are not live certification."""
import copy
import unittest
import numpy as np
import torch
from tools.abmx_replay.validate_trace import read
from tools.abmx_replay.model import IDENTITY
from tools.native_radial_target.adapter import NAMES,MODE,NativeRadialTargetRig,native_source_bindings
from tools.native_radial_target.validate_runtime import vector,verify_physical_trace,cached_mesh,independent_target
from tools.native_radial_target.test_native_target import ROOT,fixture,mixed,nonidentity_all30
from pathlib import Path
import tempfile
from src.hs2_mesh_deform import HeadRig


class RuntimeVerifierTests(unittest.TestCase):
    def test_source_gaze_scope_and_frozen_source_nuisance(self):
        artifact=read(ROOT/'outputs/native_radial_target_20261005/source_smoke_frozen_v1/compiled.json')
        contract=read(artifact['compiler_inputs']['contract_path'])
        strict=independent_target(artifact,contract)
        unsupported=[r['mesh_name'] for r in strict['evidence']['source_mesh_checks'].values() if not r['target_scope_supported']]
        self.assertEqual(set(unsupported),{'o_eyebase_L','o_eyebase_R'})
        target=independent_target(artifact,contract,source_gaze_policy='frozen_source_locals')
        self.assertEqual(len(target['vertices']),8)
        self.assertTrue(all(r['target_scope_supported'] for r in target['evidence']['source_mesh_checks'].values()))
        self.assertFalse(target['evidence']['candidate_actual_after_or_expression_used_for_target'])
        with self.assertRaises(ValueError):independent_target(artifact,contract,source_gaze_policy='infer_candidate_gaze')

    def test_float32_cpu_and_available_cuda_gradients(self):
        rig=HeadRig(2,sampling_profile='slider_unlocker_18_2')
        # Explicit indexed device matches the adapter's strict device contract.
        devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        reference=None
        for device in devices:
            model=NativeRadialTargetRig(rig,semantic_mode=MODE,source_bindings=native_source_bindings(rig),device=device,dtype=torch.float32)
            native=torch.tensor([mixed()],device=device,dtype=torch.float32,requires_grad=True)
            logical=model.logical_tensor(nonidentity_all30()).clone().requires_grad_()
            vertices=model(native,logical)
            gradients=torch.autograd.grad(vertices.square().mean(),(native,logical))
            self.assertTrue(torch.isfinite(vertices).all() and all(torch.isfinite(g).all() for g in gradients))
            if reference is None:reference=vertices.detach().cpu().numpy()
            else:np.testing.assert_allclose(vertices.detach().cpu().numpy(),reference,atol=2e-6,rtol=1e-6)

    def test_missing_trace_bone_filter_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/native_radial_target_20261005') as d:
            history,_,contract,_,_,_=fixture(Path(d));trace=read(history['trace'])
            trace['metadata']['requested_names']=list(NAMES[:-1])
            with self.assertRaises(ValueError):verify_physical_trace(trace,{},trace,read(contract))

    def test_trace_counter_and_future_sequence_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs/native_radial_target_20261005') as d:
            history,_,contract,_,_,_=fixture(Path(d));source=read(history['trace'])
            for key in ('observed_calls','dropped_events','pending_calls'):
                trace=copy.deepcopy(source);trace[key]+=1
                with self.assertRaises(ValueError):verify_physical_trace(trace,{},source,read(contract))
            trace=copy.deepcopy(source);trace['events'][-1]['sequence']+=1
            with self.assertRaises(ValueError):verify_physical_trace(trace,{},source,read(contract))

    def test_visible_submesh_cache_refuses_changed_asset(self):
        manifest=read(ROOT/'../HS2Mod/artifacts/infrastructure_live_20261005/native_radial_source_v1/live_cases.json')
        snapshot=read(manifest['source_history']['geometry']['path']);rig=HeadRig(2,sampling_profile='slider_unlocker_18_2')
        mesh=copy.deepcopy(next(m for m in snapshot['meshes'] if m['mesh_name']=='o_eyebase_L'))
        self.assertGreater(len(cached_mesh(rig,mesh).verts),0)
        mesh['source']['vertices'][0][0]+=.001
        with self.assertRaises(ValueError):cached_mesh(rig,mesh)
        mesh=copy.deepcopy(next(m for m in snapshot['meshes'] if m['mesh_name']=='o_eyebase_L'))
        mesh['bone_names'][0]='unverified'
        with self.assertRaises(ValueError):cached_mesh(rig,mesh)

    def test_all30_order_and_float32_physical_equality(self):
        self.assertEqual(len(NAMES),30)
        m=copy.deepcopy(IDENTITY);m['position']=[.001,-.002,.003]
        v=vector(m);self.assertEqual(v.shape,(10,));self.assertEqual(v.dtype,np.float32)
        self.assertEqual(float(v[3]),1.)


if __name__=='__main__':unittest.main()
