"""All30 cached-math and synthetic source-guard checks, never game evidence."""
from __future__ import annotations
import copy
import json
import pickle
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
import torch
from src.hs2_mesh_deform import HeadRig,_fk_world,_trs
from tools.abmx_multibone.geometry import parent_first_world,skin_vertices
from tools.abmx_replay.model import FLAG_FIELDS,IDENTITY,replay_apply
from tools.abmx_replay.validate_trace import read,sha
from tools.native_radial_target.adapter import (MODE,PROFILE,NAMES,NativeRadialTargetRig,
    native_source_bindings,identity_patch_map,native_numpy,target_numpy,assert_coverage)
from tools.native_radial_target.compiler import compile_native_target,prepare_execution_guard,_native,_clean

ROOT=Path(__file__).resolve().parents[2]


def mixed():return [[.48,.52,.47,.53][i%4] for i in range(59)]


def nonidentity_all30():
    m=identity_patch_map()
    for i,n in enumerate(NAMES):m[n]={'scale':[1.002,1.001,.999],'length':1.+(.002 if i%2 else -.002),
        'position':[.0001*(i%3-1),.0002,-.0001],'rotation':[.2,-.1,.15]}
    return m


def fixture(directory):
    """Fully constructed all30 identity source verifies logic only, not runtime."""
    rig=HeadRig(2,sampling_profile=PROFILE);contract_path=ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json'
    contract=read(contract_path);values=mixed();locals=_native(rig,values)
    metadata={'schema_version':1,'session_id':'synthetic-not-game','started_frame':10,'stopped_frame':10,
        'character_transform_id':123,'requested_names':list(NAMES),'bridge_mvid':'synthetic-bridge',
        'plugin_mvid':contract['plugin_mvid'],'plugin_version':contract['plugin_version'],
        'apply_method_il_sha256':contract['apply_method_il_sha256'],'pre_existing_patch_owners':[],
        'observation_policy':'void Prefix last / void Postfix first; no argument, baseline, transform or return writes'}
    events=[];bones=[];transforms=[]
    for round_index in range(2):
        for i,name in enumerate(NAMES):
            native={k:np.asarray(v,dtype=np.float32).tolist() for k,v in locals[name].items()}
            fields={**{k:k=='_hasBaseline' for k in FLAG_FIELDS},'_lenBaseline':float(np.linalg.norm(native['local_position'])),
                '_positionBaseline':native['local_position'],'_posBaseline':native['local_position'],
                '_sclBaseline':native['local_scale'],'_rotBaseline':native['local_rotation_xyzw']}
            wrapper={'frame_count':10,'modifier_type':'KKABMX.Core.BoneModifier','assembly_mvid':contract['plugin_mvid'],
                'bone_transform_id':1000+i,'missing_fields':[],'fields':fields}
            state={'bone_transform_id':1000+i,**native,'cache':wrapper}
            events.append({'sequence':len(events)+1,'frame':10,'completed_frame':10,'bone_name':name,
                'modifier_instance_identity':2000+i,'coordinate':0,'coordinate_specific':False,'additional_modifiers':[],
                'is_during_h_scene':False,'no_rotation_excluded':False,'resolved_modifier':copy.deepcopy(IDENTITY),
                'before':copy.deepcopy(state),'after':copy.deepcopy(state)})
            if round_index==0:
                bones.append({'name':name,'exists':True,**copy.deepcopy(IDENTITY),'runtime_baseline':copy.deepcopy(wrapper)})
                transforms.append({'name':name,'id':1000+i,'path':'/synthetic/'+name,**native})
    trace={'metadata':metadata,'events':events,'observed_calls':60,'active':False,'trace_complete':True,
        'dropped_events':0,'pending_calls':0,'observer_errors':[]}
    mesh={'mesh_name':'o_head','bone_names':rig.skin_bone_names,'source':{'vertices':rig.verts.tolist(),'triangles':rig.faces.reshape(-1).tolist(),
        'bone_indices':rig.bone_idx.tolist(),'bone_weights':rig.bone_w.tolist(),'bindposes':rig.bindpose.reshape(-1).tolist()}}
    snapshot={'frame_count':10,'frame_count_end':10,'game':{'game_assembly_sha256':'synthetic-game'},
        'character':{'transform_id':123,'head_id':2,'shape_value_face':values},'transforms':transforms,'meshes':[mesh],
        'abmx_runtime':{'bones':bones},'abmx_trace_cursor':{'session_id':metadata['session_id'],'frame':10,'active':True,
            'observed_calls':60,'completed_calls':60,'last_completed_sequence':60,'pending_calls':0,'dropped_events':0,'observer_error_count':0}}
    protocol={'head_id':2,'semantic_mode':MODE,'sampling_profile':PROFILE,'native59':values,'source_history_native59':values,
              'logical_patches':nonidentity_all30()}
    def save(name,value):
        p=directory/name;p.write_text(json.dumps(value,allow_nan=False),encoding='utf-8');return str(p),sha(p)
    trace_path,trace_sha=save('synthetic_identity_trace.json',trace)
    geo_path,geo_sha=save('synthetic_identity_geometry.json',snapshot)
    protocol_path,protocol_sha=save('synthetic_protocol.json',protocol)
    history={'trace':trace_path,'trace_sha256':trace_sha,'geometry':{'path':geo_path,'sha256':geo_sha}}
    declared={**protocol,'source_files':{protocol_path:protocol_sha}}
    return history,declared,contract_path,snapshot,metadata,rig


class NativeTargetMathTests(unittest.TestCase):
    def model(self,h=2):
        rig=HeadRig(h,sampling_profile=PROFILE)
        return rig,NativeRadialTargetRig(rig,semantic_mode=MODE,source_bindings=native_source_bindings(rig))

    def test_all4bases_all30_numpy_torch_fk_lbs(self):
        logical=nonidentity_all30();x=torch.tensor([mixed()],dtype=torch.float64)
        for head in range(4):
            rig,model=self.model(head);a=model.logical_tensor(logical)
            p,q,s=model.local_transforms(x,a);ref=target_numpy(native_numpy(rig,mixed()),logical,semantic_mode=MODE)
            for name in NAMES:
                i=model.name2bone[name]
                np.testing.assert_allclose(p[0,i].numpy(),ref[name]['local_position'],atol=1e-12,rtol=0)
                rq=np.asarray(ref[name]['local_rotation_xyzw']);tq=q[0,i].numpy()
                self.assertLess(min(np.abs(rq-tq).max(),np.abs(rq+tq).max()),1e-12)
                np.testing.assert_allclose(s[0,i].numpy(),ref[name]['local_scale'],atol=1e-12,rtol=0)
            world=_fk_world(rig,mixed(),None)
            local={pid:np.linalg.solve(world[rig.bones[pid]['parent']],world[pid]) if rig.bones[pid]['parent'] in world else world[pid] for pid in rig._topo}
            numpy_world=parent_first_world(rig,local,ref)
            np.testing.assert_allclose(model(x,a)[0].numpy(),skin_vertices(rig,numpy_world,np.zeros_like(rig.verts)),atol=2e-12,rtol=0)

    def test_complete_submesh_connection(self):
        rig,model=self.model();x=torch.tensor([mixed()],dtype=torch.float64);a=model.logical_tensor(nonidentity_all30());world=model.bone_world(x,a)
        paths=list((Path(rig.data_dir)/'submeshes').glob('*.npz'));self.assertGreater(len(paths),1)
        for path in paths:
            sub=model.load_submesh(path.stem);v=model.skin(sub,world)
            self.assertEqual(tuple(v.shape),(1,len(sub['verts_h']),3));self.assertTrue(torch.isfinite(v).all())

    def test_gradients_all30_and_native_direction(self):
        _,model=self.model();x=torch.tensor([mixed()],dtype=torch.float64,requires_grad=True)
        a=model.logical_tensor(nonidentity_all30()).clone().requires_grad_();rng=np.random.default_rng(3951)
        weights=torch.tensor(rng.normal(size=(1,model.n_bones,3)),dtype=torch.float64)
        def loss(native,logical):
            p,q,s=model.local_transforms(native,logical)
            return (p*weights).sum()+q.square().sum()*.03+s.square().sum()*.02
        gx,ga=torch.autograd.grad(loss(x,a),(x,a));self.assertTrue(torch.isfinite(gx).all() and torch.isfinite(ga).all())
        self.assertTrue((ga[0].abs().sum(1)>0).all())
        dx=torch.tensor(rng.normal(size=x.shape),dtype=torch.float64);dx/=dx.norm()
        da=torch.tensor(rng.normal(size=a.shape),dtype=torch.float64);da/=da.norm();eps=1e-6
        fd=(loss(x.detach()+eps*dx,a.detach()+eps*da)-loss(x.detach()-eps*dx,a.detach()-eps*da))/(2*eps)
        self.assertAlmostEqual(float((gx*dx).sum()+(ga*da).sum()),float(fd.detach()),places=7)

    def test_missing_extra_nonpositive_mode_and_default_refused(self):
        rig,model=self.model();patches=identity_patch_map()
        for kind in ('missing','extra','negative','zero'):
            p=copy.deepcopy(patches)
            if kind=='missing':del p[NAMES[0]]
            if kind=='extra':p['unexpected']=copy.deepcopy(IDENTITY)
            if kind=='negative':p[NAMES[0]]['length']=-1
            if kind=='zero':p[NAMES[0]]['scale'][0]=0
            with self.assertRaises(ValueError):model.logical_tensor(p)
        with self.assertRaises(ValueError):model(torch.tensor([mixed()],dtype=torch.float64),None)
        with self.assertRaises(ValueError):NativeRadialTargetRig(rig,semantic_mode='installed_abmx_stateful',source_bindings=native_source_bindings(rig))
        with self.assertRaises(ValueError):model.from_card('anything')
        broken=copy.deepcopy(rig);del broken.bones[broken.name2pid[NAMES[0]]]
        with self.assertRaises(ValueError):assert_coverage(broken)

    def test_zero_history_inert_pending_flag_repeated_installed_apply(self):
        native={'local_position':[0,0,0],'local_scale':[1,1,1],'local_rotation_xyzw':[0,0,0,1]}
        cache={**{k:k=='_hasBaseline' for k in FLAG_FIELDS},'_lenBaseline':0,'_positionBaseline':[0,0,0],
            '_posBaseline':[0,0,0],'_sclBaseline':[1,1,1],'_rotBaseline':[0,0,0,1],
            '_lenModForceUpdate':True,'_lenModNeedsPositionRestore':True}
        _clean(cache);m={**IDENTITY,'position':[.01,.02,-.03]}
        for _ in range(32):
            result=replay_apply(before=native,cache=cache,coordinate_modifiers=[m],coordinate=0,additional_modifiers=[],
                bone_exists=True,rotation_excluded=False,is_during_h_scene=False)
            native=result['after'];cache=result['cache_after']
            np.testing.assert_array_equal(np.asarray(native['local_position'],dtype=np.float32),np.asarray(m['position'],dtype=np.float32))
            self.assertNotIn('length_use_current_direction',result['branches'])
            self.assertTrue(cache['_lenModForceUpdate'])
        bad={**cache,'_positionBaseline':[0,1,0],'_forceApply':False,'_changedPosition':False}
        with self.assertRaises(ValueError):_clean(bad)


class SyntheticCompilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent=ROOT/'outputs/native_radial_target_20261005/synthetic_fixtures';parent.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=parent);cls.directory=Path(cls.tmp.name)
        cls.history,cls.declared,cls.contract,cls.snapshot,cls.metadata,cls.rig=fixture(cls.directory)
        cls.artifact=compile_native_target(cls.history,cls.declared,cls.contract)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_complete_compile_and_prepared_guard(self):
        self.assertEqual(self.artifact['all30_order'],list(NAMES));self.assertEqual(len(self.artifact['executed_patches']),30)
        self.assertTrue(all(r['length']==1 for r in self.artifact['executed_patches']))
        self.assertFalse(self.artifact['all30_runtime_certified']);json.dumps(self.artifact,allow_nan=False)
        ctx=prepare_execution_guard(self.artifact,self.contract)
        started=time.perf_counter();report=ctx.verify_snapshot(self.snapshot,self.metadata)
        self.assertTrue(report['passed']);self.assertFalse(report['inference_or_trace_replay_during_guard'])
        self.guard_seconds=time.perf_counter()-started
        with self.assertRaises(TypeError):pickle.dumps(ctx)

    def test_coherent_target_physical_tamper_refused(self):
        a=copy.deepcopy(self.artifact);a['executed_patches'][0]['position'][0]+=.01
        a['bone_diagnostics'][NAMES[0]]['executed_modifier']['position'][0]+=.01
        with self.assertRaises(ValueError):prepare_execution_guard(a,self.contract)

    def test_snapshot_type_counts_native_flags_extra_and_missing_refused(self):
        ctx=prepare_execution_guard(self.artifact,self.contract)
        for change in ('type','count','native','pending','extra','missing'):
            s=copy.deepcopy(self.snapshot)
            if change=='type':s['abmx_runtime']['bones'][0]['runtime_baseline']['modifier_type']='Wrong.Type'
            if change=='count':s['abmx_trace_cursor']['completed_calls']=59
            if change=='native':s['character']['shape_value_face'][0]+=.01
            if change=='pending':s['abmx_runtime']['bones'][0]['runtime_baseline']['fields']['_forceApply']=True
            if change=='extra':s['abmx_runtime']['bones'].append({'name':'foreign',**IDENTITY,'position':[.01,0,0]})
            if change=='missing':s['abmx_runtime']['bones'].pop()
            with self.subTest(change=change),self.assertRaises(ValueError):ctx.verify_snapshot(s,self.metadata)

    def test_caller_artifact_mutated_after_prepare_refused(self):
        a=copy.deepcopy(self.artifact);ctx=prepare_execution_guard(a,self.contract)
        a['executed_patches'][0]['position'][0]+=.001
        with self.assertRaises(ValueError):ctx.verify_snapshot(self.snapshot,self.metadata)

    def test_stale_protocol_bytes_refused(self):
        a=copy.deepcopy(self.artifact);ctx=prepare_execution_guard(a,self.contract)
        path=Path(next(iter(self.declared['source_files'])));old=path.read_bytes()
        try:
            path.write_bytes(old+b' ')
            with self.assertRaises(ValueError):ctx.verify_snapshot(self.snapshot,self.metadata)
        finally:path.write_bytes(old)

    def test_four_bone_history_not_all30_refused(self):
        t=read(self.history['trace']);t['metadata']['requested_names']=list(NAMES[:4])
        path=self.directory/'wrong_filter_trace.json';path.write_text(json.dumps(t),encoding='utf-8')
        history={**self.history,'trace':str(path),'trace_sha256':sha(path)}
        with self.assertRaises(ValueError):compile_native_target(history,self.declared,self.contract)


if __name__=='__main__':unittest.main()
