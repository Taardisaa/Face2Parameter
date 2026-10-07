"""New metadata revision tests; immutable v1 tests and evidence stay untouched."""
from __future__ import annotations
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from tools.abmx_replay.validate_trace import read,sha
from tools.abmx_stable_lowering import strict_metadata_guard as guard
from tools.abmx_stable_lowering.compiler import verify_execution_guard
from tools.abmx_multibone.geometry import DEFAULT_BONES

ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT.parent/'HS2Mod/artifacts/infrastructure_live_20261005/abmx_lowered_v2/live_cases.json'
MANIFEST_SHA='ab4fc82a6f1dffe665936913f3970c599a13d3286f7012b399adeff121c37aed'


def actual_inputs():
    if sha(MANIFEST)!=MANIFEST_SHA: raise ValueError('Actual V2 manifest changed')
    manifest=read(MANIFEST)
    case=manifest['cases'][0]
    artifact_row=manifest['compiler_artifact']
    geometry=case['identity_capture']['paired_geometry']
    for path,digest in ((artifact_row['path'],artifact_row['sha256']),(geometry['path'],geometry['sha256']),(case['trace'],case['trace_sha256'])):
        if sha(path)!=digest: raise ValueError('Actual V2 source changed')
    return manifest,read(artifact_row['path']),read(geometry['path']),case['trace_start'],read(case['trace'])


class StrictMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest,cls.artifact,cls.snapshot,cls.metadata,cls.trace=actual_inputs()
        cls.contract=ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json'

    def check(self,snapshot=None,metadata=None,**kw):
        return guard.verify_metadata(snapshot or self.snapshot,metadata or self.metadata,self.artifact,**kw)

    def test_actual_full_new_guard_posthoc(self):
        result=guard.verify_execution_guard_v2(self.artifact,self.snapshot,self.contract,
            current_trace_metadata=self.metadata,completed_trace=self.trace)
        self.assertTrue(result['passed'])
        self.assertTrue(result['strict_metadata']['completed_recorded_prefix_independently_verified'])
        self.assertEqual(result['strict_metadata']['trace_binding']['actual_selected_prefix_events'],12)
        self.assertFalse(result['old_live_guard_retroactively_claimed_v2'])

    def test_wrong_actual_wrapper_type_refused(self):
        snap=copy.deepcopy(self.snapshot)
        next(r for r in snap['abmx_runtime']['bones'] if r['name']==DEFAULT_BONES[0])['runtime_baseline']['modifier_type']='Wrong.Type'
        # Preserve and demonstrate the old accepted gap explicitly.
        self.assertTrue(verify_execution_guard(self.artifact,snap,self.contract,current_trace_metadata=self.metadata)['passed'])
        with self.assertRaises(ValueError):
            guard.verify_execution_guard_v2(self.artifact,snap,self.contract,current_trace_metadata=self.metadata)

    def test_inconsistent_completed_count_refused(self):
        snap=copy.deepcopy(self.snapshot);snap['abmx_trace_cursor']['completed_calls']=11
        self.assertTrue(verify_execution_guard(self.artifact,snap,self.contract,current_trace_metadata=self.metadata)['passed'])
        with self.assertRaises(ValueError):
            guard.verify_execution_guard_v2(self.artifact,snap,self.contract,current_trace_metadata=self.metadata)

    def test_bool_and_noninteger_counters_refused(self):
        for value in (True,12.,'12',None,-1):
            for key in ('frame','observed_calls','completed_calls','last_completed_sequence','pending_calls','dropped_events','observer_error_count'):
                snap=copy.deepcopy(self.snapshot);snap['abmx_trace_cursor'][key]=value
                with self.subTest(value=value,key=key),self.assertRaises(ValueError):self.check(snap)

    def test_missing_counter_and_inactive_refused(self):
        for key in ('observed_calls','completed_calls','last_completed_sequence'):
            snap=copy.deepcopy(self.snapshot);del snap['abmx_trace_cursor'][key]
            with self.assertRaises(ValueError):self.check(snap)
        for value in (False,1,'true',None):
            snap=copy.deepcopy(self.snapshot);snap['abmx_trace_cursor']['active']=value
            with self.assertRaises(ValueError):self.check(snap)

    def test_zero_prefix_explicitly_supported(self):
        snap=copy.deepcopy(self.snapshot)
        for key in ('observed_calls','completed_calls','last_completed_sequence'):snap['abmx_trace_cursor'][key]=0
        result=self.check(snap)
        self.assertTrue(result['passed'])
        self.assertFalse(result['completed_recorded_prefix_independently_verified'])

    def test_losses_pending_or_errors_refused(self):
        for key in ('pending_calls','dropped_events','observer_error_count'):
            snap=copy.deepcopy(self.snapshot);snap['abmx_trace_cursor'][key]=1
            with self.assertRaises(ValueError):self.check(snap)

    def test_frame_session_and_actor_refused(self):
        for key,value in (('frame',178795),('session_id','wrong'),('session_id','')):
            snap=copy.deepcopy(self.snapshot);snap['abmx_trace_cursor'][key]=value
            with self.assertRaises(ValueError):self.check(snap)
        meta=copy.deepcopy(self.metadata);meta['started_frame']=self.snapshot['frame_count']+1
        with self.assertRaises(ValueError):self.check(metadata=meta)
        snap=copy.deepcopy(self.snapshot);snap['character']['transform_id']+=1
        with self.assertRaises(ValueError):self.check(snap)

    def test_nonserial_prefix_and_future_event_refused(self):
        installed=read(self.contract)
        trace=copy.deepcopy(self.trace);trace['events'][0]['sequence']=2
        with self.assertRaises(ValueError):self.check(completed_trace=trace,installed_contract=installed)
        snap=copy.deepcopy(self.snapshot)
        for key in ('observed_calls','completed_calls','last_completed_sequence'):snap['abmx_trace_cursor'][key]=len(self.trace['events'])
        with self.assertRaises(ValueError):self.check(snap,completed_trace=self.trace,installed_contract=installed)

    def test_wrong_wrapper_frame_id_and_missing_fields_refused(self):
        for key,value in (('frame_count',True),('bone_transform_id',False),('missing_fields',['_lenBaseline']),('missing_fields',None)):
            snap=copy.deepcopy(self.snapshot)
            next(r for r in snap['abmx_runtime']['bones'] if r['name']==DEFAULT_BONES[0])['runtime_baseline'][key]=value
            with self.assertRaises(ValueError):self.check(snap)

    def test_stale_contract_refused(self):
        with patch.object(guard,'METADATA_CONTRACT_SHA256','bad'):
            with self.assertRaises(ValueError):self.check()

    def test_original_numerical_guard_runs_first(self):
        calls=[]
        def old(*a,**kw):calls.append('v1');return {'passed':True}
        def new(*a,**kw):calls.append('strict');raise ValueError('diagnostic strict refusal')
        with patch.object(guard,'verify_execution_guard',old),patch.object(guard,'verify_metadata',new):
            with self.assertRaises(ValueError):guard.verify_execution_guard_v2(self.artifact,self.snapshot,self.contract,current_trace_metadata=self.metadata)
        self.assertEqual(calls,['v1','strict'])


if __name__=='__main__':unittest.main()
