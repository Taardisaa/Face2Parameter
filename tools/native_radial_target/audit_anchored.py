"""Optional observed transform-only boundary proof, with independent failures.

Never changes frozen compiler, strict report, or source-only target definition.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from tools.abmx_replay.validate_trace import read,sha,validate,trs_errors,local_only
from tools.abmx_replay.model import IDENTITY,serialized,require
from tools.native_radial_target.adapter import NAMES,MODE
from tools.native_radial_target.compiler import prepare_execution_guard
from tools.native_radial_target.validate_runtime import independent_target,vector,save,verify_physical_trace
from tools.native_radial_target.observed_boundary import POLICY,prove_anchor,prove_observed_gap,diagnostic_cursor


def audit_anchored(manifest_path,contract_path,strict_report_path,out):
    out=Path(out);require(not out.exists(),'New policy output required');out.mkdir(parents=True)
    m=read(manifest_path);contract=read(contract_path);strict=read(strict_report_path)
    result={'policy':POLICY,'semantic_mode':MODE,'manifest_sha256':sha(manifest_path),
        'implementation_sha256':sha(__file__),'proof_source_sha256':sha(Path(__file__).with_name('observed_boundary.py')),
        'strict_report_path':str(Path(strict_report_path).resolve()),'strict_report_sha256':sha(strict_report_path),
        'strict_acceptance_replaced':False,'conditional_snapshot_and_surface_passed':False,
        'writer_cause_or_future_absence_certified':False,'likeness_or_quality_or_character_readiness_certified':False}
    try:
        require(strict['manifest_sha256']==sha(manifest_path) and strict['passed'] is False,'Expected exact preserved strict-false report')
        require(len(m['cases'])==1,'Exactly one actual case');case=m['cases'][0]
        artifact=m['compiler_artifact'];require(sha(artifact['path'])==artifact['sha256'],'Frozen artifact changed')
        a=read(artifact['path']);context=prepare_execution_guard(a,contract_path,artifact_path=artifact['path'],artifact_sha256=artifact['sha256'])
        identity=case['identity_capture']['paired_geometry'];require(sha(identity['path'])==identity['sha256'],'Actual identity changed')
        context.verify_snapshot(read(identity['path']),case['trace_start'])
        target=independent_target(a,contract,source_gaze_policy='frozen_source_locals')
        require(sha(case['trace'])==case['trace_sha256'],'Actual candidate trace changed')
        trace=read(case['trace']);source=read(case['source_history']['trace'])
        replay,starts,totals=verify_physical_trace(trace,a,source,contract)
        expected={r['name']:{k:r[k] for k in IDENTITY} for r in a['executed_patches']}
        active_checks=[]
        for e in trace['events']:
            name=e['bone_name']
            require(name in NAMES,'Unrecognized observed bone')
            if e['sequence']<starts[name]:continue
            physical=expected[name];proof=prove_anchor(e['before']['cache']['fields'],physical)
            fields=e['before']['cache']['fields']
            baseline={k:fields[f] for k,f in (('local_position','_posBaseline'),('local_rotation_xyzw','_rotBaseline'),('local_scale','_sclBaseline'))}
            cache_native=trs_errors(a['native_baselines'][name],baseline)
            after_error=trs_errors(proof['predicted_after_from_cache_not_incoming'],local_only(e['after']))
            source_target=trs_errors(target['targets'][name],proof['predicted_after_from_cache_not_incoming'])
            require(cache_native['passed'] and after_error['passed'] and source_target['passed'],'Anchored source-native/actual-after disagreement')
            active_checks.append({'sequence':e['sequence'],'name':name,'native_baseline_error':cache_native,
                'source_only_target_error':source_target,'actual_after_error':after_error,'branch_proof':proof})
        gaps=[]
        for b in replay['inter_call_boundaries']:
            current=trace['events'][b['sequence']-1];previous=trace['events'][b['previous_sequence']-1]
            require(previous['sequence']>=starts[current['bone_name']],'External gap touches unsupported identity/setup boundary')
            gaps.append(prove_observed_gap(previous,current,expected[current['bone_name']]))
        save(out/'active_after_targets.json',active_checks);save(out/'transform_only_gaps.json',gaps)
        windows=[]
        for w in case['windows']:
            require(sha(w['geometry']['path'])==w['geometry']['sha256'],'Actual window changed')
            snapshot=read(w['geometry']['path'])
            require(snapshot['character']['head_id']==a['source_bindings']['head_id'] and
                np.array_equal(np.asarray(snapshot['character']['shape_value_face'],dtype=np.float32),np.asarray(a['source_bindings']['native59'],dtype=np.float32)),
                'Actual window native/head differs')
            predicted,binding=diagnostic_cursor(snapshot,trace,replay,NAMES)
            direct={}
            for name in NAMES:
                transform=next(t for t in snapshot['transforms'] if t['name']==name)
                direct[name]=trs_errors(target['targets'][name],local_only(transform))
            windows.append({'window':w['window'],'snapshot_cursor_diagnostic':binding,
                'source_only_target_vs_actual_locals':direct,'passed':binding['passed'] and all(v['passed'] for v in direct.values())})
        save(out/'snapshot_targets.json',windows)
        result.update(actual_calls=len(trace['events']),active_calls_proven=len(active_checks),active_counts_per_bone=totals,
            observed_transform_only_gaps_proven=len(gaps),conditional_observed_apply_target_passed=True,
            snapshot_target_passed_windows=sum(w['passed'] for w in windows),snapshot_window_count=len(windows),
            conditional_snapshot_and_surface_passed=all(w['passed'] for w in windows) and
                all(w['passed'] for w in strict.get('windows',[])) and bool(strict.get('windows')),
            interpretation='Installed Apply reanchors all three local channels from unchanged cache; later snapshot overwrites remain independent failures')
    except (ValueError,KeyError,TypeError,OSError,StopIteration) as error:result['rejection']=str(error)
    save(out/'summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True);p.add_argument('--strict-report',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();r=audit_anchored(a.manifest,a.contract,a.strict_report,a.out_dir)
    print({k:v for k,v in r.items() if k not in ('implementation_sha256','proof_source_sha256','manifest_sha256')})
    return 0 if r['conditional_snapshot_and_surface_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
