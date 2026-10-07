"""Read-only POSTHOC strict revision audit of the immutable lowered V2 identity."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import unittest
from tools.abmx_stable_lowering.strict_metadata_guard import verify_execution_guard_v2,REVISION
from tools.abmx_stable_lowering.test_strict_metadata_guard import actual_inputs,ROOT,MANIFEST,MANIFEST_SHA
from tools.abmx_replay.validate_trace import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('New immutable output directory required')
    args.out.mkdir(parents=True)
    manifest,artifact,snapshot,metadata,trace=actual_inputs()
    contract=ROOT/'outputs/abmx_replay_20261005/installed_contract_v2.json'
    result=verify_execution_guard_v2(artifact,snapshot,contract,current_trace_metadata=metadata,completed_trace=trace)
    stream=io.StringIO()
    test_result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tools.abmx_stable_lowering.test_strict_metadata_guard'))
    files=[Path(__file__),Path(__file__).with_name('strict_metadata_guard.py'),Path(__file__).with_name('test_strict_metadata_guard.py'),
           Path(__file__).with_name('compiler.py'),Path(__file__).with_name('test_lowering.py')]
    report={'revision':REVISION,'scope':'posthoc_saved_actual_v2_identity_snapshot_only_no_new_live_calls',
        'manifest_path':str(MANIFEST),'manifest_sha256':MANIFEST_SHA,'artifact':manifest['compiler_artifact'],
        'identity_geometry':manifest['cases'][0]['identity_capture']['paired_geometry'],
        'candidate_trace':{'path':manifest['cases'][0]['trace'],'sha256':manifest['cases'][0]['trace_sha256']},
        'posthoc_strict_guard':result,'tests':{'passed':test_result.wasSuccessful(),'run':test_result.testsRun,
        'failures':[(str(t),e) for t,e in test_result.failures],'errors':[(str(t),e) for t,e in test_result.errors]},
        'implementation_sha256':{str(p.resolve()):sha(p) for p in files},
        'frozen_v1_unchanged':sha(Path(__file__).with_name('compiler.py'))=='0f721b16d3812e4e04a5e29bd57156e896e44876ec9b75adf7337061084cb8d8',
        'original_v2_live_call_claimed_new_revision':False,'new_runtime_stability_or_character_readiness_claimed':False,
        'game_actions':0,'input_manifest_unchanged':sha(MANIFEST)==MANIFEST_SHA}
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (args.out/'tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    print(json.dumps({'report':str((args.out/'report.json').resolve()),'passed':result['passed'],'tests':report['tests'],
        'prefix':result['strict_metadata']['trace_binding'],'frozen_v1_unchanged':report['frozen_v1_unchanged']},indent=2))
    if not test_result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
