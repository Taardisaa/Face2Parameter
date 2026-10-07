"""Freeze a history-only logical->executed artifact; never accesses the game."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from tools.abmx_replay.validate_trace import read,sha
from tools.abmx_stable_lowering.compiler import compile_history, verify_execution_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest',type=Path)
    source.add_argument('--history',type=Path,help='Frozen JSON with trace, trace_sha256, geometry.path/sha256')
    parser.add_argument('--manifest-sha256')
    parser.add_argument('--declared',type=Path,help='Six-key exact compiler declaration JSON')
    parser.add_argument('--declared-sha256')
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise ValueError('New immutable output path required')
    if args.manifest:
        if not args.manifest_sha256 or sha(args.manifest)!=args.manifest_sha256:
            raise ValueError('Explicit immutable manifest SHA required')
        from tools.abmx_multibone.run import restored_flags
        manifest=read(args.manifest)
        restored_flags(manifest)
        for a,b in (('before','after'),('expression_before','expression_after'),('modifiers_before','modifiers_after')):
            if manifest[a]!=manifest[b]: raise ValueError('Restoration records differ')
        if sha(manifest['protocol_path'])!=manifest['protocol_sha256']: raise ValueError('Protocol SHA changed')
        protocol=read(manifest['protocol_path'])
        cases=[c for c in manifest['cases'] if c['regime']=='combined']
        groups=[g for g in protocol['regimes'] if g['name']=='combined']
        if len(cases)!=1 or len(groups)!=1: raise ValueError('Unique combined case required')
        case=cases[0]
        history=case['source_history']
        if protocol['sampling_profile']!='slider_unlocker_18_2' or protocol['normalized_whole_head_gate']!=1e-5:
            raise ValueError('Unsupported profile/gate')
        if any(case[k]!=protocol[k] for k in ('baseline_expected_native59','candidate_expected_native59','source_history_expected_native59')):
            raise ValueError('Case native declarations disagree')
        declared={'head_id':case['head_id'],'native59':protocol['native59'],
            'source_history_native59':protocol['source_history_expected_native59'],'logical_patches':groups[0]['patches'],
            'sampling_profile':protocol['sampling_profile'],'source_files':{str(args.manifest.resolve()):args.manifest_sha256,
                manifest['protocol_path']:manifest['protocol_sha256'],case['trace']:case['trace_sha256']}}
    else:
        if not args.declared or not args.declared_sha256 or sha(args.declared)!=args.declared_sha256:
            raise ValueError('Frozen declaration with explicit SHA required')
        history,declared=read(args.history),read(args.declared)
        declared['source_files']={**declared['source_files'],str(args.declared.resolve()):args.declared_sha256,
                                  str(args.history.resolve()):sha(args.history)}
    result=compile_history(history,declared,args.contract)
    # For same-native identity sources this is a useful offline guard exercise;
    # it is explicitly not a current readback from a later game experiment.
    if declared['native59']==declared['source_history_native59']:
        result['earlier_identity_snapshot_guard']=verify_execution_guard(result,read(history['geometry']['path']),args.contract,
                current_trace_metadata=read(history['trace'])['metadata'])
        result['earlier_identity_snapshot_guard']['role']='earlier recorded snapshot only; new execution needs fresh guard'
    result['freeze_cli_sha256']=sha(__file__)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'artifact':str(args.out.resolve()),'sha256':sha(args.out),'compiler_binding_sha256':result['compiler_binding_sha256'],
                      'executed_patches':result['executed_patches'],'actual_stability_certified':False},indent=2))


if __name__=='__main__': main()
