"""New strict-metadata review contract for the installed bridge 0.31.14 (`efa943cb...`).

The guard (tools/abmx_stable_lowering/strict_metadata_guard.py) binds a review contract
whose `producer_sources` pin the bridge source that emits the ABMX runtime-baseline
wrapper, the trace cursor and the geometry frame binding. The v1 contract pinned
MakerGeometryService.cs at an uncommitted intermediate revision. This rebuilds the
same contract schema for the committed source, refusing unless:

  * every reviewed producer file in the working tree equals its blob at --revision
    (bound to a real git revision, not to uncommitted bytes);
  * the reviewed types decompiled from the installed DLL equal those decompiled from
    an offline build of that revision (--revision-build), so the installed binary runs
    the reviewed code;
  * each reviewed section still contains the reviewed statements;
  * the new live lowered_v2 record satisfies every invariant of the (unchanged)
    proposed strict revision: cursor exact types, serial complete prefix, session and
    frame binding, the four wrappers' reflected type/MVID/frame/fields, and the
    complete ordered prefix of the stopped trace.
The proposed strict revision itself, the consumer source and the guard logic are not
changed. Reading git history is read-only; nothing is committed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from tools.abmx_replay.validate_trace import read, sha

HS2MOD = Path(__file__).resolve().parents[3]/'HS2Mod'
BRIDGE = HS2MOD/'plugins/HS2_McpBridge'
INSTALLED = Path('E:/HoneySelect2_ArcticFox/BepInEx/Plugins/HS2_McpBridge/HS2_McpBridge.dll')
INSTALLED_SHA = 'efa943cb78099fcfccc11a10d4947693216e58e9525006b74ab2ee20d4ac8c21'
TYPES = ('MakerAbmxService', 'MakerAbmxTrace', 'MakerGeometryService')
# Reviewed sections of the committed MakerGeometryService.cs (edc12ea / 4a4a488 blob).
GEOMETRY_SECTIONS = [
    {'start': 43, 'end': 43, 'reason': 'Initial geometry frame captured.', 'must_contain': ['var frame = Time.frameCount;']},
    {'start': 70, 'end': 71, 'reason': 'Actual runtime-baseline read (GetBones(includeRuntimeBaseline: true)).',
     'must_contain': ['MakerAbmxService.HasController()', 'GetBones(includeRuntimeBaseline: true)']},
    {'start': 81, 'end': 88, 'reason': 'frame_count/frame_count_end, abmx_runtime and cursor export in the same sample.',
     'must_contain': ['["frame_count"] = frame,', '["frame_count_end"] = Time.frameCount,', '["abmx_runtime"] = abmxRuntime,',
                      '["abmx_trace_cursor"] = MakerAbmxTrace.Cursor(),']},
]


def git_blob(revision, path):
    rel = Path(path).resolve().relative_to(HS2MOD.resolve()).as_posix()
    return subprocess.run(['git', 'show', f'{revision}:{rel}'], cwd=HS2MOD, capture_output=True, check=True).stdout


def decompile(dll, type_name, out):
    text = subprocess.run(['ilspycmd', '-t', 'HS2Mod.McpBridge.'+type_name, str(dll)], capture_output=True, check=True).stdout
    out.write_bytes(text)
    return hashlib.sha256(text).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-contract', type=Path, required=True)
    ap.add_argument('--revision', required=True)
    ap.add_argument('--revision-build', type=Path, required=True, help='HS2_McpBridge.dll built offline from --revision')
    ap.add_argument('--manifest', type=Path, required=True, help='new live lowered_v2 live_cases.json')
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    require(not args.out.exists(), 'New immutable contract path required')
    require(sha(INSTALLED) == INSTALLED_SHA, 'Installed bridge is not efa943cb')
    old = read(args.old_contract)
    commit = subprocess.run(['git', 'rev-parse', args.revision], cwd=HS2MOD, capture_output=True, check=True, text=True).stdout.strip()
    args.work.mkdir(parents=True, exist_ok=True)
    producers = []
    for row in old['producer_sources']:
        path = Path(row['path'])
        blob = git_blob(commit, path)
        committed = hashlib.sha256(blob).hexdigest()
        require(sha(path) == committed, f'Working tree differs from {commit[:7]} for {path.name}: review would bind uncommitted bytes')
        lines = blob.decode('utf-8').splitlines()
        if path.name == 'MakerGeometryService.cs':
            sections = []
            for s in GEOMETRY_SECTIONS:
                text = '\n'.join(lines[s['start']-1:s['end']])
                require(all(t in text for t in s['must_contain']), f'Reviewed statements not at {path.name}:{s["start"]}-{s["end"]}')
                sections.append({k: s[k] for k in ('start', 'end', 'reason')})
        else:
            require(committed == row['sha256'], f'{path.name} changed since the v1 review; sections need re-review')
            sections = copy.deepcopy(row['sections'])
        producers.append({'path': row['path'], 'sha256': committed, 'git_revision': commit,
                          'git_path': path.resolve().relative_to(HS2MOD.resolve()).as_posix(), 'sections': sections})
    installed_vs_build = {}
    for t in TYPES:
        a = decompile(INSTALLED, t, args.work/f'installed_{t}.cs')
        b = decompile(args.revision_build, t, args.work/f'revision_{t}.cs')
        installed_vs_build[t] = {'installed_decompiled_sha256': a, 'revision_build_decompiled_sha256': b, 'equal': a == b}
        require(a == b, f'Installed {t} does not decompile to the {commit[:7]} build')
    consumer = old['consumer_source']
    require(sha(consumer['path']) == consumer['sha256'], 'Consumer validate_trace.py changed')
    # Actual new live record.
    manifest = read(args.manifest)
    case = manifest['cases'][0]
    geometry_row, trace_path = case['identity_capture']['paired_geometry'], case['trace']
    require(sha(geometry_row['path']) == geometry_row['sha256'] and sha(trace_path) == case['trace_sha256'], 'Live record changed')
    snapshot, trace = read(geometry_row['path']), read(trace_path)
    cursor = snapshot['abmx_trace_cursor']
    rev = old['proposed_new_strict_revision']
    for key in rev['cursor_types']['nonnegative_exact_int_not_bool']:
        require(type(cursor[key]) is int and cursor[key] >= 0, 'cursor '+key)
    require(type(cursor['active']) is bool and cursor['active'] is True and type(cursor['session_id']) is str and cursor['session_id'], 'cursor active/session')
    require(cursor['pending_calls'] == cursor['dropped_events'] == cursor['observer_error_count'] == 0, 'cursor loss')
    require(cursor['observed_calls'] == cursor['completed_calls'] == cursor['last_completed_sequence'], 'serial prefix')
    require(cursor['frame'] == snapshot['frame_count'] == snapshot['frame_count_end'], 'frame binding')
    require(case['trace_start']['session_id'] == cursor['session_id'], 'session binding')
    wrappers = []
    for name in ('cf_J_Chin_rs', 'cf_J_ChinTip_s', 'cf_J_CheekUp_L', 'cf_J_CheekUp_R'):
        rows = [r for r in snapshot['abmx_runtime']['bones'] if r['name'] == name]
        require(len(rows) == 1, 'wrapper '+name)
        w = rows[0]['runtime_baseline']
        require(w['modifier_type'] == rev['wrapper_modifier_type']['required_value'] and w['missing_fields'] == []
                and w['frame_count'] == cursor['frame'], 'wrapper fields '+name)
        wrappers.append({'name': name, 'modifier_type': w['modifier_type'], 'assembly_mvid': w['assembly_mvid'],
                         'frame_count': w['frame_count'], 'bone_transform_id': w['bone_transform_id'], 'missing_fields': w['missing_fields']})
    prefix = [e['sequence'] for e in trace['events'][:cursor['completed_calls']]]
    require(prefix == list(range(1, cursor['completed_calls'] + 1)), 'ordered complete prefix')
    contract = {**copy.deepcopy(old),
                'status': 'invariants_verified_committed_source_installed_build_equal',
                'actual_geometry_path': geometry_row['path'], 'actual_geometry_sha256': geometry_row['sha256'],
                'actual_manifest_sha256': sha(args.manifest), 'actual_candidate_trace_sha256': case['trace_sha256'],
                'actual_cursor': cursor, 'actual_four_supported_wrappers': wrappers,
                'actual_ordered_complete_prefix_sequences': prefix, 'producer_sources': producers,
                'installed_bridge': {'path': str(INSTALLED), 'sha256': INSTALLED_SHA, 'version': '0.31.14',
                                     'reviewed_types_decompile_equal_to_revision_build': installed_vs_build,
                                     'revision_build': {'path': str(args.revision_build.resolve()), 'sha256': sha(args.revision_build),
                                                        'command': 'dotnet build -c Release -p:Deterministic=true (sources exported from the revision)'}},
                'loaded_method_il_newly_decompiled_in_this_followup': True,
                'supersedes': {'path': str(args.old_contract.resolve()), 'sha256': sha(args.old_contract),
                               'reason': 'v1 pinned MakerGeometryService.cs 43e45941..., an uncommitted intermediate revision'},
                'game_actions': 0, 'production_source_edits': 0}
    contract['limits'] = list(old['limits']) + [
        'Bound to committed bridge sources at the recorded revision; the installed DLL is an IL-patched build '
        '(HttpServer.HandleRequest only, present_dispatch_source_v1/patch_audit.json) whose three reviewed types decompile '
        'identically to an offline build of that revision.']
    with args.out.open('x', encoding='utf-8') as handle:
        json.dump(contract, handle, indent=2, allow_nan=False)
    print(json.dumps({'contract': str(args.out), 'sha256': sha(args.out), 'revision': commit,
                      'installed_vs_build': {t: v['equal'] for t, v in installed_vs_build.items()}}, indent=1))


if __name__ == '__main__':
    main()
