"""Convert one probe-v7 bridge window into a candidate-renderer template (offline).

The same chain the v6.1 / v7 acceptance windows went through
(``docs/hs2_offline_frame_orchestrator.md`` §12, §13.1), recorded here as commands:

1. ``renderdoc_replay_msvc convert`` -> ``<bulk>/parse_<nonce>/capture.zip.xml`` (pinned RenderDoc 1.46)
2. ``verify_probe_v5_session.py`` -> must print ``accepted``
3. ``pose_clip_bindings.py`` -> ``<bulk>/pose_clip_bindings_<nonce>.json`` (Animator write set)
4. ``renderdoc_replay_msvc inspect`` -> driver action listing
5. ``frame_graph`` (unsegmented) -> the bridge capture target: the one 1086x1448 Texture2D written
   as render target by all 16 bridge renders and finally read as a copy source (ReadPixels)
6. ``frame_graph --segment-boundary-resource <target>``
7. ``capture_to_program --through-event <Present - 1>``
8. ``resource_acceptance --wiring-out --raw-dir``
9. ``frame_orchestrate_paired`` (template; generated + fresh oracle executed, checkpoints compared)

Every step logs stdout/stderr and its exit code into ``<out>/steps.json``; a step whose output
exists is not repeated (outputs are never overwritten).

    python tools/chin_fit/convert_window.py --bulk <session bulk dir> --nonce <nonce> \
        --card-state <state json> --out <fresh window dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / '.venv/Scripts/python.exe')
OUT = ROOT / 'outputs'
TOOL = OUT / 'offline_renderer_repair_20261013/official_action_provenance_msvc_v1/renderdoc_replay_msvc.exe'
RDDIR = OUT / 'offline_renderer_repair_20261012/renderdoc_portable_1_46/extracted/RenderDoc_1.46_64'
MANIFESTS = [OUT / 'offline_renderer_repair_20261012/shader_programs_v2/manifest.json',
             OUT / 'offline_renderer_repair_20261012/body_shader_programs_v1/manifest.json',
             OUT / 'offline_renderer_repair_20261012/sss_shader_programs_v2/manifest.json'] + [
    OUT / 'offline_renderer_frame_graph_20261020/shader_exports' / n / 'manifest.json'
    for n in ('unity_builtin_extra', 'sharedassets2', 'sharedassets3', 'resources_assets', 'hs2_betteraa_ctaa')]
LIBRARY = OUT / 'offline_renderer_shader_library_20261020/library_v1'
EXECUTOR = OUT / 'offline_renderer_executor_20261020/build_v5/frame_executor.exe'
PPV2_STATE = OUT / 'offline_renderer_ppv2_20261020/scene_state_v2/characustom_ppv2.json'
RES = Path('E:/HS2_offline_renderer_data/resources_20261020')
INDEX = RES / 'source_index_v5.json'
SIDELOADER = RES / 'sideloader_scan_v1.json'


def bridge_target(resources_json):
    fg = json.loads(Path(resources_json).read_text(encoding='utf-8'))
    hits = []
    for k, r in fg.items():
        d = r.get('descriptor') or {}
        if r.get('kind') == 'CreateTexture2D' and d.get('Width') == 1086 and d.get('Height') == 1448:
            tl = r.get('timeline') or []
            writes = [t for t in tl if t[1] == 'write' and t[2] == 'render_target']
            if len(writes) == 16 and tl and list(tl[-1][1:]) == ['read', 'copy_source']:
                hits.append(int(k))
    if len(hits) != 1:
        raise RuntimeError('bridge capture target not unique: %r' % hits)
    return hits[0]


def present_event(inspect_stdout):
    pres = []
    with open(inspect_stdout, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('{'):
                x = json.loads(line)
                if x.get('event') == 'action' and x['name'].startswith('Present('):
                    pres.append(x['event_id'])
    if len(pres) != 1:
        raise RuntimeError('expected one Present action, got %r' % pres)
    return pres[0]


class Steps:
    def __init__(self, out):
        self.out = out
        self.file = out / 'steps.json'
        self.rows = json.loads(self.file.read_text(encoding='utf-8')) if self.file.exists() else []

    def run(self, name, cmd, done, stdout_file=None, env=None):
        if done.exists():
            print('skip', name)
            return
        t0 = time.time()
        log = self.out / ('%s.log' % name)
        with open(stdout_file or log, 'wb') as so, open(self.out / ('%s.stderr' % name), 'wb') as se:
            p = subprocess.run([str(c) for c in cmd], stdout=so, stderr=se, env=env, cwd=str(ROOT))
        row = {'step': name, 'cmd': [str(c) for c in cmd], 'exit': p.returncode, 'seconds': round(time.time() - t0, 1)}
        self.rows.append(row)
        self.file.write_text(json.dumps(self.rows, indent=1), encoding='utf-8')
        print(name, 'exit', p.returncode, row['seconds'], 's', flush=True)
        if p.returncode != 0:
            raise RuntimeError('%s failed (exit %d); see %s' % (name, p.returncode, self.out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bulk', required=True, type=Path)
    ap.add_argument('--nonce', required=True)
    ap.add_argument('--card-state', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    s = Steps(a.out)
    n, B, O = a.nonce, a.bulk, a.out
    rdcs = sorted(B.glob('rdc_%s_frame*.rdc' % n))
    if len(rdcs) != 1:
        raise RuntimeError('one RDC expected for %s: %r' % (n, rdcs))
    rdc = rdcs[0]
    env = dict(os.environ, PATH=str(RDDIR) + os.pathsep + os.environ['PATH'])
    parse = B / ('parse_' + n)
    xml = parse / 'capture.zip.xml'
    if not xml.exists():
        parse.mkdir(exist_ok=True)
    s.run('convert', [TOOL, 'convert', rdc, 'rdc', xml, 'zip.xml'], xml, stdout_file=parse / 'convert_log.txt', env=env)
    verify = B / ('verify_' + n)
    s.run('verify', [PY, ROOT / 'tools/offline_renderer/frame_pair_probe/verify_probe_v5_session.py', '--xml', xml,
                     '--window', B / ('window_' + n), '--nonce', n, '--out', verify], verify / 'verification.json')
    if not json.loads((verify / 'verification.json').read_text(encoding='utf-8'))['accepted']:
        raise RuntimeError('probe window not accepted: ' + n)
    ver = json.loads((verify / 'verification.json').read_text(encoding='utf-8'))
    a.window_index = int(next(c['evidence'] for c in ver['checks'] if c['check'] == 'window_has_captured_frame'))
    pcb = B / ('pose_clip_bindings_%s.json' % n)
    s.run('pose_clip_bindings', [PY, ROOT / 'tools/offline_renderer/pose_clip_bindings.py', '--window',
                                 B / ('window_' + n), '--out', pcb], pcb)
    (O / 'inspect').mkdir(exist_ok=True)
    ins = O / 'inspect' / 'stdout.bin'
    s.run('inspect', [TOOL, 'inspect', rdc], ins, stdout_file=ins, env=env)
    man = [x for m in MANIFESTS for x in ('--shader-manifest', m)]
    fg0 = O / 'frame_graph_unsegmented'
    s.run('frame_graph_unsegmented', [PY, '-m', 'tools.offline_renderer.frame_graph', '--xml', xml, '--inspect', ins]
          + man + ['--out', fg0], fg0 / 'summary.json')
    target = bridge_target(fg0 / 'resources.json')
    fg = O / 'frame_graph'
    s.run('frame_graph', [PY, '-m', 'tools.offline_renderer.frame_graph', '--xml', xml, '--inspect', ins] + man
          + ['--segment-boundary-resource', target, '--out', fg], fg / 'summary.json')
    through = present_event(ins) - 1
    cp = O / 'capture_program'
    s.run('capture_program', [PY, '-m', 'tools.offline_renderer.capture_to_program', '--xml', xml, '--inspect', ins,
                              '--through-event', through, '--out', cp], cp / 'program.json')
    ra = O / 'resource_acceptance'
    s.run('resource_acceptance', [PY, 'tools/offline_renderer/resource_acceptance.py', '--xml', xml,
                                  '--zip', parse / 'capture.zip',
                                  '--scene', B / ('window_' + n) / 'frames' / ('scene_%03d.json' % a.window_index),
                                  '--index', INDEX, '--sideloader-scan', SIDELOADER, '--card-state', a.card_state,
                                  '--shader-library', LIBRARY / 'main', '--shader-library', LIBRARY / 'resources_standard',
                                  '--shader-library', LIBRARY / 'dhh_dhpost', '--wiring-out', O / 'resource_wiring',
                                  '--raw-dir', B / ('window_' + n), '--out', ra,
                                  '--receipt', O / 'resource_acceptance_receipt.json'],
          O / 'resource_acceptance_receipt.json')
    tpl = O / 'template_v1'
    s.run('template', [PY, '-m', 'tools.offline_renderer.frame_orchestrate_paired', '--xml', xml, '--inspect', ins,
                       '--window', B / ('window_' + n), '--window-index', a.window_index, '--library', LIBRARY,
                       '--capture-program', cp, '--executor', EXECUTOR] + man +
          ['--frame-graph', fg, '--acceptance-table', ra / 'resources_table.json',
           '--resource-wiring', O / 'resource_wiring', '--bridge-request', B / ('bridge_body_%s.json' % n),
           '--skin-geometry', B / 'prepare_geometry_actor.json', '--ppv2-scene-state', PPV2_STATE,
           '--out', tpl], tpl / 'report.json', env=env)
    rep = json.loads((tpl / 'report.json').read_text(encoding='utf-8'))
    summary = {'nonce': n, 'target_resource': target, 'through_event': through,
               'summary': rep.get('summary'), 'execution_summary': rep.get('execution_summary')}
    (O / 'conversion.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
