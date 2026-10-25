"""Plugin work of the retry bridge frame (frame 17675, 16 renders) generated in-frame.

The bridge renderstyle capture (``MakerRenderStyleService.Capture``) has no paired
scene state, so Unity's own scene work (camera renders of objects, PPv2, the
ordinary frame) is spliced from the converted capture and labelled
``external:capture``.  The plugin work is generated through
``plugin_common.PluginLowering`` and ``hs2pipe.frame.UnityForwardLowering``:

* DHH SSS: the SeparableSSS convolution blits and the convolution's final
  ``Blit(source, destination)`` of every render;
* GrabPass copies (``PluginLowering`` emits them; the forward interface supplies
  the active colour target);
* the image-effect chain: CharaCustom.CustomRender blits and every CTAA blit;
* all plugin render textures (``RenderTextureAllocator``: temporary pool LIFO,
  persistent CTAA / capture RTs).  Spliced capture commands that read or write a
  plugin texture are rewritten to the generated texture (per-render role map).

Validation: each generated blit vs the captured draw it replaces (shaders, states,
constant-buffer bytes within the declared ranges, viewport, target and texture
descriptors), and execution of the generated program and of the verbatim
capture program with identical read-backs (end of every capture pass).
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.frame_program import ProgramBuilder  # noqa: E402
from src.render.hs2pipe import frame as fr  # noqa: E402
from src.render.hs2pipe import shader_library as sl  # noqa: E402
from src.render.hs2pipe.external_capture import DRAW_OPS, CaptureProgram  # noqa: E402
from src.render.hs2pipe.plugin_common import PluginLowering, ShaderManifests, UnityRenderTexture  # noqa: E402
from tools.offline_renderer import validate_plugin_effects as vpe  # noqa: E402
from tools.offline_renderer.frame_structural_diff import LIB_ROOTS  # noqa: E402

ACTIONS = DRAW_OPS + ('dispatch',)


class NotifyDict(dict):
    def __init__(self, cb):
        super().__init__()
        self.cb = cb

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.cb(k, v)


class RetryDriver:
    """Capture cursor + the frame-specific answers of the forward interface."""

    def __init__(self, cap, fgcap, em, plan, lib):
        self.cap, self.fg, self.em, self.plan, self.lib = cap, fgcap, em, plan, lib
        self._sp = {}
        self.cursor = 0
        self.replaced = set()
        self.cap_to_gen = {}
        self.res_map, self.view_map = {}, {}
        self.views_of = collections.defaultdict(list)
        for v in cap.program['views']:
            self.views_of[v['resource']].append(v)
        self.lowering = None
        self.k = None
        self.queues = {}
        self.engine = {}
        self.role_log = []

    # ---------------------------------------------------------- cursor
    def idx(self, event):
        return self.cap.command_at_event(event)

    def pass_actions(self, p):
        out = []
        for e in p['events']:
            try:
                i = self.idx(e)
            except KeyError:
                continue
            if self.cap.commands[i]['op'] in ACTIONS or self.cap.commands[i]['op'] == 'copy_subresource_region':
                out.append(i)
        return out

    def splice_one(self, i, reason):
        g = self.em.splice(i, reason, remap=(self.res_map, self.view_map))
        self.cap_to_gen[i] = g

    def advance_to(self, ci):
        while self.cursor < ci:
            i = self.cursor
            if i not in self.replaced:
                self.splice_one(i, 'not generated (no paired scene state / Unity-internal)')
            self.cursor += 1

    def finish(self):
        self.advance_to(len(self.cap.commands))

    # ---------------------------------------------------------- segments
    def begin_segment(self, k):
        self.k = k
        ps = self.fg.seg_passes(k)
        labels = [p['label'] for p in ps]
        blur = [i for i, l in enumerate(labels) if l == 'sss_separable_blur']
        conv_final = blur[-1] + 1
        if labels[conv_final] != 'blit_copy':
            raise fr.FrameGap('segment %d: no convolution Blit after the blurs' % k)
        c0 = next(i for i, p in enumerate(ps) if p['label'] == 'blit_copy' and p['viewport']
                  and p['viewport']['Width'] == 3200.0)
        blits = [self.pass_actions(ps[i])[0] for i in blur + [conv_final]]
        blits += [a for p in ps[c0:] for a in self.pass_actions(p) if self.cap.commands[a]['op'] in DRAW_OPS]
        grabs = [a for p in ps if p['label'] == 'grabpass_copy' for a in self.pass_actions(p)]
        g0 = labels.index('grabpass_copy')
        g1 = labels.index('post_copy_std')
        fwd = [a for p in ps[g0:g1] if p['label'] == 'forward_scene' for a in self.pass_actions(p)]
        self.queues = {'blit': collections.deque(blits), 'grab': collections.deque(grabs),
                       'forward': collections.deque(fwd)}
        self.seg_passes, self.labels, self.c0, self.blur = ps, labels, c0, blur
        self.roles = vpe.segment_map(self.fg, k)
        rec = self.plan.renders[k]
        ctaa_ops = rec.image_effect_chain[-1]['ops']
        ctaa_blit = [o for o in ctaa_ops if o.kind == 'blit' and o.args['material'] == 'Hidden/CTAA_PC'][0]
        self.roles[ctaa_blit.args['dest']] = self.roles.pop('ctaa_write')
        accum = ctaa_blit.args['textures']['_Accum']
        r = self.roles.pop('ctaa_accum_read')
        self.roles.setdefault(accum, [{'resource': r}])
        # engine buffers of this render
        first_blur = self.cap.commands[blits[0]]['state']
        self.engine['conv_source'] = self.engine_rec(self._srv_res(first_blur, '_MainTex'), 'lighting camera ImageEffects source')
        self.engine['conv_destination'] = self.engine_rec(self._rt_res(self.cap.commands[blits[len(blur)]]['state']),
                                                          'lighting camera ImageEffects destination')
        chain0 = self.cap.commands[self.pass_actions(ps[c0])[0]]['state']
        self.engine['chain_source'] = self.engine_rec(self._srv_res(chain0, '_MainTex'), 'main camera ImageEffects source')
        self.engine['rtCamera'] = self.engine_rec(self._rt_res(chain0), 'CustomRender.rtCamera', depth=True)
        chain1 = self.cap.commands[self.pass_actions(ps[c0 + 1])[0]]['state']
        self.engine['chain_destination'] = self.engine_rec(self._rt_res(chain1), 'ImageEffects Temp (unity_forward)')
        ctaa_pc = [p for p in ps if p['label'] == 'temporal_aa_ctaa'][1]
        st = self.cap.commands[self.pass_actions(ctaa_pc)[0]]['state']
        self.engine['mv'] = self.engine_rec(self._srv_res(st, '_CameraMotionVectorsTexture'), '_CameraMotionVectorsTexture')
        for h, rec_ in list((self.lowering.handles if self.lowering else {}).items()):
            self.on_handle(h, rec_)

    def _srv_res(self, st, name):
        """Capture resource bound to texture ``name`` of the draw's pixel program."""
        sha = self.cap.shader_sha(st['ps']['shader'])
        if sha not in self._sp:
            m = self.lib.match(sha)[0]
            sh = self.lib.copies(m['shader'])[0]
            sp = [x for sub in sh.subshaders for pz in sub.passes for x in pz.subprograms('ps') if x.dxbc_sha256 == sha]
            if not sp:
                for c in self.lib.copies(m['shader']):
                    sp = [x for sub in c.subshaders for pz in sub.passes for x in pz.subprograms('ps')
                          if x.dxbc_sha256 == sha]
                    if sp:
                        break
            self._sp[sha] = sp[0]
        slot = [t['srv'] for t in self._sp[sha].resources()['textures'] if t['name'] == name][0]
        return [self.cap.obj(x['view'])['resource'] for x in st['ps']['srvs'] if x['slot'] == slot][0]

    def _rt_res(self, st):
        return self.cap.obj(st['om']['render_targets'][0]['view'])['resource'], (
            self.cap.obj(st['om']['depth_stencil'])['resource'] if st['om']['depth_stencil'] else None)

    def engine_rec(self, res, name, depth=False):
        if isinstance(res, tuple):
            res, dres = res
        else:
            dres = None
        d = self.cap.obj(res)['desc']
        rec = {'resource': res, 'engine': True, 'name': name,
               'rt': UnityRenderTexture(name, d['Width'], d['Height'], 0, 'ARGBHalf', 'Default')}
        for v in self.views_of[res]:
            rec.setdefault(v['kind'], v['id'])
        if dres is not None:
            for v in self.views_of[dres]:
                if v['kind'] == 'dsv':
                    rec['dsv'] = v['id']
        return rec

    # ---------------------------------------------------------- role map
    def on_handle(self, h, rec):
        """A plugin handle got a texture: capture resources of that role in the
        current render now mean the generated texture."""
        if self.k is None or 'resource' not in rec:
            return
        for t in self.roles.get(h, []):
            res = t['resource']
            depth = bool(t.get('depth')) or self.cap.obj(res)['desc'].get('BindFlags', 0) & 64
            if depth:
                if not rec.get('depth_resource'):
                    raise fr.FrameGap('%s: captured depth surface but generated RT has none' % h)
                self.res_map[res] = rec['depth_resource']
                for v in self.views_of[res]:
                    self.view_map[v['id']] = rec['dsv'] if v['kind'] == 'dsv' else rec['depth_srv']
            else:
                self.res_map[res] = rec['resource']
                for v in self.views_of[res]:
                    self.view_map[v['id']] = rec[v['kind']]
            self.role_log.append({'segment': self.k, 'handle': h, 'capture_resource': res,
                                  'generated_resource': self.res_map[res]})

    def map_view(self, v):
        return self.view_map.get(v, v)

    # ---------------------------------------------------------- forward-interface answers
    def next_generated_action(self, kind):
        ci = self.queues[kind].popleft()
        self.advance_to(ci)
        self.replaced.add(ci)
        return ci

    def generated_action_done(self, ci, gidx):
        self.cap_to_gen[ci] = gidx
        self.cursor = max(self.cursor, ci + 1)

    def render_camera(self, args, on_opaque):
        cam = args['camera']
        ps, labels = self.seg_passes, self.labels
        if cam == 'SSS profile Camera':
            i = labels.index('sss_profile') + 1
            self.advance_to(self.pass_actions(ps[i])[-1] + 1)
        elif cam == 'SSS Lighting Pass Camera':
            i = labels.index('sss_lighting')
            self.advance_to(self.pass_actions(ps[i])[-1] + 1)
            on_opaque(self.engine['conv_source'], self.engine['conv_destination'])
        else:
            raise fr.FrameGap('render_camera ' + cam)

    def draw_pass(self, op):
        ci = self.queues['forward'].popleft()
        self.advance_to(ci + 1)

    def active_color_target(self):
        """Called by PluginLowering right before it emits a GrabPass copy: the
        captured copy it replaces gives the active target (external:capture)."""
        ci = self.queues['grab'].popleft()
        self.advance_to(ci)
        self.replaced.add(ci)
        self.cursor = ci + 1
        self.cap_to_gen[ci] = len(self.em.b.program['commands'])
        src = self.cap.commands[ci]['src']
        return {'resource': self.res_map.get(src, src)}

    def engine_buffer(self, name):
        key = {'rtCamera': 'rtCamera', 'camera motion vectors (global)': 'mv'}.get(name)
        if key is None:
            raise fr.FrameGap('engine buffer ' + name)
        return self.engine[key]

    def resolve_handle(self, h):
        if self.lowering and h in self.lowering.handles:
            return self.lowering.handles[h]
        raise fr.FrameGap('global texture handle %s not live' % h)


def diff_blit(rec, cap, b, lib):
    g = b.program['commands'][rec.command]
    c = cap.commands[rec.capture_command]
    objs = {e['id']: e for t in ('shaders', 'blend_states', 'depth_states', 'raster_states', 'samplers', 'views',
                                 'resources') for e in b.program[t]}
    gs, cs = g['state'], c['state']
    row = {'event': rec.capture_event, 'shader': rec.shader, 'pass': rec.pass_index, 'external': rec.external,
           'notes': rec.notes}
    ok_all = True
    for st in ('vs', 'ps'):
        sha = objs[gs[st]['shader']]['blob']
        row[st] = sha == cap.shader_sha(cs[st]['shader'])
        refl = lib.reflection(sha)
        cbs = {}
        for x in gs[st]['constant_buffers']:
            gbytes = b.blobs.read(x['blob'])
            bound = cap.bound_cbuffer(rec.capture_command, st, x['slot'])
            n = 16 * refl['cb'][str(x['slot'])]['float4_count']
            cbs[x['slot']] = {'declared_equal': bound is not None and bound[1][:n] == gbytes[:n],
                              'full_equal': bound is not None and bound[1] == gbytes,
                              'external_vars': rec.cbuffers.get((st, x['slot']), {}).get('external', [])}
        row[st + '_cbuffers'] = cbs
        used_srv = {int(k) for k in refl['srv']}
        used_smp = {int(k) for k in refl['sampler']}

        def vdesc(view_id, table):
            v = objs[view_id] if view_id in objs else cap.obj(view_id)
            r = objs[v['resource']] if v['resource'] in objs else cap.obj(v['resource'])
            return v['desc'], r['desc']
        gsrv = [(x['slot'], vdesc(x['view'], objs)) for x in gs[st]['srvs'] if x['slot'] in used_srv]
        csrv = [(x['slot'], vdesc(x['view'], None)) for x in cs[st]['srvs'] if x['slot'] in used_srv]
        gsrv.sort(key=lambda x: x[0])
        csrv.sort(key=lambda x: x[0])
        row[st + '_srvs'] = gsrv == csrv
        if gsrv != csrv:
            row[st + '_srv_detail'] = {'generated': gsrv, 'captured': csrv}
        gsm = [(x['slot'], (objs.get(x['sampler']) or cap.obj(x['sampler']))['desc']) for x in gs[st]['samplers']
               if x['slot'] in used_smp]
        csm = [(x['slot'], cap.obj(x['sampler'])['desc']) for x in cs[st]['samplers'] if x['slot'] in used_smp]
        gsm.sort(key=lambda x: x[0])
        csm.sort(key=lambda x: x[0])
        row[st + '_samplers'] = gsm == csm
        if gsm != csm:
            row[st + '_sampler_detail'] = {'generated': [(s, d['Filter'], d['MaxAnisotropy']) for s, d in gsm],
                                           'captured': [(s, d['Filter'], d['MaxAnisotropy']) for s, d in csm]}
        ok_all &= row[st] and row[st + '_srvs'] and row[st + '_samplers'] and all(
            v['declared_equal'] for v in cbs.values())
    for name, a, k in (('blend', 'om', 'blend_state'), ('depth_stencil', 'om', 'depth_stencil_state'),
                       ('raster', 'rs', 'state')):
        gd = (objs.get(gs[a][k]) or cap.obj(gs[a][k]))['desc']
        row[name] = gd == cap.obj(cs[a][k])['desc']
        ok_all &= row[name]
    row['viewport'] = gs['rs']['viewports'] == cs['rs']['viewports']
    gt = [(objs.get(r['view']) or cap.obj(r['view'])) for r in gs['om']['render_targets']]
    ct = [cap.obj(r['view']) for r in cs['om']['render_targets']]
    row['targets'] = [(v['desc'], (objs.get(v['resource']) or cap.obj(v['resource']))['desc']) for v in gt] == \
        [(v['desc'], cap.obj(v['resource'])['desc']) for v in ct]
    row['depth_bound'] = (gs['om']['depth_stencil'] is None) == (cs['om']['depth_stencil'] is None)
    ok_all &= row['viewport'] and row['targets'] and row['depth_bound']
    row['equal'] = bool(ok_all)
    return row


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    lib = sl.ShaderLibrary([Path(args.library) / r for r in LIB_ROOTS])
    cap = CaptureProgram(args.capture_program)
    fgcap = vpe.Capture(args.frame_graph, args.captured_cbuffers)
    manifests = ShaderManifests(args.shader_manifest)
    rf = vpe.load(args.runtime_fields)
    plan, sss, conv, cam, quality, source = vpe.build_plan(rf, manifests)
    cb_names, conflicts = fr.capture_cbuffer_names(cap, lib)
    n_segments = len(plan.renders)
    # checkpoints: last action of every capture pass of the 16 renders
    checks = []
    for k in range(len(fgcap.segments)):
        for pidx in fgcap.segments[k]:
            p = fgcap.passes[pidx]
            acts = []
            for e in p['events']:
                try:
                    i = cap.command_at_event(e)
                except KeyError:
                    continue
                if cap.commands[i]['op'] in ACTIONS:
                    acts.append(i)
            if acts:
                checks.append({'segment': k, 'pass': pidx, 'label': p['label'], 'capture_command': acts[-1]})

    def outputs(st, op, obj):
        if op == 'dispatch':
            return [obj(u['view'])['resource'] for u in st['cs']['uavs']]
        t = [obj(r['view'])['resource'] for r in st['om']['render_targets']]
        if st['om']['depth_stencil'] is not None:
            t.append(obj(st['om']['depth_stencil'])['resource'])
        return t

    def build(root, generate):
        b = ProgramBuilder(root, provenance={**cap.program['provenance'], 'generator': fr.GENERATOR if generate else
                                             'capture oracle (verbatim capture program)',
                                             'purpose': 'retry bridge frame 17675 plugin work in-frame'},
                           first_id=cap.max_id + 1)
        cap.import_into(b, reason='capture objects (no paired scene state)')
        em = fr.FrameEmitter(b, cap, lib, cbuffer_names=cb_names)
        em.seed_frame_start()
        drv = RetryDriver(cap, fgcap, em, plan, lib)
        fwd = fr.UnityForwardLowering(em, lib, drv)
        if generate:
            low = PluginLowering(b, manifests, fwd, 'Linear')
            low.handles = NotifyDict(drv.on_handle)
            drv.lowering = low
            low.emit(plan.setup)
            for k in range(n_segments):
                drv.begin_segment(k)
                rec = plan.renders[k]
                low.emit(rec.sss_ops)
                low.emit(rec.transparent_ops)
                src = drv.engine['chain_source']
                for i, comp in enumerate(rec.image_effect_chain):
                    last = i == len(rec.image_effect_chain) - 1
                    dst = low.handles[comp['destination']] if last else drv.engine['chain_destination']
                    low.emit(comp['ops'], {'source': src, 'destination': dst})
                    src = dst
                low.emit(rec.post_render)
                for q, rest in drv.queues.items():
                    if rest:
                        raise fr.FrameGap('segment %d: %d unconsumed %s actions' % (k, len(rest), q))
            drv.finish()
        else:
            for i in range(len(cap.commands)):
                drv.cap_to_gen[i] = em.splice(i, 'oracle')
        views = {v['id']: v for v in b.program['views']}
        for c in checks:
            ci = c['capture_command']
            gi = drv.cap_to_gen[ci]
            g = b.program['commands'][gi]
            gouts = outputs(g['state'], g['op'], lambda v: views[v])
            couts = outputs(cap.commands[ci]['state'], cap.commands[ci]['op'], cap.obj)
            for n, (cr, gr) in enumerate(zip(couts, gouts)):
                if cap.obj(cr)['kind'] == 'buffer':
                    continue
                name = 's%02d_p%04d_%s_%d' % (c['segment'], c['pass'], c['label'], n)
                b.readback(name, gr, subresource=0, after_command=gi, orientation='d3d_rows_top_down')
        b.write()
        return b, em, drv, fwd

    gb, em, drv, fwd = build(out / 'generated', True)
    rows = [diff_blit(r, cap, gb, lib) for r in fwd.records]
    summary = {'blits': len(rows), 'blits_equal': sum(r['equal'] for r in rows)}
    fails = collections.Counter()
    for r in rows:
        for k, v in r.items():
            if v is False:
                fails[k] += 1
            if isinstance(v, dict) and k.endswith('_cbuffers'):
                for vv in v.values():
                    if not vv['declared_equal']:
                        fails[k] += 1
    summary['failing_fields'] = dict(fails)
    grab_cmds = [drv.cap_to_gen[ci] for ci in sorted(drv.replaced) if cap.commands[ci]['op'] == 'copy_subresource_region']
    report = {'schema': 'hs2_frame_orchestrator_retry/1', 'summary': summary, 'blits': rows,
              'generated_grab_copies': len(grab_cmds), 'replaced_capture_actions': len(drv.replaced),
              'role_map': drv.role_log, 'cbuffer_name_conflicts': conflicts, 'gaps': em.gaps}
    if not args.no_execute:
        if not args.reuse_oracle:
            build(out / 'oracle', False)
        ex = {}
        runs = [('generated', out / 'generated')] + ([] if args.reuse_oracle else [('oracle', out / 'oracle')])
        for name, root in runs:
            cmd = [str(args.executor), str(root), str(out / ('exec_' + name)), '--hash-only']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            (out / ('exec_%s_stdout.txt' % name)).write_text(proc.stdout, encoding='utf-8')
            (out / ('exec_%s_stderr.txt' % name)).write_text(proc.stderr, encoding='utf-8')
            if proc.returncode != 0:
                raise RuntimeError('frame_executor %s failed (%d)' % (name, proc.returncode))
        odir = Path(args.reuse_oracle) / 'exec_oracle' if args.reuse_oracle else out / 'exec_oracle'
        for name, d in (('generated', out / 'exec_generated'), ('oracle', odir)):
            ex[name] = {r['name']: r['sha256'] for r in json.loads((d / 'execution.json').read_text(encoding='utf-8'))['readbacks']}
        names = sorted(ex['oracle'])
        res = [{'name': n, 'equal': ex['generated'].get(n) == ex['oracle'][n]} for n in names]
        report['execution_summary'] = {'checkpoints': len(res), 'equal': sum(r['equal'] for r in res)}
        report['execution_mismatches'] = [r['name'] for r in res if not r['equal']]
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture-program', required=True)
    ap.add_argument('--frame-graph', required=True)
    ap.add_argument('--captured-cbuffers', required=True)
    ap.add_argument('--runtime-fields', required=True)
    ap.add_argument('--shader-manifest', action='append', default=[])
    ap.add_argument('--library', required=True)
    ap.add_argument('--executor', default=str(ROOT / 'outputs/offline_renderer_executor_20261020/build_v3/frame_executor.exe'))
    ap.add_argument('--reuse-oracle')
    ap.add_argument('--no-execute', action='store_true')
    ap.add_argument('--out', required=True)
    rep = run(ap.parse_args(argv))
    print(json.dumps(rep['summary']))
    print(json.dumps(rep.get('execution_summary')))


if __name__ == '__main__':
    main()
