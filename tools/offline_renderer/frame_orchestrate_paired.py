"""Generate the paired Maker frame (26059) as one frame program, diff it per pass
against the converted capture, and execute both.

Inputs: the paired session (scene state window + official zip.xml + driver
listing), the installed shader library and the capture program made by
``capture_to_program.py`` from the same RDC.

Generated (from scene state, ``hs2pipe.frame``): every forward-family draw the
plan reproduces (ForwardBase opaque, transparent queue): shaders, D3D11 states,
constant buffers through the frame's shared pool.  Everything else is spliced
from the capture and labelled ``external:capture`` (see the report's
``spliced_steps`` and per-draw ``external`` fields).

Outputs (bulky, under ``--out``): ``generated/`` and ``oracle/`` programs (the
oracle is the capture program with identical read-backs), executor outputs,
``report.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.frame_program import ProgramBuilder  # noqa: E402
from src.render.hs2pipe import frame as fr  # noqa: E402
from src.render.hs2pipe import forward_values as FV  # noqa: E402
from src.render.hs2pipe import shader_library as sl  # noqa: E402
from src.render.hs2pipe import unity_d3d11_state as ds  # noqa: E402
from src.render.hs2pipe import unity_forward_plan as up  # noqa: E402
from src.render.hs2pipe.external_capture import DRAW_OPS, CaptureProgram  # noqa: E402
from tools.offline_renderer.frame_structural_diff import (LIB_ROOTS, capture_passes, compare_draw_lists,  # noqa: E402
                                                           explained_by_export_precision)

REASON = 'paired capture 04d1 frame 26059 (capture_to_program --through-event 2210)'


def _copy_pass(copy, sub_index, pass_index):
    sub = [x for x in copy.subshaders if x.index == sub_index][0]
    pz = [x for x in sub.passes if x.index == pass_index][0]
    if pz.type == 'use':
        rows = copy.use_pass(pz, tier=None)
        if len(rows) != 1:
            raise fr.FrameGap('UsePass resolved to %d passes' % len(rows))
        pz = rows[0]
    return pz


def select_pass_programs(lib, pass_obj, keywords, captured):
    """Selected Subprogram per stage of one pass.  All installed copies of the
    pass's shader must agree; when they do not, the copy whose selection equals
    the captured DXBC is used and labelled external:capture (bundle of the
    material's shader not exported)."""
    rows = []
    for copy in lib.copies(pass_obj.shader.name):
        pz = _copy_pass(copy, pass_obj.subshader, pass_obj.index)
        sel = {st: pz.select(st, keywords, tier=None) for st in pz.stages()}
        rows.append((copy, pz, {st: (x.subprogram if x else None) for st, x in sel.items()}))
    shas = {tuple(sorted((st, sp.dxbc_sha256 if sp else None) for st, sp in r[2].items())) for r in rows}
    if len(shas) == 1:
        return rows[0][1], rows[0][2], None
    for copy, pz, sps in rows:
        if all((sps.get(st).dxbc_sha256 if sps.get(st) else None) == captured.get(st) for st in ('vs', 'ps')):
            return pz, sps, 'external:capture (installed copies differ; copy %s chosen by captured DXBC)' % copy.source_label
    return rows[0][1], {'vs': None, 'ps': None}, 'unresolved: no installed copy selects the captured programs'


def forward_pass_object(lib, g):
    shader, sub = lib.copies(g.material['shader'])[0].resolve(tier=None)
    return [x for x in sub.passes if x.index == g.pass_index][0]


def ppv2_roles(cap, lib):
    """CameraTarget / its depth / _CameraDepthTexture resource and view ids, from the
    capture draws of MultiScaleVO pass 2 (composite) and pass 0 (depth copy)."""
    out = {}
    for i, c in enumerate(cap.commands):
        if c['op'] not in DRAW_OPS:
            continue
        st = c['state']
        m = lib.match(cap.shader_sha(st['ps']['shader']))
        if not m or m[0]['shader'] != 'Hidden/PostProcessing/MultiScaleVO':
            continue
        if m[0]['pass'] == 2 and 'camera_target' not in out:
            rtv = st['om']['render_targets'][0]['view']
            out.update(camera_target=cap.obj(rtv)['resource'], camera_target_rtv=rtv,
                       camera_depth=cap.obj(st['om']['depth_stencil'])['resource'],
                       camera_depth_dsv=st['om']['depth_stencil'])
        if m[0]['pass'] == 0 and 'depth_texture' not in out:
            v = st['ps']['srvs'][0]['view']
            out.update(depth_texture=cap.obj(v)['resource'], depth_texture_srv=v)
    srvs = [v for v in cap.program['views'] if v['kind'] == 'srv' and v['resource'] == out['camera_target']]
    if len(srvs) != 1:
        raise fr.FrameGap('camera target SRV not unique')
    out['camera_target_srv'] = srvs[0]['id']
    return out


class PPv2Run:
    """PostProcessLayer of the Main Camera inside the frame program: shader-library
    programs, the frame's shared cbuffer pool, its own temporary-RT pool.  Called
    at OnPreCull (immediate uploads), CameraEvent.BeforeImageEffectsOpaque (opaque
    command buffer) and CameraEvent.BeforeImageEffects (post command buffer)."""

    PASS_LABELS = ('ambient_occlusion_msvo', 'post_copy_std', 'post_depth_of_field', 'color_grading_lut_bake',
                   'post_uber', 'post_smaa', 'post_final_pass')

    def __init__(self, b, em, lib, cap, scene, inputs, fv):
        from src.render.hs2pipe import ppv2, ppv2_probe as pp
        from src.render.hs2pipe.ppv2_unity import TemporaryRTPool, TextureRef
        self.ppv2, self.pp, self.TextureRef = ppv2, pp, TextureRef
        self.b, self.em, self.cap, self.scene, self.inp, self.fv = b, em, cap, scene, inputs, fv
        s_pp = inputs['prev_prev_scene']
        self.gen = ppv2.PostProcessLayerGen(b, fr.LibraryShaderSource(lib), ppv2.PPv2Assets(inputs['static']),
                                            pp.layer_config(scene), cb_pool=em.pool, rt_pool=TemporaryRTPool(b),
                                            dithering_state=pp.dithering_state(s_pp), curve_pixels=pp.curve_pixels(s_pp),
                                            anisotropic_filtering=pp.anisotropic_filtering(scene))
        roles = inputs['roles']

        def res(rid):
            return cap.obj(rid)['desc']
        ctd = res(roles['camera_target'])
        ct = TextureRef(roles['camera_target'], roles['camera_target_srv'], ctd['Width'], ctd['Height'],
                        rtv=roles['camera_target_rtv'], filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0,
                        name='CameraTarget (external:capture)', resource_desc=ctd)
        dp = TextureRef(roles['camera_depth'], None, ct.width, ct.height, dsv=roles['camera_depth_dsv'],
                        filter_mode='Point', wrap_mode='Clamp', aniso_level=0,
                        name='CameraTarget depth (external:capture)', resource_desc=res(roles['camera_depth']))
        attrs = pp.ProbeScene(scene).texture_attrs('Camera DepthTexture')
        dt = TextureRef(roles['depth_texture'], roles['depth_texture_srv'], ct.width, ct.height,
                        name='_CameraDepthTexture (external:capture)', resource_desc=res(roles['depth_texture']), **attrs)
        self.ct, self.dp, self.dt = ct, dp, dt
        self.skip = set()
        firsts = {}
        for i, c in enumerate(cap.commands):
            if c['op'] not in DRAW_OPS and c['op'] != 'dispatch':
                continue
            lab = inputs['event_pass'].get(cap.event(i), (None, None))[1]
            if lab in self.PASS_LABELS:
                self.skip.add(i)
                firsts.setdefault('opaque' if lab == 'ambient_occlusion_msvo' else 'post', i)
        pre = [i for i, c in enumerate(cap.commands)
               if inputs['event_pass'].get(cap.event(i), (None, None))[1] == 'post_curves_texture_copy']
        if not pre:
            raise fr.FrameGap('no OnPreCull upload point (post_curves_texture_copy) in the capture')
        self.segments = {pre[0]: self.on_pre_cull, firsts['opaque']: self.run_opaque, firsts['post']: self.run_post}
        self.records = {}

    def warm_up(self):
        """Frame N-1 (temporary-RT pool, Dithering and m_Pixels), into a mirror target."""
        pp, ppv2 = self.pp, self.ppv2
        d = dict(self.ct.resource_desc)
        rid = self.b.texture('tex2d', d, name='PPv2 warm-up camera target (frame N-1 mirror)')
        warm = self.TextureRef(rid, self.b.view('srv', rid, self.cap.obj(self.ct.srv)['desc']), self.ct.width,
                               self.ct.height, rtv=self.b.view('rtv', rid, self.cap.obj(self.ct.rtv)['desc']),
                               filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0, name='warm', resource_desc=d)
        s = self.inp['prev_scene']
        first = len(self.b.program['commands'])
        inputs = ppv2.PPv2FrameInputs(camera_target=warm, camera_depth=self.dp, camera_depth_texture=self.dt,
                                      globals={'_CameraDepthTexture': self.dt,
                                               '_ZBufferParams': pp.global_vector(s, '_ZBufferParams')})
        o, p = self.gen.on_pre_cull(pp.camera_state(s), pp.volumes_from_probe(s, self.inp['static']),
                                    frame=s['statics']['UnityEngine.Time']['renderedFrameCount'], inputs=inputs,
                                    render_settings=pp.render_settings(s))
        ex = ppv2.Executor(self.gen)
        ex.run(o, 'opaque')
        ex.run(p, 'post')
        self.warm_range = (first, len(self.b.program['commands']))

    def on_pre_cull(self):
        pp, ppv2 = self.pp, self.ppv2
        s = self.scene
        self.inputs = ppv2.PPv2FrameInputs(camera_target=self.ct, camera_depth=self.dp, camera_depth_texture=self.dt,
                                           globals={'_CameraDepthTexture': self.dt,
                                                    '_ZBufferParams': tuple(self.fv.camera['_ZBufferParams'])})
        first = len(self.b.program['commands'])
        self.opaque, self.post = self.gen.on_pre_cull(pp.camera_state(s), pp.volumes_from_probe(s, self.inp['static']),
                                                      frame=s['statics']['UnityEngine.Time']['renderedFrameCount'],
                                                      inputs=self.inputs, render_settings=pp.render_settings(s))
        self.precull_range = (first, len(self.b.program['commands']))
        self.ex = ppv2.Executor(self.gen)

    def run_opaque(self):
        n = len(self.ex.records)
        self.ex.run(self.opaque, 'opaque')
        self.records['opaque'] = self.ex.records[n:]

    def run_post(self):
        n = len(self.ex.records)
        self.ex.run(self.post, 'post')
        self.records['post'] = self.ex.records[n:]


def fr_mesh(r):
    if r['kind'] == 'ParticleSystemRenderer':
        return 'Buffer-2-'
    if r['kind'] == 'MeshRenderer':
        return 'o_' + r['name'][2:] if r['name'].startswith('p_') else r['name']
    return r['mesh']


def material_floats(m):
    return {k: v['value'] for k, v in (m.get('typed_properties_from_manifest') or {}).items()
            if v.get('has') and v['type'] in ('float', 'range', 'int')}


def diff_draw(rec, cap, b):
    """Per-draw structural comparison (generated objects only; external fields listed)."""
    gcmd = b.program['commands'][rec.command]
    ccmd = cap.commands[rec.capture_command]
    gs, cs = gcmd['state'], ccmd['state']
    objs = {e['id']: e for t in ('shaders', 'blend_states', 'depth_states', 'raster_states') for e in b.program[t]}

    def desc(st, path):
        a, k = path
        i = st[a][k]
        return objs[i].get('desc') if i in objs and 'desc' in objs[i] else cap.obj(i)['desc']
    row = {'pass_label': rec.pass_label, 'event': rec.capture_event, 'renderer': rec.renderer, 'shader': rec.shader, 'pass': rec.pass_index,
           'external': rec.external}
    for stage in ('vs', 'ps'):
        g = objs[gs[stage]['shader']]['blob'] if gs[stage]['shader'] in objs else cap.shader_sha(gs[stage]['shader'])
        row[stage] = 'equal' if g == cap.shader_sha(cs[stage]['shader']) else 'different'
    for name, path in (('blend', ('om', 'blend_state')), ('depth_stencil', ('om', 'depth_stencil_state')),
                       ('raster', ('rs', 'state'))):
        row[name] = 'equal' if desc(gs, path) == desc(cs, path) else 'different'
    row['stencil_ref'] = 'equal' if gs['om']['stencil_ref'] == cs['om']['stencil_ref'] else 'different'
    cbrows = {}
    for (stage, slot), cb in rec.cbuffers.items():
        bid, cbytes = cap.bound_cbuffer(rec.capture_command, stage, slot)
        g = cb['bytes']
        diff = [i for i in range(min(len(g), len(cbytes))) if g[i] != cbytes[i]]
        owner = {}
        for v in cb['layout'].variables:
            for a, z in fr.variable_ranges(v):
                for k in range(a, z):
                    owner[k] = v.name
        dvars = {}
        for k in diff:
            n = owner.get(k, '<undeclared>')
            dvars.setdefault(n, []).append(k)
        detail = {}
        for n, ks in dvars.items():
            kind = ('generated' if n in cb['generated'] else 'external' if n in cb['external'] else 'undeclared')
            if kind == 'generated' and n in getattr(rec, 'inputs', {}):
                var = cb['layout'].variable(n)
                hexv = cbytes[var.offset:var.offset + 4 * min(var.columns, 3)].hex()
                if explained_by_export_precision(rec.inputs[n], hexv):
                    kind = 'generated_export_decimal_precision'
            elif kind == 'undeclared':
                # bytes of this shared (name, size) buffer declared by the other stage's layout
                other = [r for (st2, sl2), r in rec.cbuffers.items() if r['key'] == cb['key'] and r is not cb]
                for r in other:
                    for v in r['layout'].variables:
                        if any(a <= ks[0] < z for a, z in fr.variable_ranges(v)):
                            kind = 'other_stage:' + v.name
            detail[n] = {'kind': kind, 'bytes': len(ks), 'first': ks[0],
                         'generated_hex': g[ks[0] & ~3:(ks[-1] | 3) + 1].hex(),
                         'captured_hex': cbytes[ks[0] & ~3:(ks[-1] | 3) + 1].hex()}
        cbrows['%s:%d:%s' % (stage, slot, cb['name'])] = {
            'size': cb['size'], 'capture_buffer': bid, 'capture_size': len(cbytes),
            'equal': not diff and len(g) == len(cbytes), 'differing_bytes': len(diff),
            'differences': detail, 'generated_vars': len(cb['generated']),
            'external_vars': cb['external'], 'pool_sync': cb['pool_sync'], 'size_note': cb['size_note']}
    row['cbuffers'] = cbrows
    row['status'] = 'equal' if (all(row[k] == 'equal' for k in ('vs', 'ps', 'blend', 'depth_stencil', 'raster',
                                                                 'stencil_ref'))
                                and all(c['equal'] for c in cbrows.values())) else 'different'
    return row


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    W = Path(args.window)
    scene = json.loads((W / 'frames' / ('scene_%03d.json' % args.window_index)).read_text(encoding='utf-8'))
    frame = json.loads((W / 'frames' / ('frame_%03d.json' % args.window_index)).read_text(encoding='utf-8'))
    static = json.loads((W / 'static.json').read_text(encoding='utf-8'))
    lib = sl.ShaderLibrary([Path(args.library) / r for r in LIB_ROOTS])
    cap = CaptureProgram(args.capture_program)
    ad = up.SceneAdapter(scene, frame, static, lib)
    cam = ad.camera('Main Camera')
    q = scene['statics']['UnityEngine.QualitySettings']
    quality = {'pixel_light_count': q['pixelLightCount'], 'shadows': up.enum_value(q['shadows'])}
    gk = sorted(k for k, v in scene['global_keywords'].items() if v)
    cap_passes, _ = capture_passes(Path(args.xml), Path(args.inspect))
    opaque, active = up.forward_plan(ad, cam, transparent=False, global_keywords=gk, quality=quality)
    transparent, _ = up.forward_plan(ad, cam, transparent=True, global_keywords=gk, quality=quality)
    t_draws = [d for d in transparent if d.light_mode != 'ALWAYS' or d.shader.startswith('Legacy')]
    fwd = [p for p in cap_passes if p['label'] == 'forward_scene']
    o_cap = [d for d in fwd[0]['draws'] if d['mesh'] != '']
    t_cap = [d for p in fwd[1:] for d in p['draws']]
    lists = {'forward_opaque': compare_draw_lists(opaque, o_cap), 'forward_transparent': compare_draw_lists(t_draws, t_cap)}
    for k, rows in lists.items():
        bad = [r for r in rows if r['status'] not in ('equal', 'equal_object_program_unresolved')]
        if bad:
            raise fr.FrameGap('%s draw list differs from the capture; refusing in-place emission: %r' % (k, bad[:2]))
    from src.render import paired_skin as ps
    from src.render import unity_skin_emu as ue
    prev_path = W / 'frames' / ('scene_%03d.json' % (args.window_index - 1))
    prev_scene = json.loads(prev_path.read_text(encoding='utf-8')) if prev_path.exists() else None
    fv = FV.ForwardValues(scene, ad, lib, ps.ProbeWindow(W), ue.shared_emulator(), cam, active, prev_scene)
    # generated replacements, keyed by capture command index
    plan = {}
    for label, gl, cl in (('forward_opaque', opaque, o_cap), ('forward_transparent', t_draws, t_cap)):
        for g, c in zip(gl, cl):
            ci = cap.command_at_event(c['event'])
            plan[ci] = [(label, {
                'renderer': g.renderer, 'material': g.material, 'shader': g.shader,
                'pass': forward_pass_object(lib, g), 'keywords': frozenset(g.keywords)}, c, ci)]
    from src.render.hs2pipe import unity_camera_passes as ucp
    pgaps = []
    cache = {}
    for label, cap_label, gl in (
            ('depth_texture', 'depth_texture_prepass', ucp.depth_texture_plan(ad, cam, lib, gk, quality, pgaps, cache)),
            ('depth_normals', 'depth_normals', ucp.depth_normals_plan(ad, cam, lib, gk, pgaps, cache))):
        cl = [d for p in cap_passes if p['label'] == cap_label for d in p['draws']]
        names = [fr_mesh(d['renderer']) for d in gl]
        ok = len(gl) == len(cl) and all(cm['mesh'].startswith(n) if n.endswith('-') else cm['mesh'] == n
                                        for n, cm in zip(names, cl))
        lists[label] = [{'status': 'equal' if ok else 'different'} for _ in gl]
        if not ok:
            raise fr.FrameGap('%s draw list differs from the capture: %r vs %r' % (label, names, [c['mesh'] for c in cl]))
        for g, c in zip(gl, cl):
            ci = cap.command_at_event(c['event'])
            plan[ci] = [(label, g, c, ci)]
    # motion vectors: object draws after the camera-motion quad
    mv_cap = [d for p in cap_passes if p['label'] == 'motion_vectors' for d in p['draws']][1:]
    by_mesh = {}
    for r in ad.renderers():
        by_mesh.setdefault(fr_mesh(r), []).append(r['id'])
    included = []
    for d in mv_cap:
        ids = by_mesh.get(d['mesh'], [])
        if len(ids) != 1:
            raise fr.FrameGap('motion-vector capture draw %s maps to renderers %r' % (d['mesh'], ids))
        included.append(ids[0])
    mv_plan, mv_candidates = ucp.motion_vector_plan(ad, cam, lib, gk, pgaps, set(included), cache)
    names = [fr_mesh(d['renderer']) for d in mv_plan]
    ok = names == [d['mesh'] for d in mv_cap]
    lists['motion_vectors'] = [{'status': 'equal' if ok else 'different'} for _ in mv_plan]
    report_mv = {'generated_order': names, 'captured_order': [d['mesh'] for d in mv_cap],
                 'candidates': len(mv_candidates), 'included_external': len(included)}
    # Emit the generated order at the first object draw of the pass; each generated
    # draw takes its external bindings from the capture draw of the same renderer.
    cap_of = {}
    for rid, c in zip(included, mv_cap):
        cap_of[rid] = c
    firsts = [cap.command_at_event(c['event']) for c in mv_cap]
    if args.diagnostic_capture_mv_order:
        # DIAGNOSTIC SUBSTITUTION (isolated): captured motion-vector order (the
        # RenderObject index tie-break input, scene node order, is not exported).
        by_id = {g['renderer']['id']: g for g in mv_plan}
        for rid, c in zip(included, mv_cap):
            ci = cap.command_at_event(c['event'])
            plan[ci] = [('motion_vectors', by_id[rid], c, ci)]
    else:
        plan[firsts[0]] = [('motion_vectors', g, cap_of[g['renderer']['id']],
                            cap.command_at_event(cap_of[g['renderer']['id']]['event'])) for g in mv_plan]
        for ci in firsts[1:]:
            plan[ci] = []
    event_pass = {}
    for p in cap_passes:
        for e in p['events']:
            event_pass[e] = (p['pass_index'], p['label'])

    cb_names, cb_conflicts = fr.capture_cbuffer_names(cap, lib)
    pp_inputs = None
    if args.ppv2_scene_state:
        def rd(k):
            pth = W / 'frames' / ('scene_%03d.json' % (args.window_index - k))
            return json.loads(pth.read_text(encoding='utf-8'))
        pp_inputs = {'static': json.loads(Path(args.ppv2_scene_state).read_text(encoding='utf-8')),
                     'prev_scene': rd(1), 'prev_prev_scene': rd(2), 'event_pass': event_pass,
                     'roles': ppv2_roles(cap, lib)}

    def build(root, generate):
        b = ProgramBuilder(root, provenance={**cap.program['provenance'], 'generator': fr.GENERATOR if generate else
                                             'capture oracle (verbatim capture program)', 'purpose': REASON},
                           first_id=cap.max_id + 1)
        cap.import_into(b, reason='capture objects (meshes, textures, samplers, targets, cbuffer residue)')
        em = fr.FrameEmitter(b, cap, lib, cbuffer_names=cb_names)
        dev = ds.DeviceRaster(invert_projection=True)
        last_cmd_of_pass = {}
        ppv = PPv2Run(b, em, lib, cap, scene, pp_inputs, fv) if (generate and pp_inputs is not None) else None
        segments = {}
        if ppv is not None:
            ppv.warm_up()
            segments = ppv.segments
        for i, c in enumerate(cap.commands):
            if ppv is not None and i in segments:
                segments[i]()
            if ppv is not None and i in ppv.skip:
                em.skipped.append((i, 'replaced by generated PPv2'))
                ev = cap.event(i)
                if ev in event_pass and (c['op'] in DRAW_OPS or c['op'] == 'dispatch'):
                    last_cmd_of_pass[event_pass[ev][0]] = (len(b.program['commands']) - 1, i)
                continue
            if generate and i in plan:
                idx = None
                for label, g, cd, ci in plan[i]:
                    pz, sps, why = select_pass_programs(lib, g['pass'], g['keywords'], {'vs': cd['vs'], 'ps': cd['ps']})
                    vals, notes = fv.draw_values(g['renderer'], g['material'])
                    if label == 'motion_vectors':
                        mv, src = fv.motion_vector_values(g['renderer'])
                        vals.update(mv)
                        notes = list(notes) + ['_PreviousM: ' + str(src)]
                    if args.diagnostic_capture_colours:
                        # DIAGNOSTIC SUBSTITUTION (isolated, not the delivered path): every
                        # GammaToLinear material colour is taken from the capture instead of
                        # the decimal export, to test whether the remaining residuals are
                        # only the export's colour precision.
                        for n in list(vals.get('_inputs', {})):
                            vals.pop(n, None)
                        notes = list(notes) + ['DIAGNOSTIC: material colours external:capture']
                    if why:
                        notes = list(notes) + [why]
                    rec = em.emit_draw(label=label, cap_index=ci, stage_programs=sps, pass_record=pz,
                                       light_mode=pz.light_mode, values=vals,
                                       material_floats=material_floats(g['material']), device=dev,
                                       renderer_name=g['renderer']['name'], shader_name=g['shader'],
                                       keywords=g['keywords'], notes=notes)
                    rec.inputs = vals.get('_inputs', {})
                    idx = rec.command
                if idx is None:
                    continue
            else:
                ev = cap.event(i)
                lab = event_pass.get(ev, (None, 'non-pass command'))[1]
                idx = em.splice(i, 'step not generated: ' + lab if c['op'] in DRAW_OPS or c['op'] == 'dispatch'
                                else 'capture resource update / clear / copy (%s)' % lab)
            ev = cap.event(i)
            if ev in event_pass and (c['op'] in DRAW_OPS or c['op'] == 'dispatch'):
                last_cmd_of_pass[event_pass[ev][0]] = (idx, i)
        # read-backs: outputs of the last action of every pass with actions
        checks = []
        for pidx, (gidx, cidx) in sorted(last_cmd_of_pass.items()):
            cmd = cap.commands[cidx]
            st = cmd['state']
            targets = []
            if cmd['op'] == 'dispatch':
                targets = [cap.obj(u['view'])['resource'] for u in st['cs']['uavs']]
            else:
                targets = [cap.obj(r['view'])['resource'] for r in st['om']['render_targets']]
                if st['om']['depth_stencil'] is not None:
                    targets.append(cap.obj(st['om']['depth_stencil'])['resource'])
            for res in targets:
                r = cap.obj(res)
                if r['kind'] == 'buffer':
                    continue
                name = 'p%02d_%s_r%d' % (pidx, [p for p in cap_passes if p['pass_index'] == pidx][0]['label'], res)
                b.readback(name, res, subresource=0, after_command=gidx, orientation='d3d_rows_top_down')
                checks.append({'name': name, 'pass_index': pidx, 'resource': res, 'command': gidx, 'capture_command': cidx})
        b.write()
        return b, em, checks

    gen_b, em, checks = build(out / 'generated', True)
    if not args.no_execute and not args.reuse_oracle:
        build(out / 'oracle', False)
    rows = [diff_draw(r, cap, gen_b) for r in em.records]
    report = {'schema': 'hs2_frame_orchestrator_paired/1', 'capture_program': str(args.capture_program),
              'draw_lists': {k: [r['status'] for r in v] for k, v in lists.items()},
              'generated_draws': len(em.records), 'spliced_commands': len(em.spliced),
              'spliced_by_reason': {}, 'draws': rows, 'gaps': ad.gaps + em.gaps + pgaps, 'motion_vectors': report_mv,
              'summary': {}}
    for _, reason in em.spliced:
        report['spliced_by_reason'][reason] = report['spliced_by_reason'].get(reason, 0) + 1
    tot = {}
    per_pass = {}
    for r in rows:
        pp = per_pass.setdefault(r['pass_label'], {'draws': 0, 'draws_equal': 0, 'cbuffers': {}})
        pp['draws'] += 1
        pp['draws_equal'] += r['status'] == 'equal'
        for k in ('vs', 'ps', 'blend', 'depth_stencil', 'raster', 'stencil_ref'):
            tot.setdefault(k, {}).setdefault(r[k], 0)
            tot[k][r[k]] += 1
        for cb in r['cbuffers'].values():
            kinds = {x['kind'].split(':')[0] for x in cb['differences'].values()}
            s = ('equal' if cb['equal'] else 'different_export_decimal_precision_only'
                 if kinds <= {'generated_export_decimal_precision', 'other_stage'} else 'different')
            tot.setdefault('cbuffers', {}).setdefault(s, 0)
            tot['cbuffers'][s] += 1
            pp['cbuffers'][s] = pp['cbuffers'].get(s, 0) + 1
    nvar = {'generated': 0, 'external': 0}
    ext_names = {}
    for rec in em.records:
        for cb in rec.cbuffers.values():
            nvar['generated'] += len(cb['generated'])
            nvar['external'] += len(cb['external'])
            for n in cb['external']:
                ext_names[n] = ext_names.get(n, 0) + 1
    tot['cbuffer_variables'] = nvar
    report['external_variables'] = dict(sorted(ext_names.items(), key=lambda x: -x[1]))
    report['summary'] = tot
    report['per_pass'] = per_pass
    if not args.no_execute:
        ex = {}
        runs = [('generated', out / 'generated')] + ([] if args.reuse_oracle else [('oracle', out / 'oracle')])
        if args.reuse_oracle:
            ex['oracle'] = {r['name']: r for r in json.loads((Path(args.reuse_oracle) / 'exec_oracle' / 'execution.json')
                                                               .read_text(encoding='utf-8'))['readbacks']}
            report['oracle_execution'] = str(Path(args.reuse_oracle) / 'exec_oracle')
        for name, root in runs:
            cmd = [str(args.executor), str(root), str(out / ('exec_' + name)), '--hash-only']
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            (out / ('exec_%s_stdout.txt' % name)).write_text(proc.stdout, encoding='utf-8')
            (out / ('exec_%s_stderr.txt' % name)).write_text(proc.stderr, encoding='utf-8')
            if proc.returncode != 0:
                raise RuntimeError('frame_executor %s failed (%d)' % (name, proc.returncode))
            ex[name] = {r['name']: r for r in json.loads((out / ('exec_' + name) / 'execution.json')
                                                          .read_text(encoding='utf-8'))['readbacks']}
        report['execution'] = [{**c, 'equal': ex['generated'][c['name']]['sha256'] == ex['oracle'][c['name']]['sha256'],
                                'generated_sha256': ex['generated'][c['name']]['sha256'],
                                'oracle_sha256': ex['oracle'][c['name']]['sha256']} for c in checks]
        report['execution_summary'] = {'checkpoints': len(checks),
                                       'equal': sum(1 for r in report['execution'] if r['equal'])}
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', required=True)
    ap.add_argument('--inspect', required=True)
    ap.add_argument('--window', required=True)
    ap.add_argument('--window-index', type=int, required=True)
    ap.add_argument('--library', required=True)
    ap.add_argument('--capture-program', required=True)
    ap.add_argument('--executor', default=str(ROOT / 'outputs/offline_renderer_executor_20261020/build_v3/frame_executor.exe'))
    ap.add_argument('--no-execute', action='store_true')
    ap.add_argument('--ppv2-scene-state', help='PPv2 static scene state (extract_ppv2_scene_state output)')
    ap.add_argument('--reuse-oracle', help='earlier run directory whose exec_oracle read-backs are reused')
    ap.add_argument('--diagnostic-capture-mv-order', action='store_true',
                    help='DIAGNOSTIC: emit motion vectors in the captured order (labelled)')
    ap.add_argument('--diagnostic-capture-colours', action='store_true',
                    help='DIAGNOSTIC: take GammaToLinear material colours from the capture (labelled)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rep = run(args)
    print(json.dumps(rep['summary']))
    print(json.dumps(rep.get('execution_summary')))


if __name__ == '__main__':
    main()
