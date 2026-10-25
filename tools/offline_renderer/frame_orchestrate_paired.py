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
        self.lib = lib
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
                                               '_ZBufferParams': self.zbuffer_params(s)})
        o, p = self.gen.on_pre_cull(pp.camera_state(s), pp.volumes_from_probe(s, self.inp['static']),
                                    frame=s['statics']['UnityEngine.Time']['renderedFrameCount'], inputs=inputs,
                                    render_settings=pp.render_settings(s))
        ex = ppv2.Executor(self.gen)
        ex.run(o, 'opaque')
        ex.run(p, 'post')
        self.warm_range = (first, len(self.b.program['commands']))

    def zbuffer_params(self, s):
        """_ZBufferParams of the Main Camera of scene ``s`` (Camera::SetCameraShaderProps
        through unity_forward; the v4 export's global uniform when no camera rule input)."""
        if '_v5' not in s:
            return self.pp.global_vector(s, '_ZBufferParams')
        from src.render.hs2pipe import unity_forward as uf, unity_forward_plan as up
        cam = up.SceneAdapter(s, None, None, self.lib).camera('Main Camera')
        setup = uf.camera_setup(cam, FV.GRAPHICS_D3D11)
        return tuple(uf.camera_shader_props(cam, setup, FV.GRAPHICS_D3D11)['_ZBufferParams'])

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


def diff_ppv2_action(ci, rec, cap, b, lib):
    """Generated PPv2 draw/dispatch vs the captured one it replaces: shaders, states,
    constant-buffer bytes per slot, output and input view/resource descriptors."""
    g = b.program['commands'][rec['command']]
    c = cap.commands[ci]
    objs = {e['id']: e for t in ('shaders', 'blend_states', 'depth_states', 'raster_states', 'samplers', 'views',
                                 'resources') for e in b.program[t]}
    gs, cs = g['state'], c['state']
    row = {'capture_event': cap.event(ci), 'kind': rec['kind'], 'shader': rec.get('shader') or rec.get('compute'),
           'pass': rec.get('pass') if rec['kind'] == 'draw' else rec.get('kernel')}
    stages = ('cs',) if c['op'] == 'dispatch' else ('vs', 'ps')
    eq = True
    for st in stages:
        sha = objs[gs[st]['shader']]['blob']
        ok = sha == cap.shader_sha(cs[st]['shader'])
        row[st] = ok
        eq &= ok
        refl = lib.reflection(sha)
        cb, cb_full = {}, {}
        for slot_row in gs[st]['constant_buffers']:
            gbytes = b.blobs.read(slot_row['blob'])
            bound = cap.bound_cbuffer(ci, st, slot_row['slot'])
            n = 16 * refl['cb'][str(slot_row['slot'])]['float4_count']
            ok = bound is not None and bound[1][:n] == gbytes[:n]
            cb[slot_row['slot']] = ok
            cb_full[slot_row['slot']] = bound is not None and bound[1] == gbytes
            eq &= ok
        row[st + '_cbuffers'] = cb
        row[st + '_cbuffers_full_buffer_info'] = cb_full
        used = {'srvs': {int(k) for k in refl['srv']}, 'samplers': {int(k) for k in refl['sampler']}}

        def desc_of(table_objs, sid):
            return table_objs(sid)
        for kind in ('srvs', 'samplers'):
            gv = [(x['slot'], objs[x['view' if kind == 'srvs' else 'sampler']]) for x in gs[st][kind]
                  if x['slot'] in used[kind]]
            cv = [(x['slot'], cap.obj(x['view' if kind == 'srvs' else 'sampler'])) for x in cs[st][kind]
                  if x['slot'] in used[kind]]
            if kind == 'srvs':
                ok = [(sl, v['desc'], objs[v['resource']]['desc']) for sl, v in gv] == \
                     [(sl, v['desc'], cap.obj(v['resource'])['desc']) for sl, v in cv]
            else:
                ok = [(sl, v['desc']) for sl, v in gv] == [(sl, v['desc']) for sl, v in cv]
            row[st + '_' + kind] = ok
            eq &= ok
    if c['op'] != 'dispatch':
        for name, a, k in (('blend', 'om', 'blend_state'), ('depth_stencil', 'om', 'depth_stencil_state'),
                           ('raster', 'rs', 'state')):
            ok = objs[gs[a][k]]['desc'] == cap.obj(cs[a][k])['desc']
            row[name] = ok
            eq &= ok
        ok = gs['rs']['viewports'] == cs['rs']['viewports']
        row['viewport'] = ok
        eq &= ok
        gt = [(objs[r['view']]['desc'], objs[objs[r['view']]['resource']]['desc']) for r in gs['om']['render_targets']]
        ct = [(cap.obj(r['view'])['desc'], cap.obj(cap.obj(r['view'])['resource'])['desc'])
              for r in cs['om']['render_targets']]
        row['targets'] = gt == ct
        eq &= gt == ct
        row['args'] = g['args'] == c['args']
        if not row['args']:
            row['args_detail'] = {'generated': g['args'], 'captured': c['args']}
        eq &= row['args']
    else:
        row['groups'] = g['args'] == c['args']
        eq &= row['groups']
    row['equal'] = bool(eq)
    return row


def ppv2_summary(rows):
    out = {'actions': len(rows), 'equal': sum(r['equal'] for r in rows)}
    fails = {}
    for r in rows:
        for k, v in r.items():
            if v is False:
                fails[k] = fails.get(k, 0) + 1
            if isinstance(v, dict) and not k.endswith('_info'):
                for kk, vv in v.items():
                    if vv is False:
                        fails[k] = fails.get(k, 0) + 1
    out['failing_fields'] = fails
    return out


def fr_mesh(r):
    if r['kind'] == 'ParticleSystemRenderer':
        return 'Buffer-2-'
    if r['kind'] == 'MeshRenderer':
        return 'o_' + r['name'][2:] if r['name'].startswith('p_') else r['name']
    return r['mesh']


def material_floats(m):
    if '_native_floats' in m:
        return dict(m['_native_floats'])
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


F32 = __import__('numpy').float32


def shadow_branch(cap, ci, stage_programs, candidates):
    """Outcome of the normal-bias branch whose only unexported inputs are two
    GraphicsCaps bits (+0xd0 bit 7, +0x3fc): the candidate equal to the captured
    unity_LightShadowBias of this draw (external:capture, one boolean)."""
    import numpy as np
    for stage in ('vs', 'ps'):
        sp = stage_programs.get(stage)
        if sp is None:
            continue
        for lay in sp.constant_buffers():
            if lay.slot is None or not any(v.name == 'unity_LightShadowBias' for v in lay.variables):
                continue
            v = lay.variable('unity_LightShadowBias')
            got = cap.bound_cbuffer(ci, stage, lay.slot)[1][v.offset:v.offset + 16]
            for k, val in candidates.items():
                if np.asarray(val, np.float32).tobytes() == got:
                    return k
            raise fr.FrameGap('captured unity_LightShadowBias equals neither branch outcome at command %d' % ci)
    raise fr.FrameGap('shadow caster program declares no unity_LightShadowBias at command %d' % ci)


class ChainDriver:
    """Forward-interface answers for the main camera's OnRenderImage chain of an
    ordinary frame (CharaCustom.CustomRender, CTAA_PC).  Every chain texture is an
    engine/plugin object that exists before the frame (camera intermediates, the
    CustomRender rtCamera, the CTAA history textures): their identity is the capture
    resource bound at the replaced blit (allocation external:capture, contents carried
    across frames).  Capture commands between generated blits are spliced in order."""

    def __init__(self, cap, em, queue):
        import collections
        self.cap, self.em = cap, em
        self.queue = collections.deque(queue)
        self.cursor = None
        self.handled = set()
        self.cap_to_gen = {}
        self.engine = {}
        self.splice_fn = None

    def next_generated_action(self, kind):
        if kind != 'blit':
            raise fr.FrameGap('chain driver: unexpected %s action' % kind)
        ci = self.queue.popleft()
        for i in range(self.cursor, ci):
            if i not in self.handled:
                self.cap_to_gen[i] = self.splice_fn(i)
                self.handled.add(i)
        self.handled.add(ci)
        return ci

    def generated_action_done(self, ci, gidx):
        self.cap_to_gen[ci] = gidx
        self.cursor = ci + 1

    def map_view(self, v):
        return v

    def engine_buffer(self, name):
        if name not in self.engine:
            raise fr.FrameGap('chain driver: engine buffer %s' % name)
        return self.engine[name]

    def resolve_handle(self, h):
        raise fr.FrameGap('chain driver: global texture handle %s' % h)

    def render_camera(self, args, on_opaque):
        raise fr.FrameGap('chain driver: render_camera')

    def draw_pass(self, op):
        raise fr.FrameGap('chain driver: draw_pass')

    def active_color_target(self):
        raise fr.FrameGap('chain driver: active_color_target')


DXGI_RT_FORMATS = {9: 'ARGBHalf', 10: 'ARGBHalf', 1: 'ARGBFloat', 2: 'ARGBFloat', 33: 'RGHalf', 34: 'RGHalf', 28: 'ARGB32', 29: 'ARGB32', 27: 'ARGB32', 87: 'ARGB32',
                   91: 'ARGB32', 24: 'ARGB2101010'}


def scene_rt(tex, name):
    """UnityRenderTexture of an exported RenderTexture (descriptor + properties)."""
    from src.render.hs2pipe.plugin_common import UnityRenderTexture
    p = tex['properties']
    d = tex['descriptor']
    fmt = up.enum_value(d['RenderTextureDescriptor.<colorFormat>k__BackingField'])
    return UnityRenderTexture(name, int(p['width']), int(p['height']), int(p['depth']), fmt, 'Default',
                              msaa=int(p.get('antiAliasing', 1)), filter_mode=up.enum_value(p['filterMode']),
                              wrap_mode=up.enum_value(p['wrapModeU']), aniso_level=int(p.get('anisoLevel', 1)),
                              auto_generate_mips=bool(p.get('autoGenerateMips', True)),
                              use_mipmap=bool(p.get('useMipMap', False)), lifetime='persistent')


def chain_engine_rec(cap, views_of, res, rt, name, dres=None):
    d = cap.obj(res)['desc']
    if (int(d['Width']), int(d['Height'])) != (rt.width, rt.height):
        raise fr.FrameGap('%s: captured resource %dx%d differs from %dx%d' % (name, d['Width'], d['Height'],
                                                                               rt.width, rt.height))
    rec = {'resource': res, 'engine': True, 'name': name, 'rt': rt}
    if 'SwapChain-BackBuffer' in cap.obj(res).get('name', '') or 'Swap Chain Backbuffer' in cap.obj(res).get('name', ''):
        rec['backbuffer'] = True          # OnRenderImage destination null -> the screen
    for v in views_of.get(res, []):
        rec.setdefault(v['kind'], v['id'])
    if dres is not None:
        for v in views_of.get(dres, []):
            if v['kind'] == 'dsv':
                rec['dsv'] = v['id']
    return rec


def srv_resource(cap, lib, st, name, cache):
    """Capture resource bound to texture ``name`` of the draw's pixel program."""
    sha = cap.shader_sha(st['ps']['shader'])
    if sha not in cache:
        found = None
        for m in lib.match(sha):
            for c in lib.copies(m['shader']):
                for sub in c.subshaders:
                    for pz in sub.passes:
                        for x in pz.subprograms('ps'):
                            if x.dxbc_sha256 == sha:
                                found = found or x
        if found is None:
            raise fr.FrameGap('pixel program %s not in the library' % sha)
        cache[sha] = found
    slot = [t['srv'] for t in cache[sha].resources()['textures'] if t['name'] == name]
    if not slot:
        raise fr.FrameGap('texture %s not declared by %s' % (name, sha))
    return [cap.obj(x['view'])['resource'] for x in st['ps']['srvs'] if x['slot'] == slot[0]][0]


def d3d11_clears(flags, color, depth, stencil, rtvs, dsv, reversed_z=True):
    """``GfxDeviceD3D11Base::Clear`` (0x5445b0): GfxClearFlags bit 1 -> ClearRenderTargetView
    of every bound RTV with the colour as given; bits 2/4 with a bound DSV ->
    ClearDepthStencilView(D3D11 flags depth=1 / stencil=2, depth = 1 - depth when
    GraphicsCaps+0xdd (reversed Z), stencil)."""
    import numpy as np
    out = []
    if flags & 1:
        for v in rtvs:
            out.append(('clear_rtv', {'view': v, 'color': [float(np.float32(c)) for c in color]}))
    if flags & 6 and dsv is not None:
        df = ((flags >> 1) & 1) | (2 if flags & 4 else 0)
        d = np.float32(1.0) - np.float32(depth) if reversed_z else np.float32(depth)
        out.append(('clear_dsv', {'view': dsv, 'flags': df, 'depth': float(d), 'stencil': int(stencil)}))
    return out


def camera_clear_flags(clear_flags, has_skybox, caps_0x3a1):
    """GfxClearFlags of ``Camera::ClearNoSkybox`` (0x65fac0 -> 0x65f8e0): Skybox -> 7, or
    6 + (GraphicsCaps+0x3a1 != 0) with a skybox material; SolidColor -> 7; Depth -> 6;
    Nothing -> no clear."""
    m = {'Skybox': 1, 'SolidColor': 2, 'Color': 2, 'Depth': 3, 'Nothing': 4}[clear_flags]
    if m == 1:
        return (6 + (1 if caps_0x3a1 else 0)) if has_skybox else 7
    if m == 2:
        return 7
    if m == 3:
        return 6
    return 0


def uf_projection_far(cam):
    """``Camera::GetProjectionFar`` (0x663750): Camera+0x4ec == 1 (implicit projection)
    -> far clip (+0x440).  Otherwise, when every off-diagonal element of the
    projection's upper 3x3 is zero (loop 0x663780..0x6637ad), the far plane distance
    (m15-m14) / sqrtf(((m7-m6)^2 + (m3-m2)^2) + (m11-m10)^2); else the far clip."""
    import math
    import numpy as np
    f = np.float32
    if cam.get('implicit_projection'):
        return cam.f('far_clip')
    m = [f(v) for v in cam.m('projection_matrix')]
    for r in range(3):
        for c in range(3):
            if r != c and m[r + 4 * c] != f(0.0):
                return cam.f('far_clip')
    a = m[7] - m[6]
    b = m[3] - m[2]
    c = m[11] - m[10]
    d = (a * a + b * b) + c * c
    return f(m[15] - m[14]) / f(math.sqrt(float(d)))


def scene_cam_target_texture(scene):
    cams = [c for c in scene['cameras'] if c['name'] == 'Main Camera' and c['active_in_hierarchy']
            and c['properties']['enabled']]
    return cams[0]['properties'].get('targetTexture')


def scene_cam_allow_msaa(scene):
    cams = [c for c in scene['cameras'] if c['name'] == 'Main Camera' and c['active_in_hierarchy']]
    return bool(cams[0]['properties'].get('allowMSAA')) if cams else False


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    W = Path(args.window)
    scene = json.loads((W / 'frames' / ('scene_%03d.json' % args.window_index)).read_text(encoding='utf-8'))
    v5 = int(scene.get('schema_version', 0)) >= 5
    if v5:
        from src.render.hs2pipe import scene_v5
        scene, _renders = scene_v5.load(W, args.window_index)
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
    # shadow map of the main directional light (RenderShadowMaps 0x6a18b0): cascade fit,
    # caster VP, unity_LightShadowBias generated; caster membership/order external:capture
    shadow = None
    sm_cap = [d for p in cap_passes if p['label'] == 'shadow_map' for d in p['draws']]
    if sm_cap and active['has_main_light']:
        from src.render.hs2pipe import unity_shadow_pass as usp, unity_shadows as ush
        from src.render.hs2pipe import unity_forward as uf, unity_math as um
        by_mesh_s = {}
        for r in ad.renderers():
            by_mesh_s.setdefault(fr_mesh(r), []).append(r)
        caster_ids = []
        for d in sm_cap:
            ids = [r['id'] for r in by_mesh_s.get(d['mesh'], [])]
            if len(ids) != 1:
                raise fr.FrameGap('shadow caster capture draw %s maps to renderers %r' % (d['mesh'], ids))
            if ids[0] not in caster_ids:
                caster_ids.append(ids[0])
        main = active['lights'][0]
        lp = main['raw']['properties']
        nst = scene['native']['settings']
        graphics = {'max_render_texture_size': nst['graphics_caps_max_render_texture_size_0xf8'],
                    'video_memory_mb': nst['graphics_caps_video_memory_mb_0xd8_f32']}
        seen = set()
        receivers = []
        for d in opaque:
            if d.renderer['id'] not in seen:
                seen.add(d.renderer['id'])
                receivers.append(d.renderer)
        casters = [r for i in caster_ids for r in ad.renderers() if r['id'] == i]
        tgt = cam.get('target')
        sh = usp.directional_shadow(cam, lp, q, graphics, casters, receivers, main['id'],
                                    (tgt['pixel_width'], tgt['pixel_height']))
        res = sh['result']
        cvp, cdev = ush.shadow_caster_view_projection(res)
        ok_, cinv = um.invert_general_3d(res.view)
        # GraphicsCaps+0xd0 bit 7 and +0x3fc are not exported (v6 field request); the
        # branch outcome is the only external:capture input of unity_LightShadowBias.
        cap_bias = None
        soft_inputs = {'light_shadows': {'None': 0, 'Hard': 1, 'Soft': 2}[up.enum_value(lp['shadows'])],
                       'tier_0xb4': bool(nst['graphics_settings_tier_cascaded_shadow_maps']),
                       'build_0x101': bool(nst['build_settings_0x101']), 'build_0x102': bool(nst['build_settings_0x102']),
                       'quality_shadows': {'Disable': 0, 'HardOnly': 1, 'All': 2}[up.enum_value(q['shadows'])],
                       'shadow_distance': q['shadowDistance']}
        cand = {}
        for caps in ((True, False), (False, False)):
            br = uf.soft_normal_bias_branch(caps_0xd0_bit7=caps[0], caps_0x3fc=caps[1], **soft_inputs)
            cand[br] = uf.light_shadow_bias({'shadow_bias': lp['shadowBias'], 'shadow_normal_bias': lp['shadowNormalBias']},
                                            cdev, res.texel_size[0], res.texel_size[1], br)
        c0 = cap.commands[cap.command_at_event(sm_cap[0]['event'])]
        sizes = {(int(v['Width']), int(v['Height'])) for v in c0['state']['rs']['viewports']}
        if sizes != {(sh['size'], sh['size'])}:
            raise fr.FrameGap('shadow map size %d differs from the captured viewport %r' % (sh['size'], sizes))
        shadow = {'result': res, 'size': sh['size'], 'vp': cvp, 'device_projection': cdev, 'inverse_view': cinv,
                  'bias_candidates': cand, 'caster_ids': caster_ids, 'soft_inputs': soft_inputs,
                  'split_distances': sh['split_distances']}
        sc_plan = ucp.shadow_caster_plan(ad, lib, gk, quality, pgaps, caster_ids, cache)
        names = [fr_mesh(g['renderer']) for g in sc_plan]
        if names != [d['mesh'] for d in sm_cap]:
            raise fr.FrameGap('shadow caster plan %r differs from the capture %r' % (names, [d['mesh'] for d in sm_cap]))
        lists['shadow_caster'] = [{'status': 'equal_order_external'} for _ in sc_plan]
        for g, c in zip(sc_plan, sm_cap):
            ci = cap.command_at_event(c['event'])
            plan[ci] = [('shadow_caster', g, c, ci)]
        # the caps bits of the normal-bias branch, from the first caster's captured bias
        c0i = cap.command_at_event(sm_cap[0]['event'])
        pz0, sps0, _ = select_pass_programs(lib, sc_plan[0]['pass'], sc_plan[0]['keywords'],
                                            {'vs': sm_cap[0]['vs'], 'ps': sm_cap[0]['ps']})
        branch = shadow_branch(cap, c0i, sps0, cand)
        shadow['branch'] = branch
        caps_bits = {'caps_0xd0_bit7': branch, 'caps_0x3fc': False} if branch else None
        # screen-space shadow map: ForwardShaderRenderLoop::CollectShadows 0x6c8210 ->
        # ComputeScreenSpaceShadowMap 0x69b950
        ss_cap = [d for p in cap_passes if p['label'] == 'screenspace_shadows' for d in p['draws']]
        if ss_cap and caps_bits is not None:
            ss_shader = lib.copies('Hidden/Internal-ScreenSpaceShadows')
            if len(ss_shader) != 1:
                raise fr.FrameGap('Hidden/Internal-ScreenSpaceShadows: %d installed copies' % len(ss_shader))
            ss_shader = ss_shader[0]
            # soft: same BuildSettings/QualitySettings/caps chain as the bias branch without the
            # tier byte (0x69bc69..0x69bce4); force_inv: Camera+0x4ec == 0 (projection set by
            # Camera::SetProjectionMatrix 0x66bc84; ResetProjectionMatrix writes 1, 0x6690f0)
            # and the projection's row 3 is not (0,0,0,1) (0x69bcec..0x69bd2e)
            si = dict(soft_inputs)
            si.pop('tier_0xb4')
            soft = (si['light_shadows'] == 2 and caps_bits['caps_0xd0_bit7'] and not caps_bits['caps_0x3fc']
                    and si['build_0x101'] and si['build_0x102'] and si['quality_shadows'] >= 2
                    and F32(si['shadow_distance']) > F32(0.0))
            pm = cam.m('projection_matrix')
            explicit = not cam.get('implicit_projection')
            force_inv = explicit and not (pm[3] == 0 and pm[7] == 0 and pm[11] == 0 and pm[15] == 1)

            def tag_index(value):
                for i, sub in enumerate(ss_shader.subshaders):
                    if sub.tags.get('ShadowmapFilter') == value:
                        return i
                return -1
            idx = 0
            if soft:
                idx = tag_index('PCF_SOFT_FORCE_INV_PROJECTION_IN_PS') if force_inv else -1
                if idx == -1:
                    idx = tag_index('PCF_SOFT')
            elif force_inv:
                idx = tag_index('HardShadow_FORCE_INV_PROJECTION_IN_PS')
            idx = idx if idx > 0 else 0                         # cmovg 0x69beb3
            ss_pass = ss_shader.subshaders[idx].passes[0]        # SetPassSlow(pass 0, subshader idx)
            cps = uf.cascade_shader_params([res.world_to_shadow] + [ush.unused_cascade_world_to_shadow(True)] * 3,
                                           int(q['shadowCascades']), sh['split_distances'],
                                           [[F32(0.0)] * 4] * 4)
            kw = set(gk)
            if cps['keyword_SHADOWS_SINGLE_CASCADE']:
                kw.add('SHADOWS_SINGLE_CASCADE')
            n = sh['size']
            ortho = um.set_ortho(0.0, 1.0, 0.0, 1.0, -1.0, 100.0)        # LoadFullScreenOrthoMatrix(-1, 100)
            odev = um.calculate_device_projection(ortho, False, fv.setup.invert_y, True)
            vals = dict(fv.camera)
            vals.update(fv.light)
            ident = um.identity()
            vals.update({'unity_ObjectToWorld': ident, 'unity_WorldToObject': ident, 'unity_MatrixV': ident,
                         'unity_MatrixInvV': ident, 'glstate_matrix_projection': odev,
                         'unity_MatrixVP': um.multiply(odev, ident),
                         'unity_WorldToShadow': cps['unity_WorldToShadow'],
                         '_LightSplitsNear': cps['_LightSplitsNear'], '_LightSplitsFar': cps['_LightSplitsFar'],
                         '_ShadowMapTexture_TexelSize': [F32(1.0) / F32(n), F32(1.0) / F32(n), F32(n), F32(n)]})
            if fv.global_sheet is not None:
                vals['_property_fallback'] = {'global': fv.global_sheet, 'builtin': FV.BUILTIN_SET}
            ss_notes = ['screen-space shadows: subshader %d (%s), soft=%s force_inv_projection=%s'
                        % (idx, ss_shader.subshaders[idx].tags.get('ShadowmapFilter'), soft, force_inv),
                        'GraphicsCaps+0xd0 bit 7 / +0x3fc from the caster bias branch (external:capture)',
                        'quad vertex data (rays, 0x69c0d0) external:capture (IA)']
            ci = cap.command_at_event(ss_cap[0]['event'])
            tgt = cam.get('target')
            plan[ci] = [('screenspace_shadows', {
                'renderer': {'id': None, 'name': 'ScreenSpaceShadows', 'kind': 'FullScreenQuad', 'raw': {}},
                'material': {'shader': ss_shader.name, 'render_queue': 2000}, 'shader': ss_shader.name,
                'pass': ss_pass, 'keywords': frozenset(kw), 'values': vals, 'notes': ss_notes,
                'viewport': [{'TopLeftX': 0.0, 'TopLeftY': 0.0, 'Width': float(tgt['pixel_width']),
                              'Height': float(tgt['pixel_height']), 'MinDepth': 0.0, 'MaxDepth': 1.0}]},
                ss_cap[0], ci)]
            lists['screenspace_shadows'] = [{'status': 'equal'}]
    # skybox: drawn after the opaque queue (DoRenderLoop 0x6b9ca0 RenderSkybox) with the
    # RenderSettings.skybox material, pass 0 of its active subshader
    sky_cap = [d for d in fwd[0]['draws'] if d['mesh'] == '']
    rs_sky = scene['statics']['UnityEngine.RenderSettings'].get('skybox')
    if sky_cap and rs_sky and cam.get('clear_flags') == 'Skybox':
        sm = [m for m in scene['materials'] if m['id'] == rs_sky['ref_id']][0]
        sh, sub = lib.copies(sm['shader'])[0].resolve(tier=None)
        ci = cap.command_at_event(sky_cap[0]['event'])
        plan[ci] = [('skybox', {'renderer': {'id': None, 'name': 'Skybox', 'kind': 'Skybox', 'raw': {}},
                                'material': sm, 'shader': sm['shader'], 'pass': sub.passes[0],
                                'keywords': frozenset(set(gk) | set(sm.get('shader_keywords') or []))}, sky_cap[0], ci)]
    # motion vectors: object draws after the camera-motion quad
    mv_all = [d for p in cap_passes if p['label'] == 'motion_vectors' for d in p['draws']]
    mv_cap = mv_all[1:]
    # camera-motion quad (DoMotionVectorRenderLoop 0x6c9b60): Hidden/Internal-MotionVectors
    # pass r12+1, r12 = 1 when Camera+0x558 (current render target) is null or its
    # antiAliasing (RenderTexture+0xd0) > 1 (0x6c9d69..0x6c9d88); material SetMatrix of
    # _NonJitteredVP/_PreviousVP (0x6c9ebb..0x6c9ed7); LoadFullScreenOrthoMatrix(0,
    # Camera::GetProjectionFar) (0x6c9e78 -> 0x68fc70: SetOrtho(0,1,0,1,near,far) ->
    # SetProjectionMatrix, SetViewMatrix(identity)); GfxDevice+0x21e8 == 0 (no single-pass
    # stereo) -> RenderFullScreenQuad 0x6ce370 with eye 2, whose mono path (0x6ce4a9) sets
    # no matrices; the world matrix is not set by either
    if mv_all and cam.get('render_into_texture'):
        mvsh = lib.copies('Hidden/Internal-MotionVectors')
        if len(mvsh) != 1:
            raise fr.FrameGap('Hidden/Internal-MotionVectors: %d installed copies' % len(mvsh))
        mvsh, mvsub = mvsh[0].resolve(tier=None)
        qaa = int(q.get('antiAliasing', 0))
        target_aa = qaa if (qaa > 1 and scene_cam_allow_msaa(scene)) else 1
        r12 = 1 if target_aa > 1 else 0
        from src.render.hs2pipe import unity_math as um
        far = uf_projection_far(cam)
        ortho = um.set_ortho(0.0, 1.0, 0.0, 1.0, 0.0, far)
        odev = um.calculate_device_projection(ortho, False, fv.setup.invert_y, True)
        ident = um.identity()
        vals = dict(fv.camera)
        vals.update(fv.light)
        vals.update({'unity_MatrixV': ident,
                     'unity_MatrixInvV': ident, 'glstate_matrix_projection': odev,
                     'unity_MatrixVP': um.multiply(odev, ident),
                     '_NonJitteredVP': list(fv.setup.non_jittered_vp),
                     '_PreviousVP': cam.m('previous_view_projection')})
        if fv.global_sheet is not None:
            vals['_property_fallback'] = {'global': fv.global_sheet, 'builtin': FV.BUILTIN_SET}
        ci = cap.command_at_event(mv_all[0]['event'])
        plan[ci] = [('motion_vectors_camera', {
            'renderer': {'id': None, 'name': 'CameraMotionQuad', 'kind': 'FullScreenQuad', 'raw': {}},
            'material': {'shader': mvsh.name, 'render_queue': 2000}, 'shader': mvsh.name,
            'pass': mvsub.passes[r12 + 1], 'keywords': frozenset(gk), 'values': vals,
            'notes': ['camera motion quad: pass %d; Camera+0x558 is the camera intermediate RT, antiAliasing '
                      'from QualitySettings.antiAliasing/allowMSAA (creation not transcribed: gap '
                      'camera_intermediate_rt_aa)' % (r12 + 1),
                      'quad vertex data (RenderFullScreenQuad rays) external:capture (IA)']}, mv_all[0], ci)]
        lists['motion_vectors_camera'] = [{'status': 'equal'}]
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
    # camera / pass clears (GraphicsHelper::Clear 0x70cf60 -> GfxDevice::Clear): the capture
    # clear commands in the window before the first action of the pass are one Unity Clear
    # call; their views are the targets bound by RenderTexture::SetActive (allocation
    # external:capture)
    clear_plan, clear_rows = {}, []
    rs_ = scene['statics']['UnityEngine.RenderSettings']
    clear_rules = [('depth_texture_prepass', 'Camera::RenderDepthTexture 0x6674bb', 7, (1.0, 1.0, 1.0, 1.0)),
                   ('depth_normals', 'Camera::RenderShaderReplacementPass 0x668146 (colour 0x667227)', 7,
                    (0.5, 0.5, 1.0, 1.0)),
                   ('shadow_map', 'RenderShadowMaps directional 0x6a2847', 7, (1.0, 1.0, 1.0, 1.0)),
                   ('screenspace_shadows', 'ComputeScreenSpaceShadowMap 0x69bbaa (r12 = -1 -> flags 1)', 1,
                    (1.0, 0.0, 1.0, 0.0)),
                   ('forward_scene', 'Camera::ClearNoSkybox 0x65fac0 -> 0x65f8e0', None, None)]
    order = sorted(cap_passes, key=lambda p: p['events'][0])
    for label, src_, flags, color in clear_rules:
        idx_ = [k for k, p in enumerate(order) if p['label'] == label]
        if not idx_:
            continue
        k = idx_[0]
        lo = order[k - 1]['events'][-1] if k > 0 else -1
        hi = order[k]['events'][0]
        win = [i for i, c in enumerate(cap.commands) if c['op'].startswith('clear') and lo < cap.event(i) < hi]
        if not win:
            continue
        rtvs = [cap.commands[i]['view'] for i in win if cap.commands[i]['op'] == 'clear_rtv']
        dsvs = [cap.commands[i]['view'] for i in win if cap.commands[i]['op'] == 'clear_dsv']
        note = None
        if flags is None:
            # GraphicsCaps+0x3a1 is not exported (v6 field request): the presence of a colour
            # clear in the capture decides it (external:capture, one boolean)
            has_sky = rs_.get('skybox') is not None
            caps3a1 = bool(rtvs)
            flags = camera_clear_flags(cam.get('clear_flags'), has_sky, caps3a1)
            note = 'GraphicsCaps+0x3a1 external:capture (%s)' % caps3a1
            props = [c for c in scene['cameras'] if c['name'] == 'Main Camera' and c['active_in_hierarchy']
                     and c['properties']['enabled']][0]['properties']
            from src.render.hs2pipe import unity_forward as uf_
            bg = [up.F(x) for x in props['backgroundColor']]
            color = uf_.gamma_to_linear_color(bg)        # GammaToLinearSpace 0x65fbad (Linear space)
        gen = d3d11_clears(flags, color, 1.0, 0, rtvs, dsvs[0] if dsvs else None)
        if len(gen) != len(win):
            raise fr.FrameGap('%s clear: %d generated vs %d captured commands' % (label, len(gen), len(win)))
        for (op_, args_), ci in zip(gen, win):
            c = cap.commands[ci]
            eq = c['op'] == op_ and all(c.get(kk) == vv for kk, vv in args_.items())
            clear_plan[ci] = (op_, args_, src_, note)
            clear_rows.append({'capture_command': ci, 'pass': label, 'op': op_, 'equal': eq, 'source': src_,
                               'generated': args_, 'captured': {kk: c.get(kk) for kk in args_}})

    # GrabPass copies of the transparent queue (hs2pipe.liquid: GrabPasses::ApplyGrabPass
    # 0x859da0 -> RenderTexture::GrabPixels -> GrabIntoRenderTexture): the active colour
    # target is copied whole into the grab texture when the object's grab pass is applied.
    # The grab texture is the 'GrabPass Temp' pool temporary (0x85b500; allocation
    # external:capture).
    grab_plan = {}
    from src.render.hs2pipe import liquid as liq
    for p in cap_passes:
        if p['label'] != 'grabpass_copy' or p['pass_index'] < fwd[0]['pass_index']:
            continue
        for e in p['events']:
            ci = cap.command_at_event(e)
            c = cap.commands[ci]
            if c['op'] != 'copy_subresource_region':
                continue
            nxt = [d for d in t_cap if d['event'] > e]
            if not nxt:
                raise fr.FrameGap('grab copy at event %d precedes no transparent draw' % e)
            gi = [k for k, d in enumerate(t_cap) if d['event'] == nxt[0]['event']][0]

            def has_grab(gd):
                _, sub_ = lib.copies(gd.material['shader'])[0].resolve(tier=None)
                return any(pz.type == 'grab' for pz in sub_.passes)
            # the grab runs before the still-pending ForwardBase draw of the preceding object
            # (BatchRenderer defers it; hs2pipe.liquid): the owner is the next draw, or the one
            # after a single pending ForwardBase draw
            owner = [k for k in (gi, gi + 1) if k < len(t_draws) and has_grab(t_draws[k])]
            if not owner or (owner[0] == gi + 1 and t_draws[gi].light_mode != 'FORWARDBASE'):
                raise fr.FrameGap('grab copy at event %d: no GrabPass owner after the pending draw' % e)
            g = t_draws[owner[0]]
            nst = cap.commands[cap.command_at_event(nxt[0]['event'])]['state']
            src = cap.obj(nst['om']['render_targets'][0]['view'])['resource']
            tgt_ = cam.get('target')
            grab_plan[ci] = {'src': src, 'dst': c['dst'], 'renderer': g.renderer['name'], 'shader': g.material['shader'],
                             'box': {'left': 0, 'top': 0, 'front': 0, 'right': int(tgt_['pixel_width']),
                                     'bottom': int(tgt_['pixel_height']), 'back': 1}}
    # main camera OnRenderImage chain (component order: Main Camera components with an
    # OnRenderImage, capture_loop.ON_RENDER_IMAGE), after PPv2's final pass
    chain = None
    if args.shader_manifest:
        from src.render.hs2pipe import capture_loop as cl, ctaa as ctaa_mod
        from src.render.hs2pipe.plugin_common import Op, UnityRenderTexture
        mpath = [c['path'] for c in scene['cameras'] if c['name'] == 'Main Camera' and c['active_in_hierarchy']
                 and c['properties']['enabled']]
        if len(mpath) != 1:
            raise fr.FrameGap('expected one enabled Main Camera')
        mcomps = sorted([c for c in scene['effect_components'] if c.get('path') == mpath[0]],
                        key=lambda c: c['component_index'])
        effects = [c for c in mcomps if c['enabled'] and c['type'].split(',')[0] in cl.ON_RENDER_IMAGE]
        final_idx = max(p['pass_index'] for p in cap_passes if p['label'] == 'post_final_pass')
        chain_passes = [p for p in cap_passes if p['pass_index'] > final_idx and p['label'] in
                        ('blit_copy', 'temporal_aa_ctaa')]
        chain_cis = [cap.command_at_event(d['event']) for p in chain_passes for d in p['draws']]
        tex_by_id = {t['id']: t for t in scene['textures']}
        tgt = cam.get('target')
        hdr = UnityRenderTexture('ImageEffects source (camera HDR intermediate)', tgt['pixel_width'],
                                 tgt['pixel_height'], 0, 'ARGBHalf', 'Default', filter_mode='Bilinear',
                                 wrap_mode='Clamp')
        ops_chain = []
        rts = {}
        for i, comp in enumerate(effects):
            t = comp['type'].split(',')[0]
            f = {k.split('.', 1)[1]: v for k, v in comp['fields'].items()}
            last = i == len(effects) - 1
            if t == 'CharaCustom.CustomRender':
                rtc = f.get('<rtCamera>k__BackingField')
                if rtc:
                    rts['rtCamera'] = scene_rt(tex_by_id[rtc['ref_id']], 'CustomRender.rtCamera')
                if f['update']:
                    ops = ([Op('blit', 'plugin', 'CharaCustom.CustomRender.OnRenderImage',
                               {'source': 'source', 'dest': 'rtCamera', 'material': None, 'pass': -1})] if rtc else [])
                    ops.append(Op('blit', 'plugin', 'CharaCustom.CustomRender.OnRenderImage',
                                  {'source': 'source', 'dest': 'destination', 'material': None, 'pass': -1}))
                else:
                    ops = [Op('blit', 'plugin', 'CharaCustom.CustomRender.OnRenderImage',
                              {'source': 'rtCamera', 'dest': 'destination', 'material': None, 'pass': -1})]
            elif t == 'CTAA_PC':
                st_ = ctaa_mod.CTAAState.from_audit_fields([{'field': k, 'value': v, 'read_succeeded': True}
                                                            for k, v in f.items()])
                for attr, name in (('rt_accum0', 'rtAccum0'), ('rt_accum1', 'rtAccum1'), ('txaa_out', 'txaaOut'),
                                   ('after_pre_enhance', 'afterPreEnhace')):
                    ref = f.get(name)
                    if ref:
                        rt_ = scene_rt(tex_by_id[ref['ref_id']], name)
                        setattr(st_, attr, rt_)
                        rts[name] = rt_
                ops = ctaa_mod.on_render_image(st_, camera_orthographic=bool(cam.get('orthographic')), source=hdr,
                                               screen_width=tgt['pixel_width'], screen_height=tgt['pixel_height'])
                if any(o.kind in ('create_persistent', 'destroy') for o in ops):
                    raise fr.FrameGap('CTAA recreates its history textures in this frame (not handled)')
            else:
                raise fr.FrameGap('OnRenderImage component %s not lowered' % t)
            ops_chain.append({'component': t, 'ops': ops, 'last': last})
        nblits = sum(1 for c in ops_chain for o in c['ops'] if o.kind == 'blit')
        if nblits != len(chain_cis):
            raise fr.FrameGap('OnRenderImage chain: %d generated blits vs %d captured' % (nblits, len(chain_cis)))
        chain = {'components': ops_chain, 'cis': chain_cis, 'rts': rts, 'source_rt': hdr,
                 'manifests': args.shader_manifest}

    cb_names, cb_conflicts = fr.capture_cbuffer_names(cap, lib)
    skin = None
    if args.skin_geometry:
        from src.render.hs2pipe import frame_skin
        skin = frame_skin.head_skin_uploads(args.xml, W, args.skin_geometry, args.fxc, out / 'skin_work',
                                            scene['frame_count'])
    pp_inputs = None
    if args.ppv2_scene_state:
        def rd(k):
            if v5:
                return scene_v5.load(W, args.window_index - k)[0]
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
        skin_rows = []
        skin_cmds = {}
        if generate and skin is not None:
            chunk_res = {(e.get('provenance') or {}).get('chunk'): e for e in b.program['resources']}
            chunk_cmd = {(c.get('provenance') or {}).get('chunk'): i for i, c in enumerate(cap.commands)}
            for chunk, u in sorted(skin.items()):
                row = {k: v for k, v in u.items() if k != 'bytes'}
                row['chunk'] = chunk
                if chunk in chunk_cmd:
                    skin_cmds[chunk_cmd[chunk]] = u
                    row['emitted_as'] = 'map_write (in-frame upload)'
                elif chunk in chunk_res:
                    e = chunk_res[chunk]
                    if e['kind'] != 'buffer' or e['desc']['ByteWidth'] != len(u['bytes']):
                        raise fr.FrameGap('skin upload %d does not fit resource %d' % (chunk, e['id']))
                    e['initial'] = [{'subresource': 0, 'blob': b.blob(u['bytes']), 'row_pitch': len(u['bytes']),
                                     'slice_pitch': len(u['bytes'])}]
                    e['provenance'] = {'generator': 'hs2pipe.frame_skin', 'role': u['role'], 'frame': u['frame'],
                                       'surface': u['surface'], 'replaces': 'capture-start buffer contents'}
                    row['emitted_as'] = 'initial contents (state at frame start)'
                else:
                    raise fr.FrameGap('skin upload chunk %d not in the capture program' % chunk)
                skin_rows.append(row)
        em = fr.FrameEmitter(b, cap, lib, cbuffer_names=cb_names)
        dev = ds.DeviceRaster(invert_projection=True)
        last_cmd_of_pass = {}
        ppv = PPv2Run(b, em, lib, cap, scene, pp_inputs, fv) if (generate and pp_inputs is not None) else None
        segments = {}
        if ppv is not None:
            ppv.warm_up()
            segments = ppv.segments
        em.seeded_keys = em.seed_frame_start() if generate else []
        cdrv = None
        if generate and chain is not None:
            import collections
            from src.render.hs2pipe.plugin_common import PluginLowering, ShaderManifests
            cdrv = ChainDriver(cap, em, chain['cis'])
            views_of = collections.defaultdict(list)
            for v in cap.program['views']:
                views_of[v['resource']].append(v)
            cis = chain['cis']
            sp_cache = {}

            def rt_of(ci):
                st_ = cap.commands[ci]['state']
                r0 = cap.obj(st_['om']['render_targets'][0]['view'])['resource']
                dsv = cap.obj(st_['om']['depth_stencil'])['resource'] if st_['om']['depth_stencil'] else None
                return r0, dsv
            # engine records: the blit that writes / reads each chain texture in the
            # capture names its resource (allocation external:capture)
            plan_blits = [o for c in chain['components'] for o in c['ops'] if o.kind == 'blit']
            names = {}
            for o, ci in zip(plan_blits, cis):
                st_ = cap.commands[ci]['state']
                names.setdefault(o.args['source'], ('srv', ci, '_MainTex'))
                names.setdefault(o.args['dest'], ('rt', ci, None))
                for tname, h in (o.args.get('textures') or {}).items():
                    names.setdefault(h, ('srv', ci, tname))
            hdr = chain['source_rt']
            for h, (kind, ci, tname) in names.items():
                if h in ('source', 'destination'):
                    continue
                if kind == 'rt':
                    res, dres = rt_of(ci)
                else:
                    res, dres = srv_resource(cap, lib, cap.commands[ci]['state'], tname, sp_cache), None
                rt_ = chain['rts'].get(h)
                if rt_ is None:
                    d = cap.obj(res)['desc']
                    from src.render.hs2pipe.plugin_common import UnityRenderTexture
                    rt_ = UnityRenderTexture(h, int(d['Width']), int(d['Height']), 0,
                                             DXGI_RT_FORMATS[int(d['Format'])], 'Default')
                cdrv.engine[h] = chain_engine_rec(cap, views_of, res, rt_, h, dres)
            # chain scopes: component k's source = previous destination
            src0 = srv_resource(cap, lib, cap.commands[cis[0]]['state'], '_MainTex', sp_cache)
            chain_src = chain_engine_rec(cap, views_of, src0, hdr, 'ImageEffects source')
            first_of = []
            k = 0
            for c in chain['components']:
                first_of.append(k)
                k += sum(1 for o in c['ops'] if o.kind == 'blit')
            scopes = []
            cur = chain_src
            for n, c in enumerate(chain['components']):
                blits = [ci for o, ci in zip(plan_blits, cis)][first_of[n]:first_of[n] + sum(
                    1 for o in c['ops'] if o.kind == 'blit')]
                ci_dest = [ci for o, ci in zip(plan_blits[first_of[n]:], cis[first_of[n]:])
                           if o.args['dest'] == 'destination'][0]
                res, dres = rt_of(ci_dest)
                d = cap.obj(res)['desc']
                from src.render.hs2pipe.plugin_common import UnityRenderTexture
                drt = UnityRenderTexture('%s destination' % c['component'], int(d['Width']), int(d['Height']), 0,
                                         DXGI_RT_FORMATS[int(d['Format'])], 'Default')
                dst = chain_engine_rec(cap, views_of, res, drt, drt.name, dres)
                if c['last']:
                    # the last OnRenderImage destination is the camera target: targetTexture
                    # null -> the screen backbuffer
                    if scene_cam_target_texture(scene) is None and not dst.get('backbuffer'):
                        raise fr.FrameGap('camera renders to the screen but the last chain blit does not')
                    if scene_cam_target_texture(scene) is not None and dst.get('backbuffer'):
                        raise fr.FrameGap('camera has a targetTexture but the last chain blit writes the backbuffer')
                elif dst.get('backbuffer'):
                    raise fr.FrameGap('intermediate chain destination is the backbuffer')
                scopes.append({'source': cur, 'destination': dst})
                cur = dst
            cfwd = fr.UnityForwardLowering(em, lib, cdrv)
            clow = PluginLowering(b, ShaderManifests([Path(m) for m in chain['manifests']]), cfwd, 'Linear')
            cdrv.motion = None
        # GfxDevice world matrix as left by the last generated draw (SetWorldMatrix state);
        # None after any capture-spliced or PPv2 action (its world matrix is not modelled)
        world_state = None
        grab_rows = []
        em.grab_rows = grab_rows
        for i, c in enumerate(cap.commands):
            if cdrv is not None and i in cdrv.handled:
                continue
            if cdrv is not None and i == chain['cis'][0]:
                def _splice(j):
                    ev_ = cap.event(j)
                    lab_ = event_pass.get(ev_, (None, 'non-pass command'))[1]
                    return em.splice(j, 'capture resource update / clear / copy (%s)' % lab_)
                cdrv.splice_fn = _splice
                cdrv.cursor = i
                n0 = len(em.records)
                for c_, sc in zip(chain['components'], scopes):
                    clow.emit(c_['ops'], sc)
                if cdrv.queue:
                    raise fr.FrameGap('OnRenderImage chain: %d captured blits not consumed' % len(cdrv.queue))
                for r_ in em.records[n0:]:
                    r_.pass_label = 'image_effect_chain'
                for ci_ in chain['cis']:
                    ev_ = cap.event(ci_)
                    if ev_ in event_pass:
                        last_cmd_of_pass[event_pass[ev_][0]] = (cdrv.cap_to_gen[ci_], ci_)
                world_state = None
                continue
            if i in skin_cmds:
                u = skin_cmds[i]
                if c['end'] - c['start'] != len(u['bytes']):
                    raise fr.FrameGap('skin upload size differs at capture command %d' % i)
                b.map_write(c['resource'], c['subresource'], c['map_type'], c['map_flags'], c['start'], c['end'],
                            u['bytes'], generator='hs2pipe.frame_skin', role=u['role'], surface=u['surface'],
                            frame=u['frame'], capture_command=i)
                continue
            if generate and i in clear_plan:
                op_, args_, src_, note_ = clear_plan[i]
                prov_ = {'generator': 'hs2pipe camera clear', 'source': src_, 'capture_command': i,
                         'external': ['target allocation (RenderTexture::SetActive binding)'] + ([note_] if note_ else [])}
                if op_ == 'clear_rtv':
                    b.clear_rtv(args_['view'], args_['color'], **prov_)
                else:
                    b.clear_dsv(args_['view'], args_['flags'], args_['depth'], args_['stencil'], **prov_)
                continue
            if generate and i in grab_plan:
                gp_ = grab_plan[i]
                idx = b.copy_subresource_region(gp_['dst'], 0, (0, 0, 0), gp_['src'], 0, src_box=gp_['box'],
                                                generator='hs2pipe GrabPass (liquid rule)',
                                                source='GrabPasses::ApplyGrabPass -> GrabIntoRenderTexture',
                                                object=gp_['renderer'], capture_command=i,
                                                external=['grab texture allocation (GrabPass Temp pool temporary)',
                                                          'active colour target allocation'])
                ev = cap.event(i)
                if ev in event_pass:
                    last_cmd_of_pass[event_pass[ev][0]] = (idx, i)
                grab_rows.append({'capture_command': i, 'renderer': gp_['renderer'], 'shader': gp_['shader'],
                                  'equal': all(c.get(k) == v for k, v in (('src', gp_['src']), ('dst', gp_['dst']),
                                                                          ('src_box', gp_['box']), ('dst_x', 0),
                                                                          ('dst_y', 0), ('dst_z', 0),
                                                                          ('src_subresource', 0),
                                                                          ('dst_subresource', 0)))})
                continue
            if ppv is not None and i in segments:
                segments[i]()
                world_state = None
            if ppv is not None and i in ppv.skip:
                world_state = None
                em.skipped.append((i, 'replaced by generated PPv2'))
                ev = cap.event(i)
                if ev in event_pass and (c['op'] in DRAW_OPS or c['op'] == 'dispatch'):
                    last_cmd_of_pass[event_pass[ev][0]] = (len(b.program['commands']) - 1, i)
                continue
            if generate and i in plan:
                idx = None
                for label, g, cd, ci in plan[i]:
                    pz, sps, why = select_pass_programs(lib, g['pass'], g['keywords'], {'vs': cd['vs'], 'ps': cd['ps']})
                    if 'values' in g:
                        vals, notes = dict(g['values']), list(g.get('notes', []))
                    else:
                        vals, notes = fv.draw_values(g['renderer'], g['material'])
                    if label == 'skybox':
                        from src.render.hs2pipe import unity_forward as uf, unity_math as um
                        sk = uf.skybox_matrices(cam, fv.setup, FV.GRAPHICS_D3D11)
                        ok_, w2o = um.invert_general_3d(sk['unity_ObjectToWorld'])
                        devp = um.calculate_device_projection(sk['projection'], False, fv.setup.invert_y, True)
                        vals.update({'unity_ObjectToWorld': sk['unity_ObjectToWorld'], 'unity_WorldToObject': w2o,
                                     'unity_MatrixVP': sk['unity_MatrixVP'], 'glstate_matrix_projection': devp})
                        notes = list(notes) + ['skybox: GraphicsCaps+0x404 taken as 1.0 (gap skybox_caps_404)']
                    vp_override = sc_override = None
                    devx = dev
                    if label == 'depth_texture':
                        # depth-pass render job 0x6c8840: unity_LightShadowBias = (0,0,0,0)
                        # on the device (0x6c890e..0x6c8923) before the draws
                        vals['unity_LightShadowBias'] = [F32(0.0)] * 4
                    if label == 'shadow_caster':
                        res = shadow['result']
                        vals.update({'unity_MatrixV': list(res.view), 'unity_MatrixInvV': list(shadow['inverse_view']),
                                     'unity_MatrixVP': list(shadow['vp']),
                                     'glstate_matrix_projection': list(shadow['device_projection'])})
                        bias = shadow['bias_candidates']
                        vals['unity_LightShadowBias'] = bias[shadow_branch(cap, ci, sps, bias)]
                        notes = list(notes) + ['shadow normal-bias branch: GraphicsCaps+0xd0 bit 7 / +0x3fc '
                                               'external:capture (outcome of the 0x6a29db branch)']
                        n = shadow['size']
                        vp_override = [{'TopLeftX': 0.0, 'TopLeftY': 0.0, 'Width': float(n), 'Height': float(n),
                                        'MinDepth': 0.0, 'MaxDepth': 1.0}]
                        sc_override = [ds.shadow_scissor_rect(0, 0, n, n)]
                        devx = ds.shadow_caster_device(True)
                    if label == 'screenspace_shadows':
                        vp_override = g['viewport']
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
                    if label == 'motion_vectors_camera' and world_state is not None:
                        # neither LoadFullScreenOrthoMatrix nor the mono RenderFullScreenQuad path
                        # sets the world matrix: the device keeps the last draw's (the skybox's
                        # DeviceMVPMatricesState 0x68a1d0 restores the one before it)
                        from src.render.hs2pipe import unity_math as um
                        ok_, w2o = um.invert_general_3d(world_state)
                        vals['unity_ObjectToWorld'] = list(world_state)
                        vals['unity_WorldToObject'] = w2o
                        notes = list(notes) + ['world matrix: device state left by the previous generated draw']
                    if why:
                        notes = list(notes) + [why]
                    rec = em.emit_draw(label=label, cap_index=ci, stage_programs=sps, pass_record=pz,
                                       light_mode=pz.light_mode, values=vals,
                                       material_floats=material_floats(g['material']) if 'id' in g['material'] else {},
                                       device=devx,
                                       renderer_name=g['renderer']['name'], shader_name=g['shader'],
                                       keywords=g['keywords'], notes=notes, viewports=vp_override,
                                       scissors=sc_override)
                    rec.inputs = vals.get('_inputs', {})
                    idx = rec.command
                    if label != 'skybox' and 'unity_ObjectToWorld' in vals:
                        world_state = list(vals['unity_ObjectToWorld'])
                if idx is None:
                    continue
            else:
                ev = cap.event(i)
                lab = event_pass.get(ev, (None, 'non-pass command'))[1]
                if c['op'] in DRAW_OPS or c['op'] == 'dispatch':
                    world_state = None
                idx = em.splice(i, 'step not generated: ' + lab if c['op'] in DRAW_OPS or c['op'] == 'dispatch'
                                else 'capture resource update / clear / copy (%s)' % lab)
            ev = cap.event(i)
            if ev in event_pass and (c['op'] in DRAW_OPS or c['op'] == 'dispatch' or i in grab_plan):
                last_cmd_of_pass[event_pass[ev][0]] = (idx, i)
        # PPv2 actions replace capture actions one for one, in order
        pp_pairs = {}
        if ppv is not None:
            gen_acts = [r for r in ppv.records.get('opaque', []) + ppv.records.get('post', [])
                        if r['kind'] in ('draw', 'dispatch')]
            cap_acts = sorted(ppv.skip)
            if len(gen_acts) != len(cap_acts):
                raise fr.FrameGap('PPv2: %d generated actions vs %d captured' % (len(gen_acts), len(cap_acts)))
            pp_pairs = dict(zip(cap_acts, gen_acts))
            em.ppv2_pairs = pp_pairs

        def outputs(st, op, objs):
            if op == 'dispatch':
                return [objs(u['view'])['resource'] for u in st['cs']['uavs']]
            t = [objs(r['view'])['resource'] for r in st['om']['render_targets']]
            if st['om']['depth_stencil'] is not None:
                t.append(objs(st['om']['depth_stencil'])['resource'])
            return t
        views = {v['id']: v for v in b.program['views']}
        # read-backs: outputs of the last action of every pass with actions
        checks = []
        for pidx, (gidx, cidx) in sorted(last_cmd_of_pass.items()):
            cmd = cap.commands[cidx]
            ctargets = ([cmd['dst']] if cmd['op'] == 'copy_subresource_region'
                        else outputs(cmd['state'], cmd['op'], cap.obj))
            gtargets = ctargets
            if cidx in pp_pairs:
                rec = pp_pairs[cidx]
                gidx = rec['command']
                g = b.program['commands'][gidx]
                gtargets = outputs(g['state'], g['op'], lambda v: views[v])
                if len(gtargets) != len(ctargets):
                    raise fr.FrameGap('PPv2 action output count differs at capture command %d' % cidx)
            for res, gres in zip(ctargets, gtargets):
                r = cap.obj(res)
                if r['kind'] == 'buffer':
                    continue
                name = 'p%02d_%s_r%d' % (pidx, [p for p in cap_passes if p['pass_index'] == pidx][0]['label'], res)
                b.readback(name, gres if generate else res, subresource=0, after_command=gidx,
                           orientation='d3d_rows_top_down')
                checks.append({'name': name, 'pass_index': pidx, 'resource': res, 'generated_resource': gres,
                               'command': gidx, 'capture_command': cidx})
        b.write()
        em.skin_rows = skin_rows
        return b, em, checks

    gen_b, em, checks = build(out / 'generated', True)
    if not args.no_execute and not args.reuse_oracle:
        build(out / 'oracle', False)
    rows = [diff_draw(r, cap, gen_b) for r in em.records]
    ppv2_rows = [diff_ppv2_action(ci, rec, cap, gen_b, lib) for ci, rec in sorted(getattr(em, 'ppv2_pairs', {}).items())]
    report = {'schema': 'hs2_frame_orchestrator_paired/1', 'capture_program': str(args.capture_program),
              'draw_lists': {k: [r['status'] for r in v] for k, v in lists.items()},
              'generated_draws': len(em.records), 'spliced_commands': len(em.spliced),
              'spliced_by_reason': {}, 'draws': rows, 'gaps': ad.gaps + em.gaps + pgaps, 'motion_vectors': report_mv,
              'head_skin': getattr(em, 'skin_rows', []),
              'grab_copies': getattr(em, 'grab_rows', []),
              'clears': clear_rows,
              'ppv2_actions': ppv2_rows,
              'ppv2_summary': ppv2_summary(ppv2_rows),
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
    ap.add_argument('--shader-manifest', action='append', default=[],
                    help='plugin shader manifests (CTAA, CustomRender blit): enables the OnRenderImage chain')
    ap.add_argument('--no-execute', action='store_true')
    ap.add_argument('--skin-geometry', help='same-session geometry export (ancestors above ChaControl)')
    ap.add_argument('--fxc', default='C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/fxc.exe')
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
