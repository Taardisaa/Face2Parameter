"""Byte-diff the plugin effect generators against the retry capture (frame 17675).

CPU only.  Inputs are existing evidence; nothing launches the game.

* ``--frame-graph``: ``frame_graph.py`` output (passes/actions/resources/summary).
* ``--captured-cbuffers``: ``compare_generated_cbuffers.py extract`` output
  (per-event bytes of every compiled cbuffer variable).
* ``--runtime-fields``: ``RenderAudit`` component fields
  (``runtime_audit_hook_v3/runtime_fields_completion.json``).  This is **another
  frame of another process**; every value taken from it is labelled so in the
  report and listed as a paired-capture requirement.
* ``--shader-manifest``: installed shader exports (SeparableSSS, CTAA, Liquid...).

Checks (all exact, no tolerance):
  1. SeparableSSS ``$Globals`` of the 64 blur draws; CTAA Enhance/Resolve ``$Globals``.
  2. CTAA jitter: jittered ``unity_MatrixVP`` regenerated from the same render's
     captured ``_NonJitteredVP`` and the generated jitter, per iteration.
  3. Plugin render-texture D3D11 descriptors and view formats.
  4. Plugin pass order per segment, CTAA ping-pong targets and first-frame init.
  5. Resource identity across iterations (pool reuse vs. fresh textures).
  6. GrabPass placement in the transparent loop.
  7. Graphics.Blit ortho constants of plugin blits.
  8. Sampler-state keys of plugin render textures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import capture_loop, ctaa, dhh_sss, liquid  # noqa: E402
from src.render.hs2pipe.plugin_common import (F32, ShaderManifests, UnityRenderTexture, RenderTextureAllocator,  # noqa: E402
                                              texture_slots, unity_sampler_key, jsonable)

SCHEMA = 1
MAIN_W, MAIN_H = 1086, 1448


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Capture:
    def __init__(self, fg_dir, cb_path):
        fg = Path(fg_dir)
        self.passes = load(fg / 'passes.json')
        self.summary = load(fg / 'summary.json')
        self.resources = load(fg / 'resources.json')
        self.actions = {a['event_id']: a for a in load(fg / 'actions.json')}
        self.cb = {d['event_id']: d for d in load(cb_path)['draws']}
        self.segments = [s['passes'] for s in self.summary['segments']]

    def var(self, event, stage, name):
        for cb in self.cb[event]['stages'].get(stage, {}).get('cbuffers', []):
            v = cb.get('variables', {}).get(name)
            if v:
                return v
        return None

    def globals(self, event, stage='PS'):
        for cb in self.cb[event]['stages'].get(stage, {}).get('cbuffers', []):
            if cb['name'] == '$Globals':
                return cb['variables']
        return {}

    def seg_passes(self, s):
        return [self.passes[i] for i in self.segments[s]]

    def color_target(self, p):
        return [t['resource'] for t in p['targets'] if not t.get('depth')]


def runtime_component(fields_json, type_name, aux=False):
    rows = fields_json['renderer_runtime_detail']['auxiliary_effects' if aux else 'camera_effects']
    for c in rows:
        if c['type'] == type_name:
            return c['fields']
    raise KeyError(type_name)


# ------------------------------------------------------------------ plan
def build_plan(rf, manifests):
    sss = dhh_sss.SSSState.from_audit_fields(runtime_component(rf, 'DHH.SSS'))
    conv = dhh_sss.ConvolutionState.from_audit_fields(runtime_component(rf, 'DHH.SSS_convolution', aux=True))
    ctaa_state = ctaa.CTAAState.from_audit_fields(runtime_component(rf, 'CTAA_PC'))
    # Persistent CTAA textures as they exist before the capture (normal 3200x1800 frame).
    for attr, field in (('rt_accum0', 'rtAccum0'), ('rt_accum1', 'rtAccum1'), ('txaa_out', 'txaaOut'),
                        ('after_pre_enhance', 'afterPreEnhace')):
        v = {f['field']: f.get('value') for f in runtime_component(rf, 'CTAA_PC')}[field]
        setattr(ctaa_state, attr, UnityRenderTexture(field, v['width'], v['height'], 0, v['format'], 'Default',
                                                     filter_mode=v['filter'], wrap_mode=v['wrap_u'], lifetime='persistent'))
    detail = rf['renderer_runtime_detail']
    cam_rt = rf['camera']
    quality = dhh_sss.QualityState(anti_aliasing=0, pixel_light_count=int(detail['pixel_light_count']), shadows='All',
                                   color_space=detail['quality_color_space'])
    comps = tuple((c['type'], bool(c['enabled'])) for c in detail['camera_effects'])
    cr = {f['field']: f.get('value') for f in runtime_component(rf, 'CharaCustom.CustomRender')}
    rtc = cr.get('<rtCamera>k__BackingField')
    cam = capture_loop.MainCamera(field_of_view=float(cam_rt['fov']), near_clip=float(cam_rt['near']), far_clip=float(cam_rt['far']),
                                  allow_hdr=bool(cam_rt['allow_hdr']), allow_msaa=False, components=comps,
                                  screen_width=3200, screen_height=1800, custom_render_update=bool(cr['update']),
                                  custom_render_rt=(rtc['width'], rtc['height']) if rtc else None)
    request = capture_loop.CaptureRequest(path='C:/capture.png', target=(0.0, 14.0, 0.0), ortho_size=1.0,
                                          yaw=0.0, width=MAIN_W, height=MAIN_H)
    # Unity's HDR camera intermediate handed to OnRenderImage (unity_forward descriptor;
    # captured TempBuffer 1913 / ImageEffects Temp are ARGBHalf, Bilinear).
    source = UnityRenderTexture('ImageEffects Temp', MAIN_W, MAIN_H, 0, 'ARGBHalf', 'Default', filter_mode='Bilinear',
                                wrap_mode='Clamp')
    from src.render.hs2pipe import unity_math as um
    base_projection = um.set_perspective(cam.field_of_view, float(capture_loop.capture_aspect(MAIN_W, MAIN_H)),
                                         cam.near_clip, cam.far_clip)
    m = manifests
    objects = [
        liquid.ForwardObject('o_body liquid', 'AIT/Liquid', 2999, liquid.forward_passes(m, 'AIT/Liquid')),
        liquid.ForwardObject('o_eyelashes', 'AIT/eyelashes', 3000, liquid.forward_passes(m, 'AIT/eyelashes')),
        liquid.ForwardObject('o_namida', 'AIT/main namida', 4001, liquid.forward_passes(m, 'AIT/main namida')),
        liquid.ForwardObject('particles', 'Legacy Shaders/Particles/Additive', 3000,
                             [{'pass': 0, 'kind': 'forward_base', 'grab_name': None}]),
    ]
    plan = capture_loop.plan_capture(request, cam, sss=sss, conv=conv, quality=quality, ctaa_state=ctaa_state,
                                     character_layer_mask=int(cam_rt['culling_mask']), base_projection=base_projection,
                                     transparent_objects=objects, camera_source_rt=source)
    return plan, sss, conv, cam, quality, source


# ------------------------------------------------------------------ checks
def check_blur_cbuffers(cap, manifests, plan, sss):
    rows, ok = [], True
    lt = dhh_sss.lighting_rt('LightingTexture', MAIN_W // 2, MAIN_H // 2,
                             dhh_sss.CameraState(MAIN_W, MAIN_H, 23, 0.75, 0.08, 1000, False, 0, True, False, 'SolidColor', (0, 0, 0, 1), 0),
                             dhh_sss.QualityState(0, 16, 'All', 'Linear'))
    for s in range(16):
        rec = plan.renders[s]
        render = [o for o in rec.sss_ops if o.kind == 'render_camera' and o.args['camera'] == 'SSS Lighting Pass Camera'][0]
        blits = [c for c in render.args['image_effects_opaque'] if c['kind'] == 'blit' and c['args']['material'] == 'Hidden/SeparableSSS']
        events = [p['first_event'] for p in cap.seg_passes(s) if p['label'] == 'sss_separable_blur']
        if len(events) != len(blits):
            ok = False
            rows.append({'segment': s, 'error': 'blur count %d generated vs %d captured' % (len(blits), len(events))})
            continue
        for e, b in zip(events, blits):
            tos = np.frombuffer(bytes.fromhex(b['args']['set_vector']['_TexelOffsetScale']['f32_hex']), np.float32)
            gen = dhh_sss.blur_cbuffer(manifests, sss, source=lt, texel_offset_scale=tos, global_keywords={'SSS_PROFILES'})
            got = cap.globals(e)
            diff = {k: (v['hex'], got.get(k, {}).get('hex')) for k, v in gen['variables'].items() if got.get(k, {}).get('hex') != v['hex']}
            extra = sorted(set(got) - set(gen['variables']))
            captured_kw = cap.cb[e]['stages']['PS']['programs'][0]['keywords'] if cap.cb[e]['stages']['PS']['programs'] else None
            kw_ok = sorted(captured_kw or []) == sorted(gen['keywords'])
            good = not diff and not extra and kw_ok
            ok &= good
            rows.append({'segment': s, 'event': e, 'axis': b['args']['axis'], 'variables': len(gen['variables']),
                         'keywords': gen['keywords'], 'keywords_match': kw_ok, 'mismatch': diff, 'captured_only': extra,
                         'pass': good})
    return {'pass': ok, 'draws': len(rows), 'rows': rows}


def check_ctaa(cap, manifests, plan, source):
    rows, ok = [], True
    for s in range(16):
        rec = plan.renders[s]
        ops = rec.image_effect_chain[-1]['ops']
        mats = {o.args['shader']: o.args['properties'] for o in ops if o.kind == 'material'}
        ev = {p['pixel_and_compute_shader_debug_names'][0]: p['first_event'] for p in cap.seg_passes(s)
              if p['label'] == 'temporal_aa_ctaa'}
        for shader, fn in (('Hidden/CTAA_Enhance_PC', ctaa.enhance_cbuffer), ('Hidden/CTAA_PC', ctaa.resolve_cbuffer)):
            gen = fn(manifests, mats[shader], source)
            got = cap.globals(ev[shader])
            diff = {k: (v['hex'], got.get(k, {}).get('hex')) for k, v in gen.items() if got.get(k, {}).get('hex') != v['hex']}
            good = not diff and not (set(got) - set(gen))
            ok &= good
            rows.append({'segment': s, 'event': ev[shader], 'shader': shader, 'mismatch': diff, 'pass': good,
                         'producers': {k: v.get('producers') for k, v in got.items()}})
    return {'pass': ok, 'rows': rows}


def check_jitter(cap, plan):
    rows, ok = [], True
    for s in range(16):
        pc = plan.renders[s].pre_cull
        nj = vp = None
        for p in cap.seg_passes(s):
            for e in p['events']:
                if e not in cap.cb:
                    continue
                if nj is None and p['label'] == 'motion_vectors':
                    v = cap.var(e, 'VS', '_NonJitteredVP') or cap.var(e, 'PS', '_NonJitteredVP')
                    if v:
                        nj = np.frombuffer(bytes.fromhex(v['hex']), np.float32)
                if vp is None and p['label'] == 'forward_scene':
                    v = cap.var(e, 'VS', 'unity_MatrixVP')
                    if v:
                        vp = np.frombuffer(bytes.fromhex(v['hex']), np.float32)
        gen = ctaa.jittered_gpu_view_projection(nj, float(pc['m02_add']), float(pc['m12_add']))
        good = gen.tobytes() == vp.tobytes()
        ok &= good
        rows.append({'segment': s, 'jitter_index': pc['jitter_index'], 'm02_add': float(pc['m02_add']),
                     'm12_add': float(pc['m12_add']), 'generated_vp_hex': gen.tobytes().hex(),
                     'captured_vp_hex': vp.tobytes().hex(), 'differing_floats': int(np.sum(gen.view(np.uint32) != vp.view(np.uint32))),
                     'pass': good})
    return {'pass': ok, 'rows': rows, 'model': 'VP_j = NJ + (-NJ[3]) * m02/(-m12) per column; Mono double intermediates'}


def segment_map(cap, s):
    """Capture resources of the plugin textures in segment ``s`` (by pass role)."""
    ps = cap.seg_passes(s)
    out = {}
    labels = [p['label'] for p in ps]
    i = labels.index('sss_profile')
    out['SSS_ProfileTex'] = ps[i + 1]['targets']
    li = labels.index('sss_lighting')
    out['LightingTexture'] = ps[li]['targets']
    blur = [p for p in ps if p['label'] == 'sss_separable_blur']
    out['T1'] = blur[0]['targets']
    out['T2'] = blur[1]['targets']
    out['SSSLightingTextureBlurred'] = blur[-1]['targets']
    grabs = [p for p in ps if p['label'] == 'grabpass_copy']
    out['grab:_GrabTexture'] = grabs[0]['targets']
    ct = [p for p in ps if p['label'] == 'temporal_aa_ctaa']
    out['afterPreEnhace'] = ct[0]['targets']
    k = ps.index(ct[1])
    out['ctaa_write'] = ct[1]['targets']
    out['txaaOut'] = ps[k + 1]['targets']
    out['capture rt'] = ps[-1]['targets']
    out['ctaa_accum_read'] = [r for r in cap.actions[ct[1]['first_event']]['shaders']['PS']['srvs'] if r['name'] == '_Accum'][0]['resource']
    if s == 0:
        out['rtAccum0'] = ps[ps.index(ct[0]) + 1]['targets']
    return out


def check_descriptors(cap, plan, sss, quality):
    w, h = MAIN_W // 2, MAIN_H // 2
    carrier = dhh_sss.CameraState(MAIN_W, MAIN_H, 23, 0.75, 0.08, 1000, False, 0, True, False, 'SolidColor', (0, 0, 0, 1), 0)
    gen = {
        'SSS_ProfileTex': dhh_sss.profile_rt(w, h),
        'LightingTexture': dhh_sss.lighting_rt('LightingTexture', w, h, carrier, quality),
        'SSSLightingTextureBlurred': dhh_sss.lighting_rt('SSSLightingTextureBlurred', w, h, carrier, quality),
        'T1': UnityRenderTexture('T1', w, h, 0, 'ARGBFloat', 'Linear'),
        'T2': UnityRenderTexture('T2', w, h, 0, 'ARGBFloat', 'Linear'),
        'grab:_GrabTexture': liquid.grab_rt(UnityRenderTexture('cam', MAIN_W, MAIN_H, 24, 'ARGBHalf', 'Default')),
        'afterPreEnhace': ctaa.history_rt('afterPreEnhace', UnityRenderTexture('src', MAIN_W, MAIN_H, 0, 'ARGBHalf', 'Default'), 'Bilinear'),
        'rtAccum0': ctaa.history_rt('rtAccum0', UnityRenderTexture('src', MAIN_W, MAIN_H, 0, 'ARGBHalf', 'Default'), 'Bilinear'),
        'ctaa_write': ctaa.history_rt('rtAccum1', UnityRenderTexture('src', MAIN_W, MAIN_H, 0, 'ARGBHalf', 'Default'), 'Bilinear'),
        'txaaOut': ctaa.history_rt('txaaOut', UnityRenderTexture('src', MAIN_W, MAIN_H, 0, 'ARGBHalf', 'Default'), 'Bilinear'),
        'capture rt': capture_loop.bridge_target(plan.request),
    }
    targets = segment_map(cap, 0)
    rows, ok = [], True
    for name, rt in gen.items():
        low = rt.d3d11(quality.color_space)
        cap_t = targets[name]
        color = [t for t in cap_t if not t.get('depth')][0]
        depth = [t for t in cap_t if t.get('depth')]
        cd = cap.resources[str(color['resource'])]['descriptor']
        good = cd == low['color']
        detail = {'name': name, 'resource': color['resource'], 'generated': low['color'], 'captured': cd}
        if low['depth'] or depth:
            dd = cap.resources[str(depth[0]['resource'])]['descriptor'] if depth else None
            detail['generated_depth'] = low['depth']
            detail['captured_depth'] = dd
            good &= dd == low['depth']
        # RTV view format of the writing draw
        ev = None
        for p in cap.seg_passes(0):
            if any(t['resource'] == color['resource'] for t in p['targets']) and cap.actions[p['first_event']]['kind'].startswith('Draw'):
                ev = p['first_event']
                break
        if ev is not None:
            rtv = [r for r in cap.actions[ev]['render_targets'] if r['resource'] == color['resource']]
            if rtv:
                detail['rtv_view_format'] = [rtv[0]['view_format'], low['color_view_format']]
                good &= rtv[0]['view_format'] == low['color_view_format']
        detail['pass'] = bool(good)
        ok &= bool(good)
        rows.append(detail)
    return {'pass': ok, 'rows': rows}


def check_order_and_identity(cap, plan):
    rows, ok = [], True
    ident = {}
    for s in range(16):
        rec = plan.renders[s]
        gen = [l for l, _ in capture_loop.plugin_pass_sequence(rec)]
        ps = cap.seg_passes(s)
        labels = [p['label'] for p in ps]
        a = labels.index('sss_profile')
        b = a + labels[a:].index('depth_texture_prepass', labels[a:].index('sss_lighting'))
        sss_block = labels[a:b]
        g0 = labels.index('grabpass_copy')
        g1 = labels.index('post_copy_std')
        trans_block = labels[g0:g1]
        c0 = next(i for i, p in enumerate(ps) if p['label'] == 'blit_copy' and p['viewport'] and p['viewport']['Width'] == 3200.0)
        chain_block = labels[c0:]
        cap_seq = sss_block + trans_block + chain_block
        # transparent block: the capture groups consecutive transparent draws in one pass
        gen_c = []
        for l in gen:
            if l == 'forward_scene' and gen_c and gen_c[-1] == 'forward_scene':
                continue
            gen_c.append(l)
        good = gen_c == cap_seq
        m = segment_map(cap, s)
        ctaa_ops = rec.image_effect_chain[-1]['ops']
        write = [o.args['dest'] for o in ctaa_ops if o.kind == 'blit' and o.args['material'] == 'Hidden/CTAA_PC'][0]
        accum = [o.args['textures']['_Accum'] for o in ctaa_ops if o.kind == 'blit' and o.args['material'] == 'Hidden/CTAA_PC'][0]
        for key in ('SSS_ProfileTex', 'LightingTexture', 'SSSLightingTextureBlurred', 'T1', 'T2', 'grab:_GrabTexture',
                    'afterPreEnhace', 'txaaOut', 'capture rt'):
            ident.setdefault(key, []).append([t['resource'] for t in m[key] if not t.get('depth')][0])
        ident.setdefault('ctaa_write', []).append(([t['resource'] for t in m['ctaa_write'] if not t.get('depth')][0], write))
        ident.setdefault('ctaa_accum', []).append((m['ctaa_accum_read'], accum))
        first_frame_blit = any(o.kind == 'blit' and o.args.get('note') == 'first-frame history initialisation' for o in ctaa_ops)
        ok &= good
        rows.append({'segment': s, 'pass_order_match': good, 'generated': gen_c, 'captured': cap_seq,
                     'first_frame_blit_generated': first_frame_blit})
    # identity patterns
    simulated = simulate_pool(plan)
    pattern = {}
    for key, ids in ident.items():
        if key in ('ctaa_write', 'ctaa_accum'):
            cap_ids = [x[0] for x in ids]
            gen_names = [x[1] for x in ids]
            mapping = {}
            consistent = all(mapping.setdefault(g, c) == c for g, c in zip(gen_names, cap_ids)) and len(set(mapping.values())) == len(mapping)
            pattern[key] = {'captured': cap_ids, 'generated': gen_names, 'bijective': consistent}
            ok &= consistent
        else:
            cap_distinct = len(set(ids))
            gen_ids = simulated.get(key)
            gen_distinct = len(set(gen_ids)) if gen_ids else None
            good = gen_ids is None or [ids.index(x) for x in ids] == [gen_ids.index(x) for x in gen_ids]
            pattern[key] = {'captured_resources': ids, 'captured_distinct': cap_distinct,
                            'generated_handles': gen_ids, 'generated_distinct': gen_distinct, 'pattern_match': good}
            ok &= good
    return {'pass': ok, 'rows': rows, 'identity': pattern}


def simulate_pool(plan):
    """Replay the plan's get/release ops through the Unity temporary-RT pool model."""
    alloc = RenderTextureAllocator(None, 'Linear')
    held = {}
    out = {}
    persistent = {}
    for rec in plan.renders:
        for op in rec.sss_ops:
            if op.kind == 'release_temporary':
                r = held.pop(op.args['handle'], None)
                if r is not None:
                    alloc.release_temporary(r[0], key_override=r[1])
            elif op.kind == 'get_temporary':
                rt = op.args['rt']
                r = alloc.get_temporary(rt)
                post = op.args.get('post_create') or {}
                key = rt.pool_key()
                if post.get('autoGenerateMips') is False:
                    key = key[:6] + (False,) + key[7:]
                held[op.args['handle']] = (r, key)
                out.setdefault(op.args['handle'], []).append(r['handle'])
            elif op.kind == 'render_camera' and op.args['camera'] == 'SSS Lighting Pass Camera':
                temps = {}
                for c in op.args['image_effects_opaque']:
                    if c['kind'] == 'get_temporary':
                        a = c['args']['rt']
                        rt = UnityRenderTexture(**a)
                        temps[c['args']['handle']] = alloc.get_temporary(rt)
                        out.setdefault(c['args']['handle'], []).append(temps[c['args']['handle']]['handle'])
                    elif c['kind'] == 'release_temporary':
                        alloc.release_temporary(temps.pop(c['args']['handle']))
        for eff in rec.image_effect_chain:
            for op in eff['ops']:
                if op.kind == 'create_persistent':
                    persistent[op.args['handle']] = alloc.create_persistent(op.args['rt'])
        for name in ('afterPreEnhace', 'txaaOut'):
            out.setdefault(name, []).append(persistent[name]['handle'])
        # Unnamed GrabPass: one static temp re-acquired per loop from the pool (same key).
        grab = [o for o in rec.transparent_ops if o.kind == 'get_temporary']
        for o in grab:
            r = alloc.get_temporary(o.args['rt'])
            out.setdefault(o.args['handle'], []).append(r['handle'])
            alloc.release_temporary(r)
        out.setdefault('capture rt', []).append('capture rt')
    return out


def check_grab_order(cap, plan):
    rec = plan.renders[0]
    gen = [(o.kind, o.args.get('object')) for o in rec.transparent_ops if o.kind in ('grab_copy', 'draw_pass')]
    ps = cap.seg_passes(0)
    labels = [p['label'] for p in ps]
    g0, g1 = labels.index('grabpass_copy'), labels.index('post_copy_std')
    seq = []
    for p in ps[g0:g1]:
        for e in p['events']:
            a = cap.actions[e]
            if a['kind'] == 'CopySubresourceRegion':
                seq.append(('grab_copy', a['destination']))
            elif a['kind'].startswith('Draw'):
                seq.append(('draw_pass', a['shaders']['PS']['debug_names'][0]))
    gen_kinds = [k for k, _ in gen]
    cap_kinds = [k for k, _ in seq]
    shader_of = {o.args['object']: o.args['shader'] for o in rec.transparent_ops if o.kind == 'draw_pass'}
    gen_shaders = [shader_of.get(n) if k == 'draw_pass' else 'grab' for k, n in gen]
    cap_shaders = [n if k == 'draw_pass' else 'grab' for k, n in seq]
    # which grab each grab-reading draw sampled
    reads = {}
    for p in ps[g0:g1]:
        for e in p['events']:
            a = cap.actions[e]
            if a['kind'].startswith('Draw'):
                for r in a['shaders']['PS']['srvs']:
                    if r['name'] == '_GrabTexture':
                        reads[a['shaders']['PS']['debug_names'][0]] = r['producers'][0]['base']
    good = gen_shaders == cap_shaders
    return {'pass': good, 'generated': gen_shaders, 'captured': cap_shaders, 'captured_grab_reads': reads,
            'generated_ops': [o.to_json() for o in rec.transparent_ops]}


def check_blit_constants(cap):
    from src.render.hs2pipe import unity_math as um
    # GL.LoadOrtho: projection ortho(0,1,0,1,-1,100), view identity; unity_MatrixVP =
    # MultiplyMatrices4x4(GPU projection, view) (the product maps -0 entries to +0).
    ortho = um.gl_get_gpu_projection(um.set_ortho(0, 1, 0, 1, -1, 100), True)
    gen = np.asarray(um.multiply(ortho, um.identity()), np.float32).tobytes().hex()
    ident = np.eye(4, dtype=np.float32).T.reshape(-1).tobytes().hex()
    rows, ok = [], True
    for s in range(16):
        for p in cap.seg_passes(s):
            names = p['pixel_and_compute_shader_debug_names']
            if p['label'] in ('sss_separable_blur', 'temporal_aa_ctaa') or (p['label'] == 'blit_copy' and p['viewport'] and p['viewport']['Width'] in (1086.0, 3200.0) and s < 16):
                e = p['first_event']
                vp = cap.var(e, 'VS', 'unity_MatrixVP')
                ow = cap.var(e, 'VS', 'unity_ObjectToWorld')
                if vp is None:
                    continue
                good = vp['hex'] == gen and ow['hex'] == ident
                ok &= good
                rows.append({'segment': s, 'event': e, 'label': p['label'], 'pass': good})
    return {'pass': ok, 'count': len(rows), 'generated_unity_MatrixVP': gen,
            'failures': [r for r in rows if not r['pass']]}


def check_samplers(cap, manifests):
    """Unity sampler key of the plugin RT bound to each plugin texture slot."""
    expect = {
        ('Hidden/SeparableSSS', '_MainTex'): ('Bilinear', 'Clamp'),
        ('Hidden/SeparableSSS', 'SSS_ProfileTex'): ('Point', 'Clamp'),
        ('Hidden/SeparableSSS', 'SSS_ProfileTexR'): ('Point', 'Clamp'),
        ('Hidden/CTAA_PC', '_MainTex'): ('Bilinear', 'Clamp'),
        ('Hidden/CTAA_PC', '_Accum'): ('Bilinear', 'Clamp'),
        ('Hidden/CTAA_Enhance_PC', '_MainTex'): ('Bilinear', 'Clamp'),
        ('AIT/Liquid', '_GrabTexture'): ('Bilinear', 'Clamp'),
        ('AIT/main namida', '_GrabTexture'): ('Bilinear', 'Clamp'),
    }
    rows, ok = [], True
    seen = set()
    for e, a in sorted(cap.actions.items()):
        if e > 2259 or not a['kind'].startswith('Draw'):
            continue
        ps = a['shaders'].get('PS', {})
        for m in ps.get('source_matches') or []:
            key_shader = m['shader']
            program = None
            try:
                program = manifests.program(m['shader'], m['pass'], 'progFragment', m['keywords'])
            except Exception:
                continue
            slots = texture_slots(program)
            for tex, (filt, wrap) in expect.items():
                if tex[0] != key_shader or tex[1] not in slots or (key_shader, tex[1]) in seen:
                    continue
                sslot = slots[tex[1]]['sampler']
                bound = [x for x in ps['samplers'] if x['slot'] == sslot]
                if not bound:
                    continue
                res = cap.resources[str(bound[0]['sampler'])]
                name = res['debug_names'][0]
                genkey = unity_sampler_key(filt, wrap, 1)
                good = name == genkey
                ok &= good
                seen.add((key_shader, tex[1]))
                rows.append({'shader': key_shader, 'texture': tex[1], 'event': e, 'sampler_resource': bound[0]['sampler'],
                             'captured_key': name, 'generated_key': genkey, 'd3d11_descriptor': res['descriptor'], 'pass': good})
            break
    return {'pass': ok, 'rows': rows}


def check_sss_view(cap):
    """Lighting camera VP vs main camera _NonJitteredVP of the same render."""
    rows = []
    for s in range(16):
        ps = cap.seg_passes(s)
        light = next(p for p in ps if p['label'] == 'sss_lighting')
        lvp = cap.var(light['first_event'], 'VS', 'unity_MatrixVP')
        nj = None
        for p in ps:
            if p['label'] == 'motion_vectors':
                v = cap.var(p['first_event'], 'VS', '_NonJitteredVP') or cap.var(p['first_event'], 'PS', '_NonJitteredVP')
                if v:
                    nj = v
                    break
        rows.append({'segment': s, 'lighting_vp': lvp['hex'], 'main_non_jittered_vp': nj['hex'],
                     'byte_equal': lvp['hex'] == nj['hex']})
    return {'byte_equal_all': all(r['byte_equal'] for r in rows), 'distinct_lighting_vp': len({r['lighting_vp'] for r in rows}),
            'rows': rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frame-graph', type=Path, required=True)
    ap.add_argument('--captured-cbuffers', type=Path, required=True)
    ap.add_argument('--runtime-fields', type=Path, required=True)
    ap.add_argument('--shader-manifest', type=Path, action='append', default=[])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--receipt', type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    manifests = ShaderManifests(args.shader_manifest)
    cap = Capture(args.frame_graph, args.captured_cbuffers)
    rf = load(args.runtime_fields)
    plan, sss, conv, cam, quality, source = build_plan(rf, manifests)
    report = {
        'schema_version': SCHEMA,
        'provenance': {'frame_graph': str(args.frame_graph.resolve()), 'captured_cbuffers': str(args.captured_cbuffers.resolve()),
                       'captured_cbuffers_sha256': sha(args.captured_cbuffers),
                       'runtime_fields': str(args.runtime_fields.resolve()), 'runtime_fields_sha256': sha(args.runtime_fields),
                       'runtime_fields_scope': 'other frame and process (runtime_audit_hook_v3); component fields assumed unchanged at capture',
                       'manifests': [str(p.resolve()) for p in args.shader_manifest],
                       'tool_sha256': sha(__file__),
                       'generators_sha256': {p.name: sha(p) for p in sorted((ROOT / 'src/render/hs2pipe').glob('*.py'))
                                             if p.name in ('dhh_sss.py', 'ctaa.py', 'liquid.py', 'capture_loop.py', 'plugin_common.py')}},
        'checks': {}}
    c = report['checks']
    c['sss_blur_cbuffers'] = check_blur_cbuffers(cap, manifests, plan, sss)
    c['ctaa_cbuffers'] = check_ctaa(cap, manifests, plan, source)
    c['ctaa_jitter_vp'] = check_jitter(cap, plan)
    c['rt_descriptors'] = check_descriptors(cap, plan, sss, quality)
    c['pass_order_identity'] = check_order_and_identity(cap, plan)
    c['grab_order'] = check_grab_order(cap, plan)
    c['blit_constants'] = check_blit_constants(cap)
    c['sampler_keys'] = check_samplers(cap, manifests)
    c['sss_lighting_view_vs_main_nonjittered'] = check_sss_view(cap)
    report['summary'] = {k: v.get('pass', v.get('byte_equal_all')) for k, v in c.items()}
    args.out.mkdir(parents=True)
    (args.out / 'report.json').write_text(json.dumps(jsonable(report), indent=1), encoding='utf-8')
    (args.out / 'plan.json').write_text(json.dumps(jsonable(plan.to_json()), indent=1), encoding='utf-8')
    if args.receipt:
        receipt = {'schema_version': SCHEMA, 'provenance': report['provenance'], 'summary': report['summary'],
                   'counts': {'blur_draws': c['sss_blur_cbuffers']['draws'], 'ctaa_draws': len(c['ctaa_cbuffers']['rows']),
                              'jitter_iterations': len(c['ctaa_jitter_vp']['rows']), 'blit_vs_constants': c['blit_constants']['count'],
                              'descriptors': len(c['rt_descriptors']['rows']), 'sampler_slots': len(c['sampler_keys']['rows'])},
                   'sss_lighting_vp_equals_main_nonjittered_vp': c['sss_lighting_view_vs_main_nonjittered']['byte_equal_all'],
                   'report': str((args.out / 'report.json').resolve())}
        args.receipt.write_text(json.dumps(receipt, indent=1), encoding='utf-8')
    print(json.dumps(report['summary']))


if __name__ == '__main__':
    main()
