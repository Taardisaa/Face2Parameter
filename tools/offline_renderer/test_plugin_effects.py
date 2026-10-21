"""Tests for the plugin effect generators (DHH SSS, CTAA, GrabPass/Liquid, bridge capture loop).

Expected bytes are values recorded in the retry capture (frame 17675) or follow
from the decompiled plugin source; see docs/hs2_offline_plugin_effects.md.
Evidence-backed tests skip only when the ignored evidence files are absent.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import capture_loop, ctaa, dhh_sss, liquid  # noqa: E402
from src.render.hs2pipe.plugin_common import (F32, PluginLowering, PluginStateError, ProgramRecorder,  # noqa: E402
                                              RenderTextureAllocator, ShaderManifests, UnimplementedBranch,
                                              UnityRenderTexture, mathf_lerp, unity_sampler_key)

OUT = ROOT / 'outputs'
MANIFESTS = [OUT / 'offline_renderer_repair_20261012/sss_shader_programs_v2/manifest.json',
             OUT / 'offline_renderer_frame_graph_20261020/shader_exports/hs2_betteraa_ctaa/manifest.json',
             OUT / 'offline_renderer_repair_20261012/body_shader_programs_v1/manifest.json',
             OUT / 'offline_renderer_repair_20261012/shader_programs_v2/manifest.json']
FRAME_GRAPH = OUT / 'offline_renderer_frame_graph_20261020/frame_graph_v2'
CAPTURED_CB = OUT / 'offline_renderer_unity_forward_20261020/captured_cbuffers_v1.json'
RUNTIME_FIELDS = ROOT.parent / 'HS2Mod/artifacts/chenger_reference_fit_20261012/runtime_audit_hook_v3/runtime_fields_completion.json'
HAVE_MANIFESTS = all(p.exists() for p in MANIFESTS)
HAVE_EVIDENCE = HAVE_MANIFESTS and FRAME_GRAPH.exists() and CAPTURED_CB.exists() and RUNTIME_FIELDS.exists()


def h32(v):
    return np.asarray([v], np.float32).tobytes().hex()


def sss_state(**over):
    base = dict(enabled=True, downsampling=2.0, scattering_radius=1.0, scattering_iterations=2, shader_iterations=2,
                dither=True, dither_intensity=1.0, dither_scale=5.0, depth_test=0.1, normal_test=0.1,
                profile_color_test=0.1, profile_radius_test=0.1, dither_edge_test=True, edge_offset=1.2,
                fix_pixel_leaks=False, max_distance=10000, profile_per_object=True, use_profile_test=False,
                debug_distance=False, sss_color=(1.0, 0.3, 0.1, 1.0), sss_layer=3072, toggle_texture='None',
                noise_texture={'instance_id': -96070, 'width': 256, 'height': 256, 'format': 'ARGB32'})
    base.update(over)
    return dhh_sss.SSSState(**base)


CONV = dhh_sss.ConvolutionState(init_fov=23.0, allow_msaa=False, blur_material_exists=True)
CAM = dhh_sss.CameraState(pixel_width=1086, pixel_height=1448, field_of_view=23.0, aspect=0.75, near_clip=0.08,
                          far_clip=1000.0, orthographic=False, orthographic_size=5.0, allow_hdr=True, allow_msaa=False,
                          clear_flags='SolidColor', background_color=(0.5, 0.5, 0.5, 1.0), culling_mask=525312)
QUALITY = dhh_sss.QualityState(anti_aliasing=0, pixel_light_count=16, shadows='All', color_space='Linear')


def ctaa_state():
    cfg = ctaa.CTAAConfig(temporal_stability=3, hdr_response=0.001, adaptive_enhance=0.32, temporal_jitter_scale=0.25,
                          micro_shimmer_reduction=3.0, static_stability_power=0.5)
    st = ctaa.CTAAState(True, cfg, False, False, 1, F32(0.25))
    big = UnityRenderTexture('x', 3200, 1800, 0, 'ARGBHalf', 'Default', lifetime='persistent')
    st.rt_accum0 = st.rt_accum1 = st.txaa_out = st.after_pre_enhance = big
    return st


class MonoArithmetic(unittest.TestCase):
    def test_lerp_double_intermediate(self):
        # CTAA_PC.preEnhanceStrength/Clamp captured as _AESCTAA / _AEMAXCTAA (event 2182)
        self.assertEqual(h32(mathf_lerp(0.2, 2.0, 0.32)), 'efa7463f')
        self.assertEqual(h32(mathf_lerp(0.005, 0.12, 0.32)), '7a362b3d')

    def test_capture_distance_and_aspect(self):
        # runtime_audit_hook_v3 actual_pipeline_capture: ortho_size 1.42995071 -> 7.028433
        self.assertEqual(float(capture_loop.capture_distance(1.42995071, 23.0)), float(F32(7.028433)))
        self.assertEqual(h32(capture_loop.capture_aspect(1086, 1448)), h32(F32(1086) / F32(1448)))


class SSS(unittest.TestCase):
    def test_texture_size_and_radius(self):
        w, h = dhh_sss.texture_size(sss_state(), CAM)
        self.assertEqual((int(w), int(h)), (543, 724))
        r = dhh_sss.convolution_radius(dhh_sss.lighting_blur_radius(sss_state(), CAM), 23.0, 23.0, 543, 724)
        self.assertEqual(h32(r), '1957523e')       # captured _TexelOffsetScale.y, event 579
        # normal-frame audit: 1600x900 -> BlurRadius 0.416666657
        r2 = dhh_sss.convolution_radius(F32(0.5), 23.0, 23.0, 1600, 900)
        self.assertEqual(float(r2), float(F32(0.416666657)))

    def test_material_values(self):
        props, kw = dhh_sss.blur_material(sss_state())
        self.assertEqual(h32(props['DepthTest']), '0ad7a33b')     # max(1e-5, 0.1/20)
        self.assertEqual(float(props['_SSS_NUM_SAMPLES']), 3.0)
        self.assertEqual(kw, {'RANDOMIZED_ROTATION': True, 'PROFILE_TEST': False, 'DEBUG_DISTANCE': False,
                              'OFFSET_EDGE_TEST': False, 'DITHER_EDGE_TEST': True})
        props, kw = dhh_sss.blur_material(sss_state(dither=False, use_profile_test=True))
        self.assertNotIn('DitherScale', props)
        self.assertTrue(kw['PROFILE_TEST'])

    def test_convolution_schedule(self):
        ops = dhh_sss.on_pre_render(sss_state(), CONV, CAM, QUALITY)
        render = [o for o in ops if o.kind == 'render_camera' and o.args['camera'] == 'SSS Lighting Pass Camera'][0]
        blits = [(c['args']['source'], c['args']['dest'], c['args'].get('axis')) for c in render.args['image_effects_opaque']
                 if c['kind'] == 'blit']
        self.assertEqual(blits, [('source', 'T1', 'vertical'), ('T1', 'T2', 'horizontal'), ('T2', 'T1', 'vertical'),
                                 ('T1', 'blurred', 'horizontal'), ('source', 'destination', None)])
        ops1 = dhh_sss.convolution_on_render_image(iterations=1, blur_radius=F32(0.5), init_fov=23, lighting_fov=23,
                                                   source=UnityRenderTexture('s', 10, 10, 0, 'ARGBFloat', 'Default'),
                                                   rt_format='ARGBFloat', allow_msaa_field=False,
                                                   lighting_cam_allow_msaa=False, quality_aa=0)
        self.assertEqual([o.args['dest'] for o in ops1 if o.kind == 'blit'], ['T1', 'blurred', 'destination'])
        ops0 = dhh_sss.convolution_on_render_image(iterations=0, blur_radius=F32(0.5), init_fov=23, lighting_fov=23,
                                                   source=UnityRenderTexture('s', 10, 10, 0, 'ARGBFloat', 'Default'),
                                                   rt_format='ARGBFloat', allow_msaa_field=False,
                                                   lighting_cam_allow_msaa=False, quality_aa=0)
        self.assertEqual([o.args['dest'] for o in ops0], ['blurred', 'destination'])

    def test_order_keywords_and_disabled(self):
        ops = dhh_sss.on_pre_render(sss_state(), CONV, CAM, QUALITY)
        kinds = [(o.kind, o.args.get('camera') or o.args.get('keyword') or o.args.get('name')) for o in ops]
        self.assertEqual(kinds[0], ('global_keyword', 'SCENE_VIEW'))
        prof = kinds.index(('render_camera', 'SSS profile Camera'))
        light = kinds.index(('render_camera', 'SSS Lighting Pass Camera'))
        self.assertLess(prof, light)
        self.assertLess(kinds.index(('global_keyword', 'SSS_PROFILES')), prof)
        lf = [o for o in ops if o.kind == 'camera_field' and o.args['camera'] == 'SSS Lighting Pass Camera'][-1].args['fields']
        self.assertEqual((lf['cullingMask'], lf['depthTextureMode'], lf['renderingPath']), (3072, 'DepthNormals', 'Forward'))
        off = dhh_sss.on_pre_render(sss_state(enabled=False), CONV, CAM, QUALITY)
        self.assertEqual([o.kind for o in off], ['camera_field', 'buffers_viewer'])
        with self.assertRaises(UnimplementedBranch):
            dhh_sss.texture_size(sss_state(), dhh_sss.CameraState(**dict(CAM.__dict__, stereo_enabled=True)))

    def test_rt_descriptors(self):
        lt = dhh_sss.lighting_rt('LightingTexture', 543, 724, CAM, QUALITY).d3d11('Linear')
        self.assertEqual((lt['color']['Format'], lt['color_view_format'], lt['depth']['Format'], lt['color']['BindFlags'],
                          lt['depth']['BindFlags']), (1, 2, 19, 40, 72))
        pr = dhh_sss.profile_rt(543, 724).d3d11('Linear')
        self.assertEqual((pr['color']['Format'], pr['color_view_format'], pr['depth']['Format']), (27, 28, 53))
        cap = capture_loop.bridge_target(capture_loop.CaptureRequest('a.png', (0, 0, 0), 1.0, width=1086, height=1448))
        self.assertEqual(cap.d3d11('Linear')['color_view_format'], 29)


class CTAA(unittest.TestCase):
    def test_jitter_sequence_and_state_machine(self):
        st = ctaa_state()
        st.reset_temporal()
        proj = [F32(0)] * 16
        idx, writes, first = [], [], []
        src = UnityRenderTexture('ImageEffects Temp', 1086, 1448, 0, 'ARGBHalf', 'Default')
        for k in range(16):
            pc = ctaa.on_pre_cull(st, camera_orthographic=False, base_projection=proj, pixel_rect_width=1086, pixel_rect_height=1448)
            idx.append(pc['jitter_index'])
            ops = ctaa.on_render_image(st, camera_orthographic=False, source=src, screen_width=3200, screen_height=1800)
            writes.append([o.args['dest'] for o in ops if o.kind == 'blit' and o.args['material'] == 'Hidden/CTAA_PC'][0])
            first.append(any(o.args.get('note') for o in ops if o.kind == 'blit'))
            if k == 0:
                self.assertEqual(sum(o.kind == 'create_persistent' for o in ops), 4)   # 3200x1800 -> 1086x1448
            else:
                self.assertFalse(any(o.kind == 'create_persistent' for o in ops))
        self.assertEqual(idx, list(range(16)))
        self.assertEqual(writes, ['rtAccum1', 'rtAccum0'] * 8)
        self.assertEqual(first, [True] + [False] * 15)
        self.assertEqual(st.frame_counter, 0)

    def test_jitter_bits(self):
        # Mono double intermediates: m02 for x_jit[0]=0.5, scale 0.25 -> (0.25-1)/1086
        dx, dy = ctaa.jitter_offsets(0, F32(0.25), 1086, 1448)
        self.assertEqual(h32(dx), h32(F32((0.125 * 2 - 1) / 1086.0)))
        self.assertEqual(h32(dy), h32(F32((float(F32(0.33)) * 0.25 * 2 - 1) / 1448.0)))

    def test_material_values(self):
        st = ctaa_state()
        st.reset_temporal()
        src = UnityRenderTexture('ImageEffects Temp', 1086, 1448, 0, 'ARGBHalf', 'Default')
        ops = ctaa.on_render_image(st, camera_orthographic=False, source=src, screen_width=3200, screen_height=1800)
        enh = [o for o in ops if o.kind == 'material' and o.args['shader'] == 'Hidden/CTAA_Enhance_PC'][0].args['properties']
        self.assertEqual([h32(enh[k]) for k in ('_AEXCTAA', '_AEYCTAA', '_AESCTAA', '_AEMAXCTAA')],
                         ['0ad7a339', 'b4a2113a', 'efa7463f', '7a362b3d'])
        res = [o for o in ops if o.kind == 'material' and o.args['shader'] == 'Hidden/CTAA_PC'][0].args['properties']
        self.assertEqual(res['_ControlParams'].tobytes().hex(), '0000003f000040406f12833a00004040')
        self.assertEqual(h32(res['_AdaptiveResolve']), '00803b45')

    def test_disabled_and_orthographic(self):
        st = ctaa_state()
        ops = ctaa.on_render_image(st, camera_orthographic=True, source=UnityRenderTexture('s', 8, 8, 0, 'ARGBHalf', 'Default'),
                                   screen_width=8, screen_height=8)
        self.assertEqual([(o.args['source'], o.args['dest']) for o in ops], [('source', 'destination')])
        self.assertEqual(float(st.jitter_scale), 0.0)
        self.assertIsNone(ctaa.on_pre_cull(st, camera_orthographic=True, base_projection=[F32(0)] * 16,
                                           pixel_rect_width=8, pixel_rect_height=8))


class GrabPass(unittest.TestCase):
    def objects(self, eyelash_add_lights=0):
        return [liquid.ForwardObject('liquid', 'AIT/Liquid', 2999, [{'pass': 0, 'kind': 'grab', 'grab_name': None},
                                                                    {'pass': 1, 'kind': 'forward_base', 'grab_name': None},
                                                                    {'pass': 2, 'kind': 'forward_add', 'grab_name': None}]),
                liquid.ForwardObject('lashes', 'AIT/eyelashes', 3000, [{'pass': 0, 'kind': 'forward_base', 'grab_name': None},
                                                                       {'pass': 1, 'kind': 'forward_add', 'grab_name': None}],
                                     forward_add_lights=eyelash_add_lights),
                liquid.ForwardObject('namida', 'AIT/main namida', 4001, [{'pass': 0, 'kind': 'grab', 'grab_name': None},
                                                                          {'pass': 1, 'kind': 'forward_base', 'grab_name': None}])]

    def test_grab_precedes_pending_draw(self):
        cam = UnityRenderTexture('cam', 1086, 1448, 24, 'ARGBHalf', 'Default')
        ops = liquid.schedule_forward_loop(self.objects(), cam)
        seq = [(o.kind, o.args['object']) for o in ops if o.kind in ('grab_copy', 'draw_pass')]
        self.assertEqual(seq, [('grab_copy', 'liquid'), ('draw_pass', 'liquid'), ('grab_copy', 'namida'),
                               ('draw_pass', 'lashes'), ('draw_pass', 'namida')])
        # an immediate ForwardAdd draw flushes the batch first: the grab then sees the eyelashes
        ops = liquid.schedule_forward_loop(self.objects(eyelash_add_lights=1), cam)
        seq = [(o.kind, o.args['object']) for o in ops if o.kind in ('grab_copy', 'draw_pass')]
        self.assertEqual(seq.index(('grab_copy', 'namida')), 4)

    def test_named_grab_once_per_loop(self):
        cam = UnityRenderTexture('cam', 64, 64, 24, 'ARGBHalf', 'Default')
        objs = [liquid.ForwardObject('a%d' % i, 'X', 3000, [{'pass': 0, 'kind': 'grab', 'grab_name': '_Bg'},
                                                            {'pass': 1, 'kind': 'forward_base', 'grab_name': None}]) for i in range(3)]
        ops = liquid.schedule_forward_loop(objs, cam)
        self.assertEqual(sum(o.kind == 'grab_copy' for o in ops), 1)
        unnamed = [liquid.ForwardObject('u%d' % i, 'X', 3000, [{'pass': 0, 'kind': 'grab', 'grab_name': None},
                                                               {'pass': 1, 'kind': 'forward_base', 'grab_name': None}]) for i in range(3)]
        self.assertEqual(sum(o.kind == 'grab_copy' for o in liquid.schedule_forward_loop(unnamed, cam)), 3)
        rt = liquid.grab_rt(cam)
        self.assertEqual((rt.depth_bits, rt.resolved_format(), rt.filter_mode), (0, 'ARGBHalf', 'Bilinear'))
        with self.assertRaises(UnimplementedBranch):
            liquid.grab_rt(UnityRenderTexture('m', 8, 8, 24, 'ARGBHalf', 'Default', msaa=4))


class Pool(unittest.TestCase):
    def test_lifo_and_profile_miss(self):
        alloc = RenderTextureAllocator(None, 'Linear')
        t = UnityRenderTexture('T', 8, 8, 0, 'ARGBFloat', 'Linear')
        a, b = alloc.get_temporary(t), alloc.get_temporary(t)
        alloc.release_temporary(b)
        alloc.release_temporary(a)
        self.assertEqual(alloc.get_temporary(t)['handle'], a['handle'])     # most recently released first
        p = dhh_sss.profile_rt(8, 8)
        r = alloc.get_temporary(p)
        k = p.pool_key()
        alloc.release_temporary(r, key_override=k[:6] + (False,) + k[7:])
        self.assertNotEqual(alloc.get_temporary(p)['handle'], r['handle'])  # autoGenerateMips=false -> miss

    def test_sampler_key(self):
        self.assertEqual(unity_sampler_key('Point', 'Clamp', 1), 'SamplerState-0-111-1')


class StubForward:
    def __init__(self):
        self.calls = []

    def blit(self, builder, **kw):
        self.calls.append(('blit', kw['shader'], kw['source'].get('handle', kw['source'].get('name')),
                           kw['dest'].get('handle', kw['dest'].get('name')), kw['ps_globals']))

    def render_camera(self, builder, args, on_opaque):
        self.calls.append(('render_camera', args['camera']))
        if args['image_effects_opaque']:
            on_opaque({'name': 'source', 'rt': UnityRenderTexture('LightingTexture', 543, 724, 24, 'ARGBFloat', 'Default')},
                      {'name': 'destination', 'rt': UnityRenderTexture('ImageEffects Temp', 543, 724, 0, 'ARGBFloat', 'Default')})

    def engine_state(self, op):
        self.calls.append(('state', op['kind']))

    def draw_pass(self, builder, op):
        self.calls.append(('draw', op['args']['object']))

    def active_color_target(self):
        return {'resource': 'camera_color'}

    def engine_buffer(self, name):
        return {'name': name, 'resource': 'engine:' + name,
                'rt': UnityRenderTexture(name, 1086, 1448, 0, 'ARGBHalf', 'Default')}


@unittest.skipUnless(HAVE_MANIFESTS, 'installed shader exports not present')
class Lowering(unittest.TestCase):
    def test_capture_plan_lowering(self):
        m = ShaderManifests(MANIFESTS)
        cam = capture_loop.MainCamera(23.0, 0.08, 1000.0, True, False, tuple((t, True) for t in capture_loop.MAKER_CAMERA_COMPONENTS[:-1]) +
                                      (('DHH.SSS_buffers_viewer', False),), 3200, 1800, True, (3200, 1800))
        req = capture_loop.CaptureRequest('C:/x.png', (0.0, 14.0, 0.0), 1.0, width=1086, height=1448)
        src = UnityRenderTexture('ImageEffects Temp', 1086, 1448, 0, 'ARGBHalf', 'Default')
        objs = [liquid.ForwardObject('liquid', 'AIT/Liquid', 2999, liquid.forward_passes(m, 'AIT/Liquid'))]
        plan = capture_loop.plan_capture(req, cam, sss=sss_state(), conv=CONV, quality=QUALITY, ctaa_state=ctaa_state(),
                                         character_layer_mask=525312, base_projection=[F32(0)] * 16,
                                         transparent_objects=objs, camera_source_rt=src)
        self.assertEqual(len(plan.renders), 16)
        rec = ProgramRecorder()
        fwd = StubForward()
        low = PluginLowering(rec, m, fwd, 'Linear')
        low.emit(plan.setup)
        for r in plan.renders:
            low.emit(r.sss_ops)
            low.emit(r.transparent_ops)
            for eff in r.image_effect_chain:
                low.emit(eff['ops'], {'source': fwd.engine_buffer('source'), 'destination': fwd.engine_buffer('destination')})
            low.emit(r.post_render)
        blits = [c for c in fwd.calls if c[0] == 'blit']
        # per render: 4 blur + conv copy + 2 CustomRender + enhance + resolve + 2 copies = 11; +1 first-frame init
        self.assertEqual(len(blits), 16 * 11 + 1)
        sep = [c for c in blits if c[1] == 'Hidden/SeparableSSS']
        self.assertEqual(sep[0][4]['_TexelOffsetScale']['hex'], '000000001957523e0000000000000000')
        self.assertEqual(sep[0][4]['maxDistance']['hex'], '10270000')
        self.assertEqual(sum(c['op'] == 'copy_subresource_region' for c in rec.commands), 16)
        names = [r['name'] for r in rec.resources]
        self.assertEqual(names.count('SSS_ProfileTex'), 16)       # fresh every render (pool miss)
        self.assertEqual(names.count('LightingTexture'), 1)
        self.assertEqual(names.count('rtAccum0'), 1)

    def test_real_program_builder(self):
        import tempfile
        from src.render.frame_program import ProgramBuilder
        with tempfile.TemporaryDirectory() as tmp:
            b = ProgramBuilder(Path(tmp) / 'program')
            alloc = RenderTextureAllocator(b, 'Linear')
            cam = alloc.get_temporary(UnityRenderTexture('cam', 64, 32, 24, 'ARGBHalf', 'Default'))
            lt = alloc.get_temporary(dhh_sss.lighting_rt('LightingTexture', 32, 16, CAM, QUALITY))
            grab = alloc.get_temporary(liquid.grab_rt(cam['rt']))
            liquid.grab_copy_command(b, source_resource=cam['resource'], dest_resource=grab['resource'], width=64, height=32)
            b.write()                       # frame_program validator accepts descriptors, views and the copy
            self.assertIn('dsv', lt)

    def test_missing_interface_raises(self):
        low = PluginLowering(ProgramRecorder(), ShaderManifests(MANIFESTS), object(), 'Linear')
        with self.assertRaises(UnimplementedBranch):
            low.emit(dhh_sss.on_post_render())


@unittest.skipUnless(HAVE_EVIDENCE, 'retry capture evidence not present')
class CaptureByteDiff(unittest.TestCase):
    def test_validation_against_capture(self):
        import tempfile
        from tools.offline_renderer import validate_plugin_effects as v
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'report'
            args = ['--frame-graph', str(FRAME_GRAPH), '--captured-cbuffers', str(CAPTURED_CB),
                    '--runtime-fields', str(RUNTIME_FIELDS), '--out', str(out)]
            for p in MANIFESTS:
                args += ['--shader-manifest', str(p)]
            v.main(args)
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
        self.assertTrue(all(report['summary'].values()), report['summary'])


if __name__ == '__main__':
    unittest.main()
