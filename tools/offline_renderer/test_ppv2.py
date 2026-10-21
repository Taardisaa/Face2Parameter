"""Tests for the PPv2 generator (src/render/hs2pipe/ppv2*.py).

Unit tests need no game, capture or GPU.  Integration tests use the ignored outputs
under ``outputs/offline_renderer_ppv2_20261020`` (oracle JSON, scene state, compute
export) and installed game assets; they are skipped when those are absent.

    .venv/Scripts/python.exe -m unittest tools.offline_renderer.test_ppv2 -v
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import ppv2  # noqa: E402
from src.render.hs2pipe.ppv2_settings import (EffectSettings, Param, SplineValue, Volume, VolumeBlender,  # noqa: E402
                                              parse_dhh_graphic_setting, sort_by_priority, texture_interp,
                                              TextureValue)
from src.render.hs2pipe.ppv2_unity import (F, RTDescriptor, SystemRandom, TextureRef, UnityConstantBufferPool,  # noqa: E402
                                           blend_desc, curve_cache_coeffs, depth_desc, dv, evaluate_curve, r4,
                                           sampler_desc)

OUT = ROOT / 'outputs' / 'offline_renderer_ppv2_20261020'
SCENE_STATE = OUT / 'scene_state_v2' / 'characustom_ppv2.json'
RETRY_ORACLE = OUT / 'oracle_v3' / 'oracle.json'
PAIRED_ORACLE = OUT / 'paired_04d1e76f' / 'oracle_v1' / 'oracle.json'
COMPUTE = OUT / 'compute_shaders_v1' / 'manifest.json'
SHADERS = ROOT / 'outputs' / 'offline_renderer_frame_graph_20261020' / 'shader_exports' / 'sharedassets2' / 'manifest.json'
HS2 = ROOT.parent / 'HS2Mod' / 'artifacts' / 'chenger_reference_fit_20261012'
RETRY_REQUEST = HS2 / 'native_present_session_v1' / 'capture_v1' / 'native_request.json'
PAIRED_FRAMES = HS2 / 'paired_capture_session_20261020' / 'window_04d1e76fdad14c11bbc6fe439b228e77' / 'frames'


def bits(x):
    return struct.pack('<f', float(F(x)))


class MonoNumerics(unittest.TestCase):
    def test_double_intermediate_matches_captured_msvo_table(self):
        # MultiScaleVO.PushRenderCommands for L3 (136 wide) with projection[0,0] of the retry capture;
        # captured gInvThicknessTable[0] = 0x4235eea0 (45.4830322).  Pure float32 evaluation gives 0x4235eea1.
        tan = r4(1.0 / dv(6.553542613983154))
        num = r4(2.0 * dv(tan) * 10.0 / 136.0)
        num2 = r4(1.0 / dv(num))
        th0 = F(__import__('math').sqrt(float(F(0.96))))
        self.assertEqual(bits(r4(dv(num2) / dv(th0))).hex(), 'a0ee3542')
        f32 = (F(2) * (F(1) / F(6.553542613983154)) * F(10) / F(136))
        self.assertNotEqual(bits((F(1) / f32) / th0).hex(), 'a0ee3542')

    def test_system_random_reproduces_captured_dithering(self):
        # retry capture segment 0 FinalPass _Dithering_Coords.zw follow 14193 prior Dithering.Render calls
        r = SystemRandom(1234)
        for _ in range(14193):
            r.next_double()
            r.next_double()
        z, w = F(r.next_double()), F(r.next_double())
        self.assertEqual((bits(z).hex(), bits(w).hex()), (bits(0.0611846186).hex(), bits(0.0163251646).hex()))

    def test_dithering_wraps_index(self):
        d = ppv2.Dithering()
        idx = [d.step(64)[0] for _ in range(66)]
        self.assertEqual(idx[:2], [1, 2])
        self.assertEqual(idx[62:66], [63, 0, 1, 2])


class AnimationCurveNative(unittest.TestCase):
    KEYS = [{'time': F(0), 'value': F(0.75), 'inSlope': F(-0.6292682886123657), 'outSlope': F(-0.6292682886123657),
             'weightedMode': 0, 'inWeight': F(0), 'outWeight': F(0)},
            {'time': F(0.481013), 'value': F(0.553191), 'inSlope': F(0.6323527693748474), 'outSlope': F(0.6323527693748474),
             'weightedMode': 0, 'inWeight': F(0), 'outWeight': F(0)},
            {'time': F(1), 'value': F(0.475862), 'inSlope': F(0.3650944232940674), 'outSlope': F(0.3650944232940674),
             'weightedMode': 0, 'inWeight': F(0), 'outWeight': F(0)}]

    def test_clamp_and_key_values(self):
        self.assertEqual(evaluate_curve(self.KEYS, -1), F(0.75))
        self.assertEqual(evaluate_curve(self.KEYS, 1), F(0.475862))
        self.assertEqual(evaluate_curve(self.KEYS, 0), F(0.75))
        self.assertEqual(evaluate_curve(self.KEYS, self.KEYS[1]['time']), F(0.553191))

    def test_stepped_tangent(self):
        k = [dict(self.KEYS[0], outSlope=F('inf')), self.KEYS[1]]
        self.assertEqual(curve_cache_coeffs(k[0], k[1])[2:5], (F(0), F(0), F(0)))

    @unittest.skipUnless(SCENE_STATE.exists(), 'scene state export absent')
    def test_serialized_cached_data_bit_exact(self):
        from src.render.hs2pipe.ppv2_settings import effect_from_scene
        st = json.loads(SCENE_STATE.read_text(encoding='utf-8'))
        n = 0
        for row in st['effects'].values():
            e = effect_from_scene(row, st.get('textures'))
            for p in e.params.values():
                if p.kind == 'spline' and p.value.keys:
                    self.assertEqual([bits(x) for x in p.value.cached], [bits(x) for x in p.value.serialized_cache])
                    n += 1
        self.assertGreater(n, 0)


class Blending(unittest.TestCase):
    def test_capture_ao_intensity_blend(self):
        # scene 0.2 (weight 1) then DHH 0.425532 at weight 0.8914894 -> captured AdditionalParams.y 0.401059389
        blender = VolumeBlender()
        a = EffectSettings('AmbientOcclusion')
        a.set('intensity', F(0.2))
        d = EffectSettings('AmbientOcclusion')
        d.set('intensity', F(0.425532))
        vols = [Volume('scene', 8, F(0), F(1), True, True, [a]), Volume('dhh', 8, F(1000), F(0.8914894), True, True, [d])]
        blender.update(256, vols, frame=1)
        self.assertEqual(bits(blender.layer['AmbientOcclusion']['intensity']).hex(), bits(0.401059389).hex())

    def test_inactive_settings_and_layer_mask(self):
        blender = VolumeBlender()
        a = EffectSettings('Bloom', active=False)
        a.set('intensity', F(3))
        b = EffectSettings('Bloom')
        b.set('intensity', F(5))
        blender.update(256, [Volume('x', 8, 0, 1, True, True, [a]), Volume('y', 9, 0, 1, True, True, [b])], frame=1)
        self.assertEqual(blender.layer['Bloom']['intensity'], F(0))

    def test_priority_sort_is_stable(self):
        v = [Volume(n, 8, p, 1, True, True) for n, p in (('a', 0), ('b', 1000), ('c', 0))]
        self.assertEqual([x.name for x in sort_by_priority(v)], ['a', 'c', 'b'])

    def test_bool_enum_interp_takes_to_when_weight_positive(self):
        p = Param('enum', True, 1)
        p.interp(Param('enum', True, 1), Param('enum', True, 2), F(0.1), 0)
        self.assertEqual(p.value, 2)

    def test_texture_lut2d_default(self):
        tex = TextureValue('lut', 1024, 32)
        self.assertIs(texture_interp(None, tex, F(1), 4), tex)
        with self.assertRaises(NotImplementedError):
            texture_interp(None, tex, F(0.5), 4)

    def test_spline_interp_writes_shared_base_spline(self):
        blender = VolumeBlender()
        base_obj = blender.base['ColorGrading'].params['lumVsSatCurve'].value
        s = EffectSettings('ColorGrading')
        s.override('lumVsSatCurve', SplineValue.new([], 0.25, False))
        blender.update(256, [Volume('v', 8, 0, F(1), True, True, [s])], frame=7)
        self.assertIs(blender.layer['ColorGrading'].params['lumVsSatCurve'].value, base_obj)
        self.assertEqual(base_obj.cached[0], F(0.25))


class DHHSettings(unittest.TestCase):
    def test_get_data_with_tag_semantics(self):
        text = '?tag=GSfloat_1&=2.5&&?tag=GSfloat_10&=7&&?tag=GSfloat_2&=bad&&?tag=GSbool_0&=True&&?tag=GSvector3_0&=(1.0, 2.0, 3.0)&&'
        cfg = parse_dhh_graphic_setting(text)
        self.assertEqual(cfg['floats'][1], F(2.5))     # first IndexOf hit is the real GSfloat_1
        self.assertEqual(cfg['floats'][10], F(7))
        self.assertEqual(cfg['floats'][2], F(0))       # float.TryParse failure -> 0
        self.assertEqual(cfg['floats'][46], F(80))     # SetDefault backup for a missing tag
        self.assertTrue(cfg['bools'][0])
        self.assertEqual(cfg['vectors'][0], (F(1), F(2), F(3)))


class UnityMappings(unittest.TestCase):
    def test_rt_formats(self):
        k, res, srv, rtv, uav = RTDescriptor(8, 4, 'ARGBHalf', srgb=True).d3d()
        self.assertEqual((res['Format'], srv['Format'], res['BindFlags'], uav), (9, 10, 40, None))
        k, res, srv, rtv, uav = RTDescriptor(8, 4, 'RHalf', srgb=False, enable_random_write=True).d3d()
        self.assertEqual((res['Format'], uav['Format'], res['BindFlags']), (39, 41, 168))
        k, res, srv, rtv, uav = RTDescriptor(8, 4, 'RHalf', srgb=False, dimension='Tex2DArray', volume_depth=16,
                                             enable_random_write=True).d3d()
        self.assertEqual((res['ArraySize'], srv['ViewDimension'], uav['Texture2DArray']['ArraySize']), (16, 5, 0xFFFFFFFF))
        k, res, srv, rtv, uav = RTDescriptor(33, 33, 'ARGBHalf', srgb=False, dimension='Tex3D', volume_depth=33,
                                             enable_random_write=True).d3d()
        self.assertEqual((k, res['Depth'], uav['ViewDimension']), ('tex3d', 33, 8))

    def test_sampler_rules(self):
        t = TextureRef(1, 2, 4, 4, filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=1)
        self.assertEqual((sampler_desc(t, 'ForceEnable')['Filter'], sampler_desc(t, 'ForceEnable')['MaxAnisotropy']), (85, 9))
        t.aniso_level = 0
        self.assertEqual(sampler_desc(t, 'ForceEnable')['Filter'], 20)
        t.filter_mode, t.aniso_level = 'Point', 1
        self.assertEqual((sampler_desc(t, 'ForceEnable')['Filter'], sampler_desc(t, 'ForceEnable')['MaxAnisotropy']), (0, 1))
        t.wrap_mode = 'Repeat'
        self.assertEqual(sampler_desc(t)['AddressU'], 1)

    def test_pass_states(self):
        v = lambda x: {'val': float(x), 'name': '<noninit>'}  # noqa: E731
        st = {'rtBlend0': {'srcBlend': v(0), 'destBlend': v(6), 'srcBlendAlpha': v(0), 'destBlendAlpha': v(10),
                           'blendOp': v(0), 'blendOpAlpha': v(0), 'colMask': v(15)}, 'zTest': v(8), 'zWrite': v(0)}
        rt = blend_desc(st)['RenderTarget'][0]
        self.assertEqual((rt['BlendEnable'], rt['SrcBlend'], rt['DestBlend'], rt['DestBlendAlpha']), (True, 1, 4, 6))
        self.assertEqual((depth_desc(st)['DepthFunc'], depth_desc(st)['DepthWriteMask']), (8, 0))

    def test_cbuffer_pool_keeps_residue(self):
        pool = UnityConstantBufferPool()
        pool.seed('$Globals', 32, bytes(range(32)))
        pool.write('$Globals', 32, 16, b'\0' * 4)
        self.assertEqual(pool.snapshot('$Globals', 32), bytes(range(16)) + b'\0' * 4 + bytes(range(20, 32)))


class ColorUtilitiesCapture(unittest.TestCase):
    def test_hue_sat_con_and_post_exposure(self):
        # retry capture Lut3DBaker _HueSatCon.y/z and Uber _PostExposure
        w = F(0.8914894)
        sat = r4(dv(14) + (dv(8.936147) - dv(14)) * dv(w))
        con = r4(dv(40) + (dv(-3.752232e-05) - dv(40)) * dv(w))
        self.assertEqual(bits(r4(dv(sat) / 100.0 + 1.0)).hex(), bits(1.09485626).hex())
        self.assertEqual(bits(r4(dv(con) / 100.0 + 1.0)).hex(), bits(1.04340386).hex())
        pe = r4(dv(-0.2) + (dv(0.4893677) + 0.2) * dv(w))
        self.assertEqual(bits(ppv2.exp2(pe)).hex(), bits(1.33289576).hex())


INTEGRATION = all(p.exists() for p in (SCENE_STATE, RETRY_ORACLE, COMPUTE, SHADERS, RETRY_REQUEST))


@unittest.skipUnless(INTEGRATION, 'oracle/scene/compute exports absent')
class CaptureDiff(unittest.TestCase):
    def test_retry_capture_all_16_renders_byte_exact(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ppv2_diff
        with tempfile.TemporaryDirectory() as tmp:
            args = type('A', (), dict(oracle=RETRY_ORACLE, scene_state=SCENE_STATE,
                                      dhh=Path(r'E:\HoneySelect2_ArcticFox\DHH_Data\GraphicSetting.dhs'),
                                      native_request=RETRY_REQUEST, shader_manifest=[SHADERS], compute_manifest=COMPUTE,
                                      fov=23.0, projection_m00=6.553542613983154, out=Path(tmp) / 'd'))
            oracle, b, segments, gen, external, cfg, vols = ppv2_diff.build(args)
            ppv2_diff.attach_layout_ranges(gen, segments)
            report, summary = ppv2_diff.compare(oracle, b, segments)
        self.assertEqual(summary['mismatches'], 0, [r for r in report if r.get('fail')][:3])
        self.assertEqual(summary['actions'], 16 * 26)

    @unittest.skipUnless(PAIRED_ORACLE.exists() and (PAIRED_FRAMES / 'scene_005.json').exists(), 'paired capture absent')
    def test_paired_capture_frame_byte_exact_with_scene_state(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import ppv2_diff_paired
        with tempfile.TemporaryDirectory() as tmp:
            args = type('A', (), dict(oracle=PAIRED_ORACLE, scene_state=SCENE_STATE, scene=PAIRED_FRAMES / 'scene_005.json',
                                      prev_scene=PAIRED_FRAMES / 'scene_004.json', prev_prev_scene=PAIRED_FRAMES / 'scene_003.json',
                                      shader_manifest=[SHADERS], compute_manifest=COMPUTE, out=Path(tmp) / 'd'))
            doc = ppv2_diff_paired.run(args)
        self.assertEqual(doc['summary']['mismatches'], 0)
        for f in doc['frames']:
            self.assertEqual(f['blend_vs_reflected']['fail'], 0)
            self.assertTrue(f['persistent_vs_reflected']['dithering_equal'])
            self.assertEqual(f['persistent_vs_reflected']['m_pixels_mismatch_count'], 0)


if __name__ == '__main__':
    unittest.main()
