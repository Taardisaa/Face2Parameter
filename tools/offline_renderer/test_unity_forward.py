"""Tests for the Unity built-in forward generator (src/render/hs2pipe).

Fixture bytes marked ``capture`` are cbuffer variables of the 2026-10-12 retry
capture (unaccepted_retry_frame17675.rdc, extracted by
compare_generated_cbuffers.py): they are engine outputs used as regression
oracles for the transcribed float32 rules.  Inputs never include the value
under test.
"""
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import scene_state as ss  # noqa: E402
from src.render.hs2pipe import unity_forward as uf  # noqa: E402
from src.render.hs2pipe import unity_math as um  # noqa: E402
from src.render.hs2pipe import unity_shadows as ush  # noqa: E402
from tools.offline_renderer import compare_generated_cbuffers as cgc  # noqa: E402

F = np.float32

# capture: capture camera of segment 0 (1086x1448), events 865/1164/1580/1615
CAP_V = '000080bf000000000000000000000000000000000000803f000000000000000000000000000000000000803f000000002b9bbe3d83a964c1dc0ff6c00000803f'
CAP_VP = ('9fb6d1c000000000000000000000000000000000f7489dc00000000000000000e70935bad42a173a00e0a738000080bf'
          'ac801d3fdd7a8c42ae97a23ddc0ff640')
CAP_INV_V = '000080bf000000000000008000000000000000000000803f000000800000000000000080000000800000803f000000002b9bbe3d83a96441dc0ff6400000803f'
CAP_CAM_INV_PROJ = ('5a401c3e00000000000000800000000000000000cd55503e000000800000000008f6dcacd800f6ac82f57fb3e7fbc7c0'
                    '16ffdcb8ed0af6b8000080bf1a04c840')
CAP_CAM_TO_WORLD = '000080bf000000000000000000000000000000000000803f00000000000000000000000000000000000080bf000000002b9bbe3d83a96441dc0ff6400000803f'
CAP_QUAD_VP = ('0000004000000000000000000000000000000000000000c000000000000000000000000000000000c337223c00000000'
               '000080bf0000803f21777d3f0000803f')
CAP_ZBUFFER = '004c43460000803fe8fb47410020833a'
CAP_PROJ_PARAMS = '000080bf0ad7a33d00007a446f12833a'
CAP_NONJIT_VP = ('9fb6d1c000000000000000000000000000000000f7489dc00000000000000000000000000000008000e0a738000080bf'
                 'a7241c3f227d8c42ae97a23ddc0ff640')
# capture: o_head forward draw 1338
CAP_O2W = ('e6f37f3f0cfe97bcebcda3bb0000000048cd983cd1e67f3feefaa73c000000005746973b88b6a8bc67f17f3f00000000'
           '9dcbd03db132674125c57abd0000803f')
CAP_W2O = ('e8f37f3f4acd983c5846973b000000000cfe97bcd3e67f3f89b6a8bc00000000ebcda3bbeefaa73c68f17f3f00000000'
           '91d7293e9a1e67c14877b73e0000803f')
CAP_SHAR = '961035ae3f4d433000000000b963be3e'
CAP_SHBR = '000000006368ad30c0a2643400000000'
CAP_SHC = '66529cb366529cb366529cb30000803f'
CAP_LIGHT_SHADOW_DATA = '30de1c3e55550541abaa2a3d000080c0'
# other-session export (runtime_audit_hook_v3/runtime_fields_completion.json): Maker ambient probe
AMBIENT_PROBE = [0.3317869, 6.339461e-10, 0.0, -3.67333247e-11, 0.0, 1.12575727e-09, 6.33299351e-08, 0.0, -6.494952e-08] * 3

GRAPHICS = {'reversed_z': True, 'uses_opengl_texture_coords': False}


def capture_camera(jitter=True):
    view = um.from_bytes(bytes.fromhex(CAP_V))
    vp = um.from_bytes(bytes.fromhex(CAP_VP))
    aspect = F(1086) / F(1448)
    base = um.set_perspective(23.0, aspect, 0.08, 1000.0)
    proj = list(base)
    if jitter:
        proj[8], proj[9] = vp[8], -vp[9]
    return ss.CameraRender({'implicit_projection': False, 'projection_matrix': [float(x) for x in proj],
                            'non_jittered_projection_matrix': [float(x) for x in base],
                            'world_to_camera_matrix': [float(x) for x in view], 'render_into_texture': True,
                            'near_clip': 0.08, 'far_clip': 1000.0, 'field_of_view': 23.0, 'aspect': float(aspect),
                            'orthographic': False, 'orthographic_size': 5.0,
                            'position': [0.0930693969130516, 14.29138469696045, 7.689435958862305],
                            'target': {'pixel_width': 1086, 'pixel_height': 1448}})


class UnityMathTests(unittest.TestCase):
    def test_perspective_matches_float32_path(self):
        p = um.set_perspective(23.0, F(1086) / F(1448), 0.08, 1000.0)
        self.assertEqual(p[11], F(-1.0))
        self.assertEqual(p[15], F(0.0))
        # m22/m23 follow (n+f)/(n-f) and (2n*f)/(n-f) in float32
        self.assertEqual(p[10], (F(0.08) + F(1000.0)) / (F(0.08) - F(1000.0)))
        self.assertEqual(p[14], ((F(0.08) + F(0.08)) * F(1000.0)) / (F(0.08) - F(1000.0)))

    def test_view_projection_bit_exact_against_capture(self):
        cam = capture_camera()
        setup = uf.camera_setup(cam, GRAPHICS)
        self.assertEqual(uf.pack(setup.view_projection).hex(), CAP_VP)
        self.assertEqual(uf.pack(setup.inverse_view).hex(), CAP_INV_V)
        self.assertEqual(uf.pack(setup.camera_inverse_projection).hex(), CAP_CAM_INV_PROJ)
        self.assertEqual(uf.pack(setup.camera_to_world).hex(), CAP_CAM_TO_WORLD)
        self.assertEqual(uf.pack(setup.non_jittered_vp).hex(), CAP_NONJIT_VP)

    def test_camera_params_bit_exact_against_capture(self):
        cam = capture_camera()
        props = uf.camera_shader_props(cam, uf.camera_setup(cam, GRAPHICS), GRAPHICS)
        self.assertEqual(uf.pack(props['_ZBufferParams']).hex(), CAP_ZBUFFER)
        self.assertEqual(uf.pack(props['_ProjectionParams']).hex(), CAP_PROJ_PARAMS)

    def test_explicit_unjittered_projection_takes_matrix_branch(self):
        cam = capture_camera(jitter=False)
        props = uf.camera_shader_props(cam, uf.camera_setup(cam, GRAPHICS), GRAPHICS)
        # SetCameraShaderProps derives near/far from the planes of an explicit
        # diagonal projection, which differs from the clip planes in float32.
        self.assertNotEqual(uf.pack(props['_ProjectionParams']).hex(), CAP_PROJ_PARAMS)

    def test_world_to_object_general3d_against_capture(self):
        ok, w2o = um.invert_general_3d(um.from_bytes(bytes.fromhex(CAP_O2W)))
        self.assertTrue(ok)
        self.assertEqual(um.to_bytes(w2o).hex(), CAP_W2O)

    def test_general3d_singular_returns_zero(self):
        ok, out = um.invert_general_3d([F(0.0)] * 16)
        self.assertFalse(ok)
        self.assertEqual(out, [F(0.0)] * 16)

    def test_invert_full_identity_and_singular(self):
        ok, out = um.invert_full(um.identity())
        self.assertTrue(ok)
        self.assertEqual(out, um.identity())
        ok, out = um.invert_full([F(0.0)] * 16)
        self.assertFalse(ok)

    def test_multiply_column_order(self):
        a = um.set_scale(2.0, 3.0, 4.0)
        b = um.identity()
        b[12], b[13], b[14] = F(1.0), F(1.0), F(1.0)
        out = um.multiply(a, b)            # a*b: translate then scale
        self.assertEqual(out[12:15], [F(2.0), F(3.0), F(4.0)])

    def test_device_projection_rows(self):
        p = um.set_ortho(0, 1, 0, 1, -1, 100)
        dev = um.calculate_device_projection(p, False, True, True)
        self.assertEqual(struct.unpack('<f', um.to_bytes([dev[1]]))[0], 0.0)
        self.assertEqual(um.to_bytes([dev[1]]).hex(), '00000080')      # Y-row negation keeps -0
        # capture: Internal-ScreenSpaceShadows quad VP (event 1164) = devProj(LoadOrtho)·identity view;
        # the multiply turns the -0 entries into +0 exactly as recorded.
        vp = um.multiply(dev, um.identity())
        self.assertEqual(uf.pack(vp).hex(), CAP_QUAD_VP)


class LightingTests(unittest.TestCase):
    def test_sh_constants_from_ambient_probe(self):
        rs = {'ambient_mode': 'Skybox', 'has_skybox_material': True, 'ambient_intensity': 1.053189,
              'ambient_probe': AMBIENT_PROBE}
        sh = uf.sh_shader_constants(uf.final_ambient_probe(rs))
        self.assertEqual(uf.pack(sh['unity_SHAr']).hex(), CAP_SHAR)
        self.assertEqual(uf.pack(sh['unity_SHBr']).hex(), CAP_SHBR)
        self.assertEqual(uf.pack(sh['unity_SHC']).hex(), CAP_SHC)

    def test_gamma_to_linear_branches(self):
        self.assertEqual(uf.gamma_to_linear_space(0.0), F(0.0))
        self.assertEqual(uf.gamma_to_linear_space(1.0), F(1.0))
        self.assertEqual(uf.gamma_to_linear_space(0.04), F(0.04) / F(12.92))

    def test_trilight_is_explicit_gap(self):
        with self.assertRaises(uf.GeneratorGap):
            uf.final_ambient_probe({'ambient_mode': 'Trilight', 'ambient_probe': AMBIENT_PROBE})

    def test_light_shadow_data(self):
        lsd = uf.light_shadow_data({'shadow_strength': 0.8468087}, F(120.0), F(1000.0))
        self.assertEqual(uf.pack(lsd).hex(), CAP_LIGHT_SHADOW_DATA)

    def test_time_vector(self):
        t = uf.time_shader_props({'shader_time': 239.123978, 'delta_time': 0.001, 'smooth_delta_time': 0.5})
        self.assertEqual(t['_Time'][1], F(239.123978))
        self.assertEqual(t['unity_DeltaTime'][0], F(0.005))      # clamped low
        self.assertEqual(t['unity_DeltaTime'][2], F(0.2))        # clamped high


class ShadowTests(unittest.TestCase):
    def test_shadow_map_size(self):
        g = {'video_memory_mb': 4096, 'max_render_texture_size': 16384}
        q = {'shadow_resolution': 'VeryHigh'}
        light = {'type': 'Directional', 'shadow_resolution': -1, 'shadow_custom_resolution': 0}
        self.assertEqual(uf.shadow_map_size(g, q, light, 1086, 1448), 4096)
        self.assertEqual(uf.shadow_map_size(g, q, light, 543, 724), 4096)
        self.assertEqual(uf.shadow_map_size(g, {'shadow_resolution': 'Low'}, light, 1086, 1448), 1024)
        self.assertEqual(uf.shadow_map_size(g, q, dict(light, shadow_custom_resolution=1000), 10, 10), 1024)

    def test_split_distances(self):
        dist, pct = uf.cascade_split_distances(0.08, 120.0, 1, [0.0, 0.0, 0.0])
        self.assertEqual(dist[:2], [F(0.08), F(120.0)])
        self.assertEqual(pct[:2], [F(0.0), F(1.0)])
        self.assertEqual(dist[2], F(120.0) * F(1.01))
        dist4, pct4 = uf.cascade_split_distances(0.1, 100.0, 4, [0.0666666701, 0.200000003, 0.466666669])
        self.assertEqual(dist4[4], F(100.0))
        self.assertEqual(dist4[1], F(0.0666666701) * (F(100.0) - F(0.1)) + F(0.1))

    def test_frustum_portion_endpoints(self):
        pts = [(F(x), F(y), F(z)) for x, y, z in ush.FRUSTUM_CORNERS]
        portion = ush.get_frustum_portion(pts, 0.0, 1.0)
        self.assertEqual(portion, pts)

    def test_sphere_from_4_points(self):
        pts = [(F(1), F(0), F(0)), (F(-1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))]
        c, r = ush.calculate_sphere_from_4_points(pts)
        self.assertTrue(all(abs(float(v)) < 1e-6 for v in c))
        self.assertAlmostEqual(float(r), 1.0, places=6)

    def test_determinant(self):
        self.assertEqual(ush.get_determinant(um.set_scale(2, 3, 4)), F(24.0))

    def test_stable_fit_snaps_to_texels(self):
        aspect = F(1.0)
        proj = um.set_perspective(60.0, aspect, 0.3, 100.0)
        view = um.identity()
        ok, ctw = um.invert_full(um.multiply(proj, view))
        ok2, c2w = um.invert_general_3d(um.multiply(um.set_scale(1, 1, -1), view))
        axes = ((F(1), F(0), F(0)), (F(0), F(0.6), F(0.8)), (F(0), F(-0.8), F(0.6)))
        inp = ush.CascadeInputs(projection_type='StableFit', clip_to_world=ctw, camera_near=0.3, camera_far=100.0,
                                shadow_distance=50.0, split_percentages=[0.0, 1.0, 1.01, 1.0201, 1.030301],
                                cascade_count=1, near_plane_offset=2.0, reversed_z=True, light_axes=axes,
                                caster_bounds=((-5, -5, -60), (5, 5, 0)), camera_projection=proj,
                                camera_to_world=c2w, camera_orthographic=False, camera_fov=60.0)
        res = ush.directional_cascade(inp, 0, 1024, 1024)
        self.assertTrue(res.ok)
        # Sphere fit: x/y extents are both 2r up to float32 rounding of (l+r)-(l-r).
        self.assertAlmostEqual(float(res.texel_size[0]), float(res.texel_size[1]), places=6)
        self.assertLessEqual(float(res.split_far), 50.0)
        # The texture-space shadow matrix maps the sphere centre inside [0,1].
        wc = res.sphere[0]
        p = [((F(wc[1]) * res.world_to_shadow[4 + i] + F(wc[0]) * res.world_to_shadow[i]) + F(wc[2]) * res.world_to_shadow[8 + i]) + res.world_to_shadow[12 + i] for i in range(3)]
        self.assertTrue(all(F(0.0) < v < F(1.0) for v in p))

    def test_cascade_without_casters_is_identity(self):
        inp = ush.CascadeInputs('StableFit', um.identity(), 0.1, 10, 10, [0, 1, 1, 1, 1], 1, 0, True,
                                ((1, 0, 0), (0, 1, 0), (0, 0, 1)), None)
        res = ush.directional_cascade(inp, 0, 512, 512)
        self.assertFalse(res.ok)
        self.assertEqual(res.projection, um.set_ortho(-1, 1, -1, 1, 0.1, 10.0))

    def test_cascade_params(self):
        w2s = [um.identity()] * 4
        out = uf.cascade_shader_params(w2s, 1, [F(0.1), F(10), F(10.1), F(10.2), F(10.3)], [[F(0)] * 4] * 4)
        self.assertTrue(out['keyword_SHADOWS_SINGLE_CASCADE'])
        self.assertEqual(out['unity_ShadowCascadeScales'][1], F(1.0))

    def test_bias(self):
        dev = um.calculate_device_projection(um.set_ortho(-1, 1, -1, 1, -20.0, 25.0), False, True, True)
        b = uf.light_shadow_bias({'shadow_bias': 0.04, 'shadow_normal_bias': 0.4}, dev, 0.005, 0.006)
        self.assertEqual(b[1], F(1.0))
        self.assertEqual(b[2], F(0.4) * F(0.006))
        self.assertEqual(b[0], -(dev[10] * F(0.04)))


class SceneStateTests(unittest.TestCase):
    def test_validate_reports_missing(self):
        problems = ss.validate({'schema': ss.SCHEMA, 'version': ss.VERSION})
        self.assertIn('cameras', problems)
        self.assertIn('graphics.reversed_z', problems)

    def test_float_bits_literal(self):
        self.assertEqual(ss.f32_value('0x3f800000'), F(1.0))
        with self.assertRaises(ss.SceneStateError):
            ss.f32_value(True)

    def test_field_table(self):
        md = ss.field_table_markdown()
        self.assertIn('Camera.projectionMatrix', md)


class HarnessTests(unittest.TestCase):
    def test_compare_exact_and_ulps(self):
        captured = {'draws': [{'event_id': 7, 'stages': {'VS': {'cbuffers': [
            {'name': 'UnityPerCamera', 'variables': {'_ProjectionParams': {'hex': CAP_PROJ_PARAMS}}}]}}}]}
        gen_ok = {'records': [{'events': [7], 'stage': 'VS', 'cbuffer': 'UnityPerCamera',
                               'variables': {'_ProjectionParams': {'hex': CAP_PROJ_PARAMS}}}]}
        self.assertEqual(cgc.compare(gen_ok, captured)['totals']['equal'], 1)
        bad = uf.pack([F(-1.0), F(0.08), np.nextafter(F(1000.0), F(2000.0)), F(0.001)]).hex()
        gen_bad = {'records': [{'events': [7], 'stage': 'VS', 'cbuffer': 'UnityPerCamera',
                                'variables': {'_ProjectionParams': {'hex': bad}}}]}
        rep = cgc.compare(gen_bad, captured)
        self.assertEqual(rep['totals']['different'], 1)
        diff = rep['records'][0]['variables']['_ProjectionParams']['different'][0]
        self.assertEqual(diff['max_ulps'], 1)


# paired capture rdc_04d1e76f (frame 26060, 3200x1800 main camera), scene export
# window_04d1e76f/frames/scene_005.json: matrices exported row-major (m00, m01, ...).
PAIRED_PROJ = [2.76477575, 0, -0.000261718757, 0, 0, 4.915157, -0.0003138889, 0, 0, 0, -1.0001601, -0.1600128, 0, 0, -1, 0]
PAIRED_NJ = [2.76477575, 0, 0, 0, 0, 4.915157, 0, 0, 0, 0, -1.0001601, -0.1600128, 0, 0, -1, 0]
PAIRED_W2C = [-1, 2.220446e-16, 8.74227837e-08, -1.13686838e-13, -4.006007e-09, 0.9989495, -0.0458233841, -12.9863434,
              8.733095e-08, 0.0458233841, 0.998949647, -22.41863, 0, 0, 0, 1]
PAIRED_VP = ('16f230c0635fa93228f7f52cb38abbb3da3449b78d1e9dc0c92976364db13bbdf6f188b9a9f4663edcb2a7382abb7fbf'
             '1e43c03ba04a7f42832da03d5b59b341')
PAIRED_SKY_VP = ('16f230c0635fa9325e4347a9b38abbb3da3449b78d1e9dc0626cc7b24db13bbdf6f188b9a9f4663e6edb07b52abb7fbf'
                 '1e43c03ba04a7f424cb1543a5b59b341')
PAIRED_SKY_O2W = ('00401c460000000000000000000000000000000000401c460000000000000000000000000000000000401c4600000000'
                  '5dcbff35000060416766ae410000803f')


class PairedCaptureTests(unittest.TestCase):
    def camera(self):
        rm = cgc.unity_row_major_to_memory
        return ss.CameraRender({'implicit_projection': False, 'projection_matrix': rm(PAIRED_PROJ),
                                'non_jittered_projection_matrix': rm(PAIRED_NJ), 'world_to_camera_matrix': rm(PAIRED_W2C),
                                'render_into_texture': True, 'near_clip': 0.08, 'far_clip': 1000.0, 'field_of_view': 23.0,
                                'aspect': 1.77777779, 'orthographic': False, 'orthographic_size': 5.0,
                                'position': [1.9058167e-06, 14.0, 21.8000011],
                                'target': {'pixel_width': 3200, 'pixel_height': 1800}})

    def test_view_projection(self):
        setup = uf.camera_setup(self.camera(), GRAPHICS)
        self.assertEqual(uf.pack(setup.view_projection).hex(), PAIRED_VP)

    def test_skybox(self):
        cam = self.camera()
        sk = uf.skybox_matrices(cam, uf.camera_setup(cam, GRAPHICS), GRAPHICS)
        self.assertEqual(uf.pack(sk['unity_MatrixVP']).hex(), PAIRED_SKY_VP)
        self.assertEqual(uf.pack(sk['unity_ObjectToWorld']).hex(), PAIRED_SKY_O2W)

    def test_adjust_depth_range(self):
        m = uf.adjust_depth_range(um.set_perspective(60, 1, 1, 10), 2.0, 20.0)
        self.assertEqual(m[10], -((F(2) + F(20)) / (F(20) - F(2))))


class PlanTests(unittest.TestCase):
    def test_draw_cbuffers_keeps_shadow_copy_bytes(self):
        from src.render.hs2pipe import shader_library as sl
        from src.render.hs2pipe import unity_forward_plan as up
        full = sl.CBufferLayout('UnityPerDraw', 176, [sl.Variable('unity_ObjectToWorld', 0, 'float', 4, 4, 0, True),
                                                       sl.Variable('unity_WorldToObject', 64, 'float', 4, 4, 0, True)], slot=1)
        o2w_only = sl.CBufferLayout('UnityPerDraw', 176, [sl.Variable('unity_ObjectToWorld', 0, 'float', 4, 4, 0, True)], slot=1)
        copies = {}
        r1 = {'object_to_world': [float(x) for x in um.from_bytes(bytes.fromhex(CAP_O2W))]}
        d1 = up.PlannedDraw('opaque', r1, 0, {}, 'FORWARDBASE', (), uf.per_draw_values(r1))
        out1, miss1 = up.draw_cbuffers(d1, {}, [full], copies, sl.pack_cbuffer)
        self.assertEqual(out1[1]['bytes'][64:128].hex(), CAP_W2O)
        r2 = {'object_to_world': [float(x) for x in um.identity()]}
        d2 = up.PlannedDraw('opaque', r2, 0, {}, 'FORWARDBASE', (), uf.per_draw_values(r2))
        out2, _ = up.draw_cbuffers(d2, {}, [o2w_only], copies, sl.pack_cbuffer)
        # Only O2W is written; the previous draw's W2O stays in the shared copy
        # (the "stale W2O" bytes of the capture).
        self.assertEqual(out2[1]['bytes'][0:64], um.to_bytes(um.identity()))
        self.assertEqual(out2[1]['bytes'][64:128].hex(), CAP_W2O)


if __name__ == '__main__':
    unittest.main()
