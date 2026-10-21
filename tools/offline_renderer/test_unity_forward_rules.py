"""Analytic checks for src/render/hs2pipe/unity_forward_rules.py.

Run: .venv/Scripts/python.exe -m unittest tools.offline_renderer.test_unity_forward_rules -v
"""
import functools
import math
import random
import unittest

import numpy as np

from src.render.hs2pipe import unity_forward_rules as R

F = np.float32
IDENT = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def ctx_persp(pos=(0.0, 0.0, 0.0)):
    w2c = list(IDENT)
    w2c[12], w2c[13], w2c[14] = -pos[0], -pos[1], -pos[2]
    return {'world_to_camera': w2c, 'position': pos, 'sort_mode': R.SORT_PERSPECTIVE, 'sort_axis': (0, 0, 1)}


def node(center, **kw):
    n = {'center': center, 'layering': R.layering_key(0, 0), 'materials': [
        {'shader_id': kw.pop('shader_id', 10), 'material_instance_id': kw.pop('mat_id', 100),
         'custom_render_queue': kw.pop('queue', 2000), 'shader_render_queue': 2000}]}
    n.update(kw)
    return n


def build(nodes, ctx, passes_per_ro=1):
    opaque, transparent = R.extract_render_objects(nodes, ctx)
    def entries(ros):
        out = []
        for i, _ in enumerate(ros):
            for p in range(passes_per_ro):
                out.append({'ro': i, 'data': (R.PASS_FORWARD_BASE << 8) | p | (R.FIRST_PASS_FLAG if p == 0 else 0)})
        return out
    return opaque, transparent, entries(opaque), entries(transparent)


CAM_F2B = {'opaque_sort_mode': R.OPAQUE_SORT_DEFAULT}
CAM_NODIST = {'opaque_sort_mode': R.OPAQUE_SORT_NONE}


class SortDistanceTests(unittest.TestCase):
    def test_layering_key(self):
        self.assertEqual(R.layering_key(0, 0), 0x80008000)
        self.assertEqual(R.layering_key(1, -1), 0x80017FFF)
        self.assertLess(R.layering_key(-1, 32767), R.layering_key(0, -32768))

    def test_resolve_transparency_sort(self):
        gs = {'transparency_sort_mode': 0, 'transparency_sort_axis': (0, 0, 1)}
        cam = {'implicit_transparency_sort': True, 'orthographic': True,
               'transparency_sort_mode': 3, 'transparency_sort_axis': (1, 0, 0)}
        self.assertEqual(R.resolve_transparency_sort(cam, gs)[0], R.SORT_ORTHOGRAPHIC)
        cam['implicit_transparency_sort'] = False
        self.assertEqual(R.resolve_transparency_sort(cam, gs), (3, (F(1), F(0), F(0))))

    def test_perspective_distance_is_negative_squared(self):
        d, along = R.sort_distances((3.0, 4.0, 0.0), 0.0, ctx_persp())
        self.assertEqual(d, F(-25.0))
        self.assertEqual(along, F(-0.0))
        self.assertEqual(R.sort_distances((3.0, 4.0, 0.0), 1.0, ctx_persp())[0], F(-36.0))
        # negative effective distance flips the sign of the squared value
        self.assertEqual(R.sort_distances((3.0, 4.0, 0.0), -10.0, ctx_persp())[0], F(25.0))

    def test_orthographic_and_custom_axis(self):
        ctx = ctx_persp()
        ctx['sort_mode'] = R.SORT_ORTHOGRAPHIC
        self.assertEqual(R.sort_distances((1.0, 2.0, -7.0), 0.5, ctx), (F(-7.5), F(7.0)))
        ctx['sort_mode'] = R.SORT_CUSTOM_AXIS
        ctx['sort_axis'] = (0.0, 1.0, 0.0)
        self.assertEqual(R.sort_distances((1.0, 2.0, -7.0), 0.5, ctx)[0], F(-2.5))

    def test_bucket_switch(self):
        self.assertTrue(R.opaque_distance_bucket_enabled({'opaque_sort_mode': 0}, False))
        self.assertFalse(R.opaque_distance_bucket_enabled({'opaque_sort_mode': 0}, True))
        self.assertTrue(R.opaque_distance_bucket_enabled({'opaque_sort_mode': 1}, True))
        self.assertFalse(R.opaque_distance_bucket_enabled({'opaque_sort_mode': 2}, False))


class ExtractionTests(unittest.TestCase):
    def test_queue_split_and_shadows_only(self):
        nodes = [node((0, 0, -1), queue=2500), node((0, 0, -2), queue=2501),
                 node((0, 0, -3), shadow_casting_mode=3)]
        opaque, transparent = R.extract_render_objects(nodes, ctx_persp())
        self.assertEqual([r['node'] for r in opaque], [0])
        self.assertEqual([r['node'] for r in transparent], [1])

    def test_static_batch_subset_clamp(self):
        n = node((0, 0, -1), first_subset=5, static_batch_subset_count=2)
        n['materials'] = n['materials'] * 3
        opaque, _ = R.extract_render_objects([n], ctx_persp())
        self.assertEqual([r['subset'] for r in opaque], [5, 6, 6])
        self.assertEqual([r['material_index'] for r in opaque], [0, 1, 2])

    def test_material_key(self):
        self.assertEqual(R.material_key(3, 0x123456, False), (3 << 20) | 0x23456)
        self.assertEqual(R.material_key(3, 0x123456, True), 3 << 20)
        self.assertEqual(R.material_key(-1, 0, True), 0xFFF00000)

    def test_pass_filter(self):
        passes = [{'type': R.PASS_FORWARD_BASE}, {'type': R.PASS_FORWARD_ADD},
                  {'type': R.PASS_VERTEX}, {'type': 8}]
        e = R.forward_pass_entries(0, passes, vertex_lit_subshader=False, lightmapped=False,
                                   add_light_count=0, render_options=0, multi_pass=False)
        self.assertEqual([x['data'] for x in e], [(4 << 8) | 0 | R.FIRST_PASS_FLAG])
        e = R.forward_pass_entries(0, passes, vertex_lit_subshader=False, lightmapped=False,
                                   add_light_count=1, render_options=0, multi_pass=True)
        self.assertEqual([x['data'] for x in e],
                         [(4 << 8) | R.MULTI_PASS_FLAG | R.FIRST_PASS_FLAG, (5 << 8) | 1 | R.MULTI_PASS_FLAG])
        e = R.forward_pass_entries(0, [{'type': 1}, {'type': 2}], vertex_lit_subshader=True,
                                   lightmapped=True, add_light_count=0, render_options=0, multi_pass=False)
        self.assertEqual([x['data'] & 0xFF for x in e], [1])


class SortTests(unittest.TestCase):
    def order(self, nodes, cam, opaque=True, hsr=False, passes=1):
        o, t, eo, et = build(nodes, ctx_persp(), passes)
        ros, ents = (o, eo) if opaque else (t, et)
        s = R.sort_draws(ents, cam, ros, nodes, opaque=opaque, has_hidden_surface_removal_on_gpu=hsr)
        return [(ros[e['ro']]['node'], e['data'] & 0xFF) for e in s]

    def test_opaque_front_to_back(self):
        nodes = [node((0, 0, -9.0)), node((0, 0, -2.0)), node((0, 0, -5.0))]
        self.assertEqual(self.order(nodes, CAM_F2B), [(1, 0), (2, 0), (0, 0)])

    def test_transparent_back_to_front(self):
        nodes = [node((0, 0, -2.0), queue=3000), node((0, 0, -9.0), queue=3000), node((0, 0, -5.0), queue=3000)]
        self.assertEqual(self.order(nodes, CAM_F2B, opaque=False), [(1, 0), (2, 0), (0, 0)])

    def test_queue_before_distance(self):
        nodes = [node((0, 0, -1.0), queue=2000), node((0, 0, -50.0), queue=1000)]
        self.assertEqual(self.order(nodes, CAM_F2B), [(1, 0), (0, 0)])

    def test_first_passes_before_additional_passes(self):
        nodes = [node((0, 0, -9.0)), node((0, 0, -2.0))]
        self.assertEqual(self.order(nodes, CAM_F2B, passes=2), [(1, 0), (0, 0), (1, 1), (0, 1)])

    def test_bucket_then_state_then_exact_distance(self):
        # -1 (0xbf800000) and -1.5 (0xbfc00000) share bucket 0xbf; -5 is 0xc0.
        near = node((0, 0, -1.0), shader_id=20)
        mid = node((0, 0, -math.sqrt(1.5)), shader_id=10)
        far = node((0, 0, -math.sqrt(5.0)), shader_id=5)
        # bucket on: bucket 0xbf before 0xc0; inside 0xbf the shader id decides
        self.assertEqual(self.order([near, mid, far], CAM_F2B), [(1, 0), (0, 0), (2, 0)])
        # NoDistanceSort: shader id first, distance only as a late tie-break
        self.assertEqual(self.order([near, mid, far], CAM_NODIST), [(2, 0), (1, 0), (0, 0)])
        # HSR GPU with Default mode behaves like NoDistanceSort
        self.assertEqual(self.order([near, mid, far], CAM_F2B, hsr=True), [(2, 0), (1, 0), (0, 0)])

    def test_static_batched_first_and_decisive(self):
        a = node((0, 0, -1.0), shader_id=1)
        b = node((0, 0, -9.0), shader_id=9, static_batch_subset_count=1, mesh_id=7)
        self.assertEqual(self.order([a, b], CAM_NODIST), [(1, 0), (0, 0)])

    def test_qsort_matches_total_order(self):
        rng = random.Random(1234)
        for n in (5, 31, 32, 33, 64, 66, 200, 700):
            vals = [rng.randrange(0, 50) for _ in range(n)]
            items = list(range(n))
            less = lambda a, b: (vals[a], a) < (vals[b], b)
            got = R.unity_qsort(list(items), less)
            exp = sorted(items, key=lambda i: (vals[i], i))
            self.assertEqual(got, exp)

    def test_heap_fallback_sorts(self):
        rng = random.Random(7)
        arr = [rng.random() for _ in range(100)]
        a = list(arr)
        R.unity_qsort(a, lambda x, y: x < y, depth=0)
        self.assertEqual(a, sorted(arr))

    def test_msvc_sort(self):
        rng = random.Random(9)
        # <=32 elements: insertion sort, stable for equal keys
        arr = [(i, F(rng.randrange(3))) for i in range(30)]
        a = list(arr)
        R.msvc_sort(a, lambda x, y: x[1] > y[1])
        self.assertEqual(a, sorted(arr, key=lambda e: -float(e[1])))
        for n in (33, 41, 120, 500):
            arr = [(i, F(rng.random())) for i in range(n)]
            a = list(arr)
            R.msvc_sort(a, lambda x, y: x[1] > y[1])
            self.assertEqual([e[1] for e in a], sorted([e[1] for e in arr], reverse=True))
            a = list(arr)
            R.msvc_sort(a, lambda x, y: x[1] > y[1], ideal=0)
            self.assertEqual([e[1] for e in a], sorted([e[1] for e in arr], reverse=True))


def light(t, **kw):
    l = {'type': t, 'render_mode': R.RENDER_MODE_AUTO, 'culling_mask': 0xFFFFFFFF,
         'lightmap_mode_for_render': 0, 'shadows': 0, 'range': 10.0, 'position': (0, 0, 0),
         'final_color': (1.0, 1.0, 1.0, 1.0), 'forward': (0.0, 0.0, 1.0)}
    l.update(kw)
    l['luminance_intensity'] = R.light_luminance_intensity(l.get('color', (1, 1, 1)), l.get('intensity', 1.0))
    return l


class LightTests(unittest.TestCase):
    def test_luminance_and_main_light(self):
        self.assertEqual(R.light_luminance_intensity((1, 1, 1), 1.0), (F(0.59) + F(0.3)) + F(0.11))
        dirs = [dict(color=(1, 1, 1), intensity=1.0, render_mode=0, shadows=0),
                dict(color=(1, 1, 1), intensity=0.2, render_mode=0, shadows=1),
                dict(color=(1, 1, 1), intensity=5.0, render_mode=2, shadows=0)]
        self.assertEqual(R.find_main_directional_light(dirs), 1)     # 0.2*16 > 1, ForceVertex excluded
        ordered, has_main = R.order_directional_lights(dirs)
        self.assertTrue(has_main)
        self.assertIs(ordered[0], dirs[1])
        self.assertEqual(R.find_main_directional_light([dirs[2]]), -1)

    def test_importance(self):
        d = light(R.LIGHT_DIRECTIONAL, shadows=1)
        self.assertEqual(R.light_importance(d, (0, 0, 0)), d['luminance_intensity'] * F(16))
        p = light(R.LIGHT_POINT, range=5.0, position=(0, 5, 0), render_mode=R.RENDER_MODE_FORCE_PIXEL)
        att = F(1) / ((F(25) / F(25)) * F(25) + F(1))
        self.assertEqual(R.light_importance(p, (0, 0, 0)), F(1000) + att * p['luminance_intensity'])
        v = light(R.LIGHT_POINT, render_mode=R.RENDER_MODE_FORCE_VERTEX)
        self.assertEqual(R.light_importance(v, (0, 0, 0)), F(-1000) + F(1) * v['luminance_intensity'])

    def test_cull_lights_for_object(self):
        lights = [light(R.LIGHT_DIRECTIONAL), light(R.LIGHT_DIRECTIONAL, culling_mask=0),
                  light(R.LIGHT_POINT, position=(0, 0, 3), range=2.0),       # sphere misses AABB
                  light(R.LIGHT_POINT, position=(0, 0, 1.5), range=2.0, render_mode=1)]
        n = {'layer': 0, 'center': (0, 0, 0), 'extent': (0.5, 0.5, 0.5), 'local_center': (0, 0, 0),
             'local_extent': (0.5, 0.5, 0.5), 'object_to_world': IDENT, 'transform_type': 0}
        out, off = R.cull_lights_for_object({'lights': lights, 'counts': (2, 0, 2, 0, 0)}, n)
        self.assertEqual([i for i, _ in out], [3, 0])
        self.assertFalse(off)

    def test_aabb_sphere(self):
        self.assertTrue(R.intersect_aabb_sphere((0, 0, 0), (1, 1, 1), (2.9, 0, 0), 2.0))
        self.assertFalse(R.intersect_aabb_sphere((0, 0, 0), (1, 1, 1), (3.0, 0, 0), 2.0))   # strict
        self.assertFalse(R.intersect_aabb_aabb((0, 0, 0), (1, 1, 1), (2, 0, 0), (1, 1, 1)))
        self.assertTrue(R.intersect_aabb_aabb((0, 0, 0), (1, 1, 1), (1.5, 0, 0), (1, 1, 1)))
        m = [2.0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0, 2.0, 0, 10.0, 0, 0, 1.0]
        self.assertTrue(R.intersect_transformed_aabb_sphere((0, 0, 0), (1, 1, 1), m, 1, (13.5, 0, 0), 2.0))
        self.assertFalse(R.intersect_transformed_aabb_sphere((0, 0, 0), (1, 1, 1), m, 1, (14.5, 0, 0), 2.0))
        with self.assertRaises(NotImplementedError):
            R.intersect_transformed_aabb_sphere((0, 0, 0), (1, 1, 1), m, 2, (0, 0, 0), 1.0)

    def _obj(self, culled):
        return {'culled': culled, 'has_offscreen': False, 'layer': 0, 'center': (0, 0, 0),
                'extent': (0.5, 0.5, 0.5), 'ambient_sh': [0.0] * 27,
                'flags': {'lightmapped': False, 'vertex_lit_subshader': False, 'coefficient_type': 1,
                          'no_add_pass': False, 'no_lights': False}}

    def test_classify(self):
        act = [light(R.LIGHT_DIRECTIONAL), light(R.LIGHT_POINT), light(R.LIGHT_POINT),
               light(R.LIGHT_POINT), light(R.LIGHT_DIRECTIONAL, forward=(0.0, -1.0, 0.0))]
        culled = [(0, F(10)), (1, F(8)), (2, F(6)), (3, F(4)), (4, F(2))]
        b = R.classify_lights(self._obj(culled), {'lights': act, 'has_main_light': True},
                              {'pixel_light_count': 2, 'rendering_path': 1})
        self.assertEqual(b['main'], 0)
        self.assertEqual(b['add'], [1, 2])
        blend = (F(6) - F(4)) / ((F(8) - F(4)) + R.K_BLEND_EPS)
        self.assertEqual(b['last_add_light_blend'], blend)
        # point light 2 is re-inserted as the first vertex light; vertex index uses (vert-1)+add = 2
        self.assertEqual(b['vertex'], [2, 3])
        self.assertEqual(b['last_vertex_light_blend'], blend)
        # light 4 (directional, Auto, beyond the pixel budget) went to SH: L0 red = C0*a0*k0*1
        l0 = R._SH_C0 * ((F(1) * R._SH_A0) * R.SH_NORM[0])
        self.assertEqual(b['sh'][0], l0)
        self.assertEqual(b['occlusion'], [F(1)] * 4)

    def test_classify_vertex_lit_pixel_count_zero_and_lightmapped(self):
        act = [light(R.LIGHT_DIRECTIONAL), light(R.LIGHT_POINT)]
        culled = [(0, F(10)), (1, F(8))]
        b = R.classify_lights(self._obj(culled), {'lights': act, 'has_main_light': True},
                              {'pixel_light_count': 4, 'rendering_path': 0})
        self.assertEqual((b['main'], b['add'], b['vertex']), (0, [], [1]))
        o = self._obj(culled)
        o['flags']['lightmapped'] = True
        b = R.classify_lights(o, {'lights': act, 'has_main_light': True},
                              {'pixel_light_count': 0, 'rendering_path': 1})
        self.assertEqual((b['add'], b['vertex']), ([], []))

    def test_cookie_main_light_quirk(self):
        # ActiveLights[0] has a cookie: a later directional becomes main and
        # index 0 is then skipped entirely if it comes after it in the list.
        act = [light(R.LIGHT_DIRECTIONAL, has_cookie=True), light(R.LIGHT_DIRECTIONAL)]
        b = R.classify_lights(self._obj([(1, F(5)), (0, F(3))]), {'lights': act, 'has_main_light': True},
                              {'pixel_light_count': 4, 'rendering_path': 1})
        self.assertEqual((b['main'], b['add']), (1, []))
        b = R.classify_lights(self._obj([(0, F(5)), (1, F(3))]), {'lights': act, 'has_main_light': True},
                              {'pixel_light_count': 4, 'rendering_path': 1})
        self.assertEqual((b['main'], b['add']), (1, [0]))

    def test_sh_constants(self):
        sp = math.sqrt(math.pi)
        exp = [1 / (2 * sp), -math.sqrt(3) / (3 * sp), math.sqrt(3) / (3 * sp), -math.sqrt(3) / (3 * sp),
               math.sqrt(15) / (8 * sp), -math.sqrt(15) / (8 * sp), math.sqrt(5) / (16 * sp),
               -math.sqrt(15) / (8 * sp), math.sqrt(15) / (16 * sp)]
        for k, e in zip(R.SH_NORM, exp):
            self.assertAlmostEqual(float(k), e, places=6)

    def test_sh_directional_band1_sign(self):
        sh = R.sh_add_directional_light([0.0] * 27, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
        # coefficient 2 (z) positive, coefficient 6 (3z^2-1) positive, green/blue untouched
        self.assertGreater(sh[2], 0)
        self.assertGreater(sh[6], 0)
        self.assertEqual(sh[1], 0)
        self.assertEqual(sh[9:], [F(0)] * 18)
        exp = float(R._SH_C0) * 0.48860251903533936 * math.sqrt(3) / (3 * math.sqrt(math.pi))
        self.assertAlmostEqual(float(sh[2]), exp, places=6)

    def test_sh_point_color_inside_box(self):
        p = light(R.LIGHT_POINT, position=(0.0, 0.0, 0.0), range=1.0, final_color=(2.0, 2.0, 2.0, 1.0))
        c = R.light_color_for_sh(p, (0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        # d2=0 < e2=4 -> att = 1/(4*25+1) * (1/4)
        self.assertEqual(c[0], F(2) * ((F(1) / (F(4) * F(25) + F(1))) * (F(1) / F(4))))
        with self.assertRaises(NotImplementedError):
            R.sh_light_direction(p, (1.0, 0.0, 0.0))
        d = R.sh_light_direction(p, (1.0, 0.0, 0.0), rsqrt_estimate=lambda s: F(1.0) / F(np.sqrt(s)))
        self.assertAlmostEqual(float(d[0]), -1.0, places=6)

    def test_forward_base_vertex_constants(self):
        act = [light(R.LIGHT_POINT, range=5.0, position=(1, 2, 3), vertex_color=(1, 1, 1, 1))]
        block = {'add': [], 'vertex': [0], 'last_add_light_blend': F(1), 'last_vertex_light_blend': F(0.5),
                 'occlusion': [F(1)]}
        c = R.forward_base_vertex_constants(block, {'lights': act})
        self.assertEqual(c['unity_4LightAtten0'], [F(1), F(1), F(1), F(1)])
        self.assertEqual(c['unity_LightColor'][0], [F(0.5)] * 4)
        self.assertEqual(c['unity_4LightPosX0'][0], F(1))
        self.assertEqual(c['unity_4LightAtten0'][0], F(25) / F(25))


def cam_default(**kw):
    from src.render.hs2pipe import unity_math as um
    c = {'world_to_camera': [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 1.0],
         'projection': um.set_perspective(60.0, 1.0, 0.3, 1000.0), 'position': (0.0, 0.0, 0.0),
         'near': 0.3, 'far': 1000.0, 'culling_mask': 0xFFFFFFFF, 'layer_cull_distances': [0.0] * 32,
         'layer_cull_spherical': False, 'implicit_culling_matrix': True, 'implicit_world_to_camera': True}
    c.update(kw)
    return c


class CullTests(unittest.TestCase):
    def test_planes_from_identity(self):
        planes = R.extract_projection_planes(IDENT)
        self.assertEqual([[float(v) for v in p] for p in planes],
                         [[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1], [0, 0, 1, 1], [0, 0, -1, 1]])

    def test_aabb_plane_test_is_inclusive_at_zero(self):
        planes = R.extract_projection_planes(IDENT)
        e = (0.5, 0.5, 0.5)
        self.assertTrue(R.aabb_visible((0, 0, 0), e, planes))
        self.assertFalse(R.aabb_visible((2, 0, 0), e, planes))
        self.assertTrue(R.aabb_visible((1.5, 0, 0), e, planes))       # rad + dist == 0 kept
        self.assertTrue(R.intersect_aabb_plane_bounds((1.5, 0, 0), e, planes))

    def test_camera_cull_masks_flags_and_near_far(self):
        cp = R.calculate_culling_parameters(cam_default())
        self.assertEqual([float(v) for v in cp['planes'][4]], [0.0, 0.0, 1.0, -0.30000001192092896])
        self.assertEqual([float(v) for v in cp['planes'][5]], [-0.0, -0.0, -1.0, 1000.0])
        self.assertEqual((float(cp['layer_far'][0]), cp['layer_mode']), (1000.0, 1))
        e = (0.5, 0.5, 0.5)
        rs = [{'center': (0, 0, 10), 'extent': e, 'layer': 0},
              {'center': (0, 0, -10), 'extent': e, 'layer': 0},
              {'list': 0, 'center': (0, 0, 5), 'extent': (0.1, 0.1, 0.1), 'layer': 0},
              {'center': (0, 0, 10), 'extent': e, 'layer': 3},
              {'center': (0, 0, 10), 'extent': e, 'layer': 0, 'flags': R.shadow_casting_flags(3)},
              {'center': (0, 0, 10), 'extent': e, 'layer': 0, 'flags': R.FLAG_PENDING_REMOVAL}]
        out = R.cull(rs, cam_default(culling_mask=0xFFFFFFFF & ~(1 << 3)))
        self.assertEqual(out, {0: [2], 1: [0], 2: [], 3: [], 4: [], 5: []})

    def test_layer_cull_distances(self):
        lcd = [0.0] * 32
        lcd[0] = 10.0
        e1 = (1.0, 1.0, 1.0)
        rs = [{'center': (0, 0, 10), 'extent': e1, 'layer': 0},
              {'center': (0, 0, 10.001), 'extent': e1, 'layer': 0},
              {'center': (0, 0, 20), 'extent': e1, 'layer': 1}]
        out = R.cull(rs, cam_default(layer_cull_distances=lcd, layer_cull_spherical=True))
        self.assertEqual(out[1], [0, 2])
        e = (0.5, 0.5, 0.5)
        rs = [{'center': (0, 0, 10.5), 'extent': e, 'layer': 0}, {'center': (0, 0, 10.6), 'extent': e, 'layer': 0}]
        self.assertEqual(R.cull(rs, cam_default(layer_cull_distances=lcd))[1], [0])

    def test_block_ranges_and_queue_order(self):
        self.assertEqual(R.configure_block_ranges_with_min(300, 128, 1), [(0, 100), (100, 100), (200, 100)])
        self.assertEqual(R.configure_block_ranges_with_min(300, 128, 0), [(0, 300)])
        rs = [{} for _ in range(300)]
        rs[5] = {'produces_node': False}
        order = R.render_node_queue_order({1: list(range(300))}, rs, 1)
        self.assertEqual(order, list(range(5)) + list(range(6, 100)) + [299] + list(range(100, 299)))
        rs = [{}, {'job_prepared': False}, {}, {}]
        self.assertEqual(R.render_node_queue_order({0: [0, 1], 1: [2, 3]}, rs, 4), [0, 2, 3, 1])

    def test_spot_light_object_frustum(self):
        # Spot light at the origin looking down +z (worldToLight = identity), 60 deg cone, range 10.
        cot = F(1.0) / F(math.tan(math.radians(30.0)))
        spot = light(R.LIGHT_SPOT, range=10.0, spot_cotan=cot, world_to_light=IDENT)
        o2w_in = list(IDENT)
        o2w_in[14] = 5.0                 # object 5 units in front of the light
        o2w_out = list(IDENT)
        o2w_out[14] = -5.0               # behind the light
        args = ((0, 0, 0), (0.5, 0.5, 0.5))
        self.assertFalse(R.spot_light_rejects_object(spot, *args, o2w_in, R.aabb_in_projection_frustum))
        self.assertTrue(R.spot_light_rejects_object(spot, *args, o2w_out, R.aabb_in_projection_frustum))


if __name__ == '__main__':
    unittest.main()
