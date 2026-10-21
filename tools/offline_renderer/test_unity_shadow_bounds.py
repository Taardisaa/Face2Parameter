"""Analytic checks for the shadow caster / receiver bounds transcription
(src/render/hs2pipe/unity_shadow_bounds.py; RVAs in that module and in
outputs/offline_renderer_unity_forward_20261020/shadow_bounds_spec.md)."""
from __future__ import annotations

import math
import struct
import unittest

import numpy as np

from src.render.hs2pipe import unity_math as um
from src.render.hs2pipe import unity_shadow_bounds as sb

F = np.float32


def bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


def world_to_camera_at(tx, ty, tz):
    """Unity worldToCamera of an unrotated camera at (tx, ty, tz): diag(1, 1, -1) * T(-t)."""
    m = um.identity()
    m[10] = F(-1.0)
    m[12], m[13], m[14] = F(-tx), F(-ty), F(tz)
    return m


BOX_PLANES = [  # x in [-1, 1], y in [-1, 1], z in [0, 10]
    (1, 0, 0, 1), (-1, 0, 0, 1), (0, 1, 0, 1), (0, -1, 0, 1), (0, 0, 1, 0), (0, 0, -1, 10)]


class ConstantTests(unittest.TestCase):
    def test_constant_bit_patterns(self):
        self.assertEqual(bits(sb.RECEIVER_EXPAND), 0x3c23d70a)
        self.assertEqual(bits(sb.VEC3_EPSILON), 0x3727c5ac)
        self.assertEqual(bits(sb.DEG_180), 0x43340000)
        self.assertEqual(bits(sb.STABLE_FIT_K), 0x3ecccccd)
        self.assertEqual(bits(sb.CLOSE_FIT_RADIUS), 0x58635fa9)


class MinMaxTests(unittest.TestCase):
    def test_empty_is_invalid_and_partial_infinity_rules(self):
        self.assertFalse(sb.is_valid(sb.empty_minmax()))
        mm = sb.empty_minmax()
        mm[1][0] = F(1.0)                    # max no longer -inf vector, min still +inf vector
        self.assertFalse(sb.is_valid(mm))
        mm = sb.empty_minmax()
        mm[0][2] = F(0.0)
        mm[1][1] = F(0.0)
        self.assertTrue(sb.is_valid(mm))

    def test_encapsulate_center_extent(self):
        mm = sb.encapsulate(sb.empty_minmax(), [1, 2, 3, 0.5, 1, 2])
        self.assertEqual([float(v) for v in mm[0]], [0.5, 1.0, 1.0])
        self.assertEqual([float(v) for v in mm[1]], [1.5, 3.0, 5.0])
        sb.encapsulate(mm, [-4, 2, 3, 1, 1, 1])
        self.assertEqual([float(v) for v in mm[0]], [-5.0, 1.0, 1.0])
        self.assertEqual([float(v) for v in mm[1]], [1.5, 3.0, 5.0])

    def test_union_keeps_accumulator_on_equality(self):
        # minss(new, acc) keeps acc on equality: +0 then -0 stays +0, reverse stays -0.
        a = sb.encapsulate(sb.empty_minmax(), [0.0, 0, 0, 0.0, 0, 0])
        sb.encapsulate(a, [-0.0, 0, 0, 0.0, 0, 0])
        self.assertFalse(math.copysign(1.0, float(a[0][0])) < 0)
        b = sb.encapsulate(sb.empty_minmax(), [-0.0, 0, 0, 0.0, 0, 0])
        sb.encapsulate(b, [0.0, 0, 0, 0.0, 0, 0])
        self.assertTrue(math.copysign(1.0, float(b[0][0])) < 0)

    def test_receiver_expansion_is_float32(self):
        mm = sb.encapsulate(sb.empty_minmax(), [1, 2, 3, 1, 1, 1])
        sb.expand_receiver_bounds(mm)
        e = F(0.00999999977648258)
        self.assertEqual(mm[0], [F(0) - e, F(1) - e, F(2) - e])
        self.assertEqual(mm[1], [F(2) + e, F(3) + e, F(4) + e])
        empty = sb.expand_receiver_bounds(sb.empty_minmax())
        self.assertFalse(sb.is_valid(empty))


class IntersectionTests(unittest.TestCase):
    def test_plane_bounds_touching_box_is_kept(self):
        plane = [(0, 0, -1, 10)]
        self.assertTrue(sb.intersect_aabb_plane_bounds([0, 0, 11, 0, 0, 1], plane, 1))
        self.assertFalse(sb.intersect_aabb_plane_bounds([0, 0, 11.5, 0, 0, 1], plane, 1))

    def test_frustum_full(self):
        self.assertTrue(sb.intersect_aabb_frustum_full([0, 0, 5, 0.5, 0.5, 0.5], BOX_PLANES))
        self.assertFalse(sb.intersect_aabb_frustum_full([5, 0, 5, 0.5, 0.5, 0.5], BOX_PLANES))
        self.assertTrue(sb.intersect_aabb_frustum_full([1.4, 0, 5, 0.5, 0.5, 0.5], BOX_PLANES))

    def test_sphere(self):
        self.assertTrue(sb.intersect_aabb_sphere([0, 0, 45, 1, 1, 4], (0, 0, 10), 32))     # 31^2 < 32^2
        self.assertFalse(sb.intersect_aabb_sphere([0, 0, 50, 1, 1, 1], (0, 0, 10), 32))    # 39^2 > 32^2
        self.assertFalse(sb.intersect_aabb_sphere([0, 0, 42, 1, 1, 0], (0, 0, 10), 32))    # 32^2 is not < 32^2
        self.assertTrue(sb.intersect_aabb_sphere([3, 4, 10, 0, 0, 0], (0, 0, 10), 5.5))    # 9 + 16 < 30.25


class CameraPlaneTests(unittest.TestCase):
    def test_extract_projection_planes_identity(self):
        planes = sb.extract_projection_planes(um.identity())
        expect = [(1, 0, 0, 1), (-1, 0, 0, 1), (0, 1, 0, 1), (0, -1, 0, 1), (0, 0, 1, 1), (0, 0, -1, 1)]
        for got, want in zip(planes, expect):
            self.assertEqual([float(v) for v in got], [float(v) for v in want])

    def test_extract_projection_planes_normalises(self):
        m = um.identity()
        m[0] = m[5] = m[10] = F(2.0)
        planes = sb.extract_projection_planes(m)
        self.assertEqual([float(v) for v in planes[0]], [1.0, 0.0, 0.0, 0.5])
        self.assertEqual([float(v) for v in planes[5]], [0.0, 0.0, -1.0, 0.5])

    def test_calculate_frustum_planes_near_far_from_camera_transform(self):
        w2c = world_to_camera_at(0, 0, 2)
        proj = um.set_perspective(F(90.0), F(1.0), F(0.5), F(100.0))
        planes, base_far = sb.calculate_frustum_planes(um.multiply(proj, w2c), w2c, F(10.0), F(0.5))
        self.assertEqual([float(v) for v in planes[4]], [0.0, 0.0, 1.0, -2.5])    # z >= 2 + near
        self.assertEqual(float(base_far), 2.0)
        self.assertEqual([float(v) for v in planes[5]], [0.0, 0.0, -1.0, 12.0])   # z <= 2 + maxDist
        self.assertTrue(sb.intersect_aabb_frustum_full([0, 0, 6, 0.1, 0.1, 0.1], planes))
        self.assertFalse(sb.intersect_aabb_frustum_full([0, 0, 13, 0.5, 0.5, 0.5], planes))

    def test_setup_shadow_cull_data_stable_fit(self):
        w2c = world_to_camera_at(0, 0, 2)
        proj = um.set_perspective(F(90.0), F(1.0), F(0.5), F(100.0))
        scd = sb.setup_shadow_cull_data(
            cull_world_to_clip=um.multiply(proj, w2c), world_to_camera=w2c, camera_near=0.5,
            camera_far=100.0, shadow_distance_for_rendering=40.0,
            shadow_projection=sb.SHADOW_PROJECTION_STABLE_FIT, cull_position=(0, 0, 2),
            is_orthographic=False, field_of_view=90.0)
        # k = clamp01(1 - 90/180) * 0.4f = 0.2f; -(k * 40) rounds to -8; (1 - k) rounds to 0.8f; 0.8f * 40 -> 32.
        self.assertTrue(scd['use_sphere'])
        self.assertEqual([float(v) for v in scd['sphere_center']], [0.0, 0.0, 10.0])
        self.assertEqual(float(scd['sphere_radius']), 32.0)
        self.assertEqual(float(scd['shadow_distance']), 40.0)
        self.assertEqual(float(scd['planes_b'][5][3]), 42.0)      # far at min(40, 100) from z = 2
        self.assertEqual(float(scd['planes_a'][5][3]), 102.0)     # far at camera far
        self.assertEqual(scd['planes_a'][:5], scd['planes_b'][:5])
        self.assertTrue(sb.is_object_within_shadow_range(scd, [0, 0, 40, 1, 1, 1]))
        self.assertTrue(sb.is_object_within_shadow_range(scd, [0, 0, 45, 1, 1, 4]))
        self.assertFalse(sb.is_object_within_shadow_range(scd, [0, 0, 50, 1, 1, 1]))

    def test_setup_shadow_cull_data_ortho_and_close_fit(self):
        w2c = world_to_camera_at(0, 0, 0)
        clip = um.multiply(um.set_ortho(F(-1), F(1), F(-1), F(1), F(0.5), F(100.0)), w2c)
        ortho = sb.setup_shadow_cull_data(
            cull_world_to_clip=clip, world_to_camera=w2c, camera_near=0.5, camera_far=100.0,
            shadow_distance_for_rendering=40.0, shadow_projection=sb.SHADOW_PROJECTION_STABLE_FIT,
            cull_position=(0, 0, 0), is_orthographic=True, field_of_view=60.0)
        k = F(0.4000000059604645)
        s = -(k * F(40.0))
        # z = ((col0.z*0 + col1.z*0) + col2.z*s) + pos.z with cameraToWorld col2 = (0, 0, -1).
        self.assertEqual(ortho['sphere_center'][2], ((F(0) * F(0) + F(0) * F(0)) + F(-1.0) * s) + F(0.0))
        self.assertEqual(ortho['sphere_radius'], (F(1.0) - k) * F(40.0))
        close = sb.setup_shadow_cull_data(
            cull_world_to_clip=clip, world_to_camera=w2c, camera_near=0.5, camera_far=100.0,
            shadow_distance_for_rendering=12.0, shadow_projection=sb.SHADOW_PROJECTION_CLOSE_FIT,
            cull_position=(0, 0, 0), is_orthographic=True, field_of_view=60.0)
        self.assertFalse(close['use_sphere'])
        self.assertEqual(float(close['planes_b'][5][3]), 12.0)
        self.assertTrue(sb.is_object_within_shadow_range(close, [0, 0, 13, 0, 0, 1.5]))
        self.assertFalse(sb.is_object_within_shadow_range(close, [0, 0, 13, 0, 0, 0.5]))


def caster(c, e, *, visible=True, pass_=True, mode='on'):
    return {'world_bounds': list(c) + list(e), 'shadow_cull_visible': visible,
            'has_shadow_caster_pass': pass_, 'shadow_casting_mode': mode}


class CasterBoundsTests(unittest.TestCase):
    A = caster((0, 0, 5), (0.5, 0.5, 0.5))
    B = caster((5, 0, 5), (0.5, 0.5, 0.5))

    def bounds(self, casters, **kw):
        kw.setdefault('light_type', sb.LIGHT_DIRECTIONAL)
        kw.setdefault('camera_frustum_planes', BOX_PLANES)
        r = sb.caster_bounds(casters, **kw)
        return None if r is None else ([float(v) for v in r[0]], [float(v) for v in r[1]])

    def test_directional_uses_only_casters_in_camera_frustum(self):
        self.assertEqual(self.bounds([self.A, self.B]), ([-0.5, -0.5, 4.5], [0.5, 0.5, 5.5]))

    def test_directional_falls_back_to_all_casters_when_none_visible(self):
        b2 = caster((-5, 3, 1), (1, 1, 1))
        self.assertEqual(self.bounds([self.B, b2]), ([-6.0, -0.5, 0.0], [5.5, 4.0, 5.5]))

    def test_non_directional_light_unions_all_casters(self):
        r = self.bounds([self.A, self.B], light_type='spot', camera_frustum_planes=None)
        self.assertEqual(r, ([-0.5, -0.5, 4.5], [5.5, 0.5, 5.5]))

    def test_shadow_caster_pass_and_cull_membership_filter(self):
        self.assertEqual(self.bounds([caster((0, 0, 5), (1, 1, 1), pass_=False), self.B]),
                         ([4.5, -0.5, 4.5], [5.5, 0.5, 5.5]))
        self.assertIsNone(self.bounds([caster((0, 0, 5), (1, 1, 1), visible=False)]))
        self.assertIsNone(self.bounds([]))

    def test_terrain_custom_casters_join_after_fallback_merge(self):
        r = self.bounds([self.B], terrain_custom_cull_bounds=[[0, 0, 5, 0.5, 0.5, 0.5]])
        self.assertEqual(r, ([-0.5, -0.5, 4.5], [5.5, 0.5, 5.5]))
        r = self.bounds([self.A, self.B], terrain_custom_cull_bounds=[[0, -9, 5, 0.5, 0.5, 0.5]])
        self.assertEqual(r, ([-0.5, -9.5, 4.5], [0.5, 0.5, 5.5]))

    def test_contract_errors(self):
        with self.assertRaises(sb.ShadowBoundsGap):
            sb.caster_bounds([{'world_bounds': [0] * 6, 'has_shadow_caster_pass': True,
                               'shadow_casting_mode': 'on'}],
                             light_type=sb.LIGHT_DIRECTIONAL, camera_frustum_planes=BOX_PLANES)
        with self.assertRaises(ValueError):
            sb.caster_bounds([caster((0, 0, 5), (1, 1, 1), mode='off')],
                             light_type=sb.LIGHT_DIRECTIONAL, camera_frustum_planes=BOX_PLANES)
        shadows_only = self.bounds([caster((0, 0, 5), (1, 1, 1), mode='shadows_only')])
        self.assertEqual(shadows_only, ([-1.0, -1.0, 4.0], [1.0, 1.0, 6.0]))


class DirectionalPostCullTests(unittest.TestCase):
    def ctx(self, **kw):
        w2c = world_to_camera_at(0, 0, 0)
        proj = um.set_perspective(F(90.0), F(1.0), F(0.5), F(100.0))
        scd = sb.setup_shadow_cull_data(
            cull_world_to_clip=um.multiply(proj, w2c), world_to_camera=w2c, camera_near=0.5,
            camera_far=100.0, shadow_distance_for_rendering=40.0,
            shadow_projection=sb.SHADOW_PROJECTION_STABLE_FIT, cull_position=(0, 0, 0),
            is_orthographic=False, field_of_view=90.0)
        d = [0.0] * 32
        d[3] = 10.0
        base = {'layer_mask': 0xffffffff, 'skip_baked': False, 'layer_cull_distances': d,
                'layer_cull_spherical': True, 'shadow_cull_data': scd}
        base.update(kw)
        return base

    def node(self, z, layer=0, mode='on', lightmapped=False):
        return {'shadow_casting_mode': mode, 'layer': layer, 'world_bounds': [0, 0, z, 1, 1, 1],
                'lightmap_index_is_none': not lightmapped}

    def test_mode_mask_and_baked(self):
        self.assertFalse(sb.directional_caster_post_cull(self.node(5, mode='off'), self.ctx()))
        self.assertTrue(sb.directional_caster_post_cull(self.node(5, mode='shadows_only'), self.ctx()))
        self.assertFalse(sb.directional_caster_post_cull(self.node(5, layer=4), self.ctx(layer_mask=0xffffffef)))
        self.assertFalse(sb.directional_caster_post_cull(self.node(5, lightmapped=True), self.ctx(skip_baked=True)))
        self.assertTrue(sb.directional_caster_post_cull(self.node(5, lightmapped=True), self.ctx()))

    def test_layer_cull_distance_spherical_and_planar(self):
        self.assertTrue(sb.directional_caster_post_cull(self.node(10, layer=3), self.ctx()))    # 100 <= 100
        self.assertFalse(sb.directional_caster_post_cull(self.node(10.5, layer=3), self.ctx()))
        planar = self.ctx(layer_cull_spherical=False)
        self.assertTrue(sb.directional_caster_post_cull(self.node(10.5, layer=3), planar))     # box reaches 9.5
        self.assertFalse(sb.directional_caster_post_cull(self.node(11.5, layer=3), planar))


def recv(c, e, *, main='L', receive=True, resolved=True, extra=(), extra_pass=False, terrain=None):
    o = {'world_bounds': list(c) + list(e), 'receive_shadows': receive, 'subshader_resolved': resolved,
         'forward_main_light': main, 'forward_additional_lights': list(extra),
         'additional_light_shadow_pass': extra_pass}
    if terrain is not None:
        o['terrain_receiver_bounds'] = terrain
    return o


class ReceiverBoundsTests(unittest.TestCase):
    LIGHTS = {'L': {'type': sb.LIGHT_DIRECTIONAL, 'shadows': True, 'active_flag_0x20': True},
              'L2': {'type': sb.LIGHT_DIRECTIONAL, 'shadows': True, 'active_flag_0x20': True},
              'N': {'type': sb.LIGHT_DIRECTIONAL, 'shadows': False, 'active_flag_0x20': True}}

    @classmethod
    def setUpClass(cls):
        w2c = world_to_camera_at(0, 0, 2)
        proj = um.set_perspective(F(90.0), F(1.0), F(0.5), F(100.0))
        cls.scd = sb.setup_shadow_cull_data(
            cull_world_to_clip=um.multiply(proj, w2c), world_to_camera=w2c, camera_near=0.5,
            camera_far=100.0, shadow_distance_for_rendering=40.0,
            shadow_projection=sb.SHADOW_PROJECTION_STABLE_FIT, cull_position=(0, 0, 2),
            is_orthographic=False, field_of_view=90.0)

    def raw(self, objects, light='L', **kw):
        kw.setdefault('receiver_gate', True)
        r = sb.receiver_bounds(objects, light, self.LIGHTS, self.scd, expand=False, **kw)
        return None if r is None else ([float(v) for v in r[0]], [float(v) for v in r[1]])

    def test_filters_receive_range_subshader_and_main_light(self):
        objs = [recv((0, 0, 10), (1, 1, 1)),
                recv((0, 5, 10), (1, 1, 1), receive=False),
                recv((0, 0, 80), (1, 1, 1)),                      # outside the 32 m stable-fit sphere
                recv((5, 0, 10), (1, 1, 1), resolved=False),
                recv((-7, 0, 10), (1, 1, 1), main='L2'),
                recv((9, 0, 10), (1, 1, 1), main='N'),
                recv((0, -6, 10), (1, 1, 1), main=None)]
        self.assertEqual(self.raw(objs), ([-1.0, -1.0, 9.0], [1.0, 1.0, 11.0]))
        self.assertEqual(self.raw(objs, 'L2'), ([-8.0, -1.0, 9.0], [-6.0, 1.0, 11.0]))
        self.assertIsNone(self.raw(objs, 'N'))
        self.assertIsNone(self.raw(objs, receiver_gate=False))

    def test_first_main_light_and_additional_lights(self):
        objs = [recv((0, 0, 10), (1, 1, 1), extra=['L2'], extra_pass=True),
                recv((3, 0, 10), (1, 1, 1), main='L2', extra=['L'], extra_pass=True)]
        first, boxes = sb.forward_receiver_bounds(objs, self.LIGHTS, self.scd, receiver_gate=True,
                                                  additional_light_gate=True)
        self.assertEqual(first, 'L')
        # L (first main) never takes objects that list it as an additional light.
        self.assertEqual([float(v) for v in boxes['L'][0]], [-1.0, -1.0, 9.0])
        self.assertEqual([float(v) for v in boxes['L'][1]], [1.0, 1.0, 11.0])
        # L2: created from object 0 as additional light, then object 1 as its main light.
        self.assertEqual([float(v) for v in boxes['L2'][0]], [-1.0, -1.0, 9.0])
        self.assertEqual([float(v) for v in boxes['L2'][1]], [4.0, 1.0, 11.0])
        _, no_extra = sb.forward_receiver_bounds(objs, self.LIGHTS, self.scd, receiver_gate=True,
                                                 additional_light_gate=False)
        self.assertEqual([float(v) for v in no_extra['L2'][0]], [2.0, -1.0, 9.0])

    def test_terrain_patch_uses_receiver_data_bounds(self):
        objs = [recv((0, 0, 10), (1, 1, 1), terrain=[0, 0, 10, 0.25, 0.25, 0.25])]
        self.assertEqual(self.raw(objs), ([-0.25, -0.25, 9.75], [0.25, 0.25, 10.25]))

    def test_expanded_result_and_gates(self):
        objs = [recv((1, 2, 10), (1, 1, 1))]
        r = sb.receiver_bounds(objs, 'L', self.LIGHTS, self.scd, receiver_gate=True)
        e = F(0.00999999977648258)
        self.assertEqual(list(r[0]), [F(0) - e, F(1) - e, F(9) - e])
        self.assertEqual(list(r[1]), [F(2) + e, F(3) + e, F(11) + e])
        self.assertFalse(sb.forward_receiver_gate(shadow_pass_argument=False, camera_can_do_shadows=True,
                                                  rendering_path=1))
        self.assertFalse(sb.forward_receiver_gate(shadow_pass_argument=True, camera_can_do_shadows=True,
                                                  rendering_path=sb.RENDERING_PATH_VERTEX_LIT))
        self.assertTrue(sb.forward_receiver_gate(shadow_pass_argument=True, camera_can_do_shadows=True,
                                                 rendering_path=1))
        self.assertFalse(sb.should_use_shadows(build_has_shadows=True, caps_0x39b=1, caps_0x399=1,
                                               caps_0x390=1, quality_shadows=2,
                                               shadow_distance_for_rendering=0.0))
        self.assertTrue(sb.should_use_shadows(build_has_shadows=True, caps_0x39b=1, caps_0x399=1,
                                              caps_0x390=1, quality_shadows=2,
                                              shadow_distance_for_rendering=40.0))

    def test_local_light_entry_creation_is_an_explicit_gap(self):
        lights = dict(self.LIGHTS)
        lights['S'] = {'type': 'spot', 'shadows': True, 'active_flag_0x20': True}
        objs = [recv((0, 0, 10), (1, 1, 1)), recv((0, 0, 10), (1, 1, 1), main='S')]
        with self.assertRaises(sb.ShadowBoundsGap):
            sb.forward_receiver_bounds(objs, lights, self.scd, receiver_gate=True)


if __name__ == '__main__':
    unittest.main()
