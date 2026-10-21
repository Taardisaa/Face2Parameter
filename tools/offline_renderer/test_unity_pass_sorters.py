"""Analytic checks for src/render/hs2pipe/unity_pass_sorters.py.

Run: .venv/Scripts/python.exe -m unittest tools.offline_renderer.test_unity_pass_sorters -v
"""
import random
import unittest

import numpy as np

from src.render.hs2pipe import unity_forward_rules as R
from src.render.hs2pipe import unity_pass_sorters as P

F = np.float32
IDENT = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]


def w2c_at(pos):
    m = list(IDENT)
    m[12], m[13], m[14] = -pos[0], -pos[1], -pos[2]
    return m


def sc_pass():
    return {'light_mode': 'ShadowCaster', 'type': P.PASS_TYPE_SHADOW_CASTER}


def fwd_pass():
    return {'light_mode': 'ForwardBase', 'type': R.PASS_FORWARD_BASE}


def mv_pass():
    return {'light_mode': 'MotionVectors', 'type': P.PASS_TYPE_MOTION_VECTORS}


def shader(passes=None, tags=None, active=0, extra_subshaders=()):
    subs = [{'tags': dict(tags or {}), 'passes': list(passes if passes is not None else [fwd_pass(), sc_pass()])}]
    subs.extend(extra_subshaders)
    return {'active_subshader': active, 'subshaders': subs}


def mat(mid, sh=None, queue=2000, **kw):
    m = {'material_instance_id': mid, 'shader_id': 1, 'shader': sh if sh is not None else shader(),
         'custom_render_queue': queue, 'shader_render_queue': 2000}
    m.update(kw)
    return m


def node(center, mats, **kw):
    n = {'center': center, 'materials': mats, 'layering': R.layering_key(0, 0), 'mesh_id': 0,
         'shadow_casting_mode': P.SHADOW_CASTING_ON}
    n.update(kw)
    return n


DN_SHADER = {'active_subshader': 0, 'subshaders': [
    {'tags': {'RenderType': 'Opaque'}, 'passes': [{'light_mode': ''}]},
    {'tags': {'RenderType': 'TransparentCutout'}, 'passes': [{'light_mode': ''}]},
    {'tags': {'RenderType': 'TreeTransparentCutout'}, 'passes': [{'light_mode': ''}, {'light_mode': ''}]},
]}


class SubshaderSelection(unittest.TestCase):
    def test_no_replacement_uses_active_subshader(self):
        sh = shader(active=1, extra_subshaders=[{'tags': {}, 'passes': []}])
        self.assertEqual(P.calculate_subshader_index_to_use(sh, mat(1, sh), None), 1)

    def test_empty_tag_gives_subshader_zero(self):
        self.assertEqual(P.calculate_subshader_index_to_use(shader(), mat(1), {'shader': DN_SHADER, 'tag': ''}), 0)
        self.assertEqual(P.calculate_subshader_index_to_use(shader(), mat(1), {'shader': DN_SHADER, 'tag': '<INVALID>'}), 0)

    def test_tag_lookup_case_insensitive_and_first_match(self):
        sh = shader(tags={'rendertype': 'transparentCUTOUT'})
        rep = {'shader': DN_SHADER, 'tag': 'RenderType'}
        self.assertEqual(P.calculate_subshader_index_to_use(sh, mat(1, sh), rep), 1)
        dup = {'active_subshader': 0, 'subshaders': DN_SHADER['subshaders'] + [
            {'tags': {'RenderType': 'Opaque'}, 'passes': []}]}
        self.assertEqual(P.calculate_subshader_index_to_use(shader(tags={'RenderType': 'Opaque'}), mat(1),
                                                            {'shader': dup, 'tag': 'RenderType'}), 0)

    def test_missing_tag_or_value_skips(self):
        rep = {'shader': DN_SHADER, 'tag': 'RenderType'}
        self.assertEqual(P.calculate_subshader_index_to_use(shader(), mat(1), rep), -1)
        self.assertEqual(P.calculate_subshader_index_to_use(shader(tags={'RenderType': 'Background'}), mat(1), rep), -1)

    def test_only_active_subshader_tags_are_read(self):
        sh = {'active_subshader': 1, 'subshaders': [{'tags': {'RenderType': 'Opaque'}, 'passes': []},
                                                     {'tags': {}, 'passes': []}]}
        self.assertEqual(P.calculate_subshader_index_to_use(sh, mat(1, sh), {'shader': DN_SHADER, 'tag': 'RenderType'}), -1)

    def test_material_override_tag_wins(self):
        sh = shader(tags={'RenderType': 'Opaque'})
        m = mat(1, sh, override_tags={'RenderType': 'TreeTransparentCutout'})
        self.assertEqual(P.calculate_subshader_index_to_use(sh, m, {'shader': DN_SHADER, 'tag': 'RenderType'}), 2)


class ShadowCasterPass(unittest.TestCase):
    def test_own_pass(self):
        self.assertEqual(P.get_shadow_caster_pass_to_use(shader(), 0), {'subshader': 0, 'pass': 1})

    def test_force_no_shadow_casting(self):
        sh = shader(tags={'ForceNoShadowCasting': 'TRUE'})
        self.assertIsNone(P.get_shadow_caster_pass_to_use(sh, 0))
        sh2 = shader(tags={'ForceNoShadowCasting': 'False'})
        self.assertIsNotNone(P.get_shadow_caster_pass_to_use(sh2, 0))

    def test_fallback_pass_from_copied_subshader(self):
        sh = shader(passes=[fwd_pass()], extra_subshaders=[{'tags': {}, 'passes': [fwd_pass(), fwd_pass(), sc_pass()]}])
        self.assertEqual(P.get_shadow_caster_pass_to_use(sh, 0), {'subshader': 1, 'pass': 2})

    def test_no_pass(self):
        self.assertIsNone(P.get_shadow_caster_pass_to_use(shader(passes=[fwd_pass()]), 0))


class DepthTexture(unittest.TestCase):
    def test_key_layout(self):
        n = node((0, 0, -5), [], mesh_id=0x12345, first_subset=0x1AB, static_batch_subset_count=0)
        vz = F(-5.0)
        k = P.depth_sort_key(3, n, vz, 0x203)
        self.assertEqual(k >> 56, 3)
        self.assertEqual((k >> 42) & 0x3FFF, 0x3FFF)                     # not static batched -> 0xffff
        self.assertEqual((k >> 32) & 0x3FF, R.f32_bits(vz) >> 22)
        self.assertEqual((k >> 24) & 0xFF, (0x203 >> 1) & 0xFF)
        self.assertEqual((k >> 16) & 0xFF, 0xAB)
        self.assertEqual(k & 0xFFFF, 0x2345)
        sb = dict(n, static_batch_subset_count=2)
        self.assertEqual((P.depth_sort_key(3, sb, vz, 0x203) >> 42) & 0x3FFF, 0x12345 & 0x3FFF)
        neg = P.depth_sort_key(0, n, vz, -302378)
        self.assertEqual((neg >> 24) & 0xFF, ((-302378 << 23) & 0xFF000000) >> 24)

    def test_inclusion(self):
        nodes = [node((0, 0, -2), [mat(10)]),
                 node((0, 0, -2), [mat(11, queue=2501)]),
                 node((0, 0, -2), [mat(12)], shadow_casting_mode=P.SHADOW_CASTING_SHADOWS_ONLY),
                 node((0, 0, -2), [mat(13)], shadow_casting_mode=P.SHADOW_CASTING_OFF),
                 node((0, 0, -2), [mat(14, shader(passes=[fwd_pass()]))]),
                 node((0, 0, -2), [mat(15, queue=-1, shader_render_queue=2500)])]
        ros = P.depth_pass_objects(nodes, IDENT)
        self.assertEqual([r['node'] for r in ros], [0, 3, 5])

    def test_queue_comes_from_material_shader_even_with_replacement(self):
        rep_sh = shader()
        nodes = [node((0, 0, -2), [mat(10, queue=-1, shader_render_queue=3000)])]
        self.assertEqual(P.depth_pass_objects(nodes, IDENT, {'shader': rep_sh, 'tag': ''}), [])

    def test_order_subshader_then_bucket_then_material_bits(self):
        far = node((0, 0, -40), [mat(2)])
        near = node((0, 0, -2), [mat(200)])
        sh1 = shader(active=1, extra_subshaders=[{'tags': {}, 'passes': [sc_pass()]}])
        sub1_near = node((0, 0, -1), [mat(4, sh1)])
        same_bucket_a = node((0, 0, -2.5), [mat(100)])        # -2 and -2.5 share bits >> 22? check below
        nodes = [sub1_near, far, near, same_bucket_a]
        draws = P.depth_texture_draws(nodes, IDENT)
        order = [d['node'] for d in draws]
        self.assertEqual(order[-1], 0)                       # subshader 1 after every subshader-0 entry
        self.assertEqual(R.f32_bits(F(-2.0)) >> 22, R.f32_bits(F(-2.5)) >> 22)
        self.assertEqual(order[:3], [3, 2, 1])               # same bucket: id bits 50 < 100; far last

    def test_material_index_tiebreak(self):
        n = node((0, 0, -2), [mat(10), mat(10), mat(10)])
        draws = P.depth_texture_draws([n], IDENT)
        self.assertEqual([d['material_index'] for d in draws], [0, 1, 2])
        ros = P.depth_pass_objects([n], IDENT)
        self.assertEqual([d['material_index'] for d in P.sort_depth_pass(list(reversed(ros)))], [0, 1, 2])

    def test_large_list_total_order(self):
        rng = random.Random(7)
        nodes = [node((0, 0, -rng.uniform(0.1, 900.0)), [mat(rng.randrange(1, 1 << 20))], mesh_id=i)
                 for i in range(300)]
        draws = P.depth_texture_draws(nodes, IDENT)
        keys = [(d['key'], d['material_index']) for d in draws]
        self.assertEqual(keys, sorted(keys))

    def test_update_depth_textures(self):
        self.assertEqual(P.update_depth_textures(depth_texture_mode=7, rendering_path=P.RENDERING_PATH_FORWARD),
                         ['depth', 'depth_normals'])
        self.assertEqual(P.update_depth_textures(depth_texture_mode=2, rendering_path=P.RENDERING_PATH_FORWARD),
                         ['depth_normals'])
        self.assertEqual(P.update_depth_textures(depth_texture_mode=0, rendering_path=P.RENDERING_PATH_FORWARD,
                                                 screen_space_shadows_tier=True, has_screen_space_shadow_shader=True,
                                                 cull_results_0x2f0=True, should_use_shadows=True), ['depth'])
        self.assertEqual(P.update_depth_textures(depth_texture_mode=3, rendering_path=P.RENDERING_PATH_DEFERRED_SHADING,
                                                 caps_0x39f=True), [])
        self.assertEqual(P.update_depth_textures(depth_texture_mode=0, rendering_path=P.RENDERING_PATH_DEFERRED_SHADING,
                                                 caps_0x39f=False), ['depth'])


class DepthNormals(unittest.TestCase):
    def _nodes(self):
        op = shader(tags={'RenderType': 'Opaque'})
        cut = shader(tags={'RenderType': 'TransparentCutout'})
        tree = shader(tags={'RenderType': 'TreeTransparentCutout'})
        tr = shader(tags={'RenderType': 'Transparent'})
        return [node((0, 0, -2), [mat(40, tree)]),
                node((0, 0, -2), [mat(30, cut, queue=2989)]),          # no queue filter
                node((0, 0, -2), [mat(20, op), mat(22, tr, queue=3000)]),
                node((0, 0, -2), [mat(10, op)], shadow_casting_mode=P.SHADOW_CASTING_OFF)]

    def test_draw_list_groups_by_subshader_and_expands_passes(self):
        draws = P.depth_normals_draws(self._nodes(), IDENT, DN_SHADER)
        self.assertEqual([(d['node'], d['subshader'], d['pass']) for d in draws],
                         [(3, 0, 0), (2, 0, 0), (1, 1, 0), (0, 2, 0), (0, 2, 1)])

    def test_layering_then_priority_desc(self):
        op = shader(tags={'RenderType': 'Opaque'})
        a = node((0, 0, -2), [mat(10, op)], layering=R.layering_key(0, 1))
        b = node((0, 0, -2), [mat(10, op)], priority=0)
        c = node((0, 0, -2), [mat(10, op)], priority=5)
        draws = P.depth_normals_draws([a, b, c], IDENT, DN_SHADER)
        self.assertEqual([d['node'] for d in draws], [2, 1, 0])

    def test_equal_keys_keep_insertion_order_below_32(self):
        op = shader(tags={'RenderType': 'Opaque'})
        n = node((0, 0, -2), [mat(10, op), mat(10, op)])
        ros = P.replacement_objects([n], IDENT, {'shader': DN_SHADER, 'tag': 'RenderType'})
        self.assertFalse(P.replacement_less(ros[1], ros[0]))
        self.assertEqual([r['material_index'] for r in P.sort_replacement(list(reversed(ros)))], [1, 0])


class ShadowCasters(unittest.TestCase):
    def test_parts_and_casters(self):
        nodes = [node((0, 0, -2), [mat(1, shadow_caster_hash=5), mat(2, shader(passes=[fwd_pass()])),
                                   mat(3, shadow_caster_hash=6, queue=3000)], first_subset=4, static_batch_subset_count=2),
                 node((0, 0, -2), [mat(4, shader(passes=[fwd_pass()]))]),
                 node((0, 0, -2), [mat(5, shadow_caster_hash=7)])]
        casters, parts = P.shadow_caster_parts(nodes)
        self.assertEqual(casters, [{'node': 0, 'first': 0, 'end': 2}, {'node': 2, 'first': 2, 'end': 3}])
        self.assertEqual([p['subset'] for p in parts], [4, 5, 0])     # clamp to count-1 for material 2

    def test_depth16(self):
        self.assertEqual(P.shadow_caster_depth16((0, 0, 0), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -0.25, 1]), 0)
        self.assertEqual(P.shadow_caster_depth16((0, 0, 0), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0.5, 1]),
                         int(np.trunc(float(F(0.5) * F(16383.0)))))
        self.assertEqual(P.shadow_caster_depth16((0, 0, 0), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 3.0, 1]), 16383)
        self.assertEqual(P.shadow_caster_depth16((0, 0, 0), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), 0)  # NaN

    def test_key_layout_and_order(self):
        cm = IDENT
        a = node((0, 0, 0.25), [mat(1, shadow_caster_hash=0x400001)], mesh_id=0x8003, transform_type=4)
        part = {'material': a['materials'][0], 'subset': 0x1FF}
        k = P.shadow_caster_sort_key(a, part, cm)
        self.assertEqual(k >> 63, 0)
        self.assertEqual((k >> 40) & 0x3FFFFF, 1)
        self.assertEqual((k >> 39) & 1, 1)
        self.assertEqual((k >> 24) & 0x7FFF, 3)
        self.assertEqual((k >> 16) & 0xFF, 0xFF)
        self.assertEqual(k & 0xFFFF, int(np.trunc(float(F(0.25) * F(16383.0)))))
        two = dict(a, shadow_casting_mode=P.SHADOW_CASTING_TWO_SIDED)
        self.assertEqual(P.shadow_caster_sort_key(two, part, cm) >> 63, 1)
        sb = dict(a, static_batch_subset_count=3, first_subset=0x77)
        ks = P.shadow_caster_sort_key(sb, part, cm)
        self.assertEqual(((ks >> 16) & 0xFF, ks & 0xFFFF), (0, 0x77))

    def test_draws_hash_then_depth(self):
        nodes = [node((0, 0, 0.9), [mat(1, shadow_caster_hash=2)]),
                 node((0, 0, 0.1), [mat(2, shadow_caster_hash=2)]),
                 node((0, 0, 0.5), [mat(3, shadow_caster_hash=1)]),
                 node((0, 0, 0.5), [mat(4, shadow_caster_hash=1)], shadow_casting_mode=P.SHADOW_CASTING_TWO_SIDED)]
        casters, parts = P.shadow_caster_parts(nodes)
        masks = P.directional_split_masks(len(casters), shadow_projection=0, cascade_count=1)
        draws = P.shadow_caster_draws(nodes, casters, parts, split=0, split_masks=masks, cull_matrix=IDENT)
        self.assertEqual([d['node'] for d in draws], [2, 1, 0, 3])
        masks2 = [1, 0, 1, 1]
        draws2 = P.shadow_caster_draws(nodes, casters, parts, split=0, split_masks=masks2, cull_matrix=IDENT,
                                       custom_casters=[casters[1]])
        self.assertEqual([d['node'] for d in draws2], [2, 1, 0, 3])

    def test_gaps_raise(self):
        with self.assertRaises(NotImplementedError):
            P.directional_split_masks(3, shadow_projection=1, cascade_count=1)
        with self.assertRaises(NotImplementedError):
            P.directional_split_masks(3, shadow_projection=0, cascade_count=2)
        n = node((0, 0, 0), [mat(1)])
        casters, parts = P.shadow_caster_parts([n])
        with self.assertRaises(NotImplementedError):
            P.shadow_caster_draws([n], casters, parts, split=0, split_masks=[1], cull_matrix=IDENT)


INTERNAL_MV = {'active_subshader': 0, 'subshaders': [{'tags': {}, 'passes': [mv_pass(), {'type': 0}, {'type': 0}]}]}


class MotionVectors(unittest.TestCase):
    def test_compare_approximately(self):
        a = [F(0.0)] * 16
        self.assertTrue(P.compare_approximately_matrix(a, [P.VECTOR3_EPSILON] + [F(0.0)] * 15))
        self.assertFalse(P.compare_approximately_matrix(a, [F(2e-5)] + [F(0.0)] * 15))
        self.assertFalse(P.compare_approximately_matrix(a, [F(np.nan)] + [F(0.0)] * 15))

    def _nodes(self):
        moved = list(IDENT)
        moved[12] = 1.0
        base = dict(local_to_world=IDENT, prev_local_to_world=IDENT, transform_update_frame=5)
        return [
            node((0, 0, -2), [mat(1)], motion_vector_mode=P.MV_CAMERA, skinned_last_position=True, **base),  # 0 skip
            node((0, 0, -2), [mat(2)], motion_vector_mode=P.MV_OBJECT, skinned_last_position=True, **base),  # 1 keep
            node((0, 0, -2), [mat(3)], motion_vector_mode=P.MV_OBJECT, **base),                             # 2 static: skip
            node((0, 0, -2), [mat(4)], motion_vector_mode=P.MV_OBJECT,
                 **dict(base, prev_local_to_world=moved)),                                                   # 3 moved: keep
            node((0, 0, -2), [mat(5)], motion_vector_mode=P.MV_OBJECT,
                 **dict(base, prev_local_to_world=moved, transform_update_frame=4)),                         # 4 stale: skip
            node((0, 0, -2), [mat(6, shader(passes=[fwd_pass(), mv_pass()]))], motion_vector_mode=P.MV_OBJECT, **base),  # 5 own pass
            node((0, 0, -2), [mat(7)], motion_vector_mode=P.MV_FORCE_NO_MOTION, **base),                    # 6 keep
            node((0, 0, -2), [mat(8, queue=3000)], motion_vector_mode=P.MV_OBJECT, skinned_last_position=True, **base),  # transparent
        ]

    def test_inclusion_rules(self):
        nodes = self._nodes()
        ctx = {'world_to_camera': IDENT, 'position': (0, 0, 0), 'sort_mode': R.SORT_PERSPECTIVE, 'sort_axis': (0, 0, 1)}
        opaque, _ = R.extract_render_objects(nodes, ctx)
        rp = P.motion_vector_objects(opaque, nodes, renderer_update_frame=5, internal_mv_shader=INTERNAL_MV)
        self.assertEqual([opaque[e['ro']]['node'] for e in rp], [1, 3, 5, 6])
        self.assertEqual([e['object_pass'] for e in rp], [False, False, True, False])
        self.assertEqual(rp[2]['pass'], 1)
        self.assertTrue(rp[3]['force_no_motion'])
        self.assertTrue(rp[0]['skinned'])

    def test_sort(self):
        ctx = {'world_to_camera': IDENT, 'position': (0, 0, 0), 'sort_mode': R.SORT_PERSPECTIVE, 'sort_axis': (0, 0, 1)}
        base = dict(motion_vector_mode=P.MV_OBJECT, skinned_last_position=True)
        nodes = [node((0, 0, -3), [mat(1, queue=2450)], **base),
                 node((0, 0, -2.5), [mat(2, queue=2450)], **base),
                 node((0, 0, -3), [mat(3, queue=2000)], **base),
                 node((0, 0, -100), [mat(4, queue=2000)], **base),
                 node((0, 0, -3), [mat(5, queue=2450, enable_instancing=True)], mesh_id=9, **base),
                 node((0, 0, -3), [mat(6, queue=2450)], priority=3, **base),
                 node((0, 0, -3), [mat(7, queue=2450)], **base)]
        opaque, _ = R.extract_render_objects(nodes, ctx)
        draws = P.motion_vector_draws(opaque, nodes, renderer_update_frame=0, internal_mv_shader=INTERNAL_MV)
        order = [opaque[d['ro']]['node'] for d in draws]
        # queue 2000 first (bits(dist)>>24 bucket: 0xC1 before 0xC6), then 2450: bucket 0xC0 (node 1)
        # before 0xC1, then instancing key desc (node 4), priority desc (node 5), RO index (0, 6)
        self.assertEqual(order, [2, 3, 1, 4, 5, 0, 6])


if __name__ == '__main__':
    unittest.main()
