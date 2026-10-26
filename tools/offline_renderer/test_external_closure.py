"""Checks for the external:capture closure of 2026-10-25 (orchestrator section 13.2):

* shadow-caster culling (src/render/hs2pipe/unity_shadow_cull.py): constants and tables read
  back from the installed UnityPlayer.dll, CalculateShadowCasterCull / edge planes / kernel on
  analytic cases, and (with the v7 session on E:) the generated caster set of one bridge render
  against the capture;
* motion-vector inclusion (unity_camera_passes.motion_vector_included);
* TexEnv resolution and built-in texture settings (bindings.resolve_texenv,
  builtin_texture_settings).

    .venv/Scripts/python.exe -m unittest tools.offline_renderer.test_external_closure
"""
from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import bindings as bd  # noqa: E402
from src.render.hs2pipe import unity_camera_passes as ucp  # noqa: E402
from src.render.hs2pipe import unity_math as um  # noqa: E402
from src.render.hs2pipe import unity_shadow_cull as usc  # noqa: E402

F = np.float32
DLL = Path('E:/HoneySelect2_ArcticFox/UnityPlayer.dll')
V7 = Path('E:/HS2_offline_renderer_data/paired_session_v7_20261025')
V7_TEMPLATE = Path('E:/HS2_offline_renderer_data/multiview_20261025/v7/w_2daddb57/template_v1/report.json')


def bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]


class DllTables(unittest.TestCase):
    """The transcribed constants equal the bytes of the installed DLL."""

    @classmethod
    def setUpClass(cls):
        if not DLL.exists():
            raise unittest.SkipTest('installed UnityPlayer.dll not present')
        sys.path.insert(0, str(ROOT / 'tools' / 'offline_renderer'))
        import unityplayer_disasm as u
        cls.pe = u.PE(DLL)

    def f32(self, rva, n):
        return struct.unpack('<%df' % n, self.pe.read_rva(rva, 4 * n))

    def test_plane_tables(self):
        ndc = [self.f32(0x14ae790 + 12 * i, 3) for i in range(6)]
        self.assertEqual([tuple(p) for p in usc.PLANE_NDC], [tuple(p) for p in ndc])
        self.assertEqual(usc.SIDE_CYCLE, struct.unpack('<4i', self.pe.read_rva(0x14ae7d8, 16)))
        self.assertEqual(usc.NEAR_FAR, struct.unpack('<2i', self.pe.read_rva(0x14ae7e8, 8)))
        self.assertEqual(self.pe.read_rva(0x1244ec0, 6), b'\0' * 6)       # PrepareShadowedLight exclusions

    def test_camera_motion_quad_world_matrix(self):
        """The camera motion-vector quad draws with an identity world matrix:
        LoadFullScreenOrthoMatrix calls vtable +0xe8 (0x68fcc4), GfxDevice's +0xe8 slot is
        SetViewMatrix 0x8cf290, which runs SetIdentity 0x7d0a00 on the world matrix at
        GfxDevice+0x2110 (0x8cf2ed..0x8cf2f4)."""
        self.assertEqual(self.pe.read_rva(0x68fcc4, 6), bytes.fromhex('ff90e8000000'))
        slot = struct.unpack('<Q', self.pe.read_rva(0x1264640 + 0xe8, 8))[0] - self.pe.image_base
        self.assertEqual(slot, 0x8cf290)
        self.assertEqual(self.pe.read_rva(0x8cf2ed, 7), bytes.fromhex('488d8f10210000'))   # lea rcx,[rdi+0x2110]
        call = self.pe.read_rva(0x8cf2f4, 5)
        self.assertEqual(call[0], 0xE8)
        self.assertEqual(0x8cf2f9 + struct.unpack('<i', call[1:])[0], 0x7d0a00)
        # GfxDevice::SetWorldMatrix 0x8cf370 stores the same +0x2110 matrix
        self.assertEqual(self.pe.read_rva(0x8cf37a, 7), bytes.fromhex('0f118110210000'))

    def test_float_constants(self):
        self.assertEqual(bits(usc.PARALLEL_EPS), 0x3c23d70a)
        self.assertEqual(bits(usc.NEG_PARALLEL_EPS), 0xbc23d70a)
        self.assertEqual(bits(usc.CROSS_EPS), 0x3a83126f)
        self.assertEqual(bits(usc.EPS_V3), 0x3727c5ac)

    def test_builtin_texenv_names(self):
        base = self.pe.image_base
        names = []
        for i in range(25):
            p = struct.unpack('<Q', self.pe.read_rva(0x14b8e40 + 8 * i, 8))[0] - base
            names.append(self.pe.read_rva(p, 40).split(b'\0')[0].decode())
        self.assertEqual(tuple(names), bd.BUILTIN_TEXENV_NAMES)

    def test_aniso_limits(self):
        self.assertEqual(struct.unpack('<ii', self.pe.read_rva(0x14af700, 8)), (1, 16))   # gUserMin/MaxAniso
        self.assertEqual(struct.unpack('<i', self.pe.read_rva(0x14b22dc, 4))[0], 9)        # ForceEnable min


class CasterCull(unittest.TestCase):
    def frustum(self):
        # camera at the origin looking +Z, fov 60, aspect 1, near 0.3, far 100
        proj = um.set_perspective(60.0, 1.0, 0.3, 100.0)
        w2c = um.identity()
        w2c[10] = F(-1.0)
        cull = um.multiply(proj, w2c)
        from src.render.hs2pipe import unity_shadow_bounds as sb
        planes, _ = sb.calculate_frustum_planes(cull, w2c, 100.0, 0.3)
        ok, c2w = um.invert_full(cull)
        return planes, c2w

    def test_light_from_above(self):
        planes, c2w = self.frustum()
        L = [F(0.0), F(-1.0), F(0.0)]                     # light travelling straight down
        out = usc.calculate_shadow_caster_cull(False, planes, c2w, [F(0)] * 3, F(1.0), F(1.0),
                                               usc.LIGHT_DIRECTIONAL, L)
        # a plane is kept when n.L < 0: the bottom plane (n.y > 0); left/right/near/far are parallel
        # (|n.L| <= 0.01, not usable) and give no silhouette edge
        self.assertEqual([list(p) for p in out], [list(planes[2])])
        # tilted light: bottom and far face it, the sides do not -> bottom/side and far/side edges
        L = [F(0.0), F(-0.8944272), F(0.4472136)]
        out = usc.calculate_shadow_caster_cull(False, planes, c2w, [F(0)] * 3, F(1.0), F(1.0),
                                               usc.LIGHT_DIRECTIONAL, L)
        kept = [list(p) for p in out[:2]]
        self.assertEqual(kept, [list(planes[2]), list(planes[5])])
        self.assertEqual(len(out), 8)          # 2 kept + 2 side edges + near/bottom + far/left,right,top
        for p in out[2:]:
            # every silhouette plane contains the light direction and is unit length
            self.assertLess(abs(float(p[0] * L[0] + p[1] * L[1] + p[2] * L[2])), 1e-6)
            self.assertAlmostEqual(float(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), 1.0, places=5)

    def test_kernel(self):
        planes = [[F(1), F(0), F(0), F(0)]]                   # keep x >= 0
        node = {'layer': 3, 'flags': 0, 'aabb': [F(1), F(0), F(0), F(0.5), F(0.5), F(0.5)]}
        self.assertTrue(usc.kernel_node_visible(node, planes, 1 << 3, usc.SCENE_MASK_DEFAULT))
        self.assertFalse(usc.kernel_node_visible(node, planes, 1 << 4, usc.SCENE_MASK_DEFAULT))
        outside = dict(node, aabb=[F(-1), F(0), F(0), F(0.5), F(0.5), F(0.5)])
        self.assertFalse(usc.kernel_node_visible(outside, planes, 1 << 3, usc.SCENE_MASK_DEFAULT))
        touching = dict(node, aabb=[F(-0.5), F(0), F(0), F(0.5), F(0.5), F(0.5)])   # rad == -dist: kept
        self.assertTrue(usc.kernel_node_visible(touching, planes, 1 << 3, usc.SCENE_MASK_DEFAULT))
        removed = dict(node, flags=usc.NODE_PENDING_REMOVAL)
        self.assertFalse(usc.kernel_node_visible(removed, planes, 1 << 3, usc.SCENE_MASK_DEFAULT))
        with self.assertRaises(usc.ShadowCullGap):
            usc.kernel_node_visible(dict(node, flags=5), planes, 1 << 3, usc.SCENE_MASK_DEFAULT)

    def test_layer_distance_combine(self):
        out = usc.combine_layer_distances([0, 5, 0, 7], [0, 0, 3, 2])
        self.assertEqual([float(x) for x in out], [0.0, 5.0, 3.0, 2.0])

    def test_projection_near_far(self):
        p = um.set_perspective(23.0, 0.75, 0.3, 1000.0)
        self.assertEqual(bits(usc.projection_near(True, 0.3, p)), bits(F(0.3)))
        jit = list(p)
        jit[8] = F(0.001)                                    # CTAA jitter: off-diagonal -> clip planes
        self.assertEqual(bits(usc.projection_near(False, 0.3, jit)), bits(F(0.3)))
        self.assertEqual(bits(usc.projection_far(False, 1000.0, jit)), bits(F(1000.0)))
        self.assertAlmostEqual(float(usc.projection_near(False, 0.3, p)), 0.3, places=4)


@unittest.skipUnless(V7.exists() and V7_TEMPLATE.exists(), 'v7 session / template report not present')
class CasterSetAgainstCapture(unittest.TestCase):
    def test_render0_yaw0(self):
        from src.render.hs2pipe import scene_v5, shader_library as sl, unity_forward_plan as up
        from tools.offline_renderer import frame_orchestrate_paired as fop
        from tools.offline_renderer.frame_structural_diff import LIB_ROOTS
        n = '2daddb57cebe4294bdf792ffaf382127'
        W = V7 / ('window_' + n)
        rep = json.loads(V7_TEMPLATE.read_text(encoding='utf-8'))
        cap = []
        for d in rep['draws']:
            if d['pass_label'] == 'shadow_caster' and d['render'] == 0 and d['renderer'] not in cap:
                cap.append(d['renderer'])
        scene_raw = json.loads((W / 'frames/scene_005.json').read_text(encoding='utf-8'))
        renders = json.loads((W / 'frames/renders_005.json').read_text(encoding='utf-8'))
        seed = scene_v5.prior_seed(W, 5)
        recs = [r for r in scene_v5.resolve_records(renders, seed)
                if r['camera']['name'] == 'Main Camera' and r['phase'] == 'pre_render']
        r0 = [r for r in recs if r.get('bridge_render')][0]
        scene = scene_v5.normalize(scene_raw, renders, 'Main Camera', recs.index(r0), seed)
        req = json.loads((V7 / ('bridge_body_%s.json' % n)).read_text(encoding='utf-8'))
        fop.bridge_camera_pose(scene, req, str(V7 / 'prepare_geometry_actor.json'))
        lib = sl.ShaderLibrary([ROOT / 'outputs/offline_renderer_shader_library_20261020/library_v1' / x
                                for x in LIB_ROOTS])
        ad = up.SceneAdapter(scene, None, json.loads((W / 'static.json').read_text(encoding='utf-8')), lib)
        cam = ad.camera('Main Camera')
        lights = up.light_records(ad, cam)
        q = scene['statics']['UnityEngine.QualitySettings']
        res = fop.generated_caster_cull(ad, cam, q, lights['lights'][0]['raw']['properties'], scene)
        by = {r['id']: r['name'] for r in ad.renderers()}
        self.assertEqual(sorted(by[i] for i in res['ids']), sorted(cap))
        self.assertFalse(res['cull_matrix_implicit_branch'])


class MotionVectorInclusion(unittest.TestCase):
    SHADER_NO_MV = {'active_subshader': 0, 'subshaders': [{'passes': [{'light_mode': 'FORWARDBASE'}]}]}
    SHADER_MV = {'active_subshader': 0, 'subshaders': [{'passes': [{'light_mode': 'MOTIONVECTORS'}]}]}
    I = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]

    def native(self, flags, moved=False, last=None):
        m = list(self.I)
        if moved:
            m[12] = 0.5
        n = {'base_renderer_flags_0xf0': '0x%08x' % flags, 'base_renderer_local_to_world_0x08': m,
             'base_renderer_previous_local_to_world_0x48': list(self.I), 'base_renderer_transform_update_frame_0xbc': 7}
        if last is not None:
            n['smr_has_last_position_data_0x28d'] = last
        return n

    def test_rules(self):
        smr, mr = {'kind': 'SkinnedMeshRenderer'}, {'kind': 'MeshRenderer'}
        obj = 0x400                                   # mode Object (bits 10..11 = 1)
        self.assertEqual(ucp.motion_vector_included(smr, self.native(obj, last=1), self.SHADER_NO_MV)[0], True)
        # bit 9 comes from SkinnedMeshRenderer+0x28d, not from the flattened word
        self.assertEqual(ucp.motion_vector_included(smr, self.native(obj | 0x200, last=0), self.SHADER_NO_MV)[0], False)
        self.assertEqual(ucp.motion_vector_included(mr, self.native(obj), self.SHADER_NO_MV)[0], False)
        self.assertEqual(ucp.motion_vector_included(mr, self.native(obj), self.SHADER_MV)[0], True)
        self.assertEqual(ucp.motion_vector_included(mr, self.native(0x800), self.SHADER_NO_MV)[0], True)
        self.assertEqual(ucp.motion_vector_included(mr, self.native(0), self.SHADER_MV)[0], False)
        self.assertIsNone(ucp.motion_vector_included(mr, self.native(obj, moved=True), self.SHADER_NO_MV)[0])
        self.assertTrue(ucp.motion_vector_included(mr, self.native(obj, moved=True), self.SHADER_NO_MV, 7)[0])
        self.assertFalse(ucp.motion_vector_included(mr, self.native(obj, moved=True), self.SHADER_NO_MV, 8)[0])

    def test_compare_approximately(self):
        a = list(self.I)
        b = list(self.I)
        b[3] = float(np.float32(1e-5))
        self.assertTrue(ucp.compare_approximately(a, b, usc.EPS_V3))
        b[3] = 2e-5
        self.assertFalse(ucp.compare_approximately(a, b, usc.EPS_V3))
        b[3] = float('nan')
        self.assertFalse(ucp.compare_approximately(a, b, usc.EPS_V3))


class TexEnvResolution(unittest.TestCase):
    @staticmethod
    def entry(name, tid, dim=2, bdef=0xFFFFFFFF):
        raw = struct.pack('<IiIIII', tid, dim, 2, 0, 0, bdef) + b'\0' * 12
        return {'name': name, 'type': 'texture', 'value_hex': raw.hex()}

    def test_resolve(self):
        mat = {'entries': [self.entry('_MainTex', 492, bdef=0xC0000000), self.entry('_Mask', 0, bdef=0xC0000000),
                           self.entry('_Bump', 0, bdef=0xC0000009), self.entry('_Cube', 0, dim=4, bdef=0xC000000A)]}
        glob = {'entries': [self.entry('_Global', 77, bdef=0xFFFFFFFF)]}
        self.assertEqual(bd.resolve_texenv('_MainTex', 2, mat, glob), ('texture', 492, 'material'))
        self.assertEqual(bd.resolve_texenv('_Mask', 2, mat, glob), ('builtin', 0, 'material'))
        self.assertEqual(bd.resolve_texenv('_Bump', 2, mat, glob), ('builtin', 9, 'material'))
        self.assertEqual(bd.resolve_texenv('_Cube', 4, mat, glob), ('builtin', 10, 'material'))
        self.assertEqual(bd.resolve_texenv('_Global', 2, mat, glob), ('texture', 77, 'global'))
        self.assertEqual(bd.resolve_texenv('_Absent', 3, mat, glob), ('default', 3, None))
        # a 2D texture does not serve a cube request: falls through to the default TexEnv
        self.assertEqual(bd.resolve_texenv('_MainTex', 4, mat, glob), ('default', 4, None))
        self.assertEqual(bd.BUILTIN_TEXENV_TEXTURE[0], 'UnityWhite')
        self.assertEqual(bd.DEFAULT_TEXENV_BY_DIM[3], 'UnityDefault3D')

    def test_builtin_settings(self):
        self.assertEqual(bd.builtin_texture_settings('UnityWhite'),
                         {'filterMode': 'Bilinear', 'wrapMode': 'Repeat', 'anisoLevel': 0, 'mipMapBias': 0})
        self.assertEqual(bd.builtin_texture_settings('UnityBlackCube')['wrapMode'], 'Clamp')
        self.assertEqual(bd.builtin_texture_settings('UnityBlackCube')['anisoLevel'], 1)
        self.assertEqual(bd.builtin_texture_settings('UnityDefault3D'),
                         {'filterMode': 'Bilinear', 'wrapMode': 'Repeat', 'anisoLevel': 1, 'mipMapBias': 0})
        self.assertEqual(bd.builtin_texture_settings('UnityHalo')['wrapMode'], 'Clamp')


if __name__ == '__main__':
    unittest.main()


V7_ACCEPT = Path('E:/HS2_offline_renderer_data/multiview_20261025/v7/w_2daddb57/resource_acceptance/resources_table.json')
CARD = Path('E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex/Chenger_chin_baseline_DINOv2_v1_20261011.png')


@unittest.skipUnless(V7.exists() and V7_ACCEPT.exists() and CARD.exists(), 'v7 session / acceptance table / card not present')
class MaterialSources(unittest.TestCase):
    """hs2pipe.material_sources on the yaw-0 window: each material's installed shader object."""

    @classmethod
    def setUpClass(cls):
        from src.render.hs2pipe import material_sources as ms
        from src.utils.chara_loader.AiSyoujyoCharaData import AiSyoujyoCharaData
        scene = json.loads((V7 / 'window_2daddb57cebe4294bdf792ffaf382127/frames/scene_005.json').read_text(encoding='utf-8'))
        rows = json.loads(V7_ACCEPT.read_text(encoding='utf-8'))
        hst = int(AiSyoujyoCharaData.load(str(CARD))['Custom']['hair']['shaderType'])
        lib_sources = [str(Path('E:/HoneySelect2_ArcticFox/abdata/chara') / x) for x in
                       ('mm_base.unity3d', 'oo_base.unity3d', '00/fo_inner_t_00.unity3d', '00/fo_inner_b_00.unity3d')]
        cls.out = {v['renderer']: v for v in ms.resolve(scene, rows, lib_sources, hst).values()}
        cls.hst = hst

    def test_prefab_pptr_chain(self):
        top = self.out['o_top_camisole2_a']
        self.assertTrue(top['shader_source'].lower().endswith('fo_top_00.unity3d'))
        self.assertEqual(top['shader_name'], 'AIT/Clothes True')
        tang = self.out['o_tang']
        self.assertTrue(tang['shader_source'].lower().endswith('fo_head_30.unity3d'))

    def test_runtime_rules(self):
        hair = self.out['Kamael_Valentine_Hair_03_HQ']        # ChaControl.ChangeSettingHairShader
        self.assertEqual(self.hst, 0)
        self.assertTrue(hair['shader_source'].lower().endswith('hair_shader_mat.unity3d'))
        self.assertEqual(hair['shader_path_id'], -8694255534545872754)
        self.assertIn('ChangeSettingHairShader', hair['shader_set_by'])
        head = self.out['o_head']                              # list MatData from the head's MainAB
        self.assertEqual(head['material_asset'], 'cf_m_skin_head_02')
        self.assertTrue(head['shader_source'].lower().endswith('fo_head_30.unity3d'))
        self.assertFalse(any('unresolved' in v for v in self.out.values()))


GGM = Path('E:/HoneySelect2_ArcticFox/HoneySelect2_Data/globalgamemanagers')


@unittest.skipUnless(GGM.exists(), 'installed globalgamemanagers not present')
class GraphicsSettingsBuiltins(unittest.TestCase):
    def test_builtin_shader_objects(self):
        from src.render.hs2pipe import graphics_settings as gs, shader_library as sl
        from tools.offline_renderer.frame_structural_diff import LIB_ROOTS
        ref = gs.builtin_shader_objects()
        self.assertEqual((ref['m_DepthNormals']['mode'], ref['m_DepthNormals']['path_id']), (1, 62))
        self.assertTrue(ref['m_DepthNormals']['file'].endswith('unity_builtin_extra'))
        lib = sl.ShaderLibrary([ROOT / 'outputs/offline_renderer_shader_library_20261020/library_v1' / r for r in LIB_ROOTS])
        for name, setting in (('Hidden/Internal-DepthNormalsTexture', 'm_DepthNormals'),
                              ('Hidden/Internal-MotionVectors', 'm_MotionVectors'),
                              ('Hidden/Internal-ScreenSpaceShadows', 'm_ScreenSpaceShadows')):
            self.assertEqual([gs.library_copy_is_setting(c, setting) for c in lib.copies(name)], [True])


class InputAssembler(unittest.TestCase):
    """hs2pipe.ia_bindings rules (ExtractMeshBuffers 0x7a3f40, AddDefaultStreamsToMeshBuffers 0x793760)."""
    # o_head-like format: stream 0 pos/normal/tangent, stream 1 color/uv0..2, stream 2 blend weight/indices
    CH = [(0, True), (0, True), (0, True), (1, True), (1, True), (1, True), (1, True)] + [(0, False)] * 5 + \
         [(2, True), (2, True)]
    FMT = {'channels': CH, 'streams': [40, 28, 32], 'vertex_count': 4,
           'submeshes': [{'indexCount': 6, 'firstByte': 12, 'baseVertex': 3}], 'index_size': 2}

    def test_default_stream_rule(self):
        from src.render.hs2pipe import ia_bindings as ia
        avail = ia.available_mask(self.CH)
        self.assertFalse(ia.needs_default_stream(0x17, avail))         # vertex/normal/tangent/uv0 present
        no_color = avail & ~(1 << 3)
        self.assertTrue(ia.needs_default_stream(1 << 3, no_color))      # Color missing -> default
        self.assertFalse(ia.needs_default_stream(1 << 7, avail))        # TexCoord3 missing, TexCoord0 present
        self.assertTrue(ia.needs_default_stream(1 << 4, 0x7))            # texcoords required, none present
        self.assertEqual(ia.required_stream_count(self.CH, 0x1), 1)
        self.assertEqual(ia.required_stream_count(self.CH, 0x11), 2)

    def test_stream_plan_and_args(self):
        from src.render.hs2pipe import ia_bindings as ia
        self.assertEqual(ia.stream_plan(self.FMT, 0xFF, skinned=True),
                         ['skin', ('mesh', 1), ('mesh', 2)])     # TexCoord3 missing but TexCoord0 present: no default
        self.assertEqual(ia.stream_plan(self.FMT, 0x1, skinned=True, previous_stream=True), ['skin', 'skin_previous'])
        self.assertEqual(ia.stream_plan(self.FMT, 0x11, skinned=True, previous_stream=True),
                         ['skin', ('mesh', 1), 'skin_previous'])
        self.assertEqual(ia.draw_args(self.FMT, 0), {'IndexCount': 6, 'StartIndexLocation': 6, 'BaseVertexLocation': 3})
