"""Tests for src/render/hs2pipe/shader_library.py (variant selection, routing,
render state, cbuffer layouts/packing) and the coverage receipt.

Synthetic tests need no game data.  Installed-data tests run when the ignored
extraction under outputs/offline_renderer_shader_library_20261020/library_v1
and the 2018.4.11f1 editor CGIncludes are present.
"""
from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.render.hs2pipe import shader_library as sl  # noqa: E402

LIB = ROOT / 'outputs' / 'offline_renderer_shader_library_20261020' / 'library_v1'
LIB_ROOTS = [LIB / 'main', LIB / 'resources_standard', LIB / 'dhh_dhpost']
CGINC = ROOT.parent / 'HS2Mod' / 'zips' / 'UnitySetup64-2018.4.11f1' / 'Editor' / 'Data' / 'CGIncludes' / \
    'UnityShaderVariables.cginc'
RECEIPT = Path(__file__).with_name('shader_library_coverage_20261020.json')


def _sub(order, keywords, tier=3, gpu=15, sha=None):
    return {'order': order, 'blob_index': order, 'tier': tier, 'gpu_program_type': gpu, 'keywords': list(keywords),
            'dxbc_sha256': sha or f'sha{order}', 'bindings': {}, 'channels': {}, 'shader_requirements': None}


def _state(**override):
    v = lambda x, n='<noninit>': {'val': x, 'name': n}  # noqa: E731
    rt = {'srcBlend': v(1), 'destBlend': v(0), 'srcBlendAlpha': v(1), 'destBlendAlpha': v(0), 'blendOp': v(0),
          'blendOpAlpha': v(0), 'colMask': v(15)}
    face = {'pass': v(0), 'fail': v(0), 'zFail': v(0), 'comp': v(8)}
    st = {f'rtBlend{i}': dict(rt) for i in range(8)}
    st.update({'rtSeparateBlend': False, 'zClip': v(1), 'zTest': v(4), 'zWrite': v(1), 'culling': v(2),
               'offsetFactor': v(0), 'offsetUnits': v(0), 'alphaToMask': v(0), 'stencilOp': face,
               'stencilOpFront': face, 'stencilOpBack': face, 'stencilReadMask': v(255),
               'stencilWriteMask': v(255), 'stencilRef': v(0), 'fogMode': -1, 'lighting': False})
    st.update(override)
    return st


def _pass(index, light_mode, stages, name='P', state=None, type_=0):
    return {'index': index, 'type': type_, 'name': name, 'use_name': '', 'texture_name': '', 'tags': [],
            'state_tags': [['LIGHTMODE', light_mode]] if light_mode else [], 'state_name': name,
            'state_lod': 0, 'program_mask': 6, 'name_indices': [], 'state': state or _state(), 'stages': stages}


class FakeLibrary:
    def __init__(self):
        self.shaders = {}

    def add(self, name, subshaders, fallback=''):
        rec = {'name': name, 'fallback': fallback, 'properties': [],
               'subshaders': [{'index': i, 'lod': 0, 'tags': tags, 'passes': passes}
                              for i, (tags, passes) in enumerate(subshaders)]}
        self.shaders[name] = sl.Shader(self, rec, {'path': 'synthetic'})
        return self.shaders[name]

    def shader(self, name, source=None):
        if name not in self.shaders:
            raise sl.ShaderLibraryError('Shader not in library: ' + name)
        return self.shaders[name]


class KeywordScoreTest(unittest.TestCase):
    def test_score_is_matches_minus_16_per_unrequested_variant_keyword(self):
        self.assertEqual(sl.keyword_score({'A', 'B', 'X'}, {'A', 'B'}), 2)
        self.assertEqual(sl.keyword_score({'A'}, {'A', 'C'}), 1 - 16)
        self.assertEqual(sl.keyword_score(set(), set()), 0)

    def test_exact_variant_wins_and_unknown_requests_are_ignored(self):
        variants = [set(), {'A'}, {'A', 'B'}, {'B'}]
        self.assertEqual(sl.select_index(variants, {'A', 'B', 'UNUSED'})[0], 2)
        self.assertEqual(sl.select_index(variants, {'UNUSED'})[0], 0)

    def test_closest_match_takes_unrequested_keyword_when_every_variant_has_it(self):
        # Installed namida case: every ForwardBase variant carries _ALPHAPREMULTIPLY_ON.
        variants = [{'DIRECTIONAL', 'P'}, {'DIRECTIONAL', 'LIGHTPROBE_SH', 'P'}]
        i, score = sl.select_index(variants, {'DIRECTIONAL', 'LIGHTPROBE_SH', '_ALPHABLEND_ON'})
        self.assertEqual((i, score), (1, 2 - 16))

    def test_ties_keep_the_first_serialized_variant(self):
        self.assertEqual(sl.select_index([{'A'}, {'B'}], {'A', 'B'})[0], 0)

    def test_unsupported_variants_are_skipped(self):
        self.assertEqual(sl.select_index([{'A'}, set()], {'A'}, unsupported=[{'A'}])[0], 1)


class PassSelectionTest(unittest.TestCase):
    def setUp(self):
        self.lib = FakeLibrary()

    def test_tier_and_gpu_program_type_admission(self):
        stages = {'vs': [_sub(0, ['A'], tier=0, sha='t0'), _sub(1, ['A'], tier=2, sha='t2'),
                         _sub(2, ['A'], gpu=4, sha='gles'), _sub(3, [], sha='all')]}
        p = self.lib.add('S', [([], [_pass(0, 'FORWARDBASE', stages)])]).subshaders[0].passes[0]
        self.assertEqual(p.select('vs', {'A'}, tier=2).dxbc_sha256, 't2')
        self.assertEqual(p.select('vs', {'A'}, tier=1).dxbc_sha256, 'all')
        with self.assertRaises(sl.ShaderLibraryError):
            p.select('vs', {'A'})   # tier-specific stage without an active tier

    def test_select_reports_exactness(self):
        stages = {'vs': [_sub(0, []), _sub(1, ['DIRECTIONAL'])]}
        p = self.lib.add('S', [([], [_pass(0, 'FORWARDBASE', stages)])]).subshaders[0].passes[0]
        sel = p.select('vs', {'DIRECTIONAL'})
        self.assertTrue(sel.exact)
        self.assertFalse(p.select('vs', {'DIRECTIONAL', 'X'}).exact)

    def test_light_mode_default_and_first_pass(self):
        s = self.lib.add('S', [([], [_pass(0, None, {}), _pass(1, 'SHADOWCASTER', {}, name='A'),
                                     _pass(2, 'SHADOWCASTER', {}, name='B')])])
        sub = s.subshaders[0]
        self.assertEqual(sub.passes[0].light_mode, 'ALWAYS')
        self.assertEqual(sub.first_pass_with_light_mode('ShadowCaster').index, 1)

    def test_shadow_caster_from_fallback_chain(self):
        self.lib.add('Base', [([], [_pass(0, 'SHADOWCASTER', {'vs': [_sub(0, [])]})])])
        self.lib.add('Mid', [([], [_pass(0, 'FORWARDBASE', {'vs': [_sub(0, [])]})])], fallback='Base')
        top = self.lib.add('Top', [([], [_pass(0, 'FORWARDBASE', {'vs': [_sub(0, [])]})])], fallback='Mid')
        self.assertEqual(top.shadow_caster_pass().shader.name, 'Base')

    def test_replacement_by_tag_value(self):
        repl = self.lib.add('R', [([['RenderType', 'Opaque']], [_pass(0, None, {'vs': [_sub(0, [])]})]),
                                  ([['RenderType', 'TransparentCutout']], [_pass(0, None, {'vs': [_sub(0, [])]})])])
        obj = self.lib.add('O', [([['RenderType', 'TransparentCutout']], [_pass(0, 'FORWARDBASE', {})])])
        none = self.lib.add('N', [([['RenderType', 'Transparent']], [_pass(0, 'FORWARDBASE', {})])])
        self.assertEqual(sl.replacement_subshader(repl, 'RenderType', obj).index, 1)
        self.assertIsNone(sl.replacement_subshader(repl, 'RenderType', none))
        self.assertEqual(sl.replacement_subshader(repl, '', none).index, 0)

    def test_use_pass(self):
        self.lib.add('Lib/Src', [([], [_pass(0, 'SHADOWCASTER', {'vs': [_sub(0, [])]}, name='SHADOWCASTER')])])
        user = self.lib.add('U', [([], [dict(_pass(0, None, {}), type=1, use_name='Lib/Src/SHADOWCASTER')])])
        p = user.subshaders[0].passes[0]
        self.assertEqual(p.type, 'use')
        self.assertEqual(user.use_pass(p)[0].shader.name, 'Lib/Src')


class RenderStateTest(unittest.TestCase):
    def test_property_bound_values(self):
        st = _state(zWrite={'val': 1, 'name': '_ZWrite'}, culling={'val': 2, 'name': '_Cull'})
        with self.assertRaises(sl.ShaderLibraryError):
            sl.resolve_state(st)
        r = sl.resolve_state(st, {'_ZWrite': 0, '_Cull': 0})
        self.assertEqual((r['zWrite']['value'], r['culling']['value'], r['culling']['property']), (0, 0, '_Cull'))

    def test_d3d11_mapping(self):
        rt = {'srcBlend': {'val': 2, 'name': '<noninit>'}, 'destBlend': {'val': 0, 'name': '<noninit>'},
              'srcBlendAlpha': {'val': 2, 'name': '<noninit>'}, 'destBlendAlpha': {'val': 0, 'name': '<noninit>'},
              'blendOp': {'val': 0, 'name': '<noninit>'}, 'blendOpAlpha': {'val': 0, 'name': '<noninit>'},
              'colMask': {'val': 14, 'name': '<noninit>'}}
        d = sl.to_d3d11(sl.resolve_state(_state(rtBlend0=rt)))
        b = d['blend']['RenderTarget'][0]
        # DstColor Zero: alpha slot uses DEST_ALPHA (captured eyeshadow draw 1390).
        self.assertEqual((b['BlendEnable'], b['SrcBlend'], b['DestBlend'], b['SrcBlendAlpha']), (True, 9, 1, 7))
        self.assertEqual(b['RenderTargetWriteMask'], 7)   # Unity RGB (8|4|2) -> D3D11 R|G|B
        self.assertEqual(d['depth_stencil']['DepthFunc'], 7)   # LEqual under reversed Z
        self.assertEqual(sl.to_d3d11(sl.resolve_state(_state()), reversed_z=False)['depth_stencil']['DepthFunc'], 4)
        self.assertEqual(d['cull_d3d11_as_authored'], 3)
        off = sl.to_d3d11(sl.resolve_state(_state(zTest={'val': 0, 'name': '<noninit>'})))
        self.assertFalse(off['depth_stencil']['DepthEnable'])


class CBufferTest(unittest.TestCase):
    def layout(self):
        names = {0: 'CB', 1: 'f', 2: 'v3', 3: 'arr', 4: 'm', 5: 'i', 6: 'b', 7: 'S', 8: 'sm'}
        bindings = {'m_VectorParams': [], 'm_MatrixParams': [], 'm_ConstantBufferBindings': [
            {'m_NameIndex': 0, 'm_Index': 2}], 'm_ConstantBuffers': [{
                'm_NameIndex': 0, 'm_Size': 224,
                'm_VectorParams': [{'m_NameIndex': 1, 'm_Index': 0, 'm_ArraySize': 0, 'm_Type': 0, 'm_Dim': 1},
                                   {'m_NameIndex': 2, 'm_Index': 4, 'm_ArraySize': 0, 'm_Type': 0, 'm_Dim': 3},
                                   {'m_NameIndex': 3, 'm_Index': 16, 'm_ArraySize': 2, 'm_Type': 0, 'm_Dim': 2},
                                   {'m_NameIndex': 5, 'm_Index': 112, 'm_ArraySize': 0, 'm_Type': 1, 'm_Dim': 1},
                                   {'m_NameIndex': 6, 'm_Index': 116, 'm_ArraySize': 0, 'm_Type': 2, 'm_Dim': 1}],
                'm_MatrixParams': [{'m_NameIndex': 4, 'm_Index': 48, 'm_ArraySize': 0, 'm_Type': 0, 'm_RowCount': 4}],
                'm_StructParams': [{'m_NameIndex': 7, 'm_Index': 128, 'm_ArraySize': 0, 'm_StructSize': 64,
                                    'm_VectorMembers': [],
                                    'm_MatrixMembers': [{'m_NameIndex': 8, 'm_Index': 0, 'm_ArraySize': 0,
                                                         'm_Type': 0, 'm_RowCount': 4}]}]}]}
        return sl.layouts_from_bindings(bindings, names)[0]

    def test_layout_and_packing(self):
        cb = self.layout()
        self.assertEqual((cb.name, cb.slot, cb.size), ('CB', 2, 224))
        self.assertEqual(cb.variable('S.sm').offset, 128)
        m = np.arange(16, dtype=np.float32)
        vals = {'f': 1.5, 'v3': [1, 2, 3], 'arr': [[1, 2], [3, 4]], 'm': m, 'i': -3, 'b': True, 'S.sm': m + 100}
        data = sl.pack_cbuffer(cb, vals)
        self.assertEqual(len(data), 224)
        self.assertEqual(struct.unpack_from('<f', data, 0)[0], 1.5)
        self.assertEqual(struct.unpack_from('<3f', data, 4), (1, 2, 3))
        self.assertEqual(struct.unpack_from('<2f', data, 16), (1, 2))
        self.assertEqual(struct.unpack_from('<2f', data, 32), (3, 4))   # array element stride 16
        self.assertEqual(struct.unpack_from('<16f', data, 48), tuple(range(16)))
        self.assertEqual(struct.unpack_from('<i', data, 112)[0], -3)
        self.assertEqual(struct.unpack_from('<I', data, 116)[0], 1)
        back = sl.unpack_cbuffer(cb, data)
        np.testing.assert_array_equal(back['S.sm'], m + 100)
        mask = cb.coverage_mask()
        self.assertFalse(mask[24:32].any())   # padding after arr[0]
        self.assertTrue(mask[48:112].all())

    def test_missing_value_and_base_bytes(self):
        cb = self.layout()
        with self.assertRaises(sl.ShaderLibraryError):
            sl.pack_cbuffer(cb, {'f': 1.0})
        base = bytes(range(224))
        data = sl.pack_cbuffer(cb, {'f': 2.0}, base=base, allow_missing=True)
        self.assertEqual(data[24:32], base[24:32])

    def test_wrong_component_count(self):
        with self.assertRaises(ValueError):
            sl.pack_cbuffer(self.layout(), {'v3': [1, 2]}, allow_missing=True)


@unittest.skipUnless(CGINC.exists(), 'Unity 2018.4.11f1 editor CGIncludes not present')
class BuiltinCBufferTest(unittest.TestCase):
    def test_source_layouts(self):
        b = sl.parse_builtin_cbuffers(CGINC, sl.D3D11_STANDALONE_DEFINES)
        sizes = {k: v.size for k, v in b.items()}
        self.assertEqual(sizes['UnityPerDraw'], 176)
        self.assertEqual(sizes['UnityPerFrame'], 368)
        self.assertEqual(sizes['UnityLighting'], 768)
        self.assertEqual(sizes['UnityShadows'], 416)
        self.assertEqual(sizes['UnityPerCamera'], 144)
        self.assertEqual(b['UnityPerFrame'].variable('unity_MatrixVP').offset, 272)
        self.assertEqual(b['UnityPerFrame'].variable('unity_ShadowColor').offset, 352)
        self.assertEqual(b['UnityPerCamera'].variable('_ProjectionParams').offset, 80)
        self.assertEqual(b['UnityProbeVolume'].variable('unity_ProbeVolumeMin').offset, 96)


@unittest.skipUnless(all((r / 'library.json').exists() for r in LIB_ROOTS), 'installed shader library not extracted')
class InstalledLibraryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = sl.ShaderLibrary(LIB_ROOTS)

    def test_head_forward_variant_equals_capture(self):
        p = self.lib.shader('AIT/Skin True Face', source='fo_head_38').subshaders[0].first_pass_with_light_mode(
            'ForwardBase')
        req = {'DIRECTIONAL', 'SHADOWS_SCREEN', 'LIGHTPROBE_SH', '_EMISSION', '_NODE_686_ON',
               '_TUNINGCOLOR21OR3_ON', '_TUNINGSPEC21OR3_ON', 'SSS_PROFILES'}
        self.assertTrue(p.select('vs', req).dxbc_sha256.startswith('2c91553a2681'))   # captured VS 14302
        self.assertTrue(p.select('ps', req).dxbc_sha256.startswith('54afe1452229'))   # captured PS 14303

    def test_namida_closest_match(self):
        p = self.lib.shader('AIT/main namida', source='fo_head_38').subshaders[0].first_pass_with_light_mode(
            'ForwardBase')
        sel = p.select('ps', {'DIRECTIONAL', 'LIGHTPROBE_SH', '_ALPHABLEND_ON'})
        self.assertIn('_ALPHAPREMULTIPLY_ON', sel.subprogram.keywords)
        self.assertTrue(sel.dxbc_sha256.startswith('a1df297b75e1'))

    def test_replacement_routing(self):
        lp = self.lib.shader('Hidden/LightingPass')
        face = self.lib.shader('AIT/Skin True Face', source='fo_head_38')
        eye = self.lib.shader('AIT/Eye Translucency', source='fo_head_38')
        sub = sl.replacement_subshader(lp, 'RenderType', face)
        self.assertIsNone(sub.first_pass_with_light_mode('FORWARDBASE'))   # o_head absent from SSS lighting
        self.assertEqual(sub.first_pass_with_light_mode('SHADOWCASTER').index, 0)
        eye_sub = sl.replacement_subshader(lp, 'RenderType', eye)
        self.assertIsNone(eye_sub.first_pass_with_light_mode('SHADOWCASTER'))
        self.assertEqual(lp.shadow_caster_pass().shader.name, 'Legacy Shaders/VertexLit')
        clothes = self.lib.shader('AIT/Clothes True', source='fo_top_00')
        dn = sl.replacement_subshader(self.lib.shader('Hidden/Internal-DepthNormalsTexture'), 'RenderType', clothes)
        self.assertEqual((dn.tag('RenderType'), len(dn.passes)), ('TreeTransparentCutout', 2))

    def test_reflection_and_builtin_layouts(self):
        p = self.lib.shader('AIT/Skin True Face', source='fo_head_38').subshaders[0].passes[0]
        sp = p.select('vs', {'DIRECTIONAL', 'SHADOWS_SCREEN', 'LIGHTPROBE_SH'}).subprogram
        um = sp.uniform_map()
        self.assertEqual(um['unity_ObjectToWorld'][0], 'UnityPerDraw')
        self.assertEqual(um['unity_MatrixVP'][2].offset, 272)
        self.assertTrue(all(cb.declared_float4 is not None for cb in sp.constant_buffers() if cb.slot is not None))
        if CGINC.exists():
            report = self.lib.check_builtin_layouts(sl.parse_builtin_cbuffers(CGINC, sl.D3D11_STANDALONE_DEFINES))
            self.assertTrue(report)
            self.assertFalse(any(r['mismatches'] for r in report.values()))

    def test_dxbc_bytes_are_content_addressed(self):
        import hashlib
        sha = '54afe1452229'
        full = next(h for h in self.lib.hash_index() if h.startswith(sha))
        self.assertEqual(hashlib.sha256(self.lib.dxbc(full)).hexdigest(), full)

    def test_compute_kernel(self):
        k = self.lib.compute_kernel('Lut3DBaker', 'KGenLut3D_AcesTonemap')
        self.assertTrue(k['kernel']['dxbc_sha256'].startswith('1687501a1622'))


@unittest.skipUnless(RECEIPT.exists(), 'coverage receipt not generated')
class ReceiptTest(unittest.TestCase):
    def test_every_executed_program_is_matched_and_selection_agrees(self):
        r = json.loads(RECEIPT.read_text(encoding='utf-8'))
        for name, c in r['captures'].items():
            self.assertEqual(c['coverage']['unmatched'], [], name)
            totals = c['variant_selection_anchored']['status_totals']
            self.assertFalse(set(totals) - {'equal', 'equal_material_keywords_unknown'}, name)
        rules = r['captures']['retry_frame17675']['routing_and_selection_rules']['status_totals']
        self.assertFalse(set(rules) - {'equal', 'equal_material_keywords_unknown'})
        self.assertFalse(any(v['mismatches'] for v in r['builtin_cbuffer_check'].values()))


if __name__ == '__main__':
    unittest.main()
