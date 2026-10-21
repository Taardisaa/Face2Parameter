"""Tests for src/render/hs2pipe/resources.py and sideloader_assets.py.

CPU only.  Synthetic tests always run; installed-data tests run when the game
directory exists (read-only); the receipt test checks the committed acceptance
receipt.  Run: python -m unittest tools.offline_renderer.test_resources -v
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import resources as R  # noqa: E402
from src.render.hs2pipe import sideloader_assets as S  # noqa: E402

RECEIPT = Path(__file__).with_name('resources_acceptance_20261020.json')
GAME = R.GAME_ROOT


class Computed(unittest.TestCase):
    def test_net_random(self):
        self.assertEqual(R.NetRandom(0).internal_sample(), 1559595546)
        self.assertEqual(R.NetRandom(102).next(5), 0)

    def test_builtin_textures(self):
        e = R.builtin_texture_entry('UnityNormalMap')
        self.assertEqual(e.desc['Format'], 28)
        self.assertEqual(e.initial[0]['data'], b'\x7f\x7f\xff\xff' * 16)
        w = R.builtin_texture_entry('UnityWhite')
        self.assertEqual(w.desc['Format'], 29)
        cube = R.builtin_texture_entry('UnityDefaultCube')
        self.assertEqual((cube.kind, cube.desc['ArraySize'], cube.desc['MiscFlags'], len(cube.initial)), ('texcube', 6, 4, 6))
        self.assertEqual(R.builtin_texture_entry('UnityDefault3D').initial[0]['data'], b'\x80' * 4)

    def test_default_vertex_buffer(self):
        e = R.default_vertex_buffer_entry(0, 1)
        self.assertEqual(e.initial[0]['data'], bytes.fromhex('0000ff00ff0000ffffffffff00000000'))
        self.assertEqual(R.default_vertex_buffer_entry(0, 3).desc['ByteWidth'], 64)

    def test_skybox_vb(self):
        e = R.skybox_cubemap_vertex_buffer_entry()
        self.assertEqual(e.desc['ByteWidth'], 141120)
        # byte-equal to capture resource 199 of paired frame 26059
        self.assertEqual(e.content_hash(), 'b3eab359d2cd1c56136f1ef193153988ab27006f9ba14268b731bf63ea6080eb')

    def test_ppv2_objects(self):
        vb, ib = R.ppv2_fullscreen_triangle_entries()
        self.assertEqual(struct.unpack('<9f', vb.initial[0]['data']), (-1, -1, 0, -1, 3, 0, 3, -1, 0))
        self.assertEqual(ib.initial[0]['data'], struct.pack('<3H', 0, 1, 2))
        self.assertEqual(R.ppv2_white_texture_entry().initial[0]['data'], b'\xff' * 4)


def _tree(fmt, w, h, mips, data, cs=1, images=1, dim=2, name='t'):
    return {'m_Name': name, 'm_TextureFormat': fmt, 'm_Width': w, 'm_Height': h, 'm_MipCount': mips,
            'm_CompleteImageSize': len(data) // images, 'm_ImageCount': images, 'm_TextureDimension': dim,
            'm_ColorSpace': cs, 'm_TextureSettings': {}}


class TextureUpload(unittest.TestCase):
    def test_dxt1_chain_and_srgb(self):
        sizes = [max(1, -(-max(1, 8 >> m) // 4)) * max(1, -(-max(1, 4 >> m) // 4)) * 8 for m in range(4)]
        data = bytes(range(256)) * 2
        data = data[:sum(sizes)]
        e = R.texture_entry_from_tree(_tree(10, 8, 4, 4, data), data)
        self.assertEqual(e.desc['Format'], 72)
        self.assertEqual([len(s['data']) for s in e.initial], sizes)
        lin = R.texture_entry_from_tree(_tree(10, 8, 4, 4, data, cs=0), data)
        self.assertEqual(lin.desc['Format'], 71)

    def test_rgb24_and_argb32(self):
        e = R.texture_entry_from_tree(_tree(3, 2, 1, 1, b'\x01\x02\x03\x04\x05\x06'), b'\x01\x02\x03\x04\x05\x06')
        self.assertEqual(e.initial[0]['data'], b'\x01\x02\x03\xff\x04\x05\x06\xff')
        a = R.texture_entry_from_tree(_tree(5, 1, 1, 1, b'\x0a\x01\x02\x03'), b'\x0a\x01\x02\x03')
        self.assertEqual(a.initial[0]['data'], b'\x01\x02\x03\x0a')

    def test_cubemap_order(self):
        data = b''.join(bytes([i]) * 4 for i in range(6))
        t = _tree(4, 1, 1, 1, data, images=6, dim=4)
        e = R.texture_entry_from_tree(t, data, unity_type='Cubemap')
        self.assertEqual([s['data'][0] for s in e.initial], list(range(6)))
        self.assertEqual(e.srv['ViewDimension'], 9)

    def test_rejections(self):
        with self.assertRaises(R.ResourceUnresolved):
            R.texture_entry_from_tree(_tree(28, 4, 4, 1, b'\0' * 8), b'\0' * 8)
        t = _tree(4, 2, 2, 1, b'\0' * 16)
        t['m_CompleteImageSize'] = 12
        with self.assertRaises(R.ResourceUnresolved):
            R.texture_entry_from_tree(t, b'\0' * 16)


class MeshUpload(unittest.TestCase):
    def test_streams(self):
        n = 3
        ch = [{'stream': 0, 'offset': 0, 'format': 0, 'dimension': 0}] * 14
        ch = [dict(c) for c in ch]
        ch[0] = {'stream': 0, 'offset': 0, 'format': 0, 'dimension': 3}
        ch[4] = {'stream': 1, 'offset': 0, 'format': 0, 'dimension': 2}
        s0 = bytes(range(36))
        s1 = bytes(range(100, 124))
        data = s0 + b'\0' * 12 + s1          # stream 1 starts at the next 16-byte boundary (48)
        tree = {'m_Name': 'm', 'm_MeshCompression': 0, 'm_IndexFormat': 0, 'm_IndexBuffer': b'\0\0\1\0\2\0',
                'm_SubMeshes': [], 'm_VertexData': {'m_VertexCount': n, 'm_Channels': ch, 'm_DataSize': data}}
        es = R.mesh_entries_from_tree(tree, skinned=True)
        self.assertEqual([e.role for e in es], ['vertex_stream_0', 'vertex_stream_1', 'index_buffer'])
        self.assertEqual(es[0].initial[0]['data'], s0)
        self.assertEqual(es[1].initial[0]['data'], s1)
        self.assertFalse(es[0].extra['bound_by_frame'])
        self.assertTrue(es[1].extra['bound_by_frame'])


class Sideloader(unittest.TestCase):
    def test_versions(self):
        self.assertGreater(S.compare_versions('1.0.4', '1.0.3'), 0)
        self.assertGreater(S.compare_versions('1.10', '1.9'), 0)
        self.assertEqual(S.compare_versions('2', '2'), 0)

    def test_culture_order(self):
        self.assertEqual(S.culture_order(['HS2_UncensorSelector.Base', 'com.ArcticFox.Munc_mosaic_removal_patch']),
                         ['com.ArcticFox.Munc_mosaic_removal_patch', 'HS2_UncensorSelector.Base'])
        with self.assertRaises(S.SideloaderUnresolved):
            S.culture_order(['a-b', 'ab', 'a_c'])

    def test_state(self):
        def arc(path, guid, ver, members, games=(), mtime=0):
            return {'path': path, 'mtime_ns': mtime, 'size': 1, 'schema_ver': '1', 'guid': guid, 'version': ver,
                    'games': list(games), 'unity3d': members, 'png': [], 'uncensor_selector': False}
        doc = {'schema': S.SCAN_SCHEMA, 'archives': [
            arc('a/x.zipmod', 'b.mod', '1.0', ['abdata/chara/x.unity3d']),
            arc('a/longer_x.zipmod', 'b.mod', '1.0', ['abdata/chara/x.unity3d']),
            arc('a/y.zipmod', 'a.mod', '2', ['abdata/chara/x.unity3d']),
            arc('a/z.zipmod', 'c.mod', '1', [], games=['KoikatsuSunshine'])]}
        st = S.SideloaderState(doc)
        self.assertEqual([r['guid'] for r in st.loaded], ['a.mod', 'b.mod'])
        self.assertEqual(st.duplicates[0]['loaded'], 'a/longer_x.zipmod')
        self.assertEqual([g for _, _, g in st.bundle_sources('chara/x.unity3d')], ['a.mod', 'b.mod'])
        self.assertEqual(len(st.skipped), 1)


class XUnity(unittest.TestCase):
    def test_key(self):
        self.assertEqual(R.XUnityTextureTranslations.key('SpriteAtlasTexture-sp_ai_pouch-1024x1024-fmt4'), '4F5D94683E')


@unittest.skipUnless(RECEIPT.exists(), 'acceptance receipt missing')
class Receipt(unittest.TestCase):
    def test_no_mismatch(self):
        r = json.loads(RECEIPT.read_text(encoding='utf-8'))
        # Only Maker list thumbnails resolved by name without their list row (UI state) differ.
        self.assertTrue(all(m['name'] in ('p_cf_head_02', 'preset_head_02') for m in r['mismatch']))
        self.assertGreater(len(r['byte_exact']), 150)
        for row in r['byte_exact']:
            self.assertIn(row['provenance']['class'], ('asset', 'computed'))


@unittest.skipUnless((GAME / 'abdata/chara/38/fo_head_38.unity3d').exists(), 'game install not present')
class Installed(unittest.TestCase):
    def test_o_head_streams(self):
        files = R.SourceFiles(max_open=1)
        es = R.asset_mesh_entries(files, str(GAME / 'abdata/chara/38/fo_head_38.unity3d'), 1210433472884616197, skinned=True)
        self.assertEqual([e.desc['ByteWidth'] for e in es], [177560, 124292, 142048, 51096])
        self.assertFalse(es[0].extra['bound_by_frame'])


if __name__ == '__main__':
    unittest.main()
