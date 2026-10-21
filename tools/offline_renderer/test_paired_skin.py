"""CPU checks for the same-frame pairing analyzer (synthetic data only)."""
from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import gpu_skin_inputs as g  # noqa: E402
from src.render import paired_skin as ps  # noqa: E402
from tools.offline_renderer.test_gpu_skin_inputs import Builder, rid, u  # noqa: E402

CTX = 7


def enc(a, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(a, dtype).tobytes()).decode()


def make_window(root: Path, rng):
    n = 5
    tri = np.array([0, 1, 2, 2, 3, 4], np.int32)
    w = np.array([[0.6, 0.4, 0, 0]] * n, np.float32)
    bi = np.array([[0, 1, 0, 0]] * n, np.int32)
    bind = np.stack([np.eye(4), np.eye(4)]).astype(np.float32)
    bind[1, :3, 3] = [0.01, -0.02, 0.03]
    mesh = {'name': 'o_test', 'id': 77, 'vertex_count': n, 'vertices_f32': enc(rng.random((n, 3))),
            'normals_f32': enc(rng.random((n, 3))), 'tangents_f32': enc(np.c_[rng.random((n, 3)), np.ones(n)]),
            'uv_f32': enc(rng.random((n, 2))), 'uv2_f32': enc(np.zeros((0, 2))), 'triangles_i32': enc(tri, np.int32),
            'bindposes_f32_rowmajor': enc(bind), 'bone_weights_f32': enc(w), 'bone_indices_i32': enc(bi, np.int32),
            'blendshapes': [{'name': 'x.e00', 'frames': [{'weight': 100.0, 'dv_f32': enc(rng.random((n, 3)) * 1e-3),
                                                          'dn_f32': enc(np.zeros((n, 3))), 'dt_f32': enc(np.zeros((n, 3)))}]}]}
    (root / 'frames').mkdir(parents=True)
    (root / 'mesh_00_o_test.json').write_text(json.dumps(mesh))
    static = {'transforms': [{'i': i, 'id': 100 + i, 'name': f't{i}', 'parent': i - 1} for i in range(3)],
              'renderers': [{'id': 9, 'name': 'o_test', 'transform': 0, 'root_bone': 1, 'bones': [1, 2],
                             'mesh': {'id': 77, 'name': 'o_test', 'vertex_count': n, 'data_file': 'mesh_00_o_test.json'}}]}
    (root / 'static.json').write_text(json.dumps(static))
    for k, fc in enumerate((500, 501)):
        T = np.zeros((3, 33), np.float32)
        for i in range(3):
            q = Rotation.random(random_state=k * 10 + i).as_quat()
            p = rng.random(3) + [0, 14, 0]
            M = np.eye(4)
            M[:3, :3] = Rotation.from_quat(q).as_matrix() * 0.9
            M[:3, 3] = p
            T[i, 10:13] = p
            T[i, 13:17] = q
            T[i, 17:33] = M.ravel()
        row = {'index': k, 'frame_count': fc, 'transforms_f32': enc(T),
               'renderers': [{'id': 9, 'weights_f32': enc([100.0])}]}
        (root / 'frames' / f'frame_{k:03d}.json').write_text(json.dumps(row))
    return ps.ProbeWindow(root)


class PairedSkinTests(unittest.TestCase):
    def test_float32_helpers(self):
        rng = np.random.default_rng(0)
        a, b = rng.random((4, 4)), rng.random((4, 4))
        self.assertLess(np.abs(ps.matmul_f32(a, b) - a @ b).max(), 1e-6)
        q = Rotation.random(random_state=4).as_quat()
        p = [0.1, 14.4, -0.2]
        inv = ps.trs_inverse_f32(p, q).astype(float)
        self.assertLess(np.abs(inv @ ps._trs(p, q) - np.eye(4)).max(), 5e-6)

    def test_variants_agree_with_reference_and_detect_exactness(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            win = make_window(Path(tmp), rng)
            fr = win.by_frame[501]
            r = win.static['renderers'][0]
            mats = ps.frame_matrices(win, fr, r)
            variants = ps.skin_variants(win.meshes[9], mats, fr['W'][9])
            ref = np.frombuffer(variants['f64_reference'], np.float32)
            for name, data in variants.items():
                self.assertLess(np.abs(np.frombuffer(data, np.float32) - ref).max(), 1e-4, name)
            exact = ps.compare_words(variants['f32_paletteA_blend_matrix'], variants['f32_paletteA_blend_matrix'])
            self.assertTrue(exact['bit_exact'])
            self.assertEqual(exact['max_ulp'], 0)
            other = win.by_frame[500]
            moved = ps.skin_variants(win.meshes[9], ps.frame_matrices(win, other, r), other['W'][9])
            self.assertFalse(ps.compare_words(variants['f32_paletteA_blend_matrix'], moved['f32_paletteA_blend_matrix'])['bit_exact'])

    def test_marker_found_only_inside_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            nonce = bytes(range(16))
            rec = bytearray(64)
            rec[0:16] = ps.MAGIC
            rec[16:32] = nonce
            rec[32:36] = (501).to_bytes(4, 'little', signed=True)
            rec[36:40] = (3).to_bytes(4, 'little', signed=True)
            rec[44:48] = (~501).to_bytes(4, 'little', signed=True)
            rec[48:64] = ps.TAIL
            b = Builder()
            b.chunk('Internal::DeviceInitialisation', rid('ImmediateContext', CTX))
            stale = bytearray(rec)
            stale[32:36] = (499).to_bytes(4, 'little', signed=True)
            stale[44:48] = (~499).to_bytes(4, 'little', signed=True)
            b.chunk('ID3D11Device::CreateTexture2D', b.buf('SubresourceContents', bytes(stale)) + rid('pTexture', 5))
            state = ('<struct name="state"><struct name="IA"><ResourceId name="Layout">0</ResourceId></struct></struct>')
            b.chunk('Internal::Beginning of Capture', state + u('numAnnotations', 0))
            b.chunk('ID3D11DeviceContext::UpdateSubresource', rid('Context', CTX) + rid('pDstResource', 5) + b.buf('pSrcData', bytes(rec)))
            index = g.CaptureIndex(b.write(Path(tmp)))
            try:
                rows = ps.find_markers(index, nonce)
            finally:
                index.close()
            self.assertEqual([(r['frame_count'], r['nonce_ok'], r['window_index']) for r in rows], [(501, True, 3)])


if __name__ == '__main__':
    unittest.main()


DLL = Path('E:/HoneySelect2_ArcticFox/UnityPlayer.dll')


@unittest.skipUnless(DLL.exists(), 'installed UnityPlayer.dll required')
class UnityEmuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.render.unity_skin_emu import UnityPlayerEmu
        cls.emu = UnityPlayerEmu()

    def test_multiply_with_base_order(self):
        rng = np.random.default_rng(5)
        base = rng.random((4, 4)).astype(np.float32)
        a = [rng.random((4, 4)).astype(np.float32) for _ in range(3)]
        b = [rng.random((4, 4)).astype(np.float32) for _ in range(3)]
        got = self.emu.multiply_with_base(base, a, b)
        for i in range(3):
            # Disassembly: res = base . (a . b), each column ((c0x + c1y) + c2z) + c3w.
            self.assertTrue(np.array_equal(got[i], ps.matmul_f32(base, ps.matmul_f32(a[i], b[i]))))

    def test_vertex_kernel_matches_declared_matrix_blend(self):
        rng = np.random.default_rng(6)
        n = 7
        src = np.c_[rng.random((n, 9)), np.sign(rng.random(n) - .5)].astype(np.float32)
        pal = [rng.random((4, 4)).astype(np.float32) for _ in range(5)]
        for m in pal:
            m[3] = [0, 0, 0, 1]
        w = rng.random((n, 4)).astype(np.float32)
        bi = rng.integers(0, 5, (n, 4)).astype(np.int32)
        got = self.emu.skin_vertices4(src, pal, w, bi)
        P = np.stack(pal)[bi]
        M = (P[:, 0] * w[:, 0, None, None]).astype(np.float32)
        for k in range(1, 4):
            M = (M + (P[:, k] * w[:, k, None, None]).astype(np.float32)).astype(np.float32)
        expected = ps._apply(M, src[:, 0:3], src[:, 3:6], src[:, 6:9], src[:, 9])
        self.assertEqual(got, expected)

    def test_global_position_of_single_transform(self):
        from src.render.unity_skin_emu import global_position_rotation
        q = Rotation.random(random_state=3).as_quat().astype(np.float32)
        p, r = global_position_rotation(self.emu, [([0.5, 1.0, -2.0], q, [1, 1, 1])])
        self.assertTrue(np.array_equal(p, np.float32([0.5, 1.0, -2.0])))
        self.assertTrue(np.array_equal(r, q))


@unittest.skipUnless(DLL.exists(), 'installed UnityPlayer.dll required')
class UnityChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.render.unity_skin_emu import shared_emulator
        cls.emu = shared_emulator()

    def test_setter_normalization_port_matches_original(self):
        from src.render.unity_skin_emu import emulated_normalize
        from src.render.unity_transform_math import normalize_quaternion
        rng = np.random.default_rng(9)
        for _ in range(300):
            q = (rng.normal(size=4) * rng.choice([1e-20, 1e-3, 1, 30])).astype(np.float32)
            self.assertTrue(np.array_equal(normalize_quaternion(q).view(np.uint32), emulated_normalize(self.emu, q).view(np.uint32)))

    def test_blendshape_port_matches_native(self):
        from src.render.unity_blendshape import BlendShapeData, apply_blendshapes, emulate_apply_blendshapes
        rng = np.random.default_rng(11)
        n = 40
        base = np.c_[rng.random((n, 9)), np.ones(n)].astype(np.float32)
        def frame(w):
            d = (rng.normal(size=(n, 3)) * 0.01).astype(np.float32)
            d[rng.random(n) < .5] = 0
            return (w, d, (d * 0.3).astype(np.float32), (d * -0.2).astype(np.float32))
        channels = [[frame(100.0)], [frame(30.0), frame(70.0), frame(100.0)], [frame(50.0), frame(150.0)]]
        data = BlendShapeData.from_dense_frames(channels)
        for w in ([100, 0, 0], [55.5, 12.0, 3.0], [-20, 85, 120], [250, 70, 50], [0.00001, 100, 150], [42, 29.999, 149.9]):
            for flags in ((True, True), (True, False), (False, False)):
                a = apply_blendshapes(data, np.float32(w), base, *flags)
                b = emulate_apply_blendshapes(self.emu, data, np.float32(w), base, *flags)
                self.assertTrue(np.array_equal(a.view(np.uint32), b.view(np.uint32)), (w, flags))

    def test_generated_shape_program_matches_enums(self):
        import json
        from src.render.shape_head_female_program import DST, SRC, UPDATE
        self.assertEqual(len(DST), 52)
        self.assertEqual(len(SRC), 102)
        self.assertEqual(len(UPDATE), 113)
        enums = Path(__file__).resolve().parents[2] / 'data/hs2_head/enums.json'
        if enums.exists():
            e = json.loads(enums.read_text())
            self.assertEqual((DST, SRC), (e['dst'], e['src']))
