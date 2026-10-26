"""Tests of the differentiable head geometry chain (src/render/diffgeo).

    .venv/Scripts/python.exe -m unittest tools.diffgeo.test_diffgeo

Needs the installed game (UnityPlayer.dll, IL.dll / Mono for the hosted EyeLook) and the paired
session data under E:/HS2_offline_renderer_data (read only).  The session tests build one
DiffGeoSession (the chin-fit front window): Mono initialises once per process, so every
session-based test shares it.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p_ in (str(ROOT), str(ROOT / 'tools' / 'offline_renderer')):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import numpy as np  # noqa: E402
import torch  # noqa: E402

F = np.float32
V61 = Path('E:/HS2_offline_renderer_data/paired_session_v6_1_20261025')
V61_ART = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_reference_fit_20261012/paired_capture_session_v6_1_20261025')
EFFECT = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_native_inverse_20261011/solver_context.json')
FIT_V2 = Path('E:/HS2_offline_renderer_data/chin_fit_20261025/deprecated/fit_v2_stopped_by_user_dfols_cancelled')
WINDOWS = Path('E:/HS2_offline_renderer_data/chin_fit_20261025/configs/windows_a2.json')


def npnorm(q):
    from src.render.unity_transform_math import normalize_quaternion
    return normalize_quaternion(q)


class NativePrimitiveTests(unittest.TestCase):
    """Lifted UnityPlayer.dll routines vs the original instructions under Unicorn, bit for bit."""

    @classmethod
    def setUpClass(cls):
        from src.render.diffgeo.fp import Arith
        from src.render import unity_skin_emu as ue
        cls.A = Arith('exact')
        cls.ue = ue
        cls.emu = ue.shared_emulator()
        cls.rng = np.random.default_rng(11)

    def rq(self):
        return npnorm(self.rng.normal(size=4).astype(F))

    def trs(self):
        s = (self.rng.uniform(0.5, 1.5, 3) * self.rng.choice([1, 1, 1, -1], 3)).astype(F)
        return (self.rng.normal(size=3).astype(F) * F(0.3), self.rq(), s)

    def test_euler_and_normalize(self):
        from src.render.diffgeo import unity as U
        for k in range(100):
            d = self.rng.uniform(-200, 200, 3).astype(F)
            if k % 4 == 0:
                d[self.rng.integers(3)] = 0
            a = self.ue.quaternion_euler_degrees(self.emu, d)
            b = U.quaternion_euler_degrees(self.A, self.A.f32(d)).numpy().astype(F)
            self.assertEqual(a.tobytes(), b.tobytes())
            q = (self.rng.normal(size=4) * self.rng.uniform(0.1, 3)).astype(F)
            self.assertEqual(npnorm(q).tobytes(), U.normalize_quaternion(self.A, self.A.f32(q)).numpy().astype(F).tobytes())

    def test_hierarchy_palette_skin(self):
        from src.render.diffgeo import unity as U
        A, T = self.A, self.A.f32
        for _ in range(20):
            chain = [self.trs() for _ in range(self.rng.integers(1, 8))]
            p, r = self.ue.global_position_rotation(self.emu, chain)
            pt, rt = U.global_position_rotation(A, [tuple(T(x) for x in c) for c in chain])
            self.assertEqual(p.tobytes(), pt.numpy().astype(F).tobytes())
            self.assertEqual(r.tobytes(), rt.numpy().astype(F).tobytes())
            m, _ = self.emu.world_to_local_noscale(p, r)
            self.assertEqual(np.asarray(m, F).tobytes(),
                             U.world_to_local_noscale(A, T(p), T(r)).numpy().astype(F).reshape(4, 4).T.tobytes())
        entries = []
        for i in range(10):
            t, q, s = self.trs()
            entries.append((int(self.rng.integers(0, i + 1)), t, q, s))
        bmap = [int(self.rng.integers(1, 11)) for _ in range(5)]
        bind = [(np.eye(4) + self.rng.normal(size=(4, 4)) * 0.2).astype(F) for _ in range(5)]
        for b in bind:
            b[3] = [0, 0, 0, 1]
        base = (np.eye(4) + self.rng.normal(size=(4, 4)) * 0.1).astype(F)
        base[3] = [0, 0, 0, 1]
        pal = self.ue.prepared_palette(self.emu, base, entries, bmap, bind)
        palt = U.prepared_palette(A, T(base.T.ravel()), [(p_, T(t), T(q), T(s)) for p_, t, q, s in entries], bmap,
                                  T(np.stack([b.T.ravel() for b in bind])))
        for k in range(5):
            self.assertEqual(np.asarray(pal[k], F).tobytes(), palt[k].numpy().astype(F).reshape(4, 4).T.tobytes())
        n = 300
        src = self.rng.normal(size=(n, 10)).astype(F)
        w = self.rng.dirichlet([1, 1, 1, 1], size=n).astype(F)
        w[::5, 2:] = 0
        bi = self.rng.integers(0, 5, size=(n, 4)).astype(np.int32)
        vb = self.emu.skin_vertices4(src, pal, w, bi)
        self.assertEqual(bytes(vb), U.skin_vertices4(A, T(src), palt, T(w), bi).numpy().astype(F).tobytes())

    def test_crt_sin_cos_gradient(self):
        from src.render.diffgeo.fp import Arith
        A = Arith('exact')
        x = A.f32([0.1, -0.7, 1.3]).requires_grad_(True)
        (A.sinf(x).sum() + A.cosf(x).sum()).backward()
        self.assertTrue(torch.allclose(x.grad, torch.cos(x.detach()) - torch.sin(x.detach())))

    def test_generated_file_up_to_date(self):
        from tools.diffgeo import derive_native
        self.assertEqual(derive_native.main(['--check']), 0)


class ForwardTests(unittest.TestCase):
    """Bone locals of every transform vs CapturedHeadForward (v6.1 chin_rs window, frame N-2)."""

    @classmethod
    def setUpClass(cls):
        from src.render.paired_skin import ProbeWindow
        from src.hs2_offline_state import CapturedHeadForward
        from tools.offline_renderer.paired_forward_context import build
        nonce = '0b3351d62a2d439f8af95db8de79d712'
        win = ProbeWindow(V61 / ('window_' + nonce))
        N = json.loads((V61 / ('window_' + nonce) / 'capture_receipt.json').read_text())['expected_capture_frame']
        geo, ctx, _ = build(V61 / 'prepare_geometry_actor.json', V61_ART / 'state_before_baseline.json', EFFECT, win, N - 2)
        cls.model = CapturedHeadForward(geo, ctx)
        cls.patch = json.loads((V61_ART / 'patch_chin_rs_wider_x108.json').read_text())['bones']

    def check(self, patches, native, mode='card_load'):
        from src.render.diffgeo.fp import Arith
        from src.render.diffgeo.forward import TorchHeadForward, local_f32_bytes, exact_local_bytes
        ex = self.model.evaluate(patches=patches, native=native, apply_count=2, baseline_mode=mode)
        tf = TorchHeadForward(self.model, Arith('exact'))
        nt = None if native is None else torch.tensor(np.asarray(native, float), requires_grad=True)
        res = tf.evaluate(patches, native=nt, apply_count=2, baseline_mode=mode)
        bad = [t for t, r in ex['bone_local'].items() if exact_local_bytes(r) != local_f32_bytes(res['bone_local'][t])]
        self.assertEqual(bad, [])
        self.assertEqual([b['branches'] for b in ex['branches']], [b['branches'] for b in res['branches']])

    def test_window_patch_length_branch(self):
        self.check(self.patch, None)

    def test_native59_with_key_ends(self):
        rng = np.random.default_rng(5)
        nat = rng.uniform(0, 1, 59)
        nat[[0, 13, 27, 58]] = [0.0, 1.0, 0.5, 1.0]
        self.check(self.patch, [float(F(v)) for v in nat])

    def test_captured_mode_and_update_baseline(self):
        """Recorded ABMX caches; with native59 the UpdateBaseline path (Reset -> native -> Collect)."""
        self.check(self.patch, None, mode='captured')
        nat = np.clip(np.asarray(self.model.identity['native59'], float) + 0.05, 0, 1)
        self.check(self.patch, [float(F(v)) for v in nat], mode='captured')

    def test_remove_and_new_modifier(self):
        self.check([{'name': 'cf_J_ChinLow', 'remove': True}], None)
        self.check([{'name': 'cf_J_NoseBridge_s', 'scale': [1.02, 1.0, 0.99], 'length': 1.0,
                     'position': [0.0, 0.001, 0.0], 'rotation': [0.5, 0.0, -0.25]}], None)


_SESSION = {}


def session():
    if 'front' not in _SESSION:
        from tools.diffgeo.forward_acceptance import window_config
        from src.render.diffgeo.session import DiffGeoSession
        cfg, _ = window_config('front')
        _SESSION['front'] = DiffGeoSession(cfg, renderers='head')
    return _SESSION['front']


class SessionTests(unittest.TestCase):
    """Torch streams vs the exact chain on the chin-fit front window; gradients vs float64 FD."""

    @classmethod
    def setUpClass(cls):
        from src.render.chin_fit import controls as C
        cls.s = session()
        cls.rows, _ = C.encode_persistent(cls.s.models[cls.s.N])
        cls.C = C

    def streams_equal(self, patch):
        from src.render.diffgeo.session import stream_bytes, matrix_bytes_o2w
        ex = self.s.exact_streams(patch)
        tr = self.s.torch_streams(patch)
        for (rid, fc), v in ex['skin_cache'].items():
            self.assertEqual(stream_bytes(tr['skin'][fc][rid]['vb']), bytes(v[0]), (rid, fc))
            self.assertEqual(matrix_bytes_o2w(tr['skin'][fc][rid]['o2w']), bytes(v[1]), (rid, fc))

    def test_streams_window_and_q_patch(self):
        self.streams_equal(self.s.window_patch)
        q = np.random.default_rng(3).uniform(-1, 1, len(self.C.CONTROLS))
        self.streams_equal(self.C.patch_from_q(q, self.rows))

    def test_bridge_camera_generated_for_yaw(self):
        """hs2pipe.bridge_camera at the window's yaw == record; yaw 90 == the right window's record."""
        from src.render.diffgeo import camera as cam
        from src.render.hs2pipe import scene_v5 as sv
        self.assertTrue(cam.check_record(self.s))
        W = json.loads(WINDOWS.read_text(encoding='utf-8'))
        wd = Path(W['bulk']) / ('window_' + W['views']['right']['nonce'])
        rc = json.loads((wd / 'capture_receipt.json').read_text(encoding='utf-8'))
        idx = sorted(rc['frame_counts_by_index']).index(rc['expected_capture_frame'])
        recs = sv.resolve_records(json.loads((wd / 'frames' / ('renders_%03d.json' % idx)).read_text(encoding='utf-8')),
                                  sv.prior_seed(wd, idx))
        pr = [r for r in recs if r['phase'] == 'pre_render' and r.get('bridge_render')
              and r['bridge_render']['ordinal'] == 0][0]
        w2c = np.asarray(pr['data']['camera_state']['matrices']['worldToCameraMatrix'], F).reshape(4, 4)
        self.assertEqual(np.asarray(cam.generated_camera(self.s, 90.0).w2c, F).tobytes(), w2c.tobytes())

    def functional(self, s, patch):
        rid = next(r for r in s.head_renderers if s.renderers[r]['name'] == 'o_head')
        tr = s.torch_streams(patch, frames=(s.N - 2,), rids=[rid])
        vb = tr['skin'][s.N - 2][rid]['vb']
        c = torch.as_tensor(np.random.default_rng(1).normal(size=(vb.shape[0], 3)))
        return (c * vb[:, 0:3]).sum()

    def test_gradient_vs_float64_fd(self):
        from src.render.diffgeo.fp import Arith
        s64 = self.s.with_arith(Arith('f64'))
        r = self.rows[0]
        x = torch.tensor(np.asarray(r['rotation'], F).astype(float), requires_grad=True)
        y = torch.tensor(np.asarray(r['scale'], F).astype(float), requires_grad=True)

        def patch(xr, ys):
            return {'bones': [dict(r, rotation=xr, scale=ys)] + self.rows[1:]}
        f = self.functional(s64, patch(x, y))
        gx, gy = torch.autograd.grad(f, [x, y])
        for vec, g, h, which in ((x, gx, 1e-4, 'rot'), (y, gy, 1e-6, 'scl')):
            for i in range(3):
                vals = []
                for sg in (1, -1):
                    v = vec.detach().clone()
                    v[i] += sg * h
                    with torch.no_grad():
                        vals.append(float(self.functional(s64, patch(v if which == 'rot' else x.detach(),
                                                                     v if which == 'scl' else y.detach()))))
                fd = (vals[0] - vals[1]) / (2 * h)
                self.assertLess(abs(fd - float(g[i])), 1e-5 * max(1.0, abs(fd)), (which, i, fd, float(g[i])))

    def test_identity_channel_gradients(self):
        """A new modifier at its identity values: zero gradient in the default branch mode; the
        opt-in one-sided mode matches float64 central differences (both neighbourhoods apply)."""
        from src.render.diffgeo.fp import Arith
        name = 'cf_J_MayuMid_s_L'          # an o_head bone without a card modifier

        def leaves():
            return {k: torch.tensor(v, dtype=torch.float64, requires_grad=True)
                    for k, v in (('scale', [1., 1., 1.]), ('position', [0., 0., 0.]), ('rotation', [0., 0., 0.]))}

        def patch(L):
            return {'bones': [{'name': name, 'length': 1.0, **L}]}
        L = leaves()
        f = self.functional(self.s.with_arith(Arith('f64')), patch(L))
        # the game takes the not-applied branch: the output does not depend on the leaves at all
        g = torch.autograd.grad(f, list(L.values()), allow_unused=True) if f.requires_grad else [None] * 3
        self.assertTrue(all(x is None or float(x.abs().max()) == 0.0 for x in g))
        s1 = self.s.with_arith(Arith('f64', identity_gradient='one_sided'))
        L = leaves()
        g = dict(zip(L, torch.autograd.grad(self.functional(s1, patch(L)), list(L.values()))))
        for k, h in (('scale', 1e-6), ('position', 1e-6), ('rotation', 1e-4)):
            for i in range(3):
                vals = []
                for sg in (1, -1):
                    P = {kk: vv.detach().clone() for kk, vv in L.items()}
                    P[k][i] += sg * h
                    with torch.no_grad():
                        vals.append(float(self.functional(s1, patch(P))))
                fd = (vals[0] - vals[1]) / (2 * h)
                self.assertLess(abs(fd - float(g[k][i])), 1e-5 * max(1.0, abs(fd)), (k, i, fd, float(g[k][i])))


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.render.diffgeo.chin_proxy import ChinProxy
        cls.s = session()
        cls.proxy = ChinProxy(cls.s, FIT_V2, WINDOWS)

    def test_anchored_baseline_equals_exact_and_gradient_finite(self):
        from src.render.diffgeo.chin_proxy import patch_from_q_torch
        from src.render.chin_fit import controls as C
        base = json.loads((FIT_V2 / 'baseline_score.json').read_text(encoding='utf-8'))
        rows = json.loads((FIT_V2 / 'encoding.json').read_text(encoding='utf-8'))['rows']
        q = torch.zeros(len(C.CONTROLS), dtype=torch.float64, requires_grad=True)
        total, rep, _ = self.proxy.loss(patch_from_q_torch(q, rows))
        self.assertAlmostEqual(float(total), base['cost'], places=12)
        for v in rep:
            self.assertAlmostEqual(rep[v]['rms_px'], base['views'][v]['rms_px'], places=9)
        (g,) = torch.autograd.grad(total, [q])
        self.assertTrue(bool(torch.isfinite(g).all()) and float(g.abs().max()) > 0)


if __name__ == '__main__':
    unittest.main()
