"""CPU tests of the chin-fit objective, controls, guard math and conversion helpers.

    .venv/Scripts/python.exe -m unittest tools.chin_fit.test_chin_fit -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.render.chin_fit import controls as ctl  # noqa: E402
from src.render.chin_fit import objective as obj  # noqa: E402
from src.render.chin_fit.protection import quat_matrix, stream_positions  # noqa: E402
from tools.chin_fit import convert_window as cw  # noqa: E402

BASE_ROWS = [{'name': n, 'scale': [1.0, 1.0, 1.0], 'rotation': [0.0, 0.0, 0.0], 'length': 1.0,
              'position': [0.0, 0.001, -0.002]} for n in ctl.NAMES]


class FakeDetector:
    """Returns fixed landmarks (render px) per view; raises for views listed in ``fail``."""

    def __init__(self, pts, fail=()):
        self.pts, self.fail = pts, set(fail)

    def detect(self, rgb, rect, view='?'):
        if view in self.fail or tuple(rect) == ('fail',):
            raise obj.PredictionUnavailable(view, 0)
        return self.pts[view]


def fake_reference(view, n=478):
    rng = np.random.default_rng(len(view))
    L = np.c_[rng.uniform(100, 900, (n, 2)), np.zeros(n)]
    ids = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
    return obj.ReferenceView(view, 'art.png', '0' * 64, ids, L[ids, :2].copy(), [33, 133, 263, 362, 1, 2, 10],
                             L, 250.0, None)


class TestSimilarity(unittest.TestCase):
    def test_recovers_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 1000, (12, 2))
        th, s, t = np.radians(7.0), 1.13, np.array([31.0, -12.5])
        A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        dst = src @ A.T + t
        A2, t2 = obj.fit_similarity(src, dst, np.ones(len(src)))
        np.testing.assert_allclose(A2, A, atol=1e-9)
        np.testing.assert_allclose(t2, t, atol=1e-7)

    def test_no_reflection(self):
        src = np.array([[0, 0], [1, 0], [0, 1.0]])
        dst = src * np.array([-1, 1.0])
        A, _ = obj.fit_similarity(src, dst, np.ones(3))
        self.assertGreater(np.linalg.det(A), 0)

    def test_anchor_weights_equal_groups(self):
        w = obj.anchor_weights([33, 133, 263, 362, 1, 2, 10])
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertAlmostEqual(w[:4].sum(), 1 / 3)
        self.assertAlmostEqual(w[4:6].sum(), 1 / 3)
        self.assertAlmostEqual(w[6], 1 / 3)
        with self.assertRaises(ValueError):
            obj.anchor_weights([33, 152])


class TestObjective(unittest.TestCase):
    def setUp(self):
        self.ref = fake_reference('front')
        th, s, t = np.radians(-3.0), 0.9, np.array([40.0, 25.0])
        A = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        # render points such that A @ render + t = art
        Ai = np.linalg.inv(A)
        self.render = np.c_[(self.ref.art_landmarks[:, :2] - t) @ Ai.T, np.zeros(478)]
        self.det = FakeDetector({'front': self.render})
        anchors = {i: self.render[i, :2].tolist() for i in self.ref.anchor_ids}
        self.dom = obj.ChinObjective.build_domain('front', anchors, self.ref, [0, 0, 1086, 1448], 'x')
        self.A = A

    def objective(self, det):
        return obj.ChinObjective({'front': self.dom}, {'front': self.ref}, det)

    def test_frame_recovered_from_mesh_anchors(self):
        np.testing.assert_allclose(self.dom.A, self.A, atol=1e-9)
        self.assertLess(max(r for _, r in self.dom.baseline_anchor_residual_px), 1e-6)

    def test_baseline_maps_exactly(self):
        o = self.objective(self.det)
        r, rep = o.score({'front': np.zeros((4, 4, 3), np.uint8)})
        self.assertEqual(len(r), o.n_residuals())
        self.assertLess(np.abs(r).max(), 1e-9)

    def test_chain_move_is_measured_in_eye_units(self):
        moved = self.render.copy()
        moved[152, 1] += 10.0          # chin tip 10 render px down
        r, rep = self.objective(FakeDetector({'front': moved})).score({'front': None})
        art_shift = 10.0 * np.sqrt(abs(np.linalg.det(self.dom.A)))
        self.assertAlmostEqual(rep['front']['max_px'], art_shift, places=6)
        n = len(self.ref.chain_ids)
        self.assertAlmostEqual(float(np.linalg.norm(r)), art_shift / self.ref.eye_unit_px / np.sqrt(n), places=9)

    def test_missing_binding_rejected(self):
        with self.assertRaises(ValueError):
            obj.ChinObjective.build_domain('front', {33: [0, 0]}, self.ref, None, 'x')

    def test_domain_json_round_trip(self):
        d2 = obj.ViewDomain.from_json(json.loads(json.dumps(self.dom.to_json())))
        np.testing.assert_array_equal(d2.A, self.dom.A)
        self.assertEqual(d2.rect, self.dom.rect)
        self.assertEqual(d2.measure, 'landmarks')

    def test_prediction_failure_propagates(self):
        with self.assertRaises(obj.PredictionUnavailable):
            self.objective(FakeDetector({}, fail={'front'})).score({'front': None})

    def test_select_rect_full_frame_first(self):
        rect, tried = obj.select_rect(FakeDetector({'v': self.render}), np.zeros((1448, 1086, 3), np.uint8), 'v')
        self.assertEqual(rect, [0, 0, 1086, 1448])
        self.assertEqual(len(tried), 1)

    def test_select_rect_largest_accepted(self):
        class D:
            def detect(self, rgb, rect, view):
                x0, y0, x1, y1 = rect
                if (x1 - x0) * (y1 - y0) > 1000 * 1448 or x0 > 0:
                    raise obj.PredictionUnavailable(view, 0)
                return None
        rect, tried = obj.select_rect(D(), np.zeros((1448, 1086, 3), np.uint8), 'v')
        self.assertEqual(rect, [0, 0, 1000, 1448])


def disc_image(cx, cy, r, w=400, h=400, side=1):
    """Gray gradient background with a skin-coloured disc (synthetic profile)."""
    yy, xx = np.mgrid[0:h, 0:w]
    im = np.zeros((h, w, 3), float)
    im[...] = (130 + 0.05 * yy)[..., None]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    im[inside] = (225, 185, 165)
    return im.astype(np.uint8)


class TestSilhouette(unittest.TestCase):
    def test_contour_and_offsets(self):
        a = disc_image(200, 200, 100)
        b = disc_image(206, 200, 100)       # moved 6 px towards the face side (+x)
        ca = obj.silhouette_contours(a, 1)
        cb = obj.silhouette_contours(b, 1)
        c = max(ca, key=len)
        ang = np.radians(np.arange(-40, 41, 10))
        st = np.c_[200 + 100 * np.cos(ang), 200 + 100 * np.sin(ang)]
        nv = np.c_[np.cos(ang), np.sin(ang)]
        off0 = obj.profile_offsets(ca, st, nv)
        off1 = obj.profile_offsets(cb, st, nv)
        self.assertLess(np.abs(off0).max(), 1.0)                       # edge within a pixel of r
        np.testing.assert_allclose(off1 - off0, 6 * np.cos(ang), atol=0.6)
        self.assertGreater(len(c), 100)

    def test_no_crossing_is_nan(self):
        st = np.array([[10.0, 10.0]])
        nv = np.array([[1.0, 0.0]])
        self.assertTrue(np.isnan(obj.profile_offsets([np.array([[300.0, 0], [300.0, 5]])], st, nv, reach=50)[0]))


class TestControls(unittest.TestCase):
    def test_count_and_ranges(self):
        self.assertEqual(len(ctl.CONTROLS), 36)
        pair = [c for c in ctl.CONTROLS if len(c['names']) == 2 and c['channel'] == 'position']
        self.assertTrue(all(c['range'] == 0.012 for c in pair))

    def test_zero_is_base(self):
        p = ctl.patch_from_q(np.zeros(36), BASE_ROWS)
        self.assertEqual(p['bones'], BASE_ROWS)

    def test_mirroring_and_length(self):
        q = np.zeros(36)
        i = next(k for k, c in enumerate(ctl.CONTROLS) if len(c['names']) == 2 and c['channel'] == 'position'
                 and c['axis'] == 0)
        j = next(k for k, c in enumerate(ctl.CONTROLS) if len(c['names']) == 2 and c['channel'] == 'rotation'
                 and c['axis'] == 1)
        s = next(k for k, c in enumerate(ctl.CONTROLS) if c['names'] == ['cf_J_ChinTip_s'] and c['channel'] == 'scale'
                 and c['axis'] == 2)
        q[i], q[j], q[s] = 0.5, -1.0, 1.0
        b = {r['name']: r for r in ctl.patch_from_q(q, BASE_ROWS)['bones']}
        self.assertAlmostEqual(b['cf_J_CheekLow_L']['position'][0], 0.006)
        self.assertAlmostEqual(b['cf_J_CheekLow_R']['position'][0], -0.006)
        self.assertAlmostEqual(b['cf_J_CheekLow_L']['rotation'][1], -3.0)
        self.assertAlmostEqual(b['cf_J_CheekLow_R']['rotation'][1], 3.0)
        self.assertAlmostEqual(b['cf_J_ChinTip_s']['scale'][2], 1.1)
        self.assertTrue(all(r['length'] == 1.0 for r in b.values()))

    def test_rejects_bad_vectors(self):
        with self.assertRaises(ValueError):
            ctl.patch_from_q(np.zeros(35), BASE_ROWS)
        with self.assertRaises(ValueError):
            ctl.patch_from_q(np.r_[np.zeros(35), np.nan], BASE_ROWS)

    def test_length_one_required(self):
        rows = [dict(r) for r in BASE_ROWS]
        rows[0] = dict(rows[0], length=1.01)
        with self.assertRaises(RuntimeError):
            ctl.patch_from_q(np.zeros(36), rows)


class TestGuardMath(unittest.TestCase):
    def test_quat_matrix(self):
        q = np.array([0.1, -0.2, 0.3, 0.9])
        q /= np.linalg.norm(q)
        R = quat_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0)
        # 90 deg about y maps +z to +x (Unity, left-handed)
        s = np.sqrt(0.5)
        np.testing.assert_allclose(quat_matrix([0, s, 0, s]) @ [0, 0, 1], [1, 0, 0], atol=1e-12)

    def test_stream_positions(self):
        a = np.arange(30, dtype=np.float32).reshape(3, 10)
        np.testing.assert_array_equal(stream_positions(a.tobytes()), a[:, :3])


class TestConversionHelpers(unittest.TestCase):
    def test_bridge_target(self):
        rt = [[100 * i, 'write', 'render_target'] for i in range(16)] + [[2000, 'read', 'copy_source']]
        fg = {'5': {'kind': 'CreateTexture2D', 'descriptor': {'Width': 1086, 'Height': 1448}, 'timeline': rt},
              '6': {'kind': 'CreateTexture2D', 'descriptor': {'Width': 1086, 'Height': 1448}, 'timeline': rt[:3]}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'resources.json'
            p.write_text(json.dumps(fg))
            self.assertEqual(cw.bridge_target(p), 5)
            fg['7'] = fg['5']
            p.write_text(json.dumps(fg))
            with self.assertRaises(RuntimeError):
                cw.bridge_target(p)

    def test_present_event(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'stdout.bin'
            p.write_text('\n'.join(json.dumps(x) for x in [
                {'event': 'opened'}, {'event': 'action', 'event_id': 10, 'name': 'Draw()'},
                {'event': 'action', 'event_id': 42, 'name': 'Present(ResourceId::7)'}]))
            self.assertEqual(cw.present_event(p), 42)


@unittest.skipUnless((obj.ALIGN / 'front_alignment_receipt.json').exists(), 'HS2Mod reference artifacts absent')
class TestReferences(unittest.TestCase):
    def test_reference_chains_are_art_detections(self):
        for v in obj.VIEWS:
            ref = obj.ReferenceView.load(v, verify_images=False)
            self.assertEqual(len(ref.chain_ids), 21 if v == 'front' else 11)
            np.testing.assert_array_equal(ref.art_landmarks[ref.chain_ids, :2], ref.chain_xy)
            obj.anchor_weights(ref.anchor_ids)       # every step-4 anchor belongs to a group

    def test_mesh_bindings_cover_anchor_ids(self):
        for v in obj.VIEWS:
            ref = obj.ReferenceView.load(v, verify_images=False)
            ids = [b['id'] for b in obj.load_bindings(v)]
            self.assertEqual(ids, list(ref.anchor_ids))

    @unittest.skipUnless(Path('C:/Users/13666/Workspace/Face2Parameter/outputs/chenger_stylized_arcface_20261011/'
                              'generated_side_references_v1').exists(), 'art profiles absent')
    def test_profile_stations_on_art(self):
        for v, n in (('right', 52), ('left', 41)):
            ref = obj.ReferenceView.load(v, verify_images=False)
            st, nv = obj.profile_stations(obj._load_rgb(ref.art_path), ref.art_landmarks, obj.FACE_SIDE[v])
            self.assertEqual(len(st), n)
            # first station at the lower lip, normals point into the background on the face side
            self.assertLess(np.linalg.norm(st[0] - ref.art_landmarks[17, :2]), 15)
            self.assertGreater(nv[0, 0] * obj.FACE_SIDE[v], 0)
            # the art silhouette itself sits on its own stations
            off = obj.profile_offsets(obj.silhouette_contours(obj._load_rgb(ref.art_path), obj.FACE_SIDE[v]), st, nv)
            self.assertLess(np.abs(off).max(), 0.5)


if __name__ == '__main__':
    unittest.main()
