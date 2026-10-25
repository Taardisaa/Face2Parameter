"""CPU tests of pair_skin_capture's bridge pre_cull pose loading (probe 6.1.0) and summary.

    python -m unittest tools.offline_renderer.test_pair_skin_capture
"""
from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.render import gpu_skin_inputs as g
from tools.offline_renderer import pair_skin_capture as psc

FRAMES = [997, 998, 999, 1000, 1001, 1002, 1003, 1004]


def window(tmp, mode='renderstyle', renders=16, count=4, transforms=True, skip=None, bad_count=None, dup=False, frame=1002):
    w = Path(tmp) / 'window'
    (w / 'frames').mkdir(parents=True)
    (w / 'capture_receipt.json').write_text(json.dumps({'mode': mode, 'probe': 'HS2_FramePairProbe 6.1.0',
                                                        'frame_counts_by_index': FRAMES}), encoding='utf-8')
    recs = [{'seq': 1, 'phase': 'pre_cull', 'frame_count': 1002, 'bridge_render': None, 'data': {}}]
    for o in range(renders):
        data = {}
        if transforms and o != skip:
            n = count - 1 if o == bad_count else count
            rows = np.arange(n * 33, dtype=np.float32) + o
            data['character_transforms'] = {'order': 'static.json transforms', 'count': n,
                                            'transforms_f32': base64.b64encode(rows.tobytes()).decode(),
                                            'active': base64.b64encode(bytes(n)).decode()}
        ordinal = 0 if (dup and o == 1) else o
        recs.append({'seq': 2 + o, 'phase': 'pre_cull', 'frame_count': frame, 'bridge_render': {'ordinal': ordinal}, 'data': data})
        recs.append({'seq': 50 + o, 'phase': 'pre_render', 'frame_count': frame, 'bridge_render': {'ordinal': ordinal}, 'data': {}})
    (w / 'frames' / 'renders_005.json').write_text(json.dumps({'frame_count': 1002, 'records': recs}), encoding='utf-8')
    return w


class BridgePoseTests(unittest.TestCase):
    def test_exported_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            poses, status = psc.bridge_pre_cull_poses(window(tmp), FRAMES, 1002, 4)
            self.assertEqual(status, 'exported')
            self.assertEqual(sorted(poses), list(range(16)))
            self.assertEqual(poses[3]['T'].shape, (4, 33))
            self.assertEqual(float(poses[3]['T'][0, 0]), 3.0)
            self.assertEqual(float(poses[3]['T'][1, 0]), 36.0)

    def test_older_probe_and_frame_mode_give_no_poses(self):
        with tempfile.TemporaryDirectory() as tmp:
            poses, status = psc.bridge_pre_cull_poses(window(tmp, transforms=False), FRAMES, 1002, 4)
            self.assertEqual(poses, {})
            self.assertIn('no character_transforms', status)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(psc.bridge_pre_cull_poses(window(tmp, mode='frame'), FRAMES, 1002, 4)[0], {})

    def test_partial_or_malformed_exports_fail_closed(self):
        for kw in ({'skip': 5}, {'bad_count': 2}, {'dup': True}, {'frame': 1001}):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(g.SkinPathError, msg=str(kw)):
                    psc.bridge_pre_cull_poses(window(tmp, **kw), FRAMES, 1002, 4)
        with tempfile.TemporaryDirectory() as tmp:
            w = window(tmp)
            with self.assertRaises(g.SkinPathError):
                psc.bridge_pre_cull_poses(w, FRAMES, 1002, 5)
            with self.assertRaises(g.SkinPathError):
                psc.bridge_pre_cull_poses(w, FRAMES, 990, 4)

    def test_summary_counts(self):
        rows = [{'o2w_bit_exact_frames': [1], 'w2o_bit_exact_frames': [1], 'o2w_bit_exact_bridge_renders': [], 'w2o_bit_exact_bridge_renders': []},
                {'o2w_bit_exact_frames': [], 'w2o_bit_exact_frames': [], 'o2w_bit_exact_bridge_renders': [0], 'w2o_bit_exact_bridge_renders': [0]},
                {'o2w_bit_exact_frames': [], 'w2o_bit_exact_frames': [], 'o2w_bit_exact_bridge_renders': [], 'w2o_bit_exact_bridge_renders': []},
                {'role': '_PreviousM', 'o2w_bit_exact_frames': [2], 'o2w_bit_exact_bridge_renders': []}]
        s = psc.object_matrix_summary(rows)
        self.assertEqual(s, {'rows': 4, 'o2w_rows_matching_frames': 2, 'o2w_rows_matching_bridge_renders': 1,
                             'w2o_rows_matching_frames': 1, 'w2o_rows_matching_bridge_renders': 1,
                             'o2w_rows_matching_neither': 1})


if __name__ == '__main__':
    unittest.main()
