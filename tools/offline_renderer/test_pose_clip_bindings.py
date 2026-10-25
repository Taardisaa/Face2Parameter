"""Tests of pose_clip_bindings (CPU; the integration case reads the installed Maker pose bundle).

    python -m unittest tools.offline_renderer.test_pose_clip_bindings
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

from tools.offline_renderer import pose_clip_bindings as pcb

BUNDLE = Path('E:/HoneySelect2_ArcticFox/abdata/custom/00/anim_f_00.unity3d')
# Session v6.1 baseline window (probe 6.1.0, 907 transforms) and the animator state its drivers
# export recorded (p_cf_anim, edit_F, Idle_00, fullPathHash -32934989).
V61 = Path('E:/HS2_offline_renderer_data/paired_session_v6_1_20261025/window_766d81849ec147a881b4214169a54185')


class PathTests(unittest.TestCase):
    def test_relative_paths_crc32(self):
        tf = [{'i': 0, 'name': 'chaF', 'parent': -1}, {'i': 1, 'name': 'p_cf_anim', 'parent': 0},
              {'i': 2, 'name': 'cf_J_Root', 'parent': 1}, {'i': 3, 'name': 'cf_J_Hips', 'parent': 2}]
        p = pcb.relative_paths(tf, 1)
        self.assertEqual(p[zlib.crc32(b'')], (1, ''))
        self.assertEqual(p[zlib.crc32(b'cf_J_Root/cf_J_Hips')], (3, 'cf_J_Root/cf_J_Hips'))
        self.assertNotIn(zlib.crc32(b'chaF'), {k for k, v in p.items() if v[0] == 0})

    def test_missing_animation_block_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / 'static.json').write_text(json.dumps({'transforms': []}), encoding='utf-8')
            (w / 'scene.json').write_text(json.dumps({'animation_error': 'x'}), encoding='utf-8')
            with self.assertRaises(pcb.BindingError):
                pcb.analyse(w, w / 'scene.json')


@unittest.skipUnless(BUNDLE.is_file() and (V61 / 'static.json').is_file(), 'installed bundle / v6.1 window not present')
class InstalledBundleTests(unittest.TestCase):
    def window(self, tmp, clips=('Idle_00',), full=-32934989, main_data='edit_F'):
        w = Path(tmp)
        shutil.copy(V61 / 'static.json', w / 'static.json')
        st = json.loads((w / 'static.json').read_text(encoding='utf-8'))
        root = [t for t in st['transforms'] if t['name'] == 'p_cf_anim'][0]
        scene = {'animation': {'animators': [{'animator': {'ref_name': 'p_cf_anim'}, 'transform': {'ref_id': root['id']},
                                              'controller': {'ref_name': 'edit_F'},
                                              'layer0_current': {'full_path_hash': full},
                                              'layer0_current_clips': [{'clip': {'ref_name': c}, 'weight': 1} for c in clips]}],
                               'maker_pose_source': {'pose_no': 1, 'entry': {'MainAB': 'custom/00/anim_f_00.unity3d', 'MainData': main_data,
                                                                             'Clip': 'Idle_00', 'bundle_file': str(BUNDLE),
                                                                             'bundle_file_exists': True}}}}
        (w / 'scene.json').write_text(json.dumps(scene), encoding='utf-8')
        return w

    def test_idle_00_bindings(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = self.window(tmp)
            res = pcb.analyse(w, w / 'scene.json')
            a = res['animators'][0]
            self.assertEqual(a['state']['name'], 'Idle_00')
            self.assertEqual(a['state']['clips'], ['Idle_00'])
            cur = [c for c in a['clips'] if c['in_current_state']][0]
            self.assertEqual(cur['bindings'], 1063)
            self.assertEqual(len(cur['bound']), 1042)
            self.assertEqual(len({r['path_crc32'] for r in cur['unbound'] if r['type_id'] == 4}), 7)
            self.assertEqual(len(a['transforms_written_by_current_state']), 345)   # typeID 4 only; the 7 typeID-95 rows bind the Animator root

    def test_state_and_clip_mismatches_fail_closed(self):
        for kw in ({'clips': ('adv_Idle_05_00',)}, {'full': 12345}, {'main_data': 'edit_M'}):
            with tempfile.TemporaryDirectory() as tmp:
                w = self.window(tmp, **kw)
                with self.assertRaises(pcb.BindingError, msg=str(kw)):
                    pcb.analyse(w, w / 'scene.json')


if __name__ == '__main__':
    unittest.main()
