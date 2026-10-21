"""Acceptance tests of the hosted per-frame drivers (installed game required).

Each tool runs in its own process (Mono initialises once per process).
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAME = Path('E:/HoneySelect2_ArcticFox')
SESSION = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_reference_fit_20261012/paired_capture_session_20261020')
SCRATCH = Path('E:/HS2_offline_renderer_data/skinning_20261020/drivers')


@unittest.skipUnless(GAME.exists() and SESSION.exists(), 'installed game and paired session required')
class HostedDriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=SCRATCH)
        out = Path(cls.tmp.name)
        for tool, name in (('eye_look_acceptance.py', 'eye.json'), ('fbs_composition_check.py', 'fbs.json')):
            subprocess.run([sys.executable, str(HERE / tool), '--out', str(out / name)], check=True,
                           capture_output=True, timeout=1800)
        cls.eye = json.loads((out / 'eye.json').read_text(encoding='utf-8'))
        cls.fbs = json.loads((out / 'fbs.json').read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def rows(self, window):
        return {r['frame']: r for r in self.eye['windows'][window] if r.get('role') != 'state_seed'}

    def test_scene_model_matches_exported_world(self):
        for rows in self.eye['windows'].values():
            for r in rows:
                self.assertEqual(r['world_mismatch'], [], r['frame'])

    def test_init_reference_state_from_prefab(self):
        for s in self.eye['init_state']:
            self.assertEqual(s['referenceLookDir'], [0.0, 0.0, 1.0])
            self.assertEqual(s['referenceUpDir'], [0.0, 1.0, 0.0])

    def test_eye_look_seed_invariant_frames_bit_exact(self):
        expect = {'04d1e76f': range(26057, 26063), '625aa0f2': range(12741, 12751)}
        for w, frames in expect.items():
            rows = self.rows(w)
            for f in frames:
                for bone in ('cf_J_look_L', 'cf_J_look_R'):
                    self.assertTrue(rows[f][bone]['seed_invariant'], (w, f, bone))
                    self.assertTrue(rows[f][bone]['bit_equal'], (w, f, bone))

    def test_no_seed_invariant_mismatch(self):
        for rows in self.eye['windows'].values():
            for r in rows:
                for bone in ('cf_J_look_L', 'cf_J_look_R'):
                    if bone in r and r[bone]['seed_invariant']:
                        self.assertTrue(r[bone]['bit_equal'], (r['frame'], bone))

    def test_fbs_composition(self):
        for rows in self.fbs['windows'].values():
            for r in rows:
                self.assertEqual(len(r['matches']), 1, r)
        blink = [r['matches'][0] for r in self.fbs['windows']['04d1e76f']]
        self.assertEqual(blink, ['open', 'open', 'closed', 'closed', 'closed', 'open', 'open', 'open'])
        self.assertEqual(self.fbs['mouth_adjust_width_scale'], {'open': 1.0, 'closed': 1.0})


if __name__ == '__main__':
    unittest.main()
