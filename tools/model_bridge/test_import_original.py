"""Transaction checks only: a failed/finished import must restore its character.

Synthetic transport and geometry helpers; these do not certify the real renderer,
source decoder, native card writer or photograph accuracy.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from .import_original import apply_and_save, same_source


class ImportTransactionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        self.path = self.out/'source.json'; self.path.write_text('fixture')
        self.before = {'active': False, 'reference_local_size': [1., 1., 1.], 'reference_local_center': [0., 0., 0.]}
        self.current = copy.deepcopy(self.before); self.calls = []
        self.artifact = {'source': {'fixture': True}, 'vertices': [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], 'triangles': [0, 1, 2]}
        self.fail_attachment = False; self.fail_backup = False
        self.snapshot = {'meta': {'fullname': 'synthetic original'}, 'face_shapes': {'fixture': True}}

    def transport(self, base, method, body=None, route='/maker/face/model'):
        self.calls.append((method, route))
        if route == '/health':
            return {'version': '0.31.10'}
        if route == '/maker/snapshot':
            return copy.deepcopy(self.snapshot)
        if route.startswith('/maker/render?'):
            path = Path(parse_qs(urlparse(route).query)['out'][0]); path.write_bytes(b'fixture PNG')
            return {'path': str(path)}
        if route == '/maker/card/save':
            if self.fail_backup and not self.current['active']:
                raise RuntimeError('native backup refused')
            Path(body['path']).write_bytes(b'fixture card')
            return {'source_head_embedded': self.current['active']}
        if route == '/maker/card/load':
            self.current = copy.deepcopy(self.before); return {}
        if route == '/maker/face/model/attachment':
            if self.fail_attachment:
                raise RuntimeError('body_changed')
            return copy.deepcopy(self.current)
        if method == 'POST':
            self.current = {'active': True}; return copy.deepcopy(self.current)
        return copy.deepcopy(self.current)

    def apply(self, keep=False):
        with patch('tools.model_bridge.import_original.verify', return_value={'fixture': True}), patch(
                'tools.model_bridge.import_original.check', return_value={'fixture': True}), patch(
                'tools.model_bridge.import_original.rebase', return_value={'fixture': True}):
            return apply_and_save('fixture', self.artifact, self.path,
                {'proposal': {}, 'native_state': {}, 'reference_state': {}}, {'reference_artifact': self.path},
                self.out, keep_in_game=keep, transport=self.transport)

    def test_success_saves_new_card_and_restores_ordinary_native_by_default(self):
        report = self.apply()
        self.assertTrue(report['passed'])
        self.assertTrue(report['restoration']['ordinary_native_character'])
        self.assertTrue((self.out/'source_character.png').is_file())
        self.assertEqual(self.current, self.before)

    def test_failure_after_import_restores_even_when_keep_requested(self):
        self.fail_attachment = True
        with self.assertRaisesRegex(RuntimeError, 'body_changed'):
            self.apply(keep=True)
        report = json.loads((self.out/'receipt.json').read_text())
        self.assertFalse(report['passed'])
        self.assertTrue(report['restoration']['native_parameter_snapshot_literal'])
        self.assertEqual(self.current, self.before)

    def test_failed_backup_never_imports(self):
        self.fail_backup = True
        with self.assertRaisesRegex(RuntimeError, 'native backup refused'):
            self.apply()
        self.assertNotIn(('POST', '/maker/face/model'), self.calls)
        self.assertEqual(self.current, self.before)

    def test_success_keeps_source_only_with_explicit_request(self):
        report = self.apply(keep=True)
        self.assertTrue(self.current['active'])
        self.assertNotIn('restoration', report)
        self.assertNotIn(('POST', '/maker/card/load'), self.calls)

    def test_restore_comparison_cannot_confuse_different_source_pose_or_native_state(self):
        a = {'active': True, 'artifact_sha256': 'a', 'source_pose_axis_angle': [0.], 'render_vertices': [[0., 0., 0.]],
             'render_triangles': [0], 'attachment': {'status': 'absent'}}
        for key, value in [('active', False), ('artifact_sha256', 'b'), ('source_pose_axis_angle', [1.]),
                           ('render_vertices', [[0., 0., 1.]]), ('attachment', {'status': 'active'})]:
            b = copy.deepcopy(a); b[key] = value
            self.assertFalse(same_source(a, b))


if __name__ == '__main__':
    unittest.main()
