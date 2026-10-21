"""Checks for render_patched_capture.

CPU checks (always): plan binding/rejection, target validation, exact row
orientation and PNG eligibility. They use synthetic index rows only.

GPU checks (opt-in, HS2_REPLAY_TOOL + HS2_RENDERDOC_DIR set): the pre-existing
independent 32x32 analytic fixture capture (not HS2) is replayed through the
official driver: identity, an isolated diagnostic vertex patch, a mismatched
original-range rejection inside a live session, and fast/official byte
equivalence of the frame section and every read-back.
"""
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import render_patched_capture as rpc

F2P = Path(__file__).resolve().parents[2]
FIXTURE_RDC = F2P / 'outputs/offline_renderer_repair_20261012/renderdoc_analytic/capture_capture.rdc'


def h(b):
    return hashlib.sha256(b).hexdigest()


def synthetic_index(blob=b'abcdefgh'):
    return {'schema_version': 1, 'rdc_sha256': 'r' * 64, 'frame_section_sha256': 's' * 64, 'chunk_count': 2, 'chunks': [
        {'chunk_index': 0, 'name': 'ID3D11Device::CreateBuffer', 'resources': [{'name': 'pBuffer', 'id': 38}], 'buffers': [
            {'ordinal': 0, 'sd_buffer_index': 3, 'size_offset': 100, 'data_offset': 128, 'bytes': 0, 'sha256': h(b''), 'opaque': False},
            {'ordinal': 1, 'sd_buffer_index': 4, 'size_offset': 136, 'data_offset': 192, 'bytes': len(blob), 'sha256': h(blob), 'opaque': False}]},
        {'chunk_index': 1, 'name': 'ID3D11DeviceContext::Unmap', 'resources': [{'name': 'Context', 'id': 7}, {'name': 'pResource', 'id': 50}], 'buffers': [
            {'ordinal': 0, 'sd_buffer_index': 5, 'size_offset': 300, 'data_offset': 320, 'bytes': len(blob), 'sha256': h(blob), 'opaque': False}]}]}


def write_plan(tmp, patches, rdc_sha='r' * 64):
    data = tmp / 'cand.bin'
    data.write_bytes(b'XYZ')
    base = {'chunk_index': 0, 'chunk_name': 'ID3D11Device::CreateBuffer', 'buffer_ordinal': 1, 'expected_buffer_index': 4,
            'expected_blob_sha256': h(b'abcdefgh'), 'byte_offset': 2, 'expected_range_sha256': h(b'cde'),
            'replacement_file': 'cand.bin', 'replacement_sha256': h(b'XYZ'), 'source_witness': 'synthetic', 'semantic_role': 'test'}
    rows = [dict(base, **p) for p in patches]
    path = tmp / 'plan.json'
    path.write_text(json.dumps({'source_rdc_sha256': rdc_sha, 'patches': rows}))
    return path


class PlanBinding(unittest.TestCase):
    def test_valid_patch_resolves_to_exact_section_range(self):
        with tempfile.TemporaryDirectory() as d:
            resolved, plan = rpc.resolve_plan(write_plan(Path(d), [{'resource_id': 38}]), synthetic_index(), 'r' * 64)
            self.assertEqual(resolved[0]['section_range'], [194, 197])
            self.assertEqual(resolved[0]['data_offset'], 192)
            self.assertEqual(plan['sha256'], h((Path(d) / 'plan.json').read_bytes()))

    def test_rejections(self):
        cases = [({'chunk_name': 'other'}, 'r'), ({'buffer_ordinal': 2}, 'r'), ({'expected_buffer_index': 9}, 'r'),
                 ({'expected_blob_sha256': '0' * 64}, 'r'), ({'replacement_sha256': '0' * 64}, 'r'), ({'byte_offset': 6}, 'r'),
                 ({'source_witness': ''}, 'r'), ({'resource_id': 7}, 'r'), ({'chunk_index': 5}, 'r'),
                 ({'expected_range_sha256': 'not-a-digest'}, 'r'), ({}, 'x')]
        for change, rdc in cases:
            with self.subTest(change=change, rdc=rdc), tempfile.TemporaryDirectory() as d:
                plan = write_plan(Path(d), [change], rdc_sha=rdc * 64)
                with self.assertRaises(rpc.PipelineRejected):
                    rpc.resolve_plan(plan, synthetic_index(), 'r' * 64)

    def test_overlap_rejected_and_disjoint_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(rpc.PipelineRejected):
                rpc.resolve_plan(write_plan(Path(d), [{}, {'byte_offset': 3}]), synthetic_index(), 'r' * 64)
        with tempfile.TemporaryDirectory() as d:
            ok, _ = rpc.resolve_plan(write_plan(Path(d), [{}, {'chunk_index': 1, 'chunk_name': 'ID3D11DeviceContext::Unmap', 'buffer_ordinal': 0,
                                                                'expected_buffer_index': 5}]), synthetic_index(), 'r' * 64)
            self.assertEqual(len(ok), 2)

    def test_identity_plan_is_none(self):
        self.assertEqual(rpc.resolve_plan(None, synthetic_index(), 'r' * 64), ([], None))


class Targets(unittest.TestCase):
    def test_target_validation(self):
        good = {'name': 'cam', 'kind': 'texture', 'event_id': 3, 'resource_id': 4, 'orientation': 'unity_texture_rows_bottom_up'}
        self.assertEqual(rpc.validate_targets([good])[0]['mip'], 0)
        for bad in [dict(good, orientation=None), dict(good, event_id=0), dict(good, kind='mesh'), dict(good, name='a b')]:
            with self.subTest(bad=bad), self.assertRaises(rpc.PipelineRejected):
                rpc.validate_targets([bad])
        with self.assertRaises(rpc.PipelineRejected):
            rpc.validate_targets([good, good])
        with self.assertRaises(rpc.PipelineRejected):
            rpc.validate_targets([])

    def test_row_orientation_is_exact_permutation(self):
        info = {'dimension': 2, 'format_type': 0, 'width': 2, 'height': 3}
        raw = bytes(range(24))  # 3 rows of 8 bytes
        top, _ = rpc.orient_texture(raw, info, 'unity_texture_rows_bottom_up')
        self.assertEqual(top, raw[16:24] + raw[8:16] + raw[0:8])
        same, _ = rpc.orient_texture(raw, info, 'd3d_rows_top_down')
        self.assertEqual(same, raw)
        self.assertIsNone(rpc.orient_texture(raw, dict(info, format_type=3), 'unity_texture_rows_bottom_up')[0])
        with self.assertRaises(rpc.PipelineRejected):
            rpc.orient_texture(raw[:23], info, 'unity_texture_rows_bottom_up')

    def test_png_eligibility(self):
        rgba8 = {'format_type': 0, 'component_bytes': 1, 'component_count': 4, 'component_type': 0}
        self.assertTrue(rpc.png_eligible(rgba8))
        self.assertFalse(rpc.png_eligible(dict(rgba8, component_bytes=4, component_type=1)))
        self.assertFalse(rpc.png_eligible(dict(rgba8, component_type=4)))


GPU = os.environ.get('HS2_REPLAY_TOOL') and os.environ.get('HS2_RENDERDOC_DIR')


@unittest.skipUnless(GPU, 'set HS2_REPLAY_TOOL and HS2_RENDERDOC_DIR to run official GPU replay checks')
class FixtureReplay(unittest.TestCase):
    """Independent analytic 32x32 fixture: VB 38 (CreateBuffer chunk 5 InitialData), RT 47 drawn at event 20."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='rpc_fixture_', dir=os.environ.get('HS2_REPLAY_TEST_ROOT')))
        cls.tools = rpc.Tools(os.environ['HS2_REPLAY_TOOL'], os.environ['HS2_RENDERDOC_DIR'])
        rpc.build_index(cls.tools, FIXTURE_RDC, cls.tmp / 'index.json', cls.tmp / 'log_index')
        cls.tools.run(['convert', FIXTURE_RDC, 'rdc', cls.tmp / 'original.zip.xml', 'zip.xml'], cls.tmp / 'log_export', 120)
        cls.rdc_sha = rpc.sha256_file(FIXTURE_RDC)
        cls.index = json.loads((cls.tmp / 'index.json').read_text())
        chunk = cls.index['chunks'][5]
        assert chunk['name'] == 'ID3D11Device::CreateBuffer' and chunk['resources'][0]['id'] == 38
        buf = chunk['buffers'][1]
        cls.vb = buf
        cls.targets = [{'name': 'rt', 'kind': 'texture', 'event_id': 20, 'resource_id': 47, 'orientation': 'd3d_rows_top_down'},
                       {'name': 'vb', 'kind': 'buffer', 'event_id': 20, 'resource_id': 38},
                       {'name': 'copy', 'kind': 'texture', 'event_id': 21, 'resource_id': 54, 'orientation': 'd3d_rows_top_down'}]

    def plan(self, name, range_sha=None):
        d = self.tmp / ('plan_' + name)
        d.mkdir()
        old = self.original_vb[:4]
        new = struct.pack('<f', struct.unpack('<f', old)[0] + 0.125)
        (d / 'replacement.bin').write_bytes(new)
        plan = {'source_rdc_sha256': self.rdc_sha, 'source_xml_sha256': rpc.sha256_file(self.tmp / 'original.zip.xml'),
                'source_zip_sha256': rpc.sha256_file(self.tmp / 'original.zip'), 'patches': [{
                    'chunk_index': 5, 'chunk_name': 'ID3D11Device::CreateBuffer', 'buffer_ordinal': 1, 'resource_id': 38,
                    'expected_buffer_index': self.vb['sd_buffer_index'], 'expected_blob_sha256': self.vb['sha256'], 'byte_offset': 0,
                    'expected_range_sha256': range_sha or h(old), 'replacement_file': 'replacement.bin', 'replacement_sha256': h(new),
                    'source_witness': 'independent analytic fixture VB CreateBuffer InitialData (not HS2)',
                    'semantic_role': 'DIAGNOSTIC isolated first vertex float +0.125'}]}
        (d / 'plan.json').write_text(json.dumps(plan, indent=2))
        return d / 'plan.json', old, new

    def test_fixture(self):
        with rpc.ReplaySession(self.tools, FIXTURE_RDC, self.tmp / 'index.json', self.tmp / 'session') as s:
            ident = s.render(None, self.targets, self.tmp / 'identity')
            self.__class__.original_vb = Path(ident['targets'][1]['raw']['path']).read_bytes()
            self.assertEqual(h(self.original_vb[:self.vb['bytes']]), self.vb['sha256'])
            plan, old, new = self.plan('diag')
            diag = s.render(plan, self.targets, self.tmp / 'diag', keep_candidate=True)
            vb = Path(diag['targets'][1]['raw']['path']).read_bytes()
            self.assertEqual(vb, new + self.original_vb[4:])  # exactly the declared range changed
            self.assertNotEqual(diag['targets'][0]['raw']['sha256'], ident['targets'][0]['raw']['sha256'])
            bad, _, _ = self.plan('bad', range_sha='0' * 64)
            with self.assertRaises(rpc.PipelineRejected):
                s.render(bad, self.targets, self.tmp / 'bad_out')
            self.assertEqual(sorted(p.name for p in (self.tmp / 'bad_out').iterdir()), ['receipt.json'])
            again = s.render(None, self.targets, self.tmp / 'identity_again')
            self.assertEqual([t['raw']['sha256'] for t in again['targets']], [t['raw']['sha256'] for t in ident['targets']])
        job = {'schema': rpc.JOB_SCHEMA, 'capture_rdc': str(FIXTURE_RDC), 'capture_rdc_sha256': self.rdc_sha,
               'index': str(self.tmp / 'index.json'), 'plan': str(plan), 'official_zip_xml': str(self.tmp / 'original.zip.xml'),
               'targets': self.targets}
        both = rpc.run_job(self.tools, job, self.tmp / 'both', 'both')
        self.assertTrue(both['equivalence']['all_equal'], both['equivalence'])
        ident_job = dict(job, plan=None)
        both_identity = rpc.run_job(self.tools, ident_job, self.tmp / 'both_identity', 'both')
        self.assertTrue(both_identity['equivalence']['all_equal'], both_identity['equivalence'])


if __name__ == '__main__':
    unittest.main()
