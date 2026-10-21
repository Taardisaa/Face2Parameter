"""CPU tests of src/render/frame_program.py: builder, content-addressed blobs, validator rejections.

Run: python -m tools.offline_renderer.test_frame_program
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import frame_program as fp  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ProgramSchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'p'

    def tearDown(self):
        self.tmp.cleanup()

    def minimal(self):
        b = fp.ProgramBuilder(self.root, provenance={'generator': 'test'})
        rt = b.texture('tex2d', {'Width': 4, 'Height': 4, 'MipLevels': 1, 'ArraySize': 1, 'Format': 28,
                                 'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': 0, 'BindFlags': 40,
                                 'CPUAccessFlags': 0, 'MiscFlags': 0}, name='rt')
        rtv = b.view('rtv', rt, {'Format': 28, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}})
        b.clear_rtv(rtv, [0.0, 0.5, 1.0, 1.0])
        b.readback('rt', rt)
        return b, rt, rtv

    def test_roundtrip_and_blob_addressing(self):
        b, rt, rtv = self.minimal()
        h = b.blob(b'abc')
        self.assertEqual(h, sha(b'abc'))
        self.assertEqual((self.root / 'blobs' / h).read_bytes(), b'abc')
        b.write()
        program, blobs = fp.load(self.root)
        self.assertEqual(program['commands'][0]['op'], 'clear_rtv')
        (self.root / 'blobs' / h).write_bytes(b'abd')
        with self.assertRaises(fp.ProgramInvalid):
            blobs.read(h)

    def test_hard_linked_blob_reuse(self):
        b, rt, rtv = self.minimal()
        h = b.blob(b'shared-bytes')
        b.write()
        other = fp.ProgramBuilder(Path(self.tmp.name) / 'q', blob_link_sources=[self.root])
        self.assertEqual(other.blob(b'shared-bytes'), h)
        self.assertEqual(other.blobs.linked, 1)
        self.assertEqual((Path(self.tmp.name) / 'q' / 'blobs' / h).read_bytes(), b'shared-bytes')
        (self.root / 'blobs' / sha(b'x')).write_bytes(b'not x')   # corrupt source is never linked
        self.assertEqual(other.blob(b'x'), sha(b'x'))
        self.assertEqual(other.blobs.linked, 1)

    def test_rejections(self):
        b, rt, rtv = self.minimal()
        cases = []
        p = json.loads(json.dumps(b.program))
        p['commands'].append({'op': 'present'})
        cases.append(('unknown op', p))
        p = json.loads(json.dumps(b.program))
        p['readbacks'][0]['after_command'] = 5
        cases.append(('readback beyond commands', p))
        p = json.loads(json.dumps(b.program))
        p['resources'][0]['desc']['Usage'] = fp.USAGE_IMMUTABLE
        cases.append(('immutable without data', p))
        p = json.loads(json.dumps(b.program))
        p['views'][0]['desc'] = {'Format': 28, 'ViewDimension': 3, 'Texture2D': {'MipSlice': 0}}
        cases.append(('view union mismatch', p))
        p = json.loads(json.dumps(b.program))
        p['resources'][0]['initial'] = [{'subresource': 0, 'blob': b.blob(b'\0' * 63), 'row_pitch': 16, 'slice_pitch': 64}]
        cases.append(('short initial blob', p))
        p = json.loads(json.dumps(b.program))
        p['readbacks'].append(dict(p['readbacks'][0]))
        cases.append(('duplicate readback name', p))
        p = json.loads(json.dumps(b.program))
        p['commands'][0]['extra'] = 1
        cases.append(('unknown command field', p))
        p = json.loads(json.dumps(b.program))
        p['resources'][0]['desc']['Bogus'] = 0
        cases.append(('unknown desc field', p))
        for name, prog in cases:
            with self.subTest(name), self.assertRaises(fp.ProgramInvalid):
                fp.validate(prog, b.blobs)

    def test_draw_snapshot_must_be_complete(self):
        b, rt, rtv = self.minimal()
        s = fp.DrawState()
        with self.assertRaises(fp.ProgramInvalid):
            b.draw(s, VertexCount=3)
        b.draw(s, VertexCount=3, StartVertexLocation=0)
        with self.assertRaises(fp.ProgramInvalid):  # no vertex shader
            fp.validate(b.program, b.blobs)
        cmd = b.program['commands'][-1]
        del cmd['state']['rs']
        with self.assertRaises(fp.ProgramInvalid):
            fp.validate(b.program, b.blobs)

    def test_cb_blob_rules(self):
        b, rt, rtv = self.minimal()
        vs = b.shader('vs', b'DXBC' + b'\0' * 60)
        s = fp.DrawState()
        s.vs.shader = vs
        s.vs.constant_buffers = [{'slot': 0, 'blob': b.blob(b'\0' * 20)}]
        b.draw(s, VertexCount=3, StartVertexLocation=0)
        with self.assertRaises(fp.ProgramInvalid):
            fp.validate(b.program, b.blobs)
        b.program['commands'][-1]['state']['vs']['constant_buffers'] = [{'slot': 0, 'blob': b.blob(b'\0' * 32)}]
        fp.validate(b.program, b.blobs)

    def test_layouts(self):
        ls = fp.subresource_layouts('tex2d', {'Width': 9, 'Height': 5, 'MipLevels': 0, 'ArraySize': 2, 'Format': 71})
        self.assertEqual(len(ls), 8)
        self.assertEqual((ls[0]['rows'], ls[0]['row_bytes']), (2, 24))  # BC1: 3x2 blocks of 8 bytes
        self.assertEqual(ls[4]['subresource'], 4)
        with self.assertRaises(fp.ProgramInvalid):
            fp.block_layout(130)


if __name__ == '__main__':
    unittest.main()
