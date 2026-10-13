"""CPU regression for frame-specific byte inputs and undefined discarded ranges."""
import unittest
from src.render.captured_pipeline import CapturedPipeline,UnresolvedPipeline

class BufferLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.p=CapturedPipeline.__new__(CapturedPipeline);self.p.resources={}
        self.p._resource({'pBuffer':1},'pBuffer','CreateBuffer',{'ByteWidth':16},bytes(range(16)))

    def test_partial_write_keeps_prior_draw_and_surrounding_bytes(self):
        previous=list(self.p.resources[1]['buffer_segments'])
        self.p._buffer_update(1,4,8,b'abcd')
        now=self.p.resources[1]['buffer_segments']
        self.assertEqual(self.p.buffer_range(previous,0,16),bytes(range(16)))
        self.assertEqual(self.p.buffer_range(now,0,16),bytes(range(4))+b'abcd'+bytes(range(8,16)))

    def test_discarded_unknown_bytes_cannot_be_shaded(self):
        self.p._buffer_update(1,4,8,b'abcd',discard=True)
        segments=self.p.resources[1]['buffer_segments']
        self.assertEqual(self.p.buffer_range(segments,4,8),b'abcd')
        with self.assertRaises(UnresolvedPipeline):self.p.buffer_range(segments,0,16)

    def test_overlapping_updates_preserve_latest_byte_values(self):
        self.p._buffer_update(1,2,10,b'abcdefgh')
        self.p._buffer_update(1,6,12,b'UVWXYZ')
        self.assertEqual(self.p.buffer_range(self.p.resources[1]['buffer_segments'],0,16),b'\x00\x01abcdUVWXYZ\x0c\x0d\x0e\x0f')
if __name__=='__main__':unittest.main()
