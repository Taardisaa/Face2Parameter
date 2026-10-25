"""CPU tests of FramePairProbe v5 tooling (no game, no GPU).

* the probe's Unity-free core (ProbeCore.cs) through its C# test executable;
* the shader-library manifest builder;
* the session verifier on synthetic captures/windows.

    python -m unittest tools.offline_renderer.test_frame_pair_probe_v5 -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

from tools.offline_renderer.frame_pair_probe import build_probe_manifest as bpm
from tools.offline_renderer.frame_pair_probe import verify_probe_v5_session as vps

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / 'tools' / 'offline_renderer' / 'frame_pair_probe'
LIBRARY = ROOT / 'outputs' / 'offline_renderer_shader_library_20261020' / 'library_v1'
NONCE = '0123456789abcdef0123456789abcdef'


class ProbeCoreTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('dotnet'), 'dotnet SDK required')
    def test_core_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = subprocess.run(['dotnet', 'build', str(PROBE / 'tests' / 'ProbeCoreTests' / 'ProbeCoreTests.csproj'),
                                    '-c', 'Release', '-o', tmp], capture_output=True, text=True, encoding='utf-8', errors='replace')
            self.assertEqual(build.returncode, 0, build.stdout[-3000:])
            run = subprocess.run([str(Path(tmp) / 'ProbeCoreTests.exe')], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout)
            self.assertIn('0 failed', run.stdout)


class ManifestTests(unittest.TestCase):
    def test_conflicting_types_raise(self):
        t = {}
        bpm._put(t, 's', '_A', 'float', 0)
        bpm._put(t, 's', '_A', 'float', 0)
        with self.assertRaises(bpm.ManifestError):
            bpm._put(t, 's', '_A', 'vector', 0)
        bpm._put(t, 's', '_C', 'vector', 0)
        bpm._put(t, 's', '_C', 'color', 0)
        self.assertEqual(t['_C'], ('color', 0))

    @unittest.skipUnless((LIBRARY / 'main' / 'library.json').is_file(), 'installed shader library extraction required')
    def test_library_manifest(self):
        lib = bpm.sl.ShaderLibrary([LIBRARY / r for r in bpm.LIB_ROOTS])
        shaders, keywords, globals_, stats = bpm.build(lib)
        self.assertEqual(len(shaders), len(lib.shader_names()))
        liquid = shaders['AIT/Liquid']
        for name in ('_Cutoff', '_ChromaticAberration', '_WeatheringAll'):
            self.assertEqual(liquid[name], ('float', 0))
        for name in ('_UVScalePattern', '_WeatheringUV'):
            self.assertEqual(liquid[name][0], 'vector')
        self.assertEqual(liquid['_WeatheringMap'], ('texture', 0))
        self.assertIn('AIT/main namida', shaders)
        self.assertEqual(globals_['_Time'], ('vector', 0))
        self.assertEqual(globals_['unity_ObjectToWorld'][0], 'matrix')
        self.assertIn('DIRECTIONAL', keywords)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'm.tsv'
            bpm.write(out, [LIBRARY / r for r in bpm.LIB_ROOTS], shaders, keywords, globals_, stats)
            lines = [l for l in out.read_text(encoding='utf-8').splitlines() if not l.startswith('#')]
            self.assertTrue(all(len(l.split('\t')) in (2, 4, 5) for l in lines))


@dataclass
class Ref:
    length: int
    data: bytes


@dataclass
class Chunk:
    index: int
    name: str
    buffers: list = field(default_factory=list)


class FakeIndex:
    def __init__(self, chunks):
        self.begin = Chunk(0, 'Internal::Beginning of Capture')
        self.chunks = [self.begin] + chunks

    @staticmethod
    def blob(ref):
        return ref.data


def record(magic, frame, w36, w40, nonce=NONCE):
    b = bytearray(64)
    b[0:16] = magic
    b[16:32] = bytes.fromhex(nonce)
    b[32:36] = frame.to_bytes(4, 'little', signed=True)
    b[36:40] = w36.to_bytes(4, 'little', signed=True)
    b[40:44] = w40.to_bytes(4, 'little', signed=True)
    b[44:48] = (~frame).to_bytes(4, 'little', signed=True)
    b[48:64] = vps.TAIL
    return bytes(b)


def chunk(i, payload):
    return Chunk(i, 'ID3D11DeviceContext::Unmap', [Ref(len(payload), payload)])


class VerifierTests(unittest.TestCase):
    def window(self, tmp, mode='renderstyle', captured=1002, bridge_frames=None, raw_status='complete'):
        w = Path(tmp) / 'window'
        (w / 'frames').mkdir(parents=True)
        frames = [997, 998, 999, 1000, 1001, 1002, 1003, 1004]
        k = frames.index(captured)
        for name in ('frame', 'scene'):
            (w / 'frames' / ('%s_%03d.json' % (name, k))).write_text('{}', encoding='utf-8')
        (w / 'frames' / ('renders_%03d.json' % k)).write_text(json.dumps(
            {'frame_count': captured, 'records': [{'seq': 1, 'frame_count': captured}]}), encoding='utf-8')
        rawdir = Path(tmp) / 'raw' / 'texture_5'
        rawdir.mkdir(parents=True)
        (rawdir / 'descriptor.json').write_text('{}', encoding='utf-8')
        (w / 'raw_textures.json').write_text(json.dumps({'queued_frame': 1001, 'counts': {raw_status: 1},
            'jobs': [{'status': raw_status, 'path': str(rawdir), 'error': None}]}), encoding='utf-8')
        png = Path(tmp) / 'bridge.png'
        png.write_bytes(b'png')
        bridge_frames = bridge_frames or [1002] * 16
        receipt = {'nonce': NONCE, 'mode': mode, 'aborted_before_trigger': False, 'triggered_frame': 1000,
                   'expected_capture_frame': 1002, 'frame_counts_by_index': frames, 'captures_before': 1, 'captures_after': 2,
                   'captures': [{'index': 1, 'path': 'x.rdc'}], 'bridge_expected_renders': 16,
                   'bridge_renders': [{'frame_count': f, 'ordinal': i} for i, f in enumerate(bridge_frames)],
                   'bridge': {'initial_http_status': '504', 'final_http_status': '200', 'png_path': str(png),
                              'final_http_body': json.dumps({'operation_id': 'x', 'status': 'succeeded',
                                                             'result': {'capture': {'temporal_render_count': 16}}})}}
        (w / 'capture_receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
        return w

    def index(self, frame=1002, bridge=True, other_nonce_extra=True):
        chunks = []
        if bridge:
            chunks.append(chunk(10, b'xx' + record(vps.MAGIC_BRIDGE, frame, 1, 0) + b'yy'))
            chunks.append(chunk(20, record(vps.MAGIC_BRIDGE, frame, 2, 15)))
        chunks.append(chunk(30, record(vps.MAGIC_FRAME, frame, 5, frame)))
        if other_nonce_extra:
            chunks.append(chunk(31, record(vps.MAGIC_FRAME, frame, 5, frame, nonce='ff' * 16)))
        return FakeIndex(chunks)

    def test_accepts_paired_bridge_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(), self.window(tmp), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])

    def test_rejects_capture_of_wrong_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(frame=1001), self.window(tmp, captured=1001), NONCE)
            self.assertFalse(r['accepted'])
            self.assertIn('marker_frame_equals_expected', [c['check'] for c in r['checks'] if not c['ok']])

    def test_rejects_bridge_renders_outside_captured_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(), self.window(tmp, bridge_frames=[1002] * 15 + [1003]), NONCE)
            self.assertIn('bridge_renders_in_captured_frame', [c['check'] for c in r['checks'] if not c['ok']])

    def test_rejects_missing_bridge_markers_and_failed_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.window(tmp, raw_status='failed'), NONCE)
            failed = {c['check'] for c in r['checks'] if not c['ok']}
            self.assertTrue({'bridge_markers', 'raw_jobs_resolved'} <= failed, failed)

    def v6_window(self, tmp, hooks='1', drivers=True, errors=0, missing=None):
        w = self.window(tmp, mode='frame')
        r = json.loads((w / 'capture_receipt.json').read_text(encoding='utf-8'))
        r['probe'] = 'HS2_FramePairProbe 6.0.0'
        r['driver_hooks'] = None if hooks == '0' else {'installed': list(vps.HOOKS), 'errors': errors}
        (w / 'capture_receipt.json').write_text(json.dumps(r), encoding='utf-8')
        (w / 'request.ini').write_text('Mode=frame\nDriverHooks=%s\n' % hooks, encoding='utf-8')
        if drivers:
            recs, seq = [], 0
            for h in ('chara_update_force', 'fbs', 'chara_late_update_force', 'abmx', 'neck', 'eye'):
                for p in ('prefix', 'postfix'):
                    if missing == (h, p):
                        continue
                    seq += 1
                    recs.append({'seq': seq, 'hook': h, 'phase': p, 'frame_count': 1002, 'instance_in_window_character': True})
            (w / 'frames' / 'drivers_005.json').write_text(json.dumps({'frame_count': 1002, 'records': recs}), encoding='utf-8')
        return w

    def test_v6_driver_hooks_accepted_and_order_observed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.v6_window(tmp), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])
            obs = {o['observation']: o['value'] for o in r['observations']}
            self.assertTrue(obs['late_update_hook_order']['matches'])

    def test_v6_rejects_missing_hook_and_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.v6_window(tmp, errors=2, missing=('neck', 'postfix')), NONCE)
            failed = {c['check'] for c in r['checks'] if not c['ok']}
            self.assertTrue({'driver_hook_errors_zero', 'driver_hooks_all_fired_for_character'} <= failed, failed)

    def test_v6_without_hooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.v6_window(tmp, hooks='0', drivers=False), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])

    def test_frame_mode_ignores_bridge_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.window(tmp, mode='frame'), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])


if __name__ == '__main__':
    unittest.main()
