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

    def v6_window(self, tmp, hooks='1', drivers=True, errors=0, missing=None, probe='HS2_FramePairProbe 6.0.0'):
        w = self.window(tmp, mode='frame')
        r = json.loads((w / 'capture_receipt.json').read_text(encoding='utf-8'))
        r['probe'] = probe
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

    @staticmethod
    def smr_native(**over):
        nat = {'flags_0x128': '0x0', 'smr_mesh_0x1d8': '0x1234', 'smr_mesh_is_shared_mesh': True,
               'smr_has_last_position_data_0x28d': 1, 'smr_force_matrix_recalc_0x28e': 0, 'smr_dirty_0x2c4': 0,
               'smr_skin_buffer_0x290': '0x5000', 'smr_prev_skin_buffer_0x298': '0x6000',
               'skin_buffer_holds_frame': 1000, 'prev_skin_buffer_holds_frame': 1001}
        nat.update(over)
        return nat

    def v61_window(self, tmp, scene_rows, slot_rows, renders_rows=()):
        w = self.v6_window(tmp, hooks='0', drivers=False, probe='HS2_FramePairProbe 6.1.0')
        (w / 'frames' / 'scene_005.json').write_text(json.dumps({'renderers': list(scene_rows)}), encoding='utf-8')
        (w / 'frames' / 'renders_005.json').write_text(json.dumps(
            {'frame_count': 1002, 'records': [{'seq': 1, 'frame_count': 1002, 'data': {'renderers': list(renders_rows)}}]}), encoding='utf-8')
        (w / 'frames' / 'drivers_004.json').write_text(json.dumps(
            {'frame_count': 1001, 'records': [{'seq': 1, 'frame_count': 1001, 'skin_slots': list(slot_rows)}]}), encoding='utf-8')
        return w

    def test_v61_smr_fields_accepted(self):
        smr = {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': self.smr_native()}
        mesh = {'type': 'UnityEngine.MeshRenderer', 'native': {'flags_0x128': '0x0'}}
        with tempfile.TemporaryDirectory() as tmp:
            w = self.v61_window(tmp, [smr, mesh], [{'renderer': -5, 'name': 'o_head', 'native': self.smr_native()}], [smr])
            r = vps.verify(self.index(bridge=False), w, NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])
            ev = next(c['evidence'] for c in r['checks'] if c['check'] == 'smr_native_fields_v6_1')
            self.assertEqual((ev['rows'], ev['captured_scene_rows']), (3, 1))

    def test_v61_rejects_stale_keys_errors_and_mesh_mismatch(self):
        good = {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': self.smr_native()}
        cases = [
            {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': self.smr_native(smr_skin_buffer_0x258='0x34')},
            {'type': 'UnityEngine.SkinnedMeshRenderer', 'native_error': 'NativeReadError: Renderer+0x38+0x1a0 = 0x0 is not the sharedMesh'},
            {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': self.smr_native(smr_mesh_is_shared_mesh=False)},
            {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': {'flags_0x128': '0x0'}},
        ]
        for bad in cases:
            with tempfile.TemporaryDirectory() as tmp:
                r = vps.verify(self.index(bridge=False), self.v61_window(tmp, [good], [], [bad]), NONCE)
                self.assertIn('smr_native_fields_v6_1', [c['check'] for c in r['checks'] if not c['ok']], bad)
        with tempfile.TemporaryDirectory() as tmp:     # a failing driver skin-slot row fails the window too
            slot = {'renderer': -5, 'name': 'o_head', 'native_error': 'NativeReadError: x'}
            r = vps.verify(self.index(bridge=False), self.v61_window(tmp, [good], [slot]), NONCE)
            self.assertFalse(r['accepted'])

    def test_v61_requires_smr_rows_in_captured_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            smr = {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': self.smr_native()}
            r = vps.verify(self.index(bridge=False), self.v61_window(tmp, [], [], [smr]), NONCE)
            self.assertIn('smr_native_fields_v6_1', [c['check'] for c in r['checks'] if not c['ok']])

    def v61_bridge_window(self, tmp, transforms=3, drop=None, short=None):
        import base64
        w = self.window(tmp)
        r = json.loads((w / 'capture_receipt.json').read_text(encoding='utf-8'))
        r['probe'] = 'HS2_FramePairProbe 6.1.0'
        r['driver_hooks'] = None
        (w / 'capture_receipt.json').write_text(json.dumps(r), encoding='utf-8')
        (w / 'request.ini').write_text('Mode=renderstyle\nDriverHooks=0\n', encoding='utf-8')
        (w / 'static.json').write_text(json.dumps({'transforms': [{'i': i} for i in range(transforms)]}), encoding='utf-8')
        smr = {'type': 'UnityEngine.SkinnedMeshRenderer', 'native': self.smr_native()}
        (w / 'frames' / 'scene_005.json').write_text(json.dumps({'renderers': [smr]}), encoding='utf-8')
        recs = [{'seq': 1, 'frame_count': 1002, 'phase': 'pre_cull', 'bridge_render': None, 'data': {}}]
        for o in range(16):
            if o == drop:
                data = {'character_transforms_error': 'InvalidOperationException: x'}
            else:
                n = transforms - 1 if o == short else transforms
                data = {'character_transforms': {'order': 'static.json transforms', 'count': n,
                        'transforms_f32': base64.b64encode(bytes(n * 33 * 4)).decode(),
                        'active': base64.b64encode(bytes(n)).decode()}}
            recs.append({'seq': 2 + o, 'frame_count': 1002, 'phase': 'pre_cull', 'bridge_render': {'ordinal': o}, 'data': data})
            recs.append({'seq': 100 + o, 'frame_count': 1002, 'phase': 'pre_render', 'bridge_render': {'ordinal': o},
                         'data': {'renderers': [smr]}})
        (w / 'frames' / 'renders_005.json').write_text(json.dumps({'frame_count': 1002, 'records': recs}), encoding='utf-8')
        return w

    def test_v61_bridge_pre_cull_transforms(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(), self.v61_bridge_window(tmp), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])
            self.assertIn('bridge_pre_cull_character_transforms', [c['check'] for c in r['checks']])
        for kw in ({'drop': 7}, {'short': 3}):
            with tempfile.TemporaryDirectory() as tmp:
                r = vps.verify(self.index(), self.v61_bridge_window(tmp, **kw), NONCE)
                self.assertIn('bridge_pre_cull_character_transforms', [c['check'] for c in r['checks'] if not c['ok']], kw)

    def v7_window(self, tmp, node_check='equals_cached_mesh_0x174', fresh=True, nodes_error=False, prepared=True,
                  pose_exists=True, bridge_anim=True, drop_post=False, lights_custom=True, cam_parent='CharaCamera',
                  bridge_chains=True):
        w = self.v61_bridge_window(tmp)
        r = json.loads((w / 'capture_receipt.json').read_text(encoding='utf-8'))
        r['probe'] = 'HS2_FramePairProbe 7.0.0'
        r['bridge']['body_yaw_degrees'] = 30
        (w / 'capture_receipt.json').write_text(json.dumps(r), encoding='utf-8')
        pinfo = {'smr_manager_handle_0x2f8': 3, 'prepared_renderer_info': {'flag_bytes_0_4': [0, 1, 1, 0, 1]}} if prepared else {}
        hair = {'type': 'UnityEngine.SkinnedMeshRenderer', 'name': 'hair',
                'native': self.smr_native(smr_update_when_offscreen_0x28c=1, bone_bounds_state='current', **pinfo)}
        head = {'type': 'UnityEngine.SkinnedMeshRenderer', 'name': 'o_head',
                'native': self.smr_native(smr_update_when_offscreen_0x28c=0, bone_bounds='not exported',
                                          smr_manager_handle_0x2f8=4, prepared_renderer_info=None)}
        link = lambda n: {'name': n, 'local_position': [0, 0, 0], 'local_rotation': [0, 0, 0, 1], 'local_scale': [1, 1, 1]}
        chains = {'camera_main_chain': [link('CharaCustom'), link(cam_parent), link('Main Camera')],
                  'named': {'CharaCamera': [{'chain': []}], 'Lights Custom': [{'chain': []}] if lights_custom else []},
                  'lights': [{'light': {'ref_name': 'Directional Light Key'}, 'chain': []}]}
        anim = {'animators': [{'animator': {'ref_name': 'p_cf_anim'}, 'layer0_current_clips': [{'clip': {'ref_name': 'Idle_00'}, 'weight': 1}]}],
                'maker_pose_source': {'pose_no': 1, 'entry': {'MainAB': 'custom/00/anim_f_00.unity3d', 'bundle_file_exists': pose_exists}}}
        (w / 'frames' / 'scene_005.json').write_text(json.dumps({'renderers': [head, hair], 'animation': anim,
                                                                   'transform_chains': chains}), encoding='utf-8')
        doc = json.loads((w / 'frames' / 'renders_005.json').read_text(encoding='utf-8'))
        recs = doc['records']
        for x in recs:
            if x['phase'] == 'pre_cull' and x.get('bridge_render') and bridge_anim:
                x['data']['animation'] = anim
            if x['phase'] == 'pre_cull' and x.get('bridge_render') and bridge_chains:
                x['data']['transform_chains'] = chains
        pre = [x for x in recs if x['phase'] == 'pre_cull']
        for k, x in enumerate(pre):
            if drop_post and k == 3:
                continue
            nodes = {'render_loop_0x530': '0x1', 'context': '0x2', 'fresh': fresh,
                     'queues': [{'queue': 'context_0x7d0_scene_0x8', 'count': 2,
                                 'nodes': [[0, -5, 77, node_check], [1, -9, 78, 'not_a_mesh_renderer:ParticleSystemRenderer']]},
                                {'queue': 'context_0x7d8_scene_0x10068', 'count': 0, 'nodes': []}]}
            data = {'render_nodes_error': 'NativeReadError: x'} if nodes_error and k == 2 else {'render_nodes': nodes}
            data['skinned_bounds_update_when_offscreen'] = [{'renderer': -7, 'smr_update_when_offscreen_0x28c': 1}]
            recs.append({'seq': 500 + k, 'frame_count': 1002, 'phase': 'post_render', 'render_id': k + 1,
                         'bridge_render': x.get('bridge_render'), 'data': data})
        (w / 'frames' / 'renders_005.json').write_text(json.dumps(doc), encoding='utf-8')
        return w

    def test_v7_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(), self.v7_window(tmp), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])
            names = {c['check'] for c in r['checks']}
            self.assertTrue({'render_nodes_v7', 'skinned_bounds_v7', 'animation_source_v7', 'bridge_pre_cull_animation_v7',
                             'transform_chains_v7', 'bridge_pre_cull_transform_chains_v7'} <= names)
            self.assertEqual({o['observation']: o['value'] for o in r['observations']}['bridge_yaw_degrees'], 30)

    def test_v7_rejections(self):
        cases = [({'node_check': 'mismatch'}, 'render_nodes_v7'), ({'fresh': False}, 'render_nodes_v7'),
                 ({'nodes_error': True}, 'render_nodes_v7'), ({'drop_post': True}, 'render_nodes_v7'),
                 ({'prepared': False}, 'skinned_bounds_v7'), ({'pose_exists': False}, 'animation_source_v7'),
                 ({'bridge_anim': False}, 'bridge_pre_cull_animation_v7'), ({'lights_custom': False}, 'transform_chains_v7'),
                 ({'cam_parent': 'CustomControl'}, 'transform_chains_v7'), ({'bridge_chains': False}, 'bridge_pre_cull_transform_chains_v7')]
        for kw, name in cases:
            with tempfile.TemporaryDirectory() as tmp:
                r = vps.verify(self.index(), self.v7_window(tmp, **kw), NONCE)
                self.assertIn(name, [c['check'] for c in r['checks'] if not c['ok']], kw)

    def test_v60_smr_fields_observed_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.v6_window(tmp, hooks='0', drivers=False), NONCE)
            self.assertIn('smr_fields_invalid_probe_6_0', [o['observation'] for o in r['observations']])

    def test_frame_mode_ignores_bridge_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = vps.verify(self.index(bridge=False), self.window(tmp, mode='frame'), NONCE)
            self.assertTrue(r['accepted'], [c for c in r['checks'] if not c['ok']])


if __name__ == '__main__':
    unittest.main()
