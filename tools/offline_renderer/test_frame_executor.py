"""Frame-program converter/executor tests and the HS2 acceptance runner (ladder step 1).

CPU tests (always): converter mechanics on synthetic RenderDoc XML fixtures (not HS2
evidence) - capture-start classes, Map/Unmap -> map_write, empty-range Unmaps,
D3D11RenderState hazard resolution, readback event binding, rejection branches.

GPU tests (opt-in, ``HS2_FRAME_EXECUTOR`` = built frame_executor.exe): analytic
fixture programs with HLSL compiled through the system d3dcompiler_47 (fixture only,
never HS2 content) on the hardware adapter and on WARP: exact clear/draw/copy
bytes, map_write range semantics (RenderDoc MapIntercept), blob-hash and
unknown-field rejection, and runtime hazard refusal.

HS2 acceptance (explicit command, GPU replay of the existing RDC)::

    python -m tools.offline_renderer.test_frame_executor hs2-acceptance --executor EXE --tool REPLAY_EXE \
        --renderdoc-dir DIR --rdc RDC --index INDEX --xml ZIP_XML --inspect INSPECT --frame-graph FRAME_GRAPH_DIR \
        --native-png PNG --out FRESH_DIR --receipt tools/offline_renderer/frame_executor_acceptance_20261020.json
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.render import frame_program as fp  # noqa: E402
import capture_to_program as c2p  # noqa: E402
from .test_frame_graph import (X, ids, Fixture, standard_fixture, create_buffer, create_rt,  # noqa: E402
                               initial_state, VB, CB, RT_A, RTV_A, SRV_A, RT_B, RTV_B, SRV_B)


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ===================================================================== CPU tests
class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, fx, through, readbacks=(), name='prog'):
        xml, zip_path, listing = fx.write(self.dir)
        conv = c2p.Converter(xml, zip_path, listing)
        return conv.convert(self.dir / name, through, list(readbacks))

    def test_standard_fixture(self):
        fx, ev = standard_fixture()
        fix_blend_factor(fx)
        program, report = self.convert(fx, ev['draw3'], [{'name': 'b', 'event': ev['draw3'], 'resource': RT_B},
                                                         {'name': 'a', 'event': ev['draw2'] + 1, 'resource': RT_A}])
        ops = [c['op'] for c in program['commands']]
        # RT_A/RT_B are not in NeededInitials -> no RenderDoc reset clear; VB listed (buffer: capture-start record)
        self.assertEqual(ops, ['map_write', 'clear_rtv', 'draw_indexed', 'draw_indexed', 'map_write', 'draw_indexed'])
        m = program['commands'][0]
        self.assertEqual((m['resource'], m['map_type'], m['start'], m['end']), (CB, 4, 0, 64))
        d1, d3 = program['commands'][2], program['commands'][5]
        self.assertEqual(d1['state']['ps']['srvs'], [{'slot': 0, 'view': SRV_B}])
        self.assertEqual(d3['state']['ps']['srvs'], [{'slot': 0, 'view': SRV_A}])
        self.assertEqual(d3['state']['om']['render_targets'], [{'slot': 0, 'view': RTV_B}])
        self.assertEqual(d1['state']['om']['blend_factor'], [1.0, 1.0, 1.0, 1.0])
        # OMSetRenderTargets(RTV_B) unbinds SRV_B from PS slot 0 (ChangeRefWrite -> UnbindForRead)
        self.assertEqual([(h['resolution'], h['object']) for h in report['hazard_resolutions']], [('unbind_srv_PS', SRV_B)])
        rb = {r['name']: r for r in program['readbacks']}
        self.assertEqual(rb['b']['after_command'], 5)
        self.assertEqual(rb['a']['after_command'], 3)  # last command at or before the requested event
        cls = {r['id']: r['provenance']['initial_state'] for r in program['resources']}
        self.assertEqual(cls[VB], 'capture_start_createbuffer_record')
        self.assertEqual(cls[RT_A], 'undefined_at_creation')

    def test_initial_state_classes_and_hazard_force_null(self):
        fx = Fixture()
        fx.chunk('Internal::DeviceInitialisation', [X.rid('ImmediateContext', 7)])
        create_rt(fx, RT_A, RTV_A, SRV_A)
        create_rt(fx, RT_B, RTV_B, SRV_B)
        fx.chunk('ID3D11Device::CreateTexture2D', [
            X.st('Descriptor', [X.u('Width', 4), X.u('Height', 4), X.u('MipLevels', 1), X.u('ArraySize', 1), X.u('Format', 39),
                                X.st('SampleDesc', [X.u('Count', 1), X.u('Quality', 0)]), X.u('Usage', 0), X.u('BindFlags', 72),
                                X.u('CPUAccessFlags', 0), X.u('MiscFlags', 0)]), X.arr('pInitialData', []), X.rid('pTexture', 140)])
        fx.chunk('ID3D11Device::CreateDepthStencilView', [X.rid('pResource', 140), X.st('pDesc', [
            X.u('Format', 40), X.u('ViewDimension', 3), X.u('Flags', 0), X.st('Texture2D', [X.u('MipSlice', 0)])]), X.rid('pView', 141)])
        data = bytes(range(64))
        fx.chunk('Internal::Initial Contents', [X.u('type', 4), X.rid('id', RT_B), X.u('NumSubresources', 1), X.b('OmittedContents', False),
                                                X.u('RowPitch', 32), X.buf('SubresourceContents', fx.blob(b'\x07' * 112), 112)])
        fx.chunk('Internal::List of Initial Contents Resources', [X.arr('NeededInitials', [
            X.st(None, [X.rid('id', RT_A), X.b('written', False)]), X.st(None, [X.rid('id', RT_A), X.b('written', True)]),
            X.st(None, [X.rid('id', RT_B), X.b('written', True)]), X.st(None, [X.rid('id', 140), X.b('written', False)])])])
        fx.chunk('Internal::Beginning of Capture', [c2p_initial_state(), X.arr('HiddenStreamOutCounters', []), X.u('numAnnotations', 0)])
        fx.captured = True
        fx.chunk('ID3D11DeviceContext::OMSetRenderTargets', [X.u('NumViews', 1), ids('ppRenderTargetViews', [RTV_A]), X.rid('pDepthStencilView', 141)])
        e_srv = fx.chunk('ID3D11DeviceContext::PSSetShaderResources', [X.u('StartSlot', 0), X.u('NumViews', 2), ids('ppShaderResourceViews', [SRV_A, SRV_B])])
        e_clear = fx.chunk('ID3D11DeviceContext::ClearRenderTargetView', [X.rid('pRenderTargetView', RTV_A), X.arr('ColorRGBA', [X.f(None, 1)] * 4)], action=True)
        program, report = self.convert(fx, e_clear, [{'name': 'a', 'event': e_clear, 'resource': RT_A},
                                                     {'name': 'b', 'event': e_clear, 'resource': RT_B}])
        cls = {r['id']: r['provenance']['initial_state'] for r in program['resources']}
        # first NeededInitials record wins (written=false) -> RenderDoc fast clear to zero
        self.assertEqual(cls[RT_A], 'renderdoc_clear_rtv_zero')
        self.assertEqual(cls[RT_B], 'initial_contents_chunk')
        self.assertEqual(cls[140], 'renderdoc_clear_dsv_one_zero')
        init = next(r for r in program['resources'] if r['id'] == RT_B)['initial']
        self.assertEqual((init[0]['row_pitch'], init[0]['slice_pitch']), (32, 128))
        clears = [c for c in program['commands'] if 'initial_state' in (c.get('provenance') or {})]
        self.assertEqual(sorted(c['op'] for c in clears), ['clear_dsv', 'clear_rtv'])
        dsv_clear = next(c for c in clears if c['op'] == 'clear_dsv')
        self.assertEqual((dsv_clear['flags'], dsv_clear['depth'], dsv_clear['stencil']), (3, 1.0, 0))
        view = next(v for v in program['views'] if v['id'] == dsv_clear['view'])
        self.assertEqual(view['desc']['Format'], 40)  # GetDepthTypedFormat(R32_TYPELESS) = D32_FLOAT
        self.assertIn({'event': e_srv, 'resolution': 'forced_null_srv_PS', 'slot': 0, 'object': SRV_A}, report['hazard_resolutions'])

    def test_empty_unmap_and_rejections(self):
        fx, ev = standard_fixture()
        fix_blend_factor(fx)
        xml, zip_path, listing = fx.write(self.dir)
        conv = c2p.Converter(xml, zip_path, listing)
        with self.assertRaises(c2p.Unconvertible):  # Present is not a frame-program command
            conv.convert(self.dir / 'x', ev['present'], [])
        conv = c2p.Converter(xml, zip_path, listing)
        with self.assertRaises(c2p.Unconvertible):  # not an API event of the capture
            conv.convert(self.dir / 'y', 10_000, [])
        fx2 = Fixture()
        fx2.chunk('Internal::DeviceInitialisation', [X.rid('ImmediateContext', 7)])
        create_buffer(fx2, CB, b'\0' * 64, bind=4, usage=2)
        fx2.chunk('Internal::Beginning of Capture', [c2p_initial_state(), X.arr('HiddenStreamOutCounters', []), X.u('numAnnotations', 0)])
        fx2.captured = True
        e = fx2.chunk('ID3D11DeviceContext::Unmap', [X.rid('pResource', CB), X.u('Subresource', 0), X.u('MapType', 4), X.u('MapFlags', 0),
                                                     X.u('Byte offset to start of written data', 0), X.u('Byte offset to end of written data', 0),
                                                     X.null('MapWrittenData')], action=True)
        sub = self.dir / 'empty'
        sub.mkdir()
        xml, zip_path, listing = fx2.write(sub)
        conv = c2p.Converter(xml, zip_path, listing)
        program, report = conv.convert(sub / 'p', e, [])
        self.assertEqual(program['commands'], [])
        self.assertEqual(report['skipped_without_gpu_effect'], {'Unmap_empty_range': 1})

    def test_resource_range_bitfields(self):
        a = c2p.Range(1, 0, 0xFFFFFFFF, 0, 1)  # SRV MipLevels=-1 -> 4-bit maxMip wraps to 0xE (RenderDoc bitfield)
        self.assertEqual((a.max_mip, a.full), (0xE, False))
        b = c2p.Range(1, 3, 1, 0, 1)
        self.assertTrue(a.intersects(b))
        self.assertFalse(c2p.Range(1, 0, 1, 0, 1).intersects(c2p.Range(1, 1, 1, 0, 1)))
        self.assertFalse(c2p.Range(1, 0, 1, 0, 1).intersects(c2p.Range(2, 0, 1, 0, 1)))


def c2p_initial_state():
    """frame-graph fixture state with the 4-float OM BlendFactor RenderDoc serializes for D3D11RenderState."""
    st = initial_state()
    om = next(c for c in st if c.attrib.get('name') == 'OM')
    bf = next(c for c in om if c.attrib.get('name') == 'BlendFactor')
    bf.extend([X.f(None, 1.0) for _ in range(4)])
    return st


def fix_blend_factor(fx):
    for chunk in fx.chunks:
        if chunk.attrib.get('name') == 'Internal::Beginning of Capture':
            st = next(c for c in chunk if c.attrib.get('name') == 'state')
            om = next(c for c in st if c.attrib.get('name') == 'OM')
            bf = next(c for c in om if c.attrib.get('name') == 'BlendFactor')
            bf.extend([X.f(None, 1.0) for _ in range(4)])


# ===================================================================== GPU tests
EXECUTOR = os.environ.get('HS2_FRAME_EXECUTOR')


def compile_hlsl(source, entry, target):
    """System d3dcompiler_47 (fixture shaders only)."""
    lib = ctypes.WinDLL('d3dcompiler_47.dll')
    blob, err = ctypes.c_void_p(), ctypes.c_void_p()
    src = source.encode()
    hr = lib.D3DCompile(src, len(src), None, None, None, entry.encode(), target.encode(), 0, 0,
                        ctypes.byref(blob), ctypes.byref(err))
    if hr < 0:
        raise RuntimeError('fixture HLSL compile failed')
    vt = ctypes.cast(ctypes.cast(blob, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    ptr = ctypes.WINFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)(vt[3])(blob)
    size = ctypes.WINFUNCTYPE(ctypes.c_size_t, ctypes.c_void_p)(vt[4])(blob)
    data = ctypes.string_at(ptr, size)
    ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vt[2])(blob)
    return data


FIXTURE_HLSL = '''
cbuffer C : register(b0) { float4 color; };
Texture2D<float4> T : register(t0);
float4 vs(uint id : SV_VertexID) : SV_Position { float2 p = float2((id << 1) & 2, id & 2); return float4(p * 2 - 1, 0, 1); }
float4 ps(float4 pos : SV_Position) : SV_Target { return color; }
float4 ps_tex(float4 pos : SV_Position) : SV_Target { return T.Load(int3(pos.xy, 0)); }
'''

TEX2D = {'Width': 4, 'Height': 4, 'MipLevels': 1, 'ArraySize': 1, 'Format': 28, 'SampleDesc': {'Count': 1, 'Quality': 0},
         'Usage': 0, 'BindFlags': 40, 'CPUAccessFlags': 0, 'MiscFlags': 0}


def viewport(w, h):
    return [{'TopLeftX': 0.0, 'TopLeftY': 0.0, 'Width': float(w), 'Height': float(h), 'MinDepth': 0.0, 'MaxDepth': 1.0}]


@unittest.skipUnless(EXECUTOR, 'set HS2_FRAME_EXECUTOR to the built frame_executor.exe')
class ExecutorGPUTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_exe(self, program_dir, name, *extra, expect_ok=True):
        out = self.dir / name
        r = subprocess.run([EXECUTOR, str(program_dir), str(out), *extra], capture_output=True, timeout=300)
        if expect_ok:
            self.assertEqual(r.returncode, 0, r.stderr.decode(errors='replace'))
            return json.loads((out / 'execution.json').read_text()), out
        self.assertNotEqual(r.returncode, 0)
        return r.stderr.decode(errors='replace'), out

    def fixture_program(self, name='p'):
        b = fp.ProgramBuilder(self.dir / name, provenance={'generator': 'test_frame_executor fixture'})
        vs = b.shader('vs', compile_hlsl(FIXTURE_HLSL, 'vs', 'vs_5_0'))
        ps = b.shader('ps', compile_hlsl(FIXTURE_HLSL, 'ps', 'ps_5_0'))
        ps_tex = b.shader('ps', compile_hlsl(FIXTURE_HLSL, 'ps_tex', 'ps_5_0'))
        rt = b.texture('tex2d', TEX2D, name='rt')
        rtv = b.view('rtv', rt, {'Format': 28, 'ViewDimension': 4, 'Texture2D': {'MipSlice': 0}})
        srv = b.view('srv', rt, {'Format': 28, 'ViewDimension': 4, 'Texture2D': {'MostDetailedMip': 0, 'MipLevels': 1}})
        rt2 = b.texture('tex2d', TEX2D, name='rt2')
        rtv2 = b.view('rtv', rt2, None)
        cb = b.buffer({'ByteWidth': 16, 'Usage': fp.USAGE_DYNAMIC, 'BindFlags': 4, 'CPUAccessFlags': 0x10000,
                       'MiscFlags': 0, 'StructureByteStride': 0}, initial=struct.pack('<4f', 0, 0, 0, 0), name='cb')
        rs = b.raster_state({'FillMode': 3, 'CullMode': 1, 'FrontCounterClockwise': False, 'DepthBias': 0, 'DepthBiasClamp': 0.0,
                             'SlopeScaledDepthBias': 0.0, 'DepthClipEnable': True, 'ScissorEnable': False,
                             'MultisampleEnable': False, 'AntialiasedLineEnable': False})
        return b, dict(vs=vs, ps=ps, ps_tex=ps_tex, rt=rt, rtv=rtv, srv=srv, rt2=rt2, rtv2=rtv2, cb=cb, rs=rs)

    def draw_state(self, o, ps, rtv, cb=None, cb_blob=None, srv=None):
        s = fp.DrawState()
        s.ia.topology = 4
        s.vs.shader = o['vs']
        s.ps.shader = ps
        if cb is not None:
            s.ps.constant_buffers = [{'slot': 0, 'buffer': cb, 'first_constant': 0, 'num_constants': 4096}]
        if cb_blob is not None:
            s.ps.constant_buffers = [{'slot': 0, 'blob': cb_blob}]
        if srv is not None:
            s.ps.srvs = [{'slot': 0, 'view': srv}]
        s.om.render_targets = [{'slot': 0, 'view': rtv}]
        s.rs.viewports = viewport(4, 4)
        s.rs.state = o['rs']
        return s

    def test_exact_bytes_hardware_and_warp(self):
        b, o = self.fixture_program()
        b.clear_rtv(o['rtv'], [0.0, 0.0, 1.0, 1.0])
        b.readback('cleared', o['rt'])
        b.map_write(o['cb'], 0, fp.MAP_WRITE_DISCARD, 0, 0, 16, struct.pack('<4f', 1, 0, 0, 1))
        b.draw(self.draw_state(o, o['ps'], o['rtv'], cb=o['cb']), VertexCount=3, StartVertexLocation=0)
        b.readback('red', o['rt'])
        b.draw(self.draw_state(o, o['ps'], o['rtv2'], cb_blob=b.blob(struct.pack('<4f', 0, 1, 0, 1))), VertexCount=3, StartVertexLocation=0)
        b.draw(self.draw_state(o, o['ps_tex'], o['rtv2'], srv=o['srv']), VertexCount=3, StartVertexLocation=0)
        b.readback('copied_by_shader', o['rt2'])
        b.copy_subresource_region(o['rt2'], 0, (1, 1, 0), o['rt'], 0, {'left': 0, 'top': 0, 'front': 0, 'right': 1, 'bottom': 1, 'back': 1})
        b.update_subresource(o['rt'], 0, b'\x11\x22\x33\x44' * 2, 8, 8, box={'left': 2, 'top': 3, 'front': 0, 'right': 4, 'bottom': 4, 'back': 1})
        b.readback('updated', o['rt'])
        b.write()
        for extra, name in (((), 'hw'), (('--warp',), 'warp')):
            result, out = self.run_exe(b.root, name, *extra)
            rows = {r['name']: r for r in result['readbacks']}
            self.assertEqual((out / 'cleared.raw.bin').read_bytes(), bytes([0, 0, 255, 255]) * 16)
            self.assertEqual((out / 'red.raw.bin').read_bytes(), bytes([255, 0, 0, 255]) * 16)
            self.assertEqual((out / 'copied_by_shader.raw.bin').read_bytes(), bytes([255, 0, 0, 255]) * 16)
            upd = bytearray(bytes([255, 0, 0, 255]) * 16)
            upd[3 * 16 + 8:3 * 16 + 16] = b'\x11\x22\x33\x44' * 2
            self.assertEqual((out / 'updated.raw.bin').read_bytes(), bytes(upd))
            self.assertEqual(rows['red']['sha256'], sha(bytes([255, 0, 0, 255]) * 16))
            self.assertEqual(result['adapter']['driver_type'], 'warp' if extra else 'hardware')

    def test_map_write_semantics(self):
        b = fp.ProgramBuilder(self.dir / 'm', provenance={})
        init = bytes(range(256)) * 2
        buf = b.buffer({'ByteWidth': 512, 'Usage': fp.USAGE_DYNAMIC, 'BindFlags': 1, 'CPUAccessFlags': 0x10000,
                        'MiscFlags': 0, 'StructureByteStride': 0}, initial=init)
        dst = b.buffer({'ByteWidth': 512, 'Usage': 0, 'BindFlags': 1, 'CPUAccessFlags': 0, 'MiscFlags': 0, 'StructureByteStride': 0},
                       initial=bytes(512))
        b.map_write(buf, 0, fp.MAP_WRITE_NO_OVERWRITE, 0, 100, 140, b'\xAA' * 40)   # temp-buffer copy route
        b.copy_resource(dst, buf)
        b.readback('no_overwrite', dst)
        st = b.texture('tex2d', {'Width': 64, 'Height': 1, 'MipLevels': 1, 'ArraySize': 1, 'Format': 28,
                                 'SampleDesc': {'Count': 1, 'Quality': 0}, 'Usage': fp.USAGE_STAGING, 'BindFlags': 0,
                                 'CPUAccessFlags': 0x30000, 'MiscFlags': 0}, initial=[(0, bytes([9]) * 256, 256, 256)])
        b.map_write(st, 0, fp.MAP_WRITE, 0, 1, 200, b'\x55' * 199)              # Map(WRITE) partial range
        b.readback('staging_partial', st)
        b.write()
        result, out = self.run_exe(b.root, 'hw')
        expect = bytearray(init)
        expect[100:140] = b'\xAA' * 40
        self.assertEqual((out / 'no_overwrite.raw.bin').read_bytes(), bytes(expect))
        self.assertEqual((out / 'staging_partial.raw.bin').read_bytes(), b'\x09' + b'\x55' * 199 + b'\x09' * 56)

    def test_explicit_failures(self):
        b, o = self.fixture_program('bad_blob')
        b.clear_rtv(o['rtv'], [0.0, 0.0, 0.0, 0.0])
        b.readback('x', o['rt'])
        b.write()
        cb_blob = next(r for r in b.program['resources'] if r['id'] == o['cb'])['initial'][0]['blob']
        (b.root / 'blobs' / cb_blob).write_bytes(b'\0' * 16 + b'x')
        err, _ = self.run_exe(b.root, 'e1', expect_ok=False)
        self.assertIn('SHA-256', err)
        b2, o2 = self.fixture_program('unknown_field')
        b2.clear_rtv(o2['rtv'], [0.0, 0.0, 0.0, 0.0])
        b2.readback('x', o2['rt'])
        b2.program['resources'][0]['desc']['Bogus'] = 1
        (b2.root / 'program.json').write_text(json.dumps(b2.program))
        err, _ = self.run_exe(b2.root, 'e2', expect_ok=False)
        self.assertIn("unsupported field 'Bogus'", err)
        b3, o3 = self.fixture_program('hazard')
        b3.draw(self.draw_state(o3, o3['ps_tex'], o3['rtv'], srv=o3['srv']), VertexCount=3, StartVertexLocation=0)
        b3.readback('x', o3['rt'])
        b3.write()
        err, _ = self.run_exe(b3.root, 'e3', expect_ok=False)
        self.assertIn('runtime refused SRV slot 0', err)


# ===================================================================== HS2 acceptance
CHECKPOINTS = [  # (name, pass label, colour/depth resource or None=first target, occurrence of that label+resource in segment)
    ('sss_shadow_map', 'shadow_map', 14283, 0),
    ('sss_lighting', 'sss_lighting', 59360, 0),
    ('sss_blur_1', 'sss_separable_blur', 59388, 0),
    ('sss_blur_out', 'sss_separable_blur', 59396, 0),
    ('capture_depth_texture', 'depth_texture_prepass', 59404, 0),
    ('capture_shadow_map', 'shadow_map', 14283, 1),
    ('screenspace_shadows', 'screenspace_shadows', 59416, 0),
    ('forward_opaque', 'forward_scene', 59330, 0),
    ('forward_depth', 'forward_scene', 59334, 0),
    ('motion_vectors', 'motion_vectors', 59420, 0),
    ('msvo_occlusion', 'ambient_occlusion_msvo', 59338, 0),
    ('forward_after_ao', 'ambient_occlusion_msvo', 59330, 0),
    ('forward_liquid', 'forward_scene', 59330, 1),
    ('forward_transparent', 'forward_scene', 59330, 2),
    ('post_dof', 'post_depth_of_field', 59648, 0),
    ('color_grading_lut', 'color_grading_lut_bake', 14970, 0),
    ('post_uber', 'post_uber', 59644, 0),
    ('post_smaa', 'post_smaa', 59648, 0),
    ('post_final_pass', 'post_final_pass', 59330, 0),
    ('ctaa_enhance', 'temporal_aa_ctaa', None, 0),
    ('ctaa_pc', 'temporal_aa_ctaa', None, 1),
    ('ctaa_blit_out', 'blit_copy', 59684, 0),
    ('capture_target', 'blit_copy', 59322, 0),
]


def segment_checkpoints(frame_graph_dir, segment):
    summary = json.loads((frame_graph_dir / 'summary.json').read_text())
    passes = json.loads((frame_graph_dir / 'passes.json').read_text())
    seg = summary['segments'][segment]['passes']
    out = []
    for name, label, resource, occurrence in CHECKPOINTS:
        found = []
        for i in seg:
            p = passes[i]
            if p['label'] != label:
                continue
            targets = [t['resource'] for t in p['targets']]
            if resource is None or resource in targets:
                found.append((p, targets[0] if resource is None else resource))
        if len(found) <= occurrence:
            raise RuntimeError('checkpoint %s not found in segment %d' % (name, segment))
        p, res = found[occurrence]
        out.append({'name': 's%02d_%s' % (segment, name), 'event': p['last_event'], 'resource': res, 'subresource': 0,
                    'pass_label': label, 'pass_index': p['pass_index'], 'pass_events': [p['first_event'], p['last_event']]})
    return out


def run_logged(argv, log_dir, timeout):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from logged_process import run_logged as rl
    return rl(argv, log_dir, timeout=timeout)


def diff_summary(a, b):
    if len(a) != len(b):
        return {'length_differs': [len(a), len(b)]}
    import numpy as np
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    d = np.nonzero(x != y)[0]
    return {'differing_bytes': int(d.size), 'first_differing_byte': int(d[0]) if d.size else None}


def strip_provenance(program):
    """Executed content only: drop top-level and per-entry provenance (labels, hashes, timings)."""
    out = {}
    for k, v in program.items():
        if k == 'provenance':
            continue
        if isinstance(v, list):
            out[k] = [{kk: vv for kk, vv in e.items() if kk != 'provenance'} for e in v]
        else:
            out[k] = v
    return out


def program_equivalence(new_dir, old_dir):
    new = json.loads((new_dir / 'program.json').read_text(encoding='utf-8'))
    old = json.loads((old_dir / 'program.json').read_text(encoding='utf-8'))
    same = strip_provenance(new) == strip_provenance(old)
    cmd_prov = [c.get('provenance') for c in new['commands']] == [c.get('provenance') for c in old['commands']]
    res_prov = {r['id']: r['provenance'] for r in old['resources']}
    changed = [{'resource': r['id'], 'old': res_prov.get(r['id']), 'new': r['provenance']}
               for r in new['resources'] if r['provenance'] != res_prov.get(r['id'])]
    return {'previous_program': str(old_dir), 'executed_content_identical': same, 'command_provenance_identical': cmd_prov,
            'resource_provenance_changes': changed}


def hs2_acceptance(a):
    import render_patched_capture as rpc
    import numpy as np
    from PIL import Image
    out = a.out.resolve()
    if out.exists():
        raise FileExistsError('Fresh acceptance directory required')
    out.mkdir(parents=True)
    fg_dir = a.frame_graph.resolve()
    summary = json.loads((fg_dir / 'summary.json').read_text())
    seg0_end = summary['segments'][0]['last_event']
    final_event = summary['segments'][16]['first_event']
    seg0 = segment_checkpoints(fg_dir, 0)
    seg15 = segment_checkpoints(fg_dir, 15)
    final = [{'name': 'final_capture_target', 'event': final_event, 'resource': 59322, 'subresource': 0,
              'orientation': 'unity_texture_rows_bottom_up'}]
    programs = [('segment0', seg0_end, seg0), ('frame_to_%d' % final_event, final_event, seg0 + seg15 + final)]
    tools = rpc.Tools(a.tool, a.renderdoc_dir)
    rdc_sha = rpc.sha256_file(a.rdc)
    native = np.asarray(Image.open(a.native_png).convert('RGBA')).tobytes()
    receipt = {'schema': 'hs2_frame_executor_acceptance/1', 'date': time.strftime('%Y-%m-%d'),
               'scope': 'Acceptance ladder step 1: mechanically converted frame program of the existing retry RDC executed by '
                        'frame_executor.exe on the capture adapter, compared bit-exactly with the official RenderDoc 1.46 replay '
                        'at explicit (event, resource) checkpoints. No game launch, injection or new capture.',
               'capture': {'rdc': str(a.rdc), 'rdc_sha256': rdc_sha, 'zip_xml': str(a.xml),
                           'zip_xml_sha256': rpc.sha256_file(a.xml)},
               'executor': {'exe': str(a.executor), 'sha256': rpc.sha256_file(a.executor),
                            'build_receipt': str(Path(a.executor).parent / 'build_receipt.json'),
                            'hash_only': bool(a.hash_only)},
               'converter_sha256': rpc.sha256_file(c2p.__file__), 'frame_program_sha256': rpc.sha256_file(fp.__file__),
               'oracle': tools.describe(), 'oracle_reused_from': str(a.oracle_reuse) if a.oracle_reuse else None,
               'native_png': {'path': str(a.native_png), 'rgba_sha256': sha(native)}, 'programs': [], 'out': str(out)}
    zip_path = a.xml.with_suffix('')
    for name, through, cps in programs:
        t0 = time.perf_counter()
        conv = c2p.Converter(a.xml, zip_path, a.inspect, a.shader_manifest, fg_dir / 'actions.json')
        readbacks = [{k: c[k] for k in ('name', 'event', 'resource', 'subresource')} | {'orientation': c.get('orientation', 'd3d_rows_top_down')}
                     for c in cps]
        links = [Path(d) / ('program_' + name) for d in a.link_blobs_from]
        program, report = conv.convert(out / ('program_' + name), through, readbacks, label=name, blob_link_sources=links)
        convert_s = time.perf_counter() - t0
        runs = {}
        flags = ['--hash-only'] if a.hash_only else []
        for run_name in ('hardware', 'hardware_repeat'):
            log = run_logged([a.executor, out / ('program_' + name), out / ('exec_%s_%s' % (name, run_name)), *flags],
                             out / ('log_exec_%s_%s' % (name, run_name)), 1800)
            if log['status'] != 'pass':
                raise RuntimeError('executor failed: ' + Path(log['stderr']).read_text(errors='replace'))
            runs[run_name] = json.loads((out / ('exec_%s_%s' % (name, run_name)) / 'execution.json').read_text())
        targets = rpc.validate_targets([{'name': c['name'], 'kind': 'texture', 'event_id': c['event'], 'resource_id': c['resource'],
                                         'orientation': c.get('orientation', 'd3d_rows_top_down'), 'mip': 0, 'slice': 0, 'png': False}
                                        for c in cps])
        t1 = time.perf_counter()
        if a.oracle_reuse:
            oracle_dir = a.oracle_reuse.resolve() / ('oracle_' + name)
            oracle = json.loads((oracle_dir / 'receipt.json').read_text())
            if oracle['status'] != 'pass' or oracle['capture']['rdc_sha256'] != rdc_sha or oracle['plan'] is not None or oracle['patches']:
                raise RuntimeError('reused oracle receipt is not an identity replay of this RDC')
            if oracle['tools']['exe_sha256'] != tools.exe_sha256 or oracle['tools']['renderdoc_dll_sha256'] != tools.dll_sha256:
                raise RuntimeError('reused oracle receipt was produced by a different replay tool/DLL')
            got = {(t['name'], t['event_id'], t['resource_id']) for t in oracle['targets']}
            want = {(t['name'], t['event_id'], t['resource_id']) for t in targets}
            if got != want:
                raise RuntimeError('reused oracle targets differ from the checkpoints')
            for t in oracle['targets']:
                if rpc.sha256_file(t['raw']['path']) != t['raw']['sha256']:
                    raise RuntimeError('reused oracle raw read-back changed on disk: ' + t['name'])
            oracle_src = {'receipt': str(oracle_dir / 'receipt.json'), 'reused': True,
                          'raw_files_rehashed': len(oracle['targets'])}
        else:
            with rpc.ReplaySession(tools, a.rdc, a.index, out / ('oracle_session_' + name)) as sess:
                oracle = sess.render(None, targets, out / ('oracle_' + name))
            oracle_src = {'receipt': str(out / ('oracle_' + name) / 'receipt.json'), 'reused': False}
        oracle_src.update(seconds=time.perf_counter() - t1, tool_ms=oracle['timing']['tool_ms'])
        orc = {t['name']: t for t in oracle['targets']}
        ex = {r['name']: r for r in runs['hardware']['readbacks']}
        rep = {r['name']: r for r in runs['hardware_repeat']['readbacks']}
        rows = []
        for c in cps:
            o, e = orc[c['name']], ex[c['name']]
            row = {'name': c['name'], 'event': c['event'], 'resource': c['resource'], 'pass': c.get('pass_label'),
                   'bytes': e['bytes'], 'executor_sha256': e['sha256'], 'oracle_sha256': o['raw']['sha256'],
                   'bit_exact': e['sha256'] == o['raw']['sha256'], 'repeat_identical': rep[c['name']]['sha256'] == e['sha256'],
                   'format': e.get('format'), 'extent': [e.get('width'), e.get('height'), e.get('depth')]}
            if not row['bit_exact']:
                row['difference'] = diff_summary((out / ('exec_%s_hardware' % name) / e['raw']).read_bytes(),
                                                 Path(o['raw']['path']).read_bytes()) if e['raw'] else \
                    'hash-only run; rerun without --hash-only for byte differences'
            if c['name'] == 'final_capture_target':
                row['topdown_sha256'] = e['topdown_sha256']
                row['topdown_equals_native_png_rgba'] = e['topdown_sha256'] == sha(native)
            rows.append(row)
        warp = None
        if name == 'segment0':
            log = run_logged([a.executor, out / ('program_' + name), out / ('exec_%s_warp' % name), '--warp', *flags],
                             out / ('log_exec_%s_warp' % name), 3600)
            warp = {'status': log['status']}
            if log['status'] == 'pass':
                w = json.loads((out / ('exec_%s_warp' % name) / 'execution.json').read_text())
                warp.update(adapter=w['adapter'], timing=w['timing'],
                            readbacks_equal_to_hardware=sum(1 for r in w['readbacks'] if r['sha256'] == ex[r['name']]['sha256']),
                            readbacks=len(w['readbacks']),
                            note='WARP is a different device; no WARP oracle replay exists, so equality is not an acceptance criterion')
        equivalence = [program_equivalence(out / ('program_' + name), Path(d) / ('program_' + name)) for d in a.previous_programs]
        receipt['programs'].append({
            'name': name, 'through_event': through, 'program_dir': str(out / ('program_' + name)),
            'program_json_sha256': runs['hardware']['program_json_sha256'],
            'conversion': {k: report[k] for k in ('commands', 'ops', 'objects', 'initial_state_classes', 'skipped_without_gpu_effect',
                                                  'undefined_at_creation_resources', 'undefined_content_dependencies',
                                                  'blob_count', 'blob_bytes', 'blobs_hard_linked')} |
                          {'hazard_resolutions': len(report['hazard_resolutions']), 'seconds': convert_s,
                           'frame_graph_crosscheck': {'checked': report['frame_graph_crosscheck']['checked'],
                                                      'mismatched': len(report['frame_graph_crosscheck']['mismatched'])}},
            'previous_program_equivalence': equivalence,
            'executor': {'adapter': runs['hardware']['adapter'], 'timing': runs['hardware']['timing'],
                         'repeat_timing': runs['hardware_repeat']['timing'], 'blobs_verified': runs['hardware']['blobs_verified']},
            'oracle': oracle_src, 'warp': warp, 'checkpoints': rows,
            'all_bit_exact': all(r['bit_exact'] for r in rows), 'all_repeat_identical': all(r['repeat_identical'] for r in rows)})
    receipt['passed'] = all(p['all_bit_exact'] and p['all_repeat_identical'] for p in receipt['programs']) and \
        all(r.get('topdown_equals_native_png_rgba', True) for p in receipt['programs'] for r in p['checkpoints'])
    (out / 'acceptance_full.json').write_text(json.dumps(receipt, indent=1))
    if a.receipt:
        Path(a.receipt).write_text(json.dumps(compact_receipt(receipt), indent=1))
    print(json.dumps({'passed': receipt['passed'], 'programs': [(p['name'], p['all_bit_exact'], len(p['checkpoints']))
                                                                for p in receipt['programs']]}))
    return receipt


def compact_receipt(r):
    """Tracked receipt: hashes and verdicts per checkpoint; bulky paths stay in acceptance_full.json."""
    c = {k: r[k] for k in ('schema', 'date', 'scope', 'capture', 'executor', 'converter_sha256', 'frame_program_sha256',
                           'oracle', 'oracle_reused_from', 'native_png', 'out', 'passed')}
    c['programs'] = []
    for p in r['programs']:
        q = {k: p[k] for k in ('name', 'through_event', 'program_json_sha256', 'all_bit_exact', 'all_repeat_identical',
                               'previous_program_equivalence', 'oracle')}
        q['conversion'] = {k: v for k, v in p['conversion'].items() if k != 'undefined_content_dependencies'}
        q['undefined_content_dependencies'] = [{k: d.get(k) for k in ('resource', 'first_op', 'first_event', 'undefined_bytes_kept',
                                                                      'fully_defined_by_first_write')}
                                               for d in p['conversion']['undefined_content_dependencies']]
        q['executor'] = {'adapter': p['executor']['adapter'], 'timing_ms': p['executor']['timing'],
                         'repeat_timing_ms': p['executor']['repeat_timing']}
        q['warp'] = None if not p['warp'] else {k: p['warp'].get(k) for k in ('status', 'readbacks', 'readbacks_equal_to_hardware', 'note')}
        q['checkpoints'] = [{k: x[k] for k in ('name', 'event', 'resource', 'pass', 'format', 'extent', 'executor_sha256',
                                               'oracle_sha256', 'bit_exact', 'repeat_identical', 'topdown_equals_native_png_rgba')
                             if k in x} for x in p['checkpoints']]
        c['programs'].append(q)
    return c


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'hs2-acceptance':
        ap = argparse.ArgumentParser(prog='test_frame_executor hs2-acceptance')
        for k in ('executor', 'tool', 'renderdoc-dir', 'rdc', 'index', 'xml', 'inspect', 'frame-graph', 'native-png', 'out'):
            ap.add_argument('--' + k, type=Path, required=True)
        ap.add_argument('--shader-manifest', type=Path, action='append', default=[])
        ap.add_argument('--receipt', type=Path)
        ap.add_argument('--oracle-reuse', type=Path, help='earlier acceptance dir whose identity-replay oracle read-backs are re-hashed and reused')
        ap.add_argument('--link-blobs-from', type=Path, action='append', default=[], help='earlier acceptance dir (hard-link identical blobs)')
        ap.add_argument('--previous-programs', type=Path, action='append', default=[], help='earlier acceptance dir for executed-content equivalence')
        ap.add_argument('--hash-only', action='store_true', help='executor records read-back hashes without raw files')
        hs2_acceptance(ap.parse_args(argv[1:]))
        return
    unittest.main(argv=[sys.argv[0], *argv])


if __name__ == '__main__':
    main()
