"""CPU checks for the head GPU skinning data path (no game, no GPU, no RDC).

A synthetic official-schema zip.xml capture reproduces the recorded Unity
structure: a dynamic 40-byte skinned VB whose capture-start contents live in
CreateBuffer.InitialData, static named UV/skin streams, a mesh IB and a
per-draw UnityPerDraw cbuffer written by Map/Unmap before the draw.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.render import gpu_skin_inputs as g  # noqa: E402
from src.render.capture_mutation import mutate_capture  # noqa: E402

CTX = 7


class Builder:
    def __init__(self):
        self.chunks = []
        self.blobs = []

    def blob(self, data: bytes):
        self.blobs.append(data)
        return f'<buffer name="{{name}}" typename="Byte Buffer" byteLength="{len(data)}">{len(self.blobs) - 1}</buffer>', len(self.blobs) - 1

    def buf(self, name, data):
        text, _ = self.blob(data)
        return text.replace('{name}', name)

    def chunk(self, name, body):
        self.chunks.append((name, body))

    def write(self, root: Path):
        lines = ['<?xml version="1.0"?>', '<rdc>', '<header><driver id="1">D3D11</driver></header>', '<chunks version="20">']
        for i, (name, body) in enumerate(self.chunks):
            lines.append(f'<chunk id="1" chunkIndex="{i}" name="{escape(name)}" length="0">{body}</chunk>')
        lines += ['</chunks>', '</rdc>']
        xml = root / 'cap.zip.xml'
        xml.write_text('\n'.join(lines), encoding='utf-8')
        with zipfile.ZipFile(root / 'cap.zip', 'w') as z:
            for i, data in enumerate(self.blobs):
                z.writestr('%06d' % i, data)
        return xml


def rid(name, value):
    return f'<ResourceId name="{name}">{value}</ResourceId>'


def u(name, value):
    return f'<uint name="{name}">{value}</uint>'


def arr(name, values, tag='ResourceId'):
    return f'<array name="{name}">' + ''.join(f'<{tag}>{v}</{tag}>' for v in values) + '</array>'


def stage(name, obj=0, cbs=None):
    cbs = cbs or [0] * 14
    return (f'<struct name="{name}">{rid("Object", obj)}{arr("ConstantBuffers", cbs)}'
            f'{arr("CBOffsets", [0] * 14, "uint")}</struct>')


def create_buffer(b, ident, width, usage, bind, data):
    desc = (f'<struct name="pDesc">{u("ByteWidth", width)}<enum name="Usage">{usage}</enum>'
            f'<enum name="BindFlags">{bind}</enum><enum name="CPUAccessFlags">0</enum>'
            f'<enum name="MiscFlags">0</enum>{u("StructureByteStride", 0)}</struct>')
    b.chunk('ID3D11Device::CreateBuffer', desc + '<null name="pInitialData" />' + rid('pBuffer', ident)
            + b.buf('InitialData', data) + f'<uint name="InitialDataLength">{width}</uint>')


def name_it(b, ident, text):
    b.chunk('ID3D11Resource::SetDebugName', rid('pResource', ident) + f'<string name="Name">{text}</string>')


def pose(q, t):
    M = np.eye(4)
    M[:3, :3] = g.unity_rotation_matrix(q)
    M[:3, 3] = np.asarray(t, np.float32)
    return M


class Fixture:
    """Three-vertex skinned surface; root pose A in capture, pose B in source."""

    def __init__(self, root: Path):
        rng = np.random.default_rng(3)
        self.n = 3
        self.tris = np.array([0, 1, 2], np.uint16)
        self.uv = rng.random((3, 2)).astype(np.float32)
        self.weights = np.array([[1, 0, 0, 0]] * 3, np.float32)
        self.bones = np.zeros((3, 4), np.int32)
        q = Rotation.from_euler('xyz', [3, -2, 5], degrees=True).as_quat()
        self.root = pose(q, [0.1, 14.45, -0.06])
        world = rng.random((3, 3)) + [0, 14.4, 0]
        self.world = world
        self.normals = rng.random((3, 3))
        self.tangents = rng.random((3, 3))
        self.hand = np.array([1., -1., 1.])
        self.q = q
        vb = g.skinned_vb_bytes(world, self.normals, self.tangents, self.hand, self.root)
        cb = bytearray(176)
        cb[0:64] = g.matrix_bytes(self.root)
        cb[64:128] = g.matrix_bytes(np.linalg.inv(self.root))
        b = Builder()
        b.chunk('Internal::DeviceInitialisation', rid('ImmediateContext', CTX))
        create_buffer(b, 10, 120, 2, 1, vb)
        name_it(b, 10, 'Buffer-1-120')
        create_buffer(b, 11, 24, 0, 1, self.uv.tobytes())
        name_it(b, 11, 'o_test')
        # stream2 layout: four float weights followed by four int indices per vertex.
        create_buffer(b, 12, 96, 0, 1, b''.join(self.weights[i].tobytes() + self.bones[i].tobytes() for i in range(3)))
        name_it(b, 12, 'o_test')
        create_buffer(b, 13, 6, 0, 2, self.tris.tobytes())
        name_it(b, 13, 'o_test(Clone)')
        create_buffer(b, 14, 176, 2, 4, bytes(176))
        name_it(b, 14, 'ConstantBuffer-179-176')
        elements = [('POSITION', 0, 6, 0, 0), ('NORMAL', 0, 6, 0, 12), ('TANGENT', 0, 2, 0, 24), ('TEXCOORD', 0, 16, 1, 0)]
        descs = ''.join(f'<struct><string name="SemanticName">{s}</string>{u("SemanticIndex", i)}'
                        f'<enum name="Format">{f}</enum>{u("InputSlot", sl)}{u("AlignedByteOffset", o)}'
                        f'<enum name="InputSlotClass">0</enum>{u("InstanceDataStepRate", 0)}</struct>'
                        for s, i, f, sl, o in elements)
        b.chunk('ID3D11Device::CreateInputLayout', f'<array name="pInputElementDescs">{descs}</array>'
                + b.buf('pShaderBytecodeWithInputSignature', b'\0') + rid('pInputLayout', 20))
        b.chunk('ID3D11Device::CreateVertexShader', b.buf('pShaderBytecode', b'DXBC') + rid('pClassLinkage', 0) + rid('pShader', 30))
        ia = (f'<struct name="IA">{rid("Layout", 0)}{u("Topo", 4)}{arr("VBs", [0] * 32)}'
              f'{arr("Strides", [0] * 32, "uint")}{arr("Offsets", [0] * 32, "uint")}{rid("IndexBuffer", 0)}'
              f'<enum name="IndexFormat">0</enum>{u("IndexOffset", 0)}</struct>')
        state = ('<struct name="state">' + ia + ''.join(stage(s) for s in ('VS', 'HS', 'DS', 'GS', 'PS', 'CS'))
                 + f'<struct name="SO">{arr("Buffers", [0] * 4)}</struct></struct>')
        b.chunk('Internal::Beginning of Capture', state + arr('HiddenStreamOutCounters', [], 'uint') + u('numAnnotations', 0))
        c = rid('Context', CTX)
        b.chunk('ID3D11DeviceContext::IASetVertexBuffers', c + u('StartSlot', 0) + u('NumBuffers', 3)
                + arr('ppVertexBuffers', [10, 11, 12]) + arr('pStrides', [40, 8, 32], 'uint') + arr('pOffsets', [0, 0, 0], 'uint'))
        b.chunk('ID3D11DeviceContext::IASetIndexBuffer', c + rid('pIndexBuffer', 13) + '<enum name="Format">57</enum>' + u('Offset', 0))
        b.chunk('ID3D11DeviceContext::IASetInputLayout', c + rid('pInputLayout', 20))
        b.chunk('ID3D11DeviceContext::VSSetShader', c + rid('pShader', 30) + arr('ppClassInstances', []) + u('NumClassInstances', 0))
        b.chunk('ID3D11DeviceContext::Map', c + rid('pResource', 14) + u('Subresource', 0) + '<enum name="MapType">4</enum>' + u('MapFlags', 0))
        b.chunk('ID3D11DeviceContext::Unmap', c + rid('pResource', 14) + u('Subresource', 0) + '<enum name="MapType">4</enum>'
                + u('MapFlags', 0) + u('Byte offset to start of written data', 0) + u('Byte offset to end of written data', 176)
                + b.buf('MapWrittenData', bytes(cb)))
        self.unmap_chunk = len(b.chunks) - 1
        b.chunk('ID3D11DeviceContext::VSSetConstantBuffers', c + u('StartSlot', 2) + u('NumBuffers', 1) + arr('ppConstantBuffers', [14]))
        b.chunk('ID3D11DeviceContext::DrawIndexed', c + u('IndexCount', 3) + u('StartIndexLocation', 0) + '<int name="BaseVertexLocation">0</int>')
        self.xml = b.write(root)
        self.surface = {'mesh_name': 'o_test', 'renderer_path': '/x/o_test', 'vertex_count': 3, 'active_in_hierarchy': True,
                        'source': {'triangles': self.tris.tolist(), 'uv': self.uv.tolist(),
                                   'bone_weights': self.weights.tolist(), 'bone_indices': self.bones.tolist()}}
        self.usage = {30: {'object_matrices': [{'cb_slot': 2, 'first_register': 0, 'operand': 'v0', 'inputs': ['POSITION0']}],
                           'normal_rows': [(2, 4), (2, 5), (2, 6)]}}


ASM_MV = """//
// Input signature:
//
// Name                 Index   Mask Register SysValue  Format   Used
// -------------------- ----- ------ -------- -------- ------- ------
// POSITION                 0   xyzw        0     NONE   float   xyzw
// TEXCOORD                 4   xyz         1     NONE   float   xyz
//
// Output signature:
vs_4_0
mul r0.xyzw, v0.yyyy, cb1[1].xyzw
mad r0.xyzw, cb1[0].xyzw, v0.xxxx, r0.xyzw
mad r0.xyzw, cb1[2].xyzw, v0.zzzz, r0.xyzw
mad r1.xyzw, cb1[3].xyzw, v0.wwww, r0.xyzw
mul r2.xyzw, r1.yyyy, cb0[3].xyzw
mad r2.xyzw, cb0[2].xyzw, r1.xxxx, r2.xyzw
mad r2.xyzw, cb0[4].xyzw, r1.zzzz, r2.xyzw
mad o0.xyzw, cb0[5].xyzw, r1.wwww, r2.xyzw
mov r1.xyz, v1.xyzx
mov r1.w, l(1.000000)
movc r1.xyzw, cb0[14].xxxx, r1.xyzw, v0.xyzw
mul r2.xyzw, r1.yyyy, cb0[11].xyzw
mad r2.xyzw, cb0[10].xyzw, r1.xxxx, r2.xyzw
mad r2.xyzw, cb0[12].xyzw, r1.zzzz, r2.xyzw
mad r1.xyzw, cb0[13].xyzw, r1.wwww, r2.xyzw
ret
"""


class GpuSkinInputTests(unittest.TestCase):
    def test_unity_rotation_matches_reference(self):
        for seed in range(5):
            q = Rotation.random(random_state=seed).as_quat()
            m = g.unity_rotation_matrix(q).astype(float)
            self.assertLess(np.abs(m - Rotation.from_quat(q).as_matrix()).max(), 2e-7)

    def test_world_rotation_is_product_of_local_rotations(self):
        a = Rotation.from_euler('x', 10, degrees=True).as_quat()
        b = Rotation.from_euler('y', 20, degrees=True).as_quat()
        transforms = {1: {'parent_id': None}, 2: {'parent_id': 1}}
        local = {1: {'local_rotation_xyzw': a}, 2: {'local_rotation_xyzw': b}}
        q = g.world_rotation(transforms, local, 2)
        expected = (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()
        self.assertLess(min(np.abs(q - expected).max(), np.abs(q + expected).max()), 1e-12)

    def test_root_relative_vertices_roundtrip(self):
        rng = np.random.default_rng(1)
        root = pose(Rotation.random(random_state=2).as_quat(), [0.2, 14.0, -0.1])
        world = rng.random((5, 3)) + [0, 14, 0]
        data = np.frombuffer(g.skinned_vb_bytes(world, world * 0 + 1, world * 0 + 2, np.ones(5), root), np.float32).reshape(5, 10)
        back = (np.c_[data[:, :3], np.ones(5)] @ root.T)[:, :3]
        self.assertLess(np.abs(back - world).max(), 5e-6)
        self.assertTrue(np.array_equal(data[:, 9], np.ones(5, np.float32)))

    def test_matrix_bytes_are_column_registers(self):
        M = np.arange(16, dtype=float).reshape(4, 4)
        regs = np.frombuffer(g.matrix_bytes(M), np.float32).reshape(4, 4)
        self.assertTrue(np.array_equal(regs[3], M[:, 3].astype(np.float32)))

    def test_motion_vector_shader_matrices(self):
        usage = g.object_matrix_uses(ASM_MV)
        rows = {(m['cb_slot'], m['first_register']): m['inputs'] for m in usage['object_matrices']}
        self.assertEqual(rows[(1, 0)], ['POSITION0'])
        self.assertEqual(rows[(0, 10)], ['POSITION0', 'TEXCOORD4'])
        self.assertNotIn((0, 2), rows)  # view-projection applied to a CB product

    def test_end_to_end_patch_and_mutation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            fx = Fixture(root)
            index = g.CaptureIndex(fx.xml)
            self.addCleanup(index.close)
            writes, draws = g.simulate(index)
            self.assertEqual(len(draws), 1)
            bindings = g.bind_surfaces(index, draws, [fx.surface])
            self.assertEqual(bindings[0]['status'], 'bound')
            ev = bindings[0]['static_evidence']
            self.assertTrue(ev['index_buffer_equal'] and ev['uv0_equal'] and ev['skin_stream_weights_equal'])
            points, _ = g.upload_points(index, writes, draws, bindings, fx.usage)
            roles = sorted(p['role'] for p in points)
            self.assertEqual(roles, ['per_draw_object_matrices', 'skinned_vertices_current'])
            poses = g.label_poses(index, writes, points)
            self.assertEqual(list(poses), ['pose0'])
            cbp = next(p for p in points if p['role'] == 'per_draw_object_matrices')
            self.assertEqual(cbp['world_to_object'], 'inverse:pose0')
            # Missing state: no patch, explicit gap.
            empty = g.generate_patches(index, writes, points, {}, root / 'empty')
            self.assertEqual(empty['patches'], [])
            self.assertEqual(len(empty['gaps']), 2)
            # Candidate: moved root and vertices.
            q2 = Rotation.from_euler('xyz', [1, 4, -3], degrees=True).as_quat()
            moved = pose(q2, [0.12, 14.40, -0.05])
            world2 = fx.world + [0.001, -0.002, 0.003]
            state = g.SurfaceState(0, world2, fx.normals, fx.tangents, fx.hand, moved, q2)
            plan = g.generate_patches(index, writes, points, {'pose0': {0: state}}, root / 'cand')
            offsets = sorted((p['chunk_index'], p['byte_offset']) for p in plan['patches'])
            create_chunk = index.buffers[10]['chunk']
            self.assertEqual(offsets, [(create_chunk, 0), (fx.unmap_chunk, 0), (fx.unmap_chunk, 64)])
            plan_path = root / 'cand' / 'plan.json'
            out = root / 'out'
            out.mkdir()
            result = mutate_capture(fx.xml, plan_path, out / 'cand.zip.xml')
            self.assertEqual(len(result['patches']), 3)
            mutated = g.CaptureIndex(out / 'cand.zip.xml')
            self.addCleanup(mutated.close)
            mw, md = g.simulate(mutated)
            vb = mutated.blob(mw[md[0]['bound'][('VB', 0)][1]]['buffer'])
            arr_ = np.frombuffer(vb, np.float32).reshape(3, 10).astype(float)
            back = (np.c_[arr_[:, :3], np.ones(3)] @ moved.T)[:, :3]
            self.assertLess(np.abs(back - world2).max(), 5e-6)
            cb = mutated.blob(mw[md[0]['bound'][('VSCB', 2)][1]]['buffer'])
            self.assertEqual(cb[0:64], g.matrix_bytes(moved))
            self.assertEqual(cb[128:], bytes(48))  # untouched trailing per-draw fields

    def test_shared_write_is_not_patched(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fx = Fixture(Path(tmp))
            index = g.CaptureIndex(fx.xml)
            self.addCleanup(index.close)
            writes, draws = g.simulate(index)
            bindings = g.bind_surfaces(index, draws, [fx.surface])
            points, _ = g.upload_points(index, writes, draws, bindings, fx.usage)
            for p in points:
                p['in_place_patchable'] = False
                p['non_head_consumer_draws'] = [999]
            g.label_poses(index, writes, points)
            plan = g.generate_patches(index, writes, points, {}, Path(tmp) / 'shared')
            self.assertTrue(all(gap['reason'] == 'write also consumed by non-head draws' for gap in plan['gaps']))

    def test_duplicate_inactive_surface(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            fx = Fixture(Path(tmp))
            index = g.CaptureIndex(fx.xml)
            self.addCleanup(index.close)
            writes, draws = g.simulate(index)
            inactive = dict(fx.surface, renderer_path='/y/o_test', active_in_hierarchy=False)
            rows = g.bind_surfaces(index, draws, [fx.surface, inactive])
            self.assertEqual([r['status'] for r in rows], ['bound', 'inactive_no_gpu_upload'])
            both_active = dict(inactive, active_in_hierarchy=True)
            with self.assertRaises(g.SkinPathError):
                g.bind_surfaces(index, draws, [fx.surface, both_active])


if __name__ == '__main__':
    unittest.main()
