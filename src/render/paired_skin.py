"""Same-frame pairing of a FramePairProbe window with a RenderDoc capture.

The probe (tools/offline_renderer/frame_pair_probe) records, at Unity's
end-of-frame point of consecutive frames, every character transform (float32
bits) and blendshape weight, and uploads a 64-byte marker texture holding
``HS2PAIRMARKERv01`` + nonce + Time.frameCount. The capture therefore names its
own frame: the in-frame marker upload selects exactly one export.

Skinning is recomputed from that export and compared byte for byte with the
captured skinned VB writes. Besides the float64 reference, a small declared
set of float32 evaluation orders is evaluated; each is a stated hypothesis
about Unity 2018.4's closed-source CPU skinning, reported by its exact
word-equality count, never tuned to the data.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from . import gpu_skin_inputs as g

MAGIC = b'HS2PAIRMARKERv01'
TAIL = b'HS2PAIRMARKEREND'
F = np.float32


def b64(text, dtype=np.float32):
    return np.frombuffer(base64.b64decode(text), dtype)


class ProbeWindow:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.static = json.loads((self.root / 'static.json').read_text(encoding='utf-8'))
        self.frames = []
        for path in sorted((self.root / 'frames').glob('frame_*.json')):
            row = json.loads(path.read_text(encoding='utf-8'))
            t = b64(row['transforms_f32']).reshape(-1, 33)
            if len(t) != len(self.static['transforms']):
                raise g.SkinPathError('Transform count differs from static layout')
            row['T'] = t
            row['W'] = {r['id']: b64(r['weights_f32']) for r in row['renderers']}
            self.frames.append(row)
        self.by_frame = {f['frame_count']: f for f in self.frames}
        self.meshes = {}
        for r in self.static['renderers']:
            m = r.get('mesh') or {}
            if 'data_file' in m:
                self.meshes[r['id']] = self._mesh(self.root / m['data_file'])

    @staticmethod
    def _mesh(path):
        d = json.loads(path.read_text(encoding='utf-8'))
        n = d['vertex_count']
        out = {'name': d['name'], 'id': d['id'], 'n': n,
               'v': b64(d['vertices_f32']).reshape(n, 3), 'nrm': b64(d['normals_f32']).reshape(n, 3),
               't': b64(d['tangents_f32']).reshape(n, 4), 'uv': b64(d['uv_f32']).reshape(n, 2),
               'uv2': b64(d['uv2_f32']).reshape(-1, 2), 'tri': b64(d['triangles_i32'], np.int32),
               'bind': b64(d['bindposes_f32_rowmajor']).reshape(-1, 4, 4),
               'w': b64(d['bone_weights_f32']).reshape(n, 4), 'bi': b64(d['bone_indices_i32'], np.int32).reshape(n, 4),
               'shapes': []}
        for s in d['blendshapes']:
            out['shapes'].append({'name': s['name'], 'frames': [
                {'weight': fr['weight'], 'dv': b64(fr['dv_f32']).reshape(n, 3), 'dn': b64(fr['dn_f32']).reshape(n, 3),
                 'dt': b64(fr['dt_f32']).reshape(n, 3)} for fr in s['frames']]})
        return out

    def surfaces(self):
        """Rows compatible with gpu_skin_inputs.bind_surfaces."""
        rows = []
        for r in self.static['renderers']:
            if r['id'] not in self.meshes:
                continue
            m = self.meshes[r['id']]
            rows.append({'renderer_id': r['id'], 'mesh_name': m['name'], 'renderer_path': r['name'] + '#' + str(r['id']),
                         'vertex_count': m['n'], 'active_in_hierarchy': None,
                         'source': {'triangles': m['tri'].tolist(), 'uv': m['uv'], 'uv2': m['uv2'] if len(m['uv2']) else [],
                                    'bone_weights': m['w'], 'bone_indices': m['bi']}})
        return rows


def find_markers(index: g.CaptureIndex, nonce: bytes):
    """Every in-frame data blob carrying the probe marker, with its chunk."""
    rows = []
    for c in index.chunks:
        if c.index <= index.begin.index:
            continue
        for ref in c.buffers:
            if ref.length < 64:
                continue
            data = index.blob(ref)
            at = data.find(MAGIC)
            while at >= 0:
                rec = data[at:at + 64]
                if len(rec) == 64 and rec[48:64] == TAIL:
                    frame = int.from_bytes(rec[32:36], 'little', signed=True)
                    if int.from_bytes(rec[44:48], 'little', signed=True) != ~frame:
                        raise g.SkinPathError('Marker integrity word mismatch')
                    rows.append({'chunk_index': c.index, 'chunk_name': c.name, 'nonce_ok': rec[16:32] == nonce,
                                 'frame_count': frame, 'window_index': int.from_bytes(rec[36:40], 'little', signed=True),
                                 'rendered_frame_count': int.from_bytes(rec[40:44], 'little', signed=True)})
                at = data.find(MAGIC, at + 1)
    return rows


# ---------------------------------------------------------------- arithmetic

def blend_f64(mesh, weights):
    v = mesh['v'].astype(float).copy()
    n = mesh['nrm'].astype(float).copy()
    t = mesh['t'][:, :3].astype(float).copy()
    for s, w in zip(mesh['shapes'], weights):
        if w == 0:
            continue
        frames = s['frames']
        if len(frames) != 1:
            raise g.SkinPathError('Multi-frame blendshape interpolation requires implementation: ' + s['name'])
        k = float(w) / float(frames[0]['weight'])
        v += k * frames[0]['dv']
        n += k * frames[0]['dn']
        t += k * frames[0]['dt']
    return v, n, t


def blend_f32(mesh, weights):
    """Sequential float32 accumulation in shape index order (hypothesis)."""
    v = mesh['v'].astype(F).copy()
    n = mesh['nrm'].astype(F).copy()
    t = mesh['t'][:, :3].astype(F).copy()
    for s, w in zip(mesh['shapes'], weights):
        if w == 0:
            continue
        frames = s['frames']
        if len(frames) != 1:
            raise g.SkinPathError('Multi-frame blendshape interpolation requires implementation: ' + s['name'])
        k = F(w) / F(frames[0]['weight'])
        v = (v + frames[0]['dv'] * k).astype(F)
        n = (n + frames[0]['dn'] * k).astype(F)
        t = (t + frames[0]['dt'] * k).astype(F)
    return v, n, t


def matmul_f32(a, b):
    """4x4 float32 product, k summed left to right without fused multiply-add."""
    a = np.asarray(a, F)
    b = np.asarray(b, F)
    out = (a[..., :, 0:1] * b[..., 0:1, :]).astype(F)
    for k in range(1, 4):
        out = (out + (a[..., :, k:k + 1] * b[..., k:k + 1, :]).astype(F)).astype(F)
    return out


def trs_inverse_f32(position, rotation_xyzw):
    """Inverse of TRS(p, q, 1) in float32: R(q*) and -(R(q*) p)."""
    x, y, z, w = (F(c) for c in rotation_xyzw)
    R = g.unity_rotation_matrix([-x, -y, -z, w])
    p = np.asarray(position, F)
    out = np.eye(4, dtype=F)
    out[:3, :3] = R
    for r in range(3):
        acc = F(R[r, 0] * p[0])
        acc = F(acc + F(R[r, 1] * p[1]))
        acc = F(acc + F(R[r, 2] * p[2]))
        out[r, 3] = -acc
    return out


def frame_matrices(win: ProbeWindow, frame, renderer):
    T = frame['T']
    l2w = T[:, 17:33].reshape(-1, 4, 4)
    root = renderer['root_bone']
    return {'bones_l2w': np.stack([l2w[i] for i in renderer['bones']]).astype(F),
            'root_pos': T[root, 10:13].astype(F), 'root_rot': T[root, 13:17].astype(F),
            'root_l2w': l2w[root].astype(F)}


def skin_variants(mesh, mats, weights):
    """Return {name: 40-byte vertex payload} for the declared evaluation orders."""
    out = {}
    w = mesh['w']
    bi = mesh['bi']
    safe = np.where(w > 0, bi, 0)
    hand = mesh['t'][:, 3].astype(F)
    # float64 reference (generator path)
    root = g.root_frame(_trs(mats['root_pos'], mats['root_rot']), mats['root_rot'])
    v, n, t = blend_f64(mesh, weights)
    S = np.stack([mats['bones_l2w'][k].astype(float) @ mesh['bind'][k].astype(float) for k in range(len(mesh['bind']))])
    world = np.einsum('vkij,vj,vk->vi', S[safe], np.c_[v, np.ones(len(v))], w)[:, :3]
    wn = np.einsum('vkij,vj,vk->vi', S[safe][..., :3, :3], n, w)
    wt = np.einsum('vkij,vj,vk->vi', S[safe][..., :3, :3], t, w)
    out['f64_reference'] = g.skinned_vb_bytes(world, wn, wt, hand, root)
    # float32 orders
    rinv = trs_inverse_f32(mats['root_pos'], mats['root_rot'])
    v32, n32, t32 = blend_f32(mesh, weights)
    pal_a = matmul_f32(matmul_f32(rinv, mats['bones_l2w']), mesh['bind'])   # (Rinv L2W) B
    pal_b = matmul_f32(rinv, matmul_f32(mats['bones_l2w'], mesh['bind']))   # Rinv (L2W B)
    for tag, pal in (('A', pal_a), ('B', pal_b)):
        P = pal[safe]                                   # (n,4,4,4)
        wv = w.astype(F)[..., None, None]
        M = (P[:, 0] * wv[:, 0]).astype(F)
        for k in range(1, 4):
            M = (M + (P[:, k] * wv[:, k]).astype(F)).astype(F)
        out[f'f32_palette{tag}_blend_matrix'] = _apply(M, v32, n32, t32, hand)
        acc_p = acc_n = acc_t = None
        for k in range(4):
            pk, nk, tk = _transform(P[:, k], v32, n32, t32)
            wk = w[:, k:k + 1].astype(F)
            pk, nk, tk = (pk * wk).astype(F), (nk * wk).astype(F), (tk * wk).astype(F)
            acc_p = pk if acc_p is None else (acc_p + pk).astype(F)
            acc_n = nk if acc_n is None else (acc_n + nk).astype(F)
            acc_t = tk if acc_t is None else (acc_t + tk).astype(F)
        out[f'f32_palette{tag}_blend_vertices'] = np.concatenate([acc_p, acc_n, acc_t, hand[:, None]], 1).astype(F).tobytes()
    return out


def _trs(p, q):
    M = np.eye(4)
    M[:3, :3] = g.unity_rotation_matrix(q)
    M[:3, 3] = np.asarray(p, float)
    return M


def _transform(M, v, n, t):
    def lin(x, with_t):
        acc = (M[:, :3, 0] * x[:, 0:1]).astype(F)
        acc = (acc + (M[:, :3, 1] * x[:, 1:2]).astype(F)).astype(F)
        acc = (acc + (M[:, :3, 2] * x[:, 2:3]).astype(F)).astype(F)
        return (acc + M[:, :3, 3]).astype(F) if with_t else acc
    return lin(v, True), lin(n, False), lin(t, False)


def _apply(M, v, n, t, hand):
    p, nn, tt = _transform(M, v, n, t)
    return np.concatenate([p, nn, tt, hand[:, None]], 1).astype(F).tobytes()


def compare_words(original: bytes, generated: bytes):
    a = np.frombuffer(original, np.uint32)
    b = np.frombuffer(generated, np.uint32)
    fa = a.view(np.float32).astype(float)
    fb = b.view(np.float32).astype(float)
    return {'bit_exact': original == generated, 'equal_words': int((a == b).sum()), 'words': int(a.size),
            'max_abs_diff': float(np.abs(fa - fb).max()),
            'max_ulp': int(np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64)).max())}


# ---------------------------------------------------------------- Unity-exact path

def chain_entries(win: ProbeWindow, frame, bones, ancestors):
    """Prepared-renderer hierarchy entries (parents first) for the given bones.

    ``ancestors``: local (t, q, s) of the transforms above the exported
    ChaControl root, top first (from the same session's bridge export).
    """
    tf = win.static['transforms']
    T = frame['T']
    order, pos = [], {}

    def add(i):
        if i in pos:
            return
        p = tf[i]['parent']
        if p >= 0:
            add(p)
        order.append(i)
        pos[i] = len(order)
    for b in bones:
        add(b)
    entries = [(k, t, q, s) for k, (t, q, s) in enumerate(ancestors)]
    shift = len(ancestors)
    for i in order:
        p = tf[i]['parent']
        entries.append(((pos[p] + shift) if p >= 0 else shift, T[i, 0:3], T[i, 3:7], T[i, 7:10]))
    return entries, {i: pos[i] + shift for i in order}


def unity_exact_vertices(emu, win: ProbeWindow, frame, renderer, mesh, ancestors):
    """Original UnityPlayer arithmetic: root W2L-no-scale, prepared TRS chain,
    MultiplyMatrix bindposes, float32 blendshapes, SkinVerticesGenericCPU<4,1,1>."""
    from .unity_skin_emu import prepared_palette
    mats = frame_matrices(win, frame, renderer)
    base, _ = emu.world_to_local_noscale(mats['root_pos'], mats['root_rot'])
    entries, where = chain_entries(win, frame, renderer['bones'], ancestors)
    palette = prepared_palette(emu, base, entries, [where[b] for b in renderer['bones']], list(mesh['bind']))
    src = unity_blend(mesh, frame['W'][renderer['id']])
    return emu.skin_vertices4(src, palette, mesh['w'], mesh['bi'])


def unity_object_matrices(emu, win: ProbeWindow, frame, renderer):
    """unity_ObjectToWorld = TRS(root.position, root.rotation, 1) (float32,
    UnityCsReference order) and unity_WorldToObject = InvertMatrix4x4_General3D."""
    from .unity_skin_emu import invert_general3d
    mats = frame_matrices(win, frame, renderer)
    o2w = np.eye(4, dtype=F)
    o2w[:3, :3] = g.unity_rotation_matrix(mats['root_rot'])
    o2w[:3, 3] = mats['root_pos']
    return g.matrix_bytes(o2w), g.matrix_bytes(invert_general3d(emu, o2w))


def unity_blend(mesh, weights):
    """Native-exact ApplyBlendShapes (src.render.unity_blendshape) on the mesh stream."""
    from .unity_blendshape import BlendShapeData, apply_blendshapes
    base = np.concatenate([mesh['v'], mesh['nrm'], mesh['t']], 1).astype(F)
    if not mesh['shapes']:
        return base
    data = mesh.get('_bsd')
    if data is None:
        data = BlendShapeData.from_dense_frames([[(f['weight'], f['dv'], f['dn'], f['dt']) for f in s['frames']]
                                                 for s in mesh['shapes']])
        mesh['_bsd'] = data
    return apply_blendshapes(data, weights, base)
