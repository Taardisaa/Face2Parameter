"""Unity 2018.4.11f1 CPU blendshape application, float32-exact.

Python port of the installed UnityPlayer.dll routines (RVAs, matched PDB):
  ApplyBlendShapes(SkinMeshInfo&, UInt8* out)              0x78fab0
  CalculateBlendShapeWeights(dynamic_array<FrameWeight>&,
      const BlendShapeData&, int channels, const float*, float min, float max)  0x7901c0
  ApplyBlendShapeTmpl<normals, tangents>(...)              0x78ce80 / 0x78d330 / 0x78d690
plus a Unicorn run of the original ApplyBlendShapes for cross-checking
(dynamic_array reserve/grow and the destructor are serviced by Python hooks).

Per-frame weights are clamped to [min, max] = SkinnedMeshRendererManager
0x1158/0x115c, set each frame by UpdateAllSkinnedMeshes from
PlayerSettings.legacyClampBlendShapeWeights (installed build: True -> [0, 1]).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

F = np.float32
EPS = F(9.999999974752427e-07)       # [0x11d0148]
ONE = F(1.0)                          # [0x11b6da4]
NEAR_ONE = F(0.9999989867210388)      # [0x11d5190]
LEGACY_CLAMP = (F(0.0), F(1.0))      # legacyClampBlendShapeWeights = True


@dataclass
class BlendShapeData:
    """Runtime BlendShapeData: sparse vertices, frames (shapes), channels, fullWeights."""
    index: np.ndarray                    # (m,) uint32
    dv: np.ndarray                       # (m,3) float32
    dn: np.ndarray
    dt: np.ndarray
    shapes: list = field(default_factory=list)    # (firstVertex, vertexCount, hasNormals, hasTangents)
    channels: list = field(default_factory=list)  # (frameIndex, frameCount)
    full_weights: np.ndarray = None

    @classmethod
    def from_dense_frames(cls, frames_by_channel):
        """frames_by_channel: [[(weight, dv(n,3), dn, dt), ...] per channel] (API dense frames).

        Sparse rows keep every vertex whose 9 delta words are not all +0.0 bits
        (GetBlendShapeFrameVertices expands absent rows to +0.0); a frame's
        hasNormals/hasTangents flag is set when any of its normal/tangent deltas
        is nonzero. Rows/flags whose deltas are all +0.0 only differ from the
        asset where a source component is -0.0 (documented limitation).
        """
        idx, dv, dn, dt, shapes, channels, weights = [], [], [], [], [], [], []
        for frames in frames_by_channel:
            channels.append((len(shapes), len(frames)))
            for w, a, b, c in frames:
                a, b, c = (np.asarray(x, F) for x in (a, b, c))
                rows = np.nonzero(np.any(np.concatenate([a, b, c], 1).view(np.uint32) != 0, 1))[0]
                shapes.append((sum(len(i) for i in idx), len(rows), bool(np.any(b[rows] != 0)), bool(np.any(c[rows] != 0))))
                idx.append(rows.astype(np.uint32)); dv.append(a[rows]); dn.append(b[rows]); dt.append(c[rows])
                weights.append(F(w))
        cat = lambda xs, shape: np.concatenate(xs).astype(F) if xs and sum(len(x) for x in xs) else np.zeros(shape, F)  # noqa: E731
        return cls(np.concatenate(idx).astype(np.uint32) if idx else np.zeros(0, np.uint32),
                   cat(dv, (0, 3)), cat(dn, (0, 3)), cat(dt, (0, 3)), shapes, channels, np.asarray(weights, F))


def calculate_weights(data: BlendShapeData, channel_weights, wmin=LEGACY_CLAMP[0], wmax=LEGACY_CLAMP[1]):
    """Port of CalculateBlendShapeWeights: list of (frame index, weight)."""
    out = []
    fw = data.full_weights
    for (frame_index, frame_count), w in zip(data.channels, np.asarray(channel_weights, F)):
        w = w if wmin <= w else wmin               # comiss min,[w]; cmovbe -> max(w, min)
        f0 = fw[frame_index]
        if f0 > w or frame_count == 1:
            t = F(w / f0)
            x = wmin if wmin > t else (wmax if wmax < t else t)   # minss(max, t)
            if abs(float(x)) > float(EPS):
                out.append((frame_index, x))
            continue
        j = 0
        while j + 1 < frame_count - 1 and w > fw[frame_index + j + 1]:
            j += 1
        a, b = fw[frame_index + j], fw[frame_index + j + 1]
        t = F(F(w - a) / F(b - a))
        if NEAR_ONE > t:
            out.append((frame_index + j, F(ONE - t)))
        if t > EPS:
            out.append((frame_index + j + 1, t if t <= wmax else wmax))
    return out


def apply_blendshapes(data: BlendShapeData, channel_weights, base, skin_normals=True, skin_tangents=True,
                      wmin=LEGACY_CLAMP[0], wmax=LEGACY_CLAMP[1]):
    """Port of ApplyBlendShapes on a (n,10) float32 pos/normal/tangent4 stream."""
    out = np.array(base, F, copy=True)
    for frame, w in calculate_weights(data, channel_weights, wmin, wmax):
        first, count, has_n, has_t = data.shapes[frame]
        rows = slice(first, first + count)
        idx = data.index[rows]
        normals = skin_normals and has_n
        tangents = normals and skin_tangents and has_t
        out[idx, 0:3] = (out[idx, 0:3] + (data.dv[rows] * w).astype(F)).astype(F)
        if normals:
            out[idx, 3:6] = (out[idx, 3:6] + (data.dn[rows] * w).astype(F)).astype(F)
        if tangents:
            toff = 6 if has_n else 3
            out[idx, toff:toff + 3] = (out[idx, toff:toff + 3] + (data.dt[rows] * w).astype(F)).astype(F)
    return out


# ---------------------------------------------------------------- native emulation

APPLY_BLENDSHAPES = 0x78fab0
RESERVE = 0x877530        # dynamic_array_data::reserve(size_t count, size_t elem, size_t align)
GROW = 0x1b6100           # dynamic_array<8-byte>::grow()
DTOR = 0x9fec0            # dynamic_array destructor (folded)


def emulate_apply_blendshapes(emu, data: BlendShapeData, channel_weights, base, skin_normals=True, skin_tangents=True,
                              wmin=LEGACY_CLAMP[0], wmax=LEGACY_CLAMP[1]):
    from unicorn import UC_HOOK_CODE, x86_const as x
    uc = emu.uc
    emu.reset_heap()
    base = np.ascontiguousarray(base, F)
    n = len(base)
    vb = np.zeros((len(data.index), 10), np.uint32)
    vb[:, 0] = data.index
    vb[:, 1:4] = data.dv.view(np.uint32)
    vb[:, 4:7] = data.dn.view(np.uint32)
    vb[:, 7:10] = data.dt.view(np.uint32)
    p_verts = emu.alloc(vb.tobytes() + bytes(64))
    shapes = b''.join(struct.pack('<IIBB2x', a, b, int(c), int(d)) for a, b, c, d in data.shapes)
    p_shapes = emu.alloc(shapes + bytes(16))
    p_channels = emu.alloc(b''.join(struct.pack('<QiiiI', 0, 0, fi, fc, 0) for fi, fc in data.channels) + bytes(16))
    p_fw = emu.alloc(np.asarray(data.full_weights, F).tobytes() + bytes(16))
    bsd = bytearray(0x80)
    struct.pack_into('<QQQ', bsd, 0x00, p_verts, 1, len(data.index))
    struct.pack_into('<QQQQ', bsd, 0x20, p_shapes, 1, len(data.shapes), len(data.shapes) << 1)
    struct.pack_into('<QQQ', bsd, 0x40, p_channels, p_channels + 24 * len(data.channels), p_channels + 24 * len(data.channels))
    struct.pack_into('<QQQ', bsd, 0x58, p_fw, 1, len(data.full_weights))
    p_bsd = emu.alloc(bytes(bsd))
    p_src = emu.alloc(base.tobytes() + bytes(64))
    p_w = emu.alloc(np.asarray(channel_weights, F).tobytes() + bytes(16))
    info = bytearray(0x100)
    struct.pack_into('<Q', info, 0x20, p_src)
    struct.pack_into('<i', info, 0x30, 40)
    struct.pack_into('<i', info, 0x48, n)
    info[0x4c] = int(bool(skin_normals))
    info[0x4d] = int(bool(skin_tangents))
    struct.pack_into('<Qi', info, 0x60, p_w, len(data.channels))
    struct.pack_into('<Q', info, 0x70, p_bsd)
    struct.pack_into('<ff', info, 0x78, float(wmin), float(wmax))
    p_info = emu.alloc(bytes(info))
    p_out = emu.alloc(bytes(base.nbytes + 64))
    state = {'arrays': 0}

    def ret(value=None):
        rsp = uc.reg_read(x.UC_X86_REG_RSP)
        target = struct.unpack('<Q', bytes(uc.mem_read(rsp, 8)))[0]
        uc.reg_write(x.UC_X86_REG_RSP, rsp + 8)
        if value is not None:
            uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RIP, target)

    def regrow(this, count, elem):
        ptr, label, size, cap = struct.unpack('<QQQQ', bytes(uc.mem_read(this, 32)))
        if count > (cap >> 1):
            new = emu.alloc(bytes(count * elem + 16))
            if ptr and size:
                uc.mem_write(new, bytes(uc.mem_read(ptr, size * elem)))
            uc.mem_write(this, struct.pack('<Q', new))
            uc.mem_write(this + 0x18, struct.pack('<Q', count << 1))
            state['arrays'] += 1

    def hook(uc_, address, size, user):
        rel = address - emu.base
        if rel == RESERVE:
            regrow(uc.reg_read(x.UC_X86_REG_RCX), uc.reg_read(x.UC_X86_REG_RDX), uc.reg_read(x.UC_X86_REG_R8))
            ret()
        elif rel == GROW:
            this = uc.reg_read(x.UC_X86_REG_RCX)
            cap = struct.unpack('<Q', bytes(uc.mem_read(this + 0x18, 8)))[0] >> 1
            regrow(this, max(1, cap * 2), 8)
            ret()
        elif rel == DTOR:
            ret()

    handles = [uc.hook_add(UC_HOOK_CODE, hook, begin=emu.base + a, end=emu.base + a) for a in (RESERVE, GROW, DTOR)]
    try:
        emu.call(APPLY_BLENDSHAPES, (p_info, p_out))
    finally:
        for h in handles:
            uc.hook_del(h)
    return np.frombuffer(emu.read(p_out, base.nbytes), F).reshape(n, 10).copy()
