"""Float32-exact reimplementation of UnityPlayer 2018.4.11f1 matrix routines.

Every routine below is transcribed from the disassembly of the installed
``E:/HoneySelect2_ArcticFox/UnityPlayer.dll`` (SHA-256 ``5034cdc2…``, CodeView
GUID ``BFA91A8C44734BB4AABC5A975B304CFE`` age 1, matched to Unity's public
symbol-server PDB).  Disassembly is reproduced with
``tools/offline_renderer/unityplayer_disasm.py``; the RVA of each source is
given in the docstring.  Operation order and operand grouping follow the
machine code; SSE scalar/packed single operations are IEEE binary32 with
round-to-nearest, reproduced with numpy ``float32`` scalars (no FMA exists in
the recovered code).  Two exceptions are explicit:

* ``cosf``/``sinf`` are CRT calls inside UnityPlayer (RVA 0x1171520/0x1174df0);
  they are executed from the installed binary by ``unity_crt``.
* Matrices use Unity's memory layout: 16 floats, column-major
  (``m[col*4 + row]``), which is also the byte layout of a ``float4x4`` in the
  compiled HLSL constant buffers (column_major packing).
"""
from __future__ import annotations

import math
import struct

import numpy as np

F = np.float32
ZERO = F(0.0)
ONE = F(1.0)
HALF = F(0.5)
PI_F32 = F(3.14159274)          # __real@40490fdb
DEG_360 = F(360.0)              # __real@43b40000


def f32(x):
    return F(x)


def _crt_cosf(x):
    """Installed CRT cosf executed by ``unity_crt`` (Unicorn, original instructions)."""
    from .unity_crt import cosf
    return cosf(x)


def _crt_sinf(x):
    from .unity_crt import sinf
    return sinf(x)


def identity():
    m = [ZERO] * 16
    m[0] = m[5] = m[10] = m[15] = ONE
    return m


def as_f32_list(values):
    vals = [F(v) for v in values]
    if len(vals) != 16:
        raise ValueError('Matrix4x4f requires 16 floats')
    return vals


def to_bytes(m):
    return struct.pack('<%df' % len(m), *[float(v) for v in m])


def from_bytes(data):
    return [F(v) for v in struct.unpack('<%df' % (len(data) // 4), data)]


def to_numpy(m):
    """Row-major 4x4 numpy view (for inspection only, not used in exact paths)."""
    return np.array([float(v) for v in m], dtype=np.float64).reshape(4, 4).T


# ------------------------------------------------------------- constructors
def set_perspective(fovy, aspect, znear, zfar):
    """``Matrix4x4f::SetPerspective`` (RVA 0x7d0b90)."""
    fovy, aspect, znear, zfar = F(fovy), F(aspect), F(znear), F(zfar)
    a = (fovy * HALF) / DEG_360
    a = a + a
    a = a * PI_F32
    cotan = _crt_cosf(a) / _crt_sinf(a)
    m = [ZERO] * 16
    m[5] = cotan
    m[0] = cotan / aspect
    m[10] = (znear + zfar) / (znear - zfar)
    m[14] = ((znear + znear) * zfar) / (znear - zfar)
    m[11] = F(-1.0)
    return m


def set_ortho(left, right, bottom, top, znear, zfar):
    """``Matrix4x4f::SetOrtho`` (RVA 0x7d0a70)."""
    l, r, b, t, n, f = (F(v) for v in (left, right, bottom, top, znear, zfar))
    m = [ZERO] * 16
    m[15] = ONE
    m[0] = F(2.0) / (r - l)
    m[5] = F(2.0) / (t - b)
    m[10] = F(-2.0) / (f - n)
    m[12] = -((l + r) / (r - l))
    m[13] = -((b + t) / (t - b))
    m[14] = -((n + f) / (f - n))
    return m


def set_scale(sx, sy, sz):
    """``Matrix4x4f::SetScale``: diagonal (sx, sy, sz, 1)."""
    m = [ZERO] * 16
    m[0], m[5], m[10], m[15] = F(sx), F(sy), F(sz), ONE
    return m


# ------------------------------------------------------------- operations
def multiply(a, b):
    """``MultiplyMatrices4x4(a, b, out)`` (RVA 0xe48d0): out = a·b.

    Column j of out is ((b[j].x·a.c0 + b[j].y·a.c1) + b[j].z·a.c2) + b[j].w·a.c3
    with every product rounded (mulps) before the left-to-right addps chain.
    """
    out = [ZERO] * 16
    for j in range(4):
        bx, by, bz, bw = b[4 * j], b[4 * j + 1], b[4 * j + 2], b[4 * j + 3]
        for i in range(4):
            out[4 * j + i] = ((by * a[4 + i] + bx * a[i]) + bz * a[8 + i]) + bw * a[12 + i]
    return out


def calculate_device_projection(m, uses_opengl_texture_coords=False, invert_y=False, reversed_z=True):
    """``GfxDevice::CalculateDeviceProjectionMatrix`` (RVA 0x8ca400).

    D3D11 path (``uses_opengl_texture_coords`` false): optional Y-row negate,
    then row2 = k·row2 + 0.5·row3 with k = -0.5 (reversed Z) or 0.5.
    """
    m = list(m)
    if uses_opengl_texture_coords:
        if reversed_z:
            for c in range(4):
                m[2 + 4 * c] = -m[2 + 4 * c]
        return m
    if invert_y:
        for c in range(4):
            m[1 + 4 * c] = -m[1 + 4 * c]
    k = F(-0.5) if reversed_z else HALF
    for c in range(4):
        m[2 + 4 * c] = k * m[2 + 4 * c] + m[3 + 4 * c] * HALF
    return m


def gl_get_gpu_projection(m, render_into_texture, uses_opengl_texture_coords=False, reversed_z=True):
    """``GLGetGPUProjectionMatrix`` (RVA 0x704540): invertY = !usesOpenGLTextureCoords && renderIntoTexture."""
    return calculate_device_projection(m, uses_opengl_texture_coords,
                                       (not uses_opengl_texture_coords) and bool(render_into_texture), reversed_z)


def invert_general_3d(m):
    """``InvertMatrix4x4_General3D`` (RVA 0x7cdec0): Mesa ``invert_matrix_3d_general``.

    Returns (ok, out).  Singular when (double)(det·det) < 1e-25; out is then zero.
    """
    pos = ZERO
    neg = ZERO

    def acc(t):
        nonlocal pos, neg
        if t >= ZERO:          # comiss/jb: NaN goes to the negative accumulator
            pos = pos + t
        else:
            neg = neg + t

    acc((m[0] * m[5]) * m[10])
    acc((m[1] * m[6]) * m[8])
    acc((m[2] * m[4]) * m[9])
    acc(-((m[2] * m[5]) * m[8]))
    acc(-((m[4] * m[1]) * m[10]))
    acc(-((m[6] * m[0]) * m[9]))
    det = neg + pos
    # cvtps2pd(det*det); comisd 1e-25, x; jbe continue -> singular only when
    # 1e-25 > x and ordered (NaN takes jbe and continues).
    if float(det * det) < 1e-25:
        return False, [ZERO] * 16
    inv = ONE / det
    out = [ZERO] * 16
    out[0] = (m[10] * m[5] - m[9] * m[6]) * inv
    out[4] = -((m[10] * m[4] - m[6] * m[8]) * inv)
    out[8] = (m[9] * m[4] - m[8] * m[5]) * inv
    out[1] = -((m[10] * m[1] - m[9] * m[2]) * inv)
    out[5] = (m[10] * m[0] - m[2] * m[8]) * inv
    out[9] = -((m[9] * m[0] - m[1] * m[8]) * inv)
    out[2] = (m[1] * m[6] - m[2] * m[5]) * inv
    out[6] = -((m[6] * m[0] - m[2] * m[4]) * inv)
    out[10] = (m[0] * m[5] - m[1] * m[4]) * inv
    out[12] = -((out[4] * m[13] + out[0] * m[12]) + out[8] * m[14])
    out[13] = -((out[1] * m[12] + out[5] * m[13]) + out[9] * m[14])
    out[14] = -((out[6] * m[13] + out[2] * m[12]) + out[10] * m[14])
    out[3] = out[7] = out[11] = ZERO
    out[15] = ONE
    return True, out


def invert_full(m):
    """``InvertMatrix4x4_Full`` (RVA 0x7cd530): Mesa ``invert_matrix_general``
    (Gauss-Jordan, partial pivoting on |x| with strict '>' swaps, zero-skip of
    the identity half).  Returns (ok, out); out is zero when singular."""
    def MAT(r, c):
        return m[c * 4 + r]
    rows = []
    for r in range(4):
        row = [MAT(r, 0), MAT(r, 1), MAT(r, 2), MAT(r, 3), ZERO, ZERO, ZERO, ZERO]
        row[4 + r] = ONE
        rows.append(row)
    r0, r1, r2, r3 = rows
    fail = (False, [ZERO] * 16)
    if abs(r3[0]) > abs(r2[0]):
        r3, r2 = r2, r3
    if abs(r2[0]) > abs(r1[0]):
        r2, r1 = r1, r2
    if abs(r1[0]) > abs(r0[0]):
        r1, r0 = r0, r1
    if r0[0] == ZERO:
        return fail
    m1 = r1[0] / r0[0]
    m2 = r2[0] / r0[0]
    m3 = r3[0] / r0[0]
    for k in (1, 2, 3):
        s = r0[k]
        r1[k] = r1[k] - m1 * s
        r2[k] = r2[k] - m2 * s
        r3[k] = r3[k] - m3 * s
    for k in (4, 5, 6, 7):
        s = r0[k]
        if s != ZERO:
            r1[k] = r1[k] - m1 * s
            r2[k] = r2[k] - m2 * s
            r3[k] = r3[k] - m3 * s
    if abs(r3[1]) > abs(r2[1]):
        r3, r2 = r2, r3
    if abs(r2[1]) > abs(r1[1]):
        r2, r1 = r1, r2
    if r1[1] == ZERO:
        return fail
    m2 = r2[1] / r1[1]
    m3 = r3[1] / r1[1]
    for k in (2, 3):
        r2[k] = r2[k] - m2 * r1[k]
        r3[k] = r3[k] - m3 * r1[k]
    for k in (4, 5, 6, 7):
        s = r1[k]
        if s != ZERO:
            r2[k] = r2[k] - m2 * s
            r3[k] = r3[k] - m3 * s
    if abs(r3[2]) > abs(r2[2]):
        r3, r2 = r2, r3
    if r2[2] == ZERO:
        return fail
    m3 = r3[2] / r2[2]
    for k in (3, 4, 5, 6, 7):
        r3[k] = r3[k] - m3 * r2[k]
    if r3[3] == ZERO:
        return fail
    s = ONE / r3[3]
    for k in (4, 5, 6, 7):
        r3[k] = r3[k] * s
    m2 = r2[3]
    s = ONE / r2[2]
    for k in (4, 5, 6, 7):
        r2[k] = (r2[k] - r3[k] * m2) * s
    m1 = r1[3]
    for k in (4, 5, 6, 7):
        r1[k] = r1[k] - r3[k] * m1
    m0 = r0[3]
    for k in (4, 5, 6, 7):
        r0[k] = r0[k] - r3[k] * m0
    m1 = r1[2]
    s = ONE / r1[1]
    for k in (4, 5, 6, 7):
        r1[k] = (r1[k] - r2[k] * m1) * s
    m0 = r0[2]
    for k in (4, 5, 6, 7):
        r0[k] = r0[k] - r2[k] * m0
    m0 = r0[1]
    s = ONE / r0[0]
    for k in (4, 5, 6, 7):
        r0[k] = (r0[k] - r1[k] * m0) * s
    out = [ZERO] * 16
    for c in range(4):
        out[c * 4 + 0] = r0[4 + c]
        out[c * 4 + 1] = r1[4 + c]
        out[c * 4 + 2] = r2[4 + c]
        out[c * 4 + 3] = r3[4 + c]
    return True, out


def transform_point(m, p):
    """Matrix4x4f::MultiplyPoint3 order is not yet recovered; used only for diagnostics."""
    x, y, z = (F(v) for v in p)
    return [((m[i] * x + m[4 + i] * y) + m[8 + i] * z) + m[12 + i] for i in range(3)]
