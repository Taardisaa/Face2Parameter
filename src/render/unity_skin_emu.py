"""Execute the installed UnityPlayer.dll skinning machine code under emulation.

Unity 2018.4.11f1's CPU skinning is closed source. Its exact float32 behaviour
is taken from the installed binary itself: the matched PDB (GUID
BFA91A8C44734BB4AABC5A975B304CFE, age 1) names the routines, and Unicorn runs
their original instructions on supplied inputs. Nothing is re-derived or fitted.

Routines (RVAs of UnityPlayer.dll SHA 5034cdc2…):
  SkinVerticesGenericCPU<4,1,1>        0x776730  per-vertex palette blend + transform
  MultiplyMatrixArrayWithBase4x4        0xbdd000  res[i] = base · (a[i] · b[i])
  CalculateInverseGlobalMatrixNoScale   0xbfdd40  (tail from 0xbfdd73, after the
                                                   global position/rotation call)
  GetWorldToLocalMatrixNoScale packing  0xbff997  affineX → Matrix4x4f
  CalculateSkinningMatrices prepared    0xbda7ae–0xbdabde (hierarchical TRS chain)
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

DLL = Path('E:/HoneySelect2_ArcticFox/UnityPlayer.dll')
DLL_SHA = '5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923'
RVA = {'skin4nt': 0x776730, 'mul_base': 0xbdd000, 'invglobal_tail': 0xbfdd73, 'invglobal_end': 0xbfde77,
       'w2l_pack': 0xbff997, 'w2l_pack_end': 0xbff9fb,
       'csm_prepared': 0xbda7ae, 'csm_prepared_end': 0xbdabde}
STACK, STACK_SIZE = 0x10000000, 0x200000
HEAP, HEAP_SIZE = 0x20000000, 0x10000000
RET = 0x0DEAD000


class UnityPlayerEmu:
    def __init__(self, dll=DLL):
        import pefile
        from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_PROT_ALL
        data = Path(dll).read_bytes()
        if hashlib.sha256(data).hexdigest() != DLL_SHA:
            raise RuntimeError('Installed UnityPlayer.dll differs from the audited binary')
        pe = pefile.PE(data=data, fast_load=True)
        self.base = pe.OPTIONAL_HEADER.ImageBase
        image = pe.get_memory_mapped_image(ImageBase=self.base)
        size = (len(image) + 0xfff) & ~0xfff
        self.uc = Uc(UC_ARCH_X86, UC_MODE_64)
        self.uc.mem_map(self.base, size, UC_PROT_ALL)
        self.uc.mem_write(self.base, image)
        self.uc.mem_map(STACK, STACK_SIZE, UC_PROT_ALL)
        self.uc.mem_map(HEAP, HEAP_SIZE, UC_PROT_ALL)
        self.uc.mem_map(RET, 0x1000, UC_PROT_ALL)
        self.uc.mem_write(RET, b'\xc3')
        self._heap = HEAP

    # -- memory helpers
    def alloc(self, data: bytes, align=64):
        self._heap = (self._heap + align - 1) & ~(align - 1)
        addr = self._heap
        self.uc.mem_write(addr, data)
        self._heap += max(len(data), 16)
        return addr

    def reset_heap(self):
        self._heap = HEAP

    def read(self, addr, n):
        return bytes(self.uc.mem_read(addr, n))

    def _regs(self):
        from unicorn import x86_const as x
        return x

    def call(self, rva, args, stack_args=()):
        x = self._regs()
        rsp = STACK + STACK_SIZE - 0x1000
        frame = struct.pack('<Q', RET) + b'\0' * 0x20 + b''.join(struct.pack('<Q', a) for a in stack_args)
        rsp -= len(frame) + 8
        rsp &= ~0xf
        rsp -= 8  # call pushes return address: rsp % 16 == 8 at entry
        self.uc.mem_write(rsp, frame)
        for reg, val in zip((x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9), args):
            self.uc.reg_write(reg, val)
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp)
        self.uc.emu_start(self.base + rva, RET)

    # -- routines
    def multiply_with_base(self, base, a, b):
        n = len(a)
        if len(b) != n:
            raise ValueError('Array lengths differ')
        self.reset_heap()
        pb = self.alloc(colmajor(base))
        pa = self.alloc(b''.join(colmajor(m) for m in a))
        pbb = self.alloc(b''.join(colmajor(m) for m in b))
        pr = self.alloc(bytes(64 * n))
        self.call(RVA['mul_base'], (pb, pa, pbb, pr), (n,))
        return [from_colmajor(self.read(pr + 64 * i, 64)) for i in range(n)]

    def skin_vertices4(self, src, palette, weights, indices):
        """src: (n,10) float32 pos/normal/tangent; palette list of 4x4; weights (n,4); indices (n,4)."""
        src = np.ascontiguousarray(src, np.float32)
        n = len(src)
        bw = np.zeros((n, 8), np.uint32)
        bw[:, :4] = np.ascontiguousarray(weights, np.float32).view(np.uint32)
        bw[:, 4:] = np.ascontiguousarray(indices, np.int32).view(np.uint32)
        self.reset_heap()
        ps = self.alloc(src.tobytes() + bytes(0x200))
        pd = self.alloc(bytes(src.nbytes + 64))
        pp = self.alloc(b''.join(colmajor(m) for m in palette))
        pw = self.alloc(bw.tobytes())
        self.call(RVA['skin4nt'], (ps, pd, n, pp), (pw,))
        return self.read(pd, src.nbytes)

    def world_to_local_noscale(self, position, rotation_xyzw):
        """Original arithmetic after CalculateGlobalPositionAndRotation, plus packing."""
        x = self._regs()
        rsp = STACK + STACK_SIZE - 0x4000
        rsp &= ~0xf
        # Function body expects its own frame: rot at [rsp+0x30], pos at [rsp+0x40].
        self.uc.mem_write(rsp + 0x30, np.asarray(rotation_xyzw, np.float32).tobytes())
        self.uc.mem_write(rsp + 0x40, np.asarray(list(position) + [0.0], np.float32).tobytes())
        self.reset_heap()
        out = self.alloc(bytes(64))
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp)
        self.uc.reg_write(x.UC_X86_REG_RBX, out)
        # The callee-saved xmm restores read the caller frame; their values are irrelevant to rbx output.
        self.uc.emu_start(self.base + RVA['invglobal_tail'], self.base + RVA['invglobal_end'])
        affine = self.read(out, 64)
        # Packing exactly as GetWorldToLocalMatrixNoScale (rax -> affine, rdi -> result).
        rsp2 = rsp - 0x200
        res = self.alloc(bytes(64))
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp2)
        self.uc.reg_write(x.UC_X86_REG_RAX, out)
        self.uc.reg_write(x.UC_X86_REG_RDI, res)
        self.uc.mem_write(rsp2 + 0x80, struct.pack('<Q', 0))
        self.uc.emu_start(self.base + RVA['w2l_pack'], self.base + RVA['w2l_pack_end'])
        return from_colmajor(self.read(res, 64)), affine


def colmajor(m):
    return np.asarray(m, np.float32).T.copy().tobytes()


def from_colmajor(b):
    return np.frombuffer(b, np.float32).reshape(4, 4).T.copy()


def prepared_palette(emu: UnityPlayerEmu, base, entries, bone_map, bindposes):
    """Run CalculateSkinningMatrices' prepared-renderer path (0xbda7ae–0xbdabde).

    entries: list of (parent_entry_index, position3, rotation_xyzw, scale3) for
    entry 1..N (entry 0 is the base matrix slot); parents precede children.
    bone_map[k]: entry index of renderer bone k. Returns the skinning palette
    pose[bone_map[k]] · bindpose[k], computed by the original instructions.
    Struct offsets are those dereferenced by the original code:
      info+0x10 -> hierarchy*, hierarchy+0x18 -> TRS records (48 B: t,q,s)
      info+0x20 entry count, +0x24 capacity, +0x30 parent indices,
      info+0x38 TRS indices, +0x40 bone map (uint32 each), info+3 result byte.
    """
    x = emu._regs()
    n = len(entries) + 1
    emu.reset_heap()
    trs = bytearray()
    parents = [0]
    trs_index = [0]
    for i, (parent, t, q, s) in enumerate(entries):
        if not 0 <= parent < i + 1:
            raise ValueError('Parent must precede child')
        trs += np.asarray(list(t) + [0.0], np.float32).tobytes()
        trs += np.asarray(q, np.float32).tobytes()
        trs += np.asarray(list(s) + [0.0], np.float32).tobytes()
        parents.append(parent)
        trs_index.append(i)
    p_trs = emu.alloc(bytes(trs))
    hier = emu.alloc(struct.pack('<4Q', 0, 0, 0, p_trs))
    p_par = emu.alloc(np.asarray(parents, np.uint32).tobytes())
    p_idx = emu.alloc(np.asarray(trs_index, np.uint32).tobytes())
    p_map = emu.alloc(np.asarray(bone_map, np.uint32).tobytes())
    info = bytearray(0x48)
    struct.pack_into('<Q', info, 0x10, hier)
    struct.pack_into('<II', info, 0x20, n, n)
    struct.pack_into('<QQQ', info, 0x30, p_par, p_idx, p_map)
    info[3] = 1
    p_info = emu.alloc(bytes(info))
    p_base = emu.alloc(colmajor(base))
    p_bind = emu.alloc(b''.join(colmajor(m) for m in bindposes))
    p_pose = emu.alloc(bytes(64 * n + 64))
    p_out = emu.alloc(bytes(64 * len(bone_map)))
    rbp = STACK + STACK_SIZE - 0x10000
    rbp &= ~0xf
    emu.uc.mem_write(rbp + 0x160, struct.pack('<Q', p_out))
    regs = {x.UC_X86_REG_RAX: p_pose, x.UC_X86_REG_RSI: p_base, x.UC_X86_REG_RDI: p_info,
            x.UC_X86_REG_RBX: 0, x.UC_X86_REG_R12: 1, x.UC_X86_REG_R13: 0, x.UC_X86_REG_R14: len(bone_map),
            x.UC_X86_REG_R15: p_bind, x.UC_X86_REG_RBP: rbp, x.UC_X86_REG_RSP: rbp - 0x200}
    for reg, val in regs.items():
        emu.uc.reg_write(reg, val)
    emu.uc.emu_start(emu.base + RVA['csm_prepared'], emu.base + RVA['csm_prepared_end'])
    return [from_colmajor(emu.read(p_out + 64 * k, 64)) for k in range(len(bone_map))]


INVERT_GENERAL3D = 0x7cdec0  # InvertMatrix4x4_General3D(const float*, float*) used by GfxDeviceD3D11Base::BeforeDrawCall for unity_WorldToObject


def invert_general3d(emu: UnityPlayerEmu, m):
    emu.reset_heap()
    src = emu.alloc(colmajor(m))
    dst = emu.alloc(bytes(64))
    emu.call(INVERT_GENERAL3D, (src, dst))
    return from_colmajor(emu.read(dst, 64))


GLOBAL_POS_ROT = 0xdda60  # CalculateGlobalPositionAndRotation(TransformAccessReadOnly, float3&, float4&)


def global_position_rotation(emu: UnityPlayerEmu, chain):
    """World position/rotation by the original hierarchy walk.

    chain: local (t, q, s) of the transform, then its parent, ... up to the top.
    Fake TransformHierarchy: +0x18 TRS records (48 B), +0x20 parent indices.
    """
    emu.reset_heap()
    trs = b''.join(np.asarray(list(t) + [0.0], np.float32).tobytes() + np.asarray(q, np.float32).tobytes()
                   + np.asarray(list(s) + [0.0], np.float32).tobytes() for t, q, s in chain)
    parents = list(range(1, len(chain))) + [-1]
    p_trs = emu.alloc(trs)
    p_par = emu.alloc(np.asarray(parents, np.int32).tobytes())
    hier = emu.alloc(struct.pack('<5Q', 0, 0, 0, p_trs, p_par))
    access = emu.alloc(struct.pack('<QI4x', hier, 0))
    p_pos = emu.alloc(bytes(16))
    p_rot = emu.alloc(bytes(16))
    emu.call(GLOBAL_POS_ROT, (access, p_pos, p_rot))
    return (np.frombuffer(emu.read(p_pos, 12), np.float32).copy(),
            np.frombuffer(emu.read(p_rot, 16), np.float32).copy())


def skin_from_locals(emu: UnityPlayerEmu, parent, local, bone_ids, root_id, mesh):
    """Unity-exact skinned VB and per-draw matrices from local TRS only.

    parent: id -> parent id (None at top); local: id -> (t, q, s) float32.
    mesh: dict with v (n,3), nrm (n,3), t (n,4), bind (k,4,4), w (n,4), bi (n,4),
    blend (n,3)x3 already-applied float32 deltas or None (see blend_f32).
    Returns (vertex_bytes, object_to_world_bytes, world_to_object_bytes).
    """
    def chain(i):
        out = []
        while i is not None:
            out.append(local[i])
            i = parent[i]
        return out
    pos, rot = global_position_rotation(emu, chain(root_id))
    base, _ = emu.world_to_local_noscale(pos, rot)
    order, where = [], {}

    def add(i):
        if i in where:
            return
        if parent[i] is not None:
            add(parent[i])
        order.append(i)
        where[i] = len(order)
    for b in bone_ids:
        add(b)
    entries = [((where[parent[i]] if parent[i] is not None else 0),) + tuple(local[i]) for i in order]
    palette = prepared_palette(emu, base, entries, [where[b] for b in bone_ids], list(mesh['bind']))
    src = np.concatenate([mesh['v'], mesh['nrm'], mesh['t'][:, :3], mesh['t'][:, 3:4]], 1).astype(np.float32)
    vb = emu.skin_vertices4(src, palette, mesh['w'], mesh['bi'])
    from .gpu_skin_inputs import unity_rotation_matrix, matrix_bytes
    o2w = np.eye(4, dtype=np.float32)
    o2w[:3, :3] = unity_rotation_matrix(rot)
    o2w[:3, 3] = pos
    return vb, matrix_bytes(o2w), matrix_bytes(invert_general3d(emu, o2w))


FROM_EULER_RAD = 0x9541d0  # Quaternion_CUSTOM_Internal_FromEulerRad_Injected -> EulerToQuaternion(ZXY) with UnityPlayer sinf/cosf
_SHARED = None


def shared_emulator():
    global _SHARED
    if _SHARED is None:
        _SHARED = UnityPlayerEmu()
    return _SHARED


def quaternion_euler_degrees(emu: UnityPlayerEmu, degrees):
    """UnityEngine.Quaternion.Euler: managed `euler * Mathf.Deg2Rad` (float32,
    Deg2Rad = 0.0174532924f), then the native Internal_FromEulerRad."""
    rad = np.asarray(degrees, np.float32) * np.float32(0.0174532924)
    emu.reset_heap()
    src = emu.alloc(rad.astype(np.float32).tobytes() + bytes(4))
    dst = emu.alloc(bytes(16))
    emu.call(FROM_EULER_RAD, (src, dst))
    return np.frombuffer(emu.read(dst, 16), np.float32).copy()


SETLOCALR_NORMALIZE = (0x1b130f, 0x1b1350)  # xmm2 = q in, xmm2 = normalized out


def _consts():
    import pefile
    pe = pefile.PE(str(DLL), fast_load=True)
    img = pe.get_memory_mapped_image()
    eps = np.frombuffer(img[0x1b1334 + 0x101f00c:0x1b1334 + 0x101f00c + 16], np.float32).copy()
    fb = np.frombuffer(img[0x1b134d + 0x101f103:0x1b134d + 0x101f103 + 16], np.float32).copy()
    return eps, fb


def emulated_normalize(emu: UnityPlayerEmu, q):
    from unicorn import x86_const as x
    q = np.asarray(q, np.float32)
    emu.uc.reg_write(x.UC_X86_REG_XMM2, int.from_bytes(q.tobytes(), 'little'))
    emu.uc.reg_write(x.UC_X86_REG_RAX, HEAP)  # interleaved address arithmetic reads [rax+0x18]
    emu.uc.reg_write(x.UC_X86_REG_R10, 0)
    emu.uc.emu_start(emu.base + SETLOCALR_NORMALIZE[0], emu.base + SETLOCALR_NORMALIZE[1])
    return np.frombuffer(emu.uc.reg_read(x.UC_X86_REG_XMM2).to_bytes(16, 'little'), np.float32).copy()
