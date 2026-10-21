"""UnityPlayer.dll's statically linked CRT math, executed from the installed binary.

``powf`` (RVA 0x116c7d0), ``sinf`` (0x1174df0), ``cosf`` (0x1171520) and
``fmod`` (0x117f2f8) are called by the transcribed engine routines
(GammaToLinearSpace 0x27cae0 / 0x27c970 / ShaderPropertySheet colour
conversion 0x83cde0, SetPerspective, UpdateGlobalShaderProperties, cascade
snapping).  They are not correctly rounded in general: the paired capture
shows 1-ulp differences between the material colours Unity uploads and a
correctly rounded ``pow``.  Instead of re-deriving the vectorised CRT
polynomial, the original instructions are executed with Unicorn through the
skinning sibling's ``UnityPlayerEmu`` (which checks the DLL SHA-256).

Gap ``crt_cpu_dispatch``: the CRT selects code paths by CPU features
(``__isa_available``, initialised at process start).  The emulator runs the
image's static initial value; equality with the capture is the evidence that
the executed path matches the game's on this machine.
"""
from __future__ import annotations

import struct
from functools import lru_cache

import numpy as np

RVA = {'powf': 0x116c7d0, 'sinf': 0x1174df0, 'cosf': 0x1171520, 'fmod': 0x117f2f8}
F = np.float32


def _emu():
    from src.render import unity_skin_emu as ue
    return ue, ue.shared_emulator()


def _call(rva, xmm0, xmm1=None, double=False):
    from unicorn import x86_const as x
    ue, emu = _emu()
    rsp = ue.STACK + ue.STACK_SIZE - 0x1000
    frame = struct.pack('<Q', ue.RET) + b'\0' * 0x40
    rsp -= len(frame) + 8
    rsp &= ~0xf
    rsp -= 8
    emu.uc.mem_write(rsp, frame)
    enc = (lambda v: struct.unpack('<Q', struct.pack('<d', v))[0]) if double else \
        (lambda v: struct.unpack('<I', struct.pack('<f', v))[0])
    emu.uc.reg_write(x.UC_X86_REG_XMM0, enc(xmm0))
    if xmm1 is not None:
        emu.uc.reg_write(x.UC_X86_REG_XMM1, enc(xmm1))
    emu.uc.reg_write(x.UC_X86_REG_RSP, rsp)
    emu.uc.emu_start(emu.base + rva, ue.RET)
    raw = emu.uc.reg_read(x.UC_X86_REG_XMM0)
    if double:
        return struct.unpack('<d', struct.pack('<Q', raw & 0xffffffffffffffff))[0]
    return np.frombuffer(struct.pack('<I', raw & 0xffffffff), np.float32)[0]


@lru_cache(maxsize=None)
def _powf_bits(a, b):
    return _call(RVA['powf'], a, b)


def powf(a, b):
    return F(_powf_bits(float(F(a)), float(F(b))))


def sinf(a):
    return F(_call(RVA['sinf'], float(F(a))))


def cosf(a):
    return F(_call(RVA['cosf'], float(F(a))))


def fmod(a, b):
    return _call(RVA['fmod'], float(a), float(b), double=True)
