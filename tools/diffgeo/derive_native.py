"""Derive the lane formulas of the UnityPlayer.dll skinning routines from the installed binary.

The routines that place the skinned head vertices are closed-source SSE code inside the
installed ``E:/HoneySelect2_ArcticFox/UnityPlayer.dll`` (Unity 2018.4.11f1, SHA-256
``5034cdc2...``; the matched PDB names them, see ``src/render/unity_skin_emu.py``).  The exact
CPU chain executes them under Unicorn.  For the differentiable chain the same instructions are
*lifted*: a symbolic lane interpreter executes the straight-line SSE body of each routine on
symbolic inputs (4 lane expressions per xmm register; memory operands resolved through a
register -> region map; integer/address instructions only select regions and are skipped) and
the resulting expression DAG is written out as Python code over torch tensors, with one
binary32 rounding ``R`` after every packed/scalar float operation.  Bit operations are
recognised only in the exact forms the code uses (sign-mask AND/XOR, lane-select masks
``andnps``/``orps`` with the constant (0,0,0,~0), sign-of-scale products); anything else raises.

Routines (RVA, the range lifted, how it is used):

* ``gpr_step``  CalculateGlobalPositionAndRotation 0xdda60, loop body 0xddb10..0xddc62 (setup
  0xddaa8..0xddb03): one parent step of the world position / rotation walk;
* ``inv_noscale`` CalculateInverseGlobalMatrixNoScale tail 0xbfdd73..0xbfde77 + the
  GetWorldToLocalMatrixNoScale packing 0xbff997..0xbff9fb;
* ``local_compose`` SkinnedMeshRendererManager::CalculateSkinningMatrices prepared path, entry
  loop 0xbda830..0xbdaa72 (setup 0xbda7e6..0xbda82f): pose[i] = pose[parent] * TRS(entry i);
* ``bind_mul``  same routine, bone loop 0xbdaa90..0xbdabd5: palette[k] = pose[map[k]] * bindpose[k];
* ``skin4``     SkinVerticesGenericCPU<4,1,1> 0x776730, vertex loop 0x776790..0x77695e;
* ``euler_case4`` EulerToQuaternion 0x7cc120 (called by Internal_FromEulerRad 0x9541d0 with
  order 4), case 4 0x7cc4ce..0x7cc5ed after the six CRT sinf/cosf calls.

    .venv/Scripts/python.exe tools/diffgeo/derive_native.py --out src/render/diffgeo/native_generated.py

The output is deterministic; ``--check`` regenerates in memory and compares with the file.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import struct
import sys
from pathlib import Path

DLL = Path('E:/HoneySelect2_ArcticFox/UnityPlayer.dll')
DLL_SHA = '5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923'
SIGN = 0x80000000
ONES = 0xffffffff
LANE_MASK_W = (0, 0, 0, ONES)          # __xmm@ffffffff000000000000000000000000 (0x11d05f0)


def K(bits):
    return ('k', bits & ONES)


ZERO = K(0)


def kval(n):
    return struct.unpack('<f', struct.pack('<I', n[1]))[0]


class LiftError(RuntimeError):
    pass


def simp(n):
    """Canonical forms of the bit operations the routines use; anything else stays opaque and
    is rejected by the emitter."""
    op = n[0]
    if op == 'and':
        a, b = n[1], n[2]
        for x, y in ((a, b), (b, a)):
            if x == K(ONES):
                return y
            if x == ZERO:
                return ZERO
            if x == K(SIGN):
                return K(y[1] & SIGN) if y[0] == 'k' else ('signbit', y)
        if a[0] == 'k' and b[0] == 'k':
            return K(a[1] & b[1])
    elif op == 'andn':                          # ~a & b
        a, b = n[1], n[2]
        if a == ZERO:
            return b
        if a == K(ONES):
            return ZERO
        if a[0] == 'k' and b[0] == 'k':
            return K(~a[1] & b[1])
    elif op in ('or', 'xor'):
        a, b = n[1], n[2]
        for x, y in ((a, b), (b, a)):
            if x == ZERO:
                return y
        if a[0] == 'k' and b[0] == 'k':
            return K(a[1] | b[1]) if op == 'or' else K(a[1] ^ b[1])
        if op == 'xor' and a == b:
            return ZERO
        if op == 'xor':
            for x, y in ((a, b), (b, a)):
                if x == K(SIGN):
                    return ('neg', y)
                if x[0] == 'signbit':
                    return ('xorsign', y, x[1])        # y, sign flipped where x[1] is negative
    return n


def isreg(s):
    return re.match(r'^xmm\d+$', s.strip()) is not None


class Machine:
    def __init__(self, img, regions):
        self.img = img
        self.x = {}
        self.mem = {}
        self.regions = regions

    def reg(self, r):
        if r not in self.x:
            self.x[r] = [('in', r, i) for i in range(4)]
        return list(self.x[r])

    def addr(self, s, ins):
        inner = re.search(r'\[(.*)\]', s).group(1).replace(' ', '')
        if inner.startswith('rip'):
            off = int(inner[3:], 0) if inner[3:] else 0
            return ('rip', ins.address + ins.size + off)
        mm = re.match(r'([a-z0-9]+)([+-](?:0x[0-9a-f]+|[0-9]+))?$', inner)
        if not mm:
            raise LiftError('unsupported address %s at %#x' % (inner, ins.address))
        base = mm.group(1)
        if base not in self.regions:
            raise LiftError('no region for %s at %#x' % (base, ins.address))
        return (self.regions[base], int(mm.group(2), 0) if mm.group(2) else 0)

    def load(self, a, n):
        if a[0] == 'rip':
            return [K(struct.unpack_from('<I', self.img, a[1] + 4 * i)[0]) for i in range(n)]
        return [self.mem.get((a[0], a[1] + 4 * i), ('m', a[0], a[1] + 4 * i)) for i in range(n)]

    def operand(self, s, ins):
        s = s.strip()
        if isreg(s):
            return self.reg(s)
        if 'ptr' in s:
            return self.load(self.addr(s, ins), 4 if 'xmmword' in s else 1)
        raise LiftError('operand %s at %#x' % (s, ins.address))

    def run(self, md, start, end):
        for ins in md.disasm(self.img[start:end], start):
            self.step(ins)

    def step(self, ins):
        mn, ops = ins.mnemonic, ins.op_str
        if 'xmm' not in ops:
            if mn.startswith(('mul', 'add', 'sub', 'div', 'sqrt')) and mn.endswith(('ps', 'ss')):
                raise LiftError('float op without xmm operand at %#x' % ins.address)
            return
        parts = [p.strip() for p in re.split(r',(?![^\[]*\])', ops)]
        d = parts[0]
        if mn in ('movaps', 'movups', 'movdqa', 'movdqu'):
            if isreg(d):
                self.x[d] = self.operand(parts[1], ins)
            else:
                a = self.addr(d, ins)
                for i, v in enumerate(self.reg(parts[1])):
                    self.mem[(a[0], a[1] + 4 * i)] = v
            return
        if mn == 'movss':
            if isreg(d) and isreg(parts[1]):
                v = self.reg(d)
                v[0] = self.reg(parts[1])[0]
                self.x[d] = v
            elif isreg(d):
                self.x[d] = self.operand(parts[1], ins) + [ZERO, ZERO, ZERO]
            else:
                a = self.addr(d, ins)
                self.mem[(a[0], a[1])] = self.reg(parts[1])[0]
            return
        if mn in ('xorps', 'pxor') and parts[1] == d:
            self.x[d] = [ZERO] * 4
            return
        a = self.reg(d) if isreg(d) else None
        if a is None:
            raise LiftError('memory destination %s at %#x' % (mn, ins.address))
        b = self.operand(parts[1], ins)
        if len(b) == 1:
            b = b + [ZERO] * 3
        packed = {'mulps': 'mul', 'addps': 'add', 'subps': 'sub', 'divps': 'div', 'andps': 'and', 'pand': 'and',
                  'orps': 'or', 'por': 'or', 'xorps': 'xor', 'pxor': 'xor', 'andnps': 'andn', 'pandn': 'andn'}
        scalar = {'mulss': 'mul', 'addss': 'add', 'subss': 'sub', 'divss': 'div'}
        if mn in packed:
            self.x[d] = [simp((packed[mn], a[i], b[i])) for i in range(4)]
        elif mn in scalar:
            v = list(a)
            v[0] = simp((scalar[mn], a[0], b[0]))
            self.x[d] = v
        elif mn == 'sqrtps':
            self.x[d] = [('sqrt', b[i]) for i in range(4)]
        elif mn == 'pshufd':
            imm = int(parts[2], 0)
            self.x[d] = [b[(imm >> (2 * i)) & 3] for i in range(4)]
        elif mn == 'shufps':
            imm = int(parts[2], 0)
            self.x[d] = [a[imm & 3], a[(imm >> 2) & 3], b[(imm >> 4) & 3], b[(imm >> 6) & 3]]
        elif mn == 'unpcklps':
            self.x[d] = [a[0], b[0], a[1], b[1]]
        elif mn == 'unpckhps':
            self.x[d] = [a[2], b[2], a[3], b[3]]
        elif mn == 'movlhps':
            self.x[d] = [a[0], a[1], b[0], b[1]]
        elif mn == 'movhlps':
            self.x[d] = [b[2], b[3], a[2], a[3]]
        else:
            raise LiftError('unsupported instruction %#x %s %s' % (ins.address, mn, ops))


# ------------------------------------------------------------------------------------ emitter
class Emitter:
    """Expression DAG -> SSA Python over lane tensors.  ``R`` rounds to binary32 (identity in
    the float64 mode); ``S(x)`` = copysign(1, x) (the sign-of-scale products)."""

    def __init__(self, names):
        self.names = names            # input node -> python expression
        self.var = {}
        self.lines = []

    def expr(self, n):
        if n in self.names:
            return self.names[n]
        if n[0] == 'k':
            v = kval(n)
            if n[1] == SIGN:
                return '-0.0'
            return repr(float(v))
        if n in self.var:
            return self.var[n]
        op = n[0]
        if op in ('mul', 'add', 'sub', 'div'):
            a, b = self.expr(n[1]), self.expr(n[2])
            if n[1][0] == 'k' and n[2][0] == 'k':
                raise LiftError('constant-only operation')
            s = 'R(%s %s %s)' % (a, {'mul': '*', 'add': '+', 'sub': '-', 'div': '/'}[op], b)
        elif op == 'sqrt':
            s = 'R(sqrt(%s))' % self.expr(n[1])
        elif op == 'neg':
            s = '-%s' % self.expr(n[1])
        elif op == 'xorsign':
            y, e = n[1], n[2]
            # e is a product of copysign(1, s) terms (exactly +-1); y * e flips the sign bit
            # exactly when e is negative (also for zeros)
            if y == K(0x3f800000):
                s = self.sign_of(e)
            else:
                s = '(%s * %s)' % (self.expr(y), self.sign_of(e))
        else:
            raise LiftError('unsupported node %r' % (op,))
        v = 'v%d' % len(self.var)
        self.var[n] = v
        self.lines.append('    %s = %s' % (v, s))
        return v

    def sign_of(self, e):
        if e[0] == 'xorsign' and e[1] == K(0x3f800000):
            return 'S(%s)' % self.expr(e[2])
        if e[0] == 'mul':
            return '(%s * %s)' % (self.sign_of(e[1]), self.sign_of(e[2]))
        if e[0] == 'xorsign':
            return '(%s * %s)' % (self.sign_of(e[1]), self.sign_of(e[2]))
        raise LiftError('sign expression %r' % (e,))


def function(name, args, doc, outputs, names):
    em = Emitter(names)
    outs = [em.expr(o) for o in outputs]
    body = ['def %s(%s):' % (name, ', '.join(['R', 'S', 'sqrt'] + args)), '    """%s"""' % doc]
    body += em.lines
    body.append('    return (%s,)' % ', '.join(outs))
    return '\n'.join(body), len(em.lines)


def derive():
    import capstone
    import pefile
    data = DLL.read_bytes()
    if hashlib.sha256(data).hexdigest() != DLL_SHA:
        raise LiftError('Installed UnityPlayer.dll differs from the audited binary')
    pe = pefile.PE(data=data, fast_load=True)
    img = pe.get_memory_mapped_image()
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    funcs, counts = [], {}

    # ---- CalculateGlobalPositionAndRotation: one parent step
    m = Machine(img, {'rax': 'P', 'rdx': 'pos', 'r8': 'rot', 'r11': 'save', 'rsp': 'save'})
    for i in range(4):
        m.mem[('pos', 4 * i)] = ('in', 'pos', i)
        m.mem[('rot', 4 * i)] = ('in', 'rot', i)
    m.run(md, 0xddaa8, 0xddb04)
    m.run(md, 0xddb10, 0xddc62)
    names = {('in', 'pos', i): 'pos[%d]' % i for i in range(3)}
    names.update({('in', 'rot', i): 'rot[%d]' % i for i in range(4)})
    names.update({('m', 'P', 4 * i): 'pt[%d]' % i for i in range(3)})
    names.update({('m', 'P', 0x10 + 4 * i): 'pq[%d]' % i for i in range(4)})
    names.update({('m', 'P', 0x20 + 4 * i): 'ps[%d]' % i for i in range(3)})
    outs = [m.mem[('pos', 4 * i)] for i in range(3)] + [m.mem[('rot', 4 * i)] for i in range(4)]
    src, counts['gpr_step'] = function(
        'gpr_step', ['pos', 'rot', 'pt', 'pq', 'ps'],
        'CalculateGlobalPositionAndRotation 0xdda60, loop body 0xddb10..0xddc62: (pos, rot) of the child '
        'expressed in the parent (pt, pq, ps) frame.  Returns pos x,y,z, rot x,y,z,w.', outs, names)
    funcs.append(src)

    # ---- CalculateInverseGlobalMatrixNoScale tail + GetWorldToLocalMatrixNoScale packing
    m = Machine(img, {'rsp': 'stk', 'rbx': 'aff'})
    for i in range(4):
        m.mem[('stk', 0x30 + 4 * i)] = ('in', 'rot', i)
        m.mem[('stk', 0x40 + 4 * i)] = ('in', 'pos', i)
    m.run(md, 0xbfdd73, 0xbfde77)
    m2 = Machine(img, {'rax': 'aff', 'rdi': 'res', 'rsp': 'stk'})
    m2.mem.update({k: v for k, v in m.mem.items() if k[0] == 'aff'})
    m2.run(md, 0xbff997, 0xbff9fb)
    names = {('in', 'pos', i): 'pos[%d]' % i for i in range(3)}
    names.update({('in', 'rot', i): 'rot[%d]' % i for i in range(4)})
    outs = [m2.mem[('res', 4 * i)] for i in range(16)]
    src, counts['inv_noscale'] = function(
        'inv_noscale', ['pos', 'rot'],
        'GetWorldToLocalMatrixNoScale: CalculateInverseGlobalMatrixNoScale tail 0xbfdd73..0xbfde77 and the '
        'packing 0xbff997..0xbff9fb.  Returns the 16 floats column-major.', outs, names)
    funcs.append(src)

    # ---- CalculateSkinningMatrices prepared: entry loop
    m = Machine(img, {'r8': 'L', 'rcx': 'Pp', 'r10': 'out', 'rbp': 'stk'})
    m.run(md, 0xbda7e6, 0xbda830)
    m.run(md, 0xbda830, 0xbdaa72)
    names = {('m', 'L', 4 * i): 't[%d]' % i for i in range(3)}
    names.update({('m', 'L', 0x10 + 4 * i): 'q[%d]' % i for i in range(4)})
    names.update({('m', 'L', 0x20 + 4 * i): 's[%d]' % i for i in range(3)})
    names.update({('m', 'Pp', 4 * i): 'P[%d]' % i for i in range(16)})
    outs = [m.mem[('out', -0x20 + 4 * i)] for i in range(16)]
    src, counts['local_compose'] = function(
        'local_compose', ['t', 'q', 's', 'P'],
        'CalculateSkinningMatrices prepared path, entry loop 0xbda830..0xbdaa72: parent pose P (16 floats '
        'column-major) times the entry TRS (t, q, s).  Returns 16 floats column-major.', outs, names)
    funcs.append(src)

    # ---- bone loop
    m = Machine(img, {'rax': 'B', 'rcx': 'Po', 'r8': 'out', 'rbp': 'stk'})
    m.run(md, 0xbdaa90, 0xbdabd5)
    names = {('m', 'B', 4 * i): 'B[%d]' % i for i in range(16)}
    names.update({('m', 'Po', 4 * i): 'P[%d]' % i for i in range(16)})
    outs = [m.mem[('out', 4 * i)] for i in range(16)]
    src, counts['bind_mul'] = function(
        'bind_mul', ['P', 'B'],
        'CalculateSkinningMatrices prepared path, bone loop 0xbdaa90..0xbdabd5: pose P times bindpose B '
        '(column-major).  Returns 16 floats column-major.', outs, names)
    funcs.append(src)

    # ---- SkinVerticesGenericCPU<4,1,1>
    m = Machine(img, {'r11': 'W', 'rdx': 'M0', 'rax': 'M1', 'rcx': 'M2', 'r8': 'M3', 'rbx': 'S', 'rdi': 'D'})
    m.run(md, 0x776790, 0x77695e)
    names = {('m', 'W', -8 + 4 * k): 'w[%d]' % k for k in range(4)}
    for k in range(4):
        names.update({('m', 'M%d' % k, 4 * i): 'M%d[%d]' % (k, i) for i in range(16)})
    names.update({('m', 'S', 4 * i): 'src[%d]' % i for i in range(10)})
    outs = [m.mem[('D', 4 * i)] for i in range(10)]
    src, counts['skin4'] = function(
        'skin4', ['w', 'M0', 'M1', 'M2', 'M3', 'src'],
        'SkinVerticesGenericCPU<4,1,1> 0x776730, vertex body 0x776790..0x77695e: 4 weights, the 4 palette '
        'matrices (column-major) and the 40-byte source vertex (pos, normal, tangent xyzw).  Returns the '
        '10 output floats.', outs, names)
    funcs.append(src)

    # ---- EulerToQuaternion, order 4
    m = Machine(img, {'rsi': 'out'})
    regs = {'xmm13': 'sx', 'xmm15': 'cx', 'xmm14': 'sy', 'xmm12': 'cy', 'xmm10': 'sz', 'xmm11': 'cz'}
    for r, n in regs.items():
        m.x[r] = [('in', n, 0)] + [('in', n, i) for i in range(1, 4)]
    m.run(md, 0x7cc4ce, 0x7cc5ed)
    names = {('in', n, 0): n for n in regs.values()}
    outs = [m.mem[('out', 4 * i)] for i in range(4)]
    src, counts['euler_case4'] = function(
        'euler_case4', ['sx', 'cx', 'sy', 'cy', 'sz', 'cz'],
        'EulerToQuaternion 0x7cc120, order 4 (case 0x7cc4ce..0x7cc5ed) from the CRT sinf/cosf of the half '
        'angles (x, y, z).  Returns x, y, z, w.', outs, names)
    funcs.append(src)

    header = ('"""Generated by tools/diffgeo/derive_native.py from the installed UnityPlayer.dll (do not edit).\n\n'
              'UnityPlayer.dll SHA-256 %s\n'
              'Lane formulas of the native skinning routines; every packed/scalar float operation is one\n'
              'binary32 rounding R (identity in float64 mode).  S(x) = copysign(1, x).\n'
              'SSA statements per function: %s\n"""\n\n' % (DLL_SHA, ', '.join('%s %d' % kv for kv in counts.items())))
    return header + '\n\n\n'.join(funcs) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[2] / 'src/render/diffgeo/native_generated.py')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args(argv)
    text = derive()
    if a.check:
        same = a.out.read_text(encoding='utf-8') == text
        print('generated file %s: %s' % (a.out, 'up to date' if same else 'DIFFERS'))
        return 0 if same else 1
    a.out.write_text(text, encoding='utf-8')
    print('wrote', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
