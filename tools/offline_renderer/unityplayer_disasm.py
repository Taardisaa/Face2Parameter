"""Read-only symbol lookup and disassembly of the installed UnityPlayer.dll.

Used to recover specific undocumented engine rules (shadow cascade fitting,
light sorting, culling) for ``src/render/hs2pipe/unity_forward.py``.

* Symbols come from Unity's public symbol server PDB that matches the
  installed DLL's CodeView record (RSDS GUID/age are checked, not assumed).
  Only the public symbol stream (S_PUB32 records) is read.
* Disassembly uses capstone (installed into an ignored ``pydeps`` folder of
  the output directory; pass ``--pydeps``).

Nothing is executed or patched.  Output is evidence for a human-written rule;
it is not a translator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

MSF_MAGIC = b'Microsoft C/C++ MSF 7.00\r\n\x1aDS\0\0\0'
S_PUB32 = 0x110E


class PE:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        d = self.data
        pe = struct.unpack_from('<I', d, 0x3c)[0]
        nsec = struct.unpack_from('<H', d, pe + 6)[0]
        optsz = struct.unpack_from('<H', d, pe + 20)[0]
        opt = pe + 24
        if struct.unpack_from('<H', d, opt)[0] != 0x20b:
            raise ValueError('PE32+ required')
        self.image_base = struct.unpack_from('<Q', d, opt + 24)[0]
        dd = opt + 112
        self.sections = []
        so = opt + optsz
        for i in range(nsec):
            name, vs, va, rs, ra = struct.unpack_from('<8sIIII', d, so + i * 40)
            self.sections.append({'name': name.rstrip(b'\0').decode(), 'vsize': vs, 'rva': va, 'rsize': rs, 'raw': ra})
        dbg_rva, dbg_size = struct.unpack_from('<II', d, dd + 6 * 8)
        self.codeview = None
        for i in range(dbg_size // 28):
            _, _, _, _, typ, _, _, ptr = struct.unpack_from('<IIHHIIII', d, self.rva_to_offset(dbg_rva) + i * 28)
            if typ == 2 and d[ptr:ptr + 4] == b'RSDS':
                g = struct.unpack('<IHH8s', d[ptr + 4:ptr + 20])
                self.codeview = {'guid': '%08X%04X%04X%s' % (g[0], g[1], g[2], g[3].hex().upper()),
                                 'age': struct.unpack_from('<I', d, ptr + 20)[0],
                                 'pdb': d[ptr + 24:d.index(b'\0', ptr + 24)].decode()}

    def rva_to_offset(self, rva):
        for s in self.sections:
            if s['rva'] <= rva < s['rva'] + max(s['vsize'], s['rsize']):
                return rva - s['rva'] + s['raw']
        raise ValueError('RVA outside sections: %x' % rva)

    def read_rva(self, rva, n):
        """Bytes as loaded: virtual bytes past a section's raw size are zero
        (uninitialized .data), never the next section's file bytes."""
        out = bytearray()
        for i in range(n):
            r = rva + i
            sec = next((s for s in self.sections if s['rva'] <= r < s['rva'] + max(s['vsize'], s['rsize'])), None)
            if sec is None:
                raise ValueError('RVA outside sections: %x' % r)
            off = r - sec['rva']
            out.append(self.data[sec['raw'] + off] if off < sec['rsize'] else 0)
        return bytes(out)


class PDB:
    def __init__(self, path):
        self.f = open(path, 'rb')
        sb = self.f.read(56)
        if sb[:32] != MSF_MAGIC:
            raise ValueError('MSF 7.00 PDB required')
        self.block, _, _, dir_bytes, _, map_addr = struct.unpack_from('<6I', sb, 32)
        nblocks = (dir_bytes + self.block - 1) // self.block
        self.f.seek(map_addr * self.block)
        dir_blocks = struct.unpack('<%dI' % nblocks, self.f.read(4 * nblocks))
        directory = b''.join(self._block(b) for b in dir_blocks)[:dir_bytes]
        n = struct.unpack_from('<I', directory, 0)[0]
        sizes = struct.unpack_from('<%dI' % n, directory, 4)
        pos = 4 + 4 * n
        self.streams = []
        for size in sizes:
            if size == 0xffffffff:
                self.streams.append((0, []))
                continue
            cnt = (size + self.block - 1) // self.block
            self.streams.append((size, struct.unpack_from('<%dI' % cnt, directory, pos)))
            pos += 4 * cnt

    def _block(self, b):
        self.f.seek(b * self.block)
        return self.f.read(self.block)

    def stream(self, i):
        size, blocks = self.streams[i]
        return b''.join(self._block(b) for b in blocks)[:size]

    def info(self):
        s = self.stream(1)
        _, _, age = struct.unpack_from('<III', s, 0)
        g = struct.unpack('<IHH8s', s[12:28])
        return {'guid': '%08X%04X%04X%s' % (g[0], g[1], g[2], g[3].hex().upper()), 'age': age}

    def publics(self):
        dbi = self.stream(3)
        sym_stream = struct.unpack_from('<H', dbi, 20)[0]
        data = self.stream(sym_stream)
        pos = 0
        out = []
        while pos + 4 <= len(data):
            reclen, kind = struct.unpack_from('<HH', data, pos)
            if reclen < 2:
                break
            if kind == S_PUB32:
                flags, off, seg = struct.unpack_from('<IIH', data, pos + 4)
                end = data.index(b'\0', pos + 14)
                out.append({'name': data[pos + 14:end].decode('latin-1'), 'segment': seg, 'offset': off, 'flags': flags})
            pos += reclen + 2
        return out


def load(pe_path, pdb_path):
    pe = PE(pe_path)
    pdb = PDB(pdb_path)
    info = pdb.info()
    if pe.codeview is None or info['guid'] != pe.codeview['guid'] or info['age'] != pe.codeview['age']:
        raise ValueError('PDB does not match DLL CodeView record: %r vs %r' % (info, pe.codeview))
    syms = []
    for p in pdb.publics():
        if 1 <= p['segment'] <= len(pe.sections):
            p['rva'] = pe.sections[p['segment'] - 1]['rva'] + p['offset']
            syms.append(p)
    syms.sort(key=lambda r: r['rva'])
    return pe, syms


def function_extent(syms, index, pe):
    """[rva, next public symbol rva) in .text; a function ends at or before the next public."""
    rva = syms[index]['rva']
    nxt = next((s['rva'] for s in syms[index + 1:] if s['rva'] > rva), rva + 0x1000)
    return rva, min(nxt, rva + 0x8000)


def disassemble(pe, start, end, symbols_by_rva):
    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    code = pe.read_rva(start, end - start)
    lines = []
    for ins in md.disasm(code, pe.image_base + start):
        rva = ins.address - pe.image_base
        text = '%08x  %-8s %s' % (rva, ins.mnemonic, ins.op_str)
        m = re.search(r'0x([0-9a-f]+)', ins.op_str)
        if ins.mnemonic.startswith('call') or ins.mnemonic.startswith('j') or 'rip' in ins.op_str:
            target = None
            if 'rip' in ins.op_str:
                mm = re.search(r'rip ([+-]) 0x([0-9a-f]+)', ins.op_str)
                if mm:
                    disp = int(mm.group(2), 16) * (1 if mm.group(1) == '+' else -1)
                    target = rva + ins.size + disp
            elif m:
                target = int(m.group(1), 16) - pe.image_base
            if target is not None:
                name = symbols_by_rva.get(target)
                if name:
                    text += '    ; ' + name
                elif 'rip' in ins.op_str and ins.mnemonic in ('movss', 'movaps', 'movups', 'mulss', 'addss', 'subss', 'divss',
                                                              'comiss', 'ucomiss', 'maxss', 'minss', 'andps', 'xorps',
                                                              'movsd', 'mulps', 'addps', 'subps', 'cvtss2sd', 'andnps', 'orps'):
                    try:
                        raw = pe.read_rva(target, 16)
                        text += '    ; [%x] f32 %s' % (target, ' '.join('%.9g' % v for v in struct.unpack('<4f', raw)))
                    except ValueError:
                        pass
        lines.append(text)
        if rva >= end:
            break
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dll', type=Path, default=Path('E:/HoneySelect2_ArcticFox/UnityPlayer.dll'))
    ap.add_argument('--pdb', type=Path, required=True)
    ap.add_argument('--pydeps', type=Path, help='directory with capstone (pip --target)')
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('find')
    f.add_argument('pattern')
    d = sub.add_parser('disasm')
    d.add_argument('symbol', help='exact decorated public name or 0xRVA')
    d.add_argument('--out', type=Path)
    c = sub.add_parser('callers', help='direct E8 rel32 call sites of a target RVA in .text')
    c.add_argument('target', help='0xRVA')
    args = ap.parse_args(argv)
    if args.pydeps:
        sys.path.insert(0, str(args.pydeps))
    pe, syms = load(args.dll, args.pdb)
    if args.cmd == 'find':
        rx = re.compile(args.pattern)
        for s in syms:
            if rx.search(s['name']):
                print('%08x %s' % (s['rva'], s['name']))
        return
    by_rva = {}
    for s in syms:
        by_rva.setdefault(s['rva'], s['name'])
    if args.cmd == 'callers':
        import bisect
        target = int(args.target, 16)
        text = next(s for s in pe.sections if s['name'] == '.text')
        code = pe.data[text['raw']:text['raw'] + text['rsize']]
        starts = [s['rva'] for s in syms]
        pos = code.find(b'\xe8')
        while pos != -1:
            if pos + 5 <= len(code):
                rel = struct.unpack_from('<i', code, pos + 1)[0]
                rva = text['rva'] + pos
                if rva + 5 + rel == target:
                    i = bisect.bisect_right(starts, rva) - 1
                    print('%08x  in %08x %s' % (rva, syms[i]['rva'], syms[i]['name'][:160]))
            pos = code.find(b'\xe8', pos + 1)
        return
    if args.symbol.startswith('0x'):
        # Unnamed internal function: from the given RVA to the next public symbol.
        start = int(args.symbol, 16)
        nxt = next(s['rva'] for s in syms if s['rva'] > start)
        end = min(nxt, start + 0x8000)
        name = by_rva.get(start, '<no public symbol>')
    else:
        idx = next(i for i, s in enumerate(syms) if s['name'] == args.symbol)
        start, end = function_extent(syms, idx, pe)
        name = syms[idx]['name']
    lines = disassemble(pe, start, end, by_rva)
    header = {'dll': str(args.dll), 'dll_sha256': hashlib.sha256(pe.data).hexdigest(), 'codeview': pe.codeview,
              'symbol': name, 'rva': '%x' % start, 'end_rva': '%x' % end}
    text = json.dumps(header) + '\n' + '\n'.join(lines) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text)


if __name__ == '__main__':
    main()
