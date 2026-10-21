"""Generate src/render/shape_head_female_program.py from the decompiled installed
Assembly-CSharp ShapeHeadInfoFemale (DstName/SrcName enums and Update body).

    python tools/offline_renderer/gen_shape_head_program.py DECOMPILED/ShapeHeadInfoFemale.cs
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ASM = Path('E:/HoneySelect2_ArcticFox/HoneySelect2_Data/Managed/Assembly-CSharp.dll')


def enum(text, name):
    m = re.search(r'public enum ' + name + r'\s*\{(.*?)\}', text, re.S)
    return [x.strip() for x in m.group(1).split(',') if x.strip()]


def arg(expr):
    expr = expr.strip()
    if re.fullmatch(r'-?[0-9.]+f', expr):
        return {'const': float(expr[:-1])}
    terms = []
    for t in expr.split('+'):
        m = re.fullmatch(r'dictSrc\[(\d+)\]\.vct(Pos|Rot|Scl)\.([xyz])', t.strip())
        if not m:
            raise ValueError('Unparsed term: ' + t)
        terms.append([int(m.group(1)), m.group(2).lower(), 'xyz'.index(m.group(3))])
    return {'sum': terms}


def main():
    src = Path(sys.argv[1]).read_text(encoding='utf-8')
    dst, srcn = enum(src, 'DstName'), enum(src, 'SrcName')
    body = re.search(r'public override void Update\(\)\s*\{(.*?)\n\t\}\n', src, re.S).group(1)
    ops = []
    for line in body.splitlines():
        m = re.fullmatch(r'\s*dictDst\[(\d+)\]\.trfBone\.(SetLocal\w+)\((.*)\);', line)
        if m:
            ops.append({'dst': int(m.group(1)), 'method': m.group(2), 'args': [arg(a) for a in m.group(3).split(',')]})
        elif line.strip() and line.strip() not in ('{', '}') and not line.strip().startswith('if ('):
            raise ValueError('Unparsed Update line: ' + line)
    out = Path(__file__).resolve().parents[2] / 'src/render/shape_head_female_program.py'
    out.write_text('"""Generated from installed Assembly-CSharp ShapeHeadInfoFemale (do not edit).\n\n'
                   f'Assembly-CSharp.dll SHA-256 {hashlib.sha256(ASM.read_bytes()).hexdigest()}\n'
                   'Source: ShapeHeadInfoFemale.DstName/SrcName and Update(); args are float sums.\n"""\n'
                   f'DST = {json.dumps(dst)}\nSRC = {json.dumps(srcn)}\nUPDATE = {json.dumps(ops)}\n', encoding='utf-8')
    print(len(dst), len(srcn), len(ops))


if __name__ == '__main__':
    main()
