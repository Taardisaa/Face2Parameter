"""Export installed Unity ComputeShader kernels (DXBC + Unity binding metadata).

Read-only.  ``extract_shader_programs.py`` exports only ``Shader`` assets; the
PostProcessing v2 compute shaders (MSVO, Lut3DBaker, ...) are ``ComputeShader``
assets (class 72) whose type tree is available in the player build.  For each
D3D11 variant (targetRenderer 2) and kernel this writes the exact DXBC blob,
its SHA-256, the Unity constant-buffer layouts (name, byte size, parameters with
offset/array size/row/col count), the texture/UAV/buffer bind points and the
thread-group size from the DXBC ``dcl_thread_group`` token.

Output: a fresh directory with ``manifest.json`` and ``<shader>__<kernel>.dxbc``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import UnityPy

OP_DCL_THREAD_GROUP = 155


def dxbc_thread_group(code):
    """(x, y, z) from the SHEX/SHDR ``dcl_thread_group`` declaration."""
    if code[:4] != b'DXBC':
        raise ValueError('not DXBC')
    count = struct.unpack_from('<I', code, 28)[0]
    for i in range(count):
        off = struct.unpack_from('<I', code, 32 + 4 * i)[0]
        tag = code[off:off + 4]
        if tag not in (b'SHEX', b'SHDR'):
            continue
        size = struct.unpack_from('<I', code, off + 4)[0]
        body = code[off + 8:off + 8 + size]
        pos = 8
        while pos < len(body):
            tok = struct.unpack_from('<I', body, pos)[0]
            op = tok & 0x7ff
            length = (tok >> 24) & 0x7f
            if op == 53:  # customdata: length in the next dword
                length = struct.unpack_from('<I', body, pos + 4)[0]
            if op == OP_DCL_THREAD_GROUP:
                return list(struct.unpack_from('<III', body, pos + 4))
            if length == 0:
                raise ValueError('zero-length token')
            pos += 4 * length
    raise ValueError('no dcl_thread_group')


def extract(asset, out, names=None):
    asset, out = Path(asset), Path(out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    env = UnityPy.load(str(asset))
    result = {'schema': 'hs2_compute_shader_export/1', 'asset_path': str(asset.resolve()),
              'asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(), 'compute_shaders': []}
    for obj in env.objects:
        if obj.type.name != 'ComputeShader':
            continue
        t = obj.read_typetree()
        if names and t['m_Name'] not in names:
            continue
        row = {'name': t['m_Name'], 'path_id': obj.path_id, 'variants': []}
        for v in t['variants']:
            vrow = {'targetRenderer': v['targetRenderer'], 'targetLevel': v['targetLevel'],
                    'resourcesResolved': v.get('resourcesResolved'),
                    'constant_buffers': [{'name': cb['name'], 'byte_size': cb['byteSize'],
                                          'params': [{k: p[k] for k in ('name', 'offset', 'arraySize', 'rowCount', 'colCount', 'type')}
                                                     for p in cb['params']]} for cb in v['constantBuffers']],
                    'kernels': []}
            for k in v['kernels']:
                code = bytes(k['code'])
                fname = '%s__%s__r%d.dxbc' % (t['m_Name'], k['name'], v['targetRenderer'])
                krow = {'name': k['name'], 'dxbc': fname, 'dxbc_sha256': hashlib.sha256(code).hexdigest(),
                        'cbs': [{'name': c['name'], 'bindPoint': c['bindPoint']} for c in k['cbs']],
                        'textures': [{'name': c['name'], 'bindPoint': c['bindPoint'],
                                      'samplerBindPoint': c.get('samplerBindPoint'), 'texDimension': c.get('texDimension')}
                                     for c in k['textures']],
                        'builtin_samplers': [{'sampler': c['sampler'], 'bindPoint': c['bindPoint']} for c in k['builtinSamplers']],
                        'in_buffers': [{'name': c['name'], 'bindPoint': c['bindPoint']} for c in k['inBuffers']],
                        'out_buffers': [{'name': c['name'], 'bindPoint': c['bindPoint']} for c in k['outBuffers']],
                        'keywords': k.get('keywords'), 'requirements': k.get('requirements')}
                if v['targetRenderer'] == 2 and code[:4] == b'DXBC':
                    krow['thread_group_size'] = dxbc_thread_group(code)
                (out / fname).write_bytes(code)
                vrow['kernels'].append(krow)
            row['variants'].append(vrow)
        result['compute_shaders'].append(row)
    (out / 'manifest.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--asset', type=Path, required=True)
    ap.add_argument('--name', action='append', help='restrict to ComputeShader names')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    r = extract(args.asset, args.out, set(args.name) if args.name else None)
    print(json.dumps({'compute_shaders': [c['name'] for c in r['compute_shaders']]}))


if __name__ == '__main__':
    main()
