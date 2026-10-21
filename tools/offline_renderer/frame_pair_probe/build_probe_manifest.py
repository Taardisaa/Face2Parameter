"""Shader property manifest for FramePairProbe v5, built from the installed shader library.

Replaces the provisional 37-shader ``shader_manifest.tsv`` of the first paired
session. Every shader of ``src/render/hs2pipe/shader_library.py`` (all extracted
installed copies) contributes:

* ``P`` lines (material-typed properties per shader name): the serialized
  ``Properties`` block (Float/Range -> float, Color -> color, Vector -> vector,
  Texture -> texture) and every variable of every non-built-in constant buffer
  (``$Globals`` and custom cbuffers) of every subprogram (``Subprogram.uniform_map``)
  plus every texture binding (``Subprogram.resources``);
* ``K`` lines: every keyword of every subprogram variant;
* ``G`` lines: every variable of every constant buffer (built-in ``Unity*``
  cbuffers included) and every texture binding name, i.e. every name a shader of
  the library can read from the global property sheet.

Types of cbuffer variables: matrix -> matrix, 1 column -> float (int/bool kinds
included: ShaderPropertySheet has only float/vector/matrix/texture/buffer storage
(FindPropertyData 0x710d20), Material.GetInt is (int)GetFloat, so the float is the
lossless value and the int conversion happens at cbuffer write), more columns ->
vector; ``array_size`` is kept. A name declared with two
different types for one shader (or globally) is an error, not resolved silently,
except that a serialized Color/Vector property is the material type of a float4
variable of the same name.

The probe reads the values through managed getters only for names that are
already registered natively, and dumps every material/global sheet natively, so
the manifest decides which managed cross-check reads are made, not coverage.

    python -m tools.offline_renderer.frame_pair_probe.build_probe_manifest \
        --library outputs/offline_renderer_shader_library_20261020/library_v1 --out FRESH.tsv
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src' / 'render'))
from hs2pipe import shader_library as sl  # noqa: E402

LIB_ROOTS = ('main', 'resources_standard', 'dhh_dhpost')
PROP_TYPES = {'float': 'float', 'range': 'float', 'color': 'color', 'vector': 'vector', 'texture': 'texture', 'int': 'int'}


class ManifestError(ValueError):
    pass


def is_builtin_cbuffer(name):
    return name != '$Globals' and name.startswith('Unity')


def variable_type(v):
    if v.matrix:
        return 'matrix'
    return 'float' if v.columns == 1 else 'vector'


def _put(table, owner, name, typ, size):
    old = table.get(name)
    if old is None:
        table[name] = (typ, size)
        return
    if old == (typ, size):
        return
    # serialized Color/Vector property versus its float4 cbuffer variable
    if {old[0], typ} == {'color', 'vector'} and old[1] == size == 0:
        table[name] = ('color', 0)
        return
    raise ManifestError(f'{owner}: {name} declared as {old} and {(typ, size)}')


def build(library):
    shaders = {}
    keywords = set()
    globals_ = {}
    stats = {'shader_names': 0, 'copies': 0, 'subprograms': 0}
    for name in library.shader_names():
        stats['shader_names'] += 1
        props = {}
        for shader in library.copies(name):
            stats['copies'] += 1
            for p in shader.record.get('properties') or []:
                t = PROP_TYPES.get(p['type'])
                if t is None:
                    raise ManifestError(f'{name}: unknown serialized property type {p["type"]} of {p["name"]}')
                _put(props, name, p['name'], t, 0)
            for sub in shader.subshaders:
                for ps in sub.passes:
                    for stage in ps.stages():
                        for sp in ps.subprograms(stage):
                            if not sp.dxbc_sha256:
                                continue
                            stats['subprograms'] += 1
                            keywords.update(sp.keywords)
                            for var_name, (cb, _slot, v) in sp.uniform_map().items():
                                typ, size = variable_type(v), v.array_size or 0
                                if not is_builtin_cbuffer(cb):
                                    if var_name in props and props[var_name][0] in ('color', 'vector') and typ == 'vector':
                                        pass
                                    else:
                                        _put(props, name, var_name, typ, size)
                                g_typ = 'vector' if typ in ('vector', 'color') else typ
                                if var_name in globals_ and globals_[var_name][0] != g_typ and {globals_[var_name][0], g_typ} <= {'float', 'vector'}:
                                    # a scalar in one program and a vector in another: the global
                                    # sheet stores one entry; read it as a vector (superset).
                                    globals_[var_name] = ('vector', max(size, globals_[var_name][1]))
                                else:
                                    _put(globals_, 'globals', var_name, g_typ, size)
                            for tex in sp.resources()['textures']:
                                _put(props, name, tex['name'], 'texture', 0)
                                _put(globals_, 'globals', tex['name'], 'texture', 0)
        shaders[name] = props
    return shaders, sorted(keywords), globals_, stats


def write(path, library_roots, shaders, keywords, globals_, stats):
    lines = ['# FramePairProbe v5 shader manifest (tools/offline_renderer/frame_pair_probe/build_probe_manifest.py)',
             '# generated ' + datetime.datetime.now(datetime.timezone.utc).isoformat()]
    for r in library_roots:
        lib_json = Path(r) / 'library.json'
        lines.append('# library %s library.json sha256=%s' % (r, hashlib.sha256(lib_json.read_bytes()).hexdigest()))
    lines.append('# stats %s' % ' '.join('%s=%d' % kv for kv in sorted(stats.items())))
    for shader in sorted(shaders):
        for name, (typ, size) in sorted(shaders[shader].items()):
            lines.append('\t'.join(('P', shader, name, typ, str(size))))
    for k in keywords:
        lines.append('K\t' + k)
    for name, (typ, size) in sorted(globals_.items()):
        lines.append('\t'.join(('G', name, typ, str(size))))
    for line in lines:
        if any(c in line for c in '\r\n') or (not line.startswith('#') and len(line.split('\t')) not in (2, 4, 5)):
            raise ManifestError('Unwritable manifest line: %r' % line)
    data = ('\n'.join(lines) + '\n').encode('utf-8')
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--library', type=Path, required=True, help='library_v1 directory containing main/resources_standard/dhh_dhpost')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Fresh output path required: %s' % args.out)
    roots = [args.library / r for r in LIB_ROOTS]
    library = sl.ShaderLibrary(roots)
    shaders, keywords, globals_, stats = build(library)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sha = write(args.out, roots, shaders, keywords, globals_, stats)
    print('%s sha256=%s shaders=%d P=%d K=%d G=%d' % (args.out, sha, len(shaders), sum(len(v) for v in shaders.values()),
                                                   len(keywords), len(globals_)))


if __name__ == '__main__':
    main()
