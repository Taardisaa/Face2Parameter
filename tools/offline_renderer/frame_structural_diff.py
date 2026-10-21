"""Structural diff: generated camera-render plan vs a paired capture.

Ladder step 2 (structural part) on the paired Maker frame: the draw list and
order of each pass of the main camera render, the selected programs, and the
built-in / material constant-buffer variables of every matched draw are
generated from the same-frame scene export with ``unity_forward_plan`` /
``unity_forward`` and compared with the capture (frame graph interpretation
of the official zip.xml + driver listing; per-variable bytes from
``compare_generated_cbuffers.extract``).

Values that the scene export does not contain are taken from the capture and
reported with provenance ``external:capture`` (and listed as gaps); they are
never compared as if generated.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.render.hs2pipe import shader_library as sl  # noqa: E402
from src.render.hs2pipe import unity_forward as uf  # noqa: E402
from src.render.hs2pipe import unity_forward_plan as up  # noqa: E402
from src.render.hs2pipe import forward_values as FV  # noqa: E402
from src.render.hs2pipe.forward_values import EXTERNAL_CAPTURE  # noqa: E402
from tools.offline_renderer import frame_graph as fg  # noqa: E402

F = np.float32
LIB_ROOTS = ('main', 'resources_standard', 'dhh_dhpost')


def capture_passes(xml, inspect):
    g = fg.FrameGraph(fg.load_chunks(xml), fg.load_driver_listing(inspect), fg.Blobs(str(xml)[:-4]),
                      fg.ShaderLibrary([])).build()
    passes = fg.group_passes(g)
    by = {a['event_id']: a for a in g.actions}
    rows = []
    for p in passes:
        draws = []
        for e in p['events']:
            a = by[e]
            if not a['kind'].startswith('Draw'):
                continue
            ib = (a.get('index_buffer') or {}).get('debug_names') or ['']
            draws.append({'event': e, 'mesh': ib[-1].replace('(Clone)', ''), 'args': a['arguments'],
                          'vs': a['shaders'].get('VS', {}).get('dxbc_sha256'),
                          'ps': a['shaders'].get('PS', {}).get('dxbc_sha256')})
        rows.append({'pass_index': p['pass_index'], 'label': p['label'], 'events': p['events'], 'draws': draws})
    return rows, g


def draw_mesh_name(d):
    r = d.renderer
    if r['kind'] == 'ParticleSystemRenderer':
        return 'Buffer-2-'            # dynamic particle geometry ring buffer
    if r['kind'] == 'MeshRenderer':
        return 'o_' + r['name'][2:] if r['name'].startswith('p_') else r['name']
    return r['mesh']


def compare_draw_lists(plan, cap):
    """Element-wise comparison (renderer mesh, VS, PS) of the generated and
    captured draw sequences of one pass."""
    out = []
    n = max(len(plan), len(cap))
    for i in range(n):
        g = plan[i] if i < len(plan) else None
        c = cap[i] if i < len(cap) else None
        row = {'i': i, 'generated': None if g is None else
               {'renderer': g.renderer['name'], 'submesh': g.submesh, 'shader': g.shader, 'pass': g.pass_index,
                'light_mode': g.light_mode, 'vs': g.programs.get('vs'), 'ps': g.programs.get('ps')},
               'captured': c}
        if g is None or c is None:
            row['status'] = 'count_differs'
        else:
            mesh = draw_mesh_name(g)
            mesh_ok = c['mesh'].startswith(mesh) if mesh.endswith('-') else c['mesh'] == mesh
            prog_known = g.programs.get('vs') is not None and g.programs.get('ps') is not None
            prog_ok = prog_known and g.programs.get('vs') == c['vs'] and g.programs.get('ps') == c['ps']
            row['mesh_equal'] = mesh_ok
            row['programs_equal'] = prog_ok if prog_known else None
            row['status'] = ('equal' if mesh_ok and prog_ok else
                             'equal_object_program_unresolved' if mesh_ok and not prog_known else 'different')
        out.append(row)
    return out


# ------------------------------------------------------------- cbuffers
def captured_variables(captured_doc):
    idx = {}
    for d in captured_doc['draws']:
        for st, s in d['stages'].items():
            for cb in s['cbuffers']:
                for n, v in (cb.get('variables') or {}).items():
                    idx[(d['event_id'], st, cb.get('name'), n)] = v.get('hex')
    return idx


def notes_inputs(vals):
    return vals.get('_inputs', {})


def explained_by_export_precision(raw_color, hexv):
    """True when a float32 neighbour (<= 8 ulp per channel) of the exported decimal
    colour converts to the captured bytes exactly: the probe printed the material
    colour with too few digits (field request: float bits)."""
    target = [F(x) for x in struct.unpack('<%df' % (len(hexv) // 8), bytes.fromhex(hexv))]
    for i, c in enumerate(raw_color[:3]):
        want = target[i]
        if uf.gamma_to_linear_space(F(c)) == want:
            continue
        ok = False
        for direction in (F(10.0), F(-10.0)):
            cur = F(c)
            for _ in range(8):
                cur = np.nextafter(cur, direction)
                if uf.gamma_to_linear_space(cur) == want:
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return False
    return True


def compare_draw_cbuffers(pairs, cap_vars, value_fn):
    rows = []
    totals = {}
    for g, c in pairs:
        vals, notes = value_fn(g)
        row = {'event': c['event'], 'renderer': g.renderer['name'], 'variables': {}, 'notes': notes}
        for (ev, st, cb, name), hexv in cap_vars.items():
            if ev != c['event']:
                continue
            key = '%s:%s:%s' % (st, cb, name)
            if hexv is None:
                status = 'capture_bytes_unknown'
            elif name in vals:
                gen = uf.encode(vals[name])[:len(bytes.fromhex(hexv))]
                status = 'equal' if gen.hex() == hexv else 'different'
                if status == 'different' and name in notes_inputs(vals):
                    if explained_by_export_precision(vals['_inputs'][name], hexv):
                        status = 'different_export_decimal_precision'
                if status == 'different':
                    row['variables'][key + ':generated'] = gen.hex()
                    row['variables'][key + ':captured'] = hexv
            elif name in EXTERNAL_CAPTURE:
                status = 'external:capture'
            else:
                status = 'not_generated'
            row['variables'][key] = status
            totals[status] = totals.get(status, 0) + 1
        rows.append(row)
    return rows, totals


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    W = Path(args.window)
    scene = json.loads((W / 'frames' / ('scene_%03d.json' % args.window_index)).read_text(encoding='utf-8'))
    frame = json.loads((W / 'frames' / ('frame_%03d.json' % args.window_index)).read_text(encoding='utf-8'))
    static = json.loads((W / 'static.json').read_text(encoding='utf-8'))
    lib = sl.ShaderLibrary([Path(args.library) / r for r in LIB_ROOTS])
    ad = up.SceneAdapter(scene, frame, static, lib)
    cam = ad.camera(args.camera)
    q = scene['statics']['UnityEngine.QualitySettings']
    quality = {'pixel_light_count': q['pixelLightCount'], 'shadows': up.enum_value(q['shadows'])}
    gk = sorted(k for k, v in scene['global_keywords'].items() if v)
    cap_passes, graph = capture_passes(Path(args.xml), Path(args.inspect))
    report = {'scene': str(W), 'camera': args.camera, 'global_keywords': gk, 'passes': {}}
    fwd = [p for p in cap_passes if p['label'] == 'forward_scene']
    opaque_cap = [d for d in fwd[0]['draws']] if fwd else []
    transparent_cap = [d for p in fwd[1:] for d in p['draws']]
    opaque, active = up.forward_plan(ad, cam, transparent=False, global_keywords=gk, quality=quality)
    transparent, _ = up.forward_plan(ad, cam, transparent=True, global_keywords=gk, quality=quality)
    # Skybox: drawn after the opaque queue when clearFlags == Skybox (RenderSkybox RVA 0x643680).
    sky_cap = [d for d in opaque_cap if d['mesh'] == '']
    opaque_cap = [d for d in opaque_cap if d['mesh'] != '']
    transparent_draws = [d for d in transparent if d.light_mode != 'ALWAYS' or d.shader.startswith('Legacy')]
    grabs = [d for d in transparent if d.light_mode == 'ALWAYS' and not d.shader.startswith('Legacy')]
    report['passes']['forward_opaque'] = compare_draw_lists(opaque, opaque_cap)
    report['passes']['forward_transparent'] = compare_draw_lists(transparent_draws, transparent_cap)
    report['passes']['grab_passes'] = [{'renderer': d.renderer['name'], 'shader': d.shader, 'pass': d.pass_index}
                                       for d in grabs]
    report['passes']['skybox'] = {'generated': cam.get('clear_flags') == 'Skybox', 'captured_events': [d['event'] for d in sky_cap]}
    if args.captured_cbuffers:
        from src.render import paired_skin as ps
        from src.render import unity_skin_emu as ue
        cap_vars = captured_variables(json.loads(Path(args.captured_cbuffers).read_text(encoding='utf-8')))
        win = ps.ProbeWindow(W)
        fv = FV.ForwardValues(scene, ad, lib, win, ue.shared_emulator(), cam, active)

        def value_fn(g):
            return fv.draw_values(g.renderer, g.material)
        pairs = []
        for rows_key, plan_list, cap_list in (('forward_opaque', opaque, opaque_cap),
                                              ('forward_transparent', transparent_draws, transparent_cap)):
            for g, c in zip(plan_list, cap_list):
                pairs.append((g, c))
        cb_rows, cb_totals = compare_draw_cbuffers(pairs, cap_vars, value_fn)
        report['cbuffers'] = {'totals': cb_totals, 'draws': cb_rows}
        report['summary_cbuffers'] = cb_totals
    report['gaps'] = ad.gaps
    summary = {}
    for k, rows in report['passes'].items():
        if isinstance(rows, list) and rows and 'status' in rows[0]:
            t = {}
            for r in rows:
                t[r['status']] = t.get(r['status'], 0) + 1
            summary[k] = t
    report['summary'] = summary
    out.mkdir(parents=True)
    (out / 'structural_diff.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', required=True)
    ap.add_argument('--inspect', required=True)
    ap.add_argument('--window', required=True)
    ap.add_argument('--window-index', type=int, required=True)
    ap.add_argument('--camera', default='Main Camera')
    ap.add_argument('--library', required=True)
    ap.add_argument('--captured-cbuffers')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    rep = run(args)
    print(json.dumps(rep['summary']))
    if 'summary_cbuffers' in rep:
        print(json.dumps(rep['summary_cbuffers']))


if __name__ == '__main__':
    main()
