"""Frame-pair probe v5 export (``HS2_FramePairProbe 5.x``) -> generator inputs.

The v5 export adds, per frame, ``renders_NNN.json``: one record per camera
callback (``pre_cull`` / ``pre_render`` / ``post_render``) with the camera
state, global keywords, the native global sheet, GfxDevice built-in params,
lights (transform matrices), renderers (native scene handle / flags, bounds)
and changed material sheets.  A field ``{"unchanged": true}`` means "equal to
the same field of the previous record that carried it".

Materials carry ``native_sheet``: the material ShaderPropertySheet entries with
the raw stored bytes as Unity binds them to GPU constant buffers (colours are
stored linear; ``gamma_flag`` only describes managed getters).  These bytes are
the material constant values themselves; no conversion is applied.

``normalize`` returns a scene dict in the field layout the generators read
(``global_keywords``, ``global_uniforms``, per-material ``_native_values``,
per-light ``properties.local_to_world``), plus ``_v5`` with the resolved
render record of the requested camera.
"""
from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import numpy as np

F = np.float32


class FieldDict(dict):
    """Reflected-field dict that also answers the v4 key spelling.

    Probe v4 prefixed nested struct members with their declaring type
    (``LayerMask.m_Mask``); v5 writes the bare member name (``m_Mask``) inside
    nested structs.  A ``Type.member`` lookup that is absent falls back to the
    bare member (the value is the same reflected field)."""

    def _alt(self, k):
        if isinstance(k, str) and not dict.__contains__(self, k) and '.' in k:
            tail = k.split('.', 1)[1]
            if dict.__contains__(self, tail):
                return tail
        return k

    def __getitem__(self, k):
        return dict.__getitem__(self, self._alt(k))

    def get(self, k, default=None):
        return dict.get(self, self._alt(k), default)

    def __contains__(self, k):
        return dict.__contains__(self, k) or (isinstance(k, str) and '.' in k and
                                              dict.__contains__(self, k.split('.', 1)[1]))


def tolerant(obj):
    if isinstance(obj, dict):
        return FieldDict({k: tolerant(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [tolerant(v) for v in obj]
    return obj


def resolve_records(renders_doc):
    """Records in seq order with ``unchanged`` fields replaced by the last carried value."""
    last = {}
    out = []
    for rec in sorted(renders_doc['records'], key=lambda r: r['seq']):
        data = {}
        for k, v in rec['data'].items():
            if isinstance(v, dict) and v.get('unchanged') is True and len(v) == 1:
                if k not in last:
                    raise KeyError('field %s unchanged before any value' % k)
                data[k] = last[k]
            else:
                data[k] = v
                last[k] = v
        out.append(dict(rec, data=data))
    return out


def render_record(renders_doc, camera, phase='pre_render', render_index=0):
    rows = [r for r in resolve_records(renders_doc) if r['camera']['name'] == camera and r['phase'] == phase]
    if len(rows) <= render_index:
        raise KeyError('%s %s record %d not exported' % (camera, phase, render_index))
    return rows[render_index]


def rowmajor_to_memory(values):
    v = [F(x) for x in values]
    return [v[r * 4 + c] for c in range(4) for r in range(4)]


def sheet_values(sheet):
    """{name: GPU values} of a native sheet (floats / vectors / matrices)."""
    out = {}
    for e in sheet['entries']:
        t = e['type']
        if t == 'texture':
            continue
        raw = bytes.fromhex(e['value_hex'])
        if t in ('float', 'vector', 'matrix'):
            vals = list(np.frombuffer(raw, np.float32))
        elif t == 'int':
            vals = ('int', list(np.frombuffer(raw, np.int32)))
        else:
            raise KeyError('native sheet entry type %s' % t)
        out[e['name']] = vals if t != 'float' or len(vals) != 1 else [vals[0]]
    return out


def sheet_floats(sheet):
    return {e['name']: float(np.frombuffer(bytes.fromhex(e['value_hex']), np.float32)[0])
            for e in sheet['entries'] if e['type'] == 'float'}


def normalize(scene, renders_doc, camera='Main Camera', render_index=0):
    s = copy.deepcopy(scene)
    rec = render_record(renders_doc, camera, 'pre_render', render_index)
    d = rec['data']
    s['global_keywords'] = {k['name']: True for k in d['global_keywords']['enabled']}
    gu = {}
    for e in d['global_sheet']['entries']:
        if e['type'] in ('float', 'vector'):
            v = [float(x) for x in np.frombuffer(bytes.fromhex(e['value_hex']), np.float32)]
            gu[e['name']] = {'value': v if e['type'] == 'vector' else v[0]}
    s['global_uniforms'] = gu
    changed = {}
    smd = {}
    for row in d.get('material_sheets_changed') or []:
        sh = row['sheet']
        if 'entries' in sh:
            changed[row['material']] = sh
        if row.get('shared_material_data'):
            smd[row['material']] = row['shared_material_data']
    for m in s['materials']:
        sd = smd.get(m['id'], m.get('shared_material_data'))
        if sd:
            # probe v6: SharedMaterialData+0xd0 (Material::UpdateHashes 0x84e3b0 ->
            # Pass::ComputePassValuesHash 0x855f70), the shadow-caster sort material key
            m['shadow_caster_hash'] = int(sd['caster_pass_values_hash_0xd0'])
        sheet = changed.get(m['id'], m.get('native_sheet'))
        if sheet is None:
            continue
        m['_native_sheet'] = sheet
        m['_native_values'] = sheet_values(sheet)
        m['_native_floats'] = sheet_floats(sheet)
    lights = {l['id']: l for l in d.get('lights') or []}
    for l in s['lights']:
        src = lights.get(l['id'], l)
        tr = src.get('transform') or l.get('transform')
        if tr:
            l['properties']['local_to_world'] = rowmajor_to_memory(tr['local_to_world_rowmajor'])
            l['properties']['world_rotation_xyzw'] = [F(x) for x in tr['rotation']]
            l['properties']['world_position'] = [F(x) for x in tr['position']]
    for c in s['effect_components']:
        if 'fields' in c:
            c['fields'] = tolerant(c['fields'])
    # the rendering camera as it was at this render's OnPreRender (the scene export is
    # taken once per frame; bridge renders move/jitter the camera and restore it after)
    cp = d.get('camera_properties')
    if cp:
        rows = [c for c in s['cameras'] if c.get('id') == rec['camera']['id']]
        if len(rows) != 1:
            raise KeyError('render record camera %s not in the scene export' % rec['camera']['id'])
        props = dict(rows[0]['properties'])
        props.update(cp)
        rows[0]['properties'] = props
        rows[0]['matrices'] = {}
        rows[0]['_native'] = (d.get('camera_state') or {}).get('native') or {}
        rows[0]['_render_record'] = {'seq': rec['seq'], 'bridge_render': rec.get('bridge_render')}
    # per-render renderer state (bounds as Unity held them at this render's OnPreRender;
    # float32 printed round-trip)
    s['_render_renderers'] = {int(r['id']): r for r in (d.get('renderers') or [])}
    s['_v5'] = {'render_record': rec, 'camera': camera}
    return s


def load(window, index, camera='Main Camera', render_index=0):
    W = Path(window)
    scene = json.loads((W / 'frames' / ('scene_%03d.json' % index)).read_text(encoding='utf-8'))
    renders = json.loads((W / 'frames' / ('renders_%03d.json' % index)).read_text(encoding='utf-8'))
    return normalize(scene, renders, camera, render_index), renders
