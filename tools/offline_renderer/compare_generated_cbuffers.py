"""Byte-diff harness: generated Unity built-in cbuffer bytes vs a capture.

Two subcommands:

``extract``  CPU interpretation of an official RenderDoc XML+ZIP export through
             ``frame_graph.FrameGraph`` (same Map/Unmap/initial-content rules).
             For every draw/dispatch and every declared constant-buffer slot it
             records the Unity cbuffer name, the installed export's variable
             layout and the exact bytes each variable read at that event.  Only
             bytes inside a variable of the bound program's layout are kept:
             Unity fills a cbuffer per variable of the program layout, so other
             bytes are not determined by that draw.

``compare``  Reads a generator output (``unity_forward.GeneratedCBuffers``
             JSON) and the extracted capture, and diffs every generated
             variable byte-for-byte against the captured variable at the events
             the generated record names.  No tolerance is applied; float
             differences are reported in ULPs and absolute value as evidence,
             never used to pass a variable.

The generator side chooses which variables are *invariant* for a capture taken
in another frame (projection given FOV/aspect/clip planes, shadow settings,
...).  Variables the generator cannot derive from the given scene state are
not generated, so they appear in the report as ``not_generated`` with the
reason the generator gave.
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

from tools.offline_renderer import frame_graph as fg  # noqa: E402

SCHEMA_VERSION = 1
DRAW_KINDS = ('Draw', 'DrawIndexed', 'DrawInstanced', 'DrawIndexedInstanced', 'Dispatch')


def _program_identity(info):
    matches = info.get('source_matches') or []
    return [{'shader': m['shader'], 'subshader': m['subshader'], 'pass': m['pass'], 'pass_name': m.get('pass_name'),
             'stage': m['stage'], 'keywords': m['keywords']} for m in matches]


def extract(graph, passes, segments):
    """Per-event captured bytes of every named cbuffer variable of every bound program."""
    pass_of = {}
    for p in passes:
        for e in p['events']:
            pass_of[e] = p['pass_index']
    segment_of = {}
    for s, seg in enumerate(segments):
        for pi in seg:
            segment_of[pi] = s
    learned = {}
    for res, names in graph.cb_resource_names.items():
        for name, row in names.items():
            learned.setdefault(int(res), {})[name] = row['layout']
    rows = []
    for a in graph.actions:
        if a['kind'] not in DRAW_KINDS:
            continue
        ev = a['event_id']
        pi = pass_of.get(ev)
        row = {'event_id': ev, 'chunk_index': a['chunk_index'], 'kind': a['kind'], 'pass_index': pi,
               'pass_label': passes[pi]['label'] if pi is not None else None,
               'segment': segment_of.get(pi), 'stages': {}}
        if a['kind'] != 'Dispatch':
            row['viewport'] = a['viewports'][0] if a.get('viewports') else None
            row['render_targets'] = [{'resource': t['resource'], 'debug_names': t['debug_names']} for t in a['render_targets']]
            row['depth'] = {'resource': a['depth']['resource'], 'debug_names': a['depth']['debug_names']} if a['depth'] else None
        for stage, info in a.get('shaders', {}).items():
            export = graph.reflections[info['resource']].get('export_names')
            srow = {'shader_resource': info['resource'], 'debug_names': info['debug_names'],
                    'dxbc_sha256': info['dxbc_sha256'], 'programs': _program_identity(info), 'cbuffers': []}
            for cb in info['constant_buffers']:
                entry = {'slot': cb['slot'], 'declared_float4': cb['declared_float4']}
                if cb.get('unbound'):
                    entry['unbound'] = True
                    srow['cbuffers'].append(entry)
                    continue
                name = cb.get('name')
                entry.update(name=name, name_source=cb.get('name_source'), resource=cb['resource'],
                             bytes=cb['bytes'], debug_names=cb.get('debug_names'))
                layout = None
                if name is not None:
                    layout = (export or {}).get('cb_layouts', {}).get(name) or learned.get(cb['resource'], {}).get(name)
                entry['layout_source'] = None if layout is None else (
                    'installed_export' if export and name in export.get('cb_layouts', {}) else 'same_cb_resource_witness')
                entry['variables'] = {}
                if layout is not None:
                    width = graph.resources[cb['resource']]['descriptor']['ByteWidth']
                    snap = fg.BufferContents(width, None, 'none')
                    snap.segments = graph._cb_snapshots[(ev, stage, cb['slot'])]
                    limit = cb['bytes'][1]
                    for var in layout['variables']:
                        s0 = cb['bytes'][0] + var['offset']
                        s1 = s0 + var['size']
                        if s1 > limit:
                            entry['variables'][var['name']] = {'offset': var['offset'], 'size': var['size'],
                                                               'outside_declared_range': True}
                            continue
                        data = snap.read(graph.blobs, s0, s1)
                        entry['variables'][var['name']] = {
                            'offset': var['offset'], 'size': var['size'],
                            'hex': data.hex() if data is not None else None,
                            'producers': [p['producer'] for p in fg.BufferContents.producers(snap, s0, s1)]}
                srow['cbuffers'].append(entry)
            row['stages'][stage] = srow
        rows.append(row)
    return rows


def run_extract(xml, zip_path, inspect, manifests, out, boundary_resource):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Fresh output path required: %s' % out)
    chunks = fg.load_chunks(xml)
    listing = fg.load_driver_listing(inspect) if inspect else None
    graph = fg.FrameGraph(chunks, listing, fg.Blobs(zip_path), fg.ShaderLibrary(manifests)).build()
    validation = graph.validate_events()
    passes = fg.group_passes(graph)
    segments = fg.segment_passes(passes, graph, boundary_resource)
    rows = extract(graph, passes, segments)
    doc = {'schema_version': SCHEMA_VERSION, 'kind': 'captured_cbuffer_variables',
           'provenance': {'xml': str(Path(xml).resolve()), 'xml_sha256': fg.digest(Path(xml).read_bytes()),
                          'zip': str(Path(zip_path).resolve()), 'inspect': str(Path(inspect).resolve()) if inspect else None,
                          'shader_manifests': [str(Path(m).resolve()) for m in manifests],
                          'frame_graph_tool_sha256': fg.digest(Path(fg.__file__).read_bytes()),
                          'tool_sha256': fg.digest(Path(__file__).read_bytes())},
           'event_validation': validation,
           'passes': [{k: p[k] for k in ('pass_index', 'label', 'first_event', 'last_event', 'events', 'viewport',
                                         'camera_view_signatures', 'targets', 'clear_events')} for p in passes],
           'segments': segments, 'draws': rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, default=fg.jsonable), encoding='utf-8')
    return doc


# ----------------------------------------------------------------- compare
def _floats(hexstr):
    data = bytes.fromhex(hexstr)
    return list(struct.unpack('<%df' % (len(data) // 4), data))


def _ulps(a, b):
    ia = struct.unpack('<i', struct.pack('<f', a))[0]
    ib = struct.unpack('<i', struct.pack('<f', b))[0]
    ia = ia if ia >= 0 else -2147483648 - ia
    ib = ib if ib >= 0 else -2147483648 - ib
    return abs(ia - ib)


def diff_variable(generated_hex, captured_hex, kind='float'):
    """Exact byte comparison with per-component evidence (no tolerance)."""
    g = bytes.fromhex(generated_hex)
    c = bytes.fromhex(captured_hex)
    if len(g) != len(c):
        return {'equal': False, 'reason': 'length %d vs %d' % (len(g), len(c))}
    if g == c:
        return {'equal': True}
    comps = []
    for i in range(0, len(g), 4):
        if g[i:i + 4] == c[i:i + 4]:
            continue
        if kind == 'int':
            gv, cv = struct.unpack('<i', g[i:i + 4])[0], struct.unpack('<i', c[i:i + 4])[0]
            comps.append({'component': i // 4, 'generated': gv, 'captured': cv})
        else:
            gv, cv = struct.unpack('<f', g[i:i + 4])[0], struct.unpack('<f', c[i:i + 4])[0]
            comps.append({'component': i // 4, 'generated': gv, 'captured': cv, 'abs': abs(gv - cv), 'ulps': _ulps(gv, cv)})
    return {'equal': False, 'differing_components': comps,
            'max_abs': max((d.get('abs', 0) for d in comps), default=0),
            'max_ulps': max((d.get('ulps', 0) for d in comps), default=0)}


def index_captured(doc):
    """(event, stage, cbuffer name) -> variables."""
    idx = {}
    for d in doc['draws']:
        for stage, s in d['stages'].items():
            for cb in s['cbuffers']:
                if cb.get('name'):
                    idx.setdefault((d['event_id'], stage, cb['name']), cb['variables'])
    return idx


def compare(generated, captured):
    """``generated``: {'records': [{'events': [...], 'stage': 'VS'|'PS'|..|'any', 'cbuffer': name,
    'variables': {name: {'hex', 'kind', 'rule'}}, 'not_generated': {name: reason}}]}."""
    idx = index_captured(captured)
    draws = {d['event_id']: d for d in captured['draws']}
    results = []
    totals = {'compared': 0, 'equal': 0, 'different': 0, 'absent_in_capture': 0}
    for rec in generated['records']:
        out = {'label': rec.get('label'), 'cbuffer': rec['cbuffer'], 'variables': {},
               'not_generated': rec.get('not_generated', {})}
        for ev in rec['events']:
            stages = [rec['stage']] if rec.get('stage', 'any') != 'any' else sorted(draws[ev]['stages']) if ev in draws else []
            for stage in stages:
                variables = idx.get((ev, stage, rec['cbuffer']))
                if variables is None:
                    continue
                for name, g in rec['variables'].items():
                    cap = variables.get(name)
                    slot = out['variables'].setdefault(name, {'rule': g.get('rule'), 'equal_events': [],
                                                              'different': [], 'absent_events': []})
                    if cap is None or cap.get('hex') is None:
                        slot['absent_events'].append([ev, stage])
                        totals['absent_in_capture'] += 1
                        continue
                    totals['compared'] += 1
                    d = diff_variable(g['hex'], cap['hex'], g.get('kind', 'float'))
                    if d['equal']:
                        totals['equal'] += 1
                        slot['equal_events'].append([ev, stage])
                    else:
                        totals['different'] += 1
                        slot['different'].append({'event': ev, 'stage': stage, **d})
        for slot in out['variables'].values():
            slot['equal_count'] = len(slot.pop('equal_events'))
            slot['absent_count'] = len(slot.pop('absent_events'))
            slot['different_count'] = len(slot['different'])
            slot['different'] = slot['different'][:8]
        results.append(out)
    return {'schema_version': SCHEMA_VERSION, 'totals': totals, 'records': results,
            'generator_provenance': generated.get('provenance'), 'capture_provenance': captured.get('provenance')}


# ------------------------------------------- retry-capture diagnostic inputs
def _var(draw, name, stage=None, cbuffer=None):
    for st, s in draw['stages'].items():
        if stage and st != stage:
            continue
        for cb in s['cbuffers']:
            if cbuffer and cb.get('name') != cbuffer:
                continue
            v = (cb.get('variables') or {}).get(name)
            if v and v.get('hex'):
                return bytes.fromhex(v['hex'])
    return None


def _is_perspective(vp):
    """Row 3 of an ortho VP is (0,0,0,1); any other row 3 is a perspective projection."""
    return not (vp[3] == 0 and vp[7] == 0 and vp[11] == 0 and vp[15] == 1)


def _floats32(data):
    from src.render.hs2pipe import unity_math as um
    return um.from_bytes(data)


def retry_diagnostic_records(captured, inputs):
    """Generated records for the 2026-10-12 retry capture, which has no paired
    scene state.  Every generated variable is derived by a recovered rule from
    inputs that are *not* the variable itself; each input carries its
    provenance class:

    * ``bridge_source``  - HS2_McpBridge MakerRenderStyleService.Capture code
      (aspect = (float)w/h, camera position formula, render into texture);
    * ``other_session``  - another process' runtime export (fov/near/far,
      ambient probe, light strength): same scene configuration, other frame;
    * ``capture_input``  - a different variable of the same capture used as an
      input (unity_MatrixV, CTAA jitter from VP column 2, O2W for W2O,
      shadow distance from _LightShadowData.y): labelled, never the compared one.
    """
    from src.render.hs2pipe import unity_math as um
    from src.render.hs2pipe import unity_forward as uf
    from src.render.hs2pipe.scene_state import CameraRender
    F = np.float32
    records = []
    graphics = {'reversed_z': True, 'uses_opengl_texture_coords': False}
    draws = captured['draws']
    # Views: per segment and viewport size, the perspective VS unity_MatrixVP of
    # that camera render (one per camera per segment); camera-level variables
    # are compared on every draw of that segment and viewport size.
    cams = inputs['cameras']
    by_seg = {}
    for d in draws:
        vw = d.get('viewport') or {}
        key = '%dx%d' % (int(vw.get('Width', 0)), int(vw.get('Height', 0)))
        if key in cams and d.get('segment') is not None:
            by_seg.setdefault((d['segment'], key), []).append(d)
    for (seg, key), group in sorted(by_seg.items()):
        persp = {}
        for d in group:
            vpb = _var(d, 'unity_MatrixVP', 'VS', 'UnityPerFrame')
            if vpb is not None and _is_perspective(_floats32(vpb)):
                persp.setdefault(vpb, []).append(d)
        if len(persp) != 1:
            records.append({'label': 'camera %s segment %s: %d perspective views, not checked' % (key, seg, len(persp)),
                            'cbuffer': 'UnityPerFrame', 'stage': 'any', 'events': [], 'variables': {}})
            continue
        vp_bytes, vgroup = next(iter(persp.items()))
        vp = _floats32(vp_bytes)
        cin = cams[key]
        view = next((v for v in (_var(d, 'unity_MatrixV', 'VS', 'UnityPerFrame') for d in vgroup) if v is not None), None)
        if view is None:
            continue
        aspect = F(cin['pixel_width']) / F(cin['pixel_height'])
        base = um.set_perspective(cin['fov'], aspect, cin['near'], cin['far'])
        proj = list(base)
        implicit = bool(cin.get('implicit_projection'))
        if not implicit:
            proj[8] = vp[8]                             # capture_input: CTAA jitter (GL m02)
            proj[9] = -vp[9]                            # GL m12 (device row 1 is negated)
        cam = CameraRender({'implicit_projection': implicit, 'projection_matrix': [float(x) for x in proj],
                            'non_jittered_projection_matrix': [float(x) for x in base],
                            'world_to_camera_matrix': [float(x) for x in _floats32(view)],
                            'render_into_texture': True, 'near_clip': cin['near'], 'far_clip': cin['far'],
                            'field_of_view': cin['fov'], 'aspect': float(aspect),
                            'orthographic_size': cin['orthographic_size'], 'orthographic': False,
                            'position': cin['position'],
                            'target': {'pixel_width': cin['pixel_width'], 'pixel_height': cin['pixel_height']}})
        setup = uf.camera_setup(cam, graphics)
        props = uf.camera_shader_props(cam, setup, graphics)
        def g(names):
            return {n: {'hex': uf.encode(props[n]).hex(), 'rule': 'camera_setup/camera_shader_props'} for n in names}
        inputs_note = {'fov/near/far/ortho_size': 'other_session', 'aspect/position/render_into_texture': 'bridge_source',
                       'unity_MatrixV': 'capture_input',
                       'projection': 'implicit SetPerspective' if implicit else 'SetPerspective + CTAA jitter m02/m12 from VP column 2 (capture_input)'}
        label = 'camera %s segment %d' % (key, seg)
        records.append({'label': label, 'cbuffer': 'UnityPerFrame', 'stage': 'any', 'events': [d['event_id'] for d in vgroup],
                        'variables': g(('unity_MatrixVP', 'unity_MatrixInvV')), 'inputs': inputs_note})
        all_events = [d['event_id'] for d in group]
        records.append({'label': label, 'cbuffer': 'UnityPerCamera', 'stage': 'any', 'events': all_events,
                        'variables': g(('_ProjectionParams', '_ZBufferParams', '_ScreenParams', 'unity_OrthoParams',
                                        '_WorldSpaceCameraPos')), 'inputs': inputs_note})
        records.append({'label': label, 'cbuffer': 'UnityPerCameraRare', 'stage': 'any', 'events': all_events,
                        'variables': g(('unity_CameraInvProjection', 'unity_CameraToWorld')), 'inputs': inputs_note})
        mv = [d['event_id'] for d in group if d.get('pass_label') == 'motion_vectors']
        if mv:
            records.append({'label': 'motion vectors ' + label, 'cbuffer': '$Globals', 'stage': 'any', 'events': mv,
                            'variables': {'_NonJitteredVP': {'hex': uf.pack(setup.non_jittered_vp).hex(),
                                                             'rule': 'devProj(nonJittered)·V'}},
                            'inputs': inputs_note})
    # Per-draw W2O from the same draw's O2W (all draws whose VS layout has both).
    w2o = []
    for d in draws:
        o = _var(d, 'unity_ObjectToWorld', 'VS', 'UnityPerDraw')
        if o is None or _var(d, 'unity_WorldToObject', 'VS', 'UnityPerDraw') is None:
            continue
        rec = uf.per_draw_values({'object_to_world': [float(x) for x in _floats32(o)]})
        w2o.append({'label': 'W2O event %d' % d['event_id'], 'cbuffer': 'UnityPerDraw', 'stage': 'VS',
                    'events': [d['event_id']],
                    'variables': {'unity_WorldToObject': {'hex': uf.pack(rec['unity_WorldToObject']).hex(),
                                                          'rule': 'InvertMatrix4x4_General3D(O2W)'}},
                    'inputs': {'unity_ObjectToWorld': 'capture_input (same draw)'}})
    records.extend(w2o)
    # Lighting: SH from the other-session ambient probe (static Maker scene setting).
    rs = inputs['render_settings']
    sh = uf.sh_shader_constants(uf.final_ambient_probe(rs))
    sh_events = sorted({d['event_id'] for d in draws if _var(d, 'unity_SHAr', None, 'UnityLighting') is not None
                        or _var(d, 'unity_SHBr', None, 'UnityLighting') is not None})
    records.append({'label': 'SH (ambient probe, no light probes)', 'cbuffer': 'UnityLighting', 'stage': 'any',
                    'events': sh_events,
                    'variables': {k: {'hex': uf.pack(v).hex(), 'rule': 'UpdateFinalAmbientProbe+GetShaderConstantsFromNormalizedSH'}
                                  for k, v in sh.items()},
                    'inputs': {'ambient_probe/intensity/mode': 'other_session (runtime_audit_hook_v3)'}})
    # Forward light colour (_LightColor0, $Globals) from the other-session light colour/intensity.
    lc = inputs.get('forward_light')
    if lc:
        col = uf.light_final_color(lc)
        lc_events = sorted({d['event_id'] for d in draws if _var(d, '_LightColor0', None, '$Globals') is not None})
        records.append({'label': '_LightColor0', 'cbuffer': '$Globals', 'stage': 'any', 'events': lc_events,
                        'variables': {'_LightColor0': {'hex': uf.pack(col).hex(), 'rule': 'PrecalcLightColor (gamma intensity)'}},
                        'inputs': {'color/intensity': 'other_session (Directional Light Key)',
                                   'lightsUseLinearIntensity': 'installed GraphicsSettings asset (false)'}})
    # Shadows: _LightShadowData and shadow-map texel size.
    sl = inputs['shadow_light']
    lsd_events = sorted({d['event_id'] for d in draws if _var(d, '_LightShadowData', None, 'UnityShadows') is not None})
    lsd = uf.light_shadow_data({'shadow_strength': sl['shadow_strength']}, F(sl['shadow_distance']), F(sl['camera_far']))
    records.append({'label': '_LightShadowData', 'cbuffer': 'UnityShadows', 'stage': 'any', 'events': lsd_events,
                    'variables': {'_LightShadowData': {'hex': uf.pack(lsd).hex(), 'rule': 'CalculateLightShadowFade'}},
                    'inputs': {'shadow_strength': 'other_session', 'shadow_distance': 'capture_input (_LightShadowData.y=far/d of another draw class)',
                               'camera_far': 'other_session'}})
    tex_events = []
    for d in draws:
        if _var(d, '_ShadowMapTexture_TexelSize', 'PS', '$Globals') is not None:
            tex_events.append((d['event_id'], (d.get('viewport') or {})))
    for ev, vpd in tex_events:
        size = uf.shadow_map_size(inputs['graphics'], inputs['quality'], {'type': 'Directional', 'shadow_resolution': -1,
                                                                          'shadow_custom_resolution': 0},
                                  int(vpd.get('Width')), int(vpd.get('Height')))
        records.append({'label': 'shadow map texel size event %d' % ev, 'cbuffer': '$Globals', 'stage': 'PS', 'events': [ev],
                        'variables': {'_ShadowMapTexture_TexelSize': {
                            'hex': uf.pack([F(1.0) / F(size), F(1.0) / F(size), F(size), F(size)]).hex(),
                            'rule': 'CalculateShadowMapSize + RenderTexture texel size'}},
                        'inputs': {'shadow_resolution': 'installed QualitySettings asset level 4 (VeryHigh)',
                                   'video_memory/max RT size': 'assumed >=480MB / 16384 (D3D11 caps), labelled assumption',
                                   'viewport': 'capture_input'}})
    return records


# --------------------------------------------- paired capture (same-frame state)
CAMERA_PASS_LABELS = ('depth_texture_prepass', 'depth_normals', 'shadow_map', 'screenspace_shadows',
                      'forward_scene', 'motion_vectors')


def unity_row_major_to_memory(values):
    """The probe exports Matrix4x4 as m00,m01,m02,m03,m10,... (row-major);
    Unity/HLSL memory order is column-major."""
    v = list(values)
    if len(v) != 16:
        raise ValueError('16 matrix values required')
    return [v[r * 4 + c] for c in range(4) for r in range(4)]


def _enum_value(text):
    return text.split('.', 1)[1].split('=')[0] if isinstance(text, str) and '=' in text else text


def paired_records(captured, scene, camera_name='Main Camera'):
    """Generated records for a paired capture from the probe's same-frame scene
    export (``window_*/frames/scene_NNN.json`` of the marker frame)."""
    from src.render.hs2pipe import unity_math as um
    from src.render.hs2pipe import unity_forward as uf
    from src.render.hs2pipe.scene_state import CameraRender
    F = np.float32
    st = scene['statics']
    sysinfo = st['UnityEngine.SystemInfo']
    q = st['UnityEngine.QualitySettings']
    rs = st['UnityEngine.RenderSettings']
    gs = st['UnityEngine.Rendering.GraphicsSettings']
    graphics = {'reversed_z': bool(sysinfo['usesReversedZBuffer']), 'uses_opengl_texture_coords': False,
                'video_memory_mb': int(sysinfo['graphicsMemorySize']), 'max_render_texture_size': int(sysinfo['maxTextureSize'])}
    cams = [c for c in scene['cameras'] if c['name'] == camera_name and c['active_in_hierarchy'] and c['properties']['enabled']]
    if len(cams) != 1:
        raise ValueError('expected exactly one enabled camera named %s' % camera_name)
    cp = cams[0]['properties']
    mats = cams[0].get('matrices', {})
    proj = unity_row_major_to_memory(mats.get('projectionMatrix', cp['projectionMatrix']))
    nj = unity_row_major_to_memory(mats.get('nonJitteredProjectionMatrix', cp['nonJitteredProjectionMatrix']))
    view = unity_row_major_to_memory(mats.get('worldToCameraMatrix', cp['worldToCameraMatrix']))
    c2w = unity_row_major_to_memory(mats.get('cameraToWorldMatrix', cp['cameraToWorldMatrix']))
    implicit_proj = um.set_perspective(cp['fieldOfView'], cp['aspect'], cp['nearClipPlane'], cp['farClipPlane'])
    explicit = um.to_bytes([F(v) for v in proj]) != um.to_bytes(implicit_proj)
    rect = cp['pixelRect']
    into_rt = bool(cp.get('forceIntoRenderTexture')) or cp.get('targetTexture') is not None
    cam = CameraRender({'implicit_projection': not explicit, 'projection_matrix': proj, 'non_jittered_projection_matrix': nj,
                        'world_to_camera_matrix': view, 'render_into_texture': into_rt,
                        'near_clip': cp['nearClipPlane'], 'far_clip': cp['farClipPlane'], 'field_of_view': cp['fieldOfView'],
                        'aspect': cp['aspect'], 'orthographic': cp['orthographic'], 'orthographic_size': cp['orthographicSize'],
                        'position': [c2w[12], c2w[13], c2w[14]],
                        'target': {'pixel_width': int(rect[2]), 'pixel_height': int(rect[3])}})
    setup = uf.camera_setup(cam, graphics)
    props = uf.camera_shader_props(cam, setup, graphics)
    inputs_note = {'camera': 'scene export Camera.projectionMatrix/nonJittered/worldToCamera (row-major -> memory)',
                   'implicit_projection': 'inferred: exported projection %s SetPerspective(fov,aspect,clip)' % ('!=' if explicit else '=='),
                   'render_into_texture': 'Camera.forceIntoRenderTexture/targetTexture',
                   '_WorldSpaceCameraPos': 'Camera.cameraToWorldMatrix column 3 (transform.position not exported)'}
    draws = [d for d in captured['draws'] if d.get('pass_label') in CAMERA_PASS_LABELS
             and (d.get('viewport') or {}).get('Width') == float(rect[2]) and (d.get('viewport') or {}).get('Height') == float(rect[3])]
    persp = [d for d in draws if _var(d, 'unity_MatrixVP', 'VS', 'UnityPerFrame') is not None
             and _is_perspective(_floats32(_var(d, 'unity_MatrixVP', 'VS', 'UnityPerFrame')))]
    records = []

    def g(names):
        return {n: {'hex': uf.encode(props[n]).hex(), 'rule': 'camera_setup/camera_shader_props'} for n in names}
    persp_main = [d for d in persp if d.get('pass_label') != 'shadow_map']
    records.append({'label': 'camera ' + camera_name, 'cbuffer': 'UnityPerFrame', 'stage': 'any',
                    'events': [d['event_id'] for d in persp_main],
                    'variables': g(('unity_MatrixVP', 'unity_MatrixInvV', 'unity_MatrixV')), 'inputs': inputs_note})
    ev_all = [d['event_id'] for d in draws]
    records.append({'label': 'camera ' + camera_name, 'cbuffer': 'UnityPerCamera', 'stage': 'any', 'events': ev_all,
                    'variables': g(('_ProjectionParams', '_ZBufferParams', '_ScreenParams', 'unity_OrthoParams',
                                    '_WorldSpaceCameraPos')), 'inputs': inputs_note})
    records.append({'label': 'camera ' + camera_name, 'cbuffer': 'UnityPerCameraRare', 'stage': 'any', 'events': ev_all,
                    'variables': g(('unity_CameraInvProjection', 'unity_CameraToWorld', 'unity_WorldToCamera',
                                    'unity_CameraProjection')), 'inputs': inputs_note})
    sky = [d['event_id'] for d in draws if any('Skybox' in n for s_ in d['stages'].values() for n in s_.get('debug_names', []))]
    if sky:
        sk = uf.skybox_matrices(cam, setup, graphics)
        records.append({'label': 'skybox ' + camera_name, 'cbuffer': 'UnityPerFrame', 'stage': 'any', 'events': sky,
                        'variables': {'unity_MatrixVP': {'hex': uf.pack(sk['unity_MatrixVP']).hex(), 'rule': 'RenderSkybox/GetSkyboxProjectionMatrix'}},
                        'inputs': inputs_note})
        records.append({'label': 'skybox ' + camera_name, 'cbuffer': 'UnityPerDraw', 'stage': 'any', 'events': sky,
                        'variables': {'unity_ObjectToWorld': {'hex': uf.pack(sk['unity_ObjectToWorld']).hex(), 'rule': 'RenderSkybox scale far*10'}},
                        'inputs': inputs_note})
        persp_main = [d for d in persp_main if d['event_id'] not in sky]
        records[0]['events'] = [d['event_id'] for d in persp_main]
    mv = [d['event_id'] for d in draws if d.get('pass_label') == 'motion_vectors']
    prev = unity_row_major_to_memory(cp['previousViewProjectionMatrix'])
    records.append({'label': 'motion vectors ' + camera_name, 'cbuffer': '$Globals', 'stage': 'any', 'events': mv,
                    'variables': {'_NonJitteredVP': {'hex': uf.pack(setup.non_jittered_vp).hex(), 'rule': 'devProj(nonJittered)·V'},
                                  '_PreviousVP': {'hex': uf.pack([F(v) for v in prev]).hex(),
                                                  'rule': 'pass-through of Camera previous VP (SetupMotionMatrices reads Camera+0x598)'}},
                    'inputs': inputs_note})
    # W2O from each draw's own O2W.
    for d in captured['draws']:
        o = _var(d, 'unity_ObjectToWorld', 'VS', 'UnityPerDraw')
        if o is None or _var(d, 'unity_WorldToObject', 'VS', 'UnityPerDraw') is None:
            continue
        rec = uf.per_draw_values({'object_to_world': [float(x) for x in _floats32(o)]})
        records.append({'label': 'W2O event %d' % d['event_id'], 'cbuffer': 'UnityPerDraw', 'stage': 'VS', 'events': [d['event_id']],
                        'variables': {'unity_WorldToObject': {'hex': uf.pack(rec['unity_WorldToObject']).hex(),
                                                              'rule': 'InvertMatrix4x4_General3D(O2W)'}},
                        'inputs': {'unity_ObjectToWorld': 'capture_input (same draw)'}})
    # Ambient SH and per-renderer interpolated SH (identical without light probes).
    rset = {'ambient_mode': _enum_value(rs['ambientMode']), 'has_skybox_material': rs.get('skybox') is not None,
            'ambient_intensity': rs['ambientIntensity'], 'ambient_probe': rs['ambientProbe']}
    sh = uf.sh_shader_constants(uf.final_ambient_probe(rset))
    sh_events = sorted({d['event_id'] for d in captured['draws'] if _var(d, 'unity_SHAr', None, 'UnityLighting') is not None
                        or _var(d, 'unity_SHBr', None, 'UnityLighting') is not None})
    records.append({'label': 'SH (final ambient probe)', 'cbuffer': 'UnityLighting', 'stage': 'any', 'events': sh_events,
                    'variables': {k: {'hex': uf.pack(v).hex(), 'rule': 'UpdateFinalAmbientProbe+GetShaderConstantsFromNormalizedSH'}
                                  for k, v in sh.items()}, 'inputs': {'RenderSettings': 'scene export'}})
    interp = {json.dumps(r['interpolated_sh']) for r in scene['renderers'] if r.get('interpolated_sh')}
    for i, sh_json in enumerate(sorted(interp)):
        shv = uf.sh_shader_constants([F(v) for v in json.loads(sh_json)])
        records.append({'label': 'SH (LightProbes.GetInterpolatedProbe variant %d)' % i, 'cbuffer': 'UnityLighting', 'stage': 'any',
                        'events': sh_events, 'variables': {k: {'hex': uf.pack(v).hex(), 'rule': 'GetShaderConstantsFromNormalizedSH'}
                                                           for k, v in shv.items()},
                        'inputs': {'interpolated_sh': 'scene export renderers[].interpolated_sh'}})
    # Forward main light.
    lights = [l for l in scene['lights'] if l['active_in_hierarchy'] and l['properties']['enabled']
              and _enum_value(l['properties']['type']) == 'Directional']
    sun = rs.get('sun') or {}
    key = [l for l in lights if l['id'] == sun.get('ref_id')]
    if key:
        lp = key[0]['properties']
        col = uf.light_final_color({'color': lp['color'], 'intensity': lp['intensity']},
                                   gs['lightsUseLinearIntensity'], gs['lightsUseColorTemperature'])
        lc_events = sorted({d['event_id'] for d in captured['draws'] if _var(d, '_LightColor0', None, '$Globals') is not None})
        records.append({'label': '_LightColor0 (RenderSettings.sun)', 'cbuffer': '$Globals', 'stage': 'any', 'events': lc_events,
                        'variables': {'_LightColor0': {'hex': uf.pack(col).hex(), 'rule': 'PrecalcLightColor'}},
                        'inputs': {'light': 'scene export Light %d' % key[0]['id']}})
        sd = uf.shadow_distance_for_camera({'shadow_distance': q['shadowDistance']}, cam)
        lsd = uf.light_shadow_data({'shadow_strength': lp['shadowStrength']}, sd, cam.f('far_clip'))
        lsd_events = sorted({d['event_id'] for d in captured['draws'] if _var(d, '_LightShadowData', None, 'UnityShadows') is not None})
        records.append({'label': '_LightShadowData', 'cbuffer': 'UnityShadows', 'stage': 'any', 'events': lsd_events,
                        'variables': {'_LightShadowData': {'hex': uf.pack(lsd).hex(), 'rule': 'CalculateLightShadowFade'}},
                        'inputs': {'light/quality/camera': 'scene export'}})
        size = uf.shadow_map_size(graphics, {'shadow_resolution': _enum_value(q['shadowResolution'])},
                                  {'type': 'Directional', 'shadow_resolution': -1 if 'FromQuality' in str(lp['shadowResolution']) else
                                   ('Low', 'Medium', 'High', 'VeryHigh').index(_enum_value(lp['shadowResolution'])),
                                   'shadow_custom_resolution': max(0, int(lp['shadowCustomResolution']))},
                                  int(rect[2]), int(rect[3]))
        tex_events = [d['event_id'] for d in captured['draws'] if _var(d, '_ShadowMapTexture_TexelSize', 'PS', '$Globals') is not None]
        records.append({'label': 'shadow map texel size', 'cbuffer': '$Globals', 'stage': 'PS', 'events': tex_events,
                        'variables': {'_ShadowMapTexture_TexelSize': {
                            'hex': uf.pack([F(1.0) / F(size), F(1.0) / F(size), F(size), F(size)]).hex(),
                            'rule': 'CalculateShadowMapSize'}}, 'inputs': {'quality/light/SystemInfo/viewport': 'scene export'}})
    return records


def run_paired(captured_path, scene_path, out_dir, camera_name='Main Camera'):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError('Fresh output directory required')
    captured = json.loads(Path(captured_path).read_text(encoding='utf-8'))
    scene = json.loads(Path(scene_path).read_text(encoding='utf-8'))
    records = paired_records(captured, scene, camera_name)
    generated = {'schema_version': SCHEMA_VERSION, 'kind': 'generated_cbuffer_variables',
                 'provenance': {'generator': 'src/render/hs2pipe/unity_forward.py', 'scene': str(Path(scene_path).resolve()),
                                'scene_sha256': fg.digest(Path(scene_path).read_bytes()), 'camera': camera_name},
                 'records': records}
    report = compare(generated, captured)
    out_dir.mkdir(parents=True)
    (out_dir / 'generated.json').write_text(json.dumps(generated), encoding='utf-8')
    (out_dir / 'report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def run_retry_diagnostic(captured_path, inputs_path, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError('Fresh output directory required')
    captured = json.loads(Path(captured_path).read_text(encoding='utf-8'))
    inputs = json.loads(Path(inputs_path).read_text(encoding='utf-8'))
    records = retry_diagnostic_records(captured, inputs)
    generated = {'schema_version': SCHEMA_VERSION, 'kind': 'generated_cbuffer_variables',
                 'provenance': {'generator': 'src/render/hs2pipe/unity_forward.py', 'inputs': str(Path(inputs_path).resolve()),
                                'scope': 'retry capture without paired scene state: rule checks from non-circular inputs'},
                 'records': records}
    report = compare(generated, captured)
    out_dir.mkdir(parents=True)
    (out_dir / 'generated.json').write_text(json.dumps(generated), encoding='utf-8')
    (out_dir / 'report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('retry-diagnostic', help='rule checks on the 2026-10-12 retry capture (no paired state)')
    r.add_argument('--captured', type=Path, required=True)
    r.add_argument('--inputs', type=Path, required=True)
    r.add_argument('--out', type=Path, required=True)
    e = sub.add_parser('extract')
    e.add_argument('--xml', type=Path, required=True)
    e.add_argument('--zip', type=Path)
    e.add_argument('--inspect', type=Path)
    e.add_argument('--shader-manifest', type=Path, action='append', default=[])
    e.add_argument('--segment-boundary-resource', type=int)
    e.add_argument('--out', type=Path, required=True)
    c = sub.add_parser('compare')
    c.add_argument('--generated', type=Path, required=True)
    c.add_argument('--captured', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    pr = sub.add_parser('paired', help='paired capture with same-frame scene export')
    pr.add_argument('--captured', type=Path, required=True)
    pr.add_argument('--scene', type=Path, required=True)
    pr.add_argument('--camera', default='Main Camera')
    pr.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == 'paired':
        report = run_paired(args.captured, args.scene, args.out, args.camera)
        print(json.dumps(report['totals']))
        return
    if args.cmd == 'retry-diagnostic':
        report = run_retry_diagnostic(args.captured, args.inputs, args.out)
        print(json.dumps(report['totals']))
        return
    if args.cmd == 'extract':
        doc = run_extract(args.xml, args.zip or args.xml.with_suffix(''), args.inspect, args.shader_manifest, args.out,
                          args.segment_boundary_resource)
        print(json.dumps({'draws': len(doc['draws']), 'passes': len(doc['passes'])}))
    else:
        if args.out.exists():
            raise FileExistsError('Fresh output path required')
        report = compare(json.loads(args.generated.read_text(encoding='utf-8')),
                         json.loads(args.captured.read_text(encoding='utf-8')))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
        print(json.dumps(report['totals']))


if __name__ == '__main__':
    main()
