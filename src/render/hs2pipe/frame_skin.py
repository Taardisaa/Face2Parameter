"""Head skinned vertex buffers of a paired frame, generated in-frame.

UnityPlayer 2018.4 skins the head surfaces on the CPU and uploads one DYNAMIC
40-byte-vertex buffer per SkinnedMeshRenderer (``gpu_skin_inputs`` module
docstring).  The uploads consumed by the head draws are located with the sibling
``gpu_skin_inputs`` (``simulate`` / ``bind_surfaces`` / ``upload_points``) and
regenerated from the paired probe export with ``paired_skin.unity_exact_vertices``
(UnityPlayer arithmetic executed by ``unity_skin_emu``): the current stream from
frame N, the motion-vector previous stream from frame N-1.  The capture bytes
are used only for the equality check.  Ancestors above ChaControl come from the
same session's geometry export (static Maker hierarchy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.render import gpu_skin_inputs as g
from src.render import paired_skin as ps


def ancestor_chain(win, geometry_path):
    geo = json.loads(Path(geometry_path).read_text(encoding='utf-8'))
    gt = {t['id']: t for t in geo['transforms']}
    top = win.static['chara_transform_id']
    if top not in gt:
        raise g.SkinPathError('ancestor export is not from this process (ChaControl id missing)')
    chain, cur = [], gt[top]['parent_id']
    while cur is not None:
        t = gt[cur]
        chain.append((np.asarray(t['local_position'], np.float32), np.asarray(t['local_rotation_xyzw'], np.float32),
                      np.asarray(t['local_scale'], np.float32)))
        cur = t['parent_id']
    chain.reverse()
    return chain


def head_skin_uploads(xml, window_dir, geometry_path, fxc, work_dir, frame_current, frame_rule=None):
    """{zip.xml chunk index: {bytes, role, surface, frame, captured_equal}} of every
    skinned-vertex upload read by a bound head draw.

    ``frame_rule(role, write) -> frame`` chooses the probe frame of each write; the
    default is the ordinary frame (current stream = ``frame_current``, previous stream =
    ``frame_current - 1``)."""
    from src.render.unity_skin_emu import shared_emulator
    index = g.CaptureIndex(Path(xml))
    writes, draws = g.simulate(index)
    win = ps.ProbeWindow(Path(window_dir))
    surfaces = win.surfaces()
    bindings = g.bind_surfaces(index, draws, surfaces)
    pairs = {(b['skinned_vb'], b['index_buffer']) for b in bindings if b['status'] == 'bound'}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    usage = {vs: g.object_matrix_uses(g.disassemble(index.shader_bytes(vs), Path(fxc), work, f'vs_{vs}'))
             for vs in sorted({d['shaders']['VS'] for d in draws if (d['vbs'][0], d['ib']) in pairs})}
    points, _ = g.upload_points(index, writes, draws, bindings, usage)
    emu = shared_emulator()
    anc = ancestor_chain(win, geometry_path)
    renderers = {r['id']: r for r in win.static['renderers']}
    out = {}
    for p in points:
        if p['role'] not in ('skinned_vertices_current', 'skinned_vertices_previous'):
            continue
        si = p['surfaces'][0]
        rid = surfaces[si]['renderer_id']
        if frame_rule is not None:
            fc = frame_rule(p['role'], writes[p['write_id']])
        else:
            fc = frame_current if p['role'] == 'skinned_vertices_current' else frame_current - 1
        if fc not in win.by_frame:
            raise g.SkinPathError('frame %d not in the probe window' % fc)
        data = ps.unity_exact_vertices(emu, win, win.by_frame[fc], renderers[rid], win.meshes[rid], anc)
        captured = index.blob(writes[p['write_id']]['buffer'])
        out[writes[p['write_id']]['chunk_index']] = {
            'bytes': bytes(data), 'role': p['role'], 'surface': win.meshes[rid]['name'], 'frame': fc,
            'captured_equal': bytes(data) == bytes(captured), 'bytes_len': len(data)}
    return out
