"""Success checks of one FramePairProbe v5 window against its RenderDoc capture (CPU only).

Inputs: the official ``zip.xml`` export of the RDC the probe's TriggerCapture saved,
the probe window directory (OutDir of the request) and the nonce. Checks, each
recorded with its evidence in ``verification.json`` (nothing is relaxed; a failed
check fails the window):

1. ``capture_receipt.json``: not aborted, TriggerCapture fired, exactly one new capture.
2. Exactly one in-frame v01 end-of-frame marker with this nonce; its frame equals the
   receipt's ``expected_capture_frame`` (trigger frame + 2: failed first attempt +
   automatic retry) and the window has frame/scene/renders exports of that frame.
3. ``renders_<k>.json`` of the captured index holds only records of that frame.
4. Raw texel export: every job ``complete`` or ``skipped`` with a stated reason; none
   pending or failed; every completed job has ``descriptor.json``.
5. Mode ``renderstyle`` only:
   * the receipt lists ``bridge_expected_renders`` bridge renders, ordinals 0..n-1,
     all in the captured frame;
   * exactly one HS2PAIRBRIDGEv05 begin (kind 1) and one end (kind 2) marker with
     this nonce in the capture, both naming the captured frame, begin before end and
     both before the v01 end-of-frame marker;
   * the bridge HTTP result is 200 (directly or through /operations polling) and the
     PNG it names exists.

    python tools/offline_renderer/frame_pair_probe/verify_probe_v5_session.py \
        --xml CAP.zip.xml --window E:/.../window_<nonce> --nonce <hex32> --out FRESH_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAGIC_FRAME = b'HS2PAIRMARKERv01'
MAGIC_BRIDGE = b'HS2PAIRBRIDGEv05'
TAIL = b'HS2PAIRMARKEREND'
BRIDGE_KINDS = {1: 'begin', 2: 'end'}


def find_markers(index, nonce: bytes, magic: bytes):
    """Every in-frame 64-byte probe record of one magic (same layout as paired_skin.find_markers)."""
    rows = []
    for c in index.chunks:
        if c.index <= index.begin.index:
            continue
        for ref in c.buffers:
            if ref.length < 64:
                continue
            data = index.blob(ref)
            at = data.find(magic)
            while at >= 0:
                rec = data[at:at + 64]
                if len(rec) == 64 and rec[48:64] == TAIL:
                    frame = int.from_bytes(rec[32:36], 'little', signed=True)
                    rows.append({'chunk_index': c.index, 'chunk_name': c.name, 'nonce_ok': rec[16:32] == nonce,
                                 'frame_count': frame, 'word36': int.from_bytes(rec[36:40], 'little', signed=True),
                                 'word40': int.from_bytes(rec[40:44], 'little', signed=True),
                                 'integrity_ok': int.from_bytes(rec[44:48], 'little', signed=True) == ~frame})
                at = data.find(magic, at + 1)
    return rows


def verify(index, window: Path, nonce_hex: str):
    nonce = bytes.fromhex(nonce_hex)
    receipt = json.loads((window / 'capture_receipt.json').read_text(encoding='utf-8'))
    checks = []

    def check(name, ok, evidence):
        checks.append({'check': name, 'ok': bool(ok), 'evidence': evidence})

    observations = []

    def observe(name, value):
        # Recorded game facts (e.g. the hook order); never part of acceptance.
        observations.append({'observation': name, 'value': value})

    check('receipt_nonce', receipt.get('nonce') == nonce_hex, receipt.get('nonce'))
    check('not_aborted', receipt.get('aborted_before_trigger') is False, receipt.get('aborted_before_trigger'))
    check('triggered', receipt.get('triggered_frame', -1) > 0, receipt.get('triggered_frame'))
    check('one_new_capture', receipt.get('captures_after', 0) - receipt.get('captures_before', 0) == 1 and len(receipt.get('captures', [])) == 1,
          receipt.get('captures'))
    expected = receipt.get('expected_capture_frame')
    frames = receipt.get('frame_counts_by_index', [])
    check('expected_frame_is_trigger_plus_two', expected == receipt.get('triggered_frame', -9) + 2, [receipt.get('triggered_frame'), expected])

    own = [m for m in find_markers(index, nonce, MAGIC_FRAME) if m['nonce_ok']]
    check('one_v01_marker', len(own) == 1 and own[0]['integrity_ok'], own)
    captured = own[0]['frame_count'] if len(own) == 1 else None
    check('marker_frame_equals_expected', captured == expected, [captured, expected])
    k = frames.index(captured) if captured in frames else None
    check('window_has_captured_frame', k is not None, k)
    if k is not None:
        for name in ('frame_%03d.json', 'scene_%03d.json', 'renders_%03d.json'):
            check('export_' + name.split('_')[0], (window / 'frames' / (name % k)).is_file(), name % k)
        renders = json.loads((window / 'frames' / ('renders_%03d.json' % k)).read_text(encoding='utf-8'))
        bad = [r['seq'] for r in renders['records'] if r['frame_count'] != captured]
        check('renders_single_frame', renders['frame_count'] == captured and not bad, {'file_frame': renders['frame_count'], 'other_frame_seqs': bad[:20]})

    raw = json.loads((window / 'raw_textures.json').read_text(encoding='utf-8'))
    jobs = raw.get('jobs', [])
    bad_jobs = [j for j in jobs if j['status'] not in ('complete', 'skipped') or (j['status'] == 'skipped' and not j.get('error'))]
    check('raw_jobs_resolved', jobs and not bad_jobs, {'counts': raw.get('counts'), 'bad': bad_jobs[:20]})
    missing = [j['path'] for j in jobs if j['status'] == 'complete' and not (Path(j['path']) / 'descriptor.json').is_file()]
    check('raw_descriptors_present', not missing, missing[:20])
    check('raw_export_before_captured_frame', raw.get('queued_frame') == (captured - 1 if captured else None), [raw.get('queued_frame'), captured])

    if receipt.get('mode') == 'renderstyle':
        n = receipt.get('bridge_expected_renders')
        br = receipt.get('bridge_renders', [])
        check('bridge_render_count', n in (1, 16) and len(br) == n, [n, len(br)])
        check('bridge_ordinals', [b['ordinal'] for b in br] == list(range(len(br))), [b['ordinal'] for b in br])
        check('bridge_renders_in_captured_frame', br and all(b['frame_count'] == captured for b in br), sorted({b['frame_count'] for b in br}))
        bm = [m for m in find_markers(index, nonce, MAGIC_BRIDGE) if m['nonce_ok']]
        kinds = {BRIDGE_KINDS.get(m['word36'], 'unknown'): m for m in bm}
        check('bridge_markers', len(bm) == 2 and set(kinds) == {'begin', 'end'} and all(m['integrity_ok'] for m in bm), bm)
        if len(bm) == 2 and set(kinds) == {'begin', 'end'} and len(own) == 1:
            b, e = kinds['begin'], kinds['end']
            check('bridge_markers_frame', b['frame_count'] == e['frame_count'] == captured, [b['frame_count'], e['frame_count'], captured])
            check('bridge_marker_order', b['chunk_index'] < e['chunk_index'] < own[0]['chunk_index'],
                  [b['chunk_index'], e['chunk_index'], own[0]['chunk_index']])
            check('bridge_end_ordinal', e['word40'] == n - 1, e['word40'])
        bridge = receipt.get('bridge') or {}
        final = bridge.get('final_http_status')
        body = bridge.get('final_http_body') or ''
        ok_status = final == '200'
        result = None
        if ok_status:
            doc = json.loads(body)
            result = doc.get('result', doc) if 'operation_id' in doc else doc
            if 'status' in doc and doc.get('status') != 'succeeded':
                ok_status = False
        check('bridge_http_ok', ok_status, {'initial': bridge.get('initial_http_status'), 'final': final})
        png = bridge.get('png_path')
        check('bridge_png_written', png and Path(png).is_file(), png)
        check('bridge_result_render_count', bool(result) and (result.get('capture') or {}).get('temporal_render_count') == n,
              (result or {}).get('capture', {}).get('temporal_render_count') if result else None)
    probe = receipt.get('probe') or ''
    if probe_version(probe) >= (6,):
        verify_v6(window, receipt, captured, frames, check, observe)
    if probe_version(probe) >= (7,):
        verify_v7(window, receipt, captured, frames, check, observe)
    return {'window': str(window), 'nonce': nonce_hex, 'captured_frame': captured, 'probe': probe,
            'accepted': all(c['ok'] for c in checks), 'checks': checks, 'observations': observations}


HOOKS = ('fbs', 'chara_update_force', 'chara_late_update_force', 'abmx', 'neck', 'eye')
# Installed MonoScript execution order (head_skinning.md §9.1): UniRx dispatcher -16000 (FBS),
# Manager.Character -45 (LateUpdateForce), ABMX 0, NeckLookControllerVer2 10050, EyeLookController 10800.
LATE_ORDER = ('fbs', 'chara_late_update_force', 'abmx', 'neck', 'eye')


SMR_TYPE = 'UnityEngine.SkinnedMeshRenderer'
# Probe 6.1.0: Renderer-relative offsets (Renderer+0x38 + the BaseRenderer-relative operands of
# SkinnedMeshRenderer::AddAsRenderNode 0x77abe0). 6.0.0 read the BaseRenderer offsets from the
# Renderer start; its keys below are invalid data.
SMR_KEYS_V61 = ('smr_mesh_0x1d8', 'smr_mesh_is_shared_mesh', 'smr_has_last_position_data_0x28d',
                'smr_force_matrix_recalc_0x28e', 'smr_dirty_0x2c4', 'smr_skin_buffer_0x290', 'smr_prev_skin_buffer_0x298')
SMR_KEYS_V60 = ('smr_has_last_position_data_0x255', 'smr_force_matrix_recalc_0x256', 'smr_dirty_0x28c',
                'smr_skin_buffer_0x258', 'smr_prev_skin_buffer_0x260')


def probe_version(probe: str):
    try:
        return tuple(int(x) for x in probe.split()[1].split('.'))
    except (IndexError, ValueError):
        return ()


def smr_rows(doc, path='', parent_key=None):
    """(path, row) of every SkinnedMeshRenderer row: scene/renders renderer rows by their type,
    driver skin_slots rows by position."""
    if isinstance(doc, dict):
        if doc.get('type') == SMR_TYPE or (parent_key == 'skin_slots' and 'renderer' in doc):
            yield path, doc
        for k, v in doc.items():
            yield from smr_rows(v, path + '/' + k, k)
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            yield from smr_rows(v, '%s[%d]' % (path, i), parent_key)


def smr_row_problem(row):
    if 'native_error' in row:
        return 'native_error: ' + str(row['native_error'])
    nat = row.get('native')
    if not isinstance(nat, dict):
        return 'no native section'
    stale = [k for k in SMR_KEYS_V60 if k in nat]
    if stale:
        return 'v6.0 keys present: %s' % stale
    missing = [k for k in SMR_KEYS_V61 if k not in nat]
    if missing:
        return 'missing: %s' % missing
    if nat['smr_mesh_is_shared_mesh'] is not True:
        return 'cached mesh is not sharedMesh'
    return None


def verify_smr_native(window, receipt, captured, frames, check, observe):
    """Every SMR row of every exported frame carries the 6.1 fields with the cached-mesh check."""
    version = probe_version(receipt.get('probe') or '')
    if version[:2] == (6, 0):
        observe('smr_fields_invalid_probe_6_0', 'smr_* and *_holds_frame were read from Renderer+offset instead of '
                'Renderer+0x38+offset (SkinnedMeshRenderer::AddAsRenderNode 0x77abe0); not usable')
        return
    rows, bad = 0, []
    scene_rows = 0
    for f in sorted((window / 'frames').glob('*.json')):
        if not f.name.startswith(('scene_', 'renders_', 'drivers_')):
            continue
        doc = json.loads(f.read_text(encoding='utf-8'))
        for path, row in smr_rows(doc):
            rows += 1
            if captured in frames and f.name == 'scene_%03d.json' % frames.index(captured):
                scene_rows += 1
            problem = smr_row_problem(row)
            if problem:
                bad.append({'file': f.name, 'path': path, 'problem': problem})
    check('smr_native_fields_v6_1', rows > 0 and scene_rows > 0 and not bad,
          {'rows': rows, 'captured_scene_rows': scene_rows, 'bad': bad[:20], 'bad_count': len(bad)})


def verify_bridge_transforms(window, receipt, captured, frames, check):
    """Probe 6.1.0 renderstyle: every bridge render's pre_cull record carries the character
    local/world TRS (static.json order, 33 float32 per transform) it is culled and drawn with."""
    if receipt.get('mode') != 'renderstyle' or probe_version(receipt.get('probe') or '') < (6, 1):
        return
    import base64
    count = len(json.loads((window / 'static.json').read_text(encoding='utf-8'))['transforms'])
    n = receipt.get('bridge_expected_renders')
    rows, bad = [], []
    if captured in frames:
        path = window / 'frames' / ('renders_%03d.json' % frames.index(captured))
        if path.is_file():
            for r in json.loads(path.read_text(encoding='utf-8'))['records']:
                if r.get('phase') != 'pre_cull' or not r.get('bridge_render'):
                    continue
                ordinal = r['bridge_render']['ordinal']
                rows.append(ordinal)
                ct = (r.get('data') or {}).get('character_transforms')
                if not isinstance(ct, dict):
                    bad.append({'ordinal': ordinal, 'problem': 'missing',
                                'error': (r.get('data') or {}).get('character_transforms_error') or r.get('data_error')})
                    continue
                size = len(base64.b64decode(ct.get('transforms_f32', '')))
                if ct.get('count') != count or size != count * 33 * 4 or len(base64.b64decode(ct.get('active', ''))) != count:
                    bad.append({'ordinal': ordinal, 'problem': 'size', 'count': ct.get('count'), 'bytes': size, 'static': count})
    check('bridge_pre_cull_character_transforms', sorted(rows) == list(range(n or 0)) and not bad,
          {'ordinals': rows, 'static_transforms': count, 'bad': bad[:20]})


NODE_CHECKS = ('equals_cached_mesh_0x174', 'instance_not_a_scene_renderer')


def _captured_doc(window, frames, captured, prefix):
    if captured not in frames:
        return None
    path = window / 'frames' / ('%s_%03d.json' % (prefix, frames.index(captured)))
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def verify_v7(window, receipt, captured, frames, check, observe):
    """Probe 7.0.0: RenderNode queues per render, updateWhenOffscreen bone bounds, animation source."""
    renders = _captured_doc(window, frames, captured, 'renders')
    scene = _captured_doc(window, frames, captured, 'scene')
    recs = (renders or {}).get('records', [])
    pre_cull = [r for r in recs if r.get('phase') == 'pre_cull']
    post = [r for r in recs if r.get('phase') == 'post_render']
    bad, nonempty, unknown = [], 0, 0
    for r in post:
        data = r.get('data') or {}
        rn = data.get('render_nodes')
        if not isinstance(rn, dict):
            bad.append({'render_id': r.get('render_id'), 'problem': data.get('render_nodes_error') or r.get('data_error') or 'missing'})
            continue
        qs = rn.get('queues') or []
        if rn.get('fresh') is not True or len(qs) != 2:
            bad.append({'render_id': r.get('render_id'), 'problem': 'not fresh or queue count %d' % len(qs)})
            continue
        for q in qs:
            nodes = q.get('nodes') or []
            if q.get('count') != len(nodes):
                bad.append({'render_id': r.get('render_id'), 'problem': 'count %s != %d rows' % (q.get('count'), len(nodes))})
            for n in nodes:
                if n[3] not in NODE_CHECKS and not str(n[3]).startswith('not_a_mesh_renderer:'):
                    bad.append({'render_id': r.get('render_id'), 'problem': 'node check %s' % n[3]})
                unknown += n[3] == 'instance_not_a_scene_renderer'
        nonempty += bool(qs and qs[0].get('count'))
    check('render_nodes_v7', post and len(post) == len(pre_cull) and nonempty > 0 and not bad,
          {'post_render': len(post), 'pre_cull': len(pre_cull), 'renders_with_nodes': nonempty, 'bad': bad[:20]})
    observe('render_nodes_unknown_instance_ids', unknown)

    rows = [r for r in (scene or {}).get('renderers', []) if r.get('type') == SMR_TYPE]
    sbad = []
    for r in rows:
        nat = r.get('native') or {}
        uwo = nat.get('smr_update_when_offscreen_0x28c')
        if uwo not in (0, 1):
            sbad.append({'name': r.get('name'), 'problem': 'no update_when_offscreen field'})
        if 'prepared_renderer_info' not in nat or 'smr_manager_handle_0x2f8' not in nat:
            sbad.append({'name': r.get('name'), 'problem': 'no SkinnedMeshRendererManager prepared-info fields'})
    for r in post:
        for b in (r.get('data') or {}).get('skinned_bounds_update_when_offscreen', []):
            if 'error' in b or b.get('smr_update_when_offscreen_0x28c') != 1:
                sbad.append({'render_id': r.get('render_id'), 'problem': str(b)[:200]})
    uwo_rows = [r.get('name') for r in rows if (r.get('native') or {}).get('smr_update_when_offscreen_0x28c') == 1]
    check('skinned_bounds_v7', rows and not sbad, {'smr_rows': len(rows), 'update_when_offscreen': uwo_rows, 'bad': sbad[:20]})
    # Hair bone AABBs are exported for completeness only (the orchestrator computes them exactly).
    observe('bone_bounds_state', {r.get('name'): (r.get('native') or {}).get('bone_bounds_state') for r in rows
                                  if (r.get('native') or {}).get('smr_update_when_offscreen_0x28c') == 1})

    chains = (scene or {}).get('transform_chains')
    cbad = []
    if not isinstance(chains, dict):
        cbad.append((scene or {}).get('transform_chains_error') or 'missing')
    else:
        cam = chains.get('camera_main_chain') or []
        if len(cam) < 2 or cam[-2].get('name') != 'CharaCamera':
            cbad.append("Camera.main's parent is not CharaCamera: %s" % [c.get('name') for c in cam])
        for name in ('CharaCamera', 'Lights Custom'):
            if not (chains.get('named') or {}).get(name):
                cbad.append('no transform named %s' % name)
        if not chains.get('lights'):
            cbad.append('no light chains')
    check('transform_chains_v7', not cbad, cbad)
    if receipt.get('mode') == 'renderstyle':
        bp0 = [r for r in pre_cull if r.get('bridge_render')]
        missing_chains = [r['bridge_render']['ordinal'] for r in bp0 if not isinstance((r.get('data') or {}).get('transform_chains'), dict)]
        check('bridge_pre_cull_transform_chains_v7', bp0 and not missing_chains, {'missing': missing_chains})

    anim = (scene or {}).get('animation')
    abad = []
    if not isinstance(anim, dict):
        abad.append((scene or {}).get('animation_error') or 'missing')
    else:
        ans = [a for a in anim.get('animators', []) if isinstance(a, dict) and 'error' not in a]
        if not ans or len(ans) != len(anim.get('animators', [])):
            abad.append('animator rows missing or failed')
        if not any(a.get('layer0_current_clips') for a in ans):
            abad.append('no layer-0 current clip')
        mp = anim.get('maker_pose_source')
        if not isinstance(mp, dict) or not isinstance(mp.get('entry'), dict) or mp['entry'].get('bundle_file_exists') is not True:
            abad.append('maker pose source: %s' % str(mp if mp is not None else anim.get('maker_pose_source_error'))[:200])
    check('animation_source_v7', not abad, abad)
    if receipt.get('mode') == 'renderstyle':
        bp = [r for r in pre_cull if r.get('bridge_render')]
        miss = [r['bridge_render']['ordinal'] for r in bp if not isinstance((r.get('data') or {}).get('animation'), dict)]
        check('bridge_pre_cull_animation_v7', bp and not miss, {'bridge_pre_cull': len(bp), 'missing': miss})
        observe('bridge_yaw_degrees', (receipt.get('bridge') or {}).get('body_yaw_degrees'))


def verify_v6(window, receipt, captured, frames, check, observe):
    """Probe v6 additions: driver hooks (when requested), skin slots in renders."""
    verify_smr_native(window, receipt, captured, frames, check, observe)
    verify_bridge_transforms(window, receipt, captured, frames, check)
    req = {}
    for line in (window / 'request.ini').read_text(encoding='utf-8').splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            req[k.strip()] = v.strip()
    if req.get('DriverHooks') != '1':
        check('driver_hooks_not_requested_and_absent', receipt.get('driver_hooks') is None, receipt.get('driver_hooks'))
        return
    hooks = receipt.get('driver_hooks') or {}
    check('driver_hooks_installed', len(hooks.get('installed', [])) == len(HOOKS), hooks.get('installed'))
    check('driver_hook_errors_zero', hooks.get('errors') == 0, hooks.get('errors'))
    if captured not in frames:
        return
    k = frames.index(captured)
    path = window / 'frames' / ('drivers_%03d.json' % k)
    check('drivers_export_of_captured_frame', path.is_file(), str(path))
    if not path.is_file():
        return
    doc = json.loads(path.read_text(encoding='utf-8'))
    recs = [r for r in doc['records'] if r.get('instance_in_window_character')]
    check('drivers_single_frame', doc['frame_count'] == captured and all(r['frame_count'] == captured for r in doc['records']),
          sorted({r['frame_count'] for r in doc['records']}))
    have = {(r['hook'], r['phase']) for r in recs}
    missing = [(h, p) for h in HOOKS for p in ('prefix', 'postfix') if (h, p) not in have]
    check('driver_hooks_all_fired_for_character', not missing, missing)
    first = {}
    for r in recs:
        if r['phase'] == 'prefix' and r['hook'] not in first:
            first[r['hook']] = r['seq']
    order = sorted((h for h in LATE_ORDER if h in first), key=lambda h: first[h])
    observe('late_update_hook_order', {'observed': order, 'installed_monoscript_order': list(LATE_ORDER),
                                       'matches': order == list(LATE_ORDER), 'first_prefix_seq': first})



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', type=Path, required=True)
    ap.add_argument('--window', type=Path, required=True)
    ap.add_argument('--nonce', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    from src.render import gpu_skin_inputs as g  # noqa: E402
    index = g.CaptureIndex(args.xml)
    result = verify(index, args.window, args.nonce.lower())
    result['xml_sha256'] = index.xml_sha256
    args.out.mkdir(parents=True)
    (args.out / 'verification.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('accepted' if result['accepted'] else 'REJECTED', args.out / 'verification.json')
    return 0 if result['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main())
