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
    if probe.startswith('HS2_FramePairProbe 6'):
        verify_v6(window, receipt, captured, frames, check, observe)
    return {'window': str(window), 'nonce': nonce_hex, 'captured_frame': captured, 'probe': probe,
            'accepted': all(c['ok'] for c in checks), 'checks': checks, 'observations': observations}


HOOKS = ('fbs', 'chara_update_force', 'chara_late_update_force', 'abmx', 'neck', 'eye')
# Installed MonoScript execution order (head_skinning.md §9.1): UniRx dispatcher -16000 (FBS),
# Manager.Character -45 (LateUpdateForce), ABMX 0, NeckLookControllerVer2 10050, EyeLookController 10800.
LATE_ORDER = ('fbs', 'chara_late_update_force', 'abmx', 'neck', 'eye')


def verify_v6(window, receipt, captured, frames, check, observe):
    """Probe v6 additions: driver hooks (when requested), skin slots in renders."""
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
