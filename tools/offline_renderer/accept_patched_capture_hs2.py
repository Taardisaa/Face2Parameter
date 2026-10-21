"""Acceptance of the patch->capture->replay pipeline on the existing HS2 RDC.

Uses only the pre-existing 2026-10-12 retry capture (no game launch, injection or
new capture). Cases:
  identity      no patch; camera 59322@35086 must equal the earlier exact replay
                bytes and, top-down, every RGBA pixel of the native Unity PNG.
  diagnostic    DIAGNOSTIC, head-unrelated: the last in-scope upload of Unity's
                "Internal Curves Texture" (Unmap chunk 35240 into staging 59898,
                copied to 11214 at event 32898). Row-0 half floats 1..511 are
                halved. Texture 11214 at event 32898 must equal the identity
                read-back with exactly mapped bytes [2,1024) replaced.
  rejection     same patch with a wrong expected_range_sha256 -> rejected,
                nothing but its receipt written; the session keeps working.
  identity_after identity again in the same session: byte-identical to identity.
  equivalence   official zip.xml route (capture_mutation + official Convert) for
                both identity and diagnostic: equal frame section + read-backs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import render_patched_capture as rpc

CURVES_UNMAP_CHUNK = 35240
CURVES_STAGING = 59898
CURVES_TEXTURE = 11214
CURVES_COPY_EVENT = 32898
CAMERA = 59322
CAMERA_EVENT = 35086


def h(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('tool', 'renderdoc-dir', 'rdc', 'index', 'zip-xml', 'native-png', 'reference-bin', 'out', 'receipt'):
        ap.add_argument('--' + k, required=True, type=Path)
    ap.add_argument('--timing-repeats', type=int, default=3)
    a = ap.parse_args()
    if a.out.exists() or a.receipt.exists():
        raise FileExistsError('Fresh output directory and receipt required')
    a.out.mkdir(parents=True)
    tools = rpc.Tools(a.tool, a.renderdoc_dir)
    rdc_sha = rpc.sha256_file(a.rdc)
    index = rpc.load_index(a.index, rdc_sha)
    chunk = index['chunks'][CURVES_UNMAP_CHUNK]
    assert chunk['name'] == 'ID3D11DeviceContext::Unmap' and {'name': 'pResource', 'id': CURVES_STAGING} in chunk['resources']
    buf = chunk['buffers'][0]
    zip_path = a.zip_xml.with_suffix('')
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        blob = z.read(f'{buf["sd_buffer_index"]:06d}')
    assert h(blob) == buf['sha256'] and len(blob) == buf['bytes'] == 2047
    # Unmap recorded written range [1,2048) of the tightly packed 128x2 R16G16B16A16_FLOAT staging texture:
    # blob byte j is mapped byte j+1. Mapped [2,1024) = row-0 half floats 1..511 = blob[1:1023].
    old = blob[1:1023]
    new = (np.frombuffer(old, dtype='<f2').astype(np.float32) * 0.5).astype('<f2').tobytes()
    plans = {}
    for name, range_sha in (('diagnostic', h(old)), ('rejection', '0' * 64)):
        d = a.out / f'plan_{name}'
        d.mkdir()
        (d / 'replacement.bin').write_bytes(new)
        plan = {'source_rdc_sha256': rdc_sha, 'source_xml_sha256': rpc.sha256_file(a.zip_xml), 'source_zip_sha256': rpc.sha256_file(zip_path),
                'patches': [{'chunk_index': CURVES_UNMAP_CHUNK, 'chunk_name': chunk['name'], 'buffer_ordinal': 0, 'resource_id': CURVES_STAGING,
                             'expected_buffer_index': buf['sd_buffer_index'], 'expected_blob_sha256': buf['sha256'], 'byte_offset': 1,
                             'expected_range_sha256': range_sha, 'replacement_file': 'replacement.bin', 'replacement_sha256': h(new),
                             'source_witness': 'recorded Unmap MapWrittenData of staging 59898 (Unity "Internal Curves Texture" upload, copied to 11214 at event 32898)',
                             'semantic_role': 'DIAGNOSTIC ONLY, head-unrelated colour-grading curves: row-0 half floats 1..511 halved; not an HS2 candidate'}]}
        (d / 'plan.json').write_text(json.dumps(plan, indent=2))
        plans[name] = d / 'plan.json'
    targets = [{'name': 'camera', 'kind': 'texture', 'event_id': CAMERA_EVENT, 'resource_id': CAMERA, 'orientation': 'unity_texture_rows_bottom_up'},
               {'name': 'curves', 'kind': 'texture', 'event_id': CURVES_COPY_EVENT, 'resource_id': CURVES_TEXTURE, 'orientation': 'unity_texture_rows_bottom_up', 'png': False}]
    result = {'scope': 'pipeline acceptance on the existing unaccepted retry RDC; diagnostic patch is not HS2 fitting',
              'tools': tools.describe(), 'rdc': str(a.rdc), 'rdc_sha256': rdc_sha, 'index': str(a.index), 'index_sha256': rpc.sha256_file(a.index),
              'cases': {}, 'timings': {}}
    t = time.perf_counter()
    with rpc.ReplaySession(tools, a.rdc, a.index, a.out / 'session') as s:
        result['timings']['session_start_s'] = time.perf_counter() - t
        result['session_ready'] = s.ready
        ident = s.render(None, targets, a.out / 'identity')
        cam = Path(ident['targets'][0]['raw']['path']).read_bytes()
        ref = a.reference_bin.read_bytes()
        top = Path(ident['targets'][0]['topdown']['path']).read_bytes()
        native = np.asarray(Image.open(a.native_png).convert('RGBA'))
        info = ident['targets'][0]['texture']
        top_arr = np.frombuffer(top, np.uint8).reshape(info['height'], info['width'], 4)
        png_arr = np.asarray(Image.open(ident['targets'][0]['png']['path']))
        result['cases']['identity'] = {'receipt': str(a.out / 'identity' / 'receipt.json'),
                                       'camera_raw_equals_previous_exact_replay': cam == ref, 'reference_bin_sha256': h(ref),
                                       'camera_topdown_equals_native_png_rgba': bool(np.array_equal(top_arr, native)),
                                       'written_png_pixels_equal_topdown': bool(np.array_equal(png_arr, top_arr)),
                                       'timing': ident['timing']}
        diag = s.render(plans['diagnostic'], targets, a.out / 'diagnostic', keep_candidate=True)
        curves_id = Path(ident['targets'][1]['raw']['path']).read_bytes()
        curves_dg = Path(diag['targets'][1]['raw']['path']).read_bytes()
        predicted = curves_id[:2] + new + curves_id[1024:]
        cam_dg = np.frombuffer(Path(diag['targets'][0]['topdown']['path']).read_bytes(), np.uint8).reshape(top_arr.shape)
        changed = np.any(cam_dg != top_arr, axis=2)
        result['cases']['diagnostic'] = {'receipt': str(a.out / 'diagnostic' / 'receipt.json'), 'curves_texture_bytes': len(curves_id),
                                         'curves_equals_predicted_exact_bytes': curves_dg == predicted,
                                         'curves_changed_byte_range': [int(min(i for i in range(len(curves_id)) if curves_id[i] != curves_dg[i])),
                                                                       int(max(i for i in range(len(curves_id)) if curves_id[i] != curves_dg[i])) + 1] if curves_id != curves_dg else None,
                                         'camera_changed_pixels': int(changed.sum()), 'camera_pixels': int(changed.size),
                                         'camera_max_channel_delta': int(np.abs(cam_dg.astype(int) - top_arr.astype(int)).max()),
                                         'candidate': diag['candidate'], 'timing': diag['timing']}
        try:
            s.render(plans['rejection'], targets, a.out / 'rejection')
            rejected = None
        except rpc.PipelineRejected as e:
            rejected = str(e)
        result['cases']['rejection'] = {'error': rejected, 'rejected': rejected is not None,
                                        'files': sorted(p.name for p in (a.out / 'rejection').iterdir())}
        again = s.render(None, targets, a.out / 'identity_after')
        result['cases']['identity_after'] = {'equal_to_identity': [x['raw']['sha256'] for x in again['targets']] == [x['raw']['sha256'] for x in ident['targets']]}
        reps = []
        for i in range(a.timing_repeats):
            r = s.render(plans['diagnostic'], targets[:1], a.out / f'timing_diag_{i}')
            reps.append({'python_total_s': r['timing']['python_total_s'], 'tool_ms': r['timing']['tool_ms'],
                         'camera_sha256': r['targets'][0]['raw']['sha256']})
        result['timings']['repeated_diagnostic_camera_only'] = reps
        result['timings']['repeat_camera_deterministic'] = len({r['camera_sha256'] for r in reps} | {diag['targets'][0]['raw']['sha256']}) == 1
    for name, plan in (('identity', None), ('diagnostic', plans['diagnostic'])):
        job = {'schema': rpc.JOB_SCHEMA, 'capture_rdc': str(a.rdc), 'capture_rdc_sha256': rdc_sha, 'index': str(a.index),
               'plan': None if plan is None else str(plan), 'official_zip_xml': str(a.zip_xml), 'targets': targets}
        t = time.perf_counter()
        summary = rpc.run_job(tools, job, a.out / f'equivalence_{name}', 'both')
        official = json.loads((a.out / f'equivalence_{name}' / 'official' / 'official_route.json').read_text())
        result['cases'][f'equivalence_{name}'] = {'equivalence': summary['equivalence'], 'official_timing_s': official['timing'],
                                                  'both_wall_s': time.perf_counter() - t}
    checks = [result['cases']['identity']['camera_raw_equals_previous_exact_replay'], result['cases']['identity']['camera_topdown_equals_native_png_rgba'],
              result['cases']['identity']['written_png_pixels_equal_topdown'], result['cases']['diagnostic']['curves_equals_predicted_exact_bytes'],
              result['cases']['rejection']['rejected'], result['cases']['rejection']['files'] == ['receipt.json'],
              result['cases']['identity_after']['equal_to_identity'], result['timings']['repeat_camera_deterministic'],
              result['cases']['equivalence_identity']['equivalence']['all_equal'], result['cases']['equivalence_diagnostic']['equivalence']['all_equal']]
    result['pipeline_acceptance_passed'] = all(checks)
    result.update({'candidate_geometry_complete': False, 'full_renderer_accepted': False, 'chin_accepted': False})
    a.receipt.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['pipeline_acceptance_passed'], 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
