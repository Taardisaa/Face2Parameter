"""Execute the generated PPv2 frame program on the GPU and compare per pass with the
official RenderDoc replay of the paired capture.

Same generator setup as ``ppv2_diff_paired.py`` (frame N-1 warm-up, frame N from
same-frame scene state).  Inputs owned by other generators are initialised from the
official replay read-back of the paired RDC (``render_patched_capture.py render``):

* camera target before the MSVO composite (event of the MSVO depth copy) -> target A,
* camera target before ``CopyStd`` (after the forward transparent work, which is not a
  PPv2 pass) -> target B; the generator's post command buffer runs on B,
* ``_CameraDepthTexture``.

Every PPv2 intermediate the capture exposes is read back after the matching generated
command and compared by SHA-256 of the tightly packed subresource bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.render.frame_program import ProgramBuilder  # noqa: E402
from src.render.hs2pipe import ppv2, ppv2_probe as pp  # noqa: E402
from src.render.hs2pipe.ppv2_settings import load_json  # noqa: E402
from src.render.hs2pipe.ppv2_unity import TemporaryRTPool, TextureRef, UnityConstantBufferPool  # noqa: E402
import ppv2_diff as rd  # noqa: E402
import ppv2_diff_paired as pd  # noqa: E402

# oracle read-back name -> (frame-N record selector, what)
CHECKS = [
    ('out_depth_copy', ('opaque', 'draw', 0, 'target')),
    ('out_linear_depth', ('opaque', 'dispatch', 0, 'uav0')),
    ('out_ao', ('opaque', 'dispatch', 9, 'uav0')),
    ('out_ct_composite', ('opaque', 'draw', 1, 'target')),
    ('out_copystd', ('post', 'draw', 0, 'target')),
    ('out_coc', ('post', 'draw', 1, 'target')),
    ('out_dof_prefilter', ('post', 'draw', 2, 'target')),
    ('out_dof_bokeh', ('post', 'draw', 3, 'target')),
    ('out_dof_postfilter', ('post', 'draw', 4, 'target')),
    ('out_dof', ('post', 'draw', 5, 'target')),
    ('out_lut3d_slice0', ('post', 'dispatch', 0, 'uav0')),
    ('out_uber', ('post', 'draw', 6, 'target')),
    ('out_smaa_edges', ('post', 'draw', 7, 'target')),
    ('out_smaa_weights', ('post', 'draw', 8, 'target')),
    ('out_smaa', ('post', 'draw', 9, 'target')),
    ('out_final', ('post', 'draw', 10, 'target')),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def run(args):
    oracle = load_json(args.oracle)
    static = load_json(args.scene_state)
    s_n, s_p, s_pp = (load_json(p) for p in (args.scene, args.prev_scene, args.prev_prev_scene))
    rb = load_json(Path(args.readback) / 'receipt.json')
    rb_targets = {t['name']: t for t in rb['targets']}
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    b = ProgramBuilder(out / 'program', provenance={'generator': ppv2.GENERATOR, 'purpose': 'PPv2 GPU acceptance (paired capture)'})
    shaders = ppv2.ManifestShaderSource(args.shader_manifest, [args.compute_manifest])
    gen = ppv2.PostProcessLayerGen(b, shaders, ppv2.PPv2Assets(static), pp.layer_config(s_n), cb_pool=UnityConstantBufferPool(),
                                   rt_pool=TemporaryRTPool(b), dithering_state=pp.dithering_state(s_pp),
                                   curve_pixels=pp.curve_pixels(s_pp), anisotropic_filtering=pp.anisotropic_filtering(s_n))
    R = pd.roles(oracle)

    def ext(rid, attrs, name, init=None):
        t = rd.tex_from_oracle(b, oracle, rid, rd.captured_views(oracle, 0, rid), attrs, name)
        if init is not None:
            data = Path(rb_targets[init]['raw']['path']).read_bytes()
            if sha(data) != rb_targets[init]['raw']['sha256']:
                raise RuntimeError('read-back changed: ' + init)
            res = next(r for r in b.program['resources'] if r['id'] == t.resource)
            pitch = len(data) // res['desc']['Height']
            res['initial'] = [{'subresource': 0, 'blob': b.blob(data), 'row_pitch': pitch, 'slice_pitch': len(data)}]
        return t
    ct_attrs = dict(filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0)
    probe_n = pp.ProbeScene(s_n)
    depth_tex = ext(R['depth_texture'], probe_n.texture_attrs('Camera DepthTexture'), '_CameraDepthTexture', 'in_depth_texture')
    depth = ext(R['camera_depth'], dict(filter_mode='Point', wrap_mode='Clamp', aniso_level=0), 'camera depth')
    warm = ext(R['camera_target'], ct_attrs, 'camera target (warm-up frame N-1)')
    tgt_a = ext(R['camera_target'], ct_attrs, 'camera target A (before MSVO composite)', 'in_ct_opaque')
    tgt_b = ext(R['camera_target'], ct_attrs, 'camera target B (before CopyStd)', 'in_ct_post')
    records = {}
    for label, scene, target in (('warm', s_p, warm), ('frame_n', s_n, tgt_a)):
        inputs = ppv2.PPv2FrameInputs(camera_target=target, camera_depth=depth, camera_depth_texture=depth_tex,
                                      globals={'_CameraDepthTexture': depth_tex,
                                               '_ZBufferParams': pp.global_vector(scene, '_ZBufferParams')})
        opaque, post = gen.on_pre_cull(pp.camera_state(scene), pp.volumes_from_probe(scene, static),
                                       frame=scene['statics']['UnityEngine.Time']['renderedFrameCount'], inputs=inputs,
                                       render_settings=pp.render_settings(scene))
        ex = ppv2.Executor(gen)
        ex.run(opaque, 'opaque')
        n_opaque = len(ex.records)
        if label == 'frame_n':
            inputs.camera_target = tgt_b    # forward transparent passes (not PPv2) run between the two buffers
        else:
            inputs.camera_target = warm
        ex.run(post, 'post')
        records[label] = {'opaque': ex.records[:n_opaque], 'post': ex.records[n_opaque:]}
    views = {v['id']: v for v in b.program['views']}
    plan = []
    for name, (buf, kind, k, what) in CHECKS:
        recs = [r for r in records['frame_n'][buf] if r['kind'] == kind]
        r = recs[k]
        cmd = b.program['commands'][r['command']]
        if what == 'target':
            res = r['target']['resource']
        else:
            res = views[cmd['state']['cs']['uavs'][0]['view']]['resource']
        b.readback(name, res, subresource=0, after_command=r['command'], orientation='d3d_rows_top_down')
        plan.append({'name': name, 'command': r['command'], 'resource': res, 'record_kind': kind,
                     'shader': r.get('shader') or r.get('kernel'), 'pass': r.get('pass')})
    b.write()
    cmd = [str(args.executor), str(out / 'program'), str(out / 'exec')] + (['--warp'] if args.warp else [])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    (out / 'executor_stdout.txt').write_text(proc.stdout, encoding='utf-8')
    (out / 'executor_stderr.txt').write_text(proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError('frame_executor failed (%d); see %s' % (proc.returncode, out))
    exe = json.loads((out / 'exec' / 'execution.json').read_text(encoding='utf-8'))
    got = {r['name']: r for r in exe['readbacks']}
    rows = []
    for p in plan:
        o = rb_targets[p['name']]
        g = got[p['name']]
        rows.append({**p, 'equal': g['sha256'] == o['raw']['sha256'], 'generated_sha256': g['sha256'],
                     'replay_sha256': o['raw']['sha256'], 'bytes': [g['bytes'], o['raw']['bytes']],
                     'replay_event': o['event_id'], 'replay_resource': o['resource_id']})
    doc = {'schema': 'hs2_ppv2_gpu_acceptance/1', 'adapter': exe.get('adapter'), 'executor': str(args.executor),
           'executor_sha256': sha(Path(args.executor).read_bytes()), 'program_json_sha256': exe.get('program_json_sha256'),
           'replay_receipt': str(Path(args.readback) / 'receipt.json'),
           'all_equal': all(r['equal'] for r in rows), 'checks': rows}
    (out / 'acceptance.json').write_text(json.dumps(doc, indent=1), encoding='utf-8')
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--oracle', type=Path, required=True)
    ap.add_argument('--readback', type=Path, required=True, help='render_patched_capture render output (fast/)')
    ap.add_argument('--scene-state', type=Path, required=True)
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--prev-scene', type=Path, required=True)
    ap.add_argument('--prev-prev-scene', type=Path, required=True)
    ap.add_argument('--shader-manifest', type=Path, action='append', required=True)
    ap.add_argument('--compute-manifest', type=Path, required=True)
    ap.add_argument('--executor', type=Path, required=True)
    ap.add_argument('--warp', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    doc = run(ap.parse_args(argv))
    print(json.dumps({'all_equal': doc['all_equal'], 'checks': [(r['name'], r['equal']) for r in doc['checks']]}, indent=1))


if __name__ == '__main__':
    main()
