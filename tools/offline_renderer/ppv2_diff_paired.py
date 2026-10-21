"""PPv2 generator vs the paired capture (ordinary Maker frame + same-frame scene state).

Capture: ``paired_capture_session_20261020/rdc_04d1e76f..._frame26059.rdc`` (marker
frame 26060), oracle from ``ppv2_oracle.py``.  Scene state: the frame-pair probe
``window_04d1e76f.../frames/scene_*.json``.

Procedure (no value is taken from the capture except where listed as external):
1. Frame N-1 (snapshot ``--prev-scene`` inputs, PPv2 persistent state from
   ``--prev-prev-scene``) is generated first: it warms the temporary-RT pool exactly as
   the previous frame left it and steps Dithering / m_Pixels.  After it, the generator's
   persistent state is compared with snapshot N-1 (state evolution check).
2. Frame N (snapshot ``--scene``) is generated and byte-diffed against the oracle with
   ``ppv2_diff.compare`` (same checks as the retry-capture diff).
3. The volume blend of frame N is compared, parameter by parameter (float32 bits), with
   the layer's reflected bundle settings in snapshot N; Dithering and m_Pixels after
   frame N are compared with snapshot N.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.render.frame_program import ProgramBuilder  # noqa: E402
from src.render.hs2pipe import ppv2, ppv2_probe as pp  # noqa: E402
from src.render.hs2pipe.ppv2_settings import DEFAULTS, load_json  # noqa: E402
from src.render.hs2pipe.ppv2_unity import F, TemporaryRTPool, UnityConstantBufferPool  # noqa: E402
import ppv2_diff as rd  # noqa: E402


def bits(x):
    return struct.pack('<f', float(F(x))).hex()


def compare_blend(blender, scene):
    ref, probe = pp.blended_settings(scene)
    rows, fails = [], 0
    for cls in DEFAULTS:
        fields = ref[cls]
        mine = blender.layer[cls]
        for name, p in mine.params.items():
            key = 'PostProcessEffectSettings.enabled' if name == 'enabled' else '%s.%s' % (cls, name)
            entry = fields[key]
            v = entry['ParameterOverride`1.value']
            if p.kind == 'spline':
                ok = v is not None and p.value is not None and bool(v['Spline.m_Loop']) == p.value.loop and \
                    bits(v['Spline.m_ZeroValue']) == bits(p.value.zero)
                detail = 'loop/zero only (curve and cachedData not reflected: depth limit)'
            elif p.kind == 'texture':
                ok = (v is None) == (p.value is None)
                detail = {'reflected': v, 'generated': p.value and p.value.key}
            elif p.kind in ('color', 'vector2', 'vector4'):
                ok = [bits(a) for a in v] == [bits(a) for a in p.value]
                detail = {'reflected': v, 'generated': [float(a) for a in p.value]}
            elif p.kind == 'float':
                ok = bits(v) == bits(p.value)
                detail = {'reflected': v, 'generated': float(p.value)}
            elif p.kind in ('enum',):
                ok = pp.enum_value(v) == p.value
                detail = {'reflected': v, 'generated': p.value}
            else:
                ok = v == p.value
                detail = {'reflected': v, 'generated': p.value}
            if not ok:
                fails += 1
            rows.append({'effect': cls, 'parameter': name, 'ok': ok, 'detail': detail})
    return {'parameters': len(rows), 'fail': fails, 'rows': [r for r in rows if not r['ok']]}


def compare_persistent(gen, scene):
    idx, rnd = pp.dithering_state(scene)
    px = pp.curve_pixels(scene)
    d_ok = gen.dithering.index == idx and gen.dithering.random.seed_array == rnd.seed_array and \
        gen.dithering.random.inext == rnd.inext and gen.dithering.random.inextp == rnd.inextp
    p_bad = [i for i in range(256) if [bits(c) for c in gen.curve_pixels[i]] != [bits(c) for c in px[i]]]
    return {'dithering_equal': d_ok, 'dithering_index': [gen.dithering.index, idx],
            'm_pixels_mismatched_entries': p_bad[:20], 'm_pixels_mismatch_count': len(p_bad)}


def roles(oracle):
    seg = oracle['segments'][0]
    acts = [a for a in seg['actions'] if a['kind'] == 'DrawIndexed']
    comp = [a for a in acts if a['shaders']['PS']['source_matches'] and a['shaders']['PS']['source_matches'][0]['shader'] ==
            'Hidden/PostProcessing/MultiScaleVO' and a['shaders']['PS']['source_matches'][0]['pass'] == 2][0]
    copy = [a for a in acts if a['shaders']['PS']['source_matches'] and a['shaders']['PS']['source_matches'][0]['shader'] ==
            'Hidden/PostProcessing/MultiScaleVO' and a['shaders']['PS']['source_matches'][0]['pass'] == 0][0]
    return {'camera_target': comp['render_targets'][0]['resource'], 'camera_depth': comp['depth']['resource'],
            'depth_texture': copy['shaders']['PS']['srvs'][0]['resource']}


def run(args):
    oracle = load_json(args.oracle)
    static = load_json(args.scene_state)
    scenes = {k: load_json(getattr(args, k)) for k in ('scene', 'prev_scene', 'prev_prev_scene')}
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    b = ProgramBuilder(out / 'program', provenance={'generator': ppv2.GENERATOR, 'oracle': str(Path(args.oracle).resolve()),
                                                    'scene': str(Path(args.scene).resolve())})
    shaders = ppv2.ManifestShaderSource(args.shader_manifest, [args.compute_manifest])
    assets = ppv2.PPv2Assets(static)
    s_n, s_p, s_pp = scenes['scene'], scenes['prev_scene'], scenes['prev_prev_scene']
    layer_cfg = pp.layer_config(s_n)
    gen = ppv2.PostProcessLayerGen(b, shaders, assets, layer_cfg, cb_pool=UnityConstantBufferPool(),
                                   rt_pool=TemporaryRTPool(b), dithering_state=pp.dithering_state(s_pp),
                                   curve_pixels=pp.curve_pixels(s_pp), anisotropic_filtering=pp.anisotropic_filtering(s_n))
    R = roles(oracle)
    probe_n = pp.ProbeScene(s_n)
    ext = {
        'camera_target': rd.tex_from_oracle(b, oracle, R['camera_target'], rd.captured_views(oracle, 0, R['camera_target']),
                                            dict(filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0), 'CameraTarget'),
        'camera_depth': rd.tex_from_oracle(b, oracle, R['camera_depth'], rd.captured_views(oracle, 0, R['camera_depth']),
                                           dict(filter_mode='Point', wrap_mode='Clamp', aniso_level=0), 'CameraTarget depth'),
        'depth_texture': rd.tex_from_oracle(b, oracle, R['depth_texture'], rd.captured_views(oracle, 0, R['depth_texture']),
                                            probe_n.texture_attrs('Camera DepthTexture'), '_CameraDepthTexture'),
    }
    external = {'camera_target_attributes': 'not reflected (TempBuffer of the camera); reused temporary -> aniso 0',
                'roles': R}
    frames = []
    for label, scene, seed in (('frame_n_minus_1', s_p, False), ('frame_n', s_n, True)):
        vols = pp.volumes_from_probe(scene, static)
        cam = pp.camera_state(scene)
        globs = {'_CameraDepthTexture': ext['depth_texture'], '_ZBufferParams': pp.global_vector(scene, '_ZBufferParams')}
        if seed:
            oseg = oracle['segments'][0]
            seeds = {}
            for a in oseg['actions']:
                for st in a.get('shaders', {}).values():
                    for cb in st['constant_buffers']:
                        if cb['name'] != '$Globals' or cb['data_hex'] is None:
                            continue
                        size = oracle['resources'][str(cb['resource'])]['descriptor']['ByteWidth']
                        data = bytes.fromhex(cb['data_hex'])
                        have = seeds.setdefault(size, bytearray())
                        if len(data) > len(have):
                            have.extend(data[len(have):])
            for size, data in seeds.items():
                gen.cb_pool.seed('$Globals', size, bytes(data) + bytes(size - len(data)))
        inputs = ppv2.PPv2FrameInputs(camera_target=ext['camera_target'], camera_depth=ext['camera_depth'],
                                      camera_depth_texture=ext['depth_texture'], globals=globs)
        first = len(b.program['commands'])
        opaque, post = gen.on_pre_cull(cam, vols, frame=scene['statics']['UnityEngine.Time']['renderedFrameCount'],
                                       inputs=inputs, render_settings=pp.render_settings(scene))
        pre = list(range(first, len(b.program['commands'])))
        blend = compare_blend(gen.blender, scene)
        ex = ppv2.Executor(gen)
        ex.run(opaque, 'opaque')
        ex.run(post, 'post')
        frames.append({'label': label, 'frame_count': scene['frame_count'], 'precull_commands': pre, 'records': ex.records,
                       'blend_vs_reflected': blend, 'persistent_vs_reflected': compare_persistent(gen, scene),
                       'zbuffer_params_from_state': [float(x) for x in globs['_ZBufferParams']]})
    b.write()
    seg = dict(frames[1], segment=0)
    rd.attach_layout_ranges(gen, [seg])
    report, summary = rd.compare(oracle, b, [seg])
    for r in seg['records']:
        r.pop('_ranges', None)
    doc = {'schema': 'hs2_ppv2_paired_diff/1', 'summary': summary, 'external_inputs': external,
           'layer': layer_cfg.__dict__,
           'frames': [{k: v for k, v in f.items() if k != 'records'} for f in frames],
           'actions': report, 'log': gen.log}
    (out / 'diff.json').write_text(json.dumps(doc, indent=1, default=str), encoding='utf-8')
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--oracle', type=Path, required=True)
    ap.add_argument('--scene-state', type=Path, required=True, help='extract_ppv2_scene_state output (static assets)')
    ap.add_argument('--scene', type=Path, required=True)
    ap.add_argument('--prev-scene', type=Path, required=True)
    ap.add_argument('--prev-prev-scene', type=Path, required=True)
    ap.add_argument('--shader-manifest', type=Path, action='append', required=True)
    ap.add_argument('--compute-manifest', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    doc = run(ap.parse_args(argv))
    print(json.dumps({'summary': doc['summary'], 'frames': [{k: (v if k != 'blend_vs_reflected' else {'parameters': v['parameters'], 'fail': v['fail']})
                                                              for k, v in f.items() if k not in ('precull_commands',)}
                                                             for f in doc['frames']]}, indent=1, default=str))


if __name__ == '__main__':
    main()
