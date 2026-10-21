"""Generate the PPv2 frame part for the retry capture and byte-diff it against the oracle.

Inputs
* ``--oracle``: ``ppv2_oracle.py`` output (exact captured bytes/descriptors).
* ``--scene-state``: ``extract_ppv2_scene_state.py`` output (CharaCustom layer, volumes, profiles).
* ``--dhh``: ``DHH_Data/GraphicSetting.dhs`` of the installed game.
* ``--native-request``: the bridge capture request (``ortho_size`` -> DHH focus distance).
* shader manifests (sharedassets2 export) and the compute-shader export.

Scene-state fields this capture does not record are listed in ``REQUIRED_LIVE_FIELDS``;
for the diff they are taken from documented sources (see each entry) and every value
that is not produced by the PPv2 generator itself is reported under ``external_inputs``
so that it is never counted as reproduced.

Output (fresh directory): ``program/`` (frame program of all 16 capture-camera
renders' PPv2 work), ``diff.json`` (per action and summary), ``generator_log.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.render.frame_program import ProgramBuilder  # noqa: E402
from src.render.hs2pipe import ppv2  # noqa: E402
from src.render.hs2pipe.ppv2_settings import (DHHRuntime, GraphicData, dhh_volume, load_json,  # noqa: E402
                                              parse_dhh_graphic_setting, scene_volumes)
from src.render.hs2pipe.ppv2_unity import (F, RTDescriptor, SystemRandom, TemporaryRTPool, TextureRef,  # noqa: E402
                                           UnityConstantBufferPool, dv, mathf_tan, r4)

CAPTURE_SEGMENTS = range(16)

REQUIRED_LIVE_FIELDS = [
    {'field': 'UnityEngine.Camera.fieldOfView (Camera.main)', 'type': 'float',
     'used_for': 'MakerRenderStyleService focus distance = ortho_size / Mathf.Tan(fov*Deg2Rad*0.5)',
     'diff_source': '23.0, cross-checked against captured glstate projection [1,1] (unity_MatrixVP)'},
    {'field': 'UnityEngine.Camera.nearClipPlane / farClipPlane', 'type': 'float',
     'used_for': 'MultiScaleVO.CalculateZBufferParams', 'diff_source': 'captured _ProjectionParams (0.08, 1000)'},
    {'field': 'UnityEngine.Camera.projectionMatrix[0,0]', 'type': 'float',
     'used_for': 'MultiScaleVO.CalculateTanHalfFovHeight', 'diff_source': 'captured unity_MatrixVP / view (|m00|)'},
    {'field': 'UnityEngine.Camera.allowHDR, actualRenderingPath, pixelWidth/pixelHeight', 'type': 'bool, enum, int',
     'used_for': 'source format, AO/SSR/fog branches, MSVO sizes', 'diff_source': 'DHH sets allowHDR=true; forward (SSS on); 1086x1448 request'},
    {'field': 'DHH.DHH_Main.depthOfFieldLayer.focusDistance.value at render time', 'type': 'float',
     'used_for': 'DoF _Distance/_LensCoeff', 'diff_source': 'bridge temporary focus = ortho_size/tan(fov/2) (MakerRenderStyleService)'},
    {'field': 'DHH.DHH_Main GraphicFloat[100]/GraphicBool[30]/GraphicColor[4] (live arrays)', 'type': 'float[]/bool[]/Vector3[]',
     'used_for': 'DHH_Post volume weight and overrides', 'diff_source': 'DHH_Data/GraphicSetting.dhs (may differ from live UI edits)'},
    {'field': 'CharaCustom.CustomControl._draw3D.Value (Map3D active)', 'type': 'bool',
     'used_for': 'PostProcessVolume3D enabled', 'diff_source': 'true (default) -- confirmed by DoF/LUT constants'},
    {'field': 'PostProcessManager.m_SortedVolumes[256] order of the priority-0 volumes', 'type': 'List<PostProcessVolume>',
     'used_for': 'blend order 2D -> 3D -> DHH', 'diff_source': 'selected by captured DoF/LUT constants'},
    {'field': 'Manager.Config.GraphicData.{Bloom,SSAO,SSR,Vignette}', 'type': 'bool',
     'used_for': 'ConfigEffectorVolume active flags', 'diff_source': 'UserData/config/system.xml (all True)'},
    {'field': 'QualitySettings.anisotropicFiltering', 'type': 'AnisotropicFiltering',
     'used_for': 'sampler anisotropy', 'diff_source': 'ForceEnable (DHH ApplyGraphicSetting)'},
    {'field': 'RenderSettings.fog (+fogDensity/start/end)', 'type': 'bool, float',
     'used_for': 'MSVO APPLY_FORWARD_FOG', 'diff_source': 'false (captured MSVO composite variant has no APPLY_FORWARD_FOG)'},
    {'field': 'PostProcessLayer.dithering.m_NoiseTextureIndex and m_Random state (number of prior Dithering.Render calls)',
     'type': 'int + System.Random', 'used_for': '_DitheringTex and _Dithering_Coords.zw',
     'diff_source': 'solved from the captured (z, w) of segment 0 against System.Random(1234)'},
    {'field': 'ColorGradingRenderer.m_Pixels[128..255] (last LDR-mode curves)', 'type': 'Color[256]',
     'used_for': 'curves texture row 1 bytes (unused by the HDR bake)', 'diff_source': 'scene-only (pre-DHH) LDR blend'},
    {'field': 'Unity built-in camera globals _ZBufferParams', 'type': 'Vector4',
     'used_for': 'DoF CoC pass', 'diff_source': 'external (unity_forward); taken from the capture, not counted as reproduced'},
    {'field': 'temporary RT pool free list at PPv2 start (e.g. DHH SSS "ImageEffects Temp" 543x724, anisoLevel 0)',
     'type': 'RenderTexture objects', 'used_for': 'DoF half-res target sampler', 'diff_source': 'captured resource 59384'},
    {'field': 'camera target / camera depth / _CameraDepthTexture RenderTexture objects (filter, wrap, anisoLevel)',
     'type': 'RenderTexture', 'used_for': 'samplers of CopyStd/_CameraDepthTexture', 'diff_source': 'captured resources 59330/59334/59404'},
    {'field': 'Unity constant-buffer shadows ($Globals per size) at PPv2 start', 'type': 'bytes',
     'used_for': 'residue bytes outside PPv2 program layouts', 'diff_source': 'captured CB contents (full-buffer diff only)'},
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def solve_dithering(z, w, limit=2_000_000):
    """Number n of prior Dithering.Render calls such that call n+1 draws (z, w):
    each call consumes two System.Random(1234).NextDouble() values."""
    r = SystemRandom(1234)
    for n in range(limit):
        a = F(r.next_double())
        b = F(r.next_double())
        if a == z and b == w:
            return n
    return None


def tex_from_oracle(b, oracle, rid, view_ids, attrs, name):
    """Create a program resource mirroring a captured input resource (descriptor only)."""
    R = oracle['resources']
    r = R[str(rid)]
    kind = {'CreateTexture2D': 'tex2d', 'CreateTexture3D': 'tex3d'}[r['kind']]
    desc = {k: v for k, v in r['descriptor'].items() if v is not None}
    if kind == 'tex2d':
        desc = {k: desc[k] for k in ('Width', 'Height', 'MipLevels', 'ArraySize', 'Format', 'SampleDesc', 'Usage',
                                     'BindFlags', 'CPUAccessFlags', 'MiscFlags')}
    res = b.texture(kind, desc, name=name)
    t = TextureRef(res, None, desc['Width'], desc['Height'], name=name, resource_desc=desc, **attrs)
    for vk, vid in view_ids.items():
        vdesc = R[str(vid)]['descriptor']
        setattr(t, vk, b.view(vk, res, vdesc))
    return t


def captured_views(oracle, seg, rid):
    """View ids by kind used on a captured resource within a segment."""
    out = {}
    for a in oracle['segments'][seg]['actions']:
        for st in a.get('shaders', {}).values():
            for s in st['srvs']:
                if s['resource'] == rid:
                    out.setdefault('srv', s['view'])
        for t in a.get('render_targets', []):
            if t['resource'] == rid:
                out.setdefault('rtv', t['view'])
        if a.get('depth') and a['depth']['resource'] == rid:
            out.setdefault('dsv', a['depth']['view'])
    return out


def build(args):
    oracle = load_json(args.oracle)
    scene = load_json(args.scene_state)
    dhs = Path(args.dhh).read_text(encoding='utf-8')
    request = load_json(args.native_request)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    b = ProgramBuilder(out / 'program', provenance={'generator': ppv2.GENERATOR, 'oracle': str(Path(args.oracle).resolve()),
                                                    'ppv2_assembly_sha256': ppv2.PPV2_ASSEMBLY_SHA256})
    shaders = ppv2.ManifestShaderSource(args.shader_manifest, [args.compute_manifest])
    assets = ppv2.PPv2Assets(scene)
    # ---- DHH + scene volumes
    cfg = parse_dhh_graphic_setting(dhs)
    fov = F(args.fov)
    deg2rad = F(0.0174532924)
    # MakerRenderStyleService: distance = size / Mathf.Tan(cam.fieldOfView * Mathf.Deg2Rad * .5f)
    focus = r4(dv(request['ortho_size']) / dv(mathf_tan(r4(dv(fov) * dv(deg2rad) * 0.5))))
    dhh_vol, layer_over = dhh_volume(cfg, DHHRuntime(graphic_enable=True, focus_distance=focus))
    vols = scene_volumes(scene, GraphicData(), draw3d=True, textures=scene.get('textures')) + [dhh_vol]
    main_layer = next(l for l in scene['layers'] if l['game_object']['m_Name'] == 'Main Camera')
    layer_cfg = ppv2.LayerConfig.from_scene(main_layer, layer_over)
    # ---- shared infrastructure (owned by the frame orchestrator in production)
    cb_pool = UnityConstantBufferPool()
    rt_pool = TemporaryRTPool(b)
    # dithering state solved from segment 0
    seg0 = oracle['segments'][0]
    fin = [a for a in seg0['actions'] if a['kind'] == 'DrawIndexed' and a['shaders']['PS']['debug_names'][-1] == 'Hidden/PostProcessing/FinalPass'][0]
    cbhex = fin['shaders']['PS']['constant_buffers'][0]['data_hex']
    zc, wc = struct.unpack_from('<2f', bytes.fromhex(cbhex), 456)
    prior = solve_dithering(F(zc), F(wc))
    if prior is None:
        raise RuntimeError('captured dithering (z, w) not in the System.Random(1234) sequence')
    gen = ppv2.PostProcessLayerGen(b, shaders, assets, layer_cfg, cb_pool=cb_pool, rt_pool=rt_pool,
                                   dithering_prior_calls=prior, anisotropic_filtering='ForceEnable')
    gen.seed_ldr_curve_history(scene_volumes(scene, GraphicData(), draw3d=True, textures=scene.get('textures')), frame=-1)
    # ---- external inputs mirrored from the capture (descriptors + sampler attributes)
    ext = {}
    v_ct = captured_views(oracle, 0, 59330)
    ext['camera_target'] = tex_from_oracle(b, oracle, 59330, v_ct, dict(filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0),
                                           'CameraTarget color (TempBuffer 1913)')
    ext['camera_depth'] = tex_from_oracle(b, oracle, 59334, captured_views(oracle, 0, 59334),
                                          dict(filter_mode='Point', wrap_mode='Clamp', aniso_level=0), 'CameraTarget depth')
    ext['depth_texture'] = tex_from_oracle(b, oracle, 59404, captured_views(oracle, 0, 59404),
                                           dict(filter_mode='Point', wrap_mode='Clamp', aniso_level=1), '_CameraDepthTexture')
    image_temp = tex_from_oracle(b, oracle, 59384, {}, dict(filter_mode='Bilinear', wrap_mode='Clamp', aniso_level=0),
                                 'ImageEffects Temp (DHH SSS)')
    d = RTDescriptor(543, 724, 'ARGBHalf', srgb=True)
    kind, res, srv, rtv, uav = d.d3d()
    image_temp.srv = b.view('srv', image_temp.resource, srv)
    image_temp.rtv = b.view('rtv', image_temp.resource, rtv)
    image_temp.descriptor = d
    rt_pool.adopt(d, image_temp, free=True)
    camera = ppv2.CameraState(pixel_width=request['width'], pixel_height=request['height'], near_clip=F(0.08),
                              far_clip=F(1000.0), projection_m00=F(args.projection_m00), allow_hdr=True,
                              rendering_path='Forward')
    external_inputs = {'focus_distance': float(focus), 'dithering_prior_calls': prior,
                       'camera': {k: (float(v) if isinstance(v, F) else v) for k, v in camera.__dict__.items()},
                       'zbuffer_params': {}}
    segments = []
    for seg in CAPTURE_SEGMENTS:
        oseg = oracle['segments'][seg]
        # Unity camera built-in _ZBufferParams (owned by unity_forward): taken from the capture.
        coc = [a for a in oseg['actions'] if a['kind'] == 'DrawIndexed'
               and a['shaders']['PS']['source_matches'] and a['shaders']['PS']['source_matches'][0]['shader'] == 'Hidden/PostProcessing/DepthOfField'
               and a['shaders']['PS']['source_matches'][0]['pass'] == 0][0]
        zbp = struct.unpack_from('<4f', bytes.fromhex(coc['shaders']['PS']['constant_buffers'][0]['data_hex']), 336)
        external_inputs['zbuffer_params'][seg] = list(zbp)
        # Seed the shared $Globals shadows with captured contents (residue only): the
        # progressive union, in action order, of every captured declared range.  Every
        # byte a PPv2 program layout covers is overwritten by the generator before it is
        # read, so only bytes outside all PPv2 layouts survive from the seed.
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
            full = bytes(data) + bytes(cb_pool.snapshot('$Globals', size)[len(data):])
            cb_pool.seed('$Globals', size, full)
        inputs = ppv2.PPv2FrameInputs(camera_target=ext['camera_target'], camera_depth=ext['camera_depth'],
                                      camera_depth_texture=ext['depth_texture'],
                                      globals={'_CameraDepthTexture': ext['depth_texture'],
                                               '_ZBufferParams': tuple(F(x) for x in zbp)})
        first_cmd = len(b.program['commands'])
        opaque, post = gen.on_pre_cull(camera, vols, frame=17675, inputs=inputs, render_settings=ppv2.RenderSettingsState(fog=False))
        precull_cmds = list(range(first_cmd, len(b.program['commands'])))
        ex = ppv2.Executor(gen)
        ex.run(opaque, 'opaque')
        ex.run(post, 'post')
        segments.append({'segment': seg, 'precull_commands': precull_cmds, 'records': ex.records,
                         'volume_blend': gen.log[-2] if len(gen.log) >= 2 else None})
    b.write()
    return oracle, b, segments, gen, external_inputs, layer_cfg, vols


def hexdiff(a, b):
    if a == b:
        return None
    n = min(len(a), len(b))
    bad = [i for i in range(0, n, 4) if a[i:i + 4] != b[i:i + 4]]
    return {'len': [len(a), len(b)], 'differing_dwords': len(bad), 'first_offsets': bad[:16]}


def compare(oracle, b, segments, layer_vars_only_names=()):
    R = oracle['resources']
    prog = b.program
    views = {v['id']: v for v in prog['views']}
    res = {r['id']: r for r in prog['resources']}
    samplers = {s['id']: s['desc'] for s in prog['samplers']}
    states = {t: {s['id']: s['desc'] for s in prog[t]} for t in ('blend_states', 'depth_states', 'raster_states')}
    report = []
    summary = {'actions': 0, 'mismatches': 0, 'checks': {}}

    def check(name, ok, detail=None, row=None):
        c = summary['checks'].setdefault(name, {'pass': 0, 'fail': 0})
        c['pass' if ok else 'fail'] += 1
        if not ok:
            summary['mismatches'] += 1
            if row is not None:
                row.setdefault('fail', []).append({'check': name, 'detail': detail})

    def tex_desc_cmp(row, label, cap_res_id, cap_view_id, gen_view_id):
        cr = R[str(cap_res_id)]
        cv = R[str(cap_view_id)]
        gv = views[gen_view_id]
        gr = res[gv['resource']]
        cdesc = {k: v for k, v in cr['descriptor'].items() if v is not None}
        gdesc = {k: v for k, v in gr['desc'].items()}
        check('resource_descriptor', cdesc == gdesc, {'what': label, 'captured': cdesc, 'generated': gdesc}, row)
        check('view_descriptor', cv['descriptor'] == gv['desc'], {'what': label, 'captured': cv['descriptor'], 'generated': gv['desc']}, row)

    for seg in segments:
        oseg = oracle['segments'][seg['segment']]
        cap = [a for a in oseg['actions']]
        gen_cmds = [prog['commands'][i] for i in seg['precull_commands']]
        recs = seg['records']
        # precull: staging Unmap + copy
        cmap = [a for a in cap if a['kind'] == 'Unmap']
        ccopy = [a for a in cap if a['kind'] == 'CopySubresourceRegion']
        row = {'segment': seg['segment'], 'kind': 'curves_upload', 'events': [a['event_id'] for a in cmap + ccopy]}
        gm = [c for c in gen_cmds if c['op'] == 'map_write']
        gc = [c for c in gen_cmds if c['op'] == 'copy_subresource_region']
        check('curves_upload_count', len(gm) == len(cmap) == 1 and len(gc) == len(ccopy) == 1, None, row)
        if gm and cmap:
            data = b.blobs.read(gm[0]['blob'])
            s0, s1 = cmap[0]['written_bytes']
            check('curves_upload_bytes', data[s0:s1] == bytes.fromhex(cmap[0]['data_hex']),
                  hexdiff(data[s0:s1], bytes.fromhex(cmap[0]['data_hex'])), row)
        report.append(row)
        summary['actions'] += 1
        cap_actions = [a for a in cap if a['kind'] not in ('Unmap', 'CopySubresourceRegion')]
        if len(cap_actions) != len(recs):
            check('action_count', False, {'captured': len(cap_actions), 'generated': len(recs)}, row)
        for a, r in zip(cap_actions, recs):
            summary['actions'] += 1
            row = {'segment': seg['segment'], 'event': a['event_id'], 'kind': a['kind'], 'generated_command': r['command']}
            gcmd = prog['commands'][r['command']]
            if a['kind'] == 'ClearRenderTargetView':
                check('kind', r['kind'] == 'clear_rtv', None, row)
                check('clear_color', gcmd['color'] == a['values']['ColorRGBA'], None, row)
                tex_desc_cmp(row, 'clear target', a['resource'], a['view'], gcmd['view'])
                report.append(row)
                continue
            if a['kind'] == 'Dispatch':
                check('kind', r['kind'] == 'dispatch', None, row)
                cs = a['shaders']['CS']
                check('cs_dxbc', cs['dxbc_sha256'] == r['dxbc_sha256'], {'captured': cs['dxbc_sha256'], 'generated': r['dxbc_sha256']}, row)
                args = a['arguments']
                check('dispatch_groups', [args['ThreadGroupCountX'], args['ThreadGroupCountY'], args['ThreadGroupCountZ']] == r['groups'],
                      {'captured': args, 'generated': r['groups']}, row)
                st = gcmd['state']['cs']
                for cb in cs['constant_buffers']:
                    gcb = next(x for x in st['constant_buffers'] if x['slot'] == cb['slot'])
                    gdata = b.blobs.read(gcb['blob'])
                    cdata = bytes.fromhex(cb['data_hex'])
                    check('cs_cbuffer_bytes', gdata[:len(cdata)] == cdata, hexdiff(gdata[:len(cdata)], cdata), row)
                    check('cs_cbuffer_size', len(gdata) == oracle['resources'][str(cb['resource'])]['descriptor']['ByteWidth'],
                          {'generated': len(gdata)}, row)
                for s in cs['srvs']:
                    g = next((x for x in st['srvs'] if x['slot'] == s['slot']), None)
                    check('cs_srv_bound', g is not None, s, row)
                    if g:
                        tex_desc_cmp(row, 'cs srv %d' % s['slot'], s['resource'], s['view'], g['view'])
                for u in a['uavs']:
                    g = next((x for x in st['uavs'] if x['slot'] == u['slot']), None)
                    check('cs_uav_bound', g is not None, u, row)
                    if g:
                        tex_desc_cmp(row, 'cs uav %d' % u['slot'], u['resource'], u['view'], g['view'])
                for smp in cs['samplers']:
                    g = next((x for x in st['samplers'] if x['slot'] == smp['slot']), None)
                    cd = R[str(smp['sampler'])]['descriptor']
                    check('sampler_descriptor', g is not None and samplers[g['sampler']] == cd,
                          {'slot': smp['slot'], 'captured': cd, 'generated': g and samplers[g['sampler']]}, row)
                report.append(row)
                continue
            # draws
            check('kind', r['kind'] == 'draw', None, row)
            st = gcmd['state']
            for stage, key in (('VS', 'vs'), ('PS', 'ps')):
                cs = a['shaders'][stage]
                gsha = r['vs_sha256'] if stage == 'VS' else r['ps_sha256']
                check('%s_dxbc' % stage.lower(), cs['dxbc_sha256'] == gsha, {'captured': cs['dxbc_sha256'], 'generated': gsha}, row)
                for cb in cs['constant_buffers']:
                    gcb = next((x for x in st[key]['constant_buffers'] if x['slot'] == cb['slot']), None)
                    check('cbuffer_bound', gcb is not None, cb['slot'], row)
                    if gcb is None:
                        continue
                    gdata = b.blobs.read(gcb['blob'])
                    cdata = bytes.fromhex(cb['data_hex'])
                    check('cbuffer_size', len(gdata) == oracle['resources'][str(cb['resource'])]['descriptor']['ByteWidth'],
                          {'generated': len(gdata)}, row)
                    rec = next(x for x in r['cbuffers'] if x['stage'] == stage and x['slot'] == cb['slot'])
                    layout = _layout_ranges(r, stage, cb['slot'])
                    lv_ok = all(gdata[o:o + n] == cdata[o:o + n] for o, n in layout if o + n <= len(cdata))
                    check('cbuffer_layout_variables', lv_ok,
                          {'stage': stage, 'vars': rec['variables'],
                           'diff': [o for o, n in layout if gdata[o:o + n] != cdata[o:o + n]]}, row)
                    check('cbuffer_full_bytes_seeded', gdata[:len(cdata)] == cdata, hexdiff(gdata[:len(cdata)], cdata), row)
                for s in cs['srvs']:
                    g = next((x for x in st[key]['srvs'] if x['slot'] == s['slot']), None)
                    check('srv_bound', g is not None, s, row)
                    if g:
                        tex_desc_cmp(row, '%s srv %s' % (stage, s['name']), s['resource'], s['view'], g['view'])
                for smp in cs['samplers']:
                    g = next((x for x in st[key]['samplers'] if x['slot'] == smp['slot']), None)
                    cd = R[str(smp['sampler'])]['descriptor']
                    check('sampler_descriptor', g is not None and samplers[g['sampler']] == cd,
                          {'stage': stage, 'slot': smp['slot'], 'captured': cd, 'generated': g and samplers[g['sampler']]}, row)
            for t in a['render_targets']:
                g = next((x for x in st['om']['render_targets'] if x['slot'] == t['slot']), None)
                check('rtv_bound', g is not None, t, row)
                if g:
                    tex_desc_cmp(row, 'rtv %d' % t['slot'], t['resource'], t['view'], g['view'])
            check('dsv_bound', bool(a['depth']) == bool(st['om']['depth_stencil']), None, row)
            if a['depth'] and st['om']['depth_stencil']:
                tex_desc_cmp(row, 'dsv', a['depth']['resource'], a['depth']['view'], st['om']['depth_stencil'])
            raw = a['raw_state']
            check('blend_state', states['blend_states'][st['om']['blend_state']] == R[str(raw['OM']['BlendState'])]['descriptor'],
                  {'captured': R[str(raw['OM']['BlendState'])]['descriptor'], 'generated': states['blend_states'][st['om']['blend_state']]}, row)
            check('depth_state', states['depth_states'][st['om']['depth_stencil_state']] == R[str(raw['OM']['DepthStencilState'])]['descriptor'],
                  {'captured': R[str(raw['OM']['DepthStencilState'])]['descriptor'], 'generated': states['depth_states'][st['om']['depth_stencil_state']]}, row)
            check('raster_state', states['raster_states'][st['rs']['state']] == a['rasterizer']['descriptor'],
                  {'captured': a['rasterizer']['descriptor'], 'generated': states['raster_states'][st['rs']['state']]}, row)
            check('sample_mask', st['om']['sample_mask'] == raw['OM']['SampleMask'], None, row)
            check('stencil_ref', st['om']['stencil_ref'] == raw['OM']['StencRef'], None, row)
            check('viewport', st['rs']['viewports'] == a['viewports'], {'captured': a['viewports'], 'generated': st['rs']['viewports']}, row)
            check('topology', st['ia']['topology'] == a['topology'], None, row)
            # executed geometry: index values and addressed vertex bytes (ring offsets differ by design)
            geo = raw['geometry']
            cidx = struct.unpack('<%dH' % (len(geo['index_hex']) // 4), bytes.fromhex(geo['index_hex']))
            gib = res[st['ia']['index_buffer']['buffer']]
            gvb = res[st['ia']['vertex_buffers'][0]['buffer']]
            gidx = struct.unpack('<%dH' % (gcmd['args']['IndexCount']), b.blobs.read(gib['initial'][0]['blob'])[:2 * gcmd['args']['IndexCount']])
            check('index_values', list(cidx) == list(gidx), {'captured': cidx, 'generated': gidx}, row)
            vslot = next(v for v in geo['vertex'] if v['slot'] == 0)
            gvb_bytes = b.blobs.read(gvb['initial'][0]['blob'])
            check('vertex_bytes', bytes.fromhex(vslot['hex']) == gvb_bytes[:len(bytes.fromhex(vslot['hex']))],
                  None, row)
            elems = [{k: e[k] for k in ('SemanticName', 'SemanticIndex', 'Format', 'InputSlot', 'AlignedByteOffset',
                                         'InputSlotClass', 'InstanceDataStepRate')} for e in a['input_layout']['elements']]
            gl = next(il for il in prog['input_layouts'] if il['id'] == st['ia']['input_layout'])
            check('input_layout', elems == gl['elements'], {'captured': elems, 'generated': gl['elements']}, row)
            check('vertex_stride', st['ia']['vertex_buffers'][0]['stride'] == a['vertex_buffers'][0]['stride'], None, row)
            check('index_count', gcmd['args']['IndexCount'] == a['arguments']['IndexCount'], None, row)
            report.append(row)
    return report, summary


def _layout_ranges(rec, stage, slot):
    return rec.setdefault('_ranges', {}).get((stage, slot), [])


def attach_layout_ranges(gen, segments):
    """(offset, size) of every program variable per recorded draw/stage/slot."""
    shaders = gen.shaders
    for seg in segments:
        for r in seg['records']:
            if r['kind'] != 'draw':
                continue
            ranges = {}
            for stage, key, sha in (('VS', 'progVertex', r['vs_sha256']), ('PS', 'progFragment', r['ps_sha256'])):
                prog = _program_by_sha(shaders, r['shader'], r['pass'], key, sha)
                for cbname, cb in prog['layout']['cbuffers'].items():
                    slot = prog['layout']['cb_slots'][cbname]
                    ranges[(stage, slot)] = [(v['offset'], 4 * v['dim'] * max(1, v['array'])) for v in cb['variables']]
            r['_ranges'] = ranges


def _program_by_sha(shaders, shader, pass_index, stage, sha):
    root, doc, s, p = shaders._pass(shader, 0, pass_index)
    for g in p['programs']:
        if g['stage'] == stage and g['dxbc_sha256'] == sha:
            return {'layout': ppv2.unity_layout(g)}
    raise KeyError(sha)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--oracle', type=Path, required=True)
    ap.add_argument('--scene-state', type=Path, required=True)
    ap.add_argument('--dhh', type=Path, default=Path(r'E:\HoneySelect2_ArcticFox\DHH_Data\GraphicSetting.dhs'))
    ap.add_argument('--native-request', type=Path, required=True)
    ap.add_argument('--shader-manifest', type=Path, action='append', required=True)
    ap.add_argument('--compute-manifest', type=Path, required=True)
    ap.add_argument('--fov', type=float, default=23.0)
    ap.add_argument('--projection-m00', type=float, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    oracle, b, segments, gen, external, layer_cfg, vols = build(args)
    attach_layout_ranges(gen, segments)
    report, summary = compare(oracle, b, segments)
    for seg in segments:
        for r in seg['records']:
            r.pop('_ranges', None)
    out = Path(args.out)
    doc = {'schema': 'hs2_ppv2_diff/1', 'summary': summary, 'external_inputs': external,
           'required_live_fields': REQUIRED_LIVE_FIELDS,
           'layer': layer_cfg.__dict__,
           'volumes': [{'name': v.name, 'priority': float(v.priority), 'weight': float(v.weight), 'enabled': v.enabled,
                        'effects': [s.cls for s in v.settings]} for v in vols],
           'actions': report}
    (out / 'diff.json').write_text(json.dumps(doc, indent=1, default=str), encoding='utf-8')
    (out / 'generator_log.json').write_text(json.dumps({'log': gen.log, 'segments': [
        {k: v for k, v in s.items() if k != 'records'} for s in segments]}, indent=1, default=str), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
