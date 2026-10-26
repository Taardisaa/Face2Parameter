"""Per-draw resource bindings derived from scene state and checked against the capture.

For every generated draw the bound SRVs, samplers and IA buffers are re-derived:

* SRV of a shader texture slot (reflection name):
  - material texture: the material's native ShaderPropertySheet texture entry
    (``first_dword`` = Unity texture id, probe-validated against the managed getters)
    -> the scene texture with that id -> the capture resource the resource acceptance
    paired with that scene texture (``resource_acceptance`` table: byte-equal content);
  - global texture (``Shader.SetGlobalTexture`` / engine): the global sheet texture
    entry -> a RenderTexture row of the probe v6 ``render_textures`` export (texture id)
    -> the capture resource with the RenderTexture's name and size (RenderTexture
    names are the D3D debug names), or a scene texture / Unity built-in as above;
* sampler: Unity's TextureSettings -> D3D11 sampler rule (``ppv2_unity.sampler_desc``)
  from the texture's or RenderTexture's filterMode / wrapMode / anisoLevel and
  QualitySettings.anisotropicFiltering;
* IA: vertex stream k / index buffer of the renderer's mesh (acceptance table rows
  ``scene_renderer_id`` + role), or a generated skin stream.

A derived binding is accepted only when it equals the captured binding of that slot;
otherwise the slot stays external and is listed.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ANISO_SETTING = {'Disable': 'Disable', 'Enable': 'Enable', 'ForceEnable': 'ForceEnable'}

# ---- built-in TexEnvs (UnityPlayer.dll 5034cdc2...) -----------------------------------------
# GetBuiltinTexEnvParamName 0x8a0e50 (name table .rdata 0x14b8e40): FastPropertyName 0xC0000000|k
BUILTIN_TEXENV_NAMES = ('white', 'black', 'red', 'gray', 'grey', 'linearGray', 'linearGrey', 'grayscaleRamp',
                        'greyscaleRamp', 'bump', 'blackCube', 'lightmap', 'unity_Lightmap', 'unity_LightmapInd',
                        'unity_ShadowMask', 'unity_DynamicLightmap', 'unity_DynamicDirectionality',
                        'unity_DynamicNormal', '_DitherMaskLOD', '_DitherMaskLOD2D', 'unity_RandomRotation16',
                        'unity_NHxRoughness', 'unity_SpecCube0', 'unity_SpecCube1', 'unity_ProbeVolumeSH')
# builtintex::ReinitBuiltinTextures 0x759c30: GfxDevice+0x1eb8 + 0x18*k <- SetTextureInfo of the builtin texture
# global (names from builtintex::GenerateBuiltinTextures 0x754620); 18/19 dither textures (GenerateDitherTextures
# 0x755010) are not resolved by name here; 23/24 are not set there (engine per draw)
BUILTIN_TEXENV_TEXTURE = {0: 'UnityWhite', 1: 'UnityBlack', 2: 'UnityRed', 3: 'UnityGrey', 4: 'UnityGrey',
                          5: 'UnityLinearGrey', 6: 'UnityLinearGrey', 7: 'UnityGrayscaleRamp', 8: 'UnityGrayscaleRamp',
                          9: 'UnityNormalMap', 10: 'UnityBlackCube', 11: 'UnityBlack', 12: 'UnityBlack',
                          13: 'UnityBlack', 14: 'UnityWhite', 15: 'UnityBlack', 16: 'UnityBlack', 17: 'UnityBlack',
                          20: 'UnityRandomRotation', 21: 'UnityNHxRoughness', 22: 'UnityBlackCube'}
# TextureDimension of those textures (Texture2D via BuildTexture 0x2d2290; Cubemap via 0x7504c0)
BUILTIN_TEXTURE_DIM = {'UnityBlackCube': 4, 'UnityDefaultCube': 4, 'UnityDefault3D': 3, 'UnityDefault2DArray': 5,
                       'UnityDefaultCubeArray': 6}
# builtintex::GetDefaultTexEnv(dim, false) 0x755480 -> static TexEnvs .data 0x14b2470 + 0x18*dim, filled at
# 0x754cb0..0x754e13 from builtintex::GetDefaultTexture's table .data 0x1547e28 + 8*dim
DEFAULT_TEXENV_BY_DIM = {2: 'UnityDefault2D', 3: 'UnityDefault3D', 4: 'UnityDefaultCube', 5: 'UnityDefault2DArray',
                         6: 'UnityDefaultCubeArray'}
# TextureSettings of the built-in textures (Texture+0x38: filter, +0x3c aniso, +0x40 mip bias, +0x44/+0x48/+0x4c
# wrap U/V/W).  The Texture ctor (0x74f050) calls TextureSettings::Reset 0x7074e0 (Bilinear, aniso 1, bias 0,
# Repeat).  Then:
# * BuildTexture 0x2d2290 (2D built-ins): aniso = 0 (0x2d2389);
# * the Cubemap builder 0x7504c0 (UnityBlackCube, UnityDefaultCube): wrap Clamp (0x750622..0x750635);
# * the Texture3D UnityDefault3D (0x754964..0x7549fc): Reset values;
# * GenerateBuiltinTextures sets wrap Clamp on UnityGrayscaleRamp, UnityHalo, UnityNHxRoughness (0x75484d,
#   0x7548ab, 0x754fc9).
# Texture::SetAnisoLimit 0x75b390 re-applies every texture (FindObjectsOfType 0x632d40 has no hide-flag filter),
# so the effective anisotropy follows ppv2_unity.sampler_desc's QualitySettings rule (aniso 0 stays off)
BUILTIN_TEXTURE_BUILDER = {'UnityBlackCube': 'cubemap', 'UnityDefaultCube': 'cubemap', 'UnityDefault3D': 'texture3d'}
BUILTIN_TEXTURE_CLAMPED = {'UnityHalo', 'UnityGrayscaleRamp', 'UnityNHxRoughness'}


def texenv_fields(entry):
    """(texture id, dimension, builtin default name id) of a native sheet TexEnv entry:
    dword 0 (TextureID), dword 1 (TextureDimension), dword 5 (+0x14, the builtin TexEnv the
    property default names, 0xC0000000|k, or 0xffffffff), as read by shaderprops::GetTexEnv."""
    import struct
    raw = bytes.fromhex(entry['value_hex'])
    tid, dim = struct.unpack_from('<Ii', raw, 0)
    bdef = struct.unpack_from('<I', raw, 0x14)[0]
    return tid, dim, bdef


def resolve_texenv(name, dim, material_sheet, global_sheet):
    """``shaderprops::GetTexEnv(local, global, name, dim, false)`` 0x85b850 for a property that is
    not a built-in TexEnv name: material sheet entry with a texture of the requested dimension
    (or dim == Any) -> that texture; else the entry's +0x14 built-in TexEnv when it holds a texture
    of that dimension (0x85aa80); else the same on the global sheet; else GetDefaultTexEnv(dim).
    Returns ('texture', id, sheet) / ('builtin', k) / ('default', dim)."""
    for sheet, label in ((material_sheet, 'material'), (global_sheet, 'global')):
        for e in (sheet or {}).get('entries', []):
            if e['type'] != 'texture' or e['name'] != name:
                continue
            tid, edim, bdef = texenv_fields(e)
            if tid != 0 and (dim == edim or dim == 1):
                return ('texture', tid, label)
            if bdef != 0xFFFFFFFF and bdef & 0xC0000000:
                k = bdef & 0x3FFFFFFF
                tex = BUILTIN_TEXENV_TEXTURE.get(k)
                if tex is not None and (dim == BUILTIN_TEXTURE_DIM.get(tex, 2) or dim == 1):
                    return ('builtin', k, label)
            break
    return ('default', dim, None)


def builtin_texture_settings(name):
    """TextureSettings of a built-in texture (see BUILTIN_TEXTURE_BUILDER)."""
    kind = BUILTIN_TEXTURE_BUILDER.get(name, 'build_texture_2d')
    d = {'filterMode': 'Bilinear', 'wrapMode': 'Repeat', 'anisoLevel': 1, 'mipMapBias': 0}
    if kind == 'build_texture_2d':
        d['anisoLevel'] = 0
    if kind == 'cubemap' or name in BUILTIN_TEXTURE_CLAMPED:
        d['wrapMode'] = 'Clamp'
    return d


class BindingDeriver:
    def __init__(self, cap, lib, acceptance_table, scene, skin_resources=()):
        self.cap, self.lib, self.scene = cap, lib, scene
        rows = json.loads(Path(acceptance_table).read_text(encoding='utf-8'))
        self.tex_res = collections.defaultdict(set)      # scene texture id -> capture resources
        self.builtin_res = {}
        self.mesh_res = {}                               # capture resource -> (renderer id, role)
        self.row_of = {r['resource']: r for r in rows}
        for r in rows:
            rq = r.get('request')
            if not isinstance(rq, dict) or r['status'] not in ('byte_exact', 'raw_readback_byte_exact',
                                                               'runtime_descriptor_equal'):
                continue
            if 'scene_texture_id' in rq:
                for sid in [rq['scene_texture_id']] + list(rq.get('scene_texture_ids_same_identity') or []):
                    self.tex_res[int(sid)].add(r['resource'])
            elif 'unity_builtin_texture' in rq:
                self.builtin_res[r['resource']] = rq['unity_builtin_texture']
            elif 'scene_renderer_id' in rq:
                self.mesh_res[r['resource']] = (int(rq['scene_renderer_id']), rq['role'])
        self.skin_res = set(skin_resources)
        # installed Mesh object of each renderer (acceptance provenance of its mesh streams) and the
        # engine default vertex buffer (hs2pipe.resources 'unity_default_vertex_buffer')
        self.mesh_src, self.default_vb = {}, set()
        for r in rows:
            rq = r.get('request') if isinstance(r.get('request'), dict) else {}
            p = (r.get('entry') or {}).get('provenance') or {}
            if ('scene_renderer_id' in rq and r.get('status') == 'byte_exact' and p.get('unity_type') == 'Mesh'
                    and p.get('source') and p.get('path_id') is not None):
                self.mesh_src.setdefault(int(rq['scene_renderer_id']), (p['source'], int(p['path_id'])))
            if 'unity_default_vertex_buffer' in rq and r.get('status') == 'byte_exact':
                self.default_vb.add(r['resource'])
        self._stream_eq = {}
        self._scene_key = None
        self.set_scene(scene)
        self.views_of = collections.defaultdict(list)
        for v in cap.program['views']:
            self.views_of[v['resource']].append(v)
        self.res_by_name = collections.defaultdict(list)
        for e in cap.program['resources']:
            for n in str(e.get('name', '')).split(' | '):
                self.res_by_name[n.strip()].append(e['id'])
        self.counts = collections.Counter()
        self.external = collections.Counter()

    def set_scene(self, scene):
        """Scene state of the render being emitted (texture ids, RenderTexture rows)."""
        if self._scene_key == id(scene):
            return
        self._scene_key, self.scene = id(scene), scene
        self.tex_by_native = {}
        for t in scene['textures']:
            nid = (t.get('native') or {}).get('texture_id_0x50')
            if nid is not None:
                self.tex_by_native[int(nid)] = t
        self.rt_by_native = {}
        rec = ((scene.get('_v5') or {}).get('render_record') or {}).get('data') or {}
        self.rts = rec.get('render_textures') or scene.get('render_textures_compact') or []
        for t in self.rts:
            nid = (t.get('native') or {}).get('texture_id_0x50')
            if nid is not None:
                self.rt_by_native[int(nid)] = t
        q = scene['statics']['UnityEngine.QualitySettings']
        self.aniso = str(q.get('anisotropicFiltering', 'Enable')).split('.')[-1].split('=')[0]

    # ------------------------------------------------------------ helpers
    def _sheet_textures(self, sheet):
        out = {}
        for e in (sheet or {}).get('entries', []):
            if e['type'] == 'texture':
                out[e['name']] = int(e['first_dword'])
        return out

    def _sampler(self, props):
        from .ppv2_unity import TextureRef, sampler_desc
        wu = str(props.get('wrapModeU', props.get('wrapMode'))).split('.')[-1].split('=')[0]
        wv = str(props.get('wrapModeV', props.get('wrapMode'))).split('.')[-1].split('=')[0]
        ww = str(props.get('wrapModeW', props.get('wrapMode'))).split('.')[-1].split('=')[0]
        if not (wu == wv == ww):
            return None                     # per-axis wrap not in the transcribed rule
        if float(props.get('mipMapBias', 0) or 0) != 0.0:
            return None
        filt = str(props['filterMode']).split('.')[-1].split('=')[0]
        tex = TextureRef(0, None, 1, 1, filter_mode=filt, wrap_mode=wu, aniso_level=int(props['anisoLevel']))
        return sampler_desc(tex, self.aniso)

    def _builtin(self, texname, label):
        cands = {r for r, n in self.builtin_res.items() if n == texname}
        props = builtin_texture_settings(texname)
        # when the scene export carries the built-in texture, its managed settings must equal the rule
        ex = [t for t in self.scene.get('textures') or [] if t.get('name') == texname]
        if ex:
            p = ex[0].get('properties') or {}

            def bare(v):
                return str(v).split('.')[-1].split('=')[0]
            if (bare(p.get('filterMode')) != props['filterMode'] or int(p.get('anisoLevel', -1)) != props['anisoLevel']
                    or bare(p.get('wrapMode')) != props['wrapMode']):
                return cands, None, label + ' [exported settings differ from the builder rule]'
        return cands, props, label

    def _texture_source(self, name, material_sheet, global_sheet, dim=1, renderer_id=None):
        """(candidate capture resources, sampler props, source label) of a texture slot
        (shaderprops::GetTexEnv 0x85b850, :func:`resolve_texenv`)."""
        if name in BUILTIN_TEXENV_NAMES:
            k = BUILTIN_TEXENV_NAMES.index(name)
            return self._engine_builtin(k, name, dim, renderer_id)
        kind = resolve_texenv(name, dim, material_sheet, global_sheet)
        if kind[0] == 'texture':
            tid, label = kind[1], kind[2]
            if label == 'material':
                t = self.tex_by_native.get(tid)
                if t is not None:
                    return self.tex_res.get(int(t['id']), set()), t['properties'], 'material texture %s' % t.get('name')
                rt = self.rt_by_native.get(tid)
                if rt is not None:
                    return self._rt_resources(rt), rt, 'material RenderTexture %s' % rt.get('name')
                return set(), None, 'material texture id %d not exported' % tid
            rt = self.rt_by_native.get(tid)
            if rt is not None:
                return self._rt_resources(rt), rt, 'global RenderTexture %s' % rt.get('name')
            t = self.tex_by_native.get(tid)
            if t is not None:
                return self.tex_res.get(int(t['id']), set()), t['properties'], 'global texture %s' % t.get('name')
            return set(), None, 'global texture id %d not exported' % tid
        if kind[0] == 'builtin':
            texname = BUILTIN_TEXENV_TEXTURE[kind[1]]
            return self._builtin(texname, 'builtin TexEnv %s = %s (%s sheet +0x14 default, GetTexEnv 0x85b966)'
                                 % (BUILTIN_TEXENV_NAMES[kind[1]], texname, kind[2]))
        texname = DEFAULT_TEXENV_BY_DIM.get(kind[1])
        if texname is None:
            return set(), None, 'default TexEnv for dimension %r not resolved' % (kind[1],)
        return self._builtin(texname, 'GetDefaultTexEnv(%d) = %s (no sheet entry, 0x85ba2d)' % (kind[1], texname))

    def _engine_builtin(self, k, name, dim, renderer_id):
        """Built-in TexEnvs the engine sets per draw (GetTexEnv step 1: GfxDevice+0x1eb8 + 0x18*k)."""
        if k in (22, 23):
            # SetReflections 0x6c11e0 with the default sample (no active reflection probe):
            # ReflectionProbes::GenerateDefaultSample 0x67d5d0 -> both probes use the default
            # reflection texture (black cube when null)
            if any((p.get('properties') or {}).get('enabled') and p.get('active_in_hierarchy', True)
                   for p in self.scene.get('reflection_probes') or []):
                return set(), None, 'engine %s: active reflection probes (sample blend not transcribed)' % name
            rec = ((self.scene.get('_v5') or {}).get('render_record') or {}).get('data') or {}
            d = (rec.get('reflection_probe_default') or {}).get('defaultTexture')
            texname = (d or {}).get('ref_name') if d else 'UnityBlackCube'
            return self._builtin(texname, 'engine %s = default reflection texture %s (GenerateDefaultSample 0x67d5d0, '
                                 'SetReflections 0x6c11e0)' % (name, texname))
        if k == 24:
            r = next((x for x in self.scene['renderers'] if int(x['id']) == int(renderer_id or 0)), None)
            usage = str(((r or {}).get('properties') or {}).get('lightProbeUsage', ''))
            if 'UseProxyVolume' in usage:
                return set(), None, 'engine %s: light probe proxy volume in use (not transcribed)' % name
            return self._builtin('UnityDefault3D', 'engine %s = GetDefaultTexture(3D) UnityDefault3D (no proxy volume: '
                                 'LightProbeProxyVolumeSample::Invalidate 0x6cc1d0, BatchRenderer 0x6cb18d)' % name)
        texname = BUILTIN_TEXENV_TEXTURE.get(k)
        if texname is None:
            return set(), None, 'built-in TexEnv %s texture not resolved' % name
        return self._builtin(texname, 'built-in TexEnv %s = %s (ReinitBuiltinTextures 0x759c30)' % (name, texname))

    def _rt_resources(self, rt):
        name = rt.get('name') or ''
        if not name:
            return set()
        d = rt['descriptor']
        out = set()
        for rid in self.res_by_name.get(name, []):
            desc = self.cap.obj(rid)['desc']
            if int(desc.get('Width', -1)) == int(d['width']) and int(desc.get('Height', -1)) == int(d['height']):
                out.add(rid)
        return out

    def _buffer_reason(self, b, renderer_id):
        r = self.row_of.get(b)
        if r is None:
            return 'not in the acceptance table'
        rq = r.get('request') if isinstance(r.get('request'), dict) else {}
        if 'unity_default_vertex_buffer' in rq:
            return 'engine default vertex stream: channel-to-stream rule not transcribed'
        if 'unity_builtin_geometry' in rq:
            return 'engine built-in geometry %s: binding rule not transcribed' % rq['unity_builtin_geometry']
        if 'scene_renderer_id' in rq:
            return 'mesh stream of renderer %s, not of the drawn renderer %s' % (rq['scene_renderer_id'], renderer_id)
        if r['status'] == 'runtime':
            return 'runtime buffer (%s): skin output not generated / dynamic ring' % (r.get('debug_name') or 'unnamed')
        return 'acceptance status %s' % r['status']

    # ------------------------------------------------------------ derive
    def rt_row(self, name, width, height):
        """RenderTexture row (render record ``render_textures``, probe v6) of a named engine RT."""
        hits = [t for t in self.rts if t.get('name') == name and int(t['descriptor']['width']) == int(width)
                and int(t['descriptor']['height']) == int(height)]
        return hits[0] if len(hits) == 1 else None

    # ------------------------------------------------------------ IA (hs2pipe.ia_bindings)
    def _capture_bytes(self, res):
        e = self.cap.obj(res)
        init = e.get('initial')
        return self.cap.blob(init[0]['blob']) if init else None

    def _mesh_stream_equal(self, res, src, k):
        key = (res, src, k)
        if key not in self._stream_eq:
            from . import ia_bindings as ia
            data = self._capture_bytes(res)
            self._stream_eq[key] = data is not None and data == ia.stream_bytes(src[0], src[1], k)
        return self._stream_eq[key]

    def _previous_vb(self, cap_index, slot):
        for j in range(cap_index - 1, max(-1, cap_index - 400), -1):
            st = self.cap.commands[j].get('state')
            if isinstance(st, dict) and 'ia' in st:
                for vb in st['ia']['vertex_buffers']:
                    if vb['slot'] == slot:
                        return vb['buffer']
                return None
        return None

    def derive_ia(self, state, stage_programs, renderer, label, subset, cap_index):
        """Vertex-buffer slots / draw arguments of a mesh draw from ``ia_bindings``; returns
        (derived, external, draw_args_status) or None when the mesh object is not known."""
        from . import ia_bindings as ia
        src = self.mesh_src.get(int(renderer.get('id') or 0))
        vs = stage_programs.get('vs')
        if src is None or vs is None:
            return None
        fmt = ia.mesh_format(*src)
        required = int(((vs.record.get('channels') or {}).get('m_SourceMap')) or 0)
        skinned = renderer.get('kind') == 'SkinnedMeshRenderer'
        plan = ia.stream_plan(fmt, required, skinned=skinned, previous_stream=(label == 'motion_vectors' and skinned))
        derived, external = 0, []
        for vb in state['ia']['vertex_buffers']:
            b, slot = vb['buffer'], vb['slot']
            if slot < len(plan):
                want = plan[slot]
                if want in ('skin', 'skin_previous'):
                    ok = b in self.skin_res
                elif want == 'default':
                    ok = b in self.default_vb
                else:
                    k = want[1]
                    ok = ((b in self.mesh_res and self.mesh_res[b] == (int(renderer['id']), 'vertex_stream_%d' % k))
                          or self._mesh_stream_equal(b, src, k))
                why = 'stream %s (ExtractMeshBuffers 0x7a3f40)' % (want,)
            else:
                ok = b == self._previous_vb(cap_index, slot)
                why = 'slot above streamCount %d: device state (BindBuffersForDraw 0x5438e0)' % len(plan)
            if ok:
                derived += 1
                self.counts['ia_vertex_buffer'] += 1
            else:
                external.append('ia vertex buffer %s (%s not matched)' % (b, why))
                self.external['ia vertex buffer'] += 1
        ib = (state['ia'].get('index_buffer') or {}).get('buffer')
        if ib is not None:
            if ib in self.mesh_res and self.mesh_res[ib] == (int(renderer['id']), 'index_buffer'):
                derived += 1
                self.counts['ia_index_buffer'] += 1
            else:
                external.append('ia index buffer %s (%s)' % (ib, self._buffer_reason(ib, renderer['id'])))
                self.external['ia index buffer'] += 1
        args = None
        if subset is not None and int(subset) >= 0 and fmt['submeshes']:
            # material slots beyond the submesh count draw the last submesh
            args = ia.draw_args(fmt, min(int(subset), len(fmt['submeshes']) - 1))
        cap_args = self.cap.commands[cap_index].get('args') or {}
        da = 'derived' if args is not None and all(int(cap_args.get(k_, -1)) == v for k_, v in args.items()) else 'external'
        return derived, external, da

    def derive(self, state, stage_programs, material, global_sheet, renderer_id=None, engine=None,
               renderer=None, label=None, subset=None, cap_index=None):
        """Per-slot derivation of one draw; returns {'derived': n, 'external': [labels]}.

        ``engine``: textures the engine binds during the render (not in the pre-render
        sheets): {property: (capture resource, RenderTexture name, source)}; see
        ``frame_orchestrate_paired.engine_textures``."""
        derived, external = 0, []
        engine = engine or {}
        for stage in ('vs', 'ps'):
            sp = stage_programs.get(stage)
            if sp is None:
                continue
            bound_srv = {x['slot']: x['view'] for x in state[stage]['srvs']}
            bound_smp = {x['slot']: x['sampler'] for x in state[stage]['samplers']}
            for t in sp.resources()['textures']:
                name = t['name']
                view = bound_srv.get(t['srv'])
                if view is None:
                    continue
                res = self.cap.obj(view)['resource'] if view in self.cap.by_id else None
                if name in engine:
                    eres, ename, esrc = engine[name]
                    d_ = self.cap.obj(eres)['desc']
                    row = self.rt_row(ename, d_.get('Width', -1), d_.get('Height', -1))
                    cands, props, src = {eres}, row, 'engine %s (%s)' % (ename, esrc)
                else:
                    cands, props, src = self._texture_source(name, (material or {}).get('_native_sheet'), global_sheet,
                                                             int(t.get('dim', 1)), renderer_id)
                if res is not None and res in cands:
                    derived += 1
                    self.counts['srv'] += 1
                else:
                    external.append('%s.%s srv (%s)' % (stage, name, src))
                    self.external['srv: ' + src.split(' ')[0] + ' ' + src.split(' ')[1]] += 1
                if t['sampler'] is not None and t['sampler'] >= 0 and t['sampler'] in bound_smp:
                    want = self._sampler(props) if props else None
                    got = self.cap.obj(bound_smp[t['sampler']]).get('desc')
                    if want is not None and got == want:
                        derived += 1
                        self.counts['sampler'] += 1
                    else:
                        external.append('%s.%s sampler' % (stage, name))
                        self.external['sampler'] += 1
        ia_res = (self.derive_ia(state, stage_programs, renderer, label, subset, cap_index)
                  if renderer is not None and cap_index is not None else None)
        if ia_res is not None:
            d_, e_, da = ia_res
            return {'derived': derived + d_, 'external': external + e_, 'draw_args': da,
                    'ia': {'derived': d_, 'external': len(e_)}}
        for vb in state['ia']['vertex_buffers']:
            b = vb['buffer']
            if b in self.skin_res:
                derived += 1
                self.counts['ia_skin_stream'] += 1
            elif b in self.mesh_res and self.mesh_res[b][0] == renderer_id:
                derived += 1
                self.counts['ia_mesh_stream'] += 1
            else:
                external.append('ia vertex buffer %s (%s)' % (b, self._buffer_reason(b, renderer_id)))
                self.external['ia vertex buffer'] += 1
        ib = (state['ia'].get('index_buffer') or {}).get('buffer')
        if ib is not None:
            if ib in self.mesh_res and self.mesh_res[ib][0] == renderer_id:
                derived += 1
                self.counts['ia_index_buffer'] += 1
            else:
                external.append('ia index buffer %s (%s)' % (ib, self._buffer_reason(ib, renderer_id)))
                self.external['ia index buffer'] += 1
        return {'derived': derived, 'external': external}
