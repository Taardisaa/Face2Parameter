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


class BindingDeriver:
    def __init__(self, cap, lib, acceptance_table, scene, skin_resources=()):
        self.cap, self.lib, self.scene = cap, lib, scene
        rows = json.loads(Path(acceptance_table).read_text(encoding='utf-8'))
        self.tex_res = collections.defaultdict(set)      # scene texture id -> capture resources
        self.builtin_res = {}
        self.mesh_res = {}                               # capture resource -> (renderer id, role)
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
        self.tex_by_native = {}
        for t in scene['textures']:
            nid = (t.get('native') or {}).get('texture_id_0x50')
            if nid is not None:
                self.tex_by_native[int(nid)] = t
        self.rt_by_native = {}
        for t in scene.get('render_textures_compact') or []:
            nid = (t.get('native') or {}).get('texture_id_0x50')
            if nid is not None:
                self.rt_by_native[int(nid)] = t
        q = scene['statics']['UnityEngine.QualitySettings']
        self.aniso = str(q.get('anisotropicFiltering', 'Enable')).split('.')[-1].split('=')[0]
        self.views_of = collections.defaultdict(list)
        for v in cap.program['views']:
            self.views_of[v['resource']].append(v)
        self.res_by_name = collections.defaultdict(list)
        for e in cap.program['resources']:
            for n in str(e.get('name', '')).split(' | '):
                self.res_by_name[n.strip()].append(e['id'])
        self.counts = collections.Counter()
        self.external = collections.Counter()

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

    def _texture_source(self, name, material_sheet, global_sheet):
        """(candidate capture resources, sampler props, source label) of a texture slot."""
        mt = self._sheet_textures(material_sheet)
        if name in mt:
            t = self.tex_by_native.get(mt[name])
            if t is not None:
                return self.tex_res.get(int(t['id']), set()), t['properties'], 'material texture %s' % t.get('name')
            rt = self.rt_by_native.get(mt[name])
            if rt is not None:
                return self._rt_resources(rt), rt, 'material RenderTexture %s' % rt.get('name')
            return set(), None, 'material texture id %d not exported' % mt[name]
        gt = self._sheet_textures(global_sheet)
        if name in gt:
            rt = self.rt_by_native.get(gt[name])
            if rt is not None:
                return self._rt_resources(rt), rt, 'global RenderTexture %s' % rt.get('name')
            t = self.tex_by_native.get(gt[name])
            if t is not None:
                return self.tex_res.get(int(t['id']), set()), t['properties'], 'global texture %s' % t.get('name')
            return set(), None, 'global texture id %d not exported' % gt[name]
        return set(), None, 'texture %s in no sheet (built-in default)' % name

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

    # ------------------------------------------------------------ derive
    def derive(self, state, stage_programs, material, global_sheet, renderer_id=None):
        """Per-slot derivation of one draw; returns {'derived': n, 'external': [labels]}."""
        derived, external = 0, []
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
                res = self.cap.obj(view)['resource'] if view in self.cap.objects else None
                cands, props, src = self._texture_source(name, (material or {}).get('_native_sheet'), global_sheet)
                if res is not None and res in self.builtin_res and not cands:
                    cands, src = {res}, 'Unity built-in %s' % self.builtin_res[res]
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
        for vb in state['ia']['vertex_buffers']:
            b = vb['buffer']
            if b in self.skin_res:
                derived += 1
                self.counts['ia_skin_stream'] += 1
            elif b in self.mesh_res and self.mesh_res[b][0] == renderer_id:
                derived += 1
                self.counts['ia_mesh_stream'] += 1
            else:
                external.append('ia vertex buffer %s' % b)
                self.external['ia vertex buffer'] += 1
        ib = (state['ia'].get('index_buffer') or {}).get('buffer')
        if ib is not None:
            if ib in self.mesh_res and self.mesh_res[ib][0] == renderer_id:
                derived += 1
                self.counts['ia_index_buffer'] += 1
            else:
                external.append('ia index buffer %s' % ib)
                self.external['ia index buffer'] += 1
        return {'derived': derived, 'external': external}
