"""Byte-for-byte acceptance of ``src/render/hs2pipe/resources.py`` against a paired capture.

For every resource the captured frame reads, this tool

1. takes the capture-start content RenderDoc recorded for it (``CreateBuffer``
   InitialData, texture creation ``SubresourceContents``, ``Internal::Initial
   Contents`` normalized from its RowPitch to tight rows, or the pre-capture full
   ``UpdateSubresource``), plus its creation descriptor and SRV descriptors;
2. decides from the frame graph whether the frame reads that capture-start content
   (no in-frame writer before the read) or content produced inside the frame;
3. asks the resource API for the same object *from scene state* (scene texture id,
   scene renderer id, Unity built-in name) and compares descriptor, SRV descriptor
   and every subresource's bytes (SHA-256, streamed; nothing is dumped twice).

Read-only on the game install; the capture is not modified.  Bulky outputs
(the per-resource table) go to ``--out`` (an ignored or external directory); the
compact receipt goes to ``--receipt``.

    python tools/offline_renderer/resource_acceptance.py \
        --xml .../parse_04d1e76f/capture.zip.xml --scene .../frames/scene_005.json \
        --index E:/HS2_offline_renderer_data/resources_20261020/source_index_v1.json \
        --out E:/HS2_offline_renderer_data/resources_20261020/acceptance_v1 \
        --receipt tools/offline_renderer/resources_acceptance_20261020.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.render import frame_program as fp  # noqa: E402
from src.render.hs2pipe import resources as R  # noqa: E402
from tools.offline_renderer import frame_graph as fg  # noqa: E402

TEX = {'CreateTexture1D': 'tex1d', 'CreateTexture2D': 'tex2d', 'CreateTexture3D': 'tex3d'}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _list(v):
    return list(v) if isinstance(v, fg.Repeated) else ([] if v is None else [v])


class CaptureOracle:
    """Capture-start content and descriptors of every created resource."""

    def __init__(self, xml, zip_path, chunk_cache=None):
        self.xml, self.zip_path = Path(xml), Path(zip_path)
        if chunk_cache and Path(chunk_cache).exists():
            self.chunks = pickle.loads(Path(chunk_cache).read_bytes())
        else:
            self.chunks = fg.load_chunks(self.xml)
            if chunk_cache:
                Path(chunk_cache).write_bytes(pickle.dumps(self.chunks))
        self.blobs = fg.Blobs(self.zip_path)
        self.graph = fg.FrameGraph(self.chunks, blobs=self.blobs).build()
        self.res = self.graph.resources
        self.initial = {}
        self.pre_updates = {}
        self.srvs = {}
        begun = False
        for c in self.chunks:
            n, f = c['name'], c['fields']
            if n == 'Internal::Initial Contents':
                self.initial[f['id']] = c
            elif n == 'Internal::Beginning of Capture':
                begun = True
            elif n == 'ID3D11Device::CreateShaderResourceView':
                self.srvs.setdefault(f['pResource'], []).append(f['pDesc'])
            elif not begun and n in ('ID3D11DeviceContext::UpdateSubresource', 'ID3D11DeviceContext::UpdateSubresource1'):
                self.pre_updates.setdefault(f['pDstResource'], []).append(c)

    def kind(self, rid):
        r = self.res[rid]
        if r['kind'] == 'CreateBuffer':
            return 'buffer'
        k = TEX.get(r['kind'])
        if k == 'tex2d' and r['descriptor'].get('MiscFlags', 0) & 4:
            return 'texcube'
        return k

    def capture_start(self, rid):
        """(source label, [tight subresource bytes]) or (label, None)."""
        r = self.res[rid]
        if r['kind'] == 'CreateBuffer':
            ref = r.get('initial_data')
            return ('CreateBuffer.InitialData', [self.blobs.read(ref)]) if ref else ('none', None)
        kind = self.kind(rid)
        desc = r['descriptor']
        lays = fp.subresource_layouts(kind, desc)
        if rid in self.initial:
            f = self.initial[rid]['fields']
            n = f['NumSubresources']
            refs = _list(f['SubresourceContents'])
            rps = _list(f['RowPitch'])
            dps = _list(f.get('DepthPitch')) if kind == 'tex3d' else [None] * n
            if not (n == len(lays) == len(refs) == len(rps)):
                raise R.ResourceUnresolved('initial contents count mismatch for %d' % rid)
            out = []
            for lay, ref, rp, dp in zip(lays, refs, rps, dps):
                data = self.blobs.read(ref)
                rows = []
                for z in range(lay['depth']):
                    for y in range(lay['rows']):
                        o = (z * dp if dp else 0) + y * rp
                        rows.append(data[o:o + lay['row_bytes']])
                tight = b''.join(rows)
                if len(tight) != lay['bytes']:
                    raise R.ResourceUnresolved('initial contents short for %d' % rid)
                out.append(tight)
            return 'Internal::Initial Contents (RowPitch-normalized)', out
        refs = r.get('creation_subresource_blobs') or []
        if refs:
            return 'creation SubresourceContents', [self.blobs.read(x) for x in refs]
        if rid in self.pre_updates:
            out = [None] * len(lays)
            for c in self.pre_updates[rid]:
                f = c['fields']
                if f.get('IsUpdate'):
                    return 'pre-capture boxed update (not normalized)', None
                out[f['DstSubresource']] = self.blobs.read(f['Contents'])
            if all(x is not None for x in out):
                return 'pre-capture full UpdateSubresource', out
        return 'none', None

    def frame_reads(self):
        """{resource: {'via': set, 'capture_start_read': bool}} for VB/IB/SRV reads."""
        reads = {}
        for a in self.graph.actions:
            for vb in a.get('vertex_buffers', []):
                if vb['resource']:
                    row = reads.setdefault(vb['resource'], {'via': set(), 'capture_start_read': False, 'strides': set()})
                    row['via'].add('vb')
                    row['strides'].add(vb['stride'])
                    if any(isinstance(p, dict) and p.get('kind') == 'capture_start_bytes_in_CreateBuffer_record'
                           for p in vb['producers']):
                        row['capture_start_read'] = True
            ib = a.get('index_buffer')
            if ib:
                row = reads.setdefault(ib['resource'], {'via': set(), 'capture_start_read': False, 'strides': set()})
                row['via'].add('ib')
                if any(isinstance(p, dict) and p.get('kind') == 'capture_start_bytes_in_CreateBuffer_record'
                       for p in ib['producers']):
                    row['capture_start_read'] = True
            for info in a.get('shaders', {}).values():
                for s in info['srvs']:
                    if s.get('unbound') or 'subresources' not in s:
                        continue
                    row = reads.setdefault(s['resource'], {'via': set(), 'capture_start_read': False, 'strides': set()})
                    row['via'].add('srv')
                    if any(p['base'] == 'capture_start' and not p['partial_writers_since_base'] for p in s['producers']):
                        row['capture_start_read'] = True
                    row.setdefault('view_formats', set()).add(s.get('view_format'))
        return reads


def _strip_clone(name):
    return name[:-len('(Clone)')] if name.endswith('(Clone)') else name


def compare_entry(oracle, rid, entry):
    r = oracle.res[rid]
    label, cap = oracle.capture_start(rid)
    want_desc = r['descriptor']
    desc_equal = all(entry.desc.get(k) == want_desc.get(k) for k in want_desc)
    srv_equal = None
    if entry.srv is not None:
        srv_equal = entry.srv in oracle.srvs.get(rid, [])
    if cap is None or entry.initial is None:
        bytes_equal = None
        sub = None
    else:
        mine = [s['data'] for s in entry.initial]
        sub = [hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest() for a, b in zip(mine, cap)]
        bytes_equal = len(mine) == len(cap) and all(sub)
    return {'capture_content': label, 'desc_equal': desc_equal, 'srv_desc_equal': srv_equal,
            'bytes_equal': bytes_equal, 'subresources_equal': None if sub is None else sum(sub),
            'subresources': None if cap is None else len(cap),
            'capture_desc': want_desc, 'entry': entry.summary()}


# --------------------------------------------------------------- consumers
def consumer_programs(oracle, library_dirs):
    """{resource: {'programs': installed shader names of the actions that read it,
    'semantics': vertex semantics it feeds}}."""
    lib = None
    if library_dirs:
        from src.render.hs2pipe.shader_library import ShaderLibrary
        lib = ShaderLibrary(library_dirs)
    out = {}

    def names(info):
        if lib is None or not info:
            return set()
        m = lib.match(info.get('dxbc_sha256')) or []
        return {x.get('shader') for x in m if isinstance(x, dict) and x.get('shader')}
    for a in oracle.graph.actions:
        sh = a.get('shaders', {})
        prog = set()
        for st in ('VS', 'PS', 'CS'):
            prog |= names(sh.get(st))
        for vb in a.get('vertex_buffers', []):
            if vb['resource']:
                row = out.setdefault(vb['resource'], {'programs': set(), 'semantics': set()})
                row['programs'] |= prog
                row['semantics'] |= set(vb['semantics'])
        if a.get('index_buffer'):
            out.setdefault(a['index_buffer']['resource'], {'programs': set(), 'semantics': set()})['programs'] |= prog
        for info in sh.values():
            for srv in info.get('srvs', []):
                if not srv.get('unbound'):
                    out.setdefault(srv['resource'], {'programs': set(), 'semantics': set()})['programs'] |= prog
    return out


UI_PROGRAMS = ('UI/Default',)


def classify_dynamic_buffer(name, info, consumers):
    """Runtime producer of a DYNAMIC buffer read by the frame (from its consumers)."""
    c = consumers.get('programs', set())
    sem = consumers.get('semantics', set())
    if any(p in UI_PROGRAMS for p in c):
        return {'kind': 'ui_canvas_geometry', 'class': 'b',
                'producer': 'UnityEngine.UI Canvas batching (CanvasRenderer meshes); UI layout state is not in the '
                            'scene export'}
    if 'TEXCOORD4' in sem:
        return {'kind': 'previous_skinned_vertices', 'class': 'a',
                'producer': 'Unity CPU skinning of the previous frame (SkinnedMeshRenderer motion-vector source); '
                            'owned by the skinning generator (docs/hs2_offline_head_skinning.md)'}
    if 'TANGENT0' in sem:
        return {'kind': 'skinning_output', 'class': 'a',
                'producer': 'Unity CPU skinning (SkinnedMeshRenderer) written by Map in the frame; skinning generator'}
    if any(p.startswith('Hidden/') for p in c):
        return {'kind': 'unity_dynamic_vertex_ring', 'class': 'a',
                'producer': 'GfxDevice dynamic VB/IB ring for immediate draws (Blit quads, deferred and screen-space '
                            'quads); vertex bytes recovered by the generators (hs2pipe.ppv2 blit_quad), ring placement '
                            'owned by them'}
    if 'Legacy Shaders/Particles/Additive' in c or 'Standard' in c:
        return {'kind': 'dynamic_batch_or_particles', 'class': 'b',
                'producer': 'Unity dynamic batching / ParticleSystem geometry (Maker background dust particles); '
                            'particle simulation state is not in the scene export'}
    if {'POSITION0', 'NORMAL0', 'TANGENT0'} & sem:
        return {'kind': 'skinning_output', 'class': 'a',
                'producer': 'Unity CPU skinning (SkinnedMeshRenderer) written by Map in the frame; skinning generator'}
    return {'kind': 'unclassified_dynamic_buffer', 'class': 'b', 'producer': 'unknown', 'programs': sorted(c)}


def classify_texture(name, desc, info, consumers):
    c = consumers.get('programs', set())
    if name == 'Font Texture':
        return {'kind': 'dynamic_font_atlas', 'class': 'b',
                'producer': 'UnityEngine.Font dynamic glyph atlas: FreeType rasterization of every string requested '
                            'since startup (TextGenerator); content depends on the UI text history'}
    if name == 'Internal Curves Texture':
        return {'kind': 'ppv2_curves', 'class': 'a',
                'producer': 'hs2pipe.ppv2 ColorGrading curves upload inside the frame (Map + CopySubresourceRegion)'}
    if desc.get('BindFlags', 0) & (32 | 64 | 128):
        if info['capture_start_read']:
            return {'kind': 'render_target_previous_frame', 'class': 'b',
                    'producer': 'render target read before any write in this frame (temporal history)',
                    'programs': sorted(c)}
        return {'kind': 'render_target_in_frame', 'class': 'a',
                'producer': 'render target produced inside the frame by the frame generators', 'programs': sorted(c)}
    return {'kind': 'script_texture', 'class': 'b', 'producer': 'runtime Texture2D without an installed identity',
            'programs': sorted(c)}


# ------------------------------------------------------------------- run
def run(args):
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    oracle = CaptureOracle(args.xml, args.zip or Path(args.xml).with_suffix(''), args.chunk_cache)
    scene = json.loads(Path(args.scene).read_text(encoding='utf-8'))
    index = R.SourceIndex.load(args.index)
    stale = index.stale()
    if stale:
        raise SystemExit('source index is stale for %d files, rebuild it' % len(stale))
    sideloader = None
    plugin_meshes = {}
    uncensor = None
    if args.sideloader_scan:
        from src.render.hs2pipe import sideloader_assets as S
        sideloader = S.SideloaderState.load(args.sideloader_scan)
    if args.card_state:
        if sideloader is None:
            raise SystemExit('--card-state needs --sideloader-scan')
        snap = json.loads(Path(args.card_state).read_text(encoding='utf-8'))['snapshot']
        meta = snap['meta']
        card = {'sex': 1 if snap['maker_sex_label'] == 'female' else 0, 'birth_day': meta['birth_day'],
                'personality': meta['personality'], 'voice_pitch': meta['voice_pitch']}
        uncensor = R.uncensor_selector_body(card, sideloader)
        if uncensor['source']:
            for n in R.UNCENSOR_BODY_NAMES:
                plugin_meshes[n] = {'source': uncensor['source'], 'asset': uncensor['asset'],
                                    'producer': 'UncensorSelector ReloadCharacterBody/UpdateMeshRenderer',
                                    'body_guid': uncensor['guid']}
    sr = R.SceneResources(scene, index, preferred_sources=args.preferred or (), sideloader=sideloader,
                          plugin_meshes=plugin_meshes, xunity=R.XUnityTextureTranslations())
    reads = oracle.frame_reads()
    consumers = consumer_programs(oracle, args.shader_library or ())
    ctx = Context(oracle, sr, scene, args)
    rows = []
    for rid in sorted(reads):
        info = reads[rid]
        r = oracle.res[rid]
        names = r['debug_names']
        name = names[-1] if names else None
        row = {'resource': rid, 'kind': oracle.kind(rid), 'debug_name': name, 'via': sorted(info['via']),
               'capture_start_read': info['capture_start_read'], 'usage': r['descriptor'].get('Usage'),
               'consumers': sorted(consumers.get(rid, {}).get('programs', ()))}
        try:
            entry, request = ctx.resolve(rid, r, name, info, consumers.get(rid, {}))
        except Exception as e:  # noqa: BLE001 - unresolved requests are recorded rows
            if not type(e).__name__.endswith('Unresolved'):
                raise
            row.update(status='unresolved', reason=str(e))
            rows.append(row)
            continue
        if entry is not None and entry.desc is None:
            request, entry = entry.provenance.get('producer'), None
        if entry is None:
            row.update(status='runtime', runtime=request)
            rows.append(row)
            continue
        row['request'] = request
        row.update(compare_entry(oracle, rid, entry))
        if entry.initial is None:
            ok = row['desc_equal'] and row['srv_desc_equal'] in (True, None)
            row['status'] = 'runtime_descriptor_equal' if ok else 'runtime_descriptor_mismatch'
            row['runtime'] = entry.provenance.get('producer')
        else:
            ok = row['desc_equal'] and row['bytes_equal'] and row['srv_desc_equal'] in (True, None)
            row['status'] = 'byte_exact' if ok else 'mismatch'
        rows.append(row)
    (out / 'resources_table.json').write_text(json.dumps(rows, indent=1, default=sorted), encoding='utf-8')
    run.context = {'uncensor_body': uncensor, 'sideloader': sideloader, 'index': index}
    return rows, oracle, time.time() - t0


class Context:
    """Maps captured resources to scene-state requests (Unity debug name, descriptor and,
    for script render textures, the creation order of equal descriptors)."""

    def __init__(self, oracle, sr, scene, args):
        self.oracle, self.sr, self.scene = oracle, sr, scene
        self.tex_by_name = {}
        for t in scene['textures']:
            self.tex_by_name.setdefault(t['name'], []).append(t)
        self.skinned_by_mesh = {}
        for r in scene['renderers']:
            sm = r['properties'].get('sharedMesh')
            if isinstance(sm, dict) and r['properties'].get('enabled') and r['active_in_hierarchy']:
                self.skinned_by_mesh.setdefault(sm['ref_name'], []).append(r['id'])
        self.mesh_names = dict(a.split('=', 1) for a in (args.mesh_renderer or []))
        # ChaControl composites: equal descriptors paired by creation order (capture creation
        # order == decreasing Unity instance id of runtime objects).
        self.rt_pairs = {}
        caps = {}
        for rid, res in oracle.res.items():
            d = res.get('descriptor') or {}
            if res['kind'] == 'CreateTexture2D' and d.get('MiscFlags') == 1 and d.get('Format') == 27:
                caps.setdefault((d['Width'], d['Height'], d['MipLevels']), []).append((res['chunk'], rid))
        scn = {}
        for t in scene['textures']:
            desc, _ = R.runtime_texture_desc(t)
            if desc is not None:
                scn.setdefault((desc['Width'], desc['Height'], desc['MipLevels']), []).append(t['id'])
        for key, lst in caps.items():
            ids = sorted(scn.get(key, []), reverse=True)
            lst = sorted(lst)
            if len(ids) == len(lst):
                for (_, rid), tid in zip(lst, ids):
                    self.rt_pairs[rid] = tid

    def resolve(self, rid, r, name, info, consumers):
        sr, kind, desc = self.sr, self.oracle.kind(rid), r['descriptor']
        if kind == 'buffer':
            if desc.get('Usage') == 2:
                return None, classify_dynamic_buffer(name, info, consumers)
            if name == 'Buffer-1-16' and 0 in info.get('strides', ()):
                return R.default_vertex_buffer_entry(0, 1), {'unity_default_vertex_buffer': 0}
            if name == 'Fullscreen Triangle':
                e = [x for x in R.ppv2_fullscreen_triangle_entries()
                     if (x.role == 'index_buffer') == bool(desc['BindFlags'] & 2)]
                return e[0], {'ppv2_script_object': 'RuntimeUtilities.fullscreenTriangle'}
            if 'Skybox/MyCubemap' in consumers.get('programs', set()) and desc.get('BindFlags') == 1:
                return R.skybox_cubemap_vertex_buffer_entry(), {'unity_builtin_geometry': 'SkyboxGenerator::GetCubemapSkyboxVB'}
            base = _strip_clone(name or '')
            rids = self.skinned_by_mesh.get(base, [])
            if base in self.mesh_names:
                rids = [int(self.mesh_names[base])]
            if len(rids) != 1:
                raise R.ResourceUnresolved('buffer %s: %d enabled scene renderers use mesh %r' % (rid, len(rids), base))
            entries = sr.renderer_mesh(rids[0], mesh_name=base if base in self.mesh_names else None)
            want_ib = bool(desc['BindFlags'] & 2)
            cands = [e for e in entries if (e.role == 'index_buffer') == want_ib and e.desc['ByteWidth'] == desc['ByteWidth']]
            if len(cands) != 1:
                raise R.ResourceUnresolved('buffer %s: %d mesh streams of width %d' % (rid, len(cands), desc['ByteWidth']))
            return cands[0], {'scene_renderer_id': rids[0], 'role': cands[0].role}
        if name in R.BUILTIN_TEXTURES:
            return R.builtin_texture_entry(name), {'unity_builtin_texture': name}
        if rid in self.rt_pairs:
            return sr.texture(self.rt_pairs[rid]), {'scene_texture_id': self.rt_pairs[rid],
                                                     'pairing': 'equal descriptor, creation order'}
        cands = [t for t in self.tex_by_name.get(name or '', [])
                 if name and t['type'].endswith(('Texture2D', 'Cubemap'))
                 and t['properties'].get('width') == desc['Width'] and t['properties'].get('height') == desc.get('Height')
                 and t['properties'].get('mipmapCount') == desc['MipLevels']]
        if cands:
            entries = {}
            for t in cands:
                e = sr.texture(t['id'])
                entries.setdefault(e.content_hash(), (e, t['id']))
            if len(entries) != 1:
                raise R.ResourceUnresolved('texture %s: scene textures of different content share the name' % rid)
            (e, tid), = entries.values()
            return e, {'scene_texture_id': tid, 'scene_texture_ids_same_identity': [t['id'] for t in cands]}
        if name in ('Font Texture', 'Internal Curves Texture'):
            return None, classify_texture(name, desc, info, consumers)
        if name and kind in ('tex2d', 'texcube') and not desc.get('BindFlags', 0) & (32 | 64 | 128):
            try:
                e = sr.texture_identity(name, desc['Width'], desc['Height'], desc['Format'], desc['MipLevels'],
                                        'Cubemap' if kind == 'texcube' else 'Texture2D',
                                        ui='UI/Default' in consumers.get('programs', set()))
                return e, {'texture_identity': [name, desc['Width'], desc['Height'], desc['Format'], desc['MipLevels']]}
            except R.ResourceUnresolved as err:
                if 'different-content' not in str(err) and 'no installed asset' not in str(err):
                    raise
                return self._identify_list_texture(rid, name, desc, kind, err)
        return None, classify_texture(name, desc, info, consumers)

    def _identify_list_texture(self, rid, name, desc, kind, err):
        """Same-name assets of different content (Maker list thumbnails of several mods).  The
        list row shown on screen is Maker UI state the scene export does not contain; the
        installed candidate whose bytes equal the capture is reported as identification
        evidence only, never as a resolved request."""
        label, cap = self.oracle.capture_start(rid)
        sr = self.sr
        hits = []
        for spec, pid, _ in sr.index.find('Cubemap' if kind == 'texcube' else 'Texture2D', name):
            if not sr.effective_source(spec, name):
                continue
            tree = sr.files.object(spec, pid).read_typetree()
            if (tree['m_Width'], tree['m_Height'], tree['m_MipCount']) != (desc['Width'], desc['Height'], desc['MipLevels']):
                continue
            e = R.asset_texture_entry(sr.files, spec, pid, linear_project=sr.linear)
            if e.desc['Format'] == desc['Format'] and cap is not None and [s['data'] for s in e.initial] == cap:
                hits.append({'source': spec, 'path_id': pid, 'bundle_path': sr.bundle_path_of(spec)})
        raise R.ResourceUnresolved('UI list texture %s: %s; byte-equal installed candidates (identification only): %s'
                                   % (name, err, json.dumps(hits)))


# ------------------------------------------------------------------ receipt
CODE_FILES = ('UnityPlayer.dll', 'HoneySelect2_Data/Managed/Assembly-CSharp.dll',
              'HoneySelect2_Data/Managed/Unity.Postprocessing.Runtime.dll',
              'BepInEx/Plugins/HS2_BepisPlugins/HS2_Sideloader.dll',
              'BepInEx/Plugins/HS2_Plugins/DeathWeasel1337/HS2_UncensorSelector.dll',
              'BepInEx/Plugins/HS2_Plugins/ManlyMarco/HS2_OverlayMods.dll',
              'BepInEx/Plugins/HS2_Plugins/DeathWeasel1337/HS2_MaterialEditor.dll',
              'BepInEx/Plugins/XUnity.AutoTranslator/XUnity.AutoTranslator.Plugin.Core.dll')


def _short_prov(p):
    out = {'class': p.get('class')}
    if p.get('class') == 'asset':
        out.update(source=p['source'], path_id=p['path_id'], unity_type=p.get('unity_type'))
        if p.get('equivalent_sources'):
            out['equivalent_sources'] = len(p['equivalent_sources'])
        b = p.get('binding')
        if isinstance(b, dict):
            out['binding'] = b.get('kind')
        elif b:
            out['binding'] = b
        if p.get('prefab_binding'):
            out['binding'] = 'prefab_renderer' + ('/scene_file' if any('scene_file' in x for x in p['prefab_binding']) else '')
        if p.get('plugin_mesh'):
            out['plugin'] = p['plugin_mesh'].get('producer')
    else:
        out['producer'] = p.get('producer')
        for k in ('png', 'fill'):
            if p.get(k):
                out[k] = p[k]
        if p.get('replaced_asset'):
            out['replaced_asset'] = {'source': p['replaced_asset'].get('source'), 'path_id': p['replaced_asset'].get('path_id')}
    return out


def write_receipt(args, rows, oracle, secs):
    from collections import Counter
    g = R.GAME_ROOT
    ctx = getattr(run, 'context', {})
    inputs = {'xml': str(args.xml), 'xml_sha256': file_sha256(args.xml),
              'zip': str(oracle.zip_path), 'zip_sha256': file_sha256(oracle.zip_path),
              'scene_state': str(args.scene), 'scene_state_sha256': file_sha256(args.scene),
              'source_index': str(args.index), 'source_index_sha256': file_sha256(args.index),
              'source_index_files': len(ctx['index'].doc['sources']) if ctx.get('index') else None}
    if args.sideloader_scan:
        inputs.update(sideloader_scan=str(args.sideloader_scan), sideloader_scan_sha256=file_sha256(args.sideloader_scan),
                      sideloader_loaded_mods=len(ctx['sideloader'].loaded), sideloader_duplicates=len(ctx['sideloader'].duplicates))
    if args.card_state:
        inputs.update(card_state=str(args.card_state), card_state_sha256=file_sha256(args.card_state))
    code = {c: file_sha256(g / c) for c in CODE_FILES if (g / c).exists()}
    by = Counter((r['kind'], r['status']) for r in rows)
    exact = [r for r in rows if r['status'] == 'byte_exact']
    cls = Counter()
    for r in exact:
        p = r['entry']['provenance']
        key = p.get('class')
        if key == 'asset':
            key = 'asset_mesh' if r['kind'] == 'buffer' else 'asset_texture'
        cls[key] += 1
    receipt = {
        'schema': 'hs2pipe.resources_acceptance/1', 'generator': R.GENERATOR, 'date': '2026-10-20',
        'capture': 'paired frame 26059 (rdc_04d1e76f...), official zip.xml export', 'inputs': inputs,
        'installed_code_sha256': code, 'uncensor_body': ctx.get('uncensor_body') and
        {k: ctx['uncensor_body'][k] for k in ('guid', 'oo_base', 'asset', 'source', 'evidence')},
        'counts': {'frame_read_resources': len(rows), 'by_kind_status': {'%s/%s' % k: v for k, v in sorted(by.items())},
                   'byte_exact_by_provenance_class': dict(cls)},
        'byte_exact': [{'resource': r['resource'], 'name': r['debug_name'], 'kind': r['kind'],
                        'subresources': r['subresources'], 'content_sha256': r['entry']['content_sha256'],
                        'capture_content': r['capture_content'], 'provenance': _short_prov(r['entry']['provenance'])}
                       for r in exact],
        'runtime': [{'resource': r['resource'], 'name': r['debug_name'], 'kind': r['kind'], 'status': r['status'],
                     'capture_start_read': r['capture_start_read'],
                     'producer': r.get('runtime') if not isinstance(r.get('runtime'), dict) else
                     {k: r['runtime'][k] for k in ('kind', 'class', 'producer') if k in r['runtime']}}
                    for r in rows if r['status'] in ('runtime', 'runtime_descriptor_equal', 'runtime_descriptor_mismatch')],
        'unresolved': [{'resource': r['resource'], 'name': r['debug_name'], 'kind': r['kind'], 'reason': r['reason'][:600]}
                       for r in rows if r['status'] == 'unresolved'],
        'mismatch': [{'resource': r['resource'], 'name': r['debug_name'], 'desc_equal': r['desc_equal'],
                      'srv_desc_equal': r['srv_desc_equal'], 'subresources_equal': r['subresources_equal'],
                      'subresources': r['subresources']} for r in rows if r['status'] == 'mismatch'],
        'seconds': round(secs, 1),
        'bulky_table': str(Path(args.out) / 'resources_table.json'),
    }
    Path(args.receipt).write_text(json.dumps(receipt, indent=1, default=str), encoding='utf-8')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xml', required=True)
    ap.add_argument('--zip')
    ap.add_argument('--scene', required=True)
    ap.add_argument('--index', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--receipt')
    ap.add_argument('--chunk-cache')
    ap.add_argument('--preferred', action='append')
    ap.add_argument('--mesh-renderer', action='append', help='MESHNAME=SCENE_RENDERER_ID for MeshRenderers')
    ap.add_argument('--sideloader-scan', help='sideloader_assets.scan() JSON of the installed mods')
    ap.add_argument('--card-state', help='bridge state JSON of the loaded card (ChaFileParameter for plugin selections)')
    ap.add_argument('--shader-library', action='append', help='shader library extraction dirs (consumer names)')
    args = ap.parse_args(argv)
    rows, oracle, secs = run(args)
    from collections import Counter
    print(json.dumps(Counter((r['kind'], r['status']) for r in rows).most_common(), default=str))
    for r in rows:
        if r['status'] != 'byte_exact':
            rt = r.get('runtime')
            print(r['resource'], r['kind'], r['debug_name'], r['status'], (r.get('reason') or '')[:400],
                  rt.get('kind') if isinstance(rt, dict) else rt)
    print('seconds', round(secs, 1))
    if args.receipt:
        write_receipt(args, rows, oracle, secs)


if __name__ == '__main__':
    main()
