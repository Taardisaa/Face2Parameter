"""Read-only extraction of installed Unity 2018.4 shaders for the offline shader library.

Covers ``Shader`` and ``ComputeShader`` objects in AssetBundles, serialized
``.assets`` files, Unity resource files, zipmod members and plugin-embedded
bundles (already extracted with their provenance JSON).  Nothing is converted:

* every serialized SubShader/Pass is kept with its tags, LOD, raw pass state
  (constant or property-bound values), UsePass/GrabPass names, Fallback and
  Dependencies;
* every stage's SerializedSubProgram is kept **in serialized order** (the
  order Unity 2018.4 ``Program::GetMatchingSubProgram`` scans) with tier,
  GPU program type, keyword index list, the keyword names resolved through the
  pass name table, the blob's own keyword strings, requirements and the raw
  binding tables;
* the program bytes are cut exactly at the DXBC container size and written
  content-addressed (``dxbc/<sha256>.dxbc``); leading Unity header and any
  trailing bytes are recorded, never silently dropped;
* each DXBC gets declaration reflection (cbuffer float4 sizes, SRV/sampler/UAV
  slots, ISGN/OSGN) from the mandatory SM4/5 token stream, plus
  ``D3DDisassemble`` text.

ComputeShader in 2018.4 has no keyword variants; each kernel of each
(targetRenderer, targetLevel) variant is exported with its binding tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import zipfile
from pathlib import Path

import UnityPy
from UnityPy.export.ShaderConverter import ShaderProgram
from UnityPy.helpers import CompressionHelper
from UnityPy.streams import EndianBinaryReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.native_head.audit_neck_shading import disassemble  # noqa: E402
from tools.offline_renderer.frame_graph import dxbc_chunks, parse_declarations, parse_signature  # noqa: E402

SCHEMA = 'hs2.shader_library.source.v1'
D3D11_PLATFORM = 4            # ShaderCompilerPlatform kShaderCompPlatformD3D11
D3D11_COMPUTE_RENDERER = 2    # GfxDeviceRenderer kGfxRendererD3D11
# ShaderGpuProgramType (2018.4): DX11 VS SM40/50, PS SM40/50, GS SM40/50, HS, DS.
DX11_GPU_PROGRAM_TYPES = {15: 'DX11VertexSM40', 16: 'DX11VertexSM50', 17: 'DX11PixelSM40',
                          18: 'DX11PixelSM50', 19: 'DX11GeometrySM40', 20: 'DX11GeometrySM50',
                          21: 'DX11HullSM50', 22: 'DX11DomainSM50'}
STAGES = (('progVertex', 'vs'), ('progFragment', 'ps'), ('progGeometry', 'gs'),
          ('progHull', 'hs'), ('progDomain', 'ds'))
PROPERTY_TYPES = {0: 'color', 1: 'vector', 2: 'float', 3: 'range', 4: 'texture'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def split_dxbc(data):
    """Return (prefix_len, dxbc, trailing_len) using the container's own size."""
    start = data.find(b'DXBC')
    if start < 0:
        return None
    total, = struct.unpack_from('<I', data, start + 24)
    if start + total > len(data):
        raise ValueError('DXBC container size exceeds program bytes')
    return start, data[start:start + total], len(data) - start - total


class BlobStore:
    """Content-addressed DXBC/asm/reflection store shared by all sources."""

    def __init__(self, out):
        self.out = out
        (out / 'dxbc').mkdir()
        (out / 'asm').mkdir()
        self.reflections = {}

    def add(self, dxbc):
        sha = digest(dxbc)
        if sha not in self.reflections:
            (self.out / 'dxbc' / (sha + '.dxbc')).write_bytes(dxbc)
            text, check = disassemble(dxbc)
            if check != sha:
                raise ValueError('Disassembler input digest changed')
            (self.out / 'asm' / (sha + '.asm')).write_text(text, encoding='utf-8')
            chunks = dxbc_chunks(dxbc)
            code = chunks.get('SHEX', chunks.get('SHDR'))
            refl = parse_declarations(code)
            refl = {'shader_model': refl['shader_model'], 'program_type': refl['program_type'],
                    'cb': {str(k): v for k, v in refl['cb'].items()},
                    'srv': {str(k): v for k, v in refl['srv'].items()},
                    'sampler': {str(k): v for k, v in refl['sampler'].items()},
                    'uav': {str(k): v for k, v in refl['uav'].items()},
                    'container_chunks': sorted(chunks), 'bytes': len(dxbc)}
            for fourcc, key in (('ISGN', 'input_signature'), ('OSGN', 'output_signature'),
                                ('OSG5', 'output_signature')):
                if fourcc in chunks:
                    refl[key] = parse_signature(chunks[fourcc])
            self.reflections[sha] = refl
        return sha


def name_table(pass_tree):
    table = {}
    for name, index in pass_tree['m_NameIndices']:
        if index in table and table[index] != name:
            raise ValueError(f'Pass name table maps index {index} to two names')
        table[index] = name
    return table


def tag_map(tags):
    return [[k, v] for k, v in tags['tags']]


def export_shader(obj, store, version):
    shader = obj.read()
    tree = obj.read_typetree()
    parsed = tree['m_ParsedForm']
    platforms = list(shader.platforms)
    row = {'name': parsed['m_Name'], 'path_id': obj.path_id, 'platforms': platforms,
           'fallback': parsed['m_FallbackName'], 'custom_editor': parsed['m_CustomEditorName'],
           'dependencies': [[d['first'], d['second']] if isinstance(d, dict) else d
                            for d in parsed['m_Dependencies']],
           'disable_no_subshaders_message': parsed['m_DisableNoSubshadersMessage'],
           'shader_is_baked': tree.get('m_ShaderIsBaked'),
           'properties': [{'name': p['m_Name'], 'description': p['m_Description'],
                           'attributes': p['m_Attributes'], 'type': PROPERTY_TYPES[p['m_Type']],
                           'flags': p['m_Flags'], 'default': [p[f'm_DefValue[{i}]'] for i in range(4)],
                           'default_texture': p['m_DefTexture']}
                          for p in parsed['m_PropInfo']['m_Props']],
           'subshaders': []}
    if D3D11_PLATFORM not in platforms:
        row['d3d11'] = 'absent'
        programs = None
    else:
        pi = platforms.index(D3D11_PLATFORM)
        entry = lambda field: field[pi][0] if isinstance(field[pi], list) else field[pi]  # noqa: E731
        offset, length, size = map(entry, (shader.offsets, shader.compressedLengths, shader.decompressedLengths))
        code = CompressionHelper.decompress_lz4(bytes(shader.compressedBlob)[offset:offset + length], size)
        programs = ShaderProgram(EndianBinaryReader(code, endian='<'), version).m_SubPrograms
        row['d3d11_blob'] = {'decompressed_bytes': size, 'subprogram_count': len(programs)}
    for si, sub in enumerate(parsed['m_SubShaders']):
        srow = {'index': si, 'lod': sub['m_LOD'], 'tags': tag_map(sub['m_Tags']), 'passes': []}
        for pi_, p in enumerate(sub['m_Passes']):
            names = name_table(p)
            prow = {'index': pi_, 'type': p['m_Type'], 'name': p['m_Name'], 'use_name': p['m_UseName'],
                    'texture_name': p['m_TextureName'], 'tags': tag_map(p['m_Tags']),
                    'state_tags': tag_map(p['m_State']['m_Tags']), 'state_name': p['m_State']['m_Name'],
                    'state_lod': p['m_State']['m_LOD'], 'program_mask': p['m_ProgramMask'],
                    'has_instancing_variant': p['m_HasInstancingVariant'],
                    'has_procedural_instancing_variant': p.get('m_HasProceduralInstancingVariant'),
                    'state': {k: v for k, v in p['m_State'].items() if k not in ('m_Tags', 'm_Name', 'm_LOD')},
                    'name_indices': [[n, i] for n, i in p['m_NameIndices']], 'stages': {}}
            for field, stage in STAGES:
                subs = p[field]['m_SubPrograms']
                if not subs:
                    continue
                rows = []
                for order, sp in enumerate(subs):
                    keywords = [names[i] for i in sp['m_KeywordIndices']]
                    rec = {'order': order, 'blob_index': sp['m_BlobIndex'],
                           'tier': sp['m_ShaderHardwareTier'], 'gpu_program_type': sp['m_GpuProgramType'],
                           'gpu_program_type_name': DX11_GPU_PROGRAM_TYPES.get(sp['m_GpuProgramType']),
                           'keyword_indices': list(sp['m_KeywordIndices']), 'keywords': keywords,
                           'shader_requirements': sp.get('m_ShaderRequirements'),
                           'channels': sp['m_Channels'],
                           'bindings': {k: sp[k] for k in ('m_VectorParams', 'm_MatrixParams', 'm_TextureParams',
                                                           'm_BufferParams', 'm_ConstantBuffers',
                                                           'm_ConstantBufferBindings', 'm_UAVParams', 'm_Samplers')
                                        if k in sp},
                           'dxbc_sha256': None}
                    if programs is not None and sp['m_GpuProgramType'] in DX11_GPU_PROGRAM_TYPES:
                        blob = programs[sp['m_BlobIndex']]
                        rec['blob_keywords'] = list(blob.m_Keywords)
                        rec['blob_local_keywords'] = list(blob.m_LocalKeywords or [])
                        rec['blob_program_type'] = blob.m_ProgramType
                        if sorted(rec['blob_keywords']) != sorted(keywords):
                            raise ValueError(f"{row['name']} s{si}p{pi_} {stage} #{order}: keyword index names "
                                             f"{keywords} disagree with blob keywords {rec['blob_keywords']}")
                        split = split_dxbc(bytes(blob.m_ProgramCode))
                        if split is None:
                            rec['dxbc_absent_reason'] = 'program bytes contain no DXBC container'
                        else:
                            prefix, dxbc, trailing = split
                            rec['dxbc_sha256'] = store.add(dxbc)
                            rec['program_prefix_hex'] = bytes(blob.m_ProgramCode)[:prefix].hex()
                            rec['program_trailing_bytes'] = trailing
                    rows.append(rec)
                prow['stages'][stage] = rows
            srow['passes'].append(prow)
        row['subshaders'].append(srow)
    return row


def export_compute(obj, store):
    tree = obj.read_typetree()
    row = {'name': tree['m_Name'], 'path_id': obj.path_id, 'variants': []}
    for v in tree['variants']:
        vrow = {'target_renderer': v['targetRenderer'], 'target_level': v['targetLevel'],
                'constant_buffers': v['constantBuffers'], 'resources_resolved': v.get('resourcesResolved'),
                'kernels': []}
        for k in v['kernels']:
            code = bytes(k['code'])
            krow = {key: k[key] for key in k if key != 'code'}
            krow['code_bytes'] = len(code)
            krow['dxbc_sha256'] = None
            if v['targetRenderer'] == D3D11_COMPUTE_RENDERER:
                split = split_dxbc(code)
                if split is None:
                    krow['dxbc_absent_reason'] = 'kernel code contains no DXBC container'
                else:
                    prefix, dxbc, trailing = split
                    krow['dxbc_sha256'] = store.add(dxbc)
                    krow['program_prefix_hex'] = code[:prefix].hex()
                    krow['program_trailing_bytes'] = trailing
            vrow['kernels'].append(krow)
        row['variants'].append(vrow)
    return row


def load_source(spec, work):
    """``PATH`` or ``ZIP::MEMBER``.  Zip members are copied to ``work`` first."""
    if '::' in spec:
        archive, member = spec.split('::', 1)
        with zipfile.ZipFile(archive) as z:
            data = z.read(member)
        target = work / (digest(data)[:16] + '_' + Path(member).name)
        target.write_bytes(data)
        return target, {'kind': 'zip_member', 'archive': str(Path(archive).resolve()),
                        'archive_sha256': digest(Path(archive).read_bytes()), 'member': member,
                        'member_sha256': digest(data), 'copied_to': str(target)}
    path = Path(spec)
    prov = {'kind': 'file', 'path': str(path.resolve()), 'sha256': digest(path.read_bytes())}
    sidecar = path.with_name(path.name + '.provenance.json')
    if sidecar.exists():
        prov['embedded_provenance'] = json.loads(sidecar.read_text(encoding='utf-8'))
    return path, prov


def extract(sources, out, shader_filter=None, compute_filter=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    work = out / 'zip_members'
    work.mkdir()
    store = BlobStore(out)
    library = {'schema_version': 'hs2.shader_library.v1', 'sources': []}
    for n, spec in enumerate(sources):
        path, prov = load_source(spec, work)
        env = UnityPy.load(str(path))
        objects = list(env.objects)
        version = objects[0].version if objects else None
        unity_version = getattr(objects[0].assets_file, 'unity_version', None) if objects else None
        shaders, computes = [], []
        for obj in objects:
            if obj.type.name == 'Shader':
                name = obj.read_typetree()['m_ParsedForm']['m_Name']
                if shader_filter is None or name in shader_filter:
                    shaders.append(export_shader(obj, store, version))
            elif obj.type.name == 'ComputeShader':
                name = obj.read_typetree()['m_Name']
                if compute_filter is None or name in compute_filter:
                    computes.append(export_compute(obj, store))
        manifest = {'schema_version': SCHEMA, 'source': prov, 'unity_version': unity_version,
                    'shader_filter': sorted(shader_filter) if shader_filter else None,
                    'compute_filter': sorted(compute_filter) if compute_filter else None,
                    'shaders': shaders, 'compute_shaders': computes}
        name = f'source_{n:02d}_{path.name.replace(" ", "_")}.json'
        (out / name).write_text(json.dumps(manifest, indent=1), encoding='utf-8')
        library['sources'].append({'manifest': name, 'source': prov, 'shaders': [s['name'] for s in shaders],
                                   'compute_shaders': [c['name'] for c in computes]})
    (out / 'reflection.json').write_text(json.dumps(store.reflections, indent=1), encoding='utf-8')
    library['dxbc_count'] = len(store.reflections)
    library['tool_sha256'] = digest(Path(__file__).read_bytes())
    (out / 'library.json').write_text(json.dumps(library, indent=1), encoding='utf-8')
    print(json.dumps({'library': str(out / 'library.json'), 'sources': len(sources),
                      'dxbc': len(store.reflections)}))
    return library


# ===================================================================== coverage
# Executed-program coverage and variant-selection verification against real
# captures.  Scene keywords are derived only from recorded source facts (native
# material/global keyword reads, bundle materials, decompiled plugin code); the
# derivation of every requested keyword set is written into the outputs.

SSS_VIEWPORT, CAPTURE_VIEWPORT = 543.0, 1086.0
TRANSPARENT_QUEUE_LIMIT = 2500   # Unity forward: queue > 2500 receives no screen-space shadow (documented)
MOTION_VECTOR_QUEUE_LIMIT = 2500  # Unity per-object motion vectors: opaque queues only (documented)
# The camera depth texture (ShadowCaster passes) also renders opaque queues only (documented).


def _queue(shader_record_tags, material_queue):
    if material_queue is not None and material_queue >= 0:
        return material_queue
    q = dict(shader_record_tags).get('QUEUE', 'Geometry')
    base = {'BACKGROUND': 1000, 'GEOMETRY': 2000, 'ALPHATEST': 2450, 'TRANSPARENT': 3000, 'OVERLAY': 4000}
    m = re.fullmatch(r'([A-Za-z]+)([+-]\d+)?', q)
    return base[m.group(1).upper()] + int(m.group(2) or 0)


def _bundle_renderer_materials(bundle, mesh_names):
    """{mesh: [[slot materials...] per renderer]} with keywords, queue and texture names."""
    env = UnityPy.load(str(bundle))
    out = {}
    for obj in env.objects:
        if obj.type.name != 'SkinnedMeshRenderer':
            continue
        r = obj.read()
        try:
            mesh = r.m_Mesh.read().m_Name
        except Exception:  # noqa: BLE001 - renderer without a readable mesh is not a candidate
            continue
        if mesh not in mesh_names:
            continue
        slots = []
        for mp in r.m_Materials:
            mt = mp.read()
            t = mt.object_reader.read_typetree()
            shader = mt.m_Shader.read().object_reader.read_typetree()['m_ParsedForm']['m_Name']
            texnames = set()
            for _, te in mt.m_SavedProperties.m_TexEnvs:
                # Only textures inside this bundle are named here (external PPtrs stay unnamed).
                if te.m_Texture.path_id and te.m_Texture.file_id == 0:
                    texnames.add(te.m_Texture.read().m_Name)
            slots.append({'material': t['m_Name'], 'shader': shader,
                          'keywords': sorted(k for k in t['m_ShaderKeywords'].split() if k),
                          'queue': t['m_CustomRenderQueue'], 'textures': sorted(texnames),
                          'source': f'{bundle} material {t["m_Name"]}'})
        out.setdefault(mesh, []).append(slots)
    return out


def _executed(actions):
    executed = {}
    for a in actions:
        for stage, s in (a.get('shaders') or {}).items():
            if not s:
                continue
            row = executed.setdefault((stage, s['dxbc_sha256']), {'stage': stage, 'dxbc_sha256': s['dxbc_sha256'],
                                                                   'debug_names': set(), 'uses': 0, 'chunks': []})
            row['debug_names'].update(s.get('debug_names') or [])
            row['uses'] += 1
            row['chunks'].append(a['chunk_index'])
    return executed


def _coverage(index, actions):
    programs, unmatched = [], []
    for (stage, sha), row in sorted(_executed(actions).items(), key=lambda kv: min(kv[1]['chunks'])):
        ids = index.get(sha, [])
        groups = {}
        for i in ids:
            key = (i.get('shader') or i.get('compute_shader'), i.get('subshader'), i.get('pass'), i.get('kernel'),
                   i['source'])
            groups.setdefault(key, []).append(i.get('keywords'))
        rec = {'stage': stage, 'dxbc_sha256': sha, 'debug_names': sorted(row['debug_names']), 'uses': row['uses'],
               'first_chunk': min(row['chunks']), 'matched': bool(ids), 'identities': []}
        for (shader, sub, pass_, kernel, source), kws in groups.items():
            ident = {'shader': shader, 'source': source, 'variant_count': len(kws)}
            if kernel is not None:
                ident['kernel'] = kernel
            else:
                ident.update({'subshader': sub, 'pass': pass_})
                if len(kws) <= 6:
                    ident['keyword_sets'] = kws
            rec['identities'].append(ident)
        programs.append(rec)
        if not ids:
            unmatched.append({'stage': stage, 'dxbc_sha256': sha, 'debug_names': sorted(row['debug_names']),
                              'reason': 'no installed export with this exact DXBC'})
    return programs, unmatched


def _renderer_keywords(light_mode, queue, receive_shadows, main_light_shadows):
    """Keywords the built-in forward renderer sets for one pass (scene facts given)."""
    if light_mode == 'FORWARDBASE':
        kw = {'DIRECTIONAL', 'LIGHTPROBE_SH'}
        if main_light_shadows and receive_shadows and queue <= TRANSPARENT_QUEUE_LIMIT:
            kw.add('SHADOWS_SCREEN')
        return [frozenset(kw)]
    if light_mode == 'FORWARDADD':   # which additional light is not identified per draw: both hypotheses
        return [frozenset({'DIRECTIONAL', 'SHADOWS_SCREEN'}), frozenset({'DIRECTIONAL'})]
    if light_mode == 'SHADOWCASTER':
        return [frozenset({'SHADOWS_DEPTH'})]
    return [frozenset()]


def _compare(pass_, requested, captured, tier):
    out, ok = {}, True
    for st in ('vs', 'ps'):
        sel = pass_.select(st, requested, tier=tier)
        out[st] = {'selected': sel.dxbc_sha256 if sel else None, 'captured': captured.get(st),
                   'selected_keywords': sorted(sel.subprogram.keywords) if sel else None,
                   'exact': sel.exact if sel else None, 'score': sel.score if sel else None}
        ok &= bool(sel) and sel.dxbc_sha256 == captured.get(st)
    return ok, out


def _captured(a):
    return {st.lower(): (a['shaders'].get(st) or {}).get('dxbc_sha256') for st in ('VS', 'PS')}


def verify_anchored(lib, actions, objects, globals_, tier, main_light_shadows=True):
    """Keyword selection inside the pass identified by the executed DXBC.

    ``objects``: chunk_index -> {'object', 'slots': [material dicts], 'receive_shadows'}.
    The pass is the installed (shader, subshader, pass) whose programs contain
    both executed hashes; the requested set is globals | material | renderer
    keywords for that pass's LightMode.  This isolates variant selection from
    pass/replacement routing.
    """
    sl = sys.modules['src.render.hs2pipe.shader_library']
    index = lib.hash_index()
    results = []
    for a in actions:
        if 'shaders' not in a or a['chunk_index'] not in objects:
            continue
        o = objects[a['chunk_index']]
        cap = _captured(a)
        vs_ids = {(i['shader'], i['source'], i['subshader'], i['pass']) for i in index.get(cap['vs'], []) if 'shader' in i}
        ps_ids = {(i['shader'], i['source'], i['subshader'], i['pass']) for i in index.get(cap['ps'], []) if 'shader' in i}
        passes = sorted(vs_ids & ps_ids)
        own = [p for p in passes if any(p[0] == s['shader'] for s in o['slots'])]
        candidates = own or passes
        rec = {'chunk': a['chunk_index'], 'event': a.get('event_id'), 'object': o['object'], 'checks': []}
        for shader_name, source, sub, pidx in candidates:
            shader = next(s for s in lib.copies(shader_name) if s.source_label == source)
            pass_ = shader.subshaders[sub].passes[pidx]
            slots = [s for s in o['slots'] if s['shader'] == shader_name] if own else o['slots']
            for slot in slots:
                queue = _queue(lib.copies(slot['shader'])[0].subshaders[0].record['tags']
                               if lib.copies(slot['shader']) else [], slot.get('queue'))
                for rk in _renderer_keywords(pass_.light_mode, queue, o.get('receive_shadows', True),
                                             main_light_shadows):
                    req = frozenset(globals_) | frozenset(slot['keywords'] or ()) | rk
                    ok, detail = _compare(pass_, req, cap, tier)
                    rec['checks'].append({'pass': {'shader': shader_name, 'source': source, 'subshader': sub,
                                                   'pass': pidx, 'light_mode': pass_.light_mode},
                                          'material': slot.get('material'), 'queue': queue,
                                          'material_keywords_known': slot['keywords'] is not None,
                                          'requested': sorted(req), 'equal': ok, **detail})
        if not rec['checks']:
            rec['status'] = 'no_pass_identity'
        elif all(c['equal'] for c in rec['checks']):
            rec['status'] = 'equal' if all(c['material_keywords_known'] for c in rec['checks'])                 else 'equal_material_keywords_unknown'
        elif any(c['equal'] for c in rec['checks']):
            rec['status'] = 'equal_for_some_hypothesis'
        else:
            rec['status'] = 'different'
        results.append(rec)
    return results


def verify_rules(lib, actions, passes, segments, objects, globals_, tier):
    """Full routing check on the retry capture: (camera, pass label, object) ->
    pass by engine rules -> keywords -> variant, compared with every draw.

    For each (segment, camera, label, object) the expected draw sequence is the
    renderer's eligible material slots in queue order (stable by slot index),
    each contributing every pass the rules select (replacement subshaders render
    all their passes in order).  Expected and captured sequences are compared
    element-wise; a length difference is reported, never trimmed.
    """
    sl = sys.modules['src.render.hs2pipe.shader_library']
    label_of, pass_of = {}, {}
    for p in passes:
        for e in p['events']:
            label_of[e], pass_of[e] = p['label'], p['pass_index']
    segment_of = {}
    for s in segments:
        for pi in s['passes']:
            segment_of[pi] = s['segment']

    def object_shader(slot):
        return lib.shader(slot['shader'], source=slot.get('library_source'))

    def route(camera, label, slot):
        """-> (list of passes, renderer light mode for keywords, derivation)."""
        shader = object_shader(slot)
        if label == 'motion_vectors':
            if slot['queue_resolved'] > MOTION_VECTOR_QUEUE_LIMIT:
                return [], 'per-object motion vectors skip queue > 2500'
            return [lib.shader('Hidden/Internal-MotionVectors').subshaders[0].passes[0]], \
                'Hidden/Internal-MotionVectors subshader 0 pass 0'
        if label == 'depth_normals':
            sub = sl.replacement_subshader(lib.shader('Hidden/Internal-DepthNormalsTexture'), 'RenderType', shader,
                                           tier=tier)
            return (sub.passes if sub else []), f"DepthNormals replacement RenderType={shader.tag('RenderType')}"
        if camera == 'sss' and label == 'sss_profile':
            sub = sl.replacement_subshader(lib.shader('Hidden/SSS_Profile'), 'RenderType', shader, tier=tier)
            return (sub.passes if sub else []), 'RenderWithShader(SSS_Profile, "RenderType")'
        if label == 'depth_texture_prepass' and slot['queue_resolved'] > TRANSPARENT_QUEUE_LIMIT:
            return [], 'camera depth texture renders opaque queues only (<= 2500)'
        if label in ('depth_texture_prepass', 'shadow_map'):
            if camera == 'sss':
                target = lib.shader('Hidden/LightingPass')
                sub = sl.replacement_subshader(target, 'RenderType', shader, tier=tier)
                if sub is None:
                    return [], 'LightingPass has no subshader for this RenderType'
                p = sub.first_pass_with_light_mode('SHADOWCASTER') or target.shadow_caster_pass(tier=tier)
                return ([p] if p else []), 'RenderWithShader(LightingPass) ShadowCaster (own, else Fallback chain)'
            p = shader.shadow_caster_pass(tier=tier)
            return ([p] if p else []), 'object ShadowCaster (own, else Fallback chain)'
        if camera == 'sss' and label == 'sss_lighting':
            sub = sl.replacement_subshader(lib.shader('Hidden/LightingPass'), 'RenderType', shader, tier=tier)
            p = sub.first_pass_with_light_mode('FORWARDBASE') if sub else None
            return ([p] if p else []), 'RenderWithShader(LightingPass) ForwardBase'
        if camera == 'capture' and label == 'forward_scene':
            _, sub = shader.resolve(tier=tier)
            p = sub.first_pass_with_light_mode('FORWARDBASE')
            return ([p] if p else []), 'object ForwardBase'
        return None, None

    groups = {}
    for a in actions:
        ev = a.get('event_id')
        if 'shaders' not in a or a['chunk_index'] not in objects or ev not in label_of:
            continue
        width = a['viewports'][0]['Width'] if a.get('viewports') else None
        camera = 'sss' if width == SSS_VIEWPORT else 'capture' if width == CAPTURE_VIEWPORT else None
        if camera is None:
            continue
        o = objects[a['chunk_index']]
        groups.setdefault((segment_of[pass_of[ev]], camera, label_of[ev], o['object']), []).append(a)
    results = []
    for (seg, camera, label, obj), draws in sorted(groups.items(), key=lambda kv: kv[1][0]['event_id']):
        o = objects[draws[0]['chunk_index']]
        expected, derivation, unknown = [], [], False
        for slot in sorted(o['slots'], key=lambda s: (s['queue_resolved'], s['slot'])):
            ps_, why = route(camera, label, slot)
            if ps_ is None:
                unknown = True
                break
            derivation.append(f"slot {slot['slot']} {slot.get('material')}: {why}")
            for p in ps_:
                expected.append((slot, p))
        if unknown:
            continue
        rec = {'segment': seg, 'camera': camera, 'label': label, 'object': obj, 'derivation': derivation,
               'captured_events': [d['event_id'] for d in draws], 'expected_draws': len(expected), 'draws': []}
        if len(expected) != len(draws):
            rec['status'] = 'draw_count_differs'
            results.append(rec)
            continue
        all_ok = True
        for d, (slot, p) in zip(sorted(draws, key=lambda d: d['event_id']), expected):
            rks = _renderer_keywords(p.light_mode if label != 'motion_vectors' else 'ALWAYS', slot['queue_resolved'],
                                     True, True)
            req = frozenset(globals_) | frozenset(slot['keywords'] or ()) | rks[0]
            ok, detail = _compare(p, req, _captured(d), tier)
            all_ok &= ok
            try:
                st = sl.to_d3d11(p.render_state())
                ours = [st['depth_stencil']['DepthEnable'], bool(st['depth_stencil']['DepthWriteMask']),
                        st['depth_stencil']['DepthFunc'], st['cull_d3d11_as_authored'],
                        st['blend']['RenderTarget'][0] if d.get('render_targets') else None]
                cap = [d['depth']['depth_enable'], d['depth']['depth_write'], d['depth']['depth_func'],
                       d['rasterizer']['descriptor']['CullMode'],
                       d['render_targets'][0]['blend'] if d.get('render_targets') else None]
                state = 'equal' if ours == cap else {'ours': ours, 'captured': cap}
            except sl.ShaderLibraryError as exc:
                state = 'needs material floats: ' + str(exc)
            rec['draws'].append({'event': d['event_id'], 'state_d3d11': state,
                                 'pass': {'shader': p.shader.name, 'subshader': p.subshader,
                                          'pass': p.index, 'light_mode': p.light_mode},
                                 'slot': slot['slot'], 'material': slot.get('material'),
                                 'material_keywords_known': slot['keywords'] is not None,
                                 'requested': sorted(req), 'equal': ok, **detail})
        unknown_kw = any(s['keywords'] is None for s in o['slots'])
        rec['status'] = ('equal' if not unknown_kw else 'equal_material_keywords_unknown') if all_ok else 'different'
        results.append(rec)
    return results


def _retry_objects(lib, actions, head, geometry, game_root):
    """chunk -> object with ordered material slots (runtime or installed sources)."""
    objects = {}
    for s in head['surfaces']:
        m = geometry['meshes'][s['surface_index']]
        if m['mesh_name'] != s['mesh_name']:
            raise ValueError('head_surfaces surface_index does not address the source geometry mesh')
        mats = m['material_state']['materials']
        if not mats or not s['draws']:
            continue
        slots = [{'slot': i, 'material': x['name'], 'shader': x['shader_name'], 'keywords': x['keywords'],
                  'queue': x['render_queue'], 'library_source': 'fo_head_38',
                  'source': 'source3d_v1 native material_state (runtime)'} for i, x in enumerate(mats)]
        for d in s['draws']:
            objects[d['chunk_index']] = {'object': s['mesh_name'], 'slots': slots}
    E = Path(game_root) / 'abdata' / 'chara'
    clothes = {'o_top_camisole2_a': E / '00' / 'fo_top_00.unity3d', 'o_bot_pantshot1_a': E / '00' / 'fo_bot_00.unity3d',
               'o_bra_a': E / '00' / 'fo_inner_t_00.unity3d', 'o_shorts_a': E / '00' / 'fo_inner_b_00.unity3d',
               'o_shoes_bones': E / '00' / 'fo_shoes_00.unity3d'}
    renderers = {}
    for mesh, bundle in clothes.items():
        renderers.update(_bundle_renderer_materials(bundle, {mesh}))
    body_bundle = _bundle_renderer_materials(E / 'oo_base.unity3d', {'o_body_cf'})['o_body_cf'][0]
    mm = UnityPy.load(str(E / 'mm_base.unity3d'))
    body0 = None
    for obj in mm.objects:
        if obj.type.name == 'Material' and obj.read_typetree()['m_Name'] == 'cf_m_skin_body_00':
            t = obj.read_typetree()
            body0 = {'material': 'cf_m_skin_body_00', 'shader': 'AIT/Skin True',
                     'keywords': sorted(k for k in t['m_ShaderKeywords'].split() if k),
                     'queue': t['m_CustomRenderQueue'], 'library_source': 'mm_base',
                     'source': 'chara/mm_base.unity3d cf_m_skin_body_00 (ChaControl.LoadBody drawMatName, slot 0)'}
    body = [dict(body0, slot=0)] + [dict(s, slot=i, library_source='oo_base') for i, s in enumerate(body_bundle) if i]
    bound = {}   # mesh -> union of SRV debug names over all its draws
    for a in actions:
        if 'shaders' not in a or not a.get('index_buffer'):
            continue
        for n in [n.replace('(Clone)', '') for n in a['index_buffer'].get('debug_names', [])]:
            for st in ('PS', 'VS'):
                for s in (a['shaders'].get(st) or {}).get('srvs', []):
                    bound.setdefault(n, set()).update(s.get('debug_names') or [])
    for a in actions:
        if 'shaders' not in a or a['chunk_index'] in objects or not a.get('index_buffer'):
            continue
        for n in [n.replace('(Clone)', '') for n in a['index_buffer'].get('debug_names', [])]:
            srvs = bound.get(n, set())
            if n == 'o_body_cf':
                objects[a['chunk_index']] = {'object': n, 'slots': body}
            elif n in renderers:
                # All installed renderers of this mesh agreeing on keywords resolve it without texture
                # evidence (ChaControl replaces _MainTex with composed RenderTextures, so bound names are
                # partial); otherwise keep the renderers with the largest bound-texture overlap.
                cands = renderers[n]
                if len({tuple(tuple(m['keywords']) for m in slots) for slots in cands}) > 1:
                    score = {i: sum(len(set(m['textures']) & srvs) for m in slots) for i, slots in enumerate(cands)}
                    best = max(score.values())
                    cands = [slots for i, slots in enumerate(cands) if score[i] == best]
                keysets = {tuple(tuple(m['keywords']) for m in slots) for slots in cands}
                if len(keysets) == 1:
                    slots = [dict(m, slot=i, library_source=Path(str(clothes[n])).name,
                                  source=m['source'] + ' (renderer chosen by bound texture names)')
                             for i, m in enumerate(cands[0])]
                else:
                    slots = [{'slot': 0, 'material': None, 'shader': renderers[n][0][0]['shader'], 'keywords': None,
                              'queue': renderers[n][0][0]['queue'], 'library_source': Path(str(clothes[n])).name,
                              'source': f'{len(cands)} candidate renderers by texture names'}]
                objects[a['chunk_index']] = {'object': n, 'slots': slots}
            elif n.startswith('Kamael_Valentine_Hair'):
                objects[a['chunk_index']] = {'object': n, 'slots': [
                    {'slot': 0, 'material': 'anyi', 'shader': 'Shader Forge/hair_07', 'keywords': None, 'queue': None,
                     'library_source': 'hair_shader_mat',
                     'source': 'ChaControl.ChangeSettingHairShader -> chara/hair_shader_mat.unity3d shader; '
                               'mod material keywords not read in this session'}]}
    for o in objects.values():
        for s in o['slots']:
            s['queue_resolved'] = _queue(lib.shader(s['shader'], source=s.get('library_source'))
                                         .subshaders[0].record['tags'], s.get('queue'))
    return objects


def _paired_objects(lib, actions, scene):
    """chunk -> object from the same-frame native scene export (renderers + materials)."""
    mats = {m['id']: m for m in scene['materials']}
    by_mesh = {}
    for r in scene['renderers']:
        mesh = (r['properties'].get('sharedMesh') or {}).get('ref_name')
        if mesh and r['active_in_hierarchy'] and r['properties'].get('enabled'):
            by_mesh.setdefault(mesh, []).append(r)
    objects, ambiguous = {}, set()
    for a in actions:
        if 'shaders' not in a or not a.get('index_buffer'):
            continue
        for n in [n.replace('(Clone)', '') for n in a['index_buffer'].get('debug_names', [])]:
            rs = by_mesh.get(n, [])
            if len(rs) != 1:
                if rs:
                    ambiguous.add(n)
                continue
            r = rs[0]
            slots = []
            for i, ref in enumerate(r['properties'].get('sharedMaterials') or []):
                m = mats.get(ref['ref_id']) if ref else None
                if m is None:
                    continue
                slots.append({'slot': i, 'material': m['name'], 'shader': m['shader'],
                              'keywords': m['shader_keywords'], 'queue': m['render_queue'],
                              'source': 'same-frame native material export'})
            objects[a['chunk_index']] = {'object': n, 'slots': slots,
                                         'receive_shadows': bool(r['properties'].get('receiveShadows', True))}
    # Draws without a renderer (blits, post-processing, skybox): the material is the unique
    # same-frame material whose shader is the executed program's shader.
    index = lib.hash_index()
    by_shader = {}
    for m in scene['materials']:
        by_shader.setdefault(m['shader'], []).append(m)
    for a in actions:
        if 'shaders' not in a or a['chunk_index'] in objects or not a['shaders'].get('PS'):
            continue
        names = {i['shader'] for i in index.get(a['shaders']['PS']['dxbc_sha256'], []) if 'shader' in i}
        mats = [m for n in names for m in by_shader.get(n, [])]
        sky = (scene.get('statics', {}).get('UnityEngine.RenderSettings', {}) or {}).get('skybox') or {}
        if len(mats) > 1 and any(m['id'] == sky.get('ref_id') for m in mats):
            mats = [m for m in mats if m['id'] == sky.get('ref_id')]   # skybox draw uses RenderSettings.skybox
        if len({(m['shader'], tuple(m['shader_keywords'])) for m in mats}) == 1:
            m = mats[0]
            objects[a['chunk_index']] = {'object': 'material:' + m['name'], 'receive_shadows': False, 'slots': [
                {'slot': 0, 'material': m['name'], 'shader': m['shader'], 'keywords': m['shader_keywords'],
                 'queue': m['render_queue'], 'source': 'same-frame native material export (unique for shader)'}]}
    return objects, sorted(ambiguous)


def _interpret(xml):
    from tools.offline_renderer import frame_graph as fg
    xml = Path(xml)
    graph = fg.FrameGraph(fg.load_chunks(xml), None, fg.Blobs(xml.with_suffix('')), fg.ShaderLibrary([])).build()
    return graph.actions


def coverage(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.render.hs2pipe import shader_library as sl

    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Fresh output directory required')
    out.mkdir(parents=True)
    lib = sl.ShaderLibrary(args.library)
    index = lib.hash_index()
    captures = {}

    # ---- retry capture (16 bridge renders + following Maker frame), interpreted with event IDs
    actions = json.loads(Path(args.retry_actions).read_text(encoding='utf-8'))
    passes = json.loads(Path(args.retry_passes).read_text(encoding='utf-8'))
    summary = json.loads(Path(args.retry_summary).read_text(encoding='utf-8'))
    head = json.loads(Path(args.retry_head_surfaces).read_text(encoding='utf-8'))
    geometry = json.loads(Path(args.retry_geometry).read_text(encoding='utf-8'))
    programs, unmatched = _coverage(index, actions)
    objects = _retry_objects(lib, actions, head, geometry, args.game_root)
    retry_globals = ('SSS_PROFILES',)   # DHH.SSS.OnPreRender (ProfilePerObject; profile pass present)
    rules = verify_rules(lib, actions, passes, summary['segments'], objects, retry_globals, args.tier)
    anchored = verify_anchored(lib, actions, objects, retry_globals, args.tier)
    captures['retry_frame17675'] = {'xml_sha256': summary['provenance']['xml_sha256'], 'programs': programs,
                                    'unmatched': unmatched, 'rules': rules, 'anchored': anchored,
                                    'globals': list(retry_globals),
                                    'globals_source': 'DHH.SSS.OnPreRender Shader.EnableKeyword("SSS_PROFILES"); '
                                                      'no native global keyword read in this session'}

    # ---- paired captures: ordinary Maker frames with same-frame native keyword exports
    for name, xml, scene_path in args.paired or []:
        acts = _interpret(xml)
        (out / f'{name}_actions.json').write_text(json.dumps(acts, default=str), encoding='utf-8')
        progs, unm = _coverage(index, acts)
        row = {'xml_sha256': digest(Path(xml).read_bytes()), 'programs': progs, 'unmatched': unm}
        if scene_path == '-':
            row.update({'anchored': [], 'globals': [], 'globals_source': 'no same-frame scene keyword export',
                        'scene': None, 'scene_frame_count': None, 'ambiguous_meshes': []})
        else:
            scene = json.loads(Path(scene_path).read_text(encoding='utf-8'))
            globals_ = sorted(k for k, v in scene['global_keywords'].items() if v)
            objs, ambiguous = _paired_objects(lib, acts, scene)
            row.update({'scene': str(Path(scene_path)), 'scene_frame_count': scene['frame_count'],
                        'anchored': verify_anchored(lib, acts, objs, globals_, args.tier), 'globals': globals_,
                        'ambiguous_meshes': ambiguous,
                        'globals_source': 'same-frame native Shader.IsKeywordEnabled over the manifest keywords'})
        captures[name] = row

    builtin = sl.parse_builtin_cbuffers(args.cginc, sl.D3D11_STANDALONE_DEFINES)
    layout_check = lib.check_builtin_layouts(builtin)
    (out / 'builtin_cbuffers.json').write_text(json.dumps(
        {k: {'size': v.size, 'variables': [x.__dict__ for x in v.variables]} for k, v in builtin.items()}, indent=1),
        encoding='utf-8')
    for name, c in captures.items():
        (out / f'{name}_verification.json').write_text(json.dumps(c, indent=1, default=str), encoding='utf-8')

    def totals(rows):
        t = {}
        for r in rows:
            t[r['status']] = t.get(r['status'], 0) + 1
        return t

    def rule_groups(rows):
        g = {}
        for r in rows:
            key = (r['camera'], r['label'], r['object'])
            x = g.setdefault(key, {'camera': r['camera'], 'label': r['label'], 'object': r['object'],
                                   'segments': 0, 'statuses': {}, 'derivation': r['derivation'],
                                   'expected_draws_per_segment': r['expected_draws'],
                                   'passes': sorted({json.dumps(d['pass']) for d in r['draws']}),
                                   'selected_keywords': sorted({json.dumps([d['vs']['selected_keywords'],
                                                                            d['ps']['selected_keywords']])
                                                                for d in r['draws']})})
            x['segments'] += 1
            x['statuses'][r['status']] = x['statuses'].get(r['status'], 0) + 1
            for d in r['draws']:
                k = 'equal' if d['state_d3d11'] == 'equal' else json.dumps(d['state_d3d11'])
                x.setdefault('state_d3d11', {})[k] = x.setdefault('state_d3d11', {}).get(k, 0) + 1
        for x in g.values():
            x['passes'] = [json.loads(p) for p in x['passes']]
            x['selected_keywords'] = [json.loads(p) for p in x['selected_keywords']]
        return list(g.values())

    def anchored_summary(rows):
        g = {}
        for r in rows:
            for c in r['checks'] or [{}]:
                ps = c.get('pass') or {}
                pkey = {k: ps.get(k) for k in ('shader', 'subshader', 'pass', 'light_mode')} if ps else None
                key = (r['object'], json.dumps(pkey), c.get('material'), json.dumps(c.get('requested')))
                x = g.setdefault(key, {'object': r['object'], 'pass': pkey, 'material': c.get('material'),
                                       'requested': c.get('requested'), 'draws': 0, 'equal': 0,
                                       'selected_keywords': [c.get('vs', {}).get('selected_keywords'),
                                                             c.get('ps', {}).get('selected_keywords')],
                                       'exact': [c.get('vs', {}).get('exact'), c.get('ps', {}).get('exact')]})
                x['draws'] += 1
                x['equal'] += bool(c.get('equal'))
        return list(g.values())

    receipt = {
        'schema_version': 'hs2.shader_library.coverage.v2',
        'library_roots': [str(Path(r)) for r in args.library],
        'library_sha256': {str(r): digest((Path(r) / 'library.json').read_bytes()) for r in args.library},
        'unity_player_pdb': sl.UNITYPLAYER, 'rules': sl.RULES,
        'tier_used': args.tier,
        'tier_note': 'every extracted SerializedSubProgram has m_ShaderHardwareTier 3 (tier-independent)',
        'captures': {}, 'bulky_outputs': str(out), 'tool_sha256': digest(Path(__file__).read_bytes())}
    for name, c in captures.items():
        row = {'xml_sha256': c['xml_sha256'],
               'coverage': {'executed_programs': len(c['programs']),
                            'matched': len(c['programs']) - len(c['unmatched']),
                            'by_stage': {st: sum(1 for p in c['programs'] if p['stage'] == st)
                                         for st in ('VS', 'PS', 'CS', 'GS', 'HS', 'DS')},
                            'unmatched': c['unmatched']},
               'executed_programs': [{'stage': p['stage'], 'sha256_16': p['dxbc_sha256'][:16],
                                      'debug_names': p['debug_names'], 'uses': p['uses'],
                                      'sources': sorted({f"{i['shader']} [{Path(i['source']).name}]"
                                                         for i in p['identities']}),
                                      'identity_count': sum(i['variant_count'] for i in p['identities'])}
                                     for p in c['programs']],
               'globals_used': c['globals'], 'globals_source': c['globals_source'],
               'variant_selection_anchored': {'draws': len(c['anchored']), 'status_totals': totals(c['anchored']),
                                              'groups': anchored_summary(c['anchored'])}}
        if 'rules' in c:
            row['routing_and_selection_rules'] = {'groups_checked': len(c['rules']), 'status_totals': totals(c['rules']),
                                                  'groups': rule_groups(c['rules'])}
        if c.get('scene'):
            row['scene'] = c['scene']
            row['scene_frame_count'] = c['scene_frame_count']
            row['ambiguous_meshes_skipped'] = c['ambiguous_meshes']
        for other, orow in receipt['captures'].items():
            if 'executed_programs' in orow and                     {(e['stage'], e['sha256_16']) for e in orow['executed_programs']} ==                     {(e['stage'], e['sha256_16']) for e in row['executed_programs']}:
                row['executed_programs'] = None
                row['executed_programs_identical_to'] = other
                break
        receipt['captures'][name] = row
    receipt['builtin_cbuffer_check'] = {k: {'programs': v['programs'], 'serialized_sizes': v['sizes'],
                                            'source_size': v['source_size'], 'mismatches': len(v['mismatches']),
                                            'first_mismatches': v['mismatches'][:3]} for k, v in layout_check.items()}
    text = json.dumps(receipt, indent=1)
    # Keep leaf lists on one line so the receipt stays compact and reviewable.
    text = re.sub(r'\[\s+([^\[\]{}]*?)\s+\]', lambda m: '[' + re.sub(r'\s*\n\s*', ' ', m.group(1)) + ']', text)
    Path(args.receipt).write_text(text, encoding='utf-8')
    print(json.dumps({'receipt': args.receipt,
                      **{n: {'executed': r['coverage']['executed_programs'], 'matched': r['coverage']['matched'],
                             'anchored': r['variant_selection_anchored']['status_totals'],
                             'rules': (r.get('routing_and_selection_rules') or {}).get('status_totals')}
                         for n, r in receipt['captures'].items()}}))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'coverage':
        ap = argparse.ArgumentParser(description='Executed-program coverage + variant verification receipt')
        ap.add_argument('--library', action='append', required=True)
        ap.add_argument('--retry-actions', required=True)
        ap.add_argument('--retry-passes', required=True)
        ap.add_argument('--retry-head-surfaces', required=True)
        ap.add_argument('--retry-summary', required=True)
        ap.add_argument('--retry-geometry', required=True, help='source3d baseline_front_geometry.json')
        ap.add_argument('--paired', nargs=3, action='append', metavar=('NAME', 'XML', 'SCENE_JSON'),
                        help="paired capture: official zip.xml and its same-frame scene export ('-' if none)")
        ap.add_argument('--cginc', required=True, help='2018.4.11f1 editor CGIncludes/UnityShaderVariables.cginc')
        ap.add_argument('--game-root', default='E:/HoneySelect2_ArcticFox')
        ap.add_argument('--tier', type=int, default=None,
                        help='GraphicsCaps active tier (0..2); omit when all subprograms are tier-independent')
        ap.add_argument('--out', required=True)
        ap.add_argument('--receipt', required=True)
        coverage(ap.parse_args(argv[1:]))
        return
    if argv and argv[0] == 'extract':
        argv = argv[1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', action='append', required=True, help='Asset file, bundle, or ZIP::MEMBER (repeatable)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--shader', action='append', help='Shader name filter (repeatable)')
    ap.add_argument('--compute', action='append', help='ComputeShader name filter (repeatable)')
    args = ap.parse_args(argv)
    extract(args.source, args.out, set(args.shader) if args.shader else None,
            set(args.compute) if args.compute else None)


if __name__ == '__main__':
    main()
