"""Whole-frame orchestrator of the HS2 offline renderer (Unity 2018.4.11f1, D3D11).

From scene state plus resources this module emits one complete frame program
(``src/render/frame_program.py``) in Unity's real camera/callback order.  The
order is transcribed from the installed UnityPlayer.dll (SHA-256 ``5034cdc2…``):

``RenderManager::RenderCameras`` (0x6669f0): cameras of each stack in depth
order; per camera ``Camera::CustomCull`` (0x660f70; ``OnPreCull`` is the
``SendMessage(kPreCull)`` at 0x6610a8, before culling), ``UpdateVelocity``,
then ``Camera::CustomRender`` (0x661400):

==========================  =========  ==========================================
step                        call site  meaning
==========================  =========  ==========================================
``on_pre_render``           0x66157f   ``SendMessage(kPreRender)`` + Camera.onPreRender
``update_depth_textures``   0x6616fa   depth texture then depth-normals (0x66e200)
``setup_render``            0x661746   ``Camera::SetupRender`` (0x66ca70)
``render_loop``             0x6617d7   PerformRenderFunction -> ``DoRenderLoop`` 0x6b9890
``on_post_render``          0x6617fc   ``SendMessage(kPostRender)`` unless RenderFlag & 0x40
``render_post_layers``      0x6618d9   only when CullResults+0x7a8 == 0
``image_effects``           0x6618ef   ``RenderImageFilters(loop, false)``
``gui_layer``               0x66196e   ``DoRenderGUILayer``
``after_everything``        0x6619c3   command buffers, CameraEvent 20
==========================  =========  ==========================================

``DoRenderLoop`` (forward path, 0x6b9890): ``DoForwardShaderRenderLoop``
(opaque; 0x6b9c98) -> ``RenderSkybox`` (0x6b9ca0) -> ``DoMotionVectorRenderLoop``
(0x6b9ca9) -> halos -> ``RenderImageFilters(loop, true)`` (0x6b9da5) ->
``DoForwardShaderRenderLoop`` (transparent; 0x6b9de3).
``ForwardShaderRenderLoop::PerformRendering`` (0x6ccb30): light shadow maps
(``RenderLightShadowMaps`` 0x6ce990, incl. the screen-space collect), SetupRender,
``ClearNoSkybox`` (opaque), CameraEvent 10/16 (Before opaque/alpha), the draws,
CameraEvent 11/17.  ``RenderImageFilters`` (0x6bca80): CameraEvent 12 (opaque)
/ 18, the component ``OnRenderImage`` chain (``ApplyAfterOpaqueFilters`` /
``ApplyAfterTransparentFilters``), SetupRender, CameraEvent 13 / 19.

Note the order the binary fixes: ``OnPostRender`` runs *before* the
``OnRenderImage`` chain of the same camera.

Generated work and ``external:capture`` work
--------------------------------------------
A step whose generator exists is emitted from scene state (shaders from the
installed shader library, D3D11 states from ``unity_d3d11_state``, constant
buffers through the frame's shared ``UnityConstantBufferPool`` with Unity's
leftover-byte rule).  Bytes the scene state lacks (meshes, textures, samplers,
camera target allocation, cbuffer residue left by work that is not generated)
come from the converted capture (``external_capture``) and are labelled
``external:capture`` per field.  Steps without a generator yet are spliced
verbatim from the capture, labelled per command, and listed as gaps.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from src.render.frame_program import DrawState
from . import shader_library as sl
from . import unity_d3d11_state as ds
from .external_capture import DRAW_OPS, link_command_blobs
from .ppv2_unity import UnityConstantBufferPool

F = np.float32
GENERATOR = 'hs2pipe.frame/1'

CAMERA_STEPS = ('on_pre_cull', 'cull', 'update_velocity', 'on_pre_render', 'update_depth_textures', 'setup_render',
                'render_loop', 'on_post_render', 'render_post_layers', 'image_effects', 'gui_layer', 'after_everything')
FORWARD_LOOP_STEPS = ('shadow_maps', 'clear', 'before_forward_opaque', 'forward_opaque', 'after_forward_opaque',
                      'skybox', 'motion_vectors', 'halos', 'image_effects_opaque', 'before_forward_alpha',
                      'forward_transparent', 'after_forward_alpha')
# capture frame-graph pass label -> generator step of the main-camera forward path
CAPTURE_LABEL_STEP = {
    'depth_texture_prepass': 'update_depth_textures', 'depth_normals': 'update_depth_textures',
    'shadow_map': 'shadow_maps', 'screenspace_shadows': 'shadow_maps', 'forward_scene': 'forward',
    'motion_vectors': 'motion_vectors', 'ambient_occlusion_msvo': 'image_effects_opaque',
    'grabpass_copy': 'forward_transparent', 'post_copy_std': 'image_effects', 'post_depth_of_field': 'image_effects',
    'color_grading_lut_bake': 'image_effects', 'post_uber': 'image_effects', 'post_smaa': 'image_effects',
    'post_final_pass': 'image_effects', 'temporal_aa_ctaa': 'image_effects', 'blit_copy': 'image_effects',
}


class FrameGap(RuntimeError):
    pass


def cbuffer_key(debug_name, names):
    """(cbuffer name, size) of a capture buffer named ``ConstantBuffer-<id>-<size>``."""
    parts = debug_name.split('-')
    if len(parts) != 3 or parts[0] != 'ConstantBuffer':
        return None
    nid, size = int(parts[1]), int(parts[2])
    return (names[nid], size) if nid in names else None


def capture_cbuffer_names(capture, library):
    """CbKey name id -> name, from every capture draw whose shaders are in the
    library (the id is the debug-name field; the name is the serialized layout's)."""
    layouts = {}
    for name in library.shader_names():
        for sh in library.copies(name):
            for sub in sh.subshaders:
                for p in sub.passes:
                    for stage in p.stages():
                        for sp in p.subprograms(stage):
                            if sp.dxbc_sha256 and sp.dxbc_sha256 not in layouts:
                                layouts[sp.dxbc_sha256] = sp
    names, conflicts = {}, []
    for i, c in enumerate(capture.commands):
        if c['op'] not in DRAW_OPS:
            continue
        for stage in ('vs', 'ps'):
            sp = layouts.get(capture.shader_sha(c['state'][stage]['shader']))
            if sp is None:
                continue
            for layout in sp.constant_buffers():
                bound = capture.bound_cbuffer(i, stage, layout.slot) if layout.slot is not None else None
                if bound is None:
                    continue
                parts = capture.obj(bound[0]).get('name', '').split('-')
                if len(parts) == 3 and parts[0] == 'ConstantBuffer':
                    nid = int(parts[1])
                    if names.setdefault(nid, layout.name) != layout.name:
                        conflicts.append((nid, names[nid], layout.name))
    return names, conflicts


def unwrap(v):
    return v[1] if isinstance(v, tuple) else v


def variable_ranges(v):
    """Byte ranges a declared variable occupies (Unity writes these only)."""
    out = []
    for e in range(v.array_size or 1):
        if v.matrix:
            base = v.offset + 64 * e if v.rows == 4 else v.offset + 16 * v.rows * e
            out.append((base, base + 16 * v.rows))
        else:
            base = v.offset + 16 * e
            out.append((base, base + 4 * v.columns))
    return out


@dataclass
class DrawRecord:
    command: int
    pass_label: str
    capture_command: int
    capture_event: int
    renderer: str
    shader: str
    pass_index: int
    keywords: tuple
    programs: dict
    cbuffers: dict = field(default_factory=dict)      # (stage, slot) -> {name, size, bytes, provenance ranges}
    external: dict = field(default_factory=dict)      # field -> reason
    notes: list = field(default_factory=list)


class FrameEmitter:
    """Generated-command emission into one ProgramBuilder with the frame's shared
    constant-buffer pool and the capture as the ``external:capture`` source."""

    def __init__(self, builder, capture, library, *, pool=None, cbuffer_names=None):
        self.b = builder
        self.cb_names = cbuffer_names or {}   # capture CbKey name id -> cbuffer name
        self.cap = capture
        self.lib = library
        self.pool = pool or UnityConstantBufferPool()
        self.pool_last_capture = {}      # (name, size) -> capture command index of the last generated sync
        self.key_buffer = {}             # (name, size) -> capture buffer id
        self.shaders = {}
        self.states = {}
        self.records = []
        self.gaps = []
        self.spliced = []
        self.skipped = []

    # ------------------------------------------------------------ objects
    def shader(self, stage, sha, source):
        if sha not in self.shaders:
            self.shaders[sha] = self.b.shader(stage, self.lib.dxbc(sha), source=source)
        return self.shaders[sha]

    def state(self, table, desc):
        import json
        key = (table, json.dumps(desc, sort_keys=True))
        if key not in self.states:
            self.states[key] = getattr(self.b, table)(desc)
        return self.states[key]

    # ------------------------------------------------------------ splice
    def splice(self, index, reason):
        """Copy one capture command verbatim (labelled external:capture)."""
        cmd = copy.deepcopy(self.cap.commands[index])
        prov = dict(cmd.get('provenance') or {})
        prov.update(external='external:capture', reason=reason, capture_command=index)
        cmd['provenance'] = prov
        link_command_blobs(self.b, self.cap, cmd)
        self.b.program['commands'].append(cmd)
        self.spliced.append((index, reason))
        if cmd['op'] in DRAW_OPS or cmd['op'] == 'dispatch':
            self._mirror_pool(index)
        return len(self.b.program['commands']) - 1

    def _mirror_pool(self, index):
        """A spliced draw went through Unity's pool too: its uploaded shadow copies
        become the pool state (keys from the capture buffer names
        ``ConstantBuffer-<name id>-<size>``, CreateCB 0x545db0)."""
        for (stage, slot), (bid, data) in (self.cap.bound_at(index) or {}).items():
            key = cbuffer_key(self.cap.obj(bid).get('name', ''), self.cb_names)
            if key is None or key[1] != len(data):
                continue
            self.pool.seed(key[0], key[1], data)
            self.pool_last_capture[key] = index

    # ------------------------------------------------------------ cbuffers
    def _sync_pool_from_capture(self, key, buffer_id, cap_index):
        """When work that is not generated wrote this Unity cbuffer since the last
        generated draw that used it, the shadow copy is taken from the capture
        (bytes before this draw's own upload)."""
        last = self.pool_last_capture.get(key)
        written = self.cap.writes_between(buffer_id, -1 if last is None else last, self.cap.upload_index(cap_index, buffer_id))
        if last is None or written:
            self.pool.seed(key[0], key[1], self.cap.contents_before_upload(cap_index, buffer_id))
            return 'external:capture (pool shadow left by %s)' % ('earlier frames / spliced work' if last is None
                                                                   else 'spliced work')
        return None

    def draw_cbuffers(self, stage_programs, values, cap_index):
        """Unity BeforeDrawCall rule over the shared pool: per stage (VS then PS)
        each layout's declared variables are written into the (name, size)
        shadow -- one shadow (and one D3D11 buffer) per key, shared by both
        stages; declared variables without a generated value take the captured
        bytes (external:capture); the whole shadow is uploaded once, so every
        binding of a key sees the bytes after all stages' writes.

        The key size is the serialized layout size (``CbKey`` = name id, size;
        ``ConstantBuffersD3D11Base::CreateCB`` 0x545db0 names the buffer
        ``ConstantBuffer-<name>-<size>``).  When the capture binds a buffer of
        another size, the material's installed shader copy is not the library
        copy: the capture's size is used and labelled."""
        out = {}
        for stage in ('vs', 'ps'):
            sp = stage_programs.get(stage)
            if sp is None:
                continue
            for layout in sp.constant_buffers():
                if layout.slot is None:
                    continue
                bound = self.cap.bound_cbuffer(cap_index, stage, layout.slot)
                if bound is None:
                    raise FrameGap('capture draw %d has no cbuffer at %s slot %d' % (cap_index, stage, layout.slot))
                buf_id, cap_bytes = bound
                size = layout.size
                size_note = None
                if len(cap_bytes) != size:
                    size_note = ('external:capture: captured %s buffer is %d bytes, library layout %d (material '
                                 'shader copy not in the library)' % (layout.name, len(cap_bytes), size))
                    self.gaps.append({'gap': 'shader_copy_layout_size', 'detail': size_note})
                    size = len(cap_bytes)
                variables = [v for v in layout.variables if max(z for _, z in variable_ranges(v)) <= size]
                key = (layout.name, size)
                prev = self.key_buffer.setdefault(key, buf_id)
                if prev != buf_id:
                    self.gaps.append({'gap': 'cbuffer_identity', 'detail': '%s/%d bound as capture buffers %d and %d'
                                      % (key[0], key[1], prev, buf_id)})
                sync = self._sync_pool_from_capture(key, buf_id, cap_index)
                shadow = bytearray(self.pool.shadow(*key))
                generated, external = [], []
                have = {}
                for v in variables:
                    if v.name in values:
                        val = unwrap(values[v.name])
                        n = np.asarray(val).size
                        if not v.matrix and not v.array_size and n == 4 and v.columns < 4:
                            # material Vector4 / Color into a narrower uniform: the
                            # vector param copies its declared dim (BeforeDrawCall
                            # per-layout write of dim*4 bytes)
                            val = list(np.asarray(val).reshape(-1)[:v.columns])
                        have[v.name] = val
                        generated.append(v.name)
                    else:
                        external.append(v.name)
                sub = sl.CBufferLayout(layout.name, size, variables, slot=layout.slot)
                data = bytearray(sl.pack_cbuffer(sub, have, base=bytes(shadow), allow_missing=True))
                for v in variables:
                    if v.name in external:
                        for a, z in variable_ranges(v):
                            data[a:z] = cap_bytes[a:z]
                self.pool.seed(key[0], key[1], bytes(data))
                self.pool_last_capture[key] = cap_index
                row = out.setdefault((stage, layout.slot), {'name': layout.name, 'size': size, 'key': key,
                                                            'generated': [], 'external': [], 'pool_sync': sync,
                                                            'capture_buffer': buf_id, 'size_note': size_note,
                                                            'layout': sub})
                row['generated'] += generated
                row['external'] += external
        for row in out.values():
            row['bytes'] = self.pool.snapshot(*row['key'])
        return out

    # ------------------------------------------------------------ draws
    def emit_draw(self, *, label, cap_index, stage_programs, pass_record, light_mode, values, material_floats,
                  device: ds.DeviceRaster, renderer_name, shader_name, keywords, notes=()):
        cap_cmd = self.cap.commands[cap_index]
        cst = cap_cmd['state']
        st = DrawState()
        rec = DrawRecord(command=-1, pass_label=label, capture_command=cap_index,
                         capture_event=self.cap.event(cap_index), renderer=renderer_name, shader=shader_name,
                         pass_index=pass_record.index if pass_record is not None else -1, keywords=tuple(keywords),
                         programs={k: (v.dxbc_sha256 if v is not None else None) for k, v in stage_programs.items()},
                         notes=list(notes))
        # IA: mesh / skinning output buffers and their input layout (external:capture)
        st.ia.input_layout = cst['ia']['input_layout']
        st.ia.topology = cst['ia']['topology']
        st.ia.vertex_buffers = copy.deepcopy(cst['ia']['vertex_buffers'])
        st.ia.index_buffer = copy.deepcopy(cst['ia']['index_buffer'])
        rec.external['ia'] = 'mesh/skin buffers and input layout not in the scene state'
        # shaders (generated) and cbuffers (pool)
        for stage in ('vs', 'ps'):
            sp = stage_programs.get(stage)
            b = getattr(st, stage)
            if sp is None:
                b.shader = cst[stage]['shader']
                rec.external[stage + '.shader'] = 'program unresolved from scene state'
                continue
            b.shader = self.shader(stage, sp.dxbc_sha256, sp.identity())
        cbs = self.draw_cbuffers(stage_programs, values, cap_index)
        rec.cbuffers = cbs
        for (stage, slot), cb in sorted(cbs.items()):
            getattr(st, stage).constant_buffers.append({'slot': slot, 'blob': self.b.blob(cb['bytes'])})
        for stage in ('vs', 'ps'):
            if stage_programs.get(stage) is None:
                getattr(st, stage).constant_buffers = copy.deepcopy(cst[stage]['constant_buffers'])
            getattr(st, stage).srvs = copy.deepcopy(cst[stage]['srvs'])
            getattr(st, stage).samplers = copy.deepcopy(cst[stage]['samplers'])
        rec.external['srvs_samplers'] = 'material/engine textures and samplers not in the scene state'
        # OM / RS
        st.om.render_targets = copy.deepcopy(cst['om']['render_targets'])
        st.om.depth_stencil = cst['om']['depth_stencil']
        rec.external['om_targets'] = 'camera target allocation not generated'
        if pass_record is not None:
            gs = ds.gfx_states(pass_record.record['state'], light_mode=light_mode, material_floats=material_floats)
            st.om.blend_state = self.state('blend_state', ds.d3d11_blend_desc(gs.blend))
            st.om.depth_stencil_state = self.state('depth_state', ds.d3d11_depth_stencil_desc(gs.depth, gs.stencil))
            st.om.stencil_ref = gs.stencil['ref']
            st.rs.state = self.state('raster_state', ds.d3d11_raster_desc(gs.raster, device))
        else:
            st.om.blend_state = cst['om']['blend_state']
            st.om.depth_stencil_state = cst['om']['depth_stencil_state']
            st.om.stencil_ref = cst['om']['stencil_ref']
            st.rs.state = cst['rs']['state']
            rec.external['states'] = 'pass state unresolved'
        st.om.blend_factor = list(cst['om']['blend_factor'])
        st.om.sample_mask = cst['om']['sample_mask']
        st.rs.viewports = copy.deepcopy(cst['rs']['viewports'])
        st.rs.scissors = copy.deepcopy(cst['rs']['scissors'])
        rec.external['viewport'] = 'camera pixel rect binding taken from the capture'
        prov = {'generator': GENERATOR, 'pass': label, 'renderer': renderer_name, 'shader': shader_name,
                'capture_event': rec.capture_event, 'external': sorted(rec.external)}
        op = cap_cmd['op']
        if op not in DRAW_OPS:
            raise FrameGap('capture command %d is not a draw' % cap_index)
        rec.external['draw_args'] = 'submesh ranges not in the scene state'
        rec.command = getattr(self.b, op)(st, prov, **cap_cmd['args'])
        self.records.append(rec)
        return rec


# ------------------------------------------------------------ PPv2 shader source
class LibraryShaderSource:
    """``hs2pipe.ppv2`` shader-source interface (``program`` / ``pass_state`` /
    ``kernel``) served from the installed shader library.

    Variant selection is the library's transcription of Unity's
    ``ShaderKeywordSet`` scoring (``Pass.select``, RVA 0x85a400/0x85aff0), not
    the manifest adapter's exact-set match; binding layouts come from the same
    serialized subprogram records (``ppv2.unity_layout``)."""

    def __init__(self, library):
        self.lib = library

    def _pass(self, shader, subshader, pass_index):
        sh = self.lib.shader(shader)
        sub = [s for s in sh.subshaders if s.index == subshader][0]
        return sh, [p for p in sub.passes if p.index == pass_index][0]

    def pass_state(self, shader, subshader, pass_index):
        return self._pass(shader, subshader, pass_index)[1].record['state']

    def program(self, shader, subshader, pass_index, stage, keywords):
        from .ppv2 import unity_layout
        sh, pz = self._pass(shader, subshader, pass_index)
        st = {'progVertex': 'vs', 'progFragment': 'ps'}.get(stage, stage)
        sel = pz.select(st, frozenset(keywords), tier=None)
        if sel is None:
            raise KeyError('%s pass %d %s: no loadable variant' % (shader, pass_index, stage))
        sp = sel.subprogram
        data = sp.dxbc()
        layout = unity_layout({'bindings': sp.record['bindings'], 'name_indices': pz.record['name_indices']})
        return {'dxbc': data, 'sha256': sp.dxbc_sha256, 'layout': layout,
                'source': {'shader': shader, 'subshader': subshader, 'pass': pass_index, 'keywords': sorted(sp.keywords),
                           'asset': sh.source_label, 'library': 'hs2pipe.shader_library', 'score': sel.score,
                           'exact': sel.exact}}

    def kernel(self, compute, kernel_name):
        k = self.lib.compute_kernel(compute, kernel_name)
        kr = k['kernel']
        data = self.lib.dxbc(kr['dxbc_sha256'])

        def res(rows):
            return [{'name': r['name'], 'bindPoint': r['bindPoint'], 'samplerBindPoint': r.get('samplerBindPoint'),
                     'texDimension': r.get('texDimension')} for r in rows]
        kernel = {'name': kr['name'], 'cbs': [{'name': c['name'], 'bindPoint': c['bindPoint']} for c in kr['cbs']],
                  'textures': res(kr['textures']), 'builtin_samplers': list(kr['builtinSamplers']),
                  'in_buffers': res(kr['inBuffers']), 'out_buffers': res(kr['outBuffers']),
                  'thread_group_size': list(kr['threadGroupSize']), 'dxbc_sha256': kr['dxbc_sha256']}
        cbs = {c['name']: {'name': c['name'], 'byte_size': c['byteSize'],
                           'params': [{'name': p['name'], 'offset': p['offset'], 'arraySize': p['arraySize'],
                                       'rowCount': p['rowCount'], 'colCount': p['colCount'], 'type': p['type']}
                                      for p in c['params']]} for c in k['constant_buffers']}
        return {'dxbc': data, 'sha256': kr['dxbc_sha256'], 'kernel': kernel, 'cbuffers': cbs,
                'source': {'compute': compute, 'kernel': kernel_name, 'library': 'hs2pipe.shader_library'}}
