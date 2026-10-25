"""Candidate rendering on the paired bridge frame (v5.0.1 mode b, offline).

A candidate is the session card plus a patch: ABMX bone modifiers in the bridge
``{"bones": [...]}`` format and/or the 59 native face sliders (``native59``).
The chain, all from installed code or source-derived ports:

1. ``CapturedHeadForward(baseline_mode='card_load')`` per probe frame: head prefab
   locals -> ShapeHeadInfo/ShapeBody (native59 / body) -> ABMX CollectBaselineCo ->
   ABMX Apply -> bone local TRS (bit-exact to the probe for the baseline card);
2. the hosted EyeLookController (installed IL on the game's Mono JIT,
   ``tools/offline_renderer/hosted_eye_look.py``), which reads the face bones
   (cf_J_FaceUp_tz, cf_J_EyePos_rz_L/R) and writes cf_J_look_L/R;
3. ``unity_skin_emu.skin_from_locals`` with the probe blendshape weights: the skinned
   vertex streams the bridge frame reads -- slot 0 = frame N-2, slot 1 (motion-vector
   previous) = N-1 (capture-start contents), and the frame's own post-bridge skinning
   N with previous N-1 (in-frame uploads);
4. the bridge frame program (``frame_orchestrate_paired`` run, the template) with those
   streams substituted, executed by ``frame_executor --serve``; the 1086x1448 capture
   target is read back and written as a top-down PNG.

Assumption (stated, as required): the candidate is held constant over the probe
frames N-5..N (a static card that was loaded with these values), and the hidden
EyeLook state at the window's first frame is the one reconstructed from the
baseline export.  On this window the hosted EyeLook converges from deliberately
wrong seeds by N-3, before the first consumed frame N-2.

Dependencies that read head geometry or bones are checked, not frozen (see
``dependency_check``): a candidate whose changed bones (or their descendants)
include the root bone of a head surface, any bone/root/transform of another
renderer, raises ``CandidateUnsupported``.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

F = np.float32

# Source citations of the invariance argument (UnityPlayer.dll 2018.4.11f1, sha 5034cdc2...)
SOURCE_CITATIONS = {
    'skinned_bounds': 'SkinnedMeshRendererManager::InitializePreparedInfo 0xbdce90 copies the renderer local AABB '
                      '(SkinnedMeshRenderer+0x270/+0x27c, written only by SetLocalAABB 0x788010) into the prepared '
                      'info (+0x58..) and sets the bone-matrix flag +4 = m_UpdateWhenOffscreen (+0x28c, '
                      'SetUpdateWhenOffscreen 0x788313) && HasBoneWeights; CalculateTransformInfo_Prepared 0xbdaca0 '
                      'transforms that AABB by the root bone global matrix (CalculateGlobalMatrix 0xbdad6d); the '
                      'skin-matrix path (ReadySkinMatricesTask 0xbdb528) runs only when the prepared gate is set',
    'object_to_world': 'skinned unity_ObjectToWorld = TRS(rootBone.position, rootBone.rotation, 1) '
                       '(paired_skin.unity_object_matrices; bit-exact 16/16 in forward_chain_acceptance); '
                       'WorldToObject = InvertMatrix4x4_General3D of it',
    'previous_matrix': '_PreviousM = RenderNode+0x40 previous-frame matrix of the same rootBone TRS',
    'shadow_order': 'shadow caster sort key material field SharedMaterialData+0xd0 = Material::UpdateHashes '
                    '0x84e3b0 -> Pass::ComputePassValuesHash 0x855f70: XXH32 over the ShaderLab::Pass* address, '
                    'the 32-byte keyword set and the per-stage PrepareOverridingValues bytes; no mesh, bone or '
                    'bounds input',
}


class CandidateUnsupported(RuntimeError):
    """A candidate changes an input the bridge program takes from the capture."""


@dataclass
class SessionConfig:
    session: Path = Path('E:/HS2_offline_renderer_data/paired_session_v5_1_20261021')
    nonce: str = '5af92a0873c440e38ff76ff7b4cd1134'
    artifacts: Path = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_reference_fit_20261012/'
                           'paired_capture_session_v5_1_20261021')
    loaded_state: str = 'state_before_renderstyle.json'
    effect_context: Path = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_native_inverse_20261011/'
                                'solver_context.json')
    template_run: Path = Path('E:/HS2_offline_renderer_data/orchestrator_20261024/modeb_5af9/run_v7')
    capture_program: Path = Path('E:/HS2_offline_renderer_data/orchestrator_20261024/modeb_5af9/capture_program')
    executor: Path = Path('C:/Users/13666/Workspace/Face2Parameter/outputs/offline_renderer_executor_20261020/'
                          'build_v5/frame_executor.exe')
    fxc: Path = Path('C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/fxc.exe')
    work: Path = Path('E:/HS2_offline_renderer_data/candidate_20261025/work')
    target_resource: int = 31690
    target_size: tuple = (1086, 1448)
    apply_count: int = 2

    @property
    def window(self):
        return self.session / ('window_' + self.nonce)

    @property
    def xml(self):
        return self.session / ('parse_' + self.nonce) / 'capture.zip.xml'


@dataclass
class CandidateResult:
    png: Path
    pixels: object
    target_sha256: str
    timings: dict
    changed_bones: list
    patched: dict
    dependency: dict
    notes: list = field(default_factory=list)


def _frame_locals(bone_local, tf, by_id):
    """Forward bone locals keyed by probe transform index (+ ancestors above the probe root)."""
    parent, loc = {}, {}
    for i, t in enumerate(tf):
        L = bone_local[t['id']]
        loc[i] = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F),
                  np.asarray(L['local_scale'], F))
        parent[i] = t['parent'] if t['parent'] >= 0 else ('anc', by_id[t['id']]['parent_id'])
    for key in list(parent.values()):
        cur = key
        while isinstance(cur, tuple) and cur not in loc:
            tid = cur[1]
            L = bone_local[tid]
            loc[cur] = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F),
                        np.asarray(L['local_scale'], F))
            pid = by_id[tid]['parent_id']
            parent[cur] = ('anc', pid) if pid is not None else None
            cur = parent[cur]
    return parent, loc


class BridgeCandidateRenderer:
    LOOK = ('cf_J_look_L', 'cf_J_look_R')

    def __init__(self, cfg: SessionConfig | None = None, *, start_executor=True):
        import sys
        here = Path(__file__).resolve().parents[3]
        for p_ in (str(here), str(here / 'tools' / 'offline_renderer')):
            if p_ not in sys.path:
                sys.path.insert(0, p_)
        from src.render import paired_skin as ps
        from src.render import gpu_skin_inputs as g
        from src.render.unity_skin_emu import UnityPlayerEmu
        from src.hs2_offline_state import CapturedHeadForward
        from tools.offline_renderer.paired_forward_context import build
        import eye_look_acceptance as ela
        t0 = time.time()
        self.cfg = cfg = cfg or SessionConfig()
        self.ps, self.g = ps, g
        self.win = ps.ProbeWindow(cfg.window)
        self.tf = self.win.static['transforms']
        self.emu = UnityPlayerEmu()
        frames = sorted(self.win.by_frame)
        self.N = json.loads((cfg.window / 'capture_receipt.json').read_text(encoding='utf-8'))['expected_capture_frame']
        self.eye_frames = [f for f in frames if f <= self.N]          # seed frame + updates up to N
        self.skin_frames = (self.N - 2, self.N - 1, self.N)
        # per-frame forward models (transforms of that probe frame)
        self.models, self.geo = {}, None
        for fc in self.eye_frames[1:]:
            geo, ctx, info = build(cfg.session / 'prepare_geometry_actor.json', cfg.artifacts / cfg.loaded_state,
                                   cfg.effect_context, self.win, fc)
            self.models[fc] = CapturedHeadForward(geo, ctx)
            self.geo = geo
        self.by_id = {t['id']: t for t in self.geo['transforms']}
        self.card_native = list(self.models[self.N].identity['native59'])
        # hosted EyeLook (seeded at the window's first frame from the baseline export)
        self.eye = ela.Harness(self.win.static)
        self.ela = ela
        # Camera.main position per eye-look frame (scene export of that frame; EyeLook target)
        self.cameras, cam = {}, None
        for k, fc in enumerate(sorted(self.win.by_frame)):
            sc = cfg.window / 'frames' / ('scene_%03d.json' % k)
            cam = ela.camera_from_scene(sc)[0] if sc.exists() else cam
            self.cameras[fc] = cam
        # skin upload points of the bridge capture (gpu_skin_inputs path of frame_skin)
        index = g.CaptureIndex(cfg.xml)
        writes, draws = g.simulate(index)
        surfaces = self.win.surfaces()
        bindings = g.bind_surfaces(index, draws, surfaces)
        pairs = {(b['skinned_vb'], b['index_buffer']) for b in bindings if b['status'] == 'bound'}
        work = Path(cfg.work)
        work.mkdir(parents=True, exist_ok=True)
        usage = {vs: g.object_matrix_uses(g.disassemble(index.shader_bytes(vs), cfg.fxc, work, f'vs_{vs}'))
                 for vs in sorted({d['shaders']['VS'] for d in draws if (d['vbs'][0], d['ib']) in pairs})}
        points, _ = g.upload_points(index, writes, draws, bindings, usage)
        self.renderers = {r['id']: r for r in self.win.static['renderers']}
        self.points = []
        for p in points:
            if p['role'] not in ('skinned_vertices_current', 'skinned_vertices_previous'):
                continue
            w = writes[p['write_id']]
            start = w['kind'] == 'capture_start_contents'
            if p['role'] == 'skinned_vertices_current':
                fc = self.N - 2 if start else self.N
            else:
                fc = self.N - 1
            rid = surfaces[p['surfaces'][0]]['renderer_id']
            self.points.append({'chunk': w['chunk_index'], 'role': p['role'], 'renderer': rid, 'frame': fc,
                                'capture_start': start, 'captured': bytes(index.blob(w['buffer']))})
        self.head_renderers = sorted({p['renderer'] for p in self.points})
        # template program locations of every point
        self._bind_template()
        self.scene = json.loads((cfg.window / 'frames' / ('scene_%03d.json' % self._window_index())).read_text(
            encoding='utf-8'))
        self._build_dependency_tables()
        self.baseline = None
        self.baseline = self._streams(patches=(), native=None)
        mism = [(p['chunk'], p['role']) for p in self.points if self.baseline['bytes'][p['chunk']] != p['captured']]
        if mism:
            raise RuntimeError('baseline card does not reproduce the captured skin uploads: %r' % mism)
        self.timings_init = {'init_s': time.time() - t0}
        self.proc = None
        if start_executor:
            self._start_executor()

    # ------------------------------------------------------------ template
    def _window_index(self):
        for i, f in enumerate(sorted(self.win.by_frame)):
            if f == self.N:
                return i
        raise RuntimeError('capture frame not in the probe window')

    def _bind_template(self):
        cfg = self.cfg
        from src.render.hs2pipe.external_capture import CaptureProgram
        cap = CaptureProgram(cfg.capture_program)
        res_chunk = {(e.get('provenance') or {}).get('chunk'): e['id'] for e in cap.program['resources']}
        cmd_chunk = {(c.get('provenance') or {}).get('chunk'): i for i, c in enumerate(cap.commands)}
        prog = json.loads((cfg.template_run / 'generated' / 'program.json').read_text(encoding='utf-8'))
        gen_cmd = {}
        for i, c in enumerate(prog['commands']):
            pv = c.get('provenance') or {}
            if c['op'] == 'map_write' and pv.get('generator') == 'hs2pipe.frame_skin':
                gen_cmd[pv['capture_command']] = i
        res_by_id = {e['id']: e for e in prog['resources']}
        for p in self.points:
            if p['chunk'] in cmd_chunk:
                ci = cmd_chunk[p['chunk']]
                if ci not in gen_cmd:
                    raise RuntimeError('skin upload command %d not generated in the template' % ci)
                p['program'] = ('map_write', gen_cmd[ci])
            elif p['chunk'] in res_chunk:
                rid = res_chunk[p['chunk']]
                e = res_by_id[rid]
                if (e.get('provenance') or {}).get('generator') != 'hs2pipe.frame_skin':
                    raise RuntimeError('template resource %d is not the generated skin stream' % rid)
                p['program'] = ('resource', rid)
            else:
                raise RuntimeError('skin chunk %d not in the capture program' % p['chunk'])
        rep = json.loads((cfg.template_run / 'report.json').read_text(encoding='utf-8'))
        fin = [x for x in rep['execution'] if x['resource'] == cfg.target_resource]
        if not fin:
            raise RuntimeError('template has no checkpoint of the capture target')
        self.target_command = fin[-1]['command']
        self.template_target_sha = fin[-1]['generated_sha256']

    # ------------------------------------------------------------ dependencies
    def _build_dependency_tables(self):
        tf = self.tf
        self.children = {}
        for i, t in enumerate(tf):
            if t['parent'] >= 0:
                self.children.setdefault(t['parent'], []).append(i)
        self.index_of = {t['id']: i for i, t in enumerate(tf)}
        rows = []
        head = set(self.head_renderers)
        for r in self.scene['renderers']:
            p = r['properties']
            if not (r['active_in_hierarchy'] and p.get('enabled')):
                continue
            rid = int(r['id'])
            ids = set()
            for b in p.get('bones') or []:
                if b:
                    ids.add(int(b['ref_id']))
            if p.get('rootBone'):
                ids.add(int(p['rootBone']['ref_id']))
            if p.get('transform'):
                ids.add(int(p['transform']['ref_id']))
            rows.append({'id': rid, 'name': r['name'], 'head_surface': rid in head,
                         'root': int(p['rootBone']['ref_id']) if p.get('rootBone') else None,
                         'transform': int(p['transform']['ref_id']) if p.get('transform') else None,
                         'deps': ids, 'update_when_offscreen': str(p.get('updateWhenOffscreen')) == 'True'})
        self.dep_rows = rows
        for rr in rows:
            if rr['head_surface'] and rr['update_when_offscreen']:
                raise CandidateUnsupported('head surface %s updates bounds from bones (updateWhenOffscreen): '
                                           'bounds would depend on the candidate' % rr['name'])

    def dependency_check(self, changed_ids):
        """Which program inputs a set of changed transforms reaches (computed)."""
        moved = set()
        stack = [self.index_of[t] for t in changed_ids if t in self.index_of]
        while stack:
            i = stack.pop()
            if i in moved:
                continue
            moved.add(i)
            stack.extend(self.children.get(i, []))
        moved_ids = {self.tf[i]['id'] for i in moved}
        problems, head = [], []
        for rr in self.dep_rows:
            if rr['head_surface']:
                if rr['root'] in moved_ids or rr['transform'] in moved_ids:
                    problems.append('%s: root bone/transform moved -> ObjectToWorld, bounds, culling, shadow fit and '
                                    '_PreviousM would change' % rr['name'])
                else:
                    head.append(rr['name'])
            elif rr['deps'] & moved_ids:
                problems.append('%s: bones/transform moved but its skin/matrices are taken from the capture'
                                % rr['name'])
        return {'moved_transforms': len(moved_ids), 'head_surfaces_regenerated': sorted(set(head)),
                'problems': problems,
                'invariant_inputs': ['renderer bounds (local AABB x unchanged root) -> culling, draw lists',
                                     'shadow caster cull set and cascade fit (caster/receiver bounds unchanged)',
                                     'shadow caster order (material hash, no geometry input)',
                                     'ObjectToWorld / WorldToObject / _PreviousM (root TRS unchanged)'],
                'citations': SOURCE_CITATIONS}

    # ------------------------------------------------------------ streams
    def _streams(self, patches, native):
        """Skinned stream bytes per upload chunk for a candidate (and the frame locals)."""
        from src.render.unity_skin_emu import skin_from_locals
        tf = self.tf
        t0 = time.time()
        locals_by_frame = {}
        for fc in self.eye_frames[1:]:
            res = self.models[fc].evaluate(patches=list(patches), native=native, apply_count=self.cfg.apply_count,
                                           baseline_mode='card_load')
            locals_by_frame[fc] = res['bone_local']
        t1 = time.time()
        # hosted EyeLook: seed at the first window frame, run every later frame with the candidate locals
        h, ela = self.eye, self.ela
        first = self.win.by_frame[self.eye_frames[0]]
        h.apply_frame(first['T'])
        h.el.set_state(ela.seed_state('reconstructed', first['T'], h))
        cam = None
        for fc in self.eye_frames[1:]:
            fr = self.win.by_frame[fc]
            T = fr['T'].copy()
            L = locals_by_frame[fc]
            for i, t in enumerate(tf):
                r = L[t['id']]
                T[i, 0:3] = np.asarray(r['local_position'], F)
                T[i, 3:7] = np.asarray(r['local_rotation_xyzw'], F)
                T[i, 7:10] = np.asarray(r['local_scale'], F)
            cam = self.cameras[fc]
            got = h.run({'T': T, 'delta_time': fr['delta_time'], 'time': fr['time'], 'frame_count': fr['frame_count']},
                        cam)
            for n in self.LOOK:
                tid = tf[h.ids.index(h.names[n].index)]['id']
                L[tid] = dict(L[tid], local_rotation_xyzw=[float(x) for x in got[n]])
        t2 = time.time()
        out = {}
        cache = {}
        # exact reuse: a surface's skin at a frame is a deterministic function of the locals of
        # its bone closure (bones, root and their ancestors); unchanged closure -> baseline bytes
        changed = None
        if getattr(self, 'baseline', None) is not None:
            changed = self._changed_by_frame(locals_by_frame)
        for p in self.points:
            key = (p['renderer'], p['frame'])
            if key not in cache and changed is not None and not (self._closure(p['renderer']) & changed[p['frame']]):
                cache[key] = self.baseline['skin_cache'][key]
            if key not in cache:
                fc = p['frame']
                parent, loc = _frame_locals(locals_by_frame[fc], tf, self.by_id)
                r = self.renderers[p['renderer']]
                m = dict(self.win.meshes[r['id']])
                blended = self.ps.unity_blend(self.win.meshes[r['id']], self.win.by_frame[fc]['W'][r['id']])
                m['v'], m['nrm'], m['t'] = blended[:, 0:3], blended[:, 3:6], blended[:, 6:10]
                cache[key] = skin_from_locals(self.emu, parent, loc, r['bones'], r['root_bone'], m)
            out[p['chunk']] = bytes(cache[key][0])
        t3 = time.time()
        return {'bytes': out, 'locals': locals_by_frame, 'skin_cache': cache,
                'timing': {'forward_s': t1 - t0, 'eye_look_s': t2 - t1, 'skin_s': t3 - t2},
                'object_matrices': {k: (bytes(v[1]), bytes(v[2])) for k, v in cache.items()}}

    def _closure(self, renderer):
        c = self.__dict__.setdefault('_closures', {})
        if renderer not in c:
            r = self.renderers[renderer]
            ids, tf = set(), self.tf
            for b in list(r['bones']) + [r['root_bone']]:
                j = b
                while j is not None and j >= 0:
                    if tf[j]['id'] in ids:
                        break
                    ids.add(tf[j]['id'])
                    j = tf[j]['parent']
            # ancestors above the probe root (forward model transform table)
            frontier = [self.by_id[t['id']]['parent_id'] for t in tf if t['parent'] < 0]
            while frontier:
                tid = frontier.pop()
                if tid is None or tid in ids:
                    continue
                ids.add(tid)
                frontier.append(self.by_id[tid]['parent_id'])
            c[renderer] = ids
        return c[renderer]

    def _changed_by_frame(self, locals_by_frame):
        out = {}
        for fc, L in locals_by_frame.items():
            B = self.baseline['locals'][fc]
            ch = set()
            for tid, r in L.items():
                b = B[tid]
                # float32 bit comparison (a list != check would equate -0.0 and 0.0)
                if any(np.asarray(r[k], F).tobytes() != np.asarray(b[k], F).tobytes()
                       for k in ('local_position', 'local_rotation_xyzw', 'local_scale')):
                    ch.add(tid)
            out[fc] = ch
        return out

    def changed_transforms(self, streams):
        ch = set()
        for v in self._changed_by_frame(streams['locals']).values():
            ch |= v
        return ch

    # ------------------------------------------------------------ executor
    def _start_executor(self):
        cfg = self.cfg
        out = Path(cfg.work) / ('serve_%d' % int(time.time() * 1000))
        self.proc = subprocess.Popen([str(cfg.executor), str(cfg.template_run / 'generated'), str(out), '--serve'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                     bufsize=1)
        line = self.proc.stdout.readline()
        ready = json.loads(line)
        if ready.get('status') != 'ready':
            raise RuntimeError('executor serve did not start: %s %s' % (line, self.proc.stderr.read()))
        self.ready = ready

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
            self.proc = None

    def _execute(self, out_dir, resource_data, map_write_blobs):
        req = {'out': str(out_dir / 'exec'), 'hash_only': False, 'stop_after': self.target_command,
               'resource_data': resource_data, 'map_write_blobs': map_write_blobs,
               'readbacks': [{'name': 'bridge_target', 'after_command': self.target_command,
                              'resource': self.cfg.target_resource, 'subresource': 0,
                              'orientation': 'unity_texture_rows_bottom_up'}]}
        self.proc.stdin.write(json.dumps(req) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        r = json.loads(line)
        if r.get('status') != 'pass':
            raise RuntimeError('executor request failed: %s' % line)
        ex = json.loads((out_dir / 'exec' / 'execution.json').read_text(encoding='utf-8'))
        return r, ex['readbacks'][0]

    # ------------------------------------------------------------ public
    def render(self, candidate: dict, out_dir, *, patch_all=False, write_png=True) -> CandidateResult:
        """Render one candidate; ``patch_all`` substitutes every regenerated stream even
        when it equals the template (identity check)."""
        from PIL import Image
        t0 = time.time()
        out_dir = Path(out_dir)
        if out_dir.exists():
            raise FileExistsError('Fresh candidate directory required: %s' % out_dir)
        out_dir.mkdir(parents=True)
        bones = candidate.get('bones') or []
        native = candidate.get('native59')
        if native is not None and len(native) != 59:
            raise ValueError('native59 needs 59 values')
        if set(candidate) - {'bones', 'native59', 'name', 'card_value', 'notes'}:
            raise ValueError('candidate keys: bones, native59 (name, card_value, notes are descriptive)')
        st = self._streams(patches=bones, native=(None if native is None else [float(x) for x in native]))
        changed = self.changed_transforms(st)
        dep = self.dependency_check(changed)
        if dep['problems']:
            raise CandidateUnsupported('; '.join(dep['problems']))
        # object matrices of the head surfaces must be the template's (root unchanged)
        for k, v in st['object_matrices'].items():
            if v != self.baseline['object_matrices'][k]:
                raise CandidateUnsupported('ObjectToWorld/WorldToObject of renderer %s frame %s changed' % k)
        t1 = time.time()
        blobs = out_dir / 'streams'
        blobs.mkdir()
        resource_data, map_write = {}, {}
        patched = {'resources': 0, 'map_writes': 0, 'unchanged_skipped': 0}
        for p in self.points:
            data = st['bytes'][p['chunk']]
            if not patch_all and data == self.baseline['bytes'][p['chunk']]:
                patched['unchanged_skipped'] += 1
                continue
            f = blobs / ('chunk_%d.bin' % p['chunk'])
            f.write_bytes(data)
            kind, where = p['program']
            if kind == 'resource':
                resource_data[str(where)] = [{'subresource': 0, 'file': str(f), 'row_pitch': len(data),
                                              'slice_pitch': len(data)}]
                patched['resources'] += 1
            else:
                map_write[str(where)] = str(f)
                patched['map_writes'] += 1
        t2 = time.time()
        r, rb = self._execute(out_dir, resource_data, map_write)
        t3 = time.time()
        w, hgt = self.cfg.target_size
        raw = (out_dir / 'exec' / rb['topdown']).read_bytes()
        pixels = np.frombuffer(raw, np.uint8).reshape(hgt, w, 4)
        png = None
        if write_png:
            png = out_dir / 'bridge_target.png'
            Image.frombytes('RGBA', (w, hgt), raw).save(png)
        t4 = time.time()
        timings = {**st['timing'], 'dependency_s': t1 - t0 - sum(st['timing'].values()), 'write_streams_s': t2 - t1,
                   'executor_s': t3 - t2, 'executor_internal_ms': r, 'png_s': t4 - t3, 'total_s': t4 - t0}
        res = CandidateResult(png=png, pixels=pixels, target_sha256=rb['sha256'], timings=timings,
                              changed_bones=sorted(self.tf[self.index_of[t]]['name'] for t in changed
                                                   if t in self.index_of),
                              patched=patched, dependency=dep)
        (out_dir / 'result.json').write_text(json.dumps({'candidate': candidate, 'png': str(png),
                                                         'target_sha256_raw_d3d': rb['sha256'],
                                                         'timings': timings, 'changed_bones': res.changed_bones,
                                                         'patched': patched, 'dependency': {
                                                             k: v for k, v in dep.items() if k != 'citations'}},
                                                        indent=1, default=str), encoding='utf-8')
        return res
