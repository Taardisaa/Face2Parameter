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

Candidate model (stated): the candidate is applied from the probe frame N-4 on (the
locals of N-4..N come from card + patch).  The hidden EyeLook state entering N-4 is the
game's own state exported by the probe's EyeLookController.LateUpdate hook (eye postfix of
N-5 = prefix of N-4, ``eye_seed_state``) and the EyeLook target is the hook's
EyeUpdateCalc target argument; for the window's own card + patch both are exact.  Windows
without hook records (probe < 6) fall back to the reconstruction from the first frame's
look rotation (labelled).  The mouth width written before the bridge freeze
(UpdateBlendShapeVoice) is the exported FBSCtrlMouth.adjustWidthScale of N-1.

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
    # the patch the window was captured with (card + this patch is what the game rendered);
    # the template's skin streams are the window's, so the reference streams are generated
    # from card + window_patch
    window_patch: dict | None = None

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
        # probe driver hook records (drivers_NNN.json, probe v6+): EyeLookController.LateUpdate
        # prefix/postfix (EyeUpdateCalc target and EyeObject hidden state), FaceBlendShape
        # OnLateUpdate postfix (FBSCtrlMouth.adjustWidthScale), ChaControl.UpdateForce postfix
        self.hooks = {}
        for k in range(len(self.win.by_frame)):
            dp = cfg.window / 'frames' / ('drivers_%03d.json' % k)
            if dp.exists():
                for rec in json.loads(dp.read_text(encoding='utf-8'))['records']:
                    self.hooks[(int(rec['frame_count']), rec['hook'], rec['phase'])] = rec
        # Camera.main position per eye-look frame: the EyeUpdateCalc target argument of that frame's
        # eye prefix record (Camera.main.transform.position as the game passed it); without hook
        # records, the scene export's cameraToWorld translation (labelled stand-in)
        self.cameras, self.camera_source, cam = {}, {}, None
        for k, fc in enumerate(sorted(self.win.by_frame)):
            sc = cfg.window / 'frames' / ('scene_%03d.json' % k)
            cam = ela.camera_from_scene(sc)[0] if sc.exists() else cam
            pre = self.hooks.get((fc, 'eye', 'prefix'))
            if pre is not None:
                self.cameras[fc] = np.asarray(pre['call_arguments']['target_position'], F)
                self.camera_source[fc] = 'eye prefix call_arguments.target_position'
            else:
                self.cameras[fc] = cam
                self.camera_source[fc] = 'scene cameraToWorld translation (stand-in)'
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
        if cfg.template_run is not None:
            self._bind_template()
        self.scene = json.loads((cfg.window / 'frames' / ('scene_%03d.json' % self._window_index())).read_text(
            encoding='utf-8'))
        self._build_dependency_tables()
        self.baseline = None
        wp = cfg.window_patch or {}
        self.baseline = self._streams(patches=wp.get('bones') or (), native=wp.get('native59'))
        mism = [(p['chunk'], p['role']) for p in self.points if self.baseline['bytes'][p['chunk']] != p['captured']]
        if mism:
            raise RuntimeError('card + window patch does not reproduce the captured skin uploads: %r' % mism)
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
        # head surfaces whose world bounds come from their bones (updateWhenOffscreen: the
        # bone-AABB path of CalculateTransformInfo_Prepared 0xbdb41e..0xbdb9cc over Mesh bone
        # AABBs, which the probe does not export; not transcribed). Their bounds are taken from
        # the template and a candidate that moves any of their bones/root raises.
        self.bone_bounds_surfaces = sorted({rr['name'] for rr in rows
                                            if rr['head_surface'] and rr['update_when_offscreen']})

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
            # updateWhenOffscreen surfaces: bone-driven bounds are generated and checked in
            # frozen_pose_check (a changed box differs from the template and raises there)
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
    def eye_seed_state(self):
        """EyeObject hidden state (angleH, angleV, dirUp) entering the first recomputed frame.

        Probe v6+ windows: the game's own state, the eye postfix record of the window's first
        frame (EyeLookController.LateUpdate postfix), checked equal to the prefix record of the
        next frame (no writer between them).  For the window's card + patch this is exact; a
        different candidate starts from the window's state and is applied from the next frame
        on (the candidate model, stated in the report).  Older windows: reconstructed from
        the first frame's look rotation (eye_look_acceptance.seed_state, labelled)."""
        f0, f1 = self.eye_frames[0], self.eye_frames[1]
        post, pre = self.hooks.get((f0, 'eye', 'postfix')), self.hooks.get((f1, 'eye', 'prefix'))
        if post is None:
            self.eye_seed_source = 'reconstructed from the first frame look rotation (no hook records)'
            first = self.win.by_frame[f0]
            return self.ela.seed_state('reconstructed', first['T'], self.eye)

        def st(rec):
            return [{'angleH': F(e['EyeObject.angleH']), 'angleV': F(e['EyeObject.angleV']),
                     'dirUp': np.asarray(e['EyeObject.dirUp'], F)}
                    for e in rec['eye_look_calc_fields']['EyeLookCalc.eyeObjs']]
        s = st(post)
        if pre is not None:
            s1 = st(pre)
            for a, b in zip(s, s1):
                if (a['angleH'].tobytes() != b['angleH'].tobytes() or a['angleV'].tobytes() != b['angleV'].tobytes()
                        or a['dirUp'].tobytes() != b['dirUp'].tobytes()):
                    raise CandidateUnsupported('EyeLook state changes between the frame %d postfix and the '
                                               'frame %d prefix' % (f0, f1))
        self.eye_seed_source = 'game state: eye postfix record of frame %d (= prefix of %d)' % (f0, f1)
        return s

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
        h.el.set_state(self.eye_seed_state())
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

    # ------------------------------------------------------------ frozen bridge pose
    def frozen_pose_check(self, streams):
        """Per bridge render: generate the frozen-pose locals, then every enabled
        SkinnedMeshRenderer's rootBone world TRS (ObjectToWorld) and, for local-AABB
        renderers, the world bounds (``frozen_pose``); compare them bit for bit with the
        render record values the template program was built from.

        Frozen locals (MakerRenderService.PoseFreeze: Animator Play(hash, 0, 0) + Update(0)
        inside frame N, before frame N's animation and LateUpdate): transforms the Animator
        rewrites carry the phase-zero sample (bridge pre_cull export; the clip bindings are
        not parsed, so 'rewritten' = differs from the N-1 end-of-frame export); every other
        transform keeps its N-1 end-of-frame local, which for the forward model's outputs
        is the forward chain (card + patch) at N-1."""
        import base64
        from src.render.unity_skin_emu import global_position_rotation
        from . import frozen_pose as fpz, scene_v5 as sv
        cfg = self.cfg
        idx = self._window_index()
        recs = sv.resolve_records(json.loads((cfg.window / 'frames' / ('renders_%03d.json' % idx)).read_text(
            encoding='utf-8')), sv.prior_seed(cfg.window, idx))
        pc = {r['bridge_render']['ordinal']: r for r in recs if r['phase'] == 'pre_cull' and r.get('bridge_render')}
        pr = {r['bridge_render']['ordinal']: r for r in recs if r['phase'] == 'pre_render' and r.get('bridge_render')}
        if not pc:
            raise CandidateUnsupported('window has no bridge pre_cull character transforms (probe < 6.1)')
        tf, prev = self.tf, self.N - 1
        fwd = streams['locals'][prev]
        T1 = self.win.by_frame[prev]['T']
        parent = {}
        for t in tf:
            parent[t['id']] = tf[t['parent']]['id'] if t['parent'] >= 0 else self.by_id[t['id']]['parent_id']
        for tid in list(parent.values()):
            cur = tid
            while cur is not None and cur not in parent:
                parent[cur] = self.by_id[cur]['parent_id']
                cur = parent[cur]
        out = {'renders': 0, 'rows': 0, 'root_trs_equal': 0, 'bounds_equal': 0, 'bounds_rows': 0,
               'animator_rewritten': None, 'forward_frozen_mismatch': [], 'mismatch': []}
        # probe v7: the transforms the pose Animator writes (clip bindings of the current state,
        # write-default-values: every bound transform); PoseFreeze's Update(0) writes exactly these
        written = None
        cb = cfg.session / ('pose_clip_bindings_%s.json' % cfg.nonce)
        if cb.exists():
            anim = json.loads(cb.read_text(encoding='utf-8'))['animators']
            written = {tf[i]['id'] for a in anim for i in a['transforms_written_by_current_state']}
            out['clip_binding_written'] = len(written)
        bone_bounds = set()
        self.__dict__.setdefault('_bone_aabbs', {})
        for k in sorted(pr):
            A = np.frombuffer(base64.b64decode(pc[k]['data']['character_transforms']['transforms_f32']),
                              F).reshape(-1, 33)
            differs = {tf[i]['id'] for i in range(len(tf)) if A[i, 0:10].tobytes() != T1[i, 0:10].tobytes()}
            # second writer inside frame N before the bridge: Manager.Character.Update ->
            # ChaControl.UpdateForce -> UpdateBlendShapeVoice (ChaControl.cs 5996..6025) sets
            # cf_J_MouthMove localScale.x = FBSCtrlMouth.GetAdjustWidthScale() when
            # fileStatus.mouthAdjustWidth; adjustWidthScale is 1f unless openRate > 0.2
            # (FBSCtrlMouth.AdjustWidthScale); ABMX re-applies its modifier only in LateUpdate
            mouth_ids = {t['id'] for t in tf if t['name'] == 'cf_J_MouthMove'}
            # GetAdjustWidthScale() at frame N's Update = FBSCtrlMouth.adjustWidthScale left by the
            # frame N-1 FaceBlendShape OnLateUpdate (fbs postfix record, probe v6+); the call is made
            # only when fileStatus.mouthAdjustWidth (UpdateForce postfix mouth record of frame N).
            # FBS reads timers, Random and the expression state, never geometry: the value holds
            # for every candidate.  TransformScaleEx.SetLocalScaleX writes scale.x only.
            fbs_post = self.hooks.get((self.N - 1, 'fbs', 'postfix'))
            uf_post = self.hooks.get((self.N, 'chara_update_force', 'postfix'))
            if fbs_post is not None and uf_post is not None:
                adjust_x = F(fbs_post['face_blend_shape_fields']['FaceBlendShape.MouthCtrl']
                             ['FBSCtrlMouth.adjustWidthScale'])
                adjust_on = bool(uf_post['mouth']['file_status_mouth_adjust_width'])
                adjust_src = 'FBSCtrlMouth.adjustWidthScale of the frame N-1 fbs postfix (%r), fileStatus %r' % (
                    float(adjust_x), adjust_on)
            else:
                adjust_x, adjust_on = F(1.0), True
                adjust_src = 'asserted 1f (no driver hook records: openRate not exported)'
            if written is not None:
                if not differs <= (written | mouth_ids):
                    raise CandidateUnsupported('frozen locals differ from N-1 outside the clip-binding write set: %r'
                                               % sorted(tf[self.index_of[t]]['name'] for t in differs - written)[:8])
                rewritten = set(written)
                out['diff_set_within_write_set'] = True
                out['written_but_equal_to_n1'] = len(written - differs)
            else:
                rewritten = differs
            if out['animator_rewritten'] is None:
                out['animator_rewritten'] = sorted(tf[self.index_of[t]]['name'] for t in rewritten)
                out['animator_set_source'] = ('probe v7 pose_clip_bindings (current state)' if written is not None
                                              else 'differs from the N-1 export (clip bindings not exported)')
            loc = {}
            for i, t in enumerate(tf):
                tid = t['id']
                if tid in mouth_ids and tid in fwd and tid not in rewritten:
                    L = fwd[tid]
                    sc = np.asarray(L['local_scale'], F).copy()
                    if adjust_on:
                        sc[0] = adjust_x           # SetLocalScaleX(GetAdjustWidthScale())
                    g_ = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F), sc)
                    if np.concatenate(g_).tobytes() != A[i, 0:10].tobytes():
                        raise CandidateUnsupported('cf_J_MouthMove frozen local differs from the UpdateBlendShapeVoice '
                                                   'rule (%s)' % adjust_src)
                    out['mouth_adjust_width_rule'] = ('cf_J_MouthMove scale.x = %s (UpdateBlendShapeVoice), equal'
                                                      % adjust_src)
                    loc[tid] = g_
                    continue
                if tid in fwd and tid not in rewritten:
                    L = fwd[tid]
                    g_ = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F),
                          np.asarray(L['local_scale'], F))
                    if np.concatenate(g_).tobytes() != A[i, 0:10].tobytes() and k == min(pr):
                        out['forward_frozen_mismatch'].append(t['name'])
                    loc[tid] = g_
                else:
                    loc[tid] = (A[i, 0:3].copy(), A[i, 3:7].copy(), A[i, 7:10].copy())
            for tid in parent:
                if tid is not None and tid not in loc:
                    L = fwd[tid]
                    loc[tid] = (np.asarray(L['local_position'], F), np.asarray(L['local_rotation_xyzw'], F),
                                np.asarray(L['local_scale'], F))
            out['renders'] += 1
            for r in pr[k]['data']['renderers']:
                if r['type'] != 'UnityEngine.SkinnedMeshRenderer' or not r.get('root_bone') or not r['enabled']:
                    continue
                rb = r['root_bone']
                ch = fpz.chain_of(loc, parent, rb['id'])
                pos, rot = global_position_rotation(self.emu, ch)
                out['rows'] += 1
                if (pos.tobytes() == np.asarray(rb['position'], F).tobytes()
                        and rot.tobytes() == np.asarray(rb['rotation'], F).tobytes()):
                    out['root_trs_equal'] += 1
                else:
                    out['mismatch'].append({'render': k, 'renderer': r['id'], 'field': 'root_trs'})
                if r.get('update_when_offscreen'):
                    # bone-driven bounds (CalculateTransformInfo_Prepared bone path): Mesh bone
                    # AABBs from the exported mesh (GetCachedBonesBounds rule) and the bone
                    # matrices composed from the hierarchy top (frozen_pose.prepared_*)
                    st_ = self.renderers.get(r['id'])
                    mesh_ = self.win.meshes.get(r['id']) if st_ is not None else None
                    if mesh_ is None or mesh_.get('shapes'):
                        bone_bounds.add(r['id'])      # blend-shape branch 0x7a2ce3 not transcribed
                        continue
                    if r['id'] not in self._bone_aabbs:
                        self._bone_aabbs[r['id']] = fpz.bone_aabbs(mesh_['v'], mesh_['w'], mesh_['bi'], mesh_['bind'])
                    ident = [np.array(c_, F) for c_ in np.eye(4)]
                    mats = []
                    for b in st_['bones']:
                        ids_, c_ = [], tf[b]['id']
                        while c_ is not None:
                            ids_.append(c_)
                            c_ = parent[c_]
                        M = ident
                        for tid_ in reversed(ids_):
                            M = fpz.prepared_compose(M, fpz.prepared_local_matrix(*loc[tid_]))
                        mats.append(M)
                    out['bounds_rows'] += 1
                    c, e = fpz.bone_world_aabb(mats, self._bone_aabbs[r['id']])
                    if np.concatenate([c, e]).tobytes() == np.asarray(r['bounds'], F).tobytes():
                        out['bounds_equal'] += 1
                        out['bone_bounds_generated'] = out.get('bone_bounds_generated', 0) + 1
                    else:
                        out['mismatch'].append({'render': k, 'renderer': r['id'], 'field': 'bone_bounds'})
                    continue
                out['bounds_rows'] += 1
                c, e = fpz.world_aabb(fpz.global_matrix(self.emu, ch), r['local_bounds'][0:3], r['local_bounds'][3:6])
                if np.concatenate([c, e]).tobytes() == np.asarray(r['bounds'], F).tobytes():
                    out['bounds_equal'] += 1
                else:
                    out['mismatch'].append({'render': k, 'renderer': r['id'], 'field': 'bounds'})
        out['bone_bounds_renderers'] = sorted(bone_bounds)
        return out

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
        frozen = None
        if getattr(self, 'has_frozen_export', None) is not False:
            try:
                frozen = self.frozen_pose_check(st)
                self.has_frozen_export = True
            except CandidateUnsupported:
                self.has_frozen_export = False
        if frozen is not None and frozen['mismatch']:
            raise CandidateUnsupported('generated frozen-pose matrices/bounds differ from the template: %r'
                                       % frozen['mismatch'][:8])
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
        res.frozen = frozen
        (out_dir / 'result.json').write_text(json.dumps({'candidate': candidate, 'png': str(png),
                                                         'target_sha256_raw_d3d': rb['sha256'],
                                                         'timings': timings, 'changed_bones': res.changed_bones,
                                                         'patched': patched, 'frozen_pose': frozen,
                                                         'dependency': {
                                                             k: v for k, v in dep.items() if k != 'citations'}},
                                                        indent=1, default=str), encoding='utf-8')
        return res
