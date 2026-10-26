"""Differentiable head geometry on one paired bridge window (probe v6+), next to the exact chain.

``DiffGeoSession`` is a ``BridgeCandidateRenderer`` (``src/render/hs2pipe/candidate.py``) without
the capture program: it loads the window exactly as the exact chain does (probe export, the
per-frame ``paired_forward_context.build`` + ``CapturedHeadForward`` models, the driver hook
records, the hosted EyeLook harness, the scene export) and keeps the exact chain's own methods
(``_streams``, ``eye_seed_state``, ``frozen_pose_check``, ``dependency_check``).  Its skin
"upload points" are synthetic -- one per (renderer, frame) of every mesh renderer of the window at
N-2 / N-1 / N -- so ``_streams`` returns the exact chain's ``skin_from_locals`` bytes for any
patch without a RenderDoc capture.

The torch chain (``torch_streams``) recomputes the same quantities with autograd:

1. per frame, ``forward.TorchHeadForward.evaluate(baseline_mode='card_load')``;
2. EyeLook (``cf_J_look_L/R``): **not differentiable** -- installed IL on the game's Mono JIT.  It
   is evaluated exactly at the current iterate and held constant: if no transform of the
   EyeLook input closure (the world TRS of ``cf_J_FaceUp_tz`` and ``cf_J_EyePos_rz_L/R`` read by
   ``EyeLookCalc.EyeUpdateCalc``: those transforms and all their ancestors) changes bitwise from
   the window's own state, the look rotations are the exact chain's (deterministic function of
   unchanged inputs); otherwise the hosted controller runs over N-4..N on the torch forward's
   own (detached, binary32) locals, as ``_streams`` does;
3. FBS blend shapes: the probe's frame weights (FaceBlendShape reads timers, Random and the
   expression state, never geometry -- hs2_offline_head_skinning.md section 9.3), blended by the
   exact port ``paired_skin.unity_blend``; a constant of every parameter;
4. per renderer and frame: root world TRS, root world-to-local, prepared palette and
   SkinVerticesGenericCPU (``unity``), plus unity_ObjectToWorld;
5. the bridge render's frozen root TRS (``frozen_root``): PoseFreeze phase-0 locals for the
   Animator write set (probe v7 clip bindings), the forward's N-1 locals otherwise.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from src.render.hs2pipe.candidate import BridgeCandidateRenderer, SessionConfig, CandidateUnsupported

from . import unity as U
from .fp import Arith, DT
from .forward import TorchHeadForward, local_f32_bytes

F = np.float32
EYE_INPUTS = ('cf_J_FaceUp_tz', 'cf_J_EyePos_rz_L', 'cf_J_EyePos_rz_R')


class DiffGeoSession(BridgeCandidateRenderer):
    def __init__(self, cfg: SessionConfig, *, arith: Arith | None = None, renderers='all'):
        """Window data exactly as ``BridgeCandidateRenderer.__init__`` (candidate.py lines 139-196),
        no capture program / executor.  ``renderers``: 'all' mesh renderers of the window, or
        'head' (renderers whose root and bones lie under cf_J_Head)."""
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
        self.cfg = cfg
        self.ps, self.g = ps, g
        self.A = arith or Arith('exact')
        self.win = ps.ProbeWindow(cfg.window)
        self.tf = self.win.static['transforms']
        self.emu = UnityPlayerEmu()
        frames = sorted(self.win.by_frame)
        self.N = json.loads((cfg.window / 'capture_receipt.json').read_text(encoding='utf-8'))['expected_capture_frame']
        self.eye_frames = [f for f in frames if f <= self.N]
        self.skin_frames = (self.N - 2, self.N - 1, self.N)
        self.models, self.geo = {}, None
        for fc in self.eye_frames[1:]:
            geo, ctx, info = build(cfg.session / 'prepare_geometry_actor.json', cfg.artifacts / cfg.loaded_state,
                                   cfg.effect_context, self.win, fc)
            self.models[fc] = CapturedHeadForward(geo, ctx)
            self.geo = geo
        self.by_id = {t['id']: t for t in self.geo['transforms']}
        self.card_native = list(self.models[self.N].identity['native59'])
        self.eye = None                    # hosted EyeLook harness, created on first use (Mono: once per process)
        self.ela = ela
        self.hooks = {}
        for k in range(len(self.win.by_frame)):
            dp = cfg.window / 'frames' / ('drivers_%03d.json' % k)
            if dp.exists():
                for rec in json.loads(dp.read_text(encoding='utf-8'))['records']:
                    self.hooks[(int(rec['frame_count']), rec['hook'], rec['phase'])] = rec
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
        self.renderers = {r['id']: r for r in self.win.static['renderers']}
        self.index_of = {t['id']: i for i, t in enumerate(self.tf)}
        head_root = self.index_of[self._unique_probe('cf_J_Head')]
        under = self._subtree(head_root)
        active = {}
        for fr in self.win.frames:
            for row in fr['renderers']:
                if 'active' in row:
                    active.setdefault(row['id'], set()).add(bool(row['active']))
        rows = []
        for rid, r in self.renderers.items():
            if rid not in self.win.meshes or active.get(rid) == {False}:
                continue
            is_head = r['root_bone'] in under and all(b in under for b in r['bones'])
            if renderers == 'head' and not is_head:
                continue
            rows.append(rid)
        self.mesh_renderers = sorted(rows)
        # synthetic upload points: one per (renderer, frame), keyed by that pair
        self.points = []
        for rid in self.mesh_renderers:
            for fc in self.skin_frames:
                role = 'skinned_vertices_previous' if fc == self.N - 1 else 'skinned_vertices_current'
                self.points.append({'chunk': (rid, fc), 'role': role, 'renderer': rid, 'frame': fc,
                                    'capture_start': fc == self.N - 2, 'captured': None})
        self.head_renderers = sorted(rid for rid in self.mesh_renderers if self.renderers[rid]['root_bone'] in under)
        self.scene = json.loads((cfg.window / 'frames' / ('scene_%03d.json' % self._window_index())).read_text(
            encoding='utf-8'))
        self._build_dependency_tables()
        self.tforward = {fc: TorchHeadForward(m, self.A) for fc, m in self.models.items()}
        self._blend = {}
        self._memo = {}
        self.baseline = None
        wp = cfg.window_patch or {}
        self.window_patch = wp
        self.baseline = self._streams(patches=wp.get('bones') or (), native=wp.get('native59'))
        self.timings_init = {'init_s': time.time() - t0}
        self.proc = None

    def with_arith(self, arith):
        """A view of this session evaluating the torch chain in another arithmetic mode (the exact
        chain, window data and EyeLook harness are shared)."""
        import copy as _copy
        other = _copy.copy(self)
        other.A = arith
        other.tforward = {fc: TorchHeadForward(m, arith) for fc, m in self.models.items()}
        other._memo = {}
        return other

    # ------------------------------------------------------------------ helpers
    def _unique_probe(self, name):
        ids = [t['id'] for t in self.tf if t['name'] == name]
        if len(ids) != 1:
            raise ValueError('one probe transform required: ' + name)
        return ids[0]

    def _subtree(self, root_index):
        kids = {}
        for i, t in enumerate(self.tf):
            if t['parent'] >= 0:
                kids.setdefault(t['parent'], []).append(i)
        out, stack = set(), [root_index]
        while stack:
            i = stack.pop()
            if i in out:
                continue
            out.add(i)
            stack.extend(kids.get(i, []))
        return out

    def _start_executor(self):
        raise RuntimeError('DiffGeoSession has no frame executor (geometry only)')

    def _ensure_eye(self):
        if self.eye is None:
            self.eye = self.ela.Harness(self.win.static)
        return self.eye

    def _streams(self, patches, native):
        self._ensure_eye()
        return BridgeCandidateRenderer._streams(self, patches, native)

    def exact_streams(self, patch):
        """The exact chain (numpy + emulated UnityPlayer + hosted EyeLook) for a bridge patch dict."""
        native = patch.get('native59')
        return self._streams(patches=patch.get('bones') or (),
                             native=None if native is None else [float(x) for x in native])

    def blended_source(self, rid, fc):
        """ApplyBlendShapes of the probe frame weights (exact port; parameter-independent), (n, 10)."""
        key = (rid, fc)
        if key not in self._blend:
            src = self.ps.unity_blend(self.win.meshes[rid], self.win.by_frame[fc]['W'][rid])
            self._blend[key] = torch.as_tensor(np.asarray(src, F).astype(np.float64), dtype=DT)
        return self._blend[key]

    # ------------------------------------------------------------------ torch chain
    def torch_locals(self, patches=(), native=None, frames=None, apply_count=None):
        frames = frames or self.skin_frames
        ac = apply_count or self.cfg.apply_count
        out, branches = {}, {}
        for fc in frames:
            res = self.tforward[fc].evaluate(patches, native=native, apply_count=ac, baseline_mode='card_load')
            out[fc] = res['bone_local']
            branches[fc] = res['branches']
        return out, branches

    def eye_input_closure(self):
        if not hasattr(self, '_eye_closure'):
            ids = set()
            for name in EYE_INPUTS:
                tid = self._unique_probe(name)
                while tid is not None:
                    ids.add(tid)
                    i = self.index_of.get(tid)
                    tid = (self.tf[self.tf[i]['parent']]['id'] if i is not None and self.tf[i]['parent'] >= 0
                           else self.by_id[tid]['parent_id'])
            self._eye_closure = ids
        return self._eye_closure

    def eye_inputs_untouched(self, patch):
        """Structural rule (source): BoneModifier.Apply writes only its own bone, so the EyeLook
        input closure keeps the window's locals when no patched / removed bone lies in it and
        native59 is the window's (the card-load baselines and the native writes are then the
        same).  The bitwise check on the evaluated frames confirms it."""
        names = {self.tf[self.index_of[t]]['name'] if t in self.index_of else self.by_id[t]['name']
                 for t in self.eye_input_closure()}
        bones = list(patch.get('bones') or ()) + list(self.window_patch.get('bones') or ())
        if any(b['name'] in names or b.get('remove') for b in bones):
            return False
        nat = patch.get('native59')
        ref = self.window_patch.get('native59') or self.card_native
        if nat is not None:
            v = nat.detach().numpy() if isinstance(nat, torch.Tensor) else np.asarray(nat, float)
            if np.asarray(v, F).tobytes() != np.asarray(ref, F).tobytes():
                return False
        return True

    def look_rotations(self, locals_by_frame, patch=None, *, force_hosted=False):
        """{frame: {look tid: (4,) constant tensor}} -- exact EyeLook at this iterate (see module doc).
        Returns (rotations, source label)."""
        look_ids = [self._unique_probe(n) for n in self.LOOK]
        closure = self.eye_input_closure()
        base = self.baseline['locals']
        if not force_hosted and patch is not None and self.eye_inputs_untouched(patch):
            same = (not self.A.exact) or all(local_f32_bytes(L[t]) == _row_bytes(base[fc][t])
                                             for fc, L in locals_by_frame.items() for t in closure)
            if not same:
                raise RuntimeError('EyeLook input closure changed although no parameter reaches it (bug)')
            rot = {fc: {t: self.A.f32(base[fc][t]['local_rotation_xyzw']) for t in look_ids}
                   for fc in self.eye_frames[1:]}
            return rot, 'exact chain window state (EyeLook input closure untouched; checked bitwise)'
        missing = [fc for fc in self.eye_frames[1:] if fc not in locals_by_frame]
        if missing:
            raise CandidateUnsupported('EyeLook inputs changed; torch locals of frames %r are required' % missing)
        h = self._ensure_eye()
        first = self.win.by_frame[self.eye_frames[0]]
        h.apply_frame(first['T'])
        h.el.set_state(self.eye_seed_state())
        out = {}
        for fc in self.eye_frames[1:]:
            fr = self.win.by_frame[fc]
            T = fr['T'].copy()
            L = locals_by_frame[fc]
            for i, t in enumerate(self.tf):
                r = L[t['id']]
                T[i, 0:3] = np.asarray(r['pos'].detach().numpy(), F)
                T[i, 3:7] = np.asarray(r['rot'].detach().numpy(), F)
                T[i, 7:10] = np.asarray(r['scl'].detach().numpy(), F)
            got = h.run({'T': T, 'delta_time': fr['delta_time'], 'time': fr['time'], 'frame_count': fr['frame_count']},
                        self.cameras[fc])
            out[fc] = {}
            for n in self.LOOK:
                tid = self.tf[h.ids.index(h.names[n].index)]['id']
                out[fc][tid] = self.A.f32(got[n])
        return out, 'hosted EyeLookController on the torch locals (detached, held constant)'

    def _parent_map(self):
        if not hasattr(self, '_parents'):
            parent = {}
            for t in self.tf:
                parent[t['id']] = self.tf[t['parent']]['id'] if t['parent'] >= 0 else self.by_id[t['id']]['parent_id']
            for tid in list(parent.values()):
                cur = tid
                while cur is not None and cur not in parent:
                    parent[cur] = self.by_id[cur]['parent_id']
                    cur = parent[cur]
            self._parents = parent
        return self._parents

    def _chain(self, L, tid):
        parent = self._parent_map()
        out = []
        while tid is not None:
            r = L[tid]
            out.append((r['pos'], r['rot'], r['scl']))
            tid = parent[tid]
        return out

    def skin(self, L, fc, rids=None):
        """{renderer: {'vb': (n, 10), 'o2w': (4, 4), 'root_pos', 'root_rot'}} from frame locals L
        (look rotations already set).  Renderers with the same root bone share one prepared
        hierarchy: a pose row is the same deterministic product whatever the entry order, so
        the union of their closures is evaluated once (bit-identical to per-renderer)."""
        A = self.A
        parent = self._parent_map()
        rids = rids or self.mesh_renderers
        groups = {}
        for rid in rids:
            groups.setdefault(self.tf[self.renderers[rid]['root_bone']]['id'], []).append(rid)
        out = {}
        for root, members in groups.items():
            pos, rot = U.global_position_rotation(A, self._chain(L, root), self._memo)
            base = U.world_to_local_noscale(A, pos, rot)
            o2w = U.object_to_world(A, pos, rot)
            order, where = [], {}

            def add(tid):
                if tid in where:
                    return
                if parent[tid] is not None:
                    add(parent[tid])
                order.append(tid)
                where[tid] = len(order)
            for rid in members:
                for b in self.renderers[rid]['bones']:
                    add(self.tf[b]['id'])
            entries = [((where[parent[t]] if parent[t] is not None else 0), L[t]['pos'], L[t]['rot'], L[t]['scl'])
                       for t in order]
            rows = U.prepared_poses(A, base, entries, self._memo)
            for rid in members:
                m = self.win.meshes[rid]
                key = ('bind', rid)
                if key not in self._blend:
                    self._blend[key] = torch.as_tensor(np.stack([np.asarray(b, F).T.ravel() for b in m['bind']])
                                                       .astype(np.float64), dtype=DT)
                    self._blend[('w', rid)] = torch.as_tensor(np.asarray(m['w'], F).astype(np.float64), dtype=DT)
                bones = [self.tf[b]['id'] for b in self.renderers[rid]['bones']]
                palette = U.palette_from_poses(A, [rows[where[b]] for b in bones], self._blend[key])
                vb = U.skin_vertices4(A, self.blended_source(rid, fc), palette, self._blend[('w', rid)], m['bi'])
                out[rid] = {'vb': vb, 'o2w': o2w, 'root_pos': pos, 'root_rot': rot}
        return out

    def torch_streams(self, patch=None, *, frames=None, rids=None, force_hosted_eye=False):
        """Torch chain for a bridge patch ``{'bones': [...], 'native59': [...]}`` (values may be
        tensors).  Returns {'skin': {frame: {renderer: ...}}, 'locals', 'branches', 'eye_source'}."""
        patch = patch or {}
        frames = tuple(frames or self.skin_frames)
        need = set(frames)
        locs, branches = self.torch_locals(patch.get('bones') or (), patch.get('native59'), frames=sorted(need))
        if force_hosted_eye or not self.eye_inputs_untouched(patch):
            extra = [fc for fc in self.eye_frames[1:] if fc not in locs]
            if extra:
                more, _ = self.torch_locals(patch.get('bones') or (), patch.get('native59'), frames=extra)
                locs.update(more)
        look, src = self.look_rotations(locs, patch, force_hosted=force_hosted_eye)
        skin = {}
        for fc in frames:
            L = dict(locs[fc])
            for tid, q in look[fc].items():
                L[tid] = dict(L[tid], rot=q)
            locs[fc] = L
            skin[fc] = self.skin(L, fc, rids)
        return {'skin': skin, 'locals': locs, 'branches': branches, 'eye_source': src}

    # ------------------------------------------------------------------ frozen bridge pose
    def frozen_written(self):
        """(rewritten transform ids, pre_cull export arrays per render, mouth rule) of the bridge freeze
        (BridgeCandidateRenderer.frozen_pose_check, probe v7 clip bindings)."""
        if hasattr(self, '_frozen'):
            return self._frozen
        import base64
        from src.render.hs2pipe import scene_v5 as sv
        cfg = self.cfg
        idx = self._window_index()
        recs = sv.resolve_records(json.loads((cfg.window / 'frames' / ('renders_%03d.json' % idx)).read_text(
            encoding='utf-8')), sv.prior_seed(cfg.window, idx))
        pc = {r['bridge_render']['ordinal']: r for r in recs if r['phase'] == 'pre_cull' and r.get('bridge_render')}
        pr = {r['bridge_render']['ordinal']: r for r in recs if r['phase'] == 'pre_render' and r.get('bridge_render')}
        if not pc:
            raise CandidateUnsupported('window has no bridge pre_cull character transforms (probe < 6.1)')
        arrays = {k: np.frombuffer(base64.b64decode(pc[k]['data']['character_transforms']['transforms_f32']),
                                   F).reshape(-1, 33) for k in pc}
        cb = cfg.session / ('pose_clip_bindings_%s.json' % cfg.nonce)
        if cb.exists():
            anim = json.loads(cb.read_text(encoding='utf-8'))['animators']
            ws = {self.tf[i]['id'] for a in anim for i in a['transforms_written_by_current_state']}
            written = {k: ws for k in arrays}
            source = 'probe v7 pose_clip_bindings (current state)'
        else:
            # probe 6.1 (frozen_pose_check fallback, labelled there): the transforms whose frozen
            # export differs from the N-1 end-of-frame export
            T1 = self.win.by_frame[self.N - 1]['T']
            written = {k: {self.tf[i]['id'] for i in range(len(self.tf)) if A_[i, 0:10].tobytes() != T1[i, 0:10].tobytes()}
                       for k, A_ in arrays.items()}
            source = 'differs from the N-1 export (clip bindings not exported, probe < 7)'
        self.frozen_set_source = source
        fbs_post = self.hooks.get((self.N - 1, 'fbs', 'postfix'))
        uf_post = self.hooks.get((self.N, 'chara_update_force', 'postfix'))
        if fbs_post is None or uf_post is None:
            raise CandidateUnsupported('driver hook records required for the UpdateBlendShapeVoice mouth rule')
        adjust_x = F(fbs_post['face_blend_shape_fields']['FaceBlendShape.MouthCtrl']['FBSCtrlMouth.adjustWidthScale'])
        adjust_on = bool(uf_post['mouth']['file_status_mouth_adjust_width'])
        mouth = {t['id'] for t in self.tf if t['name'] == 'cf_J_MouthMove'}
        self._frozen = {'written': written, 'arrays': arrays, 'records': pr, 'mouth': mouth,
                        'adjust_x': adjust_x, 'adjust_on': adjust_on}
        return self._frozen

    def frozen_root(self, L_prev, rid, render=0):
        """Root world (pos, rot) and unity_ObjectToWorld of renderer ``rid`` at bridge render ``render``
        from the forward locals of N-1 (``L_prev``): the frozen-pose rule of frozen_pose_check."""
        A = self.A
        fz = self.frozen_written()
        Arr = fz['arrays'][render]
        parent = self._parent_map()
        tid = self.tf[self.renderers[rid]['root_bone']]['id']
        chain = []
        while tid is not None:
            i = self.index_of.get(tid)
            if i is not None and tid in fz['written'][render]:
                chain.append((A.f32(Arr[i, 0:3]), A.f32(Arr[i, 3:7]), A.f32(Arr[i, 7:10])))
            elif tid in fz['mouth'] and fz['adjust_on']:
                r = L_prev[tid]
                scl = torch.cat([A.f32([fz['adjust_x']]), r['scl'][1:]])
                chain.append((r['pos'], r['rot'], scl))
            else:
                r = L_prev[tid]
                chain.append((r['pos'], r['rot'], r['scl']))
            tid = parent[tid]
        pos, rot = U.global_position_rotation(A, chain)
        return pos, rot, U.object_to_world(A, pos, rot)


def _row_bytes(row):
    return np.concatenate([np.asarray(row[k], F) for k in ('local_position', 'local_rotation_xyzw',
                                                            'local_scale')]).tobytes()


def stream_bytes(vb):
    return np.asarray(vb.detach().numpy(), F).tobytes()


def matrix_bytes_o2w(o2w):
    """unity_ObjectToWorld bytes as the exact chain writes them (column-major float32)."""
    return np.asarray(o2w.detach().numpy(), np.float64).T.astype(F).tobytes()
