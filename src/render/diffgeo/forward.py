"""Head bone forward (card + patch -> bone local TRS) over torch.

Mirror of the exact chain's ``CapturedHeadForward.evaluate`` (``src/hs2_offline_state.py``),
statement for statement, with its data taken from the exact chain's own model object (recorded
transforms, ordered ABMX modifiers and private caches, head prefab rig, accessory/AdvIK effect
modifiers, rotation exclusions -- ``paired_forward_context.build`` + ``CapturedHeadForward``):

* ``baseline_mode='card_load'`` -- ``BoneController.CollectBaselineCo`` after a card load:
  head prefab locals -> ShapeHeadInfoFemale (native59, ``native_shape``) -> ShapeBodyInfoFemale
  head-chain writes -> fresh ``CollectBaseline`` of the prefab-bone modifiers; native-written
  channels persist into the frame state;
* ``baseline_mode='captured'`` -- recorded caches; a given ``native`` follows
  ``BoneController.UpdateBaseline`` (Reset -> UpdateShapeFace/Body -> CollectBaseline);
* patches (bridge ``{"bones": [...]}`` rows; values may be tensors that require grad), modifier
  removal (Reset, then the body / face update), then ``apply_count`` LateUpdate passes of
  ``BoneModifier.Apply`` over the ordered modifiers (``abmx``) with the effect modifiers.

The ShapeBodyInfoFemale writes (cf_N_height, cf_J_Head_s) depend only on the card's body values,
not on any parameter of this chain; they are evaluated by the exact port and enter as constants.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from src.render.shape_head_female_program import DST, UPDATE

from . import abmx as X
from . import native_shape as NS
from .fp import DT

PREFIX = 'KKABMX.Core.BoneModifier.'
FIELDS = {'ScaleModifier': 'scale', 'LengthModifier': 'length', 'PositionModifier': 'position',
          'RotationModifier': 'rotation'}
BODY_NAMES = ('cf_N_height', 'cf_J_Head_s')
EXACT_PATCH_FIELDS = set(X.IDENTITY) | {'name', 'remove'}


def _copy_local(local):
    return {tid: dict(v) for tid, v in local.items()}


def _copy_modifiers(mods):
    return [{'id': m['id'], 'name': m['name'], 'cache': dict(m['cache']),
             'coordinate_modifiers': [None if c is None else dict(c) for c in m['coordinate_modifiers']]}
            for m in mods]


class TorchHeadForward:
    def __init__(self, model, A):
        self.m = model
        self.A = A
        self._base = None
        self._body = None
        self._prefab = None
        self._card_cache = {}
        self._apply_memo = {}
        self.mods0 = []
        for m in model.modifiers:
            self.mods0.append({'id': m['id'], 'name': m['name'], 'cache': X.cache_from_record(A, m['cache']),
                               'coordinate_modifiers': [X.modifier(A, c) for c in m['coordinate_modifiers']]})
        self.additional = {name: [X.modifier(A, d) for d in rows] for name, rows in model.additional.items()}
        self.coordinate = 0 if model.identity['abmx_coordinate'] == 'Unknown' else int(model.identity['abmx_coordinate'])
        if self.coordinate < 0:
            raise ValueError('Invalid coordinate index')

    # ------------------------------------------------------------------ constants
    def _trs(self, pos, rot, scl):
        A = self.A
        return {'pos': A.f32(pos), 'rot': A.f32(rot), 'scl': A.f32(scl)}

    def base_locals(self):
        if self._base is None:
            self._base = {i: self._trs(r['local_position'], r['local_rotation_xyzw'], r['local_scale'])
                          for i, r in self.m.transforms.items()}
        return _copy_local(self._base)

    def body_scales(self):
        """ShapeBodyInfoFemale head-chain writes from the exact port (parameter-independent)."""
        if self._body is None:
            m = self.m
            ids = {name: m.unique(name) for name in BODY_NAMES}
            tmp = {ids[n]: {k: list(m.transforms[ids[n]][k]) for k in ('local_position', 'local_rotation_xyzw',
                                                                       'local_scale')} for n in BODY_NAMES}
            m._body_update(tmp)
            self._body = {ids[n]: np.asarray(tmp[ids[n]]['local_scale'], np.float32) for n in BODY_NAMES}
        return self._body

    def prefab(self):
        if self._prefab is None:
            pre = {}
            for b in self.m.rig.bones.values():
                pre.setdefault(b['name'], b)
            self._prefab = pre
        return self._prefab

    # ------------------------------------------------------------------ native updates
    def face_update(self, local, native):
        m = self.m
        targets = {name: local[m.unique(name)] for name in DST
                   if any(i in m.needed for i in m.names.get(name, []))}
        NS.face_update(self.A, m.rig, native, targets, mouth_shape_mask=m.identity['mouth_shape_mask'])

    def body_update(self, local):
        for tid, scl in self.body_scales().items():
            local[tid]['scl'] = self.A.f32(scl)

    def card_load_baselines(self, local, modifiers, native):
        m, A = self.m, self.A
        prefab = self.prefab()
        L0 = _copy_local(local)
        for tid in L0:
            name = m.transforms[tid]['name']
            if name in prefab and m.names.get(name) == [tid]:
                b = prefab[name]
                L0[tid] = self._trs(b['pos'], b['rot'], b['scale'])
        self.face_update(L0, native)
        self.body_update(L0)
        for mod in modifiers:
            if mod['name'] in prefab or mod['name'] in BODY_NAMES:
                cache = X.fresh_cache(A)
                X.collect(A, L0[mod['id']], cache)
                mod['cache'] = cache
        written = {}
        for op in UPDATE:
            channel = 'pos' if 'Position' in op['method'] else 'rot' if 'Rotation' in op['method'] else 'scl'
            written.setdefault(DST[op['dst']], set()).add(channel)
        written.setdefault('cf_N_height', set()).add('scl')
        written.setdefault('cf_J_Head_s', set()).add('scl')
        for name, channels in written.items():
            for tid in m.names.get(name, []):
                if tid in local:
                    for channel in channels:
                        local[tid][channel] = L0[tid][channel]

    def _native(self, native):
        if native is None:
            return None
        t = native if isinstance(native, torch.Tensor) else torch.as_tensor(np.asarray(native, np.float64), dtype=DT)
        return t.to(DT)

    # ------------------------------------------------------------------ evaluate
    def evaluate(self, patches=(), *, native=None, apply_count=1, baseline_mode='captured'):
        """Same contract as ``CapturedHeadForward.evaluate`` (``bone_local`` and ``branches``)."""
        A, m = self.A, self.m
        if type(apply_count) != int or not 1 <= apply_count <= 32:
            raise ValueError('Declared Apply count 1..32 required')
        if baseline_mode not in ('captured', 'card_load'):
            raise ValueError('baseline_mode must be captured or card_load')
        native = self._native(native)
        if baseline_mode == 'card_load':
            card_native = native if native is not None else self._native(m.identity['native59'])
            key = card_native.detach().to(torch.float32).numpy().tobytes()
            if card_native.requires_grad or key not in self._card_cache:
                local = self.base_locals()
                modifiers = _copy_modifiers(self.mods0)
                self.card_load_baselines(local, modifiers, card_native)
                if not card_native.requires_grad:
                    self._card_cache[key] = (local, modifiers)
            else:
                local, modifiers = self._card_cache[key]
            local = _copy_local(local)
            modifiers = _copy_modifiers(modifiers)
            native = None
        else:
            local = self.base_locals()
            modifiers = _copy_modifiers(self.mods0)
        if any(not isinstance(p, dict) or set(p) - EXACT_PATCH_FIELDS or 'name' not in p for p in patches):
            raise ValueError('Explicit supported patch fields required')
        if len({p['name'] for p in patches}) != len(patches):
            raise ValueError('Duplicate patch bone names')
        if native is not None:
            self.face_update(local, native)
        removed = False
        for patch in patches:
            match = next((x for x in modifiers if x['name'] == patch['name']), None)
            if patch.get('remove'):
                if match is not None:
                    X.reset(A, local[match['id']], match['cache'])
                    modifiers.remove(match)
                    removed = True
                continue
            if match is None:
                ident = m.unique(patch['name'])
                cache = X.fresh_cache(A)
                X.collect(A, local[ident], cache)
                match = {'id': ident, 'name': patch['name'], 'cache': cache,
                         'coordinate_modifiers': [X.modifier(A, X.IDENTITY)]}
                modifiers.append(match)
            if match['id'] not in local:
                raise ValueError('Requested patch targets a transform outside captured surfaces')
            cm = match['coordinate_modifiers'][0]
            cm = dict(cm) if cm is not None else None
            for field in X.IDENTITY:
                if field in patch:
                    if cm is None:
                        raise ValueError('patch of a null coordinate modifier')
                    cm[field] = X.t32(A, patch[field])
            match['coordinate_modifiers'][0] = cm
        if removed:
            self.body_update(local)
            self.face_update(local, native if native is not None else self._native(m.identity['native59']))
        if native is not None or m.identity['abmx_baseline_refresh_pending']:
            if native is None:
                native = self._native(m.identity['native59'])
            dst = set(m.rig.enums['dst']) | set(BODY_NAMES)
            refreshing = [x for x in modifiers if x['name'] in dst and x['id'] in local]
            for x in refreshing:
                X.reset(A, local[x['id']], x['cache'])
            self.face_update(local, native)
            self.body_update(local)
            for x in refreshing:
                X.collect(A, local[x['id']], x['cache'])
        branches = []
        for step in range(apply_count):
            for name in self.additional:
                if name not in {x['name'] for x in modifiers}:
                    ident = m.unique(name)
                    cache = X.fresh_cache(A)
                    X.collect(A, local[ident], cache)
                    modifiers.append({'id': ident, 'name': name, 'cache': cache,
                                      'coordinate_modifiers': [X.modifier(A, X.IDENTITY)]})
            first_by_name = {}
            for index, x in enumerate(modifiers):
                first_by_name.setdefault(x['name'], index)
            for index, x in enumerate(modifiers):
                if x['id'] not in m.needed:
                    branches.append({'step': step, 'modifier_index': index, 'name': x['name'],
                                     'branches': ['disjoint_from_all_requested_surface_bones_and_ancestors']})
                    continue
                if x['name'].startswith(('cf_J_SiriDam', 'cf_J_Mune00')):
                    X.reset(A, local[x['id']], x['cache'])
                    X.collect(A, local[x['id']], x['cache'])
                add = self.additional.get(x['name'], []) if first_by_name[x['name']] == index else []
                eff = X.effective_modifier(A, x['coordinate_modifiers'], add)
                b = X.apply_memo(A, local[x['id']], x['cache'], eff, self._apply_memo,
                                 rotation_excluded=x['name'] in m.excluded,
                                 is_during_h_scene=m.identity['abmx_during_h_scene'])
                branches.append({'step': step, 'modifier_index': index, 'name': x['name'], 'branches': b})
        return {'bone_local': local, 'branches': branches, 'modifiers': modifiers}


def local_f32_bytes(trs):
    """float32 bytes of (pos, rot, scl) -- the comparison key of the exact chain."""
    return np.concatenate([np.asarray(trs[k].detach().numpy(), np.float32) for k in ('pos', 'rot', 'scl')]).tobytes()


def exact_local_bytes(row):
    return np.concatenate([np.asarray(row[k], np.float32) for k in ('local_position', 'local_rotation_xyzw',
                                                                    'local_scale')]).tobytes()
