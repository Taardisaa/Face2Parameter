"""Installed HS2ABMX 4.4.6 BoneModifier state transitions over torch.

Port of the exact chain's ``tools/abmx_replay/model.py`` (``replay_apply``,
``effective_modifier``) and ``src/hs2_offline_state.py`` (``reset``, ``collect``) with the same
rounding points: binary32 per Vector3 / Quaternion operator (numpy float32 in the exact port),
Mono double evaluation for ``Vector3.SqrMagnitude`` and ``Quaternion.operator*``, the native
Quaternion.Euler and the SetLocalR normalisation of every stored rotation.

State: a local TRS is ``{'pos': (3,), 'rot': (4,), 'scl': (3,)}`` float64 tensors; a cache is
``{'_sclBaseline', '_posBaseline', '_positionBaseline', '_rotBaseline': tensors,
'_lenBaseline': 0-d tensor, flags: bool}``; a modifier is ``{'scale': (3,), 'length': (),
'position': (3,), 'rotation': (3,)}`` tensors (patch values may require grad).

Branches are decided on forward values exactly as installed ABMX does (``IsEmpty`` / ``CanApply``
tests ``!= 1`` / ``!= 0`` per component, Vector3 approximate-zero tests, the length < 0.1
fallback, the rotation exclusion list).  The gradient is the derivative of the selected branch;
it is zero for a channel whose modifier equals its identity value exactly (that channel is then
not applied by the game) -- see ``docs/hs2_diffgeo.md``.
"""
from __future__ import annotations

import numpy as np
import torch

from . import unity as U
from .fp import DT

FLAG_FIELDS = ("_hasBaseline", "_lenModForceUpdate", "_lenModNeedsPositionRestore",
               "_changedScale", "_changedRotation", "_changedPosition", "_forceApply")
IDENTITY = {"scale": [1., 1., 1.], "length": 1., "position": [0., 0., 0.], "rotation": [0., 0., 0.]}
LEN_MIN = float(np.float32(.1))


def t32(A, v):
    """A recorded binary32 value (cache/modifier/patch) as a tensor; tensors are rounded by R."""
    if isinstance(v, torch.Tensor):
        return A.R(v.to(DT))
    return A.f32(v)


def cache_from_record(A, rec):
    out = {k: bool(rec[k]) for k in FLAG_FIELDS}
    for k in ('_sclBaseline', '_posBaseline', '_positionBaseline', '_rotBaseline'):
        out[k] = t32(A, rec[k])
    out['_lenBaseline'] = t32(A, rec['_lenBaseline'])
    return out


def fresh_cache(A):
    """Native BoneModifier constructor defaults (before CollectBaseline)."""
    c = {k: False for k in FLAG_FIELDS}
    c.update({'_sclBaseline': A.f32([1., 1., 1.]), '_posBaseline': A.f32([0., 0., 0.]),
              '_positionBaseline': A.f32([0., 0., 0.]), '_rotBaseline': A.f32([0., 0., 0., 0.]),
              '_lenBaseline': A.f32(0.)})
    return c


def modifier(A, value):
    if value is None:
        return None
    return {'scale': t32(A, value['scale']), 'length': t32(A, value['length']),
            'position': t32(A, value['position']), 'rotation': t32(A, value['rotation'])}


def collect(A, local, cache):
    """CollectBaseline: preserves already established length history."""
    cache['_sclBaseline'] = local['scl']
    cache['_posBaseline'] = local['pos']
    cache['_rotBaseline'] = local['rot']
    if U.unity_zero(A, cache['_positionBaseline']):
        cache['_positionBaseline'] = local['pos']
        cache['_lenBaseline'] = A.R(torch.sqrt(U.sqrmag(A, local['pos'])))
        cache['_lenModNeedsPositionRestore'] = False
    cache['_hasBaseline'] = True


def reset(A, local, cache):
    """BoneModifier.Reset (flags not silently cleared)."""
    if not cache['_hasBaseline']:
        return
    local['scl'] = cache['_sclBaseline']
    local['rot'] = U.normalize_quaternion(A, cache['_rotBaseline'])
    position = cache['_posBaseline']
    if not U.unity_zero(A, cache['_positionBaseline']):
        if cache['_lenModNeedsPositionRestore'] or U.unity_zero(A, position):
            position = cache['_positionBaseline']
            cache['_lenModNeedsPositionRestore'] = False
        else:
            position = A.R(A.R(position / A.R(torch.sqrt(U.sqrmag(A, position)))) * cache['_lenBaseline'])
    local['pos'] = position


def effective_modifier(A, coordinate_modifiers, additional_modifiers):
    """Installed HS2 IsCoordinateSpecific is false: modifier 0, then additional effect products."""
    if not coordinate_modifiers:
        raise ValueError('Missing coordinate modifiers')
    base = coordinate_modifiers[0]
    if not additional_modifiers:
        return base
    base = dict(base) if base is not None else modifier(A, IDENTITY)
    for add in additional_modifiers:
        base = {'scale': A.R(base['scale'] * add['scale']), 'length': A.R(base['length'] * add['length']),
                'position': A.R(base['position'] + add['position']),
                'rotation': A.R(base['rotation'] + add['rotation'])}
    return base


def _any_ne(v, x):
    """IsEmpty-style component test on the binary32 value (both arithmetic modes)."""
    return bool(np.any(np.float32(v.detach().cpu().numpy()) != np.float32(x)))


MEMO_LIMIT = 50000
_TENSOR_CACHE = ('_sclBaseline', '_posBaseline', '_positionBaseline', '_rotBaseline', '_lenBaseline')


def apply_memo(A, local, cache, effective, memo, **kw):
    """``apply`` with a content-addressed memo: when no input carries a gradient the transition is
    a pure function of the input bits (TRS, private cache, flags, effective modifier, options),
    so a repeated transition returns the stored outputs (bit-identical by construction)."""
    ts = [local[k] for k in ('pos', 'rot', 'scl')] + [cache[k] for k in _TENSOR_CACHE]
    if effective is not None:
        ts += [effective[k] for k in ('scale', 'length', 'position', 'rotation')]
    if memo is None or any(t.requires_grad for t in ts):
        return apply(A, local, cache, effective, **kw)
    key = (A.mode, A.one_sided, effective is None, tuple(cache[k] for k in FLAG_FIELDS), tuple(sorted(kw.items())),
           b''.join(t.detach().numpy().tobytes() for t in ts))
    hit = memo.get(key)
    if hit is None:
        if len(memo) > MEMO_LIMIT:
            memo.clear()
        dec = apply(A, local, cache, effective, **kw)
        memo[key] = ({k: local[k] for k in ('pos', 'rot', 'scl')}, dict(cache), list(dec))
        return dec
    out_local, out_cache, dec = hit
    local.update(out_local)
    cache.update(out_cache)
    return list(dec)


def apply(A, local, cache, effective, *, rotation_excluded, is_during_h_scene):
    """BoneModifier.Apply on (local, cache) in place; returns the branch list."""
    decisions = []
    length_result = None
    if effective is None:
        return ['return_null_modifier']
    has_scale = _any_ne(effective['scale'], 1.0)
    has_length = _any_ne(effective['length'], 1.0)
    has_position = _any_ne(effective['position'], 0.0)
    has_rotation = _any_ne(effective['rotation'], 0.0)
    nonempty = has_scale or has_length or has_position or has_rotation
    if nonempty:
        cache['_forceApply'] = True
        decisions.append('CanApply_nonempty')
    elif cache['_forceApply']:
        cache['_forceApply'] = False
        cache['_lenModForceUpdate'] = True
        decisions.append('CanApply_force_to_length_restore')
    elif cache['_lenModForceUpdate']:
        decisions.append('CanApply_pending_length_restore')
    else:
        decisions = ['return_CanApply_false']
        if A.one_sided:
            _one_sided(A, local, cache, effective, decisions, flags=(has_scale, has_length, has_position, has_rotation),
                       rotation_excluded=rotation_excluded, is_during_h_scene=is_during_h_scene, length_result=None)
        return decisions
    if has_scale:
        local['scl'] = A.R(cache['_sclBaseline'] * effective['scale'])
        cache['_changedScale'] = True
        decisions.append('scale_from_cached_baseline')
    elif cache['_changedScale']:
        local['scl'] = cache['_sclBaseline']
        cache['_changedScale'] = False
        decisions.append('restore_cached_scale')
    if has_rotation and not rotation_excluded:
        q = U.managed_quaternion_mul(A, cache['_rotBaseline'], U.quaternion_euler_degrees(A, effective['rotation']))
        local['rot'] = U.normalize_quaternion(A, q)
        if not float(torch.linalg.norm(local['rot'].detach())) > 0:
            raise ValueError('Cannot certify Unity setter of zero quaternion')
        cache['_changedRotation'] = True
        decisions.append('rotation_baseline_times_euler_zxy')
    elif cache['_changedRotation']:
        local['rot'] = U.normalize_quaternion(A, cache['_rotBaseline'])
        cache['_changedRotation'] = False
        decisions.append('restore_cached_rotation_including_excluded_bone')
    if (cache['_lenModForceUpdate'] or has_length) and not U.unity_zero(A, cache['_positionBaseline']):
        direction = local['pos']
        if np.float32(effective['length'].detach().cpu().numpy()) < np.float32(LEN_MIN) or U.unity_zero(A, direction) or is_during_h_scene:
            direction = cache['_positionBaseline']
            cache['_lenModNeedsPositionRestore'] = True
            decisions.append('length_use_historical_direction')
        else:
            decisions.append('length_use_current_direction')
        magnitude = A.R(torch.sqrt(U.sqrmag(A, direction)))
        if not (bool(torch.isfinite(magnitude)) and float(magnitude.detach()) > 0):
            raise ValueError('Cannot replay zero/overflow normalization divisor')
        local['pos'] = A.R(A.R(A.R(direction / magnitude) * cache['_lenBaseline']) * effective['length'])
        length_result = local['pos']
        cache['_lenModForceUpdate'] = False
        if has_position:
            local['pos'] = A.R(local['pos'] + effective['position'])
            cache['_changedPosition'] = True
            decisions.append('length_then_add_position')
        elif cache['_changedPosition']:
            local['pos'] = cache['_posBaseline']
            cache['_changedPosition'] = False
            decisions.append('length_result_overwritten_by_position_restore')
    elif has_position:
        local['pos'] = A.R(cache['_posBaseline'] + effective['position'])
        cache['_changedPosition'] = True
        decisions.append('position_only_cached_baseline_plus_offset')
    elif cache['_changedPosition']:
        local['pos'] = cache['_posBaseline']
        cache['_changedPosition'] = False
        decisions.append('restore_cached_position')
    else:
        decisions.append('position_unchanged')
    if A.one_sided:
        _one_sided(A, local, cache, effective, decisions, flags=(has_scale, has_length, has_position, has_rotation),
                   rotation_excluded=rotation_excluded, is_during_h_scene=is_during_h_scene,
                   length_result=length_result)
    return decisions


def _continuous(taken, alt):
    """Branch values agree up to binary32 rounding (a few ulp of the larger magnitude)."""
    t, a = taken.detach(), alt.detach()
    tol = 4.0 * 2.0 ** -23 * torch.clamp(torch.maximum(t.abs(), a.abs()), min=1.0)
    return bool(((t - a).abs() <= tol).all())


def _one_sided(A, local, cache, effective, decisions, *, flags, rotation_excluded, is_during_h_scene, length_result):
    """Opt-in gradient for channels whose modifier equals its identity value exactly (``Arith(...,
    identity_gradient='one_sided')``).  The game does not apply such a channel (IsEmpty), so the
    forward keeps the taken branch's value bit for bit; when the applying branch evaluated at the
    same point gives the same value up to binary32 rounding (the function is continuous there and
    both one-sided neighbourhoods use the applying branch), the backward uses the applying
    branch's derivative.  A discontinuous channel (e.g. a length of exactly 1 with a position
    offset: the length branch renormalises the current position) keeps the zero derivative and
    is labelled.  Does not change any forward value or cache flag."""
    has_scale, has_length, has_position, has_rotation = flags
    from .fp import forward_from

    def pick(name, taken, alt):
        if _continuous(taken, alt):
            decisions.append(name + '_identity_one_sided_gradient')
            return forward_from(taken, alt)
        decisions.append(name + '_identity_discontinuous_zero_gradient')
        return taken
    if not has_scale and effective['scale'].requires_grad:
        local['scl'] = pick('scale', local['scl'], A.R(cache['_sclBaseline'] * effective['scale']))
    if not has_rotation and not rotation_excluded and effective['rotation'].requires_grad:
        q = U.managed_quaternion_mul(A, cache['_rotBaseline'], U.quaternion_euler_degrees(A, effective['rotation']))
        local['rot'] = pick('rotation', local['rot'], U.normalize_quaternion(A, q))
    if not has_position and effective['position'].requires_grad:
        base = length_result if length_result is not None else cache['_posBaseline']
        local['pos'] = pick('position', local['pos'], A.R(base + effective['position']))
    if (not has_length and effective['length'].requires_grad and length_result is None
            and not U.unity_zero(A, cache['_positionBaseline'])):
        direction = local['pos']
        if U.unity_zero(A, direction) or is_during_h_scene:
            direction = cache['_positionBaseline']
        magnitude = A.R(torch.sqrt(U.sqrmag(A, direction)))
        alt = A.R(A.R(A.R(direction / magnitude) * cache['_lenBaseline']) * effective['length'])
        if has_position:
            alt = A.R(alt + effective['position'])
        local['pos'] = pick('length', local['pos'], alt)
