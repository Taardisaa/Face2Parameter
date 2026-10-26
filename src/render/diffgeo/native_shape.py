"""ShapeHeadInfoFemale native face shape over torch (port of ``src.hs2_native_shape``).

Line-by-line port of the exact chain's ``ShapeInfoPort`` (installed Assembly-CSharp
``ShapeInfoBase.ChangeValue`` + ``AnimationKeyInfo.GetInfo`` with the SliderUnlocker range
clamp, and the generated ``ShapeHeadInfoFemale.Update`` program
``src/render/shape_head_female_program.py``) with the same Mono rounding points:

* GetInfo: the float local ``num = (count - 1) * rate`` stays double; ``FloorToInt`` receives it
  as a binary32 argument; ``t`` is rounded when passed on; ``Vector3.Lerp`` / ``Mathf.LerpAngle``
  evaluate in double and round once per stored component.
* Update arguments: left-to-right double sums of the src fields, rounded once (float argument).
* SetLocalRotation: native Quaternion.Euler (``unity.quaternion_euler_degrees``) and the
  SetLocalR normalisation.

Discrete branches keep their game semantics (gradient piecewise, see docs/hs2_diffgeo.md):
``rate == 0`` / ``rate == 1`` return the first / last key (zero derivative at exactly those
values); the key segment ``i = FloorToInt(num)`` is selected from the forward value (the
derivative is the segment's, one-sided at segment boundaries); ``Clamp01(t)`` clamps.  The
SliderUnlocker out-of-range extrapolation (``SliderMath``) is not ported in the exact chain
either; a rate outside [0, 1] raises.
"""
from __future__ import annotations

import numpy as np
import torch

from src.render import unity_transform_math as um
from src.render.shape_head_female_program import DST, SRC, UPDATE

from . import unity as U
from .fp import DT

F = np.float32


def _clamp01(t):
    return torch.where(t < 0, torch.zeros_like(t), torch.where(t > 1, torch.ones_like(t), t))


class TorchShapeInfo:
    """One ShapeInfoBase over a key table and category rows (same inputs as ShapeInfoPort)."""

    def __init__(self, A, keys, rows):
        self.A = A
        self.keys = {name: [(np.asarray(f['pos'], F), np.asarray(f['rot'], F), np.asarray(f['scl'], F)) for f in frames]
                     for name, frames in keys.items()}
        self.categories = {}
        for row in rows:
            use = [bool(u) for u in row['use']]
            self.categories.setdefault(int(row['category']), []).append(
                {'name': row['bone'], 'id': row['bone'], 'use': [use[0:3], use[3:6], use[6:9]],
                 'getflag': [any(use[0:3]), any(use[3:6]), any(use[6:9])]})
        # lerp_angle's wrapped key difference depends on the keys only (exact numpy port)
        self._angle_num = {}

    def fresh_src(self):
        z = lambda v: [torch.tensor(float(v), dtype=DT) for _ in range(3)]
        return {c['id']: {'pos': z(0.0), 'rot': z(0.0), 'scl': z(1.0)}
                for rows in self.categories.values() for c in rows}

    def _num(self, name, i, c):
        key = (name, i, c)
        if key not in self._angle_num:
            a, b = self.keys[name][i][1][c], self.keys[name][i + 1][1][c]
            num = float(um.repeat(F(float(F(b)) - float(F(a))), 360.0))
            if num > 180.0:
                num = num - 360.0
            self._angle_num[key] = num
        return self._angle_num[key]

    def get_info(self, name, rate, flags):
        A = self.A
        rate = A.R(rate)
        r = float(np.float32(rate.detach()))
        if r > 1 or r < 0:
            raise NotImplementedError('SliderUnlocker out-of-range extrapolation (SliderMath) not yet ported')
        keys = self.keys[name]
        out = [None, None, None]
        for k in range(3):
            if not flags[k]:
                continue
            if r == 0:
                out[k] = [torch.tensor(float(v), dtype=DT) for v in keys[0][k]]
            elif r == 1:
                out[k] = [torch.tensor(float(v), dtype=DT) for v in keys[-1][k]]
            else:
                num = float(F(len(keys) - 1)) * rate                      # double register local
                i = um.floor_to_int(F(float(num.detach())))               # FloorToInt(float arg)
                t = A.R(num - float(F(i)))
                if i + 1 >= len(keys):
                    raise IndexError('GetInfo key segment past the table end')
                ct = _clamp01(t)
                if k == 1:
                    out[k] = [A.R(float(keys[i][1][c]) + self._num(name, i, c) * ct) for c in range(3)]
                else:
                    a, b = keys[i][k], keys[i + 1][k]
                    out[k] = [A.R(float(a[c]) + (float(b[c]) - float(a[c])) * ct) for c in range(3)]
        return out

    def change_value(self, src, category, value):
        for c in self.categories.get(category, []):
            info = src[c['id']]
            got = self.get_info(c['name'], value, c['getflag'])
            for k, field in enumerate(('pos', 'rot', 'scl')):
                for axis in range(3):
                    if c['use'][k][axis]:
                        info[field][axis] = got[k][axis]

    def arg(self, src, a):
        if 'const' in a:
            return torch.tensor(float(F(a['const'])), dtype=DT)
        total = None
        for i, (sid, field, axis) in enumerate(a['sum']):
            v = src[SRC[sid]][field][axis]
            total = v if i == 0 else total + v
        return self.A.R(total)

    def update_head(self, src, locals_by_name):
        """ShapeHeadInfoFemale.Update on {dst name: TRS dict of tensors}, in place."""
        A = self.A
        for op in UPDATE:
            name = DST[op['dst']]
            if name not in locals_by_name:
                continue
            loc = locals_by_name[name]
            args = [self.arg(src, a) for a in op['args']]
            m = op['method']
            if m.startswith('SetLocalPosition') and len(m) == len('SetLocalPositionX'):
                axis = 'XYZ'.index(m[-1])
                p = loc['pos']
                loc['pos'] = torch.stack([args[0] if j == axis else p[j] for j in range(3)])
            elif m == 'SetLocalPosition':
                loc['pos'] = torch.stack(args)
            elif m == 'SetLocalRotation':
                loc['rot'] = U.normalize_quaternion(A, U.quaternion_euler_degrees(A, torch.stack(args)))
            elif m == 'SetLocalScale':
                loc['scl'] = torch.stack(args)
            else:
                raise NotImplementedError(m)
        return locals_by_name


def face_update(A, rig, native, locals_by_name, mouth_shape_mask=False):
    """Installed ShapeHeadInfoFemale ChangeValue(0..58) + Update (CapturedHeadForward._face_update)."""
    values = native
    if values.shape != (59,) or not torch.isfinite(values).all():
        raise ValueError('All 59 finite native values required')
    if mouth_shape_mask:
        # ChaFileDefine.cf_MouthShapeMaskID with cf_MouthShapeDefault (all 0.5f): constant
        mask = torch.zeros(59, dtype=torch.bool)
        mask[[3, 6, 11, 47, 48, 49, 50, 51, 52, 53]] = True
        values = torch.where(mask, torch.full_like(values, 0.5), values)
    shape = TorchShapeInfo(A, rig.anm, rig.customhead)
    src = shape.fresh_src()
    for category in range(59):
        shape.change_value(src, category, values[category])
    shape.update_head(src, locals_by_name)
    return locals_by_name
