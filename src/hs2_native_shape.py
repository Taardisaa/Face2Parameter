"""ShapeHeadInfoFemale native face-shape evaluation, float32-exact.

Ports the installed Assembly-CSharp ShapeInfoBase.ChangeValue,
AnimationKeyInfo.GetInfo (+ SliderUnlocker range clamp) and the generated
ShapeHeadInfoFemale.Update program, with Mono double-evaluation semantics
(src.render.unity_transform_math) and the native setter chain
(Quaternion.Euler -> EulerToQuaternion, Transform SetLocalR normalization).
"""
from __future__ import annotations

import numpy as np

from src.render import unity_transform_math as um
from src.render.shape_head_female_program import DST, SRC, UPDATE

F = np.float32
SRC_ID = {n: i for i, n in enumerate(SRC)}


class ShapeInfoPort:
    """ShapeInfoBase.ChangeValue + AnimationKeyInfo.GetInfo over one key/category table.

    keys: {src name: [{'pos','rot','scl'}...]} (asset list order);
    rows: [{'category', 'bone', 'use'[9]}] in the category TextAsset's line order.
    dictSrc is keyed by src name (bijective with the SrcName enum id).
    """

    def __init__(self, keys, rows):
        self.keys = {name: [(np.asarray(f['pos'], F), np.asarray(f['rot'], F), np.asarray(f['scl'], F)) for f in frames]
                     for name, frames in keys.items()}
        self.categories = {}
        for row in rows:
            use = [bool(u) for u in row['use']]
            self.categories.setdefault(int(row['category']), []).append(
                {'name': row['bone'], 'id': row['bone'], 'use': [use[0:3], use[3:6], use[6:9]],
                 'getflag': [any(use[0:3]), any(use[3:6]), any(use[6:9])]})
        # GetSrcBoneInfo: one BoneInfo per referenced src id (pos 0, rot 0, scl 1).
        self.src = {c['id']: {'pos': np.zeros(3, F), 'rot': np.zeros(3, F), 'scl': np.ones(3, F)}
                    for rows in self.categories.values() for c in rows}

    def get_info(self, name, rate, flags):
        rate = F(rate)
        state = rate
        if rate > 1:                       # SliderUnlocker GetInfoPreHook clamp
            rate = F(1)
        if rate < 0:
            rate = F(0)
        keys = self.keys[name]
        out = [np.zeros(3, F), np.zeros(3, F), np.zeros(3, F)]
        for k in range(3):
            if not flags[k]:
                continue
            if rate == 0:
                out[k] = keys[0][k].copy()
            elif rate == 1:
                out[k] = keys[-1][k].copy()
            else:
                # Mono keeps the float local `num` in a double register; FloorToInt
                # receives it as a float32 argument; `t` is rounded when passed on.
                num = float(F(len(keys) - 1)) * float(rate)
                i = um.floor_to_int(F(num))
                t = F(num - float(F(i)))
                if k == 1:
                    out[k] = np.array([um.lerp_angle(keys[i][1][c], keys[i + 1][1][c], t) for c in range(3)], F)
                else:
                    out[k] = um.vector3_lerp(keys[i][k], keys[i + 1][k], t)
        if state < 0 or state > 1:
            raise NotImplementedError('SliderUnlocker out-of-range extrapolation (SliderMath) not yet ported')
        return out

    def change_value(self, category, value):
        for c in self.categories.get(category, []):
            info = self.src[c['id']]
            got = self.get_info(c['name'], value, c['getflag'])
            for k, field in enumerate(('pos', 'rot', 'scl')):
                for axis in range(3):
                    if c['use'][k][axis]:
                        info[field][axis] = got[k][axis]

    def _arg(self, a):
        if 'const' in a:
            return F(a['const'])
        total = 0.0
        for i, (sid, field, axis) in enumerate(a['sum']):
            v = float(self.src[SRC[sid]][field][axis])
            total = v if i == 0 else total + v     # left-to-right C# sum in double
        return F(total)

    def update_head(self, locals_by_name):
        """Apply ShapeHeadInfoFemale.Update to {dst name: local dict} in place."""
        from src.render.unity_skin_emu import shared_emulator, quaternion_euler_degrees
        emu = shared_emulator()
        for op in UPDATE:
            name = DST[op['dst']]
            if name not in locals_by_name:
                continue
            loc = locals_by_name[name]
            args = [self._arg(a) for a in op['args']]
            m = op['method']
            if m.startswith('SetLocalPosition') and len(m) == len('SetLocalPositionX'):
                p = np.asarray(loc['local_position'], F).copy()
                p['XYZ'.index(m[-1])] = args[0]
                loc['local_position'] = p
            elif m == 'SetLocalPosition':
                loc['local_position'] = np.array(args, F)
            elif m == 'SetLocalRotation':
                loc['local_rotation_xyzw'] = um.normalize_quaternion(quaternion_euler_degrees(emu, np.array(args, F)))
            elif m == 'SetLocalScale':
                loc['local_scale'] = np.array(args, F)
            else:
                raise NotImplementedError(m)
        return locals_by_name


class NativeShapeHead(ShapeInfoPort):
    def __init__(self, rig):
        super().__init__(rig.anm, rig.customhead)

    def update(self, locals_by_name):
        return self.update_head(locals_by_name)


# ShapeBodyInfoFemale.Update (updateMask & 0x10) writes reaching head skin chains:
#   dictDst[cf_N_height].SetLocalScale(src[cf_s_height].vctScl.x, .y, .z)
#   dictDst[cf_J_Head_s].SetLocalScale(src[cf_s_Head_h].vctScl.c * src[cf_s_Head_s].vctScl.c ...)
# All other body writes are disjoint from every head surface's bones and ancestors
# (checked against the paired probe hierarchy).
class NativeShapeBodyHeadChain(ShapeInfoPort):
    def update(self, locals_by_name, update_mask=31):
        if not update_mask & 0x10:
            return locals_by_name
        if 'cf_N_height' in locals_by_name:
            locals_by_name['cf_N_height']['local_scale'] = self.src['cf_s_height']['scl'].astype(F).copy()
        if 'cf_J_Head_s' in locals_by_name:
            h, s_ = self.src['cf_s_Head_h']['scl'], self.src['cf_s_Head_s']['scl']
            locals_by_name['cf_J_Head_s']['local_scale'] = np.array([float(h[c]) * float(s_[c]) for c in range(3)]).astype(F)
        return locals_by_name


def body_tables():
    """cf_anmShapeBody keys and cf_custombody rows from the installed customshape bundle."""
    import UnityPy
    from scripts.hs2_extract_head import CUSTOMSHAPE, parse_anm_shape
    rows = []
    for reader in UnityPy.load(CUSTOMSHAPE).objects:
        if reader.type.name != 'TextAsset':
            continue
        asset = reader.read_typetree()
        if asset['m_Name'] != 'cf_custombody':
            continue
        text = asset['m_Script']
        if not isinstance(text, str):
            text = bytes(text).decode('utf-8')
        for line in text.splitlines():
            if line.strip():
                f = line.split('	')
                rows.append({'category': int(f[0]), 'bone': f[1], 'use': [v != '0' for v in f[2:11]]})
    return parse_anm_shape(CUSTOMSHAPE, 'cf_anmShapeBody'), rows
