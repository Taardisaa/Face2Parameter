"""Per-frame driver acceptance seeded from the probe v6 driver-hook pre-state.

For every hooked frame of a v6 window:
  * eye: the installed EyeLookController.LateUpdate runs on the hosted game JIT
    from the eye *prefix* record (EyeObject angleH/angleV/dirUp, ptnNo, target
    position, deltaTime) and that frame's exported character transforms. All
    EyeLookCalc/EyeObject fields are compared bitwise with the *postfix* record,
    and cf_J_look_L/R with the end-of-frame export.
  * fbs: the installed FaceBlendShape.OnLateUpdate runs from the fbs prefix
    record (blink control, three controllers incl. blend timers, mouth random
    timer/scale state, voiceValue, UnityEngine.Random.state). Every field, the
    postfix Random.state and the end-of-frame weights of the head renderers
    are compared bitwise.
No seeds, no reconstruction: the hidden state is the game's own recorded state.
"""
from __future__ import annotations

import argparse
import base64
import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eye_look_acceptance import Harness, LOOK  # noqa: E402
from hosted_face_blendshape import HostedFaceBlendShape, load_fbs_asset  # noqa: E402

F = np.float32


def f32(v):
    return np.asarray(v, F)


def load(window: Path):
    static = json.loads((window/'static.json').read_text(encoding='utf-8'))
    frames, hooks = {}, {}
    for p in sorted((window/'frames').glob('frame_*.json')):
        row = json.loads(p.read_text(encoding='utf-8'))
        row['T'] = np.frombuffer(base64.b64decode(row['transforms_f32']), F).reshape(-1, 33)
        row['W'] = {r['id']: np.frombuffer(base64.b64decode(r['weights_f32']), F) for r in row['renderers']}
        frames[row['frame_count']] = row
    for p in sorted((window/'frames').glob('drivers_*.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        for r in d['records']:
            if r.get('instance_in_window_character') is not True:
                continue                       # other actors' instances (none expected in the Maker)
            key = (r['frame_count'], r['hook'], r['phase'])
            if key in hooks:
                raise RuntimeError(f'two window-character records for {key}')
            hooks[key] = r
    return static, frames, hooks


def same(a, b):
    a, b = f32(a), f32(b)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# ------------------------------------------------------------------ eye
def eye_state(h):
    rt, calc = h.rt, h.el.calc
    arr = lambda name, size, n: np.frombuffer(rt.array_bytes(rt.get_ref(calc, name), size), F).reshape(n, -1)  # noqa: E731
    out = {'EyeLookCalc.fixAngle': arr('fixAngle', 16, 2), 'EyeLookCalc.angleHRate': arr('angleHRate', 4, 2).reshape(-1),
           'EyeLookCalc.angleVRate': rt.floats(calc, 'angleVRate', 1)[0], 'EyeLookCalc.targetPos': rt.floats(calc, 'targetPos', 3),
           'EyeLookCalc.nowPtnNo': int(rt.get(calc, 'nowPtnNo', C.c_int32).value)}
    for i, s in enumerate(h.el.state()):
        for k in ('angleH', 'angleV', 'dirUp', 'referenceLookDir', 'referenceUpDir', 'origRotation'):
            out[f'eyeObjs[{i}].{k}'] = s[k]
    return out


def eye_record_state(rec):
    f = rec['eye_look_calc_fields']
    out = {k: f[k] for k in ('EyeLookCalc.fixAngle', 'EyeLookCalc.angleHRate', 'EyeLookCalc.angleVRate', 'EyeLookCalc.targetPos',
                             'EyeLookCalc.nowPtnNo')}
    for i, e in enumerate(f['EyeLookCalc.eyeObjs']):
        for k in ('angleH', 'angleV', 'dirUp', 'referenceLookDir', 'referenceUpDir', 'origRotation'):
            out[f'eyeObjs[{i}].{k}'] = e['EyeObject.'+k]
    return out


def run_eye(h, frames, hooks):
    rows = []
    for fc in sorted(frames):
        pre, post = hooks.get((fc, 'eye', 'prefix')), hooks.get((fc, 'eye', 'postfix'))
        if pre is None:
            continue
        fr = frames[fc]
        h.apply_frame(fr['T'])
        world_bad = h.world_check(fr['T'], ['cf_J_FaceUp_tz', 'cf_J_EyePos_rz_L', 'cf_J_EyePos_rz_R', 'chaF_001'])
        eyes = pre['eye_look_calc_fields']['EyeLookCalc.eyeObjs']
        h.el.set_state([{'angleH': F(e['EyeObject.angleH']), 'angleV': F(e['EyeObject.angleV']), 'dirUp': f32(e['EyeObject.dirUp'])}
                        for e in eyes])
        before = eye_state(h)
        pre_state = eye_record_state(pre)
        seeded = {k: same(before[k], pre_state[k]) for k in before if not k.startswith('EyeLookCalc.')}
        h.rt.set(h.el.ctrl, 'ptnNo', np.array([pre['call_arguments']['ptnNo']], np.int32))
        h.rt.time.update(deltaTime=float(F(pre['delta_time'])), time=float(F(pre['time'])), frameCount=fc)
        h.camera.pos = f32(pre['call_arguments']['target_position'])
        h.el.late_update()
        after, expect = eye_state(h), eye_record_state(post)
        fields = {k: same(after[k], expect[k]) for k in expect}
        bones = {}
        for n in LOOK:
            exp = fr['T'][h.ids.index(h.names[n].index), 3:7]
            bones[n] = {'bit_equal': h.names[n].rot.tobytes() == exp.tobytes(), 'got': h.names[n].rot.tolist(), 'export': exp.tolist()}
        rows.append({'frame': fc, 'delta_time': pre['delta_time'], 'world_mismatch': world_bad, 'seed_fields_equal_prefix': seeded,
                     'postfix_fields_bit_equal': fields, 'all_postfix_fields_equal': all(fields.values()), 'bones': bones,
                     'mismatched': {k: [np.asarray(after[k]).tolist(), np.asarray(expect[k]).tolist()] for k, ok in fields.items() if not ok}})
    return rows


# ------------------------------------------------------------------ fbs
BASE_FIELDS = (('openRate', F), ('OpenMin', F), ('OpenMax', F), ('FixedRate', F), ('correctOpenMax', F))
BLINK_FIELDS = (('fixedFlags', np.uint8), ('BlinkFrequency', np.uint8), ('blinkMode', np.int8), ('BaseSpeed', F),
                ('calcSpeed', F), ('blinkTime', F), ('count', np.int32), ('openRate', F))
TIMER_FIELDS = (('count', F), ('rate', F), ('progressTime', F))
MOUTH_FIELDS = (('useAjustWidthScale', np.uint8), ('randTimeMin', F), ('randTimeMax', F), ('randScaleMin', F),
                ('randScaleMax', F), ('openRefValue', F), ('sclNow', F), ('sclStart', F), ('sclEnd', F), ('adjustWidthScale', F))


class FbsState:
    """Read/write the recorded FBS field set on the hosted objects."""

    def __init__(self, hf: HostedFaceBlendShape):
        self.hf, self.rt = hf, hf.rt

    def _obj(self, path):
        rt, fbs = self.rt, self.hf.fbs
        if path == 'blink':
            return self.hf.blink
        ctrl = self.hf.ctrls[path][0]
        return ctrl

    def write(self, rec):
        rt, f = self.rt, rec['face_blend_shape_fields']
        b = f['FaceBlendShape.BlinkCtrl']
        for name, t in BLINK_FIELDS:
            rt.set(self.hf.blink, name, np.array([b['FBSBlinkControl.'+name]], t))
        for ctrl_name, cls in (('EyebrowCtrl', 'FBSCtrlEyebrow'), ('EyesCtrl', 'FBSCtrlEyes'), ('MouthCtrl', 'FBSCtrlMouth')):
            c, obj = f['FaceBlendShape.'+ctrl_name], self.hf.ctrls[ctrl_name][0]
            for name, t in BASE_FIELDS:
                rt.set(obj, name, np.array([c['FBSBase.'+name]], t))
            timer = rt.get_ref(obj, 'blendTimeCtrl')
            for name, t in TIMER_FIELDS:
                rt.set(timer, name, np.array([c['FBSBase.blendTimeCtrl']['TimeProgressCtrl.'+name]], t))
            if ctrl_name == 'EyebrowCtrl':
                rt.set(obj, 'SyncBlink', np.array([c['FBSCtrlEyebrow.SyncBlink']], np.uint8))
            if ctrl_name == 'MouthCtrl':
                for name, t in MOUTH_FIELDS:
                    rt.set(obj, name, np.array([c['FBSCtrlMouth.'+name]], t))
                tpc = rt.get_ref(obj, 'tpcRand')
                rec_t = c['FBSCtrlMouth.tpcRand']
                for name, t in TIMER_FIELDS:
                    rt.set(tpc, name, np.array([rec_t['TimeProgressCtrl.'+name]], t))
                for name in ('minTime', 'maxTime'):
                    rt.set(tpc, name, np.array([rec_t['TimeProgressCtrlRandom.'+name]], F))
        rt.set(self.hf.fbs, 'voiceValue', np.array([f['FaceBlendShape.voiceValue']], F))
        for name in ('EyeLookUpCorrect', 'EyeLookDownCorrect', 'EyeLookSideCorrect'):
            rt.set(self.hf.fbs, name, np.array([f['FaceBlendShape.'+name]], F))
        s = rec['random_state']
        self.rt.random_state = [np.uint32(s['State.s%d' % i] & 0xffffffff) for i in range(4)]

    def read(self):
        rt, out = self.rt, {}
        for name, t in BLINK_FIELDS:
            out['blink.'+name] = np.asarray(self._get(self.hf.blink, name, t))
        for ctrl_name in ('EyebrowCtrl', 'EyesCtrl', 'MouthCtrl'):
            obj = self.hf.ctrls[ctrl_name][0]
            for name, t in BASE_FIELDS:
                out[f'{ctrl_name}.{name}'] = self._get(obj, name, t)
            timer = rt.get_ref(obj, 'blendTimeCtrl')
            for name, t in TIMER_FIELDS:
                out[f'{ctrl_name}.blendTimeCtrl.{name}'] = self._get(timer, name, t)
            if ctrl_name == 'MouthCtrl':
                for name, t in MOUTH_FIELDS:
                    out[f'MouthCtrl.{name}'] = self._get(obj, name, t)
                tpc = rt.get_ref(obj, 'tpcRand')
                for name, t in TIMER_FIELDS + (('minTime', F), ('maxTime', F)):
                    out[f'MouthCtrl.tpcRand.{name}'] = self._get(tpc, name, t)
        out['voiceValue'] = self._get(self.hf.fbs, 'voiceValue', F)
        out['random_state'] = self.rt.random_state.astype(np.int64)
        return out

    def _get(self, obj, name, t):
        ct = {F: C.c_float, np.uint8: C.c_uint8, np.int8: C.c_int8, np.int32: C.c_int32}[t]
        return np.asarray(self.rt.get(obj, name, ct).value, t)

    @staticmethod
    def from_record(rec):
        f, out = rec['face_blend_shape_fields'], {}
        b = f['FaceBlendShape.BlinkCtrl']
        for name, t in BLINK_FIELDS:
            out['blink.'+name] = np.asarray(b['FBSBlinkControl.'+name], t)
        for ctrl_name in ('EyebrowCtrl', 'EyesCtrl', 'MouthCtrl'):
            c = f['FaceBlendShape.'+ctrl_name]
            for name, t in BASE_FIELDS:
                out[f'{ctrl_name}.{name}'] = np.asarray(c['FBSBase.'+name], t)
            for name, t in TIMER_FIELDS:
                out[f'{ctrl_name}.blendTimeCtrl.{name}'] = np.asarray(c['FBSBase.blendTimeCtrl']['TimeProgressCtrl.'+name], t)
            if ctrl_name == 'MouthCtrl':
                for name, t in MOUTH_FIELDS:
                    out[f'MouthCtrl.{name}'] = np.asarray(c['FBSCtrlMouth.'+name], t)
                for name, t in TIMER_FIELDS:
                    out[f'MouthCtrl.tpcRand.{name}'] = np.asarray(c['FBSCtrlMouth.tpcRand']['TimeProgressCtrl.'+name], t)
                for name in ('minTime', 'maxTime'):
                    out[f'MouthCtrl.tpcRand.{name}'] = np.asarray(c['FBSCtrlMouth.tpcRand']['TimeProgressCtrlRandom.'+name], F)
        out['voiceValue'] = np.asarray(f['FaceBlendShape.voiceValue'], F)
        s = rec['random_state']
        out['random_state'] = np.asarray([s['State.s%d' % i] & 0xffffffff for i in range(4)], np.int64)
        return out

    @staticmethod
    def dicts_ok(rec):
        f = rec['face_blend_shape_fields']
        return all(f['FaceBlendShape.'+c]['FBSBase.'+d] == {'0': 1} for c in ('EyebrowCtrl', 'EyesCtrl', 'MouthCtrl')
                   for d in ('dictBackFace', 'dictNowFace'))


def run_fbs(hf, static, frames, hooks):
    rt = hf.rt
    io = FbsState(hf)
    tf = static['transforms']

    def under_head(i):
        while i >= 0:
            if tf[i]['name'] == 'ct_head':
                return True
            i = tf[i]['parent']
        return False
    rid = {r['name']: r['id'] for r in static['renderers'] if under_head(r['transform'])}
    rows = []
    for fc in sorted(frames):
        pre, post = hooks.get((fc, 'fbs', 'prefix')), hooks.get((fc, 'fbs', 'postfix'))
        if pre is None:
            continue
        if not (FbsState.dicts_ok(pre) and FbsState.dicts_ok(post)):
            raise RuntimeError(f'frame {fc}: dictBackFace/dictNowFace not the hosted {{0:1}}; extend the writer')
        io.write(pre)
        seeded = io.read()
        expect_pre = FbsState.from_record(pre)
        seed_ok = {k: seeded[k].tobytes() == expect_pre[k].tobytes() for k in expect_pre}
        rt.time.update(deltaTime=float(F(pre['delta_time'])), time=float(F(pre['time'])), frameCount=fc)
        writes = hf.late_update()
        got, expect = io.read(), FbsState.from_record(post)
        fields = {k: got[k].tobytes() == expect[k].tobytes() for k in expect}
        weights = {}
        for name, d in writes.items():
            w = frames[fc]['W'][rid[name]]
            weights[name] = {'written': len(d), 'bit_equal': all(w[i].tobytes() == np.float32(v).tobytes() for i, v in d.items())}
        rows.append({'frame': fc, 'seed_fields_equal_prefix': all(seed_ok.values()), 'seed_mismatch': [k for k, v in seed_ok.items() if not v],
                     'postfix_fields_bit_equal': fields, 'all_postfix_fields_equal': all(fields.values()),
                     'mismatched': {k: [got[k].tolist(), expect[k].tolist()] for k, ok in fields.items() if not ok},
                     'weights': weights, 'all_weights_equal': all(v['bit_equal'] for v in weights.values()),
                     'blink_mode_pre_post': [pre['face_blend_shape_fields']['FaceBlendShape.BlinkCtrl']['FBSBlinkControl.blinkMode'],
                                             post['face_blend_shape_fields']['FaceBlendShape.BlinkCtrl']['FBSBlinkControl.blinkMode']]})
    return rows


# ------------------------------------------------------------------ mouth width
def run_mouth(h, frames, hooks):
    """ChaControl.UpdateBlendShapeVoice (Update, -45): SetLocalScaleX(cf_J_MouthMove, mouthCtrl.GetAdjustWidthScale()).

    Input state = end-of-frame F-1 export (no writer between frames) and the F-1 FBS
    postfix adjustWidthScale; the installed TransformScaleEx.SetLocalScaleX runs hosted.
    Checked against the UpdateForce postfix scale, then ABMX's LateUpdate output
    (abmx postfix) against the end-of-frame export."""
    rt = h.rt
    k = rt.klass('IL.dll', 'TransformScaleEx', 'IllusionUtility.SetUtility')
    method = rt.method(k, 'SetLocalScaleX', 2)
    bone = h.names['cf_J_MouthMove']
    i = h.ids.index(bone.index)
    rows = []
    for fc in sorted(frames):
        post = hooks.get((fc, 'chara_update_force', 'postfix'))
        prev_fbs = hooks.get((fc - 1, 'fbs', 'postfix'))
        if post is None or prev_fbs is None or (fc - 1) not in frames:
            continue
        m = post['mouth']
        h.apply_frame(frames[fc - 1]['T'])
        x = prev_fbs['face_blend_shape_fields']['FaceBlendShape.MouthCtrl']['FBSCtrlMouth.adjustWidthScale']
        voice_silent = (not m['as_voice_present'] or not m['as_voice_playing'])
        if m['file_status_mouth_adjust_width']:
            rt.invoke(method, 0, bone.obj, f32([x]))
        rec = m['scales']['trf_mouth_adjust_width']
        abmx_rec = hooks.get((fc, 'abmx', 'postfix'))
        if abmx_rec is None or 'mouth' not in abmx_rec:
            raise RuntimeError(f'frame {fc}: ABMX postfix mouth record missing')
        abmx = abmx_rec['mouth']['trf_mouth_adjust_width']           # abmx records hold the scales directly
        export = frames[fc]['T'][i]
        rows.append({'frame': fc, 'voice_silent': voice_silent, 'adjust_width_scale_from_fbs_f_minus_1': x,
                     'update_force_scale_bit_equal': same(bone.scl, rec['local_scale']),
                     'update_force_position_rotation_unchanged': same(bone.pos, rec['local_position']) and same(bone.rot, rec['local_rotation']),
                     'abmx_postfix_equals_export': same(export[7:10], abmx['local_scale']) and same(export[0:3], abmx['local_position'])
                                                    and same(export[3:7], abmx['local_rotation']),
                     'hosted_scale': bone.scl.tolist(), 'recorded_update_force_scale': rec['local_scale'],
                     'abmx_scale': abmx['local_scale']})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--window', type=Path, required=True, action='append')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('Fresh output path required')
    report, h, hf = {'schema': 'hs2_driver_hook_acceptance_v1', 'windows': {}}, None, None
    for window in args.window:
        static, frames, hooks = load(window)
        if h is None:
            h = Harness(static)
            hf = HostedFaceBlendShape(h.rt, load_fbs_asset())   # one instance: the weight icall is registered once
            hf.create()
        eye = run_eye(h, frames, hooks)
        fbs = run_fbs(hf, static, frames, hooks)
        mouth = run_mouth(h, frames, hooks)
        report['windows'][window.name] = {'eye': eye, 'fbs': fbs, 'mouth': mouth}
        for r in mouth:
            print(window.name[:15], 'mouth', r['frame'], 'update', r['update_force_scale_bit_equal'], r['update_force_position_rotation_unchanged'],
                  'abmx=export', r['abmx_postfix_equals_export'], r['hosted_scale'], r['abmx_scale'])
        for r in eye:
            print(window.name[:15], 'eye', r['frame'], 'fields', r['all_postfix_fields_equal'],
                  {n: b['bit_equal'] for n, b in r['bones'].items()}, 'world', r['world_mismatch'], list(r['mismatched'])[:4])
        for r in fbs:
            print(window.name[:15], 'fbs', r['frame'], 'seed', r['seed_fields_equal_prefix'], 'fields', r['all_postfix_fields_equal'],
                  'weights', r['all_weights_equal'], 'blink', r['blink_mode_pre_post'], list(r['mismatched'])[:4])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)), encoding='utf-8')


if __name__ == '__main__':
    main()
