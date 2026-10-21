"""FaceBlendShape (FBS) per-frame driver, executed as HS2's own code.

Mechanism (installed IL.dll, decompiled_IL/FaceBlendShape.cs, FBSBlinkControl.cs,
FBSBase.cs, FBSCtrlEyebrow/Eyes/Mouth.cs, FBSAssist/TimeProgressCtrl*.cs):
  * FaceBlendShape.Start subscribes OnLateUpdate to UniRx Observable.EveryLateUpdate,
    i.e. MainThreadDispatcher.LateUpdate (installed MonoScript execution order
    -16000: the first LateUpdate of the frame).
  * OnLateUpdate: BlinkCtrl.CalcBlink(); eyebrow/eyes CalcBlend(openRate or -1 when
    blinking is fixed); MouthCtrl.CalcBlend(voiceValue). The head prefab's
    FaceBlendShape.EyeLookController is null (p_cf_head_02 serialized m_PathID 0,
    never assigned by ChaControl), so the eye-look open correction is inactive.
  * Each controller writes every Close/Open index of its PtnSet on each target
    SkinnedMeshRenderer; later controllers overwrite earlier ones on shared
    indices (eyebrow -> eyes -> mouth).
  * Hidden state: FBSBlinkControl (fixedFlags, blinkMode, calcSpeed, blinkTime,
    count, openRate), each FBSBase (openRate, OpenMin/Max, FixedRate,
    correctOpenMax, dictBackFace/dictNowFace, blendTimeCtrl), the mouth's
    TimeProgressCtrlRandom and scale lerp, and UnityEngine.Random.state.
"""
from __future__ import annotations

import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.render.hs2_hosted_runtime import HostedRuntime  # noqa: E402

F = np.float32
IL = 'IL.dll'
HEAD_BUNDLE = 'E:/HoneySelect2_ArcticFox/abdata/chara/38/fo_head_38.unity3d'


def load_fbs_asset(prefab='p_cf_head_02', path=HEAD_BUNDLE):
    import UnityPy
    env = UnityPy.load(path)
    objs = {o.path_id: o for o in env.objects}
    for o in env.objects:
        if o.type.name != 'MonoBehaviour':
            continue
        d = o.read()
        try:
            if d.m_Script.read().m_ClassName != 'FaceBlendShape':
                continue
        except Exception:
            continue
        if d.m_GameObject.read().m_Name != prefab:
            continue
        tt = o.read_typetree()
        for ctrl in ('EyebrowCtrl', 'EyesCtrl', 'MouthCtrl'):
            for t in tt[ctrl]['FBSTarget']:
                t['target_name'] = objs[t['ObjTarget']['m_PathID']].read().m_Name
        tt['path_id'] = o.path_id
        return tt
    raise RuntimeError(f'FaceBlendShape of {prefab} not found')


class HostedFaceBlendShape:
    def __init__(self, rt: HostedRuntime, asset):
        self.rt = rt
        self.asset = asset
        k = lambda n: rt.klass(IL, n)  # noqa: E731
        self.k_fbs, self.k_blink = k('FaceBlendShape'), k('FBSBlinkControl')
        self.k_brow, self.k_eyes, self.k_mouth = k('FBSCtrlEyebrow'), k('FBSCtrlEyes'), k('FBSCtrlMouth')
        self.k_target = k('FBSTargetInfo')
        self.k_closeopen = rt.nested(self.k_target, 'CloseOpen')
        self.k_smr = rt.klass('UnityEngine.CoreModule.dll', 'SkinnedMeshRenderer', 'UnityEngine')
        self.smr = {}
        self.writes = []
        self._install()

    def _install(self):
        def set_weight(this, index, value):
            self.writes.append((self.smr_names[this], int(index), np.float32(value)))
        self.smr_names = {}
        self.rt.icall('UnityEngine.SkinnedMeshRenderer::SetBlendShapeWeight', None,
                      [C.c_void_p, C.c_int, C.c_float], set_weight)

    def _renderer(self, name):
        if name not in self.smr:
            obj = self.rt.new(self.k_smr, init=False)
            self.rt.set(obj, 'm_CachedPtr', np.array([0x200000 + len(self.smr)], np.int64))
            self.smr[name] = obj
            self.smr_names[obj] = name
        return self.smr[name]

    def _controller(self, klass, data):
        rt = self.rt
        ctrl = rt.new(klass)
        targets = []
        for t in data['FBSTarget']:
            ti = rt.new(self.k_target)
            sets = []
            for p in t['PtnSet']:
                co = rt.new(self.k_closeopen)
                rt.set(co, 'Close', np.array([p['Close']], np.int32))
                rt.set(co, 'Open', np.array([p['Open']], np.int32))
                sets.append(co)
            rt.set_ref(ti, 'PtnSet', rt.array(self.k_closeopen, sets))
            rt.set_ref(ti, 'ObjTarget', 0)            # resolved below as smrTarget (GetComponent result)
            targets.append((ti, t['target_name']))
        rt.set_ref(ctrl, 'FBSTarget', rt.array(self.k_target, [ti for ti, _ in targets]))
        for f in ('OpenMin', 'OpenMax', 'FixedRate'):
            rt.set(ctrl, f, np.array([data[f]], F))
        for f, ct in (('SyncBlink', np.uint8), ('useAjustWidthScale', np.uint8)):
            if f in data:
                rt.set(ctrl, f, np.array([data[f]], ct))
        for f in ('randTimeMin', 'randTimeMax', 'randScaleMin', 'randScaleMax', 'openRefValue'):
            if f in data:
                rt.set(ctrl, f, np.array([data[f]], F))
        if 'objAdjustWidthScale' in data:
            rt.set_ref(ctrl, 'objAdjustWidthScale', 0)
        return ctrl, targets

    def create(self):
        rt, a = self.rt, self.asset
        fbs = rt.new(self.k_fbs)
        rt.set(fbs, 'm_CachedPtr', np.array([0x300000], np.int64))
        blink = rt.new(self.k_blink)
        rt.set(blink, 'BlinkFrequency', np.array([a['BlinkCtrl']['BlinkFrequency']], np.uint8))
        rt.set(blink, 'BaseSpeed', np.array([a['BlinkCtrl']['BaseSpeed']], F))
        rt.set_ref(fbs, 'BlinkCtrl', blink)
        self.ctrls = {}
        for name, klass in (('EyebrowCtrl', self.k_brow), ('EyesCtrl', self.k_eyes), ('MouthCtrl', self.k_mouth)):
            ctrl, targets = self._controller(klass, a[name])
            rt.set_ref(fbs, name, ctrl)
            self.ctrls[name] = (ctrl, targets)
        rt.set_ref(fbs, 'EyeLookController', 0)
        for f in ('EyeLookUpCorrect', 'EyeLookDownCorrect', 'EyeLookSideCorrect'):
            rt.set(fbs, f, np.array([a[f]], F))
        self.fbs, self.blink = fbs, blink
        # Awake: EyebrowCtrl.Init(); EyesCtrl.Init(); MouthCtrl.Init()
        rt.invoke(rt.method(self.k_fbs, 'Awake', 0), fbs)
        # FBSTargetInfo.SetSkinnedMeshRenderer() result: the target object's SkinnedMeshRenderer.
        for ctrl, targets in self.ctrls.values():
            for ti, name in targets:
                rt.set_ref(ti, 'smrTarget', self._renderer(name))
        return fbs

    # ChaControl.ChangeHeadAsync -> fbs setup calls with the card's fileStatus.
    def apply_file_status(self, eyes_blink=True, eyes_open_max=1.0, eyebrow_open_max=1.0, mouth_open_max=1.0,
                          mouth_open_min=0.0, mouth_fixed=False):
        rt = self.rt
        rt.invoke(rt.method(self.k_blink, 'SetFixedFlags', 1), self.blink, np.array([0 if eyes_blink else 1], np.uint8))
        brow, eyes, mouth = (self.ctrls[n][0] for n in ('EyebrowCtrl', 'EyesCtrl', 'MouthCtrl'))
        rt.set(brow, 'OpenMax', np.array([eyebrow_open_max], F))
        rt.set(eyes, 'OpenMax', np.array([eyes_open_max], F))
        rt.set(mouth, 'OpenMax', np.array([mouth_open_max], F))
        rt.set(mouth, 'OpenMin', np.array([mouth_open_min], F))
        if mouth_fixed:
            rt.set(mouth, 'FixedRate', np.array([mouth_open_max], F))

    def set_blink_state(self, mode, blink_time, calc_speed, count, open_rate=1.0):
        rt = self.rt
        rt.set(self.blink, 'blinkMode', np.array([mode], np.int8))
        rt.set(self.blink, 'blinkTime', np.array([blink_time], F))
        rt.set(self.blink, 'calcSpeed', np.array([calc_speed], F))
        rt.set(self.blink, 'count', np.array([count], np.int32))
        rt.set(self.blink, 'openRate', np.array([open_rate], F))

    def late_update(self):
        self.writes = []
        self.rt.invoke(self.rt.method(self.k_fbs, 'OnLateUpdate', 0), self.fbs)
        final = {}
        for name, idx, v in self.writes:
            final.setdefault(name, {})[idx] = v
        return final

    def adjust_width_scale(self):
        mouth = self.ctrls['MouthCtrl'][0]
        r = self.rt.invoke(self.rt.method(self.k_mouth, 'GetAdjustWidthScale', 0), mouth)
        return np.float32(self.rt.unbox(r, C.c_float))

    def random_state(self):
        rt = self.rt
        k = rt.klass('UnityEngine.CoreModule.dll', 'Random', 'UnityEngine')
        r = rt.invoke(rt.method(k, 'get_state', 0), 0)
        return C.string_at(rt.m.object_unbox(r), 16).hex()
