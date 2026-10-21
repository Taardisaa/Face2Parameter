"""EyeLookController/EyeLookCalc per-frame driver, executed as HS2's own code.

Mechanism (installed IL.dll, decompiled_IL/EyeLookController.cs, EyeLookCalc.cs):
  * p_cf_head_bone (chara/oo_base.unity3d) carries EyeLookCalc + EyeLookController.
    Awake -> Init() runs during CommonLib.LoadAsset's Instantiate, before
    ChaControl parents the clone under trfHeadParent and before InitShapeFace,
    so referenceLookDir/referenceUpDir/origRotation come from prefab TRS only.
  * EyeLookController.LateUpdate (script execution order 10800, from the
    installed MonoScript) calls EyeUpdateCalc(target.position, ptnNo). Maker:
    ptnNo = 1 (CustomControl.Initialize ChangeLookEyesPtn(1); CustomBase _eyelook=0
    -> ChangeLookEyesPtn(1)), target = Camera.main.transform (eyesTargetType 0).
  * EyeUpdateCalc reads rootNode = cf_J_FaceUp_tz and the eye parents
    cf_J_EyePos_rz_L/R world TRS (face-slider dependent) and writes
    cf_J_look_L/R. Hidden per-eye state: angleH, angleV, dirUp.

The managed code runs on the installed Mono JIT (src/render/hs2_hosted_runtime.py);
native Unity calls are the original UnityPlayer.dll routines.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.render.hs2_hosted_runtime import HostedRuntime, TransformScene  # noqa: E402

F = np.float32
IL = 'IL.dll'

# Serialized EyeLookCalc of p_cf_head_bone (oo_base.unity3d path_id -8722742763119948945).
EYE_TYPE_FIELDS = ('thresholdAngleDifference', 'bendingMultiplier', 'maxAngleDifference', 'upBendingAngle',
                   'downBendingAngle', 'minBendingAngle', 'maxBendingAngle', 'leapSpeed', 'forntTagDis',
                   'nearDis', 'hAngleLimit', 'vAngleLimit')


def load_eye_look_asset(path='E:/HoneySelect2_ArcticFox/abdata/chara/oo_base.unity3d'):
    import UnityPy
    env = UnityPy.load(path)
    objs = {o.path_id: o for o in env.objects}
    calc = None
    for o in env.objects:
        if o.type.name != 'MonoBehaviour':
            continue
        d = o.read()
        try:
            name = d.m_Script.read().m_ClassName
        except Exception:
            continue
        if name == 'EyeLookCalc' and d.m_GameObject.read().m_Name == 'p_cf_head_bone':
            calc = (o.path_id, o.read_typetree(), d.m_GameObject.m_PathID)
            break
    if calc is None:
        raise RuntimeError('EyeLookCalc of p_cf_head_bone not found')
    go = objs[calc[2]].read()
    root = None
    for c in go.m_Component:
        p = c.component if hasattr(c, 'component') else c
        oo = objs.get(p.m_PathID)
        if oo is not None and oo.type.name == 'Transform':
            root = oo
    rows = []

    def walk(o, parent):
        t = o.read()
        tt = o.read_typetree()
        idx = len(rows)
        rows.append({'path_id': o.path_id, 'name': t.m_GameObject.read().m_Name, 'parent': parent,
                     'pos': [tt['m_LocalPosition'][k] for k in 'xyz'],
                     'rot': [tt['m_LocalRotation'][k] for k in 'xyzw'],
                     'scl': [tt['m_LocalScale'][k] for k in 'xyz']})
        for ch in t.m_Children:
            walk(objs[ch.m_PathID], idx)
    walk(root, None)
    return {'calc_path_id': calc[0], 'calc': calc[1], 'prefab': rows}


class HostedEyeLook:
    def __init__(self, rt: HostedRuntime, asset):
        self.rt = rt
        self.asset = asset
        self.k_calc = rt.klass(IL, 'EyeLookCalc')
        self.k_ctrl = rt.klass(IL, 'EyeLookController')
        self.k_obj = rt.klass(IL, 'EyeObject')
        self.k_state = rt.klass(IL, 'EyeTypeState')
        self.k_quat = rt.klass('UnityEngine.CoreModule.dll', 'Quaternion', 'UnityEngine')
        self.k_single = rt.m.get_single_class()
        self.by_pid = {r['path_id']: r['name'] for r in asset['prefab']}

    # ------------------------------------------------------------ construction
    def create(self, scene: TransformScene, names: dict):
        """EyeLookCalc + EyeLookController as deserialized from the prefab."""
        rt, cal = self.rt, self.asset['calc']
        calc = rt.new(self.k_calc)
        scene.attach(calc, names['p_cf_head_bone'])
        rt.set_ref(calc, 'rootNode', names[self.by_pid[cal['rootNode']['m_PathID']]].obj)
        eyes = []
        for e in cal['eyeObjs']:
            eo = rt.new(self.k_obj)
            rt.set_ref(eo, 'eyeTransform', names[self.by_pid[e['eyeTransform']['m_PathID']]].obj)
            rt.set(eo, 'eyeLR', np.array([e['eyeLR']], np.int32))
            eyes.append(eo)
        rt.set_ref(calc, 'eyeObjs', rt.array(self.k_obj, eyes))
        rt.set(calc, 'headLookVector', np.array([cal['headLookVector'][k] for k in 'xyz'], F))
        rt.set(calc, 'headUpVector', np.array([cal['headUpVector'][k] for k in 'xyz'], F))
        states = []
        for s in cal['eyeTypeStates']:
            so = rt.new(self.k_state)
            for f in EYE_TYPE_FIELDS:
                rt.set(so, f, np.array([s[f]], F))
            rt.set(so, 'lookType', np.array([s['lookType']], np.int32))
            states.append(so)
        rt.set_ref(calc, 'eyeTypeStates', rt.array(self.k_state, states))
        rt.set_ref(calc, 'angleHRate', rt.array(self.k_single, [np.array(v, F).tobytes() for v in cal['angleHRate']],
                                                elem_size=4, refs=False))
        rt.set(calc, 'angleVRate', np.array([cal['angleVRate']], F))
        rt.set(calc, 'sorasiRate', np.array([cal['sorasiRate']], F))
        rt.set(calc, 'isDebugDraw', np.array([cal['isDebugDraw']], np.uint8))
        rt.set(calc, 'ptnDraw', np.array([cal['ptnDraw']], np.int32))
        rt.set(calc, 'drawLineLength', np.array([cal['drawLineLength']], F))
        rt.set_ref(calc, 'targetObj', 0)
        rt.set_ref(calc, 'fixAngle', rt.array(self.k_quat, [np.array([q[k] for k in 'xyzw'], F).tobytes()
                                                          for q in cal['fixAngle']], elem_size=16, refs=False))
        self.calc, self.eyes = calc, eyes
        return calc

    def controller(self, scene, names, target, ptn):
        rt = self.rt
        ctrl = rt.new(self.k_ctrl)
        scene.attach(ctrl, names['p_cf_head_bone'])
        rt.set_ref(ctrl, 'eyeLookScript', self.calc)
        rt.set_ref(ctrl, 'target', target.obj)
        rt.set(ctrl, 'ptnNo', np.array([ptn], np.int32))
        self.ctrl = ctrl
        return ctrl

    def awake(self):
        self.rt.invoke(self.rt.method(self.k_calc, 'Awake', 0), self.calc)

    def late_update(self):
        self.rt.invoke(self.rt.method(self.k_ctrl, 'LateUpdate', 0), self.ctrl)

    def retarget(self, names):
        rt, cal = self.rt, self.asset['calc']
        rt.set_ref(self.calc, 'rootNode', names[self.by_pid[cal['rootNode']['m_PathID']]].obj)
        for eo, e in zip(self.eyes, cal['eyeObjs']):
            rt.set_ref(eo, 'eyeTransform', names[self.by_pid[e['eyeTransform']['m_PathID']]].obj)

    # ------------------------------------------------------------ state
    def state(self):
        rt = self.rt
        out = []
        for eo in self.eyes:
            out.append({'angleH': rt.floats(eo, 'angleH', 1)[0], 'angleV': rt.floats(eo, 'angleV', 1)[0],
                        'dirUp': rt.floats(eo, 'dirUp', 3), 'referenceLookDir': rt.floats(eo, 'referenceLookDir', 3),
                        'referenceUpDir': rt.floats(eo, 'referenceUpDir', 3),
                        'origRotation': rt.floats(eo, 'origRotation', 4)})
        return out

    def set_state(self, states):
        rt = self.rt
        for eo, s in zip(self.eyes, states):
            for k in ('angleH', 'angleV'):
                if k in s:
                    rt.set(eo, k, np.array([s[k]], F))
            if 'dirUp' in s:
                rt.set(eo, 'dirUp', np.asarray(s['dirUp'], F))
