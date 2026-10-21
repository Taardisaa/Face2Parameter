"""Frame F-1 -> F acceptance of the hosted EyeLook driver on the paired probe windows.

For every probe frame F (after the first) the installed EyeLookController.LateUpdate
is executed on the game's Mono JIT with
  * the frame-F local TRS of the whole exported character hierarchy (the eye
    look reads only ancestors of the eyes; it runs at execution order 10800,
    after the animator, ChaControl.LateUpdateForce (-45), ABMX (0), IK (9996-10001)
    and NeckLookControllerVer2 (10050); only DynamicBone (13100) and
    CorrectLightAngle (12400) follow it, neither writes an eye ancestor),
  * Time.deltaTime of frame F,
  * the camera position (Camera.main transform position; see CAMERA below),
  * the hidden EyeObject state carried from the hosted run of frame F-1.
cf_J_look_L/R local rotations are compared bit for bit with the frame-F export.

The hidden state entering the first frame of a window is not exported by probe
v4. It is reconstructed from that frame's exported look rotations (angleH,
angleV, dirUp of the decomposition L = LookRotation(n, u) used by EyeUpdateCalc
with referenceLookDir = +Z, referenceUpDir = +Y, origRotation = identity), and the
frames that depend on that reconstruction are reported separately.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.render.hs2_hosted_runtime import HostedRuntime, TransformScene  # noqa: E402
from hosted_eye_look import HostedEyeLook, load_eye_look_asset  # noqa: E402

F = np.float32
SESSION = Path('C:/Users/13666/Workspace/HS2Mod/artifacts/chenger_reference_fit_20261012/paired_capture_session_20261020')
WINDOWS = {'625aa0f2': 'window_625aa0f2a0b44eb498f711d82b74dd9f', '04d1e76f': 'window_04d1e76fdad14c11bbc6fe439b228e77'}
LOOK = ('cf_J_look_L', 'cf_J_look_R')


def load_window(name):
    root = SESSION / WINDOWS[name]
    static = json.loads((root / 'static.json').read_text(encoding='utf-8'))
    frames = []
    for path in sorted((root / 'frames').glob('frame_*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        row['T'] = np.frombuffer(base64.b64decode(row['transforms_f32']), F).reshape(-1, 33)
        scene = path.with_name(path.name.replace('frame_', 'scene_'))
        row['scene_path'] = scene if scene.exists() else None
        frames.append(row)
    return static, frames


def camera_from_scene(scene_path):
    d = json.loads(Path(scene_path).read_text(encoding='utf-8'))
    cams = [c for c in d['cameras'] if c['properties'].get('tag') == 'MainCamera' and c['properties'].get('enabled')
            and c['active_in_hierarchy']]
    if len(cams) != 1:
        raise RuntimeError('Camera.main is not unique in the scene export')
    m = np.asarray(cams[0]['properties']['cameraToWorldMatrix'], F).reshape(4, 4)
    return m[:3, 3].copy(), cams[0]['path']


def bits(a):
    return np.asarray(a, F).view(np.uint32).tolist()


class Harness:
    def __init__(self, static):
        self.rt = HostedRuntime()
        self.asset = load_eye_look_asset()
        self.el = HostedEyeLook(self.rt, self.asset)
        # Awake/Init in the instantiated prefab, before parenting and face shape.
        self.prefab = TransformScene(self.rt)
        pnames = {}
        for r in self.asset['prefab']:
            pnames[r['name']] = self.prefab.add(r['name'], r['parent'], r['pos'], r['rot'], r['scl'])
        self.el.create(self.prefab, pnames)
        self.el.awake()
        self.init_state = self.el.state()
        # Character scene: CharaCustom/CustomControl ancestors of chaF_001 (identity
        # TRS; probe v4 exports chaF_001 world == local with rotation (-0,-0,-0,1)).
        self.scene = TransformScene(self.rt)
        top = self.scene.add('CharaCustom', None, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
        mid = self.scene.add('CustomControl', top.index, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
        self.ids = []
        self.names = {}
        offset = mid.index + 1
        for t in static['transforms']:
            parent = mid.index if t['parent'] < 0 else t['parent'] + offset
            if t['i'] + offset != len(self.scene.items):
                raise RuntimeError('static transforms are not in parent-before-child order')
            st = self.scene.add(t['name'], parent, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
            self.ids.append(st.index)
            self.names.setdefault(t['name'], st)
        self.camera = self.scene.add('Main Camera(target)', None, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1])
        self.el.retarget(self.names)
        self.scene.attach(self.el.calc, self.names['p_cf_head_bone'])
        self.el.controller(self.scene, self.names, self.camera, ptn=1)

    def apply_frame(self, T):
        for row, idx in zip(T, self.ids):
            t = self.scene.items[idx]
            t.pos, t.rot, t.scl = row[0:3].copy(), row[3:7].copy(), row[7:10].copy()

    def world_check(self, T, names):
        bad = []
        for n in names:
            i = self.ids.index(self.names[n].index)
            p, q = self.scene.world(self.names[n])
            if p.tobytes() != T[i, 10:13].tobytes() or q.tobytes() != T[i, 13:17].tobytes():
                bad.append(n)
        return bad

    def run(self, frame, camera):
        self.apply_frame(frame['T'])
        self.camera.pos = np.asarray(camera, F)
        self.rt.time.update(deltaTime=float(frame['delta_time']), time=float(frame['time']),
                            frameCount=int(frame['frame_count']))
        self.el.late_update()
        return {n: self.names[n].rot.copy() for n in LOOK}


def reconstruct_state(T, ids, names):
    """Hidden EyeObject state implied (approximately) by an exported look rotation."""
    out = []
    for n in LOOK:
        q = T[ids.index(names[n].index), 3:7].astype(np.float64)
        x, y, z, w = q
        fwd = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
        up = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
        h = math.degrees(math.atan2(fwd[0], fwd[2]))
        v = math.degrees(-math.asin(max(-1.0, min(1.0, fwd[1]))))
        out.append({'angleH': F(h), 'angleV': F(v), 'dirUp': up.astype(F)})
    return out


SEEDS = ('reconstructed', 'init_state', 'reconstructed_up_tilted')


def seed_state(kind, T, h):
    if kind == 'init_state':
        return [{'angleH': F(0), 'angleV': F(0), 'dirUp': np.array([0, 1, 0], F)} for _ in LOOK]
    rec = reconstruct_state(T, h.ids, h.names)
    if kind == 'reconstructed_up_tilted':      # 1e-3 rad about +X: a deliberately wrong seed
        c, s_ = math.cos(1e-3), math.sin(1e-3)
        for r in rec:
            x, y, z = (float(v) for v in r['dirUp'])
            r['dirUp'] = np.array([x, c * y - s_ * z, s_ * y + c * z], F)
    return rec


def run_window(h, frames, cam_w2, seed):
    out = []
    for k, fr in enumerate(frames):
        if k == 0:
            h.apply_frame(fr['T'])
            h.el.set_state(seed_state(seed, fr['T'], h))
            continue
        if fr['scene_path'] is not None:
            cam, src = camera_from_scene(fr['scene_path'])[0], 'scene_export_cameraToWorld_translation'
        else:
            cam, src = cam_w2, 'window2_scene_export_camera'
        got = h.run(fr, cam)
        out.append((fr, src, got, [{kk: (vv.tolist() if hasattr(vv, 'tolist') else float(vv))
                                    for kk, vv in st.items() if kk in ('angleH', 'angleV', 'dirUp')} for st in h.el.state()]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    report = {'schema': 'hs2_eye_look_acceptance_v2', 'seeds': SEEDS, 'windows': {}}
    w2_static, w2_frames = load_window('04d1e76f')
    cam_w2, cam_path = camera_from_scene(w2_frames[0]['scene_path'])
    h = None
    chain = ['cf_J_FaceUp_tz', 'cf_J_EyePos_rz_L', 'cf_J_EyePos_rz_R', 'cf_J_look_L', 'cf_J_look_R', 'cf_J_Head', 'chaF_001']
    for wname in ('04d1e76f', '625aa0f2'):
        static, frames = load_window(wname)
        if h is None:
            h = Harness(static)
        world = {}
        for fr in frames:
            h.apply_frame(fr['T'])
            world[fr['frame_count']] = h.world_check(fr['T'], chain)
        runs = {seed: run_window(h, frames, cam_w2, seed) for seed in SEEDS}
        rows = [{'frame': frames[0]['frame_count'], 'role': 'state_seed', 'world_mismatch': world[frames[0]['frame_count']]}]
        for i, (fr, src, got, state) in enumerate(runs['reconstructed']):
            row = {'frame': fr['frame_count'], 'delta_time': fr['delta_time'], 'camera_source': src,
                   'world_mismatch': world[fr['frame_count']], 'state_after': state}
            for n in LOOK:
                exp = fr['T'][h.ids.index(h.names[n].index), 3:7]
                g = got[n]
                variants = {seed: runs[seed][i][2][n].tobytes() for seed in SEEDS}
                row[n] = {'bit_equal': g.tobytes() == exp.tobytes(), 'got': g.tolist(), 'export': exp.tolist(),
                          'ulp': (g.view(np.int32).astype(np.int64) - exp.view(np.int32).astype(np.int64)).tolist(),
                          'seed_invariant': len(set(variants.values())) == 1,
                          'equal_by_seed': {seed: v == exp.tobytes() for seed, v in variants.items()}}
            rows.append(row)
        report['windows'][wname] = rows
    report['init_state'] = [{k: v.tolist() if hasattr(v, 'tolist') else float(v) for k, v in s_.items()} for s_ in h.init_state]
    report['camera_main_path'] = cam_path
    report['camera_window2'] = cam_w2.tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    for w, rows in report['windows'].items():
        for r in rows:
            if r.get('role') == 'state_seed':
                print(w, r['frame'], 'seed', 'world_mismatch', r['world_mismatch'])
                continue
            print(w, r['frame'], f"dt={r['delta_time']}",
                  *[f"{n[-1]} eq={r[n]['bit_equal']} inv={r[n]['seed_invariant']} {r[n]['equal_by_seed']}" for n in LOOK],
                  'world_mismatch', r['world_mismatch'])


if __name__ == '__main__':
    main()
