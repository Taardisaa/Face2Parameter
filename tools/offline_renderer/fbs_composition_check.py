"""Hosted FaceBlendShape composition check against the paired probe windows.

The blink timer state and UnityEngine.Random.state are not exported by probe v4,
so which frames are mid-blink cannot be predicted. This check runs the installed
OnLateUpdate in the two steady blink states the source defines -- open
(blinkMode 0: openRate 1) and closed (blinkMode 1 after blinkTime, count > 0:
openRate 0) -- and verifies that every exported FBS-written weight of every probe
frame equals exactly one of the two hosted compositions (all controllers, all
targets, overwrite order). It is not a blink-timing acceptance.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.render.hs2_hosted_runtime import HostedRuntime  # noqa: E402
from hosted_face_blendshape import HostedFaceBlendShape, load_fbs_asset  # noqa: E402
from eye_look_acceptance import SESSION, WINDOWS  # noqa: E402

F = np.float32


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(argv)
    rt = HostedRuntime()
    h = HostedFaceBlendShape(rt, load_fbs_asset())
    h.create()
    loaded = json.loads((SESSION / 'prepare' / 'loaded_state.json').read_text(encoding='utf-8'))['expression']
    h.apply_file_status(eyes_blink=loaded['eyes_blink'], eyes_open_max=loaded['eyes_open_max'],
                        mouth_open_max=loaded['mouth_open_max'], mouth_open_min=loaded['mouth_open_min'],
                        mouth_fixed=loaded['mouth_fixed'])
    rt.time.update(deltaTime=0.0166835, time=1000.0)
    h.set_blink_state(0, 2000.0, 0.15, 0)
    h.late_update()                                   # settle blendTimeCtrl (ended) and dicts
    open_w = h.late_update()
    open_scale = h.adjust_width_scale()
    h.set_blink_state(1, 999.0, 0.15, 1000)
    closed_w = h.late_update()
    closed_scale = h.adjust_width_scale()
    report = {'schema': 'hs2_fbs_composition_check_v1', 'expression_state': loaded,
              'open': {k: {str(i): float(v) for i, v in d.items()} for k, d in open_w.items()},
              'closed': {k: {str(i): float(v) for i, v in d.items()} for k, d in closed_w.items()},
              'mouth_adjust_width_scale': {'open': float(open_scale), 'closed': float(closed_scale)},
              'windows': {}}
    for wname, wdir in WINDOWS.items():
        static = json.loads((SESSION / wdir / 'static.json').read_text(encoding='utf-8'))
        tf = static['transforms']

        def under_head(i):                     # FBS targets are children of the head object (ct_head)
            while i >= 0:
                if tf[i]['name'] == 'ct_head':
                    return True
                i = tf[i]['parent']
            return False
        rid = {r['name']: r['id'] for r in static['renderers'] if under_head(r['transform'])}
        rows = []
        for path in sorted((SESSION / wdir / 'frames').glob('frame_*.json')):
            fr = json.loads(path.read_text(encoding='utf-8'))
            weights = {r['id']: np.frombuffer(base64.b64decode(r['weights_f32']), F) for r in fr['renderers']}
            match = []
            for label, comp in (('open', open_w), ('closed', closed_w)):
                ok = all(weights[rid[name]][idx].tobytes() == np.float32(v).tobytes()
                         for name, d in comp.items() for idx, v in d.items())
                if ok:
                    match.append(label)
            rows.append({'frame': fr['frame_count'], 'matches': match})
        report['windows'][wname] = rows
        print(wname, [(r['frame'], r['matches']) for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print('adjust width scale', open_scale, closed_scale)


if __name__ == '__main__':
    main()
