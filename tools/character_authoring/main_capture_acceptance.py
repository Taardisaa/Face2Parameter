"""Two unchanged native main-camera captures; verify pose and public restoration."""
from __future__ import annotations
import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
from PIL import Image


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request('http://127.0.0.1:43127'+path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def state():
    return {k: request(p) for k, p in {
        'snapshot': '/maker/snapshot?region=all', 'abmx': '/maker/abmx',
        'style': '/maker/renderstyle', 'lights': '/maker/lights',
        'expression': '/maker/face/express'}.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    before = state()
    shots = []
    options = dict(yaw=0, target=[.09548324, 14.50, .04006075],
                   ortho_size=1.43, width=768, height=1024, freeze_pose=True)
    for i in range(2):
        shot = request('/maker/renderstyle/capture', dict(options, path=str((a.out/f'front_{i}.png').resolve())))
        shots.append(shot['capture'])
    after = state()
    pose_deltas = {}
    for key in ('position', 'rotation_xyzw', 'scale'):
        pose_deltas[key] = float(np.max(np.abs(np.array(shots[0]['head_pose'][key])-
                                              np.array(shots[1]['head_pose'][key]))))
    invalid_rejected = False
    try:
        request('/maker/renderstyle/capture', dict(options, freeze_pose='true',
                path=str((a.out/'invalid_should_not_exist.png').resolve())))
    except urllib.error.HTTPError as e:
        invalid_rejected = e.code == 422
    invalid_unchanged = state() == after
    pixels = [np.asarray(Image.open(a.out/f'front_{i}.png').convert('RGBA'), dtype=int) for i in range(2)]
    pd = np.abs(pixels[0]-pixels[1])
    report = dict(scope='This character, two unchanged synchronous actual-main-camera captures. Head world TRS and public state only; no private neck/gaze-state, entire actor, or identity certification.',
                  shots=shots, head_pose_max_abs_delta=pose_deltas,
                  head_pose_equal_within_1e_6=all(x <= 1e-6 for x in pose_deltas.values()),
                  public_state_equal=before == after,
                  same_call_pose_restored=all(s['pose_restoration']['local_trs_equal'] and
                      s['pose_restoration']['blink_flag_equal'] for s in shots),
                  face_surfaces_equal=shots[0]['face_surfaces'] == shots[1]['face_surfaces'],
                  invalid_type_rejected=invalid_rejected,
                  invalid_request_state_equal=invalid_unchanged,
                  invalid_png_absent=not (a.out/'invalid_should_not_exist.png').exists(),
                  pixel_delta={'max': int(pd.max()), 'mean': float(pd.mean()),
                               'scope': 'Observed only; temporal/material changes are not converted into an invented acceptance threshold.'})
    for name, value in [('before', before), ('after', after), ('report', report)]:
        (a.out/f'{name}.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'shots'}, indent=2))
    checks = ['head_pose_equal_within_1e_6', 'public_state_equal', 'same_call_pose_restored', 'face_surfaces_equal',
              'invalid_type_rejected', 'invalid_request_state_equal', 'invalid_png_absent']
    if not all(report[k] for k in checks):
        raise SystemExit('Capture acceptance failed; preserve evidence and fix responsible source path.')


if __name__ == '__main__':
    main()
