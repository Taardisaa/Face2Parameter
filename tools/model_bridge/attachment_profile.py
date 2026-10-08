"""Lock existing validated cut inputs into a small local attachment profile.

Only paths/hashes and support declarations; no game/model arrays are distributed.
Actual installed body compatibility is still checked by the native runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import host_path, sha
from .attachment_runtime import package


FORMAT = 'hs2_native_attachment_profile_v1'
FIELDS = {'proposal', 'native_state', 'reference_state', 'reference_artifact'}


def read_profile(path):
    profile = json.loads(path.read_text(encoding='utf-8'))
    if profile.get('format') != FORMAT or set(profile.get('inputs', {})) != FIELDS:
        raise ValueError('Explicit native attachment profile required')
    paths = {}
    for key, row in profile['inputs'].items():
        p = host_path(row['path']).resolve()
        if sha(p) != row['sha256']:
            raise ValueError('Attachment profile input changed: '+key)
        paths[key] = p
    inputs = {key: json.loads(p.read_text(encoding='utf-8')) for key, p in paths.items() if key != 'reference_artifact'}
    package(inputs['proposal'], inputs['native_state'], inputs['reference_state'])
    if sha(paths['reference_artifact']) != inputs['reference_state']['artifact_sha256']:
        raise ValueError('Profile reference artifact differs from the validated design')
    return inputs, paths


def build_profile(proposal, native_state, reference_state, out):
    paths = {'proposal': proposal.resolve(), 'native_state': native_state.resolve(), 'reference_state': reference_state.resolve()}
    reference = json.loads(reference_state.read_text(encoding='utf-8'))
    paths['reference_artifact'] = host_path(reference['artifact_path']).resolve()
    package(json.loads(proposal.read_text(encoding='utf-8')), json.loads(native_state.read_text(encoding='utf-8')), reference)
    if sha(paths['reference_artifact']) != reference['artifact_sha256']:
        raise ValueError('Reference artifact changed')
    result = {'format': FORMAT, 'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
        'scope': 'Original unscaled single-submesh/no-blendshape body definition; actual immutable body signatures rechecked at activation',
        'source_accuracy_certified': False, 'collar_self_intersection_certified': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--native-state', type=Path, required=True)
    parser.add_argument('--reference-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    build_profile(args.proposal, args.native_state, args.reference_state, args.out)
    print(json.dumps({'profile': str(args.out.resolve()), 'sha256': sha(args.out), 'runtime_body_validation_required': True}))


if __name__ == '__main__':
    main()
