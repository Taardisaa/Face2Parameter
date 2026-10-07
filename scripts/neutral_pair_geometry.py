"""Original decoder geometry for an associated neutral case; no image renderer.

SMIRK identity-only neutralization affects a separate diagnostic copy. MICA's
original canonical output stays literal. This export never fits either model.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.model_bridge.artifact import ModelArtifact, host_path, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smirk-manifest', type=Path, required=True)
    parser.add_argument('--mica-manifest', type=Path, required=True)
    parser.add_argument('--embedding', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Preserve prior geometry; use a new output directory')
    first, mica = ModelArtifact(args.smirk_manifest), ModelArtifact(args.mica_manifest)
    first.verify_sources(); mica.verify_sources()
    if not np.array_equal(first.faces, mica.faces) or not np.array_equal(first.state['v_template'], mica.state['v_template']):
        raise ValueError('Shared FLAME semantic topology is unverified')
    embedding_source = next(r for r in first.manifest['sources'] if 'mediapipe_landmark_embedding.npz' in r['path'])
    if sha(args.embedding) != embedding_source['sha256']:
        raise ValueError('Original semantic embedding changed')
    with np.load(args.embedding, allow_pickle=False) as e:
        face_ids = torch.as_tensor(e['lmk_face_idx'].astype(np.int64))[None]
        bary = torch.as_tensor(e['lmk_b_coords'].astype(np.float32))[None]
        landmark_ids = e['landmark_indices'].tolist()
    lbs_source = next(r for r in first.manifest['sources'] if r['path'].endswith('src/FLAME/lbs.py'))
    spec = importlib.util.spec_from_file_location('neutral_original_lbs', host_path(lbs_source['path']))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    faces = torch.as_tensor(first.faces.astype(np.int64))
    def landmarks(vertices):
        return module.vertices2landmarks(torch.as_tensor(vertices[None].astype(np.float32)), faces, face_ids, bary)[0].numpy()
    mica_by_hash = {r['input_sha256']: i for i, r in enumerate(mica.manifest['images'])}
    args.out.mkdir(parents=True)
    report = {'format':'neutral_pair_original_geometry_v1','raw_parameters_modified':False,
              'smirk_manifest':str(args.smirk_manifest.resolve()),'smirk_manifest_sha256':sha(args.smirk_manifest),
              'mica_manifest':str(args.mica_manifest.resolve()),'mica_manifest_sha256':sha(args.mica_manifest),
              'embedding_sha256':sha(args.embedding),'landmark_ids':landmark_ids,
              'original_lbs':lbs_source,'exporter_sha256':sha(Path(__file__)),'images':[]}
    for i, row in enumerate(first.manifest['images']):
        s = ModelArtifact(args.smirk_manifest, i); m = ModelArtifact(args.mica_manifest, mica_by_hash[row['input_sha256']])
        s.verify_sources(); m.verify_sources()
        posed = s.mesh()[0]; neutral = s.replay(neutral_identity=True)[0]; original_mica = m.mesh()[0]
        for a in (s,m):
            if np.max(np.abs(a.replay()[0]-a.mesh()[0])) > 1e-6:
                raise ValueError('Original source decoder does not replay exported mesh')
        if np.max(np.abs(landmarks(posed)-s.arrays['geometry__landmarks_mp'][0])) > 1e-6:
            raise ValueError('Original landmark evaluation differs')
        path = args.out/(host_path(row['input']).stem+'.npz')
        np.savez_compressed(path, faces=s.faces, smirk_raw_posed=posed, smirk_identity_diagnostic=neutral,
                            mica_original=original_mica, smirk_posed_landmarks=landmarks(posed),
                            smirk_identity_landmarks=landmarks(neutral), mica_landmarks=landmarks(original_mica),
                            detected_landmarks=s.arrays['detected_landmarks'])
        report['images'].append({'input_sha256':row['input_sha256'],'input':row['input'],
                                'file':path.name,'sha256':sha(path),'smirk_artifact_sha256':row['sha256'],
                                'mica_artifact_sha256':m.image['sha256']})
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'manifest':str(args.out/'manifest.json'),'inputs':len(report['images']),'accuracy_certified':False}))


if __name__ == '__main__':
    main()
