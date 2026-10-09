"""Check intact source/candidate surfaces, including adjacent triangle folds.

This checks one finished geometry stage, not game parameter sampling. Contact
and real crossing are retained separately; source intersections are not silently
discarded, nor are they evidence that new crossings are acceptable.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.geometry_quality.mesh_quality import self_intersections
from tools.model_bridge.artifact import sha
from tools.native_head.mother_template_inputs import save_json, source_file


def crossing_pairs(report):
    return {tuple(sorted(r['triangles'])) for r in report['pairs']
            if r['kind'] in ('proper_crossing', 'coplanar_overlap')}


def check(candidate, out):
    if out.exists():
        raise FileExistsError('Preserve prior quality evidence; use fresh directory')
    arrays = dict(np.load(candidate/'o_head_candidate.npz', allow_pickle=False))
    receipt = json.loads((candidate/'receipt.json').read_text())
    if sha(receipt['inputs']['path']) != receipt['inputs']['sha256']:
        raise ValueError('Prepared candidate inputs changed')
    recorded = receipt.get('candidate_arrays')
    if recorded and sha(recorded['path']) != recorded['sha256']:
        raise ValueError('Candidate geometry changed')
    faces = arrays['faces']
    reports = []
    out.mkdir(parents=True)
    for name, vertices in (('original_default', arrays['original_vertices']), ('candidate', arrays['verts'])):
        tri = vertices[faces]
        area2 = np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
        if np.any(area2 == 0):
            raise ValueError('Source/candidate has a collapsed triangle')
        report = self_intersections(vertices, faces, tri, np.ones(len(faces), bool), 1e-8,
                                    exclude_shared_vertex=False)
        save_json(out/(name+'_intersections.json'), report)
        reports.append(report)
        print(name+' triangle classification finished', flush=True)
    before, after = map(crossing_pairs, reports)
    added = after-before
    save_json(out/'receipt.json', dict(format='native_mother_surface_quality_v1',
        candidate=source_file(candidate/'receipt.json'), arrays=source_file(candidate/'o_head_candidate.npz'),
        legacy_candidate_receipt_without_array_hash=recorded is None, code=source_file(Path(__file__)),
        narrow_phase_code=source_file(Path('tools/geometry_quality/mesh_quality.py')),
        reports=[source_file(out/(n+'_intersections.json')) for n in ('original_default','candidate')],
        source_crossings=sorted(before), candidate_crossings=sorted(after), added_crossings=sorted(added),
        no_new_crossings=not added, shared_vertex_pairs_checked=True,
        installed=False, game_mutated=False, deliverable=False,
        limitations=['Floating point triangle/plane interval and coplanar clipping with explicit length epsilon 1e-8',
                     'Reports point/edge contacts separately; not an exact-arithmetic topology proof',
                     'Only head surface checked; component intersections and semantic matching remain required']))
    print(json.dumps(dict(output=str(out.resolve()), no_new_crossings=not added, deliverable=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    check(args.candidate.resolve(),args.out.resolve())
