"""Package the constructed collar for the native character-owned runtime.

This packages exact saved body topology/skin/UV and source placement. It does not
fit a face or substitute a renderer/skin algorithm. Mutable native normals are
intentionally not identity hashes: ChaControl.BustNormal updates them in place.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .artifact import sha
from .neck_patch import digest_array


def boundary(faces, aliases=None):
    aliases = aliases or {}
    counts = Counter()
    for tri in np.asarray(faces).reshape(-1, 3):
        for a, b in zip(tri, np.roll(tri, -1)):
            a, b = aliases.get(int(a), int(a)), aliases.get(int(b), int(b))
            counts[(a, b)] += 1
    if any(v > 1 for v in counts.values()):
        raise ValueError("Duplicate directed edge in attachment surface")
    return {(a, b) for a, b in counts if (b, a) not in counts}


def package(proposal, native, source, replacement_artifact=None):
    if proposal.get("format") != "native_neck_cut_v1" or not proposal.get("collar_implemented"):
        raise ValueError("Constructed cut/collar required")
    if len(native['meshes']) != 1:
        raise ValueError("One saved actual body required")
    mesh = native['meshes'][0]
    if mesh['blendshapes'] or len(mesh['source']['submeshes']) != 1 or mesh['renderer_lossy_scale'] != [1., 1., 1.]:
        raise ValueError("Only the explicit unscaled single-submesh/no-blendshape branch is supported")
    if not source.get('active') or not source.get('source_vertices_unchanged') or not source.get('source_triangle_indices_unchanged'):
        raise ValueError("Original source-pose arrays required for the attachment design")
    design = proposal['attachment_design_inputs']
    if design['source_artifact_sha256'] != source['artifact_sha256'] or design['native_source_geometry_sha256'] != mesh['source_geometry_sha256']:
        raise ValueError("Saved source/body differs from the design's exact input")
    original = mesh['source']
    signatures = {field+'_sha256': digest_array(original[field], dtype) for field, dtype in [
        ('vertices', '<f4'), ('triangles', '<i4'), ('bone_indices', '<i4'),
        ('bone_weights', '<f4'), ('bindposes', '<f4'), ('uv', '<f4'), ('uv2', '<f4')]}
    if signatures['vertices_sha256'] != proposal['native_vertex_sha256'] or signatures['triangles_sha256'] != proposal['native_triangle_sha256']:
        raise ValueError("Cut topology differs from actual source arrays")
    count = mesh['vertex_count']
    edges = np.asarray(proposal['edge_interpolation'], dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 3 or not np.isfinite(edges).all() or (
            np.any(edges[:, :2] != edges[:, :2].astype(int)) or
            np.any(edges[:, :2] < 0) or np.any(edges[:, :2] >= count) or
            np.any(edges[:, 2] <= 0) or np.any(edges[:, 2] >= 1)):
        raise ValueError("Invalid native edge descriptor")
    if count + len(edges) > 65535:
        raise ValueError("Native replacement exceeds current UInt16 mesh support")
    tangents = np.asarray(original['tangents'])
    if len(tangents) and np.any(tangents[edges[:, 0].astype(int), 3] != tangents[edges[:, 1].astype(int), 3]):
        raise ValueError("Cut crosses native tangent handedness seam")
    collar = proposal['collar']
    src = collar['source_ring_indices']; dst = collar['native_ring_cut_indices']
    source_edges = boundary(source['render_triangles'])
    native_edges = boundary(proposal['triangles'], dict(proposal['exact_cut_seam_aliases']))
    src_wanted, dst_wanted = set(src), set(dst)
    # Source and body index spaces may overlap numerically. Validate separately,
    # rather than unioning their indices and pretending they share a vertex space.
    n = len(src)
    collar_edges = boundary(collar['triangles'])
    source_join = {(src[b], src[a]) for a, b in collar_edges if a < n and b < n}
    native_join = {(dst[b-n], dst[a-n]) for a, b in collar_edges if a >= n and b >= n}
    if source_join != {(a, b) for a, b in source_edges if a in src_wanted and b in src_wanted} or (
            native_join != {(a, b) for a, b in native_edges if a in dst_wanted and b in dst_wanted}):
        raise ValueError("Collar does not oppose the source/native boundary half-edges")
    if len(source_join) != len(src) or len(native_join) != len(dst) or len(collar['triangles']) // 3 != len(src) + len(dst):
        raise ValueError("Incomplete collar annulus")
    result = {
        'format': 'hs2_source_neck_attachment_v1',
        'source_artifact_sha256': source['artifact_sha256'],
        'source_placement': {'scale': source['local_scale'][0], 'translation': source['local_position']},
        'native': {'vertex_count': count, 'bone_names': mesh['bone_names'], **signatures},
        'edge_interpolation': proposal['edge_interpolation'], 'triangles': proposal['triangles'],
        'source_ring_indices': src, 'native_ring_cut_indices': dst,
        'collar_triangles': collar['triangles'],
        'design': {'policy': 'New collar only; source face and untouched native triangles retained',
                   'input_provenance': design, 'source_accuracy_certified': False,
                   'normals_hash_exclusion': 'Native BustNormal mutates normals; runtime uses actual BakeMesh normals'},
    }
    if replacement_artifact is not None:
        artifact = json.loads(replacement_artifact.read_text(encoding='utf-8'))
        if artifact.get('format') not in {'hs2_source_head_mesh_v1', 'hs2_source_head_mesh_v2', 'hs2_source_head_mesh_v3'} or artifact.get('geometry_mode') != 'head_local' or (
                not np.array_equal(np.asarray(artifact['vertices'], dtype=np.float32), np.asarray(source['render_vertices'], dtype=np.float32)) or
                not np.array_equal(np.asarray(artifact['triangles'], dtype=np.int32), np.asarray(source['render_triangles'], dtype=np.int32))):
            raise ValueError("Alternate source artifact must preserve every original vertex and triangle")
        result['source_artifact_sha256'] = sha(replacement_artifact)
        result['design']['equivalent_source_artifact'] = {'path': str(replacement_artifact.resolve()),
            'sha256': sha(replacement_artifact), 'literal_mesh_arrays_equal': True}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal', type=Path, required=True)
    parser.add_argument('--native-state', type=Path, required=True)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-artifact', type=Path,
                        help='Optional original-rig artifact with byte-identical decoded mesh arrays')
    args = parser.parse_args()
    result = package(*(json.loads(p.read_text(encoding='utf-8')) for p in
                       (args.proposal, args.native_state, args.source_state)),
                     replacement_artifact=args.source_artifact)
    result['provenance'] = [{'path': str(p.resolve()), 'sha256': sha(p)} for p in
                            (args.proposal, args.native_state, args.source_state)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'attachment': str(args.out.resolve()), 'sha256': sha(args.out)}))


if __name__ == '__main__':
    main()
