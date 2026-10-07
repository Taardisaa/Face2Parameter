"""Prepare a native neck cut from installed body topology and bindposes.

This creates new attachment geometry; it does not infer or replace the game's
deformation. Runtime endpoints must be interpolated from actual native BakeMesh
vertices, rather than interpolating weights and assuming skinning commutes.
Source face vertices are never inputs to the cut and are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .attachment_audit import body_asset
from .artifact import sha


def digest_array(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def prepare(arrays, bone_names, *, geometry_vertices=None, plane=None):
    vertices = np.asarray(arrays["verts"] if geometry_vertices is None else geometry_vertices, dtype=np.float64)
    if vertices.shape != arrays["verts"].shape or not np.isfinite(vertices).all():
        raise ValueError("Actual native body vertex order/count required")
    faces = arrays["faces"].astype(np.int64)
    neck = bone_names.index("cf_J_Neck_s")
    head = bone_names.index("cf_J_Head_s")
    neck_to_mesh = np.linalg.inv(arrays["bindpose"][neck].astype(np.float64))
    origin = neck_to_mesh[:3, 3]
    normal = np.linalg.inv(neck_to_mesh[:3, :3]).T @ np.array([0., 1., 0.])
    normal /= np.linalg.norm(normal)
    if plane is not None:
        origin, normal = (np.asarray(value, dtype=np.float64) for value in plane)
        if origin.shape != (3,) or normal.shape != (3,) or not np.isfinite([origin, normal]).all():
            raise ValueError("Explicit finite cutting plane required")
        normal /= np.linalg.norm(normal)
    signed = (vertices - origin) @ normal
    above = signed > 0
    head_weight = (arrays["bone_w"] * (arrays["bone_idx"] == head)).sum(1)
    seeds = np.flatnonzero(above & (head_weight == 1))
    if not len(seeds):
        raise ValueError("No audited head-support seed above original neck plane")
    # Connectivity above the explicit plane excludes detached shoulder/hand caps.
    # UV seams may give multiple components; all with pure head seeds are retained.
    neighbours = {}
    for tri in faces:
        ids = [int(i) for i in tri if above[i]]
        for a in ids:
            neighbours.setdefault(a, set()).update(i for i in ids if i != a)
    region = set()
    pending = list(map(int, seeds))
    while pending:
        item = pending.pop()
        if item in region:
            continue
        region.add(item)
        pending.extend(neighbours.get(item, ()))
    edges, edge_indices = [], {}
    # Authored duplicate vertices may differ only in UV/normals. Equivalence is
    # exact bind position + ordered skin data, never a spatial welding tolerance.
    signatures = [np.asarray(arrays['verts'][i], dtype='<f4').tobytes()
                  + np.asarray(arrays['bone_idx'][i], dtype='<i4').tobytes()
                  + np.asarray(arrays['bone_w'][i], dtype='<f4').tobytes() for i in range(len(vertices))]
    original_count = len(vertices)
    def intersection(a, b):
        key = tuple(sorted((int(a), int(b))))
        if key not in edge_indices:
            i, j = sorted(key, key=lambda item: signatures[item])
            t = float(signed[i] / (signed[i] - signed[j]))
            if not 0 < t < 1:
                raise ValueError("Plane exactly hits a vertex; this branch remains unsupported")
            edge_indices[key] = original_count + len(edges)
            edges.append([i, j, t])
        return edge_indices[key]
    output, cut_segments, untouched = [], [], []
    for face_id, tri in enumerate(faces):
        if not any(int(i) in region for i in tri):
            output.append(tri.tolist())
            untouched.append(face_id)
            continue
        if any(above[i] and int(i) not in region for i in tri):
            raise ValueError("Mixed selected/unselected positive region in one triangle")
        polygon, crossings = [], []
        for a, b in zip(tri, np.roll(tri, -1)):
            inside_a, inside_b = signed[a] <= 0, signed[b] <= 0
            if inside_a:
                polygon.append(int(a))
            if inside_a != inside_b:
                item = intersection(a, b)
                polygon.append(item)
                crossings.append(item)
        if len(crossings) == 2:
            cut_segments.append(crossings)
        for j in range(1, len(polygon) - 1):
            output.append([polygon[0], polygon[j], polygon[j + 1]])
    if not edges or not cut_segments:
        raise ValueError("Selected body region has no explicit neck cut")
    # Preserve render seam splits but recover connectivity at exactly equivalent
    # authored edges. Use the same endpoint evaluation order on both seam copies.
    canonical_edges, aliases = {}, {}
    for offset, (a, b, _) in enumerate(edges):
        key = (signatures[int(a)], signatures[int(b)])
        item = original_count + offset
        canonical_edges.setdefault(key, item)
        aliases[item] = canonical_edges[key]
    adjacency = {}
    for a, b in cut_segments:
        a, b = aliases[a], aliases[b]
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    open_ends = sorted(i for i, values in adjacency.items() if len(values) != 2)
    edge_points = np.asarray([vertices[int(a)] * (1 - t) + vertices[int(b)] * t for a, b, t in edges])
    return {"format": "native_neck_cut_v1", "native_vertex_count": original_count,
        "native_vertex_sha256": digest_array(arrays["verts"], "<f4"),
        "native_triangle_sha256": digest_array(faces, "<i4"),
        "neck_bone": "cf_J_Neck_s", "head_seed_bone": "cf_J_Head_s",
        "plane_origin_mesh": origin.tolist(), "plane_normal_mesh": normal.tolist(),
        "selected_positive_vertices": sorted(region), "pure_head_seeds": seeds.tolist(),
        "triangles": np.asarray(output).reshape(-1).tolist(), "edge_interpolation": edges,
        "cut_segments": cut_segments, "unresolved_seam_endpoints": open_ends,
        "exact_cut_seam_aliases": [[i, aliases[i]] for i in sorted(aliases)],
        "untouched_original_faces": untouched,
        "selected_region_bounds_min": vertices[sorted(region)].min(0).tolist(),
        "selected_region_bounds_max": vertices[sorted(region)].max(0).tolist(),
        "edge_plane_max_abs_error": float(np.abs((edge_points - origin) @ normal).max()),
        "cut_geometry": "bind geometry" if geometry_vertices is None else "actual saved native posed geometry",
        "runtime_policy": "Keep original native vertices; interpolate each new endpoint from actual posed native endpoints. Do not interpolate skin weights to approximate posed endpoints.",
        "source_face_vertices_modified": False,
        "runtime_body_replacement_implemented": False,
        "collar_implemented": False}


def connect(result, native_vertices, source_vertices, source_faces, source_loop):
    """Add a collar using only existing ring points, preserving face geometry."""
    count = result['native_vertex_count']
    cut = np.asarray([native_vertices[int(a)] * (1-t) + native_vertices[int(b)] * t
                      for a,b,t in result['edge_interpolation']])
    aliases = dict(result['exact_cut_seam_aliases'])
    cut_edges = set()
    for tri in np.asarray(result['triangles']).reshape(-1,3):
        for a,b in zip(tri,np.roll(tri,-1)):
            if int(a) >= count and int(b) >= count:
                a,b=aliases[int(a)],aliases[int(b)]
                if (b,a) in cut_edges: cut_edges.remove((b,a))
                else: cut_edges.add((a,b))
    following={}
    for a,b in cut_edges:
        if a in following: raise ValueError('Branched native cut boundary')
        following[a]=b
    if result['unresolved_seam_endpoints'] or not following:
        raise ValueError('Native cut has unresolved indexed/seam boundaries')
    start=min(following);ring=[];item=start
    while item not in ring:
        ring.append(item);item=following[item]
    if item!=start or len(ring)!=len(following):
        raise ValueError('Native cut is not a single closed ring')
    wanted=set(source_loop); source_edges=set()
    for tri in np.asarray(source_faces).reshape(-1,3):
        for a,b in zip(tri,np.roll(tri,-1)):
            if int(a) in wanted and int(b) in wanted:
                pair=(int(a),int(b))
                if pair[::-1] in source_edges:source_edges.remove(pair[::-1])
                else:source_edges.add(pair)
    forward=dict(source_edges); source_order=[];item=min(wanted)
    while item not in source_order:
        source_order.append(item);item=forward[item]
    if item!=source_order[0] or set(source_order)!=wanted:
        raise ValueError('Source neck loop differs from its directed boundary')
    # Opposing seam half-edges are required for a consistently oriented annulus.
    ring=list(reversed(ring))
    a=np.asarray(source_vertices)[source_order];b=cut[np.asarray(ring)-count]
    first=int(np.argmin(np.square(b-a[0]).sum(1)))
    ring=ring[first:]+ring[:first];b=cut[np.asarray(ring)-count]
    def progress(points):
        distances=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
        if np.any(distances==0): raise ValueError('Degenerate neck ring edge')
        return np.r_[0.,np.cumsum(distances)/distances.sum()]
    pa,pb=progress(a),progress(b); n,m=len(a),len(b);i=j=0;faces=[]
    while i<n or j<m:
        if i<n and (j==m or pa[i+1]<=pb[j+1]):
            faces.append([(i+1)%n,i%n,n+j%m]);i+=1
        else:
            faces.append([i%n,n+j%m,n+(j+1)%m]);j+=1
    points=np.concatenate([a,b]);faces=np.asarray(faces,dtype=np.int32)
    if np.any(np.linalg.norm(np.cross(points[faces[:,1]]-points[faces[:,0]],
                                     points[faces[:,2]]-points[faces[:,0]]),axis=1)==0):
        raise ValueError('Degenerate connector triangle')
    result['collar']={'source_ring_indices':source_order,'native_ring_cut_indices':ring,
        'triangles':faces.reshape(-1).tolist(),'design_renderer_local_vertices':points.tolist(),
        'policy':'Exact existing source/native endpoints; deterministic cyclic zipper adds collar triangles only. Geometry is new attachment, not recovered game deformation.'}
    result['collar_implemented']=True
    result['collar_scope']='Offline attachment construction only; game integration remains incomplete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--native-state", type=Path)
    parser.add_argument("--source-state", type=Path)
    parser.add_argument("--attachment-audit", type=Path)
    parser.add_argument("--neck-boundary-index", type=int, default=1,
                        help="Explicit audited source neck loop, not inferred from a generic largest-loop rule")
    parser.add_argument("--clearance-ratio", type=float, default=.05,
                        help="New collar design clearance as a fraction of native neck-to-head joint distance; no face fitting")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use new output path")
    bundle = args.game_root / "abdata/chara/oo_base.unity3d"
    arrays, metadata = body_asset(bundle)
    state_paths = (args.native_state, args.source_state, args.attachment_audit)
    if any(state_paths) and not all(state_paths):
        parser.error("Use native/source state and attachment audit together")
    inputs = None
    if all(state_paths):
        native = json.loads(args.native_state.read_text())
        source = json.loads(args.source_state.read_text())
        audit = json.loads(args.attachment_audit.read_text())
        if len(native["meshes"]) != 1 or native["meshes"][0]["mesh_name"] != metadata["mesh"]:
            raise ValueError("One actual audited native body required")
        mesh = native["meshes"][0]
        if mesh['blendshapes']:
            raise ValueError('Body blendshape seam equivalence remains unresolved; refuse attachment')
        if mesh["renderer_lossy_scale"] != [1., 1., 1.]:
            raise ValueError("Unresolved body BakeMesh scale branch; no guessed world conversion")
        if not source.get("active") or not source.get("source_vertices_unchanged"):
            raise ValueError("Original source-pose mesh required for this attachment design")
        transforms = {row["id"]: row for row in native["transforms"]}
        src = [row for row in native["transforms"] if row["name"] == "HS2_SourceModelHead"]
        if len(src) != 1:
            raise ValueError("Exactly one source head in the saved body frame required")
        def matrix(values):
            return np.asarray(values, dtype=float).reshape(4, 4)
        src_matrix = matrix(src[0]["local_to_world"])
        body_matrix = matrix(mesh["renderer_local_to_world"])
        neck_row = transforms[mesh["bone_transform_ids"][mesh["bone_names"].index("cf_J_Neck_s")]]
        head_row = transforms[mesh["bone_transform_ids"][mesh["bone_names"].index("cf_J_Head_s")]]
        neck_matrix = matrix(neck_row["local_to_world"])
        normal_world = np.linalg.inv(neck_matrix[:3, :3]).T @ [0., 1., 0.]
        normal_world /= np.linalg.norm(normal_world)
        neck_loop = audit["source_template_topology"]["boundaries"][args.neck_boundary_index]
        if neck_loop.get("degrees") != [2] or not neck_loop.get("loop"):
            raise ValueError("Audited simple source neck loop required")
        vertices = np.asarray(source["render_vertices"])
        ring = vertices[neck_loop["loop"]]
        world_ring = ring @ src_matrix[:3, :3].T + src_matrix[:3, 3]
        if not 0 < args.clearance_ratio < .25:
            raise ValueError("Explicit small positive new-collar clearance required")
        length = float(np.linalg.norm(np.asarray(neck_row["world_position"]) - head_row["world_position"]))
        clearance = args.clearance_ratio * length
        origin_world = world_ring[np.argmin(world_ring @ normal_world)] - normal_world * clearance
        inverse_body = np.linalg.inv(body_matrix)
        origin_local = (inverse_body @ np.r_[origin_world, 1.])[:3]
        normal_local = body_matrix[:3, :3].T @ normal_world
        actual = np.asarray(mesh["baked"]["vertices"])
        exact_native = mesh["source"]
        runtime_arrays = {}
        static_equal = True
        for key, field in [("verts", "vertices"), ("faces", "triangles"), ("bone_idx", "bone_indices"),
                           ("bone_w", "bone_weights"), ("bindpose", "bindposes")]:
            value = np.asarray(exact_native[field], dtype=arrays[key].dtype)
            width = 3 if key in ["verts", "faces"] else 4
            value = value.reshape(-1, 4, 4) if key == "bindpose" else value.reshape(-1, width)
            runtime_arrays[key] = value
            static_equal = static_equal and np.array_equal(arrays[key], value)
        if len(runtime_arrays["verts"]) != mesh["vertex_count"] or len(runtime_arrays["bindpose"]) != len(mesh["bone_names"]):
            raise ValueError("Native source snapshot dimensions differ")
        result = prepare(runtime_arrays, mesh["bone_names"], geometry_vertices=actual, plane=(origin_local, normal_local))
        source_world = vertices @ src_matrix[:3,:3].T + src_matrix[:3,3]
        source_body = source_world @ inverse_body[:3,:3].T + inverse_body[:3,3]
        connect(result, actual, source_body, source['render_triangles'], neck_loop['loop'])
        inputs = {"paths_sha256": [{"path": str(p.resolve()), "sha256": sha(p)} for p in state_paths],
                  "frame_count": native["frame_count"], "source_artifact_sha256": source["artifact_sha256"],
                  "source_neck_loop": neck_loop["loop"], "clearance_ratio": args.clearance_ratio,
                  "native_source_geometry_sha256": mesh["source_geometry_sha256"],
                  "native_mesh_instance_id": mesh["mesh_instance_id"], "native_renderer_path": mesh["renderer_path"],
                  "vanilla_static_body_equal": static_equal, "selected_mod_asset_resolved": False,
                  "clearance_world": clearance, "plane_policy": "Below the lowest source neck-loop point along actual native neck-up; new collar construction, not a game deformation rule"}
    else:
        result = prepare(arrays, metadata["skin_bones"])
    result["attachment_design_inputs"] = inputs
    result["vanilla_reference"] = {"bundle": str(bundle.resolve()), "bundle_sha256": sha(bundle),
                                  "mesh": metadata["mesh"], "mesh_path_id": metadata["mesh_path_id"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.out.resolve()), "new_cut_vertices": len(result["edge_interpolation"]),
        "unresolved_seam_endpoints": len(result["unresolved_seam_endpoints"]),
        "runtime_replacement": False, "source_face_vertices_modified": False}))


if __name__ == "__main__":
    main()
