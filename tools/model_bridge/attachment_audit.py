"""Inspect original head/body topology before shape-preserving base integration.

Reads source/model and installed render assets. No runtime sampling, fitted
coordinates, threshold welding, vertex editing or inferred bone correspondence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from .artifact import ModelArtifact, sha


def topology(vertices, faces):
    """Index-based components and directed boundaries, without merging seams."""
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Out-of-range source triangle")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = undirected[counts == 1]
    neighbours = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), set()).add(int(b))
        neighbours.setdefault(int(b), set()).add(int(a))
    unseen = set(neighbours)
    boundaries = []
    while unseen:
        seed = min(unseen)
        pending, found = [seed], set()
        while pending:
            item = pending.pop()
            if item in found:
                continue
            found.add(item)
            pending.extend(neighbours[item])
        unseen -= found
        ids = sorted(found)
        row = {"vertices": ids, "degrees": sorted({len(neighbours[i]) for i in ids}),
               "bounds_min": vertices[ids].min(0).tolist(), "bounds_max": vertices[ids].max(0).tolist()}
        if row["degrees"] == [2]:
            loop, previous, current = [], None, seed
            while current not in loop:
                loop.append(current)
                nxt = sorted(neighbours[current] - ({previous} if previous is not None else set()))[0]
                previous, current = current, nxt
            if current != seed or len(loop) != len(ids):
                raise ValueError("Unexpected boundary walk")
            row["loop"] = loop
        boundaries.append(row)
    parent = np.arange(len(vertices))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, c in faces:
        parent[root(b)] = root(a)
        parent[root(c)] = root(a)
    components = {}
    for item in np.unique(faces):
        components.setdefault(int(root(item)), []).append(int(item))
    return {"components": [{"vertices": ids, "bounds_min": vertices[ids].min(0).tolist(),
                              "bounds_max": vertices[ids].max(0).tolist()} for ids in components.values()],
            "boundaries": boundaries, "nonmanifold_edge_count": int((counts > 2).sum()),
            "welding_policy": "Original vertex indices; UV/material seam duplicates are not merged"}


def source_uv(obj, expected_faces):
    """Read the source renderer's actual OBJ corner correspondence, not OBJ shape."""
    uv, faces, corners = [], [], []
    vertex_count = 0
    for row in obj.read_text(encoding="utf-8").splitlines():
        fields = row.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertex_count += 1
        elif fields[0] == "vt":
            uv.append([float(x) for x in fields[1:3]])
        elif fields[0] == "f":
            if len(fields) != 4:
                raise ValueError("Audited source renderer requires triangular OBJ")
            indices = [item.split("/") for item in fields[1:]]
            faces.append([int(item[0]) - 1 for item in indices])
            corners.append([int(item[1]) - 1 for item in indices])
    uv, faces, corners = np.asarray(uv), np.asarray(faces), np.asarray(corners)
    if not np.array_equal(faces, expected_faces) or corners.min() < 0 or corners.max() >= len(uv):
        raise ValueError("Source OBJ topology/UV correspondence differs")
    # Unity has one UV per render vertex; duplicated corner pairs preserve geometry.
    pairs = np.unique(np.stack([faces.reshape(-1), corners.reshape(-1)], 1), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=vertex_count)
    return {"path": str(obj.resolve()), "sha256": sha(obj), "faces_match": True,
            "uv_vertices": len(uv), "render_corner_pairs": len(pairs),
            "source_vertices_requiring_uv_split": int((counts > 1).sum()),
            "policy": "UV data only; OBJ template positions never replace model output geometry"}


def body_asset(bundle):
    env = UnityPy.load(str(bundle))
    objects = list(env.objects)
    transforms, gameobjects = _read_transforms(objects)
    matches = []
    for obj in objects:
        if obj.type.name != "SkinnedMeshRenderer":
            continue
        renderer = obj.read()
        if renderer.m_GameObject.path_id not in gameobjects:
            continue
        chain = _chain(transforms, gameobjects[renderer.m_GameObject.path_id])
        if "p_cf_body_00" not in chain:
            continue
        mesh = renderer.m_Mesh.read()
        if mesh.m_Name == "o_body_cf":
            matches.append((obj, renderer, mesh, chain))
    if len(matches) != 1:
        raise ValueError("Expected exact female render body; collision/silhouette excluded")
    obj, renderer, mesh, chain = matches[0]
    arrays = _read_smr_mesh(mesh)
    names = [bone.read().m_GameObject.read().m_Name for bone in renderer.m_Bones]
    parent_matches = []
    for component in objects:
        if component.type.name != "MonoBehaviour":
            continue
        try:
            fields = component.read_typetree()
        except Exception:
            continue
        target = fields.get("targetEtc", {})
        if not isinstance(target, dict) or "trfHeadParent" not in target:
            continue
        reference = target["trfHeadParent"]
        if reference["m_FileID"] != 0:
            raise ValueError("Unimplemented external head-parent reference")
        pid = reference["m_PathID"]
        ancestry = _chain(transforms, pid)
        if "p_cf_anim" in ancestry:
            parent_matches.append({"path_id": pid, "transform": transforms[pid], "chain": ancestry})
    if len(parent_matches) != 1:
        raise ValueError("Expected source-selected p_cf_anim head attachment")
    index = names.index("cf_J_Head_s")
    head_weights = (arrays["bone_w"] * (arrays["bone_idx"] == index)).sum(1)
    active = np.flatnonzero(head_weights > 0)
    pure = np.flatnonzero(head_weights == 1)
    used = np.unique(arrays["faces"])
    bind_head = arrays["bindpose"][index].astype(np.float64)
    head_local = arrays["verts"][active] @ bind_head[:3, :3].T + bind_head[:3, 3]
    return arrays, {"bundle": str(bundle.resolve()), "bundle_sha256": sha(bundle),
        "prefab": "p_cf_body_00", "mesh": mesh.m_Name, "renderer_path_id": obj.path_id,
        "mesh_path_id": mesh.object_reader.path_id, "transform_chain": chain,
        "skin_bones": names, "head_bone": "cf_J_Head_s", "head_weight_positive_vertices": active.tolist(),
        "actual_cmp_bone_body_head_parent": parent_matches[0],
        "head_weight_one_vertices": pure.tolist(), "used_y_max": float(arrays["verts"][used, 1].max()),
        "head_influenced_bounds_min": arrays["verts"][active].min(0).tolist(),
        "head_influenced_bounds_max": arrays["verts"][active].max(0).tolist(),
        "head_influenced_bounds_in_head_bind_frame": {
            "min": head_local.min(0).tolist(), "max": head_local.max(0).tolist(),
            "policy": "Literal native body Head_s bindpose; head-weight support is not an anatomical posterior-skull label"},
        "topology": topology(arrays["verts"], arrays["faces"]),
        "head_support_is_not_a_cut_mask": True,
        "warning": "Body retains head-weighted surface when only o_head/eye renderers are hidden; positive head weights alone do not define an anatomical neck cut"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-obj", type=Path, required=True)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--cha-control-source", type=Path, required=True)
    parser.add_argument("--cha-ab-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use new output directory")
    control = args.cha_control_source.read_text(encoding="utf-8-sig")
    ab = args.cha_ab_source.read_text(encoding="utf-8-sig")
    if 'ChaABDefine.BodyAsset(base.sex)' not in control or 'FemaleBodyAsset = "p_cf_body_00"' not in ab:
        raise ValueError("Installed decompilation does not establish audited body selection")
    artifact = ModelArtifact(args.manifest)
    artifact.verify_sources()
    arrays, body = body_asset(args.game_root / "abdata/chara/oo_base.unity3d")
    report = {"format": "source_native_attachment_audit_v1", "source_manifest_sha256": sha(args.manifest),
        "game_assembly_sha256": sha(args.game_root / "HoneySelect2_Data/Managed/Assembly-CSharp.dll"),
        "selection_sources": [{"path": str(p.resolve()), "sha256": sha(p)} for p in [args.cha_control_source, args.cha_ab_source]],
        "source_template_topology": topology(artifact.state["v_template"], artifact.faces),
        "source_uv": source_uv(args.source_obj, artifact.faces), "native_female_body": body,
        "implemented_attachment": False,
        "remaining": ["Identify anatomical native head/body seam from authored topology and transforms",
                      "Resolve retained native upper-neck/head interface against source full neck; do not classify posterior skull from weights alone",
                      "Add explicit connector without changing source face vertices",
                      "Carry original UV corner mapping with exact seam duplication",
                      "Keep source eyeballs; native eye components require separate explicit correspondence"]}
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / "native_body.npz", **arrays)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str((args.out / "report.json").resolve()),
                      "source_boundaries": len(report["source_template_topology"]["boundaries"]),
                      "native_body_retains_head_surface": bool(body["head_weight_one_vertices"]),
                      "attachment_implemented": False}))


if __name__ == "__main__":
    main()
