"""Prepare complete, source-bound native/FLAME inputs for mother-template authoring.

No mesh edits, runtime calls, material substitution or expression suppression.
Logical seam aliases are for graph queries only; serialized indices stay intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import UnityPy

from scripts.hs2_extract_head import _chain, _read_smr_mesh, _read_transforms
from tools.model_bridge.artifact import ModelArtifact, sha
from tools.model_bridge.attachment_audit import topology
from tools.native_head.native_seam_audit import aliases
from tools.native_head.neck_geometry import ordered_loops


def source_file(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def _json_value(value):
    if isinstance(value, (bytes, bytearray)):
        return {"encoding": "hex_bytes", "data": bytes(value).hex()}
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2)+"\n",
                    encoding="utf-8")


def native_regions(arrays, bone_names):
    """Recognize source topology/bone-supported boundaries without UV boxes.

    MouthCavity alone is not a skin mask. It identifies the unique separate
    index-connected inner component; all its original faces/boundaries survive.
    """
    v, f = arrays["verts"], arrays["faces"]
    raw = topology(v, f)
    mapping = aliases(arrays)
    root = bone_names.index("cf_J_FaceRoot_s")
    cavity = bone_names.index("cf_J_MouthCavity")
    root_weights = (arrays["bone_w"]*(arrays["bone_idx"] == root)).sum(1)
    cavity_weights = (arrays["bone_w"]*(arrays["bone_idx"] == cavity)).sum(1)
    loops = ordered_loops(f, mapping)
    neck, eyes = [], {}
    unresolved = []
    for loop in loops:
        if np.all(root_weights[loop] == 1):
            neck.append(loop)
            continue
        active = np.unique(arrays["bone_idx"][loop][arrays["bone_w"][loop] > 0])
        sides = {side for side in ("L", "R") if any(
            bone_names[i].startswith("cf_J_Eye") and bone_names[i].endswith("_"+side)
            for i in active)}
        if len(sides) == 1:
            side = next(iter(sides))
            if side in eyes:
                raise ValueError("Ambiguous native eye boundary for "+side)
            eyes[side] = loop
        else:
            unresolved.append(loop)
    if len(neck) != 1 or set(eyes) != {"L", "R"} or unresolved:
        raise ValueError("Unresolved physical native boundaries; do not classify by height")
    oral = [c for c in raw["components"] if np.any(cavity_weights[c["vertices"]] == 1)]
    if len(oral) != 1:
        raise ValueError("No unique index-connected MouthCavity component")
    oral_ids = np.asarray(oral[0]["vertices"], int)
    oral_faces = np.flatnonzero(np.isin(f, oral_ids).all(1))
    oral_boundaries = [b for b in raw["boundaries"]
                       if np.isin(b["vertices"], oral_ids).all()]
    if not oral_boundaries or any("loop" not in b for b in oral_boundaries):
        raise ValueError("Inner-component boundary is not an ordered loop")
    groups = {}
    for i, canonical in mapping.items():
        groups.setdefault(canonical, []).append(i)
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    neck_canonical = set(neck[0])
    neck_copies = [i for i, canonical in mapping.items() if canonical in neck_canonical]
    # This label is structural, not an invented outer/inner vermilion boundary.
    # The two-sided native lip aperture still needs its authored contour profile.
    return dict(format="native_mother_topology_regions_v1",
        index_components=raw["components"], original_index_boundaries=raw["boundaries"],
        physical_neck_boundary=neck[0], neck_boundary_all_render_copies=neck_copies,
        physical_eye_boundaries=eyes,
        inner_mouth_component_vertex_ids=oral_ids.tolist(),
        inner_mouth_component_face_ids=oral_faces.tolist(),
        inner_mouth_component_boundaries=oral_boundaries,
        identical_bind_position_and_skin_groups=duplicate_groups,
        graph_aliases={str(k): v for k, v in mapping.items() if k != v},
        mesh_welded=False, cut_faces=[], semantic_upper_lower_lips_ready=False,
        semantic_ear_root_ready=False,
        limitations=["Bone-supported component and physical loops are structural evidence",
                     "Inner-component boundary is not automatically the visible lip aperture",
                     "No upper/lower eyelid, inner/outer lip or ear-root contour authored yet",
                     "No closest-point snapping, UV-region classification or native deformation implemented"])


def prepare(reference, source_manifest, out):
    if out.exists():
        raise FileExistsError("Preserve earlier evidence; use a fresh output directory")
    reference = reference.resolve()
    audit = json.loads(reference.read_text(encoding="utf-8-sig"))
    for row in audit["assemblies"]+audit["decompiled_sources"]:
        if sha(row["path"]) != row["sha256"]:
            raise ValueError("Source oracle changed: "+row["path"])
    native = audit["native"]
    bundle = Path(native["bundle"])
    if sha(bundle) != native["bundle_sha256"]:
        raise ValueError("Native head donor changed")
    model = ModelArtifact(source_manifest, 0)
    model.verify_sources()
    if model.manifest["format"] != "mica_flame_raw_export_v1":
        raise ValueError("This prototype uses the source-bound official MICA reference")
    if np.any(model.state["eye_pose"]) or np.any(model.state["neck_pose"]):
        raise ValueError("Nonzero source default pose needs a separate reference contract")
    env = UnityPy.load(str(bundle))
    objects = list(env.objects)
    transforms, gameobjects = _read_transforms(objects)
    prefab = native["prefab"]
    roots = [pid for pid in transforms if transforms[pid]["name"] == prefab]
    if len(roots) != 1:
        raise ValueError("Native prefab root is ambiguous")
    selected_transforms = {pid: t for pid, t in transforms.items()
                           if prefab in _chain(transforms, pid)}
    out.mkdir(parents=True)
    # This exact copy preserves the complete serialized package, including any
    # opaque data not decoded in the array exports. Never rebuild from NPZ only.
    shutil.copyfile(bundle, out/"donor_original.unity3d")
    if sha(out/"donor_original.unity3d") != native["bundle_sha256"]:
        raise ValueError("Donor copy differs")
    renderer_rows, behaviours, heads = [], [], []
    seen_names = set()
    for obj in objects:
        if obj.type.name not in ("SkinnedMeshRenderer", "MonoBehaviour"):
            continue
        value = obj.read()
        go = value.m_GameObject.path_id
        if go not in gameobjects or gameobjects[go] not in selected_transforms:
            continue
        if obj.type.name == "MonoBehaviour":
            script = value.m_Script.read()
            behaviours.append(dict(path_id=obj.path_id, transform=gameobjects[go],
                class_name=script.m_ClassName, assembly=script.m_AssemblyName,
                original_typetree=obj.read_typetree()))
            continue
        if value.m_Mesh.file_id != 0:
            raise ValueError("Unimplemented external native mesh")
        mesh = value.m_Mesh.read()
        name = mesh.m_Name
        if name in seen_names or not name.replace("_", "").isalnum():
            raise ValueError("Duplicate/unsafe renderer mesh name")
        seen_names.add(name)
        arrays = _read_smr_mesh(mesh)
        bones = [b.read().m_GameObject.read().m_Name for b in value.m_Bones]
        if len(bones) != len(arrays["bindpose"]):
            raise ValueError("Native bindpose/bone order differs")
        if not np.isfinite(arrays["verts"]).all() or not arrays["faces"].size:
            raise ValueError("Do not substitute empty native geometry")
        data_path = out/(name+".npz")
        np.savez_compressed(data_path, **arrays, bone_names=np.asarray(bones))
        shapes = mesh.object_reader.read_typetree()["m_Shapes"]
        shape_path = out/(name+"_blendshapes.json")
        save_json(shape_path, shapes)
        reopened = np.load(data_path, allow_pickle=False)
        if any(not np.array_equal(reopened[k], arrays[k]) for k in arrays):
            raise ValueError("Native array export changed "+name)
        renderer_rows.append(dict(mesh=name, mesh_path_id=mesh.object_reader.path_id,
            renderer_path_id=obj.path_id, transform=gameobjects[go],
            vertex_count=len(arrays["verts"]), triangle_count=len(arrays["faces"]),
            array_export=source_file(data_path), blendshape_export=source_file(shape_path),
            channels=[c["name"] for c in shapes["channels"]],
            original_renderer_typetree=obj.read_typetree()))
        if name == "o_head":
            heads.append((arrays, bones))
    required = {"o_head", "o_eyebase_L", "o_eyebase_R", "o_eyelashes", "o_eyeshadow",
                "o_tooth", "o_tang", "o_namida"}
    if len(heads) != 1 or not required.issubset(seen_names):
        raise ValueError("Complete original native head parts were not exported")
    if len([b for b in behaviours if b["class_name"] == "FaceBlendShape"]) != 1:
        raise ValueError("Expected exactly one original expression controller")
    regions = native_regions(*heads[0])
    save_json(out/"native_regions.json", regions)
    save_json(out/"native_prefab.json", dict(root_transform=roots[0],
        transforms=selected_transforms, renderers=renderer_rows, behaviours=behaviours))
    # Reference geometry is the actual decoder mean, never an OBJ template's
    # unrelated positions or one person's predicted shape.
    zero_path = out/"flame_zero_identity.npz"
    np.savez_compressed(zero_path, **model.state)
    report = dict(format="native_flame_mother_inputs_v1", game_mutated=False,
        extraction_source=source_file(Path(__file__)), reference_audit=source_file(reference),
        native_bundle=source_file(bundle), prefab=prefab, native_original_copy=source_file(out/"donor_original.unity3d"),
        source_manifest=source_file(model.path), source_decoder_files=model.manifest["sources"],
        flame_zero_identity=source_file(zero_path),
        native_prefab_export=source_file(out/"native_prefab.json"),
        regions=source_file(out/"native_regions.json"),
        all_renderers=[{k: v for k, v in row.items() if k != "original_renderer_typetree"}
                       for row in renderer_rows],
        original_body_contract="Native FaceRoot boundary unchanged; actual BP body contract still required before delivery",
        reference_policy="Official saved v_template; zero identity/expression and saved zero default articulation",
        native_rest_and_closed_state_distinguished=True,
        closed_reference_policy="Original controller patterns and all sparse frames exported; no closed frame baked into bind mesh",
        mother_candidate_generated=False, identity_basis_migrated=False,
        semantics_complete=False, external_material_dependencies_preserved_in_original_bundle=True,
        external_dependency_closure_exported=False)
    save_json(out/"receipt.json", report)
    print(json.dumps(dict(output=str(out.resolve()), all_native_renderers_exported=True,
                         physical_neck_identified=True, inner_mouth_component_identified=True,
                         mother_candidate_generated=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.reference, args.source_manifest, args.out)
