"""Follow the fixed head2 support material point through saved native/ABMX probes.

Old unscoped captures are inventoried but never projected in diagnostic mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.surface_calibration.core import Camera, ContractError, SurfaceMesh, follow, meshes_from_geometry, validate_reprojection
from tools.surface_calibration.pixel_certificate import certify_pixel_contract, digest, read_json, scope, _same
from tools.unity_parity.geometry import analyze_snapshot, skin_world
from .diagnostic import render_support_diagnostics


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.asarray(a) @ np.asarray(b), -1, 1))))


def geometry_transport(geometry, anchor):
    heads = [m for m in geometry["meshes"] if m["mesh_name"] == "o_head"]
    if geometry.get("character", {}).get("head_id") != 2 or len(heads) != 1:
        raise ContractError("Fixed head2 material definition cannot be transferred to another head")
    head = heads[0]
    if head["renderer_path"] != anchor["renderer_path"] or head["source_geometry_sha256"] != anchor["source_geometry_sha256"]:
        raise ContractError("Renderer/source geometry identity changed; recalibration required; no replacement selected")
    fresh = analyze_snapshot(geometry, normalized_tolerance=1e-5)
    certification = {"world_policy_validated": True, "world_candidate": "scale_free_trs", "mesh_source_hashes": {}}
    for row in fresh["meshes"]:
        if not row["certified"] or "scale_free_trs" not in row.get("matching_candidates", []):
            raise ContractError("Independent actual LBS rejects selected world conversion")
        certification["mesh_source_hashes"][row["renderer_path"]] = row["source_geometry_sha256"]
    meshes = meshes_from_geometry(geometry, candidate="scale_free_trs", certification=certification)
    baked = follow(anchor, meshes)  # Strict renderer, source, topology and weights.
    transforms = {t["id"]: t for t in geometry["transforms"]}
    world, active = skin_world(head, transforms, influences=4)
    lbs_mesh = SurfaceMesh(head["renderer_path"], world, np.asarray(head["source"]["triangles"], dtype=int).reshape(-1, 3),
                           head["source_geometry_sha256"], visible=head["enabled"] and head["active_in_hierarchy"],
                           world_policy_certified=True)
    independent = follow(anchor, [lbs_mesh])
    position_error = float(np.linalg.norm(np.asarray(baked["world"]) - independent["world"]))
    diagonal = float(np.linalg.norm(np.ptp(world, axis=0)))
    normal_error = angle(baked["geometric_normal"], independent["geometric_normal"])
    head_row = next(r for r in fresh["meshes"] if r["renderer_path"] == head["renderer_path"])
    weights = np.asarray(head["source"]["bone_weights"])[3468]
    indices = np.asarray(head["source"]["bone_indices"])[3468]
    influence = [{"bone_name": head["bone_names"][int(i)], "weight": float(w)} for i, w in zip(indices, weights) if w > 0]
    return {"status": "source_bound_numeric_material_transport_verified",
            "fixed_vertex_id": 3468, "triangle_id": 6044, "barycentric": [0, 0, 1],
            "extremum_reselected": False, "source_and_ordered_topology_binding_passed": True,
            "baked_material": baked, "independent_lbs_material": independent,
            "anchor_lbs_vs_baked_error_game_units": position_error,
            "anchor_lbs_vs_baked_error_normalized": position_error / diagonal,
            "geometric_normal_lbs_vs_baked_angle_degrees": normal_error,
            "head_world_bbox_diagonal_game_units": diagonal, "independent_lbs_head_report": head_row,
            "native_face_values_actual": geometry["character"]["shape_value_face"],
            "active_expression_blendshapes_applied": active, "source_vertex_bone_influences": influence,
            "renderer_enabled": head["enabled"], "renderer_active_in_hierarchy": head["active_in_hierarchy"],
            "anatomical_correspondence_validated": False}, meshes


def paired_with_geometry(view, geometry, geometry_sha):
    pair = view.get("paired_geometry") or {}
    signature = geometry.get("pose_signature")
    return bool(isinstance(signature, str) and signature and pair.get("sha256") == geometry_sha
                and pair.get("pose_signature") == signature
                and view.get("pose_signature_before_render") == signature == view.get("pose_signature_after_render")
                and view.get("paired_pose_unchanged") is True
                and all(type(v) is int for v in [view.get("frame_count_before_render"), view.get("frame_count"), pair.get("frame_count"), geometry.get("frame_count"), geometry.get("frame_count_end")])
                and view["frame_count_before_render"] == view["frame_count"] == pair["frame_count"] == geometry["frame_count"] == geometry["frame_count_end"])


def run(args):
    definition = read_json(args.anchor)
    anchor = definition["support_definition"]["material"]
    if anchor["triangle_id"] != 6044 or anchor["vertex_ids"] != [3462, 3459, 3468] or anchor["barycentric"] != [0, 0, 1]:
        raise ContractError("Expected existing fixed vertex3468/triangle6044 definition; cannot substitute or refit")
    baseline_source = definition["source_bindings"]["geometry"]
    if digest(baseline_source["path"]) != baseline_source["sha256"]:
        raise ContractError("Original definition's actual geometry changed")
    reference, _ = geometry_transport(read_json(baseline_source["path"]), anchor)
    reference_world = np.asarray(reference["independent_lbs_material"]["world"])
    reference_normal = reference["independent_lbs_material"]["geometric_normal"]
    source_scopes = [certificate["matched_scope"] for certificate in read_json(args.reference_report)["pixel_certificates"]]
    results = []
    manifest_sources = []
    manifests = [path for path in [args.head2_manifest, args.paired_manifest] if path] + args.manifest
    for manifest_path in manifests:
        manifest = read_json(manifest_path)
        manifest_sources.append({"path": str(manifest_path.resolve()), "sha256": digest(manifest_path)})
        group = manifest_path.parent.name
        cases = [c for c in manifest["cases"] if group != "paired_base_cases" or c["name"].startswith("head_2_")]
        for case in cases:
            name = case["name"]
            geometry_path = Path(case["geometry"]["path"])
            geometry_sha = digest(geometry_path)
            result = {"group": group, "case": name, "geometry_path": str(geometry_path.resolve()),
                      "geometry_sha256": geometry_sha, "image_acceptance": False,
                      "anatomical_correspondence_validated": False, "native_or_abmx_effect_is_isolated": False}
            if geometry_sha != case["geometry"]["sha256"]:
                raise ContractError("Saved case geometry changed since live manifest")
            geometry = read_json(geometry_path)
            try:
                transported, meshes = geometry_transport(geometry, anchor)
                result.update(transported)
                current = np.asarray(transported["independent_lbs_material"]["world"])
                result["world_displacement_from_original_aa1_anchor_game_units"] = (current - reference_world).tolist()
                result["world_displacement_from_original_aa1_anchor_l2"] = float(np.linalg.norm(current - reference_world))
                result["normal_rotation_from_original_aa1_degrees"] = angle(transported["independent_lbs_material"]["geometric_normal"], reference_normal)
            except ContractError as exc:
                result.update(status="material_transport_rejected", reason=str(exc))
                meshes = None
            capture = case["capture"]
            views = capture.get("views") or [capture]
            image_records = []
            for index, view in enumerate(views):
                image_path = Path(view["path"])
                record = {"view_index": index, "yaw": view.get("yaw"), "path": str(image_path.resolve()),
                          "png_sha256": digest(image_path), "camera_projection": view["capture_camera"]["projection"],
                          "pose_and_frame_pairing_passed": paired_with_geometry(view, geometry, geometry_sha),
                          "pixel_contract_validated": False, "geometry_visibility_evaluated": False,
                          "material_visibility_validated": False, "anatomical_correspondence_validated": False}
                try:
                    captured_scope = scope(view)
                    record["scope_mismatches_against_aa1_reference"] = [key for key in captured_scope if not _same(captured_scope[key], source_scopes[0][key])]
                except ContractError as exc:
                    record["scope_inventory_rejection"] = str(exc)
                    # Present fields can still be compared without filling the
                    # missing fields or granting a certificate.
                    source = source_scopes[0]
                    record["present_camera_scope_mismatches_against_aa1_reference"] = [key for key, value in view["capture_camera"].items() if key in source and not _same(value, source[key])]
                try:
                    certificate = certify_pixel_contract(args.pixel_report, view)
                    record.update(pixel_contract_validated=True, pixel_certificate=certificate.evidence())
                    if not record["pose_and_frame_pairing_passed"]:
                        raise ContractError("Capture is not actual signature/frame paired to transported geometry")
                    camera = Camera.from_capture(view, pixel_certificate=certificate)
                    if meshes is None:
                        raise ContractError("Material identity/world transport rejected")
                    record.update(geometry_visibility_evaluated=True,
                                  geometry_visibility=validate_reprojection(anchor, camera, meshes),
                                  status="certified_geometry_projection_without_independent_semantic_observation")
                except ContractError as exc:
                    record.update(status="image_projection_and_visibility_acceptance_rejected", reason=str(exc),
                                  projected_xy=None, diagnostic_camera_created=False)
                image_records.append(record)
            result["images"] = image_records
            result["all_views_strict_pixel_and_pose_paired"] = all(i["pixel_contract_validated"] and i["pose_and_frame_pairing_passed"] for i in image_records)
            result["all_views_numeric_geometry_projection_passed"] = all(i.get("geometry_visibility", {}).get("status") == "projected_without_independent_observation" for i in image_records)
            result["image_acceptance_kind"] = "Independent semantic/anatomical acceptance remains false; calibrated geometric projection is counted separately."
            if result["all_views_strict_pixel_and_pose_paired"] and result["all_views_numeric_geometry_projection_passed"]:
                diagnostic_definition = {"id": "head_front_support_vertex", "maximizing_vertex_ids": [3468],
                                         "views": [i["geometry_visibility"] for i in image_records]}
                diagnostic_observations = [{"view_index": i, "path": v["path"], "yaw": v["yaw"]} for i, v in enumerate(views)]
                result["visual_review_diagnostics"] = render_support_diagnostics(diagnostic_definition, diagnostic_observations,
                                                                                 args.out / "visual_review" / (group+"__"+name))
            results.append(result)
            print(json.dumps({"case": group+"/"+name, "geometry": result["status"], "images_certified": sum(r["pixel_contract_validated"] for r in image_records)}), flush=True)
    for group in {r["group"] for r in results}:
        group_results = [r for r in results if r["group"] == group]
        baseline_name = "head_2_baseline" if group == "paired_base_cases" else "baseline"
        baseline = next((r for r in group_results if r["case"] == baseline_name and "independent_lbs_material" in r), None)
        if baseline is None:
            continue
        position = np.asarray(baseline["independent_lbs_material"]["world"])
        normal = baseline["independent_lbs_material"]["geometric_normal"]
        for row in group_results:
            if "independent_lbs_material" not in row:
                continue
            displacement = np.asarray(row["independent_lbs_material"]["world"]) - position
            row["world_displacement_from_group_baseline_game_units"] = displacement.tolist()
            row["world_displacement_from_group_baseline_l2"] = float(np.linalg.norm(displacement))
            row["normal_rotation_from_group_baseline_degrees"] = angle(row["independent_lbs_material"]["geometric_normal"], normal)
            row["group_baseline_comparison_limit"] = "Actual world snapshot difference includes inherited pose/runtime changes; not an isolated derivative or semantic nose response."
    report = {"schema_version": 1, "evidence_kind": "fixed_actual_material_point_saved_probe_transport",
              "anchor_path": str(args.anchor.resolve()), "anchor_sha256": digest(args.anchor),
              "original_geometry_source": baseline_source, "manifest_sources": manifest_sources,
              "pixel_report_path": str(args.pixel_report.resolve()), "pixel_report_sha256": digest(args.pixel_report),
              "scope_comparison_reference_report_path": str(args.reference_report.resolve()), "scope_comparison_reference_report_sha256": digest(args.reference_report),
              "implementation_sha256": digest(Path(__file__)), "fixed_point": anchor,
              "case_count": len(results), "actual_png_count": sum(len(r["images"]) for r in results),
              "source_bound_numeric_transport_verified_count": sum(r["status"] == "source_bound_numeric_material_transport_verified" for r in results),
              "strict_pixel_certified_png_count": sum(i["pixel_contract_validated"] for r in results for i in r["images"]),
              "actual_pose_paired_png_count": sum(i["pose_and_frame_pairing_passed"] for r in results for i in r["images"]),
              "strict_numeric_geometry_projection_png_count": sum(i.get("geometry_visibility", {}).get("status") == "projected_without_independent_observation" for r in results for i in r["images"]),
              "anatomical_correspondence_validated": False, "results": results,
              "limits": ["The fixed support vertex is an operational geometric material point, not nose anatomy.",
                         "Identity is frozen: no extremum reselection, substitute triangle, UV heuristic or bone proxy.",
                         "LBS reconstructs actual saved bones and active expression deltas; this does not certify the offline native parameter model or ABMX configuration replay.",
                         "Old image scope is never inferred from later captures, bridge version, filenames or same resolution. No uncertified camera is created and no rejected PNG is projected.",
                         "Renderer enabled/active flags are reported, but camera occlusion/material visibility require a correctly paired strictly pixel-certified capture and complete material evidence.",
                         "Probe differences are actual world snapshot differences; different animation/parent poses can confound isolated native/ABMX effect claims."]}
    args.out.mkdir(parents=True, exist_ok=True)
    for row in results:
        (args.out / (row["group"]+"__"+row["case"]+".json")).write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["case_count", "actual_png_count", "source_bound_numeric_transport_verified_count", "strict_pixel_certified_png_count", "actual_pose_paired_png_count"]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["anchor", "pixel-report", "reference-report", "out"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--head2-manifest", type=Path)
    parser.add_argument("--paired-manifest", type=Path)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if not args.head2_manifest and not args.paired_manifest and not args.manifest:
        parser.error("At least one actual saved capture/geometry manifest required")
    try:
        run(args)
    except (ContractError, ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Fixed material transport rejected: {exc}\n")


if __name__ == "__main__":
    main()
