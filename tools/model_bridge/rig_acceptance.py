"""Small final original-decoder acceptance for the implemented HS2 source rig.

Uses recorded model outputs and one declared composite pose per model. Does not
infer game behavior, fit geometry, change source identity or capture photographs.
Always restores the supplied pre-existing character backup after a live run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .artifact import ModelArtifact, sha
from .game_import import request
from .rig import replay_source_pose


# Chosen before inspecting game results: root, neck, jaw, and both eye joints.
COMPOSITE_POSE = [0., .07, 0., .025, -.035, 0., .12, 0., 0., 0., .07, 0., 0., -.07, 0.]
RAW_TOLERANCE = 1e-6


def separate(state):
    state = dict(state)
    vertices = np.asarray(state.pop("render_vertices", []), dtype=np.float32)
    triangles = np.asarray(state.pop("render_triangles", []), dtype=np.int32)
    return state, vertices, triangles


def load_card(base, card, digest):
    request(base, "POST", {"path": str(card.resolve())}, route="/maker/card/load")
    deadline = time.monotonic() + 30
    while True:
        result = request(base, "GET")
        if result.get("active") and result.get("artifact_sha256") == digest:
            return result
        if time.monotonic() >= deadline:
            raise RuntimeError("Card source record did not finish restoring")
        time.sleep(.25)


def accept_model(base, manifest, mesh, out, thumbnail):
    artifact = ModelArtifact(manifest)
    source = json.loads(mesh.read_text(encoding="utf-8"))
    expected, faces = artifact.mesh(head_local=True)
    faces = faces.reshape(-1)
    if source.get("format") != "hs2_source_head_mesh_v2":
        raise ValueError("Rig interchange v2 required")
    if not np.array_equal(np.asarray(source["vertices"], dtype=np.float32), expected) or not np.array_equal(
            np.asarray(source["triangles"], dtype=np.int32), faces):
        raise ValueError("Interchange differs from the original model output")
    reference = request(base, "GET")
    low, high = expected.min(0), expected.max(0)
    scale = float(reference["reference_local_size"][1] / (high[1] - low[1]))
    translation = np.asarray(reference["reference_local_center"]) - scale * ((low.astype(np.float64) + high) / 2)
    digest = sha(mesh)
    imported, actual, actual_faces = separate(request(base, "POST", {
        "path": str(mesh.resolve()), "sha256": digest, "scale": scale, "translation": translation.tolist()}))
    literal = np.array_equal(actual, expected) and np.array_equal(actual_faces, faces)
    report = {"source_manifest": str(manifest.resolve()), "source_mesh_sha256": digest,
              "imported": imported, "initial_literal_arrays_equal": bool(literal),
              "raw_tolerance": RAW_TOLERANCE, "composite_pose": COMPOSITE_POSE}
    evidence = {"original_expected": expected, "original_actual": actual, "faces": faces}
    try:
        if not literal or not imported.get("source_rig") or imported["initial_rig_replay_max"] > RAW_TOLERANCE:
            raise RuntimeError("Original source rig import failed")
        reference_vertices, reference_joints = replay_source_pose(artifact, COMPOSITE_POSE)
        posed, actual, actual_faces = separate(request(base, "POST", {"pose": COMPOSITE_POSE},
                                                     route="/maker/face/model/pose"))
        evidence.update(pose_expected=reference_vertices, pose_actual=actual, pose_joints_expected=reference_joints)
        pose_error = float(np.max(np.abs(actual - reference_vertices)))
        joint_error = float(np.max(np.abs(np.asarray(posed["source_posed_joints"]) - reference_joints)))
        report.update(posed=posed, pose_max_raw_error=pose_error, pose_joint_max_raw_error=joint_error)
        if pose_error > RAW_TOLERANCE or joint_error > RAW_TOLERANCE or not np.array_equal(actual_faces, faces):
            raise RuntimeError("Source pose differs from original LBS decoder")
        card = out / "posed_source_card.png"
        saved = request(base, "POST", {"path": str(card.resolve()), "thumbnail": str(thumbnail.resolve())},
                        route="/maker/face/model/save")
        request(base, "DELETE")
        restored, reloaded_vertices, reloaded_faces = separate(load_card(base, card, digest))
        roundtrip = (np.array_equal(actual, reloaded_vertices) and np.array_equal(faces, reloaded_faces)
                     and posed["source_pose_axis_angle"] == restored.get("source_pose_axis_angle")
                     and posed["local_scale"] == restored.get("local_scale")
                     and posed["local_position"] == restored.get("local_position"))
        evidence["reloaded_pose_actual"] = reloaded_vertices
        report.update(saved=saved, restored=restored, posed_card_sha256=sha(card), posed_card_preserved=bool(roundtrip))
        if not roundtrip:
            raise RuntimeError("Posed source card changed geometry, pose or placement")
        reset, reset_vertices, reset_faces = separate(request(base, "POST", {"reset": True},
                                                            route="/maker/face/model/pose"))
        evidence["reset_actual"] = reset_vertices
        reset_equal = np.array_equal(reset_vertices, expected) and np.array_equal(reset_faces, faces)
        report.update(reset=reset, reset_literal_arrays_equal=bool(reset_equal))
        if not reset_equal:
            raise RuntimeError("Reset did not recover literal original source output")
        report["passed"] = True
    except Exception as exc:
        report.update(passed=False, failure=str(exc))
        raise
    finally:
        np.savez_compressed(out / "geometry_evidence.npz", **evidence)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"passed": report["passed"], "report": str((out / "report.json").resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:43127")
    parser.add_argument("--smirk-manifest", type=Path, required=True)
    parser.add_argument("--smirk-mesh", type=Path, required=True)
    parser.add_argument("--mica-manifest", type=Path, required=True)
    parser.add_argument("--mica-mesh", type=Path, required=True)
    parser.add_argument("--backup-card", type=Path, required=True)
    parser.add_argument("--thumbnail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new evidence directory; failed evidence is retained")
    if not args.backup_card.is_file() or not args.thumbnail.is_file():
        raise FileNotFoundError("Existing backup card and thumbnail required")
    before, before_vertices, before_faces = separate(request(args.base, "GET"))
    if not before.get("active"):
        raise RuntimeError("Load the backup card before running acceptance")
    args.out.mkdir(parents=True)
    receipt = {"backup_card": str(args.backup_card.resolve()), "backup_sha256": sha(args.backup_card),
               "before": before, "models": {}}
    try:
        for name in ("smirk", "mica"):
            out = args.out / name
            out.mkdir()
            receipt["models"][name] = accept_model(args.base, getattr(args, name + "_manifest"),
                getattr(args, name + "_mesh"), out, args.thumbnail)
    except Exception as exc:
        receipt["failure"] = str(exc)
        raise
    finally:
        after, vertices, faces = separate(load_card(args.base, args.backup_card, before["artifact_sha256"]))
        restored = (np.array_equal(vertices, before_vertices) and np.array_equal(faces, before_faces)
                    and before["local_position"] == after.get("local_position")
                    and before["local_scale"] == after.get("local_scale")
                    and before.get("source_pose_axis_angle") == after.get("source_pose_axis_angle"))
        receipt.update(after=after, original_character_restored=bool(restored))
        (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if not restored:
            raise RuntimeError("Original character restoration differs; evidence retained")
    print(json.dumps({"receipt": str((args.out / "receipt.json").resolve()),
                      "models_passed": list(receipt["models"]), "original_character_restored": restored}))


if __name__ == "__main__":
    main()
