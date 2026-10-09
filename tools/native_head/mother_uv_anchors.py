"""Source-bound UV annotations lifted onto the actual native triangles.

These are candidate anatomical constraints, not an anatomical validation claim.
Lip annotations use original donor pigmentation; no UV rectangle is a cut mask.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import sparse

from tools.model_bridge.artifact import sha
from tools.native_head.atlas_retarget import lip_contour
from tools.native_head.mother_template_inputs import source_file


def embed(uv_point, arrays):
    corners = arrays["uv"][arrays["faces"]].astype(float)
    a = corners[:, 0]
    b, c = corners[:, 1]-a, corners[:, 2]-a
    det = b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0]
    offset = np.asarray(uv_point)-a
    valid = np.abs(det) > np.finfo(float).eps
    u = np.divide(offset[:, 0]*c[:, 1]-offset[:, 1]*c[:, 0], det,
                  out=np.zeros(len(det)), where=valid)
    v = np.divide(b[:, 0]*offset[:, 1]-b[:, 1]*offset[:, 0], det,
                  out=np.zeros(len(det)), where=valid)
    hit = np.flatnonzero(valid & (u >= 0) & (v >= 0) & (u+v <= 1))
    if len(hit) != 1:
        raise ValueError("Annotation must identify one actual UV triangle; got "+str(len(hit)))
    i = int(hit[0]); weights = np.asarray([1-u[i]-v[i], u[i], v[i]])
    return arrays["faces"][i], weights, dict(face_id=i, barycentric=weights.tolist(), uv=list(uv_point))


def constraints(arrays, inverse, target_landmarks, native_profile, texture_manifest):
    profile = json.loads(native_profile.read_text(encoding="utf-8"))
    manifest = json.loads(texture_manifest.read_text(encoding="utf-8"))
    texture = next(t for t in manifest["textures"] if t["asset"] == "cf_head_02_00_t")
    if sha(texture["source_bundle"]) != texture["source_bundle_sha256"] or sha(texture["output"]) != texture["output_sha256"]:
        raise ValueError("Original native lip input changed")
    lips = lip_contour(Image.open(texture["output"]), profile["lip_pigment"])
    # Installed head2 UV U reverses mesh X; FLAME embedding is in mesh X.
    mirror = {31: 35, 32: 34, 33: 33, 34: 32, 35: 31,
              48: 54, 49: 53, 50: 52, 51: 51, 52: 50, 53: 49,
              54: 48, 55: 59, 56: 58, 57: 57, 58: 56, 59: 55}
    points = [(int(i), p) for i, p in profile["landmark_uv"].items() if 27 <= int(i) <= 35 and int(i) != 30]
    points += [(48+i, p) for i, p in enumerate(lips)]
    rows, cols, data, targets, annotations = [], [], [], [], []
    for source_id, uv_point in points:
        ids, weights, annotation = embed(uv_point, arrays)
        target_id = mirror.get(source_id, source_id)
        row = len(targets)
        rows.extend([row]*3); cols.extend(inverse[ids]); data.extend(weights)
        targets.append(target_landmarks[target_id])
        annotation.update(target_flame_landmark=target_id,
                          native_reference_position=(arrays["verts"][ids]*weights[:, None]).sum(0).tolist())
        annotations.append(annotation)
    # A visually inspectable forward support vertex is used as the nose-tip
    # candidate; the older 2D UV annotation was below the actual nose tip.
    forward = arrays["verts"][:, 2]
    ids = np.flatnonzero(forward == forward.max())
    if len(ids) != 1:
        raise ValueError("Native forward-support nose-tip candidate is ambiguous")
    tip = int(ids[0])
    row = len(targets); rows.append(row); cols.append(int(inverse[tip])); data.append(1.)
    targets.append(target_landmarks[30])
    annotations.append(dict(native_vertex_id=tip, target_flame_landmark=30,
        native_reference_position=arrays["verts"][tip].tolist(),
        policy="Unique forward support candidate; requires anatomical visual review"))
    matrix = sparse.coo_matrix((data, (rows, cols)),
                              shape=(len(targets), int(inverse.max())+1)).tocsr()
    return matrix, np.asarray(targets), dict(profile=source_file(native_profile),
        texture_manifest=source_file(texture_manifest), original_texture=texture,
        annotations=annotations, semantically_reviewed=False,
        limitations=["Pigment outline is a candidate outer-lip annotation, not an exact anatomical contour",
                     "Native nasal UV annotations need full 3D visual review",
                     "No inner lip/ear-root correspondence or anatomy certificate"])
