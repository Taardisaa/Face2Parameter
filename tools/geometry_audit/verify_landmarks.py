"""Read-only integration audit of real cards, cached head dispatch and bone proxies.

Run from the repository with .venv/Scripts/python.exe. No game connection,
card writes, model downloads or inference are performed. Triangle distances
diagnose a proxy's surface offset; they do not calibrate anatomical identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.face_data_utils.utils import FaceData  # noqa: E402
from src.face_metrics.landmarks import MESH_BONE, card_landmarks  # noqa: E402
from src.hs2_mesh import _rig, card_head_id, card_to_mesh, fd_to_inputs  # noqa: E402
from src.hs2_mesh_deform import available_heads  # noqa: E402


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nearest_triangle_distance(point, triangles):
    """Exact unsigned distance to a triangle union, including degenerate faces."""
    a, b, c = np.moveaxis(np.asarray(triangles, dtype=float), 1, 0)
    p = np.asarray(point, dtype=float)
    ab, ac, ap = b - a, c - a, p - a
    normal = np.cross(ab, ac)
    normal2 = np.einsum("ij,ij->i", normal, normal)
    nondegenerate = normal2 > 0
    safe = np.where(nondegenerate, normal2, 1.0)
    offset = np.einsum("ij,ij->i", ap, normal)
    projected = p - (offset / safe)[:, None] * normal
    aq = projected - a
    uu = np.einsum("ij,ij->i", ab, ab)
    uv = np.einsum("ij,ij->i", ab, ac)
    vv = np.einsum("ij,ij->i", ac, ac)
    qu = np.einsum("ij,ij->i", aq, ab)
    qv = np.einsum("ij,ij->i", aq, ac)
    det = uu * vv - uv * uv
    safe_det = np.where(det > 0, det, 1.0)
    u = (qu * vv - qv * uv) / safe_det
    v = (qv * uu - qu * uv) / safe_det
    inside = nondegenerate & (det > 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
    best2 = np.where(inside, offset * offset / safe, np.inf)
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        edge2 = np.einsum("ij,ij->i", edge, edge)
        t = np.einsum("ij,ij->i", p - start, edge) / np.where(edge2 > 0, edge2, 1.0)
        nearest = start + np.clip(t, 0, 1)[:, None] * edge
        best2 = np.minimum(best2, np.einsum("ij,ij->i", p - nearest, p - nearest))
    return float(np.sqrt(best2.min()))


def verify_distance_oracle():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    # Interior projection, exterior projection and a degenerate segment.
    np.testing.assert_allclose(nearest_triangle_distance([0.2, 0.2, 2], tri), 2)
    np.testing.assert_allclose(nearest_triangle_distance([1, 1, 0], tri), np.sqrt(0.5))
    segment = np.array([[[0, 0, 0], [1, 0, 0], [1, 0, 0]]], dtype=float)
    np.testing.assert_allclose(nearest_triangle_distance([0.5, 2, 0], segment), 2)


def discover_cards(heads):
    """Use extraction manifests, verifying actual card metadata rather than manifest IDs."""
    candidates = [ROOT / "tests/HS2ChaF_20240901192905747.png"]
    for path in sorted((ROOT / "data/hs2_head/cards").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        card = Path(manifest["card"])
        candidates.append(card if card.is_absolute() else ROOT / card)
    selected = {}
    for card in candidates:
        if not card.is_file():
            continue
        head_id = card_head_id(str(card))
        if head_id in heads:
            selected.setdefault(head_id, card)
        if set(selected) == set(heads):
            break
    missing = sorted(set(heads) - set(selected))
    require(not missing, f"No existing card found for cached heads {missing}; pass --card paths.")
    return [selected[head] for head in heads]


def audit_card(card):
    card_digest = sha256(card)
    head_id = card_head_id(str(card))
    fd = FaceData(str(card))
    shape, _ = fd_to_inputs(fd)
    stored = fd.card_data.Custom["face"]["shapeValueFace"]
    require(len(shape) == 59, f"{card}: rig input must include 59 native values")
    require(len(fd.base_data) == 54, f"{card}: model native block must contain 54 values")
    np.testing.assert_array_equal(shape[54:], stored[54:59])
    vector = fd.to_vector(is_simplify=True, without_right=True, normalize=True, use_gaussian=False)
    require(vector.shape == (205,), f"{card}: unexpected model vector shape {vector.shape}")
    # Observe the real rig call while loading actual assets. Using a wrong existing
    # head could otherwise produce finite points and falsely pass this audit.
    with patch("src.hs2_mesh._rig", wraps=_rig) as select_rig:
        landmarks = card_landmarks(str(card))
        require(select_rig.call_count == 1, "Expected one rig selection for card landmarks")
        selected = select_rig.call_args
        require(selected.args == (head_id,) and not selected.kwargs,
                f"{card}: selected rig {selected}, card metadata requires head {head_id}")
    require(set(landmarks) == set(MESH_BONE) | {"GLABELLA"}, "Incomplete landmark dictionary")
    require(all(np.asarray(v).shape == (3,) and np.isfinite(v).all() for v in landmarks.values()),
            f"{card}: invalid landmark coordinates")
    verts, faces = card_to_mesh(str(card))
    require(np.isfinite(verts).all(), f"{card}: non-finite skinned mesh")
    triangles = verts[faces]
    extent = np.ptp(verts, axis=0)
    require(np.all(extent > 0), f"{card}: collapsed mesh extent")
    area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    icd = float(np.linalg.norm(landmarks["EYE_INNER_L"] - landmarks["EYE_INNER_R"]))
    require(icd > 0, f"{card}: collapsed intercanthal proxy distance")
    distances = {name: nearest_triangle_distance(point, triangles)
                 for name, point in landmarks.items()}
    return {
        "card": str(card.resolve()), "card_sha256": card_digest, "head_id": head_id,
        "head_mesh_sha256": sha256(ROOT / f"data/hs2_head/head_{head_id}/o_head_mesh.npz"),
        "native_count": len(shape), "model_native_count": len(fd.base_data),
        "model_vector_count": len(vector), "ear_values_preserved": True,
        "landmark_count": len(landmarks), "vertex_count": len(verts), "triangle_count": len(faces),
        "mesh_extent_game_units": extent.tolist(), "surface_area_game_units_squared": float(area2.sum() / 2),
        "degenerate_triangles": int((area2 <= np.linalg.norm(extent) ** 2 * 1e-12).sum()),
        "intercanthal_proxy_distance_game_units": icd,
        "bone_proxy_to_head_surface_game_units": distances,
        "bone_proxy_to_head_surface_over_intercanthal": {name: d / icd for name, d in distances.items()},
        "landmarks": {name: point.tolist() for name, point in landmarks.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, action="append", help="Repeat for real cards of different heads")
    parser.add_argument("--out", type=Path, help="Optional audit JSON; card and cache files remain untouched")
    args = parser.parse_args()
    verify_distance_oracle()
    heads = available_heads()
    require(len(heads) >= 2, "This regression audit requires at least two cached heads")
    cards = args.card or discover_cards(heads)
    reports = [audit_card(card) for card in cards]
    tested_heads = sorted({entry["head_id"] for entry in reports})
    require(len(tested_heads) >= 2, "Provide cards covering at least two different cached head IDs")
    if not args.card:
        require(tested_heads == heads, "Auto-discovered cards did not cover every cached head")
    # Revisit the first head after other heads have loaded: catches shared-rig cache contamination.
    repeated = card_landmarks(str(cards[0]))
    for name, expected in reports[0]["landmarks"].items():
        np.testing.assert_array_equal(repeated[name], expected)
    for card, report in zip(cards, reports):
        require(sha256(card) == report["card_sha256"], f"Card was unexpectedly modified: {card}")
    output = {
        "schema_version": 1, "scope": "offline read-only integration; not likeness acceptance",
        "cached_heads": heads, "tested_heads": tested_heads,
        "checks": {"distance_oracle": "pass", "multi_head_card_dispatch": "pass",
                   "repeat_after_other_heads": "pass", "native59_model54_vector205": "pass",
                   "ear_values_preserved": "pass", "cards_unchanged": "pass"},
        "surface_distance_caveat": "Distance to o_head triangles only, not anatomical correspondence; eye/teeth submeshes excluded.",
        "cards": reports,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": output["checks"], "tested_heads": tested_heads,
                      "vertex_counts": {entry["head_id"]: entry["vertex_count"] for entry in reports},
                      "out": str(args.out) if args.out else None}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
