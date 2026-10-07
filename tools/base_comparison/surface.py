"""Area quadrature and exact point-to-triangle distances; no fitted alignment."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.spatial import cKDTree


def content_hash(vertices, faces):
    digest = hashlib.sha256()
    for array, dtype in ((vertices, "<f8"), (faces, "<i8")):
        array = np.asarray(array, dtype=dtype)
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def topology_hash(faces):
    array = np.asarray(faces, dtype="<i8")
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def geometry(vertices, faces):
    triangles = np.asarray(vertices, float)[np.asarray(faces, int)]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, double_area[:, None], out=np.zeros_like(cross), where=double_area[:, None] > 0)
    return triangles, double_area / 2, normals


@dataclass
class Quadrature:
    face_ids: np.ndarray
    barycentric: np.ndarray
    weights: np.ndarray

    def points(self, vertices, faces):
        return np.einsum("nij,ni->nj", np.asarray(vertices)[np.asarray(faces)[self.face_ids]], self.barycentric)


def quadrature(vertices, faces, count=4096, seed=0, cover_every_face=True):
    """One sample per positive-area face plus area-proportional extra samples.

    Each face's total quadrature weight is its exact area. Thus small dense
    triangles cannot outweigh a large coarse face just by vertex/face count.
    Optimization may choose random area-proportional samples instead.
    """
    _, areas, _ = geometry(vertices, faces)
    valid = np.flatnonzero(areas > 0)
    if not len(valid) or not np.isfinite(areas).all():
        raise ValueError("No finite positive-area surface")
    rng = np.random.default_rng(seed)
    if cover_every_face:
        counts = np.zeros(len(faces), dtype=int)
        counts[valid] = 1
        counts += rng.multinomial(max(0, count - len(valid)), areas / areas.sum())
        ids = np.repeat(np.arange(len(faces)), counts)
        weights = areas[ids] / counts[ids]
    else:
        ids = rng.choice(len(faces), size=count, p=areas / areas.sum())
        weights = np.full(count, areas.sum() / count)
    u = rng.random((len(ids), 2))
    root = np.sqrt(u[:, 0])
    barycentric = np.column_stack([1 - root, root * (1 - u[:, 1]), root * u[:, 1]])
    return Quadrature(ids, barycentric, weights)


def closest_on_triangles(point, triangles):
    """Closest points/barycentrics on each supplied triangle, including edges.

    Plane projection is used only inside a nondegenerate triangle; all three
    edge projections are candidates. Degenerate faces reduce to segments.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    aa, cc, bc = (ab * ab).sum(1), (ac * ac).sum(1), (ab * ac).sum(1)
    pa, pc = (ap * ab).sum(1), (ap * ac).sum(1)
    denominator = aa * cc - bc * bc
    v = np.divide(cc * pa - bc * pc, denominator, out=np.zeros_like(pa), where=denominator > 0)
    w = np.divide(aa * pc - bc * pa, denominator, out=np.zeros_like(pc), where=denominator > 0)
    bary = np.column_stack([1 - v - w, v, w])
    plane = np.einsum("nij,ni->nj", triangles, bary)
    distance2 = ((plane - point) ** 2).sum(1)
    distance2[(denominator <= 0) | (bary.min(1) < 0)] = np.inf
    for start, end in ((0, 1), (1, 2), (2, 0)):
        edge = triangles[:, end] - triangles[:, start]
        edge2 = (edge * edge).sum(1)
        fraction = np.divide(((point - triangles[:, start]) * edge).sum(1), edge2,
                             out=np.zeros_like(edge2), where=edge2 > 0).clip(0, 1)
        edge_bary = np.zeros_like(bary)
        edge_bary[:, start], edge_bary[:, end] = 1 - fraction, fraction
        edge_point = triangles[:, start] + fraction[:, None] * edge
        edge_distance2 = ((edge_point - point) ** 2).sum(1)
        better = edge_distance2 < distance2
        distance2[better], plane[better], bary[better] = edge_distance2[better], edge_point[better], edge_bary[better]
    return distance2, plane, bary


class TriangleSurface:
    def __init__(self, vertices, faces):
        self.triangles, self.areas, self.normals = geometry(vertices, faces)
        self.centers = self.triangles.mean(1)
        self.radii = np.linalg.norm(self.triangles - self.centers[:, None], axis=2).max(1)
        self.tree = cKDTree(self.centers)
        self.maximum_radius = float(self.radii.max())

    def closest(self, points):
        points = np.asarray(points, float)
        distances, face_ids, barycentrics = [], [], []
        nearest_centers = self.tree.query(points)[1]
        for point, initial in zip(points, nearest_centers):
            upper2 = closest_on_triangles(point, self.triangles[initial:initial + 1])[0][0]
            # Any closer triangle's center lies within upper + its radius.
            ids = np.asarray(self.tree.query_ball_point(point, np.sqrt(upper2) + self.maximum_radius + 1e-12), int)
            center_distance = np.linalg.norm(self.centers[ids] - point, axis=1)
            ids = ids[center_distance - self.radii[ids] <= np.sqrt(upper2) + 1e-12]
            d2, _, bary = closest_on_triangles(point, self.triangles[ids])
            best = int(d2.argmin())
            distances.append(np.sqrt(max(0, d2[best])))
            face_ids.append(ids[best])
            barycentrics.append(bary[best])
        return np.asarray(distances), np.asarray(face_ids), np.asarray(barycentrics)


def weighted_quantile(values, weights, percentile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][min(len(order) - 1, np.searchsorted(cumulative, percentile * cumulative[-1]))])


def statistics(distances, weights):
    return {"rms": float(np.sqrt(np.average(distances ** 2, weights=weights))),
            "mean": float(np.average(distances, weights=weights)),
            "p95": weighted_quantile(distances, weights, 0.95), "max_sampled": float(distances.max())}


def evaluate_surface(source_vertices, source_faces, target_vertices, target_faces, *, count=4096, seed=0, regions=None):
    """Symmetric area-weighted point-to-surface evaluation over entire meshes."""
    source, target = TriangleSurface(source_vertices, source_faces), TriangleSurface(target_vertices, target_faces)
    sq = quadrature(source_vertices, source_faces, count, seed)
    tq = quadrature(target_vertices, target_faces, count, seed)
    sd, st, _ = target.closest(sq.points(source_vertices, source_faces))
    td, ts, _ = source.closest(tq.points(target_vertices, target_faces))
    sn = np.degrees(np.arccos(np.einsum("ij,ij->i", source.normals[sq.face_ids], target.normals[st]).clip(-1, 1)))
    tn = np.degrees(np.arccos(np.einsum("ij,ij->i", target.normals[tq.face_ids], source.normals[ts]).clip(-1, 1)))
    equal_surface_weights = np.r_[sq.weights / sq.weights.sum() / 2, tq.weights / tq.weights.sum() / 2]
    report = {"source_to_target": statistics(sd, sq.weights), "target_to_source": statistics(td, tq.weights),
              "symmetric": statistics(np.r_[sd, td], equal_surface_weights),
              "oriented_normal_degrees": statistics(np.r_[sn, tn], equal_surface_weights),
              "source_area": float(source.areas.sum()), "target_area": float(target.areas.sum()),
              "source_query_count": len(sd), "target_query_count": len(td), "seed": seed,
              "sampling": "one per nonzero-area triangle plus area-proportional extra; per-face weights sum to exact area",
              "distance": "exact closest triangle for each sampled query; KDTree bounds only prune candidates",
              "maximum_caveat": "max_sampled is not a continuous Hausdorff bound; no vertex-density weighted loss",
              "alignment": "none fitted; only input-declared proper rigid transforms may have been removed"}
    report["regions"] = []
    for region in regions or []:
        if not region.get("provenance") or region.get("side") not in ("source", "target"):
            raise ValueError("Named region needs provenance and explicit source/target side")
        q, d, n, faces = (sq, sd, sn, source_faces) if region["side"] == "source" else (tq, td, tn, target_faces)
        if region.get("topology_sha256") != topology_hash(faces):
            raise ValueError("Named region must bind to the exact source/target topology hash")
        face_ids = np.asarray(region["face_ids"])
        if not np.issubdtype(face_ids.dtype, np.integer) or not len(face_ids) or face_ids.min() < 0 or face_ids.max() >= len(faces):
            raise ValueError("Region triangle IDs outside declared mesh topology")
        selected = np.isin(q.face_ids, face_ids)
        if not selected.any():
            raise ValueError("Region has no positive-area query samples")
        report["regions"].append({**region, "distance": statistics(d[selected], q.weights[selected]),
                                  "normal_degrees": statistics(n[selected], q.weights[selected])})
    return report
