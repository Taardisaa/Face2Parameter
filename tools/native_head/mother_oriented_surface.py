"""Closest target triangle on the same oriented surface side for asset fitting.

The normal hemisphere is a correspondence restriction, not an anatomical label
or a proof of bijection. The final minimum distance is searched conservatively
over every compatible triangle that could improve the initial bound.
"""
import numpy as np

from tools.model_bridge.scan_accuracy import TriangleSurface, closest_on_triangles


class OrientedSurface(TriangleSurface):
    def __init__(self, triangles):
        super().__init__(triangles)
        cross = np.cross(self.triangles[:, 1]-self.triangles[:, 0],
                         self.triangles[:, 2]-self.triangles[:, 0])
        length = np.linalg.norm(cross, axis=1)
        if np.any(length == 0):
            raise ValueError('Degenerate target triangle has no surface side')
        self.normals = cross/length[:, None]

    def closest_oriented(self, points, normals):
        points, normals = np.asarray(points, float), np.asarray(normals, float)
        if points.shape != normals.shape or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals, axis=1)==0):
            raise ValueError('Finite nonzero correspondence normal required per point')
        result, face_ids, counts = [], [], []
        max_radius = float(self.radii.max())
        for point, normal in zip(points, normals):
            # Find one legal triangle for an upper bound; an incompatible nearest
            # centroid must not make the compatible search region too small.
            size = min(8, len(self.triangles))
            while True:
                _, initial_ids = self.tree.query(point, k=size)
                initial_ids = np.atleast_1d(initial_ids)
                compatible = initial_ids[self.normals[initial_ids]@normal > 0]
                if len(compatible):
                    nearest = closest_on_triangles(point, self.triangles[compatible])
                    best = int(np.argmin(np.linalg.norm(nearest-point, axis=1)))
                    upper = float(np.linalg.norm(nearest[best]-point))
                    break
                if size == len(self.triangles):
                    raise ValueError('No target triangle faces this source surface side')
                size = min(size*2, len(self.triangles))
            slack = 1e-12*max(1., np.linalg.norm(point), upper, max_radius)
            ids = np.asarray(self.tree.query_ball_point(point, upper+max_radius+slack), dtype=int)
            lower = np.linalg.norm(self.centers[ids]-point, axis=1)-self.radii[ids]
            ids = ids[(lower<=upper+slack) & (self.normals[ids]@normal>0)]
            closest = closest_on_triangles(point, self.triangles[ids])
            best = int(np.argmin(np.sum((closest-point)**2, axis=1)))
            result.append(closest[best]); face_ids.append(int(ids[best])); counts.append(len(ids))
        return np.asarray(result), np.asarray(face_ids), np.asarray(counts)


def logical_normals(vertices, faces):
    """Area weighted geometric normals, after logical UV copies are identified."""
    tri = vertices[faces]
    cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    normal = np.zeros_like(vertices, dtype=float)
    for k in range(3):
        np.add.at(normal, faces[:,k], cross)
    length = np.linalg.norm(normal, axis=1)
    if np.any(length == 0):
        raise ValueError('Logical vertex has no oriented surface normal')
    return normal/length[:,None]
