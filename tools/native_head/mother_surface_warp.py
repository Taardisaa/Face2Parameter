"""Local surface-gradient transport for native mother-template authoring.

This defines new asset geometry; it is not a recovered HS2 deformation driver.
Triangle correspondence and Jacobians are explicit and reproducible. Positive
local determinants do not prove a globally injective volumetric mapping.
"""
import numpy as np

from tools.model_bridge.scan_accuracy import TriangleSurface


def triangle_frames(vertices, faces):
    tri = vertices[faces]
    ab, ac = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    cross = np.cross(ab, ac)
    area2 = np.linalg.norm(cross, axis=1)
    if np.any(area2 <= 0):
        raise ValueError('Collapsed triangle cannot transport a surface frame')
    # Normal-column length has the same units as the two tangent edges. This
    # explicit area-derived extension avoids inventing a constant thickness.
    normal_edge = cross/np.sqrt(area2)[:, None]
    return np.stack([ab, ac, normal_edge], axis=2), area2


class SurfaceWarp:
    def __init__(self, original, authored, faces):
        self.original = np.asarray(original, float)
        self.authored = np.asarray(authored, float)
        self.faces = np.asarray(faces, int)
        rest, self.areas = triangle_frames(self.original, self.faces)
        new, _ = triangle_frames(self.authored, self.faces)
        self.gradients = new@np.linalg.inv(rest)
        if np.any(np.linalg.det(self.gradients) <= 0):
            raise ValueError('Non-positive local surface transport')
        self.surface = TriangleSurface(self.original[self.faces])

    def map(self, points):
        points = np.asarray(points, float)
        closest, ids, _ = self.surface.closest(points)
        tri = self.original[self.faces[ids]]
        ab, ac, ap = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0], closest-tri[:, 0]
        dot = lambda a, b: np.sum(a*b, axis=1)
        aa, cc, cross = dot(ab, ab), dot(ac, ac), dot(ab, ac)
        denominator = aa*cc-cross*cross
        if np.any(denominator <= 0):
            raise ValueError('Degenerate selected source triangle')
        b = (cc*dot(ap, ab)-cross*dot(ap, ac))/denominator
        c = (aa*dot(ap, ac)-cross*dot(ap, ab))/denominator
        bary = np.stack([1-b-c, b, c], axis=1)
        q = (self.authored[self.faces[ids]]*bary[:, :, None]).sum(1)
        jac = self.gradients[ids]
        mapped = q+np.einsum('nij,nj->ni', jac, points-closest)
        return mapped, jac, dict(source_triangle_ids=ids.tolist(),
            barycentric=bary.tolist(), source_normal_offsets=(points-closest).tolist())

    def vertex_gradients(self, aliases):
        # All UV/skin-identical source copies accumulate one common geometric
        # gradient. Shader normal/tangent directions remain per-render-vertex.
        canonical = np.array([aliases.get(i, i) for i in range(len(self.original))])
        total = np.zeros((len(self.original), 3, 3))
        mass = np.zeros(len(self.original))
        for k in range(3):
            ids = canonical[self.faces[:, k]]
            np.add.at(total, ids, self.gradients*self.areas[:, None, None])
            np.add.at(mass, ids, self.areas)
        if np.any(mass[canonical] == 0):
            raise ValueError('Unconnected native vertex')
        jac = total[canonical]/mass[canonical, None, None]
        if np.any(np.linalg.det(jac) <= 0):
            raise ValueError('Incident surface gradients produce an inverted vertex frame')
        return jac


def direction_maps(jacobian, normal, tangent):
    """Transfer the native authored frame, keeping its tangent handedness."""
    normal_map = np.linalg.inv(jacobian).transpose(0, 2, 1)
    new_n = np.einsum('nij,nj->ni', normal_map, normal)
    normal_length = np.linalg.norm(new_n, axis=1)
    if np.any(normal_length == 0):
        raise ValueError('Zero authored normal')
    normal_map /= normal_length[:, None, None]
    new_n /= normal_length[:, None]
    projector = np.eye(3)[None]-np.einsum('ni,nj->nij', new_n, new_n)
    tangent_map = projector@jacobian
    new_t = np.einsum('nij,nj->ni', tangent_map, tangent)
    tangent_length = np.linalg.norm(new_t, axis=1)
    if np.any(tangent_length == 0):
        raise ValueError('Zero authored tangent')
    tangent_map /= tangent_length[:, None, None]
    return normal_map, tangent_map
