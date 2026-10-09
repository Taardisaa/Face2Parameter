"""Positive-edge spokes ARAP for asset authoring, not HS2 deformation logic.

Local rotations and the global right-hand side follow the spokes energy in
Sorkine/ Alexa 2007. Positive inverse-edge-length weights are explicit here;
they are not libigl's default cotangent weights or a claim of equivalence.
Reference: https://libigl.github.io/tutorial/#as-rigid-as-possible
"""
import numpy as np
from scipy import sparse


class SpokesARAP:
    def __init__(self, vertices, faces):
        self.rest = np.asarray(vertices, float)
        edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]],
            faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
        self.a, self.b = edges.T
        self.edge = self.rest[self.a]-self.rest[self.b]
        lengths = np.linalg.norm(self.edge, axis=1)
        if np.any(lengths == 0):
            raise ValueError("Degenerate logical edge")
        w = 1/lengths
        # Mean weighted degree = 1; fixes units of the authored data term.
        self.w = w/(2*w.sum()/len(vertices))
        n = len(vertices)
        adj = sparse.coo_matrix((np.r_[self.w, self.w],
            (np.r_[self.a, self.b], np.r_[self.b, self.a])), shape=(n, n)).tocsr()
        self.stiffness = sparse.diags(np.asarray(adj.sum(1)).reshape(-1))-adj

    def rotations(self, positions):
        edge = positions[self.a]-positions[self.b]
        cov_edges = self.w[:, None, None]*np.einsum('ni,nj->nij', self.edge, edge)
        covariance = np.zeros((len(self.rest), 3, 3))
        np.add.at(covariance, self.a, cov_edges)
        np.add.at(covariance, self.b, cov_edges)
        u, _, vt = np.linalg.svd(covariance)
        v = vt.transpose(0, 2, 1)
        correction = np.ones((len(self.rest), 3))
        correction[:, 2] = np.linalg.det(v@u.transpose(0, 2, 1))
        return (v*correction[:, None, :])@u.transpose(0, 2, 1)

    def rhs(self, positions):
        rotation = self.rotations(positions)
        local_edges = np.einsum('nij,nj->ni',
            (rotation[self.a]+rotation[self.b])/2, self.edge)*self.w[:, None]
        result = np.zeros_like(self.rest)
        np.add.at(result, self.a, local_edges)
        np.add.at(result, self.b, -local_edges)
        return result
