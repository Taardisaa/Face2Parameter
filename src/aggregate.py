"""Robust aggregation of per-image vectors for multi-image inference.

Pure numpy (no torch) so it works on either backbone features or 205-dim param
vectors. See docs/multi-image-aggregation.md for the rationale: several photos of
one person are consolidated into one stable vector, with outlier robustness.

- ``aggregate``     : per-dim median (default) / mean / trimmed-mean over a set of vectors.
- ``medoid_index``  : the most central member (for picking a representative thumbnail).
- ``ensemble``      : (weighted) mean of per-backbone 205-dim vectors, in param space.
"""

from __future__ import annotations

import numpy as np

_MAD_TO_STD = 1.4826  # scale so MAD estimates the std of a normal distribution


def aggregate(vectors: np.ndarray, method: str = "median") -> tuple[np.ndarray, np.ndarray]:
    """Aggregate ``(N, D)`` vectors into one ``(D,)``.

    Returns ``(agg, kept_mask)`` where ``kept_mask`` is an ``(N,)`` bool of which rows
    contributed (all True except for ``trimmed``, which drops outliers).

    - ``median``  : per-dimension median. Zero-parameter, robust to a minority of bad rows.
    - ``mean``    : per-dimension mean. Baseline, no robustness.
    - ``trimmed`` : drop whole rows whose L2 distance to the centroid exceeds
                    ``median(d) + 3*MAD``, then average the rest.
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"expected (N, D) vectors, got shape {v.shape}")
    n = v.shape[0]
    if n == 0:
        raise ValueError("no vectors to aggregate")

    if method == "median":
        return np.median(v, axis=0).astype(np.float32), np.ones(n, dtype=bool)
    if method == "mean":
        return v.mean(axis=0).astype(np.float32), np.ones(n, dtype=bool)
    if method == "trimmed":
        centroid = np.median(v, axis=0)
        d = np.linalg.norm(v - centroid, axis=1)
        med = np.median(d)
        mad = _MAD_TO_STD * np.median(np.abs(d - med))
        keep = np.ones(n, dtype=bool) if mad == 0 else (d <= med + 3 * mad)
        if not keep.any():  # degenerate; fall back to keeping all
            keep = np.ones(n, dtype=bool)
        return v[keep].mean(axis=0).astype(np.float32), keep
    raise ValueError(f"unknown aggregate method '{method}' (median|mean|trimmed)")


def medoid_index(vectors: np.ndarray, reference: np.ndarray) -> int:
    """Index of the row in ``vectors`` (N, D) closest (L2) to ``reference`` (D,)."""
    v = np.asarray(vectors, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    return int(np.argmin(np.linalg.norm(v - ref, axis=1)))


def ensemble(vectors: np.ndarray, weights=None) -> np.ndarray:
    """(Weighted) mean of per-backbone 205-dim vectors ``(M, D)`` -> ``(D,)``.

    Cross-backbone merging only makes sense in the shared 205-dim param space (the
    backbones' feature spaces differ in dim and are not comparable). ``weights`` is an
    optional length-M sequence (defaults to equal weight).
    """
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"expected (M, D) per-backbone vectors, got shape {v.shape}")
    if weights is None:
        out = v.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (v.shape[0],):
            raise ValueError(f"weights must have length {v.shape[0]}, got {w.shape}")
        out = (v * w[:, None]).sum(axis=0) / w.sum()
    return out.astype(np.float32)
