"""Can feature-space normalisation recover the signal that renders collapse?

Motivating question (user): the beauty scores on game renders sit in a narrow band — can we
rescale them to spread them out?

Rescaling the SCORE cannot work, and it is worth being precise about why: Spearman rho depends
only on order, and every monotone transform (affine, power, sigmoid, per-domain standardisation)
preserves order exactly. The narrow band is a symptom, not the disease.

Normalising the FEATURES is a different proposition and might work. Renders collapse into a
narrow cone, so a large shared component ("this is a 3D render of a face") dominates the small
between-character differences. If the nuisance (a 3-degree yaw) lives in a FEW directions, we can
project them out and amplify what is left. If nuisance and signal share directions, we cannot —
whitening would amplify both and change nothing. That is testable, so test it.

Protocol note: the nuisance subspace is estimated on one half of the characters and evaluated on
the other. With 48 samples in 384 dimensions, fitting and scoring on the same set would "work"
for free and mean nothing.

Metric is rank-1 retrieval (given the 3-degree render, is the nearest 0-degree render the same
character?) — parameter-free, and it upper-bounds what any head on these features can do.

    .venv/Scripts/python.exe scripts/hs2_feature_normalization.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.hs2_backbone_probe import feats_dinov2, pairs  # noqa: E402


def rank1(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return float(((A @ B.T).argmax(0) == np.arange(len(B))).mean())


def ratio(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    noise = (1.0 - (A * B).sum(1)).mean()
    off = A @ A.T
    sig = (1.0 - off[np.triu_indices(len(A), 1)]).mean()
    return float(noise / max(sig, 1e-9))


def main():
    ps = pairs()
    if not ps:
        raise SystemExit("no control pairs — run hs2_baldness_ablation.py --control first")
    A = feats_dinov2([a for a, _ in ps]).astype(np.float64)      # yaw 0
    B = feats_dinov2([b for _, b in ps]).astype(np.float64)      # yaw 3
    n = len(A)
    rng = np.random.default_rng(0)
    fit = rng.permutation(n)[: n // 2]
    ev = np.setdiff1d(np.arange(n), fit)
    print(f"[features] {n} pairs, dim={A.shape[1]}; fit nuisance on {len(fit)}, evaluate on {len(ev)}\n")

    def show(label, f):
        Af, Bf = f(A), f(B)
        print(f"  {label:42s} rank-1={rank1(Af[ev], Bf[ev]) * 100:5.1f}%   "
              f"ratio={ratio(Af[ev], Bf[ev]):.3f}")

    show("raw (baseline)", lambda X: X)

    # 1. Centre on the render-domain mean: removes the single biggest shared component, the one
    #    that says "3D render" rather than "which character".
    mu = A[fit].mean(0)
    show("centred on render-domain mean", lambda X: X - mu)

    # 2. Project out the nuisance subspace, estimated as the principal directions of the
    #    (yaw3 - yaw0) differences. This is the direction the perturbation actually moves things.
    D = (B[fit] - A[fit])
    D = D - D.mean(0)
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    for k in (1, 3, 5, 10):
        P = np.eye(A.shape[1]) - Vt[:k].T @ Vt[:k]
        show(f"centred + nuisance subspace removed (k={k:2d})", lambda X, P=P: (X - mu) @ P)

    # 3. PCA-whiten within the render domain: equalise the variance of every within-domain
    #    direction, so tiny between-character directions stop being drowned out. Shrinkage is
    #    mandatory here — 24 fitting samples cannot support a 384x384 covariance.
    Xc = A[fit] - mu
    C = Xc.T @ Xc / max(len(fit) - 1, 1)
    for lam in (1e-2, 1e-1):
        Csh = (1 - lam) * C + lam * np.trace(C) / C.shape[0] * np.eye(C.shape[0])
        w, V = np.linalg.eigh(Csh)
        W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        show(f"centred + PCA-whitened (shrinkage={lam})", lambda X, W=W: (X - mu) @ W)

    print("\n  Read it against the baseline. A large jump means the signal was there and merely\n"
          "  swamped — worth retraining the reward on normalised features. Flat or worse means\n"
          "  nuisance and signal share directions, and no normalisation will separate them.")


if __name__ == "__main__":
    main()
